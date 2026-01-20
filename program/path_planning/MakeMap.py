import os
import json
from typing import Callable, Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd


# =========================================================
# Save dir helper
# =========================================================
def make_save_dir(img_path: str):
    DIR = os.path.dirname(__file__)
    dirname = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = f"{DIR}/../../outputs/{dirname}/{os.path.splitext(os.path.basename(img_path))[0]}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# =========================================================
# (A) Green dotted grid detection (lat/lon lines)
# =========================================================
def _cluster_centers(idxs, gap=3):
    idxs = np.asarray(idxs)
    if idxs.size == 0:
        return []
    clusters = []
    s = int(idxs[0])
    p = int(idxs[0])
    for a in idxs[1:]:
        a = int(a)
        if a - p > gap:
            clusters.append((s, p))
            s = a
        p = a
    clusters.append((s, p))
    return [int((a + b) // 2) for a, b in clusters]


def detect_latlon_grid_lines(
    img_bgr: np.ndarray,
    out_dir: str | None = None,
    roi=(90, 80, None, None),
    hsv_h_range=(45, 95),
    s_min=80,
    v_min=40,
    v_max=200,
    peak_ratio=0.50,
    merge_gap=3,
):
    """
    緑点線の格子（緯線・経線）を検出して、
    緯線(y px)・経線(x px)の配列を返す。
    """
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = roi
    if x1 is None:
        x1 = W
    if y1 is None:
        y1 = H
    crop = img_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h1, h2 = hsv_h_range
    lower = np.array([h1, s_min, v_min], np.uint8)
    upper = np.array([h2, 255, v_max], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "_mask_grid.png"), mask)

    row_sum = mask.sum(axis=1)
    col_sum = mask.sum(axis=0)
    if row_sum.max() == 0 or col_sum.max() == 0:
        return [], []

    ys_idx = np.where(row_sum >= row_sum.max() * peak_ratio)[0]
    xs_idx = np.where(col_sum >= col_sum.max() * peak_ratio)[0]

    ys = _cluster_centers(ys_idx, gap=merge_gap)
    xs = _cluster_centers(xs_idx, gap=merge_gap)

    return [y + y0 for y in ys], [x + x0 for x in xs]


# =========================================================
# (B) Lab distance utils
# =========================================================
def rgb_to_lab_int32(rgb: np.ndarray) -> np.ndarray:
    bgr = rgb[::-1].reshape(1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
    return lab.astype(np.int32)


def dist2_lab_img_to_lab_ref(lab_img_int32: np.ndarray, lab_ref_int32: np.ndarray) -> np.ndarray:
    d = lab_img_int32 - lab_ref_int32[None, None, :]
    return np.sum(d * d, axis=2).astype(np.int32)


# =========================================================
# (C) Land / shallow / magenta masks
# =========================================================
def extract_land_shallow_magenta_masks(
    img_bgr: np.ndarray,
    land_lower=(15, 50, 80),
    land_upper=(40, 255, 255),
    brown_excl_lower=(5, 40, 0),
    brown_excl_upper=(35, 255, 150),
    rgb_cyan=(182, 235, 219),
    rgb_s1=(167, 224, 233),
    rgb_s2=(129, 195, 226),
    T_SHALLOW=35,
    MARGIN=1,
    rgb_magenta=(198, 71, 186),
    T_MAG=30,
):
    """
    元画像から
      - mask_land    : 陸地（2値）
      - mask_shallow : 浅瀬（2値）
      - mask_magenta : マゼンタ線（2値）
    を返す
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int32)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # ---- land (HSV range) ----
    mask_land = cv2.inRange(hsv, np.array(land_lower, np.uint8), np.array(land_upper, np.uint8))
    mask_brown_excl = cv2.inRange(hsv, np.array(brown_excl_lower, np.uint8), np.array(brown_excl_upper, np.uint8))
    mask_land[mask_brown_excl > 0] = 0

    # remove small square-like components (insurance)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_land > 0).astype(np.uint8), connectivity=8)
    MAX_SIDE, MIN_SIDE = 60, 3
    MAX_AREA, MIN_AREA = 5000, 20
    ASPECT_MIN, ASPECT_MAX = 0.7, 1.4

    mask_land2 = mask_land.copy()
    for k in range(1, num):
        w = stats[k, cv2.CC_STAT_WIDTH]
        h = stats[k, cv2.CC_STAT_HEIGHT]
        area = stats[k, cv2.CC_STAT_AREA]
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
            continue
        aspect = w / max(h, 1e-9)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        mask_land2[labels == k] = 0

    mask_land = cv2.morphologyEx(mask_land2, cv2.MORPH_OPEN, k_open, iterations=1)
    mask_land = cv2.morphologyEx(mask_land, cv2.MORPH_CLOSE, k_close, iterations=1)

    # ---- shallow (Lab nearest + threshold) ----
    lab_cyan = rgb_to_lab_int32(np.array(rgb_cyan, np.uint8))
    lab_s1 = rgb_to_lab_int32(np.array(rgb_s1, np.uint8))
    lab_s2 = rgb_to_lab_int32(np.array(rgb_s2, np.uint8))

    d_cyan = dist2_lab_img_to_lab_ref(lab_img, lab_cyan)
    d_s1 = dist2_lab_img_to_lab_ref(lab_img, lab_s1)
    d_s2 = dist2_lab_img_to_lab_ref(lab_img, lab_s2)

    nearest = np.argmin(np.stack([d_cyan, d_s1, d_s2], axis=2), axis=2)
    mask_shallow = ((nearest == 1) | (nearest == 2))

    d_shallow = np.minimum(d_s1, d_s2)
    mask_shallow &= (d_shallow <= (T_SHALLOW * T_SHALLOW))
    mask_shallow &= (np.sqrt(d_shallow.astype(np.float32)) + float(MARGIN) <= np.sqrt(d_cyan.astype(np.float32)))

    mask_shallow_u8 = (mask_shallow.astype(np.uint8) * 255)
    mask_shallow_u8[mask_land > 0] = 0
    mask_shallow_u8 = cv2.morphologyEx(mask_shallow_u8, cv2.MORPH_OPEN, k_open, iterations=1)
    mask_shallow_u8 = cv2.morphologyEx(mask_shallow_u8, cv2.MORPH_CLOSE, k_close, iterations=1)

    # ---- magenta line (Lab distance) ----
    lab_mag = rgb_to_lab_int32(np.array(rgb_magenta, np.uint8))
    d_mag2 = dist2_lab_img_to_lab_ref(lab_img, lab_mag)
    mask_magenta = (d_mag2 <= (T_MAG * T_MAG)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_magenta = cv2.morphologyEx(mask_magenta, cv2.MORPH_OPEN, k3, iterations=1)

    return mask_land, mask_shallow_u8, mask_magenta


# =========================================================
# (D) Tiling + save geo meta into tiles_manifest.csv
# =========================================================
def split_by_lines_no_title_with_geo(
    img_bgr,
    ys,
    xs,
    out_dir,
    lat0_deg,
    lat0_min,
    lat_anchor_idx,
    lon0_deg,
    lon0_min,
    lon_anchor_idx,
    border=2,
):
    os.makedirs(out_dir, exist_ok=True)
    H, W = img_bgr.shape[:2]
    ys = sorted(map(int, ys))
    xs = sorted(map(int, xs))

    # ---- fixed assumptions (your request) ----
    lat_sign = -1
    lon_sign = +1
    dlat_min_per_line = 1.0
    dlon_min_per_line = 1.0

    # ---- convert to decimal degrees for saving/compute ----
    lat0 = float(lat0_deg) + float(lat0_min) / 60.0
    lon0 = float(lon0_deg) + float(lon0_min) / 60.0
    dlat = float(dlat_min_per_line) / 60.0
    dlon = float(dlon_min_per_line) / 60.0

    lat_anchor_idx = int(lat_anchor_idx)
    lon_anchor_idx = int(lon_anchor_idx)

    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("ys/xs is empty (grid lines not detected).")

    if not (0 <= lat_anchor_idx < len(ys)):
        raise ValueError(f"lat_anchor_idx out of range: {lat_anchor_idx} (len(ys)={len(ys)})")
    if not (0 <= lon_anchor_idx < len(xs)):
        raise ValueError(f"lon_anchor_idx out of range: {lon_anchor_idx} (len(xs)={len(xs)})")

    # line values (decimal degrees)
    lat_lines = [lat0 + (i - lat_anchor_idx) * dlat * lat_sign for i in range(len(ys))]
    lon_lines = [lon0 + (j - lon_anchor_idx) * dlon * lon_sign for j in range(len(xs))]

    # boundaries include outer image edge
    ys_b = [0] + ys + [H - 1]
    xs_b = [0] + xs + [W - 1]

    # outer bands: extrapolate 1 step outward (practical)
    def band_lat_bounds(i_band):
        if i_band == 0:
            top = lat_lines[0] + dlat * lat_sign
            bot = lat_lines[0]
        elif 1 <= i_band <= len(ys) - 1:
            top = lat_lines[i_band - 1]
            bot = lat_lines[i_band]
        else:
            top = lat_lines[-1]
            bot = lat_lines[-1] - dlat * lat_sign
        return (min(top, bot), max(top, bot))

    def band_lon_bounds(j_band):
        if j_band == 0:
            left = lon_lines[0] - dlon * lon_sign
            right = lon_lines[0]
        elif 1 <= j_band <= len(xs) - 1:
            left = lon_lines[j_band - 1]
            right = lon_lines[j_band]
        else:
            left = lon_lines[-1]
            right = lon_lines[-1] + dlon * lon_sign
        return (min(left, right), max(left, right))

    rows = []
    for i in range(len(ys_b) - 1):
        y0, y1 = ys_b[i], ys_b[i + 1]
        yy0 = max(0, y0 + border)
        yy1 = min(H, y1 - border)
        if yy1 - yy0 < 20:
            continue

        lat_min, lat_max = band_lat_bounds(i)

        for j in range(len(xs_b) - 1):
            x0, x1 = xs_b[j], xs_b[j + 1]
            xx0 = max(0, x0 + border)
            xx1 = min(W, x1 - border)
            if xx1 - xx0 < 20:
                continue

            lon_min, lon_max = band_lon_bounds(j)

            tile = img_bgr[yy0:yy1, xx0:xx1].copy()
            fname = f"tile_r{i:02d}_c{j:02d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), tile)

            rows.append(
                {
                    "tile": fname,
                    "x0": xx0,
                    "x1": xx1,
                    "y0": yy0,
                    "y1": yy1,
                    # per-tile bounds (decimal degrees)
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    # global meta (decimal degrees, for compute)
                    "lat0": lat0,
                    "lon0": lon0,
                    "dlat": dlat,
                    "dlon": dlon,
                    "lat_sign": lat_sign,
                    "lon_sign": lon_sign,
                    "lat_anchor_idx": lat_anchor_idx,
                    "lon_anchor_idx": lon_anchor_idx,
                    # optional: original input (deg/min) for human check
                    "lat0_deg": float(lat0_deg),
                    "lat0_min": float(lat0_min),
                    "lon0_deg": float(lon0_deg),
                    "lon0_min": float(lon0_min),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "tiles_manifest.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] saved tiles = {len(df)} -> {out_dir}")
    return df


# =========================================================
# (F) Polygon extraction utils
# =========================================================
def _extract_polygons_from_mask(
    mask_u8: np.ndarray,
    min_area: float = 2000.0,
    eps_factor: float = 0.004,
    close_ksize: int = 9,
    close_iter: int = 2,
    open_ksize: int = 5,
    open_iter: int = 1,
) -> List[np.ndarray]:
    mask = mask_u8.copy()

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_ksize), int(open_ksize)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=int(close_iter))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=int(open_iter))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polys: List[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < float(min_area):
            continue
        peri = cv2.arcLength(c, True)
        eps = float(eps_factor) * peri
        approx = cv2.approxPolyDP(c, eps, True)
        polys.append(approx.reshape(-1, 2).astype(int))
    return polys


def extract_impassable_polygons(
    image_path: str,
    min_area: float = 2000.0,
    eps_factor: float = 0.004,
    close_ksize: int = 9,
    close_iter: int = 2,
    open_ksize: int = 5,
    open_iter: int = 1,
):
    """
    impassable_map.png から
      - 灰 + シアン（浅瀬）を侵入不可として抽出
      - マゼンタは除外
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    cyan_lo = np.array([75, 35, 120], dtype=np.uint8)
    cyan_hi = np.array([110, 255, 255], dtype=np.uint8)
    mask_cyan = cv2.inRange(hsv, cyan_lo, cyan_hi)

    gray_lo = np.array([0, 0, 70], dtype=np.uint8)
    gray_hi = np.array([180, 35, 245], dtype=np.uint8)
    mask_gray = cv2.inRange(hsv, gray_lo, gray_hi)

    mask = cv2.bitwise_or(mask_cyan, mask_gray)

    mag_lo = np.array([135, 60, 60], dtype=np.uint8)
    mag_hi = np.array([175, 255, 255], dtype=np.uint8)
    mask_mag = cv2.inRange(hsv, mag_lo, mag_hi)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_mag))

    polys = _extract_polygons_from_mask(
        mask_u8=mask,
        min_area=min_area,
        eps_factor=eps_factor,
        close_ksize=close_ksize,
        close_iter=close_iter,
        open_ksize=open_ksize,
        open_iter=open_iter,
    )
    return img_bgr, polys


def extract_land_polygons_from_mask_png(
    mask_land_png_path: str,
    min_area: float = 2000.0,
    eps_factor: float = 0.004,
    close_ksize: int = 9,
    close_iter: int = 2,
    open_ksize: int = 5,
    open_iter: int = 1,
):
    """
    陸地だけは mask_land.png（2値）から直接ポリゴン抽出する。
    """
    mask = cv2.imread(mask_land_png_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read: {mask_land_png_path}")

    # 念のため2値化
    _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    polys = _extract_polygons_from_mask(
        mask_u8=mask_bin,
        min_area=min_area,
        eps_factor=eps_factor,
        close_ksize=close_ksize,
        close_iter=close_iter,
        open_ksize=open_ksize,
        open_iter=open_iter,
    )
    return polys


def extract_polygons_from_mask_png(
    mask_png_path: str,
    min_area: float = 200.0,
    eps_factor: float = 0.004,
    close_ksize: int = 9,
    close_iter: int = 2,
    open_ksize: int = 3,
    open_iter: int = 1,
    # for thin lines (magenta)
    dilate_ksize: int = 7,
    dilate_iter: int = 1,
) -> List[np.ndarray]:
    """
    任意の2値マスクpng(0/255)からポリゴン抽出
    - マゼンタ線など細いものは dilate で太らせてから輪郭抽出すると安定する
    """
    mask = cv2.imread(mask_png_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read: {mask_png_path}")

    _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # 線が細い場合の保険（太らせる）
    if dilate_ksize and dilate_ksize > 1 and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        mask_bin = cv2.dilate(mask_bin, k, iterations=int(dilate_iter))

    polys = _extract_polygons_from_mask(
        mask_u8=mask_bin,
        min_area=min_area,
        eps_factor=eps_factor,
        close_ksize=close_ksize,
        close_iter=close_iter,
        open_ksize=open_ksize,
        open_iter=open_iter,
    )
    return polys

# =========================================================
# (G) Pixel -> LatLon mapping (grid-based, with extrapolation)
# =========================================================
def read_grid_params_from_csv(grid_info_csv: str) -> Dict[str, float]:
    df = pd.read_csv(grid_info_csv)
    required = ["lat0", "lon0", "dlat", "dlon", "lat_sign", "lon_sign", "lat_anchor_idx", "lon_anchor_idx"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"grid_info_csv missing columns: {missing}")

    r0 = df.iloc[0]
    return {
        "lat0": float(r0["lat0"]),
        "lon0": float(r0["lon0"]),
        "dlat": float(r0["dlat"]),
        "dlon": float(r0["dlon"]),
        "lat_sign": int(r0["lat_sign"]),
        "lon_sign": int(r0["lon_sign"]),
        "lat_anchor_idx": int(r0["lat_anchor_idx"]),
        "lon_anchor_idx": int(r0["lon_anchor_idx"]),
    }


def interp_extrap(x, xp, fp):
    xp = np.asarray(xp, dtype=float).ravel()
    fp = np.asarray(fp, dtype=float).ravel()
    if xp.size != fp.size or xp.size < 2:
        raise ValueError("xp and fp must have same length >= 2")

    x_arr = np.asarray(x, dtype=float)
    y = np.empty_like(x_arr, dtype=float)

    left = x_arr <= xp[0]
    right = x_arr >= xp[-1]
    mid = (~left) & (~right)

    sL = (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[left] = fp[0] + sL * (x_arr[left] - xp[0])

    sR = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y[right] = fp[-1] + sR * (x_arr[right] - xp[-1])

    if np.any(mid):
        y[mid] = np.interp(x_arr[mid], xp, fp)

    if np.isscalar(x):
        return float(y.item())
    return y


def mercator_m(lat_deg):
    lat_rad = np.deg2rad(lat_deg)
    return np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))


def inv_mercator_lat(m):
    lat_rad = 2.0 * np.arctan(np.exp(m)) - (np.pi / 2.0)
    return np.rad2deg(lat_rad)


def build_pixel_to_latlon_from_grid(
    lon_xs_px: np.ndarray,
    lat_ys_px: np.ndarray,
    grid_params: dict,
    lat_model: str = "mercator",
) -> Tuple[Callable[[float, float], Tuple[float, float]], dict]:
    lon_xs_px = np.asarray(lon_xs_px, float).ravel()
    lat_ys_px = np.asarray(lat_ys_px, float).ravel()

    n_lon = lon_xs_px.size
    n_lat = lat_ys_px.size

    lon0 = float(grid_params["lon0"])
    lat0 = float(grid_params["lat0"])
    dlon = float(grid_params["dlon"])
    dlat = float(grid_params["dlat"])
    lon_sign = int(grid_params["lon_sign"])
    lat_sign = int(grid_params["lat_sign"])
    lon_anchor_idx = int(grid_params["lon_anchor_idx"])
    lat_anchor_idx = int(grid_params["lat_anchor_idx"])

    lon_values = lon0 + lon_sign * (np.arange(n_lon) - lon_anchor_idx) * dlon
    lat_values = lat0 + lat_sign * (np.arange(n_lat) - lat_anchor_idx) * dlat

    if lat_model == "linear":

        def pixel_to_latlon(x, y):
            lon = interp_extrap(x, lon_xs_px, lon_values)
            lat = interp_extrap(y, lat_ys_px, lat_values)
            return float(lat), float(lon)

        model = {
            "kind": "grid_based_pixel_to_latlon",
            "lat_model": "linear",
            "lon_xs_px": [float(v) for v in lon_xs_px],
            "lat_ys_px": [float(v) for v in lat_ys_px],
            "lon_values_deg": [float(v) for v in lon_values],
            "lat_values_deg": [float(v) for v in lat_values],
            "grid_params": grid_params,
        }
        return pixel_to_latlon, model

    if lat_model == "mercator":
        m_values = mercator_m(lat_values)

        def pixel_to_latlon(x, y):
            lon = interp_extrap(x, lon_xs_px, lon_values)
            m = interp_extrap(y, lat_ys_px, m_values)
            lat = inv_mercator_lat(m)
            return float(lat), float(lon)

        model = {
            "kind": "grid_based_pixel_to_latlon",
            "lat_model": "mercator",
            "lon_xs_px": [float(v) for v in lon_xs_px],
            "lat_ys_px": [float(v) for v in lat_ys_px],
            "lon_values_deg": [float(v) for v in lon_values],
            "lat_values_deg": [float(v) for v in lat_values],
            "grid_params": grid_params,
        }
        return pixel_to_latlon, model

    raise ValueError("lat_model must be 'linear' or 'mercator'")


# =========================================================
# (H) Save overlay / geojson / vertices(latlon) / model.json
# =========================================================
def save_outputs(
    base_image_bgr,
    polys,
    pixel_to_latlon_func,
    model_dict,
    out_dir: str,
    overlay_name: str,
    geojson_name: str,
    vertices_csv_name: str,
    model_json_name: str = "pixel_to_latlon_model.json",
):
    os.makedirs(out_dir, exist_ok=True)

    # overlay
    overlay = base_image_bgr.copy()
    for poly in polys:
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=3)
    overlay_path = os.path.join(out_dir, overlay_name)
    cv2.imwrite(overlay_path, overlay)

    # geojson (px)
    geojson_path = os.path.join(out_dir, geojson_name)
    features = []
    for pid, poly in enumerate(polys):
        coords = poly.tolist()
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "properties": {"polygon_id": pid, "kind": os.path.splitext(geojson_name)[0]},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

    # vertices latlon csv
    rows = []
    for pid, poly in enumerate(polys):
        for vid, (x, y) in enumerate(poly):
            lat, lon = pixel_to_latlon_func(float(x), float(y))
            rows.append({"polygon_id": pid, "vertex_id": vid, "x_px": int(x), "y_px": int(y), "lat": lat, "lon": lon})

    vdf = pd.DataFrame(rows)
    vertices_csv_path = os.path.join(out_dir, vertices_csv_name)
    vdf.to_csv(vertices_csv_path, index=False, encoding="utf-8-sig")

    # model json
    model_json_path = os.path.join(out_dir, model_json_name)
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(model_dict, f, ensure_ascii=False, indent=2)

    return overlay_path, geojson_path, vertices_csv_path, model_json_path


# =========================================================
# (E+All) Unified pipeline:
#   raw image -> detect green grid -> build impassable_map
#   -> save tiles_manifest (geo meta)
#   -> extract polygons (impassable + land-only) -> pixel->latlon -> export
# =========================================================
def process_all_and_export_outline(
    img_path: str,
    grid_roi=(90, 80, None, None),
    border=2,
    # manual inputs (deg/min) + anchor indices
    lat0_deg=34,
    lat0_min=57.0,
    lat_anchor_idx=0,
    lon0_deg=136,
    lon0_min=38.0,
    lon_anchor_idx=0,
    # mapping model
    lat_model="mercator",  # "mercator" or "linear"
    # polygon params
    poly_min_area=2000.0,
    poly_eps_factor=0.002,
):
    save_dir = make_save_dir(img_path)
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {img_path}")

    # 1) green grid detect
    ys, xs = detect_latlon_grid_lines(
        img,
        out_dir=save_dir,
        roi=grid_roi,
        hsv_h_range=(45, 95),
        s_min=80,
        v_min=40,
        v_max=200,
        peak_ratio=0.50,
        merge_gap=3,
    )
    print("Detected ys (lat lines px):", ys)
    print("Detected xs (lon lines px):", xs)

    # 2) land/shallow/magenta masks
    mask_land, mask_shallow, mask_magenta = extract_land_shallow_magenta_masks(img)
    mask_land_path = os.path.join(save_dir, "mask_land.png")
    mask_shallow_path = os.path.join(save_dir, "mask_shallow.png")
    mask_magenta_path = os.path.join(save_dir, "mask_magenta.png")
    cv2.imwrite(mask_land_path, mask_land)
    cv2.imwrite(mask_shallow_path, mask_shallow)
    cv2.imwrite(mask_magenta_path, mask_magenta)

    # 3) build impassable_map (for visualization / impassable polygon threshold)
    final_map = np.full_like(img, 255)  # white background
    LAND_GRAY = (220, 220, 220)         # BGR
    SHALLOW_BGR = (233, 224, 167)       # RGB(167,224,233) -> BGR
    MAGENTA_BGR = (186, 71, 198)        # RGB(198,71,186)  -> BGR

    final_map[mask_land > 0] = LAND_GRAY
    final_map[mask_shallow > 0] = SHALLOW_BGR
    final_map[mask_magenta > 0] = MAGENTA_BGR

    impassable_map_path = os.path.join(save_dir, "impassable_map.png")
    cv2.imwrite(impassable_map_path, final_map)
    print(f"[OK] Saved: {impassable_map_path}")

    # 4) save grid-index image (draw on final_map for later visual check)
    grid_vis = final_map.copy()
    H, W = grid_vis.shape[:2]
    for y in ys:
        cv2.line(grid_vis, (0, int(y)), (W - 1, int(y)), (0, 0, 255), 2)
    for x in xs:
        cv2.line(grid_vis, (int(x), 0), (int(x), H - 1), (0, 0, 255), 2)

    tiles_dir = os.path.join(save_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    grid_image_path = os.path.join(tiles_dir, "_grid_detected_indexed.png")
    cv2.imwrite(grid_image_path, grid_vis)

    # 5) tiling + tiles_manifest.csv (geo meta included)
    grid_info_csv = os.path.join(tiles_dir, "tiles_manifest.csv")
    if len(ys) >= 2 and len(xs) >= 2:
        split_by_lines_no_title_with_geo(
            final_map,
            ys,
            xs,
            tiles_dir,
            lat0_deg=lat0_deg,
            lat0_min=lat0_min,
            lat_anchor_idx=lat_anchor_idx,
            lon0_deg=lon0_deg,
            lon0_min=lon0_min,
            lon_anchor_idx=lon_anchor_idx,
            border=border,
        )
    else:
        raise ValueError("[ERROR] grid lines not detected sufficiently (need >=2 each).")

    # 6-A) impassable polygons from impassable_map.png
    base_img_bgr, polys_impassable = extract_impassable_polygons(
        image_path=impassable_map_path,
        eps_factor=poly_eps_factor,
        min_area=float(poly_min_area),
    )
    print(f"[OK] impassable polygons: {len(polys_impassable)}")

    # 6-B) land-only polygons directly from mask_land.png  ★ここが変更点
    polys_land_only = extract_land_polygons_from_mask_png(
        mask_land_png_path=mask_land_path,
        eps_factor=poly_eps_factor,
        min_area=float(poly_min_area),
    )
    print(f"[OK] land-only polygons (from mask_land.png): {len(polys_land_only)}")

    # 6-C) magenta polygons directly from mask_magenta.png
    polys_magenta = extract_polygons_from_mask_png(
        mask_png_path=mask_magenta_path,
        min_area=200.0,          # 線なので小さめ
        eps_factor=poly_eps_factor,
        close_ksize=7,
        close_iter=1,
        open_ksize=3,
        open_iter=1,
        dilate_ksize=7,          # 線を少し太らせる
        dilate_iter=1,
    )
    print(f"[OK] magenta polygons (from mask_magenta.png): {len(polys_magenta)}")

    # 7) pixel->latlon mapping (xs/ys from GREEN grid detection)
    grid_params = read_grid_params_from_csv(grid_info_csv)
    pixel_to_latlon_func, model_dict = build_pixel_to_latlon_from_grid(
        lon_xs_px=np.array(xs, dtype=float),
        lat_ys_px=np.array(ys, dtype=float),
        grid_params=grid_params,
        lat_model=str(lat_model),
    )

    base_model_dict = {
        **model_dict,
        "img_path": img_path,
        "impassable_map_path": impassable_map_path,
        "grid_image_path": grid_image_path,
        "grid_info_csv": grid_info_csv,
        "mask_land_path": mask_land_path,
        "mask_shallow_path": mask_shallow_path,
        "mask_magenta_path": mask_magenta_path,
        "note": "lon_xs_px/lat_ys_px are from GREEN grid detection.",
    }

    # 8-A) export: impassable outline (gray + shallow)
    out_imp = os.path.join(save_dir, "outline_impassable")
    imp_overlay, imp_geojson, imp_vertices, imp_model = save_outputs(
        base_image_bgr=base_img_bgr,
        polys=polys_impassable,
        pixel_to_latlon_func=pixel_to_latlon_func,
        model_dict={**base_model_dict, "outline_kind": "impassable(gray+shallow)"},
        out_dir=out_imp,
        overlay_name="impassable_outline.png",
        geojson_name="impassable_outline_px.geojson",
        vertices_csv_name="impassable_outline_vertices_latlon.csv",
        model_json_name="pixel_to_latlon_model.json",
    )

    # 8-B) export: land-only outline (from mask_land.png)
    out_land = os.path.join(save_dir, "outline_land_only")
    land_overlay, land_geojson, land_vertices, land_model = save_outputs(
        base_image_bgr=base_img_bgr,
        polys=polys_land_only,
        pixel_to_latlon_func=pixel_to_latlon_func,
        model_dict={**base_model_dict, "outline_kind": "land_only(from mask_land.png)"},
        out_dir=out_land,
        overlay_name="land_only_outline.png",
        geojson_name="land_only_outline_px.geojson",
        vertices_csv_name="land_only_outline_vertices_latlon.csv",
        model_json_name="pixel_to_latlon_model.json",
    )

    # 8-C) export: magenta outline (from mask_magenta.png)
    out_mag = os.path.join(save_dir, "outline_magenta")
    mag_overlay, mag_geojson, mag_vertices, mag_model = save_outputs(
        base_image_bgr=base_img_bgr,
        polys=polys_magenta,
        pixel_to_latlon_func=pixel_to_latlon_func,
        model_dict={**base_model_dict, "outline_kind": "magenta(from mask_magenta.png)"},
        out_dir=out_mag,
        overlay_name="magenta_outline.png",
        geojson_name="magenta_outline_px.geojson",
        vertices_csv_name="magenta_outline_vertices_latlon.csv",
        model_json_name="pixel_to_latlon_model.json",
    )

    print("---- saved ----")
    print("grid image        :", grid_image_path)
    print("manifest          :", grid_info_csv)
    print("mask_land         :", mask_land_path)
    print("[impassable] dir   :", out_imp)
    print("[land-only] dir    :", out_land)
    print("[magenta] dir     :", out_mag)

    return save_dir


# =========================================================
# Run example
# =========================================================
if __name__ == "__main__":
    IMG_PATH = "raw_datas/海岸線データ/Yokkaichi.PNG"

    process_all_and_export_outline(
        IMG_PATH,
        grid_roi=(90, 80, None, None),
        border=2,
        # manual input (deg/min + anchor indices)
        lat0_deg=34,
        lat0_min=57.0,
        lat_anchor_idx=0,
        lon0_deg=136,
        lon0_min=38.0,
        lon_anchor_idx=0,
        # mapping model
        lat_model="mercator",  # or "linear"
        # polygon extraction tuning
        poly_min_area=2000.0,
        poly_eps_factor=0.002,
    )