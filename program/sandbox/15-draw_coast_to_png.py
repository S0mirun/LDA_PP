import os
import json
from typing import Optional, Callable, Tuple, Dict

import cv2
import numpy as np
import pandas as pd


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)


def extract_impassable_polygons(
    image_path: str,
    min_area: float = 2000.0,
    eps_factor: float = 0.004,
    close_ksize: int = 9,
    close_iter: int = 2,
    open_ksize: int = 5,
    open_iter: int = 1,
):
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

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=open_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        eps = eps_factor * peri
        approx = cv2.approxPolyDP(c, eps, True)
        polys.append(approx.reshape(-1, 2).astype(int))

    return img_bgr, polys


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


def _smooth_1d(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    kernel = np.ones(int(k), dtype=float) / float(k)
    return np.convolve(a.astype(float), kernel, mode="same")


def _segments_over_threshold(v: np.ndarray, thr: float):
    idx = np.where(v > thr)[0]
    if idx.size == 0:
        return []
    segs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        segs.append((start, prev + 1))
        start = i
        prev = i
    segs.append((start, prev + 1))
    return segs


def detect_red_grid_lines(
    grid_image_path: str,
    smooth_k: int = 21,
    thr_frac: float = 0.35,
    min_seg_width: int = 3,
    debug_out_path: Optional[str] = None,
):
    img = cv2.imread(grid_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {grid_image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
    mask = cv2.bitwise_or(red1, red2)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, k, iterations=1)

    proj_x = mask.sum(axis=0).astype(float)
    proj_y = mask.sum(axis=1).astype(float)

    proj_xs = _smooth_1d(proj_x, smooth_k)
    proj_ys = _smooth_1d(proj_y, smooth_k)

    if proj_xs.max() <= 0 or proj_ys.max() <= 0:
        raise ValueError("No red pixels detected. Check that grid lines are red.")

    thr_x = thr_frac * float(proj_xs.max())
    thr_y = thr_frac * float(proj_ys.max())

    segs_x = [(l, r) for (l, r) in _segments_over_threshold(proj_xs, thr_x) if (r - l) >= min_seg_width]
    segs_y = [(l, r) for (l, r) in _segments_over_threshold(proj_ys, thr_y) if (r - l) >= min_seg_width]

    xs = []
    for l, r in segs_x:
        w = proj_x[l:r]
        xs.append(((l + r - 1) * 0.5) if w.sum() <= 0 else (np.arange(l, r) * w).sum() / w.sum())

    ys = []
    for l, r in segs_y:
        w = proj_y[l:r]
        ys.append(((l + r - 1) * 0.5) if w.sum() <= 0 else (np.arange(l, r) * w).sum() / w.sum())

    xs = np.array(sorted(xs), dtype=float)
    ys = np.array(sorted(ys), dtype=float)

    if xs.size < 2 or ys.size < 2:
        raise ValueError(
            f"Detected too few grid lines (lon={xs.size}, lat={ys.size}). "
            f"Try lowering thr_frac or ensure grid lines are clearly red."
        )

    if debug_out_path is not None:
        dbg = img.copy()
        H, W = dbg.shape[:2]
        for x in xs:
            xx = int(round(x))
            cv2.line(dbg, (xx, 0), (xx, H - 1), (0, 255, 255), 2)
        for y in ys:
            yy = int(round(y))
            cv2.line(dbg, (0, yy), (W - 1, yy), (0, 255, 255), 2)
        cv2.imwrite(debug_out_path, dbg)

    return xs, ys


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


def save_outputs(
    base_image_bgr,
    polys,
    pixel_to_latlon_func,
    model_dict,
    out_dir: str,
    overlay_name: str = "impassable_outline.png",
    geojson_name: str = "impassable_outline_px.geojson",
    vertices_csv_name: str = "impassable_outline_vertices_latlon.csv",
    model_json_name: str = "pixel_to_latlon_model.json",
):
    os.makedirs(out_dir, exist_ok=True)

    overlay = base_image_bgr.copy()
    for poly in polys:
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=3)
    overlay_path = os.path.join(out_dir, overlay_name)
    cv2.imwrite(overlay_path, overlay)

    geojson_path = os.path.join(out_dir, geojson_name)
    features = []
    for pid, poly in enumerate(polys):
        coords = poly.tolist()
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "properties": {"polygon_id": pid, "kind": "impassable_outline_px"},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

    rows = []
    for pid, poly in enumerate(polys):
        for vid, (x, y) in enumerate(poly):
            lat, lon = pixel_to_latlon_func(float(x), float(y))
            rows.append({"polygon_id": pid, "vertex_id": vid, "x_px": int(x), "y_px": int(y), "lat": lat, "lon": lon})

    vdf = pd.DataFrame(rows)
    vertices_csv_path = os.path.join(out_dir, vertices_csv_name)
    vdf.to_csv(vertices_csv_path, index=False, encoding="utf-8-sig")

    model_json_path = os.path.join(out_dir, model_json_name)
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(model_dict, f, ensure_ascii=False, indent=2)

    return overlay_path, geojson_path, vertices_csv_path, model_json_path


if __name__ == "__main__":
    in_path = "outputs/14-separate_png/Yokkaichi/tiles/_grid_detected_indexed.png"
    grid_image_path = in_path
    grid_info_csv = f"{os.path.dirname(in_path)}/tiles_manifest.csv"

    port_name = os.path.basename(os.path.dirname(os.path.dirname(in_path)))
    port_save_dir = os.path.join(SAVE_DIR, port_name)
    os.makedirs(port_save_dir, exist_ok=True)

    base_img_bgr, polys = extract_impassable_polygons(
        image_path=in_path,
        eps_factor=0.002,
        min_area=2000,
    )

    debug_grid_path = os.path.join(port_save_dir, "debug_detected_grid.png")
    lon_xs_px, lat_ys_px = detect_red_grid_lines(
        grid_image_path=grid_image_path,
        smooth_k=21,
        thr_frac=0.35,
        min_seg_width=3,
        debug_out_path=debug_grid_path,
    )

    grid_params = read_grid_params_from_csv(grid_info_csv)

    pixel_to_latlon_func, model_dict = build_pixel_to_latlon_from_grid(
        lon_xs_px=lon_xs_px,
        lat_ys_px=lat_ys_px,
        grid_params=grid_params,
        lat_model="mercator",
    )

    overlay_path, geojson_path, vertices_csv_path, model_json_path = save_outputs(
        base_image_bgr=base_img_bgr,
        polys=polys,
        pixel_to_latlon_func=pixel_to_latlon_func,
        model_dict={**model_dict, "grid_image_path": grid_image_path, "grid_info_csv": grid_info_csv},
        out_dir=port_save_dir,
    )

    print(f"port_name: {port_name}")
    print(f"polygons: {len(polys)}")
    print(f"detected lon lines: {len(lon_xs_px)}")
    print(f"detected lat lines: {len(lat_ys_px)}")
    print("saved overlay :", overlay_path)
    print("saved geojson  :", geojson_path)
    print("saved vertices :", vertices_csv_path)
    print("saved model    :", model_json_path)
    print("saved debug grid:", debug_grid_path)
