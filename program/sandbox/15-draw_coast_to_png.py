import os
import json
import cv2
import numpy as np
import pandas as pd


# =============================
# save dirs (as you requested)
# =============================
DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS_DIR = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)


def extract_impassable_polygons(
    image_path: str,
    min_area: float = 2000.0,
    eps_factor: float = 0.004,   # 粗さ：大きいほど粗い
    close_ksize: int = 9,
    close_iter: int = 2,
    open_ksize: int = 5,
    open_iter: int = 1,
):
    """灰色＋シアンを通行不可として外形ポリゴン(ピクセル座標)を抽出"""
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # シアン
    cyan_lo = np.array([75,  35, 120], dtype=np.uint8)
    cyan_hi = np.array([110, 255, 255], dtype=np.uint8)
    mask_cyan = cv2.inRange(hsv, cyan_lo, cyan_hi)

    # 灰色（低彩度・中明度）
    gray_lo = np.array([0,   0,  70], dtype=np.uint8)
    gray_hi = np.array([180, 35, 245], dtype=np.uint8)
    mask_gray = cv2.inRange(hsv, gray_lo, gray_hi)

    mask = cv2.bitwise_or(mask_cyan, mask_gray)

    # マゼンタ枠が混ざる場合の保険
    mag_lo = np.array([135, 60, 60], dtype=np.uint8)
    mag_hi = np.array([175, 255, 255], dtype=np.uint8)
    mask_mag = cv2.inRange(hsv, mag_lo, mag_hi)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_mag))

    # ギャップ埋め＆ノイズ除去
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize,  open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=open_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        eps = eps_factor * peri
        approx = cv2.approxPolyDP(c, eps, True)
        polys.append(approx.reshape(-1, 2).astype(int))

    return img_bgr, polys


def build_manifest_index(manifest_csv: str):
    """
    tiles_manifest.csv を読み込み、点(x,y)→該当タイル→lat/lon線形補間に使う配列を作る
    必要列: x0,x1,y0,y1, lat_min,lat_max, lon_min,lon_max
    """
    df = pd.read_csv(manifest_csv)

    required = ["x0", "x1", "y0", "y1", "lat_min", "lat_max", "lon_min", "lon_max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"tiles_manifest.csv missing columns: {missing}")

    # numpy 化（高速化）
    x0 = df["x0"].to_numpy(dtype=float)
    x1 = df["x1"].to_numpy(dtype=float)
    y0 = df["y0"].to_numpy(dtype=float)
    y1 = df["y1"].to_numpy(dtype=float)
    lat_min = df["lat_min"].to_numpy(dtype=float)
    lat_max = df["lat_max"].to_numpy(dtype=float)
    lon_min = df["lon_min"].to_numpy(dtype=float)
    lon_max = df["lon_max"].to_numpy(dtype=float)

    # 上下左右を明示（latは上が大きいとは限らないのでmax/minで吸収）
    lat_top = np.maximum(lat_min, lat_max)
    lat_bot = np.minimum(lat_min, lat_max)
    lon_left = np.minimum(lon_min, lon_max)
    lon_right = np.maximum(lon_min, lon_max)

    return {
        "df": df,
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "lat_top": lat_top, "lat_bot": lat_bot,
        "lon_left": lon_left, "lon_right": lon_right,
    }


def pixel_to_latlon(x: float, y: float, M):
    """
    manifest から (x,y) を含むタイルを探し、線形補間で (lat,lon) を返す
    見つからなければ (nan,nan)
    """
    # 該当タイル探索（包含）
    hit = (M["x0"] <= x) & (x <= M["x1"]) & (M["y0"] <= y) & (y <= M["y1"])
    idxs = np.where(hit)[0]
    if idxs.size == 0:
        return float("nan"), float("nan")

    i = int(idxs[0])

    # タイル内相対位置 u,v（0..1）
    dx = (M["x1"][i] - M["x0"][i])
    dy = (M["y1"][i] - M["y0"][i])
    if dx <= 0 or dy <= 0:
        return float("nan"), float("nan")

    u = (x - M["x0"][i]) / dx
    v = (y - M["y0"][i]) / dy

    # 線形補間：yが下に行くほど v↑、latは top→bottom に変化
    lon = M["lon_left"][i] + u * (M["lon_right"][i] - M["lon_left"][i])
    lat = M["lat_top"][i]  + v * (M["lat_bot"][i]   - M["lat_top"][i])

    return float(lat), float(lon)


def save_outputs(
    img_bgr,
    polys,
    manifest_csv: str,
    out_dir: str,
    overlay_name: str = "impassable_outline.png",
    geojson_name: str = "impassable_outline_px.geojson",
    vertices_csv_name: str = "impassable_outline_vertices_latlon.csv",
):
    os.makedirs(out_dir, exist_ok=True)

    # overlay
    overlay = img_bgr.copy()
    for poly in polys:
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=3)
    overlay_path = os.path.join(out_dir, overlay_name)
    cv2.imwrite(overlay_path, overlay)

    # geojson (pixel polygon)
    geojson_path = os.path.join(out_dir, geojson_name)
    features = []
    for pid, poly in enumerate(polys):
        coords = poly.tolist()
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "properties": {"polygon_id": pid, "kind": "impassable_outline_px"},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

    # vertices csv (pixel -> lat/lon)
    M = build_manifest_index(manifest_csv)

    rows = []
    for pid, poly in enumerate(polys):
        for vid, (x, y) in enumerate(poly):
            lat, lon = pixel_to_latlon(float(x), float(y), M)
            rows.append({
                "polygon_id": pid,
                "vertex_id": vid,
                "x_px": int(x),
                "y_px": int(y),
                "lat": lat,
                "lon": lon,
            })

    vdf = pd.DataFrame(rows)
    vertices_csv_path = os.path.join(out_dir, vertices_csv_name)
    vdf.to_csv(vertices_csv_path, index=False, encoding="utf-8-sig")

    # warn if any nan
    nan_cnt = int(vdf[["lat", "lon"]].isna().any(axis=1).sum())
    return overlay_path, geojson_path, vertices_csv_path, nan_cnt


if __name__ == "__main__":
    # 入力
    in_path = "/Users/tokudashintaro/Desktop/LDA_PP/outputs/14-separate_png/Yokkaichi/impassable_map.png"
    manifest_csv = f"{os.path.dirname(in_path)}/tiles/tiles_manifest.csv"

    # 保存先：SAVE_DIR/<港名>/
    port_name = os.path.basename(os.path.dirname(in_path))  # "Yokkaichi"
    port_save_dir = os.path.join(SAVE_DIR, port_name)

    img_bgr, polys = extract_impassable_polygons(
        image_path=in_path,
        eps_factor=0.004,
        min_area=2000,
    )

    overlay_path, geojson_path, vertices_csv_path, nan_cnt = save_outputs(
        img_bgr=img_bgr,
        polys=polys,
        manifest_csv=manifest_csv,
        out_dir=port_save_dir,
    )

    print(f"port_name: {port_name}")
    print(f"polygons: {len(polys)}")
    print("saved overlay :", overlay_path)
    print("saved geojson  :", geojson_path)
    print("saved vertices :", vertices_csv_path)
    if nan_cnt > 0:
        print(f"WARNING: {nan_cnt} vertices were outside all tiles (lat/lon = NaN).")