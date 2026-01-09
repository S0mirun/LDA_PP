import os
import cv2
import numpy as np
import pandas as pd

# =========================================================
# 保存先ルール
# =========================================================
def make_save_dir(img_path: str):
    try:
        DIR = os.path.dirname(__file__)
        dirname = os.path.splitext(os.path.basename(__file__))[0]
    except NameError:
        DIR = os.getcwd()
        dirname = "merged_script"

    save_dir = f"{DIR}/../../outputs/{dirname}/{os.path.splitext(os.path.basename(img_path))[0]}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# =========================================================
# (A) 緑点線グリッド検出
# =========================================================
def _cluster_centers(idxs, gap=3):
    idxs = np.asarray(idxs)
    if idxs.size == 0:
        return []
    clusters = []
    s = idxs[0]
    p = idxs[0]
    for a in idxs[1:]:
        if a - p > gap:
            clusters.append((s, p))
            s = a
        p = a
    clusters.append((s, p))
    return [int((a + b) // 2) for a, b in clusters]

def detect_latlon_grid_lines(
    img_bgr,
    out_dir=None,
    roi=(90, 80, None, None),
    hsv_h_range=(45, 95),
    s_min=80,
    v_min=40,
    v_max=200,
    peak_ratio=0.50,
    merge_gap=3,
):
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = roi
    if x1 is None: x1 = W
    if y1 is None: y1 = H
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
# (B) Lab距離ユーティリティ
# =========================================================
def rgb_to_lab_int32(rgb: np.ndarray) -> np.ndarray:
    bgr = rgb[::-1].reshape(1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
    return lab.astype(np.int32)

def dist2_lab_img_to_lab_ref(lab_img_int32: np.ndarray, lab_ref_int32: np.ndarray) -> np.ndarray:
    d = lab_img_int32 - lab_ref_int32[None, None, :]
    return np.sum(d * d, axis=2).astype(np.int32)

# =========================================================
# (C) 陸/浅瀬/マゼンタ線マスク抽出（2本目のロジック）
# =========================================================
def extract_land_shallow_magenta_masks(
    img_bgr: np.ndarray,
    land_lower=(15,  50,  80),
    land_upper=(40, 255, 255),
    brown_excl_lower=(5,  40,   0),
    brown_excl_upper=(35, 255, 150),
    rgb_cyan=(182, 235, 219),
    rgb_s1=(167, 224, 233),
    rgb_s2=(129, 195, 226),
    T_SHALLOW=22,
    MARGIN=3,
    rgb_magenta=(198, 71, 186),
    T_MAG=22,
):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int32)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # ---- land ----
    mask_land = cv2.inRange(hsv, np.array(land_lower, np.uint8), np.array(land_upper, np.uint8))

    mask_brown_excl = cv2.inRange(hsv, np.array(brown_excl_lower, np.uint8), np.array(brown_excl_upper, np.uint8))
    mask_land[mask_brown_excl > 0] = 0

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

    mask_land = cv2.morphologyEx(mask_land2, cv2.MORPH_OPEN,  k_open,  iterations=1)
    mask_land = cv2.morphologyEx(mask_land,  cv2.MORPH_CLOSE, k_close, iterations=1)

    # ---- shallow ----
    lab_cyan = rgb_to_lab_int32(np.array(rgb_cyan, np.uint8))
    lab_s1   = rgb_to_lab_int32(np.array(rgb_s1, np.uint8))
    lab_s2   = rgb_to_lab_int32(np.array(rgb_s2, np.uint8))

    d_cyan = dist2_lab_img_to_lab_ref(lab_img, lab_cyan)
    d_s1   = dist2_lab_img_to_lab_ref(lab_img, lab_s1)
    d_s2   = dist2_lab_img_to_lab_ref(lab_img, lab_s2)

    nearest = np.argmin(np.stack([d_cyan, d_s1, d_s2], axis=2), axis=2)
    mask_shallow = ((nearest == 1) | (nearest == 2))

    d_shallow = np.minimum(d_s1, d_s2)
    mask_shallow &= (d_shallow <= (T_SHALLOW * T_SHALLOW))
    mask_shallow &= (np.sqrt(d_shallow.astype(np.float32)) + float(MARGIN) <= np.sqrt(d_cyan.astype(np.float32)))

    mask_shallow_u8 = (mask_shallow.astype(np.uint8) * 255)
    mask_shallow_u8[mask_land > 0] = 0
    mask_shallow_u8 = cv2.morphologyEx(mask_shallow_u8, cv2.MORPH_OPEN,  k_open,  iterations=1)
    mask_shallow_u8 = cv2.morphologyEx(mask_shallow_u8, cv2.MORPH_CLOSE, k_close, iterations=1)

    # ---- magenta line ----
    lab_mag = rgb_to_lab_int32(np.array(rgb_magenta, np.uint8))
    d_mag2 = dist2_lab_img_to_lab_ref(lab_img, lab_mag)
    mask_magenta = (d_mag2 <= (T_MAG * T_MAG)).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_magenta = cv2.morphologyEx(mask_magenta, cv2.MORPH_OPEN, k3, iterations=1)

    return mask_land, mask_shallow_u8, mask_magenta

# =========================================================
# (D) 分割（ラベル焼き込み無し：単純タイル保存）
# =========================================================
def split_by_lines_no_title(img_bgr, ys, xs, out_dir, border=2):
    os.makedirs(out_dir, exist_ok=True)
    H, W = img_bgr.shape[:2]
    ys = sorted(map(int, ys))
    xs = sorted(map(int, xs))

    # デバッグ用：検出線を描いた画像（任意）
    vis = img_bgr.copy()
    for i, y in enumerate(ys):
        cv2.line(vis, (0, y), (W - 1, y), (0, 0, 255), 2)
        cv2.putText(vis, f"lat#{i}", (10, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    for j, x in enumerate(xs):
        cv2.line(vis, (x, 0), (x, H - 1), (0, 0, 255), 2)
        cv2.putText(vis, f"lon#{j}", (max(5, x + 6), 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "_grid_detected_indexed.png"), vis)

    ys_b = [0] + ys + [H - 1]
    xs_b = [0] + xs + [W - 1]

    rows = []
    for i in range(len(ys_b) - 1):
        y0, y1 = ys_b[i], ys_b[i + 1]
        yy0 = max(0, y0 + border)
        yy1 = min(H, y1 - border)
        if yy1 - yy0 < 20:
            continue

        for j in range(len(xs_b) - 1):
            x0, x1 = xs_b[j], xs_b[j + 1]
            xx0 = max(0, x0 + border)
            xx1 = min(W, x1 - border)
            if xx1 - xx0 < 20:
                continue

            tile = img_bgr[yy0:yy1, xx0:xx1].copy()
            fname = f"tile_r{i:02d}_c{j:02d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), tile)

            rows.append({
                "tile": fname,
                "x0": xx0, "x1": xx1, "y0": yy0, "y1": yy1,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "tiles_manifest.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] saved tiles = {len(df)} -> {out_dir}")
    return df

# =========================================================
# (E) 統合パイプライン（塗りつぶし無し／タイルにタイトル無し）
# =========================================================
def process_all(
    img_path: str,
    grid_roi=(90, 80, None, None),
    border=2,
):
    save_dir = make_save_dir(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {img_path}")

    # 1) グリッド線検出
    ys, xs = detect_latlon_grid_lines(
        img, out_dir=save_dir,
        roi=grid_roi,
        hsv_h_range=(45, 95),
        s_min=80, v_min=40, v_max=200,
        peak_ratio=0.50,
        merge_gap=3,
    )
    print("Detected ys:", ys)
    print("Detected xs:", xs)

    # 2) land / shallow / magenta line
    mask_land, mask_shallow, mask_magenta = extract_land_shallow_magenta_masks(img)

    cv2.imwrite(os.path.join(save_dir, "mask_land.png"), mask_land)
    cv2.imwrite(os.path.join(save_dir, "mask_shallow.png"), mask_shallow)
    cv2.imwrite(os.path.join(save_dir, "mask_magenta.png"), mask_magenta)

    # 3) 最終可視化（塗りつぶし無し）
    final_map = np.full_like(img, 255)  # 白背景

    LAND_GRAY = (220, 220, 220)         # 薄いグレー（BGR）
    SHALLOW_BGR = (233, 224, 167)       # RGB(167,224,233) -> BGR
    MAGENTA_BGR = (186, 71, 198)        # RGB(198,71,186)  -> BGR

    final_map[mask_land > 0] = LAND_GRAY
    final_map[mask_shallow > 0] = SHALLOW_BGR
    final_map[mask_magenta > 0] = MAGENTA_BGR

    cv2.imwrite(os.path.join(save_dir, "impassable_map.png"), final_map)
    print(f"[OK] Saved: {os.path.join(save_dir, 'impassable_map.png')}")

    # 4) 分割（タイルにタイトル無し）
    split_dir = os.path.join(save_dir, "tiles")
    if len(ys) >= 2 and len(xs) >= 2:
        split_by_lines_no_title(final_map, ys, xs, split_dir, border=border)
    else:
        print("[WARN] grid lines not detected sufficiently; skip tiling.")

    return save_dir

# =========================================================
# 実行例
# =========================================================
if __name__ == "__main__":
    IMG_PATH = "/Users/tokudashintaro/Desktop/LDA_PP/raw_datas/海岸線データ/清水.PNG"

    process_all(
        IMG_PATH,
        grid_roi=(90, 80, None, None),
        border=2,
    )