import os
import cv2
import numpy as np

# =========================================================
# ユーザー指定：保存先ルール
# =========================================================
DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
IMG_PATH = "/Users/tokudashintaro/Desktop/LDA_PP/raw_datas/海岸線データ/堺.PNG"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}/{os.path.splitext(os.path.basename(IMG_PATH))[0]}"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# 画像読み込み
# =========================================================
img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)  # BGR
if img is None:
    raise FileNotFoundError(f"画像が読めません: {IMG_PATH}")

H, W = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 形態学フィルタ用（共通）
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# =========================================================
# Lab 距離用ユーティリティ（int32でオーバーフロー回避）
# =========================================================
def rgb_to_lab_int32(rgb: np.ndarray) -> np.ndarray:
    bgr = rgb[::-1].reshape(1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
    return lab.astype(np.int32)

def dist2_lab_img_to_lab_ref(lab_img_int32: np.ndarray, lab_ref_int32: np.ndarray) -> np.ndarray:
    d = lab_img_int32 - lab_ref_int32[None, None, :]
    return np.sum(d * d, axis=2).astype(np.int32)

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int32)

# =========================================================
# 1) 陸（黄土）マスク抽出（焦茶四角は陸ではないので除外）
# =========================================================
land_lower = np.array([15,  50,  80], dtype=np.uint8)
land_upper = np.array([40, 255, 255], dtype=np.uint8)
mask_land = cv2.inRange(hsv, land_lower, land_upper)  # 0/255

# 焦茶の四角（除外対象）：暗い茶=V低め
brown_excl_lower = np.array([5,  40,   0], dtype=np.uint8)
brown_excl_upper = np.array([35, 255, 150], dtype=np.uint8)
mask_brown_excl = cv2.inRange(hsv, brown_excl_lower, brown_excl_upper)
mask_land[mask_brown_excl > 0] = 0

# 小さい四角っぽい成分除外（保険）
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

# =========================================================
# 2) 浅水（青）マスク抽出：代表色への距離（Lab）で抽出しシアンを除外
# =========================================================
rgb_cyan = np.array([182, 235, 219], dtype=np.uint8)
rgb_s1   = np.array([167, 224, 233], dtype=np.uint8)  # 浅水1（指定色）
rgb_s2   = np.array([129, 195, 226], dtype=np.uint8)

lab_cyan = rgb_to_lab_int32(rgb_cyan)
lab_s1   = rgb_to_lab_int32(rgb_s1)
lab_s2   = rgb_to_lab_int32(rgb_s2)

d_cyan = dist2_lab_img_to_lab_ref(lab_img, lab_cyan)
d_s1   = dist2_lab_img_to_lab_ref(lab_img, lab_s1)
d_s2   = dist2_lab_img_to_lab_ref(lab_img, lab_s2)

nearest = np.argmin(np.stack([d_cyan, d_s1, d_s2], axis=2), axis=2)
mask_shallow = ((nearest == 1) | (nearest == 2))

T_SHALLOW = 22
d_shallow = np.minimum(d_s1, d_s2)
mask_shallow &= (d_shallow <= (T_SHALLOW * T_SHALLOW))

MARGIN = 3
mask_shallow &= (
    np.sqrt(d_shallow.astype(np.float32)) + MARGIN
    <= np.sqrt(d_cyan.astype(np.float32))
)

mask_shallow_u8 = (mask_shallow.astype(np.uint8) * 255)
mask_shallow_u8[mask_land > 0] = 0
mask_shallow_u8 = cv2.morphologyEx(mask_shallow_u8, cv2.MORPH_OPEN,  k_open,  iterations=1)
mask_shallow_u8 = cv2.morphologyEx(mask_shallow_u8, cv2.MORPH_CLOSE, k_close, iterations=1)

# =========================================================
# 3) マゼンタ線マスク抽出（指定RGBへの距離：Lab）
# =========================================================
rgb_magenta = np.array([198, 71, 186], dtype=np.uint8)
lab_mag = rgb_to_lab_int32(rgb_magenta)
d_mag2 = dist2_lab_img_to_lab_ref(lab_img, lab_mag)

T_MAG = 22  # 途切れるなら 26→30、拾いすぎなら 18
mask_magenta = (d_mag2 <= (T_MAG * T_MAG)).astype(np.uint8) * 255

# 細線なので軽く整形
k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_magenta = cv2.morphologyEx(mask_magenta, cv2.MORPH_OPEN, k3, iterations=1)

# =========================================================
# 4) マゼンタで囲われた領域（四角以外も）を全部塗りつぶす
#    - マゼンタ線を close+dilate で「壁」にする（破線を繋ぐ）
#    - 壁以外を connectedComponents
#    - 画像外周に接している成分 = 外側
#    - それ以外 = 内側（全部）→ fill
# =========================================================
def fill_regions_closed_by_magenta_and_border(
    mask_magenta_u8: np.ndarray,
    close_ksize=17, close_iter=2,
    dilate_ksize=9, dilate_iter=1,
    min_area=2000,
    save_debug_dir=None
) -> np.ndarray:
    H, W = mask_magenta_u8.shape

    wall = mask_magenta_u8.copy()

    # 破線を繋ぐ（ここが重要）
    wall = cv2.morphologyEx(
        wall, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize)),
        iterations=close_iter
    )
    wall = cv2.dilate(
        wall,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize)),
        iterations=dilate_iter
    )

    free = (wall == 0).astype(np.uint8)  # 0/1

    # 連結成分
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=8)

    # 外周に接している成分 = outside
    outside_labels = set()

    # 上下端
    outside_labels.update(np.unique(lab[0, :]).tolist())
    outside_labels.update(np.unique(lab[H-1, :]).tolist())
    # 左右端
    outside_labels.update(np.unique(lab[:, 0]).tolist())
    outside_labels.update(np.unique(lab[:, W-1]).tolist())

    fill = np.zeros((H, W), np.uint8)

    for k in range(1, n):
        if k in outside_labels:
            continue
        area = st[k, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        fill[lab == k] = 255

    if save_debug_dir is not None:
        os.makedirs(save_debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_debug_dir, "_wall_after_close_dilate.png"), wall)
        cv2.imwrite(os.path.join(save_debug_dir, "mask_magenta_fill.png"), fill)

    return fill

mask_fill = fill_regions_closed_by_magenta_and_border(
    mask_magenta,
    close_ksize=17, close_iter=2,
    dilate_ksize=9, dilate_iter=1,
    min_area=2000,
    save_debug_dir=SAVE_DIR
)

# =========================================================
# 5) 最終可視化画像（指定配色）＋マゼンタ線＋薄いマゼンタ塗り
#   背景: 白
#   陸  : 薄いグレー
#   浅瀬: RGB=[167,224,233]
#   マゼンタ線: RGB=[198,71,186]
#   塗りつぶし: 薄いマゼンタ
# =========================================================
final_map = np.full_like(img, 255)  # 白背景（BGR）

LAND_GRAY = (220, 220, 220)         # 薄いグレー（BGR）
SHALLOW_BGR = (233, 224, 167)       # RGB(167,224,233) -> BGR
MAGENTA_BGR = (186, 71, 198)        # RGB(198,71,186)  -> BGR

# 薄いマゼンタ（好みで調整）
LIGHT_MAGENTA_RGB = (230, 170, 225)
LIGHT_MAGENTA_BGR = (LIGHT_MAGENTA_RGB[2], LIGHT_MAGENTA_RGB[1], LIGHT_MAGENTA_RGB[0])

# 塗り順（線は最前面）
final_map[mask_land > 0] = LAND_GRAY
final_map[mask_shallow_u8 > 0] = SHALLOW_BGR
final_map[mask_fill > 0] = LIGHT_MAGENTA_BGR
final_map[mask_magenta > 0] = MAGENTA_BGR

# =========================================================
# 6) 保存（最終名は impassable_map.png）
# =========================================================
cv2.imwrite(os.path.join(SAVE_DIR, "mask_land.png"), mask_land)
cv2.imwrite(os.path.join(SAVE_DIR, "mask_shallow.png"), mask_shallow_u8)
cv2.imwrite(os.path.join(SAVE_DIR, "mask_magenta.png"), mask_magenta)
cv2.imwrite(os.path.join(SAVE_DIR, "mask_magenta_fill.png"), mask_fill)

cv2.imwrite(os.path.join(SAVE_DIR, "impassable_map.png"), final_map)

print(f"[OK] IMG_PATH: {IMG_PATH}")
print(f"[OK] SAVE_DIR: {SAVE_DIR}")
print(f"land pixels   : {int((mask_land > 0).sum())}")
print(f"shallow pixels: {int((mask_shallow_u8 > 0).sum())}")
print(f"magenta pixels: {int((mask_magenta > 0).sum())}")
print(f"fill pixels   : {int((mask_fill > 0).sum())}")
print("Saved: impassable_map.png")