import os

import cv2
import glob
import numpy as np
from tqdm import tqdm

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS_DIR = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

img_path = glob.glob(f"{RAW_DATAS_DIR}/海岸線データ/四日市20251007/四日市コンビナート_0.75_10.PNG")[0]
# 画像を読み込む
img = cv2.imread(img_path)
height = img.shape[0]
width = img.shape[1]

output = np.zeros((height, width, 1), np.uint8)

target_bgr = np.array([255, 250, 220], dtype=np.uint8)
alpha_new = 50

mask = np.all(img == target_bgr, axis=2)

# 色変更
out = img.copy().astype(np.int16)
out[mask, 1] = (out[mask, 1] * 0.75).astype(np.int16)
out[mask, 2] = (out[mask, 2] * 0.55).astype(np.int16)
out = np.clip(out, 0, 255).astype(np.uint8)

output_show = cv2.resize(
    out,
    (int(width * 0.5), int(height * 0.5)),
    interpolation=cv2.INTER_AREA  # 縮小向き
)

# 読み込んだ画像を表示する
save_path_show = os.path.join(SAVE_DIR, "Yokkaichi.png")
cv2.imwrite(save_path_show, output)
cv2.imshow('imshow_test', output_show)
cv2.waitKey(0)
cv2.destroyAllWindows()