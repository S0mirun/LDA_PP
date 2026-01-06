import glob
import os

import cv2
import numpy as np
import pandas as pd

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS_DIR = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

img_path = glob.glob(f"{DIR}/../../outputs/*/Yokkaichi.png")
img = cv2.imread(img_path[0], cv2.IMREAD_COLOR)  # BGR (H,W,3)
if img is None:
    raise FileNotFoundError("Yokkaichi.png が見つかりません")

target = np.array([255,187,121], np.uint8)
tol = 3  # まずは 2〜5 で調整

lower = np.clip(target.astype(int) - tol, 0, 255).astype(np.uint8)
upper = np.clip(target.astype(int) + tol, 0, 255).astype(np.uint8)

mask255 = cv2.inRange(img, lower, upper)  # 0/255

# 細い白線(抜け)を埋める
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask255 = cv2.morphologyEx(mask255, cv2.MORPH_CLOSE, k, iterations=2)

mask = (mask255 > 0)
ys, xs = np.where(mask)

# 一致画素の色は全て同じなので、b,g,r は固定値でもOK
df = pd.DataFrame({
    "x": xs.astype(int),
    "y": ys.astype(int),
    "b": int(target[0]),
    "g": int(target[1]),
    "r": int(target[2]),
})

df.to_csv(f"{SAVE_DIR}/bgr_exact_255_187_121.csv", index=False)
print(f"saved {len(df)} pixels to bgr_exact_255_187_121.csv")