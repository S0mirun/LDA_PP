import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

csv_path = glob.glob(f"{DIR}/../../outputs/*/bgr_exact_255_187_121.csv")[0]
df = pd.read_csv(csv_path)
df_port = pd.read_csv(f"{RAW_DATAS}/tmp/coordinates_of_port/_Yokkaichi_port2B.csv")

# OpenCV系のCSVなら b,g,r の順になっている想定 → matplotlib用RGBに並べ替え
rgb = df[["r", "g", "b"]].to_numpy(dtype=np.float32) / 255.0

x = (df["x"]).to_numpy()
y = (df["y"]).to_numpy()
psi = df_port['Psi[deg]'].iloc[0]
# theta = -np.deg2rad(psi)
theta = 0

X = x * np.cos(theta) - y * np.sin(theta)
Y = x * np.sin(theta) + y * np.cos(theta)

plt.figure()
plt.scatter(
    X,
    Y,
    c=rgb,
    s=1,          # 点が多いなら 0.5〜2 あたりで調整
    marker="s",   # 画素っぽくしたいなら "s" が見やすい
    linewidths=0
)
plt.gca().invert_yaxis()   # 画像座標に合わせる（重要）
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Extracted pixels (colored)")
# plt.grid()
plt.savefig(os.path.join(SAVE_DIR, "Yokkaichi.png"))
# plt.show()