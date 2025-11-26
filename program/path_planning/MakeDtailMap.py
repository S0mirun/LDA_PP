"""
detail_mapを作るファイル。
portから半径5000[m]の範囲を抽出する。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon

from utils.LDA.ship_geometry import *
from utils.PP.stay_ports import Hokkaido, Honsyu

DIR = os.path.dirname(__file__)
RAW_DATAS = f"{DIR}/../../raw_datas"

PORT = Honsyu.Pasific.ibaragi
SAVE_DIR = f"{DIR}/../../outputs/data/{PORT.name}"
coast_file = f"{RAW_DATAS}/国土交通省/C23-06_{PORT.num}_GML/C23-06_{PORT.num}-g.csv"
port_file = f"{RAW_DATAS}/tmp/coordinates_of_port/{PORT.name}.csv"

# ---- 原点と向き ----
# LAT_ORIGIN = 41.81017
# LON_ORIGIN = 140.70401
# ANGLE_FROM_NORTH = 0.0
df_coord = pd.read_csv(port_file)
LAT_ORIGIN = df_coord["Latitude"].iloc[0]
LON_ORIGIN = df_coord["Longitude"].iloc[0]
ANGLE_FROM_NORTH = df_coord["Psi[deg]"].iloc[0]

# ---- 海岸線 CSV 読み込み（curve_id, lat, lon を想定）----
df_coast = pd.read_csv(coast_file)

R_MAX = 7500

fig, ax = plt.subplots(figsize=(8, 8))

for curve_id, g in df_coast.groupby("curve_id"):
    xs = []
    ys = []

    # lat, lon -> x, y に変換しつつ、範囲内だけ拾う
    for lat, lon in zip(g["lat"], g["lon"]):
        y, x = convert_to_xy(
            lat,
            lon,
            LAT_ORIGIN,
            LON_ORIGIN,
            ANGLE_FROM_NORTH,
        )

        if abs(x) < R_MAX and abs(y) < R_MAX:
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        continue

    pts = np.column_stack([xs, ys])

    # 始点と終点が近ければ閉じたポリゴンとみなす
    dist = np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1])
    closed = dist < 50.0  # ここのしきい値は適宜調整

    if closed:
        # 陸地などの「島」は塗りつぶし
        poly = Polygon(
            pts,
            closed=True,
            facecolor="0.8",   # 塗りつぶし色
            edgecolor="0.3",   # 枠線色
            linewidth=0.3,
            alpha=0.9,
        )
        ax.add_patch(poly)
    else:
        # 岸壁などの開いた線は線だけ描く
        ax.plot(xs, ys, color="0.3", linewidth=0.5)

ax.set_aspect("equal", "box")
ax.set_xlim(-R_MAX, R_MAX)
ax.set_ylim(-R_MAX, R_MAX)

plt.show()
