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
coast_file = f"{RAW_DATAS}/国土交通省/C23-06_{PORT.num}_GML/C23-06_{PORT.num}-g.csv"
port_file = f"{RAW_DATAS}/tmp/coordinates_of_port/_{PORT.name}.csv"
SAVE_DIR = f"{DIR}/../../outputs/data/detail_map"

SORT = False
SAVE = True
SHOW = False
R_MAX = 3000

# ---- 原点と向き ----
df_coord = pd.read_csv(port_file)
LAT_ORIGIN = df_coord["Latitude"].iloc[0]
LON_ORIGIN = df_coord["Longitude"].iloc[0]
ANGLE_FROM_NORTH = df_coord["Psi[deg]"].iloc[0]

def sort_points(arr: np.ndarray) -> np.ndarray:
    n_points = arr.shape[0]
    start_idx = int(np.argmax(arr[:, 1]))
    open_idx = list(range(n_points))
    open_idx.remove(start_idx)
    closed_idx = [start_idx]

    while open_idx:
        current_pt = arr[closed_idx[-1]]
        candidates = arr[open_idx]
        dists = np.linalg.norm(candidates - current_pt, axis=1)
        next_rel = int(np.argmin(dists))
        next_idx = open_idx[next_rel]
        closed_idx.append(next_idx)
        open_idx.remove(next_idx)

    return arr[closed_idx]

# ---- 海岸線 CSV 読み込み（curve_id, lat, lon を想定）----
df_coast = pd.read_csv(coast_file)

fig, ax = plt.subplots(figsize=(8, 8))

pts_list = []
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
    pts_list.append(pts)

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
        ax.plot(xs, ys, linewidth=0.5)

ax.set_aspect("equal", "box")
ax.set_xlim(-R_MAX, R_MAX)
ax.set_ylim(-R_MAX, R_MAX)
all_pts = np.vstack(pts_list)

if SAVE:
    if SORT:
        # all_pts = np.vstack([all_pts, ADD])
        sorted_pts = sort_points(all_pts)
        df = pd.DataFrame({
            "x [m]": sorted_pts[:, 1],
            "y [m]": sorted_pts[:, 0],
        })
    else:
        df = pd.DataFrame({
            "x [m]": all_pts[:, 1],
            "y [m]": all_pts[:, 0],
        })
    df.to_csv(os.path.join(SAVE_DIR, f"{PORT.name}.csv"))
    print("\nSAVED\n")

if SHOW:
    plt.show()