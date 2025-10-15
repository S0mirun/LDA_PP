import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import unicodedata

from utils.LDA.ship_geometry import *
from utils.LDA.visualization import *

DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS_DIR = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
LAT_ORIGIN = 35.00627778
LON_ORIGIN = 136.6740283
ANGLE_FROM_NORTH = 0.0
#
sheet_names = [
    'Passage-3-LNG',
    'Passage-3-South',
    'Passage-2-East',
    'Passage-1'
]
BINS   = [-np.inf, 0, 5, 10, np.inf]
LABELS = ["<0", "0–5", "5–10", ">=10"]
DEPTH_BIN_DTYPE = CategoricalDtype(categories=LABELS, ordered=True)

PALETTE = {  # ビンごとの固定色（必要なら色を変更）
    "<0":   "#1f77b4",
    "0–5":  "#2ca02c",
    "5–10": "#ff7f0e",
    ">=10": "#d62728",
}
"""
Index(['Unnamed: 0', 'Lat\n°', 'Lat\n’', 'Long\n°', 'Long\n’', 'depth\n(m)',
       'Lat', 'Long', 'depth\n(m).1', 'latitude [deg]', 'longitude [deg]',
       'p_x [m]', 'p_y [m]'],
      dtype='object')
"""



def df_to_xy(df):
    lat_idx = df.columns.get_loc("latitude [deg]")
    lon_idx = df.columns.get_loc("longitude [deg]")
    #
    px = np.empty(len(df), dtype=np.float64)
    py = np.empty(len(df), dtype=np.float64)
    #
    for i in range(len(df)):
        y_m, x_m = convert_to_xy(
            float(df.iat[i, lat_idx]), float(df.iat[i, lon_idx]),
            LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
        )
        px[i] = x_m
        py[i] = y_m
    
    return np.column_stack([px, py]).astype(float)

depth_path = f"{RAW_DATAS_DIR}/内航船-要素/水深-Yokkaichi/水深-ブイ等/2-Yokkaichi_waterdepth.xlsx"
raw_df = pd.read_excel(
    depth_path,
    sheet_name=None
)
depth_df = raw_df.copy()
fig, ax = plt.subplots(figsize=(10, 8))

def plot_by_bins(df, ax, xcol="x", ycol="y", valcol="depth"):
    # 2) 同じビンで切る → 同じカテゴリdtypeに統一
    df = df.copy()
    df["depth_bin"] = pd.cut(df[valcol], bins=BINS, labels=LABELS, ordered=True).astype(DEPTH_BIN_DTYPE)

    # 3) 観測されたビンだけ回す（将来の仕様変更対策）
    for name, g in df.groupby("depth_bin", observed=True):
        if g.empty: 
            continue
        color = PALETTE[str(name)]
        ax.scatter(g[xcol], g[ycol], s=12, color=color, edgecolors="none")

for name in sheet_names:
    df = depth_df[name]
    df["latitude [deg]"] = df["Lat"]
    df["longitude [deg]"] = df["Long"]
    conv_df = df_to_xy(df)
    df["p_x [m]"] = conv_df[:, 0]
    df["p_y [m]"] = conv_df[:, 1]
    df.to_csv(os.path.join(SAVE_DIR, f"{name}.csv"))
    # plot
    plot_by_bins(df, ax, xcol='p_x [m]', ycol='p_y [m]', valcol='depth\n(m)')

ax.legend(title="depth (binned)", frameon=False)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal", adjustable="box")
plt.tight_layout()

plt.show()
print("\nDone\n")