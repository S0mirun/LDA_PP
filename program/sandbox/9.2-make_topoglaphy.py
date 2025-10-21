import glob
import os
import re
import unicodedata

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from utils.LDA.ship_geometry import *
from utils.LDA.visualization import *



DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS_DIR = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
coast_path = f"{RAW_DATAS_DIR}/海岸線データ/四日市港 海岸線データ(国土地理院地図から抽出).csv"
depth_path = f"{RAW_DATAS_DIR}/内航船-要素/水深-Yokkaichi/水深-ブイ等/2-Yokkaichi_waterdepth.xlsx"
#
TOP_HEADER = [
    "date (JST)", "time (JST)", "latitude [deg]",
    "longitude [deg]", "GPS deg [deg]", "gyro deg [deg]",
    "GPS speed [knot]", "log speed [knot]", "wind dir [deg]", "wind sped [knot]"
]
SHEET_NAMES = [
    'Passage-3-LNG',
    'Passage-3-South',
    'Passage-2-East',
    'Passage-1'
]
#
LAT_ORIGIN = 35.00627778
LON_ORIGIN = 136.6740283
ANGLE_FROM_NORTH = 0.0

def prepare(coast_path):
    #
    raw_coast_df = pd.read_csv(
        coast_path,
        encoding="shift-jis"
    )
    coast_df = pd.DataFrame({"latitude [deg]": raw_coast_df.iloc[:, 0], "longitude [deg]": raw_coast_df.iloc[:, 1]})

    return coast_df

def convert_coordinate(value):
    s = unicodedata.normalize("NFKC", str(value)).strip()
    s = s.replace("’", "'").replace("′", "'").replace("”", '"').replace("″", '"')
    m = re.match(r'^([+-]?\d+(?:\.\d+)?)(?:[°\s]*?(\d+(?:\.\d+)?))?(?:[\'\s]*?(\d+(?:\.\d+)?)(?:"|″)?)?\s*([NnSsEeWw])?$', s)
    if not m:
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        if not nums:
            return float("nan")
        deg = float(nums[0])
        if len(nums) >= 2:
            deg += float(nums[1]) / 60.0
        if len(nums) >= 3:
            deg += float(nums[2]) / 3600.0
        return deg
    #
    deg = float(m.group(1))
    mi = float(m.group(2)) if m.group(2) else 0.0
    se = float(m.group(3)) if m.group(3) else 0.0
    hem = (m.group(4) or "").upper()
    val = deg + mi / 60.0 + se / 3600.0
    #
    if hem in ("S", "W"):
        val = -abs(val)
    elif hem in ("N", "E"):
        val = abs(val)

    return val

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

def maybe_add_extra(coords, use_flag=True, x_const=-6000.0):
    if not use_flag:
        return coords
    #
    y_min = float(coords[:, 1].min())
    y_max = float(coords[:, 1].max())
    extra = np.array(
        [
            [x_const, y_min], [x_const, y_max],
            [float(coords[0, 0]), float(coords[0, 1])]
        ]
        ,dtype=float
    )
    
    return np.vstack([coords, extra])

def JST_str_to_float(str):
    l = str.split(":")
    t = float(l[0]) * 3600.0 + float(l[1]) * 60.0 + float(l[2])
    return t

def draw_base_map(ax, coast_df, apply_coast_extra=True, x_const=-6000.0):
    coords_coast = df_to_xy(coast_df)
    coords_coast = maybe_add_extra(coords_coast, apply_coast_extra, x_const)
    #
    if not np.allclose(coords_coast[0], coords_coast[-1]):
        coords_coast = np.vstack([coords_coast, coords_coast[0]])
    #
    poly_coast = Polygon(coords_coast, closed=True, facecolor=Colors.black,
                         edgecolor="none", alpha=0.5, linewidth=0, zorder=1)
    #
    ax.add_patch(poly_coast)
    ax.set_xlim(-4500, 1500)
    y_min = float(coords_coast[:, 1].min())
    ax.set_ylim(y_min, -3000)
    ax.set_aspect("equal")
    x_ticks = np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1000, 1000)
    y_ticks = np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + 1000, 1000)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)



def draw_waterdepth(ax):
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
    #
    raw_df = pd.read_excel(
        depth_path,
        sheet_name=None
    )
    depth_df = raw_df.copy()
    #
    BINS   = [-np.inf, 0, 5, 10, np.inf]
    LABELS = ["<0", "0–5", "5–10", ">=10"]
    DEPTH_BIN_DTYPE = CategoricalDtype(categories=LABELS, ordered=True)

    PALETTE = {  # ビンごとの固定色（必要なら色を変更）
        "<0":   "#1f77b4",
        "0–5":  "red",
        "5–10": "orange",
        ">=10": "blue",
    }
    for name in SHEET_NAMES:
        df = depth_df[name]
        df["latitude [deg]"] = df["Lat"]
        df["longitude [deg]"] = df["Long"]
        conv_df = df_to_xy(df)
        df["p_x [m]"] = conv_df[:, 0]
        df["p_y [m]"] = conv_df[:, 1]
        # plot
        plot_by_bins(df, ax, xcol='p_x [m]', ycol='p_y [m]', valcol='depth\n(m)')
        
    handles = [Line2D([0],[0], marker="o", linestyle="", color=PALETTE[l], label=l, markersize=6)
               for l in LABELS]
    ax.legend(handles=handles, title="water depth (bins)", frameon=False)
    

def plot_one_route_and_save(ax, csv_path, coast_df, linewidth=0.5):
    raw_df = pd.read_csv(
        csv_path,
        encoding="shift-jis"
    )
    raw_df.columns = TOP_HEADER
    df = raw_df.copy()
    #
    df["latitude [deg]"] = raw_df["latitude [deg]"].map(convert_coordinate)
    df["longitude [deg]"] = raw_df["longitude [deg]"].map(convert_coordinate)
    #
    time_arr = np.empty(len(df))
    time_origin = JST_str_to_float(df.iloc[0, df.columns.get_loc("time (JST)")])
    for i in range(len(df)):
        #
        time_arr[i] = JST_str_to_float(df.iloc[i, df.columns.get_loc("time (JST)")]) - time_origin
    #
    conv_df = df_to_xy(df)
    #
    df["t [s]"] = time_arr
    df["p_x [m]"] = conv_df[:, 0]
    df["p_y [m]"] = conv_df[:, 1]
    df["gyro deg [rad]"] = np.deg2rad(df["gyro deg [deg]"].values)
    #save
    folder = os.path.basename(os.path.dirname(csv_path))
    name = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(f"{SAVE_DIR}/csv", exist_ok=True)
    df.to_csv(os.path.join(f"{SAVE_DIR}/csv", f"{folder}__{name}.csv"))
    # root
    ax.plot(df["p_x [m]"], df["p_y [m]"], c=Colors.black,
            linewidth=linewidth, alpha=0.5, zorder=3)
    # ship 
    ax = plot_ship(ax, df)
    #zoom
    axins = inset_axes(ax, width="35%", height="35%", loc="upper right", borderpad=0.6)
    axins.set_aspect("equal")
    draw_base_map(axins, coast_df,apply_coast_extra=True, x_const=-6000.0)
    axins.plot(df["p_x [m]"], df["p_y [m]"], c=Colors.black, lw=1.0, zorder=3)
    if folder == "_Yokkaichi_port1A":
        axins.set_xlim(-2500, -1500)
        axins.set_ylim(-4500, -3500)
    else:
        axins.set_xlim(-3500, -2500)
        axins.set_ylim(-7500, -6500)      
    axins = plot_ship(axins, df)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", lw=0.8)
    #save
    os.makedirs(f"{SAVE_DIR}/fig", exist_ok=True)
    plt.savefig(os.path.join(f"{SAVE_DIR}/fig", f"{folder}__{name}.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)

def plot_ship(ax,df):
    for j in range(len(df)):
        p = df.iloc[
            j,
            [
                df.columns.get_loc("p_x [m]"),
                df.columns.get_loc("p_y [m]"),
                df.columns.get_loc("gyro deg [rad]")
            ]
        ].to_numpy(dtype=float)
        ax.add_patch(
            plt.Polygon(
                ship_shape_poly(p, 100, 16, scale=1.0, z_axis_upward=True),
                fill=True, alpha=0.5,
                color=Colors.red, linewidth=0.3, zorder=4
            )
        )
        ax.add_patch(
            plt.Polygon(
                ship_shape_poly(p, 100, 16, scale=1.0, z_axis_upward=True),
                fill=False, color=Colors.red, linewidth=0.3, zorder=4
            )
        )
    return ax

def main():
    coast_df = prepare(coast_path)
    set_rcParams()
    paths = sorted(glob.glob(f"{DIR}/../../raw_datas/tmp/_Yokkaichi_port*/*.csv"))
    for csv_path in paths:
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_base_map(ax, coast_df,apply_coast_extra=True, x_const=-6000.0)
        #draw_waterdepth(ax)
        plot_one_route_and_save(ax, csv_path, coast_df, linewidth=0.5)
        plt.close(fig)
        print(f"\nsaved:    {os.path.splitext(os.path.basename(csv_path))[0]}\n")

if __name__ == "__main__":
    main()
