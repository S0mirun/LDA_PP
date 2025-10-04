import glob
import os
import re
import unicodedata

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from utils.LDA.ship_geometry import *
from utils.LDA.time_series_figure import TimeSeries,plot_traj
from utils.LDA.visualization import *



DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
top_path = f"{DIR}/../../raw_datas/tmp/csv/yokkaichi_port2.csv"
coast_path = f"{DIR}/../../raw_datas/海岸線データ/四日市港 海岸線データ(国土地理院地図から抽出).csv"
#
TOP_HEADER = [
    "date (JST)", "time (JST)", "latitude [deg]",
    "longitude [deg]", "GPS deg [deg]", "gyro deg [deg]",
    "GPS speed [knot]", "log speed [knot]", "wind dir [deg]", "wind sped [knot]"
]
#
LAT_ORIGIN = 35.00627778
LON_ORIGIN = 136.6740283
ANGLE_FROM_NORTH = 0.0

def prepare(top_path, coast_path):
    raw_top_df = pd.read_csv(
        top_path,
        usecols=[0, 1],
        encoding="shift-jis"
    )
    top_df = pd.DataFrame({"latitude [deg]": raw_top_df.iloc[:, 1], "longitude [deg]": raw_top_df.iloc[:, 0]})
    #
    raw_coast_df = pd.read_csv(
        coast_path,
        encoding="shift-jis"
    )
    coast_df = pd.DataFrame({"latitude [deg]": raw_coast_df.iloc[:, 0], "longitude [deg]": raw_coast_df.iloc[:, 1]})

    return top_df, coast_df

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

def draw_base_map(ax, top_df, coast_df, apply_port_extra=False, apply_coast_extra=True, x_const=-6000.0):
    coords_coast = df_to_xy(coast_df)
    coords_port = df_to_xy(top_df)
    coords_coast = maybe_add_extra(coords_coast, apply_coast_extra, x_const)
    coords_port = maybe_add_extra(coords_port, apply_port_extra, x_const)
    #
    if not np.allclose(coords_coast[0], coords_coast[-1]):
        coords_coast = np.vstack([coords_coast, coords_coast[0]])
    if not np.allclose(coords_port[0], coords_port[-1]):
        coords_port = np.vstack([coords_port, coords_port[0]])
    #
    poly_coast = Polygon(coords_coast, closed=True, facecolor=Colors.black,
                         edgecolor="none", alpha=0.5, linewidth=0, zorder=1)
    poly_port = Polygon(coords_port, closed=True, facecolor=Colors.red,
                        edgecolor="none", alpha=1.0, linewidth=0, zorder=2)
    #
    ax.add_patch(poly_coast)
    ax.add_patch(poly_port)
    ax.set_xlim(-4500, 1500)
    y_min = float(min(coords_coast[:, 1].min(), coords_port[:, 1].min()))
    ax.set_ylim(y_min, -3000)
    ax.set_aspect("equal")
    x_ticks = np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1000, 1000)
    y_ticks = np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + 1000, 1000)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

def plot_one_route_and_save(ax, csv_path, linewidth=0.5):
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
    # ax.plot(df["p_x [m]"], df["p_y [m]"], c=Colors.black,
    #         linewidth=linewidth, alpha=0.9, zorder=3)
    # ship 
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
    #save
    os.makedirs(f"{SAVE_DIR}/fig", exist_ok=True)
    plt.savefig(os.path.join(f"{SAVE_DIR}/fig", f"{folder}__{name}.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)

def main():
    top_df, coast_df = prepare(top_path, coast_path)
    set_rcParams()
    paths = sorted(glob.glob(f"{DIR}/../../raw_datas/tmp/_Yokkaichi_port*/*.csv"))
    for csv_path in paths:
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_base_map(ax, top_df, coast_df, apply_port_extra=False, apply_coast_extra=True, x_const=-6000.0)
        plot_one_route_and_save(ax, csv_path, linewidth=0.5)
        plt.close(fig)
        print(f"\nsaved:    {os.path.splitext(os.path.basename(csv_path))[0]}\n")

if __name__ == "__main__":
    main()
