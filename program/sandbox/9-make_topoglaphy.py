import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import unicodedata

from utils.LDA.ship_geometry import *
from utils.LDA.visualization import *
from utils.LDA.kml import kml_based_txt_to_csv


DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
top_path = f"{DIR}/../../raw_datas/tmp/csv/yokkaichi_port2.csv"
coast_path = f"{DIR}/../../raw_datas/海岸線データ/四日市港 海岸線データ(国土地理院地図から抽出).csv"
#
def prepare(top_path, coast_path):
    raw_top_df = pd.read_csv(
        top_path,
        usecols=[0,1],
        encoding='shift-jis'
    )
    #
    top_df = pd.DataFrame(columns=['latitude', 'longitude'])
    top_df['latitude'] = raw_top_df.iloc[:,1]
    top_df['longitude'] =raw_top_df.iloc[:,0]
    #
    raw_coast_df = pd.read_csv(
        coast_path,
        encoding='shift-jis'
    )
    #
    coast_df = pd.DataFrame(columns=['latitude', 'longitude'])
    coast_df['latitude'] = raw_coast_df.iloc[:,0]
    coast_df['longitude'] =raw_coast_df.iloc[:,1]
    return top_df, coast_df

def MAKE_YOKKAICHI(df):
    #
    LAT_ORIGIN = 34.941639
    LON_ORIGIN = 136.639538
    ANGLE_FROM_NORTH = 0.0
    #
    df_tpgrph = df
    p_x_arrtpgrph = np.empty(len(df_tpgrph))
    p_y_arrtpgrph = np.empty(len(df_tpgrph))
    for i in range(len(df_tpgrph)):
        #
        p_y_temp, p_x_temp = convert_to_xy(
            df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("latitude")],
            df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("longitude")],
            LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
        )
        p_x_arrtpgrph[i] = p_x_temp
        p_y_arrtpgrph[i] = p_y_temp
    #
    set_rcParams()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1, 1, 1)
    # ax setting
    ax.set_xlim(-100, 500)
    ax.set_ylim(-100, 500)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # plot topography
    ax.add_patch(
        plt.Polygon(
            np.transpose(np.array([
                p_x_arrtpgrph,
                p_y_arrtpgrph,
            ])),
            fill=True, alpha=0.5,
            color=Colors.red, linewidth=0,
            # label = "Topography",
        )
    )
    # plot scale
    scalebar_length = 10000
    x0, y0 = -25000, 25000  # left and right edges of scale bar
    tick_interval = 1000
    lw_m = 2.5
    ax.plot([x0, x0 + scalebar_length], [y0, y0], color=Colors.black, linewidth=lw_m)
    num_ticks = scalebar_length // tick_interval + 1
    for i in range(num_ticks):
        xtick = x0 + i * tick_interval
        ax.plot(
            [xtick, xtick],
            [y0, y0 + 1000],
            color=Colors.red,
            linewidth=lw_m if i==0 or i==int(num_ticks / 2) or i==num_ticks-1 else 1,
        )
        if i == 0:
            ax.text(xtick, y0 + 3000, "0", ha='center', va='top',)
        if i == num_ticks - 1:
            ax.text(xtick, y0 + 3000, f'{scalebar_length} m' , ha='center', va='top',)
    #
    fig.align_labels()
    #fig.tight_layout()
    #
    plt.savefig(os.path.join(SAVE_DIR, "YOKKAICHI"))
    print("\nfigure saved   : YOKKAICHI\n")
    
def MAKE_YOKKAICHI_BAY(df):
    #
    LAT_ORIGIN = 35.00627778
    LON_ORIGIN = 136.6740283
    ANGLE_FROM_NORTH = 0.0
    #
    df_tpgrph = df
    p_x_arrtpgrph = np.empty(len(df_tpgrph))
    p_y_arrtpgrph = np.empty(len(df_tpgrph))
    for i in range(len(df_tpgrph)):
        #
        p_y_temp, p_x_temp = convert_to_xy(
            df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("latitude")],
            df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("longitude")],
            LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
        )
        p_x_arrtpgrph[i] = p_x_temp
        p_y_arrtpgrph[i] = p_y_temp
    #
    set_rcParams()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1, 1, 1)
    # ax setting
    ax.set_xlim(-6000, 2500)
    ax.set_ylim(p_y_arrtpgrph.min(), p_y_arrtpgrph.max())
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # plot topography
    coords = np.column_stack([p_x_arrtpgrph, p_y_arrtpgrph]).astype(float)
    extra_pts = np.array([[-6000, float(p_y_arrtpgrph.min())],
                         [-6000, float(p_y_arrtpgrph.max())],
                         [float(coords[0,0]), float(coords[0,1])]  
                         ])
    coords = np.vstack([coords, extra_pts])
    ax.add_patch(
        Polygon(
            coords,
            closed=True,
            facecolor=Colors.black,
            linewidth=0,
            alpha=0.5
        )
    )
    #fig.tight_layout()
    #
    plt.savefig(os.path.join(SAVE_DIR, "YOKKAICHI_BAY.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nfigure saved   : YOKKAICHI BAY\n")

from matplotlib.patches import Polygon
import numpy as np

def MAKE_YOKKAICHI_SUM(top_df, coast_df,
                       apply_port_extra=False,
                       apply_coast_extra=True,
                       x_const=-6000.0):
    LAT_ORIGIN = 35.00627778
    LON_ORIGIN = 136.6740283
    ANGLE_FROM_NORTH = 0.0

    def df_to_coords(df):
        lat_idx = df.columns.get_loc("latitude")
        lon_idx = df.columns.get_loc("longitude")
        px = np.empty(len(df), dtype=np.float64)
        py = np.empty(len(df), dtype=np.float64)
        for i in range(len(df)):
            y_m, x_m = convert_to_xy(
                float(df.iat[i, lat_idx]), float(df.iat[i, lon_idx]),
                LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
            )
            px[i] = x_m; py[i] = y_m
        return np.column_stack([px, py]).astype(float)

    def maybe_add_extra(coords, use_flag):
        if not use_flag:
            return coords
        y_min = float(coords[:,1].min())
        y_max = float(coords[:,1].max())
        extra = np.array([
            [x_const, y_min],
            [x_const, y_max],
            [float(coords[0,0]), float(coords[0,1])]
        ], dtype=float)
        return np.vstack([coords, extra])

    def convert_coordinate(value):
        if value is None or value == '':
            return float('nan')
        s = unicodedata.normalize("NFKC", str(value)).strip()
        s = s.replace("’", "'").replace("′", "'").replace("”", '"').replace("″", '"')
        m = re.match(r'^([+-]?\d+(?:\.\d+)?)(?:[°\s]*?(\d+(?:\.\d+)?))?(?:[\'\s]*?(\d+(?:\.\d+)?)(?:"|″)?)?\s*([NnSsEeWw])?$', s)
        if not m:
            nums = re.findall(r'\d+(?:\.\d+)?', s)
            if not nums: return float('nan')
            deg = float(nums[0])
            if len(nums) >= 2:
                deg += float(nums[1]) / 60.0
            if len(nums) >= 3:
                deg += float(nums[2]) / 3600.0
            return deg
        deg = float(m.group(1))
        mi  = float(m.group(2)) if m.group(2) else 0.0
        se  = float(m.group(3)) if m.group(3) else 0.0
        hem = (m.group(4) or '').upper()
        val = deg + mi/60.0 + se/3600.0
        if hem in ('S', 'W'):
            val = -abs(val)
        elif hem in ('N', 'E'):
            val = abs(val)
        return val

    coords_coast = df_to_coords(coast_df)
    coords_port  = df_to_coords(top_df)

    coords_coast = maybe_add_extra(coords_coast, apply_coast_extra)
    coords_port  = maybe_add_extra(coords_port,  apply_port_extra)

    if not np.allclose(coords_coast[0], coords_coast[-1]):
        coords_coast = np.vstack([coords_coast, coords_coast[0]])
    if not np.allclose(coords_port[0], coords_port[-1]):
        coords_port = np.vstack([coords_port, coords_port[0]])

    poly_coast = Polygon(coords_coast, closed=True, facecolor=Colors.black, edgecolor="none", alpha=0.5, linewidth=0)
    poly_port  = Polygon(coords_port,  closed=True, facecolor=Colors.red,   edgecolor="none", alpha=1.0, linewidth=0)

    set_rcParams()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-4500, 1500)
    y_min = float(min(coords_coast[:,1].min(), coords_port[:,1].min()))
    y_max = float(max(coords_coast[:,1].max(), coords_port[:,1].max()))
    ax.set_ylim(y_min, -3000)
    ax.set_aspect('equal')
    #ax.set_xticks([]); ax.set_yticks([])
    x_ticks = np.arange(ax.get_xlim()[0], ax.get_xlim()[1]+1000, 1000)
    y_ticks = np.arange(ax.get_ylim()[0], ax.get_ylim()[1]+1000, 1000)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.add_patch(poly_coast)
    ax.add_patch(poly_port)

    # plt.savefig(os.path.join(SAVE_DIR, "YOKKAICHI_SUM.png"),
    #             dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nfigure saved   : YOKKAICHI_SUM\n")

    for i_path, path in enumerate(glob.glob(f"{DIR}/../../raw_datas/tmp/_Yokkaichi_port*/*.csv")):
        raw_df = pd.read_csv(path, usecols=[2, 3], encoding='shift-jis')
        raw_df.iloc[:,0] = raw_df.iloc[:,0].map(convert_coordinate)
        raw_df.iloc[:,1] = raw_df.iloc[:,1].map(convert_coordinate)

        df = pd.DataFrame(columns=['latitude', 'longitude'])
        df['latitude']  = raw_df.iloc[:,0]
        df['longitude'] = raw_df.iloc[:,1]

        lat = df["latitude"].to_numpy(dtype=np.float64, copy=False)
        lon = df["longitude"].to_numpy(dtype=np.float64, copy=False)
        x = np.empty_like(lat, dtype=np.float64)
        y = np.empty_like(lat, dtype=np.float64)
        for i in range(lat.size):
            y[i], x[i] = convert_to_xy(float(lat[i]), float(lon[i]),
                                       LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            continue
        ax.plot(x[m], y[m], c=Colors.black, linewidth=0.5, alpha=0.9,
                label="Path" if i_path == 0 else None, zorder=3)

    plt.savefig(os.path.join(SAVE_DIR, "YOKKAICHI_PATH_1A2B_ZOOM.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nfigure saved   : YOKKAICHI_PATH\n")



if __name__ == "__main__":
    top_df, coast_df = prepare(top_path, coast_path)
    print("\nprepare finished\n")
    #MAKE_YOKKAICHI(top_df)
    #MAKE_YOKKAICHI_BAY(coast_df)
    MAKE_YOKKAICHI_SUM(top_df, coast_df)
