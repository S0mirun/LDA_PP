import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import unicodedata

from utils.LDA.ship_geometry import *
from utils.LDA.visualization import *
from utils.PP.stay_ports import Hokkaido, Honsyu, Sea

DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
AIS_DIR = f"{DIR}/../../raw_datas/【秘密情報】航海記録"
TPGRPH_DIR = f"{DIR}/../../raw_datas/tmp/csv"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
LAT_ORIGIN = 34.57597199
LON_ORIGIN = 135.4275805
ANGLE_FROM_NORTH = 0
#
ZOOM = True
#
REGION = Hokkaido.ALL + Honsyu.ALL
SEA = Sea.ALL
#
counts = {getattr(r, "name"): 0 for r in REGION if getattr(r, "name", None)}
keys   = [getattr(r, "name") for r in REGION]

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

def file_date(path: str) -> pd.Timestamp:
    s = os.path.basename(path)
    return pd.to_datetime(s[1:9], format="%Y%m%d")

def to_float(s):
    if pd.isna(s): return np.nan
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = re.sub(r"^(?P<sgn>[+\-])\s+", r"\g<sgn>", s)
    m = re.match(r"[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", s)
    return float(m.group()) if m else np.nan

def read_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=[0],
        usecols=[2, 3, 6],     # lat, lon, u
        encoding="shift-jis",
        converters={6: to_float},
    )
    df.columns = ["latitude", "longitude", "u"]
    df["latitude"]  = pd.to_numeric(df["latitude"].map(convert_coordinate),  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"].map(convert_coordinate), errors="coerce")
    df["file_date"] = file_date(path)
    return df

def label_region(lat: float, lon: float) -> str | None:
    for region_cls in REGION:
        inside = (
            region_cls.minY_lat  <= lat <= region_cls.maxY_lat and
            region_cls.minY_long <= lon <= region_cls.maxY_long
        )
        if inside:
            return region_cls.name


def draw_Japan_Poly(ax):
    for data in glob.glob(f"{TPGRPH_DIR}/hokkaidou_*_LATLONG.csv"):
        raw_df = pd.read_csv(
            data,
            usecols=[0,1],
            encoding='shift-jis'
        )
        #
        df_tpgrph = pd.DataFrame({
            "latitude":  raw_df.iloc[:, 1].map(convert_coordinate),
            "longitude": raw_df.iloc[:, 0].map(convert_coordinate),
        })
        #
        p_x_arrtpgrph = np.empty(len(df_tpgrph))
        p_y_arrtpgrph = np.empty(len(df_tpgrph))
        for i in range(len(df_tpgrph)):
            p_y_temp, p_x_temp = convert_to_xy(
                df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("latitude")],
                df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("longitude")],
                LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
            )
            p_x_arrtpgrph[i] = p_x_temp
            p_y_arrtpgrph[i] = p_y_temp
        #
        xy = np.vstack([p_x_arrtpgrph, p_y_arrtpgrph]).T
        set_rcParams()
        ax.add_patch(
            Polygon(
                xy,
                closed=True,
                edgecolor='black',
                facecolor='gray',
                linewidth=0.5,
                alpha=0.3,
            )
        )
    # settings
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # save
    ax.autoscale()
    ax.set_aspect('equal')
    plt.savefig(os.path.join(SAVE_DIR, "Japan.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nJapan fig saved\n")

def read_AIS():
    paths = sorted(glob.glob(f"{AIS_DIR}/*/S*.csv"))
    raw_dfs = []
    raw_dfs = [read_one(p) for p in tqdm(paths, total=len(paths), desc="Reading CSVs", unit="file")]
    df = pd.concat(raw_dfs, ignore_index=True)
    df = df.sort_values("file_date", kind="mergesort").reset_index(drop=True)
    print("\nfinished   : read AIS\n")
    return df

def draw_AIS(ax, df):
    p_x_arrtpgrph = np.empty(len(df))
    p_y_arrtpgrph = np.empty(len(df))
    for i in tqdm(range(len(df)), desc="Drawing Paths", unit="file"):
        p_y_temp, p_x_temp = convert_to_xy(
            df.iloc[i, df.columns.get_loc("latitude")],
            df.iloc[i, df.columns.get_loc("longitude")],
            LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
        )
        p_x_arrtpgrph[i] = p_x_temp
        p_y_arrtpgrph[i] = p_y_temp
    ax.plot(p_x_arrtpgrph, p_y_arrtpgrph, color='black', linewidth=0.5, alpha=0.5)
    # save
    plt.savefig(os.path.join(SAVE_DIR, "AIS.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    if ZOOM :
        os.makedirs(f"{SAVE_DIR}/AIS", exist_ok=True)
        for port in REGION + SEA:
            y_min, x_min = convert_to_xy(
                port.minY_lat,
                port.minY_long,
                LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
            )
            y_max, x_max = convert_to_xy(
                port.maxY_lat,
                port.maxY_long,
                LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            plt.savefig(os.path.join(f"{SAVE_DIR}/AIS", f"{port.name}.png"),
                        dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nAIS fig saved\n")

def count_stay_port(ax, df):
    # chaeck
    place_before = None
    for lat, lon in df[["latitude", "longitude"]].itertuples(index=False, name=None):
        #
        place = label_region(lat, lon)
        if place_before is None and place is not None:
            counts[place] += 1
            # plot
            y, x = convert_to_xy(lat, lon,
                                 LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH)
            ax.scatter(x, y, color='red', s=2, zorder=10)
        place_before = place

    # result
    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    width = max((len(k) for k, _ in sorted_items), default=0)
    out_path = Path(SAVE_DIR) / "result.txt"
    with out_path.open("w", encoding='utf-8') as f:
        for name, cnt in sorted_items:
            f.write(f"{name:<{width}} : {cnt}\n")
        f.write(f"(total={sum(v for _, v in sorted_items)}, unique={len(sorted_items)})\n") 
    # save fig
    ax.autoscale()
    ax.set_aspect('equal')
    plt.savefig(os.path.join(SAVE_DIR, "stay_port.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_Japan_Poly(ax)
    #
    df = read_AIS()
    draw_AIS(ax, df)
    count_stay_port(ax, df)
    #
    print("\nDone\n")