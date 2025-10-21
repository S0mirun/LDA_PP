import glob
import os
import re
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from tqdm import tqdm
import unicodedata

from utils.LDA.ship_geometry import *
from utils.LDA.visualization import *
from utils.PP.subroutine import (yokkaichi_bay, Tokyo_bay, Hokkaido, Honsyu)

DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
AIS_DIR = f"{DIR}/../../raw_datas/【秘密情報】航海記録"
TOPO_DIR = f"{DIR}/../../outputs/Japan_xy"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
# LAT_ORIGIN = 34.57597199
# LON_ORIGIN = 135.4275805
LAT_ORIGIN = 34.5868 # +
LON_ORIGIN = 135.4255 # -
ANGLE_FROM_NORTH = -53.25
#
ZOOM = True
x_lim = (325000, 345000)
y_lim = (-283000, -263000)

#
REGION =  [Honsyu.osaka_bay, yokkaichi_bay, Tokyo_bay,
           Hokkaido.hakodate_bay, Hokkaido.ishikari_bay, Hokkaido.tomakomai, Hokkaido.kushiro,
           Honsyu.akita, Honsyu.aomori_bay, Honsyu.hachinohe,
           Honsyu.isinomaki_bay, Honsyu.kagoshima_bay, Honsyu.kanazawa,
           Honsyu.miho_bay, Honsyu.nigata, Honsyu.onahama, Honsyu.suruga_bay,
           Honsyu.tokuyama_bay, Honsyu.osaka_bay_2
            ]
counts = {getattr(r, "name"): 0 for r in REGION if getattr(r, "name", None)}
keys   = [getattr(r, "name") for r in REGION]



def show_counts(d, keys, first=False):
    # redraw N lines in place
    lines = [f"{k:<12}: {d.get(k, 0):6d}" for k in keys]
    if not first:
        sys.stdout.write(f"\x1b[{len(lines)}A")  # move cursor up N
    for s in lines:
        sys.stdout.write("\r\x1b[2K" + s + "\n")  # clear + rewrite
    sys.stdout.flush()

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



def draw_Japan(ax):
    for data in glob.glob(f"{TOPO_DIR}/*.csv"):
        df = pd.read_csv(
            data,
            encoding='shift-jis'
        )
        ax.plot(df["y [m]"] * (-1), df["x [m]"] * (-1), c='gray', linewidth=0.5)
    
    ax.set_aspect('equal')
    # save
    plt.savefig(os.path.join(SAVE_DIR, "Japan.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nJapan fig saned\n")

def draw_Japan_Poly(ax):
    for data in glob.glob(f"{TOPO_DIR}/*.csv"):
        df = pd.read_csv(
            data,
            encoding='shift-jis'
        )
        x = -1 * df["y [m]"].to_numpy()
        y = -1 * df["x [m]"].to_numpy()
        xy = np.column_stack([x, y])
        # settings
        # ax.set_xticks([])
        # ax.set_xticklabels([])
        # ax.set_yticks([])
        # ax.set_yticklabels([])
        # plot
        ax.add_patch(
            Polygon(
                xy,
                closed=True,
                edgecolor='gray',
                facecolor='gray',
                linewidth=0.5,
            )
        )
        ax.update_datalim(xy)
    # save
    if ZOOM:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        # zoom
        plt.savefig(os.path.join(SAVE_DIR, "Japan_Poly_Zoom.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
    else:
        ax.autoscale()
        ax.set_aspect('equal')
        # save
        plt.savefig(os.path.join(SAVE_DIR, "Japan_Poly.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nJapan fig saved\n")
    
def draw_AIS(ax):
    for dir in tqdm(glob.glob(f"{AIS_DIR}/*"), desc="Reading CSVs", unit="file"):
        for path in glob.glob(f"{dir}/*.csv"):
            raw_df = pd.read_csv(
                path,
                skiprows=[0],
                usecols=[2,3,6],
                encoding='shift-jis'
            )
            raw_df.iloc[:,0] = raw_df.iloc[:,0].map(convert_coordinate)
            raw_df.iloc[:,1] = raw_df.iloc[:,1].map(convert_coordinate)
            raw_df.iloc[:, 2] = pd.to_numeric(raw_df.iloc[:, 2], errors="coerce")
            # latlon → xy
            df = pd.DataFrame({
                "latitude":  raw_df.iloc[:, 0].map(convert_coordinate),
                "longitude": raw_df.iloc[:, 1].map(convert_coordinate),
                "u":         raw_df.iloc[:, 2],
            })
            #
            lat = df["latitude"].to_numpy(dtype=np.float64, copy=False)
            lon = df["longitude"].to_numpy(dtype=np.float64, copy=False)
            x = np.empty_like(lat, dtype=np.float64)
            y = np.empty_like(lat, dtype=np.float64)
            # plot
            for i in range(lat.size):
                y[i], x[i] = convert_to_xy(float(lat[i]), float(lon[i]),
                                        LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 2:
                continue
            ax.plot(x[m], y[m], c=Colors.black, linewidth=1, alpha=0.5)
    # save
    if ZOOM :
        plt.savefig(os.path.join(SAVE_DIR, "AIS_Zoom.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.savefig(os.path.join(SAVE_DIR, "AIS.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nAIS fig saved\n")
    

def count_stay_port(ax):
    paths = sorted(glob.glob(f"{AIS_DIR}/*/S*.csv"))
    raw_dfs = []
    raw_dfs = [read_one(p) for p in tqdm(paths, total=len(paths), desc="Reading CSVs", unit="file")]
    df = pd.concat(raw_dfs, ignore_index=True)
    df["u"] = pd.to_numeric(df["u"], errors="coerce")
    df = df.sort_values("file_date", kind="mergesort").reset_index(drop=True)
    # chaeck
    #show_counts(counts, keys, first=True)
    place_before = None
    for idx, lat, lon in df[["latitude", "longitude"]].itertuples(index=True, name=None):
        #
        place = label_region(lat, lon)
        if place_before is None and place is not None:
            counts[place] += 1
            # plot
            y, x = convert_to_xy(lat, lon,
                                 LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH)
            ax.scatter(x, y, color='green', s=1)
        place_before = place

        # if idx % 50 == 0:
        #     show_counts(counts, keys)
    show_counts(counts, keys)
    # save
    plt.savefig(os.path.join(SAVE_DIR, "stay_port.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))
    #draw_Japan(ax)
    draw_Japan_Poly(ax)
    draw_AIS(ax)
    #count_stay_port(ax)

    print("\nDone\n")