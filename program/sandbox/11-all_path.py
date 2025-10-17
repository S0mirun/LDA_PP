import glob
import os
import re

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from tqdm import tqdm
import unicodedata

from utils.LDA.ship_geometry import *
from utils.LDA.visualization import *

DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
AIS_DIR = f"{DIR}/../../raw_datas/【秘密情報】航海記録"
TOPO_DIR = f"{DIR}/../../outputs/Japan_xy"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
LAT_ORIGIN = 34.57597199
LON_ORIGIN = 135.4275805
ANGLE_FROM_NORTH = -53


def draw_Japan(ax):
    for data in glob.glob(f"{TOPO_DIR}/*.csv"):
        csv_name = os.path.splitext(os.path.basename(data))[0]
        df = pd.read_csv(
            data,
            encoding='shift-jis'
        )
        ax.plot(df["y [m]"] * (-1), df["x [m]"] * (-1), linewidth=0.5)
    
    ax.set_aspect('equal')
    plt.savefig(os.path.join(SAVE_DIR, "Japan.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    
def draw_AIS(ax):
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
    
    for dir in tqdm(glob.glob(f"{AIS_DIR}/*")):
        for path in glob.glob(f"{dir}/*.csv"):
            raw_df = pd.read_csv(
                path,
                skiprows=[0],
                usecols=[2,3,6],
                encoding='shift-jis'
            )
            raw_df.iloc[:,0] = raw_df.iloc[:,0].map(convert_coordinate)
            raw_df.iloc[:,1] = raw_df.iloc[:,1].map(convert_coordinate)
            # latlon → xy
            df = pd.DataFrame(columns=['latitude', 'longitude'])
            df['latitude']  = raw_df.iloc[:,0]
            df['longitude'] = raw_df.iloc[:,1]
            df['u'] = raw_df.iloc[:,2]
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
            ax.plot(x[m], y[m], c=Colors.black, linewidth=0.5, alpha=0.9)
            # chaeck 停泊
            

    plt.savefig(os.path.join(SAVE_DIR, "AIS.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    



if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_Japan(ax)
    draw_AIS(ax)

    print("\nDone\n")