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


DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
coast_path = f"{DIR}/../../raw_datas/海岸線データ/四日市港 海岸線データ(国土地理院地図から抽出).csv"
AIS_path = f"{DIR}/../../raw_datas/tmp/_Yokkaichi_port*/*.csv"
#
def preprocess(csv_path):
    raw_coast_df = pd.read_csv(
        csv_path,
        encoding='shift-jis'
    )
    #
    coast_df = pd.DataFrame(columns=['latitude', 'longitude'])
    coast_df['latitude'] = raw_coast_df.iloc[:,0]
    coast_df['longitude'] =raw_coast_df.iloc[:,1]
    return coast_df
    
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

if __name__ == "__main__":
    coast_df = preprocess(coast_path)
    print("\nprepare finished\n")
    MAKE_YOKKAICHI_BAY(coast_df)
