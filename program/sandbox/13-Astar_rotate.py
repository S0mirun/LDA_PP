from __future__ import annotations

import heapq
from dataclasses import dataclass
import glob
import os
from typing import List, Tuple, Optional, Set, Dict

import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.LDA.ship_geometry import *
from utils.LDA.visualization import *


DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
# astar map
batrh_path = f"{DIR}/../../raw_datas/tmp/csv/yokkaichi_port2.csv"
port_path = f"{DIR}/../../raw_datas/tmp/coordinates_of_port/_Yokkaichi_port2B.csv"
# astar settings
OFFSETS: List[Tuple[int, int]] = [
    (0, -1),  (0, 1),   (-1, 0),  (1, 0),
    (-1, -1), (-1, 1),  (1, -1),  (1, 1),
]
HEADINGS: np.ndarray = np.deg2rad([180, 0, -90, 90, -135, -45, 135, 45])

start_pt = [200.0, 100.0] # [ver, hor]
start_psi = 175 # [deg]
end_pt = [0,0] # [ver, hor]
end_psi = 0 # deg

@dataclass(eq=False)
class Node:
    parent: Optional["Node"]
    position: Tuple[int, int]
    psi: Optional[float] = None
    g: float = 0.0
    h: float = 0.0
    f: float = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.position == other.position


def read_csv():
    raw_df = pd.read_csv(
        batrh_path,
        usecols=[0,1],
        encoding='shift-jis'
    )
    df = pd.DataFrame(columns=['latitude', 'longitude'])
    df['latitude'] = raw_df.iloc[:,1]
    df['longitude'] =raw_df.iloc[:,0]
    #
    port_df = pd.read_csv(port_path)
    LAT_ORIGIN = port_df['Latitude'].iloc[0]
    LON_ORIGIN = port_df['Longitude'].iloc[0]
    ANGLE_FROM_NORTH = port_df['Psi[deg]'].iloc[0]
    #
    p_x_arrtpgrph = np.empty(len(df))
    p_y_arrtpgrph = np.empty(len(df))
    for i in range(len(df)):
        #
        p_y_temp, p_x_temp = convert_to_xy(
            df.iloc[i, df.columns.get_loc("latitude")],
            df.iloc[i, df.columns.get_loc("longitude")],
            LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
        )
        p_x_arrtpgrph[i] = p_x_temp
        p_y_arrtpgrph[i] = p_y_temp
    poly_xy = np.vstack([p_x_arrtpgrph, p_y_arrtpgrph])
    return poly_xy # (2, N)

def drow_basemap(ax, poly_xy):
    p_x_arrtpgrph = poly_xy[0]
    p_y_arrtpgrph = poly_xy[1]
    set_rcParams()
    # ax setting
    ax.set_xlim(-200, 200)
    ax.set_ylim(-150, 250)
    ax.set_aspect('equal')
    ax.grid()
    # plot topography
    ax.add_patch(
        plt.Polygon(
            np.transpose(np.array([
                p_x_arrtpgrph,
                p_y_arrtpgrph,
            ])),
            fill=True, alpha=0.5,
            color="gray", linewidth=0,
        )
    )


def occupancy_from_polygon(poly_xy, pitch=5.0, bounds=None, include_boundary=True,
                           inside_val=1, outside_val=0):
    """
    poly_xy: (N,2) 頂点配列（x,y）。最初と最後が同一点でなくてOK
    pitch  : グリッド間隔 [同じ座標系の単位]
    bounds : (xmin, xmax, ymin, ymax) なければポリゴンの外接矩形を自動算出
    include_boundary: Trueなら境界点も「内側」とみなす
    """
    poly_xy = np.asarray(poly_xy, float).T
    if bounds is None:
        xmin, ymin = poly_xy.min(axis=0)
        xmax, ymax = poly_xy.max(axis=0)
        # 少しマージンを足す
        m = 1e-6
        xmin, xmax = xmin - m, xmax + m
        ymin, ymax = ymin - m, ymax + m
    else:
        xmin, xmax, ymin, ymax = bounds

    x = np.arange(xmin, xmax + pitch, pitch)
    y = np.arange(ymin, ymax + pitch, pitch)
    X, Y = np.meshgrid(x, y)                        # 形状: (ny, nx)
    pts = np.c_[X.ravel(), Y.ravel()]               # (ny*nx, 2)

    path = Path(poly_xy, closed=True)
    # Path.contains_points は境界点を False にしがちなので半径で内側寄りにバイアス
    r = -1e-12 if include_boundary else 0.0
    mask = path.contains_points(pts, radius=r)      # bool (ny*nx,)
    grid = np.where(mask.reshape(Y.shape), inside_val, outside_val).astype(np.uint8)
    return grid, x, y

def astar(maze, start_pt, end_pt, start_psi, end_psi):
    #set up
    start_node = Node(None, start_pt, psi=start_psi, g=0.0, h=0.0, f=0.0)
    end_node = Node(None, end_pt, psi=end_psi, g=0.0, h=0.0, f=0.0)

    pass


def show_result():
    pass


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(5,5))
    poly_xy = read_csv()
    #
    drow_basemap(ax, poly_xy)
    grid, xs, ys = occupancy_from_polygon(poly_xy,
                                          pitch=5.0, include_boundary=True)
    A = astar(
        grid,
        start_pt,
        end_pt,
        start_psi,
        end_psi
    )
    plt.savefig(os.path.join(SAVE_DIR, "Astar.png"))
    print("\nDone\n")


