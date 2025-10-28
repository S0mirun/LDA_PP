"""
CMA-ESのためのBezier Curve初期経路計画アルゴリズム

"""

import glob
import os

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from utils.PP.MultiPlot import Buoy
from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
TMP_DIR = f"{DIR}/../../../raw_datas/tmp"
Buoy_DIR = f"{DIR}/../../../raw_datas/buoy"
#
dictionary_of_port = {
    0: {
        "name": "Osaka_port1A",
        "bay": sakai_bay.port1A,
        "start": [-1400.0, -800.0],
        "end": [0.0, -10.0],
        "psi_start": 40,
        "psi_end": 10,
        "berth_type": 2,
        "ver_range": [-1500, 500],
        "hor_range": [-1000, 500],
    },
    1: {
        "name": "Tokyo_port2C",
        "bay": Tokyo_bay.port2C,
        "start": [-1400.0, -1100.0],
        "end": [0.0, 0.0],
        "psi_start": 45,
        "psi_end": 10,
        "berth_type": 2,
        "ver_range": [-2500, 500],
        "hor_range": [-2500, 500],
    },
    2: {
        "name": "Yokkaichi_port2B",
        "bay": yokkaichi_bay.port2B,
        "start": [2050.0, 2000.0],
        "end": [200.0, 100.0],
        "psi_start": -125,
        "psi_end": 175,
        "berth_type": 1,
        "ver_range": [-500, 2500],
        "hor_range": [-500, 2500],
    },
    3: {
        "name": "Else_port1",
        "bay": else_bay.port1,
        "start": [2500.0, 0.0],
        "end": [350.0, 20.0],
        "psi_start": -145,
        "psi_end": 160,
        "berth_type": 1,
        "ver_range": [0, 3000],
        "hor_range": [-1000, 2000],
    },
    4: {
        "name": "Osaka_port1B",
        "bay": sakai_bay.port1B,
        "start": [-1400.0, -800.0],
        "end": [0.0, 15.0],
        "psi_start": 40,
        "psi_end": -10,
        "berth_type": 2,
        "ver_range": [-1500, 500],
        "hor_range": [-1000, 500],
    },
}

def _bernstein_matrix(n: int, t: np.ndarray) -> np.ndarray:
    """
    n 次ベジェのバーンスタイン基底行列 B(t) を返す。
    形状: (len(t), n+1)
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)             # (M,1)
    k = np.arange(n + 1, dtype=int)                           # (n+1,)
    coeff = np.array([math.comb(n, i) for i in k], float)      # (n+1,)
    # ブロードキャストで (M, n+1) を作る
    T = t ** k                                                # (M, n+1)
    U = (1.0 - t) ** (n - k)                                  # (M, n+1)
    return coeff * T * U 


def bezier(buoy_xy: list, start_xy: list, end_xy: list, num: int = 400):
    buoy_xy = np.column_stack([buoy_xy[0], buoy_xy[1]])
    xy = np.vstack([start_xy, buoy_xy, end_xy])
    #
    d   = np.linalg.norm(xy - xy[-1], axis=1)
    D0  = np.linalg.norm(xy[0] - xy[-1])

    mask = d <= D0                      # これより遠いものを除外
    keep = xy[mask]
    dk   = d[mask]
    pts = keep[np.argsort(-dk)]

    n = pts.shape[0] - 1
    t = np.linspace(0.0, 1.0, num)
    B = _bernstein_matrix(n, t)                      # (num, n+1)
    C = B @ pts                                      # (num, 2)
    # get psi
    dpts = n * (pts[1:] - pts[:-1])
    B1   = _bernstein_matrix(n-1, t)
    dC   = B1 @ dpts 
    psi_rad = ((-np.arctan2(dC[:, 1], dC[:, 0])) + np.pi/2) % (2*np.pi) # [0, 2π)
    psi_deg = (np.degrees(psi_rad)) % 360.0

    return C, psi_rad, psi_deg

if __name__ == '__main__':
    time_start_bezier = time.time()
    port = dictionary_of_port[2]
    df_buoy = glob.glob(f"{Buoy_DIR}/_{port['name']}.csv")
    buoy = Buoy()
    buoy.input_csv(df_buoy[0], f"{TMP_DIR}/coordinates_of_port/{port['bay'].name}.csv")
    #
    C, _, _ = bezier([buoy.X, buoy.Y], [511.0, 616.0], [177.0, 303.0])
    print(C)
    #
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(C[:, 1], C[:, 0], linewidth=2, label="bezier")
    #
    ax.scatter(buoy.Y, buoy.X, color='orange', s=20, zorder=4)
    for i, (x, y) in enumerate(zip(buoy.Y, buoy.X)):
        ax.annotate(f"{i}", (x, y),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=8, ha="left", va="bottom",
                    zorder=5, clip_on=True)

    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.set_title("Bézier curve")
    plt.savefig(os.path.join(SAVE_DIR, "BezierCurve.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    print("\nDone\n")