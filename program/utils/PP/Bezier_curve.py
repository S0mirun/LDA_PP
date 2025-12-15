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
from typing import Optional, Sequence

from utils.PP.MultiPlot import Buoy
from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
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

def calcurate_intersection(sm):
    """
    概要
        start-originを結ぶ直線とlast-endを結ぶ直線の交点を求める。
    出力
        交点の座標[ver, hor]
    """
    start = sm.start_xy[0] ; origin = sm.origin_xy[0]
    end = sm.end_xy[0] ; last = sm.last_xy[0]
    #
    a = (start[0] - origin[0]) / (start[1] - origin[1])
    b = start[0] - a * start[1] # b = y - ax
    c = (end[0] - last[0]) / (end[1] - last[1])
    d = end[0] - c * end[1]
    hor = (d - b) / (a - c)
    ver = (a*d - b*c) / (a - c)
    return np.array([ver, hor], dtype=float)

def compute_control_points(sm, *, k=0.9, beta=0.5, phi_min=0.05, phi_max=0.7):
    """
    P1: closest axis-intercept to start. 
    P2: length uses (5) P1-ratio, then (6) angle-based shrink. (ver,hor)=(y,x)
    """

    def intercept_points(p, q):
        """Return (x_intercept, y_intercept) in (y,x); None if undefined."""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        dy = p[0] - q[0]; dx = p[1] - q[1]
        if np.isclose(dx, 0.0):
            x_int = np.array([0.0, p[1]])                 # y=0 crossing
            y_int = np.array([0.0, 0.0]) if np.isclose(p[1], 0.0) else None  # x=0 crossing only if x==0
            return x_int, y_int
        a = dy / dx; b = p[0] - a * p[1]                  # y = a x + b
        y_int = np.array([b, 0.0])                        # x=0
        x_int = None if np.isclose(a, 0.0) else np.array([0.0, -b / a])  # y=0
        return x_int, y_int

    # endpoints (y,x)
    start  = np.asarray(sm.start_xy[0],  float)
    origin = np.asarray(sm.origin_xy[0], float)
    end    = np.asarray(sm.end_xy[0],    float)
    last   = np.asarray(sm.last_xy[0],   float)

    # --- P1: via intercepts on line (start-origin)
    x_int1, y_int1 = intercept_points(start, origin)
    # cands1 = [pt for pt in (x_int1, y_int1) if pt is not None]
    # if not cands1:
    #     raise ValueError("No intercept candidate for (start, origin).")
    # P1 = cands1[np.argmin([np.linalg.norm(start - pt) for pt in cands1])]
    P1 = (x_int1 + y_int1) / 2

    # --- base lengths
    L1 = np.linalg.norm(P1 - start)
    dist_se = np.linalg.norm(end - start)

    # --- (5) P1-ratio: base φ in (0,1)
    base_phi = k * (L1 / (L1 + dist_se + 1e-9))          # larger L1 -> relatively shorter P2
    phi = float(np.clip(base_phi, phi_min, phi_max))

    # --- (6) angle shrink by entry/exit directions
    u = origin - start                                   # start -> origin
    v = last - end                                       # end -> last
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu > 0.0 and nv > 0.0:
        cos_th = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        w = 0.5 * (1.0 - cos_th)                         # 0..1 (sharper -> larger)
        phi = phi * (1.0 - beta * w)
        phi = float(np.clip(phi, phi_min, phi_max))

    # --- P2 direction and placement
    if nv <= 0.0:                                        # fallback dir if end==last
        wdir = end - start
        nw = np.linalg.norm(wdir)
        dir_vec = wdir / (nw + 1e-9)
    else:
        dir_vec = v / nv

    s = L1 * phi
    P2 = end + s * dir_vec

    return np.vstack([P1, P2], dtype=float)

def calculate_angle(pt1, pt2, pt3):
    hor_1, ver_1 = pt1
    hor_2, ver_2 = pt2
    hor_3, ver_3 = pt3

    v1 = np.array([hor_1 - hor_3, ver_1 - ver_3], dtype=float)
    v2 = np.array([hor_2 - hor_3, ver_2 - ver_3], dtype=float)
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    if m1 == 0.0 or m2 == 0.0:
        angle_deg = 0.0
    else:
        cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_theta)))

    return angle_deg

def bezier(sm, buoy_xy: Optional[Sequence]=None, num: int = 400):
    """Allow buoy_xy=None; accept (ys, xs) or (N,2). Return stacked control polygon."""
    start_xy   = np.asarray(sm.origin_xy[0], dtype=float)
    isect_xy   = np.asarray(calcurate_intersection(sm), dtype=float)
    end_xy     = np.asarray(sm.last_xy[0],   dtype=float)

    if buoy_xy is None:
        buoy_mat = np.empty((0, 2), dtype=float)  # no buoys
    else:
        arr = np.asarray(buoy_xy, dtype=float)
        if arr.ndim == 2 and arr.shape[-1] == 2:   # already (N,2)
            buoy_mat = arr
        else:                                      # expect (ys, xs)
            by, bx = buoy_xy
            buoy_mat = np.column_stack([by, bx])

    if abs(calculate_angle(start_xy, isect_xy, end_xy)) < 90:
        xy = np.vstack([start_xy, buoy_mat, isect_xy, end_xy])
    else:
        xy = np.vstack([start_xy, buoy_mat, end_xy])
    #
    d   = np.linalg.norm(xy - xy[-1], axis=1)
    D0  = np.linalg.norm(xy[0] - xy[-1])

    mask = d <= D0
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
    psi_rad = (-np.arctan2(dC[:, 1], dC[:, 0])) % (2*np.pi) # [0, 2π)
    psi_deg = (np.degrees(psi_rad)) % 360.0

    return C, psi_deg, isect_xy

if __name__ == '__main__':
    SAVE_DIR = f"{DIR}/../../../outputs/{dirname}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    time_start_bezier = time.time()
    port = dictionary_of_port[2]
    df_buoy = glob.glob(f"{Buoy_DIR}/_{port['name']}.csv")
    buoy = Buoy()
    buoy.input_csv(df_buoy[0], f"{TMP_DIR}/coordinates_of_port/{port['bay'].name}.csv")
    #
    C, _, _ = bezier([buoy.X, buoy.Y], [1880.0, 1775.0], [700.0, 85.0], [315.0, 85.0])
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