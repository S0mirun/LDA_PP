"""
CMA-ESのためのBezier Curve初期経路計画アルゴリズム

"""
import os

import math
import matplotlib.pyplot as plt
import numpy as np


class Calcurate():
    def __init__(self):
        pass

    def intersection(self, sm):
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

    def angle(self, origin, to_a, to_b):
        origin = np.asarray(origin, dtype=float)
        to_a   = np.asarray(to_a,   dtype=float)
        to_b   = np.asarray(to_b,   dtype=float)

        v1 = to_a - origin
        v2 = to_b - origin
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2, axis=-1)

        angle_deg = np.zeros_like(m2, dtype=float)

        valid = (m1 > 0.0) & (m2 > 0.0)
        if np.any(valid):
            cos_theta = np.sum(v2[valid] * v1, axis=-1) / (m1 * m2[valid])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_deg[valid] = np.degrees(np.arccos(cos_theta))

        return angle_deg

def stack(sm):
    cal = Calcurate()

    start_xy   = np.asarray(sm.origin_xy[0], dtype=float)
    isect_xy   = np.asarray(cal.intersection(sm), dtype=float)
    end_xy     = np.asarray(sm.last_xy[0],   dtype=float)

    if sm.buoy_xy is None:
        buoy_mat = np.empty((0, 2), dtype=float)  # no buoys
    else:
        arr = np.asarray(sm.buoy_xy, dtype=float)
        if arr.ndim == 2 and arr.shape[-1] == 2:   # already (N,2)
            buoy_mat = arr
        else:                                      # expect (ys, xs)
            by, bx = sm.buoy_xy
            buoy_mat = np.column_stack([by, bx])

    xy = np.vstack([start_xy, buoy_mat, isect_xy, end_xy])
    return xy, isect_xy

def sort(pts, start, end):
    cal = Calcurate()
    d   = np.linalg.norm(pts - end, axis=1)
    D0  = np.linalg.norm(start - end)
    angle = cal.angle(end, start, pts)

    mask = (d <= D0) & (angle < 45)
    keep = pts[mask]
    dk   = d[mask]
    pts = keep[np.argsort(-dk)]

    return pts

def _bernstein_matrix(n: int, t: np.ndarray) -> np.ndarray:
    """
    n 次ベジェのバーンスタイン基底行列 B(t) を返す。
    形状: (len(t), n+1)
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)             # (M,1)
    k = np.arange(n + 1, dtype=int)                           # (n+1,)
    coeff = np.array([math.comb(n, i) for i in k], float)      # (n+1,)

    T = t ** k                                                # (M, n+1)
    U = (1.0 - t) ** (n - k)                                  # (M, n+1)
    return coeff * T * U 

def bezier(pts, num: int = 400):
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

    return C, psi_deg