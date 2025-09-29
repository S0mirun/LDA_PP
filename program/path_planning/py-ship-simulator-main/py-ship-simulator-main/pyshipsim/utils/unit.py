import numpy as np


__all__ = ["d2r", "r2d"]


def d2r(deg):
    return deg * np.pi / 180


def r2d(rad):
    return rad * 180 / np.pi
