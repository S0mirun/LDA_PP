"""
指定した着桟時の時系列データを要素に分類する。
生角度(raw)を用い、全て度数(degree)。角速度はdeg/m、船速はknotで扱う。
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd

from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay
from utils.PP.MultiPlot import RealTraj

# elements: 0 stop, 1 steady, 2 lateral, 3 oblique, 4 on-site turn, 5 turning

def diff_centered(x: np.ndarray) -> np.ndarray:
    """
    方位角系列x(度)から中心差分の角速度(度/分)を求める。
    wrap(±180)後に/2して、実際の変化量/分に整える。
    """
    res = x[2:] - x[:-2]
    res = ((res + 180) % 360 - 180) / 2
    return res


def diff_gyro_gps(gyro_angle: float, psi_angle: float) -> float:
    """
    ジャイロ方位(gyro)と進行方向方位(psi)の差(度)を[0,180]に正規化して返す。
    """
    d = gyro_angle - psi_angle
    return abs((d + 180) % 360 - 180)


# thresholds
EM_u1 = 0.5
EM_u2 = 1.5
EM_lamda1 = 5
EM_lamda2 = 3
EM_lamda3 = 10
EM_lamda4 = 60

# targets
PortList = [
    sakai_bay.port1A,
    sakai_bay.port1B,
    Tokyo_bay.port2B,
    Tokyo_bay.port2C,
    yokkaichi_bay.port1A,
    yokkaichi_bay.port2B,
    else_bay.port1,
    else_bay.port2,
]

# output dir
DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
TMP_DIR = f"{DIR}/../../raw_datas/tmp"

all_data = []

for target_port in PortList:
    files = glob.glob(f"{TMP_DIR}/{target_port.name}/*.csv")
    for file_i, path in enumerate(files):
        rt = RealTraj()
        rt.input_csv(path, f"{TMP_DIR}/coordinates_of_port/{target_port.name}.csv")

        # centered diff for turn rate
        rt.diff_psi_raw = diff_centered(rt.psi_raw)

        # N-2 because of centered diff window
        rt.elements = np.empty(len(rt.time) - 2, dtype=int)

        # classify
        for i in range(len(rt.elements)):
            if rt.u_knot[i + 1] < EM_u1:
                rt.elements[i] = 0
            else:
                if abs(rt.diff_psi_raw[i]) < EM_lamda1:
                    # heading diff (wrapped)
                    d_heading = diff_gyro_gps(rt.psi_raw[i + 1], rt.psi_GPS_raw[i + 1])
                    if 0 <= d_heading < EM_lamda2:
                        rt.elements[i] = 1
                    elif EM_lamda2 <= d_heading < EM_lamda3:
                        rt.elements[i] = 3
                    else:
                        rt.elements[i] = 2
                else:
                    if rt.u_knot[i + 1] < EM_u2:
                        rt.elements[i] = 4
                    elif abs(rt.diff_psi_raw[i]) >= EM_lamda4:
                        rt.elements[i] = 4
                    else:
                        rt.elements[i] = 5

        # build dataframe
        df = pd.DataFrame(
            {
                "port": target_port.name,
                "time": rt.time[1:-1],
                "u": rt.u[1:-1],
                "knot": rt.u_knot[1:-1],
                "X": rt.X[1:-1],
                "Y": rt.Y[1:-1],
                "psi_raw": rt.psi_raw[1:-1],
                "psi_GPS_raw": rt.psi_GPS_raw[1:-1],
                "diff_psi_raw": rt.diff_psi_raw,
                "element": rt.elements,
            }
        )

        # save per file
        os.makedirs(f"{SAVE_DIR}/{target_port.name}", exist_ok=True)
        df.to_csv(
            os.path.join(
                f"{SAVE_DIR}/{target_port.name}",
                f"{file_i}{target_port.name}_classified_elements_fixed.csv",
            ),
            index=False,
        )
        print(f"\nfinish    : {file_i}{target_port.name}\n")

        all_data.append(df)

# save merged
all_df = pd.concat(all_data, ignore_index=True)
all_df.to_csv(os.path.join(SAVE_DIR, "all_ports_classified_elements_fixed.csv"), index=False)
print(f"Output saved to {SAVE_DIR}")
