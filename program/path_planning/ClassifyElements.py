"""
指定した着桟時の時系列データを要素に分類する
関さんが設定した港固有の角度を引くことのない生角度（raw)を用いる
全て度数(degree)表記であることに注意
角速度はdeg/sではなくdeg/mを用いる
船速はknotで扱う
"""

import glob
import os
import numpy as np
import pandas as pd

from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay
from utils.PP.MultiPlot import RealTraj

DIR = os.path.dirname(__file__)
TMP_DIR = f"{DIR}/../../raw_datas/tmp"
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"

# elements_list 0:停止、1:保針、2:横移動、3:斜航、4:その場回頭、5:変針(旋回)
EM_u1 = 0.5
EM_u2 = 1.5
EM_lamda1 = 5
EM_lamda2 = 3
EM_lamda3 = 10
EM_lamda4 = 60

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


def diff_centered(x: np.ndarray) -> np.ndarray:
    """
    方位角系列x(度)の中心差分による変化量(度/サンプル)を返す。
    角度の巻き込み(±180度)を考慮して差分化する。
    """
    res = x[2:] - x[:-2]
    res = ((res + 180) % 360) - 180
    return res / 2.0


def diff_gyro_gps(gyro_angle: float, psi_angle: float) -> float:
    """
    ジャイロ方位(gyro)と進行方向方位(psi)の差(度)を[0,180]に正規化して返す。
    """
    d = gyro_angle - psi_angle
    d = (d + 180) % 360 - 180
    return abs(d)


def classify(current_pt: dict, child_pt: dict) -> int:
    """
    現在点と次(子)点の情報から要素ラベル(0..5)を判定して返す。

    Parameters
    ----------
    current_pt : dict
        現在点の情報。以下のキーを持つ:
        - "diff_psi_raw" : float  現在点での方位角中心差分(度/サンプル; 実質deg/mとして扱う)
    child_pt : dict
        次(子)点の情報。以下のキーを持つ:
        - "u_knot"       : float  子点での船速[knot]
        - "psi_raw"      : float  子点でのジャイロ(生)方位[deg]
        - "psi_GPS_raw"  : float  子点での進行方向方位[deg]

    Returns
    -------
    int
        要素ラベル(0:停止, 1:保針, 2:横移動, 3:斜航, 4:その場回頭, 5:変針(旋回))
    """
    u_knot = child_pt["u_knot"]
    dpsi = abs(current_pt["diff_psi_raw"])  # turn rate proxy
    d_heading = diff_gyro_gps(child_pt["psi_raw"], child_pt["psi_GPS_raw"])

    if u_knot < EM_u1:
        return 0
    if dpsi < EM_lamda1:
        if d_heading < EM_lamda2:
            return 1
        elif EM_lamda2 <= d_heading < EM_lamda3:
            return 3
        else:
            return 2
    else:
        if u_knot < EM_u2:
            return 4
        elif dpsi >= EM_lamda4:
            return 4
        else:
            return 5


def classify_file(csv_path: str, coord_csv_path: str, port_name: str) -> pd.DataFrame:
    """
    単一CSVを読み込み，分類結果をDataFrameで返す（ファイル出力は行わない）。
    """
    rt = RealTraj()
    rt.input_csv(csv_path, coord_csv_path)
    rt.diff_psi_raw = diff_centered(rt.psi_raw)
    elements = np.empty(len(rt.time) - 2, dtype="int")

    # per-segment classification
    for i in range(len(elements)):
        current_pt = {"diff_psi_raw": rt.diff_psi_raw[i]}
        child_pt = {
            "u_knot": rt.u_knot[i + 1],
            "psi_raw": rt.psi_raw[i + 1],
            "psi_GPS_raw": rt.psi_GPS_raw[i + 1],
        }
        elements[i] = classify(current_pt, child_pt)

    df = pd.DataFrame(
        {
            "port": port_name,
            "time": rt.time[1:-1],
            "u": rt.u[1:-1],
            "knot": rt.u_knot[1:-1],
            "X": rt.X[1:-1],
            "Y": rt.Y[1:-1],
            "psi_raw": rt.psi_raw[1:-1],
            "psi_GPS_raw": rt.psi_GPS_raw[1:-1],
            "diff_psi_raw": rt.diff_psi_raw,
            "element": elements,
        }
    )
    return df


if __name__ == "__main__":
    # run & export only when executed as a script
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data = []

    for target_port in PortList:
        files = glob.glob(f"{TMP_DIR}/{target_port.name}/*.csv")
        port_save_dir = f"{SAVE_DIR}/{target_port.name}"
        os.makedirs(port_save_dir, exist_ok=True)

        for file_i, path in enumerate(files):
            df = classify_file(
                csv_path=path,
                coord_csv_path=f"{TMP_DIR}/coordinates_of_port/{target_port.name}.csv",
                port_name=target_port.name,
            )
            # per-file export
            out_path = os.path.join(
                port_save_dir, f"{str(file_i)}{target_port.name}_classified_elements_fixed.csv"
            )
            df.to_csv(out_path, index=False)
            all_data.append(df)

        print(f"\nfinish    : {target_port.name}\n")

    all_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    all_df.to_csv(os.path.join(SAVE_DIR, "all_ports_classified_elements_fixed.csv"), index=False)
    print("\nDone\n")
