"""
指定した着桟時の時系列データを要素に分類する
関さんが設定した港固有の角度を引くことのない生角度（raw)を用いる
全て度数(degree)表記であることに注意
角速度はdeg/sではなくdeg/mを用いる
船速はknotで扱う
"""


import numpy as np
import pandas as pd
import glob
from datetime import datetime
import os

from subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay
from MultiPlot import RealTraj

#elements_list 0:停止、1:保針、2:横移動、3:斜航、4:その場回頭、5:変針(旋回)
DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

def diff_centered(x):
    res = (x[2:] - x[:-2])
    res = ((res + 180) % 360 - 180) / 2
    return res

def diff_gyro_gps(gyro_angle, psi_angle):
    difference_of_psi = gyro_angle - psi_angle
    difference_of_psi = abs((difference_of_psi + 180) % 360 - 180)
    return difference_of_psi

EM_u1 = 0.5# 研究1にならい決定
EM_u2 = 1.5# 研究1の舵きき最低速度にならい決定
EM_lamda1 = 5 ###1分間に5度変針する角速度[deg/m]
EM_lamda2 = 3 ###方位角と進行方向ベクトルの差が3度未満のとき保針[deg]
EM_lamda3 = 10 ###斜航か横移動かを判定する方位角と進行方向ベクトルとの差[deg]
EM_lamda4 = 60 ### 要素5(旋回)の上限角速度[deg/m]

PortList = [
        sakai_bay.port1A,
        sakai_bay.port1B,
        Tokyo_bay.port2B,
        Tokyo_bay.port2C,
        yokkaichi_bay.port1A,
        yokkaichi_bay.port2B,
        else_bay.port1,
        else_bay.port2
    ]
all_data = []

for target_port in PortList:
    files = glob.glob("tmp/" + target_port.name + "/*.csv")
    for file_i in range(len(files)):
        test_real = RealTraj()
        test_real.input_csv(files[file_i], 'tmp/coordinates_of_port/' + target_port.name + '.csv')

        # 角度の中心差分から角速度を得る
        test_real.diff_psi_raw = diff_centered(test_real.psi_raw)

        # 中心差分を使用するため、elements 配列の長さを-2で調整
        test_real.elements = np.empty(len(test_real.time) - 2, dtype='int') #ここでlen(test_real.time)-2しているので次でlen(test_real.elemets)としてよい

# 2が横移動,3が斜航
        for i in range(len(test_real.elements)):
            if test_real.u_knot[i+1] < EM_u1:
                test_real.elements[i] = 0
            else:
                if abs(test_real.diff_psi_raw[i]) < EM_lamda1:
                    # test_real.psi_raw[i+1] - test_real.psi_GPS_raw[i+1]を解く、関数を作ろうか
                    if 0 <= abs(test_real.psi_raw[i+1] - test_real.psi_GPS_raw[i+1]) < EM_lamda2:
                        test_real.elements[i] = 1
                        #ここが0と360の境目になっていたら値が大きくなってしまう
                    elif EM_lamda2 <= diff_gyro_gps(test_real.psi_raw[i+1], test_real.psi_GPS_raw[i+1]) < EM_lamda3:
                            test_real.elements[i] = 3
                    else:
                        test_real.elements[i] = 2#
                else:
                    if test_real.u_knot[i+1] < EM_u2:
                        test_real.elements[i] = 4
                    elif abs(test_real.diff_psi_raw[i]) >= EM_lamda4:
                        test_real.elements[i] = 4
                    else:
                        test_real.elements[i] = 5

        # 各ファイルのデータをリストに追加
        #8/11 x,yを追加
        port_data = pd.DataFrame({
            'port' : target_port.name,
            'time' : test_real.time[1:-1],  # 時間も調整
            'u' : test_real.u[1:-1],
            'knot' : test_real.u_knot[1:-1],
            'X' : test_real.X[1:-1],
            'Y' : test_real.Y[1:-1],
            'psi_raw' : test_real.psi_raw[1:-1],
            'psi_GPS_raw' : test_real.psi_GPS_raw[1:-1],
            'diff_psi_raw' : test_real.diff_psi_raw,
            'element' : test_real.elements
        })
        #
        os.makedirs(f"{SAVE_DIR}/{target_port.name}", exist_ok=True)
        port_data.to_csv(
            os.path.join(
                f"{SAVE_DIR}/{target_port.name}",
                f"{str(file_i)}{target_port.name}_classified_elements_fixed.csv",
            ),
            index=False
        )
        print(f"\nfinish    : {str(file_i)}{target_port.name}\n")
        #
        all_data.append(port_data)

# 全てのデータを1つのDataFrameに結合
all_data_df = pd.concat(all_data, ignore_index=True)
# まとめてCSVに出力。
all_data_df.to_csv(os.path.join(SAVE_DIR, "all_ports_classified_elements_fixed.csv"), index=False)
print(f"Output saved to {SAVE_DIR}")