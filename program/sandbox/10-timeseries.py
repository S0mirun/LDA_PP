import glob
import os

import numpy as np
import pandas as pd


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
chain = 2
count = np.zeros([6, 6]).astype(int)


for path in glob.glob(f"{DIR}/../../outputs/ClassifyElements/*/*.csv"):
    port_name = os.path.basename(os.path.dirname(path))
    csv_name = os.path.splitext(os.path.basename(path))[0]
    #
    raw_df = pd.read_csv(
        path,
        encoding='shift-jis'
    )
    #
    df = raw_df.copy()
    for pre in range(chain):
        df[f"element_prev{pre}"] = df["element"].shift(pre)
    #
    os.makedirs(f"{SAVE_DIR}/dataframe/{port_name}", exist_ok=True)
    df.to_csv(os.path.join(f"{SAVE_DIR}/dataframe/{port_name}", f"{csv_name}.csv"))
    #
    for pre_1 in range(6):
        for now in range(6):
            cnt = ((df["element_prev1"] == pre_1) & (df["element_prev0"] == now)).sum()
            count[pre_1][now] = count[pre_1][now] + cnt
            # print(f"要素{pre_1}→要素{now}： {cnt} 個")
    #print(f"\ncomplete:     {path}\n")

print(count)
for pre_1 in range(6):
    for now in range(6):
        print(f"要素{pre_1}→要素{now}： {count[pre_1][now]} 個")
print("\nDone\n")