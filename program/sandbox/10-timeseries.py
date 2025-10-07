import glob
import os

import numpy as np
import pandas as pd


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
pre_number = 2


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
    for pre in range(pre_number):
        df[f"element_prev{pre + 1}"] = df["element"].shift(pre + 1)
    #
    os.makedirs(f"{SAVE_DIR}/{port_name}", exist_ok=True)
    df.to_csv(os.path.join(f"{SAVE_DIR}/{port_name}", f"{csv_name}.csv"))
    #
    
    print(f"\ncomplete:     {path}\n")

print("\nDone\n")