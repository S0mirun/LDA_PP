import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DIR = os.path.dirname(__file__)
CSV_DIR = f"{DIR}/../Path_Planning/output/elements/20250924"
SAVE_DIR = f"{DIR}/outputs/{os.path.splitext(os.path.basename(__file__))[0]}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
paths = glob.glob(f"{CSV_DIR}/_*.csv")
#
u_0 = 0.1
#
def CLASSIFY(path):
    df = pd.read_csv(
        path,
        encoding='shift-jis'
    )
    diff_u = df["u"].diff()
    #
    acc_or_dec = np.zeros((3, 6))
    #
    for num in range(len(diff_u)):
        row = judge(diff_u[num])
        col = df["element"][num]
        acc_or_dec[row][col] = acc_or_dec[row][col] + 1
    #
    return acc_or_dec

def judge(u):
    if u >= u_0:
        return 0 # accel
    elif u <= (-1 * u_0):
        return 2 # decel
    else:
        return 1 # const
    
def MAKE_TABLE(table):
    path_id = os.path.basename(path)
    name = os.path.splitext(path_id)[0]
    target_port = name.split("_classified")[0]
    #
    fig, ax = plt.subplots()
    ax.axis("off")
    #
    columnnames = [
        "Stop",
        "Going Straight",
        "Drift",
        "Moving Sideway",
        "Pivot Turning",
        "Turning"
    ]
    indexnames = [
        "acceleration",
        "constant",
        "deceleration"
    ]
    the_table = ax.table(
        cellText=table,
        colLabels=columnnames,
        rowLabels=indexnames,
        loc="center",
        cellLoc="center"
    )
    #
    fig.savefig(
        os.path.join(SAVE_DIR, f"{target_port}.png"),
        bbox_inches='tight',
        pad_inches=0.0
    )
    plt.close


if __name__ == '__main__':
    for path in paths:
        table = CLASSIFY(path)
        MAKE_TABLE(table)
    print("\nDone\n")
