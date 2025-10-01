import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection


DIR = os.path.dirname(__file__)
CSV_DIR = f"{DIR}/../../../Path_Planning/output/elements/20250927"
SAVE_DIR = f"{DIR}/../../outputs/{os.path.splitext(os.path.basename(__file__))[0]}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
paths = glob.glob(f"{CSV_DIR}/*/*")
#
plot_color = ["black", "coral", "orange", "gold", "navy", "blue"]
elements = [
    "stop",
    "Going straight",
    "moving sideway",
    "Drift",
    "Pivot turning",
    "Turning",
]


"""
elements
    0 : stop
    1 : Going straight
    2 : moving sideway
    3 : Drift
    4 : Pivot Turning
    5 : Turning
"""

def calcurate_goal():
    goal_path = glob.glob(f"{DIR}/../../raw_datas/tmp/coordinates_of_port/*port*.csv")
    for path in goal_path:
        raw_df =pd.read_csv(
            path,
            encoding="shift-jis"
        )

def make_glaph():
    for num, path in enumerate(paths):
        target_port = os.path.basename(os.path.dirname(path))
        #
        df = pd.read_csv(
        path,
        encoding = "shift-jis"
        )
        #
        ax = plt.subplot()
        #
        x = df["time"].values
        y = df["u"].values
        elem = df["element"].values
        for e, g in df.groupby("element"):
            plt.scatter(g["time"], g["u"], color=plot_color[e], label=elements[e])
            points = np.column_stack([x, y])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            colors = [plot_color[e] for e in elem[:-1]]
            #
            lc = LineCollection(segments, colors=colors)
            plt.gca().add_collection(lc)
        
        ax.set_title({str(target_port)})
        # ax.set_title(f"""
        #              {str(target_port)}
        #              \ncsv:{os.path.splitext(os.path.basename(path))[0]}
        #              """)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("u (m/s)")
        ax.set_ylim(0, 6)
        ax.grid()
        ax.legend()
        #
        plt.savefig(os.path.join(f"{SAVE_DIR}", f"{num}{target_port}.png"))
        print(f"\n Figure Saved : {num}{target_port}\n")
        #
        ax.cla()


if __name__ == "__main__":
    calcurate_goal()
    make_glaph()
    print("\n Done \n")