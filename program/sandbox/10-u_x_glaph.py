import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = os.path.dirname(__file__)
TS_DIR = f"{DIR}/../../raw_datas/tmp"
TS_HEADER = [
    "date (JST)", "time (JST)", "latitude [deg]",
    "longitude [deg]", "GPS deg [deg]", "gyro deg [deg]",
    "GPS speed [knot]", "log speed [knot]", "wind dir [deg]", "wind sped [knot]"
]
#
def prepare():
    def convert_to_xy():
        pass
    
    paths = glob.glob(f"{TS_DIR}/_Yokkaichi_port*/*.csv")
    for path in paths:
        raw_df = pd.read_csv(
            path,
            skiprows=[0],
            encoding='shift-jis'
        )
        raw_df.head = TS_HEADER
        #



if __name__ == '__main__':
    df = prepare()
    #main(df)