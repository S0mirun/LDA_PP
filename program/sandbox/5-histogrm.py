import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

DIR = os.path.dirname(__file__)
TS_DIR = f"{DIR}/../**/raw_data/888-送付データ/2-3_一次解析/"
TS_DIR_PATH = f"{TS_DIR}/1_2023_0216_PCC_着/1-運動/"
SAVE_DIR = f"{DIR}/outputs/"

data = glob.glob(f"{SAVE_DIR}/4.3*.csv")

def prepare(data):
    raw_df = pd.read_csv(
        data[0],
        encoding="shift-jis"
    )
    df = raw_df["(11)acceleration:Kont/s"]
    return df

def Make_frequency_Table(df, stur=False):
    if stur is True:
        bins = sturges_formula(len(df))
    else:
        bins = np.arange(-2, 3, 0.025)

    hist, bins = np.histogram(df, bins=bins)
    #
    df_freq = pd.DataFrame(
        {
            'or more': bins[:-1],
            'or less': bins[1:],
            'midpoint': (bins[:-1] + bins[1:]) / 2,
            'frequency': hist
        }
    )
    #
    save_name = "5-frequency_table.csv"
    SAVE_PATH = os.path.join(SAVE_DIR, save_name)
    df_freq.to_csv(SAVE_PATH)

def Make_Histogram(df, stur=False):
    if stur is True:
        bins = sturges_formula(len(df))
    else:
        bins = np.arange(-2, 3, 0.025)
    #
    fig = plt.subplot()
    fig.hist(
        df, rwidth=1.6,
        orientation='vertical',
        histtype="stepfilled",
        range=(df.min(), df.max()),
        bins=bins
    )
    fig.set_xlabel("acceletarion:Knot/s")
    fig.set_ylabel("frequency")
    fig.grid()
    #
    save_name = "5.1-histogram.png"
    SAVE_PATH = os.path.join(SAVE_DIR, save_name)
    plt.savefig(SAVE_PATH)

def sturges_formula(data_size):
    return int(round(1+np.log2(data_size), 0))



if __name__ == "__main__":
    df = prepare(data)
    Make_frequency_Table(df, stur=False)
    Make_Histogram(df, stur=False)
