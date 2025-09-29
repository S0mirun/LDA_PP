
import os
import pandas as pd


#
DIR = os.path.dirname(__file__)
TS_INFO_TABLE_PATH = f"{DIR}/../../raw_data/ts_info_table.xlsx"

def read_ship_shape(i):
    ts_info_df = pd.read_excel(TS_INFO_TABLE_PATH, header=0)
    ts_info_series = ts_info_df[ts_info_df["No."]==i]
    L = ts_info_series["L"].values[0]
    B = ts_info_series["B"].values[0]
    return L, B
