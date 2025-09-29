
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import re
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import unicodedata

from utils.LDA.ship_geometry import *
from utils.LDA.time_series_figure  \
    import TimeSeries, make_traj_fig, make_ts_fig, make_traj_and_velo_fig
from utils.LDA.visualization import *
from utils.LDA.kml import kml_based_txt_to_csv


DIR = os.path.dirname(__file__)
RAW_TS_DIR = f"{DIR}/../../tmp/"
TS_HEADER = [
    "date (JST)", "time (JST)", "latitude [deg]",
    "longitude [deg]", "GPS deg [deg]", "gyro deg [deg]",
    "GPS speed [knot]", "log speed [knot]", "wind dir [deg]", "wind sped [knot]"
]
#
BLANK_IDS = [15, 16,]
n_cols = 2


def preprocess():
    #
    for dir in glob.glob(f"{RAW_TS_DIR}/*[A-C]"):
        #
        for path in glob.glob(f"{dir}/*"):
            csv_path = path
            ts_id = os.path.basename(csv_path)
            name= os.path.splitext(ts_id)[0]
            date, num, port, port_num, rest = name.split("_", 4)
            ts_id_str = f"{port}_{port_num}_{date}_{num}_{rest}.csv"
            raw_df = pd.read_csv(
                csv_path,
                skiprows=[0,1],
                encoding='shift-jis'
            )
            #
            raw_df.columns = TS_HEADER
            raw_df["latitude [deg]"] = raw_df["latitude [deg]"].map(convert_coordinate)
            raw_df["longitude [deg]"] = raw_df["longitude [deg]"].map(convert_coordinate)
            #
            # basename = os.path.basename(dir)
            # save_dir = os.path.join(f"{DIR}/../../outputs", basename)
            # os.makedirs(save_dir, exist_ok=True)
            # raw_df.to_csv(os.path.join(save_dir, os.path.basename(csv_path)))
            #print(raw_df.dtypes)

            df = prepare_df(raw_df)
            log_dir = f"{DIR}/ts_data/original/"
            os.makedirs(f"{log_dir}csv/", exist_ok=True)
            df.to_csv(os.path.join(f"{log_dir}/csv/", f"{ts_id_str}"))
            #
            ts = TimeSeries(
                df=df,
                label=ts_id_str, L=100, B=16,
                color=Colors.black, line_style=(0, (1, 0)),
                dt=1.0,
            )
            make_traj_fig(
                ts_list=[ts],
                ship_plot_step_period=100, alpha_ship_shape=0.5,
                fig_size=(5, 5), legend_flag=True,
            )
            save_fig(f"{log_dir}/fig/traj/", f"{ts_id_str}_traj",)
            make_ts_fig(ts_list=[ts], fig_size=(10, 5,))
            save_fig(f"{log_dir}/fig/state/", f"{ts_id_str}_state",)
            make_traj_and_velo_fig(
                ts_list=[ts],
                ship_plot_step_period=100, alpha_ship_shape=0.5,
                fig_size=(14, 7)
            )
            save_fig(f"{log_dir}/fig/traj_and_velo/", f"{ts_id_str}_traj_and_velo",)

def convert_coordinate(value):
    if value is None or value == '':
        return float('nan')
    value = unicodedata.normalize("NFKC", str(value))
    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    if not numbers:
        return float('nan')
    deg = float(numbers[0])
    if len(numbers) > 1:
        minutes = float(numbers[1])
        return deg + minutes / 60.0
    return deg


def prepare_df(raw_df):
    #
    df = raw_df.copy()
    #
    smoothen_latlon(df)
    #
    time_arr = np.empty(len(df))
    p_x_arr = np.empty(len(df))
    p_y_arr = np.empty(len(df))
    time_origin = JST_str_to_float(df.iloc[0, df.columns.get_loc("time (JST)")])
    lat_origin = df.iloc[-1, df.columns.get_loc("latitude [deg]")]
    lon_origin = df.iloc[-1, df.columns.get_loc("longitude [deg]")]
    angle_from_north = 0.0
    for i in range(len(df)):
        #
        time_arr[i] = JST_str_to_float(df.iloc[i, df.columns.get_loc("time (JST)")]) - time_origin
        #
        p_x_temp, p_y_temp = convert_to_xy(
            df.iloc[i, df.columns.get_loc("latitude [deg]")],
            df.iloc[i, df.columns.get_loc("longitude [deg]")],
            lat_origin, lon_origin, angle_from_north
        )
        p_x_arr[i] = p_x_temp
        p_y_arr[i] = p_y_temp
    #
    df["t [s]"] = time_arr
    df["p_x [m]"] = p_x_arr
    df["p_y [m]"] = p_y_arr
    #
    df["GPS deg [rad]"] = np.deg2rad(df["GPS deg [deg]"].values)
    df["gyro deg [rad]"] = np.deg2rad(df["gyro deg [deg]"].values)
    #
    df["U [m/s]"] = knot_to_ms(df["GPS speed [knot]"])
    df["beta [rad]"] = clip_angle(df["gyro deg [rad]"] - df["GPS deg [rad]"])
    df["beta [deg]"] = np.rad2deg(df["beta [rad]"].values)    
    df["u [m/s]"] = df["U [m/s]"].values * np.cos(df["beta [rad]"].values)
    df["vm [m/s]"] = -df["U [m/s]"].values * np.sin(df["beta [rad]"].values)
    #
    smoothen_ts_1D(df, "t [s]", 1e2)
    #
    return df

def smoothen_ts_1D(df, label, thr):
    ts_diffs = df[label].diff()
    for i in range(len(df)-1):
        diff = ts_diffs.iloc[i+1]
        if np.abs(diff) > thr:
            smoothen = 0.5 * (
                df.iloc[i, df.columns.get_loc(label)]
                + df.iloc[i+2, df.columns.get_loc(label)]
            )
            df.iloc[i+1, df.columns.get_loc(label)] = smoothen

def smoothen_latlon(df):
    lat_diffs = df["latitude [deg]"].diff()
    lon_diffs = df["longitude [deg]"].diff()
    for i in range(len(df)-1):
        lat_diff = lat_diffs.iloc[i+1]
        lon_diff = lon_diffs.iloc[i+1]
        if np.abs(lat_diff) > 1.0:
            lat_smoothen = 0.5 * (
                df.iloc[i, df.columns.get_loc("latitude [deg]")]
                + df.iloc[i+2, df.columns.get_loc("latitude [deg]")]
            )
            df.iloc[i+1, df.columns.get_loc("latitude [deg]")] = lat_smoothen
        if np.abs(lon_diff) > 1.0:
            lon_smoothen = 0.5 * (
                df.iloc[i, df.columns.get_loc("longitude [deg]")]
                + df.iloc[i+2, df.columns.get_loc("longitude [deg]")]
            )
            df.iloc[i+1, df.columns.get_loc("longitude [deg]")] = lon_smoothen

def JST_str_to_float(str):
    l = str.split(":")
    t = float(l[0]) * 3600.0 + float(l[1]) * 60.0 + float(l[2])
    return t





if __name__ == "__main__":
    preprocess()
    #
    print("\nDone\n")
