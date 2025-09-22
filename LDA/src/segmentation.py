
import numpy as np
import pandas as pd
import os
import pickle
import glob
from multiprocessing import Pool
from tqdm import tqdm

from utils.ship_geometry import transform_EF
from utils.time_series_figure import   \
    TimeSeries, make_traj_fig, make_ts_fig
from utils.visualization import Colors, save_fig
from utils.read_ts_info_table import read_ship_shape


#
N_PRCSS = 2
#
DT = 1.0  # [s]
#
port_name = [
    "_Osaka_port1A",
    "_Osaka_port1B",
    "_Tokyo_port2B",
    "_Tokyo_port2C",
    "_Yokkaichi_port1A",
    "_Yokkaichi_port2B"
]



class ProblemSetting:
    def __init__(self) -> None:
        # vq
        self.TARGET_ELEMENT = [
            "u [m/s]",
            "vm [m/s]",
        ]
        self.dim_vec = len(self.TARGET_ELEMENT)
        self.K_EACH = 5
        self.n_code = self.K_EACH ** self.dim_vec
        self.vq_id = f"dim_vec_{str(self.dim_vec)}_n_code_{str(self.n_code)}"
        self.vq_log_dir = f"./outputs/{target_port}/{self.vq_id}/"
        # segmentation
        self.PERIOD = 1  # [time step], in terms of DT, period of downsampling
        self.L_DOC = 100  # minimum of time steps of a segment, in terms of DT * self.PERIOD
        self.DELTA_TS_SHIFT = 100  # num of time steps shifted to the start of the next doc
        #
        self.trial_id = f"PERIOD_{self.PERIOD}_L_DOC_{self.L_DOC}_DELTA_{self.DELTA_TS_SHIFT}"
        self.log_dir = f"{self.vq_log_dir}sgmntd/{self.trial_id}/"
        #
        self.save()
    
    def save(self):
        os.makedirs(f"{self.log_dir}setting/", exist_ok=True)
        with open(f"{self.log_dir}setting/setting.bf", 'wb') as f:
            pickle.dump(self, f)
        with open(f"{self.log_dir}setting/setting.txt", mode="w") as f:
            for index_name in self.__dict__:
                f.write('{} : {}, \n'.format(index_name, self.__dict__[index_name]))


class Segmentation:
    def __init__(self, ps) -> None:
        self.ps = ps

    def main(self,):
        #
        os.makedirs(f"{self.ps.log_dir}csv/", exist_ok=True)
        pool = Pool(processes=N_PRCSS)
        paths = glob.glob(f"{self.ps.vq_log_dir}/encoded_ts/*.csv")
        with pool:
            n_sgmnts_l = list(tqdm(
                pool.imap(self.segment, paths),
                total=len(paths),
            ))
        # save number of segmented time series
        n_sgmnts = sum(n_sgmnts_l)
        with open(f"{self.ps.log_dir}info.txt", mode="w") as f:
            f.write(f"n_segments: {n_sgmnts}")
        print(f"\nall segmented time series saved: {n_sgmnts}\n")
    
    def segment(self, path):
        #
        n_sgmnts = 0
        #
        id_str = os.path.split(path)[1].split(".")[0]
        original_ts = pd.read_csv(path, index_col=0)
        df_down = original_ts[::self.ps.PERIOD]
        #
        sgmntd_ts_starts = range(
            0,
            len(df_down)-self.ps.L_DOC+1,
            self.ps.DELTA_TS_SHIFT,
        )  # for down-sampled df
        for start in sgmntd_ts_starts:
            start_str = str(start).zfill(5)
            sgmntd_ts = df_down[start : start+self.ps.L_DOC].copy()
            sgmntd_ts_f = self.format_sgmntd_df(sgmntd_ts)
            fname = f"{id_str}_start_{start_str}"
            sgmntd_ts_f.to_csv(f"{self.ps.log_dir}csv/{fname}.csv")
            n_sgmnts += 1
        #
        return n_sgmnts
    
    def format_sgmntd_df(self, sgmntd_df):
        #
        df = sgmntd_df.rename(
            columns = {
                "t [s]": "t (original) [s]",
                "p_x [m]": "p_x (original) [m]",
                "p_y [m]": "p_y (original) [m]",
                "gyro deg [deg]": "gyro deg (original) [deg]",
                "gyro deg [rad]": "gyro deg (original) [rad]",
            }
        )
        # time
        df["t [s]"] = np.linspace(0.0, self.ps.PERIOD*(len(df)-1), len(df))
        # position, heading
        df["p_x [m]"] = 0.0
        df["p_y [m]"] = 0.0
        df["gyro deg [rad]"] = 0.0
        Ox = df.iloc[0, df.columns.get_loc("p_x (original) [m]")]
        Oy = df.iloc[0, df.columns.get_loc("p_y (original) [m]")]
        theta = df.iloc[0, df.columns.get_loc("gyro deg (original) [rad]")]
        for i in range(len(df)):
            px = df.iloc[i, df.columns.get_loc("p_x (original) [m]")]
            py = df.iloc[i, df.columns.get_loc("p_y (original) [m]")]
            gyro_deg = df.iloc[i, df.columns.get_loc("gyro deg (original) [rad]")]
            p_nEF = transform_EF(px, py, gyro_deg, Ox, Oy, theta,)
            df.iloc[
                i,
                [
                    df.columns.get_loc("p_x [m]"),
                    df.columns.get_loc("p_y [m]"),
                    df.columns.get_loc("gyro deg [rad]"),
                ]
            ] = p_nEF
            df["gyro deg [deg]"] = np.rad2deg(df["gyro deg [rad]"].values)
        #
        return df


if __name__ == "__main__":
    for port in port_name:
        target_port = port
        print("port name:    " + str(target_port))
        #
        ps = ProblemSetting()
        s = Segmentation(ps=ps)
        #
        s.main()
    #
    print("\nDone\n")
