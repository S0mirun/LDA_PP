import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = os.path.dirname(__file__)
TS_DIR = f"{DIR}/raw_datas/tmp/*"
SAVE_DIR = f"{DIR}/outputs/{os.path.splitext(os.path.basename(__file__))[0]}/"
os.makedirs(SAVE_DIR, exist_ok=True)
#
HEADER = [
    "date (JST)", "time (JST)", "latitude [deg]",
    "longitude [deg]", "GPS deg [deg]", "gyro deg [deg]",
    "GPS speed [knot]", "log speed [knot]", "wind dir [deg]", "wind sped [knot]"
]
#
paths = glob.glob(f"{TS_DIR}/2*.csv")  

class CalculateAcceleration:
    def __init__(self):
        self.paths = paths
        self.SAVE_DIR = SAVE_DIR
        #
        self.path = None
        self.target_port = None
        self.raw_df = None
        self.pre_df = None
        #self.lpf_df = None
        self.acc_df = None

    def main(self):
        for path in self.paths:
            self.path = path
            #
            self.read_raw_data()
            self.prepare_data()
            self.calculate_acceleration()
            self.make_glaphs()
            #
            print(f"\ncomplete:   {self.target_port}\n")
    
    def read_raw_data(self):
        raw_df = pd.read_csv(
            self.path,
            skiprows=[0],
            encoding="shift-jis",
        )
        raw_df.columns = HEADER
        self.raw_df = raw_df
        CASE_DIR = self.make_CASE_DIR("raw_df")
        #save
        raw_df.to_csv(os.path.join(CASE_DIR, f"{self.target_port}.csv"))

    def make_CASE_DIR(self, dirname):
        path = self.path
        path_id = os.path.basename(path)
        name = os.path.splitext(path_id)[0]
        target_port = name.split("_classified")[0]
        self.target_port = target_port

        case_dir = os.path.join(self.SAVE_DIR, dirname)
        os.makedirs(case_dir, exist_ok=True)
        return case_dir
    
    def prepare_data(self):
        df = self.raw_df
        #
        t = df["time (JST)"]
        u = df["GPS speed [knot]"] * 1852 / 3600
        pre_df = pd.DataFrame({
            "time (JST)" : t,
            "u(m/s)" : u
        })
        self.pre_df = pre_df

    def low_pass_filter(self, df, col):
        sog = df[col]
        sigma = self.sigma
        #Gaussian convolution
        kernel = np.zeros(int(round(3*sigma))*2+1)
        for i in range(kernel.shape[0]):
            kernel[i] =  1.0/np.sqrt(2*np.pi)/sigma * np.exp((i - round(3*sigma))**2/(- 2*sigma**2))
            
        kernel = kernel / kernel.sum()
        sog_long = np.zeros(df.shape[0] + kernel.shape[0])
        sog_long[kernel.shape[0]//2 :-kernel.shape[0]//2] = sog
        sog_long[:kernel.shape[0]//2 ] = sog.iloc[0]
        sog_long[-kernel.shape[0]//2 :] = sog.iloc[-1]
            
        x_GC = np.convolve(sog_long,kernel,'same')

        lpf_df = df.copy()
        lpf_df['LPF_' + col] = x_GC[kernel.shape[0]//2 :-kernel.shape[0]//2]
        self.lpf_df = lpf_df

    def calculate_acceleration(self):
        df = self.pre_df
        #original
        t = pd.to_datetime(df["time (JST)"], format="%H:%M:%S", errors="coerce")
        t = (t - t.iloc[0]).dt.total_seconds().to_numpy()
        #
        u = df["u(m/s)"].astype(float).to_numpy()
        #
        dy_dt = np.full_like(u, np.nan, dtype=float)
        dy_dt[0] = (u[1] - u[0]) / (t[1] - t[0])
        dy_dt[1:-1] = (u[2:] - u[:-2]) / (t[2:] - t[:-2])
        dy_dt[-1] = (u[-1] - u[-2]) / (t[-1] - t[-2])      
        #save
        acc_df = df.copy()
        acc_df['elapsed:s'] = t
        acc_df['accel:m/s^2'] = dy_dt
        CASE_DIR = self.make_CASE_DIR("acc_df")
        acc_df.to_csv(os.path.join(CASE_DIR, f"{self.target_port}.csv"))
        self.acc_df = acc_df

    def make_glaphs(self):
        self.make_time_series()
        # self.make_frequency_table()
        # self.make_histogram()

    def make_time_series(self):
        df = self.acc_df
        #
        path_name = os.path.splitext(os.path.basename(self.path))[0]
        target_port = re.search(r'([A-Za-z]+_port\d+[A-Za-z]?)', path_name).group(1)
        #
        fig, axes = plt.subplots(2, 1, figsize = (10, 6))
        plt.title(f"{target_port}")
        for ax in axes.flat: ax.grid()
        axes[0].plot(df.index, df["u(m/s)"])
        axes[0].set_xlabel("time [min]")
        axes[0].set_ylabel("u(m/s)")
        #
        axes[1].plot(df.index, df['accel:m/s^2'])
        axes[1].set_xlabel("time [min]")
        axes[1].set_ylabel("acceleration(m/s^2)")
        #
        plt.tight_layout()
        CASE_DIR = self.make_CASE_DIR("time_series")
        plt.savefig(os.path.join(CASE_DIR, f"{self.target_port}.png"))
        plt.close()

    def make_frequency_table(self):
        df = self.acc_df['accel:Konts/s^2']
        #
        hist, bins = np.histogram(df, bins=self.bins)
        #
        freq_df = pd.DataFrame(
            {
                'or more': bins[:-1],
                'or less': bins[1:],
                'midpoint': (bins[:-1] + bins[1:]) / 2,
                'frequency': hist
            }
        )
        #
        freq_df.to_csv(os.path.join(self.CASE_DIR, "4-freq_table.csv"))

    def make_histogram(self):
        df = self.acc_df['accel:Konts/s^2']
        lpf_df = self.acc_df['LPF_accel:Konts/s^2']
        #
        fig = plt.subplot()
        fig.hist(
            df,
            rwidth=0.9,
            weights = np.ones_like(df) / len(df),
            orientation='vertical',
            histtype="stepfilled",
            color='skyblue',
            alpha=0.5,
            bins=self.bins
        )
        fig.hist(
            lpf_df,
            rwidth=0.9,
            weights = np.ones_like(df) / len(df),
            orientation='vertical',
            histtype="stepfilled",
            color='red',
            alpha=0.5,
            bins=self.bins
        )
        fig.set_xlabel("accel:Knot/s^2")
        fig.set_ylabel("frequency")
        fig.grid()
        #
        plt.savefig(os.path.join(self.CASE_DIR, "5-histogram.png"))
        fig.cla()

if __name__ == '__main__':
    CalAcc = CalculateAcceleration()
    CalAcc.main()
    print("\n   Done    \n")
