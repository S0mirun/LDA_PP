import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = os.path.dirname(__file__)
TS_DIR = f"{DIR}/../**/raw_data/888-送付データ/2-3_一次解析/"
SAVE_DIR = f"{DIR}/outputs/{os.path.splitext(os.path.basename(__file__))[0]}/"
os.makedirs(SAVE_DIR, exist_ok=True)
#
paths = glob.glob(f"{TS_DIR}/**/1-運動/*.xyz")  
#
colums = ['(1)count', '(2)JST(HH:MM:SS)', '(3)Lat:deg', '(4)Long:deg',
       '(5)GPSquality', '(6)Sog:Kont', '(7)Cog:deg', '(8)HDT(deg)',
       '(9)HDT_Cal:deg', '(10)ROT:deg/m ']
#
sigma = 5
exclude = [1, 14, 19, 23] # 一旦tab区切りを除外
bins = np.linspace(-0.1, 0.1, 41) 
#######  case22見る  #########
 
class CalculateAcceleration:
    def __init__(self):
        self.paths = paths
        self.SAVE_DIR = SAVE_DIR
        self.sigma = sigma
        self.exclude = exclude
        self.bins = bins
        #
        self.CASE_DIR = None
        self.path = None
        self.num = None
        self.raw_df = None
        self.lpf_df = None
        self.acc_df = None

    def main(self):
        for num, path in enumerate(self.paths, start=1):
            self.path = path
            self.num = num
            #
            self.read_raw_data()
            if num not in exclude:
                self.prepare_data()
                self.calculate_acceleration()
                self.make_glaphs()
                #
                print("case" + str(num) + "    complete")
            else:
                print("case" + str(num) + "    pass")
    
    def read_raw_data(self):
        raw_df = pd.read_csv(
            self.path,
            skiprows=[0],
            encoding="shift-jis",
        )
        self.raw_df = raw_df
        self.CASE_DIR = self.make_CASE_DIR()
        #save
        raw_df.to_csv(os.path.join(self.CASE_DIR, "1-raw_df.csv"))

    def make_CASE_DIR(self):
        basename = os.path.basename(self.path)
        date1 = re.search(r"\d{8}", basename)
        date2 = re.search(r"\d{4}_\d{4}", basename)

        if date1:
            date_str = date1.group()
        elif date2:
            date_str = date2.group().replace("_", "")
        else:
            date_str = "nodate"

        case_dir = os.path.join(self.SAVE_DIR, f"{date_str}_case{self.num}")
        os.makedirs(case_dir, exist_ok=True)
        return case_dir
    
    def prepare_data(self):
        pre_df = self.raw_df[['(2)JST(HH:MM:SS)', '(6)Sog:Kont']]
        pre_df.columns = ['JST(HH:MM:SS)', 'Sog:Kont']
        self.low_pass_filter(pre_df, 'Sog:Kont')
        #save
        self.lpf_df.to_csv(os.path.join(self.CASE_DIR, "1.1-LPF_data.csv"))

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
        df = self.lpf_df
        #original
        t = pd.to_datetime(df['JST(HH:MM:SS)'], format="%H:%M:%S", errors="coerce")
        t = (t - t.iloc[0]).dt.total_seconds().to_numpy()
        #
        u = df['Sog:Kont'].astype(float).to_numpy()
        #
        dy_dt = np.full_like(u, np.nan, dtype=float)
        dy_dt[0] = (u[1] - u[0]) / (t[1] - t[0])
        dy_dt[1:-1] = (u[2:] - u[:-2]) / (t[2:] - t[:-2])
        dy_dt[-1] = (u[-1] - u[-2]) / (t[-1] - t[-2])      
        #save
        acc_df = df.copy()
        acc_df['elapsed:s'] = t
        acc_df['accel:Konts/s'] = dy_dt
        acc_df.to_csv(os.path.join(self.CASE_DIR, "2-acc_df.csv"))
        #filtering
        self.low_pass_filter(acc_df, 'accel:Konts/s')
        #save
        self.lpf_df.to_csv(os.path.join(self.CASE_DIR, "2.1-LPF_acc_df.csv"))
        self.acc_df = self.lpf_df

    def make_glaphs(self):
        self.make_time_series()
        self.make_frequency_table()
        self.make_histogram()

    def make_time_series(self):
        df = self.acc_df
        #
        fig, axes = plt.subplots(2, 1, figsize = (10, 6))
        for ax in axes.flat: ax.grid()
        axes[0].plot(df.index, df['Sog:Kont'], color='skyblue')
        axes[0].plot(df.index, df['LPF_Sog:Kont'], color='red', alpha=0.8)
        axes[0].set_ylabel("SOG:Knot")
        #
        axes[1].plot(df.index, df['accel:Konts/s'], color='skyblue')
        axes[1].plot(df.index, df['LPF_accel:Konts/s'], color='red', alpha=0.8)
        axes[1].set_ylabel("acceleration:Kont/s")
        axes[1].set_ylim(-0.1, 0.1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.CASE_DIR, "3-time_series.png"))
        plt.close()

    def make_frequency_table(self):
        df = self.acc_df['accel:Konts/s']
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
        df = self.acc_df['accel:Konts/s']
        lpf_df = self.acc_df['LPF_accel:Konts/s']
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
        fig.set_xlabel("accel:Knot/s")
        fig.set_ylabel("frequency")
        fig.grid()
        #
        plt.savefig(os.path.join(self.CASE_DIR, "5-histogram.png"))
        fig.cla()

if __name__ == '__main__':
    CalAcc = CalculateAcceleration()
    CalAcc.main()
    print("complete")
