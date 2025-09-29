import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DIR = os.path.dirname(__file__)
TS_DIR = f"{DIR}/../**/raw_data/888-送付データ/2-3_一次解析/"
TS_DIR_PATH = f"{TS_DIR}/17-20230830_180820_コンテナ_離/1-運動/"
SAVE_DIR = f"{DIR}/outputs/"

col = ['(1)count', '(2)JST(HH:MM:SS)', '(3)Lat:deg', '(4)Long:deg',
       '(5)GPSquality', '(6)Sog:Kont', '(7)Cog:deg', '(8)HDT(deg)',
       '(9)HDT_Cal:deg', '(10)ROT:deg/m ']


data = glob.glob(f"{TS_DIR_PATH}/*.xyz")
df = pd.read_csv(
    data[0],
    skiprows=[0],
    encoding = "shift-jis",
)
sigma = 3

def LPF_GC(df,sigma):
    x = df['(6)Sog:Kont']
    sigma_k = sigma 
    kernel = np.zeros(int(round(3*sigma_k))*2+1)
    for i in range(kernel.shape[0]):
        kernel[i] =  1.0/np.sqrt(2*np.pi)/sigma_k * np.exp((i - round(3*sigma_k))**2/(- 2*sigma_k**2))
        
    kernel = kernel / kernel.sum()
    x_long = np.zeros(x.shape[0] + kernel.shape[0])
    x_long[kernel.shape[0]//2 :-kernel.shape[0]//2] = x
    x_long[:kernel.shape[0]//2 ] = x.iloc[0]
    x_long[-kernel.shape[0]//2 :] = x.iloc[-1]
        
    x_GC = np.convolve(x_long,kernel,'same')
    df['filtered_Sog'] = x_GC[kernel.shape[0]//2 :-kernel.shape[0]//2]
    df.to_csv(os.path.join(SAVE_DIR, "7-LPF_data.csv"))
    return df


def make_time_series(df):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df[col[5]], color='skyblue')
    ax.plot(df.index, df['filtered_Sog'], color='red')
    ax.set_ylabel('filtered_Sog')

    SAVE_DIR = f"{DIR}/outputs"
    save_name = "7-remake_time_series.png"
    plt.savefig(os.path.join(SAVE_DIR, save_name))
    plt.close()

if __name__ == '__main__':
    pre_df = LPF_GC(df,sigma)
    #print(pre_df)
    make_time_series(pre_df)
    
