import pandas as pd
import os
import glob
import matplotlib.pyplot as plt


DIR = os.path.dirname(__file__)
TS_DIR = f"{DIR}/../**/raw_data/888-送付データ/2-3_一次解析/"
TS_DIR_PATH = f"{TS_DIR}/1_2023_0216_PCC_着/1-運動/"
SAVE_DIR = f"{DIR}/outputs/{os.path.splitext(os.path.basename(__file__))[0]}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
col = ['(1)count', '(2)JST(HH:MM:SS)', '(3)Lat:deg', '(4)Long:deg',
       '(5)GPSquality', '(6)Sog:Kont', '(7)Cog:deg', '(8)HDT(deg)',
       '(9)HDT_Cal:deg', '(10)ROT:deg/m ']



data = glob.glob(f"{TS_DIR_PATH}/*.xyz")
raw_df = pd.read_csv(
    data[0],
    skiprows = [0],
    encoding = "shift-jis"
)

csv_name = "raw_df.csv"
SAVE_PATH = os.path.join(SAVE_DIR, csv_name)

raw_df.to_csv(SAVE_PATH)
print("raw dara flame saved")
#
df = raw_df[['(2)JST(HH:MM:SS)', '(6)Sog:Kont']]

csv_name = "time_series.csv"
SAVE_PATH = os.path.join(SAVE_DIR, csv_name)

df.to_csv(SAVE_PATH)

print("time series saved")
#
df = df.copy()
accel = df['(6)Sog:Kont'].diff()
df['(11)acceleration:Kont/s'] = accel

csv_name = "acceleration.csv"
SAVE_PATH = os.path.join(SAVE_DIR, csv_name)

df.to_csv(SAVE_PATH)
print("acceleration csv saved")
#
fig,axes = plt.subplots(3, 1, figsize = (10, 9))
for ax in axes.flat: ax.grid()
axes[0].plot(df.index, df['(6)Sog:Kont'])
axes[0].set_ylabel("SOG:Knot")
#
axes[1].plot(df.index, df['(11)acceleration:Kont/s'])
axes[1].set_ylabel("acceleration:Kont/s")
#
axes[2].plot(df.index, df['(11)acceleration:Kont/s'])
axes[2].set_ylabel("acceleration:Kont/s")
axes[2].set_ylim(-0.1,0.1)


plt.tight_layout()

save_name = "4-time_series.png"
SAVE_PATH = os.path.join(SAVE_DIR, save_name)
plt.savefig(SAVE_PATH)
plt.close()

print('figure saved')