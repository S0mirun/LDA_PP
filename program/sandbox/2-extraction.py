import os
import glob
import pandas as pd

DIR = os.path.dirname(__file__)
TS_INFO_DIR = f"{DIR}/../**/raw_data/888-送付データ/2-3_一次解析/1_2023_0216_PCC_着/1-運動/"

data = glob.glob(f"{TS_INFO_DIR}/*.xyz")
#
raw_df = pd.read_csv(
    data[0],
    encoding = "shift-jis",
    skiprows = [0]
    )

print(raw_df)
#
col = ['(1)count', '(2)JST(HH:MM:SS)', '(3)Lat:deg', '(4)Long:deg',
       '(5)GPSquality', '(6)Sog:Kont', '(7)Cog:deg', '(8)HDT(deg)',
       '(9)HDT_Cal:deg', '(10)ROT:deg/m ']

df = raw_df[[col[1], col[5]]]
print(df)
#
SAVE_DIR = f"{DIR}/outputs"
save_name = "2-time_series.csv"

SAVE_PATH = os.path.join(SAVE_DIR, save_name)

df.to_csv(SAVE_PATH)

