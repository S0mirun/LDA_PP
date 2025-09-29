import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)
MAKE_GLAPH_DIR = glob.glob(f"{DIR}/**/*.csv")

print(MAKE_GLAPH_DIR)
#
df = pd.read_csv(
    MAKE_GLAPH_DIR[0],
    encoding = "shift-jis"
)

col = ['(1)count', '(2)JST(HH:MM:SS)', '(3)Lat:deg', '(4)Long:deg',
       '(5)GPSquality', '(6)Sog:Kont', '(7)Cog:deg', '(8)HDT(deg)',
       '(9)HDT_Cal:deg', '(10)ROT:deg/m ']

ax = plt.subplot()
ax.plot(df.index, df[col[5]])
ax.set_xlabel(col[1])
ax.set_ylabel(col[5])

SAVE_DIR = f"{DIR}/outputs"
save_name = "3-time_series.png"

SAVE_PATH = os.path.join(SAVE_DIR, save_name)

plt.savefig(SAVE_PATH)
plt.close()
print('figure saved')