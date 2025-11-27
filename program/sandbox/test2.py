import os
import matplotlib.pyplot as plt
import pandas as pd

DIR = os.path.dirname(__file__)
DATA = f"{DIR}/../../outputs/data"
# file = f"{DATA}/detail_map/Yokkaichi_port2B.csv"
file = f"{DATA}/detail_map/Osaka_port1A.csv"

df = pd.read_csv(file)
# plt.fill(df["x [m]"], df["y [m]"])
plt.plot(df["x [m]"], df["y [m]"])
plt.show()
