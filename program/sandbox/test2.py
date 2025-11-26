import os
import matplotlib.pyplot as plt
import pandas as pd

DIR = os.path.dirname(__file__)
DATA = f"{DIR}/../../outputs/data"
file = f"{DATA}/detail_map/Osaka_port1A.csv"

df = pd.read_csv(file)
plt.fill_betweenx(df["y [m]"], df["x [m]"])
plt.show()