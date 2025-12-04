import os
import matplotlib.pyplot as plt
import pandas as pd

DIR = os.path.dirname(__file__)
DATA = f"{DIR}/../../outputs/data"
# file = f"{DATA}/detail_map/Yokkaichi_port2B.csv"
file = f"{DATA}/rough_map/Kashima.csv"
SAVE_DIR = os.path.dirname(file)

df = pd.read_csv(file)
# plt.fill(df["x [m]"], df["y [m]"])
plt.plot(df["x [m]"], df["y [m]"])
plt.grid()
plt.show()

# plt.savefig(os.path.join(f"{SAVE_DIR}", "_kashima.png"))
print("Done\n")
