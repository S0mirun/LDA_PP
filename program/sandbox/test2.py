import os
import matplotlib.pyplot as plt
import pandas as pd

DIR = os.path.dirname(__file__)
DATA = f"{DIR}/../../outputs/data"
file = f"{DATA}/detail_map/Shimizu.csv"
# file = f"{DATA}/rough_map/Kamaishi.csv"
SAVE_DIR = os.path.dirname(file)

df = pd.read_csv(file)
plt.plot(df["y [m]"], df["x [m]"])
plt.grid()
plt.axis("equal")
plt.show()

# plt.savefig(os.path.join(f"{SAVE_DIR}",
#                          f"_{os.path.splitext(os.path.basename(file))[0]}.png"))
