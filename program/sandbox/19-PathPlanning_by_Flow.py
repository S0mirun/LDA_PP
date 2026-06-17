import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.PP.Flow import *
from path_planning.PathPlanning import convert

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

port_number: int = 8
port = dictionary()[port_number]
# 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Sakaide, 4: Osaka_1B
# 5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu
# 10: Tomakomai, 11: KIX

Q = -10e5
Gamma = Q * (-0.5)
margin = 500
interval = 100

xmin, xmax = port["ver_range"]
ymin, ymax = port["hor_range"]

x_range = (xmin - margin, xmax + margin)
y_range = (ymin - margin, ymax + margin)

x = np.arange(x_range[0], x_range[1] + interval, interval)
y = np.arange(y_range[0], y_range[1] + interval, interval)

X, Y = np.meshgrid(x, y)
U = np.zeros_like(X)
V = np.zeros_like(Y)

port_csv=f"raw_datas/tmp/coordinates_of_port/_{port["name"]}.csv"
df_buoy = pd.read_csv(f"outputs/data/buoy/{port['name']}.csv")
df_buoy["x [m]"], df_buoy["y [m]"] = convert(df_buoy, port_csv, "latitude", "longitude")
df_buoy["COLOUR"] = df_buoy["COLOUR"].astype(str).str.strip()

flows = []
source_params = [
    {"z0": 0 + 0j,    "Q": Q}, # berth
]
vortex_params = []
for x, y, c in zip(df_buoy["x [m]"], df_buoy["y [m]"], df_buoy["COLOUR"]):
    vortex_params.append(
        {"z0": y + 1j*x,    "Gamma": Gamma if c==str(4) else -Gamma},
    )

flows = set_flows(flows, source_params, vortex_params)

for i in range(X.shape[0]):
    for k in range(X.shape[1]):
        z = X[i, k] + 1j * Y[i, k]
        velocity = sum(flow(z) for flow in flows)

        U[i, k] = velocity.real
        V[i, k] = velocity.imag

fig, ax = plt.subplots(figsize=(7,7))
ax.quiver(Y, X, V, U, 
            angles='xy', scale_units='xy', scale=4.5)

for p in vortex_params:
    z0 = p["z0"]
    Gamma = p["Gamma"]

    color = "green" if Gamma > 0 else "red"
    ax.scatter(z0.imag, z0.real, 
            color=color, s=10, marker="o", zorder=5)

ax.set_xlim(port["hor_range"])
ax.set_ylim(port["ver_range"])
ax.set_aspect("equal")
ax.grid(True)
ax.tick_params(
    axis="both",
    labelbottom=False,
    labelleft=False
)
plt.savefig(os.path.join(SAVE_DIR, f"{port['name']}.png"), 
            dpi=400, bbox_inches="tight", pad_inches=0.05)

print("\n finished \n")