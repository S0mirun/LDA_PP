import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from utils.LDA.ship_geometry import *

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

land_csv = "outputs/MakeMap/Yokkaichi/outline_land_only/land_only_outline_vertices_latlon.csv"
no_go_csv = "outputs/MakeMap/Yokkaichi/outline_impassable/impassable_outline_vertices_latlon.csv"
magenta_csv = "outputs/MakeMap/Yokkaichi/outline_magenta/magenta_outline_vertices_latlon.csv"
port_csv = "raw_datas/tmp/coordinates_of_port/_Yokkaichi.csv"

def convert(df):
    df_coord = pd.read_csv(port_csv)
    LAT_ORIGIN = df_coord["Latitude"].iloc[0]
    LON_ORIGIN = df_coord["Longitude"].iloc[0]
    # ANGLE_FROM_NORTH = df_coord["Psi[deg]"].iloc[0]
    ANGLE_FROM_NORTH = 0

    xs = []; ys = []
    for lat, lon in zip(df["lat"].to_numpy(), df["lon"].to_numpy()):
        if pd.isna(lat) or pd.isna(lon):
            continue
        y, x = convert_to_xy(
            lat,
            lon,
            LAT_ORIGIN,
            LON_ORIGIN,
            ANGLE_FROM_NORTH,
        )
        xs.append(x); ys.append(y)
    
    return xs, ys

df_land = pd.read_csv(land_csv)
df_nogo = pd.read_csv(no_go_csv)
df_magenta = pd.read_csv(magenta_csv)

# convert
df_land["x [m]"], df_land["y [m]"] = convert(df_land)
df_nogo["x [m]"], df_nogo["y [m]"] = convert(df_nogo)
df_magenta["x [m]"], df_magenta["y [m]"] = convert(df_magenta)

fig, ax = plt.subplots(figsize=(8, 8))
for pid in df_nogo["polygon_id"].unique():
    sub = df_nogo[df_nogo["polygon_id"] == pid]

    xy = sub[["x [m]", "y [m]"]].to_numpy()

    sea = Polygon(xy, closed=True, facecolor="skyblue", alpha=0.3)
    ax.add_patch(sea)

for pid in df_magenta["polygon_id"].unique():
    sub = df_magenta[df_magenta["polygon_id"] == pid]

    xy = sub[["x [m]", "y [m]"]].to_numpy()

    go = Polygon(xy, closed=True, facecolor="magenta")
    ax.add_patch(go)

for pid in df_land["polygon_id"].unique():
    sub = df_land[df_land["polygon_id"] == pid]

    xy = sub[["x [m]", "y [m]"]].to_numpy()

    land = Polygon(xy, closed=True, facecolor="0.7")
    ax.add_patch(land)

ax.set_aspect("equal", adjustable="box")
ax.relim()
ax.autoscale_view()
plt.show()