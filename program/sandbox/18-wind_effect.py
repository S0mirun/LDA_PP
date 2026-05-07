import os
import glob

import numpy as np
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon as MplPolygon
import pandas as pd
from scipy import ndimage

from path_planning.PathPlanning import convert
from utils.PP.dictionary_of_port import dictionary
from utils.PP.MultiPlot import RealTraj


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]

port_number: int = 10
port = dictionary()[port_number]
# 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Sakaide, 4: Osaka_1B
# 5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu
# 10: Tomakomai, 11: KIX

port_csv = None
SAVE_DIR = None

def preset(fig, ax):
    set_target_port()
    setup_save_dir()
    draw_basemap(fig, ax)


def set_target_port():
    global port_csv
    port_csv=f"raw_datas/tmp/coordinates_of_port/_{port["name"]}.csv"
    print(f"\ntarget : {port["name"]}")


def setup_save_dir():
    global SAVE_DIR
    SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
    os.makedirs(SAVE_DIR, exist_ok=True)


def setup_figure():
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(port["hor_range"])
    ax.set_ylim(port["ver_range"])
    ax.set_aspect("equal")
    ax.grid(True)
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

    print("\nFigure setup complete")

    return fig, ax


def draw_basemap(fig, ax):
    _draw_land(fig, ax)
    _draw_shipping_lane(fig, ax)
    _draw_buoy(fig, ax)
    _add_compass_image(fig, ax)
    _save_fig(fig, "basemap")
    print("\nDraw basemap complete")


def _draw_land(fig, ax):
    df_land = pd.read_csv(f"outputs/data/detail_map/{port["name"]}.csv")

    map_X, map_Y = df_land["x [m]"].values, df_land["y [m]"].values
    ax.fill_betweenx(map_X, map_Y, facecolor="gray", alpha=0.3, zorder=0)
    ax.plot(map_Y, map_X, color="k", linestyle="--", lw=0.5, alpha=0.8, zorder=0)


def _draw_shipping_lane(fig, ax):
    df_shipping_lane = pd.read_csv(f"outputs/data/Shipping_lane/{port["name"]}.csv")
    df_lane = df_shipping_lane.copy()
    df_lane["x [m]"], df_lane["y [m]"] = convert(df_lane, port_csv)

    for pid, g in df_lane.groupby("polygon_id", sort=True):
        xy = g[["x [m]", "y [m]"]].to_numpy(float)
        xy = np.vstack([xy, xy[0]])  # close
        patch = MplPolygon(
            xy,
            closed=True,
            fill=True,
            facecolor='magenta',
            alpha=0.2, zorder=1
        )
        ax.add_patch(patch)


def _draw_buoy(fig, ax):
    df_buoy = pd.read_csv(f"outputs/data/buoy/{port['name']}.csv")
    df_buoy["x [m]"], df_buoy["y [m]"] = convert(df_buoy, port_csv, "latitude", "longitude")

    ax.scatter(df_buoy["x [m]"].values, df_buoy["y [m]"].values,
                color='orange', s=20, zorder=2)
    _draw_buoy_color(fig, ax, df_buoy)


def _draw_buoy_color(fig, ax, df_buoy):
    df_buoy["COLOUR"] = df_buoy["COLOUR"].astype(str).str.strip()
    COLOR = ["white", "black", "red", "green", "blue", "yellow"]
    for i in range(1, 7):
        ax.scatter(
            df_buoy.loc[df_buoy["COLOUR"] == str(i), "x [m]"].values,
            df_buoy.loc[df_buoy["COLOUR"] == str(i), "y [m]"].values,
            color=COLOR[i-1], s=10, zorder=3)
        

def _add_compass_image(fig, ax):
    img = mpimg.imread("raw_datas/compass icon2.png")
    df = pd.read_csv(port_csv)
    angle = float(df['Psi[deg]'].iloc[0])
    img_rot = ndimage.rotate(img, angle, reshape=True)
    img_rot = np.clip(img_rot, 0.0, 1.0)
    imagebox = OffsetImage(img_rot, zoom=0.3)
    ab = AnnotationBbox(
        imagebox,
        (0, 1),
        xycoords='axes fraction',
        box_alignment=(0, 1),
        frameon=False,
        pad=0.0,
    )
    ax.add_artist(ab)


def _save_fig(fig, name):
    fig.savefig(os.path.join(SAVE_DIR, f"{name}_{port['name']}.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    

def draw_captain_path(fig, ax):
    df_captain = glob.glob(f"raw_datas/tmp/_{port['name']}/*.csv")
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    for i, df in enumerate(df_captain):
        traj = RealTraj()
        traj.input_csv(df, port_csv)

        points = np.array([traj.Y, traj.X]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, 
                            linewidth=1.0, alpha=0.6, zorder=3)
        lc.set_array(traj.windv[:-1])
        ax.add_collection(lc)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, shrink=0.8, ax=ax)
    cbar.set_label("Wind speed [m/s]")
    cbar.set_ticks([0, 2, 4, 6, 8, 10])
    
    _save_fig(fig, "captain_path")
    

if __name__ == '__main__':
    fig, ax = setup_figure()
    preset(fig, ax)
    draw_captain_path(fig, ax)
