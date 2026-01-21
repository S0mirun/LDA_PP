import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from utils.LDA.ship_geometry import *
from utils.PP.MultiPlot import RealTraj

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]

num = 2
L = 103.8; B = 16.0

class Setting:
    def __init__(self):
        # port
        self.port_number: int = 2

        self.L = 103.8
        self.B = 16.0

def convert(df, port_file, col_lat="lat", col_lon="lon"):
    df_coord = pd.read_csv(port_file)
    LAT_ORIGIN = df_coord["Latitude"].iloc[0]
    LON_ORIGIN = df_coord["Longitude"].iloc[0]
    ANGLE_FROM_NORTH = df_coord["Psi[deg]"].iloc[0]

    xs = []; ys = []
    for lat, lon in zip(df[col_lat].to_numpy(), df[col_lon].to_numpy()):
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

class MakeLine:
    def __init__(self, ps):
        self.ps = ps

    def main(self):
        self.prepare()


    def prepare(self):
        self.read_csv()
        self.init_fig()
        self.draw_basemap()
        self.make_init_route()


    def read_csv(self):
        self.port = self.dict_of_port(self.ps.port_number)
        SAVE_DIR = f"{DIR}/../../outputs/{dirname}/{self.port["name"]}"
        os.makedirs(SAVE_DIR, exist_ok=True)
        # Data Frames
        ## port
        port_csv=f"raw_datas/tmp/coordinates_of_port/_{self.port["name"]}.csv"

        ## map
        detail_csv = f"outputs/data/detail_map/{self.port["name"]}.csv"
        df_map = pd.read_csv(detail_csv)

        ## shipping lane
        lane_csv = f"outputs/data/Shipping_lane/{self.port["name"]}.csv"
        df_lane = pd.read_csv(lane_csv)
        df_lane["x [m]"], df_lane["y [m]"] = convert(df_lane, port_csv)

        ## buoy
        buoy_csv = f"outputs/data/buoy/{self.port['name']}.csv"
        df_buoy = pd.read_csv(buoy_csv)
        df_buoy["x [m]"], df_buoy["y [m]"] = convert(df_buoy, port_csv, "latitude", "longitude")

        ## captain's loute
        df_cap = glob.glob(f"raw_datas/tmp/_{self.port['name']}/*.csv")

        self.port_csv = port_csv
        self.df_map = df_map
        self.df_lane = df_lane
        self.df_buoy = df_buoy
        self.df_cap = df_cap
        self.SAVE_DIR = SAVE_DIR


    def init_fig(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))


    def draw_basemap(self):
        port = self.port
        fig, ax = self.fig, self.ax
        # map
        map_X, map_Y = self.df_map["x [m]"].values, self.df_map["y [m]"].values
        ax.fill_betweenx(map_X, map_Y, facecolor="gray", alpha=0.3)
        ax.plot(map_Y, map_X, color="k", linestyle="--", lw=0.5, alpha=0.8)

        # shipping lane
        for pid, g in self.df_lane.groupby("polygon_id", sort=True):
            xy = g[["x [m]", "y [m]"]].to_numpy(float)
            xy = np.vstack([xy, xy[0]])  # close
            patch = Polygon(
                xy,
                closed=True,
                fill=True,
                facecolor='magenta',
                alpha=0.2
            )
            ax.add_patch(patch)
        
        # buoy
        ax.scatter(self.df_buoy["x [m]"].values, self.df_buoy["y [m]"].values,
                    color='orange', s=20, zorder=4)
        legend_buoy = plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor="orange", markersize=2, label="Buoy Point")

        # captain's route
        for df in self.df_cap:
            traj = RealTraj()
            traj.input_csv(df, self.port_csv)
            ax.plot(traj.Y, traj.X, 
                        color = 'gray', ls = '-', marker = 'D',
                        markersize = 2, alpha = 0.8, lw = 1.0, zorder = 1)
        legend_captain = plt.Line2D([0], [0],
                                    color = 'gray', ls = '-', marker = 'D',
                                    markersize = 2, alpha = 0.8, lw = 1.0, label="captain's Route"
        )

        ax.set_xlim(port["hor_range"])
        ax.set_ylim(port["ver_range"])
        ax.set_aspect("equal")
        ax.grid()
        ax.set_xlabel(r"$Y\,\rm{[m]}$")
        ax.set_ylabel(r"$X\,\rm{[m]}$")
        # ax.legend(handles=[legend_initial, legend_way, legend_fixed, legend_buoy, legend_captain, legend_SD])
        fig.savefig(os.path.join(self.SAVE_DIR, "base_map.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        print("\nbase map saved")

    def make_init_route(self):
        port = self.port
        ax = self.ax
        # from birth point
        segs = []

        pt1 = (2*B, port["ver_range"][0])
        pt2 = (2*B, port["ver_range"][1])
        segs.append(np.array([pt1, pt2], dtype=float))

        lines_np = np.stack(segs, axis=0)
        ax.plot(lines_np[0], color="red", linestyle='-')

        # from shipping lane
        df = self.df_lane[self.df_lane['polygon_id'] == 1]
        lane_pts = df[['x [m]', 'y [m]']].to_numpy()
            

        # plt.savefig(os.path.join(self.SAVE_DIR, "first route.png"))
        # print("\nfirst route saved")
        

    def dict_of_port(self, num):
        dictionary_of_port = {
            0: {
                "name": "Osaka_port1A",
                "buoy": "1-堺",
                "start": [-1400.0, -800.0],
                "end": [0.0, -10.0],
                "psi_start": 40,
                "psi_end": 10,
                "berth_type": 2,
                "ver_range": [-1500, 500],
                "hor_range": [-1000, 500],
            },
            1: {
                "name": "Tokyo_port2C",
                "buoy": "千葉",
                "start": [-2400.0, -1600.0],
                "end": [0.0, 0.0],
                "psi_start": 25,
                "psi_end": 10,
                "berth_type": 2,
                "ver_range": [-2500, 500],
                "hor_range": [-2500, 500],
            },
            2: {
                "name": "Yokkaichi_port2B",
                "buoy": "四日市",
                "side": "star board",
                "start": [2000.0, 2000.0],
                "end": [100.0, 80.0], # 300, 80
                "psi_start": -125,
                "psi_end": 175,
                "berth_type": 1,
                "ver_range": [-500, 2500],
                "hor_range": [-500, 2500],
            },
            3: {
                "name": "Else_port1",
                "buoy": "2-坂出",
                "start": [2500.0, 0.0],
                "end": [450.0, 20.0], # [450.0, 20.0]
                "psi_start": -150,
                "psi_end": 135, # 135
                "berth_type": 1,
                "ver_range": [0, 3000],
                "hor_range": [-1000, 2000],
            },
            4: {
                "name": "Osaka_port1B",
                "buoy": "1-堺",
                "start": [-3000.0, -1080.0],
                "end": [-480.0, -80.0], # [-480.0, -80.0]
                "psi_start": -5,
                "psi_end": 45, # 45
                "berth_type": 2,
                "ver_range": [-3200, 500],
                "hor_range": [-1600, 500],
            },
            5: {
                "name": "Else_port2",
                "buoy": "5-函館",
                "start": [-1900.0, 0.0],
                "end": [0.0, 0.0],
                "psi_start": 50,
                "psi_end": -30,
                "berth_type": 2,
                "ver_range": [-1900, 300],
                "hor_range": [-1000, 1200],
            },
            6: {
                "name": "Kashima",
                "buoy": "4-鹿島",
                "start": [1750.0, 1900.0],
                "end": [250.0, -150.0],
                "psi_start": -120,
                "psi_end": -170,
                "berth_type": 2,
                "ver_range": [-1000, 2000],
                "hor_range": [-1500, 2000],
            },
            7: {
                "name": "Aomori",
                "buoy": "6-青森",
                "start": [350, 3400.0],
                "end": [0, 100],
                "psi_start": -115,
                "psi_end": -90,
                "berth_type": 2,
                "ver_range": [-1500, 1500],
                "hor_range": [-1000, 3500],
            },
            8: {
                "name": "Hachinohe",
                "buoy": "3-八戸",
                "start": [1350, 2500.0],
                "end": [100, 250],
                "psi_start": -110,
                "psi_end": -160,
                "berth_type": 2,
                "ver_range": [-1000, 2500],
                "hor_range": [-1000, 3000],
            },
            9: {
                "name": "Shimizu",
                "buoy": "8-清水",
                "start": [1400, -2000],
                "end": [150, 100],
                "psi_start": 100,
                "psi_end": 175,
                "berth_type": 2,
                "ver_range": [-1000, 2000],
                "hor_range": [-3000, 1000],
            },
            10: {
                "name": "Tomakomai",
                "buoy": "11-苫小牧",
                "start": [-1300, 1500],
                "end": [-200.0, -80.0],
                "psi_start": -70,
                "psi_end": 0,
                "berth_type": 2,
                "ver_range": [-2000, 500],
                "hor_range": [-1000, 2000],
            },
            11: {
                "name": "KIX",
                "buoy": "大阪港",
                "start": [-2500, 800],
                "end": [-300, 250],
                "psi_start": -10,
                "psi_end": -30,
                "berth_type": 2,
                "ver_range": [-3000, 500],
                "hor_range": [-2000, 2000],
            },
        }
        return dictionary_of_port[num]

if __name__ == "__main__":
    ps = Setting()
    makeroute = MakeLine(ps)
    makeroute.main()
