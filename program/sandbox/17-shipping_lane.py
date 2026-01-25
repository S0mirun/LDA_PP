import glob
import os

from dataclasses import dataclass
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from scipy import ndimage
from typing import ClassVar, Tuple
from tqdm import tqdm

from utils.LDA.ship_geometry import *
from utils.PP.MultiPlot import RealTraj

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]


class Setting:
    def __init__(self):
        # port
        self.port_number: int = 7
         # 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Else_1, 4: Osaka_1B
         # 5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu
         # 10: Tomakomai, 11: KIX

        self.L = 103.8
        self.B = 16.0


@dataclass
class Line:
    """
    theta : 真上が0 時計回りが負
    """
    fixed_pt:Tuple[float, float]
    end_pt:Tuple[float, float] = None
    theta:float = None # [rad]
    parent=None

    ver_range:ClassVar[Tuple[float, float] | None] = None
    hor_range:ClassVar[Tuple[float, float] | None] = None

    def __post_init__(self):
        if self.end_pt == None:
            end_pt = self.fixed_pt
            while self.check(end_pt):
                end_pt = end_pt + np.array([np.cos(self.theta), np.sin(self.theta)])
            while not self.check(end_pt):
                end_pt = end_pt + np.array([np.cos(self.theta), np.sin(self.theta)])
            self.end_pt = end_pt

    def check(self, pt):
        """
        点が画面の外かを判定する。
        True : 外, False : 内
        """
        return (pt[0] < self.ver_range[0] or self.ver_range[1] < pt[0] 
                    or pt[1] < self.hor_range[0] or self.hor_range[1] < pt[1])
    
    def swap(self):
        fixed_pt = self.fixed_pt; end_pt = self.end_pt; theta = self.theta
        self.fixed_pt = end_pt; self.end_pt = fixed_pt; self.theta = theta + np.pi

    def extent_fixed_pt(self):
        fixed_pt = self.fixed_pt
        if not self.check(fixed_pt):
            while self.check(fixed_pt):
                fixed_pt = fixed_pt - np.array([np.cos(self.theta), np.sin(self.theta)])
            while not self.check(fixed_pt):
                fixed_pt = fixed_pt - np.array([np.cos(self.theta), np.sin(self.theta)])
            self.fixed_pt = fixed_pt

    def set_parent(self, ln):
        self.fixed_pt = cal_intersect_pt(self, ln)
        self.parent = ln


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

def cal_cross(u, v):
    return u[0]*v[1] - u[1]*v[0]

def cal_angle(from_pt, to_pt):
    """
    符号付きの角度
    真上を0としている
    """
    u = [1.0, 0.0]
    v = np.asarray(to_pt) - np.asarray(from_pt)
    dot = np.dot(u, v)
    cross = cal_cross(u, v)
    return np.arctan2(cross, dot)

def cal_intersect_pt(ln1, ln2):
    p1 = ln1.fixed_pt; p2 = ln1.end_pt
    p3 = ln2.fixed_pt; p4 = ln2.end_pt

    a = p2 - p1; b = p4 - p3; c = p3 - p1

    cross_ab = cal_cross(a, b); cross_ca = cal_cross(c, a)
    u = cross_ca / cross_ab
    return p3 + u * b


def cross_judge(l1, l2):
    """
    l1, l2 : Lineで定義された直線
    """
    pt1, pt2 = l1.fixed_pt, l1.end_pt
    pt3, pt4 = l2.fixed_pt, l2.end_pt

    cross1_34 = cal_cross(pt3-pt1, pt4-pt1); cross2_34 = cal_cross(pt3-pt2, pt4-pt2)
    cross3_12 = cal_cross(pt1-pt3, pt2-pt3); cross4_12 = cal_cross(pt1-pt4, pt2-pt4)

    if (cross1_34 * cross2_34 <= 0) and (cross3_12 * cross4_12 <= 0):
        return True
    
    return False

class MakeLine:
    def __init__(self, ps):
        self.ps = ps


    def main(self):
        self.prepare()

    def prepare(self):
        self.read_csv()
        self.init_fig()
        self.draw_basemap()
        self.make_init_line()
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

        Line.hor_range = self.port["hor_range"]
        Line.ver_range = self.port["ver_range"]

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
        # compas
        img = mpimg.imread("raw_datas/compass icon2.png")
        df = pd.read_csv(self.port_csv)
        angle = float(df['Psi[deg]'].iloc[0])
        img_rot = ndimage.rotate(img, angle, reshape=True)
        img_rot = np.clip(img_rot, 0.0, 1.0)
        imagebox = OffsetImage(img_rot, zoom=0.5)
        ab = AnnotationBbox(
            imagebox,
            (0, 1), # ax1's upper left
            xycoords='axes fraction',
            box_alignment=(0, 1), # .png's upper left
            frameon=False,
            pad=0.0,
        )
        ax.add_artist(ab)

        ax.set_xlim(port["hor_range"])
        ax.set_ylim(port["ver_range"])
        ax.set_aspect("equal")
        ax.grid()
        ax.set_xlabel(r"$Y\,\rm{[m]}$")
        ax.set_ylabel(r"$X\,\rm{[m]}$")
        # ax.legend(handles=[legend_initial, legend_way, legend_fixed, legend_buoy, legend_captain, legend_SD])
        fig.savefig(os.path.join(self.SAVE_DIR, "base_map.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        print("\nbase map saved\n")

    def make_init_line(self):
        port = self.port
        ax1 = self.ax
        lines = []

        # from start point
        L_start = Line(fixed_pt=port["start"], theta=np.deg2rad(port["psi_start"]))
        lines.append(L_start)

        # from shipping lane
        for pid, g in self.df_lane.groupby("polygon_id", sort=True):
            lane_pts = g[['y [m]', 'x [m]']].to_numpy()

            mid_1 = (lane_pts[0] + lane_pts[1]) / 2
            mid_2 = (lane_pts[2] + lane_pts[3]) / 2

            L_lane = Line(fixed_pt=np.array((mid_1)), theta=cal_angle(mid_1, mid_2))
            L_lane.extent_fixed_pt()
            lines.append(L_lane)
            print(f"shipping lane {pid} complete")

        # from birth point
        margin = 2*self.ps.B
        if port["psi_end"] == 0:
            if port["side"] == "port":
                theta = theta + 10
                margin = 2*self.ps.B
            elif port["side"] == "starboard":
                margin = -self.ps.B
            if port["style"] == "head in":
                theta = 180 - theta
            theta = np.deg2rad(theta)
        else:
            theta = np.deg2rad(port["psi_end"] + 10)
        L_birth = Line(fixed_pt=np.array((0.0, margin)), theta=theta)
        L_birth.swap()
        lines.append(L_birth)

        handles = []
        for ln in lines:
            pts = np.vstack([ln.fixed_pt, ln.end_pt])
            h, = ax1.plot(pts[:, 1], pts[:, 0], color="red", linestyle='-')
            handles.append(h)
        plt.savefig(os.path.join(self.SAVE_DIR, "init line.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        print("\ninit line saved\n")
        for h in handles:
            h.remove()

        self.lines = lines

    def make_init_route(self):
        lines = self.lines

        # set parent
        for i in range(len(lines) - 2):
            lines[i+1].set_parent(lines[i])

        # check nearlest intersection
        L_birth = lines[-1]
        shortest = np.linalg.norm(L_birth.fixed_pt - L_birth.end_pt); idx = None
        for i in range(len(lines) - 1):
            if cross_judge(lines[i], L_birth):
                intersect_pt = cal_intersect_pt(lines[i], L_birth)
                length = np.linalg.norm(intersect_pt - L_birth.end_pt)
                if length < shortest:
                    shortest = length; idx = i

        if idx != None:
            L_birth.set_parent(lines[idx])
        
        self.show_init_route()

    def show_init_route(self):
        ax2 = self.ax
        lines = self.lines

        ln = lines[-1]
        pts_list = [np.asarray(ln.end_pt), np.asarray(ln.fixed_pt)]
        while ln.parent is not None:
            ln = ln.parent
            pts_list.append(np.asarray(ln.fixed_pt))
        pts = np.vstack(pts_list[::-1])

        ax2.plot(pts[:, 1], pts[:, 0], color="red", linestyle='-')
        plt.savefig(os.path.join(self.SAVE_DIR, "init route.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        print("init route saved\n")

    def dict_of_port(self, num):
        dictionary_of_port = {
            0: {
                "name": "Osaka_port1A",
                "start": [-1400.0, -800.0],
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
                "side": "port",
                "style": "head out",
                "start": [2450.0, 2300.0],
                "psi_start": -145,
                "psi_end": 0,
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
                "side": "port",
                "style": "head in",
                "start": [-3000.0, -1080.0],
                "psi_start": -15,
                "psi_end": 0,
                "berth_type": 2,
                "ver_range": [-3200, 500],
                "hor_range": [-1500, 1500],
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
                "start": [350, 3400.0],
                "psi_start": -115,
                "psi_end": 90,
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
