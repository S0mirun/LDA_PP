import glob
import os
import time

from dataclasses import dataclass
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from scipy import ndimage
import shapely
from shapely import intersects_xy
from shapely.geometry import Polygon, Point, LineString
from shapely.prepared import prep
from shapely.validation import make_valid
from typing import ClassVar, Tuple
from tqdm.auto import tqdm

from utils.LDA.ship_geometry import *
from utils.PP import Bezier_curve as Bezier
from utils.PP.dictionary_of_port import dictionary
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.fillet import fillet
from utils.PP.graph_by_taneichi import ShipDomain_proposal
from utils.PP.MultiPlot import RealTraj

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
theta_list = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(3))


class Setting:
    def __init__(self):
        # port
        self.port_number: int = 2
         # 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Sakaide, 4: Osaka_1B
         # 5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu
         # 10: Tomakomai, 11: KIX

        # ship
        self.L = 103.8
        self.B = 16.0

        # approach
        self.approach_algo = "ARC"
        self.straight = self.L

        # CMA-ES
        self.seed: int = 42
        self.MAX_SPEED_KTS: float = 9.5  # [knots]
        self.MIN_SPEED_KTS: float = 1.5  # [knots]
        self.speed_interval: float = 1.0
        self.MAX_ANGLE_DEG: float = 60  # [deg]
        self.MIN_ANGLE_DEG: float = 0  # [deg]
        self.angle_interval: float = 5

        # weight
        self.SD = 1.0
        self.angle = 1.0
        self.xy = 1.0

        # restart
        self.restarts: int = 3
        self.increase_popsize_on_restart: bool = False


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

def sigmoid(x, a, b, c):
    return a / (b + np.exp(c * x))


class ShipDomain:
    def __init__(self):
        SD = ShipDomain_proposal()
        SD_setup_csv = "outputs/303/mirror5/fitting_parameter.csv"

        SD.initial_setting(SD_setup_csv, sigmoid)

        SD.a_ave = self.df_debug["a_ave"].values[0]
        SD.b_ave = self.df_debug["b_ave"].values[0]
        SD.a_SD = self.df_debug["a_SD"].values[0]
        SD.b_SD = self.df_debug["b_SD"].values[0]


class Calculator:
    def __init__(self, ps, sd):
        self.ps = ps
        self.SD = sd

    def cross(self, u, v):
        return u[0]*v[1] - u[1]*v[0]
    
    def unit(self, v):
        nv = np.linalg.norm(v)
        return v / nv
    
    def angle(self, from_pt, to_pt):
        """
        符号付きの角度
        真上を0としている
        """
        u = [1.0, 0.0]
        v = np.asarray(to_pt) - np.asarray(from_pt)
        dot = np.dot(u, v)
        cross = self.cross(u, v)
        return np.arctan2(cross, dot)
    
    def psi(self, parent_pt, current_pt, child_pt):
        """
        船首方位角の計算。北が0, 時計回りが正
        """
        ver_p, hor_p = parent_pt
        ver_c, hor_c = current_pt
        ver_n, hor_n = child_pt

        v1 = np.array([hor_c - hor_p, ver_c - ver_p], dtype=float)
        v2 = np.array([hor_n - hor_c, ver_n - ver_c], dtype=float)
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)

        if m1 == 0.0 or m2 == 0.0:
            theta = 0.0
        else:
            dot = float(np.dot(v1, v2))
            cross = float(self.cross(v1, v2))
            theta = float(np.arctan2(cross, dot))  # CCW:+, CW:-

        psi_in = np.pi/2 - np.arctan2(v1[1], v1[0])  # 0=North, CW:+
        psi = psi_in - 0.5 * theta
        psi = (psi + np.pi) % (2.0 * np.pi) - np.pi
        return psi
    
    def speed(self, pt, base_pt):
        distance = np.linalg.norm(pt - base_pt)
        speed = self.SD.b_ave * distance ** self.SD.a_ave + self.SD.b_SD * distance ** self.SD.a_SD
        if self.ps.MAX_SPEED_KTS < speed:
            return self.ps.MAX_SPEED_KTS
        if self.ps.MIN_SPEED_KTS > speed:
            return self.ps.MIN_SPEED_KTS
        return speed
    

@dataclass
class Line:
    """
    theta : 真上が0 時計回りが負
    """
    cal:Calculator

    fixed_pt:Tuple[float, float]
    end_pt:Tuple[float, float] = None
    theta:float = None # [rad]
    parent=None

    ver_range:ClassVar[Tuple[float, float] | None] = None
    hor_range:ClassVar[Tuple[float, float] | None] = None
    map_poly:ClassVar = None
    map_poly_prep:ClassVar = None
    lane:ClassVar = None

    def __post_init__(self):
        self.theta = self.theta % (2 * np.pi)
        if self.end_pt == None:
            end_pt = self.fixed_pt
            if self.check(end_pt):
                while self.check(end_pt):
                    end_pt = end_pt + np.array([np.cos(self.theta), np.sin(self.theta)])
            while not self.check(end_pt):
                end_pt = end_pt + np.array([np.cos(self.theta), np.sin(self.theta)])

            self.end_pt = end_pt

    def check(self, pt):
        """
        点が画面の外かを判定する。
        地形と接触していないかも判定する。
        True : 外, False : 内
        """
        # outside or not
        outside =  (pt[0] < self.ver_range[0] or self.ver_range[1] < pt[0] 
                    or pt[1] < self.hor_range[0] or self.hor_range[1] < pt[1])
        if outside:
            return True
        
        # in Polygon or not
        x, y = pt[1], pt[0]
        if self.map_poly_prep.intersects(Point(x, y)):
            return True
        
        return False
    
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
        self.fixed_pt = self.intersect(self, ln)
        self.parent = ln

    def cross_judge(self, l1, l2):
        """
        l1, l2 : Lineで定義された直線
        """
        pt1, pt2 = l1.fixed_pt, l1.end_pt
        pt3, pt4 = l2.fixed_pt, l2.end_pt

        cross1_34 = self.cal.cross(pt3-pt1, pt4-pt1); cross2_34 = self.cal.cross(pt3-pt2, pt4-pt2)
        cross3_12 = self.cal.cross(pt1-pt3, pt2-pt3); cross4_12 = self.cal.cross(pt1-pt4, pt2-pt4)

        if (cross1_34 * cross2_34 <= 0) and (cross3_12 * cross4_12 <= 0):
            return True
        
        return False

    def intersect(self, ln1, ln2):
        p1 = ln1.fixed_pt; p2 = ln1.end_pt
        p3 = ln2.fixed_pt; p4 = ln2.end_pt

        a = p2 - p1; b = p4 - p3; c = p3 - p1

        cross_ab = self.cal.cross(a, b); cross_ca = self.cal.cross(c, a)
        u = cross_ca / cross_ab
        return p3 + u * b
    

class CostCalculator:
    def __init__(self, ps, sd, cal):
        self.ps = ps
        self.SD = sd
        self.cal = cal

        self.lines = None

    def SD_penalty(self, pt, psi):
        SD = self.SD
        lines = self.lines
        
        speed = self.cal.speed(pt, lines[-1].end_pt)
        r_list = []
        for theta_i in theta_list:
            r_list.append(SD.distance(speed, theta_i))

        r = np.asarray(r_list, dtype=float)
        domain_xy = np.column_stack([
            pt[0] + r * np.cos(theta_list + psi),
            pt[1] + r * np.sin(theta_list + psi),
        ])
        sd_poly = Polygon(domain_xy)

        # area
        sd_area = sd_poly.area
        if Line.map_poly_prep.intersects(sd_poly):
            inter_area = sd_poly.intersection(Line.map_poly).area
            pen = (inter_area / sd_area ) * 100 # [%]
        else:
            pen = 0.0

        return pen


class PathPlanning():
    def __init__(self, ps, sd, cal):
        self.ps = ps
        self.SD = sd
        self.cal = cal

    def main(self):
        self.preset()
        self.make_path()
        self.save_result()


    def preset(self):
        self.set_target_port()
        self.setup_figure()
        self.draw_basemap()

    def make_path(self):
        self.build_lines_by_shipping_lane()
        self.supplement_lines()
        self.build_path_from_lines()
        self.generate_path()
        self.refine_path()

    def save_result(self):
        self.set_save_dir()
        self.save_base_map()
        self.save_init_lines()
        self.save_supplied_lines()
        self.save_generated_path()
        self.save_refined_path()

    
    def set_target_port(self):
        self.port = dictionary()[self.ps.port_number]
        self.port_csv=f"raw_datas/tmp/coordinates_of_port/_{self.port["name"]}.csv"


    def setup_figure(self):
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.set_xlim(self.port["hor_range"])
        ax.set_ylim(self.port["ver_range"])
        ax.set_aspect("equal")
        ax.grid(True)
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

        self.fig, self.ax = fig, ax


    def draw_basemap(self):
        fig, ax = self.fig, self.ax

        self._draw_land(fig, ax)
        self._draw_shipping_lane(fig, ax)
        self._draw_buoy(fig, ax)
        self._add_compass_image(fig, ax)


    def _draw_land(self, fig, ax):
        df_land = pd.read_csv(f"outputs/data/detail_map/{self.port["name"]}.csv")

        map_X, map_Y = df_land["x [m]"].values, df_land["y [m]"].values
        ax.fill_betweenx(map_X, map_Y, facecolor="gray", alpha=0.3, zorder=0)
        ax.plot(map_Y, map_X, color="k", linestyle="--", lw=0.5, alpha=0.8, zorder=0)

        self.df_land = df_land


    def _draw_shipping_lane(self, fig, ax):
        df_shipping_lane = pd.read_csv(f"outputs/data/Shipping_lane/{self.port["name"]}.csv")
        df_lane = df_shipping_lane.copy()
        df_lane["x [m]"], df_lane["y [m]"] = convert(df_lane, self.port_csv)

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

        self.df_shipping_lane = df_shipping_lane


    def _draw_buoy(self, fig, ax):
        df_buoy = pd.read_csv(buoy_csv = f"outputs/data/buoy/{self.port['name']}.csv")
        df_buoy["x [m]"], df_buoy["y [m]"] = convert(df_buoy, self.port_csv, "latitude", "longitude")

        ax.scatter(self.df_buoy["x [m]"].values, self.df_buoy["y [m]"].values,
                    color='orange', s=20, zorder=2)
        
        
    def _add_compass_image(self, fig, ax):
        img = mpimg.imread("raw_datas/compass icon2.png")
        df = pd.read_csv(self.port_csv)
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
        

    def build_lines_by_shipping_lane(self):
        self._setup_lines()
        self._build_lines_from_berth()
        self._build_lines_from_shipping_lane()
        self._build_lines_from_start()
        self._define_DAG()

    def _setup_lines(self):
        self.lines = []

        # for crossing algorithm
        coords_map = self.df_land[["y [m]", "x [m]"]].to_numpy(dtype=float)
        poly_map = make_valid(Polygon(coords_map))
        Line.map_poly = poly_map
        Line.map_poly_prep = prep(poly_map)


    def _build_lines_from_berth(self):
        port = self.port

        theta = 0
        margin = 2 * self.ps.B
        if port["psi_end"] == 0:
            if port["style"] == "head out" and port["side"]== "starboard":
                theta = 0;    margin = -margin
            elif port["style"] == "head out" and port["side"]== "port":
                theta = 0;    margin = margin
            elif port["style"] == "head in" and port["side"] == "starboard":
                theta = -170;  margin = -margin
            elif port["style"] == "head in" and port["side"] == "port":
                theta = 180; margin = margin
            theta = np.deg2rad(theta)
        else:
            theta = np.deg2rad(port["psi_end"])

        L_berth = Line(fixed_pt=np.array((0.0, margin)), theta=theta)
        L_berth.swap()
        self.lines.append(L_berth)
        self.pp_end = np.array((0.0, margin))

    def _build_lines_from_shipping_lane(self):
        cal = self.cal
        lines = self.lines

        lane_polys = []
        dist_both_ship = 0.5 * self.ps.L + self.ps.B
        for pid, g in self.df_shipping_lane.groupby("polygon_id", sort=True):
            lane_pts = g[['y [m]', 'x [m]']].to_numpy()

            mid_1 = (lane_pts[0] + lane_pts[1]) / 2
            mid_2 = (lane_pts[2] + lane_pts[3]) / 2
            theta = cal.angle(mid_1, mid_2)

            B_shiplane = np.linalg.norm(lane_pts[1] - lane_pts[0])
            d = min(B_shiplane / 4, dist_both_ship)
            mid_1 = mid_1 + np.array([-d * np.sin(theta), d * np.cos(theta)])
            L_lane = Line(fixed_pt=np.array((mid_1)), theta=theta)
            L_lane.extent_fixed_pt()
            lines.append(L_lane)

            # for crossing algorithm
            poly_lane = Polygon(lane_pts[:, [1, 0]])
            lane_polys.append(poly_lane)
            print(f"shipping lane {pid} complete")

        Line.lane = shapely.union_all(lane_polys)
        lines[:] = lines[1:] + lines[:1]


    def _build_lines_from_start(self):
        port = self.port
        lines = self.lines

        L_start = Line(fixed_pt=port["start"], theta=np.deg2rad(port["psi_start"]))
        lines.insert(0, L_start)
        self.pp_start = port["start"]


    def _define_DAG(self):
        lines = self.lines

        for i in range(len(lines) - 2):
            lines[i+1].set_parent(lines[i])


    def supplement_lines(self):
        lines = self.lines

        self.base_idx = len(lines) - 1
        while True:
            self._seek_nearest_line()
            if self.cross_line_idx is not None:
                L_base = lines[self.base_idx]
                L_base.set_parent(lines[self.cross_line_idx])
            else:
                self._supplement_line()


    def _seek_nearest_line(self):
        lines = self.lines

        L_base = lines[self.base_idx]
        cross_line_idx = None

        if len(lines) == 2:
            if Line.cross_judge(lines[0], L_base):
                cross_line_idx = 0
        elif len(lines) > 2:
            shortest = np.inf
            for i in range(1, self.base_idx):
                if Line.cross_judge(lines[i], L_base):
                    intersect_pt = Line.intersect(lines[i], L_base)
                    length = np.linalg.norm(intersect_pt - L_base.end_pt)
                    if length < shortest:
                        shortest = length
                        cross_line_idx = i
        
        self.cross_line_idx = cross_line_idx

        
    def _supplement_line(self):
        lines = self.lines
        L_base = lines[self.base_idx]

        mid = (L_base.fixed_pt + L_base.end_pt) / 2
        if len(lines) == 2:
            ln = lines[0]
            self._find_visible_range(ln, mid)
        elif len(lines) > 2:
            for ln in lines[-2:0:-1]:
                self._find_visible_range(ln, mid)

                if self.idx_hit != 99:
                    break

        self._build_supplement_line(L_base, mid)

    
    def _find_visible_range(self, ln, mid):
        def to_xy(p_yx):
            return (float(p_yx[1]), float(p_yx[0]))
        
        pts = np.linspace(ln.end_pt, ln.fixed_pt, 99)
        
        idx = 0
        while idx < 99 and Line.map_poly_prep.intersects(LineString([to_xy(pts[idx]), to_xy(mid)])):
            idx += 1
        idx_through = idx
        while idx < 99 and (not Line.map_poly_prep.intersects(LineString([to_xy(pts[idx]), to_xy(mid)]))):
            idx += 1
        idx_hit = idx
        idx_hit = np.clip(idx_hit, 0, len(pts)-1)

        self.pts = pts
        self.idx_through = idx_through
        self.idx_hit = idx_hit


    def _build_supplement_line(self, L_base, mid):
        cal = self.cal

        fixed_pt = (self.pts[self.idx_through] + self.pts[self.idx_hit]) / 2
        theta = cal.angle(mid, fixed_pt)
        L_append = Line(fixed_pt=fixed_pt, theta=theta)
        L_append.swap()
        L_base.set_parent(L_append)
        self.lines.insert(self.base_idx, L_append)

    
    def build_path_from_lines(self):
        self._get_WP_from_lines()
        WP = self.way_points


        if self.ps.approach_algo == "ARC":
            for i in range(len(self.way_points)):
                self._find_best_fillet_arc(WP[i], WP[i+1], WP[i+2])


    def _get_WP_from_lines(self):
        lines = self.lines

        ln = lines[-1]
        WP_list = []
        while ln.parent is not None:
            WP_list.append(np.asarray(ln.fixed_pt))
            ln = ln.parent

        way_points = np.vstack(WP_list[::-1])
        
        self.way_points = way_points


    def _find_best_fillet_arc(self, pt1, pt2, pt3):
        pass

if __name__ == '__main__':
    ps = Setting()
    sd = ShipDomain()
    cal = Calculator(ps, sd)
    
    pp = PathPlanning(ps, sd, cal)
    pp.main()
