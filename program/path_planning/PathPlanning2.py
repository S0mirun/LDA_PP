import argparse
import glob
import os
import time

from dataclasses import dataclass
from enum import Enum, auto
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from openpyxl.utils import get_column_letter
import pandas as pd
from scipy import ndimage
import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.prepared import prep
from shapely.validation import make_valid
from tqdm.auto import tqdm
from typing import ClassVar, Tuple

from utils.LDA.ship_geometry import *
from utils.PP.clothoid import build_biclothoid, check_constraints, compute_corner_geometry
from utils.PP.dictionary_of_port import dictionary
from utils.PP.E_ddCMA import Checker, DdCma, Logger
from utils.PP.fillet import fillet
from utils.PP.graph_by_taneichi import ShipDomain_proposal
from utils.PP.MultiPlot import RealTraj
from utils.PP.Seek_pairs import pair_points_min_distance_df
from utils.PP import save_figures

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
theta_list = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(3))


class SupplementMode(Enum):
    MIDPOINT = auto()
    ANGLE_BISECTOR = auto()

class ApproachAlgo(Enum):
    ARC = auto()
    CLOTHOID = auto()


class Setting:
    def __init__(self):
        # port
        self.port_number: int = 0
         # 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Sakaide, 4: Osaka_1B
         # 5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu
         # 10: Tomakomai, 11: KIX

        # ship
        self.L = 103.8
        self.B = 16.0

        # approach
        self.approach_algo = ApproachAlgo.CLOTHOID
        self.SupplementMode = SupplementMode.MIDPOINT
        self.redraw_by_AI = True

        # clothoid (bi-clothoid corner connection)
        self.MIN_TURN_RADIUS_COEF: float = 3.0
        self.MAX_YAW_ACCEL_DEGS2: float = 5.0
        self.DELTA_S: float = 5.0
        self.LC_GRID_N: int = 51
        self.CLOTHOID_ETA: float = 0.5
        self.W_SD: float = 1.0
        self.W_L: float = 1.0

        # clothoid: multi-corner resolution
        self.MAX_RESOLVE_ITERS: int = 5
        self.JOINT_LC_GRID_N: int = 5
        self.ETA_GRID: list = [0.3, 0.4, 0.5, 0.6, 0.7]

        # clothoid: CMA-ES optimization
        self.CLOTHOID_OPTIMIZER: str = "cma-es"        # "grid" or "cma-es"
        self.CMA_SEED: int = 42
        self.CMA_RESTARTS: int = 3
        self.CMA_INCREASE_POPSIZE_ON_RESTART: bool = False
        self.CMA_ETA_BOUNDS: tuple = (0.1, 0.9)
        self.CMA_SIGMA0_LC_FRAC: float = 0.1
        self.CMA_SIGMA0_ETA: float = 0.1
        self.CMA_OVERLAP_PENALTY_COEF: float = 100.0
        self.CMA_INFEASIBLE_PENALTY: float = 1.0e4

        # CMA-ES
        self.seed: int = 42
        self.MAX_SPEED_KTS: float = 9.5  # [knots]
        self.MIN_SPEED_KTS: float = 1.5  # [knots]
        self.speed_interval: float = 1.0
        self.MAX_ANGLE_DEG: float = 60  # [deg]
        self.MIN_ANGLE_DEG: float = 0  # [deg]
        self.angle_interval: float = 5

        # display toggles
        self.SHOW_SHIP_SHAPES: bool = True

        # others
        self.PDF = True

    def min_turn_radius(self) -> float:
        return self.MIN_TURN_RADIUS_COEF * self.L



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
        self.SD_p = ShipDomain_proposal()

        SD_setup_csv = "outputs/303/mirror5/fitting_parameter.csv"
        self.SD_p.initial_setting(SD_setup_csv, sigmoid)

        df_debug = pd.read_csv("raw_datas/tmp/GuidelineFit_debug.csv")
        self.a_ave = df_debug["a_ave"].values[0]
        self.b_ave = df_debug["b_ave"].values[0]
        self.a_SD = df_debug["a_SD"].values[0]
        self.b_SD = df_debug["b_SD"].values[0]

        print("\nShip Domain setup complete")


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
class _CornerResult:
    entry_point: np.ndarray
    exit_point: np.ndarray
    din: float
    dout: float
    curve_points: np.ndarray
    is_arc: bool = False
    Lc: float = 0.0      # クロソイド合計長 (is_arc=Trueの場合は未使用)
    eta: float = 0.5      # 非対称率 Lin/Lc (is_arc=Trueの場合は未使用)



@dataclass
class Line:
    """
    theta : 真上が0 時計回りが負
    """
    cal: ClassVar[Calculator] = None

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
        if self.end_pt is None:
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
        self.fixed_pt = self.intersect(ln)
        self.parent = ln


    def cross_judge(self, other, eps=1e-1):

        def sgn(x, eps=1e-1):
            if x > eps:
                return 1
            elif x < -eps:
                return -1
            else:
                return 0

        p1 = self.fixed_pt; p2 = self.end_pt
        p3 = other.fixed_pt; p4 = other.end_pt

        cross1_34 = self.cal.cross(p3-p1, p4-p1); cross2_34 = self.cal.cross(p3-p2, p4-p2)
        cross3_12 = self.cal.cross(p1-p3, p2-p3); cross4_12 = self.cal.cross(p1-p4, p2-p4)

        s1 = sgn(cross1_34, eps); s2 = sgn(cross2_34, eps)
        s3 = sgn(cross3_12, eps); s4 = sgn(cross4_12, eps)

        return (s1 * s2 <= 0) and (s3 * s4 <= 0)


    def intersect(self, other):
        p1 = self.fixed_pt; p2 = self.end_pt
        p3 = other.fixed_pt; p4 = other.end_pt

        a = p2 - p1; b = p4 - p3; c = p3 - p1

        cross_ab = self.cal.cross(a, b); cross_ca = self.cal.cross(c, a)
        u = cross_ca / cross_ab
        return p3 + u * b


    def angle(self, other):
        p1 = self.fixed_pt; p2 = self.end_pt
        p3 = other.fixed_pt; p4 = other.end_pt

        v1 = p2 - p1; norm1 = np.linalg.norm(v1)
        v2 = p4 - p3; norm2 = np.linalg.norm(v2)

        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
        return np.degrees(theta)



class CostCalculator:
    def __init__(self, ps, sd, cal):
        self.ps = ps
        self.SD = sd
        self.cal = cal


    def SD_penalty(self, lines, pt, psi):
        SD = self.SD
        SD_p = SD.SD_p

        speed = self.cal.speed(pt, lines[-1].end_pt)
        r_list = []
        for theta_i in theta_list:
            r_list.append(SD_p.distance(speed, theta_i))

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



class PathPlanning:
    def __init__(self, ps, sd, cal, cost_cal):
        self.ps = ps
        self.SD = sd
        self.cal = cal
        self.cost_cal = cost_cal


    def main(self):
        self.preset()
        self.make_path()
        self.save_results()
        print(
            "\n##### All tasks complete #####"
            f"\ntarget : {self.port['name']}"
            f"\nMode   : {self.ps.approach_algo.name}"
            f"\nRedraw : {self.ps.redraw_by_AI}"
        )


    def preset(self):
        print("\n######  preset start    ######")
        self.set_target_port()
        self.setup_save_dir()
        self.setup_Line()
        self.setup_figure()
        self.draw_basemap()


    def make_path(self):
        print("\n######  make path start    #####")
        self.build_lines_by_shipping_lane()
        self.supplement_lines()
        self.generate_path()


    def save_results(self):
        self.save_result_fig()


    def set_target_port(self):
        self.port = dictionary()[self.ps.port_number]
        self.port_csv=f"raw_datas/tmp/coordinates_of_port/_{self.port["name"]}.csv"
        print(f"\ntarget : {self.port['name']}")


    def setup_save_dir(self):
        self.save_dir_path = f"{DIR}/../../outputs/{dirname}/{self.port["name"]}"
        folder_name = self._make_folder_name()
        SAVE_DIR = f"{self.save_dir_path}/{folder_name}"
        os.makedirs(SAVE_DIR, exist_ok=True)
        if self.ps.PDF:
            os.makedirs(f"{SAVE_DIR}/{self.port["legend"]}", exist_ok=True)

        self.SAVE_DIR = SAVE_DIR


    def _make_folder_name(self):
        ApproachAlgo_str = self.ps.approach_algo.name.lower()
        SupplementMode_str = self.ps.SupplementMode.name.lower()
        AI_str = "ai_on" if self.ps.redraw_by_AI else "ai_off"
        return f"{ApproachAlgo_str}_{SupplementMode_str}_{AI_str}"


    def setup_Line(self):
        Line.cal = self.cal
        Line.ver_range = self.port["ver_range"]
        Line.hor_range = self.port["hor_range"]
        print("\nLine setup complete")


    def setup_figure(self):
        fig, ax = plt.subplots(figsize=(8, 11))

        ax.set_xlim(self.port["hor_range"])
        ax.set_ylim(self.port["ver_range"])
        ax.set_aspect("equal")
        ax.grid(True)
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

        self.fig, self.ax = fig, ax
        self.handles = []
        print("\nFigure setup complete")


    def draw_basemap(self):
        fig, ax = self.fig, self.ax
        self.legends = []

        self._draw_land(fig, ax)
        self._draw_shipping_lane(fig, ax)
        self._draw_buoy(fig, ax)
        self._add_compass_image(fig, ax)
        self._plot_pts(fig, ax)
        self._save_fig(fig, ax, "basemap")
        print("\nDraw basemap complete")


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

        self.df_shipping_lane = df_lane
        self.legends.append(save_figures.LEGEND_TRAFFIC_LANE)


    def _draw_buoy(self, fig, ax):
        df_buoy = pd.read_csv(f"outputs/data/buoy/{self.port['name']}.csv")
        df_buoy["x [m]"], df_buoy["y [m]"] = convert(df_buoy, self.port_csv, "latitude", "longitude")

        ax.scatter(df_buoy["x [m]"].values, df_buoy["y [m]"].values,
                   **save_figures.BUOY_SCATTER_KWARGS)
        self.legends.append(save_figures.LEGEND_BUOY)
        self._draw_buoy_color(fig, ax, df_buoy)
        self._draw_buoy_pair(fig, ax, df_buoy)



    def _draw_buoy_color(self, fig, ax, df_buoy):
        df_buoy["COLOUR"] = df_buoy["COLOUR"].astype(str).str.strip()
        COLOR = save_figures.BUOY_COLOR_LIST
        for i in range(1, 7):
            ax.scatter(
                df_buoy.loc[df_buoy["COLOUR"] == str(i), "x [m]"].values,
                df_buoy.loc[df_buoy["COLOUR"] == str(i), "y [m]"].values,
                color=COLOR[i-1], **save_figures.BUOY_COLOR_SCATTER_KWARGS)


    def _draw_buoy_pair(self, fig, ax, df_buoy):
        df_pairs, _ = pair_points_min_distance_df(df=df_buoy, x_col="x [m]", y_col="y [m]", max_distance=1000)
        self._save_excel(df_pairs, "buoy_pair")

        self.buoy_lines = []
        for _, row in df_pairs[df_pairs["type"] == "pair"].iterrows():
            ax.plot(
                [row["x3"], row["x4"]],
                [row["y3"], row["y4"]],
                color="orange", lw=3, linestyle="-", zorder=2)

            self._set_buoy_lines(row["x3"], row["x4"], row["y3"], row["y4"])

        self.legends.append(save_figures.LEGEND_BUOY_LINE)


    def _save_excel(self, df, name):
        SAVE_DIR = f"{self.SAVE_DIR}/excel"
        os.makedirs(SAVE_DIR, exist_ok=True)

        file_path = os.path.join(SAVE_DIR, f"{name}.xlsx")

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
            ws = writer.sheets["Sheet1"]

            for i, col_name in enumerate(df.columns, 1):
                max_length = max(
                    len(str(col_name)),
                    df[col_name].astype(str).map(len).max() if len(df) > 0 else 0
                )
                ws.column_dimensions[get_column_letter(i)].width = max_length * 1.5 + 2


    def _set_buoy_lines(self, y_red, y_green, x_red, x_green):
        cal = self.cal

        pt_red = np.array([x_red, y_red])
        pt_green = np.array([x_green, y_green])
        theta = cal.angle(pt_red, pt_green)
        line = Line(fixed_pt=pt_red, end_pt=pt_green, theta=theta)

        self.buoy_lines.append(line)


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


    def _plot_pts(self, fig, ax):
        pt_start = self.port["start"]
        pt_end = [0, 0]

        annotations = self.port["annotations"]

        ax.scatter(pt_start[1], pt_start[0], **save_figures.START_END_SCATTER_KWARGS)
        ax.scatter(pt_end[1], pt_end[0], **save_figures.START_END_SCATTER_KWARGS)

        ann = annotations["approach_start"]
        ax.annotate("Start", xy=(pt_start[1], pt_start[0]),
                    xytext=ann["xytext"], textcoords="offset points", ha=ann["ha"], va=ann["va"],
                    fontsize=save_figures.START_END_ANNOTATE_FONTSIZE)
        ann = annotations["turn_end"]
        ax.annotate("Goal", xy=(pt_end[1], pt_end[0]),
                    xytext=ann["xytext"], textcoords="offset points", ha=ann["ha"], va=ann["va"],
                    fontsize=save_figures.START_END_ANNOTATE_FONTSIZE)


    def _save_fig(self, fig, ax, name):
        pdf_dir = f"{self.SAVE_DIR}/{self.port["legend"]}" if self.ps.PDF else None
        save_figures.save_fig(fig, ax, self.SAVE_DIR, name, self.legends, self.handles,
                               pdf=self.ps.PDF, pdf_dir=pdf_dir)

    def build_lines_by_shipping_lane(self):
        self._setup_lines()
        self._build_lines_from_berth()
        self._build_lines_from_shipping_lane()
        self._build_lines_from_start()
        print("\nbuilt lines complete")
        self._define_DAG()


    def _setup_lines(self):
        self.lines = []

        # for crossing algorithm
        coords_map = self.df_land[["y [m]", "x [m]"]].to_numpy(dtype=float)
        poly_map = make_valid(Polygon(coords_map))
        Line.map_poly = poly_map
        Line.map_poly_prep = prep(poly_map)

        self.legends.append(save_figures.LEGEND_CANDIDATE_LINES)


    def _build_lines_from_berth(self):
        port = self.port

        theta = 0
        if port["psi_end"] == 0:
            if port["style"] == "head out" and port["side"]== "starboard":
                theta = 0;    margin = -self.ps.L
            elif port["style"] == "head out" and port["side"]== "port":
                theta = 0;    margin = self.ps.L
            elif port["style"] == "head in" and port["side"] == "starboard":
                theta = -180;  margin = -2 * self.ps.B
            elif port["style"] == "head in" and port["side"] == "port":
                theta = 180; margin = 2 * self.ps.B
            theta = np.deg2rad(theta)
        else:
            theta = np.deg2rad(port["psi_end"]); margin = 2 * self.ps.B

        L_berth = Line(fixed_pt=np.array((0.0, margin)), theta=theta)
        L_berth.swap()
        self.lines.append(L_berth)
        self.pp_end = np.array((0.0, margin))

        self._save_lines("line_from_berth")


    def _save_lines(self, name):
        pdf_dir = f"{self.SAVE_DIR}/{self.port["legend"]}" if self.ps.PDF else None
        save_figures.save_lines(self.fig, self.ax, self.lines, self.handles, self.SAVE_DIR, name,
                                 self.legends, pdf=self.ps.PDF, pdf_dir=pdf_dir)


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
            # mid_1 = mid_1 + np.array([-d * np.sin(theta), d * np.cos(theta)])
            L_lane = Line(fixed_pt=np.array((mid_1)), theta=theta)
            L_lane.extent_fixed_pt()
            lines.append(L_lane)

            # for crossing algorithm
            poly_lane = Polygon(lane_pts[:, [1, 0]])
            lane_polys.append(poly_lane)
            print(f"shipping lane {pid} complete")
            self._save_lines(f"line_from_shipping_lane_{pid}")

        Line.lane = shapely.union_all(lane_polys)
        lines[:] = lines[1:] + lines[:1]


    def _build_lines_from_start(self):
        port = self.port
        lines = self.lines

        L_start = Line(fixed_pt=port["start"], theta=np.deg2rad(port["psi_start"]))
        lines.insert(0, L_start)
        self.pp_start = port["start"]

        self._save_lines("line_from_start")


    def _define_DAG(self):
        lines = self.lines

        for i in range(len(lines) - 2):
            lines[i+1].set_parent(lines[i])

        self._save_lines("line_DAG")


    def supplement_lines(self):
        L_base = self.lines[-1]
        self.len_lines = len(self.lines)

        idx = 0
        while True:
            self._seek_nearest_line(L_base)
            if self.cross_line_idx is not None:
                print("OK")
                L_base.set_parent(self.lines[self.cross_line_idx])
                break
            else:
                print("BAD")
                idx += 1
                self._supplement_line(idx)
                if self.ps.redraw_by_AI:
                    self._redraw_line_by_buoy(idx)
                self._set_parent(self.lines[-2])

        print("\nsupplement lines complete")


    def _seek_nearest_line(self, L_base, minus=1):
        lines = self.lines

        cross_line_idx = None
        if len(lines) == 2:
            if L_base.cross_judge(lines[0]):
                cross_line_idx = 0
        elif len(lines) > 2 and self.len_lines == 2:
            shortest = np.inf
            for i in range(len(lines) - minus):
                if L_base.cross_judge(lines[i]):
                    intersect_pt = L_base.intersect(lines[i])
                    length = np.linalg.norm(intersect_pt - L_base.end_pt)
                    if length < shortest:
                        shortest = length
                        cross_line_idx = i
        elif len(lines) > 2 and self.len_lines > 2:
            shortest = np.inf
            for i in range(1, len(lines) - minus):
                if L_base.cross_judge(lines[i]):
                    intersect_pt = L_base.intersect(lines[i])
                    length = np.linalg.norm(intersect_pt - L_base.end_pt)
                    if length < shortest:
                        shortest = length
                        cross_line_idx = i

        self.cross_line_idx = cross_line_idx


    def _supplement_line(self, idx):
        lines = self.lines
        L_base = lines[-1]

        mid = (L_base.fixed_pt + L_base.end_pt) / 2
        if len(lines) == 2:
            ln = lines[0]
            self._find_visible_range(ln, mid)
        elif len(lines) > 2:
            for ln in lines[-2:0:-1]:
                self._find_visible_range(ln, mid)

                if self.idx_hit != 98:
                    break

        self._build_supplement_line(mid, idx)


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


    def _build_supplement_line(self, mid, idx):
        cal = self.cal

        if self.ps.SupplementMode == SupplementMode.MIDPOINT:
            fixed_pt = (self.pts[self.idx_through] + self.pts[self.idx_hit]) / 2
            theta = cal.angle(fixed_pt, mid)
            L_append = Line(fixed_pt=fixed_pt, end_pt=mid, theta=theta)
        elif self.ps.SupplementMode == SupplementMode.ANGLE_BISECTOR:
            theta_through = cal.angle(mid, self.pts[self.idx_through])
            theta_hit = cal.angle(mid, self.pts[self.idx_hit])
            theta = (theta_through + theta_hit) / 2
            L_append = Line(fixed_pt=mid, theta=theta)
            L_append.swap()

        self.lines.insert(-1, L_append)

        self._save_lines(f"lines_supplement_{idx}")


    def _redraw_line_by_buoy(self, idx):
        target_line = self.lines[-2]

        for ln in self.buoy_lines:
            angle = target_line.angle(ln)
            if target_line.cross_judge(ln) and abs(angle - 90) > 10:
                mid = (ln.fixed_pt + ln.end_pt) / 2
                theta = ln.theta + np.pi / 2
                L_replacement = Line(fixed_pt=mid, theta=theta)
                L_replacement.extent_fixed_pt()
                self.lines[-2] = L_replacement

                self._save_lines(f"lines_supplement_{idx}(replaced)")
                print(f"redraw Line No.{idx},   angle : {angle}")
                break


    def _set_parent(self, L_base):
        self._seek_nearest_line(L_base=L_base, minus=2)
        L_base.set_parent(self.lines[self.cross_line_idx])


    def generate_path(self):
        print("\n##### Generate path Start #####")
        self._get_WP_from_lines()

        self.legends.append(save_figures.LEGEND_CAPTAIN_ROUTE)
        if self.ps.SHOW_SHIP_SHAPES:
            self.legends.append(save_figures.LEGEND_SHIP_SHAPE)
        self._save_pts(self.way_points, "way_points")

        WP = self.way_points
        if self.ps.approach_algo == ApproachAlgo.ARC:
            full_pts = np.vstack([self.pp_start, self.way_points, self.pp_end])

            arc_list = []
            for i in range(len(self.way_points)):
                self._find_best_fillet_arc(full_pts[i], full_pts[i+1], full_pts[i+2], arc_list)

            arcs = np.concatenate(arc_list, axis=0)
            self.result_pts = arcs
            print("\nFillet arc path complete")

        elif self.ps.approach_algo == ApproachAlgo.CLOTHOID:
            corners, full_seq = self._resolve_and_generate_clothoid_path()
            if self.ps.CLOTHOID_OPTIMIZER == "cma-es":
                corners = self._optimize_clothoid_cma_es(corners, full_seq)
            self.result_pts = self._assemble_full_path(corners, full_seq)
            print("\nClothoid pair path complete (multi-corner resolved)")

        self.legends.append(save_figures.LEGEND_PLANNED_PATH)
        self._save_pts(self.result_pts, "generated_path", pt_size=5)


    def _get_WP_from_lines(self):
        lines = self.lines

        ln = lines[-1]
        WP_list = []
        while ln.parent is not None:
            WP_list.append(np.asarray(ln.fixed_pt))
            ln = ln.parent

        self.way_points = np.vstack(WP_list[::-1])
        print("\nExtract Way Points complete")


    def _save_pts(self, pts, name, pt_size = 20):
        self._draw_captain_path(self.fig, self.ax)

        pdf_dir = f"{self.SAVE_DIR}/{self.port["legend"]}" if self.ps.PDF else None
        save_figures.save_pts(self.fig, self.ax, pts, self.pp_start, self.pp_end, self.handles,
                               self.SAVE_DIR, name, self.legends, pdf=self.ps.PDF, pdf_dir=pdf_dir,
                               pt_size=pt_size)


    def _draw_captain_path(self, fig, ax):
        df_captain = glob.glob(f"raw_datas/tmp/_{self.port['name']}/*.csv")
        for i, df in enumerate(df_captain):
            traj = RealTraj()
            traj.input_csv(df, self.port_csv)
            h, = ax.plot(traj.Y, traj.X,
                        color = 'gray', ls = '-', marker = 'D',
                        markersize = 2, alpha = 0.3, lw = 1.0, zorder = 3)

            self.handles.append(h)

        if self.ps.SHOW_SHIP_SHAPES:
            ship_shapes = self._compute_ship_shapes()
            self.handles.extend(save_figures.draw_ship_shapes(ax, ship_shapes))


    def _search_arc_corner(self, pt1, pt2, pt3):
        cal = self.cal
        cost_cal = self.cost_cal

        SD_least = np.inf
        L1 = np.linalg.norm(pt1 - pt2)
        L2 = np.linalg.norm(pt2 - pt3)
        alpha = np.arccos(np.clip(np.dot(cal.unit(pt1-pt2), cal.unit(pt3-pt2)), -1, 1))
        r_min = self.ps.min_turn_radius()
        r_max = min(L1, L2, r_min) * np.tan(alpha/2)
        R_list = np.linspace(r_min, r_max, 51)

        best = None
        for r in R_list:
            t1, t2, arc, psi, _ = fillet(pt1, pt2, pt3, r, n=20)
            SD_cost = 0.0
            for j in range(len(arc)):
                SD_cost += cost_cal.SD_penalty(self.lines, arc[j], psi[j])

            if SD_cost < SD_least:
                SD_least = SD_cost
                best = _CornerResult(
                    entry_point=t1, exit_point=t2,
                    din=float(np.linalg.norm(pt2 - t1)),
                    dout=float(np.linalg.norm(t2 - pt2)),
                    curve_points=arc, is_arc=True,
                )
        return best


    def _find_best_fillet_arc(self, pt1, pt2, pt3, arc_list):
        best = self._search_arc_corner(pt1, pt2, pt3)
        arc_list.append(best.curve_points)


    def _build_and_score_clothoid(self, pt1_xy, pt2_xy, pt3_xy, Lin, Lout, R_min, sigma_max):
        """
        座標は (x,y)=(hor,ver) で受け取り、(ver,hor)に変換して返す。
        """
        ps = self.ps
        try:
            result = build_biclothoid(pt1_xy, pt2_xy, pt3_xy, Lin, Lout)
        except ValueError:
            return None, np.inf

        cons = check_constraints(result, min_turn_radius=R_min, sigma_max=sigma_max)
        if not (cons["max_curvature_ok"] and cons["sigma_in_ok"] and cons["sigma_out_ok"]
                and cons["din_in_range"] and cons["dout_in_range"]):
            return None, np.inf

        s_total = result.s[-1]
        n_samples = max(2, int(np.ceil(s_total / ps.DELTA_S)) + 1)
        s_samples = np.linspace(0.0, s_total, n_samples)
        x_samples = np.interp(s_samples, result.s, result.p[:, 0])
        y_samples = np.interp(s_samples, result.s, result.p[:, 1])
        theta_samples = np.interp(s_samples, result.s, result.psi)

        pts_verhor = np.column_stack([y_samples, x_samples])
        # psi_compass = pi/2 - theta_ccw (北0, 時計回り正)
        psi_compass = np.pi / 2 - theta_samples
        psi_compass = (psi_compass + np.pi) % (2.0 * np.pi) - np.pi

        # ship domain コスト
        SD_cost = 0.0
        for j in range(len(pts_verhor)):
            SD_cost += self.cost_cal.SD_penalty(self.lines, pts_verhor[j], psi_compass[j])

        # 経路長コスト
        Lc = Lin + Lout
        J = ps.W_SD * SD_cost + ps.W_L * Lc

        cr = _CornerResult(
            entry_point=result.entry_point[::-1],
            exit_point=result.exit_point[::-1],
            din=result.din, dout=result.dout,
            curve_points=pts_verhor, is_arc=False,
            Lc=Lc, eta=Lin / Lc if Lc > 0 else 0.5,
        )
        return cr, J


    def _search_clothoid_corner(self, pt1, pt2, pt3, eta_list=None, lc_grid_n=None):
        """
        制約を満たす Lc が存在しない場合は None (呼び出し側でARCへフォールバック)。
        """
        ps = self.ps
        if eta_list is None:
            eta_list = [ps.CLOTHOID_ETA]
        if lc_grid_n is None:
            lc_grid_n = ps.LC_GRID_N

        pt1 = np.asarray(pt1, dtype=float)
        pt2 = np.asarray(pt2, dtype=float)
        pt3 = np.asarray(pt3, dtype=float)

        # (ver,hor) -> (x,y)=(hor,ver)
        pt1_xy, pt2_xy, pt3_xy = pt1[::-1], pt2[::-1], pt3[::-1]

        geom = compute_corner_geometry(pt1_xy, pt2_xy, pt3_xy)
        L1, L2, dpsi = geom.L1, geom.L2, geom.dpsi

        if abs(dpsi) < 1e-9:
            return _CornerResult(
                entry_point=pt2.copy(), exit_point=pt2.copy(),
                din=0.0, dout=0.0, curve_points=pt2.reshape(1, 2), is_arc=False,
                Lc=0.0, eta=0.5,
            )

        R_min = ps.min_turn_radius()
        Lc_max = min(L1, L2)
        Lc_min = 2.0 * abs(dpsi) * R_min
        if Lc_min > Lc_max:
            return None

        Lc_list = np.linspace(Lc_min, Lc_max, lc_grid_n)

        U_kts = self.cal.speed(pt2, self.lines[-1].end_pt)
        U_ms = knot_to_ms(U_kts)
        sigma_max = np.deg2rad(ps.MAX_YAW_ACCEL_DEGS2) / (U_ms ** 2)

        J_least = np.inf
        best = None
        for Lc in Lc_list:
            for eta in eta_list:
                cr, J = self._build_and_score_clothoid(
                    pt1_xy, pt2_xy, pt3_xy, eta * Lc, (1.0 - eta) * Lc, R_min, sigma_max)
                if cr is not None and J < J_least:
                    J_least = J
                    best = cr

        return best


    def _point_line_deviation(self, pt, line_a, line_b):
        pt = np.asarray(pt, dtype=float)
        line_a = np.asarray(line_a, dtype=float)
        line_b = np.asarray(line_b, dtype=float)
        d = line_b - line_a
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-9:
            return float(np.linalg.norm(pt - line_a))
        t = np.dot(pt - line_a, d) / (norm_d ** 2)
        proj = line_a + t * d
        return float(np.linalg.norm(pt - proj))


    def _joint_reoptimize_pair(self, p0, p1, p2, p3):
        """
        隣接2屈曲点(頂点p1,p2)を (Lc,eta) 同時探索し、
        dout_1 + din_2 <= |p2-p1| を満たす組を探す。見つからなければ None。
        """
        ps = self.ps
        p0 = np.asarray(p0, dtype=float); p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float); p3 = np.asarray(p3, dtype=float)
        L_shared = np.linalg.norm(p2 - p1)

        p0_xy, p1_xy, p2_xy, p3_xy = p0[::-1], p1[::-1], p2[::-1], p3[::-1]
        geom_a = compute_corner_geometry(p0_xy, p1_xy, p2_xy)
        geom_b = compute_corner_geometry(p1_xy, p2_xy, p3_xy)
        R_min = ps.min_turn_radius()

        Lc_max_a = min(geom_a.L1, geom_a.L2)
        Lc_min_a = 2.0 * abs(geom_a.dpsi) * R_min
        Lc_max_b = min(geom_b.L1, geom_b.L2)
        Lc_min_b = 2.0 * abs(geom_b.dpsi) * R_min
        if Lc_min_a > Lc_max_a or Lc_min_b > Lc_max_b:
            return None

        U_a = knot_to_ms(self.cal.speed(p1, self.lines[-1].end_pt))
        U_b = knot_to_ms(self.cal.speed(p2, self.lines[-1].end_pt))
        sigma_max_a = np.deg2rad(ps.MAX_YAW_ACCEL_DEGS2) / (U_a ** 2)
        sigma_max_b = np.deg2rad(ps.MAX_YAW_ACCEL_DEGS2) / (U_b ** 2)

        Lc_list_a = np.linspace(Lc_min_a, Lc_max_a, ps.JOINT_LC_GRID_N)
        Lc_list_b = np.linspace(Lc_min_b, Lc_max_b, ps.JOINT_LC_GRID_N)

        cand_a = []
        for Lc_a in Lc_list_a:
            for eta_a in ps.ETA_GRID:
                cr_a, J_a = self._build_and_score_clothoid(
                    p0_xy, p1_xy, p2_xy, eta_a * Lc_a, (1.0 - eta_a) * Lc_a, R_min, sigma_max_a)
                if cr_a is not None:
                    cand_a.append((cr_a, J_a))

        cand_b = []
        for Lc_b in Lc_list_b:
            for eta_b in ps.ETA_GRID:
                cr_b, J_b = self._build_and_score_clothoid(
                    p1_xy, p2_xy, p3_xy, eta_b * Lc_b, (1.0 - eta_b) * Lc_b, R_min, sigma_max_b)
                if cr_b is not None:
                    cand_b.append((cr_b, J_b))

        J_least = np.inf
        best = None
        for cr_a, J_a in cand_a:
            for cr_b, J_b in cand_b:
                if cr_a.dout + cr_b.din > L_shared:
                    continue
                J = J_a + J_b
                if J < J_least:
                    J_least = J
                    best = (cr_a, cr_b)

        return best


    def _resolve_and_generate_clothoid_path(self):
        """
        Stage1(単独探索) -> Stage2(隣接ペア共同再最適化) -> Stage3(WP削除)
        -> Stage4(強制ARC) の順で隣接屈曲点間のトリム距離重複を解消する。

        注: eta=0.5固定のクロソイド単独探索は Lc<=min(L1,L2)<=L_shared という
        性質上、クロソイド同士では原理的に重複しない。重複が起きるのはARCへ
        フォールバックした場合のみ(ARCのr_min/r_maxが鋭角で逆転するケース)。
        """
        ps = self.ps
        way_points_list = [np.asarray(p, dtype=float) for p in self.way_points]

        corners = None
        full_seq = None

        for iteration in range(ps.MAX_RESOLVE_ITERS):
            full_seq = [self.pp_start] + way_points_list + [self.pp_end]
            n = len(way_points_list)
            corners = [None] * n
            forced_arc = [False] * n

            for k in range(n):
                res = self._search_clothoid_corner(full_seq[k], full_seq[k+1], full_seq[k+2])
                if res is None:
                    corners[k] = self._search_arc_corner(full_seq[k], full_seq[k+1], full_seq[k+2])
                    forced_arc[k] = True
                else:
                    corners[k] = res

            conflicts = []
            for k in range(n - 1):
                L_shared = np.linalg.norm(full_seq[k+2] - full_seq[k+1])
                if corners[k].dout + corners[k+1].din > L_shared:
                    conflicts.append(k)

            if not conflicts:
                break

            resolved_by_removal = False
            for k in conflicts:
                joint = self._joint_reoptimize_pair(
                    full_seq[k], full_seq[k+1], full_seq[k+2], full_seq[k+3])
                if joint is not None:
                    corners[k], corners[k+1] = joint
                    continue

                d_k = self._point_line_deviation(full_seq[k+1], full_seq[k], full_seq[k+2])
                d_k1 = self._point_line_deviation(full_seq[k+2], full_seq[k+1], full_seq[k+3])
                remove_local_idx = k if d_k <= d_k1 else (k + 1)
                print(f"[clothoid] pt={way_points_list[remove_local_idx]} を削除して"
                      f"trim重複を解消します (d={min(d_k, d_k1):.2f})。")
                del way_points_list[remove_local_idx]
                resolved_by_removal = True
                break

            if resolved_by_removal:
                continue
        else:
            print(f"[clothoid] warning: MAX_RESOLVE_ITERS({ps.MAX_RESOLVE_ITERS})に到達。"
                  f"残存する重複を強制的にARCへ切替えます。")
            full_seq = [self.pp_start] + way_points_list + [self.pp_end]
            n = len(way_points_list)
            for k in range(n - 1):
                L_shared = np.linalg.norm(full_seq[k+2] - full_seq[k+1])
                if corners[k].dout + corners[k+1].din > L_shared:
                    corners[k] = self._search_arc_corner(full_seq[k], full_seq[k+1], full_seq[k+2])
                    corners[k+1] = self._search_arc_corner(full_seq[k+1], full_seq[k+2], full_seq[k+3])

        self.way_points = np.vstack(way_points_list) if way_points_list else self.way_points
        return corners, full_seq


    def _assemble_full_path(self, corners, full_seq):
        ps = self.ps

        def sample_line(a, b):
            a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
            L = np.linalg.norm(b - a)
            if L < 1e-9:
                return a.reshape(1, 2)
            n_pts = max(2, int(np.ceil(L / ps.DELTA_S)) + 1)
            t = np.linspace(0.0, 1.0, n_pts).reshape(-1, 1)
            return a + t * (b - a)

        segments = []
        prev_point = full_seq[0]
        for corner in corners:
            segments.append(sample_line(prev_point, corner.entry_point))
            segments.append(corner.curve_points)
            prev_point = corner.exit_point
        segments.append(sample_line(prev_point, full_seq[-1]))

        return np.concatenate(segments, axis=0)


    def _clothoid_corner_contexts(self, corners, full_seq):
        ps = self.ps
        n = len(corners)
        contexts = []
        optimizable = []
        for k in range(n):
            if corners[k].is_arc:
                contexts.append(None)
                optimizable.append(False)
                continue
            pt1, pt2, pt3 = full_seq[k], full_seq[k+1], full_seq[k+2]
            pt1_xy, pt2_xy, pt3_xy = pt1[::-1], pt2[::-1], pt3[::-1]
            geom = compute_corner_geometry(pt1_xy, pt2_xy, pt3_xy)
            R_min = ps.min_turn_radius()
            U_ms = knot_to_ms(self.cal.speed(pt2, self.lines[-1].end_pt))
            sigma_max = np.deg2rad(ps.MAX_YAW_ACCEL_DEGS2) / (U_ms ** 2)
            contexts.append(dict(
                pt1_xy=pt1_xy, pt2_xy=pt2_xy, pt3_xy=pt3_xy,
                R_min=R_min, sigma_max=sigma_max,
                Lc_min=2.0 * abs(geom.dpsi) * R_min,
                Lc_max=min(geom.L1, geom.L2),
            ))
            optimizable.append(True)

        L_shared = [np.linalg.norm(full_seq[k+2] - full_seq[k+1]) for k in range(n - 1)]
        return contexts, optimizable, L_shared


    def _build_cma_initial_vector(self, corners, optimizable):
        ps = self.ps
        x0, sigma0 = [], []
        for k, opt in enumerate(optimizable):
            if not opt:
                continue
            Lc_k, eta_k = corners[k].Lc, corners[k].eta
            x0.extend([Lc_k, eta_k])
            sigma0.extend([ps.CMA_SIGMA0_LC_FRAC * max(Lc_k, 1.0), ps.CMA_SIGMA0_ETA])
        return np.array(x0, dtype=float), np.array(sigma0, dtype=float)


    def _clip_clothoid_vars(self, X):
        arr = np.asarray(X, dtype=float)
        single = (arr.ndim == 1)
        if single:
            arr = arr[None, :]
        out = arr.copy()
        eta_lo, eta_hi = self.ps.CMA_ETA_BOUNDS
        n_opt = len(self._cma_opt_indices)

        for row in range(out.shape[0]):
            vec = out[row].reshape(n_opt, 2)
            for i, k in enumerate(self._cma_opt_indices):
                ctx = self._cma_contexts[k]
                vec[i, 0] = np.clip(vec[i, 0], ctx["Lc_min"], ctx["Lc_max"])
                vec[i, 1] = np.clip(vec[i, 1], eta_lo, eta_hi)
            out[row] = vec.reshape(-1)

        return out[0] if single else out


    def _clothoid_cma_objective(self, X):
        ps = self.ps
        arr = np.asarray(X, dtype=float)
        batched = True
        if arr.ndim == 1:
            arr = arr[None, :]
            batched = False

        n_opt = len(self._cma_opt_indices)
        costs = np.zeros(arr.shape[0], dtype=float)

        for row in range(arr.shape[0]):
            vec = arr[row].reshape(n_opt, 2)
            results = {}
            total = 0.0

            for i, k in enumerate(self._cma_opt_indices):
                Lc, eta = vec[i]
                ctx = self._cma_contexts[k]
                cr, J = self._build_and_score_clothoid(
                    ctx["pt1_xy"], ctx["pt2_xy"], ctx["pt3_xy"],
                    eta * Lc, (1.0 - eta) * Lc, ctx["R_min"], ctx["sigma_max"])
                if cr is None:
                    total += ps.CMA_INFEASIBLE_PENALTY
                    continue
                total += J
                results[k] = cr

            for k in range(len(self._cma_corners_fixed) - 1):
                cr_k = results.get(k, self._cma_corners_fixed[k])
                cr_k1 = results.get(k + 1, self._cma_corners_fixed[k + 1])
                overlap = cr_k.dout + cr_k1.din - self._cma_L_shared[k]
                if overlap > 0:
                    total += ps.CMA_OVERLAP_PENALTY_COEF * overlap ** 2

            costs[row] = total

        return costs if batched else float(costs[0])


    def _save_cma_restart_fig(self, restart, best_x, opt_indices, contexts, corners_fixed, full_seq):
        cma_dir = f"{self.SAVE_DIR}/cma"
        os.makedirs(cma_dir, exist_ok=True)

        best_vec = self._clip_clothoid_vars(best_x).reshape(len(opt_indices), 2)
        corners_snapshot = list(corners_fixed)
        for i, k in enumerate(opt_indices):
            Lc, eta = best_vec[i]
            ctx = contexts[k]
            cr, _ = self._build_and_score_clothoid(
                ctx["pt1_xy"], ctx["pt2_xy"], ctx["pt3_xy"],
                eta * Lc, (1.0 - eta) * Lc, ctx["R_min"], ctx["sigma_max"])
            if cr is not None:
                corners_snapshot[k] = cr

        path_pts = self._assemble_full_path(corners_snapshot, full_seq)
        way_pts = np.vstack(full_seq)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(path_pts[:, 1], path_pts[:, 0], color="tab:blue", lw=2.0, label="path")
        ax.scatter(way_pts[:, 1], way_pts[:, 0], color="tab:orange", zorder=5, s=20, label="way points")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(f"CMA-ES restart {restart}")
        fig.tight_layout()
        fig.savefig(f"{cma_dir}/restart_{restart:02d}.png", dpi=150)
        plt.close(fig)


    def _optimize_clothoid_cma_es(self, corners, full_seq):
        ps = self.ps
        contexts, optimizable, L_shared = self._clothoid_corner_contexts(corners, full_seq)
        opt_indices = [k for k, opt in enumerate(optimizable) if opt]

        if not opt_indices:
            print("[clothoid-cma] no optimizable corners, skipping")
            return corners

        self._cma_contexts = contexts
        self._cma_opt_indices = opt_indices
        self._cma_corners_fixed = corners
        self._cma_L_shared = L_shared

        x0, sigma0 = self._build_cma_initial_vector(corners, optimizable)

        ddcma = DdCma(xmean0=x0, sigma0=sigma0, seed=ps.CMA_SEED)
        checker = Checker(ddcma)
        logger = Logger(ddcma, prefix=f"{self.SAVE_DIR}/clothoid_cma_log")

        NEVAL_STANDARD = ddcma.lam * 5000
        print(f"[clothoid-cma] dimension N={ddcma.N}  population lam={ddcma.lam}  "
              f"NEVAL_STANDARD={NEVAL_STANDARD}")

        total_neval = 0
        cur_seed = int(ps.CMA_SEED)
        best_cost = np.inf
        best_x = x0.copy()
        time_start = time.time()

        for restart in range(ps.CMA_RESTARTS):
            is_satisfied = False
            t0 = time.time()

            pbar = tqdm(
                total=NEVAL_STANDARD,
                desc=f"Restart {restart}",
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:.0f}%|{bar}| {postfix}",
                mininterval=0.2,
                smoothing=0.1,
            )
            last_neval = ddcma.neval

            def _refresh_postfix():
                rate = pbar.format_dict.get("rate")
                eval_per_s = f"{rate:.1f}" if rate is not None else "-"
                pbar.set_postfix_str(
                    f"eval/s={eval_per_s}  neval={ddcma.neval}  best={best_cost:.6g}"
                )

            while not is_satisfied:
                ddcma.onestep(func=self._clothoid_cma_objective, check=self._clip_clothoid_vars)

                cur_best = float(np.min(ddcma.arf))
                if cur_best < best_cost:
                    best_cost = cur_best
                    best_x = ddcma.arx[int(ddcma.idx[0])].copy()

                is_satisfied, condition = checker()

                if ddcma.neval > last_neval:
                    pbar.update(ddcma.neval - last_neval)
                    last_neval = ddcma.neval
                    _refresh_postfix()

                if ddcma.t % 10 == 0:
                    pbar.write(f"neval:{ddcma.neval:<6}  cost:{cur_best:<10.6g}  best:{best_cost:<10.6g}")

            _refresh_postfix()
            pbar.close()

            logger(condition)
            elapsed = time.time() - t0
            total_neval += ddcma.neval
            print(f"[clothoid-cma] restart {restart} terminated: condition={condition}  "
                  f"best_cost={best_cost:.6g}  neval={ddcma.neval}  time={elapsed:.2f}s")

            self._save_cma_restart_fig(restart, best_x, opt_indices, contexts, corners, full_seq)

            if total_neval < NEVAL_STANDARD:
                popsize = ddcma.lam if not ps.CMA_INCREASE_POPSIZE_ON_RESTART else ddcma.lam * 2
                cur_seed *= 2
                ddcma = DdCma(xmean0=x0, sigma0=sigma0, lam=popsize, seed=cur_seed)
                checker = Checker(ddcma)
                logger.setcma(ddcma)
                print(f"[clothoid-cma] restarting with popsize={ddcma.lam}")
            else:
                break

        print(f"[clothoid-cma] optimization complete in {time.time() - time_start:.2f}s  "
              f"total_neval={total_neval}  best_cost={best_cost:.6g}")

        best_vec = self._clip_clothoid_vars(best_x).reshape(len(opt_indices), 2)

        new_corners = list(corners)
        for i, k in enumerate(opt_indices):
            Lc, eta = best_vec[i]
            ctx = contexts[k]
            cr, J = self._build_and_score_clothoid(
                ctx["pt1_xy"], ctx["pt2_xy"], ctx["pt3_xy"],
                eta * Lc, (1.0 - eta) * Lc, ctx["R_min"], ctx["sigma_max"])
            if cr is not None:
                new_corners[k] = cr

        return new_corners


    def _compute_ship_poses(self, interval_sec=60, dt=1.0, max_markers=500):
        """
        着桟位置(self.pp_end)から Calculator.speed に基づいて interval_sec 秒
        (既定1分)ごとの位置を推定し、複数csv(実航跡)の平均座標として
        (ver, hor, psi) のリストを返す。

        各csvは配列の末尾が着桟側なので、末尾を起点に逆順へたどりながら
        累積距離(弧長)を測る。ある時点で到達できたcsvだけを平均に使い、
        1つも到達できなくなった時点で打ち切る。
        """
        traj_files = glob.glob(f"raw_datas/tmp/_{self.port['name']}/*.csv")

        curves = []
        for file in traj_files:
            traj = RealTraj()
            traj.input_csv(file, self.port_csv)
            if len(traj.X) < 2:
                continue

            ver_rev = np.asarray(traj.X, dtype=float)[::-1]
            hor_rev = np.asarray(traj.Y, dtype=float)[::-1]
            seg = np.hypot(np.diff(ver_rev), np.diff(hor_rev))
            cum_dist = np.concatenate([[0.0], np.cumsum(seg)])
            curves.append((cum_dist, ver_rev, hor_rev))

        if not curves:
            return []

        avg_pts = []
        k = 1
        while k <= max_markers:
            marks = []
            target_t = k * interval_sec
            for cum_dist, ver_rev, hor_rev in curves:
                s, t = 0.0, 0.0
                max_s = cum_dist[-1]
                while t < target_t and s < max_s:
                    pt = np.array([np.interp(s, cum_dist, ver_rev),
                                   np.interp(s, cum_dist, hor_rev)])
                    speed_kts = self.cal.speed(pt, self.pp_end)
                    s += knot_to_ms(speed_kts) * dt
                    t += dt
                if t >= target_t:
                    marks.append([np.interp(s, cum_dist, ver_rev),
                                  np.interp(s, cum_dist, hor_rev)])

            if not marks:
                break

            avg_pts.append(np.mean(marks, axis=0))
            k += 1

        points = [np.array([0.0, 0.0])] + avg_pts
        poses = []
        for i in range(1, len(points)):
            current_pt = points[i]
            nearer_pt = points[i - 1]

            v_out = nearer_pt - current_pt
            if np.linalg.norm(v_out) < 1e-9:
                farther_pt = points[i + 1] if i + 1 < len(points) else points[i]
                v_out = current_pt - farther_pt

            psi = self.cal.angle(np.zeros(2), v_out)
            poses.append((current_pt[0], current_pt[1], psi))

        return poses


    def _compute_ship_shapes(self):
        """
        _compute_ship_poses で得た (ver, hor, psi) を、船体多角形(hull)の
        座標列に変換する。返す座標は ax.fill にそのまま渡せる (plot_x, plot_y) 順。

        加えて、バース位置(原点)にも船首方位を真上(0度)に固定した船型を1つ追加する。
        """
        shapes = []

        berth_hull = ship_shape_poly((0, 0, 0.0), L=self.ps.L, B=self.ps.B)
        shapes.append(np.asarray(berth_hull))

        for ver, hor, psi in self._compute_ship_poses():
            hull = ship_shape_poly((ver, hor, psi), L=self.ps.L, B=self.ps.B)
            shapes.append(np.asarray(hull))
        return shapes


    def save_result_fig(self):
        self._draw_captain_path(self.fig, self.ax)

        save_figures.save_result_fig(
            self.fig, self.ax, self.save_dir_path, self._make_folder_name(),
            self.pp_start, self.pp_end, self.result_pts, self.way_points,
            self.ps.approach_algo.name, self.ps.SupplementMode.name, self.ps.redraw_by_AI,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Path planning for ship berthing approach")
    parser.add_argument(
        "port_number", type=int, nargs="?", default=0,
        help="0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Sakaide, 4: Osaka_1B, "
             "5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu, "
             "10: Tomakomai, 11: KIX"
    )
    args = parser.parse_args()

    ps = Setting()
    ps.port_number = args.port_number
    sd = ShipDomain()
    cal = Calculator(ps, sd)
    cost_cal = CostCalculator(ps, sd, cal)

    pp = PathPlanning(ps, sd, cal, cost_cal)
    pp.main()
