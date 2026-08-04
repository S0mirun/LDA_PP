import argparse
import glob
import os

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
from typing import ClassVar, Tuple

from utils.LDA.ship_geometry import *
from utils.PP.dictionary_of_port import dictionary
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
        self.approach_algo = ApproachAlgo.ARC
        self.SupplementMode = SupplementMode.MIDPOINT
        self.redraw_by_AI = True

        # CMA-ES
        self.seed: int = 42
        self.MAX_SPEED_KTS: float = 9.5  # [knots]
        self.MIN_SPEED_KTS: float = 1.5  # [knots]
        self.speed_interval: float = 1.0
        self.MAX_ANGLE_DEG: float = 60  # [deg]
        self.MIN_ANGLE_DEG: float = 0  # [deg]
        self.angle_interval: float = 5

        # others
        self.PDF = True



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


    def SD_penalty(self, lines, pt, psi, speed_base_pt=None):
        SD = self.SD
        SD_p = SD.SD_p

        base_pt = speed_base_pt if speed_base_pt is not None else lines[-1].end_pt
        speed = self.cal.speed(pt, base_pt)
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
        self.legends.append(save_figures.LEGEND_SHIP_SHAPE)
        self._save_pts(self.way_points, "way_points")

        WP = self.way_points
        if self.ps.approach_algo == ApproachAlgo.ARC:
            full_pts = np.vstack([self.pp_start, self.way_points, self.pp_end])

            arc_list = []
            psi_list = []
            for i in range(len(self.way_points)):
                self._find_best_fillet_arc(full_pts[i], full_pts[i+1], full_pts[i+2], arc_list, psi_list)

            arcs = np.concatenate(arc_list, axis=0)
            self.result_pts = arcs
            self.arc_list = arc_list
            self.psi_list = psi_list
            print("\nFillet arc path complete")

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

        ship_shapes = self._compute_ship_shapes()
        self.handles.extend(save_figures.draw_ship_shapes(ax, ship_shapes))


    def _find_best_fillet_arc(self, pt1, pt2, pt3, arc_list, psi_list):
        cal = self.cal
        cost_cal = self.cost_cal

        SD_least = np.inf
        L1 = np.linalg.norm(pt1 - pt2)
        L2 = np.linalg.norm(pt2 - pt3)
        alpha = np.arccos(np.clip(np.dot(cal.unit(pt1-pt2), cal.unit(pt3-pt2)), -1, 1))
        r_min = (3.3 * self.ps.L * 2) / 2
        r_max = min(L1, L2, r_min) * np.tan(alpha/2)
        R_list = np.linspace(r_min, r_max, 51)

        for r in R_list:
            _, _, arc, psi, _ = fillet(pt1, pt2, pt3, r, n=20)
            # ship domain
            SD_cost = 0.0
            for j in range(len(arc)):
                SD_cost += cost_cal.SD_penalty(self.lines, arc[j], psi[j])

            if SD_least > SD_cost:
                arc_best = arc
                psi_best = psi
                SD_least = SD_cost

        arc_list.append(arc_best)
        psi_list.append(psi_best)


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


import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.Filtered_Dict import new_filtered_dict


def _segment_psi(p_from: np.ndarray, p_to: np.ndarray) -> float:
    p_from = np.asarray(p_from, dtype=float)
    p_to = np.asarray(p_to, dtype=float)
    dver = p_to[0] - p_from[0]
    dhor = p_to[1] - p_from[1]
    return float(np.arctan2(dhor, dver))


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_goal_point(pp) -> np.ndarray:
    return np.array([0.0, 0.0], dtype=float)


def build_path_segments(pp) -> List[tuple]:
    segs = []
    prev_pt = np.asarray(pp.pp_start, dtype=float)
    arc_list = getattr(pp, "arc_list", [])
    psi_list = getattr(pp, "psi_list", [])

    for arc, psi in zip(arc_list, psi_list):
        t1 = np.asarray(arc[0], dtype=float)
        if np.linalg.norm(t1 - prev_pt) > 1e-6:
            segs.append(("straight", prev_pt, t1))
        segs.append(("arc", np.asarray(arc, dtype=float), np.asarray(psi, dtype=float)))
        prev_pt = np.asarray(arc[-1], dtype=float)

    pp_end = np.asarray(pp.pp_end, dtype=float)
    if np.linalg.norm(pp_end - prev_pt) > 1e-6:
        segs.append(("straight", prev_pt, pp_end))
    prev_pt = pp_end

    goal = get_goal_point(pp)
    if np.linalg.norm(goal - prev_pt) > 1e-6:
        segs.append(("straight", prev_pt, goal))

    return segs


def _segment_length(seg: tuple) -> float:
    if seg[0] == "straight":
        return float(np.linalg.norm(seg[2] - seg[1]))
    arc = seg[1]
    return float(np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1)))


def point_at_distance_from_end(segs: List[tuple], distance: float) -> Tuple[np.ndarray, float]:
    remaining = float(distance)
    for seg_idx in range(len(segs) - 1, -1, -1):
        seg = segs[seg_idx]
        seg_len = _segment_length(seg)
        if remaining <= seg_len or seg_idx == 0:
            if seg[0] == "straight":
                p_a, p_b = seg[1], seg[2]
                frac = 0.0 if seg_len < 1e-9 else min(remaining / seg_len, 1.0)
                pt = p_b - frac * (p_b - p_a)
                phi = _segment_psi(p_a, p_b)
            else:
                arc, psi = seg[1], seg[2]
                seg_dists = np.concatenate(
                    [[0.0], np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1))]
                )
                target = max(seg_len - remaining, 0.0)
                j = int(np.searchsorted(seg_dists, target))
                j = int(np.clip(j, 0, len(arc) - 1))
                pt = arc[j]
                phi = float(psi[j])
            return np.asarray(pt, dtype=float), float(phi)
        remaining -= seg_len

    first = segs[0]
    if first[0] == "straight":
        return np.asarray(first[1], dtype=float), _segment_psi(first[1], first[2])
    return np.asarray(first[1][0], dtype=float), float(first[2][0])


def _distance_from_end_along(segs: List[tuple], point: np.ndarray, tol: float = 1.0) -> float:
    point = np.asarray(point, dtype=float)
    remaining_from_pp_end = 0.0
    for seg in reversed(segs):
        if seg[0] == "straight":
            p_a, p_b = seg[1], seg[2]
            if np.linalg.norm(point - p_a) < tol:
                return remaining_from_pp_end + float(np.linalg.norm(point - p_b))
            remaining_from_pp_end += float(np.linalg.norm(p_b - p_a))
        else:
            arc = seg[1]
            seg_dists = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1))]
            )
            diffs = np.linalg.norm(arc - point, axis=1)
            j = int(np.argmin(diffs))
            if diffs[j] < tol:
                return remaining_from_pp_end + float(seg_dists[-1] - seg_dists[j])
            remaining_from_pp_end += float(seg_dists[-1])
    return remaining_from_pp_end


def determine_boundary(pp) -> Tuple[np.ndarray, float, List[tuple]]:
    L = float(pp.ps.L)
    d5 = 5.0 * L
    segs = build_path_segments(pp)

    last_arc_seg = None
    for seg in reversed(segs):
        if seg[0] == "arc":
            last_arc_seg = seg
            break

    if last_arc_seg is not None:
        arc, psi = last_arc_seg[1], last_arc_seg[2]
        t1, t2 = arc[0], arc[-1]
        d_t1 = _distance_from_end_along(segs, t1)
        d_t2 = _distance_from_end_along(segs, t2)
        if d_t2 < d5 <= d_t1:
            return np.asarray(t1, dtype=float), float(psi[0]), segs

    boundary_pt, boundary_phi = point_at_distance_from_end(segs, d5)
    return boundary_pt, boundary_phi, segs


def _densify_segs_from_point(segs: List[tuple], boundary_pt: np.ndarray, tol: float = 1.0) -> np.ndarray:
    boundary_pt = np.asarray(boundary_pt, dtype=float)
    out = []
    started = False
    for seg in segs:
        if seg[0] == "straight":
            p_a, p_b = seg[1], seg[2]
            if not started:
                if _point_on_segment(boundary_pt, p_a, p_b, tol):
                    started = True
                    out.append(boundary_pt[None, :])
                    out.append(p_b[None, :])
                continue
            out.append(p_b[None, :])
        else:
            arc = seg[1]
            if not started:
                if np.linalg.norm(boundary_pt - arc[0]) < tol:
                    started = True
                    out.append(arc)
                elif _point_in_arc(boundary_pt, arc, tol):
                    started = True
                    j = int(np.argmin(np.linalg.norm(arc - boundary_pt, axis=1)))
                    out.append(boundary_pt[None, :])
                    out.append(arc[j:])
                continue
            out.append(arc)
    if not out:
        raise RuntimeError("boundary_pt が経路(segs)上に見つかりませんでした。")
    return np.vstack(out)


def _point_on_segment(pt, p_a, p_b, tol=1.0) -> bool:
    d = p_b - p_a
    if np.linalg.norm(d) < 1e-9:
        return bool(np.linalg.norm(pt - p_a) < tol)
    t = np.dot(pt - p_a, d) / np.dot(d, d)
    if t < -1e-6 or t > 1 + 1e-6:
        return False
    proj = p_a + np.clip(t, 0.0, 1.0) * d
    return bool(np.linalg.norm(proj - pt) < tol)


def _point_in_arc(pt, arc, tol=1.0) -> bool:
    diffs = np.linalg.norm(arc - pt, axis=1)
    return bool(np.min(diffs) < tol)


def sample_1min_points(pp, boundary_pt: np.ndarray, segs: List[tuple], calculator) -> np.ndarray:
    goal = get_goal_point(pp)
    dense_path = _densify_segs_from_point(segs, boundary_pt)

    seg_d = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(dense_path, axis=0), axis=1))]
    )
    total = float(seg_d[-1])

    turning_points = []
    idx = 0
    while idx < len(dense_path) - 1:
        cur = dense_path[idx]
        speed = calculator.speed(cur, goal)
        minute_distance = speed * 1852.0 / 60.0

        target = seg_d[idx] + minute_distance
        if target >= total - 1e-6:
            break
        j = int(np.searchsorted(seg_d, target))
        j = int(np.clip(j, idx + 1, len(dense_path) - 1))
        turning_points.append(dense_path[j].copy())
        idx = j
        return np.array(turning_points) if turning_points else np.empty((0, 2))


def _sigmoid(x, a, b, c):
    return a / (b + np.exp(c * x))


class BerthCostCalculator:
    def __init__(self, ps, cal, base_cost_cal):
        self.ps = ps
        self.cal = cal
        self.base = base_cost_cal

        self.MIN_SPEED_KTS = float(ps.MIN_SPEED_KTS)
        self.MAX_SPEED_KTS = float(ps.MAX_SPEED_KTS)
        self.speed_interval = float(ps.speed_interval)
        self.angle_interval = int(ps.angle_interval)
        self.angle_min = int(ps.MIN_ANGLE_DEG)
        self.angle_max = int(ps.MAX_ANGLE_DEG)
        self.speed_bins = np.arange(self.MIN_SPEED_KTS, self.MAX_SPEED_KTS, self.speed_interval)
        self.angle_bins = np.arange(self.angle_min, self.angle_max, self.angle_interval)
        self.new_filtered_dict = new_filtered_dict()

    def distance_cost_between(self, current_pt, child_pt, speed_ref_pt) -> float:
        speed = self.cal.speed(current_pt, speed_ref_pt)
        ideal = speed * 1852.0 / 60.0
        real = float(np.linalg.norm(np.asarray(child_pt) - np.asarray(current_pt)))
        if ideal <= 1e-9:
            return 0.0
        return float(abs(ideal - real) / ideal * 100.0)

    def elem_from_phi(self, current_pt, phi_cur: float, phi_prev: float, speed_ref_pt) -> float:
        current_speed = self.cal.speed(current_pt, speed_ref_pt)

        if current_speed < self.MIN_SPEED_KTS:
            speed_key = self.MIN_SPEED_KTS
        elif current_speed >= self.MAX_SPEED_KTS:
            speed_key = self.MAX_SPEED_KTS
        else:
            speed_key = None
            for s0 in self.speed_bins:
                if s0 <= current_speed < s0 + self.speed_interval:
                    speed_key = float(s0)
                    break
            if speed_key is None:
                speed_key = self.MAX_SPEED_KTS

        dphi = _wrap_pi(np.array([phi_cur - phi_prev]))[0]
        angle_deg = float(np.degrees(abs(dphi)))

        if angle_deg >= self.angle_max:
            angle_key = float(self.angle_max)
        else:
            angle_key = None
            for a0 in self.angle_bins:
                if a0 <= angle_deg < a0 + self.angle_interval:
                    angle_key = float(a0)
                    break
            if angle_key is None:
                angle_key = float(self.angle_max)

        occ = self.new_filtered_dict[speed_key][angle_key]
        return float(100.0 - occ)

    def angle_diff_cost(self, boundary_dir, end_dir, dist_total, boundary_pt,
                         current_pt, child_pt) -> float:
        ver_c, hor_c = current_pt
        ver_ch, hor_ch = child_pt

        v1 = np.asarray(boundary_dir, dtype=float)
        v2 = np.asarray(end_dir, dtype=float)
        v3 = np.array([hor_ch - hor_c, ver_ch - ver_c], dtype=float)
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)
        m3 = np.linalg.norm(v3)

        if m1 == 0.0 or m3 == 0.0:
            angle_deg_s = 0.0
        else:
            cos_s = np.clip(np.dot(v1, v3) / (m1 * m3), -1.0, 1.0)
            angle_deg_s = float(np.degrees(np.arccos(cos_s)))
        if m2 == 0.0 or m3 == 0.0:
            angle_deg_e = 0.0
        else:
            cos_e = np.clip(np.dot(v2, v3) / (m2 * m3), -1.0, 1.0)
            angle_deg_e = float(np.degrees(np.arccos(cos_e)))

        dist_c = float(np.linalg.norm(np.asarray(current_pt) - np.asarray(boundary_pt)))
        progress = dist_c / dist_total if dist_total > 1e-9 else 0.0

        alpha = 0.3
        w_boundary = _sigmoid(progress - alpha, a=angle_deg_s, b=1.0, c=30.0)
        w_end = _sigmoid(progress - (1 - alpha), a=angle_deg_e, b=1.0, c=30.0)

        return -(w_boundary + w_end)

    def deviation_cost(self, current_pt, init_pt, sigma_pos_m) -> float:
        d = float(np.linalg.norm(np.asarray(current_pt) - np.asarray(init_pt)))
        return (d / sigma_pos_m) ** 2 * 100.0


@dataclass
class BerthOptimizerSettings:
    seed: int = 42
    restarts: int = 3
    increase_popsize_on_restart: bool = False

    SD_ratio: float = 0.5
    element_ratio: float = 1.0
    distance_ratio: float = 0.2
    angle_diff_ratio: float = 1.5
    deviation_ratio: float = 1.0

    sigma_pos_m: float = 10.0
    sigma_phi_deg: float = 10.0


class BerthApproachOptimizer:
    def __init__(self, pp, settings: Optional[BerthOptimizerSettings] = None):
        self.pp = pp
        self.os = settings or BerthOptimizerSettings()
        self.cal = pp.cal
        self.cost_cal = BerthCostCalculator(pp.ps, pp.cal, pp.cost_cal)

    def setup(self):
        boundary_pt, boundary_phi, segs = determine_boundary(self.pp)
        self.boundary_pt = boundary_pt
        self.boundary_phi = boundary_phi
        self.segs = segs

        interior_pts = sample_1min_points(self.pp, boundary_pt, segs, self.cal)
        if len(interior_pts) == 0:
            raise RuntimeError(
                "境界点からgoal(0,0)までの区間で1分刻みの分割点が得られませんでした。"
                f" boundary_pt={boundary_pt}, pp_end={self.pp.pp_end}, goal={get_goal_point(self.pp)}"
            )
        self.interior_pts = interior_pts

        interior_phi = []
        for pt in interior_pts:
            dist_from_goal = self._nearest_path_distance(pt, segs)
            _, phi = point_at_distance_from_end(segs, dist_from_goal)
            interior_phi.append(phi)
        self.interior_phi0 = np.array(interior_phi)

        self.initial_vec = np.column_stack(
            [interior_pts[:, 0], interior_pts[:, 1], self.interior_phi0]
        ).ravel()
        self.N = len(self.initial_vec)
        self.initial_D = self._build_sigma_vector(len(interior_pts))

        self.pp_end = np.asarray(self.pp.pp_end, dtype=float)
        self.goal = get_goal_point(self.pp)
        self.psi_end = float(np.deg2rad(self.pp.port["psi_end"]))
        self.boundary_dir = np.array([np.sin(self.boundary_phi), np.cos(self.boundary_phi)])
        self.end_dir = np.array([np.sin(self.psi_end), np.cos(self.psi_end)])
        self.dist_total = float(np.linalg.norm(self.goal - self.boundary_pt))

        print(
            f"[setup] boundary_pt={self.boundary_pt} (phi={np.degrees(self.boundary_phi):.1f}deg), "
            f"内部分割点数={len(interior_pts)}, 次元数N={self.N}"
        )

    def _nearest_path_distance(self, pt: np.ndarray, segs: List[tuple]) -> float:
        best = None
        remaining_from_goal = 0.0
        for seg in reversed(segs):
            if seg[0] == "straight":
                p_a, p_b = seg[1], seg[2]
                d = _point_on_segment_distance(pt, p_a, p_b)
                if d is not None:
                    cand = remaining_from_goal + float(np.linalg.norm(pt - p_b))
                    if best is None or cand < best:
                        best = cand
                remaining_from_goal += float(np.linalg.norm(p_b - p_a))
            else:
                arc = seg[1]
                diffs = np.linalg.norm(arc - pt, axis=1)
                j = int(np.argmin(diffs))
                if diffs[j] < 1.0:
                    seg_dists = np.concatenate(
                        [[0.0], np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1))]
                    )
                    cand = remaining_from_goal + float(seg_dists[-1] - seg_dists[j])
                    if best is None or cand < best:
                        best = cand
                remaining_from_goal += float(_segment_length(seg))
        return best if best is not None else remaining_from_goal

    def _build_sigma_vector(self, n_points: int) -> np.ndarray:
        sigma_pos = np.full(n_points, self.os.sigma_pos_m, dtype=float)
        sigma_phi = np.full(n_points, np.deg2rad(self.os.sigma_phi_deg), dtype=float)
        return np.column_stack([sigma_pos, sigma_pos, sigma_phi]).ravel()

    def _raw_costs(self, pts: np.ndarray, phis: np.ndarray) -> Tuple[float, float, float, float, float]:
        full_pts = np.vstack([self.boundary_pt, pts, self.goal])
        full_phi = np.concatenate([[self.boundary_phi], phis, [self.psi_end]])

        SD_cost = 0.0
        for j in range(1, len(full_pts) - 1):
            SD_cost += self.cost_cal.base.SD_penalty(
                self.pp.lines, full_pts[j], full_phi[j], speed_base_pt=self.goal
            )

        elem_cost = 0.0
        for j in range(1, len(full_pts) - 1):
            elem_cost += self.cost_cal.elem_from_phi(
                full_pts[j], full_phi[j], full_phi[j - 1], self.goal
            )

        dist_cost = 0.0
        for j in range(len(full_pts) - 1):
            dist_cost += self.cost_cal.distance_cost_between(
                full_pts[j], full_pts[j + 1], self.goal
            )

        angle_cost = 0.0
        for j in range(1, len(full_pts) - 1):
            angle_cost += self.cost_cal.angle_diff_cost(
                self.boundary_dir, self.end_dir, self.dist_total,
                self.boundary_pt, full_pts[j], full_pts[j + 1]
            )

        deviation_cost = 0.0
        for j in range(len(pts)):
            deviation_cost += self.cost_cal.deviation_cost(
                pts[j], self.interior_pts[j], self.os.sigma_pos_m
            )

        return SD_cost, elem_cost, dist_cost, angle_cost, deviation_cost

    def compute_cost_weights(self):
        SD_cost, elem_cost, dist_cost, angle_cost, deviation_cost = self._raw_costs(
            self.interior_pts, self.interior_phi0
        )

        n_points = len(self.interior_pts)
        self.element_coeff = 1.0 * self.os.element_ratio
        self.SD_coeff = (elem_cost / SD_cost) * self.os.SD_ratio if SD_cost > 0 else 10.0
        self.distance_coeff = (elem_cost / dist_cost) * self.os.distance_ratio if dist_cost > 0 else 1.0
        self.angle_diff_coeff = (
            (elem_cost / abs(angle_cost)) * self.os.angle_diff_ratio if abs(angle_cost) > 1e-9 else 0.0
        )
        self.deviation_coeff = (elem_cost / n_points / 100.0) * self.os.deviation_ratio if n_points > 0 else 0.0

        print(
            "[compute_cost_weights] 初期解のコスト生値: "
            f"SD={SD_cost:.4g} elem={elem_cost:.4g} dist={dist_cost:.4g} angle={angle_cost:.4g} "
            f"deviation={deviation_cost:.4g}"
        )
        print(
            "[compute_cost_weights] 係数: "
            f"SD_coeff={self.SD_coeff:.4g} element_coeff={self.element_coeff:.4g} "
            f"distance_coeff={self.distance_coeff:.4g} angle_diff_coeff={self.angle_diff_coeff:.4g} "
            f"deviation_coeff={self.deviation_coeff:.4g}"
        )

    def path_evaluate(self, X: np.ndarray):
        arr = np.asarray(X, dtype=float)
        batched = True
        if arr.ndim == 1:
            arr = arr[None, :]
            batched = False

        costs = np.zeros(arr.shape[0], dtype=float)
        for i in range(arr.shape[0]):
            triplets = arr[i].reshape(-1, 3)
            pts = triplets[:, :2]
            phis = _wrap_pi(triplets[:, 2])

            SD_cost, elem_cost, dist_cost, angle_cost, deviation_cost = self._raw_costs(pts, phis)

            costs[i] = (
                self.SD_coeff * SD_cost
                + self.element_coeff * elem_cost
                + self.distance_coeff * dist_cost
                + self.angle_diff_coeff * angle_cost
                + self.deviation_coeff * deviation_cost
            )

        return float(costs[0]) if not batched else costs

    def run(self):
        ddcma = DdCma(xmean0=self.initial_vec, sigma0=self.initial_D, seed=self.os.seed)
        checker = Checker(ddcma)

        NEVAL_STANDARD = ddcma.lam * 5000
        print(f"[run] population size={ddcma.lam}, dimension={ddcma.N}, "
              f"NEVAL_STANDARD={NEVAL_STANDARD}")

        total_neval = 0
        best_dict = {}
        cur_seed = int(self.os.seed)
        t0 = time.time()

        for restart in range(self.os.restarts):
            best_dict[restart] = {"best_cost_so_far": float("inf"), "best_mean_sofar": None}
            is_satisfied = False

            while not is_satisfied:
                ddcma.onestep(func=self.path_evaluate, check=None)

                best_cost = float(np.min(ddcma.arf))
                if best_cost < best_dict[restart]["best_cost_so_far"]:
                    best_dict[restart]["best_cost_so_far"] = best_cost
                    best_dict[restart]["best_mean_sofar"] = ddcma.arx[int(ddcma.idx[0])].copy()

                is_satisfied, condition = checker()

                if ddcma.t % 10 == 0:
                    print(f"restart={restart} t={ddcma.t} neval={ddcma.neval} "
                          f"best={best_dict[restart]['best_cost_so_far']:.6g}")

            print(f"[run] restart {restart} terminated: {condition}")
            total_neval += ddcma.neval

            if restart < self.os.restarts - 1 and total_neval < NEVAL_STANDARD:
                popsize = ddcma.lam if not self.os.increase_popsize_on_restart else ddcma.lam * 2
                cur_seed *= 2
                ddcma = DdCma(xmean0=self.initial_vec, sigma0=self.initial_D,
                               lam=popsize, seed=cur_seed)
                checker = Checker(ddcma)
            else:
                break

        self.best_dict = best_dict
        self.cma_caltime = time.time() - t0
        self.best_key = min(best_dict, key=lambda k: best_dict[k]["best_cost_so_far"])
        best_mean = best_dict[self.best_key]["best_mean_sofar"]

        triplets = best_mean.reshape(-1, 3)
        self.optimized_pts = triplets[:, :2]
        self.optimized_phi = _wrap_pi(triplets[:, 2])
        self.optimized_full_path = np.vstack([self.boundary_pt, self.optimized_pts, self.goal])

        print(
            f"[run] 完了(所要時間 {self.cma_caltime:.1f}s)。"
            f"best_cost={best_dict[self.best_key]['best_cost_so_far']:.6g}"
        )
        return self.optimized_full_path, self.optimized_phi

    def save_result_fig(self, name: str = "berth_optimized_path"):
        self._draw_optimized_overlay()
        self.pp._save_pts(self.optimized_full_path, name, pt_size=8)
        print(f"[save_result_fig] '{name}' を保存しました(SAVE_DIR={self.pp.SAVE_DIR})。")

    def _draw_optimized_overlay(self):
        ax = self.pp.ax
        pts = self.optimized_pts
        phis = self.optimized_phi

        ax.scatter(pts[:, 1], pts[:, 0], color="red", s=25, zorder=6)
        for (ver, hor), psi in zip(pts, phis):
            hull = np.asarray(ship_shape_poly((ver, hor, psi), L=self.pp.ps.L, B=self.pp.ps.B))
            ax.fill(hull[:, 0], hull[:, 1], facecolor="red", alpha=0.3,
                    edgecolor="red", linewidth=1.0, zorder=6)


def _point_on_segment_distance(pt, p_a, p_b, tol=1.0):
    d = p_b - p_a
    if np.linalg.norm(d) < 1e-9:
        return None
    t = np.dot(pt - p_a, d) / np.dot(d, d)
    if t < -1e-6 or t > 1 + 1e-6:
        return None
    proj = p_a + np.clip(t, 0.0, 1.0) * d
    if np.linalg.norm(proj - pt) < tol:
        return float(np.linalg.norm(pt - p_a))
    return None


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

    opt = BerthApproachOptimizer(pp, BerthOptimizerSettings())
    opt.setup()
    opt.compute_cost_weights()
    opt.run()
    opt.save_result_fig()