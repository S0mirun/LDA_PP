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
from shapely import contains_xy, intersects_xy, prepare
from shapely.geometry import Polygon, Point, LineString
from shapely.prepared import prep
from shapely.validation import make_valid
from typing import ClassVar, Tuple
from tqdm.auto import tqdm

from utils.LDA.ship_geometry import *
from utils.PP import Bezier_curve as Bezier
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.dictionary_of_port import dictionary
from utils.PP.graph_by_taneichi import ShipDomain_proposal
from utils.PP.MultiPlot import RealTraj

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
theta_list = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(3))


class Setting:
    def __init__(self):
        # port
        self.port_number: int = 7
         # 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Else_1, 4: Osaka_1B
         # 5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu
         # 10: Tomakomai, 11: KIX

        # ship
        self.L = 103.8
        self.B = 16.0

        # CMA-ES
        self.seed: int = 42
        self.MAX_SPEED_KTS: float = 9.5  # [knots]
        self.MIN_SPEED_KTS: float = 1.5  # [knots]
        self.speed_interval: float = 1.0
        self.MAX_ANGLE_DEG: float = 60  # [deg]
        self.MIN_ANGLE_DEG: float = 0  # [deg]
        self.angle_interval: float = 5

        # weight
        self.SD = 1.5
        self.angle = 1.0
        self.xy = 0.1

        # restart
        self.restarts: int = 3
        self.increase_popsize_on_restart: bool = False


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
        self.fixed_pt = cal_intersect_pt(self, ln)
        self.parent = ln


class CostCalculator:
    def __init__(self):
        self.ps = None
        self.SD = None
        self.lines = None

    def ShipDomain_penalty(self, parent_pt, current_pt, child_pt):
        """
        岸壁との接触を shapely により判定。
        壁の中にある点の数をばつとして与える。
        """
        SD = self.SD
        lines = self.lines

        speed = cal_speed(self, current_pt, lines[-1].end_pt)
        psi = cal_psi(parent_pt, current_pt, child_pt)

        r_list = []
        for theta_i in theta_list:
            r_list.append(SD.distance(speed, theta_i))

        r = np.asarray(r_list, dtype=float)
        domain_xy = np.column_stack([
            current_pt[0] + r * np.cos(theta_list + psi),
            current_pt[1] + r * np.sin(theta_list + psi),
        ])
        # count
        xs = domain_xy[:, 1]; ys = domain_xy[:, 0]
        hit = intersects_xy(Line.map_poly, xs, ys)
        n_hit = int(np.count_nonzero(hit))

        return n_hit
    
    def spacing_penalty(self, current_pt, child_pt):
        """
        CMA-ESで動かしている点は1分後の船体の位置を表している。
        これが近すぎる/遠すぎるときに罰を与える。
        """
        lines = self.lines

        d_max = self.ps.MAX_SPEED_KTS * 1852 / 60 # [m]
        d_min = self.ps.MIN_SPEED_KTS * 1852 / 60 # [m]
        dist = np.linalg.norm(current_pt - child_pt)

        # rangeの外にあると大きな罰を与える
        out_penalty = max(0, dist - d_min) ** 2 + max(0, d_max - dist) ** 2

        # 理想的な距離からの変化に応じて軽い罰を与える
        speed = cal_speed(self, current_pt, lines[-1].end_pt)
        ideal_dist = speed * 1852 / 60 # [m/min]
        space_penalty = (abs(dist - ideal_dist) / ideal_dist) * 100 # [%]
        return out_penalty + space_penalty
    
    def ShippingLaneInit(self, init_pts):
        self.lane_poly = Line.lane
        self.init_pts = init_pts
        pts = shapely.points(init_pts[:, 1], init_pts[:, 0])

        self.inside0 = shapely.covers(self.lane_poly, pts)
        inside = shapely.covers(Line.lane, pts)
        print("inside count:", inside.sum(), "/", len(inside), "\n")

    def ShippingLane(self, pts_vh):
        """
        船は基本的に航路帯を航行する。
        初期経路で航路帯にあったものは、航路帯から外れると大きな罰を与える
        """
        p = 2; eps=self.ps.B
        w = 1e2; w_out = 1e4
        pts = shapely.points(pts_vh[:, 1], pts_vh[:, 0])
        inside = shapely.contains(self.lane_poly, pts)

        base = self.inside0
        in_mask = base & inside

        pen = 0.0
        if np.any(in_mask):
            dvh = pts_vh[in_mask] - self.init_pts[in_mask]
            d = np.linalg.norm(dvh, axis=1)
            pen += w * float(np.sum((d / eps) ** p))

        out_mask = base & (~inside)
        if np.any(out_mask):
            dvh = pts_vh[out_mask] - self.init_pts[out_mask]
            d = np.linalg.norm(dvh, axis=1)
            pen += w_out * float(np.sum((d / eps) ** p))

        return pen
    
    def SD_penalty(self, pt, psi):
        SD = self.SD
        lines = self.lines
        
        speed = cal_speed(self, pt, lines[-1].end_pt)
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

def sigmoid(x, a, b, c):
    return a / (b + np.exp(c * x))

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

def cal_psi(parent_pt, current_pt, child_pt):
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
        cross = float(cal_cross(v1, v2))
        theta = float(np.arctan2(cross, dot))  # CCW:+, CW:-

    psi_in = np.pi/2 - np.arctan2(v1[1], v1[0])  # 0=North, CW:+
    psi = psi_in - 0.5 * theta
    psi = (psi + np.pi) % (2.0 * np.pi) - np.pi
    return psi

def cal_speed(self, pt, base_pt):
    SD = self.SD
    distance = np.linalg.norm(pt - base_pt)
    speed = SD.b_ave * distance ** SD.a_ave + SD.b_SD * distance ** SD.a_SD
    if self.ps.MAX_SPEED_KTS < speed:
        return self.ps.MAX_SPEED_KTS
    if self.ps.MIN_SPEED_KTS > speed:
        return self.ps.MIN_SPEED_KTS
    return speed

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
        self.CMAES()
        self.result()

    def prepare(self):
        self.read_csv()
        self.prepare_for_SD()
        self.prepare_for_init_route()
        self.prepare_for_CMAES()

    def result(self):
        self.print_result(self.best_dict)
        self.show_best_result(self.best_dict)

    def prepare_for_SD(self):
        SD = ShipDomain_proposal()
        SD.initial_setting(self.SD_setup_csv, sigmoid)

        SD.a_ave = self.df_debug["a_ave"].values[0]
        SD.b_ave = self.df_debug["b_ave"].values[0]
        SD.a_SD = self.df_debug["a_SD"].values[0]
        SD.b_SD = self.df_debug["b_SD"].values[0]

        self.SD = SD
        print("\nShip Domain set up complete\n")

    def prepare_for_init_route(self):
        self.init_fig()
        self.draw_basemap()
        self.make_init_line()
        self.make_captain_line()
        self.make_init_route()

    def prepare_for_CMAES(self):
        # cost
        cal = CostCalculator()
        cal.SD = self.SD
        cal.lines = self.lines
        cal.ps= ps
        self.cal = cal

        print("initial point calculated\n")
        self.initial_D = self.cal_sigma_for_ddCMA(points=self.target_list,
                                                  last_point=self.init_list[-1]
                                                  )
        self.initial_vec = self.target_list.ravel()  # <<< important: flatten for CMA-ES
        self.N = len(self.initial_vec)
        print(
            f"この最適化問題の次元Nは {self.N} です\n"
            "### INITIAL CHECKPOINTS AND sigma0 SETUP COMPLETED ###\n"
            "### MOVED TO THE OPTIMIZATION PROCESS ###\n"
        )

    def CMAES(self):
        # compute auto scaling coefficients from initial solution
        self.compute_cost_weights(self.target_list)

        # --- CMA-ES expects 1D mean (N,) and sigma0 of same length ---
        ddcma = DdCma(xmean0=self.initial_vec, sigma0=self.initial_D, seed=self.ps.seed)
        checker = Checker(ddcma)
        logger = Logger(ddcma, prefix=f"{self.SAVE_DIR}/log")

        NEVAL_STANDARD = ddcma.lam * 5000
        print("Start with first population size:", ddcma.lam)
        print("Dimension:", ddcma.N)
        print(f"NEVAL_STANDARD: {NEVAL_STANDARD}")
        print("Path optimization start\n")

        total_neval = 0
        best_dict: dict[int, dict] = {}
        time_start = time.time()
        cur_seed = int(self.ps.seed)

        for restart in range(self.ps.restarts):
            is_satisfied = False
            best_dict[restart] = {
                "best_cost_so_far": float("inf"),
                "best_mean_sofar": None,
                "calculation_time": None,
                "cp_list": None,
                "mp_list": None,
                "psi_list_at_cp": None,
                "psi_list_at_mp": None,
            }

            t0 = time.time()

            # --- Progress bar: show only Restart, %, eval/s, 試行数/Max, best_sofar ---
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
                eval_per_s = f"{rate:.1f}" if rate is not None else "–"
                trials_str = f"{ddcma.neval}/{NEVAL_STANDARD}"
                pbar.set_postfix_str(
                    f"eval/s={eval_per_s}  trials={trials_str}"
                )

            while not is_satisfied:
                # ddcma.onestep(func=self.path_evaluate, check=self.enforce_max_turn_angle)
                ddcma.onestep(func=self.path_evaluate, check=None)

                best_cost = float(np.min(ddcma.arf))
                best_mean = ddcma.arx[int(ddcma.idx[0])].copy()

                if best_cost < best_dict[restart]["best_cost_so_far"]:
                    best_dict[restart]["best_cost_so_far"] = best_cost
                    best_dict[restart]["best_mean_sofar"] = best_mean

                is_satisfied, condition = checker()

                # Update progress by increase in evaluation count
                if ddcma.neval > last_neval:
                    pbar.update(ddcma.neval - last_neval)
                    last_neval = ddcma.neval
                    _refresh_postfix()

                if ddcma.t % 10 == 0:
                    pbar.write(
                        f"neval:{ddcma.neval :<6}  "
                        f"cost:{best_cost:<10.9g}  "
                        f"best:{best_dict[restart]['best_cost_so_far']:<10.9g}"
                    )
                    logger()

            # final bar state
            _refresh_postfix()
            pbar.close()

            logger(condition)
            elapsed = time.time() - t0
            best_dict[restart]["calculation_time"] = elapsed
            print(f"Terminated with condition: {condition}")
            print(f"Restart {restart} time: {elapsed:.2f} s")
            
            # show result
            self.show_CMA_path(best_dict[restart]["best_mean_sofar"], restart)
            self.CMA_result()
            best_dict[restart]["cp_list"] = self.path
            best_dict[restart]["mp_list"] = self.MP_list
            best_dict[restart]["psi_list_at_cp"] = self.CP_psi_list
            best_dict[restart]["psi_list_at_mp"] = self.MP_psi_list

            total_neval += ddcma.neval
            print(f"total number of evaluate function calls: {total_neval}\n")

            if total_neval < NEVAL_STANDARD:
                popsize = ddcma.lam if not self.ps.increase_popsize_on_restart else ddcma.lam * 2
                cur_seed *= 2
                # restart from the SAME initial mean as legacy code
                ddcma = DdCma(xmean0=self.initial_vec, sigma0=self.initial_D, lam=popsize, seed=cur_seed)
                checker = Checker(ddcma)
                logger.setcma(ddcma)
                print(f"Restart with popsize: {ddcma.lam}")
            else:
                print("Path optimization completed")
                break

        self.cma_caltime = time.time() - time_start
        print(
            f"Path optimization completed in {self.cma_caltime:.2f} s.\n\n"
            f"best_cost_so_far の値とその値を記録した平均の遷移:\n{'='*50}\n"
        )

        self.logger = logger
        self.best_dict = best_dict

    def read_csv(self):
        self.port = dictionary()[self.ps.port_number]
        SAVE_DIR = f"{DIR}/../../outputs/{dirname}/{self.port["name"]}"
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(f"{SAVE_DIR}/卒論", exist_ok=True)
        print(f"\ntarget : {self.port["name"]}")

        # Map
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

        # CMA-ES
        ## Ship Domain
        ### setup
        SD_setup_csv = "outputs/303/mirror5/fitting_parameter.csv"

        ### something important
        debug_csv = "raw_datas/tmp/GuidelineFit_debug.csv"
        df_debug = pd.read_csv(debug_csv)

        # save
        self.SAVE_DIR = SAVE_DIR
        self.port_csv = port_csv
        self.df_map = df_map
        self.df_lane = df_lane
        self.df_buoy = df_buoy
        self.df_cap = df_cap
        self.SD_setup_csv = SD_setup_csv
        self.df_debug = df_debug

        Line.hor_range = self.port["hor_range"]
        Line.ver_range = self.port["ver_range"]

        print("\nall csv read")

    def init_fig(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

    def draw_basemap(self):
        port = self.port
        fig, ax = self.fig, self.ax
        legends = []
        h_list = []

        # map
        map_X, map_Y = self.df_map["x [m]"].values, self.df_map["y [m]"].values
        ax.fill_betweenx(map_X, map_Y, facecolor="gray", alpha=0.3, zorder=0)
        ax.plot(map_Y, map_X, color="k", linestyle="--", lw=0.5, alpha=0.8, zorder=0)

        # shipping lane
        for pid, g in self.df_lane.groupby("polygon_id", sort=True):
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
        
        # buoy
        ax.scatter(self.df_buoy["x [m]"].values, self.df_buoy["y [m]"].values,
                    color='orange', s=20, zorder=4)
        legend_buoy = plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor="orange", markersize=2, label="Buoy")
        legends.append(legend_buoy)

        # compas
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

        ax.set_xlim(port["hor_range"])
        ax.set_ylim(port["ver_range"])
        ax.set_aspect("equal")
        ax.grid()
        ax.set_xlabel(r"$Y\,\rm{[m]}$")
        ax.set_ylabel(r"$X\,\rm{[m]}$")
        h = ax.legend(handles=legends)
        h_list.append(h)
        fig.savefig(os.path.join(self.SAVE_DIR, "base_map.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "base_map.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        print("base map saved\n")

        for h in h_list:
            h.remove()
        self.legends = legends

    def make_init_line(self):
        port = self.port
        ax = self.ax
        lines = []
        h_list = []

        # for crossing algorithm
        coords_map = self.df_map[["y [m]", "x [m]"]].to_numpy(dtype=float)
        poly_map = make_valid(Polygon(coords_map))
        Line.map_poly = poly_map
        Line.map_poly_prep = prep(poly_map)

        # from berth point
        theta = 0
        margin = 2*self.ps.B
        if port["psi_end"] == 0:
            if port["side"] == "starboard":
                margin = -self.ps.B
            if port["style"] == "head in":
                theta = 180
            theta = np.deg2rad(theta)
        else:
            theta = np.deg2rad(port["psi_end"])
        L_berth = Line(fixed_pt=np.array((0.0, margin)), theta=theta)
        L_berth.swap()
        lines.append(L_berth)
        self.lines = lines
        self.turn_end = np.array((0.0, margin))

        ## save fig
        self.add_text(ax, h_list, p_te=True)
        self.show_init_lines(ax, "init lines step.1")

        # from shipping lane
        lane_polys = []
        dist_both_ship = 0.5 * self.ps.L + self.ps.B
        for pid, g in self.df_lane.groupby("polygon_id", sort=True):
            lane_pts = g[['y [m]', 'x [m]']].to_numpy()

            mid_1 = (lane_pts[0] + lane_pts[1]) / 2
            mid_2 = (lane_pts[2] + lane_pts[3]) / 2
            theta = cal_angle(mid_1, mid_2)

            mid_1 = mid_1 + np.array([-dist_both_ship * np.sin(theta), dist_both_ship * np.cos(theta)])
            L_lane = Line(fixed_pt=np.array((mid_1)), theta=theta)
            L_lane.extent_fixed_pt()
            lines.append(L_lane)

            # for crossing algorithm
            poly_lane = Polygon(lane_pts[:, [1, 0]])
            lane_polys.append(poly_lane)
            print(f"shipping lane {pid} complete")
        Line.lane = shapely.union_all(lane_polys)
        lines[:] = lines[1:] + lines[:1]

        ## save fig
        self.show_init_lines(ax, "init lines step.2")

        # from start point
        L_start = Line(fixed_pt=port["start"], theta=np.deg2rad(port["psi_start"]))
        lines.insert(0, L_start)
        self.approach_start = port["start"]

        ## save fig
        self.add_text(ax, h_list, p_as=True)
        self.show_init_lines(ax, "init lines step.3")
        print("\ninit lines saved\n")

        for h in h_list:
            h.remove()


    def make_captain_line(self):
        lines = self.lines

        # set parent
        for i in range(len(lines) - 2):
            lines[i+1].set_parent(lines[i])
        self.show_captain_line("captain's line prepare")

        base_idx = len(lines) - 1
        while True:
            # check nearlest intersection
            L_berth = lines[base_idx]; idx = None
            shortest = np.linalg.norm(L_berth.fixed_pt - L_berth.end_pt)

            if base_idx == 1:
                if cross_judge(lines[0], L_berth):
                    intersect_pt = cal_intersect_pt(lines[0], L_berth)
                    length = np.linalg.norm(intersect_pt - L_berth.end_pt)
                    if length < shortest:
                        shortest = length; idx = 0
            else:
                for i in range(1, base_idx):
                    if cross_judge(lines[i], L_berth):
                        intersect_pt = cal_intersect_pt(lines[i], L_berth)
                        length = np.linalg.norm(intersect_pt - L_berth.end_pt)
                        if length < shortest:
                            shortest = length; idx = i      

            if idx != None:
                L_berth.set_parent(lines[idx])
                break
            else:
                mid = (L_berth.fixed_pt + L_berth.end_pt) / 2

                def to_xy(p_yx):
                    return (float(p_yx[1]), float(p_yx[0]))
                
                if len(lines) == 2:
                    ln = lines[0]
                    fixed_pt = ln.fixed_pt; end_pt = ln.end_pt
                    pts = np.linspace(end_pt, fixed_pt, 99)
                    idx = 0
                    while idx < 99 and Line.map_poly_prep.intersects(LineString([to_xy(pts[idx]), to_xy(mid)])):
                        idx += 1
                    idx_through = idx
                    while idx < 99 and (not Line.map_poly_prep.intersects(LineString([to_xy(pts[idx]), to_xy(mid)]))):
                        idx += 1
                    idx_hit = idx
                    idx_hit = np.clip(idx_hit, 0, len(pts)-1)
                
                else:
                    for ln in lines[-2:0:-1]:
                        fixed_pt = ln.fixed_pt; end_pt = ln.end_pt
                        pts = np.linspace(end_pt, fixed_pt, 99)
                        idx = 0
                        while idx < 99 and Line.map_poly_prep.intersects(LineString([to_xy(pts[idx]), to_xy(mid)])):
                            idx += 1
                        idx_through = idx
                        while idx < 99 and (not Line.map_poly_prep.intersects(LineString([to_xy(pts[idx]), to_xy(mid)]))):
                            idx += 1
                        idx_hit = idx

                        if idx_hit != 99:
                            break

                ln_mid = (pts[idx_through] + pts[idx_hit]) / 2
                theta = cal_angle(mid, ln_mid)
                L_append = Line(fixed_pt=mid, theta=theta)
                L_append.swap()
                L_berth.set_parent(L_append)
                lines.insert(base_idx, L_append)
            
        self.show_captain_line("captain's line before", last=True)

    def make_init_route(self):
        """
        航路をapproachとbarthingに分割
        berthingをstraightとturnに分割
        各航路ですること
            approach  : 大雑把でいいのでBezier
            straight : 着桟の前は直進したい.一旦 3L [m]としている.
            turn     : 着桟姿勢に応じて回頭,変針
        """
        port = self.port
        pts = self.captain_pts
        margin = 2 * self.ps.L

        def cal_total_len(pts):
            seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            total_len = s_cum[-1]
            return seg_len, s_cum, total_len

        def interp(u, pts):
            """
            u : 開始地点からの距離[m]
            """
            i = np.searchsorted(s_cum, u, side="right") - 1
            i = int(np.clip(i, 0, len(seg_len) - 1))
            a = (u - s_cum[i]) / seg_len[i]
            return pts[i] + (pts[i + 1] - pts[i]) * a
        
        # separate phase
        seg_len, s_cum, total_len = cal_total_len(pts)
        u_split = np.clip(total_len - 500, 0.0, total_len)
        i = np.searchsorted(s_cum, u_split, side="right") - 1
        i = int(np.clip(i, 0, len(seg_len) - 1))
        turn_start_pt = port["turn_start"]
        nearest_ln = self.get_nearest_line(turn_start_pt)
        theta = nearest_ln.theta # [rad]
        berth_start_pt = turn_start_pt + margin * np.array([np.cos(theta - np.pi), np.sin(theta - np.pi)])
        pts_for_bezier = np.vstack([pts[1:i+1], berth_start_pt])

        bezier_pts, _ = Bezier.bezier(pts_for_bezier, 100)
        straight_pts = np.vstack([berth_start_pt, turn_start_pt])
        curve_pts = np.vstack([turn_start_pt, pts[-1]])

        # approach, straight
        ## besier pts spread by speed
        ap_and_st_pts = np.vstack([pts[0], bezier_pts, turn_start_pt])
        seg_len, s_cum, total_len = cal_total_len(ap_and_st_pts)
        approach_pts = [pts[0].copy()]
        u = 0.0
        while u < total_len:
            curent_pt = interp(u, ap_and_st_pts)
            speed_kt = cal_speed(self, curent_pt, pts[-1])
            speed = speed_kt *1852 / 60 # [m/min]
            u = min(u + speed, total_len)
            approach_pts.append(interp(u, ap_and_st_pts) if u < total_len else ap_and_st_pts[-1].copy())
        
        ## psi
        psi_list = np.array([
            cal_angle(approach_pts[i], approach_pts[i+1]) % (2*np.pi)
            for i in range(len(approach_pts) - 1)
        ])
        approach_pts.pop()

        # turn
        seg_len, s_cum, total_len = cal_total_len(curve_pts)
        turn_pts = [curve_pts[0].copy()]
        u = 0.0
        while u < total_len:
            curent_pt = interp(u, curve_pts)
            speed_kt = cal_speed(self, curent_pt, pts[-1])
            speed = speed_kt *1852 / 60 # [m/min]
            u = min(u + speed, total_len)
            turn_pts.append(interp(u, curve_pts) if u < total_len else curve_pts[-1].copy())

        ## psi
        psi_list = np.append(psi_list, psi_list[-1])
        n = len(turn_pts)
        if port["side"] == "port":
            beta = (2 * np.pi - psi_list[-1]) / (n-1)
        elif port["side"] == "starboard":
            beta = -(psi_list[-1] - 0)  / (n-1)

        for i in range(n-1):
            psi = psi_list[-1] + beta
            psi_list = np.append(psi_list, psi)

        # init list
        init_path = np.vstack([approach_pts, turn_pts])
        init_list = np.hstack([init_path, psi_list.reshape(-1, 1)])

        self.init_list = init_list
        self.approach_list = init_list[:(len(init_list) - n)]
        self.turn_list = init_list[(len(init_list) - n):]
        self.target_list = self.turn_list[1:-1]

        self.berth_start = berth_start_pt
        self.turn_start = turn_start_pt

        print(init_list)
        self.show_init_route()

    def get_nearest_line(self, pt):
        """
        pt に最も近い線分を返す
        if 一番近いのが fixed_pt
        elif 一番近いのが end_pt
        else 点と直線の距離
        """
        lines = self.lines
        pt = np.asarray(pt, dtype=float)
        longest = 1e5

        for ln in lines:
            fixed_pt = np.asarray(ln.fixed_pt, dtype=float); end_pt = np.asarray(ln.end_pt, dtype=float)

            v_fe = end_pt - fixed_pt; v_ef = fixed_pt - end_pt
            v_fp = pt - fixed_pt    ; v_ep = pt - end_pt
            if np.dot(v_fp, v_fe) < 0:
                dist = np.linalg.norm(v_fp)
            elif np.dot(v_ep, v_ef) < 0:
                dist = np.linalg.norm(v_ep)
            else:
                FH_norm = np.dot(v_fp, v_fe)/np.linalg.norm(v_fe) # 垂線の足
                neighbor_point = fixed_pt + (v_fe)/np.linalg.norm(v_fe)*FH_norm
                dist = np.linalg.norm(pt - neighbor_point)
            
            if longest > dist:
                longest = dist
                nearest_line = ln

        return nearest_line
    
    # CMA-ES

    def cal_sigma_for_ddCMA(
        self,
        points: np.ndarray,
        last_point: Tuple[float, float],
        *,
        min_sigma: float = 5.0,
        min_sigma_psi: float = 5.0,
        scale: float = 0.5,
        scale_psi: float = 0.5,
    ) -> np.ndarray:
        ver = points[:, 0]
        hor = points[:, 1]
        ver_diffs = np.abs(np.diff(ver, append=last_point[0])) * scale
        hor_diffs = np.abs(np.diff(hor, append=last_point[1])) * scale
        ver_diffs = np.maximum(ver_diffs, min_sigma)
        hor_diffs = np.maximum(hor_diffs, min_sigma)

        psi = points[:, 2]
        dpsi = np.diff(psi, append=last_point[2])
        dpsi = dpsi % (2 * np.pi)
        psi_diffs = np.abs(dpsi) * scale_psi
        psi_diffs = np.maximum(psi_diffs, min_sigma_psi)

        return np.column_stack((ver_diffs, hor_diffs, psi_diffs)).ravel()
    
    def compute_cost_weights(self, pts):
        """
        評価関数を計算するときに、ある1つの要素のみが影響しすぎないようにするための関数
        ex) L:100, SD:1000000 とかだとSDだけでほとんど決まってしまう
        """
        cal = self.cal

        start = self.turn_list[0]
        end = self.turn_list[-1]
        poly = np.vstack([start, pts, end])
        pts = poly[:, :2]; head = poly[:, 2]

        # ship domain
        SD_cost = 0.0
        for j in range(len(poly)):
            SD_cost += cal.SD_penalty(pts[j], head[j])
        
        # psi smooth
        angle_cost = 0.0
        for j in range(len(head)-1):
            angle_cost += (head[j+1] - head[j]) ** 2 # 1次差分

        # (x, y) smooth
        xy_cost = 0.0
        for j in range(1, len(pts)-1):
            # xy_cost += np.linalg.norm(pts[j+1] - pts[j]) ** 2 # 1次差分
            xy_cost += np.linalg.norm(pts[j-1] - 2*pts[j] + pts[j+1]) ** 2 # 2次差分

        # xy/cost がでかいから合わせる
        self.sd_coeff = xy_cost / SD_cost
        self.angle_coeff = xy_cost / angle_cost
        self.xy_coeff = 0.1
        

    def path_evaluate(self, X):
        batched = True
        arr = np.asarray(X, float)
        if arr.ndim == 1:
            arr = arr[None, :]
            batched = False

        cal = self.cal
        start = self.turn_list[0]
        end = self.turn_list[-1]

        # weight
        w_SD = self.ps.SD
        w_angle = self.ps.angle
        w_xy = self.ps.xy

        costs = np.zeros(arr.shape[0], dtype=float)
        for i in range(arr.shape[0]):
            arr_i = arr[i].reshape(-1, 3)
            poly = np.vstack([start, arr_i, end])
            pts = poly[:, :2]; head = poly[:, 2]

            # ship domain
            SD_cost = 0.0
            for j in range(len(poly)):
                SD_cost += cal.SD_penalty(pts[j], head[j])
            
            # psi smooth
            angle_cost = 0.0
            for j in range(len(head)-1):
                angle_cost += (head[j+1] - head[j]) ** 2 # 1次差分

            # (x, y) smooth
            xy_cost = 0.0
            for j in range(1, len(pts)-1):
                # xy_cost += np.linalg.norm(pts[j+1] - pts[j]) ** 2 # 1次差分
                xy_cost += np.linalg.norm(pts[j-1] - 2*pts[j] + pts[j+1]) ** 2 # 2次差分

            total = (w_SD * (self.sd_coeff * SD_cost)
                        + w_angle * (self.angle_coeff * angle_cost) 
                        + w_xy * (self.xy_coeff * xy_cost))
            costs[i] = total

        return float(costs[0]) if not batched else costs

    
    def enforce_max_turn_angle(self, X: np.ndarray) -> np.ndarray:

        def selective_laplacian_smoothing(poly, max_deg=60.0, n_iter=10, alpha=0.7):
            poly = np.asarray(poly, float).copy()
            for _ in range(n_iter):
                v1 = poly[1:-1] - poly[:-2]
                v2 = poly[2:]   - poly[1:-1]
                a1 = np.arctan2(v1[:,1], v1[:,0])
                a2 = np.arctan2(v2[:,1], v2[:,0])
                deg  = (a2 - a1 + np.pi) % (2*np.pi) - np.pi
                ang = np.degrees(np.abs(deg))

                bad = np.where(ang > max_deg)[0] + 1  # 折れ点の index（1..M-2）
                for i in bad:
                    mid = (poly[i-1] + poly[i+1]) / 2
                    poly[i] = (1 - alpha) * poly[i] + alpha * mid
            return poly
        
        arr = np.asarray(X, float)
        if arr.ndim == 1:
            arr = arr[None, :]

        out = arr.copy()
        start = self.init_pts[0]; end = self.init_pts[-1]

        for k in range(out.shape[0]):
            pts = out[k].reshape(-1, 2)
            poly = np.vstack([start, pts, end])

            poly2 = selective_laplacian_smoothing(
                poly,
                max_deg=self.ps.MAX_ANGLE_DEG,
                n_iter=10,
                alpha=0.7
            )

            pts[:] = poly2[1:-1]
            out[k] = pts.reshape(-1)

        return out if X.ndim == 2 else out[0]
     
    # show map
    def show_init_lines(self, ax, name):
        lines = self.lines

        handles = []
        for ln in lines:
            pts = np.vstack([ln.fixed_pt, ln.end_pt])
            h, = ax.plot(pts[:, 1], pts[:, 0], color="red", linestyle='-')
            handles.append(h)
        plt.savefig(os.path.join(self.SAVE_DIR, f"{name}.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", f"{name}.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        
        for h in handles:
            h.remove()

    def show_captain_line(self, name, last=False):
        ax = self.ax
        lines = self.lines
        h_list = []

        ln = lines[-1]
        pts_list = [np.asarray(ln.end_pt), np.asarray(ln.fixed_pt)]
        while ln.parent is not None:
            ln = ln.parent
            pts_list.append(np.asarray(ln.fixed_pt))
        
        self.add_text(ax, h_list, p_as=True, p_te=True)
        self.show_init_lines(ax, name)

        if last:
            pts = np.vstack(pts_list[::-1])
            self.way_pts = pts[1:-1]
            legend_WP = plt.Line2D([0], [0], marker="o", color="w",
                                        markerfacecolor="green", markersize=2, label="Way Point")
            self.legends.append(legend_WP)

            h, =ax.plot(pts[:, 1], pts[:, 0], color="blue", linestyle='-')
            self.add_text(ax, h_list, p_as=True, p_te=True, wp=True)
            plt.savefig(os.path.join(self.SAVE_DIR, "captain's line.png"),
                        dpi=400, bbox_inches="tight", pad_inches=0.05)
            plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "captain's line.pdf"),
                        bbox_inches="tight", pad_inches=0.05)
            h_list.append(h)
            print("captain's line saved\n")
            self.captain_pts = pts

        for h in h_list:
            h.remove()


    def show_init_route(self):
        ax = self.ax
        list = self.init_list
        legends = self.legends
        port = self.port
        h_list = []

        # captain's route
        for df in self.df_cap:
            traj = RealTraj()
            traj.input_csv(df, self.port_csv)
            h, = ax.plot(traj.Y, traj.X, 
                        color = 'gray', ls = '-', marker = 'D',
                        markersize = 2, alpha = 0.8, lw = 1.0, zorder = 3)
            h_list.append(h)
        legend_captain = plt.Line2D([0], [0],
                                    color = 'gray', ls = '-', marker = 'D',
                                    markersize = 2, alpha = 0.8, lw = 1.0, label=f"Berthing Path\n({port["legend"]}-port)")
        legends.append(legend_captain)

        # text
        self.add_text(ax, h_list, p_as=True, p_bs=True, p_te=True, p_ts=True)

        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "phase change.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        
        self.add_text(ax, h_list, wp=True)

        # ship shape
        for pose in list:
            shipshape = MplPolygon(
                ship_shape_poly(
                pose=pose,
                L=self.ps.L, B=self.ps.B,
                ),
                facecolor='blue',
                alpha=0.7, zorder=6
            )
            ax.add_patch(shipshape)

        # setting
        legend_init_path = plt.Line2D([0], [0],
                                    color = 'blue', ls = '-', marker = 'D',
                                    markersize = 2, lw = 1.0, label="Initial Path")
        legends.insert(-1, legend_init_path)
        h = ax.legend(handles=legends)
        h_list.append(h)

        plt.savefig(os.path.join(self.SAVE_DIR, "init path.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "init path.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        # zoom
        ax.set_xlim([-750, 750])
        ax.set_ylim([-750, 750])
        plt.savefig(os.path.join(self.SAVE_DIR, "init path zoom.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "init path zoom.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        print("init path saved\n")

        for h in h_list:
            h.remove()


    def add_text(self, ax, h_list,
                 p_as=False, p_bs=False, p_ts=False, p_te=False, wp=False, all=False):
        port = self.port

        if all:
            p_as=True; p_bs=True; p_ts=True; p_te=True; wp=True

        if wp:
            pts = self.way_pts
            p = ax.scatter(pts[:, 1], pts[:, 0], c="green", s=20, zorder=10)
            h_list.append(p)

        port_ann = port["annotations"]
        items = [
            ("approach_start", p_as),
            ("berth_start",    p_bs),
            ("turn_start",     p_ts),
            ("turn_end",       p_te),
        ]

        for key, flag in items:
            if not flag:
                continue
            
            x, y = getattr(self, key)
            p = ax.scatter(y, x, c="black", s=20, zorder=10)
            ann = port_ann[key]
            t = ax.annotate(
                ann["text"],
                xy=(y, x),
                xytext=ann["xytext"],
                textcoords=ann.get("textcoords", "offset points"),
                ha=ann["ha"],
                va=ann["va"],
                fontsize=20,
                zorder=10,
            )

            h_list.extend([t, p])

    def show_CMA_path(self, best_mean, restart):
        ax = self.ax
        SD = self.SD
        h_list = []

        start = self.turn_list[0]; end = self.turn_list[-1]

        list = np.asarray(best_mean, float).reshape(-1, 3)
        poly = np.vstack([start, list, end])
        path = poly[:, :2]; head = poly[:, 2]

        # path, point
        h1, = ax.plot(path[:, 1], path[:, 0], color="red", linestyle='-')
        h2 = ax.scatter(path[:, 1], path[:, 0], c="r",s=12, marker="o")
        h_list.append(h1); h_list.append(h2)

        # ship shape
        for pose in poly:
            shipshape = MplPolygon(
                ship_shape_poly(
                pose=pose,
                L=self.ps.L, B=self.ps.B,
                ),
                facecolor='red',
                alpha=0.7, zorder=7
            )
            h = ax.add_patch(shipshape)
            h_list.append(h)

        # ship domain
        for pts, psi in zip(path, head):
            speed = cal_speed(self, pts, path[-1])
            r_list = []
            for theta_i in theta_list:
                r_list.append(SD.distance(speed, theta_i))

            r = np.asarray(r_list, dtype=float)
            domain_xy = np.column_stack([
                pts[0] + r * np.cos(theta_list + psi),
                pts[1] + r * np.sin(theta_list + psi),
            ])
            h3, = ax.plot(domain_xy[:, 1], domain_xy[:, 0], color="red", linestyle='--')
            h_list.append(h3)

        if restart == 0:
            legend_path = plt.Line2D([0], [0],
                                        color = 'red', ls = '-', marker = 'D',
                                        markersize = 2, lw = 1.0, label="CMA result")
            self.legends.insert(-1, legend_path)

        h3 = ax.legend(handles=self.legends)
        ax.set_xlim([-750, 750]); ax.set_ylim([-750, 750])
        plt.savefig(os.path.join(self.SAVE_DIR, f"CMA route ver {restart}.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        ax.set_xlim([-300, 300]); ax.set_ylim([-400, 400])
        h3.remove()
        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", f"CMA route ver {restart}.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        print(f"CMA route ver {restart} saved")

        for h in h_list:
            h.remove()
        self.path = path

    def CMA_result(self):
        path = self.path
        # check point
        CP_psi_list = []
        for i in range(1, len(path) - 1):
            psi_at_cp = cal_psi(path[i-1], path[i], path[i+1])
            CP_psi_list.append(psi_at_cp)

        # mid point
        MP_list = []
        MP_psi_list = []
        for i in range(0, len(path) - 1):
            mid = (path[i] + path[i + 1]) / 2
            parent = path[i - 1] if i - 1 >= 0 else path[i]
            child  = path[i + 2] if i + 2 < len(path) else path[i+1]
            psi_at_mp = cal_psi(parent, mid, child)
            MP_list.append(mid); MP_psi_list.append(psi_at_mp)
        
        self.CP_psi_list = CP_psi_list
        self.MP_list = MP_list
        self.MP_psi_list = MP_psi_list

    def print_result(self, best_dict):
        for restart, values in best_dict.items():
            best_cost_so_far = values["best_cost_so_far"]
            best_mean_sofar = values["best_mean_sofar"]
            calculation_time = values["calculation_time"]
            pairs = "\n".join(
                f"  ({best_mean_sofar[i]:.6f}, {best_mean_sofar[i+1]:.6f}, {best_mean_sofar[i+2]:.6f})"
                for i in range(0, len(best_mean_sofar), 3)
            )
            print(
                f"\n[Restart {restart}]\n"
                f"  best_cost_so_far: {best_cost_so_far:.6f}    計算時間: {calculation_time:.2f} s\n"
                f"  best_mean_sofar:\n{pairs}"
            )
        print("\n" + "=" * 50 + "\n")
        smallest_evaluation_key = min(best_dict, key=lambda k: best_dict[k]["best_cost_so_far"])
        print(f"最も評価値が小さかった試行は {smallest_evaluation_key} 番目\n")

        self.smallest_evaluation_key = smallest_evaluation_key

    def show_best_result(self, best_dict):
        ax = self.ax
        port = self.port
        legends = self.legends
        h_list = []

        # captain's route
        for df in self.df_cap:
            traj = RealTraj()
            traj.input_csv(df, self.port_csv)
            ax.plot(traj.Y, traj.X, 
                        color = 'gray', ls = '-', marker = 'D',
                        markersize = 2, alpha = 0.8, lw = 1.0, zorder = 3)
        # legend_captain = plt.Line2D([0], [0],
        #                             color = 'gray', ls = '-', marker = 'D',
        #                             markersize = 2, alpha = 0.8, lw = 1.0, label=f"Berthing Path\n({port["legend"]}-port)")
        # legends.append(legend_captain)

        # best path
        opt_list = best_dict[self.smallest_evaluation_key]["best_mean_sofar"].reshape(-1, 3)
        n = len(opt_list)
        best_list = self.init_list.copy()
        best_list[-(n+1):-1] = opt_list
        best_pts = best_list[:, :2]; best_psi = best_list[:, 2]

        # ship shape
        for pose in best_list:
            shipshape = MplPolygon(
                ship_shape_poly(
                pose=pose,
                L=self.ps.L, B=self.ps.B,
                ),
                facecolor='red',
                alpha=0.7, zorder=7,
            )
            ax.add_patch(shipshape)

        # text
        self.add_text(ax, h_list, all=True)
        
        ## save
        ax.set_xlim(port["hor_range"])
        ax.set_ylim(port["ver_range"])
        ax.legend(handles=legends)
        plt.savefig(os.path.join(self.SAVE_DIR, "CMA Result.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "CMA Result.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        
        # init path
        for pose in self.init_list:
            shipshape = MplPolygon(
                ship_shape_poly(
                pose=pose,
                L=self.ps.L, B=self.ps.B,
                ),
                facecolor='blue',
                alpha=0.7, zorder=6,
            )
            ax.add_patch(shipshape)

        plt.savefig(os.path.join(self.SAVE_DIR, "CMA Result with init.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "CMA Result with init.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        
        # zoom
        for h in h_list:
            h.remove()
        ax.set_xlim([-750, 750])
        ax.set_ylim([-750, 750])
        plt.savefig(os.path.join(self.SAVE_DIR, "CMA Result zoom.png"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(os.path.join(f"{self.SAVE_DIR}/卒論", "CMA Result zoom.pdf"),
                    bbox_inches="tight", pad_inches=0.05)
        print(f"Result fig saved\n")
        print(f"{port["name"]} end\n")

if __name__ == "__main__":
    ps = Setting()
    makeroute = MakeLine(ps)
    makeroute.main()
