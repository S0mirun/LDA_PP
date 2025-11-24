"""
CMA-ES path optimization (A* init → element-based turning points → CMA-ES)
- Max speed: 9.5 knots
- Ship Domain at segment midpoint + checkpoint
- Angle convention: vertical (X) = 0 deg, clockwise positive
- Coordinate note: (ver, hor) = (Y, X)
"""

from __future__ import annotations
from enum import StrEnum
import glob
import os
import sys
import time
from typing import Tuple

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
from scipy import ndimage
from tqdm.auto import tqdm

# --- external project modules ---
import utils.PP.Astar_for_CMAES as Astar
import utils.PP.Bezier_curve as Bezier
import utils.PP.graph_by_taneichi as Glaph
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.Filtered_Dict import new_filtered_dict
from utils.PP.graph_by_taneichi import ShipDomain_proposal
from utils.PP.MultiPlot import RealTraj, Buoy

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
PYSIM_DIR = os.path.join(PROGRAM_DIR, "py-ship-simulator-main/py-ship-simulator-main")
if PYSIM_DIR not in sys.path:
    sys.path.append(PYSIM_DIR)
import pyshipsim


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
RAW_DATAS = f"{DIR}/../../raw_datas"
DATA = f"{DIR}/../../outputs/data"


class ParamMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class InitPathAlgo(StrEnum):
    ASTAR = "astar"
    BEZIER = "bezier"


class Settings:
    def __init__(self):
        # port
        self.port_number: int = 3
         # 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Else_1, 4: Osaka_1B
        # ship
        self.L = 100

        # setup / initial path
        self.start_end_mode: ParamMode = ParamMode.AUTO
        self.psi_mode: ParamMode = ParamMode.AUTO
        self.steady_course_coeff_mode: ParamMode = ParamMode.AUTO
        self.init_path_algo: InitPathAlgo = InitPathAlgo.BEZIER
        self.enable_pre_berthing_straight_segment: bool = True

        self.save_init_path: bool = True
        self.show_SD_on_init_path: bool = False
        self.gridpitch: float = 5.0  # [m]
        self.gridpitch_for_Astar: float = 5.0  # [m]
        self.range_type: float = 1

        # CMA-ES
        self.seed: int = 42
        self.MAX_SPEED_KTS: float = 9.5  # [knots]
        self.MIN_SPEED_KTS: float = 1.5  # [knots]
        self.speed_interval: float = 1.0
        self.MAX_ANGLE_DEG: float = 60  # [deg]
        self.MIN_ANGLE_DEG: float = 0  # [deg]
        self.angle_interval: float = 5

        # initial weight ratios (auto-scaled from initial solution)
        self.length_ratio: float = 0.1
        self.SD_ratio: float = 0.5
        self.element_ratio: float = 1.0
        self.distance_ratio: float = 0.2
        self.straight_ratio: float = 1.0
        self.near_buoy_ratio: float = 1.0

        # restart
        self.restarts: int = 3
        self.increase_popsize_on_restart: bool = False

        # output
        self.show_SD_on_optimized_path: bool = True
        self.save_opt_path: bool = True
        self.enable_multiplot: bool = True
        self.save_csv: bool = True

        # label offsets (same as legacy behavior)
        self.start_label_offset_sign: int = +1
        self.end_label_offset_sign: int = -1


class CostCalculator:
    """
    Helper for costs (Ship Domain, element, distance, speed binning).
    PathPlanning wires sample_map / SD / enclosing and bins after map creation.
    """

    def __init__(self):
        self.sample_map = None
        self.SD = None
        self.enclosing = None
        self.MAX_SPEED_KTS = 9.5
        self.MIN_SPEED_KTS = 1.5
        self.speed_interval = 1.0
        self.angle_interval = 5
        self.angle_min = 0
        self.angle_max = 60
        self.speed_bins = None
        self.angle_bins = None
        self.new_filtered_dict = None

    def speed_from_end_distance(self, ver: float, hor: float) -> float:
        """
        Parameters
        ----------
        ver, hor : float
            Current point (X=ver, Y=hor).

        Returns
        -------
        float
            Estimated speed [kts] at (ver,hor) using the guideline fit,
            clipped to [MIN_SPEED_KTS, MAX_SPEED_KTS].
        """
        sm = self.sample_map
        d = np.hypot(ver - sm.end_xy[0, 0], hor - sm.end_xy[0, 1])
        v = sm.b_ave * d ** sm.a_ave + sm.b_SD * d ** sm.a_SD
        return float(np.clip(v, self.MIN_SPEED_KTS, self.MAX_SPEED_KTS))

    def minute_distance(self, ver: float, hor: float) -> float:
        """
        Returns
        -------
        float
            Ideal 1-minute travel distance [m] from (ver,hor).
        """
        return self.speed_from_end_distance(ver, hor) * 1852.0 / 60.0

    def SD_midpoint(self, parent_pt: np.ndarray, child_pt: np.ndarray) -> float:
        """
        Parameters
        ----------
        parent_pt, child_pt : array-like
            (ver, hor) pairs (X, Y).

        Returns
        -------
        float
            Normalized Ship Domain cost [%] evaluated at the segment midpoint.

        Notes
        -----
        - Heading psi is computed using the angle convention: vertical=0°, clockwise positive.
        - Normalized by Glaph.length_of_theta_list × 100.
        """
        ver_p, hor_p = parent_pt
        ver_mid, hor_mid = (parent_pt + child_pt) / 2.0
        ver_c, hor_c = child_pt

        psi = np.deg2rad(90.0) - np.arctan2(ver_c - ver_p, hor_c - hor_p)
        if psi > np.deg2rad(180.0):
            psi = (np.deg2rad(360.0) - psi) * (-1.0)

        contact_mid = self.sample_map.ship_domain_cost(
            ver_mid, hor_mid, psi, self.SD, self.enclosing
        )
        normalized = (contact_mid / Glaph.length_of_theta_list) * 100.0
        return float(normalized)

    def SD_checkpoint(self, parent_pt: np.ndarray, current_pt: np.ndarray, child_pt: np.ndarray) -> float:
        """
        Parameters
        ----------
        parent_pt, current_pt, child_pt : array-like
            (ver, hor) pairs (X, Y).

        Returns
        -------
        float
            Normalized Ship Domain cost [%] evaluated at the "checkpoint" (current_pt).

        Notes
        -----
        - We take the local turning angle at current_pt (0..π) and adjust heading:
          psi ← psi + 0.5 * angle * direction, where direction is CW(+1)/CCW(-1).
        - Angle convention: vertical=0°, clockwise positive.
        - Normalized by Glaph.length_of_theta_list × 100.
        """
        ver_p, hor_p = parent_pt
        ver_c, hor_c = current_pt
        ver_n, hor_n = child_pt

        v1 = np.array([hor_c - hor_p, ver_c - ver_p], dtype=float)
        v2 = np.array([hor_n - hor_c, ver_n - ver_c], dtype=float)
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)

        if m1 == 0.0 or m2 == 0.0:
            angle_rad = 0.0
            direction = 0
        else:
            cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
            angle_rad = float(np.arccos(cos_theta))
            cross = np.cross(v1, v2)
            direction = -1 if cross > 0 else (1 if cross < 0 else 0)  # CCW:-1, CW:+1

        psi = np.deg2rad(90.0) - np.arctan2(ver_c - ver_p, hor_c - hor_p)
        if psi > np.deg2rad(180.0):
            psi = (np.deg2rad(360.0) - psi) * (-1.0)

        psi = psi + 0.5 * angle_rad * direction
        psi = (psi + np.pi) % (2.0 * np.pi) - np.pi  # normalize to [-pi, pi]

        contact_cp = self.sample_map.ship_domain_cost(
            ver_c, hor_c, psi, self.SD, self.enclosing
        )
        normalized = (contact_cp / Glaph.length_of_theta_list) * 100.0
        return float(normalized)

    def elem(self, parent_pt: np.ndarray, current_pt: np.ndarray, child_pt: np.ndarray) -> float:
        """
        Parameters
        ----------
        parent_pt, current_pt, child_pt : array-like
            (ver, hor) pairs (X, Y).

        Returns
        -------
        float
            Element cost defined as (100 - occurrence count).

        Notes
        -----
        - Speed bin at current_pt is determined by distance to END (guideline model).
        - Turning angle at current_pt is the interior angle [0, 180] deg (no ×0.5).
        - Keys are numeric and aligned with `new_filtered_dict`.
        """
        ver_p, hor_p = parent_pt
        ver_c, hor_c = current_pt
        ver_ch, hor_ch = child_pt

        # estimated speed at current_pt (kts)
        current_speed = self.speed_from_end_distance(ver_c, hor_c)

        # speed bin key
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

        # turning angle at current (deg, 0..180)
        v1 = np.array([hor_c - hor_p, ver_c - ver_p], dtype=float)
        v2 = np.array([hor_ch - hor_c, ver_ch - ver_c], dtype=float)
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)
        if m1 == 0.0 or m2 == 0.0:
            angle_deg = 0.0
        else:
            cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_theta)))

        # angle bin key
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

    def distance_cost_between(self, current_pt: np.ndarray, child_pt: np.ndarray) -> float:
        """
        Parameters
        ----------
        current_pt, child_pt : array-like
            (ver, hor) pairs (X, Y).

        Returns
        -------
        float
            Point-to-point distance cost [%] = relative deviation from
            the ideal 1-minute distance based on current speed.
        """
        ver_c, hor_c = current_pt
        ver_n, hor_n = child_pt
        ideal = self.minute_distance(ver_c, hor_c)
        real = float(np.hypot(ver_n - ver_c, hor_n - hor_c))
        if ideal <= 1e-9:
            return 0.0
        return float(abs(ideal - real) / ideal * 100.0)
    
    def carvature(self, parent_pt: np.ndarray, current_pt: np.ndarray, child_pt: np.ndarray) -> float:
        def square(x1, x2, x3):
            x1_ver, x1_hor = x1
            x2_ver, x2_hor = x2
            x3_ver, x3_hor = x3

            a = x2_ver - x3_hor; b = x1_hor - x3_hor
            c = x2_hor - x3_ver; d = x1_ver - x3_ver

            return 0.5 * abs(a*b - c*d)

        a = norm(parent_pt, current_pt); b = norm(current_pt, child_pt); c = norm(child_pt, parent_pt)
        S = square(parent_pt, current_pt, child_pt)
        return 4 * S / (a * b * c)


def sigmoid(x, a, b, c):
    return a / (b + np.exp(c * x))

def norm(x1, x2):
    x1_ver, x1_hor = x1
    x2_ver, x2_hor = x2
    return ((x2_ver - x1_ver)**2 + (x2_hor - x1_hor)**2)**0.5


def round_by_pitch(value, pitch):
    return int(np.round(value / pitch) * pitch)


def undo_conversion(reference_hor_index, reference_ver_index, end_hor_coord, end_ver_coord, indices, grid_pitch):
    """
    Convert grid indices (hor, ver) to real coords using END as reference.

    Parameters
    ----------
    reference_hor_index, reference_ver_index : int
    end_hor_coord, end_ver_coord : float
    indices : array-like, shape (N, 2)
        Grid (hor, ver) pairs.
    grid_pitch : float
        Cell size [m].

    Returns
    -------
    ndarray, shape (N, 2)
        Real coords: (ver, hor) = (y, x) [m]
    """
    idx = np.asarray(indices, dtype=float)  # (N, 2) as (hor_idx, ver_idx)
    ref_idx = np.array([reference_hor_index, reference_ver_index], dtype=float)
    ref_xy = np.array([end_hor_coord, end_ver_coord], dtype=float)
    original = ref_xy - (ref_idx - idx) * grid_pitch  # (hor, ver)
    return original[:, ::-1]  # (ver, hor)


def calculate_turning_points(initial_coords: np.ndarray, sample_map, last_pt: np.ndarray, port: dict) -> list[tuple[float, float]]:
    """
    Parameters
    ----------
    initial_coords : (N, 2) ndarray
        Polyline in (ver, hor).
    sample_map : Map
        Map object with guideline coefficients (a_ave, b_ave, a_SD, b_SD).
    last_pt : array-like
        Last point (ver, hor) before berthing straight segment.
    port : dict
        Port info including 'berth_type' (1: 出船, 2: 入船).

    Returns
    -------
    list[(ver, hor)]
        Turning points spaced by the "1-minute distance" at each base point.
        The search stops once a point is closer than the berth-dependent threshold
        to `last_pt` (80 m for type=1, 120 m for type=2).
    """
    turning_points = []
    current_index = 0
    bt = port["berth_type"]
    stop_thresh = 80.0 if bt == 1 else 120.0

    while current_index < len(initial_coords) - 1:
        current_point = initial_coords[current_index]
        d = np.hypot(current_point[0] - sample_map.end_xy[0, 0], current_point[1] - sample_map.end_xy[0, 1])
        current_speed = sample_map.b_ave * d ** sample_map.a_ave + sample_map.b_SD * d ** sample_map.a_SD
        current_speed = min(current_speed, 9.5)
        minute_distance = current_speed * 1852.0 / 60.0 # km/min

        sum_of_distance = 0.0
        broke = False
        for i in range(current_index, len(initial_coords) - 1):
            seg = np.hypot(
                initial_coords[i + 1][0] - initial_coords[i][0],
                initial_coords[i + 1][1] - initial_coords[i][1],
            )
            sum_of_distance += seg
            if sum_of_distance >= minute_distance:
                distance_to_last = np.hypot(
                    initial_coords[i + 1][0] - last_pt[0],
                    initial_coords[i + 1][1] - last_pt[1],
                )
                if distance_to_last < stop_thresh:
                    broke = True
                    break
                turning_points.append(tuple(initial_coords[i + 1]))
                current_index = i + 1
                break
        else:
            break
        if broke:
            break
    return turning_points



class PathPlanning:
    def __init__(self, ps: Settings, cal: CostCalculator):
        self.ps = ps
        self.cal = cal

        self.sample_map = None
        self.SD = None
        self.enclosing = None

        # cost coeffs (auto-tuned from initial solution)
        self.length_coeff = 1.0
        self.SD_coeff = 1.0
        self.element_coeff = 1.0
        self.distance_coeff = 1.0

    def main(self):
        self.setup()
        self.init_path()
        self.CMAES()
        self.print_result(self.best_dict)
        self.show_result_fig(self.best_dict)
        if self.ps.save_csv:
            self.save_csv(self.best_dict, self.cma_caltime)

    def setup(self):
        self.update_planning_settings()
        os.makedirs(f"{SAVE_DIR}/{self.port['name']}", exist_ok=True)
        self.shipdomain()
        self.prepare_plots_and_variables()
        print(f"### SET UP COMPLETE ###\n")

    def init_path(self):
        self.gen_init_path()
        # sigma vector matches flattened xmean vector length
        self.initial_D = self.cal_sigma_for_ddCMA(self.initial_points, self.last_pt)
        self.initial_vec = self.initial_points.ravel()  # <<< important: flatten for CMA-ES
        self.N = len(self.initial_vec)
        print(
            f"この最適化問題の次元Nは {self.N} です\n"
            "### INITIAL CHECKPOINTS AND sigma0 SETUP COMPLETED ###\n"
            "### MOVED TO THE OPTIMIZATION PROCESS ###\n"
        )

    def CMAES(self):
        # compute auto scaling coefficients from initial solution
        self.compute_cost_weights(self.initial_points)

        # --- CMA-ES expects 1D mean (N,) and sigma0 of same length ---
        ddcma = DdCma(xmean0=self.initial_vec, sigma0=self.initial_D, seed=self.ps.seed)
        checker = Checker(ddcma)
        logger = Logger(ddcma)

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
                best_sofar = best_dict[restart]["best_cost_so_far"]
                pbar.set_postfix_str(
                    f"eval/s={eval_per_s}  trials={trials_str}  best={best_sofar:.6g}"
                )

            while not is_satisfied:
                ddcma.onestep(func=self.path_evaluate)

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
                        f"neval:{ddcma.neval}  "
                        f"cost:{best_cost:.9g}  "
                        f"best:{best_dict[restart]['best_cost_so_far']:.9g}"
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

            cp_list, mp_list, psi_list_at_cp, psi_list_at_mp = self.figure_output(
                best_dict[restart]["best_mean_sofar"], restart, initial_points=self.initial_points
            )
            best_dict[restart]["cp_list"] = cp_list
            best_dict[restart]["mp_list"] = mp_list
            best_dict[restart]["psi_list_at_cp"] = psi_list_at_cp
            best_dict[restart]["psi_list_at_mp"] = psi_list_at_mp

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

    # ---------------- internal helpers ----------------

    def compute_cost_weights(self, initial_pts: np.ndarray):
        """
        Auto-scale four cost coefficients so that their contributions match
        the desired ratios in Settings (length/SD/element/distance), based on
        the initial path.
        """
        sm = self.sample_map
        cal = self.cal
        SD = self.SD

        pts = np.asarray(initial_pts, float).reshape(-1, 2)
        origin = np.asarray(self.origin_pt, float)
        last = np.asarray(self.last_pt, float)
        end = sm.end_xy[0].astype(float)

        # length (normalized by straight origin-last)
        straight = float(np.hypot(last[0] - origin[0], last[1] - origin[1]))
        poly = np.vstack([origin, pts, last])
        seg = np.diff(poly, axis=0)
        total = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
        length_cost = (total / straight) * 100.0 - 100.0
        if length_cost < 0:
            print("total_distance is negative value!")
            sys.exit(1)

        # SD (checkpoint + midpoint)
        SD_cost = 0.0
        if len(pts) >= 2:
            SD_cost += cal.SD_checkpoint(origin, pts[0], pts[1])
            SD_cost += cal.SD_checkpoint(pts[-2], pts[-1], last)
        elif len(pts) == 1:
            SD_cost += cal.SD_checkpoint(origin, pts[0], last)
        for j in range(1, len(pts) - 1):
            SD_cost += cal.SD_checkpoint(pts[j - 1], pts[j], pts[j + 1])

        SD_cost += cal.SD_midpoint(origin, pts[0]) if len(pts) >= 1 else cal.SD_midpoint(origin, last)
        if len(pts) >= 1:
            SD_cost += cal.SD_midpoint(pts[-1], last)
        for j in range(len(pts) - 1):
            SD_cost += cal.SD_midpoint(pts[j], pts[j + 1])

        # element (legacy end-side weights)
        elem_cost = 0.0
        if len(pts) >= 1:
            elem_cost += cal.elem(sm.start_xy[0], origin, pts[0])
        if len(pts) >= 2:
            elem_cost += cal.elem(origin, pts[0], pts[1])
        if len(pts) >= 2:
            elem_cost += 2.0 * cal.elem(pts[-2], pts[-1], last)
        if len(pts) >= 1:
            elem_cost += 3.0 * cal.elem(pts[-1], last, end)
        for j in range(1, len(pts) - 1):
            elem_cost += cal.elem(pts[j - 1], pts[j], pts[j + 1])

        # point-to-point distance
        dist_cost = 0.0
        if len(pts) >= 1:
            dist_cost += cal.distance_cost_between(origin, pts[0])
            dist_cost += cal.distance_cost_between(pts[-1], last)
        else:
            dist_cost += cal.distance_cost_between(origin, last)
        for j in range(len(pts) - 1):
            dist_cost += cal.distance_cost_between(pts[j], pts[j + 1])

        # straightness
        straight_cost = 0.0
        if len(pts) >= 1:
            straight_cost += cal.carvature(sm.start_xy[0], origin, pts[0])
        if len(pts) >= 2:
            straight_cost += cal.carvature(origin, pts[0], pts[1])
        if len(pts) >= 2:
            straight_cost += cal.carvature(pts[-2], pts[-1], last)
        if len(pts) >= 1:
            straight_cost += cal.carvature(pts[-1], last, end)
        for j in range(1, len(pts) - 1):
            straight_cost += cal.carvature(pts[j - 1], pts[j], pts[j + 1])
        
        # nearest_buoy_distance
        # near_buoy_cost = 0.0
        # poly = np.vstack([origin, pts, last])
        # for pt in pts:
        #     for buoy in sm.buoy_xy:
        #         dist = norm(buoy, pt)
        #         theta = 60
        #         speed = sm.b_ave * dist ** (sm.a_ave) + sm.b_SD * dist ** (sm.a_SD)
        #         near_buoy_cost += 1 / (dist / SD.distance(speed, theta))

        # coefficients
        self.element_coeff = 1.0 * self.ps.element_ratio
        self.length_coeff = (elem_cost / length_cost) * self.ps.length_ratio if length_cost > 0 else 1.0
        self.SD_coeff = (elem_cost / SD_cost) * self.ps.SD_ratio if SD_cost > 0 else 10.0
        self.distance_coeff = (elem_cost / dist_cost) * self.ps.distance_ratio if dist_cost > 0 else 1.0
        self.straight_coeff = (elem_cost / straight_cost) * self.ps.straight_ratio
        # self.near_bay_coeff = (elem_cost / near_buoy_cost) * self.ps.near_buoy_ratio 


    def path_evaluate(self, X: np.ndarray) -> np.ndarray | float:
        """
        CMA-ES objective.

        Parameters
        ----------
        X : ndarray
            Either shape (lam, N) or (N,), where coordinates are flattened
            as [ver1, hor1, ver2, hor2, ...].

        Returns
        -------
        ndarray or float
            Total cost(s) = a*Length + b*SD + c*Element + d*Distance.
        """
        batched = True
        arr = np.asarray(X, float)
        if arr.ndim == 1:
            arr = arr[None, :]
            batched = False

        sm = self.sample_map
        cal = self.cal
        SD = self.SD
        origin = np.asarray(self.origin_pt, float)
        last = np.asarray(self.last_pt, float)
        end = sm.end_xy[0].astype(float)

        straight = float(np.hypot(last[0] - origin[0], last[1] - origin[1]))

        costs = np.zeros(arr.shape[0], dtype=float)
        for i in range(arr.shape[0]):
            pts = arr[i].reshape(-1, 2)

            # length
            poly = np.vstack([origin, pts, last])
            seg = np.diff(poly, axis=0)
            total_len = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
            length_cost = (total_len / straight) * 100.0 - 100.0
            if length_cost < 0:
                length_cost = 0.0

            # SD
            SD_cost = 0.0
            if len(pts) >= 2:
                SD_cost += cal.SD_checkpoint(origin, pts[0], pts[1])
                SD_cost += cal.SD_checkpoint(pts[-2], pts[-1], last)
            elif len(pts) == 1:
                SD_cost += cal.SD_checkpoint(origin, pts[0], last)
            for j in range(1, len(pts) - 1):
                SD_cost += cal.SD_checkpoint(pts[j - 1], pts[j], pts[j + 1])

            if len(pts) >= 1:
                SD_cost += cal.SD_midpoint(origin, pts[0])
                SD_cost += cal.SD_midpoint(pts[-1], last)
            else:
                SD_cost += cal.SD_midpoint(origin, last)
            for j in range(len(pts) - 1):
                SD_cost += cal.SD_midpoint(pts[j], pts[j + 1])

            # element
            elem_cost = 0.0
            if len(pts) >= 1:
                elem_cost += cal.elem(sm.start_xy[0], origin, pts[0])
            if len(pts) >= 2:
                elem_cost += cal.elem(origin, pts[0], pts[1])
            if len(pts) >= 2:
                elem_cost += 2.0 * cal.elem(pts[-2], pts[-1], last)
            if len(pts) >= 1:
                elem_cost += 3.0 * cal.elem(pts[-1], last, end)
            for j in range(1, len(pts) - 1):
                elem_cost += cal.elem(pts[j - 1], pts[j], pts[j + 1])

            # distance
            dist_cost = 0.0
            if len(pts) >= 1:
                dist_cost += cal.distance_cost_between(origin, pts[0])
                dist_cost += cal.distance_cost_between(pts[-1], last)
            else:
                dist_cost += cal.distance_cost_between(origin, last)
            for j in range(len(pts) - 1):
                dist_cost += cal.distance_cost_between(pts[j], pts[j + 1])

            # straightness
            straight_cost = 0.0
            if len(pts) >= 1:
                straight_cost += cal.carvature(sm.start_xy[0], origin, pts[0])
            if len(pts) >= 2:
                straight_cost += cal.carvature(origin, pts[0], pts[1])
            if len(pts) >= 2:
                straight_cost += cal.carvature(pts[-2], pts[-1], last)
            if len(pts) >= 1:
                straight_cost += cal.carvature(pts[-1], last, end)
            for j in range(1, len(pts) - 1):
                straight_cost += cal.carvature(pts[j - 1], pts[j], pts[j + 1])

            # nearest_buoy_distance
            # near_buoy_cost = 0.0
            # poly = np.vstack([origin, pts, last])
            # for pt in pts:
            #     for buoy in sm.buoy_xy:
            #         dist = norm(buoy, pt)
            #         theta = 60
            #         speed = sm.b_ave * dist ** (sm.a_ave) + sm.b_SD * dist ** (sm.a_SD)
            #         near_buoy_cost += 1 / (dist / SD.distance(speed, theta))


            total = (
                self.length_coeff * length_cost
                + self.SD_coeff * SD_cost
                + self.element_coeff * elem_cost
                + self.distance_coeff * dist_cost
                + self.straight_coeff * straight_cost
            )
            costs[i] = total

        return float(costs[0]) if not batched else costs

    def figure_output(self, best_mean, restart, initial_points):
        """
        Save optimized path figure via sample_map.ShowMap, and return cp/mp lists with psi.
        """
        path_xy = np.asarray(best_mean, float)
        sm = self.sample_map

        path_coord = path_xy.reshape(-1, 2)
        path_coord = [
            (round_by_pitch(ver, self.ps.gridpitch), round_by_pitch(hor, self.ps.gridpitch))
            for ver, hor in path_coord
        ]
        path_coord = [tuple(coord) for coord in path_coord]
        path_coord_idx = [
            (
                int(np.argmin(np.abs(sm.ver_range - ver))),
                int(np.argmin(np.abs(sm.hor_range - hor))),
            )
            for ver, hor in path_coord
        ]
        sm.path_node = path_coord_idx

        sm.path_xy = np.empty((0, 2))
        for i in range(len(sm.path_node)):
            sm.path_xy = np.append(
                sm.path_xy,
                np.array([[sm.ver_range[sm.path_node[i][0]], sm.hor_range[sm.path_node[i][1]]]]),
                axis=0,
            )
        print(f"\n{sm.path_xy}")

        cp_list, mp_list, psi_at_cp, psi_at_mp = sm.ShowMap(
            filename=f"{SAVE_DIR}/{self.port['name']}/Path_by_CMA_{self.port['name']}_{restart}.png",
            SD=self.SD,
            initial_point_list=initial_points,
            optimized_point_list=path_coord,
            SD_sw=self.ps.show_SD_on_optimized_path,
        )

        return cp_list, mp_list, psi_at_cp, psi_at_mp

    def shipdomain(self):
        port = self.port
        target_csv = f"{DATA}/rough_map/{port['name']}.csv"
        target_csv_for_pyship = f"{DATA}/rough_map_for_pyship/{port['name']}.csv"

        df_world = pd.read_csv(target_csv_for_pyship)
        world_polys = [df_world[["x [m]", "y [m]"]].to_numpy()]
        enclosing = pyshipsim.EnclosingPointCollisionChecker()
        enclosing.reset(world_polys)
        self.enclosing = enclosing
        print(f"Successfully imported data from csv\n")

        SD = ShipDomain_proposal()
        SD.initial_setting(f"{DIR}/../../outputs/303/mirror5/fitting_parameter.csv", sigmoid)
        self.SD = SD
        print(f"Generating map from data\n")

        time_start_map_generation = time.time()
        sample_map = Glaph.Map.GenerateMapFromCSV(target_csv, self.ps.gridpitch_for_Astar)
        time_end_map_generation = time.time()
        print(f"Map generation is complete.\nCalculation time : {time_end_map_generation - time_start_map_generation}\n")

        df = pd.read_csv(f"{RAW_DATAS}/tmp/GuidelineFit_debug.csv")
        sample_map.a_ave = df["a_ave"].values[0]
        sample_map.b_ave = df["b_ave"].values[0]
        sample_map.a_SD = df["a_SD"].values[0]
        sample_map.b_SD = df["b_SD"].values[0]

        self.sample_map = sample_map

        # wire CostCalculator
        self.cal.sample_map = self.sample_map
        self.cal.SD = self.SD
        self.cal.enclosing = self.enclosing
        self.cal.new_filtered_dict = new_filtered_dict()
        self.cal.MAX_SPEED_KTS = float(self.ps.MAX_SPEED_KTS)
        self.cal.MIN_SPEED_KTS = float(self.ps.MIN_SPEED_KTS)
        self.cal.speed_interval = float(self.ps.speed_interval)
        self.cal.angle_interval = int(self.ps.angle_interval)
        self.cal.angle_min = int(self.ps.MIN_ANGLE_DEG)
        self.cal.angle_max = int(self.ps.MAX_ANGLE_DEG)
        self.cal.speed_bins = np.arange(self.cal.MIN_SPEED_KTS, self.cal.MAX_SPEED_KTS, self.cal.speed_interval)
        self.cal.angle_bins = np.arange(self.cal.angle_min, self.cal.angle_max, self.cal.angle_interval)

    def prepare_plots_and_variables(self):
        port = self.port
        sm = self.sample_map

        if self.ps.start_end_mode == ParamMode.AUTO:
            sm.start_raw = np.array([port["start"]])
            sm.end_raw = np.array([port["end"]])
        else:
            sm.start_raw = np.array([self.start_coord])
            sm.end_raw = np.array([self.end_coord])

        sm.start_xy = sm.FindNodeOfThePoint(sm.start_raw[0, :])
        sm.end_xy = sm.FindNodeOfThePoint(sm.end_raw[0, :])

        if self.ps.range_type == 1:
            ver_margin = 200
            hor_margin = 200

            ver_min = min((np.amin(sm.ver_range), sm.start_raw[0, 0], sm.end_raw[0, 0])) - ver_margin
            ver_max = max((np.amax(sm.ver_range), sm.start_raw[0, 0], sm.end_raw[0, 0])) + ver_margin
            hor_min = min((np.amin(sm.hor_range), sm.start_raw[0, 1], sm.end_raw[0, 1])) - hor_margin
            hor_max = max((np.amax(sm.hor_range), sm.start_raw[0, 1], sm.end_raw[0, 1])) + hor_margin
        else:
            ver_min = -15
            ver_max = +10
            hor_min = -20
            hor_max = +20

        ver_min_round = Glaph.Map.RoundRange(None, ver_min, sm.grid_pitch, "min")
        ver_max_round = Glaph.Map.RoundRange(None, ver_max, sm.grid_pitch, "max")
        hor_min_round = Glaph.Map.RoundRange(None, hor_min, sm.grid_pitch, "min")
        hor_max_round = Glaph.Map.RoundRange(None, hor_max, sm.grid_pitch, "max")
        sm.ver_range = np.arange(ver_min_round, ver_max_round + sm.grid_pitch / 10, sm.grid_pitch)
        sm.hor_range = np.arange(hor_min_round, hor_max_round + sm.grid_pitch / 10, sm.grid_pitch)

        self.start_ver_idx = np.where(sm.ver_range == sm.start_xy[0, 0])
        self.start_hor_idx = np.where(sm.hor_range == sm.start_xy[0, 1])
        self.end_ver_idx = np.where(sm.ver_range == sm.end_xy[0, 0])
        self.end_hor_idx = np.where(sm.hor_range == sm.end_xy[0, 1])

        sx, sy = sm.start_xy[0, 0], sm.start_xy[0, 1]
        ex, ey = sm.end_xy[0, 0], sm.end_xy[0, 1]
        d_se = float(np.hypot(sx - ex, sy - ey))

        if self.ps.psi_mode == ParamMode.AUTO:
            self.psi_start = np.deg2rad(self.port["psi_start"])
            self.psi_end = np.deg2rad(self.port["psi_end"])
            print("\npsi    :default\n")
        else:
            self.psi_start = np.deg2rad(self.manual_psi_start)
            self.psi_end = np.deg2rad(self.manual_psi_end)

        start_speed = min(sm.b_ave * d_se ** sm.a_ave + sm.b_SD * d_se ** sm.a_SD, self.ps.MAX_SPEED_KTS)
        print(f"start speed :{start_speed}[knots]")

        origin_navigation_distance = start_speed * 1852.0 / 60.0
        u_start = np.array([np.cos(self.psi_start), np.sin(self.psi_start)])
        origin_pt = sm.start_xy[0] + origin_navigation_distance * u_start
        sm.origin_xy = sm.FindNodeOfThePoint(origin_pt)

        self.origin_pt = origin_pt
        self.origin_ver_idx = np.where(sm.ver_range == sm.origin_xy[0, 0])
        self.origin_hor_idx = np.where(sm.hor_range == sm.origin_xy[0, 1])

        if self.ps.steady_course_coeff_mode == ParamMode.AUTO:
            self.steady_course_coeff = 1.2
            print(f"steady_course_coeff : default\n")
        else:
            self.steady_course_coeff = 0.0
            print(f"steady_course_coeff : {self.steady_course_coeff}\n")

        straight_dist = self.ps.L * self.steady_course_coeff if self.ps.enable_pre_berthing_straight_segment else 0.0
        u_end = np.array([np.cos(self.psi_end), np.sin(self.psi_end)])
        last_pt = sm.end_xy[0] - straight_dist * u_end
        sm.last_xy = sm.FindNodeOfThePoint(last_pt)

        self.last_pt = last_pt
        self.last_ver_idx = np.where(sm.ver_range == sm.last_xy[0, 0])
        self.last_hor_idx = np.where(sm.hor_range == sm.last_xy[0, 1])

    def gen_init_path(self):
        print("Initial Path generation starts")
        time_start_init_path = time.time()
        sm = self.sample_map
        sm.path_xy = np.empty((0, 2))
        #
        if self.ps.init_path_algo == InitPathAlgo.ASTAR:
            Glaph.Map.SetMaze(sm)
            weight = sm.grid_pitch * 20  # default
            #
            sm.path_node, sm.psi, _ = Astar.astar(
                sm,
                (self.origin_hor_idx[0][0], self.origin_ver_idx[0][0]),
                (self.last_hor_idx[0][0], self.last_ver_idx[0][0]),
                psi_start=self.psi_start,
                psi_end=self.psi_end,
                SD=self.SD,
                weight=weight,
                enclosing_checker=self.enclosing,
            ) # return : index=[i, j]
            # [i, j] -> [ver, hor]
            initial_coord_xy = undo_conversion(
                self.end_hor_idx[0][0],
                self.end_ver_idx[0][0],
                sm.end_xy[0, 1],
                sm.end_xy[0, 0],
                sm.path_node,
                self.ps.gridpitch,
            )
            caltime = time.time() - time_start_init_path
            print(f"Astar algorithm took {caltime:.3f} [s]\n")

        elif self.ps.init_path_algo == InitPathAlgo.BEZIER:
            port = self.port
            if (files := glob.glob(f"{DATA}/buoy/{port['name']}.csv")):
                buoy = Buoy()
                buoy.input_csv(files[0], f"{RAW_DATAS}/tmp/coordinates_of_port/_{port['name']}.csv")
                sm.buoy_xy = [buoy.X, buoy.Y]
            else:
                sm.buoy_xy = None

            initial_coord_xy, sm.psi, sm.isect_xy = Bezier.bezier(sm=sm, buoy_xy=sm.buoy_xy, num=400)
            caltime = time.time() - time_start_init_path
            print(f"Bezier algorithm took {caltime:.3f} [s]\n")

        else:
            # manual configuration (not used here)
            pass

        initial_points = calculate_turning_points(initial_coord_xy, sm, self.last_pt, self.port)
        print("Initial Turning Points:\n",)
        for i, (x, y) in enumerate(initial_points, 1):
            print(f"  P{i:02d}: ({x:.1f}, {y:.1f})")

        if self.ps.save_init_path:
            sm.path_xy = initial_coord_xy
            self.save_init_path(sm, initial_points)

        self.initial_points = np.array(initial_points, dtype=float)

    def save_init_path(self, sm, initial_points):
        filename = (
            f"{SAVE_DIR}/{self.port['name']}/Initial_Path_by_{self.ps.init_path_algo.name}_with_SD.png"
            if self.ps.show_SD_on_init_path
            else f"{SAVE_DIR}/{self.port['name']}/Initial_Path_by_{self.ps.init_path_algo.name}_without_SD.png"
        )
        sm.ShowMap(
            filename=filename,
            SD=self.SD,
            initial_point_list=initial_points,
            optimized_point_list=None,
            SD_sw=self.ps.show_SD_on_init_path,
        )

    def cal_sigma_for_ddCMA(
        self,
        points: np.ndarray,
        last_point: Tuple[float, float],
        *,
        min_sigma: float = 5.0,
        scale: float = 0.5,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        points : (N,2) ndarray
            Initial mean points (ver, hor).
        last_point : (ver, hor)
            Last point before berthing.
        min_sigma : float
            Lower bound for each coordinate sigma to avoid zeros.
        scale : float
            Scale for difference-based initialization.

        Returns
        -------
        (2N,) ndarray
            Sigma vector interleaving (ver, hor) sigmas.
        """
        ver = points[:, 0]
        hor = points[:, 1]
        ver_diffs = np.abs(np.diff(ver, append=last_point[0])) * scale
        hor_diffs = np.abs(np.diff(hor, append=last_point[1])) * scale
        ver_diffs = np.maximum(ver_diffs, min_sigma)
        hor_diffs = np.maximum(hor_diffs, min_sigma)
        return np.column_stack((ver_diffs, hor_diffs)).ravel()

    def update_planning_settings(self):
        port = self.dict_of_port(self.ps.port_number)
        self.port = port

        # ---- pretty print helper ----
        LABELS = (
            "target", "start_end", "psi", "steady_course_coeff",
            "init_path_algo", "weight_SD", "save_init_path", "init_SD",
            "save_opt_path", "csv"
        )
        LABEL_W = max(len(s) for s in LABELS)
        def p(key, value):
            print(f"{key:<{LABEL_W}}  : {value}")

        # --- ratios for cost weights (concise one-liner) ---
        print(
            f"{'項目':<12}{'比率'}\n"
            f"{'-'*25}\n"
            f"{'Length':<12}{self.ps.length_ratio}\n"
            f"{'SD':<12}{self.ps.SD_ratio}\n"
            f"{'Element':<12}{self.ps.element_ratio}\n"
            f"{'Distance':<12}{self.ps.distance_ratio}\n"
            f"{'Straight':<12}{self.ps.straight_ratio}\n"
        )

        # --- start & end ---
        self.weight_of_SD = 20
        if self.ps.start_end_mode == ParamMode.AUTO:
            p("start_end", "default")
        else:
            self.start_coord = [-600.0, -400.0]
            self.end_coord = [0.0, 0.0]
            p("start_end", "manual")

        # --- psi (heading) ---
        if self.ps.psi_mode == ParamMode.AUTO:
            p("psi", "default")
        else:
            self.manual_psi_start = -20
            self.manual_psi_end = 10
            p("psi", "manual")

        # --- steady course coefficient ---
        if self.ps.steady_course_coeff_mode == ParamMode.AUTO:
            p("steady_course_coeff", "default")
        else:
            p("steady_course_coeff", "0")

        # --- init path algorithm & related flags ---
        if self.ps.init_path_algo == InitPathAlgo.ASTAR:
            p("init_path_algo", "Astar")
            p("weight_SD", f"grid_pitch*{self.weight_of_SD}")
        elif self.ps.init_path_algo == InitPathAlgo.BEZIER:
            p("init_path_algo", "Bezier")    
        else:
            p("init_path_algo", "Manual")

        # -- init path fig --
        if self.ps.save_init_path:
            p("save_init_path", "on")
            p("init_SD", "on" if self.ps.show_SD_on_init_path else "off")
        else:
            p("save_init_path", "off")

        # --- save options ---
        p("save_opt_path", "on" if self.ps.save_opt_path else "off")
        p("csv", "on" if self.ps.save_csv else "off")

        # --- target ---
        p("target", f"{port['name']}")

    def dict_of_port(self, num):
        dictionary_of_port = {
            0: {
                "name": "Osaka_port1A",
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
                "start": [2000.0, 2000.0],
                "end": [300.0, 80.0],
                "psi_start": -125,
                "psi_end": 175,
                "berth_type": 1,
                "ver_range": [-500, 2500],
                "hor_range": [-500, 2500],
            },
            3: {
                "name": "Else_port1",
                "start": [2500.0, 0.0],
                "end": [450.0, 20.0],
                "psi_start": -150,
                "psi_end": 135,
                "berth_type": 1,
                "ver_range": [0, 3000],
                "hor_range": [-1000, 2000],
            },
            4: {
                "name": "Osaka_port1B",
                "start": [-3000.0, -1080.0],
                "end": [-480.0, -80.0],
                "psi_start": -7,
                "psi_end": 45,
                "berth_type": 2,
                "ver_range": [-3200, 500],
                "hor_range": [-1600, 500],
            },
            5: {
                "name": "Else_port2",
                "start": [-1000.0, 650.0],
                "end": [0.0, 0.0],
                "psi_start": -45,
                "psi_end": -15,
                "berth_type": 2,
                "ver_range": [-1900, 300],
                "hor_range": [-1000, 1200],
            },
        }
        return dictionary_of_port[num]

    def print_result(self, best_dict):
        for restart, values in best_dict.items():
            best_cost_so_far = values["best_cost_so_far"]
            best_mean_sofar = values["best_mean_sofar"]
            calculation_time = values["calculation_time"]
            pairs = "\n".join(
                f"  ({best_mean_sofar[i]:.6f}, {best_mean_sofar[i+1]:.6f})"
                for i in range(0, len(best_mean_sofar), 2)
            )
            print(
                f"\n[Restart {restart}]\n"
                f"  best_cost_so_far: {best_cost_so_far:.6f}\n"
                f"  計算時間: {calculation_time:.2f} s\n"
                f"  best_mean_sofar:\n{pairs}"
            )
        print("\n" + "=" * 50 + "\n")
        smallest_evaluation_key = min(best_dict, key=lambda k: best_dict[k]["best_cost_so_far"])
        print(f"最も評価値が小さかった試行は {smallest_evaluation_key} 番目\n")
        print(f"最小評価値: {best_dict[smallest_evaluation_key]['best_cost_so_far']}\n")
        print(f"  対応する最適解 (ver, hor):")
        best_solution = best_dict[smallest_evaluation_key]["best_mean_sofar"]
        for i in range(0, len(best_solution), 2):
            print(f"({best_solution[i]:.6f}, {best_solution[i+1]:.6f})")

        self.smallest_evaluation_key = smallest_evaluation_key

    def show_result_fig(self, best_dict):
        """
        Plot CMA-ES logger results and a multi-plot map (without captain routes).
        """
        port = self.port
        points = self.initial_points
        SD = self.SD
        sm = self.sample_map
        logger = self.logger
        best_key = self.smallest_evaluation_key
        folder_path = f"{SAVE_DIR}/{port['name']}"

        # logger plot
        fig, axdict = logger.plot()
        for ax_name, ax in axdict.items():
            if ax_name != "xmean":
                ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(f"{folder_path}/Result_of_CMA_{port['name']}.png")

        if not self.ps.enable_multiplot:
            return

        # ---- multiplot ----
        pointsize = 2
        fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
        gs = gridspec.GridSpec(4, 3, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0:2])

        # map
        df_map = pd.read_csv(f"{DATA}/detail_map/{port['name']}.csv")
        map_X, map_Y = df_map["x [m]"].values, df_map["y [m]"].values
        ax1.fill_betweenx(map_X, map_Y, facecolor="gray", alpha=0.3)
        ax1.plot(map_Y, map_X, color="k", linestyle="--", lw=0.5, alpha=0.8)

        # captain's route
        df_cap = glob.glob(f"{RAW_DATAS}/tmp/_{port['name']}/*.csv")
        for df in df_cap:
            traj = RealTraj()
            traj.input_csv(df, f"{RAW_DATAS}/tmp/coordinates_of_port/_{port['name']}.csv")
            ax1.plot(traj.Y, traj.X, 
                     color = 'gray', ls = '-', marker = 'D',
                     markersize = 2, alpha = 0.8, lw = 1.0, zorder = 1)
            legend_captain = plt.Line2D([0], [0],
                                        color = 'gray', ls = '-', marker = 'D',
                                        markersize = 2, alpha = 0.8, lw = 1.0, label="captain's Route"
            )

        # buoy
        df_buoy = glob.glob(f"{DATA}/buoy/{port['name']}.csv")
        if len(df_buoy) != 0:
            buoy = Buoy()
            buoy.input_csv(df_buoy[0], f"{RAW_DATAS}/tmp/coordinates_of_port/_{port['name']}.csv")
            ax1.scatter(buoy.Y, buoy.X,
                        color='orange', s=20, zorder=4)
        legend_buoy = plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor="orange", markersize=pointsize, label="Buoy Point")

        # key points
        start_point = (sm.start_xy[0, 0], sm.start_xy[0, 1])
        end_point = (sm.end_xy[0, 0], sm.end_xy[0, 1])
        origin_point = (sm.origin_xy[0, 0], sm.origin_xy[0, 1])
        last_point = (sm.last_xy[0, 0], sm.last_xy[0, 1])

        # initial points and path
        pts = np.asarray(points, dtype=float)
        ax1.scatter(pts[:, 1], pts[:, 0], color="#03AF7A", s=20, zorder=4)
        legend_initial = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#03AF7A", markersize=pointsize, label="Initial Point")
        full_initial_path = np.vstack([np.asarray(start_point), np.asarray(origin_point), pts, np.asarray(last_point), np.asarray(end_point)])
        ax1.plot(full_initial_path[:, 1], full_initial_path[:, 0], color="#03AF7A", linestyle="-", linewidth=1.5, alpha=0.8, zorder=2)

        # optimized path + SD
        cp_list = best_dict[best_key]["cp_list"]
        mp_list = best_dict[best_key]["mp_list"]
        psi_at_cp = best_dict[best_key]["psi_list_at_cp"]
        psi_at_mp = best_dict[best_key]["psi_list_at_mp"]

        cp_hor = [h for v, h in cp_list]
        cp_ver = [v for v, h in cp_list]
        ax1.scatter(cp_hor, cp_ver, color="#005AFF", marker="o", s=25, zorder=4)
        legend_way = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#005AFF", markersize=pointsize, label="Optimized Point")

        theta = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(10))
        theta_closed = np.append(theta, theta[0])

        # SD at CP
        for (v, h), psi in zip(cp_list, psi_at_cp):
            dist = np.hypot(h - sm.end_xy[0, 1], v - sm.end_xy[0, 0])
            speed = sm.b_ave * dist ** sm.a_ave + sm.b_SD * dist ** sm.a_SD
            r = np.array([SD.distance(speed, t) for t in theta] + [SD.distance(speed, theta[0])])
            ax1.plot(h + r * np.sin(theta_closed + psi), v + r * np.cos(theta_closed + psi), lw=0.6, color="#005AFF", ls="--", zorder=3)
        legend_SD = plt.Line2D([0], [0], linestyle="--", color="#005AFF", lw=1.0, label="Ship Domain")

        # SD at MP
        for (v, h), psi in zip(mp_list, psi_at_mp):
            dist = np.hypot(h - sm.end_xy[0, 1], v - sm.end_xy[0, 0])
            speed = sm.b_ave * dist ** sm.a_ave + sm.b_SD * dist ** sm.a_SD
            r = np.array([SD.distance(speed, t) for t in theta] + [SD.distance(speed, theta[0])])
            ax1.plot(h + r * np.sin(theta_closed + psi), v + r * np.cos(theta_closed + psi), lw=0.8, color="#005AFF", ls="--")

        path_points = [start_point, origin_point] + [(v, h) for v, h in cp_list] + [last_point, end_point]
        path_points = np.asarray(path_points, float)
        ax1.plot(path_points[:, 1], path_points[:, 0], color="#005AFF", linestyle="-", linewidth=2.5, alpha=0.8, zorder=3)

        # start/end/origin/last markers
        ax1.scatter(sm.start_xy[0, 1], sm.start_xy[0, 0], color="k", s=20, zorder=4)
        ax1.text(sm.start_xy[0, 1], sm.start_xy[0, 0] + (60 * self.ps.start_label_offset_sign), "start", va="center", ha="right", fontsize=20)
        ax1.scatter(sm.end_xy[0, 1], sm.end_xy[0, 0], color="k", s=20, zorder=4)
        ax1.text(sm.end_xy[0, 1], sm.end_xy[0, 0] + (60 * self.ps.end_label_offset_sign), "end", va="center", ha="left", fontsize=20)
        ax1.scatter(sm.origin_xy[0, 1], sm.origin_xy[0, 0], color="#FF4B00", s=20, zorder=4)
        ax1.scatter(sm.last_xy[0, 1], sm.last_xy[0, 0], color="#FF4B00", s=20, zorder=4)
        ax1.scatter(sm.isect_xy[1], sm.isect_xy[0], color="#FF4B00", s=20, zorder=4)
        legend_fixed = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF4B00", markersize=pointsize, label="Fixed Point")

        # compas
        img = mpimg.imread(f"{RAW_DATAS}/compass icon.png")
        df = pd.read_csv(f"{RAW_DATAS}/tmp/coordinates_of_port/_{port['name']}.csv")
        angle = float(df['Psi[deg]'].iloc[0])
        img_rot = ndimage.rotate(img, -angle, reshape=True)
        imagebox = OffsetImage(img_rot)
        ab = AnnotationBbox(
            imagebox,
            (1, 1), # upper left
            xycoords='axes fraction',
            box_alignment=(0, 1),
            frameon=False,
            pad=0.0
        )
        ax.add_artist(ab)

        # axes/ticks
        hor_lim = [port["hor_range"][0], port["hor_range"][1]]
        ver_lim = [port["ver_range"][0], port["ver_range"][1]]
        ax1.set_xlim(*hor_lim)
        ax1.set_ylim(*ver_lim)

        tick_int = 500
        x_start = int(np.floor(hor_lim[0] / tick_int) * tick_int)
        x_end = int(np.ceil(hor_lim[1] / tick_int) * tick_int)
        y_start = int(np.floor(ver_lim[0] / tick_int) * tick_int)
        y_end = int(np.ceil(ver_lim[1] / tick_int) * tick_int)
        ax1.set_xticks(np.arange(x_start, x_end + tick_int, tick_int))
        ax1.set_yticks(np.arange(y_start, y_end + tick_int, tick_int))
        ax1.set_xticklabels(np.arange(x_start, x_end + tick_int, tick_int).astype(int), rotation=90)
        ax1.set_yticklabels(np.arange(y_start, y_end + tick_int, tick_int).astype(int))

        ax1.set_aspect("equal")
        ax1.grid()
        ax1.set_xlabel(r"$Y\,\rm{[m]}$")
        ax1.set_ylabel(r"$X\,\rm{[m]}$")
        ax1.legend(handles=[legend_initial, legend_way, legend_fixed, legend_buoy, legend_captain, legend_SD])

        fig.savefig(f"{folder_path}/Multiplot_{port['name']}_{self.ps.init_path_algo.name}.png",
                    bbox_inches="tight", pad_inches=0.05)
        plt.close()

    def save_csv(self, best_dict: dict, cma_caltime: float):
        """
        Save optimization summary CSV (legacy-compatible columns).
        """
        folder_path = f"{SAVE_DIR}/{self.port['name']}"
        csv_file_path = f"{folder_path}/csv_{self.port['name']}.csv"

        basic_header = [
            "start_ver",
            "start_hor",
            "end_ver",
            "end_hor",
            "number_of_trials",
            "total_cal_time",
        ]
        sm = self.sample_map
        basic_data = [[
            round(float(sm.start_xy[0, 0]), 2),
            round(float(sm.start_xy[0, 1]), 2),
            round(float(sm.end_xy[0, 0]), 2),
            round(float(sm.end_xy[0, 1]), 2),
            len(best_dict),
            round(float(cma_caltime), 2),
        ]]
        df = pd.DataFrame(basic_data, columns=basic_header)

        init_cols = ["initial_point_ver", "initial_point_hor"]
        df_init = pd.DataFrame(self.initial_points, columns=init_cols)
        df = pd.concat([df, df_init], axis=1)

        best_key = min(best_dict, key=lambda k: best_dict[k]["best_cost_so_far"])
        sol = np.asarray(best_dict[best_key]["best_mean_sofar"], float)
        df_opt = pd.DataFrame([sol[i : i + 2] for i in range(0, len(sol), 2)], columns=["optimal_point_ver", "optimal_point_hor"])
        df = pd.concat([df, df_opt], axis=1)

        df.to_csv(csv_file_path, index=False)
        print(f"CSV saved\n")

        print(
            f"{'-'*25}\n"
            f"{'Length':<12}{self.ps.length_ratio}\n"
            f"{'SD':<12}{self.ps.SD_ratio}\n"
            f"{'Element':<12}{self.ps.element_ratio}\n"
            f"{'Distance':<12}{self.ps.distance_ratio}\n"
            f"{'Straight':<12}{self.ps.straight_ratio}\n"
        )


if __name__ == "__main__":
    ps = Settings()
    cal = CostCalculator()
    path_planning = PathPlanning(ps, cal)
    path_planning.main()
    print("\nDone\n")
