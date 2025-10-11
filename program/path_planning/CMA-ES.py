"""
CMA-ES path optimization (A* init → element-based turning points → CMA-ES)
- Max speed: 9.5 knots
- Ship Domain at segment midpoint
- Angle convention: vertical(X) = 0 deg, clockwise positive
- Coordinate note: vertical = X (ver), horizontal = Y (hor)
"""

from __future__ import annotations
import copy
from enum import StrEnum
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import utils.PP.Astar_for_CMAES as Astar
import utils.PP.graph_by_taneichi as Glaph
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.MakeDictionary_and_StackedBarGraph import new_filtered_dict
from utils.PP.graph_by_taneichi import ShipDomain_proposal
from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay
from utils.PP.MultiPlot import RealTraj

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
PYSIM_DIR = os.path.join(PROGRAM_DIR, "py-ship-simulator-main/py-ship-simulator-main")
if PYSIM_DIR not in sys.path:
    sys.path.append(PYSIM_DIR)
import pyshipsim 

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
TMP_DIR = f"{DIR}/../../raw_datas/tmp"


class ParamMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class InitPathAlgo(StrEnum):
    ASTAR = "astar"


class Settings:
    def __init__(self):
        # port
        self.port_number: int = 2
        # ship
        self.L = 100

        # setup  /  initial path
        self.start_end_mode: ParamMode = ParamMode.AUTO
        self.psi_mode: ParamMode = ParamMode.AUTO
        self.steady_course_coeff_mode: ParamMode = ParamMode.AUTO
        self.init_path_algo: InitPathAlgo = InitPathAlgo.ASTAR
        self.enable_pre_berthing_straight_segment: bool = True
        
        self.save_init_path: bool = True
        self.show_SD_on_init_path: bool = True
        self.gridpitch: float = 5.0 #[m]
        self.gridpitch_for_Astar: float = 5.0 #[m]
        self.range_type:float = 1
        # CMA-ES
        self.seed: float = 42
        self.MAX_SPEED_KTS:float = 9.5 # [knots]
        self.MIN_SPEED_KTS:float = 1.5 # [knots]
        self.speed_interval:float = 1.0
        self.MAX_ANGLE_DEG:float = 60 # [deg]
        self.MIN_ANGLE_DEG:float = 0 # [deg]
        self.angle_interval:float = 5
        self.length_ratio: float = 0.1
        self.SD_ratio: float = 0.5
        self.element_ratio: float = 1.0
        self.distance_ratio: float = 0.2
        # restart
        self.restarts: int = 3
        self.increase_popsize_on_restart: bool = False
        
        self.show_SD_on_optimized_path: bool = True
        self.save_opt_path: bool = True
        self.enable_multiplot: bool = True

class CostCalculator:
    def __init__(self):
        pass

    def SD_midpoint(self, parent_pt, child_pt):
        """
        parent_pt, child_pt: (ver, hor)
        戻り値: 正規化Ship Domainコスト（%）
        """
        ver_p, hor_p = parent_pt
        ver_mid, hor_mid = (parent_pt + child_pt) /2
        ver_c, hor_c = child_pt
        #
        psi = np.deg2rad(90) - np.arctan2(ver_c - ver_p, hor_c - hor_p)
        if psi > np.deg2rad(180):
            psi = (np.deg2rad(360) - psi) * (-1.0)
        
        # nomalize
        contact_mid = self.sample_map.ship_domain_cost(
            ver_mid, hor_mid, psi, self.SD, self.enclosing
        )
        normalized_contact_cost_mid = (contact_mid / Glaph.length_of_theta_list) * 100.0
        return normalized_contact_cost_mid
    
    def SD_checkpoint(self, parent_pt, current_pt, child_pt):
        """
        parent_pt, current_pt, child_pt: (ver, hor)
        戻り値: 正規化 Ship Domain コスト（%）
        """
        ver_p, hor_p = parent_pt
        ver_c, hor_c = current_pt
        ver_n, hor_n = child_pt

        v1 = np.array([hor_c - hor_p, ver_c - ver_p])
        v2 = np.array([hor_n - hor_c, ver_n - ver_c])

        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 == 0.0 or mag2 == 0.0:
            angle_rad = 0.0
            direction = 0
        else:
            cos_theta = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            cross = np.cross(v1, v2)
            direction = -1 if cross > 0 else (1 if cross < 0 else 0)  # 反時計回り: -1, 時計回り: 1

        psi = np.deg2rad(90.0) - np.arctan2(ver_c - ver_p, hor_c - hor_p)
        if psi > np.deg2rad(180.0):
            psi = (np.deg2rad(360.0) - psi) * (-1.0)

        psi = psi + 0.5 * angle_rad * direction
        psi = (psi + np.pi) % (2.0 * np.pi) - np.pi  # [-pi, pi] に正規化

        contact_cp = self.sample_map.ship_domain_cost(ver_c, hor_c, psi, self.SD, self.enclosing)
        normalized_contact_cost_cp = (contact_cp / Glaph.length_of_theta_list) * 100.0
        return normalized_contact_cost_cp
    
    def elem(self, parent_pt, current_pt, child_pt):
        """
        parent_pt, current_pt, child_pt: (ver, hor)
        戻り値: 要素コスト（100 - 出現回数）
        """
        ver_p, hor_p = parent_pt
        ver_c, hor_c = current_pt
        ver_ch, hor_ch = child_pt

        # --- 船速[kts]の推定（current→endの距離でモデル化） ---
        end_ver, end_hor = self.sample_map.end_xy[0, 0], self.sample_map.end_xy[0, 1]
        dist_ce = np.hypot(ver_c - end_ver, hor_c - end_hor)
        current_speed = (
            self.sample_map.b_ave * dist_ce ** (self.sample_map.a_ave)
            + self.sample_map.b_SD * dist_ce ** (self.sample_map.a_SD)
        )

        # --- 速度ビンの決定 ---
        if current_speed < self.speed_min:
            speed_key = self.speed_min
        elif current_speed >= self.speed_max:
            speed_key = self.speed_max
        else:
            speed_key = None
            for s0 in self.speed_bins:
                if s0 <= current_speed < s0 + self.speed_interval:
                    speed_key = s0
                    break
            if speed_key is None:
                speed_key = self.speed_max  # フォールバック

        # --- 角度（currentでの折れ角度：0~180°） ---
        v1 = np.array([hor_c - hor_p,  ver_c - ver_p],  dtype=float)
        v2 = np.array([hor_ch - hor_c, ver_ch - ver_c], dtype=float)
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)
        if m1 == 0.0 or m2 == 0.0:
            angle_deg = 0.0
        else:
            cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_theta)))

        # --- 角度ビンの決定 ---
        if angle_deg >= self.angle_max:
            angle_key = self.angle_max
        else:
            angle_key = None
            for a0 in self.angle_bins:
                if a0 <= angle_deg < a0 + self.angle_interval:
                    angle_key = a0
                    break
            if angle_key is None:
                angle_key = self.angle_max  # フォールバック

        # --- 出現頻度からコスト化 ---
        occurrences = self.new_filtered_dict[speed_key][angle_key]
        return float(100 - occurrences)
    

def sigmoid(x, a, b, c):
    return a/(b + np.exp(c*x))

# Function to round a number
def round_by_pitch(value, pitch):
    return int(np.round(value / pitch) * pitch)

def undo_conversion(reference_hor_index, reference_ver_index, end_hor_coord, end_ver_coord, indices, grid_pitch):
    idx = np.asarray(indices, dtype=float)
    ref_idx = np.array([reference_hor_index, reference_ver_index], dtype=float)
    ref_xy = np.array([end_hor_coord, end_ver_coord], dtype=float)
    return ref_xy - (ref_idx - idx) * grid_pitch


def calculate_turning_points(initial_coords, sample_map, last_pt, port):
    turning_points = []
    current_index = 0
    while current_index < len(initial_coords) - 1:
        current_point = initial_coords[current_index]
        d = np.hypot(current_point[0] - sample_map.end_xy[0, 0], current_point[1] - sample_map.end_xy[0, 1])
        current_speed = sample_map.b_ave * d ** sample_map.a_ave + sample_map.b_SD * d ** sample_map.a_SD
        if current_speed > 9.5:
            current_speed = 9.5
        minute_distance = current_speed * 1852 / 60
        sum_of_distance = 0.0
        broke = False
        for i in range(current_index, len(initial_coords) - 1):
            seg = np.hypot(initial_coords[i + 1][0] - initial_coords[i][0], initial_coords[i + 1][1] - initial_coords[i][1])
            sum_of_distance += seg
            if sum_of_distance >= minute_distance:
                distance_to_last = np.hypot(initial_coords[i + 1][0] - last_pt[0], initial_coords[i + 1][1] - last_pt[1])
                bt = port["berth_type"]
                if bt == 1:
                    if distance_to_last < 80:
                        broke = True
                        break
                    turning_points.append(initial_coords[i + 1])
                    current_index = i + 1
                    break
                elif bt == 2:
                    if distance_to_last < 120:
                        broke = True
                        break
                    turning_points.append(initial_coords[i + 1])
                    current_index = i + 1
                    break
        else:
            break
        if broke:
            break
    return turning_points


class PathPlanning():
    def __init__(self, ps):
        self.ps = ps

    def main(self):
        self.setup()
        self.init_path()
        self.CMAES()
        # result
        self.print_result(self.best_dict)
        self.show_result_fig(self.best_dict)

    def setup(self):
        self.update_planning_settings()
        os.makedirs(f"{SAVE_DIR}/{self.port}", exist_ok=True)
        self.shipdomain()
        self.prepare_plots_and_variables()
        print(f"### SET UP COMPLETE ###\n")

    def init_path(self):
        self.gen_init_path()
        self.initial_D = self.cal_sigma_for_ddCMA(self.initial_points, self.last_pt)
        self.N = len(self.initial_points.flatten())
        print(
            f"この最適化問題の次元Nは {self.N} です\n"
            "### INITIAL CHECKPOINTS AND sigma0 SETUP COMPLETED ###\n"
            "### MOVED TO THE OPTIMIZATION PROCESS ###\n"
        )

    def CMAES(self):
        w_len, w_SD, w_elem, w_dist = self.compute_cost_weights(self.initial_points)
        # cmaをインスタンス化
        ddcma = DdCma(xmean0=self.initial_points, sigma0=self.initial_D, seed=self.ps.seed)
        checker = Checker(ddcma)
        logger = Logger(ddcma)
        
        # 評価関数の呼び出し回数に関する初期設定
        NEVAL_STANDARD = ddcma.lam * 5000

        print("Start with first population size: " + str(ddcma.lam))
        print("Dimension: " + str(ddcma.N))
        print(f"NEVAL_STANDARD: {NEVAL_STANDARD}")
        print('Path optimization start\n')
        #
        total_neval = 0 # total number of f-calls(number of evaluation)
        best_dict = {}
        time_start = time.time()
        # per-restart loop
        for restart in range(self.ps.restarts):
            # init per-restart state
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

            # CMA-ES main loop
            while not is_satisfied:
                ddcma.onestep(func=self.path_evaluate)

                # best of this iteration (onestep sorts; idx[0] is best)
                best_cost = float(np.min(ddcma.arf))
                best_mean = ddcma.arx[int(ddcma.idx[0])].copy()

                # update global best in this restart
                if best_cost < best_dict[restart]["best_cost_so_far"]:
                    best_dict[restart]["best_cost_so_far"] = best_cost
                    best_dict[restart]["best_mean_sofar"] = best_mean
                    best_dict[restart]["cp_list"] = cp_list
                    best_dict[restart]["mp_list"] = mp_list
                    best_dict[restart]["psi_list_at_cp"] = psi_list_at_cp
                    best_dict[restart]["psi_list_at_mp"] = psi_list_at_mp

                # check stopping criteria
                is_satisfied, condition = checker()

                # periodic logging
                if ddcma.t % 10 == 0:
                    print(ddcma.t, ddcma.neval, best_cost, best_dict[restart]["best_cost_so_far"])
                    logger()

            # result
            logger(condition)
            elapsed = time.time() - t0
            best_dict[restart]["calculation_time"] = elapsed
            print(f"Terminated with condition: {condition}")
            print(f"Restart {restart} time: {elapsed:.2f} s")

            # save
            cp_list, mp_list, psi_list_at_cp, psi_list_at_mp = self.figure_output(
                best_dict[restart]["best_mean_sofar"],
                restart,
                initial_point_list=self.initial_points,
            )
            best_dict[restart]["cp_list"] = cp_list
            best_dict[restart]["mp_list"] = mp_list
            best_dict[restart]["psi_list_at_cp"] = psi_list_at_cp
            best_dict[restart]["psi_list_at_mp"] = psi_list_at_mp

            # prepare next restart
            total_neval += ddcma.neval
            print(f"total number of evaluate function calls: {total_neval}\n")

            # restart if below the evaluation budget
            if total_neval < NEVAL_STANDARD:
                if not self.ps.increase_popsize_on_restart:
                    popsize = ddcma.lam
                else:
                    popsize = ddcma.lam * 2
                seed *= 2  # change seed to avoid converging to the same solution
                ddcma = DdCma(xmean0=self.initial_points, sigma0=self.initial_D, lam=popsize, seed=self.seed)
                checker = Checker(ddcma)
                logger.setcma(ddcma)
                print(f"Restart with popsize: {ddcma.lam}")
            else:
                print("Path optimization completed")
                break
    
        time_end = time.time()
        cma_caltime = time_end - time_start
        print(f"""Path optimization completed in {cma_caltime:.2f} s.

        best_cost_so_far の値とその値を記録した平均の遷移:
        {'='*50}
        """)

        self.logger = logger
        self.best_dict = best_dict


    def compute_cost_weights(self, initial_pt):
        origin_pt = self.origin_pt
        last_pt = self.last_pt
        points = np.vstack([origin_pt, initial_pt, last_pt])
        #
        total_dist = 0.0                    # total_distance
        SD_cost = 0.0                  # total_contact_cost
        elem_ratio = 0.0                    # proportion_of_elements
        p2p_dist_ratio = 0.0                # proportion_of_distance_between_point_and_point
        #
        # 1.length weight
        straight_dist = ((origin_pt[0] - last_pt[0]) ** 2 + (origin_pt[1] - last_pt[1]) ** 2) ** 0.5
        seg  = np.diff(points, axis=0)
        total_dist = np.sum(np.hypot(seg[:, 0], seg[:, 1]))
        # nomalize
        w_len = total_dist / straight_dist * 100 - 100
        if w_len < 0:
            print(f"total_distance is negative value!")
            sys.exit()
        
        # 2.SD weight
        w_SD_midpoint = self.calculate_SD_cost_midpoint(origin_pt, initial_pt)
        w_SD_ccheckpoint = self.calculate_SD_cost_checkpoint(initial_pt[-2], initial_pt[-1], last_pt)
        w_SD = w_SD_midpoint + w_SD_ccheckpoint

        # 3.element weight
        w_elem = self.caluculate_elem_cost()
        # distance coefficient
        return w_len, w_SD, w_elem, w_dist
    
        
    def path_evaluate(self):
        
        pass

    def figure_output(self, best_mean, restart, initial_points):
        path_xy = best_mean
        sample_map = self.sample_map
        #
        path_coord = path_xy.reshape(-1, 2)
        path_coord = [
            (round_by_pitch(ver, self.ps.gridpitch), round_by_pitch(hor, self.ps.gridpitch))
            for ver, hor in path_coord
        ]
        path_coord = [tuple(coord) for coord in path_coord]
        path_coord_idx = [
            (int(np.argmin(np.abs(sample_map.ver_range - ver))),
            int(np.argmin(np.abs(sample_map.hor_range - hor))))
            for ver, hor in path_coord
        ]
        sample_map.path_node = path_coord_idx
        #
        sample_map.path_xy = np.empty((0,2))
        path_coord_fig = copy.deepcopy(path_coord)

        for i in range(len(sample_map.path_node)):
            sample_map.path_xy = np.append(sample_map.path_xy,
                    np.array([[sample_map.ver_range[sample_map.path_node[i][0]],
                                sample_map.hor_range[sample_map.path_node[i][1]]]]),
                    axis = 0)
        print(f"\nsample_map.path_xy:     {sample_map.path_xy}")
        # save the map
        cp_list, mp_list, psi_at_cp, psi_at_mp = sample_map.ShowMap(
            filename=f"{SAVE_DIR}/{self.port}/Path_by_CMA_{self.port['name']}_{restart}.png",
            SD=self.SD,
            initial_point_list=initial_points,
            optimized_point_list=path_coord_fig,
            SD_sw=self.ps.show_SD_on_optimized_path,
        )

        return cp_list, mp_list, psi_at_cp, psi_at_mp

    def shipdomain(self):
        port = self.port
        TARGET_DIR = f"{DIR}/../../outputs/port"
        target_csv = f"{TARGET_DIR}/detail_port_csv_{port['name']}.csv"
        target_csv_for_pyship = f"{TARGET_DIR}/detail_port_csv_{port['name']}_for_pyship.csv"
        #
        df_world = pd.read_csv(target_csv_for_pyship)
        world_polys = []
        world_polys.append(df_world[['x [m]', 'y [m]']].to_numpy())
        enclosing = pyshipsim.EnclosingPointCollisionChecker()
        enclosing.reset(world_polys)
        self.enclosing = enclosing
        print(f"Successfully imported data from csv\n")
        #
        SD = ShipDomain_proposal()
        SD.initial_setting(f"{DIR}/../../outputs/303/mirror5/fitting_parameter.csv", sigmoid)
        self.SD = SD
        print(f"Generating map from data\n")
        #
        time_start_map_generation = time.time()
        sample_map = Glaph.Map.GenerateMapFromCSV(target_csv, self.ps.gridpitch_for_Astar)
        #
        time_end_map_generation = time.time()
        map_generation_caltime = time_end_map_generation - time_start_map_generation
        print(f"Map generation is complete.\nCalculation time : {map_generation_caltime}\n")
        # Coefficients for the speed-decay approximation (orange guideline curve)
        df = pd.read_csv(f"{DIR}/../../raw_datas/tmp/GuidelineFit_debug.csv")
        sample_map.a_ave = df['a_ave'].values[0]
        sample_map.b_ave = df['b_ave'].values[0]
        sample_map.a_SD = df['a_SD'].values[0]
        sample_map.b_SD = df['b_SD'].values[0]
        #
        self.sample_map = sample_map

    def prepare_plots_and_variables(self):
        port = self.port
        sample_map = self.sample_map
        #
        if self.ps.start_end_mode == 'auto':
            sample_map.start_raw = np.array([port["start"]])
            sample_map.end_raw   = np.array([port["end"]])
        else: # manual
            sample_map.start_raw = np.array([self.start_coord])
            sample_map.end_raw   = np.array([self.end_coord])
        sample_map.start_xy = sample_map.FindNodeOfThePoint(sample_map.start_raw[0,:])
        sample_map.end_xy   = sample_map.FindNodeOfThePoint(sample_map.end_raw[0,:])
        #
        if self.ps.range_type == 1:
            ver_margin = 200 # [m]
            hor_margin = 200 # [m]

            ver_min = min((np.amin(sample_map.ver_range),sample_map.start_raw[0,0], sample_map.end_raw[0,0])) - ver_margin
            ver_max = max((np.amax(sample_map.ver_range),sample_map.start_raw[0,0], sample_map.end_raw[0,0])) + ver_margin
            hor_min = min((np.amin(sample_map.hor_range),sample_map.start_raw[0,1], sample_map.end_raw[0,1])) - hor_margin
            hor_max = max((np.amax(sample_map.hor_range),sample_map.start_raw[0,1], sample_map.end_raw[0,1])) + hor_margin
        else:
            ver_min = -15
            ver_max = +10
            hor_min = -20
            hor_max = +20
        ver_min_round = Glaph.Map.RoundRange(None, ver_min, sample_map.grid_pitch, 'min')
        ver_max_round = Glaph.Map.RoundRange(None, ver_max, sample_map.grid_pitch, 'max')
        hor_min_round = Glaph.Map.RoundRange(None, hor_min, sample_map.grid_pitch, 'min')
        hor_max_round = Glaph.Map.RoundRange(None, hor_max, sample_map.grid_pitch, 'max')
        sample_map.ver_range = np.arange(ver_min_round, ver_max_round+sample_map.grid_pitch/10, sample_map.grid_pitch)
        sample_map.hor_range = np.arange(hor_min_round, hor_max_round+sample_map.grid_pitch/10, sample_map.grid_pitch)
        #
        self.start_ver_idx = np.where(sample_map.ver_range == sample_map.start_xy[0, 0])
        self.start_hor_idx = np.where(sample_map.hor_range == sample_map.start_xy[0, 1])
        self.end_ver_idx   = np.where(sample_map.ver_range == sample_map.end_xy[0, 0])
        self.end_hor_idx   = np.where(sample_map.hor_range == sample_map.end_xy[0, 1])
        #
        sx, sy = sample_map.start_xy[0, 0], sample_map.start_xy[0, 1]
        ex, ey = sample_map.end_xy[0, 0],   sample_map.end_xy[0, 1]
        distance_between_start_and_end = ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5

        start_speed = min(
            sample_map.b_ave * distance_between_start_and_end ** sample_map.a_ave
            + sample_map.b_SD  * distance_between_start_and_end ** sample_map.a_SD,
            self.ps.MAX_SPEED_KTS,
        )
        print(f"start speed is {start_speed} knots.\n")
        # start
        origin_navigation_distance = start_speed * 1852 / 60
        u_start = np.array([np.cos(self.psi_start), np.sin(self.psi_start)])
        origin_pt = sample_map.start_xy[0] + origin_navigation_distance * u_start
        sample_map.origin_xy = sample_map.FindNodeOfThePoint(origin_pt)

        self.origin_pt = origin_pt
        self.origin_ver_idx = np.where(sample_map.ver_range == sample_map.origin_xy[0, 0])
        self.origin_hor_idx = np.where(sample_map.hor_range == sample_map.origin_xy[0, 1])

        # end
        straight_dist = self.ps.L * self.steady_course_coeff if self.ps.enable_pre_berthing_straight_segment else 0.0
        u_end = np.array([np.cos(self.psi_end), np.sin(self.psi_end)])
        last_pt = sample_map.end_xy[0] - straight_dist * u_end
        sample_map.last_xy = sample_map.FindNodeOfThePoint(last_pt)

        self.last_pt = last_pt
        self.last_ver_idx = np.where(sample_map.ver_range == sample_map.last_xy[0, 0])
        self.last_hor_idx = np.where(sample_map.hor_range == sample_map.last_xy[0, 1])


    def gen_init_path(self):
        if self.ps.init_path_algo == 'astar':
            print(f'Initial Path generation starts')
            time_start_astar = time.time()
            sample_map = self.sample_map
            #
            Glaph.Map.SetMaze(sample_map)
            weight = sample_map.grid_pitch * self.weight_of_SD
            sample_map.path_node, sample_map.psi, astar_iteration = Astar.astar(
                sample_map, 
                (self.origin_hor_idx[0][0], self.origin_ver_idx[0][0]),
                (self.last_hor_idx[0][0], self.last_ver_idx[0][0]),
                psi_start = self.psi_start,
                psi_end = self.psi_end,
                SD = self.SD,
                weight = weight,
                enclosing_checker = self.enclosing
            )
            #
            time_end_astar = time.time()
            astar_caltime = time_end_astar - time_start_astar
            print(f'Astar algorithm took {astar_caltime}[s]\n')

            original_initial_coord = undo_conversion(
                self.end_hor_idx[0][0],
                self.end_ver_idx[0][0],
                sample_map.end_xy[0, 1],
                sample_map.end_xy[0, 0],
                sample_map.path_node,
                self.ps.gridpitch,
            )
            original_initial_coord = original_initial_coord[:, ::-1]

            initial_points = calculate_turning_points(
                original_initial_coord,
                sample_map,
                self.last_pt,
                self.port
            )
            for i, (x, y) in enumerate(initial_points, 1):
                print(f"  P{i:02d}: ({x:.1f}, {y:.1f})")
            # save
            if self.ps.save_init_path:
                self.save_init_path(sample_map, initial_points)
            #
            initial_points = np.array(initial_points)
            self.initial_points = initial_points
        else: # manual
            for i, (x, y) in enumerate(self.initial_points, 1):
                print(f"  P{i:02d}: ({x:.1f}, {y:.1f})")
    
    def save_init_path(self, sample_map, initial_points):
        sample_map.path_xy = np.empty((0, 2))
        for i in range(len(sample_map.path_node)):
            sample_map.path_xy = np.append(
                sample_map.path_xy,
                np.array(
                    [
                        [
                            sample_map.ver_range[sample_map.path_node[i][1]],
                            sample_map.hor_range[sample_map.path_node[i][0]],
                        ]
                    ]
                ),
                axis=0,
            )
        sample_map.ShowMap_for_astar(
            filename=self.filename_astar,
            SD=self.SD,
            SD_sw=self.ps.show_SD_on_init_path,
            initial_point_list=initial_points,
        )

    def cal_sigma_for_ddCMA(
            self,
            points: np.ndarray,
            last_point: Tuple[float, float], 
            *, min_sigma: float = 5.0, scale: float = 0.5
    ) -> np.ndarray:
        ver = points[:, 0]
        hor = points[:, 1]
        # 最終点は last_point との距離で補う（np.diff の append を利用）
        ver_diffs = np.abs(np.diff(ver, append=last_point[0])) * scale
        hor_diffs = np.abs(np.diff(hor, append=last_point[1])) * scale
        # 直線経路で 0 が出ないように下限を設ける
        ver_diffs = np.maximum(ver_diffs, min_sigma)
        hor_diffs = np.maximum(hor_diffs, min_sigma)

        return np.column_stack((ver_diffs, hor_diffs)).ravel()

    def update_planning_settings(self):
        port = self.dict_of_port(self.ps.port_number)
        self.port = port

        #
        if self.ps.start_end_mode == 'auto':
            self.weight_of_SD = 20
            print("\nstartとendの座標はデフォルト値です")
        else: # Manual
            self.weight_of_SD = 20
            self.start_coord = [-600.0, -400.0]
            self.end_coord = [0.0, 0.0]
            print(
                f"startの座標は{self.start_coord}です",
                f"endの座標は{self.end_coord}です",
                sep="\n",
            )
        #
        if self.ps.psi_mode == 'auto':
            self.psi_start = np.deg2rad(port["psi_start"])
            self.psi_end = np.deg2rad(port["psi_end"])
            print("\npsi_startとpsi_endの値はデフォルト値です\n")
        else: # manual
            self.psi_start = np.deg2rad(-20)
            self.psi_end = np.deg2rad(10)
            print(
                f"psi_startの値は{self.psi_start}です",
                f"psi_endの値は{self.psi_end}です",
                sep="\n"
            )
        #
        if self.ps.steady_course_coeff_mode == 'auto':
            self.steady_course_coeff = 1.2
            print(f"保針区間の長さを決める係数はデフォルト値です\n")
        else: # Manual
            self.steady_course_coeff = 0
            print(f"保針区間の長さを決める係数は{self.steady_course_coeff}です\n")
        #
        if self.ps.init_path_algo == 'astar':
            print(
                f"Astarアルゴリズムによって初期経路が探索され、その後、初期チェックポイントが与えられます",
                f"探索におけるShip Domainの重み係数は sample_map.grid_pitch * {self.weight_of_SD} です",
                sep="\n"
            )
            if self.ps.save_init_path:
                if self.ps.show_SD_on_init_path:
                    self.filename_astar = f"{SAVE_DIR}/Initial_Path_by_Astar_with_SD.png"
                    print(f"初期経路の図は Ship Domain の表示'有り'で {self.filename_astar} に保存されます\n")
                else:
                    self.filename_astar = f"{SAVE_DIR}/Initial_Path_by_Astar_without_SD.png"
                    print(f"初期経路の図は Ship Domain の表示'無し'で {self.filename_astar} に保存されます\n")
            else:
                print("初期経路は図に保存されません\n")
        else: # Manual
            init_check_points = [
                (1990.0, -300.0),
                (1750.0, -400.0),
                (1550.0, -500.0),
                (1350.0, -550.0),
                (1200.0, -600.0),
                (1050.0, -600.0),
                (960.0,  -500.0),
                (880.0,  -400.0),
                (790.0,  -300.0),
                (710.0,  -250.0),
                (650.0,  -200.0),
                (600.0,  -150.0),
            ]
            self.initial_points = np.asarray(init_check_points, dtype=float)
            print(f"初期チェックポイントは手動で設定されました")
            for i, (x, y) in enumerate(self.initial_points, 1):
                print(f"  P{i:02d}: ({x:.1f}, {y:.1f})")
        #
        print(
            f"最適化における各コストの重み係数は、初期のコスト比が以下になるように調整されます\n"
            f"{'項目':<12}{'比率'}\n"
            f"{'-'*25}\n"
            f"{'Length':<12}{self.ps.length_ratio}\n"
            f"{'SD':<12}{self.ps.SD_ratio}\n"
            f"{'Element':<12}{self.ps.element_ratio}\n"
            f"{'Distance':<12}{self.ps.distance_ratio}\n"
        )
        #
        if self.ps.save_opt_path:
            print("最適化の詳細なデータはcsvファイルに保存されます\n")
        else:
            print("最適化の詳細なデータはcsvファイルに保存されません\n")

    def dict_of_port(self, num):
        dictionary_of_port = {
            0: {
                "name": "Osaka_port1A",
                "bay": sakai_bay.port1A,
                "start": [-1400.0, -800.0], # [-1400.0, -700.0]
                "end": [0.0, -10.0],
                "psi_start": 40,  # psi of start（degree）
                "psi_end": 10,    # psi of end（degree）
                "berth_type": 2, # 着桟姿勢（1：出船、2：入船）
                "ver_range": [-1500, 500], # [-2500, 500]
                "hor_range": [-1000, 500] # [-2000, 1000]
            },
            1: {
                "name": "Tokyo_port2C",
                "bay": Tokyo_bay.port2C,
                "start": [-1400.0, -1100.0],
                "end": [0.0, 0.0],
                "psi_start": 45,
                "psi_end": 10,
                "berth_type": 2,
                "ver_range": [-2500, 500],
                "hor_range": [-2500, 500]
            },
            2: {
                "name": "Yokkaichi_port2B",
                "bay": yokkaichi_bay.port2B,
                "start": [2050.0, 2000.0], # [2200.0, 2000.0]
                "end": [200.0, 100.0],
                "psi_start": -125, # -140
                "psi_end": 175,
                "berth_type": 1,
                "ver_range": [-500, 2500],
                "hor_range": [-500, 2500]
            },
            3: {
                "name": "Else_port1",
                "bay": else_bay.port1,
                "start": [2500.0, 0.0],
                "end": [350.0, 20.0],
                "psi_start": -145,
                "psi_end": 160,
                "berth_type": 1,
                "ver_range": [0, 3000],
                "hor_range": [-1000, 2000]
            },
            4: {
                "name": "Osaka_port1B",
                "bay": sakai_bay.port1B,
                "start": [-1400.0, -800.0], # [-1400.0, -700.0]
                "end": [0.0, 15.0],
                "psi_start": 40,  # psi of start（degree）
                "psi_end": -10,    # psi of end（degree）
                "berth_type": 2, # 着桟姿勢（1：出船、2：入船）
                "ver_range": [-1500, 500], # [-2500, 500]
                "hor_range": [-1000, 500] # [-2000, 1000]
            }
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
        print("\n"+"=" * 50+"\n")
        smallest_evaluation_key = min(best_dict, key=lambda k: best_dict[k]["best_cost_so_far"])
        print(f"最も評価値が小さかった試行は {smallest_evaluation_key} 番目\n")
        print(f"最小評価値: {best_dict[smallest_evaluation_key]['best_cost_so_far']}\n")
        print(f"  対応する最適解 (ver, hor):")
        best_solution = best_dict[smallest_evaluation_key]['best_mean_sofar']
        for i in range(0, len(best_solution), 2):
            print(f"({best_solution[i]:.6f}, {best_solution[i+1]:.6f})")
        
        self.smallest_evaluation_key = smallest_evaluation_key
    
    def show_result_fig(self, best_dict):
        """
        Contents of Figure
        ------------------
        fmin  xmean  D
        S     sigma  beta
        ------------------
        fmin : Minimum evaluation value
        xmean : Mean of x (x is an array containing veriables)
        D : Deviation
        S : Standat deviation
        sigma : Parameters for the scale of the search range of the optimization
        beta : Parameters that control the frequency and intensity of matrix updates
        """
        #
        port        = self.port
        points      = self.initial_points
        SD          = self.SD
        sample_map  = self.sample_map
        logger      = self.logger
        best_key    = self.smallest_evaluation_key
        folder_path = f"{SAVE_DIR}/{port['name']}"

        # logger plot
        fig, axdict = logger.plot()
        for ax_name, ax in axdict.items():
            if ax_name != "xmean":
                ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(f"{folder_path}/Result_of_CMA_{port['name']}.png")

        # multiplot
        if self.ps.enable_multiplot:
            pointsize = 2
            fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
            gs = gridspec.GridSpec(4, 3, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0:2])

            # map
            df_map = pd.read_csv(f"output/real_port_csv/for_thesis_{port['name']}.csv")
            map_X, map_Y = df_map["x [m]"].values, df_map["y [m]"].values
            ax1.fill_betweenx(map_X, map_Y, facecolor="gray", alpha=0.3)
            ax1.plot(map_Y, map_X, color="k", linestyle="--", lw=0.5, alpha=0.8)

            # captain's routes
            for f in glob.glob(f'tmp/_{port["name"]}/*.csv'):
                rt = RealTraj()
                rt.input_csv(f, f"{TMP_DIR}/coordinates_of_port/{port['bay'].name}.csv")
                ax1.plot(rt.Y, rt.X, color="gray", ls="-", marker="D", markersize=2, alpha=0.8, lw=1.0, zorder=1)
            legend_captain = plt.Line2D([0], [0], linestyle="-", marker="D", markersize=2, color="gray", alpha=0.8, lw=1.0, label="Captain's Route")

            # key points
            start_point  = (sample_map.start_xy[0, 0],  sample_map.start_xy[0, 1])
            end_point    = (sample_map.end_xy[0, 0],    sample_map.end_xy[0, 1])
            origin_point = (sample_map.origin_xy[0, 0], sample_map.origin_xy[0, 1])
            last_point   = (sample_map.last_xy[0, 0],   sample_map.last_xy[0, 1])

            # initial points
            pts = np.asarray(points, dtype=float)
            ax1.scatter(pts[:, 1], pts[:, 0], color="#03AF7A", s=20, zorder=4)
            legend_initial = plt.Line2D([0], [0], marker="o", color="w",
                                        markerfacecolor="#03AF7A", markersize=pointsize,
                                        label="Initial Point")

            full_initial_path = np.vstack([
                np.asarray(start_point,  float),
                np.asarray(origin_point, float),
                pts,
                np.asarray(last_point,   float),
                np.asarray(end_point,    float),
            ])
            ax1.plot(full_initial_path[:, 1], full_initial_path[:, 0],
                    color="#03AF7A", linestyle="-", linewidth=1.5, alpha=0.8, zorder=2)

            # optimized path + SD
            cp_list       = best_dict[best_key]["cp_list"]
            mp_list       = best_dict[best_key]["mp_list"]
            psi_at_cp     = best_dict[best_key]["psi_list_at_cp"]
            psi_at_mp     = best_dict[best_key]["psi_list_at_mp"]

            cp_hor = [h for v, h in cp_list]
            cp_ver = [v for v, h in cp_list]
            ax1.scatter(cp_hor, cp_ver, color="#005AFF", marker="o", s=25, zorder=4)
            legend_way = plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor="#005AFF", markersize=pointsize,
                                    label="Optimized Point")

            theta = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(10))
            theta_closed = np.append(theta, theta[0])

            # SD at CP
            for (v, h), psi in zip(cp_list, psi_at_cp):
                dist  = np.hypot(h - sample_map.end_xy[0, 1], v - sample_map.end_xy[0, 0])
                speed = sample_map.b_ave * dist**sample_map.a_ave + sample_map.b_SD * dist**sample_map.a_SD
                r     = np.array([SD.distance(speed, t) for t in theta] + [SD.distance(speed, theta[0])])
                ax1.plot(h + r * np.sin(theta_closed + psi),
                        v + r * np.cos(theta_closed + psi),
                        lw=0.6, color="#005AFF", ls="--", zorder=3)
            legend_SD = plt.Line2D([0], [0], linestyle="--", color="#005AFF", lw=1.0, label="Ship Domain")

            # SD at MP
            for (v, h), psi in zip(mp_list, psi_at_mp):
                dist  = np.hypot(h - sample_map.end_xy[0, 1], v - sample_map.end_xy[0, 0])
                speed = sample_map.b_ave * dist**sample_map.a_ave + sample_map.b_SD * dist**sample_map.a_SD
                r     = np.array([SD.distance(speed, t) for t in theta] + [SD.distance(speed, theta[0])])
                ax1.plot(h + r * np.sin(theta_closed + psi),
                        v + r * np.cos(theta_closed + psi),
                        lw=0.8, color="#005AFF", ls="--")

            # optimized path polyline
            path_points = [start_point, origin_point] + [(v, h) for v, h in cp_list] + [last_point, end_point]
            path_points = np.asarray(path_points, float)
            ax1.plot(path_points[:, 1], path_points[:, 0],
                    color="#005AFF", linestyle="-", linewidth=2.5, alpha=0.8, zorder=3)

            # start/end/origin/last
            ax1.scatter(sample_map.start_xy[0, 1], sample_map.start_xy[0, 0], color="k", s=20, zorder=4)
            ax1.text(sample_map.start_xy[0, 1], sample_map.start_xy[0, 0] + (60 * start_positive_minus),
                    "start", va="center", ha="right", fontsize=20)
            ax1.scatter(sample_map.end_xy[0, 1], sample_map.end_xy[0, 0], color="k", s=20, zorder=4)
            ax1.text(sample_map.end_xy[0, 1], sample_map.end_xy[0, 0] + (60 * end_positive_minus),
                    "end", va="center", ha="left", fontsize=20)
            ax1.scatter(sample_map.origin_xy[0, 1], sample_map.origin_xy[0, 0], color="#FF4B00", s=20, zorder=4)
            ax1.scatter(sample_map.last_xy[0, 1],   sample_map.last_xy[0, 0],   color="#FF4B00", s=20, zorder=4)
            legend_fixed = plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor="#FF4B00", markersize=pointsize,
                                    label="Fixed Point")

            # axes/ticks
            hor_lim = [port["hor_range"][0], port["hor_range"][1]]
            ver_lim = [port["ver_range"][0], port["ver_range"][1]]
            ax1.set_xlim(*hor_lim)
            ax1.set_ylim(*ver_lim)

            tick_int = 500
            x_start = int(np.floor(hor_lim[0] / tick_int) * tick_int)
            x_end   = int(np.ceil(hor_lim[1] / tick_int) * tick_int)
            y_start = int(np.floor(ver_lim[0] / tick_int) * tick_int)
            y_end   = int(np.ceil(ver_lim[1] / tick_int) * tick_int)
            ax1.set_xticks(np.arange(x_start, x_end + tick_int, tick_int))
            ax1.set_yticks(np.arange(y_start, y_end + tick_int, tick_int))
            ax1.set_xticklabels(np.arange(x_start, x_end + tick_int, tick_int).astype(int), rotation=90)
            ax1.set_yticklabels(np.arange(y_start, y_end + tick_int, tick_int).astype(int))

            ax1.set_aspect("equal")
            ax1.grid()
            ax1.set_xlabel(r"$Y\,\rm{[m]}$")
            ax1.set_ylabel(r"$X\,\rm{[m]}$")
            ax1.legend(handles=[legend_initial, legend_way, legend_fixed, legend_captain, legend_SD])

            plt.tight_layout()
            fig.savefig(f"{folder_path}/Multiplot_{port['name']}.png", bbox_inches="tight", pad_inches=0.05)
            plt.close()


if __name__ == "__main__":
    ps = Settings()
    cal = CostCalculator()
    path_planning = PathPlanning(ps, cal)
    #
    path_planning.main()
    #
    print("\nDone\n")