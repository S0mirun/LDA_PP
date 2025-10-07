"""
CMA-ES path optimization (A* init → element-based turning points → CMA-ES)
- Max speed: 9.5 knots
- Ship Domain at segment midpoint
- Angle convention: vertical(X) = 0 deg, clockwise positive
- Coordinate note: vertical = X (ver), horizontal = Y (hor)
"""

from __future__ import annotations
from enum import StrEnum
import os
import sys
import time
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.PP.Astar_for_CMAES
import utils.PP.graph_by_taneichi
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.MakeDictionary_and_StackedBarGraph import new_filtered_dict
from utils.PP.graph_by_taneichi import ShipDomain_proposal
from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
PYSIM_DIR = os.path.join(PROGRAM_DIR, "py-ship-simulator-main/py-ship-simulator-main")
if PYSIM_DIR not in sys.path:
    sys.path.append(PYSIM_DIR)
import pyshipsim 

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)


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

        # CMA-ES / parameter modes
        self.seed: float = 42
        self.start_end_mode: ParamMode = ParamMode.AUTO
        self.psi_mode: ParamMode = ParamMode.AUTO
        self.steady_course_coeff_mode: ParamMode = ParamMode.AUTO
        self.init_path_algo: InitPathAlgo = InitPathAlgo.ASTAR
        self.MAX_SPEED_KTS = 9.5 # [knots]
        self.length_ratio: float = 0.1
        self.SD_ratio: float = 0.5
        self.element_ratio: float = 1.0
        self.distance_ratio: float = 0.2
        self.restarts: int = 3
        self.increase_popsize_on_restart: bool = False
        self.enable_pre_berthing_straight_segment: bool = True

        # figure
        self.save_init_path: bool = True
        self.show_SD_on_init_path: bool = True
        self.show_SD_on_optimized_path: bool = True
        self.enable_multiplot: bool = True
        self.gridpitch: float = 5.0 #[m]
        self.gridpitch_for_Astar: float = 5.0 #[m]
        self.range_type = 1
        # csv
        self.save_opt_path: bool = True

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
        self.gen_init_path()
        self.PP()

    def setup(self):
        self.update_planning_settings()
        self.shipdomain()
        self.prepare_plots_and_variables()

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
        sample_map = utils.PP.graph_by_taneichi.Map.GenerateMapFromCSV(target_csv, self.ps.gridpitch_for_Astar)
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
        self.port = port
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
        ver_min_round = utils.PP.graph_by_taneichi.Map.RoundRange(None, ver_min, sample_map.grid_pitch, 'min')
        ver_max_round = utils.PP.graph_by_taneichi.Map.RoundRange(None, ver_max, sample_map.grid_pitch, 'max')
        hor_min_round = utils.PP.graph_by_taneichi.Map.RoundRange(None, hor_min, sample_map.grid_pitch, 'min')
        hor_max_round = utils.PP.graph_by_taneichi.Map.RoundRange(None, hor_max, sample_map.grid_pitch, 'max')
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

        print(f"### SET UP COMPLETE ###\n")

    def gen_init_path(self):
        if self.ps.init_path_algo == 'astar':
            print(f'Initial Path generation starts')
            time_start_astar = time.time()
            sample_map = self.sample_map
            #
            utils.PP.graph_by_taneichi.Map.SetMaze(sample_map)
            weight = sample_map.grid_pitch * self.weight_of_SD
            sample_map.path_node, sample_map.psi, astar_iteration = utils.PP.Astar_for_CMAES.astar(
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

            initial_point_list = calculate_turning_points(
                original_initial_coord,
                sample_map,
                self.last_pt,
                self.port
            )
            for i, (x, y) in enumerate(initial_point_list, 1):
                print(f"  P{i:02d}: ({x:.1f}, {y:.1f})")
            # save
            if self.ps.save_init_path:
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
                    initial_point_list=initial_point_list,
                )
        else:
            for i, (x, y) in enumerate(self.initial_points, 1):
                print(f"  P{i:02d}: ({x:.1f}, {y:.1f})")


    def PP(self):
        pass

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

if __name__ == "__main__":
    ps = Settings()
    path_planning = PathPlanning(ps)
    #
    path_planning.main()
    #
    print("\nDone\n")