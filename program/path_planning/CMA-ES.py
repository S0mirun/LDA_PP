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
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.PP.Astar_for_CMAES
import utils.PP.graph
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.MakeDictionary_and_StackedBarGraph import new_filtered_dict
from utils.PP.graph import ShipDomain_proposal
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
        self.length_ratio: float = 0.1
        self.SD_ratio: float = 0.5
        self.element_ratio: float = 1.0
        self.distance_ratio: float = 0.2
        self.restarts: int = 3
        self.increase_popsize_on_restart: bool = False  # True: restarts use increased population size (IPOP); False: keep same size
        # figure
        self.save_init_path: bool = True
        self.show_SD_on_init_path: bool = True
        self.show_SD_on_optimized_path: bool = True
        self.enable_multiplot: bool = True
        self.gridpitch: float = 5.0 #[m]
        self.gridpitch_for_Astar: float = 5.0 #[m]
        # csv
        self.save_opt_path: bool = True

def sigmoid(x, a, b, c):
    return a/(b + np.exp(c*x))

# Function to round a number
def round_by_pitch(value, pitch):
    return int(np.round(value / pitch) * pitch)

class PathPlanning():
    def __init__(self, ps):
        self.ps = ps

    def main(self):
        self.setup()
        self.PP()

    def setup(self):
        self.update_planning_settings()
        self.shipdomain()
        self.prepare_plots_and_variables()

    def shipdomain(self):
        port = self.port
        TARGET_DIR = "../../outputs/port"
        target_csv = f"{TARGET_DIR}/detail_port_csv_{port['name']}.csv"
        target_csv_for_pyship = f"{TARGET_DIR}/detail_port_csv_{port['name']}_for_pyship.csv"
        #
        df_world = pd.read_csv(target_csv_for_pyship)
        world_polys = []
        world_polys.append(df_world[['x [m]', 'y [m]']].to_numpy())
        enclosing = pyshipsim.EnclosingPointCollisionChecker()
        enclosing.reset(world_polys)
        print(f"Successfully imported data from csv\n")
        #
        SD = ShipDomain_proposal()
        SD.initial_setting('../../output/303/mirror5/fitting_parameter.csv', sigmoid)
        print(f"Generating map from data\n")
        #
        time_start_map_generation = time.time()
        sample_map = utils.PP.graph.Map.GenerateMapFromCSV(target_csv, self.gridpitch_for_astar)
        time_end_map_generation = time.time()
        map_generation_caltime = time_end_map_generation - time_start_map_generation
        print(f"Map generation is complete.\nCalculation time : {map_generation_caltime}\n")
        # Coefficients for the speed-decay approximation (orange guideline curve)
        df = pd.read_csv('../../raw_datas/tmp/GuidelineFit_debug.csv')
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

        if set_start_end_sw == 0: # defalt
            sample_map.start_raw = np.array([port["start"]])
            sample_map.end_raw   = np.array([port["end"]])
        else: # manual
            sample_map.start_raw = np.array([manual_start_coord])
            sample_map.end_raw   = np.array([manual_end_coord])

    def PP(self):
        pass

    def update_planning_settings(self):
        self.port = self.dict_of_port(self.ps.port_number)
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
            print("\npsi_startとpsi_endの値はデフォルト値です\n")
        else: # manual
            self.psi_start = -20
            self.psi_end = 10
            print(
                f"psi_startの値は{self.psi_start}です",
                f"psi_endの値は{self.psi_end}です",
                sep="\n"
            )
        #
        if self.ps.steady_course_coeff_mode == 'auto':
            print(f"保針区間の長さを決める係数はデフォルト値です\n")
        else:
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
                    filename_astar = f"{SAVE_DIR}/Initial_Path_by_Astar_with_SD.png"
                    print(f"初期経路の図は Ship Domain の表示'有り'で {filename_astar} に保存されます\n")
                else:
                    filename_astar = f"{SAVE_DIR}/Initial_Path_by_Astar_without_SD.png"
                    print(f"初期経路の図は Ship Domain の表示'無し'で {filename_astar} に保存されます\n")
            else:
                print("初期経路は図に保存されません\n")
        else:
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