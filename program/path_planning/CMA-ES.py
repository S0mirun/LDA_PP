"""
CMA-ES path optimization (A* init → element-based turning points → CMA-ES)
- Max speed: 9.5 knots
- Ship Domain at segment midpoint
- Angle convention: vertical(X) = 0 deg, clockwise positive
- Coordinate note: vertical = X (ver), horizontal = Y (hor)
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- External modules ----
import utils.PP.Astar_for_CMAES
import utils.PP.graph
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.E_MakeDictionary_and_StackedBarGraph import new_filtered_dict
from utils.PP.graph import ShipDomain_proposal
from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay

# ---- Path setup for pyshipsim ----
PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
PYSIM_DIR = os.path.join(PROGRAM_DIR, "py-ship-simulator-main/py-ship-simulator-main")
if PYSIM_DIR not in sys.path:
    sys.path.append(PYSIM_DIR)
import pyshipsim  # noqa: E402


# ========== Utils ==========
def sigmoid(x: float, a: float, b: float, c: float) -> float:
    return a / (b + np.exp(c * x))

def round_by_pitch(value: float, pitch: float) -> int:
    return int(np.round(value / pitch) * pitch)


# ========== Config ==========
@dataclass
class Port:
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    psi_start_deg: float
    psi_end_deg: float
    berth_type: int  # 1: outbound, 2: inbound
    ver_range: Tuple[float, float]
    hor_range: Tuple[float, float]

class Config:
    """All knobs live here."""
    def __init__(self) -> None:
        self.today = datetime.now().strftime("%Y%m%d")

        # switches
        self.test_sw = False
        self.port_number = 2  # target port index
        self.set_start_end_sw = 0  # 0: default, 1: manual
        self.set_psi_sw = 0        # 0: default, 1: manual
        self.set_coeff_of_steady_course_sw = 0  # 0: default, 1: manual
        self.initial_path_sw = 0    # 0: A*, 1: manual
        self.initial_path_figure_sw = 1  # 0: no, 1: save
        self.initial_path_SD_sw = 0      # 0: show SD, 1: hide SD

        # A* / SD weight
        self.weight_of_SD = 20

        # CMAES ratios
        self.length_ratio = 0.1
        self.SD_ratio = 0.5
        self.element_ratio = 1.0
        self.distance_ratio = 0.2

        # CMAES control
        self.NUM_RESTART = 3
        self.restart_with_incresed_population_sw = False

        # outputs
        self.optimized_SD_sw = 0  # 0: show SD, 1: hide SD
        self.csv_sw = True
        self.Multiplot_sw = True
        self.start_positive_minus = 1
        self.end_positive_minus = -1

        # constants
        self.gridpitch = 5.0
        self.gridpitch_for_astar = 5.0
        self.Lpp = 100.0
        self.seed = 42

        # manual overrides (if used)
        self.manual_start = (-600.0, -400.0)
        self.manual_end = (0.0, 0.0)
        self.manual_psi_start = -20.0
        self.manual_psi_end = 10.0
        self.manual_coeff_steady = 0.0

        # id + folder
        self.id_number = 1
        self.folder = f"output/fig/{self.today}"
        os.makedirs(self.folder, exist_ok=True)

        # matplotlib
        plt.rcParams["font.family"] = "Times New Roman"

        # ports
        self.ports: Dict[int, Port] = {
            0: Port("Osaka_port1A", (-1400.0, -800.0), (0.0, -10.0), 40, 10, 2, (-1500, 500), (-1000, 500)),
            1: Port("Tokyo_port2C", (-1400.0, -1100.0), (0.0, 0.0), 45, 10, 2, (-2500, 500), (-2500, 500)),
            2: Port("Yokkaichi_port2B", (2050.0, 2000.0), (200.0, 100.0), -125, 175, 1, (-500, 2500), (-500, 2500)),
            3: Port("Else_port1", (2500.0, 0.0), (350.0, 20.0), -145, 160, 1, (0, 3000), (-1000, 2000)),
            4: Port("Osaka_port1B", (-1400.0, -800.0), (0.0, 15.0), 40, -10, 2, (-1500, 500), (-1000, 500)),
        }

    @property
    def port(self) -> Port:
        return self.ports[self.port_number]

    @property
    def run_dir(self) -> str:
        d = os.path.join(self.folder, f"{self.port.name}_{self.id_number}")
        os.makedirs(d, exist_ok=True)
        return d


# ========== Environment (Map, SD, terrain) ==========
class Environment:
    """Load map, SD, terrain, ranges, start/end/origin/last, speed model."""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sample_map = None  # graph_test_1211.Map
        self.SD = ShipDomain_proposal()
        self.enclosing = pyshipsim.EnclosingPointCollisionChecker()

        # file paths
        self.target_csv = f"output/port/detail_port_csv_{cfg.port.name}.csv"
        self.target_csv_for_pyship = f"output/port/detail_port_csv_{cfg.port.name}_for_pyship.csv"

        # placeholders
        self.psi_start = 0.0
        self.psi_end = 0.0
        self.origin_xy = None
        self.last_xy = None
        self.shortest_distance = 0.0

    def load(self) -> None:
        # SD params
        self.SD.initial_setting('output/303/mirror5/fitting_parameter.csv', sigmoid)

        # PyShip terrain (polygons)
        df_world = pd.read_csv(self.target_csv_for_pyship)
        world_polys = [df_world[['x [m]', 'y [m]']].to_numpy()]
        self.enclosing.reset(world_polys)

        # Map
        self.sample_map = graph_test_1211.Map.GenerateMapFromCSV(self.target_csv, float(self.cfg.gridpitch_for_astar))

        # Speed model
        df = pd.read_csv('tmp/GuidelineFit_debug.csv')
        self.sample_map.a_ave = df['a_ave'].values[0]
        self.sample_map.b_ave = df['b_ave'].values[0]
        self.sample_map.a_SD = df['a_SD'].values[0]
        self.sample_map.b_SD = df['b_SD'].values[0]

        # Start/End (raw)
        if self.cfg.set_start_end_sw == 0:
            self.sample_map.start_raw = np.array([self.cfg.port.start])
            self.sample_map.end_raw = np.array([self.cfg.port.end])
        else:
            self.sample_map.start_raw = np.array([self.cfg.manual_start])
            self.sample_map.end_raw = np.array([self.cfg.manual_end])

        # Angles (rad)
        if self.cfg.set_psi_sw == 0:
            self.psi_start = np.deg2rad(self.cfg.port.psi_start_deg)
            self.psi_end = np.deg2rad(self.cfg.port.psi_end_deg)
        else:
            self.psi_start = np.deg2rad(self.cfg.manual_psi_start)
            self.psi_end = np.deg2rad(self.cfg.manual_psi_end)

        # Node indices (rounded by pitch)
        self.sample_map.start_xy = self.sample_map.FindNodeOfThePoint(self.sample_map.start_raw[0, :])
        self.sample_map.end_xy = self.sample_map.FindNodeOfThePoint(self.sample_map.end_raw[0, :])

        # Auto ranges from start/end/obstacles + margins
        self._set_ranges_auto(ver_margin=200, hor_margin=200)

        # Origin point (1 minute forward from start by modeled speed)
        self.origin_xy = self._compute_origin()
        self.sample_map.origin_xy = self.origin_xy

        # Last point (straight-through before berthing)
        self.last_xy = self._compute_last_point()
        self.sample_map.last_xy = self.last_xy

        # Shortest straight distance between origin and last
        self.shortest_distance = float(np.hypot(self.last_xy[0, 0] - self.origin_xy[0, 0],
                                                self.last_xy[0, 1] - self.origin_xy[0, 1]))

    def _set_ranges_auto(self, ver_margin: float, hor_margin: float) -> None:
        ver_min = min((np.amin(self.sample_map.ver_range), self.sample_map.start_raw[0, 0], self.sample_map.end_raw[0, 0])) - ver_margin
        ver_max = max((np.amax(self.sample_map.ver_range), self.sample_map.start_raw[0, 0], self.sample_map.end_raw[0, 0])) + ver_margin
        hor_min = min((np.amin(self.sample_map.hor_range), self.sample_map.start_raw[0, 1], self.sample_map.end_raw[0, 1])) - hor_margin
        hor_max = max((np.amax(self.sample_map.hor_range), self.sample_map.start_raw[0, 1], self.sample_map.end_raw[0, 1])) + hor_margin

        vr0 = graph_test_1211.Map.RoundRange(None, ver_min, self.sample_map.grid_pitch, 'min')
        vr1 = graph_test_1211.Map.RoundRange(None, ver_max, self.sample_map.grid_pitch, 'max')
        hr0 = graph_test_1211.Map.RoundRange(None, hor_min, self.sample_map.grid_pitch, 'min')
        hr1 = graph_test_1211.Map.RoundRange(None, hor_max, self.sample_map.grid_pitch, 'max')

        self.sample_map.ver_range = np.arange(vr0, vr1 + self.sample_map.grid_pitch / 10, self.sample_map.grid_pitch)
        self.sample_map.hor_range = np.arange(hr0, hr1 + self.sample_map.grid_pitch / 10, self.sample_map.grid_pitch)

    def _speed_by_distance_to_end(self, ver: float, hor: float) -> float:
        d = np.hypot(ver - self.sample_map.end_xy[0, 0], hor - self.sample_map.end_xy[0, 1])
        v = self.sample_map.b_ave * (d ** self.sample_map.a_ave) + self.sample_map.b_SD * (d ** self.sample_map.a_SD)
        return min(v, 9.5)

    def _compute_origin(self) -> np.ndarray:
        start_v, start_h = self.sample_map.start_xy[0, 0], self.sample_map.start_xy[0, 1]
        spd = self._speed_by_distance_to_end(start_v, start_h)
        d1min = spd * 1852.0 / 60.0
        ov = start_v + d1min * np.cos(self.psi_start)
        oh = start_h + d1min * np.sin(self.psi_start)
        return self.sample_map.FindNodeOfThePoint([ov, oh])

    def _compute_last_point(self) -> np.ndarray:
        if self.cfg.set_coeff_of_steady_course_sw == 0:
            straight_len = self.cfg.Lpp * 1.2
        else:
            straight_len = self.cfg.Lpp * self.cfg.manual_coeff_steady
        unit_v = np.cos(self.psi_end)
        unit_h = np.sin(self.psi_end)
        lv = self.sample_map.end_xy[0, 0] - straight_len * unit_v
        lh = self.sample_map.end_xy[0, 1] - straight_len * unit_h
        return self.sample_map.FindNodeOfThePoint([lv, lh])


# ========== Initial path (A*, turning points) ==========
class PathInitializer:
    """Run A* and derive turning points based on 1-minute distance."""
    def __init__(self, cfg: Config, env: Environment) -> None:
        self.cfg = cfg
        self.env = env

    @staticmethod
    def _undo_conversion(ref_h_idx: int, ref_v_idx: int,
                         end_h: float, end_v: float,
                         indices: np.ndarray, pitch: float) -> np.ndarray:
        idx_diff = np.array([ref_h_idx, ref_v_idx]) - indices
        dxy = idx_diff * pitch
        return np.array([end_h, end_v]) - dxy  # (hor, ver)

    def _turning_points(self, coords_vh: np.ndarray) -> List[np.ndarray]:
        """Return turning points list in (ver, hor)."""
        tp: List[np.ndarray] = []
        current_idx = 0
        berth_type = self.cfg.port.berth_type
        last_v, last_h = self.env.sample_map.last_xy[0]

        while current_idx < len(coords_vh) - 1:
            current_point = coords_vh[current_idx]  # (ver, hor)
            spd = self.env._speed_by_distance_to_end(current_point[0], current_point[1])
            minute_dist = spd * 1852.0 / 60.0

            acc = 0.0
            for i in range(current_idx, len(coords_vh) - 1):
                seg = np.hypot(coords_vh[i + 1][0] - coords_vh[i][0],
                               coords_vh[i + 1][1] - coords_vh[i][1])
                acc += seg
                if acc >= minute_dist:
                    d2last = np.hypot(coords_vh[i + 1][0] - last_v,
                                      coords_vh[i + 1][1] - last_h)
                    th = 80 if berth_type == 1 else 120
                    if d2last < th:
                        return tp  # stop adding near last
                    tp.append(coords_vh[i + 1])
                    current_idx = i + 1
                    break
            else:
                break
        return tp

    def run(self) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Return (initial_point_array, turning_points, path_node)."""
        sm = self.env.sample_map

        if self.cfg.initial_path_sw == 1:
            manual = [1990., -300., 1750., -400., 1550., -500., 1350., -550., 1200., -600.,
                      1050., -600., 960., -500., 880., -400., 790., -300., 710., -250.,
                      650., -200., 600., -150.]
            init_list = [np.array([manual[i], manual[i + 1]]) for i in range(0, len(manual), 2)]
            return np.array(init_list), init_list, np.empty((0, 2), dtype=int)

        # A*: set maze and run
        graph_test_1211.Map.SetMaze(sm)
        w = sm.grid_pitch * self.cfg.weight_of_SD
        start_h_idx = np.where(sm.hor_range == self.env.sample_map.origin_xy[0, 1])[0][0]
        start_v_idx = np.where(sm.ver_range == self.env.sample_map.origin_xy[0, 0])[0][0]
        last_h_idx = np.where(sm.hor_range == self.env.sample_map.last_xy[0, 1])[0][0]
        last_v_idx = np.where(sm.ver_range == self.env.sample_map.last_xy[0, 0])[0][0]

        t0 = time.time()
        path_node, psi_list, _ = Astar_test_1211.astar(
            sm,
            (start_h_idx, start_v_idx),
            (last_h_idx, last_v_idx),
            psi_start=self.env.psi_start,
            psi_end=self.env.psi_end,
            SD=self.env.SD,
            weight=w,
            enclosing_checker=self.env.enclosing
        )
        _ = psi_list  # unused here
        elapsed = time.time() - t0
        print(f"A* finished in {elapsed:.2f}s")

        # Back to real coords (hor, ver) then swap to (ver, hor)
        end_h, end_v = sm.end_xy[0, 1], sm.end_xy[0, 0]
        path_hv = self._undo_conversion(last_h_idx, last_v_idx, end_h, end_v, np.array(path_node), self.cfg.gridpitch)
        coords_vh = path_hv[:, ::-1]  # (ver, hor)

        # Turning points
        tp = self._turning_points(coords_vh)
        tp_arr = np.array(tp)

        # Save initial path figure if needed
        if self.cfg.initial_path_figure_sw == 1:
            sm.path_xy = np.empty((0, 2))
            for h_idx, v_idx in path_node:
                sm.path_xy = np.append(
                    sm.path_xy,
                    np.array([[sm.ver_range[v_idx], sm.hor_range[h_idx]]]),
                    axis=0
                )
            fname = f"{self.cfg.run_dir}/Initial_Path_by_Astar_{'with' if self.cfg.initial_path_SD_sw==0 else 'without'}_SD.png"
            sm.ShowMap_for_astar(filename=fname, SD=self.env.SD, SD_sw=self.cfg.initial_path_SD_sw, initial_point_list=tp)

        return tp_arr, tp, np.array(path_node)


# ========== Cost model & evaluation ==========
class CostModel:
    """All cost terms + coefficient calibration."""
    def __init__(self, cfg: Config, env: Environment, init_tp: np.ndarray) -> None:
        self.cfg = cfg
        self.env = env
        self.init_tp = init_tp  # (m, 2) in (ver, hor)

        # speed & angle bins
        self.speed_min = 1.5
        self.speed_max = 9.5
        self.speed_interval = 1.0
        self.speed_bins = np.arange(self.speed_min, self.speed_max, self.speed_interval)

        self.angle_min = 0
        self.angle_max = 60
        self.angle_interval = 5
        self.angle_bins = np.arange(self.angle_min, self.angle_max, self.angle_interval)

        # shortest distance (origin ↔ last)
        self.shortest_distance = self.env.shortest_distance

        # coefficients (calibrated later)
        self.length_coeff = 1.0
        self.SD_coeff = 1.0
        self.element_coeff = 1.0
        self.distance_coeff = 1.0

    def _element_cost_one(self, vp: float, hp: float, vc: float, hc: float, vd: float, hd: float) -> float:
        # speed key
        spd = self.env._speed_by_distance_to_end(vc, hc)
        if spd < self.speed_min:
            speed_key = self.speed_min
        elif spd >= self.speed_max:
            speed_key = self.speed_max
        else:
            speed_key = None
            for s in self.speed_bins:
                if s <= spd < s + self.speed_interval:
                    speed_key = float(s)
                    break
            if speed_key is None:
                raise RuntimeError("Speed key not found")

        # angle key
        v1 = np.array([hc - hp, vc - vp])
        v2 = np.array([hd - hc, vd - vc])
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        c = float(np.clip(c, -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(c)))
        ang_key = None
        for a in self.angle_bins:
            if a <= ang_deg < a + self.angle_interval:
                ang_key = int(a)
                break
        if ang_key is None:
            ang_key = self.angle_max

        return (100.0 - new_filtered_dict[speed_key][ang_key])

    def _distance_cost_one(self, vc: float, hc: float, vd: float, hd: float) -> float:
        spd = self.env._speed_by_distance_to_end(vc, hc)
        ideal = spd * 1852.0 / 60.0
        real = np.hypot(vd - vc, hd - hc)
        return abs(ideal - real) / ideal * 100.0

    def _sd_mid_cost(self, v1: float, h1: float, v2: float, h2: float) -> float:
        psi = np.deg2rad(90) - np.arctan2(v2 - v1, h2 - h1)
        if psi > np.deg2rad(180):
            psi = (np.deg2rad(360) - psi) * (-1)
        mv, mh = (v1 + v2) / 2.0, (h1 + h2) / 2.0
        contact_mid = self.env.sample_map.ship_domain_cost(mv, mh, psi, self.env.SD, self.env.enclosing)
        return (contact_mid / graph_test_1211.length_of_theta_list) * 100.0

    def _sd_cp_cost(self, vp: float, hp: float, vc: float, hc: float, vd: float, hd: float) -> float:
        v1 = np.array([hc - hp, vc - vp])
        v2 = np.array([hd - hc, vd - vc])
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        c = float(np.clip(c, -1.0, 1.0))
        ang_rad = float(np.arccos(c))

        cross = np.cross(v1, v2)
        direction = -1 if cross > 0 else (1 if cross < 0 else 0)

        psi = np.deg2rad(90) - np.arctan2(vc - vp, hc - hp)
        if psi > np.deg2rad(180):
            psi = (np.deg2rad(360) - psi) * (-1)
        psi = psi + (ang_rad / 2.0) * direction
        if psi > np.deg2rad(180):
            psi = (np.deg2rad(360) - psi) * (-1)

        contact_cp = self.env.sample_map.ship_domain_cost(vc, hc, psi, self.env.SD, self.env.enclosing)
        return (contact_cp / graph_test_1211.length_of_theta_list) * 100.0

    def _length_cost(self, x: np.ndarray) -> float:
        """Normalized path length (origin, checkpoints, last)."""
        col = x.shape[0] // 2
        total = 0.0
        for j in range(col - 1):
            v1, h1 = x[2 * j], x[2 * j + 1]
            v2, h2 = x[2 * (j + 1)], x[2 * (j + 1) + 1]
            total += np.hypot(v2 - v1, h2 - h1)
        ov, oh = self.env.sample_map.origin_xy[0]
        lv, lh = self.env.sample_map.last_xy[0]
        total += np.hypot(x[0] - ov, x[1] - oh)
        total += np.hypot(lv - v2, lh - h2)
        val = (total / self.shortest_distance) * 100.0 - 100.0
        if val < 0:
            raise RuntimeError("negative length cost")
        return float(val)

    def determine_coefficients(self) -> None:
        """Calibrate coefficients from initial mean."""
        x = self.init_tp.flatten()
        col = x.shape[0] // 2
        ov, oh = self.env.sample_map.origin_xy[0]
        lv, lh = self.env.sample_map.last_xy[0]

        # length
        length_cost = self._length_cost(x)

        # SD cost (cp + mp)
        sd_cost = 0.0
        sd_cost += self._sd_cp_cost(ov, oh, x[0], x[1], x[2], x[3])
        sd_cost += self._sd_cp_cost(x[-4], x[-3], x[-2], x[-1], lv, lh)
        for j in range(1, col - 1):
            sd_cost += self._sd_cp_cost(x[2 * (j - 1)], x[2 * (j - 1) + 1],
                                        x[2 * j], x[2 * j + 1],
                                        x[2 * (j + 1)], x[2 * (j + 1) + 1])
        sd_cost += self._sd_mid_cost(ov, oh, x[0], x[1])
        sd_cost += self._sd_mid_cost(x[-2], x[-1], lv, lh)
        for j in range(col - 1):
            sd_cost += self._sd_mid_cost(x[2 * j], x[2 * j + 1],
                                         x[2 * (j + 1)], x[2 * (j + 1) + 1])

        # element cost
        elem = 0.0
        elem += self._element_cost_one(self.env.sample_map.start_xy[0, 0], self.env.sample_map.start_xy[0, 1], ov, oh, x[0], x[1])
        elem += self._element_cost_one(ov, oh, x[0], x[1], x[2], x[3])
        elem += 2 * self._element_cost_one(x[-4], x[-3], x[-2], x[-1], lv, lh)
        elem += 3 * self._element_cost_one(x[-2], x[-1], lv, lh, self.env.sample_map.end_xy[0, 0], self.env.sample_map.end_xy[0, 1])
        for j in range(1, col - 1):
            elem += self._element_cost_one(x[2 * (j - 1)], x[2 * (j - 1) + 1],
                                           x[2 * j], x[2 * j + 1],
                                           x[2 * (j + 1)], x[2 * (j + 1) + 1])

        # distance cost
        dist = 0.0
        dist += self._distance_cost_one(ov, oh, x[0], x[1])
        dist += self._distance_cost_one(x[-2], x[-1], lv, lh)
        for j in range(col - 1):
            dist += self._distance_cost_one(x[2 * j], x[2 * j + 1], x[2 * (j + 1)], x[2 * (j + 1) + 1])

        # coefficients
        self.length_coeff = (elem / length_cost) * self.cfg.length_ratio
        self.SD_coeff = (elem / sd_cost) * self.cfg.SD_ratio if sd_cost > 0 else 10.0
        self.element_coeff = 1.0
        self.distance_coeff = (elem / dist) * self.cfg.distance_ratio

    def evaluate_batch(self, P: np.ndarray) -> np.ndarray:
        """Vectorized evaluation for CMAES: returns cost array (lam,)."""
        row, col = P.shape[0], P.shape[1] // 2
        costs = np.zeros(row, dtype=float)

        ov, oh = self.env.sample_map.origin_xy[0]
        lv, lh = self.env.sample_map.last_xy[0]

        for i in range(row):
            x = P[i]
            # length
            length = self._length_cost(x)

            # SD
            sd = 0.0
            sd += self._sd_cp_cost(ov, oh, x[0], x[1], x[2], x[3])
            sd += self._sd_cp_cost(x[-4], x[-3], x[-2], x[-1], lv, lh)
            for j in range(1, col - 1):
                sd += self._sd_cp_cost(x[2 * (j - 1)], x[2 * (j - 1) + 1],
                                       x[2 * j], x[2 * j + 1],
                                       x[2 * (j + 1)], x[2 * (j + 1) + 1])
            sd += self._sd_mid_cost(ov, oh, x[0], x[1])
            sd += self._sd_mid_cost(x[-2], x[-1], lv, lh)
            for j in range(col - 1):
                sd += self._sd_mid_cost(x[2 * j], x[2 * j + 1],
                                        x[2 * (j + 1)], x[2 * (j + 1) + 1])

            # element
            elem = 0.0
            elem += self._element_cost_one(self.env.sample_map.start_xy[0, 0], self.env.sample_map.start_xy[0, 1], ov, oh, x[0], x[1])
            elem += self._element_cost_one(ov, oh, x[0], x[1], x[2], x[3])
            elem += 2 * self._element_cost_one(x[-4], x[-3], x[-2], x[-1], lv, lh)
            elem += 3 * self._element_cost_one(x[-2], x[-1], lv, lh, self.env.sample_map.end_xy[0, 0], self.env.sample_map.end_xy[0, 1])
            for j in range(1, col - 1):
                elem += self._element_cost_one(x[2 * (j - 1)], x[2 * (j - 1) + 1],
                                               x[2 * j], x[2 * j + 1],
                                               x[2 * (j + 1)], x[2 * (j + 1) + 1])

            # distance
            dist = 0.0
            dist += self._distance_cost_one(ov, oh, x[0], x[1])
            dist += self._distance_cost_one(x[-2], x[-1], lv, lh)
            for j in range(col - 1):
                dist += self._distance_cost_one(x[2 * j], x[2 * j + 1], x[2 * (j + 1)], x[2 * (j + 1) + 1])

            costs[i] = (self.length_coeff * length
                        + self.SD_coeff * sd
                        + self.element_coeff * elem
                        + self.distance_coeff * dist)

            if self.cfg.test_sw:
                print(f"[test] length={length:.3f}, SD={sd:.3f}, elem={elem:.3f}, dist={dist:.3f} -> {costs[i]:.3f}")
        return costs


# ========== Figures ==========
class FigureExporter:
    """Map figure after optimization + Multiplot."""
    def __init__(self, cfg: Config, env: Environment) -> None:
        self.cfg = cfg
        self.env = env

    def map_after_cma(self, best_x: np.ndarray, restart: int, initial_points: List[np.ndarray]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[float], List[float]]:
        sm = self.env.sample_map

        # round to grid pitch
        coords = best_x.reshape(-1, 2)
        coords = [(round_by_pitch(v, self.cfg.gridpitch), round_by_pitch(h, self.cfg.gridpitch)) for v, h in coords]
        coords_idx = []
        for v, h in coords:
            vi = int(np.where(sm.ver_range == v)[0][0])
            hi = int(np.where(sm.hor_range == h)[0][0])
            coords_idx.append((vi, hi))

        sm.path_node = coords_idx
        sm.path_xy = np.empty((0, 2))
        for vi, hi in sm.path_node:
            sm.path_xy = np.append(sm.path_xy, np.array([[sm.ver_range[vi], sm.hor_range[hi]]]), axis=0)

        fname = f"{self.cfg.run_dir}/Path_by_CMA_{self.cfg.port.name}_{restart}.png"
        cp_list, mp_list, psi_cp, psi_mp = sm.ShowMap(
            filename=fname,
            SD=self.env.SD,
            initial_point_list=initial_points,
            optimized_point_list=[tuple(c) for c in coords],
            SD_sw=self.cfg.optimized_SD_sw
        )
        return cp_list, mp_list, psi_cp, psi_mp

    def multiplot(self, best_key: int, best_dict: Dict[int, Any]) -> None:
        if not self.cfg.Multiplot_sw:
            return

        from matplotlib import gridspec
        import glob
        from E_Multiplot import RealTraj

        pointsize = 2
        bay_dict = {
            0: sakai_bay.port1A,
            1: Tokyo_bay.port2C,
            2: yokkaichi_bay.port2B,
            3: else_bay.port1,
            4: sakai_bay.port1B,
        }
        target_port = bay_dict[self.cfg.port_number]

        fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
        gs = gridspec.GridSpec(4, 3, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0:2])

        # terrain
        df_map = pd.read_csv(f"output/real_port_csv/for_thesis_{self.cfg.port.name}.csv")
        X = df_map['x [m]'][:]
        Y = df_map['y [m]'][:]
        ax1.fill_betweenx(X, Y, facecolor='gray', alpha=0.3)
        ax1.plot(Y, X, color="k", linestyle="--", lw=0.5, alpha=0.8)

        # captain routes
        for file in glob.glob(f'tmp/_{self.cfg.port.name}/*.csv'):
            rt = RealTraj()
            rt.input_csv(file, 'tmp/coordinates_of_port/' + target_port.name + '.csv')
            ax1.plot(rt.Y, rt.X, color='gray', ls='-', marker='D', markersize=2, alpha=0.8, lw=1.0, zorder=1)
        legend_captain = plt.Line2D([0], [0], linestyle='-', marker='D', markersize=2, color='gray', alpha=0.8, lw=1.0, label="Captain's Route")

        # points
        start = (self.env.sample_map.start_xy[0, 0], self.env.sample_map.start_xy[0, 1])
        end = (self.env.sample_map.end_xy[0, 0], self.env.sample_map.end_xy[0, 1])
        origin = (self.env.sample_map.origin_xy[0, 0], self.env.sample_map.origin_xy[0, 1])
        last = (self.env.sample_map.last_xy[0, 0], self.env.sample_map.last_xy[0, 1])

        initial_points = best_dict[best_key]["initial_points"]
        ip = np.array(initial_points).reshape(-1, 2)
        ax1.scatter(ip[:, 1], ip[:, 0], color='#03AF7A', s=20, zorder=4)
        legend_initial = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#03AF7A', markersize=pointsize, label="Initial Point")

        full_init = [start, origin] + initial_points + [last, end]
        ih = [p[1] for p in full_init]
        iv = [p[0] for p in full_init]
        ax1.plot(ih, iv, color='#03AF7A', linestyle='-', linewidth=1.5, alpha=0.8, zorder=2)

        # optimized path + SD outlines at CP/MP
        cp_list = best_dict[best_key]["cp_list"]
        mp_list = best_dict[best_key]["mp_list"]
        psi_cp = best_dict[best_key]["psi_list_at_cp"]
        psi_mp = best_dict[best_key]["psi_list_at_mp"]

        cph = [h for v, h in cp_list]
        cpv = [v for v, h in cp_list]
        ax1.scatter(cph, cpv, color='#005AFF', marker='o', s=25, zorder=4)
        legend_way = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#005AFF', markersize=pointsize, label="Optimized Point")

        theta = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(10))
        for (v, h), psi in zip(cp_list, psi_cp):
            d = np.hypot(h - end[1], v - end[0])
            spd = self.env.sample_map.b_ave * d ** self.env.sample_map.a_ave + self.env.sample_map.b_SD * d ** self.env.sample_map.a_SD
            r = [self.env.SD.distance(spd, t) for t in theta]
            r.append(r[0])
            th_closed = np.append(theta, theta[0])
            ax1.plot(h + np.array(r) * np.sin(th_closed + psi),
                     v + np.array(r) * np.cos(th_closed + psi),
                     lw=0.6, color='#005AFF', ls='--', zorder=3)
        legend_SD = plt.Line2D([0], [0], linestyle='--', color='#005AFF', lw=1.0, label="Ship Domain")

        path = [start, origin] + cp_list + [last, end]
        ph = [p[1] for p in path]
        pv = [p[0] for p in path]
        ax1.plot(ph, pv, color='#005AFF', linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)

        ax1.scatter(self.env.sample_map.start_xy[0, 1], self.env.sample_map.start_xy[0, 0], color='k', s=20, zorder=4)
        ax1.text(self.env.sample_map.start_xy[0, 1], self.env.sample_map.start_xy[0, 0] + (60 * self.cfg.start_positive_minus), 'start', va='center', ha='right', fontsize=20)
        ax1.scatter(self.env.sample_map.end_xy[0, 1], self.env.sample_map.end_xy[0, 0], color='k', s=20, zorder=4)
        ax1.text(self.env.sample_map.end_xy[0, 1], self.env.sample_map.end_xy[0, 0] + (60 * self.cfg.end_positive_minus), 'end', va='center', ha='left', fontsize=20)

        ax1.scatter(self.env.sample_map.origin_xy[0, 1], self.env.sample_map.origin_xy[0, 0], color='#FF4B00', s=20, zorder=4)
        ax1.scatter(self.env.sample_map.last_xy[0, 1], self.env.sample_map.last_xy[0, 0], color='#FF4B00', s=20, zorder=4)
        legend_fixed = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4B00', markersize=pointsize, label="Fixed Point")

        hor_lim = list(self.cfg.port.hor_range)
        ver_lim = list(self.cfg.port.ver_range)
        ax1.set_xlim(hor_lim[0], hor_lim[1])
        ax1.set_ylim(ver_lim[0], ver_lim[1])

        tick = 500
        xs = int(np.floor(hor_lim[0] / tick) * tick)
        xe = int(np.ceil(hor_lim[1] / tick) * tick)
        ys = int(np.floor(ver_lim[0] / tick) * tick)
        ye = int(np.ceil(ver_lim[1] / tick) * tick)
        xt = np.arange(xs, xe + tick, tick)
        yt = np.arange(ys, ye + tick, tick)
        ax1.set_xticks(xt); ax1.set_xticklabels(xt.astype(int), rotation=90)
        ax1.set_yticks(yt); ax1.set_yticklabels(yt.astype(int))
        ax1.set_aspect('equal'); ax1.grid()
        ax1.set_xlabel('$Y\\,\\rm{[m]}$')
        ax1.set_ylabel('$X\\,\\rm{[m]}$')
        ax1.legend(handles=[legend_initial, legend_way, legend_fixed, legend_captain, legend_SD])

        out = f"{self.cfg.run_dir}/Multiplot_{self.cfg.port.name}.png"
        plt.tight_layout()
        fig.savefig(out, bbox_inches='tight', pad_inches=0.05)
        plt.close()


# ========== Optimizer ==========
class Optimizer:
    """CMA-ES loop, logging, restarts, figures, CSV."""
    def __init__(self, cfg: Config, env: Environment, cost: CostModel, initial_points: List[np.ndarray]) -> None:
        self.cfg = cfg
        self.env = env
        self.cost = cost
        self.initial_points = initial_points
        self.fig = FigureExporter(cfg, env)

    @staticmethod
    def _sigma0_from_points(arr: np.ndarray, last_v: float, last_h: float) -> np.ndarray:
        v = arr[:, 0]; h = arr[:, 1]
        vd = np.abs(np.diff(v)) / 2.0
        hd = np.abs(np.diff(h)) / 2.0
        vd = np.append(vd, np.abs(v[-1] - last_v) / 2.0)
        hd = np.append(hd, np.abs(h[-1] - last_h) / 2.0)
        vd[vd == 0] = 5.0; hd[hd == 0] = 5.0
        return np.column_stack((vd, hd)).flatten()

    def run(self, x0_points: np.ndarray) -> Dict[int, Any]:
        last_v, last_h = self.env.sample_map.last_xy[0]
        sigma0 = self._sigma0_from_points(x0_points, last_v, last_h)
        xmean0 = x0_points.flatten()

        ddcma = DdCma(xmean0=xmean0, sigma0=sigma0, seed=self.cfg.seed)
        checker = Checker(ddcma)
        logger = Logger(ddcma)

        NEVAL_STANDARD = ddcma.lam * 5000
        total_neval = 0
        best_dict: Dict[int, Any] = {}
        print(f"Start popsize={ddcma.lam}, dim={ddcma.N}, NEVAL_STANDARD={NEVAL_STANDARD}")

        t_start = time.time()
        for restart in range(self.cfg.NUM_RESTART):
            issatisfied = False
            fbestsofar = np.inf
            best_mean_sofar = None
            best_dict[restart] = dict(
                fbestsofar=fbestsofar, best_mean_sofar=best_mean_sofar,
                calculation_time=None, cp_list=None, mp_list=None,
                psi_list_at_cp=None, psi_list_at_mp=None,
                initial_points=[list(p) for p in self.initial_points],
            )
            t0 = time.time()

            while not issatisfied:
                ddcma.onestep(func=lambda X: self.cost.evaluate_batch(X))
                fbest = float(np.min(ddcma.arf))
                best_mean = ddcma.arx[ddcma.idx[0]]

                if fbest < fbestsofar:
                    fbestsofar = fbest
                    best_mean_sofar = best_mean
                    best_dict[restart].update(
                        fbestsofar=fbestsofar,
                        best_mean_sofar=best_mean_sofar,
                    )

                issatisfied, condition = checker()
                if ddcma.t % 10 == 0:
                    print(ddcma.t, ddcma.neval, fbest, fbestsofar)
                    logger()

            logger(condition)
            print("Terminated with:", condition)
            t1 = time.time()
            best_dict[restart]["calculation_time"] = (t1 - t0)

            # figure + cp/mp/psi export
            cp, mp, psi_cp, psi_mp = self.fig.map_after_cma(best_mean_sofar, restart, self.initial_points)
            best_dict[restart]["cp_list"] = cp
            best_dict[restart]["mp_list"] = mp
            best_dict[restart]["psi_list_at_cp"] = psi_cp
            best_dict[restart]["psi_list_at_mp"] = psi_mp

            total_neval += ddcma.neval
            print(f"total eval calls: {total_neval}")

            if total_neval < NEVAL_STANDARD:
                if not self.cfg.restart_with_incresed_population_sw:
                    pop = ddcma.lam
                    self.cfg.seed *= 2
                    ddcma = DdCma(xmean0=xmean0, sigma0=sigma0, lam=pop, seed=self.cfg.seed)
                else:
                    pop = ddcma.lam * 2
                    self.cfg.seed *= 2
                    ddcma = DdCma(xmean0=xmean0, sigma0=sigma0, lam=pop, seed=self.cfg.seed)
                checker = Checker(ddcma)
                logger.setcma(ddcma)
                print("Restart with popsize:", ddcma.lam)
            else:
                print("Path optimization completed")
                break

        elapsed = time.time() - t_start
        print(f"Path optimization completed in {elapsed:.2f}s")

        # CMA logger plot
        fig, axdict = logger.plot()
        for k in axdict:
            if k != 'xmean':
                axdict[k].set_yscale('log')
        plt.tight_layout()
        plt.savefig(f"{self.cfg.run_dir}/Result_of_CMA_{self.cfg.port.name}.png")

        # CSV
        if self.cfg.csv_sw:
            self._save_csv(best_dict, elapsed)

        # Multiplot
        best_key = min(best_dict, key=lambda k: best_dict[k]["fbestsofar"])
        self.fig.multiplot(best_key, best_dict)

        # Summary
        print("=" * 50)
        for k, v in best_dict.items():
            print(f"[Restart {k}] fbestsofar={v['fbestsofar']:.6f}, time={v['calculation_time']:.2f}s")
        print("=" * 50)
        print(f"Best trial: {best_key}")
        print("Best solution (ver, hor):")
        sol = best_dict[best_key]["best_mean_sofar"]
        for i in range(0, len(sol), 2):
            print(f"({sol[i]:.6f}, {sol[i+1]:.6f})")
        return best_dict

    def _save_csv(self, best_dict: Dict[int, Any], total_time: float) -> None:
        best_key = min(best_dict, key=lambda k: best_dict[k]["fbestsofar"])
        sol = best_dict[best_key]["best_mean_sofar"]

        header_basic = ["start_ver", "start_hor", "end_ver", "end_hor", "number_of_trials", "total_cal_time"]
        basic = [[round(self.env.sample_map.start_xy[0, 0], 2),
                  round(self.env.sample_map.start_xy[0, 1], 2),
                  round(self.env.sample_map.end_xy[0, 0], 2),
                  round(self.env.sample_map.end_xy[0, 1], 2),
                  len(best_dict),
                  round(total_time, 2)]]
        df = pd.DataFrame(basic, columns=header_basic)

        init_df = pd.DataFrame(self.initial_points, columns=["initial_point_ver", "initial_point_hor"])
        df = pd.concat([df, init_df], axis=1)

        opt_df = pd.DataFrame([sol[i:i + 2] for i in range(0, len(sol), 2)], columns=["optimal_point_ver", "optimal_point_hor"])
        df = pd.concat([df, opt_df], axis=1)

        csv_path = f"{self.cfg.run_dir}/csv_{self.cfg.port.name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")


# ========== Main ==========
def main() -> None:
    cfg = Config()
    print(f"Target port: {cfg.port.name}")
    print(f"Run folder: {cfg.run_dir}")

    env = Environment(cfg)
    print("Generating map...")
    t0 = time.time()
    env.load()
    print(f"Map ready in {time.time() - t0:.2f}s")

    # Initial path and turning points
    initializer = PathInitializer(cfg, env)
    tp_arr, tp_list, _ = initializer.run()
    print("Initial turning points:")
    for p in tp_arr:
        print(p)

    # Cost model and coefficients
    cost = CostModel(cfg, env, tp_arr)
    cost.determine_coefficients()

    # Optimize
    opt = Optimizer(cfg, env, cost, tp_list)
    best = opt.run(tp_arr)

    print("**** ALL TASKS COMPLETED ****")


if __name__ == "__main__":
    main()
