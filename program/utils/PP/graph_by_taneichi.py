# 新しい接触判定を作るためのファイル
import os
import sys

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.PP.subroutine import mpl_config

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
PYSIM_DIR = os.path.join(PROGRAM_DIR, "py-ship-simulator-main/py-ship-simulator-main")
if PYSIM_DIR not in sys.path:
    sys.path.append(PYSIM_DIR)
import pyshipsim
plt.rcParams.update(mpl_config)

# Angle list to approximate ellipse polygon (0–360 deg, step=10 deg)
theta_list = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(10))
length_of_theta_list = len(theta_list)  # 36


class Map():
    """Grid map with obstacles and utilities for collision checking and plotting."""

    def __init__(self, ver_range, hor_range, grid_pitch=None, start=None, end=None, Miss_Corner=False):
        """Initialize map ranges and basic state."""
        # inputs
        self.start = start
        self.end = end
        self.grid_pitch = grid_pitch
        self.ver_range = ver_range
        self.hor_range = hor_range
        self.Miss_Corner = Miss_Corner

        # obstacle containers
        self.obstacle_dict = {}
        self.obstacle_node = np.empty((0, 2))

    def GenerateMapFromCSV(file, grid_pitch):
        """
        Build a Map from a CSV of obstacle polyline.
        Args:
            file (str): CSV path with columns 'x [m]', 'y [m]'.
            grid_pitch (float): Grid spacing.
        Returns:
            Map: Initialized map with obstacle line & nodes added.
        """

        with tqdm(total=13, desc="Build target map", unit="step") as pbar:
            df = pd.read_csv(file); pbar.update()
            ver = np.array(df['x [m]']); pbar.update()
            hor = np.array(df['y [m]']); pbar.update()
            obstacles = np.stack([ver, hor], 1); pbar.update()

            ver_min_round = Map.RoundRange(None, np.amin(ver), grid_pitch, 'min'); pbar.update()
            ver_max_round = Map.RoundRange(None, np.amax(ver), grid_pitch, 'max'); pbar.update()
            hor_min_round = Map.RoundRange(None, np.amin(hor), grid_pitch, 'min'); pbar.update()
            hor_max_round = Map.RoundRange(None, np.amax(hor), grid_pitch, 'max'); pbar.update()

            ver_range = np.arange(ver_min_round, ver_max_round + grid_pitch / 10, grid_pitch); pbar.update()
            hor_range = np.arange(hor_min_round, hor_max_round + grid_pitch / 10, grid_pitch); pbar.update()

            target_map = Map(ver_range, hor_range, grid_pitch, Miss_Corner=True); pbar.update()
            Map.AddObstacleLine(target_map, obstacles, 'berth'); pbar.update()
            Map.AddObstacleNode(target_map, obstacles, 'berth'); pbar.update()

        return target_map


    def ShowInitialMap(self, filename=None, SD=None, SD_sw=True, initial_point_list=None):
        """
        Plot map for initial path (start/origin/path/last/end + optional SD).
        Args:
            filename (str|None): Output image path.
            SD: Ship-domain model (has distance()).
            SD_sw (bool): Show SD polygons if True.
            initial_point_list (list|ndarray|None): Initial points (ver, hor).
        Returns:
            None
        """
        # figure sizing by aspect
        x_range = self.hor_range[-1] - self.hor_range[0]
        y_range = self.ver_range[-1] - self.ver_range[0]
        aspect_ratio = x_range / y_range if y_range != 0 else 1
        base_size = 6
        figsize_x = base_size * aspect_ratio
        figsize_y = base_size
        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=300, linewidth=0, edgecolor='w')
        ax = fig.add_subplot(111)

        # obstacle lines
        for key in self.obstacle_dict:
            ax.plot(self.obstacle_dict[key][:, 1], self.obstacle_dict[key][:, 0], color='k', ls='-', lw=0.8)

        # start/end
        if 'start_xy' in dir(self):
            ax.scatter(self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.5, color='k', s=15, edgecolors='k', zorder=3)
            ax.text(self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.6, 'start', va='bottom', ha='center', color='k', fontsize=10)
        if 'end_xy' in dir(self):
            ax.scatter(self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.5, color='k', s=15, edgecolors='k', zorder=3)
            ax.text(self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.6, 'end', va='bottom', ha='center', color='k', fontsize=10)

        # origin/last
        ax.scatter(self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5, color='purple', s=15, zorder=3, label='Fixed Points')
        ax.scatter(self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5, color='purple', s=15, zorder=3)

        # initial points
        if initial_point_list is not None:
            initial_points = np.array(initial_point_list).reshape(-1, 2)
            ax.scatter(initial_points[:, 1], initial_points[:, 0], color='blue', s=10, label='Initial Points', zorder=3)

        # path chain (start -> origin -> path -> last -> end)
        if 'path_xy' in dir(self):
            all_points = []
            all_points.append([self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.5])
            all_points.append([self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5])
            all_points.extend(self.path_xy[:, [1, 0]] + 0.5)
            all_points.append([self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5])
            all_points.append([self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.5])
            all_points = np.array(all_points)
            ax.plot(all_points[:, 0], all_points[:, 1], lw=1.0, color='r', ls='-')

        # ship-domain polygons along path
        if 'psi' in dir(self) and not SD == None and SD_sw == True:
            for j in range(len(self.path_xy)):
                if j % 30 == 0:  # draw at interval
                    distance = ((self.path_xy[j, 1] + 0.5 - self.end_xy[0, 1] + 0.5) ** 2 +
                                (self.path_xy[j, 0] + 0.5 - self.end_xy[0, 0] + 0.5) ** 2) ** (1 / 2)
                    speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
                    r_list = []
                    for theta_i in theta_list:
                        r_list.append(SD.distance(speed, theta_i))

                    # close polygon
                    r_list.append(r_list[0])
                    theta_list_closed = np.append(theta_list, theta_list[0])

                    ax.plot(self.path_xy[j, 1] + 0.5 + np.array(r_list) * np.sin(theta_list_closed + self.psi[j]),
                            self.path_xy[j, 0] + 0.5 + np.array(r_list) * np.cos(theta_list_closed + self.psi[j]),
                            lw=0.5, color='g', ls='--')

        # axes limits
        ax.set_xlim(self.hor_range[0], self.hor_range[-1] + self.grid_pitch)
        ax.set_ylim(self.ver_range[0], self.ver_range[-1] + self.grid_pitch)

        # major ticks (200 m)
        x_start = np.floor(self.hor_range[0] / 200) * 200
        x_end = np.ceil(self.hor_range[-1] / 200) * 200
        ax.set_xticks(np.arange(x_start, x_end + 200, 200))

        y_start = np.floor(self.ver_range[0] / 200) * 200
        y_end = np.ceil(self.ver_range[-1] / 200) * 200
        ax.set_yticks(np.arange(y_start, y_end + 200, 200))

        # tick labels
        ax.set_xticklabels(np.arange(x_start, x_end + 200, 200), rotation=90)
        ax.set_yticklabels(np.arange(y_start, y_end + 200, 200), rotation=0)

        ax.set_xlabel(r'$Y\,\rm{[m]}$')
        ax.set_ylabel(r'$X\,\rm{[m]}$')

        plt.grid(which='major', color='k', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')

        # legend
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles,
            labels,
            loc='lower right',
            fontsize=9,
            scatterpoints=1,
            handletextpad=0.3,
            labelspacing=0.6,
            borderpad=0.6,
            frameon=True,
            markerscale=1.2
        )
        plt.tight_layout()
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        plt.close()

    def ShowMap(self,
        filename=None,
        SD=None,
        initial_point_list=None,
        optimized_point_list=None,
        SD_sw=True,
    ):
        """
        Plot map with path and optional SD at checkpoints & midpoints.
        Args:
            filename (str|None): Output image path.
            SD: Ship-domain model (has distance()).
            initial_point_list (list|ndarray|None): Initial points (ver, hor).
            optimized_point_list (list|ndarray|None): Optimized points (ver, hor).
            SD_sw (bool): Show SD polygons if True.
        Returns:
            tuple: (cp_list, mp_list, psi_list_at_cp, psi_list_at_mp)
        """
        # figure sizing by aspect
        x_range = self.hor_range[-1] - self.hor_range[0]
        y_range = self.ver_range[-1] - self.ver_range[0]
        aspect_ratio = x_range / y_range if y_range != 0 else 1
        base_size = 6
        figsize_x = base_size * aspect_ratio
        figsize_y = base_size
        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=300, linewidth=0, edgecolor='w')
        ax = fig.add_subplot(111)

        # obstacle lines
        for key in self.obstacle_dict:
            ax.plot(self.obstacle_dict[key][:, 1], self.obstacle_dict[key][:, 0], color='k', ls='-', lw=0.8)

        # start/end
        if 'start_xy' in dir(self):
            ax.scatter(self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.5, color='k', s=15, edgecolors='k', zorder=3)
            ax.text(self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.6, 'start', va='bottom', ha='center', color='k', fontsize=10)
        if 'end_xy' in dir(self):
            ax.scatter(self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.5, color='k', s=15, edgecolors='k', zorder=3)
            ax.text(self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.6, 'end', va='bottom', ha='center', color='k', fontsize=10)

        # origin/last
        ax.scatter(self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5, color='purple', s=15, zorder=3, label='Fixed Points')
        ax.scatter(self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5, color='purple', s=15, zorder=3)

        # initial points
        if initial_point_list is not None:
            initial_points = np.array(initial_point_list).reshape(-1, 2)
            ax.scatter(initial_points[:, 1], initial_points[:, 0], color='blue', s=10, label='Initial Points', zorder=3)

        # optimized points
        if optimized_point_list is not None:
            optimized_points = np.array(optimized_point_list)
            ax.scatter(
                optimized_points[:, 1],
                optimized_points[:, 0],
                color='green', s=15, label='Optimized Points', zorder=3
            )

        # path chain (start -> origin -> path -> last -> end)
        if 'path_xy' in dir(self):
            all_points = []
            all_points.append([self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.5])
            all_points.append([self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5])
            all_points.extend(self.path_xy[:, [1, 0]] + 0.5)
            all_points.append([self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5])
            all_points.append([self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.5])
            all_points = np.array(all_points)
            ax.plot(all_points[:, 0], all_points[:, 1], lw=1.0, color='r', ls='-')

        # SD at CP & MP
        if SD_sw:
            # --- CP psi from three points ---
            def cal_psi_at_checkpoint(ver_parent, hor_parent, ver_current, hor_current, ver_child, hor_child):
                """
                Heading at checkpoint via bisected angle of two segments.
                Args:
                    *_parent/current/child (float): Coordinates (ver, hor).
                Returns:
                    float: psi [rad]
                """
                v1 = np.array([hor_current - hor_parent, ver_current - ver_parent])
                v2 = np.array([hor_child - hor_current, ver_child - ver_current])

                m1 = np.linalg.norm(v1)
                m2 = np.linalg.norm(v2)
                dot = np.dot(v1, v2)
                cos_theta = dot / (m1 * m2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_rad = np.arccos(cos_theta)

                cross = np.cross(v1, v2)
                if cross > 0:
                    direction = -1  # CCW
                elif cross < 0:
                    direction = 1   # CW
                else:
                    direction = 0

                psi = np.deg2rad(90) - np.arctan2(ver_current - ver_parent, hor_current - hor_parent)
                if psi > np.deg2rad(180):
                    psi = (np.deg2rad(360) - psi) * (-1)
                psi = psi + (angle_rad / 2) * direction
                if psi > np.deg2rad(180):
                    psi = (np.deg2rad(360) - psi) * (-1)
                return psi

            # --- MP psi from two points ---
            def cal_psi_at_midpoint(ver_parent, hor_parent, ver_current, hor_current):
                """
                Heading at midpoint along a segment.
                Args:
                    ver_parent, hor_parent, ver_current, hor_current (float)
                Returns:
                    float: psi [rad]
                """
                psi = np.deg2rad(90) - np.arctan2(ver_current - ver_parent, hor_current - hor_parent)
                if psi > np.deg2rad(180):
                    psi = (np.deg2rad(360) - psi) * (-1)
                return psi

            path_for_cal = np.vstack([self.origin_xy, self.path_xy, self.last_xy])

            # CP psi list
            psi_list_at_cp = []
            cp_list = []
            for i in range(1, len(path_for_cal) - 1):
                ver_parent, hor_parent = path_for_cal[i - 1]
                ver_current, hor_current = path_for_cal[i]
                ver_child, hor_child = path_for_cal[i + 1]
                cp_list.append((ver_current, hor_current))
                psi_at_cp = cal_psi_at_checkpoint(ver_parent, hor_parent, ver_current, hor_current, ver_child, hor_child)
                psi_list_at_cp.append(psi_at_cp)

            # draw SD at CP
            for j in range(len(self.path_xy)):
                distance = ((self.path_xy[j, 1] - self.end_xy[0, 1]) ** 2 + (self.path_xy[j, 0] - self.end_xy[0, 0]) ** 2) ** 0.5
                speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
                r_list = []
                for theta_i in theta_list:
                    r_list.append(SD.distance(speed, theta_i))
                r_list.append(r_list[0])
                theta_list_closed = np.append(theta_list, theta_list[0])
                ax.plot(self.path_xy[j, 1] + np.array(r_list) * np.sin(theta_list_closed + psi_list_at_cp[j]),
                        self.path_xy[j, 0] + np.array(r_list) * np.cos(theta_list_closed + psi_list_at_cp[j]),
                        lw=0.5, color='g', ls='--')

            # MP psi list
            psi_list_at_mp = []
            for i in range(len(path_for_cal) - 1):
                ver_parent, hor_parent = path_for_cal[i]
                ver_current, hor_current = path_for_cal[i + 1]
                psi_st_mp = cal_psi_at_midpoint(ver_parent, hor_parent, ver_current, hor_current)
                psi_list_at_mp.append(psi_st_mp)

            # MP coords and draw SD at MP
            mp_list = []
            for j in range(len(path_for_cal) - 1):
                mid_point_hor = (path_for_cal[j, 1] + path_for_cal[j + 1, 1]) / 2
                mid_point_ver = (path_for_cal[j, 0] + path_for_cal[j + 1, 0]) / 2
                mp_list.append((mid_point_ver, mid_point_hor))
                distance = ((mid_point_hor - self.end_xy[0, 1]) ** 2 + (mid_point_ver - self.end_xy[0, 0]) ** 2) ** 0.5
                speed = self.b_ave * distance ** self.a_ave + self.b_SD * distance ** self.a_SD
                r_list = [SD.distance(speed, theta_i) for theta_i in theta_list]
                r_list.append(r_list[0])
                theta_list_closed = np.append(theta_list, theta_list[0])
                ax.plot(mid_point_hor + np.array(r_list) * np.sin(theta_list_closed + psi_list_at_mp[j]),
                        mid_point_ver + np.array(r_list) * np.cos(theta_list_closed + psi_list_at_mp[j]),
                        lw=0.5, color='b', ls='--')
                
            print("Psi List (degrees):", np.degrees(psi_list_at_cp))

        # axes limits
        ax.set_xlim(self.hor_range[0], self.hor_range[-1] + self.grid_pitch)
        ax.set_ylim(self.ver_range[0], self.ver_range[-1] + self.grid_pitch)

        # major ticks (200 m)
        x_start = np.floor(self.hor_range[0] / 200) * 200
        x_end = np.ceil(self.hor_range[-1] / 200) * 200
        ax.set_xticks(np.arange(x_start, x_end + 200, 200))

        y_start = np.floor(self.ver_range[0] / 200) * 200
        y_end = np.ceil(self.ver_range[-1] / 200) * 200
        ax.set_yticks(np.arange(y_start, y_end + 200, 200))

        # tick labels
        ax.set_xticklabels(np.arange(x_start, x_end + 200, 200), rotation=90)
        ax.set_yticklabels(np.arange(y_start, y_end + 200, 200), rotation=0)

        ax.set_xlabel(r'$Y\,\rm{[m]}$')
        ax.set_ylabel(r'$X\,\rm{[m]}$')

        plt.grid(which='major', color='k', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')

        # legend
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles,
            labels,
            loc='lower right',
            fontsize=9,
            scatterpoints=1,
            handletextpad=0.3,
            labelspacing=0.6,
            borderpad=0.6,
            frameon=True,
            markerscale=1.2
        )
        plt.tight_layout()
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        # return (ver, hor) lists and psi lists
        if optimized_point_list is not None:
            return cp_list, mp_list, psi_list_at_cp, psi_list_at_mp

    def DetictCollision(self, start, end):
        """
        Find crossed grid cells by a line segment (start -> end).
        Args:
            start, end (ndarray): Points (ver, hor).
        Returns:
            ndarray: Unique passed nodes (ver, hor).
        """
        # line ax + by + c = 0 (x/y swapped: vertical is x)
        a = end[1] - start[1]
        b = start[0] - end[0]
        c = start[1] * end[0] - start[0] * end[1]

        # I: intersections with vertical grid lines (x)
        ver_list = []
        for ver_i in self.ver_range:
            if ver_i >= min(start[0], end[0]) and ver_i <= max(start[0], end[0]):
                ver_list.append(ver_i)
        ver_array = np.empty((len(ver_list), 2))
        ver_array[:, 0] = ver_list[:]
        ver_array[:, 1] = start[1] if b == 0 else -(a / b) * ver_array[:, 0] - (c / b)

        # II: intersections with horizontal grid lines (y)
        hor_list = []
        for hor_i in self.hor_range:
            if hor_i >= min(start[1], end[1]) and hor_i <= max(start[1], end[1]):
                hor_list.append(hor_i)
        hor_array = np.empty((len(hor_list), 2))
        hor_array[:, 1] = hor_list[:]
        hor_array[:, 0] = start[0] if a == 0 else -(b / a) * hor_array[:, 1] - (c / a)

        # merge intersections and add endpoints
        ans_array = np.concatenate([hor_array, ver_array])
        ans_array = np.unique(ans_array, axis=0)
        ans_array = np.block([[ans_array], [start], [end]])

        # exact grid points among intersections
        grid_point_array = np.empty((0, 2))
        for i in range(len(ans_array)):
            if ans_array[i, 0] in self.ver_range and ans_array[i, 1] in self.hor_range:
                grid_point_array = np.append(grid_point_array, np.array([[ans_array[i, 0], ans_array[i, 1]]]), axis=0)

        # midpoints between consecutive intersections
        middle_point_array = np.empty((0, 2))
        for i in range(len(ans_array) - 1):
            middle_point_array = np.append(
                middle_point_array,
                np.array([[np.mean(ans_array[i:i + 2, 0]), np.mean(ans_array[i:i + 2, 1])]]),
                axis=0
            )

        # passed nodes (anchor is lower-left)
        pass_node_array = np.empty((0, 2))
        pass_node = self.FindNodeOfThePoint(start)
        pass_node_array = np.append(pass_node_array, pass_node, axis=0)
        pass_node = self.FindNodeOfThePoint(end)
        pass_node_array = np.append(pass_node_array, pass_node, axis=0)

        # include four nodes around exact grid points if needed
        if self.Miss_Corner == False:
            for i in range(len(grid_point_array)):
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i, 0],                      grid_point_array[i, 1]                      ]]), axis=0)
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i, 0] - self.grid_pitch, grid_point_array[i, 1]                      ]]), axis=0)
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i, 0],                      grid_point_array[i, 1] - self.grid_pitch]]), axis=0)
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i, 0] - self.grid_pitch, grid_point_array[i, 1] - self.grid_pitch]]), axis=0)

        # nodes from segment midpoints
        for i in range(len(middle_point_array)):
            pass_node = self.FindNodeOfThePoint(middle_point_array[i, :])
            pass_node_array = np.append(pass_node_array, pass_node, axis=0)
        pass_node_array = np.unique(pass_node_array, axis=0)

        return pass_node_array

    def RoundRange(self, num, pitch, TYPE):
        """
        Round value to nearest grid boundary.
        Args:
            num (float): Value to round.
            pitch (float): Grid spacing.
            TYPE (str): 'min' => floor to <= num, 'max' => ceil to >= num.
        Returns:
            float: Rounded boundary value.
        """
        num_int = 0
        if TYPE == 'max':
            num_sign = 1
        elif TYPE == 'min':
            num_sign = -1
        while True:
            if abs(num_int) >= abs(num):
                break
            else:
                num_int = num_int + num_sign * pitch
        return num_int

    def AddObstacleLine(self, array, name='Noname'):
        """
        Add obstacle polyline into dict with unique key.
        Args:
            array (ndarray): Vertex array (ver, hor).
            name (str): Base key.
        Returns:
            None
        """
        name = name
        num = 0
        with tqdm(total=None, desc=f"AddObstacleLine[{name}]", unit='probe', leave=False) as pbar:
            while True:
                key = name + '-' + str(num)
                if key in self.obstacle_dict:
                    num += 1
                    pbar.update(1)
                else:
                    self.obstacle_dict[key] = array
                    pbar.update(1)
                    break

    def AddObstacleNode(self, array, name='Noname'):
        """
        Rasterize obstacle: collect all grid nodes along edges + filled interior.
        Args:
            array (ndarray): Obstacle vertices (ver, hor).
            name (str): Unused label (kept for symmetry).
        Returns:
            None
        """
        inner_node = self.fill_inner_concave_obstacle(array, self.grid_pitch)
        for i in tqdm(range(len(array) - 1), desc='Add Node'):
            tmp_array = self.DetictCollision(array[i, :], array[i + 1, :])
            self.obstacle_node = np.concatenate([self.obstacle_node, tmp_array])
        self.obstacle_node = np.append(self.obstacle_node, inner_node, axis=0)
        self.obstacle_node = np.unique(self.obstacle_node, axis=0)
        self.obstacle_map = set([tuple(x) for x in self.obstacle_node])

    def FindNodeOfThePoint(self, point):
        """
        Anchor a point to lower-left grid node.
        Args:
            point (ndarray): (ver, hor).
        Returns:
            ndarray: Node (1,2) (ver, hor).
        """
        node = np.empty((1, 2))
        node[0, 0] = self.FloorByDesiredPitch(point[0], self.grid_pitch)
        node[0, 1] = self.FloorByDesiredPitch(point[1], self.grid_pitch)
        return node

    def FloorByDesiredPitch(self, num, grid):
        """
        Floor by custom pitch.
        Args:
            num (float): Value.
            grid (float): Pitch.
        Returns:
            float: Floored value.
        """
        tmp = grid * math.floor(num / grid)
        return tmp

    def SetMaze(self):
        """
        Build binary maze array (1=obstacle) for A* (transposed to (hor, ver)).
        Returns:
            None
        """
        maze_np = np.zeros((len(self.ver_range), len(self.hor_range)), int)
        # set obstacle cells
        for i in range(len(self.obstacle_node)):
            ver_i = np.where(self.ver_range == self.obstacle_node[i, 0])
            hor_i = np.where(self.hor_range == self.obstacle_node[i, 1])
            maze_np[ver_i, hor_i] = 1
        # transpose for A* (hor, ver)
        maze_np = maze_np.T
        self.maze = maze_np.tolist()

    def ship_domain_cost_astar(self, node, SD, weight, enclosing_checker):
        """
        Cost = weight * (# of SD vertices inside obstacles) for A* node.
        Args:
            node: Node with .position and .psi.
            SD: Ship-domain model (distance()).
            weight (float): Weight factor.
            enclosing_checker: pyshipsim.EnclosingPointCollisionChecker instance.
        Returns:
            float: Cost.
        """
        distance = ((self.ver_range[node.position[1]] - self.end_xy[0, 1]) ** 2 +
                    (self.hor_range[node.position[0]] - self.end_xy[0, 1]) ** 2) ** (1 / 2)
        speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)

        # SD polygon around node center
        r_list = []
        for theta_i in theta_list:
            r_list.append(SD.distance(speed, theta_i))
        domain_xy = np.array([self.ver_range[node.position[1]] + r_list[:] * np.cos(theta_list[:] + node.psi),
                              self.hor_range[node.position[0]] + r_list[:] * np.sin(theta_list[:] + node.psi)])
        domain_xy = domain_xy.T
        domain_xy.tolist()

        contact_node = np.empty((0, 2))
        contact_node_array = enclosing_checker.check(domain_xy, contact_node)
        cost = weight * len(contact_node_array)
        return cost

    def ship_domain_cost(self, child_ver, child_hor, psi, SD, enclosing_checker):
        """
        Cost = (# of SD vertices inside obstacles) for raw coordinates (CMA-ES use).
        Args:
            child_ver (float): ver (y).
            child_hor (float): hor (x).
            psi (float): heading [rad].
            SD: Ship-domain model (distance()).
            enclosing_checker: pyshipsim.EnclosingPointCollisionChecker instance.
        Returns:
            int: Number of contact vertices.
        """
        distance = ((child_ver - self.end_xy[0, 1]) ** 2 + (child_hor - self.end_xy[0, 1]) ** 2) ** 0.5
        speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
        if speed > 6.8:
            speed = 6.8

        r_list = []
        for theta_i in theta_list:
            r_list.append(SD.distance(speed, theta_i))

        domain_xy = np.array([child_ver + r_list[:] * np.cos(theta_list[:] + psi),
                              child_hor + r_list[:] * np.sin(theta_list[:] + psi)])
        domain_xy = domain_xy.T
        domain_xy.tolist()

        contact_node = np.empty((0, 2))
        contact_node_array = enclosing_checker.check(domain_xy, contact_node)
        cost = len(contact_node_array)
        return cost

    def fill_inner_concave_obstacle(self, array, pitch):
        """
        Fill interior grid nodes inside a (possibly concave) polygon.
        Args:
            array (ndarray): Polygon vertices (ver, hor).
            pitch (float): Grid spacing.
        Returns:
            ndarray: Interior nodes (ver, hor).
        """
        inner_array = np.empty((0, 2))
        ver_min_round = self.RoundRange(np.amin(array[:, 0]), pitch, 'min')

        ver_list = np.arange(self.RoundRange(np.amin(array[:, 0]), pitch, 'min'), np.amax(array[:, 0]) + pitch / 2, pitch)
        hor_list = np.arange(self.RoundRange(np.amin(array[:, 1]), pitch, 'min'), np.amax(array[:, 1]) + pitch / 2, pitch)

        for vi in range(len(ver_list)):
            for hi in range(len(hor_list)):
                if self.crossing_number(np.array([ver_list[vi], hor_list[hi]]), array):
                    inner_array = np.append(inner_array, np.array([[ver_list[vi], hor_list[hi]]]), axis=0)

        return inner_array

    def crossing_number(self, point, obstacle_array):
        """
        Point-in-polygon test (Crossing Number algorithm).
        Args:
            point (ndarray): (ver, hor).
            obstacle_array (ndarray): Polygon vertices (ver, hor).
        Returns:
            bool: True if inside.
        """
        cross_count = 0
        for i in range(len(obstacle_array) - 1):
            edge_start = obstacle_array[i, :]
            edge_end = obstacle_array[i + 1, :]

            if edge_start[0] - edge_end[0] > 0:
                direction = 'down'
            elif edge_start[0] - edge_end[0] < 0:
                direction = 'up'
            else:
                continue

            if (edge_start[0] - point[0]) * (edge_end[0] - point[0]) < 0:
                pass
            elif direction == 'up' and (edge_start[0] == point[0]):
                pass
            elif direction == 'down' and (edge_end[0] == point[0]):
                pass
            else:
                continue

            cross_point_y = (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]) / (edge_end[0] - edge_start[0]) + edge_start[1]
            if cross_point_y > point[1]:
                cross_count += 1

        return (cross_count % 2 == 1)


# Seki says that there is no relationship between SD and ship speed when ship speed is greater than 6.8knots.
# Therefore, when the ship speed is greater than 6.8knots, the SD should be the SD at 6.8knots.
class ShipDomain_proposal():
    """Ship-domain model with speed-dependent anisotropic ellipse parameters."""

    def __init__(self, **kwargs):
        """Store arbitrary attributes from kwargs."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initial_setting(self, filename, func):
        """
        Load SD parameters from CSV and set evaluator.
        Args:
            filename (str): CSV path (expects rows: gf, gp, gs).
            func (callable): G(speed, a, b, c)-like evaluator.
        Returns:
            None
        """
        df = pd.read_csv(filename, index_col=0)
        self.gf_parameters = df.loc['gf'][:].to_numpy()
        self.ga_parameters = df.loc['gp'][:].to_numpy()
        self.gp_parameters = df.loc['gp'][:].to_numpy()
        self.gs_parameters = df.loc['gs'][:].to_numpy()
        self.func = func

    def distance(self, speed, theta):
        """
        Return SD radius G at heading theta [rad] for given speed.
        Args:
            speed (float): Ship speed [kts].
            theta (float): Body-fixed angle [rad], 0=fwd, CW positive.
        Returns:
            float: Radius G along theta.
        """
        gf = self.func(speed, self.gf_parameters[0], self.gf_parameters[1], self.gf_parameters[2])
        ga = self.func(speed, self.ga_parameters[0], self.ga_parameters[1], self.ga_parameters[2]) * 2
        gp = self.func(speed, self.gp_parameters[0], self.gp_parameters[1], self.gp_parameters[2])
        gs = self.func(speed, self.gs_parameters[0], self.gs_parameters[1], self.gs_parameters[2])

        gx = gf if np.cos(theta) >= 0 else ga
        gy = gs if np.sin(theta) >= 0 else gp

        G = gx * gy / (((gx * np.sin(theta)) ** 2 + (gy * np.cos(theta)) ** 2) ** (1 / 2))
        return G
