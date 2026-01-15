# 新しい接触判定を作るためのファイル
import os
import sys

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
from matplotlib.path import Path
from tqdm import tqdm

from utils.PP.subroutine import mpl_config
from utils.LDA.ship_geometry import *
plt.rcParams.update(mpl_config)

# Angle list to approximate ellipse polygon (0–360 deg, step=10 deg)
theta_list = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(10))
length_of_theta_list = len(theta_list)  # 36

def convert(df, port_file):
    df_coord = pd.read_csv(port_file)
    LAT_ORIGIN = df_coord["Latitude"].iloc[0]
    LON_ORIGIN = df_coord["Longitude"].iloc[0]
    ANGLE_FROM_NORTH = df_coord["Psi[deg]"].iloc[0]

    xs = []; ys = []
    for lat, lon in zip(df["lat"].to_numpy(), df["lon"].to_numpy()):
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


@dataclass
class Map:
    grid_pitch: float
    xs: np.ndarray
    ys: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    nodes: np.ndarray          # (H,W,2)  [x,y]
    flat: np.ndarray           # (H*W,2)

    land_polys: List[np.ndarray]
    nogo_polys: List[np.ndarray]

    mask_land: np.ndarray      # (H,W) bool
    mask_nogo: np.ndarray      # (H,W) bool
    label: np.ndarray          # (H,W) uint8: 0 free, 1 land, 2 nogo, 3 both

    @staticmethod
    def _load_polys(csv_path: str,
                    port_file: str,
                    ) -> List[np.ndarray]:
        df = pd.read_csv(csv_path)

        if ("x [m]" not in df.columns) or ("y [m]" not in df.columns):
            xs, ys = convert(df, port_file)
            df["x [m]"] = xs; df["y [m]"] = ys

        polys = []
        for _, g in df.groupby("polygon_id", sort=False):
            xy = g[["x [m]", "y [m]"]].to_numpy()
            polys.append(xy)

        return polys

    @staticmethod
    def _build_grid_from_layers(layers: List[List[np.ndarray]],
                                grid_pitch: float,
                                margin_cells: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_polys = [p for layer in layers for p in layer]
        if len(all_polys) == 0:
            raise ValueError("No polygons found.")

        allxy = np.vstack(all_polys)
        xmin, xmax = float(allxy[:, 0].min()), float(allxy[:, 0].max())
        ymin, ymax = float(allxy[:, 1].min()), float(allxy[:, 1].max())

        margin = float(grid_pitch) * float(margin_cells)
        xmin -= margin; xmax += margin
        ymin -= margin; ymax += margin

        xs = np.arange(xmin, xmax + grid_pitch, grid_pitch, dtype=float)
        ys = np.arange(ymin, ymax + grid_pitch, grid_pitch, dtype=float)
        X, Y = np.meshgrid(xs, ys)
        nodes = np.stack([X, Y], axis=-1) # これを使って座標を呼び出す。(y, x, 0or1)
        flat = nodes.reshape(-1, 2)
        return xs, ys, X, Y, nodes, flat

    @staticmethod
    def _mask_from_polys(polys: List[np.ndarray],
                         flat: np.ndarray,
                         shape_hw: Tuple[int, int],
                         include_boundary: bool = True) -> np.ndarray:
        N = len(flat)
        inside = np.zeros(N, dtype=bool)
        if len(polys) == 0:
            return inside.reshape(shape_hw)

        # 境界を inside 寄りに（必要なら）
        radius = 1e-9 if include_boundary else 0.0

        fx = flat[:, 0]
        fy = flat[:, 1]

        for p in polys:
            if len(p) < 3:
                continue

            # bbox で候補を絞る（高速化）
            xmin, xmax = float(p[:, 0].min()), float(p[:, 0].max())
            ymin, ymax = float(p[:, 1].min()), float(p[:, 1].max())

            cand = (xmin <= fx) & (fx <= xmax) & (ymin <= fy) & (fy <= ymax)
            if not np.any(cand):
                continue

            idx = np.where(cand)[0]
            path = Path(p, closed=True)
            inside[idx] |= path.contains_points(flat[idx], radius=radius)

        return inside.reshape(shape_hw)

    @classmethod
    def GenerateMapFromCSV(cls,
                           land_file: str,
                           no_go_file: str,
                           grid_pitch: float,
                           port_file: str,
                           *,
                           margin_cells: int = 1,
                           include_boundary: bool = True) -> "Map":
        land_polys = cls._load_polys(land_file, port_file)
        nogo_polys = cls._load_polys(no_go_file, port_file)

        xs, ys, X, Y, nodes, flat = cls._build_grid_from_layers(
            [land_polys, nogo_polys],
            grid_pitch=grid_pitch,
            margin_cells=margin_cells
        )
        H, W = Y.shape

        mask_land = cls._mask_from_polys(land_polys, flat, (H, W), include_boundary=include_boundary)
        mask_nogo = cls._mask_from_polys(nogo_polys, flat, (H, W), include_boundary=include_boundary)

        label = np.zeros((H, W), dtype=np.uint8)
        label[mask_land] |= 1
        label[mask_nogo] |= 2

        return cls(
            grid_pitch=grid_pitch,
            xs=xs, ys=ys, X=X, Y=Y, nodes=nodes, flat=flat,
            land_polys=land_polys, nogo_polys=nogo_polys,
            mask_land=mask_land, mask_nogo=mask_nogo, label=label
        )

    # A* 用の障害物を切替（landだけ / land+nogo）
    def blocked_for_astar(self, use_nogo_as_obstacle: bool = False) -> np.ndarray:
        return (self.mask_land | self.mask_nogo) if use_nogo_as_obstacle else self.mask_land
        

    def ShowInitialMap(self, filename=None, SD=None, SD_sw=True, initial_point_list=None):
        """
        初期経路（start/origin/path/last/end と任意のSD）を描画する
        引数:
            filename (str|None): 出力画像ファイルのパス
            SD: Ship-domain モデル（distance() を持つオブジェクト）
            SD_sw (bool): True のときSDポリゴンを描画する
            initial_point_list (list|ndarray|None): 初期点のリスト（ver, hor）
        戻り値:
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
            ax.scatter(initial_points[:, 1], initial_points[:, 0], color='green', s=10, label='Initial Points', zorder=3)

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
        経路とCP・MP上のSD（任意）を重ねてマップを描画する
        引数:
            filename (str|None): 出力画像ファイルのパス
            SD: Ship-domain モデル（distance() を持つオブジェクト）
            initial_point_list (list|ndarray|None): 初期点のリスト（ver, hor）
            optimized_point_list (list|ndarray|None): 最適化後の点のリスト（ver, hor）
            SD_sw (bool): True のときSDポリゴンを描画する
        戻り値:
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

        # buoy, intersection
        if getattr(self, 'buoy_xy', None) is not None : 
            ax.scatter(self.buoy_xy[1] + 0.5, self.buoy_xy[0] + 0.5, color='orange', s=15, zorder=3)
        if getattr(self, "isect_xy", None) is not None:
            ax.scatter(self.isect_xy[1] + 0.5, self.isect_xy[0] + 0.5, color='orange', s=15, zorder=3, label='control point')

        # initial points
        if initial_point_list is not None:
            initial_points = np.array(initial_point_list).reshape(-1, 2)
            ax.scatter(initial_points[:, 1], initial_points[:, 0], color='green', s=10, label='Initial Points', zorder=3)

        # optimized points
        if optimized_point_list is not None:
            optimized_points = np.array(optimized_point_list)
            ax.scatter(
                optimized_points[:, 1],
                optimized_points[:, 0],
                color='blue', s=15, label='Optimized Points', zorder=3
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
                2本の線分が成す角を2等分して,チェックポイントでの船首方位を求める
                引数:
                    *_parent/current/child (float): 各点の座標（ver, hor）
                戻り値:
                    float: 船首方位 psi [rad]
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
                1本の線分の中点における船首方位を求める
                引数:
                    ver_parent, hor_parent, ver_current, hor_current (float)
                戻り値:
                    float: 船首方位 psi [rad]
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
        線分（start -> end）が通過するグリッドセルを列挙する
        引数:
            start, end (ndarray): 始点・終点の座標（ver, hor）
        戻り値:
            ndarray: 通過したグリッドノードの集合（ver, hor）
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
        値を最も近いグリッド境界に丸める
        引数:
            num (float): 丸め対象の値
            pitch (float): グリッド間隔
            TYPE (str): 'min' の場合は num 以下となるように切り下げ，'max' の場合は num 以上となるように切り上げる。
        戻り値:
            float: 丸め後の境界値
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

    def FindNodeOfThePoint(self, point):
        """
        与えられた点を，左下のグリッドノードに対応付ける
        引数:
            point (ndarray): 座標（ver, hor）
        戻り値:
            ndarray: ノード座標 (1, 2)（ver, hor）
        """
        node = np.empty((1, 2))
        node[0, 0] = self.FloorByDesiredPitch(point[0], self.grid_pitch)
        node[0, 1] = self.FloorByDesiredPitch(point[1], self.grid_pitch)
        return node

    def FloorByDesiredPitch(self, num, grid):
        """
        指定したピッチで切り下げる
        引数:
            num (float): 対象となる値
            grid (float): ピッチ
        戻り値:
            float: 切り下げ後の値
        """
        tmp = grid * math.floor(num / grid)
        return tmp

    def ship_domain_cost_astar(self, node, SD, weight, maze):
        v = node.position[0] * self.grid_pitch
        h = node.position[1] * self.grid_pitch

        distance = ((v - self.end_vh[0, 1]) ** 2 +(h - self.end_vh[0, 1]) ** 2) ** (1 / 2)
        speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)

        # SD polygon around node center
        r_list = np.array([SD.distance(speed, th) for th in theta_list], dtype=float)
        domain_xy = np.stack([
            v + r_list * np.cos(theta_list + node.psi),
            h + r_list * np.sin(theta_list + node.psi),
        ], axis=1)

        # maze を uint8(0/1)へ
        maze_u8 = (maze != 0).astype(np.uint8)
        Ny, Nx = maze_u8.shape

        # world -> grid index へ変換（ver_range/hor_range は等間隔前提）
        #   x = ver_range[col], y = hor_range[row]
        ver0 = float(self.ver_range[0])
        hor0 = float(self.hor_range[0])
        dx = float(self.ver_range[1] - self.ver_range[0])
        dy = float(self.hor_range[1] - self.hor_range[0])

        # col(i) と row(j) の float index
        i_f = (domain_xy[:, 0] - ver0) / dx
        j_f = (domain_xy[:, 1] - hor0) / dy

        # ROI (bbox) を作る
        margin = 2
        imin = int(np.floor(i_f.min())) - margin
        imax = int(np.ceil(i_f.max())) + margin
        jmin = int(np.floor(j_f.min())) - margin
        jmax = int(np.ceil(j_f.max())) + margin

        # 画面内にクリップ
        imin = max(0, imin); jmin = max(0, jmin)
        imax = min(Nx - 1, imax); jmax = min(Ny - 1, jmax)

        if imin > imax or jmin > jmax:
            return 0.0

        roi_w = imax - imin + 1
        roi_h = jmax - jmin + 1

        # ROI 座標へ（OpenCV は (x=col, y=row) の int32）
        pts = np.stack([i_f - imin, j_f - jmin], axis=1)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, roi_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, roi_h - 1)

        # ShipDomain の塗りつぶしマスク（ROIのみ）
        sd_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(sd_mask, [pts], 1)

        # maze ROI と重なりセル数
        maze_roi = maze_u8[jmin:jmax+1, imin:imax+1]
        overlap = (sd_mask & maze_roi)
        contact_count = int(overlap.sum())

        cost = weight * contact_count
        return cost

    def ship_domain_cost(self, child_ver, child_hor, psi, SD, enclosing_checker):
        """
        生の座標に対してコストを「障害物内部に入ったSD頂点数」として評価する（CMA-ES 用）
        引数:
            child_ver (float): ver（y座標）
            child_hor (float): hor（x座標）
            psi (float): 船首方位 [rad]
            SD: Ship-domain モデル（distance() を持つオブジェクト）
            enclosing_checker: pyshipsim.EnclosingPointCollisionChecker インスタンス
        戻り値:
            int: 接触した頂点数。
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
        凹多角形を含むポリゴン内部のグリッドノードを塗りつぶして取得する
        引数:
            array (ndarray): ポリゴン頂点（ver, hor）
            pitch (float): グリッド間隔
        戻り値:
            ndarray: 内部ノード（ver, hor）
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
        Crossing Number 法による，点がポリゴン内部にあるかどうかの判定
        引数:
            point (ndarray): 判定する点の座標（ver, hor）
            obstacle_array (ndarray): ポリゴン頂点（ver, hor）
        戻り値:
            bool: 内部にあれば True
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
    """船速依存の異方性楕円パラメータを持つ Ship-domain モデル"""

    def __init__(self, **kwargs):
        """kwargs から渡された任意の属性をインスタンス変数として保持する"""
        """Store arbitrary attributes from kwargs."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initial_setting(self, filename, func):
        """
        CSV からSDパラメータを読み込み,評価関数を設定する。
        引数:
            filename (str): CSVファイルのパス（行名として gf, gp, gs を想定）
            func (callable): G(speed, a, b, c) 形式の評価関数
        戻り値:
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
        与えられた船速と方位 theta [rad] に対する SD 半径 G を返す
        引数:
            speed (float): 船速 [kts]
            theta (float): 船体座標系での角度 [rad]（前方0,時計回り正）
        戻り値:
            float: 方向 theta に沿った半径 G
        """
        gf = self.func(speed, self.gf_parameters[0], self.gf_parameters[1], self.gf_parameters[2])
        ga = self.func(speed, self.ga_parameters[0], self.ga_parameters[1], self.ga_parameters[2]) * 2
        gp = self.func(speed, self.gp_parameters[0], self.gp_parameters[1], self.gp_parameters[2])
        gs = self.func(speed, self.gs_parameters[0], self.gs_parameters[1], self.gs_parameters[2])

        gx = gf if np.cos(theta) >= 0 else ga
        gy = gs if np.sin(theta) >= 0 else gp

        G = gx * gy / (((gx * np.sin(theta)) ** 2 + (gy * np.cos(theta)) ** 2) ** (1 / 2))
        return G