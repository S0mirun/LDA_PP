# 新しい接触判定を作るためのファイル
import os
import sys

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.PP.subroutine import  mpl_config

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
PYSIM_DIR = os.path.join(PROGRAM_DIR, "py-ship-simulator-main/py-ship-simulator-main")
if PYSIM_DIR not in sys.path:
    sys.path.append(PYSIM_DIR)
import pyshipsim 
plt.rcParams.update(mpl_config)

# 楕円の近似多角形を何度刻みで得るか決める角度リスト
theta_list = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(10)) #0~360を10刻みで
length_of_theta_list = len(theta_list) # 36

class Map():
    
    def __init__(self, ver_range, hor_range, grid_pitch=None, start=None, end=None, Miss_Corner=False):
        # input parameter
        self.start = start
        self.end = end
        self.grid_pitch = grid_pitch
        self.ver_range = ver_range
        self.hor_range = hor_range
        self.Miss_Corner = Miss_Corner
        
        # input parameter
        self.obstacle_dict = {}
        self.obstacle_node = np.empty((0,2))
        
    def GenerateMapFromCSV(file, grid_pitch):
        df = pd.read_csv(file)
        ver = np.array(df['x [m]'])
        hor = np.array(df['y [m]'])
        obstacles = np.stack([ver, hor] ,1)
        #
        ver_min_round = Map.RoundRange(None, np.amin(ver), grid_pitch, 'min')
        ver_max_round = Map.RoundRange(None, np.amax(ver), grid_pitch, 'max')
        hor_min_round = Map.RoundRange(None, np.amin(hor), grid_pitch, 'min')
        hor_max_round = Map.RoundRange(None, np.amax(hor), grid_pitch, 'max')
        # 
        ver_range = np.arange(ver_min_round, ver_max_round+grid_pitch/10, grid_pitch)
        hor_range = np.arange(hor_min_round, hor_max_round+grid_pitch/10, grid_pitch)

        target_map = Map(ver_range, hor_range, grid_pitch, Miss_Corner = True)
        Map.AddObstacleLine(target_map, obstacles, 'berth')
        Map.AddObstacleNode(target_map, obstacles, 'berth')
        return target_map
    
    def ShowMap_for_astar(self, filename = None, SD = None, SD_sw = True, initial_point_list = None):

        # set glaph
        x_range = self.hor_range[-1] - self.hor_range[0]
        y_range = self.ver_range[-1] - self.ver_range[0]
        aspect_ratio = x_range / y_range if y_range != 0 else 1
        base_size = 6  # 画像の基本サイズ（小さい場合は大きくする）
        figsize_x = base_size * aspect_ratio
        figsize_y = base_size
        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=300, linewidth=0, edgecolor='w')
        ax = fig.add_subplot(111)

        # show obstacle lines
        for key in self.obstacle_dict:
            ax.plot(self.obstacle_dict[key][:,1], self.obstacle_dict[key][:,0], color = 'k', ls='-', lw = 0.8)

        # show start and end position with adjusted point size
        if 'start_xy' in dir(self):
            ax.scatter(self.start_xy[0,1]+0.5, self.start_xy[0,0]+0.5, color='k', s=15, edgecolors='k', zorder=3)  # ← `s` で点のサイズを指定
            ax.text(self.start_xy[0,1]+0.5, self.start_xy[0,0]+0.6, 'start', va='bottom', ha='center', color='k', fontsize=10)
        if 'end_xy' in dir(self):
            ax.scatter(self.end_xy[0,1]+0.5, self.end_xy[0,0]+0.5, color='k', s=15, edgecolors='k', zorder=3)  # ← `s` で点のサイズを指定
            ax.text(self.end_xy[0,1]+0.5, self.end_xy[0,0]+0.6, 'end', va='bottom', ha='center', color='k', fontsize=10)
            
        # Plot the origin and last point
        ax.scatter(self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5, color='purple', s=15, zorder=3, label='Fixed Points')
        # ax.text(self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.6, 'origin', va='bottom', ha='center', color='k', fontsize=10)
        ax.scatter(self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5, color='purple', s=15, zorder=3)
        # ax.text(self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.6, 'last', va='bottom', ha='center', color='k', fontsize=10)
        
        # Show initial points
        if initial_point_list is not None:
            initial_points = np.array(initial_point_list).reshape(-1, 2)
            ax.scatter(initial_points[:, 1], initial_points[:, 0], color='blue', s=10, label='Initial Points', zorder=3)

        # Combine start, origin,  initial path, last, and end points
        if 'path_xy' in dir(self):
            # Create a list to store all points
            all_points = []
            # Add start point
            all_points.append([self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.5])
            # Add origin point
            all_points.append([self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5])
            # Add all points from path_xy
            all_points.extend(self.path_xy[:, [1, 0]] + 0.5)
            # Add last point
            all_points.append([self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5])
            # Add end point
            all_points.append([self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.5])
            # Convert the list of points to a numpy array for plotting
            all_points = np.array(all_points)
            # Plot the combined path
            ax.plot(all_points[:, 0], all_points[:, 1], lw=1.0, color='r', ls='-')

        # show ship domain
        if 'psi' in dir(self) and not SD == None and SD_sw == True:
            for j in range(len(self.path_xy)):
                # SDを表示する間隔
                if j % 30 == 0:
                    distance = ((self.path_xy[j,1]+0.5 - self.end_xy[0,1]+0.5) ** 2 + (self.path_xy[j,0]+0.5 - self.end_xy[0,0]+0.5) ** 2) ** (1/2)
                    speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
                    r_list = []
                    for theta_i in theta_list:
                        r_list.append(SD.distance(speed, theta_i))
                    
                    # 多角形を閉じるために最初の点を最後に追加
                    r_list.append(r_list[0])
                    theta_list_closed = np.append(theta_list, theta_list[0])

                    ax.plot(self.path_xy[j,1] + 0.5 + np.array(r_list) * np.sin(theta_list_closed + self.psi[j]),
                            self.path_xy[j,0] + 0.5 + np.array(r_list) * np.cos(theta_list_closed + self.psi[j]), 
                            lw=0.5, color='g', ls='--')
                        
                    """  
                    ax.plot(self.path_xy[j,1]+ 0.5 + r_list[:] * np.sin(theta_list[:] + self.psi[j]),
                            self.path_xy[j,0]+ 0.5 + r_list[:] * np.cos(theta_list[:] + self.psi[j]), 
                            lw = 0.5, color = 'g', ls = '--'
                            )
                    """
                    
        # Set axis limits
        ax.set_xlim(self.hor_range[0], self.hor_range[-1] + self.grid_pitch)
        ax.set_ylim(self.ver_range[0], self.ver_range[-1] + self.grid_pitch)

        # Set major ticks for grid
        # X軸
        x_start = np.floor(self.hor_range[0] / 200) * 200  # 最小値を200の倍数に丸める
        x_end = np.ceil(self.hor_range[-1] / 200) * 200   # 最大値を200の倍数に丸める
        ax.set_xticks(np.arange(x_start, x_end + 200, 200))

        # Y軸
        y_start = np.floor(self.ver_range[0] / 200) * 200  # 最小値を200の倍数に丸める
        y_end = np.ceil(self.ver_range[-1] / 200) * 200   # 最大値を200の倍数に丸める
        ax.set_yticks(np.arange(y_start, y_end + 200, 200))

        # Apply the labels to the plot
        ax.set_xticklabels(np.arange(x_start, x_end + 200, 200), rotation=90)
        ax.set_yticklabels(np.arange(y_start, y_end + 200, 200), rotation=0)

        ax.set_xlabel(r'$Y\,\rm{[m]}$')
        ax.set_ylabel(r'$X\,\rm{[m]}$')

        # Add grid and aspect ratio
        plt.grid(which='major', color='k', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')

        # 凡例を追加（カスタマイズ）
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


        # ax.scatter(self.path_xy[:,1]+0.5, self.path_xy[:,0]+0.5, lw = 0.1, color = 'r')
        # show the smoothed path
        # if 'smoothed_path_xy' in dir(self):
            # ax.plot(   self.smoothed_path_xy[:,1]+0.5, self.smoothed_path_xy[:,0]+0.5, lw = 1.5, color = 'b', zorder = 2)
        """
        ax.set_xlim(self.hor_range[0], self.hor_range[-1]+self.grid_pitch)
        ax.set_ylim(self.ver_range[0], self.ver_range[-1]+self.grid_pitch)

        ax.set_xticks(self.hor_range[::20])
        ax.set_yticks(self.ver_range[::20])

        ax.set_xticklabels(self.hor_range[::20], rotation=90)
        ax.set_yticklabels(self.ver_range[::20], rotation=0)


        ax.set_xlabel(r'$Y\\,\\rm{[m]}$')
        ax.set_ylabel(r'$X\\,\\rm{[m]}$')

        if 'txt' in dir(self):
            ax.text(self.hor_range[-1]+self.grid_pitch, self.ver_range[-1]+self.grid_pitch,self.txt,  ha = 'right', va = 'top')

        plt.grid(which='major',color='k',linestyle='--',linewidth = 0.4,alpha = 0.5)
        # plt.grid(which='minor',color='k',linestyle='--',linewidth = 0.3,alpha = 0.5)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        fig.savefig(filename, pad_inches=0.05)
        plt.close()
        """

    def ShowMap(self, filename=None, SD=None, initial_point_list=None, optimized_point_list=None, SD_sw=True):
        # set glaph
        x_range = self.hor_range[-1] - self.hor_range[0]
        y_range = self.ver_range[-1] - self.ver_range[0]
        aspect_ratio = x_range / y_range if y_range != 0 else 1
        base_size = 6
        figsize_x = base_size * aspect_ratio
        figsize_y = base_size
        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=300, linewidth=0, edgecolor='w')
        ax = fig.add_subplot(111)

        # Show obstacle lines
        for key in self.obstacle_dict:
            ax.plot(self.obstacle_dict[key][:, 1], self.obstacle_dict[key][:, 0], color='k', ls='-', lw=0.8)

        # show start and end position with adjusted point size
        if 'start_xy' in dir(self):
            ax.scatter(self.start_xy[0,1]+0.5, self.start_xy[0,0]+0.5, color='k', s=15, edgecolors='k', zorder=3)  # ← `s` で点のサイズを指定
            ax.text(self.start_xy[0,1]+0.5, self.start_xy[0,0]+0.6, 'start', va='bottom', ha='center', color='k', fontsize=10)
        if 'end_xy' in dir(self):
            ax.scatter(self.end_xy[0,1]+0.5, self.end_xy[0,0]+0.5, color='k', s=15, edgecolors='k', zorder=3)  # ← `s` で点のサイズを指定
            ax.text(self.end_xy[0,1]+0.5, self.end_xy[0,0]+0.6, 'end', va='bottom', ha='center', color='k', fontsize=10)
            
        # Plot the origin and last point
        ax.scatter(self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5, color='purple', s=15, zorder=3, label='Fixed Points')
        # ax.text(self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.6, 'origin', va='bottom', ha='center', color='k', fontsize=10)
        ax.scatter(self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5, color='purple', s=15, zorder=3)
        # ax.text(self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.6, 'last', va='bottom', ha='center', color='k', fontsize=10)

        # Show initial points
        if initial_point_list is not None:
            initial_points = np.array(initial_point_list).reshape(-1, 2)
            ax.scatter(initial_points[:, 1], initial_points[:, 0], color='blue', s=10, label='Initial Points', zorder=3)
            # 点の番号を表示する
            # for idx, (x, y) in enumerate(initial_points):
                # ax.text(y, x, f"P{idx+1}", color='blue', fontsize=4, va='bottom', ha='center')
                
        # Show optimized points (actual coordinates)
        if optimized_point_list is not None:
            # Convert tuple list to numpy array
            optimized_points = np.array(optimized_point_list)
            ax.scatter(
                optimized_points[:, 1],  # x-coordinates
                optimized_points[:, 0],  # y-coordinates
                color='green', s=15, label='Optimized Points', zorder=3
            )
            # 点の番号を表示する
            # for idx, (x, y) in enumerate(optimized_points):
                # ax.text(y, x, f"O{idx+1}", color='green', fontsize=4, va='bottom', ha='center')

        # Combine start, origin, optimized path, last, and end points
        if 'path_xy' in dir(self):
            # Create a list to store all points
            all_points = []
            # Add start point
            all_points.append([self.start_xy[0, 1] + 0.5, self.start_xy[0, 0] + 0.5])
            # Add origin point
            all_points.append([self.origin_xy[0, 1] + 0.5, self.origin_xy[0, 0] + 0.5])
            # Add all points from path_xy
            all_points.extend(self.path_xy[:, [1, 0]] + 0.5)
            # Add last point
            all_points.append([self.last_xy[0, 1] + 0.5, self.last_xy[0, 0] + 0.5])
            # Add end point
            all_points.append([self.end_xy[0, 1] + 0.5, self.end_xy[0, 0] + 0.5])
            # Convert the list of points to a numpy array for plotting
            all_points = np.array(all_points)
            # Plot the combined path
            ax.plot(all_points[:, 0], all_points[:, 1], lw=1.0, color='r', ls='-')

        # Show the SD
        if SD_sw:
            # check pointでのSDの描画
            # CPでのSDの方向を与える関数
            def cal_psi_at_checkpoint(ver_parent, hor_parent, ver_current, hor_current, ver_child, hor_child):
                
                vector_1 = np.array([hor_current - hor_parent, ver_current - ver_parent])
                vector_2 = np.array([hor_child - hor_current, ver_child - ver_current])
                
                magnitude_vector_1 = np.linalg.norm(vector_1)
                magnitude_vector_2 = np.linalg.norm(vector_2)
                
                dot_product = np.dot(vector_1, vector_2)
                cos_theta = dot_product / (magnitude_vector_1 * magnitude_vector_2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # [-1, 1] の範囲に収める
                angle_rad = np.arccos(cos_theta)
                # angle_deg = np.degrees(angle_rad)  # 角度（0~180°）
                # print(f"内積を用いて計算した２つのベクトルの成す角は{angle_deg}です")

                cross_product = np.cross(vector_1, vector_2)
                if cross_product > 0:
                    direction = -1  # 反時計回り
                elif cross_product < 0:
                    direction = 1   # 時計回り
                else:
                    direction = 0   # 直線上にある（角度は0° or 180°）
                # print(f"回転方向は{direction}です")

                psi = np.deg2rad(90) - np.arctan2(ver_current - ver_parent, hor_current - hor_parent)
                # print(f"調整前のpsi:{psi}")
                if psi > np.deg2rad(180):
                    psi = (np.deg2rad(360) - psi) * (-1)
                psi = psi + (angle_rad / 2) * direction
                # print(f"調整後のpsi:{psi}")
                # 再調整
                if psi > np.deg2rad(180):
                    psi = (np.deg2rad(360) - psi) * (-1)
                # print(f"再調整後のpsi:{psi}")
                return psi
            
            # MidpointでのSDの描画
            # MPでのSDの方向を与える関数
            # 描画ポイントを手動で与える必要がある
            def cal_psi_at_midpoint(ver_parent, hor_parent, ver_current, hor_current):
                psi = np.deg2rad(90) - np.arctan2(ver_current - ver_parent, hor_current - hor_parent)
                # 11/19のノートに角度計算のノートあり。ここは計算大丈夫！
                if psi > np.deg2rad(180):
                    psi = (np.deg2rad(360) - psi) * (-1)
                    #print("psi is over 180. Changing method is done")
                return psi
            
            path_for_cal = np.vstack([self.origin_xy, self.path_xy, self.last_xy])  # pathにoriginとlastを追加し、必要な角度計算を行えるようにする
            
            # CPでの描画
            # psi_list の作成（origin, last を除くすべての checkpoint）
            psi_list_at_cp = []
            # Cp_listの作成
            cp_list = []
            for i in range(1, len(path_for_cal) - 1):  # 1 から len(path_for_cal) - 2 まで
                ver_parent, hor_parent = path_for_cal[i - 1]
                ver_current, hor_current = path_for_cal[i]
                ver_child, hor_child = path_for_cal[i + 1]
                cp_list.append((ver_current, hor_current))
                psi_at_cp = cal_psi_at_checkpoint(ver_parent, hor_parent, ver_current, hor_current, ver_child, hor_child)
                psi_list_at_cp.append(psi_at_cp)
            for j in range(len(self.path_xy)):
                distance = ((self.path_xy[j,1] - self.end_xy[0,1]) ** 2 + (self.path_xy[j,0] - self.end_xy[0,0]) ** 2) ** 0.5
                speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
                r_list = []
                for theta_i in theta_list:
                    r_list.append(SD.distance(speed, theta_i))
                # 多角形を閉じるために最初の点を最後に追加
                r_list.append(r_list[0])
                theta_list_closed = np.append(theta_list, theta_list[0])
                # path_xyは(ver, hor)の順で格納されているため、通常の縦横で出力するax.plotに合わせるために(hor, ver)の順になっているので注意せよ!!
                ax.plot(self.path_xy[j,1] + np.array(r_list) * np.sin(theta_list_closed + psi_list_at_cp[j]),
                        self.path_xy[j,0] + np.array(r_list) * np.cos(theta_list_closed + psi_list_at_cp[j]), 
                        lw=0.5, color='g', ls='--')
                
            # MPでの描画
            # psiリストの作成
            psi_list_at_mp = []
            for i in range(len(path_for_cal) - 1):
                ver_parent, hor_parent = path_for_cal[i]
                ver_current, hor_current = path_for_cal[i + 1]
                psi_st_mp = cal_psi_at_midpoint(ver_parent, hor_parent, ver_current, hor_current)
                psi_list_at_mp.append(psi_st_mp)
            # mid pointリストの作成とmpでのSD描画
            mp_list = []
            for j in range(len(path_for_cal) - 1):
                mid_point_hor = (path_for_cal[j, 1] + path_for_cal[j + 1, 1]) / 2
                mid_point_ver = (path_for_cal[j, 0] + path_for_cal[j + 1, 0]) / 2
                mp_list.append((mid_point_ver, mid_point_hor))
                distance = ((mid_point_hor - self.end_xy[0,1]) ** 2 + (mid_point_ver - self.end_xy[0,0]) ** 2) ** 0.5
                speed = self.b_ave * distance ** self.a_ave + self.b_SD * distance ** self.a_SD
                r_list = [SD.distance(speed, theta_i) for theta_i in theta_list]
                r_list.append(r_list[0])  # 多角形を閉じるために最初の点を追加
                theta_list_closed = np.append(theta_list, theta_list[0])
                ax.plot(mid_point_hor + np.array(r_list) * np.sin(theta_list_closed + psi_list_at_mp[j]),
                        mid_point_ver + np.array(r_list) * np.cos(theta_list_closed + psi_list_at_mp[j]),
                        lw=0.5, color='b', ls='--')  # 色は青 (b) に設定
                
        print(f"出力された図を見て、CPでの角度がおかしくないか確認せよ")
        print("Psi List (degrees):", np.degrees(psi_list_at_cp))
            
        # Set axis limits
        ax.set_xlim(self.hor_range[0], self.hor_range[-1] + self.grid_pitch)
        ax.set_ylim(self.ver_range[0], self.ver_range[-1] + self.grid_pitch)

        # Set major ticks for grid
        # X軸
        x_start = np.floor(self.hor_range[0] / 200) * 200  # 最小値を200の倍数に丸める
        x_end = np.ceil(self.hor_range[-1] / 200) * 200   # 最大値を200の倍数に丸める
        ax.set_xticks(np.arange(x_start, x_end + 200, 200))

        # Y軸
        y_start = np.floor(self.ver_range[0] / 200) * 200  # 最小値を200の倍数に丸める
        y_end = np.ceil(self.ver_range[-1] / 200) * 200   # 最大値を200の倍数に丸める
        ax.set_yticks(np.arange(y_start, y_end + 200, 200))

        # Apply the labels to the plot
        ax.set_xticklabels(np.arange(x_start, x_end + 200, 200), rotation=90)
        ax.set_yticklabels(np.arange(y_start, y_end + 200, 200), rotation=0)

        ax.set_xlabel(r'$Y\,\rm{[m]}$')
        ax.set_ylabel(r'$X\,\rm{[m]}$')

        # Add grid and aspect ratio
        plt.grid(which='major', color='k', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')

        # 凡例を追加（カスタマイズ）
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles,
            labels,
            loc='lower right',
            fontsize=9,  # フォントサイズを大きく
            scatterpoints=1,
            handletextpad=0.3,  # テキストとマーカーの間隔を広げる
            labelspacing=0.6,  # 凡例の行間を広げる
            borderpad=0.6,  # 凡例の枠内の余白を増やす
            frameon=True,
            markerscale=1.2  # マーカーのサイズを大きくする
        )
        plt.tight_layout()
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        # (ver, hor)順のcp, mpリストと角度[rad]リストを返す
        return cp_list, mp_list, psi_list_at_cp, psi_list_at_mp
        
    # 線分（start から end まで。任意に設定できる）とグリッド(網目、縦横の線)の交点を計算し、通過するグリッドセル（ノード）を特定するための関数
    # @profile
    def DetictCollision(self, start, end):
        
        # ax + by +c = 0 縦軸がxであることに注意。普通のxとyを入れ替えたものだと思えばいい
        a = end[1] - start[1]
        b = start[0] - end[0]
        c = start[1]*end[0] - start[0]*end[1]

        # 手順Ⅰ：「線分」と「縦軸(=x軸)に並行なグリッド」の交点を求める
        ver_list = []
        for ver_i in self.ver_range:
            if ver_i >= min(start[0], end[0]) and ver_i <= max(start[0], end[0]):
                ver_list.append(ver_i)
        ver_array = np.empty((len(ver_list),2))
        ver_array[:,0] = ver_list[:]
        
        # 線分が縦軸(x軸)に平行かどうかでver_array[:,1]の値を決め方を変えている
        if b == 0:
            ver_array[:,1] = start[1]
        else:
            ver_array[:,1] = -(a/b) * ver_array[:,0] - (c/b)

        # 手順Ⅱ：「線分」と「横軸(=y軸)に並行なグリッド」の交点を求める
        hor_list = []
        for hor_i in self.hor_range:
            if hor_i >= min(start[1], end[1]) and hor_i <= max(start[1], end[1]):
                hor_list.append(hor_i)
        hor_array = np.empty((len(hor_list),2))
        hor_array[:,1] = hor_list[:]
        
        # 線分が横軸(y軸)に平行かどうかでhor_array[:,0]の値を決め方を変えている
        if a == 0:
            hor_array[:,0] = start[0]
        else:
            hor_array[:,0] = -(b/a) * hor_array[:,1] - (c/a)

        # 手順ⅠⅡで求めたgridとの交点を合わせ，重複部分を解消
        ans_array = np.concatenate([hor_array, ver_array]) #配列を統合し、ans_array に格納。horもverも2次元のndarray。行と列成分を合わせているとかではない(これをしたいならnp.stack)。求めた交点を一元管理し重複を削除するための前準備
        ans_array = np.unique(ans_array, axis = 0) # 重複する交点を削除。さらに順番はそのままではなくソートされる
        
        ans_array = np.block([[ans_array], [start], [end]]) # ans_array に start と end を追加。これもconcatenateやstackと似たようなもの

        # ans_arrayの内，格子点に重なる点を抽出（start，endが格子点の場合も考慮される）
        grid_point_array = np.empty((0,2))
        
        # ans_array
        for i in range(len(ans_array)):
            # ans_array の各交点に対して、その x座標が self.ver_range（x軸方向のグリッド範囲）に含まれ、かつ y座標が self.hor_range（y軸方向のグリッド範囲）に含まれるかどうかをチェック
            if ans_array[i,0] in self.ver_range and ans_array[i,1] in self.hor_range:
                grid_point_array = np.append(grid_point_array, np.array([[ans_array[i,0], ans_array[i,1]]]), axis = 0) #array([])はルール。二つの要素を合わせたいのでさらに[]で囲っている。axis=0で新たな行として追加（縦に連結）、axis=1で新たな列として追加（横に連結）される。

        # ans_arrayの隣り合う点を結んだ点の中点リストを作成。uniqueでソートしたからできる
        middle_point_array = np.empty((0,2))
        
        # 中点は両端の点の数―1個になるから－1
        for i in range(len(ans_array)-1):
            middle_point_array = np.append(middle_point_array, np.array([[np.mean(ans_array[i:i+2, 0]), np.mean(ans_array[i:i+2, 1])]]), axis = 0) #ans_array[i:i+2, 0] は i番目の行から i+1 番目の行までを取得(隣り合う点だからiとi+1)し、その中から列 0（x座標）の要素を取り出します。

        ### 通過ノード判定  アンカーは左下
        pass_node_array = np.empty((0,2))
        
        # start位置に該当するノードを特定し、pass_node_arrayに追加
        pass_node = self.FindNodeOfThePoint(start)
        pass_node_array = np.append(pass_node_array, pass_node, axis = 0)
        
        # end位置に該当するノードを特定し、pass_node_arrayに追加
        pass_node = self.FindNodeOfThePoint(end)
        pass_node_array = np.append(pass_node_array, pass_node, axis = 0)
                
        # 格子点
        if self.Miss_Corner == False:
            # 各格子点に対して、その格子点を左下アンカーとするノードの周りにさらに左下、左上、右下のアンカーを持つノードを追加します。これにより、1つの格子点に対して4つのノードが追加されます。現在のノード
            for i in range(len(grid_point_array)):
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i,0]                      , grid_point_array[i,1]                      ]]), axis = 0) # 現在のノード
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i,0]-self.grid_pitch, grid_point_array[i,1]                      ]]), axis = 0) # 左隣のノード
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i,0]                      , grid_point_array[i,1]-self.grid_pitch]]), axis = 0) # 下隣のノード
                pass_node_array = np.append(pass_node_array, np.array([[grid_point_array[i,0]-self.grid_pitch, grid_point_array[i,1]-self.grid_pitch]]), axis = 0) # 斜め左下のノード
                        
        # 中点
        for i in range(len(middle_point_array)):
            pass_node = self.FindNodeOfThePoint(middle_point_array[i,:])
            pass_node_array = np.append(pass_node_array, pass_node, axis = 0) # axis=0で新たな行として追加（縦に連結）
        pass_node_array = np.unique(pass_node_array, axis = 0)

        return pass_node_array

    # RoundRangeメソッドには、切り捨てる方向を指定するための引数TYPEがあります。これには、'min'または'max'を指定します。'min'を指定すると、与えられた数値以下の最大のpitchの倍数を返し、'max'を指定すると、与えられた数値以上の最小のpitchの倍数を返します。
    def RoundRange(self, num, pitch, TYPE):
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
    
    # このメソッドは、指定された障害物を一意の名前で辞書に追加するためのもの。キーの重複を避けるために、同じ名前の障害物が既に存在する場合はキーに連番を付けて一意にしている。
    def AddObstacleLine(self, array, name = 'Noname'):
        name = name
        num = 0
        while True:
            key = name + '-' + str(num)
            if key in self.obstacle_dict:
                num += 1
            else:
                self.obstacle_dict[key] = array
                break

    # 障害物の頂点リストとグリッドピッチを用いて、障害物ノードリスト（地形内部に含まれるすべての点）をピッチに合ったものに更新する
    def AddObstacleNode(self, array, name = 'Noname'):
        inner_node = self.fill_inner_concave_obstacle(array, self.grid_pitch)
        # arrayは後で障害物の頂点を示すobstaclesという2次元配列が代入される。辺の数は頂点の数-1
        for i in range(len(array) - 1):
            tmp_array = self.DetictCollision(array[i,:], array[i+1,:])
            self.obstacle_node = np.concatenate([self.obstacle_node, tmp_array])
        self.obstacle_node = np.append(self.obstacle_node, inner_node, axis = 0)
        self.obstacle_node = np.unique(self.obstacle_node, axis = 0)
        self.obstacle_map = set([tuple(x) for x in self.obstacle_node]) # self.obstacle_node は2次元の numpy 配列で、各行がノードの座標[y,x]を示す。内包表記で各行(配列)が tuple に変換される。set は、重複する要素を持たないデータ構造。重複しないタプルが要素となったセットが作られる{(),(),()}

    # 指定されたピッチ（grid_pitch）に基づいて、与えられた点の座標をグリッドの左下アンカーに揃えることで、点が属するノードを特定します。
    def FindNodeOfThePoint(self, point): # 与えられた点がどのノードに存在するか調べる point must be ndarray(1,2) 1行2列で(y,x)を表す
        node = np.empty((1,2))
        # ver np.emptyで生成するときは0を使うと本当に0行となって存在しないリストを作ることになるが、要素を指定するときのインデクスは0から始まる
        node[0,0] = self.FloorByDesiredPitch(point[0], self.grid_pitch)
        #hor
        node[0,1] = self.FloorByDesiredPitch(point[1], self.grid_pitch)
        return node #アンカーは左下

    # 与えられた数を，任意の刻み幅で切り捨てる。これは、例えばグリッド状の座標を計算する際などに有用です。FindNodeThePointでgrid は grid_pitchに代入されている
    def FloorByDesiredPitch(self, num, grid):
        tmp = grid * math.floor(num/grid)
        return tmp

    # Set a maze for Astar. Using binary to determinate if it is an obstacle
    def SetMaze(self):
        maze_np = np.zeros((len(self.ver_range), len(self.hor_range)), int)
        ### set the obstacle node
        # get the coordinate of obstacle node from self
        for i in range(len(self.obstacle_node)):
            ver_i = np.where(self.ver_range == self.obstacle_node[i, 0]) #np.where()の第二引数, 第三引数を省略すると、条件を満たす要素のインデックス（位置）が返される。
            hor_i = np.where(self.hor_range == self.obstacle_node[i, 1])
            maze_np[ver_i, hor_i] = 1  #障害物は１で表す
        # 配列の転置
        # 後々A*で使うときに（hor、ver）の順にする必要があるためであると考えられる。
        # ややこしいが、A＊は通常の横縦の順で入力し、横縦の順で出力される
        maze_np = maze_np.T
        count_ones = np.sum(maze_np == 1)
        # print(f"マップ内に含まれる1の数: {count_ones}")
        self.maze = maze_np.tolist()
        
    #Nodeクラスを用いるver. A*ではこっちを使う
    # @profile
    def ship_domain_cost_astar(self, node, SD, weight, enclosing_checker):
        #ゴール地点から現在地までの距離distance
        distance = ((self.ver_range[node.position[1]] - self.end_xy[0,1]) ** 2 + (self.hor_range[node.position[0]] - self.end_xy[0,1]) ** 2) ** (1/2)
        # distanceから速力逓減ガイドラインに基づき速度を決定
        speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
        # r_listは、各角度方向でのドメインの頂点までの距離[1]がx,[0]がy
        r_list = []
        for theta_i in theta_list:
            r_list.append(SD.distance(speed, theta_i)) #distanceメソッドによりある角度での楕円の半径が得られるので、下のコードである角度刻みの半径が得られる）
        domain_xy = np.array([self.ver_range[node.position[1]] + r_list[:] * np.cos(theta_list[:] + node.psi),
                              self.hor_range[node.position[0]] + r_list[:] * np.sin(theta_list[:] + node.psi)
                        ])
        # 横並びになったdomain_xyを縦にして(x,y)のペアにして扱うために転置している
        domain_xy = domain_xy.T
        domain_xy.tolist()
        
        contact_node = np.empty((0,2))
        # enclosing_checkerはメインのファイルで既にインスタンス化しているのでOK
        # contact_nodeに格納する
        contact_node_array = enclosing_checker.check(domain_xy, contact_node)#checkで内部にあるものはcontact_nodeに格納される。少しややこしいけど第2引数が格納先という構造
        # print(f"接触した点は{contact_node_array}")
        # print(f"接触した点の数は{len(contact_node_array)}")
        cost = weight * len(contact_node_array)
        # print(f"最終コストは{cost}")
        
        """
        df_world = pd.read_csv(port_csv)
        world_polys = [] #coreでself.OBSTACLE_PLYGONS=[]のように定義されているため
        world_polys.append(df_world[['x [m]', 'y [m]']].to_numpy())
        enclosing = pyshipsim.EnclosingPointCollisionChecker()
        enclosing.reset(world_polys)
        contact_node = np.empty((0,2))
        contact_node_array = enclosing.check(domain_xy, contact_node)#checkで内部にあるものはcontact_nodeに格納される
        cost = weight * len(contact_node_array)
        """
        
        '''
        domain_passed_node = np.empty((0,2))
        for i in range(len(domain_xy)-1):
            add_domain_passed_node = cy_DDA_2D(domain_xy[i,:], domain_xy[i+1,:], self.grid_pitch,len(self.ver_range) * len(self.hor_range) + 1)
            # add_domain_passed_node = self.DetictCollision(domain_xy[i,:], domain_xy[i+1,:])
            domain_passed_node = np.append(domain_passed_node, add_domain_passed_node, axis = 0)
        domain_passed_node = self.CancellDuplication(domain_passed_node)
        # self.tmp_array1 = domain_passed_node
        # self.ShowMap()

        # domain内部の格子点の数え上げ,加算
        domain_passed_node = np.append(domain_passed_node, self.fill_inner_domain(domain_passed_node, self.grid_pitch), axis = 0)

        # domainが通過したノードを記録
        contact_node = extract_common_elements(domain_passed_node, self.obstacle_node)

        # 重複した場合，重み係数と，接触ノード数に応じてコスト加算
        # 将来的には，重み係数と，現在地との距離に応じてコスト加算に変更する可能性あり
        cost = weight * len(contact_node)
        '''
        return cost
    
    # CMAを用いるためにNodeクラスを使用せずにノーマルな座標系での計算を可能にしたver.
    # 分かりやすくするためにスタート側をparent、ゴール側をchildとしているが、A*のような親子関係はない
    # @profile
    def ship_domain_cost(self, child_ver, child_hor, psi, SD, enclosing_checker):
        # 船の現在地からship domainを作成
        #ゴール地点から現在地までの距離
        distance = ((child_ver - self.end_xy[0,1]) ** 2 + (child_hor - self.end_xy[0,1]) ** 2) ** 0.5
        #distanceに基づいて速力逓減ガイドラインに基づき速度を決定
        speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
        # Seki says that there is no relationship between SD and ship speed when ship speed is greater than 6.8knots. 
        # Therefore, when the ship speed is greater than 6.8knots, the SD should be the SD at 6.8knots.
        if speed > 6.8:
            speed = 6.8
        #r_listは、各角度方向でのSDの頂点までの距離[1]がx(ver),[0]がy(hor)
        r_list = []
        for theta_i in theta_list:
            r_list.append(SD.distance(speed, theta_i)) #distanceメソッドによりある角度での楕円の半径が得られるので、下のコードである角度刻みの半径が得られる）
        """
        pitch / 2 が意味不明
        domain_xy = np.array([child_ver + pitch / 2 + r_list[:] * np.cos(theta_list[:] + psi),
                              child_hor + pitch / 2 + r_list[:] * np.sin(theta_list[:] + psi)
                        ])
        """
        # 半径の並び順と角度の並び順はリンクしている[:]
        # 手計算で確かめたがこの式は合っている！
        domain_xy = np.array([child_ver + r_list[:] * np.cos(theta_list[:] + psi),
                              child_hor + r_list[:] * np.sin(theta_list[:] + psi)
                        ])
        #横並びになったdomain_xyを縦にして(x,y)のペアにして扱うために転置している
        domain_xy = domain_xy.T
        # Numpy配列をPythonのリスト形式に変換する
        domain_xy.tolist()
        # SDを形成する頂点が地形内部に存在するかどうか判定する
        # 地形を毎度生成することを避けるため、pyshipsimulatorのインスタンス化はCMAES_PathPlanningファイルの方で行っている
        contact_node = np.empty((0,2))
        #enclosing_checkerはメインのファイルで既にインスタンス化しているのでOK
        contact_node_array = enclosing_checker.check(domain_xy, contact_node)#checkで内部にあるものはcontact_nodeに格納される。少しややこしいけど第2引数が格納先という構造
        cost = len(contact_node_array)
        # 接触点数と円周上の点の数を返す
        # len(theta_list)はgeneral定数にしてもいい。変わらないから。
        return cost

    # 凹地形の内部にある点群を調べる
    def fill_inner_concave_obstacle(self, array, pitch):
        inner_array = np.empty((0,2))
        ver_min_round = self.RoundRange(np.amin(array[:,0]), pitch, 'min')

        ver_list = np.arange(self.RoundRange(np.amin(array[:,0]), pitch, 'min'), np.amax(array[:,0]) + pitch / 2, pitch) #最大値をamaxにしてしまうと最大値が含まれないから、+pitch/2して最大値まで含めるようにしている
        hor_list = np.arange(self.RoundRange(np.amin(array[:,1]), pitch, 'min'), np.amax(array[:,1]) + pitch / 2, pitch)

        for vi in range(len(ver_list)):
            for hi in range(len(hor_list)):
                # Trueで点が内部にある判定
                if self.crossing_number(np.array([ver_list[vi], hor_list[hi]]), array):
                    inner_array = np.append(inner_array, np.array([[ver_list[vi], hor_list[hi]]]), axis = 0) #axis=0で新たな行として追加する場合は列数が、axis=1で新たな列として追加する場合は行数が、元の配列と一致している必要がある。足りない部分が欠損値NaNで埋められたりはしない。

        return inner_array

    def crossing_number(self, point, obstacle_array):
        # obstacle_arrayが，時計または反時計回りのベクトルの集合になっているか確認
        # self.ShowMap(filename = 'output/fig/20240110/test.png')
        # debug
        # point = np.array([2,1])
        # obstacle_array = np.array([[1,2], [3,2]])

        # 点から右側に伸ばした半直線と辺との交差判定を各辺で実施
        # Crossing Number Algorithm(https://www.nttpc.co.jp/technology/number_algorithm.html)
        #!![0]がy座標、[1]がx座標である!!
        cross_count = 0
        for i in range(len(obstacle_array)-1):
            edge_start = obstacle_array[i,:]
            edge_end = obstacle_array[i+1,:]
            # 辺の方向判定
            if edge_start[0] - edge_end[0] > 0:
                direction = 'down'
            elif edge_start[0] - edge_end[0] < 0:
                direction = 'up'
            else: # edge_start[0] == edge_end[0]
                continue
            # 直線と辺の交差判定
            if (edge_start[0] - point[0]) * (edge_end[0] - point[0]) < 0 :
                pass
            elif direction == 'up' and (edge_start[0] == point[0]):
                pass #上向きの時は始点を含むから
            elif direction == 'down' and (edge_end[0] == point[0]):
                pass #下向きの時は終点を含むから
            else:
                continue
            cross_point_y = (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]) / (edge_end[0] - edge_start[0]) + edge_start[1] 
            if cross_point_y > point[1]:
                cross_count += 1

        # Trueで内部にある判定
        if cross_count % 2 == 1:
            return True
        else:
            return False


# Seki says that there is no relationship between SD and ship speed when ship speed is greater than 6.8knots. 
# Therefore, when the ship speed is greater than 6.8knots, the SD should be the SD at 6.8knots.
class ShipDomain_proposal():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def initial_setting(self,filename, func):
        df = pd.read_csv(filename, index_col = 0)
        self.gf_parameters = df.loc['gf'][:].to_numpy()
        self.ga_parameters = df.loc['gp'][:].to_numpy()
        self.gp_parameters = df.loc['gp'][:].to_numpy()
        self.gs_parameters = df.loc['gs'][:].to_numpy()
        self.func = func
    # 船速speedと船体固定角度であるthetaを入力するとその方向の楕円の半径Gを返すメソッド
    # thetaは船首方向を０に取り、時計回りである。
    # ここの角度計算は問題ない
    def distance(self, speed, theta):
        gf = self.func(speed, self.gf_parameters[0], self.gf_parameters[1], self.gf_parameters[2])
        ga = self.func(speed, self.ga_parameters[0], self.ga_parameters[1], self.ga_parameters[2]) * 2
        gp = self.func(speed, self.gp_parameters[0], self.gp_parameters[1], self.gp_parameters[2])
        gs = self.func(speed, self.gs_parameters[0], self.gs_parameters[1], self.gs_parameters[2])
        if np.cos(theta) >= 0:
            gx = gf
        else:
            gx = ga
        if np.sin(theta) >= 0:
            gy = gs
        else:
            gy = gp
        G = gx * gy / (((gx * np.sin(theta)) ** 2 + (gy * np.cos(theta)) ** 2) ** (1/2))
        return G