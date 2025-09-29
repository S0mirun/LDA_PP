import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from subroutine import ship_status, extract_common_elements, extract_common_elements_map, sigmoid
from cython_func import cy_DDA_2D

from subroutine import  mpl_config
plt.rcParams.update(mpl_config)
# 楕円の近似多角形を何度刻みで得るか決める角度リスト
theta_list = np.arange(np.deg2rad(0), np.deg2rad(360) + np.deg2rad(30) / 10, np.deg2rad(30)) #0~360を30刻みで

# vartical(y座標)は[0]、horizontal(x座標)は[1]
# 以下約500行はmapクラスの記述
class map():
    def __init__(self, var_range, hor_range, grid_pitch=None, start=None, end=None, Miss_Corner=False):
        # input parameter
        self.start = start
        self.end = end
        self.grid_pitch = grid_pitch
        self.var_range = var_range
        self.hor_range = hor_range
        self.Miss_Corner = Miss_Corner

        # input parameter
        self.obstacle_dict = {}
        self.obstacle_node = np.empty((0,2)) #0行2列の空配列。後で行を追加していくときに0行で初期化する

        # # set the main maze
        # self.maze = np.zeros((len(self.var_range), len(self.hor_range)), dtype = 'int')

        # obstacle info

    # 読み込んだxy座標(緯度経度ではない）を基に障害物の線とノードをマップに追加し、最終的に地形マップを返します。gridpitchは400で5mと定義されている。
    def GenerateMapFromCSV(file, grid_pitch):
        # 地形情報(obstacle)の情報が記載されたファイル読み込み
        df = pd.read_csv(file)
        var = np.array(df['x [m]'])
        hor = np.array(df['y [m]'])
        obstacles = np.stack([var, hor] ,1) # shape(N,)の2つの1次元配列同士を、axis=1を指定してnp.stack()で重ねると、それぞれの配列の要素を列として重ねたshape(N, 2)の2次元配列となります。


        #仮の範囲を設定。RoundRange関数は後で出てくる
        var_min_round = map.RoundRange(None, np.amin(var), grid_pitch, 'min') #最小値を下の数値に丸めることで余裕を持たせることができる
        var_max_round = map.RoundRange(None, np.amax(var), grid_pitch, 'max') #最大値を上の数値に丸めることで余裕を持たせることができる
        hor_min_round = map.RoundRange(None, np.amin(hor), grid_pitch, 'min')
        hor_max_round = map.RoundRange(None, np.amax(hor), grid_pitch, 'max')

        # 刻み幅と仮の範囲からグリッドを設定。最大値を含めるために最大値より少し大きい値を設定
        var_range = np.arange(var_min_round, var_max_round+grid_pitch/10, grid_pitch)
        hor_range = np.arange(hor_min_round, hor_max_round+grid_pitch/10, grid_pitch)

        target_map = map(var_range, hor_range, grid_pitch, Miss_Corner = True)
        map.AddObstacleLine(target_map, obstacles, 'berth')
        map.AddObstacleNode(target_map, obstacles, 'berth')
        return target_map

    def ShowMap(self, filename = None, SD = None):
        fig = plt.figure(figsize=(4*1.414, 4*1.414), dpi=300, linewidth=0, edgecolor='w')
        ax = fig.add_subplot(111)

        # show obstacle lines
        for key in self.obstacle_dict:
            ax.plot(self.obstacle_dict[key][:,1], self.obstacle_dict[key][:,0], color = 'k', ls='-', lw = 0.8)
            # for i in range(len(self.obstacle_dict[key])-1):
            #     ax.annotate('', xy=[self.obstacle_dict[key][i+1,1],self.obstacle_dict[key][i+1,0]], 
            #                     xytext=[self.obstacle_dict[key][i,1],self.obstacle_dict[key][i,0]],
            #     arrowprops=dict(shrink=0, width=1, headwidth=8, 
            #                     headlength=10, connectionstyle='arc3',
            #                     facecolor='gray', edgecolor='gray')
            #    )

        # show obstacle nodes
        # for i in range(len(self.obstacle_node)):
        #     ax.fill_between(
        #         self.hor_range,
        #         self.obstacle_node[i,0],
        #         self.obstacle_node[i,0]+self.grid_pitch,
        #         where = (self.hor_range >= self.obstacle_node[i,1]) &
        #                 (self.hor_range <= self.obstacle_node[i,1]+self.grid_pitch),
        #         facecolor='gray',
        #         alpha=0.8,
        #         zorder = 2
        #         )

        # show tmp array
        if 'tmp_array1' in dir(self):
            for i in range(len(self.tmp_array1)):
                ax.fill_between(
                    self.hor_range,
                    self.tmp_array1[i,0],
                    self.tmp_array1[i,0]+self.grid_pitch,
                    where = (self.hor_range >= self.tmp_array1[i,1]) &
                            (self.hor_range <= self.tmp_array1[i,1]+self.grid_pitch),
                    facecolor='g',
                    alpha=0.8,
                    zorder = 2
                    )     
        if 'tmp_array2' in dir(self):
            for i in range(len(self.tmp_array2)):
                ax.fill_between(
                    self.hor_range,
                    self.tmp_array2[i,0],
                    self.tmp_array2[i,0]+self.grid_pitch,
                    where = (self.hor_range >= self.tmp_array2[i,1]) &
                            (self.hor_range <= self.tmp_array2[i,1]+self.grid_pitch),
                    facecolor='r',
                    alpha=0.8,
                    zorder = 3
                    )
        # show open and closed list
        if 'open_array' in dir(self):
            for i in range(len(self.open_array)):
                ax.fill_between(
                    self.hor_range,
                    self.open_array[i,0],
                    self.open_array[i,0]+self.grid_pitch,
                    where = (self.hor_range >= self.open_array[i,1]) &
                            (self.hor_range <= self.open_array[i,1]+self.grid_pitch),
                    facecolor='y',
                    alpha=0.8,
                    zorder = 2
                    )

        if 'closed_array' in dir(self):
            for i in range(len(self.closed_array)):
                ax.fill_between(
                    self.hor_range,
                    self.closed_array[i,0],
                    self.closed_array[i,0]+self.grid_pitch,
                    where = (self.hor_range >= self.closed_array[i,1]) &
                            (self.hor_range <= self.closed_array[i,1]+self.grid_pitch),
                    facecolor='gray',
                    alpha=0.8,
                    zorder = 2
                    )

        # show start and goal position
        if 'start_xy' in dir(self):
            ax.scatter(self.start_xy[0,1]+0.5, self.start_xy[0,0]+0.5, color = 'k', lw = 1.5, zorder = 2)
            ax.text(self.start_xy[0,1]+0.5, self.start_xy[0,0]+0.6, 'start', va = 'bottom', ha = 'center', color = 'k')
        if 'end_xy' in dir(self):
            ax.scatter(self.end_xy[0,1]+0.5,   self.end_xy[0,0]+0.5,   color = 'k', lw = 1.5, zorder = 2)
            ax.text(self.end_xy[0,1]+0.5, self.end_xy[0,0]+0.6, 'end', va = 'bottom', ha = 'center', color = 'k')

        # show the Astar path
        if 'path_xy' in dir(self):
            ax.plot(   self.path_xy[:,1]+0.5, self.path_xy[:,0]+0.5, lw = 1.5, color = 'r')

        # show ship domain

        if 'psi' in dir(self) and not SD == None:
            for j in range(len(self.path_xy)):
                if j % 5 == 0:
                    distance = ((self.path_xy[j,1]+0.5 - self.end_xy[0,1]+0.5) ** 2 + (self.path_xy[j,0]+0.5 - self.end_xy[0,0]+0.5) ** 2) ** (1/2)
                    speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
                    r_list = []
                    for theta_i in theta_list:
                        r_list.append(SD.distance(speed, theta_i))
                    # ax.plot([self.path_xy[j,1]+0.5, self.path_xy[j,1]+0.5+100*np.sin(self.psi[j])],
                    #         [self.path_xy[j,0]+0.5, self.path_xy[j,0]+0.5+100*np.cos(self.psi[j])],
                    #         lw = 1.5, color = 'y'
                    #         )
                    ax.plot(self.path_xy[j,1]+ 0.5 + r_list[:] * np.sin(theta_list[:] + self.psi[j]),
                            self.path_xy[j,0]+ 0.5 + r_list[:] * np.cos(theta_list[:] + self.psi[j]), 
                            lw = 0.5, color = 'g', ls = '--'
                            )


        # ax.scatter(self.path_xy[:,1]+0.5, self.path_xy[:,0]+0.5, lw = 0.1, color = 'r')
        # show the smoothed path
        if 'smoothed_path_xy' in dir(self):
            ax.plot(   self.smoothed_path_xy[:,1]+0.5, self.smoothed_path_xy[:,0]+0.5, lw = 1.5, color = 'b', zorder = 2)

        ax.set_xlim(self.hor_range[0], self.hor_range[-1]+self.grid_pitch)
        ax.set_ylim(self.var_range[0], self.var_range[-1]+self.grid_pitch)

        ax.set_xticks(self.hor_range[::20])
        ax.set_yticks(self.var_range[::20])

        ax.set_xticklabels(self.hor_range[::20], rotation=90)
        ax.set_yticklabels(self.var_range[::20], rotation=0)


        ax.set_xlabel(r'$Y\,\rm{[m]}$')
        ax.set_ylabel(r'$X\,\rm{[m]}$')

        if 'txt' in dir(self):
            ax.text(self.hor_range[-1]+self.grid_pitch, self.var_range[-1]+self.grid_pitch,self.txt,  ha = 'right', va = 'top')

        plt.grid(which='major',color='k',linestyle='--',linewidth = 0.4,alpha = 0.5)
        # plt.grid(which='minor',color='k',linestyle='--',linewidth = 0.3,alpha = 0.5)

        plt.gca().set_aspect('equal', adjustable='box')

        plt.tight_layout()

        if filename == None:
            fig.savefig('test_astar10.png', pad_inches=0.05)
        else:
            fig.savefig(filename, pad_inches=0.05)
        plt.close()

    # 線分（start から end まで。任意に設定できる）とグリッド(網目、縦横の線)の交点を計算し、通過するグリッドセル（ノード）を特定するための関数
    # @profile
    def DetictCollision(self, start, end):
        # ax + by +c = 0 縦軸がxであることに注意。普通のxとyを入れ替えたものだと思えばいい
        a = end[1] - start[1]
        b = start[0] - end[0]
        c = start[1]*end[0] - start[0]*end[1]

        # 手順Ⅰ：「線分」と「縦軸(=x軸)に並行なグリッド」の交点を求める
        var_list = []
        for var_i in self.var_range:
            if var_i >= min(start[0], end[0]) and var_i <= max(start[0], end[0]):
                var_list.append(var_i)
        var_array = np.empty((len(var_list),2))
        var_array[:,0] = var_list[:]
        
        # 線分が縦軸(x軸)に平行かどうかでvar_array[:,1]の値を決め方を変えている
        if b == 0:
            var_array[:,1] = start[1]
        else:
            var_array[:,1] = -(a/b) * var_array[:,0] - (c/b)

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
        ans_array = np.concatenate([hor_array, var_array]) #配列を統合し、ans_array に格納。horもvarも2次元のndarray。行と列成分を合わせているとかではない(これをしたいならnp.stack)。求めた交点を一元管理し重複を削除するための前準備
        ans_array = np.unique(ans_array, axis = 0) # 重複する交点を削除。さらに順番はそのままではなくソートされる
        
        ans_array = np.block([[ans_array], [start], [end]]) # ans_array に start と end を追加。これもconcatenateやstackと似たようなもの

        # ans_arrayの内，格子点に重なる点を抽出（start，endが格子点の場合も考慮される）
        grid_point_array = np.empty((0,2))
        
        # ans_array
        for i in range(len(ans_array)):
            # ans_array の各交点に対して、その x座標が self.var_range（x軸方向のグリッド範囲）に含まれ、かつ y座標が self.hor_range（y軸方向のグリッド範囲）に含まれるかどうかをチェック
            if ans_array[i,0] in self.var_range and ans_array[i,1] in self.hor_range:
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

    # @profile
    def DDA_2D(self, start, end, pitch, n_loops):
        '''
        Output the voxels of self.grid map that the line segment passes
        This algorithm is the application of 2D DDA, explained in the URLs below:
        https://kanamori.cs.tsukuba.ac.jp/jikken/inner/uniform_grid.pdf
        http://www.cse.yorku.ca/~amana/research/grid.pdf
        '''
        ### initialize
        start_node =np.array([pitch * math.floor(start[0]/pitch), pitch * math.floor(start[1]/pitch)])
        end_node =np.array([pitch * math.floor(end[0]/pitch), pitch * math.floor(end[1]/pitch)])
        # end_node = self.FindNodeOfThePoint(end)[0]
        direction_vector = end - start
        step_x = np.sign(direction_vector[0])
        step_y = np.sign(direction_vector[1])

        # cal initial t
        if direction_vector[0] > 0: # up
            tx = (start_node[0] + pitch - start[0]) / abs(direction_vector[0])
        elif direction_vector[0] < 0: # down
            tx = (start[0] - start_node[0]) / abs(direction_vector[0])
        else:
            tx = np.inf

        if direction_vector[1] > 0: # right
            ty = (start_node[1] + pitch - start[1]) / abs(direction_vector[1])
        elif direction_vector[1] < 0: # left
            ty = (start[1] - start_node[1]) / abs(direction_vector[1])
        else:
            ty = np.inf

        ### main loop
        current_node = np.array([start_node])
        desired_node_list = np.empty((n_loops, 2))
        # while not np.array_equal(current_node[0,:], end_node):
        for i in range(n_loops):
            desired_node_list[i] = current_node
            if np.array_equal(current_node[0,:], end_node):
                desired_node_list = desired_node_list[:i+1]
                break
            if tx < ty:
                tx += pitch / abs(direction_vector[0])
                current_node[0,0] += step_x * pitch
            elif ty < tx:
                ty += pitch /abs(direction_vector[1])
                current_node[0,1] += step_y * pitch
            else:
                tx += pitch / abs(direction_vector[0])
                ty += pitch /abs(direction_vector[1])
                current_node[0,0] += step_x * pitch
                current_node[0,1] += step_y * pitch

        return desired_node_list

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

    # 障害物の頂点リストとグリッドピッチを用いて、障害物ノードのリストを更新する
    def AddObstacleNode(self, array, name = 'Noname'):
        inner_node = self.fill_inner_concave_obstacle(array, self.grid_pitch)
        # arrayは後で障害物の頂点を示すobstaclesという2次元配列が代入される。辺の数は頂点の数-1
        for i in range(len(array) - 1):
            tmp_array = self.DetictCollision(array[i,:], array[i+1,:])
            self.obstacle_node = np.concatenate([self.obstacle_node, tmp_array])
        self.obstacle_node = np.append(self.obstacle_node, inner_node, axis = 0)
        self.obstacle_node = np.unique(self.obstacle_node, axis = 0)
        self.obstacle_map = set([tuple(x) for x in self.obstacle_node]) # self.obstacle_node は2次元の numpy 配列で、各行がノードの座標[y,x]を示す。内包表記で各行(配列)が tuple に変換される。set は、重複する要素を持たないデータ構造。重複しないタプルが要素となったセットが作られる{(),(),()}

    def CancellDuplication(self, array):
        # 重複を削除して新しい配列を生成
        unique_array = np.unique(array, axis=0)
        return unique_array

    # 指定されたピッチ（grid_pitch）に基づいて、与えられた点の座標をグリッドの左下アンカーに揃えることで、点が属するノードを特定します。
    def FindNodeOfThePoint(self, point): # 与えられた点がどのノードに存在するか調べる point must be ndarray(1,2) 1行2列で(y,x)を表す
        node = np.empty((1,2))
        # var np.emptyで生成するときは0を使うと本当に0行となって存在しないリストを作ることになるが、要素を指定するときのインデクスは0から始まる
        node[0,0] = self.FloorByDesiredPitch(point[0], self.grid_pitch)
        #hor
        node[0,1] = self.FloorByDesiredPitch(point[1], self.grid_pitch)
        return node #アンカーは左下

    # 与えられた数を，任意の刻み幅で切り捨てる。これは、例えばグリッド状の座標を計算する際などに有用です。FindNodeThePointでgrid は grid_pitchに代入されている
    def FloorByDesiredPitch(self, num, grid):
        tmp = grid * math.floor(num/grid)
        return tmp
    # Astar用の０、１バイナリの地図を作る
    def SetMaze(self):
        maze_np = np.zeros((len(self.var_range), len(self.hor_range)), int)
        ### set the obstacle node
        # get the coordinate of obstacle node from self
        for i in range(len(self.obstacle_node)):
            var_i = np.where(self.var_range == self.obstacle_node[i, 0]) #np.where()の第二引数, 第三引数を省略すると、条件を満たす要素のインデックス（位置）が返される。
            hor_i = np.where(self.hor_range == self.obstacle_node[i, 1])
            maze_np[var_i, hor_i] = 1  #障害物は１で表す

        # 配列の転置
        maze_np = maze_np.T

        self.maze = maze_np.tolist()

    #使われていない
    def smoothing(self, path):
        smoothed_path = np.array([path[0,:]])
        completed = False
        while True:
            if completed == True:
                break
            for i in range(len(path)):
                CanGo = True
                passed_node_array = self.DetictCollision(smoothed_path[-1], path[-1-i])
                for j in range(len(passed_node_array)):
                    if (self.obstacle_node == passed_node_array[j]).all(axis=1).any() == True:
                        CanGo = False
                        break
                if CanGo == True:
                    print(str(smoothed_path[-1]) + ' to ' + str(path[-1-i]) + ' is ' + str(CanGo))
                    smoothed_path = np.append(smoothed_path, np.array([path[-1-i]]),axis = 0)
                    if i == 0:
                        completed = True
                        break
                    else:
                        break

        # smoothed_path = None
        return smoothed_path

    # @profile
    def fill_inner_domain(self, array, pitch):
        #debug
        # self.tmp_array1 = array
        # self.ShowMap()
        # arrayの最初の列の最小値から最大値にかけて，内部の格子点を数え上げる
        horizontal_list = np.arange(np.amin(array[:,0]), np.amax(array[:,0]) + pitch / 2 , pitch)
        target_array = np.empty((0,2))
        for i in range(len(horizontal_list)):
            tmp_array = array[array[:, 0] == horizontal_list[i]]
            if len(tmp_array) == 0:
                continue
            else:
                vertical_list = np.arange(np.amin(tmp_array[:,1]), np.amax(tmp_array[:,1]) + pitch / 2 , pitch)

            for j in range(len(vertical_list)):
                candidate_array = np.array([[horizontal_list[i], vertical_list[j]]])
                if not np.any(np.all(tmp_array == candidate_array, axis=1)):
                    target_array = np.append(target_array, candidate_array, axis = 0)
        # self.tmp_array2 = target_array
        # self.ShowMap()

        return target_array

    # @profile
    def ship_domain_cost(self, node, pitch, SD, weight):
        # 船の現在地からship domainを作成
        #ゴール地点から現在地までの距離distance
        distance = ((self.var_range[node.position[1]] - self.end_xy[0,1]) ** 2 + (self.hor_range[node.position[0]] - self.end_xy[0,1]) ** 2) ** (1/2)
        #distanceに基づいて速力逓減ガイドラインに基づき速度を決定
        speed = self.b_ave * distance ** (self.a_ave) + self.b_SD * distance ** (self.a_SD)
        #r_listは、各角度方向でのドメインの頂点までの距離
        r_list = []
        for theta_i in theta_list:
            r_list.append(SD.distance(speed, theta_i)) #distanceメソッドによりある角度での楕円の半径が得られるので、下のコードで30度刻みの半径が得られる）
        domain_xy = np.array([self.var_range[node.position[1]] + pitch / 2 + r_list[:] * np.cos(theta_list[:] + node.psi),
                              self.hor_range[node.position[0]] + pitch / 2 + r_list[:] * np.sin(theta_list[:] + node.psi)
                        ])
        domain_xy = domain_xy.T
        # 線分で表現したship domainと地形ノードとの接触判定（交差，または内部）
        # domainが通過したノードを判定
        contact_node = np.empty((0,2))
        domain_passed_node = np.empty((0,2))
        for i in range(len(domain_xy)-1):
            add_domain_passed_node = cy_DDA_2D(domain_xy[i,:], domain_xy[i+1,:], self.grid_pitch,len(self.var_range) * len(self.hor_range) + 1)
            # add_domain_passed_node = self.DetictCollision(domain_xy[i,:], domain_xy[i+1,:])
            domain_passed_node = np.append(domain_passed_node, add_domain_passed_node, axis = 0)
        domain_passed_node = self.CancellDuplication(domain_passed_node)
        # self.tmp_array1 = domain_passed_node
        # self.ShowMap()

        # domain内部の格子点の数え上げ，加算
        domain_passed_node = np.append(domain_passed_node, self.fill_inner_domain(domain_passed_node, self.grid_pitch), axis = 0)

        # domainが通過したノードを記録
        contact_node = extract_common_elements(domain_passed_node, self.obstacle_node)

        # 重複した場合，重み係数と，接触ノード数に応じてコスト加算
        # 将来的には，重み係数と，現在地との距離に応じてコスト加算に変更する可能性あり
        cost = weight * len(contact_node)
        return cost

    # 凹地形の内部にある点群を調べる
    def fill_inner_concave_obstacle(self, array, pitch):
        inner_array = np.empty((0,2))
        var_min_round = self.RoundRange(np.amin(array[:,0]), pitch, 'min')

        var_list = np.arange(self.RoundRange(np.amin(array[:,0]), pitch, 'min'), np.amax(array[:,0]) + pitch / 2, pitch) #最大値をamaxにしてしまうと最大値が含まれないから、+pitch/2して最大値まで含めるようにしている
        hor_list = np.arange(self.RoundRange(np.amin(array[:,1]), pitch, 'min'), np.amax(array[:,1]) + pitch / 2, pitch)

        for vi in range(len(var_list)):
            for hi in range(len(hor_list)):
                # Trueで点が内部にある判定
                if self.crossing_number(np.array([var_list[vi], hor_list[hi]]), array):
                    inner_array = np.append(inner_array, np.array([[var_list[vi], hor_list[hi]]]), axis = 0) #axis=0で新たな行として追加する場合は列数が、axis=1で新たな列として追加する場合は行数が、元の配列と一致している必要がある。足りない部分が欠損値NaNで埋められたりはしない。

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

    def line_of_sight(self, parent, child, pitch):
        #pitch/2 を追加して、ノードの中心を視線の始点と終点にします。
        los_xy = np.array([[self.var_range[parent.position[1]] + pitch / 2 , self.hor_range[parent.position[0]] + pitch / 2 ],
                            [self.var_range[child.position[1]] + pitch / 2 , self.hor_range[child.position[0]] + pitch / 2 ]
                        ])
        #LOSが通過したノードと地形ノードとの接触判定
        los_passed_node = cy_DDA_2D(los_xy[0,:], los_xy[1,:], self.grid_pitch,len(self.var_range) * len(self.hor_range) + 1)
        #視線が通過するノードと障害物ノードの共通要素（接触しているノード）を抽出
        contact_node = extract_common_elements_map(set([tuple(x) for x in los_passed_node]), self.obstacle_map)
        if len(contact_node) == 0:
            return True #clear
        else:
            return False #obstacle exist

    def hit_obstacle_box(self, start, end):
        direction_vector = (end - start)
        direction_vector_unit = direction_vector / np.linalg.norm(direction_vector)

        for obstacle_key in self.obstacle_dict:
            obstacle_array = self.obstacle_dict[obstacle_key]
            min_xy = [np.amin(obstacle_array[:,0]), np.amin(obstacle_array[:,1])]
            max_xy = [np.amax(obstacle_array[:,0]), np.amax(obstacle_array[:,1])]

            t_near = 0
            t_far = np.inf

            # 各方向でのt区間計算
            for i in range(len(min_xy)):
                if direction_vector_unit[i] == 0:
                    if not min_xy[i] <= direction_vector_unit[i] and direction_vector_unit[i] <= max_xy[i]:
                        break
                else:
                    if direction_vector[i] > 0: # up of right
                        tt_near = (min_xy[i] - start[i]) / direction_vector_unit[i]
                        tt_far = (max_xy[i] - start[i]) / direction_vector_unit[i]

                    else: # down or left
                        tt_near = (max_xy[i] - start[i]) / direction_vector_unit[i]
                        tt_far = (min_xy[i] - start[i]) / direction_vector_unit[i]

                    if t_near < tt_near:
                        t_near  =  tt_near
                    if t_far > tt_far:
                        t_far = tt_far

                    if t_far < t_near:
                        break
                return True
        return False
    
    def pixel_Dubins_curve(self, start, end, pitch, n_loops):
        return pitch

class ShipDomain_Wang():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initial_setting(self, ship_status, U_min = 1, U_max = 6, B_max = 3.08 * ship_status.Lpp):
        self.gf_max = 0.75 * B_max - 0.5 * ship_status.Lpp + 0.5 * ship_status.Lpp
        self.gf_min = 0.25 * ship_status.Lpp + 0.5 * ship_status.Lpp
        self.ga_max = 0.5 * B_max - 0.5 * ship_status.Lpp + 0.5 * ship_status.Lpp
        self.ga_min = 0.25 * ship_status.Lpp + 0.5 * ship_status.Lpp
        self.gy_max = 0.25 * B_max - 0.5 * ship_status.B + 0.5 * ship_status.B
        self.gy_min = ship_status.B + 0.5 * ship_status.B
        self.U_min = U_min
        self.U_max = U_max

    def distance(self, speed, theta):
        if speed > self.U_max:
            gf = self.gf_max
            ga = self.ga_max
            gp = self.gy_max
            gs = self.gy_max
        elif speed < self.U_min:
            gf = self.gf_min
            ga = self.ga_min
            gp = self.gy_min
            gs = self.gy_min
        else:
            gf = self.gf_min + (self.gf_max - self.gf_min) * (speed - self.U_min) / (self.U_max - self.U_min)
            ga = self.ga_min + (self.ga_max - self.ga_min) * (speed - self.U_min) / (self.U_max - self.U_min)
            gp = self.gy_min + (self.gy_max - self.gy_min) * (speed - self.U_min) / (self.U_max - self.U_min)
            gs = self.gy_min + (self.gy_max - self.gy_min) * (speed - self.U_min) / (self.U_max - self.U_min)

        if np.cos(theta) >= 0:
            gx = gf
        else:
            gx = ga
        if np.sin(theta) >= 0:
            gy = gs
        else:
            gy = gp
        r = gx * gy / (((gx * np.sin(theta)) ** 2 + (gy * np.cos(theta)) ** 2) ** (1/2))
        return r
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
    #theta方向の楕円の半径Gを返すメソッド（近似SDを用いるときに30度刻みで半径を得る）
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
class ShipDomain_Fujii():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def initial_setting(self, ship_status):
        self.gf = ship_status.Lpp * 6.4
        self.ga = ship_status.Lpp * 1.6
        self.gp = ship_status.Lpp * 1.6
        self.gs = ship_status.Lpp * 1.6

    def distance(self, theta):
        gf = self.gf
        ga = self.ga
        gp = self.gp
        gs = self.gs
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
class ShipDomain_shape():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def initial_setting(self, ship_status):
        self.gf = ship_status.Lpp / 2
        self.ga = ship_status.Lpp / 2
        self.gp = ship_status.B   / 2
        self.gs = ship_status.B   / 2

    def distance(self, speed, theta):
        gf = self.gf
        ga = self.ga
        gp = self.gp
        gs = self.gs
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
