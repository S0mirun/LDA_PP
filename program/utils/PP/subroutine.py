### The common functions and classes are saved in this file

import pandas as pd
import coord_conv
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import matplotlib.cm as cm


#Matplotlibのグラフ作成におけるスタイル設定を行っています。具体的には、mpl_configという辞書に各種設定を定義し、plt.rcParams.update(mpl_config)でその設定をMatplotlibに適用しています。
mpl_config =     {"lines.linestyle":"dotted",
                  'font.family'       : 'Times New Roman',
                  'xtick.direction'   : 'in',
                  'ytick.direction'   : 'in',
                  'xtick.major.width' : 1.0,
                  'ytick.major.width' : 1.0,
                  'mathtext.fontset'  : 'stix',
                  'font.size'         : 15,
                  'axes.linewidth'    : 1.0,
                  "legend.framealpha" : 1.0,
                  "xtick.minor.visible" : True,  #x軸補助目盛りの追加
                  "ytick.minor.visible" : True,  #y軸補助目盛りの追加
                  "legend.fancybox" : False,  # 丸角OFF
                  "legend.framealpha" : 1,  # 透明度の指定、0で塗りつぶしなし
                  "legend.edgecolor" : 'black',  # edgeの色を変更
                  "legend.markerscale" : 5 #markerサイズの倍率
}
plt.rcParams.update(mpl_config) #rcParamsとは、Matplotlibのランタイム設定（runtime configuration）を管理する辞書です。この辞書を使って、グラフのスタイルやレイアウトに関するさまざまな設定を一元管理することができます。rcParamsを使うことで、グラフの表示方法を簡単にカスタマイズできます。

# 任意の形状の点の集合を指定した回転角度と平行移動量に基づいて変換することができます。例えば、船の形状を表す座標を回転させたり移動させたりする際に便利です。
# ship_shapeを返す関数は下のload_target_ship_shape_parameter関数
def ship_shape_fix(shape, r, x, y, psi):
    x_fix = np.cos(shape[:] + psi) * r[:] + x
    y_fix = np.sin(shape[:] + psi) * r[:] + y
    return x_fix, y_fix

# class of port and bay
class overall:
    gridspace = 0
    minY_lat = 20.2
    maxY_lat = 45.4
    minY_long = 135.0
    maxY_long = 122.5
    name = '_all'
    type = 'all'
    
class sakai_bay:
    gridspace = 1
    minY_lat = 34.3
    maxY_lat = 34.8
    minY_long = 135.0
    maxY_long = 135.5
    name = '_Osaka'
    type = 'bay'
    class port1:
        gridspace = 2
        minY_lat = 34.565
        maxY_lat = 34.610
        minY_long = 135.380
        maxY_long = 135.430
        name = '_Osaka_port1'
        showname = 'port1'
        type = 'port'
    class port1ABC: #ABCとは？
        gridspace = 3
        minY_lat = 34.574
        maxY_lat = 34.577
        minY_long = 135.425
        maxY_long = 135.430
        name = '_Osaka_port1ABC'
        showname = 'port1ABC'
        type = 'port'
    class port1A:
        gridspace = 4
        minY_lat = 34.5763
        maxY_lat = 34.5768
        minY_long = 135.4275
        maxY_long = 135.4280
        name = '_Osaka_port1A'
        showname = 'port1A'
        type = 'port'
    class port1B:
        gridspace = 4
        minY_lat = 34.5757
        maxY_lat = 34.5762
        minY_long = 135.4272
        maxY_long = 135.4280
        name = '_Osaka_port1B'
        showname = 'port1B'
        type = 'port'
    class port1C:
        gridspace = 4
        minY_lat = 34.5742
        maxY_lat = 34.5748
        minY_long = 135.4277
        maxY_long = 135.4282
        name = '_Osaka_port1C'
        showname = 'port1C'
        type = 'port'
    class port2:
        gridspace = 2
        minY_lat = 34.650
        maxY_lat = 34.690
        minY_long = 135.170
        maxY_long = 135.210
        name = '_Osaka_port2'
        showname = 'port2'
        type = 'port'
    class port3:
        gridspace = 2
        minY_lat = 34.380
        maxY_lat = 34.430
        minY_long = 135.200
        maxY_long = 135.250
        name = '_Osaka_port3'
        showname = 'port3'
        type = 'port'
    class port3ABC:
        gridspace = 3
        minY_lat = 34.415
        maxY_lat = 34.422
        minY_long = 135.235
        maxY_long = 135.241
        name = '_Osaka_port3ABC'
        showname = 'port3ABC'
        type = 'port'    
    class port3A:
        gridspace = 4
        minY_lat = 34.4192
        maxY_lat = 34.4200
        minY_long = 135.2393
        maxY_long = 135.2398
        name = '_Osaka_port3A'
        showname = 'port3A'
        type = 'port'
    class port3B:
        gridspace = 4
        minY_lat = 34.4185
        maxY_lat = 34.4190
        minY_long = 135.2380
        maxY_long = 135.2386
        name = '_Osaka_port3B'
        showname = 'port3B'
        type = 'port'
    class port3C:
        gridspace = 4
        minY_lat = 34.4178
        maxY_lat = 34.4182
        minY_long = 135.2370
        maxY_long = 135.2375
        name = '_Osaka_port3C'
        showname = 'port3C'
        type = 'port'
    port_list = [port1, port2, port3]
    
class yokkaichi_bay:
    gridspace = 1
    minY_lat = 34.9
    maxY_lat = 35.0
    minY_long = 136.63
    maxY_long = 136.75
    name = '_Yokkaichi'
    type = 'bay'
    class port1:
        gridspace = 2
        minY_lat = 34.950
        maxY_lat = 34.975
        minY_long = 136.635
        maxY_long = 136.665
        name = '_Yokkaichi_port1'
        showname = 'port1'
        type = 'port'
    class port1A:
        gridspace = 4
        minY_lat = 34.9714
        maxY_lat = 34.9722
        minY_long = 136.650
        maxY_long = 136.651
        name = '_Yokkaichi_port1A'
        showname = 'port1A'
        type = 'port'
    class port1B:
        gridspace = 4
        minY_lat = 34.969
        maxY_lat = 34.9700
        minY_long = 136.649
        maxY_long = 136.650
        name = '_Yokkaichi_port1B'
        showname = 'port1B'
        type = 'port'
    class port1C:
        gridspace = 4
        minY_lat = 34.965
        maxY_lat = 34.9658
        minY_long = 136.648
        maxY_long = 136.6485
        name = '_Yokkaichi_port1C'
        showname = 'port1C'
        type = 'port'
    class port1D:
        gridspace = 4
        minY_lat = 34.960
        maxY_lat = 34.9615
        minY_long = 136.646
        maxY_long = 136.648
        name = '_Yokkaichi_port1D'
        showname = 'port1D'
        type = 'port'    
    class port1E:
        gridspace = 4
        minY_lat = 34.959
        maxY_lat = 34.960
        minY_long = 136.6455
        maxY_long = 136.6475
        name = '_Yokkaichi_port1E'
        showname = 'port1E'
        type = 'port'
    class port2:
        gridspace = 2
        minY_lat = 34.935
        maxY_lat = 34.952
        minY_long = 136.635
        maxY_long = 136.655
        name = '_Yokkaichi_port2'
        showname = 'port2'
        type = 'port'
    class port2A:
        gridspace = 4
        minY_lat = 34.9445
        maxY_lat = 34.9452
        minY_long = 136.641
        maxY_long = 136.6418
        name = '_Yokkaichi_port2A'
        showname = 'port2A'
        type = 'port'
    class port2B:
        gridspace = 4
        minY_lat = 34.944
        maxY_lat = 34.945
        minY_long = 136.6419
        maxY_long = 136.6425
        name = '_Yokkaichi_port2B'
        showname = 'port2B'
        type = 'port'
    class port2C:
        gridspace = 4
        minY_lat = 34.9405
        maxY_lat = 34.9415
        minY_long = 136.640
        maxY_long = 136.641
        name = '_Yokkaichi_port2C'
        showname = 'port2C'
        type = 'port'
    port_list = [port1A, port1B, port1C, port1D, port1E, port2A, port2B, port2C]
    
class Tokyo_bay:
    gridspace = 1
    minY_lat = 35.2
    maxY_lat = 35.7
    minY_long = 139.6
    maxY_long = 140.2
    name = '_Tokyo'
    type = 'bay'
    class port1:
        gridspace = 2
        minY_lat = 35.575
        maxY_lat = 35.615
        minY_long = 140.06
        maxY_long = 140.12
        name = '_Tokyo_port1'
        showname = 'port1'
        type = 'port'
    class port1A:
        gridspace = 3
        minY_lat = 35.600
        maxY_lat = 35.605
        minY_long = 140.095
        maxY_long = 140.100
        name = '_Tokyo_port1A'
        showname = 'port1A'
        type = 'port'
    class port1B:
        gridspace = 3
        minY_lat = 35.595
        maxY_lat = 35.6025
        minY_long = 140.075
        maxY_long = 140.085
        name = '_Tokyo_port1B'
        showname = 'port1B'
        type = 'port'
    class port1C:
        gridspace = 3
        minY_lat = 35.5925
        maxY_lat = 35.6025
        minY_long = 140.1025
        maxY_long = 140.1075
        name = '_Tokyo_port1C'
        showname = 'port1C'
        type = 'port'
    class port1D:
        gridspace = 3
        minY_lat = 35.594
        maxY_lat = 35.596
        minY_long = 140.102
        maxY_long = 140.103
        name = '_Tokyo_port1D'
        showname = 'port1D'
        type = 'port'
    class port1E:
        gridspace = 3
        minY_lat = 35.5935
        maxY_lat = 35.5945
        minY_long = 140.1005
        maxY_long = 140.1015
        name = '_Tokyo_port1E'
        showname = 'port1E'
        type = 'port'
    class port1F:
        gridspace = 3
        minY_lat = 35.590
        maxY_lat = 35.595
        minY_long = 140.110
        maxY_long = 140.115
        name = '_Tokyo_port1F'
        showname = 'port1F'
        type = 'port'
    class port2:
        gridspace = 2
        minY_lat = 35.52
        maxY_lat = 35.57
        minY_long = 140.05
        maxY_long = 140.10
        name = '_Tokyo_port2'
        showname = 'port2'
        type = 'port'    
    class port2A:
        gridspace = 3
        minY_lat = 35.5425
        maxY_lat = 35.5475
        minY_long = 140.0775
        maxY_long = 140.0825
        name = '_Tokyo_port2A'
        showname = 'port2A'
        type = 'port'    
    class port2B:
        gridspace = 4
        minY_lat = 35.533
        maxY_lat = 35.5335
        minY_long = 140.0552
        maxY_long = 140.0557
        name = '_Tokyo_port2B'
        showname = 'port2B'
        type = 'port'
    class port2C:
        gridspace = 4
        minY_lat = 35.5325
        maxY_lat = 35.533
        minY_long = 140.0563
        maxY_long = 140.0568
        name = '_Tokyo_port2C'
        showname = 'port2C'
        type = 'port'
    class port2D:
        gridspace = 4
        minY_lat = 35.53225
        maxY_lat = 35.53275
        minY_long = 140.0567
        maxY_long = 140.0572
        name = '_Tokyo_port2D'
        showname = 'port2D'
        type = 'port'
    class port3:
        gridspace = 2
        minY_lat = 35.50
        maxY_lat = 35.55
        minY_long = 139.90
        maxY_long = 139.95
        name = '_Tokyo_port3'
        showname = 'port3'
        type = 'port'
    port_list = [port1A, port1B, port1C, port1D, port1E, port1F, port2A, port2B, port2C, port2D]
    
class else_bay:
    class port1:
        gridspace = 4
        minY_lat = 34.354000
        maxY_lat = 34.355000
        minY_long = 133.8390
        maxY_long = 133.8400
        name = '_Else_port1'
        showname = 'port1'
        type = 'port'    
    class port2:
        gridspace = 2
        minY_lat = 41.810199 - 0.03
        maxY_lat = 41.810199 + 0.03
        minY_long = 140.704094 - 0.03
        maxY_long = 140.704094 + 0.03
        name = '_Else_port2'
        showname = 'port2'
        type = 'port'
    port_list = [port1, port2]



class latlon_parameter: #地球を楕円体と見た時のパラメータだけど、使ってない？
    POLE_RADIUS = 6356752    # 極半径(m)(短半径)
    EQUATOR_RADIUS = 6378137 # 赤道半径(m)(長半径)
    E = 0.081819191042815790 # 離心率
    E2= 0.006694380022900788 # 離心率の２乗

#船の形状を定義し、その形状を座標で表現するための準備。座標はデカルト座標（x, y）から極座標（r, θ）に変換され、船の形状の情報を含む辞書として保存さる。このクラスを使うことで、船の形状に関する座標情報を扱いやすくしている。
class ship_status: 
    Lpp = 103.8 #[m]
    B   = 16    #[m]
    ## input pp and status
    L_latlon   = Lpp * 0.0000090138
    B_latlon   = B   * 0.0000109544
    ## get X-Y position of ship shape (船を簡単な5角形で表している）
    X = np.array([-L_latlon/2, L_latlon/4, L_latlon/2,L_latlon/4,-L_latlon/2,-L_latlon/2])
    Y = np.array([-B_latlon/2,-B_latlon/2, 0,         B_latlon/2, B_latlon/2,-B_latlon/2])
    ## convert X-Y to r-theta（極座標に変換）
    R     = np.empty(6)
    THETA = np.empty(6)
    R[:]     = (X[:] ** 2 + Y[:] ** 2) ** (1/2)
    THETA[:] = np.arctan2(Y[:],X[:]) #返り値は-piからpiの間になる。第2象限、第3象限における角度も正しく取得できるので、極座標平面で考える場合はnp.arctan()よりもnp.arctan2()のほうが適当。
    
    #shapeは、船の形状を表す辞書
    shape = {
        'R':R[:],
        'THETA':THETA[:],
        'INTERVAL':12
    }

#300で地図の名前付けに使われてたりする
map_chip = [
    'honshu',
    'shikoku',
    'hokkaidou',
    'hokkaidou_bouhatei1',
    'hokkaidou_bouhatei2',
    'hokkaidou_bouhatei3',
    'hokkaidou_bouhatei4',
    'sadoshima',
    'awazishima',
    'island_sakai1',
    'island_sakai2',
    'kyusyu',
    'shimoshima_kyusyu',
    'kasumigaura',
    'island_tokyo1',
    'island_tokyo2',
    'island_tokyo3',
    'keiyou_seabeath'
]

# filename setting by OS 　使ってない？
OS_list = [
    'unknown',
    'run',
    'rest',
    'beathing',
    'unbeathing',
    'lowspeed1',
    'lowspeed2'
]

#これも300で名前付けのために使われている
elements_list = [ # 0:停止 1:保針 2:横移動　3:斜航 4:その場回頭 5:旋回(変針)
    'Stop',
    'Keep',
    'Shift',
    'Drift',
    'PivotTurn',
    'Turn'

]

#緯度経度データを船の基準点と方位を基にしてXY平面上の座標に変換するためのもの
#関数の目的は、各点の緯度経度を基にして基準点からのXY座標を計算し、新しいデータフレームとして返すこと
def convert_to_xy(df, lat_origin, lon_origin, theta_berth):

    lonlat_df = df

    xy_array = np.empty((len(lonlat_df), 2))
    for i in range(len(lonlat_df)):
        longitude_point = lonlat_df.loc[lonlat_df.index[i], 'longitude']
        latitude_point = lonlat_df.loc[lonlat_df.index[i], 'latitude']
        longitude_array = np.array([lon_origin, longitude_point])
        latitude_array = np.array([lat_origin, latitude_point])
        distance_min, course_to_north = coord_conv.lat_lon.caldistco(longitude_array, latitude_array)
        distance_meter = distance_min * 1852
        course_radian = (course_to_north - theta_berth) * np.pi / 180
        x_point, y_point = distance_meter * np.cos(course_radian), distance_meter * np.sin(course_radian)
        # slite modification to match GPS value on experiment
        # x_point, y_point = x_point - 0.06352340616454502, y_point - 0.10797887276750123
        xy_array[i] = np.array([x_point, y_point])
    
    xy_df = pd.DataFrame(xy_array, columns = ['x [m]', 'y [m]'])
    return xy_df
    # xy_df.to_csv('test/{}_xy.csv'.format(name))

    # L = 103.8
    # xy_over_Lpp_df = xy_df / L
    # xy_over_Lpp_df.to_csv('test/{}_xy_over_Lpp.csv'.format(name))

# CSVファイルから船の形状データを読み込み、それを極座標（r, θ）に変換する。また、角度を調整し、指定された間隔（interval）と共に辞書として返す。めちゃ使われてる関数
def load_target_ship_shape_parameter(filepath, interval):
    df = pd.read_csv(filepath)
    X = df['x'].values
    Y = df['y'].values
    R     = np.empty(len(X))
    THETA = np.empty(len(X))
    R[:] = (X[:] ** 2 + Y[:] ** 2) ** (1/2)
    THETA[:] = np.arctan2(Y[:],X[:])
    ship_shape = {
        'R':R[:],
        'THETA':THETA[:] - np.pi/2,
        'INTERVAL':interval
    }
    return ship_shape

# 2つのNumPy配列 array1 と array2 に共通する要素を抽出する
# 関数の主な処理は、各配列の行をセット{}に変換し、セットの共通部分を求めてから、結果をNumPy配列に戻すこと
# graphで使っている。下の_mapヴァージョンも
def extract_common_elements(array1, array2):
    # aset = set([tuple(x) for x in array1])
    # bset = set([tuple(x) for x in array2])
    # result = np.array([x for x in aset & bset])
    # NumPy配列をセットに変換して共通する要素を取得
    set_a = set(map(tuple, array1)) #ここでのタプルは単にタプルデータを入れることを表すもの
    set_b = set(map(tuple, array2))

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Calculate set intersection for unique rows in parallel
        common_rows_list = list(executor.map(np.array, set_a & set_b))#＆はandとは異なり、引数のビット単位での論理積を返す

    result = np.array(common_rows_list)

    return result

def extract_common_elements_map(map1, map2):

    # Calculate set intersection for unique rows in parallel
    common_rows_list = list(map(np.array, map1 & map2))

    result = np.array(common_rows_list) # リスト形式だった共通部分がNumPy配列形式になる。

    return result

### 300 class and function　グラフの設定を管理するためのクラスを定義
class FigSetting():
    # 300でGuidelineを図示する際の各種設定。300でも定義されている。ここで定義されているものは301で使用されている。
    # クラス変数=初期設定
    PortList = [
        sakai_bay.port1A,
        sakai_bay.port1B,
        Tokyo_bay.port2B,
        Tokyo_bay.port2C,
        yokkaichi_bay.port1A,
        yokkaichi_bay.port2B,
        else_bay.port1,
        else_bay.port2
    ]

    MeshSize = 50 #[m]
    LimitLength = 1000 #[m] 増加する場合はGuideline再作成が必要
    ax = None
    color_map =cm.Wistia #cm.Blues #cm.YlGnBu 
    AngleWidth = np.deg2rad(1)
    AngleList = np.arange(0, 2*np.pi, AngleWidth)
    target_dir_root = 'output/300/'
    coordinate_system = 'rectangular'

    # TargetElements
    # WindVMin = 0 #[m/s]
    # WindVMin = 100 #[m/s]
    # WindThetaMin = 0 #[rad]
    # WindThetaMax = np.pi * 2 #[rad]
    # UminKts = 0
    # UmaxKts = 100
    # TargetElement = 1 # 0:停止 1:保針 2:横移動　3:斜航 4:その場回頭 5:旋回(変針)
    # ship position

    #　コンストラクタは可変長引数を用いてインスタンスごとに異なる設定を適用できるようにしている
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


        # TargetU = [self.UminKts*1852/3600,self.UmaxKts*1852/3600] # min, max, mps

def sigmoid(x, a, b, c): #シグモイド関数
	return a/(b + np.exp(c*x))
