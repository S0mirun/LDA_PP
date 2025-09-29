'''
指定した情報(航路，地形など)を図に出力するためのファイル
!気を付けること!
論文用の画像出力を行う際、if __name__ == '__main__':直後で対象の港を選択する
最後のpngの出力先のパスを設定
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime as dt
import datetime

import sys
import os
import glob

from Input import CONVERT_LATLONG_FROM_DM_TO_D, GET_MIDSHIP_LATLONG
from subroutine import overall, sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay #if __name__ == '__main__':直後で対象の港を選択する箇所がある
from subroutine import ship_status, convert_to_xy, load_target_ship_shape_parameter, sigmoid, ship_shape_fix
from graph import ShipDomain_Wang, ShipDomain_proposal, ShipDomain_shape

#Matplotlibのグラフ作成におけるスタイル設定を行っています。具体的には、mpl_configという辞書に各種設定を定義し、plt.rcParams.update(mpl_config)でその設定をMatplotlibに適用しています。
from subroutine import mpl_config
plt.rcParams.update(mpl_config)

### Program to plot the simulated traj., real traj., and generated check points in one figure
## class setting
class Traj:
    def __init__(self):
        self.L = ship_status.Lpp
        self.B = ship_status.B
        self.color = 'gray'
        self.print_span = 60 #[sec]
    
    def set_params(self, **kwargs): #kwargsで渡された引数を基に、color、print_span、およびlabelの値を設定します。kwargs.getメソッドを使って、指定されたキーが存在するか確認し、存在すれば対応する属性を更新します。
        if not kwargs.get('color') is None:
            self.color = kwargs['color']
        if not kwargs.get('print_span') is None:
            self.print_span = kwargs['print_span']
        if not kwargs.get('label') is None:
            self.label = kwargs['label']



class RealTraj(Traj):
    def __init__(self):
        super().__init__()
    #
    def input_csv(self, filename_traj, filename_berth):
        # check the existance of csv file
        filelist = [filename_berth, filename_traj]
        for file in filelist:
            if os.path.isfile(file) == True:
                pass
            else:
                print(file + ' does not exist.')
                sys.exit()
        #
        df = pd.read_csv(filename_traj
                        , header=None, skiprows=1
                        , names = ['date','time', 'latitude', 'longitude','GPS_psi', 'gyro_psi', 'GPS_u', 'log_u', 'wind_dir', 'wind_v']
                        , encoding='shift-JIS'
                         )
        
        # initialize
        datetime_first = str(df['date'][0])         + ' ' + str(df['time'][0])
        datetime_last  = str(df['date'][len(df)-1]) + ' ' + str(df['time'][len(df)-1])

        #1分間隔でタイムスタンプを作成する。len(TIME)は何分かに等しい。1時間のデータなら長さは60になる
        TIME = pd.date_range(
        start = dt.datetime.strptime(datetime_first       , '%Y/%m/%d %H:%M:%S')
        , end = dt.datetime.strptime(datetime_last        , '%Y/%m/%d %H:%M:%S')
        , freq = '1min'
        )
        step_len = len(TIME) 
        NORTH_LATITUDE_DECIMAL = np.zeros(step_len)
        EAST_LONGITUDE_DECIMAL = np.zeros(step_len)
        HEAD_ANGLE_GPS         = np.zeros(step_len)
        HEAD_ANGLE_GYRO        = np.zeros(step_len)
        SHIP_SPEED_GPS         = np.zeros(step_len)
        SHIP_SPEED_LOG         = np.zeros(step_len)
        WIND_DIRECTION_DEG     = np.zeros(step_len)
        WIND_VELOCITY_MPS      = np.zeros(step_len)

        # replace string in dataframe to zero
        name = ['GPS_psi', 'gyro_psi', 'GPS_u', 'log_u', 'wind_dir', 'wind_v']
        for name in name:
            df[name] = pd.to_numeric(df[name],errors= 'coerce').fillna(0)
              
        # check the continuity of the AIS log, and interpolate loss
        # check that AIS not act at 23:59. If not, append the last log to dataframe
        if not df['time'][len(df)-1] == TIME[-1].strftime('%X'):
            df_last = pd.DataFrame(
                data = {
                    'date': [df['date'][len(df)-1]],
                    'time':[TIME[-1].strftime('%X')],
                    'latitude':[df['latitude'][len(df)-1]],
                    'longitude':[df['longitude'][len(df)-1]],
                    'GPS_psi':[df['GPS_psi'][len(df)-1]],
                    'gyro_psi':[df['gyro_psi'][len(df)-1]],
                    'GPS_u':[df['GPS_u'][len(df)-1]],
                    'log_u':[df['log_u'][len(df)-1]],
                    'wind_dir':[0.0],
                    'wind_v':[0.0]
                })
            df = pd.concat([df, df_last], ignore_index = True)
        error_sw = 1 # 0: AIS act, 1: AIS not act, 2: AIS not act from 00:00(AISの動作状態をチェックし、動作していない場合のデータを処理します。)
        ni  = 0
        nni = 0
        AIS_OFF = []
        for i in range(step_len):
            while True:
                date_i = int(dt.datetime.strptime(str(df['time'][ni]), '%H:%M:%S').strftime('%H%M%S'))
                if date_i % 100 == 0:
                    break
                else:
                    ni += 1
                    nni += 1

            if date_i == int(TIME[i].strftime('%H%M%S')):
                if error_sw == 2:
                    for j in range(i): #CONVERT_LATLONG_FROM_DM_TO_D: 緯度経度の形式を変換する関数。
                        NORTH_LATITUDE_DECIMAL[j] = CONVERT_LATLONG_FROM_DM_TO_D(df['latitude'][ni])
                        EAST_LONGITUDE_DECIMAL[j] = CONVERT_LATLONG_FROM_DM_TO_D(df['longitude'][ni])
                        HEAD_ANGLE_GPS[j]         = df['GPS_psi'][ni]
                        HEAD_ANGLE_GYRO[j]        = df['gyro_psi'][ni]
                        SHIP_SPEED_GPS[j]         = df['GPS_u'][ni]
                        SHIP_SPEED_LOG[j]         = df['log_u'][ni]
                        WIND_DIRECTION_DEG[j]     = df['wind_dir'][ni]
                        WIND_VELOCITY_MPS[j]      = df['wind_v'][ni]

                NORTH_LATITUDE_DECIMAL[i] = CONVERT_LATLONG_FROM_DM_TO_D(df['latitude'][ni])
                EAST_LONGITUDE_DECIMAL[i] = CONVERT_LATLONG_FROM_DM_TO_D(df['longitude'][ni])
                HEAD_ANGLE_GPS[i]         = df['GPS_psi'][ni]
                HEAD_ANGLE_GYRO[i]        = df['gyro_psi'][ni]
                SHIP_SPEED_GPS[i]         = df['GPS_u'][ni]
                SHIP_SPEED_LOG[i]         = df['log_u'][ni]
                WIND_DIRECTION_DEG[i]     = df['wind_dir'][ni]
                WIND_VELOCITY_MPS[i]      = df['wind_v'][ni]
                error_sw = 0
                ni += 1
            elif i == 0 or error_sw == 2:
                error_sw = 2
            else:
                NORTH_LATITUDE_DECIMAL[i] = NORTH_LATITUDE_DECIMAL[ni-(nni+1)]
                EAST_LONGITUDE_DECIMAL[i] = EAST_LONGITUDE_DECIMAL[ni-(nni+1)]
                HEAD_ANGLE_GPS[i]         = df['GPS_psi'][ni-(nni+1)]
                HEAD_ANGLE_GYRO[i]        = df['gyro_psi'][ni-(nni+1)]
                SHIP_SPEED_GPS[i]         = df['GPS_u'][ni-(nni+1)]
                SHIP_SPEED_LOG[i]         = df['log_u'][ni-(nni+1)]
                WIND_DIRECTION_DEG[i]     = df['wind_dir'][ni-(nni+1)]
                WIND_VELOCITY_MPS[i]      = df['wind_v'][ni-(nni+1)]
                error_sw = 1

            if error_sw == 1 or error_sw == 2: # AIS not act
                AIS_OFF.append(TIME[i])

        OFF_start = []
        OFF_end   = []
        for i in range(len(AIS_OFF)):
            if i == 0:
                OFF_start.append(AIS_OFF[i])
            elif not AIS_OFF[i] == AIS_OFF[i-1] + datetime.timedelta(minutes=1):
                OFF_end.append(AIS_OFF[i-1])
                OFF_start.append(AIS_OFF[i])
        if not len(AIS_OFF) == 0:
            OFF_end.append(AIS_OFF[-1])

        #GET_MIDSHIP_LATLONG: 中間船首緯度経度を計算する関数。
        MS_NORTH_LATITUDE_DECIMAL, MS_EAST_LONGITUDE_DECIMAL = GET_MIDSHIP_LATLONG(NORTH_LATITUDE_DECIMAL, EAST_LONGITUDE_DECIMAL, HEAD_ANGLE_GYRO, self.L, self.B)

        # convert latlong to XY (緯度経度データをXY座標に変換します。)
        df_berth = pd.read_csv(filename_berth)
        latlon = np.empty((len(MS_NORTH_LATITUDE_DECIMAL), 2))
        latlon[:, 0] = MS_NORTH_LATITUDE_DECIMAL[:]
        latlon[:, 1] = MS_EAST_LONGITUDE_DECIMAL[:]
        latlon_df = pd.DataFrame(data = latlon, columns= ['latitude', 'longitude'])
        xy = convert_to_xy(latlon_df, df_berth['Latitude'][0], df_berth['Longitude'][0], df_berth['Psi[deg]'][0]) #convert_to_xy: 緯度経度をXY座標に変換する関数。
        xy = xy.values

        #各種データ列を初期化し、必要な変換を行います。
        # TIMEは1分間隔なのでlen(TIME)は1時間のデータなら長さは60になる
        self.time    = np.empty(len(TIME))
        self.X       = np.empty(len(TIME)); self.u        = np.empty(len(TIME))
        self.Y       = np.empty(len(TIME)); self.vm       = np.empty(len(TIME))
        self.psi     = np.empty(len(TIME)); self.r        = np.empty(len(TIME))
        self.delta_d = np.empty(len(TIME)); self.n_prop_d = np.empty(len(TIME))
        self.delta_r = np.empty(len(TIME)); self.n_prop_r = np.empty(len(TIME))
        self.n_bt    = np.empty(len(TIME)); self.n_st     = np.empty(len(TIME))
        self.windd   = np.empty(len(TIME)); self.windv    = np.empty(len(TIME))
        self.psi_GPS = np.empty(len(TIME))
        self.psi_raw = np.empty(len(TIME))
        self.psi_GPS_raw = np.empty(len(TIME))

        #各種データ列（X、Y、船速、船首角度、風向、風速など）を計算し、適切な単位とフォーマットに変換します。1分ごとのタイムスタンプを設定します。
        for i in range(len(self.time)):
            self.time[i] = 60.0 * i
        self.X[:]     = xy[:,0]; self.u[:] = SHIP_SPEED_GPS[:] * 1852 / 3600
        self.Y[:]     = xy[:,1]
        self.psi[:]   = np.deg2rad(HEAD_ANGLE_GYRO[:] - df_berth['Psi[deg]'][0]) #radian
        # self.windd[:] = np.deg2rad(WIND_DIRECTION_DEG[:])- df_berth['Psi[deg]'][0]
        self.windd[:] = np.rad2deg(np.arctan2(np.sin(np.deg2rad(WIND_DIRECTION_DEG[:] - (df_berth['Psi[deg]'][0]))), np.cos(np.deg2rad(WIND_DIRECTION_DEG[i] - (df_berth['Psi[deg]'][0])))))

        self.windv[:] = WIND_VELOCITY_MPS[:]

        self.psi_GPS = np.deg2rad(HEAD_ANGLE_GPS[:] - df_berth['Psi[deg]'][0]) #radian
        
        self.u_knot = SHIP_SPEED_GPS[:]
        self.psi_raw = HEAD_ANGLE_GYRO[:] #degree
        self.psi_GPS_raw = HEAD_ANGLE_GPS[:] #degree
        
        



class SimlationTraj(Traj):
    def __init__(self):
        super().__init__()


    def input_csv(self, filename):
        # check the existance of csv file
        filelist = [filename]
        for file in filelist:
            if os.path.isfile(file) == True:
                pass
            else:
                print(file + ' does not exist.')
                sys.exit()
        df = pd.read_csv(filename)
        step_len = len(df['time'])
        self.time     = np.empty(step_len)
        self.X        = np.empty(step_len); self.u  =       np.empty(step_len)
        self.Y        = np.empty(step_len); self.vm =       np.empty(step_len)
        self.psi      = np.empty(step_len); self.r  =       np.empty(step_len)
        self.delta_d  = np.empty(step_len); self.n_prop_d = np.empty(step_len)
        self.delta_r  = np.empty(step_len); self.n_prop_r = np.empty(step_len)
        self.n_bt     = np.empty(step_len); self.n_st     = np.empty(step_len)
        self.windd    = np.empty(step_len); self.windv    = np.empty(step_len)

        self.time[:]    = df['time'][:]
        self.X[:]       = df[' X'][:]                      ; self.u[:]        = df[' u'][:]
        self.Y[:]       = df[' Y'][:]                      ; self.vm[:]       = df[' vm'][:]
        self.psi[:]     = df[' psi'][:]                    ; self.r[:]        = df[' r'][:]
        self.delta_d[:] = df[' delta_rudder(direction)'][:]; self.n_prop_d[:] = df[' n_prop(direction)'][:]
        self.delta_r[:] = df[' delta_rudder(real)'][:]     ; self.n_prop_r[:] = df[' n_prop(real)'][:]
        self.n_bt[:]    = df[' n_bt'][:]                   ; self.n_st[:]     = df[' n_st'][:]
        self.windd[:]   = df[' windd'][:]                  ; self.windv[:]    = df[' windv'][:]



class CheckPoint(Traj): #CheckPoint使ってない？
    def __init__(self):
        super().__init__()

        self.U_min = 2.0 * 1852 / 3600 #[mps] (U_minとKの初期値を設定します。U_minは最低速度、Kは標準偏差の係数です。)
        self.K     = 1                 # Coefficient of SD

    def input_csv(self, filename):
        # check the existance of csv file
        filelist = [filename]
        for file in filelist:
            if os.path.isfile(file) == True:
                pass
            else:
                print(file + ' does not exist.')
                sys.exit()
        df = pd.read_csv(filename)

        self.X = np.empty(len(df['x[m]'])); self.cc = np.empty(len(df['x[m]']))
        self.A = np.empty(len(df['x[m]'])); self.B  = np.empty(len(df['x[m]']))

        self.X = df['x[m]'][:]; self.cc = df['cc'][:]
        self.A = df['A'][:]   ; self.B  = df['B'][:]

    def input_guideline(self, filename):
        # check the existance of csv file
        filelist = [filename]
        for file in filelist:
            if os.path.isfile(file) == True:
                pass
            else:
                print(file + ' does not exist.')
                sys.exit()
        df = pd.read_csv(filename)

        #データフレームからガイドラインの平均値（a_ave、b_ave）および標準偏差（a_SD、b_SD）を読み込み、各属性に格納します。
        self.a_ave = df['a_ave']; self.b_ave = df['b_ave']
        self.a_SD = df['a_SD']  ; self.b_SD = df['b_SD']




# tmpファイルから港の情報等を、outoutファイルから数字付きファイルで得た結果を読み取り、論文に載せるような図を作成する。ファイルパスを指定する箇所には###を付ける。
if __name__ == '__main__':

    target_port = sakai_bay.port1A #ここで対象の港を選択する(overall, sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay)

    test_sim = SimlationTraj()
    test_sim.input_csv('tmp/result/simulation_traj.csv') ###
    test_sim.set_params(color = 'r', label = 'sim')

    test_real = RealTraj()
    test_real.input_csv('tmp/result/20210702_1048_Osaka_port1A_beathing.csv', 'tmp/coordinates_of_port/' + target_port.name + '.csv') ###
    test_real.set_params(color = 'b', label = 'real')

    test_CP = CheckPoint()
    test_CP.input_csv('tmp/result/CheckPoint/GeneratedTraj_0.0[rad].csv') ###
    test_CP.input_guideline('tmp/result/CheckPoint/GuidelineFit.csv') ###
    test_CP.set_params(color = 'olive')

    # これの中身を有効にすると全ての点で船型が表示されてしまうのでoffにしている？
    traj_list = [
        # test_sim,
        # test_real,
    ]
    CP_list = [
        test_CP
    ]

    fig = plt.figure(figsize=(12, 8), dpi = 300, constrained_layout=True)

    gs = gridspec.GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0:2]) # XY
    ax2 = fig.add_subplot(gs[0, 2]) # X
    ax3 = fig.add_subplot(gs[1, 2]) # Y
    ax4 = fig.add_subplot(gs[2, 2]) # u
    ax5 = fig.add_subplot(gs[3, 2]) # psi
    # ax6 = fig.add_subplot(gs[4, 2]) # windd 風外乱は無視

    # surrondings plot
    # df_map_desclibed = pd.read_csv(
    #               'tmp/csv/honshu_LATLONG.csv'
    #             , header=None, index_col = None
    #             , names = ['latitude', 'longitude']
    #                 )

    # latlon = np.array([df_map_desclibed['latitude'], df_map_desclibed['longitude']]).T
    # df_berth = pd.read_csv('tmp/coordinates_of_port/' + target_port.name + '.csv')

    # latlon_df = pd.DataFrame(data = latlon, columns= ['latitude', 'longitude']),
    # xy = convert_to_xy(latlon_df, df_berth['Latitude'][0], df_berth['Longitude'][0], df_berth['Psi[deg]'][0])
    # xy = xy.values
    
    
    
    
    df_map_simple = pd.read_csv('tmp/result/berth_simplified_xy.csv') ###修論で使ってた簡易な堺港のｘｙ座標
    map_X = df_map_simple['x [m]'][:]
    map_Y = df_map_simple['y [m]'][:]

    ax1.fill_betweenx(
        map_X,
        map_Y,
        facecolor='gray',
        alpha=0.3
    )
    ax1.plot(
        map_Y,
        map_X,
        color = "k",
        linestyle ="--",
        lw = 0.5,
        alpha = 0.8
    )

    ### ship shape plot これ使ってる？
    ship_shape = load_target_ship_shape_parameter('tmp/Targetship_fixed.csv', 1) ###船を上から見た図
    interval = 60 # sec
    for traj in traj_list:
        for i in range(len(traj.time)):
            if traj.time[i] % interval == 0:
                XX,YY = ship_shape_fix(
                    ship_shape['THETA'][:],
                    ship_shape['R'][:],
                    traj.Y[i],
                    traj.X[i],
                    np.pi/2 - traj.psi[i]
                    )
                ax1.plot(XX,YY,color = traj.color,linestyle ="-",lw = 0.8, zorder = 3)
        XX,YY = ship_shape_fix(
            ship_shape['THETA'][:],
            ship_shape['R'][:],
            traj.Y[-1],
            traj.X[-1],
            np.pi/2 - traj.psi[-1]
            )
        ax1.plot(XX,YY,color = traj.color,linestyle ="-",lw = 0.8, zorder = 3)

    # # Wind annotate plot
    # AnnotateScale = 30
    # for i in range(len(test_real.time)):
    #     theta =   np.pi/2 - np.deg2rad(test_real.windd[i])
    #     r     =  test_real.windv[i]
    #     ax1.annotate(
    #         '',
    #         xy     = [test_real.Y[i], test_real.X[i]],
    #         xytext = [test_real.Y[i] + AnnotateScale * r * np.cos(theta), test_real.X[i] + AnnotateScale * r * np.sin(theta)],
    #         arrowprops=dict(shrink=0, width=2, headwidth=8,
    #                         headlength=5, connectionstyle='arc3',
    #                         facecolor='skyblue', edgecolor='skyblue'),
    #         alpha = 0.4
    #     )
    # # Wind annotate legend
    # AnnotateLegendX = -780
    # AnnotateLegendY = -30

    # ax1.annotate(
    #     '',
    #     xy     = [AnnotateLegendY+AnnotateScale/2, AnnotateLegendX],
    #     xytext = [AnnotateLegendY-AnnotateScale/2, AnnotateLegendX],
    #     arrowprops=dict(shrink=0, width=2, headwidth=8,
    #                     headlength=5, connectionstyle='arc3',
    #                     facecolor='skyblue', edgecolor='skyblue'),
    #     alpha = 1
    # )
    # ax1.text(AnnotateLegendY, AnnotateLegendX+20, 'True wind\n1.0[m/s]', va = 'bottom', ha = 'center')




    # XY plot
    for traj in traj_list:
        ax1.plot(traj.Y, traj.X, color = traj.color, ls = '--', label = traj.label, zorder = 2)

    traj_other_files =glob.glob('tmp/result/_Osaka_port1A/*.csv') ###ここで実航跡をすべて取得
    # traj_other_files = []
    x_list = []
    y_list = []
    psi_list = []
    for file_i, file in enumerate(traj_other_files):
        if file_i == 0:
            label = "Captain's routes"
        else:
            label = None
        test_real_other = RealTraj() #インスタンス化
        test_real_other.input_csv(file, 'tmp/coordinates_of_port/' + target_port.name + '.csv') ###Real_trajクラスのinput_csvメソッドを実行。XYがほしいだけなのに他の処理も全てしてるから無駄なメモリ消費が多そう。
        ax1.plot(test_real_other.Y, test_real_other.X, color = 'gray', ls = '-', marker = 'D', markersize = 2, alpha = 0.8, lw = 1.0, zorder = 0, label = label)
        # idx = np.argmin(np.abs(test_real_other.u*3600 / 1852 - 6))
        # x_list.append(test_real_other.X[idx])
        # y_list.append(test_real_other.Y[idx])
        # psi_list.append(test_real_other.psi[idx])
        # ax1.plot(test_real_other.Y[idx], test_real_other.X[idx], color = 'k',marker = '.', alpha = 0.6, lw = 0.7, zorder = 1)
        # ax1.annotate('', xy=[test_real_other.Y[idx] + 100 * np.cos(np.pi/2 -test_real_other.psi[idx]), 
        #                      test_real_other.X[idx] + 100 * np.sin(np.pi/2 -test_real_other.psi[idx])], 
        #              xytext=[test_real_other.Y[idx], test_real_other.X[idx]],
        #         arrowprops=dict(shrink=0, width=1, headwidth=8, 
        #                         headlength=10, connectionstyle='arc3',
        #                         facecolor='gray', edgecolor='gray')
        #        )
    # ax1.plot(np.mean(y_list), np.mean(x_list), color = 'k',marker = '.', alpha = 0.6, lw = 0.7, zorder = 1)
    # ax1.annotate('', xy=[np.mean(y_list) + 50 * np.cos(np.deg2rad(45)), np.mean(x_list) + 50 * np.sin(np.deg2rad(45))], 
    #                  xytext=[np.mean(y_list), np.mean(x_list)],
    #             arrowprops=dict(shrink=0, width=1, headwidth=8, 
    #                             headlength=10, connectionstyle='arc3',
    #                             facecolor='tomato', edgecolor='tomato')
    #            )






    # A* result plot　(このあたりも修論の内容と思われる)
    SD_Miyauchi = ShipDomain_Wang()
    SD_Miyauchi.initial_setting(ship_status, B_max = 681.7220531840326)
    SD_proposal = ShipDomain_proposal()
    SD_proposal.initial_setting('output/303/mirror5/fitting_parameter.csv', sigmoid) ###
    SD_shape = ShipDomain_shape()
    SD_shape.initial_setting(ship_status)
    df = pd.read_csv('tmp/GuidelineFit_debug.csv') ###
    a_ave = df['a_ave'].values[0]
    b_ave = df['b_ave'].values[0]
    a_SD = df['a_SD'].values[0]
    b_SD = df['b_SD'].values[0]
    theta_list = np.arange(np.deg2rad(0),np.deg2rad(360) + np.deg2rad(30) / 10, np.deg2rad(30))

    ###
    csv_list = [
                'output/fig/20240212/scenario1_wo_domain/result.csv',
                # 'output/fig/20240212/scenario2_wo_domain/result.csv',
                'output/fig/20240212/scenario1_Wang/result.csv',
                # 'output/fig/20240212/scenario2_Wang/result.csv',
                'output/fig/20240212/scenario1_proposal/result.csv',
                # 'output/fig/20240212/scenario2_proposal/result.csv',
                ]
    label_list = [
                'Without domain',
                #   'without domain',
                  'Miyauchi model',
                #   'Miyauchi model',
                  'Present model',
                #   'presented model',
                ]
    color_list = [
                '#FF4B00', 
                #   '#FF4B00', 
                  '#005AFF', 
                #   '#005AFF', 
                  '#03AF7A', 
                #   '#03AF7A'
                  ]
    marker_list = [
                '^',
                #    'v',
                   '^',
                #    'v',
                   '^',
                #    'v',
    ]
    SD_list = [
                SD_shape,
            #    SD_shape,
               SD_Miyauchi,
            #    SD_Miyauchi,
               SD_proposal,
            #    SD_proposal
    ]
    for i,file in enumerate(csv_list):
        df_astar = pd.read_csv(file, names = ['x [m]', 'y [m]', 'psi [rad]'], header = None, skiprows = 1)
        SD = SD_list[i]
        ax1.plot(df_astar['y [m]'[:]]+2.5,df_astar['x [m]'][:]+2.5, 
                 markeredgecolor = color_list[i], color = color_list[i],ls = '-', alpha = 0.6, lw = 2, marker = marker_list[i], zorder = 4, label = label_list[i], markersize=2, markeredgewidth=1)
        for j in range(len(df_astar['y [m]'[:]])):
                if j % 10 == 0:
                    distance = ((df_astar['y [m]'][j]+0.5 - df_astar['y [m]'][len(df_astar['y [m]'[:]])-1]+0.5) ** 2 + (df_astar['x [m]'][j]+0.5 - df_astar['x [m]'][len(df_astar['y [m]'[:]])-1]+0.5) ** 2) ** (1/2)
                    speed = b_ave * distance ** (a_ave) + b_SD * distance ** (a_SD)
                    r_list = []
                    for theta_i in theta_list:
                        r_list.append(SD.distance(speed, theta_i))
                    # ax.plot([self.path_xy[j,1]+0.5, self.path_xy[j,1]+0.5+100*np.sin(self.psi[j])],
                    #         [self.path_xy[j,0]+0.5, self.path_xy[j,0]+0.5+100*np.cos(self.psi[j])],
                    #         lw = 1.5, color = 'y'
                    #         )
                    ax1.plot(df_astar['y [m]'][j]+ 2.5 + r_list[:] * np.sin(theta_list[:] +df_astar['psi [rad]'][j]),
                             df_astar['x [m]'][j]+ 2.5 + r_list[:] * np.cos(theta_list[:] + df_astar['psi [rad]'][j]), 
                            lw = 0.7, color = color_list[i], ls = '--', alpha = 0.7
                            )


    # CP plot
    # ax1.scatter([-28.912919843298535, -21.431753950226557],[-90, -40], color = test_CP.color, s = 30, zorder = 3)
    # for i in range(len(test_CP.X)):
    #     if test_CP.cc[i] >= 0.70:
    #         ax1.scatter(test_CP.B[i],test_CP.X[i], color = test_CP.color, s = 10, zorder = 0)

    # c = patches.Circle(xy=(-28.912919843298535, -90), radius=5.1899999999999995, fc=test_CP.color, ec='k', zorder = 1)
    # ax1.add_patch(c)
    # c = patches.Circle(xy=(-21.431753950226557, -40), radius=5.1899999999999995, fc=test_CP.color, ec='k', zorder = 1)
    # ax1.add_patch(c)


    xlim = [-900,150]#[-120, 50]#(-600, 50)
    ylim = [-1500,50 ]#[-200, 60]#(-900, 50)
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_xticks(np.arange(xlim[0], xlim[1]+10, 100))#100, 20
    ax1.set_yticks(np.arange(ylim[0], ylim[1]+10, 100))
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel(r'$Y\,\rm{[m]}$')
    ax1.set_ylabel(r'$X\,\rm{[m]}$')
    ax1.legend()

    # start plot
    ax1.scatter(-800, -1400, color = 'k', s = 30, zorder = 4)
    ax1.text(-800, -1400, '    start(scenario1)  ', va = 'center', ha = 'left')
    # ax1.scatter(-400, -600, color = 'k', s = 30, zorder = 4)
    # ax1.text(-400, -600, 'start(scenario2)  ', va = 'center', ha = 'right')

    # end plot
    end_XY = [1.391202639567448, -11.329940376783341]
    ax1.scatter(end_XY[1], end_XY[0], color = 'k', s = 30, zorder = 4)
    ax1.text(end_XY[1], end_XY[0], 'goal  ', va = 'top', ha = 'right')

    # get the timing sim and real are equal
    for i in range(len(test_real.time)):
        if abs(test_real.X[i] - test_sim.X[0]) <= 1e-6:
            if abs(test_real.Y[i] - test_sim.Y[0]) <= 1e-6:
                sim_start_i = i
                break

    # t_max
    t_max = 0
    for traj in traj_list:
        if traj.time[-1] > t_max:
            t_max = traj.time[-1] - 60 * sim_start_i

    # X plot
    ax2.plot(test_sim.time, test_sim.X, color = test_sim.color)
    ax2.plot(test_real.time[:-sim_start_i], test_real.X[sim_start_i:], color = test_real.color)
    ax2.grid()
    ax2.set_xlim(0, t_max)
    ax2.set_ylabel(r'$X\,\rm{[m]}$')
    ax2.set_yticks(np.arange(-1000, 50+10, 500))
    ax2.set_xticklabels([])

    # Y plot
    ax3.plot(test_sim.time, test_sim.Y, color = test_sim.color)
    ax3.plot(test_real.time[:-sim_start_i], test_real.Y[sim_start_i:], color = test_real.color)
    ax3.grid()
    ax3.set_xlim(0, t_max)
    ax3.set_ylabel(r'$Y\,\rm{[m]}$')
    ax3.set_yticks(np.arange(-1000, 50+10, 500))
    ax3.set_xticklabels([])

    # U plot
    ax4.plot(test_sim.time, test_sim.u, color = test_sim.color)
    ax4.plot(test_real.time[:-sim_start_i], test_real.u[sim_start_i:], color = test_real.color)
    ax4.grid()
    ax4.set_xlim(0, t_max)
    ax4.set_ylabel(r'$U\,\rm{[kts]}$')
    ax4.set_yticks(np.arange(0.0, 5.0, 1.0))
    ax4.set_xticklabels([])

    # psi plot
    ax5.plot(test_sim.time, np.rad2deg(test_sim.psi), color = test_sim.color)
    ax5.plot(test_real.time[:-sim_start_i], np.rad2deg(test_real.psi[sim_start_i:]), color = test_real.color)
    ax5.grid()
    ax5.set_xlim(0, t_max)
    ax5.set_ylabel(r'$\psi\,\rm{[deg.]}$')
    ax5.set_yticks(np.arange(-90, 100, 45))
    ax5.set_xlabel('time[s]')

    plt.tight_layout()

    fig.savefig('output/500/MultiPlot_0618test.png', pad_inches=0.05) ###output/500/MultiPlot.png
