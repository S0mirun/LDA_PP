import numpy as np
import pandas as pd
import pickle

#ファイルパスは数字付きのファイルで選択すると自動で代入される仕組みになってそう

#オブジェクトを指定されたファイルに保存します。
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

#ファイルからオブジェクトを読み込みます。
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

#CSVファイルから緯度と経度のデータを読み込みます。
def READ_LATLONG(csv_filepath):
    df = pd.read_csv(csv_filepath, header = None, names = ['EAST_LONGITUDE', 'NORTH_LATITUDE', 'dummy'])
    EAST_LONGITUDE = df['EAST_LONGITUDE']
    NORTH_LATITUDE = df['NORTH_LATITUDE']

    return EAST_LONGITUDE, NORTH_LATITUDE

#度分表記の緯度経度を10進表記に変換します。(航跡データファイル2020/10csvファイルに計算方法の記載あり)
def CONVERT_LATLONG_FROM_DM_TO_D(string):
    # N = +. S = - and E = +, W = -
    deg = float(string.split("°")[0])
    min = float(str(string.split("°")[1]).split("’")[0]) / 60
    if str(str(string.split("°")[1]).split(".")[1]).split("’")[1] == 'N' or str(str(string.split("°")[1]).split(".")[1]).split("’")[1] == 'E':
        direction_sign = 1
    elif str(str(string.split("°")[1]).split(".")[1]).split("’")[1] == 'S' or str(str(string.split("°")[1]).split(".")[1]).split("’")[1] == 'W':
        direction_sign = -1
    LATLONG_DECIMAL = (deg + min) * direction_sign

    return LATLONG_DECIMAL

#船の中央位置（MidShip）の緯度経度を計算します。研究ノート70ページに記載有
def GET_MIDSHIP_LATLONG(lat, long, psi, L, B):
    X_gps = 14 # distance from port-side[m]
    Y_gps = 24 # distance from stern[m]
    ms_lat  = np.empty(len(lat))
    ms_long = np.empty(len(lat))
    ms_lat[:]  = - ((B/2) - X_gps ) * np.cos(np.deg2rad(90 - psi[:])) + ((L/2) - Y_gps) * np.sin(np.deg2rad(90 - psi[:]))
    ms_long[:] =   ((B/2) - X_gps ) * np.sin(np.deg2rad(90 - psi[:])) + ((L/2) - Y_gps) * np.cos(np.deg2rad(90 - psi[:]))
    ms_lat[:]  = lat[:]  + (ms_lat[:]  * 360 / (40000 * 1000))
    ms_long[:] = long[:] + (ms_long[:] * 360 / (40000 * 1000 * np.cos(np.deg2rad(ms_lat[:]))))
    return ms_lat, ms_long