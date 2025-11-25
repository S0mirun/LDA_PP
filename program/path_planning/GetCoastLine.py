import os

import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET


# データ取得関数
# file_path：海岸線データのパス
# prefecture_name: 都道府県名
def get_coastlines_df(file_path, prefecture_name):
    if not(os.path.isfile(file_path)):
        print("file path not found")
        return pd.DataFrame()
        
    #xmlデータを読み込みます
    tree = ET.parse(file_path)
    #一番上の階層の要素を取り出します
    root = tree.getroot()
    
    lat = [] #緯度
    lon = [] #経度

    for value in tqdm(root.iter('DirectPosition.coordinate'), leave=False):
        tmp = value.text
        tmp = tmp.split(' ')
        lat.append(tmp[0])
        lon.append(tmp[1])
        
    df = pd.DataFrame()
    df['緯度'] = lat
    df['経度'] = lon
    
    df.index = [prefecture_name] * len(lat)
    
    return df

# 実行
prefecture_name = "茨城県"
DIR = os.path.dirname(__file__)
RAW_DATAS = f"{DIR}/../../raw_datas"
file_path = f"{RAW_DATAS}/*/C23-06_08-g.xml"
print(os.path.abspath(file_path))
coastline = get_coastlines_df(file_path, prefecture_name)
