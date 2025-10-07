import os
import glob

def num(DIR_PATH):
    ex = os.path.exists(DIR_PATH)
    #今のディレクトリ内を検索
    files0 = len(glob.glob(f"{DIR_PATH}/*.xyz")) 
    files1 = len(glob.glob(f"{DIR_PATH}/**/*.xyz"))
    #今のディレクトリ内のファイルの中のファイルの中の文字列まで探す
    files2 = len(glob.glob(f"{DIR_PATH}/**/**/*.xyz")) 
    where = os.path.abspath(DIR_PATH)
    print("result:   ", ex, files0, where)

DIR = os.path.dirname(__file__)
num(DIR)
TS_INFO_DIR = f"{DIR}/../LDA-learn/raw_data/888-送付データ/2-3_一次解析/"
num(TS_INFO_DIR)
TS_INFO_TABLE_PATH = f"{TS_INFO_DIR}/1_2023_0216_PCC_着/1-運動/"
num(TS_INFO_TABLE_PATH)


#
print(glob.glob('./../*.py'))



