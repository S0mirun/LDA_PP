'''
CMA-ESで経路最適化を行うための最終ファイル
1:A*で初期経路を生成
2:船速プロファイルに基づいて変針点を求める。
3:最後にCMA-ESを用いて最適化
速度の上限は9.5knots
辞書のKeyは数字に統一する
SDを線分の中央で計算
elementの角度計算の1/2を消した
角度系 : 縦軸方向を0とし、時計回りを正、反時計回りを負に取る
縦がX(ver)、横がY(hor)であることに注意！!
'''
# 入船の場合の初期点探索打ち切り距離は120m, リスタートは3回でやっている

import copy
from datetime import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

program_folder = os.path.dirname(os.path.abspath(__file__))
pyshipsim_path = os.path.join(program_folder, "py-ship-simulator-main/py-ship-simulator-main")
sys.path.append(pyshipsim_path)

import pyshipsim 
import utils.PP.Astar_for_CMAES
import utils.PP.graph
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.MakeDictionary_and_StackedBarGraph import new_filtered_dict
from utils.PP.graph import ShipDomain_proposal
from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay
