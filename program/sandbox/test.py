import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import unicodedata

from utils.ship_geometry import *
from utils.visualization import *
from utils.kml import kml_based_txt_to_csv

DIR = os.path.dirname(__file__)
dirname =os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
top_path = f"{DIR}/../../raw_datas"
print(f"\n{os.path.exists(top_path)}\n")
print(f"\n{os.path.abspath(top_path)}\n")