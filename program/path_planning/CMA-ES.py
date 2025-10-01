"""
CMA-ES path optimization (A* init → element-based turning points → CMA-ES)
- Max speed: 9.5 knots
- Ship Domain at segment midpoint
- Angle convention: vertical(X) = 0 deg, clockwise positive
- Coordinate note: vertical = X (ver), horizontal = Y (hor)
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.PP.Astar_for_CMAES
import utils.PP.graph
from utils.PP.E_ddCMA import DdCma, Checker, Logger
from utils.PP.E_MakeDictionary_and_StackedBarGraph import new_filtered_dict
from utils.PP.graph import ShipDomain_proposal
from utils.PP.subroutine import sakai_bay, yokkaichi_bay, Tokyo_bay, else_bay

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
PYSIM_DIR = os.path.join(PROGRAM_DIR, "py-ship-simulator-main/py-ship-simulator-main")
if PYSIM_DIR not in sys.path:
    sys.path.append(PYSIM_DIR)
import pyshipsim 