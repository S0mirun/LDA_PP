import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.PP.Bezier_curve as Bezier

DIR = os.path.dirname(__file__)
DATA = f"{DIR}/../../outputs/data"
file = f"{DATA}/detail_map/Hachinohe.csv"
SAVE_DIR = os.path.dirname(file)

df_map = pd.read_csv(file)

start = np.asarray([2500, 1350], dtype=float)
end   = np.asarray([0, 0], dtype=float)
pts = np.array([df_map["y [m]"], df_map["x [m]"]]).T
pts = np.vstack([start, pts, end])
pts = Bezier.sort(pts=pts, start=start, end=end)
C, _ = Bezier.bezier(pts, num=400)

plt.plot(df_map["y [m]"], df_map["x [m]"])
plt.plot(C[:,0], C[:, 1])
plt.grid()
plt.axis("equal")
plt.show()
