import os
import glob

import numpy as np
import pandas as pd
import cv2
import heapq
import matplotlib.pyplot as plt
from math import sqrt

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_walkable_from_csv(csv_path: str, W: int, H: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    walkable = np.zeros((H, W), dtype=bool)
    x = df["x"].to_numpy(dtype=int)
    y = df["y"].to_numpy(dtype=int)
    ok = (0 <= x) & (x < W) & (0 <= y) & (y < H)
    walkable[y[ok], x[ok]] = True
    return walkable

def snap_to_nearest_walkable(walkable: np.ndarray, p: tuple[int,int], max_r: int = 2000) -> tuple[int,int]:
    H, W = walkable.shape
    x0, y0 = p
    x0 = int(np.clip(x0, 0, W-1))
    y0 = int(np.clip(y0, 0, H-1))
    if walkable[y0, x0]:
        return (x0, y0)

    for r in range(1, max_r + 1):
        xmin, xmax = max(0, x0 - r), min(W - 1, x0 + r)
        ymin, ymax = max(0, y0 - r), min(H - 1, y0 + r)

        xs = np.arange(xmin, xmax + 1)

        # 上下
        if walkable[ymin, xs].any():
            i = xs[np.where(walkable[ymin, xs])[0][0]]
            return (int(i), int(ymin))
        if walkable[ymax, xs].any():
            i = xs[np.where(walkable[ymax, xs])[0][0]]
            return (int(i), int(ymax))

        ys = np.arange(ymin, ymax + 1)
        # 左右
        if walkable[ys, xmin].any():
            j = ys[np.where(walkable[ys, xmin])[0][0]]
            return (int(xmin), int(j))
        if walkable[ys, xmax].any():
            j = ys[np.where(walkable[ys, xmax])[0][0]]
            return (int(xmax), int(j))

    raise ValueError("近傍に walkable が見つかりません（max_r を増やすか抽出条件を見直してください）")

def astar_on_mask(walkable: np.ndarray, start: tuple[int,int], goal: tuple[int,int], diag: bool = True):
    H, W = walkable.shape
    start = snap_to_nearest_walkable(walkable, start)
    goal  = snap_to_nearest_walkable(walkable, goal)
    sx, sy = start
    gx, gy = goal

    if diag:
        moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                 (-1,-1,sqrt(2)),(1,-1,sqrt(2)),(-1,1,sqrt(2)),(1,1,sqrt(2))]
        def h(x,y):  # Euclid
            return sqrt((x-gx)**2 + (y-gy)**2)
    else:
        moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0)]
        def h(x,y):  # Manhattan
            return abs(x-gx) + abs(y-gy)

    INF = 1e18
    gscore = np.full((H, W), INF, dtype=float)
    parent = np.full((H, W, 2), -1, dtype=int)

    gscore[sy, sx] = 0.0
    pq = [(h(sx, sy), 0.0, sx, sy)]
    visited = np.zeros((H, W), dtype=bool)

    while pq:
        f, g, x, y = heapq.heappop(pq)
        if visited[y, x]:
            continue
        visited[y, x] = True

        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) != (sx, sy):
                px, py = parent[y, x]
                x, y = int(px), int(py)
                path.append((x, y))
            path.reverse()
            return path, start, goal, gscore[gy, gx]

        for dx, dy, cost in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not walkable[ny, nx]:
                continue
            ng = g + cost
            if ng < gscore[ny, nx]:
                gscore[ny, nx] = ng
                parent[ny, nx] = (x, y)
                heapq.heappush(pq, (ng + h(nx, ny), ng, nx, ny))

    return None, start, goal, None

# -------------------- 実行 --------------------
W, H = 950, 600
img_path = glob.glob(f"{DIR}/../../outputs/13.2*/Yokkaichi.png")[0]
csv_path = glob.glob(f"{DIR}/../../outputs/*/bgr_exact_255_187_121.csv")[0]

walkable = load_walkable_from_csv(csv_path, W=W, H=H)

start = (450, 260)  # (x,y)
goal  = (150, 300)

path, start2, goal2, cost = astar_on_mask(walkable, start, goal, diag=True)
print("snapped start:", start2, "snapped goal:", goal2, "cost:", cost, "path_len:", None if path is None else len(path))

# ---- 可視化（赤線で経路）----
img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img_rgb, origin="upper")

if path is None:
    plt.scatter([start2[0], goal2[0]], [start2[1], goal2[1]], s=30)
    plt.title("No path found")
else:
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    plt.plot(xs, ys, linewidth=2)   # ← デフォルト色を赤にしたいなら次行へ
    # plt.plot(xs, ys, color="red", linewidth=2)  # 赤線で固定したい場合

    plt.scatter([start2[0]], [start2[1]], s=40)
    plt.scatter([goal2[0]],  [goal2[1]],  s=40)
    plt.title("A* path overlay (red line if color='red')")

plt.axis("off")
plt.show()