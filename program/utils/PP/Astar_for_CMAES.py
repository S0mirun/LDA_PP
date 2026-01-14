"""
CMA-ESのためのA*初期経路計画アルゴリズム

- 目的: A*で障害物とShip Domain（SD）罰則を考慮した初期経路を生成し、CMA-ESの初期解に供する。
- 角度系:
  - ユーザー定義の角度系: 横軸0°, 時計回りが負、反時計回りが正。
  - A*内部表示用の角度系: 縦軸0°, 時計回りが正、反時計回りが負（図示のための psi_set）。
  - angle_adaptor() はユーザー角度をA*内部角度系に変換する（必要なら開始・終了方位に使用）。
- コスト:
  - g: 1近傍（8方向）の移動コストを「平方距離（直交=1, 斜め=2）」とし、これに SD 罰則 map.ship_domain_cost_astar(...) を加算。
  - h: 許容ヒューリスティックとしてマンハッタン距離を採用（直交=1, 斜め=2 の体系では L1 が最短下界になるため）。
- 実装ポイント:
  - 可読性を優先しつつ、open は優先度付きキュー（heap）で管理。
  - visited/closed は position セットで管理。
  - 同一 position のより良い g を見つけたときのみ更新・push。
  - tqdm でオープン／クローズ数の進捗を表示。
- 返り値:
  - path_list: [(row, col), ...] スタート→ゴール,座標ではない
  - psi_list : 各ステップの方位（図示用）。開始・終了方位の厳密制約は未適用（必要なら拡張）。
  - itr      : 展開ノード数カウンタ
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict

import numpy as np
from tqdm import tqdm

# 8-neighborhood (row, col)
OFFSETS: List[Tuple[int, int]] = [
    (0, -1),  (0, 1),   (-1, 0),  (1, 0),
    (-1, -1), (-1, 1),  (1, -1),  (1, 1),
]

# headings (rad) aligned with OFFSETS for drawing SD etc.
# vertical-axis=0°, CW positive, CCW negative
HEADINGS: np.ndarray = np.deg2rad([180, 0, -90, 90, -135, -45, 135, 45])


@dataclass(eq=False)
class Node:
    parent: Optional["Node"]
    position: Tuple[int, int]
    psi: Optional[float] = None
    g: float = 0.0
    h: float = 0.0
    f: float = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.position == other.position


def angle_adaptor(angle_rad: float) -> float:
    """convert user's angle to internal (vertical-axis=0°, CW positive)."""
    new_angle = -angle_rad + (np.pi / 2)
    if new_angle > np.pi:
        new_angle -= 2 * np.pi
    return new_angle


def manhattan(p: Tuple[int, int], q: Tuple[int, int]) -> int:
    """L1 distance as admissible heuristic."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def reconstruct_path(n: Node) -> Tuple[List[Tuple[int, int]], List[float]]:
    """collect path/psi from goal back to start."""
    path, psi = [], []
    cur = n
    while cur is not None:
        path.append(cur.position)
        psi.append(cur.psi if cur.psi is not None else 0.0)
        cur = cur.parent
    path.reverse()
    psi.reverse()
    return path, psi


def astar(
    map,  # expects: map.maze (2D), map.ship_domain_cost_astar(node, SD, weight, enclosing_checker)
    start: Tuple[int, int],
    end: Tuple[int, int],
    psi_start: float,
    psi_end: float,
    SD,
    weight: float,
    enclosing_checker,
):
    # setup
    maze = map.maze
    rows, cols = len(maze), len(maze[0])
    psi_start_astar = angle_adaptor(psi_start)
    psi_end_astar = angle_adaptor(psi_end)

    start_node = Node(None, start, psi=psi_start_astar, g=0.0, h=0.0, f=0.0)
    end_node = Node(None, end, psi=psi_end_astar, g=0.0, h=0.0, f=0.0)

    # containers
    open_heap: List[Tuple[float, int, Node]] = []
    heap_counter = 0
    g_best: Dict[Tuple[int, int], float] = {start_node.position: 0.0}
    closed: Set[Tuple[int, int]] = set()

    heapq.heappush(open_heap, (start_node.f, heap_counter, start_node))
    heap_counter += 1

    itr = 0

    with tqdm(total=1, desc="A*", unit="node") as pbar:
        while open_heap:
            # pick best
            _, _, current = heapq.heappop(open_heap)
            if current.position in closed:
                continue

            closed.add(current.position)
            itr += 1

            # goal?
            if current.position == end_node.position:
                path_list, psi_list = reconstruct_path(current)
                pbar.n = len(closed)
                pbar.total = len(closed) + len(open_heap)
                pbar.set_postfix(open=len(open_heap), closed=len(closed))
                pbar.refresh()
                return path_list, psi_list, itr

            # expand neighbors
            for i, (dr, dc) in enumerate(OFFSETS):
                nr, nc = current.position[0] + dr, current.position[1] + dc

                # bounds
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue

                # obstacle
                if maze[nr][nc] != 0:
                    continue

                if (nr, nc) in closed:
                    continue

                # step cost (squared step) + SD penalty
                step_cost = (dr * dr + dc * dc)
                candidate = Node(parent=current, position=(nr, nc), psi=HEADINGS[i])

                sd_penalty = map.ship_domain_cost_astar(candidate, SD, weight, enclosing_checker)
                tentative_g = current.g + step_cost + sd_penalty

                # not better than known?
                if tentative_g >= g_best.get((nr, nc), float("inf")):
                    continue

                # update best
                g_best[(nr, nc)] = tentative_g
                candidate.g = tentative_g
                candidate.h = manhattan(candidate.position, end_node.position)
                candidate.f = candidate.g + candidate.h

                heapq.heappush(open_heap, (candidate.f, heap_counter, candidate))
                heap_counter += 1

            # progress (rough)
            pbar.n = len(closed)
            pbar.total = len(closed) + len(open_heap)
            pbar.set_postfix(open=len(open_heap), closed=len(closed))
            pbar.refresh()

    # not found
    raise RuntimeError("A* failed to find a path")

def astar2(
    map,  # expects: map.maze (2D), map.ship_domain_cost_astar(node, SD, weight, enclosing_checker)
    start: Tuple[int, int],
    end: Tuple[int, int],
    psi_start: float,
    psi_end: float,
    SD,
    weight: float,
    enclosing_checker,
):
    pass