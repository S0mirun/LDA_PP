'''
CMA-ESのためのAstarによる初期経路計画アルゴリズム
角度系 : 横軸方向を０に取り、時計回りを負、反時計回りを正に取る
'''
import numpy as np
from tqdm import tqdm

# これはA*の角度系ではなく、縦軸方向を0とし、時計回りを正、反時計回りを負に取る角度系
# 単に図にSDを表示させるためだけの角度リスト
psi_set = np.deg2rad([180, 0, -90, 90, -135, -45, 135, 45])
# new_positionと対応している(これは通常の横縦座標)
# [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, psi=None):
        self.parent = parent
        self.position = position
        self.psi = psi
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def angle_adaptor(angle):
    new_angle = -angle + (np.pi / 2)
    if new_angle > np.pi:
        new_angle = new_angle - 2 * np.pi
    return new_angle
    
###A*はSDの種類を選択する必要なし###
# ややこしいが、A＊は通常の横縦の順で入力し、横縦の順で出力される
# SDを用いるためにpsiも計算するが、最後に返す必要はない。しかし、A*の中で角度がどのように用いられているのか気になるので、psiリスト出すのもありかもしれない
def astar(map, start, end, psi_start, psi_end, SD, weight, enclosing_checker):
    maze = map.maze

    # psi_startとpsi_endをA*の角度系に変換する
    psi_start_astar = angle_adaptor(psi_start)
    psi_end_astar = angle_adaptor(psi_end)

    # Create start and end node
    start_node = Node(None, start, psi = psi_start_astar)
    end_node = Node(None, end, psi = psi_end_astar)
    start_node.g = start_node.h = start_node.f = 0
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []
    open_list.append(start_node)
    #
    itr = 0
    with tqdm(total=len(open_list) + len(closed_list), desc="A*", unit="node") as pbar:
        while open_list:
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            # Pop current node off open list and add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node.position == end_node.position:
                path_list = []
                psi_list = []
                current = current_node
                while current is not None:
                    path_list.append(current.position)
                    psi_list.append(current.psi)
                    current = current.parent
                # path_list.append(start_node.position)
                # psi_list.append(start_node.psi)
                return path_list[::-1], psi_list[::-1], itr

            # Generate children
            children = []
            for point_i, new_position in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]):
                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                    continue

                # Check the terrain　where progress can be made
                if maze[node_position[0]][node_position[1]] != 0:
                    continue
                
                """
                一旦角度制約なしでやってみてうまくいったら角度制約も入れてやってみる
                # Check the angle
                if current_node.parent is not None:
                    vector_parent_to_current  = np.array(current_node.parent.position[0] - current_node.position[0],
                                            current_node.parent.position[1] - current_node.position[1])
                    vector_current_to_child = np.array(current_node.position[0] - node_position[0],
                                                    current_node.position[1] - node_position[1])
                    inner_product = np.dot(vector_current_to_child, vector_parent_to_current)
                    c_norm = np.linalg.norm(vector_current_to_child)
                    p_norm = np.linalg.norm(vector_parent_to_current)
                    # 内積の値を [-1, 1] の範囲にクリップ
                    cos_angle = np.clip(inner_product / (c_norm * p_norm), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    if angle < 50:
                        pass
                    else:
                        continue
                else:
                    print(f'start node has no parent')
                """
                
                # Create new node
                new_node = Node(current_node, node_position, psi = psi_set[point_i])
                
                # Make sure child is Not in closed_list
                if new_node in closed_list:
                    continue

                children.append(new_node)
                
                for child in children:
                    
                    # when you do not use SD,  child.g = current_node.g + ((child.position[0] - current_node.position[0]) ** 2) + ((child.position[1] - current_node.position[1]) ** 2)
                    # 2乗から平方に戻すと計算負荷が高すぎて全然計算が終わらないため、あえて2乗のまま計算を行うことにする
                    # child.g = current_node.g + np.sqrt(((child.position[0] - current_node.position[0]) ** 2) + ((child.position[1] - current_node.position[1]) ** 2)) + map.ship_domain_cost_astar(child, SD, weight, enclosing_checker)
                    # child.h = np.sqrt(((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2))
                    # child.f = child.g + child.h
                    child.g = current_node.g + ((child.position[0] - current_node.position[0]) ** 2) + ((child.position[1] - current_node.position[1]) ** 2) + map.ship_domain_cost_astar(child, SD, weight, enclosing_checker)
                    # print(f"SDの割合は{(map.ship_domain_cost_astar(child, SD, weight, enclosing_checker)) / (current_node.g + (child.position[0] - current_node.position[0]) ** 2 + ((child.position[1] - current_node.position[1]) ** 2))}")
                    child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                    child.f = child.g + child.h
                    
                # Make sure cost of new child
                if any(open_node for open_node in open_list if child == open_node and child.g > open_node.g):
                    continue
                
                open_list.append(child)
            pbar.n = len(closed_list)
            pbar.total = len(open_list) + len(closed_list)
            pbar.set_postfix(open=len(open_list), closed=len(closed_list))
            pbar.refresh()
                
            