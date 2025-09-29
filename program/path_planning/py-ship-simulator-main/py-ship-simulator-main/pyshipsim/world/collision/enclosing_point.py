import sys
import numpy as np

from ...utils import PointsArray, PolyArray


class EnclosingPointCollisionChecker:

    def reset(self, obstacle_polygons: PolyArray | list[PolyArray]) -> None:
        if obstacle_polygons is PolyArray:
            obstacle_polygons = [obstacle_polygons]
        self.obstacle_polygons = obstacle_polygons
    '''
    def check(self, ship_polygon: PolyArray) -> bool:
        for obstacle_polygon in self.obstacle_polygons:
            if enclosing_point(ship_polygon, obstacle_polygon):
                return True
        return False
    '''
    def check(self, ship_polygon: PolyArray, node_array):
        for obstacle_polygon in self.obstacle_polygons:
            common_node_array = enclosing_point(ship_polygon, obstacle_polygon, node_array)
        return common_node_array
            

'''
def enclosing_point(points: PointsArray, polygon: PolyArray):
    for point in points:
        l1 = polygon - point
        l2 = np.concatenate([l1[1:, :], l1[0:1, :]], axis=0)
        #
        cross_product = l1[:, 0] * l2[:, 1] - l1[:, 1] * l2[:, 0]
        dot_product = l1[:, 0] * l2[:, 0] + l1[:, 1] * l2[:, 1]
        total_angle = np.sum(np.arctan2(cross_product, dot_product))
        if np.abs(np.abs(total_angle) - 2 * np.pi) < sys.float_info.epsilon:
            return True
    return False
'''
def enclosing_point(points: PointsArray, polygon: PolyArray, node_array):
    for point in points:
        l1 = polygon - point
        l2 = np.concatenate([l1[1:, :], l1[0:1, :]], axis=0)
        #
        cross_product = l1[:, 0] * l2[:, 1] - l1[:, 1] * l2[:, 0]
        dot_product = l1[:, 0] * l2[:, 0] + l1[:, 1] * l2[:, 1]
        total_angle = np.sum(np.arctan2(cross_product, dot_product))
        if np.abs(np.abs(total_angle) - 2 * np.pi) < sys.float_info.epsilon:
            #print(f'Contact')
            node_array = np.vstack([node_array, point])
        #else:
            #print(f'No Contact')
    return node_array
