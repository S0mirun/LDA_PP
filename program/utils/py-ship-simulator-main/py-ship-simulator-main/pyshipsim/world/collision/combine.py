import sys
import numpy as np

from .enclosing_point import EnclosingPointCollisionChecker
from .segments_intersection import SegmentsIntersectCollisionChecker
from ...utils import PolyArray


class StrictCollisionChecker:

    def __init__(self) -> None:
        self.intersect = SegmentsIntersectCollisionChecker()
        self.enclosing = EnclosingPointCollisionChecker()

    def reset(self, obstacle_polygons: PolyArray | list[PolyArray]) -> None:
        self.intersect.reset(obstacle_polygons)
        self.enclosing.reset(obstacle_polygons)

    def check(self, ship_polygon: PolyArray) -> bool:
        intersect = self.intersect.check(ship_polygon)
        enclosing = self.enclosing.check(ship_polygon)
        return intersect or enclosing
