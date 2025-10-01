import numpy as np
import numpy.typing as npt

from ...utils import PointsArray, SegmentsArray, PolyArray


class SegmentsIntersectCollisionChecker:

    def reset(self, obstacle_polygons: PolyArray | list[PolyArray]) -> None:
        if obstacle_polygons is PolyArray:
            obstacle_polygons = [obstacle_polygons]
        self.obstacle_polygons = obstacle_polygons
        #
        self.obstacle_segments = []
        for polygon in obstacle_polygons:
            self.obstacle_segments.append(self.ploygon2segments(polygon))
        self.obstacle_segments = np.concatenate(self.obstacle_segments, axis=0)

    def check(self, ship_polygon: PolyArray) -> bool:
        self.ship_segments = self.ploygon2segments(ship_polygon)
        collide = segments_intersect(self.ship_segments, self.obstacle_segments)
        return collide

    def ploygon2segments(self, polygon: PolyArray) -> SegmentsArray:
        polygon_ = np.concatenate([polygon[1:, :], polygon[0:1, :]], axis=0)
        segments = np.concatenate(
            [polygon[:, np.newaxis, :], polygon_[:, np.newaxis, :]],
            axis=1,
        )
        return segments


def segments_intersect(
    segments_A: SegmentsArray,
    segments_B: SegmentsArray,
) -> bool:
    PointsP1, PointsQ1, PointsP2, PointsQ2 = [], [], [], []
    for i in range(len(segments_A)):
        segments_B_num = len(segments_B)
        PointsP1.append(np.tile(segments_A[i, 0, :], (segments_B_num, 1)))
        PointsQ1.append(np.tile(segments_A[i, 1, :], (segments_B_num, 1)))
        PointsP2.append(segments_B[:, 0, :])
        PointsQ2.append(segments_B[:, 1, :])
    PointsP1 = np.concatenate(PointsP1, axis=0)
    PointsQ1 = np.concatenate(PointsQ1, axis=0)
    PointsP2 = np.concatenate(PointsP2, axis=0)
    PointsQ2 = np.concatenate(PointsQ2, axis=0)
    #
    ori1 = orientation(PointsP1, PointsQ1, PointsP2)
    ori2 = orientation(PointsP1, PointsQ1, PointsQ2)
    ori3 = orientation(PointsP2, PointsQ2, PointsP1)
    ori4 = orientation(PointsP2, PointsQ2, PointsQ1)
    if np.any((ori1 != ori2) & (ori3 != ori4)):
        return True
    #
    if np.any(ori1 == 0):
        on1 = onSegment(PointsP1, PointsP2, PointsQ1)
        if np.any((ori1 == 0) & (on1 == 1)):
            return True
    if np.any(ori2 == 0):
        on2 = onSegment(PointsP1, PointsQ2, PointsQ1)
        if np.any((ori2 == 0) & (on2 == 1)):
            return True
    if np.any(ori3 == 0):
        on3 = onSegment(PointsP2, PointsP1, PointsQ2)
        if np.any((ori3 == 0) & (on3 == 1)):
            return True
    if np.any(ori4 == 0):
        on4 = onSegment(PointsP2, PointsQ1, PointsP2)
        if np.any((ori4 == 0) & (on4 == 1)):
            return True
    return False


def orientation(
    A: PointsArray,
    B: PointsArray,
    C: PointsArray,
) -> npt.NDArray[np.int64]:
    fir = (B[:, 1] - A[:, 1]) * (C[:, 0] - B[:, 0])
    sec = (B[:, 0] - A[:, 0]) * (C[:, 1] - B[:, 1])
    cross = fir - sec
    ori = np.where(cross < 0, -1, 0)
    ori = np.where(cross > 0, 1, ori)
    return ori


def onSegment(
    A: PointsArray,
    B: PointsArray,
    C: PointsArray,
) -> npt.NDArray[np.int64]:
    BC = np.array([B, C])
    BCmax = np.max(BC, axis=0)
    BCmin = np.min(BC, axis=0)
    cond1 = (A[:, 0] <= BCmax[:, 0]) & (A[:, 0] >= BCmin[:, 0])
    cond2 = (A[:, 1] <= BCmax[:, 1]) & (A[:, 1] >= BCmin[:, 1])
    return np.where(cond1 & cond2, 1, 0)


# def check_interect(base_segment: NDArray, segments: list[NDArray]) -> bool:
#     p1 = base_segment[0, :]
#     p2 = base_segment[1, :]
#     p3 = np.array(segments)[:, 0, :]
#     p4 = np.array(segments)[:, 1, :]
#     #
#     t1 = (p1[0] - p2[0]) * (p3[:, 1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[:, 0])
#     t2 = (p1[0] - p2[0]) * (p4[:, 1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[:, 0])
#     t3 = (p3[:, 0] - p4[:, 0]) * (p1[1] - p3[:, 1]) + (p3[:, 1] - p4[:, 1]) * (
#         p3[:, 0] - p1[0]
#     )
#     t4 = (p3[:, 0] - p4[:, 0]) * (p2[1] - p3[:, 1]) + (p3[:, 1] - p4[:, 1]) * (
#         p3[:, 0] - p2[0]
#     )
#     return (t1 * t2 < 0) & (t3 * t4 < 0)
