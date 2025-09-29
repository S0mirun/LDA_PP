import typing
import numpy as np
import numpy.typing as npt

__all__ = [
    "Variables",
    "PointArray",
    "PointsArray",
    "SegmentArray",
    "SegmentsArray",
    "PolyArray",
    "Pose",
]

Variables     = npt.NDArray[np.float64]  # shape: arbitrary
PointArray    = npt.NDArray[np.float64]  # shape: (2,)
PointsArray   = npt.NDArray[np.float64]  # shape: (N, 2)
SegmentArray  = npt.NDArray[np.float64]  # shape: (2, 2)
SegmentsArray = npt.NDArray[np.float64]  # shape: (N, 2, 2)
PolyArray     = npt.NDArray[np.float64]  # shape: (N, 2)

Pose          = npt.NDArray[np.float64]  # shape: (3,)
