import typing
import numpy as np
import numpy.typing as npt

from .wind import WindProcess
from .collision import SegmentsIntersectCollisionChecker
from ..utils import Variables, PolyArray


class World(object):

    def __init__(
        self,
        wind=WindProcess(),
        collision=SegmentsIntersectCollisionChecker(),
    ):
        self.STATE_NAME = []
        self.STATE_UPPER_BOUND = []
        self.STATE_LOWER_BOUND = []
        self.OBSERVATION_NAME = []
        self.OBSERVATION_UPPER_BOUND = []
        self.OBSERVATION_LOWER_BOUND = []
        self.OBSTACLE_POLYGONS = []
        #
        self.wind = wind
        self.collision = collision

    def reg_state_var(
        self,
        name: str,
        ub: float = np.inf,
        lb: float = -np.inf,
        replaced_name: str = None,
    ):
        if replaced_name in self.STATE_NAME:
            i = self.STATE_NAME.index(replaced_name)
            self.STATE_NAME[i] = name
            self.STATE_UPPER_BOUND[i] = ub
            self.STATE_LOWER_BOUND[i] = lb
        else:
            self.STATE_NAME.append(name)
            self.STATE_UPPER_BOUND.append(ub)
            self.STATE_LOWER_BOUND.append(lb)

    def reg_observation_var(
        self,
        name: str,
        ub: float = np.inf,
        lb: float = -np.inf,
        replaced_name: str = None,
    ):
        if replaced_name in self.OBSERVATION_NAME:
            i = self.OBSERVATION_NAME.index(replaced_name)
            self.OBSERVATION_NAME[i] = name
            self.OBSERVATION_UPPER_BOUND[i] = ub
            self.OBSERVATION_LOWER_BOUND[i] = lb
        else:
            self.OBSERVATION_NAME.append(name)
            self.OBSERVATION_UPPER_BOUND.append(ub)
            self.OBSERVATION_LOWER_BOUND.append(lb)

    def reg_obstacle_poly(self, poly: PolyArray):
        self.OBSTACLE_POLYGONS.append(poly)

    def reset(self, state: Variables):
        self.t = 0.0
        self.wind.reset(state)
        if len(self.OBSTACLE_POLYGONS) > 0:
            self.collision.reset(self.OBSTACLE_POLYGONS)

    def step(self, dt: float, np_random=None):
        np_random = np.random if np_random is None else np_random
        state_n = self.wind.step(dt, np_random=np_random)
        self.t += dt
        return state_n

    def get_time(self):
        return self.t

    def get_state(self) -> Variables:
        return self.wind.get_state()

    def observe_state(self, np_random: np.random.Generator = None) -> Variables:
        return self.wind.get_state()

    @staticmethod
    def additive_normal_noise(
        state: Variables,
        scale: npt.ArrayLike,
        np_random: np.random.Generator = None,
    ) -> Variables:
        np_random = np.random if np_random is None else np_random
        additive_noise = np_random.normal(
            loc=np.zeros_like(scale),
            scale=np.array(scale),
        )
        observation = state + additive_noise
        return observation

    def check_collision(self, ship_poly: PolyArray) -> bool:
        if len(self.OBSTACLE_POLYGONS) > 0:
            return self.collision.check(ship_poly)
        return False
