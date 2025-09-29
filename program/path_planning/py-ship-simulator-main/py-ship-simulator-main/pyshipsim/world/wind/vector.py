import numpy as np
import numpy.typing as npt

from .random_wind import RandomWindSpeedMaki, RandomWindSpeedEM, RandomWindDirectionEM
from .stationary_wind import StationaryWindSpeed, StationaryWindDrection


class WindProcess:
    implemented_mode = ["stationary", "random", "random_EM", "random_MAKI"]

    def __init__(self, mode: str = "random"):
        """Constructor

        Args:
            mode (str, optional): Generation method handle. Defaults to "random".
        """
        assert mode is None or mode in self.implemented_mode
        self.mode = mode
        if mode == "stationary":
            self._u = StationaryWindSpeed()
            self._gamma = StationaryWindDrection()
        elif mode == "random" or mode == "random_EM":
            self._u = RandomWindSpeedEM()
            self._gamma = RandomWindDirectionEM()
        else:
            self._u = RandomWindSpeedMaki()
            self._gamma = RandomWindDirectionEM()

    def reset(self, w: npt.ArrayLike):
        u, gamma = w
        self._u.reset(u)
        if self.mode == "stationary":
            self._gamma.reset(gamma)
        else:
            self._gamma.reset(gamma, u)
        return w

    def get_time(self):
        return self._u.get_time()

    def get_state(self):
        u = self._u.get_state()
        gamma = self._gamma.get_state()
        return np.array([u, gamma])

    def step(self, dt: float, np_random: np.random.Generator = None):
        u = self._u.step(dt, np_random=np_random)
        gamma = self._gamma.step(dt, np_random=np_random)
        return np.array([u, gamma])
