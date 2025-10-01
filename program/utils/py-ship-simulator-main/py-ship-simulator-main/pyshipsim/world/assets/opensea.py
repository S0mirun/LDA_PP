import numpy as np

from ..core import World
from ..wind import WindProcess
from ..collision import StrictCollisionChecker


class OpenSea(World):
    def __init__(self, wind=WindProcess(), col=StrictCollisionChecker()):
        super().__init__(wind, col)
        #
        self.reg_state_var(name="true_wind_speed [m/s]", ub=np.inf, lb=0.0)
        self.reg_state_var(name="true_wind_direction [rad]", ub=2 * np.pi, lb=0.0)
        self.reg_observation_var(name="true_wind_speed_hat [m/s]", ub=np.inf, lb=0.0)
        self.reg_observation_var(
            name="true_wind_direction_hat [rad]", ub=2 * np.pi, lb=0.0
        )
