import numpy as np


class StationaryWindSpeed:

    def reset(self, u):
        # initialize
        u = check_plus(u)
        self.u = u
        self.t = 0.0
        return u

    def get_time(self):
        return self.t

    def get_state(self):
        return self.u

    def step(self, dt: float, np_random: np.random.Generator = None):
        self.t += dt
        return self.u


class StationaryWindDrection:

    def reset(self, gamma: float):
        # initialize
        gamma = gamma % (2 * np.pi)
        self.gamma = gamma
        self.t = 0.0
        return gamma

    def get_time(self):
        return self.t

    def get_state(self):
        return self.gamma

    def step(self, dt: float, np_random: np.random.Generator = None):
        self.t += dt
        return self.gamma


def check_plus(var):
    if var <= 0.0:
        return 1.0e-16
    return var
