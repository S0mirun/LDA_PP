import numpy as np
import numpy.typing as npt

from ..utils import Variables, PolyArray


class SurfaceShip(object):
    implemented_solve_method = ["euler", "rk4"]

    def __init__(self, solve_method: str = "euler"):
        assert solve_method is None or solve_method in self.implemented_solve_method
        self.solve_method = solve_method
        #
        self.STATE_NAME = []
        self.STATE_UPPER_BOUND = []
        self.STATE_LOWER_BOUND = []
        self.ACTION_NAME = []
        self.ACTION_UPPER_BOUND = []
        self.ACTION_LOWER_BOUND = []
        self.OBSERVATION_NAME = []
        self.OBSERVATION_UPPER_BOUND = []
        self.OBSERVATION_LOWER_BOUND = []

    def reg_principal_particular(self, L: float, B: float):
        self.L = L
        self.B = B

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

    def reg_action_var(
        self,
        name: str,
        ub: float = np.inf,
        lb: float = -np.inf,
        replaced_name: str = None,
    ):
        if replaced_name in self.ACTION_NAME:
            i = self.ACTION_NAME.index(replaced_name)
            self.ACTION_NAME[i] = name
            self.ACTION_UPPER_BOUND[i] = ub
            self.ACTION_LOWER_BOUND[i] = lb
        else:
            self.ACTION_NAME.append(name)
            self.ACTION_UPPER_BOUND.append(ub)
            self.ACTION_LOWER_BOUND.append(lb)

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

    def reset(self, state: Variables):
        self.state = state

    def step(self, action: Variables, dt: float):
        state = self.state
        if self.solve_method == "euler":
            dstate = self.ode_rhs(state, action)
            state_n = state + dt * dstate
        elif self.solve_method == "rk4":
            k1 = self.ode_rhs(state, action)
            k2 = self.ode_rhs(state + 0.5 * k1 * dt, action)
            k3 = self.ode_rhs(state + 0.5 * k2 * dt, action)
            k4 = self.ode_rhs(state + 1.0 * k3 * dt, action)
            dstate = (1.0 * k1 + 2.0 * k2 + 2.0 * k3 + 1.0 * k4) / 6.0
            state_n = state + dt * dstate
        self.state = state_n
        return state_n

    def ode_rhs(self, state: Variables, action: Variables) -> Variables:
        raise NotImplementedError

    def get_state(self) -> Variables:
        return self.state

    def observe_state(self, np_random: np.random.Generator = None) -> Variables:
        return self.state

    @staticmethod
    def additive_normal_noise(
        state: Variables, scale: npt.ArrayLike, np_random: np.random.Generator = None
    ) -> Variables:
        np_random = np.random if np_random is None else np_random
        additive_noise = np_random.normal(
            loc=np.zeros_like(scale),
            scale=np.array(scale),
        )
        observation = state + additive_noise
        return observation

    def ship_shape(self, state: Variables) -> PolyArray:
        raise NotImplementedError

    def ship_appearance(self, state: Variables) -> PolyArray:
        return self.ship_shape(state)
