import numpy as np

from ..core import SurfaceShip, Variables, PolyArray
from ..maneuver.mmg import esso_osaka_150m
from ..response import linear_delay_ode_rhs
from ..utils import detail_ship_poly, rectangle_ship_poly

d2r = np.deg2rad


class EssoOsaka150mMMG(SurfaceShip):
    def __init__(self, solve_method: str = "euler", f2py: bool = True):
        super().__init__(solve_method)
        if f2py and hasattr(esso_osaka_150m, "f2pyMMGModel"):
            self.maneuvering_model = esso_osaka_150m.f2pyMMGModel()
        elif f2py and not hasattr(esso_osaka_150m, "f2pyMMGModel"):
            msg = "You can't use esso_osaka_150m's f2py mpdule. Please compile fortran code."
            raise Exception(msg)
        else:
            msg = "esso_osaka_150m's py mpdule is not implemented."
            raise Exception(msg)
        #
        self.reg_principal_particular(L=50 * 3.0, B=50 * 0.48925)
        # register state variables
        self.reg_state_var(name="x_position_mid [m]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="u_velo [m/s]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="y_position_mid [m]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="vm_velo [m/s]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="psi [rad]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="r_angvelo [rad/s]", ub=np.inf, lb=-np.inf)
        # register action variables
        self.reg_action_var(name="delta_rudder [rad]", ub=d2r(35), lb=-d2r(35))
        self.reg_action_var(name="n_prop [rps]", ub=20, lb=-20)
        self.reg_action_var(name="true_wind_speed [m/s]", ub=np.inf, lb=0.0)
        self.reg_action_var(name="true_wind_direction [rad]", ub=2 * np.pi, lb=0.0)
        # register observation variables
        self.reg_observation_var(name="x_position_mid_hat [m]", ub=np.inf, lb=-np.inf)
        self.reg_observation_var(name="u_velo_hat [m/s]", ub=np.inf, lb=-np.inf)
        self.reg_observation_var(name="y_position_mid_hat [m]", ub=np.inf, lb=-np.inf)
        self.reg_observation_var(name="vm_velo_hat [m/s]", ub=np.inf, lb=-np.inf)
        self.reg_observation_var(name="psi_hat [rad]", ub=np.inf, lb=-np.inf)
        self.reg_observation_var(name="r_angvelo_hat [rad/s]", ub=np.inf, lb=-np.inf)

    def ode_rhs(self, state: Variables, action: Variables) -> Variables:
        x, u, w = state[0:6], [action[0], action[1], 0.0, 0.0], action[2:4]
        derivative_state = np.empty_like(state)
        derivative_state[0:6] = self.maneuvering_model.ode_rhs(x, u, w)
        return derivative_state

    def observe_state(self, np_random: np.random.Generator = None) -> Variables:
        scale = [0.03, 0.01, 0.03, 0.01, d2r(0.1), d2r(0.1)]
        return self.additive_normal_noise(self.state, scale, np_random=np_random)

    def ship_shape(self, state: Variables) -> PolyArray:
        eta = state[[0, 2, 4]]
        polygon = rectangle_ship_poly(eta, self.L, self.B)
        return polygon

    def ship_appearance(self, state: Variables) -> PolyArray:
        eta = state[[0, 2, 4]]
        polygon = detail_ship_poly(eta, self.L, self.B)
        return polygon


class EssoOsaka150mMMGWithActuaorResponse(EssoOsaka150mMMG):
    def __init__(self, solve_method: str = "euler", f2py: bool = True):
        super().__init__(solve_method, f2py)
        # register state variables
        self.reg_state_var(name="delta_rudder [rad]", ub=d2r(35), lb=-d2r(35))
        self.reg_state_var(name="n_prop [rps]", ub=20, lb=-20)
        # register action variables
        self.reg_action_var(
            name="delta_rudder_cmd [rad]",
            replaced_name="delta_rudder [rad]",
            ub=d2r(35),
            lb=-d2r(35),
        )
        self.reg_action_var(
            name="n_prop_cmd [rps]",
            replaced_name="n_prop [rps]",
            ub=20,
            lb=-20,
        )
        # register observation variables
        self.reg_observation_var(name="delta_rudder_hat [rad]", ub=d2r(35), lb=-d2r(35))
        self.reg_observation_var(name="n_prop_hat [rps]", ub=20, lb=-20)

    def ode_rhs(self, state: Variables, action: Variables) -> Variables:
        x = state[0:6]
        u = [state[6], state[7], 0.0, 0.0]
        u_cmd = action[0:2]
        w = action[2:4]
        derivative_state = np.empty_like(state)
        derivative_state[0:6] = self.maneuvering_model.ode_rhs(x, u, w)
        derivative_state[6] = linear_delay_ode_rhs(u[0], u_cmd[0], K=d2r(20))
        derivative_state[7] = linear_delay_ode_rhs(u[1], u_cmd[1], K=20)
        return derivative_state

    def observe_state(self, np_random: np.random.Generator = None) -> Variables:
        scale = [0.03, 0.01, 0.03, 0.01, d2r(0.1), d2r(0.1), 0.0, 0.0]
        return self.additive_normal_noise(self.state, scale, np_random=np_random)
