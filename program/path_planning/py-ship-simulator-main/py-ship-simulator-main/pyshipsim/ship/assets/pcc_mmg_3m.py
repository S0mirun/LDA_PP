import numpy as np

from ..core import SurfaceShip, Variables, PolyArray
from ..maneuver.mmg import pcc_3m
from ..response import linear_delay_ode_rhs
from ..utils import detail_ship_poly, rectangle_ship_poly

d2r = np.deg2rad


class PCC3mMMG(SurfaceShip):
    def __init__(self, solve_method: str = "euler", f2py: bool = True):
        super().__init__(solve_method)
        if f2py:
            msg = "esso_osaka_150m's f2py mpdule is not implemented."
            raise Exception(msg)
        else:
            self.maneuvering_model = pcc_3m.pyMMGModel()
        #
        self.reg_principal_particular(L=3.3663, B=0.6022)
        # register state variables
        self.reg_state_var(name="x_position_mid [m]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="u_velo [m/s]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="y_position_mid [m]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="vm_velo [m/s]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="psi [rad]", ub=np.inf, lb=-np.inf)
        self.reg_state_var(name="r_angvelo [rad/s]", ub=np.inf, lb=-np.inf)
        # register action variables
        self.reg_action_var(name="delta_rudder_p [rad]", ub=d2r(35), lb=d2r(105))
        self.reg_action_var(name="delta_rudder_s [rad]", ub=-d2r(105), lb=-d2r(35))
        self.reg_action_var(name="n_prop [rps]", ub=20, lb=0)
        self.reg_action_var(name="pitch_ang [deg]", ub=40, lb=-20)
        self.reg_action_var(name="n_bt [rps]", ub=30, lb=-30)
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
        x, u, w = state[0:6], action[0:5], action[5:7]
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


class PCC3mMMGWithActuaorResponse(PCC3mMMG):
    def __init__(self, solve_method: str = "euler", f2py: bool = True):
        super().__init__(solve_method, f2py)
        # register state variables
        self.reg_state_var(name="delta_rudder_p [rad]", ub=d2r(35), lb=d2r(105))
        self.reg_state_var(name="delta_rudder_s [rad]", ub=-d2r(105), lb=-d2r(35))
        self.reg_state_var(name="n_prop [rps]", ub=20, lb=0)
        self.reg_state_var(name="pitch_ang [deg]", ub=40, lb=-20)
        self.reg_state_var(name="n_bt [rps]", ub=30, lb=-30)
        # register action variables
        self.reg_action_var(
            name="delta_rudder_p_cmd [rad]",
            replaced_name="delta_rudder_p [rad]",
            ub=d2r(35),
            lb=d2r(105),
        )
        self.reg_action_var(
            name="delta_rudder_s_cmd [rad]",
            replaced_name="delta_rudder_s [rad]",
            ub=-d2r(105),
            lb=-d2r(35),
        )
        self.reg_action_var(
            name="n_prop_cmd [rps]",
            replaced_name="n_prop [rps]",
            ub=20,
            lb=0,
        )
        self.reg_action_var(
            name="pitch_ang_cmd [deg]",
            replaced_name="pitch_ang [deg]",
            ub=40,
            lb=-20,
        )
        self.reg_action_var(
            name="n_bt_cmd [rps]",
            replaced_name="n_bt [rps]",
            ub=30,
            lb=-30,
        )
        # register observation variables
        self.reg_observation_var(
            name="delta_rudder_p_hat [rad]", ub=d2r(35), lb=d2r(105)
        )
        self.reg_observation_var(
            name="delta_rudder_s_hat [rad]", ub=-d2r(105), lb=-d2r(35)
        )
        self.reg_observation_var(name="n_prop_hat [rps]", ub=20, lb=0)
        self.reg_observation_var(name="pitch_ang_hat [deg]", ub=40, lb=-20)
        self.reg_observation_var(name="n_bt_hat [rps]", ub=30, lb=-30)

    def ode_rhs(self, state: Variables, action: Variables) -> Variables:
        x = state[0:6]
        u = state[6:11]
        u_cmd = action[0:5]
        w = action[5:7]
        derivative_state = np.empty_like(state)
        derivative_state[0:6] = self.maneuvering_model.ode_rhs(x, u, w)
        derivative_state[6] = linear_delay_ode_rhs(u[0], u_cmd[0], K=d2r(20))
        derivative_state[7] = linear_delay_ode_rhs(u[1], u_cmd[1], K=d2r(20))
        derivative_state[8] = linear_delay_ode_rhs(u[2], u_cmd[2], K=20)
        derivative_state[9] = linear_delay_ode_rhs(u[3], u_cmd[3], K=20)
        derivative_state[10] = linear_delay_ode_rhs(u[4], u_cmd[4], K=20)
        return derivative_state

    def observe_state(self, np_random: np.random.Generator = None) -> Variables:
        scale1 = [0.03, 0.01, 0.03, 0.01, d2r(0.1), d2r(0.1)]
        scale2 = [0.0, 0.0, 0.0, 0.0, 0.0]
        return self.additive_normal_noise(
            self.state, scale1 + scale2, np_random=np_random
        )
