import numpy as np


def SideThrusterForce(
    rho_fresh,
    u_velo,
    lpp,
    D_bthrust,
    D_sthrust,
    n_bow_thruster,
    n_stern_thruster,
    aY_bow,
    aY_stern,
    aN_bow,
    aN_stern,
    KT_bow_forward,
    KT_bow_reverse,
    KT_stern_forward,
    KT_stern_reverse,
    x_bowthrust_loc,
    x_stnthrust_loc,
    Thruster_speed_max,
):
    g = 9.81
    side_thrust_force = np.empty((len(u_velo), 3))
    for i in range(len(side_thrust_force)):
        Fn = np.abs(u_velo[i] / np.sqrt(lpp * g))
        # Bow
        if n_bow_thruster[i] >= 0:
            BowThrust = (
                rho_fresh * D_bthrust**4 * n_bow_thruster[i] ** 2 * KT_bow_forward
            )
            BowMoment = BowThrust * x_bowthrust_loc
        else:
            BowThrust = (
                rho_fresh * D_bthrust**4 * n_bow_thruster[i] ** 2 * KT_bow_reverse
            )
            BowMoment = BowThrust * x_bowthrust_loc
        # Stern
        if n_stern_thruster[i] >= 0:
            SternThrust = (
                rho_fresh * D_sthrust**4 * n_stern_thruster[i] ** 2 * KT_stern_forward
            )
            SternMoment = SternThrust * x_stnthrust_loc
        else:
            SternThrust = (
                rho_fresh * D_sthrust**4 * n_stern_thruster[i] ** 2 * KT_stern_reverse
            )
            SternMoment = SternThrust * x_stnthrust_loc
        X_SideThruster = 0
        Y_SideThruster = (1 + aY_bow * Fn) * BowThrust + (
            1 + aY_stern * Fn
        ) * SternThrust
        N_SideThruster = (1 + aN_bow * Fn) * BowMoment + (
            1 + aN_stern * Fn
        ) * SternMoment
        # limitation of u_velocity
        if np.abs(u_velo) >= Thruster_speed_max:
            X_SideThruster, Y_SideThruster, N_SideThruster = 0, 0, 0
        side_thrust_force[i][0] = X_SideThruster
        side_thrust_force[i][1] = Y_SideThruster
        side_thrust_force[i][2] = N_SideThruster
    return side_thrust_force


def SideThrusterForce_2ndOrder(
    rho_fresh,
    u_velo,
    lpp,
    D_bthrust,
    D_sthrust,
    n_bow_thruster,
    n_stern_thruster,
    aY_bow,
    aY_stern,
    aN_bow,
    aN_stern,
    KT_bow_forward,
    KT_bow_reverse,
    KT_stern_forward,
    KT_stern_reverse,
    x_bowthrust_loc,
    x_stnthrust_loc,
    Thruster_speed_max,
):
    g = 9.81
    side_thrust_force = np.empty((len(u_velo), 3))
    for i in range(len(side_thrust_force)):
        Fn = np.abs(u_velo[i] / np.sqrt(lpp * g))
        # Bow
        if n_bow_thruster[i] >= 0:
            BowThrust = (
                rho_fresh * D_bthrust**4 * n_bow_thruster[i] ** 2 * KT_bow_forward
            )
            BowMoment = BowThrust * x_bowthrust_loc
        else:
            BowThrust = (
                rho_fresh * D_bthrust**4 * n_bow_thruster[i] ** 2 * KT_bow_reverse
            )
            BowMoment = BowThrust * x_bowthrust_loc
        # Stern
        if n_stern_thruster[i] >= 0:
            SternThrust = (
                rho_fresh * D_sthrust**4 * n_stern_thruster[i] ** 2 * KT_stern_forward
            )
            SternMoment = SternThrust * x_stnthrust_loc
        else:
            SternThrust = (
                rho_fresh * D_sthrust**4 * n_stern_thruster[i] ** 2 * KT_stern_reverse
            )
            SternMoment = SternThrust * x_stnthrust_loc
        X_SideThruster = 0
        Y_SideThruster = (
            1 + aY_bow[0] + aY_bow[1] * Fn + aY_bow[2] * Fn**2
        ) * BowThrust + (
            1 + aY_stern[0] + aY_stern[1] * Fn + aY_stern[2] * Fn**2
        ) * SternThrust
        N_SideThruster = (
            1 + aN_bow[0] + aN_bow[1] * Fn + aN_bow[2] * Fn**2
        ) * BowMoment + (
            1 + aN_stern[0] + aN_stern[1] * Fn + aN_stern[2] * Fn**2
        ) * SternMoment
        # limitation of u_velocity
        if np.abs(u_velo) >= Thruster_speed_max:
            X_SideThruster, Y_SideThruster, N_SideThruster = 0, 0, 0
        side_thrust_force[i][0] = X_SideThruster
        side_thrust_force[i][1] = Y_SideThruster
        side_thrust_force[i][2] = N_SideThruster
    return side_thrust_force
