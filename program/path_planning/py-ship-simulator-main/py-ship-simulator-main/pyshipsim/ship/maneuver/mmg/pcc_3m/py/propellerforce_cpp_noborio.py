"""
小池さんの逆転時の不均衡力を考慮したCPPモデル。

! 引数 !
u_velo         : surge船速(m/s)
vm_velo        : sway船速(m/s)
r_angvelo      : yaw船速(rad/s)
n_prop         : プロペラ回転数(rps)
rho_fresh      : 密度ρ(?)
lpp            : Lpp(?)
dia_prop       : プロペラ直径(?)
pitch_ang_prop : プロペラ翼角(rad)
draft, pitchr, xP_nd, AAi, BBi, CCi, Jmin, alpha_p, kt_coeff_x,kt_coeff_y,kt_coeff_N, degrees_propmodel : 使用しない

! 変数 !
propforce : 4変数のndarray。それぞれ(XP, YP, NP, Js)
"""

import numpy as np


def KT(w, x, y, M, N):
    vec_size = (M + 1) * (N + 1)
    X = np.empty((vec_size, y.size))
    i_col = 0
    for i in range(M + 1):
        for j in range(N + 1):
            X[i_col, :] = (x**i) * (y**j)
            i_col += 1
    return np.dot(w[:vec_size].T, X)


def get_propellerforce4(
    u_velo,
    vm_velo,
    r_angvelo,
    n_prop,
    rho_fresh,
    lpp,
    draft,
    dia_prop,
    pitchr,
    xP_nd,
    AAi,
    BBi,
    CCi,
    Jmin,
    alpha_p,
    kt_coeff_x,
    kt_coeff_y,
    kt_coeff_N,
    pitch_ang_prop,
    degrees_propmodel,
):
    ##########################
    ### Force of Propeller ###
    ##########################
    U_ship = np.sqrt(u_velo**2 + vm_velo**2)
    v_nd = np.where(
        np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(vm_velo.shape), vm_velo / U_ship
    )
    r_nd = np.where(
        np.abs(U_ship) < 1.0e-5,
        1.0e-7 * np.ones(r_angvelo.shape),
        r_angvelo * lpp / U_ship,
    )
    U_ship = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(U_ship.shape), U_ship)

    Js = np.where(
        np.abs(n_prop) < np.finfo(float).eps,  # avoid 0 divid for J_prop at small
        np.array(1.0e4),
        np.where(
            np.abs(u_velo) < np.finfo(float).eps,  # avoid j==0
            np.array(np.finfo(float).eps),
            u_velo / (dia_prop * n_prop),
        ),
    )

    # Thrust of propeller
    def kt_x_general2(pitch_ang_prop, pitch_ang_prop0, pitch_ang_max, Js, Jsx, C0, C1):
        def func1(alpha, Js, beta):
            # ktx = -1 / 120 * np.abs( pitch_ang_prop - 10 ) * Js + 1 / 120 * (pitch_ang_prop - 10)
            ktx = -alpha * Js + beta
            return ktx

        def func2(X0, X1, Jsx, Js, Jx0, alpha):
            ktx = (X1 - X0) * Js / Jsx + alpha * Jx0
            return ktx

        def func3(alpha, Js, Jx0):
            ktx = -alpha * (Js - Jx0)
            return ktx

        Jx0_max = 0.70 * np.pi * np.tan(np.deg2rad(pitch_ang_max - pitch_ang_prop0))
        Jx0 = 0.70 * np.pi * np.tan(np.deg2rad(pitch_ang_prop - pitch_ang_prop0))
        alpha = C1 / Jx0_max
        # if np.abs(pitch_ang_max - pitch_ang_prop0) < 1.0e-5:
        #     X1 = 0.0
        # else:
        X1 = alpha * Jx0
        X0 = C0 * (pitch_ang_prop - pitch_ang_prop0) / (pitch_ang_max - pitch_ang_prop0)
        # Jsx = np.abs( Jsx0 * (pitch_ang_prop-pitch_ang_prop0) / (pitch_ang_max-pitch_ang_prop0) )

        # if np.abs(Jx0) < 1.0e-5:
        #     beta = 0.0
        # else:
        #     beta    = X0 - alpha * Jsx
        beta = X0 - alpha * Jsx
        beta1 = X0 + alpha * Jsx

        if pitch_ang_prop > pitch_ang_prop0:
            if Js <= -Jsx:
                ktx = func1(alpha, Js, beta)
            elif -Jsx < Js <= 0:
                ktx = func2(X0, X1, Jsx, Js, Jx0, alpha)
            else:
                ktx = func3(alpha, Js, Jx0)

        else:
            if Jsx <= Js:
                ktx = func1(alpha, Js, beta1)
            elif 0.0 < Js <= Jsx:
                ktx = func2(X0, X1, -Jsx, Js, Jx0, alpha)
            else:
                ktx = func3(alpha, Js, Jx0)

        return ktx

    def kt_y_N_general3(Fy_m20, pitch_ang_prop, pitch_ang_prop0, pitch_ang_min):
        kty = (
            Fy_m20
            * ((pitch_ang_prop0 - pitch_ang_prop) / (pitch_ang_prop0 - pitch_ang_min))
            ** 2
        )
        ktN = -0.5 * kty
        if pitch_ang_prop > 10:
            kty = 0.0
            ktN = 0.0
        return kty, ktN

    pitch_ang_prop0 = 10
    pitch_ang_max = 40
    pitch_ang_min = -20

    # X parameters
    Jsx = 0.5
    C0 = 0.4
    C1 = 0.6

    # if Js > 1.0:
    #     Js = 1.0
    # elif Js < -1.0:
    #     Js = -1.0

    KT_x = kt_x_general2(
        np.rad2deg(pitch_ang_prop), pitch_ang_prop0, pitch_ang_max, Js, Jsx, C0, C1
    )

    # KT_y = kt_y_linear(np.rad2deg(pitch_ang_prop), Js)
    # KM_n = kt_N_linear(np.rad2deg(pitch_ang_prop), Js)

    # Js_fy_index     = np.where((pitch==-20*np.pi/180)&(np.abs(Js)<0.01))
    # Fx_ave_m20      = np.mean(Fx[Js_fy_index])
    # Fy_ave_m20      = 0.33 * C1 - 0.03 # Simple method
    Fy_ave_m20 = 0.178  # Js =0
    # Fy_ave_m20 =0.20 #-0.3 < Js < 0.3

    KT_y, KM_n = kt_y_N_general3(
        Fy_ave_m20, np.rad2deg(pitch_ang_prop), pitch_ang_prop0, pitch_ang_min
    )

    XP = np.zeros((1, 1))
    YP = np.zeros((1, 1))
    NP = np.zeros((1, 1))

    XP[:, :] = np.where(
        n_prop >= 0, rho_fresh * n_prop**2.0 * dia_prop**4.0 * KT_x, 0.0
    )

    if pitch_ang_prop <= np.deg2rad(10):
        YP[:, :] = np.where(
            n_prop >= 0, rho_fresh * n_prop**2.0 * dia_prop**4.0 * KT_y, 0.0
        )
        NP[:, :] = np.where(
            n_prop >= 0, rho_fresh * n_prop**2.0 * dia_prop**4.0 * lpp * KM_n, 0.0
        )
    else:
        YP[:, :] = 0.0
        NP[:, :] = 0.0

    propforce = np.concatenate((XP, YP, NP, Js), axis=1)

    return propforce
