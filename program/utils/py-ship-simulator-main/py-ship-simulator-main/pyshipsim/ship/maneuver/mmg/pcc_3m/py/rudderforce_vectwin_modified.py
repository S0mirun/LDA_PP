from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

pcc_dir = Path(__file__).parents[1]


def get_rudderforceVecTwin(
    ei,
    fi,
    gi,
    asi,
    api,
    bsi,
    bpi,
    dsi,
    dpi,
    esi,
    epi,
    gsi,
    gpi,
    hsi,
    hpi,
    ksi,
    kpi,
    rsi,
    rpi,
    delta_rudder_s,
    delta_rudder_p,
    n_prop,
    J_prop,
    pitchr,
    beta,
    x_location_rudder,
    r_angvelo,
    Csi,
    Cpi,
    one_minus_wprop,
    kx_rudder,
    lambda_rudder,
    dia_prop,
    vm_velo,
    u_velo,
    lpp,
    rho_fresh,
    area_rudder,
    XP,
    one_minus_t_prop,
    switch_ur_rudder,
    switch_cn_rudder,
):

    # なぜかこのファイルがsubroutine内にいないことになっていたので
    """os.chdir('Hydro/MMG/subroutine')
    path = os.getcwd()
    print('path=',path)"""

    # interaction between ship and rudder
    t_rudder = ei[0] + ei[1] * J_prop + ei[2] * J_prop**2
    ah_rudder_cal = fi[0] + fi[1] * J_prop + fi[2] * J_prop**2
    xh_rudder_nd_cal = gi[0] + gi[1] * J_prop + gi[2] * J_prop**2

    ah_rudder = np.where(
        ah_rudder_cal > 0.60556,
        0.60556,
        np.where(ah_rudder_cal < 0.016, 0.016, ah_rudder_cal),
    )
    xh_rudder_nd = np.where(
        xh_rudder_nd_cal > -0.159,
        -0.159,
        np.where(xh_rudder_nd_cal < -0.4344, -0.4344, xh_rudder_nd_cal),
    )

    xH_rudder = xh_rudder_nd * lpp
    uprop = u_velo * one_minus_wprop
    pitch = dia_prop * pitchr

    # rudder inflow velocity due to propeller for starboard and port rudder
    hight_rudder = np.sqrt(area_rudder * lambda_rudder)
    eta_rudder = dia_prop / hight_rudder

    # effective wake fraction for starboard and port rudder
    one_minus_wrudder_s = (
        asi[0]
        + asi[1] * delta_rudder_s
        + asi[2] * delta_rudder_s**2
        + asi[3] * delta_rudder_s**3
    )
    one_minus_wrudder_p = (
        api[0]
        + api[1] * delta_rudder_p
        + api[2] * delta_rudder_p**2
        + api[3] * delta_rudder_p**3
    )

    epsilon_rudder_s = one_minus_wrudder_s / one_minus_wprop
    epsilon_rudder_p = one_minus_wrudder_p / one_minus_wprop

    kappa_rudder_s = kx_rudder / epsilon_rudder_s
    kappa_rudder_p = kx_rudder / epsilon_rudder_p

    KT = np.where(
        n_prop > np.finfo(float).eps,  # n>0
        XP / (rho_fresh * dia_prop**4 * n_prop**2 * (one_minus_t_prop)),
        np.where(
            np.abs(n_prop) > np.finfo(float).eps,  # n<0
            XP / (rho_fresh * dia_prop**4 * n_prop**2),
            np.array(0.0),  # n = 0
        ),
    )

    if switch_ur_rudder == 1:
        uR_p_s = np.where(
            (n_prop >= 0) & (KT > 0),
            epsilon_rudder_s
            * np.sqrt(
                eta_rudder
                * (
                    uprop
                    + kappa_rudder_s
                    * (
                        np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2 / (np.pi))
                        - uprop
                    )
                )
                ** 2
                + (1 - eta_rudder) * uprop**2
            ),  # <= normarl mmg model for low speed (Yoshimura's)
            0.0,
        )  ### note:J_prop=Js at backward !!!!)
        uR_p_p = np.where(
            (n_prop >= 0) & (KT > 0),
            epsilon_rudder_p
            * np.sqrt(
                eta_rudder
                * (
                    uprop
                    + kappa_rudder_p
                    * (
                        np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2 / (np.pi))
                        - uprop
                    )
                )
                ** 2
                + (1 - eta_rudder) * uprop**2
            ),  # <= normarl mmg model for low speed (Yoshimura's)
            0.0,
        )  ### note:J_prop=Js at backward !!!!)
    else:
        print("switch_ur error")
        sys.exit()

    # hydrodynamic neutral angle of starboard and port rudder
    slip_ratio_cal = np.where(
        np.abs(n_prop) < np.finfo(float).eps,  # avoid 0 divid for slip ratio at small n
        np.array(1.2),
        1 - uprop / (n_prop * pitch),
    )

    slip_ratio = np.where(
        slip_ratio_cal > 1.2, 1.2, np.where(slip_ratio_cal < 0, 0, slip_ratio_cal)
    )

    # Rudder angle
    delta_rudder_s0 = bsi[0] + bsi[1] * slip_ratio
    delta_rudder_p0 = bpi[0] + bpi[1] * slip_ratio

    # effective rudder angle 1 considering delta_rudder0
    delta_rudder_s1 = delta_rudder_s - delta_rudder_s0
    delta_rudder_p1 = delta_rudder_p - delta_rudder_p0

    # variation of inflow rudder angle due to interaction between starboard and port rudder
    cs1 = dsi[0] + dsi[1] * slip_ratio + dsi[2] * slip_ratio**2
    cp1 = dpi[0] + dpi[1] * slip_ratio + dpi[2] * slip_ratio**2
    cs2 = esi[0] + esi[1] * slip_ratio + esi[2] * slip_ratio**2
    cp2 = epi[0] + epi[1] * slip_ratio + epi[2] * slip_ratio**2
    delta_rudder_s_p = np.where(
        delta_rudder_s1 > 0,
        (cs1 * delta_rudder_p + cs2 * delta_rudder_p**2)
        * np.abs(np.cos(delta_rudder_s1)),
        cs1 * delta_rudder_p + cs2 * delta_rudder_p**2,
    )
    delta_rudder_p_s = np.where(
        delta_rudder_p1 < 0,
        (cp1 * delta_rudder_s + cp2 * delta_rudder_s**2)
        * np.abs(np.cos(delta_rudder_p1)),
        cp1 * delta_rudder_s + cp2 * delta_rudder_s**2,
    )

    # effective rudder angle 2 considering delta_rudder1 and delta_rudder_(s_p)(p_s)
    delta_rudder_s2 = delta_rudder_s1 - delta_rudder_s_p
    delta_rudder_p2 = delta_rudder_p1 - delta_rudder_p_s

    # flow straightening coefficient of sway velocity for starboard and port rudder
    ganma_rudder_s_cal1 = (
        rsi[0]
        + rsi[1] * np.abs(beta)
        + (rsi[2] + rsi[3] * np.abs(beta)) * delta_rudder_s2
    ) * np.abs(beta)
    ganma_rudder_s_cal2 = (
        rsi[4]
        + rsi[5] * np.abs(beta)
        + (rsi[6] + rsi[7] * np.abs(beta)) * delta_rudder_s2
    ) * np.abs(beta)
    ganma_rudder_s_cal3 = (
        rsi[8]
        + rsi[9] * np.abs(beta)
        + (rsi[10] + rsi[11] * np.abs(beta)) * delta_rudder_s2
    ) * np.abs(beta)
    ganma_rudder_s_cal4 = (
        rsi[12]
        + rsi[13] * np.abs(beta)
        + (rsi[14] + rsi[15] * np.abs(beta)) * delta_rudder_s2
    ) * np.abs(beta)
    ganma_rudder_p_cal1 = (
        rpi[0]
        + rpi[1] * np.abs(beta)
        + (rpi[2] + rpi[3] * np.abs(beta)) * delta_rudder_p2
    ) * np.abs(beta)
    ganma_rudder_p_cal2 = (
        rpi[4]
        + rpi[5] * np.abs(beta)
        + (rpi[6] + rpi[7] * np.abs(beta)) * delta_rudder_p2
    ) * np.abs(beta)
    ganma_rudder_p_cal3 = (
        rpi[8]
        + rpi[9] * np.abs(beta)
        + (rpi[10] + rpi[11] * np.abs(beta)) * delta_rudder_p2
    ) * np.abs(beta)
    ganma_rudder_p_cal4 = (
        rpi[12]
        + rpi[13] * np.abs(beta)
        + (rpi[14] + rpi[15] * np.abs(beta)) * delta_rudder_p2
    ) * np.abs(beta)

    ganma_rudder_s1 = np.where(
        ganma_rudder_s_cal1 < 0.0,
        0.0,
        np.where(ganma_rudder_s_cal1 > 0.8, 0.8, ganma_rudder_s_cal1),
    )
    ganma_rudder_s2 = np.where(
        ganma_rudder_s_cal2 < 0.0,
        0.0,
        np.where(ganma_rudder_s_cal2 > 0.8, 0.8, ganma_rudder_s_cal2),
    )
    ganma_rudder_s3 = np.where(
        ganma_rudder_s_cal3 < 0.0,
        0.0,
        np.where(ganma_rudder_s_cal3 > 0.8, 0.8, ganma_rudder_s_cal3),
    )
    ganma_rudder_s4 = np.where(
        ganma_rudder_s_cal4 < 0.0,
        0.0,
        np.where(ganma_rudder_s_cal4 > 1.0, 1.0, ganma_rudder_s_cal4),
    )
    ganma_rudder_p1 = np.where(
        ganma_rudder_p_cal1 < 0.0,
        0.0,
        np.where(ganma_rudder_p_cal1 > 1.0, 1.0, ganma_rudder_p_cal1),
    )
    ganma_rudder_p2 = np.where(
        ganma_rudder_p_cal2 < 0.0,
        0.0,
        np.where(ganma_rudder_p_cal2 > 0.8, 0.8, ganma_rudder_p_cal2),
    )
    ganma_rudder_p3 = np.where(
        ganma_rudder_p_cal3 < 0.0,
        0.0,
        np.where(ganma_rudder_p_cal3 > 0.8, 0.8, ganma_rudder_p_cal3),
    )
    ganma_rudder_p4 = np.where(
        ganma_rudder_p_cal4 < 0.0,
        0.0,
        np.where(ganma_rudder_p_cal4 > 0.8, 0.8, ganma_rudder_p_cal4),
    )

    ganma_rudder_s = np.where(
        beta >= 0,
        np.where(delta_rudder_s2 < 0, ganma_rudder_s1, ganma_rudder_s3),
        np.where(delta_rudder_s2 < 0, ganma_rudder_s2, ganma_rudder_s4),
    )

    ganma_rudder_p = np.where(
        beta >= 0,
        np.where(delta_rudder_p2 <= 0, ganma_rudder_p1, ganma_rudder_p3),
        np.where(delta_rudder_p2 <= 0, ganma_rudder_p2, ganma_rudder_p4),
    )

    # flow straightening coefficient of yaw rate for starboard and port rudder
    lR_s = np.where(
        r_angvelo >= 0,
        np.where(
            delta_rudder_s2 >= 0, Csi[0] * x_location_rudder, Csi[1] * x_location_rudder
        ),
        np.where(
            delta_rudder_s2 >= 0, Csi[2] * x_location_rudder, Csi[3] * x_location_rudder
        ),
    )

    lR_p = np.where(
        r_angvelo >= 0,
        np.where(
            delta_rudder_p2 >= 0, Cpi[0] * x_location_rudder, Cpi[1] * x_location_rudder
        ),
        np.where(
            delta_rudder_p2 >= 0, Cpi[2] * x_location_rudder, Cpi[3] * x_location_rudder
        ),
    )

    # decrement ratio of inflow velocity for starboard and port rudder
    fs1 = gsi[0] + gsi[1] * slip_ratio + gsi[2] * slip_ratio**2
    fp1 = gpi[0] + gpi[1] * slip_ratio + gpi[2] * slip_ratio**2
    fs2 = hsi[0] + hsi[1] * slip_ratio + hsi[2] * slip_ratio**2
    fp2 = hpi[0] + hpi[1] * slip_ratio + hpi[2] * slip_ratio**2
    fs3 = ksi[0] + ksi[1] * slip_ratio + ksi[2] * slip_ratio**2
    fp3 = kpi[0] + kpi[1] * slip_ratio + kpi[2] * slip_ratio**2
    uR_decrease_ratio_s_cal = np.where(
        delta_rudder_s2 > 0, fs1 + fs2 * delta_rudder_s2 + fs3 * delta_rudder_s2**2, 1.0
    )
    uR_decrease_ratio_p_cal = np.where(
        delta_rudder_p2 < 0, fp1 + fp2 * delta_rudder_p2 + fp3 * delta_rudder_p2**2, 1.0
    )
    uR_decrease_ratio_s = np.where(
        uR_decrease_ratio_s_cal > 1.0, 1.0, uR_decrease_ratio_s_cal
    )
    uR_decrease_ratio_p = np.where(
        uR_decrease_ratio_p_cal > 1.0, 1.0, uR_decrease_ratio_p_cal
    )

    # effective rudder inflow velocity for starboard and port rudder
    vR_s = -ganma_rudder_s * vm_velo - lR_s * r_angvelo
    vR_p = -ganma_rudder_p * vm_velo - lR_p * r_angvelo

    uR_s = uR_decrease_ratio_s * uR_p_s
    uR_p = uR_decrease_ratio_p * uR_p_p

    u_rudder_s = uR_s
    u_rudder_p = uR_p

    UUR_s = np.sqrt(uR_s**2 + (-ganma_rudder_s * vm_velo) ** 2)
    UUR_p = np.sqrt(uR_p**2 + (-ganma_rudder_s * vm_velo) ** 2)

    # data to watch [deg.] Do not use in code!!
    # aR_s_deg               = aR_s             * 180/ np.pi
    # aR_p_deg               = aR_p             * 180/ np.pi
    # delta_rudder_s_deg     = delta_rudder_s   * 180/ np.pi
    # delta_rudder_p_deg     = delta_rudder_p   * 180/ np.pi
    # delta_rudder_s0_deg    = delta_rudder_s0  * 180/ np.pi
    # delta_rudder_p0_deg    = delta_rudder_p0  * 180/ np.pi
    # delta_rudder_s1_deg    = delta_rudder_s1  * 180/ np.pi
    # delta_rudder_p1_deg    = delta_rudder_p1  * 180/ np.pi
    # delta_rudder_s_p_deg   = delta_rudder_s_p * 180/ np.pi
    # delta_rudder_p_s_deg   = delta_rudder_p_s * 180/ np.pi
    # delta_rudder_s2_deg    = delta_rudder_s2  * 180/ np.pi
    # delta_rudder_p2_deg    = delta_rudder_p2  * 180/ np.pi

    # calculating cn from csv by spline
    VecTwin_Cns = pd.read_csv(
        f"{pcc_dir}/input_csv/VecTwin_Cns.csv", header=0, index_col=0
    )
    VecTwin_Cnp = pd.read_csv(
        f"{pcc_dir}/input_csv/VecTwin_Cnp.csv", header=0, index_col=0
    )

    Cns_curve = interp1d(
        VecTwin_Cns.loc[:, "rudderangle_s"], VecTwin_Cns.loc[:, "cns"], kind="cubic"
    )
    Cnp_curve = interp1d(
        VecTwin_Cnp.loc[:, "rudderangle_p"], VecTwin_Cnp.loc[:, "cnp"], kind="cubic"
    )

    aR_s = delta_rudder_s2 - np.arctan2(vR_s, uR_s)
    aR_p = delta_rudder_p2 - np.arctan2(vR_p, uR_p)

    if switch_cn_rudder == 0:
        Cns = Cns_curve(aR_s)  # CFD result spline
        Cnp = Cnp_curve(aR_p)  # CFD result spline
    elif switch_cn_rudder == 1:
        Cns = np.where(
            aR_s > 0.654498469,
            -0.8449 * aR_s**2 + 2.0621 * aR_s + 0.6387,
            np.where(
                aR_s < -0.654498469,
                0.8449 * aR_s**2 + 2.0621 * aR_s - 0.6387,
                2.6252 * np.sin(aR_s),
            ),
        )  # Kang's cns model
        Cnp = np.where(
            aR_p > 0.654498469,
            -0.8449 * aR_p**2 + 2.0621 * aR_p + 0.6387,
            np.where(
                aR_p < -0.654498469,
                0.8449 * aR_p**2 + 2.0621 * aR_p - 0.6387,
                2.6252 * np.sin(aR_p),
            ),
        )  # Kang's cnp model

    FNs = 0.5 * rho_fresh * area_rudder * Cns * UUR_s**2  # / 9.80665
    FNp = 0.5 * rho_fresh * area_rudder * Cnp * UUR_p**2  # / 9.80665

    # calculating rudder force
    XR = -(1 - t_rudder) * (FNs * np.sin(delta_rudder_s) + FNp * np.sin(delta_rudder_p))
    YR = -(1 + ah_rudder) * (
        FNs * np.cos(delta_rudder_s) + FNp * np.cos(delta_rudder_p)
    )
    NR = -(x_location_rudder + ah_rudder * xH_rudder) * (
        FNs * np.cos(delta_rudder_s) + FNp * np.cos(delta_rudder_p)
    )

    if u_velo < 0:
        XR = XR * 0.0
        YR = YR * 0.0
        NR = NR * 0.0

    rudderforce = np.concatenate(
        (XR, YR, NR, aR_s, aR_p, UUR_s, UUR_p, u_rudder_s, u_rudder_p), axis=1
    )

    return rudderforce


def get_rudderforceVecTwin_forValidation(
    ei,
    fi,
    gi,
    asi,
    api,
    bsi,
    bpi,
    dsi,
    dpi,
    esi,
    epi,
    gsi,
    gpi,
    hsi,
    hpi,
    ksi,
    kpi,
    rsi,
    rpi,
    delta_rudder_s,
    delta_rudder_p,
    n_prop,
    J_prop,
    pitchr,
    beta,
    x_location_rudder,
    r_angvelo,
    Csi,
    Cpi,
    one_minus_wprop,
    kx_rudder,
    lambda_rudder,
    dia_prop,
    vm_velo,
    u_velo,
    lpp,
    rho_fresh,
    area_rudder,
    XP,
    one_minus_t_prop,
    switch_ur_rudder,
    switch_cn_rudder,
):

    # なぜかこのファイルがsubroutine内にいないことになっていたので
    """os.chdir('Hydro/MMG/subroutine')
    path = os.getcwd()
    print('path=',path)"""

    # interaction between ship and rudder
    t_rudder = ei[0] + ei[1] * J_prop + ei[2] * J_prop**2
    ah_rudder = fi[0] + fi[1] * J_prop + fi[2] * J_prop**2
    xh_rudder_nd = gi[0] + gi[1] * J_prop + gi[2] * J_prop**2
    xH_rudder = xh_rudder_nd * lpp
    uprop = u_velo * one_minus_wprop
    pitch = dia_prop * pitchr

    # hydrodynamic neutral angle of starboard and port rudder
    slip_ratio = np.where(
        np.abs(n_prop) < np.finfo(float).eps,  # avoid 0 divid for slip ratio at small n
        np.array(1.0),
        np.where(
            np.abs(uprop) < np.finfo(float).eps,  # avoid n==0
            np.array(np.finfo(float).eps),
            1 - uprop / (n_prop * pitch),
        ),
    )

    delta_rudder_s0 = bsi[0] + bsi[1] * slip_ratio
    delta_rudder_p0 = bpi[0] + bpi[1] * slip_ratio

    # effective rudder angle 1 considering delta_rudder0
    delta_rudder_s1 = delta_rudder_s - delta_rudder_s0
    delta_rudder_p1 = delta_rudder_p - delta_rudder_p0

    # variation of inflow rudder angle due to interaction between starboard and port rudder
    cs1 = dsi[0] + dsi[1] * slip_ratio + dsi[2] * slip_ratio**2
    cp1 = dpi[0] + dpi[1] * slip_ratio + dpi[2] * slip_ratio**2
    cs2 = esi[0] + esi[1] * slip_ratio + esi[2] * slip_ratio**2
    cp2 = epi[0] + epi[1] * slip_ratio + epi[2] * slip_ratio**2
    delta_rudder_s_p = np.where(
        delta_rudder_s1 > 0,
        (cs1 * delta_rudder_p + cs2 * delta_rudder_p**2)
        * np.abs(np.cos(delta_rudder_s1)),
        cs1 * delta_rudder_p + cs2 * delta_rudder_p**2,
    )
    delta_rudder_p_s = np.where(
        delta_rudder_p1 < 0,
        (cp1 * delta_rudder_s + cp2 * delta_rudder_s**2)
        * np.abs(np.cos(delta_rudder_p1)),
        cp1 * delta_rudder_s + cp2 * delta_rudder_s**2,
    )

    # effective rudder angle 2 considering delta_rudder1 and delta_rudder_(s_p)(p_s)
    delta_rudder_s2 = delta_rudder_s1 - delta_rudder_s_p
    delta_rudder_p2 = delta_rudder_p1 - delta_rudder_p_s

    # effective wake fraction for starboard and port rudder
    one_minus_wrudder_s = (
        asi[0]
        + asi[1] * delta_rudder_s
        + asi[2] * delta_rudder_s**2
        + asi[3] * delta_rudder_s**3
    )
    one_minus_wrudder_p = (
        api[0]
        + api[1] * delta_rudder_p
        + api[2] * delta_rudder_p**2
        + api[3] * delta_rudder_p**3
    )

    # rudder inflow velocity due to propeller for starboard and port rudder
    hight_rudder = np.sqrt(area_rudder * lambda_rudder)
    eta_rudder = dia_prop / hight_rudder

    epsilon_rudder_s = one_minus_wrudder_s / one_minus_wprop
    epsilon_rudder_p = one_minus_wrudder_p / one_minus_wprop

    kappa_rudder_s = kx_rudder / epsilon_rudder_s
    kappa_rudder_p = kx_rudder / epsilon_rudder_p

    KT = np.where(
        n_prop > np.finfo(float).eps,  # n>0
        XP / (rho_fresh * dia_prop**4 * n_prop**2 * (one_minus_t_prop)),
        np.where(
            np.abs(n_prop) > np.finfo(float).eps,  # n<0
            XP / (rho_fresh * dia_prop**4 * n_prop**2),
            np.array(0.0),  # n = 0
        ),
    )

    # urpr1_s = u_velo * one_minus_wrudder_s  + n_prop * dia_prop * kx_rudder_reverse * np.sqrt(8 * np.abs(KT)/ np.pi)
    # urpr1_p = u_velo * one_minus_wrudder_p  + n_prop * dia_prop * kx_rudder_reverse * np.sqrt(8 * np.abs(KT)/ np.pi)
    # urpr2_s = u_velo * one_minus_wrudder_s
    # urpr2_p = u_velo * one_minus_wrudder_p
    # ursq_s  = eta_rudder * np.sign(urpr1_s) * urpr1_s**2 + (1- eta_rudder) * urpr2_s**2 + cpr_rudder * u_velo**2
    # ursq_p  = eta_rudder * np.sign(urpr1_p) * urpr1_p**2 + (1- eta_rudder) * urpr2_p**2 + cpr_rudder * u_velo**2

    if switch_ur_rudder == 1:
        uR_p_s = np.where(
            (n_prop >= 0) & (KT > 0),
            epsilon_rudder_s
            * np.sqrt(
                eta_rudder
                * (
                    uprop
                    + kappa_rudder_s
                    * (
                        np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2 / (np.pi))
                        - uprop
                    )
                )
                ** 2
                + (1 - eta_rudder) * uprop**2
            ),  # <= normarl mmg model for low speed (Yoshimura's)
            # uprop * epsilon_rudder * np.sqrt(eta_rudder
            # * (1.0 + kappa_rudder * (np.sqrt(1 + 8 * KT / (np.pi * J_prop**2)) - 1.0))**2 + (1- eta_rudder)), # <= normarl mmg model
            0.0,
            # np.where( u_velo >= 0,
            #         np.sign(ursq_s) * np.sqrt(np.abs(ursq_s)),  # <= kitagawa's model for n<0 (3rd q)
            #         u_velo # <= uR=u at 4th quadrant
            #     )
        )  ### note:J_prop=Js at backward !!!!)
        uR_p_p = np.where(
            (n_prop >= 0) & (KT > 0),
            epsilon_rudder_p
            * np.sqrt(
                eta_rudder
                * (
                    uprop
                    + kappa_rudder_p
                    * (
                        np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2 / (np.pi))
                        - uprop
                    )
                )
                ** 2
                + (1 - eta_rudder) * uprop**2
            ),  # <= normarl mmg model for low speed (Yoshimura's)
            # uprop * epsilon_rudder * np.sqrt(eta_rudder
            # * (1.0 + kappa_rudder * (np.sqrt(1 + 8 * KT / (np.pi * J_prop**2)) - 1.0))**2 + (1- eta_rudder)), # <= normarl mmg model
            0.0,
            # np.where( u_velo >= 0,
            #         np.sign(ursq_p) * np.sqrt(np.abs(ursq_p)),  # <= kitagawa's model for n<0 (3rd q)
            #         u_velo # <= uR=u at 4th quadrant
            #     )
        )  ### note:J_prop=Js at backward !!!!)
    else:
        print("switch_ur error")
        sys.exit()

    # flow straightening coefficient of sway velocity for starboard and port rudder
    ganma_rudder_s = np.where(
        beta >= 0,
        np.where(
            delta_rudder_s2 < 0,
            (
                rsi[0]
                + rsi[1] * np.abs(beta)
                + (rsi[2] + rsi[3] * np.abs(beta)) * delta_rudder_s2
            )
            * np.abs(beta),
            (
                rsi[8]
                + rsi[9] * np.abs(beta)
                + (rsi[10] + rsi[11] * np.abs(beta)) * delta_rudder_s2
            )
            * np.abs(beta),
        ),
        np.where(
            delta_rudder_s2 < 0,
            (
                rsi[4]
                + rsi[5] * np.abs(beta)
                + (rsi[6] + rsi[7] * np.abs(beta)) * delta_rudder_s2
            )
            * np.abs(beta),
            (
                rsi[12]
                + rsi[13] * np.abs(beta)
                + (rsi[14] + rsi[15] * np.abs(beta)) * delta_rudder_s2
            )
            * np.abs(beta),
        ),
    )

    ganma_rudder_p = np.where(
        beta >= 0,
        np.where(
            delta_rudder_p2 <= 0,
            (
                rpi[0]
                + rpi[1] * np.abs(beta)
                + (rpi[2] + rpi[3] * np.abs(beta)) * delta_rudder_p2
            )
            * np.abs(beta),
            (
                rpi[8]
                + rpi[9] * np.abs(beta)
                + (rpi[10] + rpi[11] * np.abs(beta)) * delta_rudder_p2
            )
            * np.abs(beta),
        ),
        np.where(
            delta_rudder_p2 <= 0,
            (
                rpi[4]
                + rpi[5] * np.abs(beta)
                + (rpi[6] + rpi[7] * np.abs(beta)) * delta_rudder_p2
            )
            * np.abs(beta),
            (
                rpi[12]
                + rpi[13] * np.abs(beta)
                + (rpi[14] + rpi[15] * np.abs(beta)) * delta_rudder_p2
            )
            * np.abs(beta),
        ),
    )

    # flow straightening coefficient of yaw rate for starboard and port rudder
    lR_s = np.where(
        r_angvelo >= 0,
        np.where(
            delta_rudder_s2 >= 0, Csi[0] * x_location_rudder, Csi[1] * x_location_rudder
        ),
        np.where(
            delta_rudder_s2 >= 0, Csi[2] * x_location_rudder, Csi[3] * x_location_rudder
        ),
    )

    lR_p = np.where(
        r_angvelo >= 0,
        np.where(
            delta_rudder_p2 >= 0, Cpi[0] * x_location_rudder, Cpi[1] * x_location_rudder
        ),
        np.where(
            delta_rudder_p2 >= 0, Cpi[2] * x_location_rudder, Cpi[3] * x_location_rudder
        ),
    )

    # decrement ratio of inflow velocity for starboard and port rudder
    fs1 = gsi[0] + gsi[1] * slip_ratio + gsi[2] * slip_ratio**2
    fp1 = gpi[0] + gpi[1] * slip_ratio + gpi[2] * slip_ratio**2
    fs2 = hsi[0] + hsi[1] * slip_ratio + hsi[2] * slip_ratio**2
    fp2 = hpi[0] + hpi[1] * slip_ratio + hpi[2] * slip_ratio**2
    fs3 = ksi[0] + ksi[1] * slip_ratio + ksi[2] * slip_ratio**2
    fp3 = kpi[0] + kpi[1] * slip_ratio + kpi[2] * slip_ratio**2
    uR_decrease_ratio_s_cal = np.where(
        delta_rudder_s2 > 0, fs1 + fs2 * delta_rudder_s2 + fs3 * delta_rudder_s2**2, 1.0
    )
    uR_decrease_ratio_p_cal = np.where(
        delta_rudder_p2 < 0, fp1 + fp2 * delta_rudder_p2 + fp3 * delta_rudder_p2**2, 1.0
    )
    uR_decrease_ratio_s = np.where(
        uR_decrease_ratio_s_cal > 1.0, 1.0, uR_decrease_ratio_s_cal
    )
    uR_decrease_ratio_p = np.where(
        uR_decrease_ratio_p_cal > 1.0, 1.0, uR_decrease_ratio_p_cal
    )

    # effective rudder inflow velocity for starboard and port rudder
    vR_s = -ganma_rudder_s * vm_velo - lR_s * r_angvelo
    vR_p = -ganma_rudder_p * vm_velo - lR_p * r_angvelo

    uR_s = uR_decrease_ratio_s * uR_p_s
    uR_p = uR_decrease_ratio_p * uR_p_p

    u_rudder_s = uR_s
    u_rudder_p = uR_p

    resultant_U_rudder_s = np.sqrt(uR_s**2 + vR_s**2)
    resultant_U_rudder_p = np.sqrt(uR_p**2 + vR_p**2)

    aR_s = delta_rudder_s2 - np.arctan2(vR_s, uR_s)
    aR_p = delta_rudder_p2 - np.arctan2(vR_p, uR_p)

    UUR_s = np.sqrt(uR_s**2 + vR_s**2)
    UUR_p = np.sqrt(uR_p**2 + vR_p**2)

    # data to watch [deg.] Do not use in code!!
    aR_s_deg = aR_s * 180 / np.pi
    aR_p_deg = aR_p * 180 / np.pi
    delta_rudder_s_deg = delta_rudder_s * 180 / np.pi
    delta_rudder_p_deg = delta_rudder_p * 180 / np.pi
    delta_rudder_s1_deg = delta_rudder_s1 * 180 / np.pi
    delta_rudder_p1_deg = delta_rudder_p1 * 180 / np.pi
    delta_rudder_s_p_deg = delta_rudder_s_p * 180 / np.pi
    delta_rudder_p_s_deg = delta_rudder_p_s * 180 / np.pi
    delta_rudder_s2_deg = delta_rudder_s2 * 180 / np.pi
    delta_rudder_p2_deg = delta_rudder_p2 * 180 / np.pi

    # calculating cn from csv by spline
    VecTwin_Cns = pd.read_csv(
        "/Users/u870452j/Documents/Aoki_Code/5th-Lab/Hydro/MMG/input_csv/VecTwin_Cns.csv",
        header=0,
        index_col=0,
    )
    VecTwin_Cnp = pd.read_csv(
        "/Users/u870452j/Documents/Aoki_Code/5th-Lab/Hydro/MMG/input_csv/VecTwin_Cnp.csv",
        header=0,
        index_col=0,
    )

    Cns_curve = interp1d(
        VecTwin_Cns.loc[:, "rudderangle_s"], VecTwin_Cns.loc[:, "cns"], kind="cubic"
    )
    Cnp_curve = interp1d(
        VecTwin_Cnp.loc[:, "rudderangle_p"], VecTwin_Cnp.loc[:, "cnp"], kind="cubic"
    )

    if switch_cn_rudder == 0:
        Cns = Cns_curve(aR_s)  # CFD result spline
        Cnp = Cnp_curve(aR_p)  # CFD result spline
    elif switch_cn_rudder == 1:
        Cns = np.where(
            aR_s > 0.654498469,
            -0.8449 * aR_s**2 + 2.0621 * aR_s + 0.6387,
            np.where(
                aR_s < -0.654498469,
                0.8449 * aR_s**2 + 2.0621 * aR_s + -0.6387,
                2.6252 * np.sin(aR_s),
            ),
        )  # Kang's cns model
        Cnp = np.where(
            aR_p > 0.654498469,
            -0.8449 * aR_p**2 + 2.0621 * aR_p + 0.6387,
            np.where(
                aR_p < -0.654498469,
                0.8449 * aR_p**2 + 2.0621 * aR_p + -0.6387,
                2.6252 * np.sin(aR_p),
            ),
        )  # Kang's cnp model

    FNs = 0.5 * rho_fresh * area_rudder * Cns * UUR_s**2  # / 9.80665
    FNp = 0.5 * rho_fresh * area_rudder * Cnp * UUR_p**2  # / 9.80665

    # calculating rudder force
    XR = -(1 - t_rudder) * (FNs * np.sin(delta_rudder_s) + FNp * np.sin(delta_rudder_p))
    YR = -(1 + ah_rudder) * (
        FNs * np.cos(delta_rudder_s) + FNp * np.cos(delta_rudder_p)
    )
    NR = -(x_location_rudder + ah_rudder * xH_rudder) * (
        FNs * np.cos(delta_rudder_s) + FNp * np.cos(delta_rudder_p)
    )

    ### compute each componet of rudder force
    # if switch_rudder == 1:
    #     XR = - (1-t_rudder)                         * (FNs * np.sin(delta_rudder_s) + FNp *np.sin(delta_rudder_p))
    #     YR = - (1+ah_rudder)                        * (FNs * np.cos(delta_rudder_s) + FNp *np.cos(delta_rudder_p))
    #     NR = - (x_location_rudder + ah_rudder * xH) * (FNs * np.cos(delta_rudder_s) + FNp * np.sin(delta_rudder_p))

    # ### original rudder model (XR=YR=NR=0 at n<0)
    # elif switch_rudder == 0:
    #     XR = np.where(n_prop > 0,  - (1-t_rudder)                         * (FNs * np.sin(delta_rudder_s) + FNp *np.sin(delta_rudder_p)), np.array(0.0))
    #     YR = np.where(n_prop > 0,  - (1+ah_rudder)                        * (FNs * np.cos(delta_rudder_s) + FNp *np.cos(delta_rudder_p)), np.array(0.0))
    #     NR = np.where(n_prop > 0,  - (x_location_rudder + ah_rudder * xH) * (FNs * np.cos(delta_rudder_s) + FNp * np.sin(delta_rudder_p)), np.array(0.0))

    rudderforce = [
        XR,
        YR,
        NR,
        aR_s,
        aR_p,
        resultant_U_rudder_s,
        resultant_U_rudder_p,
        u_rudder_s,
        u_rudder_p,
        FNs,
        FNp,
        aR_s_deg,
        aR_p_deg,
        delta_rudder_s_deg,
        delta_rudder_p_deg,
        delta_rudder_s2_deg,
        delta_rudder_p2_deg,
        Cns,
        Cnp,
        delta_rudder_s1_deg,
        delta_rudder_p1_deg,
        delta_rudder_s_p_deg,
        delta_rudder_p_s_deg,
        uprop,
        uR_p_s,
        uR_p_p,
        uR_decrease_ratio_s,
        uR_decrease_ratio_p,
        slip_ratio,
    ]
    rudderforce = np.array(rudderforce)
    rudderforce = rudderforce.reshape(1, -1)

    return rudderforce
