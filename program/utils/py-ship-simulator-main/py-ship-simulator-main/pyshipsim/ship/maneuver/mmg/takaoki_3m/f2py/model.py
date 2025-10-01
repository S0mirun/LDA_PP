import os
import numpy as np
import pandas as pd

from . import f2py_mmg_takaoki_3m

__dir__ = os.path.dirname(__file__)


class MMGModel(object):
    def __init__(self):
        self.set_principal_particulars()
        self.set_mmg_model_parameter()
        self.set_switch()

    def set_principal_particulars(self):
        principal_particulars = pd.read_csv(
            f"{__dir__}/input_csv/principal_particulars_cement.csv",
            header=0,
            index_col=0,
        )
        #
        lpp = principal_particulars.at["lpp", "value"]
        B = principal_particulars.at["breadth", "value"]
        draft = principal_particulars.at["draft", "value"]
        Mass_nd = principal_particulars.at["mass_nd", "value"]
        xG_nd = principal_particulars.at["x_lcg", "value"]
        rho_fresh = principal_particulars.at["rho_fresh", "value"] / 9.80665
        # Propeller
        dia_prop = principal_particulars.at["dia_prop", "value"]
        pitchr = principal_particulars.at["pitchr", "value"]
        # Rudder
        area_rudder = principal_particulars.at["area_rudder", "value"]
        lambda_rudder = principal_particulars.at["lambda_rudder", "value"]
        x_location_rudder = principal_particulars.at["x_location_rudder", "value"]
        maxtickness_rudder = principal_particulars.at["maxtickness_rudder", "value"]
        # parameters for wind force computation
        rho_air = principal_particulars.at["rho_air", "value"] / 9.80665
        area_projected_trans = principal_particulars.at["area_projected_trans", "value"]
        area_projected_lateral = principal_particulars.at[
            "area_projected_lateral", "value"
        ]
        area_projected_lateral_superstructure = principal_particulars.at["AOD", "value"]
        lcw = principal_particulars.at["LCW", "value"]
        lcbr = principal_particulars.at["LCBR", "value"]
        hbr = principal_particulars.at["HBR", "value"]
        hc_AL = principal_particulars.at["HC", "value"]
        breadth_wind = principal_particulars.at["SBW", "value"]
        swayforce_to_cg = principal_particulars.at["Lz", "value"]
        # Side Thruster
        D_bthrust = principal_particulars.at["D_bthrust", "value"]
        D_sthrust = principal_particulars.at["D_sthrust", "value"]
        x_bowthrust_loc = principal_particulars.at["x_bowthrust_loc", "value"]
        x_stnthrust_loc = principal_particulars.at["x_stnthrust_loc", "value"]
        #
        self.pp_vector = np.array(
            [
                rho_fresh,
                rho_air,
                lpp,
                B,
                draft,
                Mass_nd,
                xG_nd,
                dia_prop,
                pitchr,
                area_rudder,
                lambda_rudder,
                x_location_rudder,
                maxtickness_rudder,
                area_projected_trans,
                area_projected_lateral,
                area_projected_lateral_superstructure,
                lcw,
                lcbr,
                hbr,
                hc_AL,
                breadth_wind,
                swayforce_to_cg,
                D_bthrust,
                D_sthrust,
                x_bowthrust_loc,
                x_stnthrust_loc,
            ]
        )

    def set_mmg_model_parameter(self):
        parameter_init = pd.read_csv(
            f"{__dir__}/input_csv/MMG_params_cement.csv", header=0, index_col=0
        )
        #
        MassX_nd = parameter_init.at["massx_nd", "params_init"]
        MassY_nd = parameter_init.at["massy_nd", "params_init"]
        IJzz_nd = parameter_init.at["IzzJzz_nd", "params_init"]
        # Hull
        Xuu_nd = parameter_init.at["xuu_nd", "params_init"]
        Xvr_nd = parameter_init.at["xvr_nd", "params_init"]
        Yv_nd = parameter_init.at["yv_nd", "params_init"]
        Yr_nd = parameter_init.at["yr_nd", "params_init"]
        Nv_nd = parameter_init.at["nv_nd", "params_init"]
        Nr_nd = parameter_init.at["nr_nd", "params_init"]
        CD = parameter_init.at["coeff_drag_sway", "params_init"]
        C_rY = parameter_init.at["cry_cross_flow", "params_init"]
        C_rN = parameter_init.at["crn_cross_flow", "params_init"]
        X_0F_nd = Xuu_nd
        X_0A_nd = parameter_init.at["coeff_drag_aft", "params_init"]
        # Propeller
        ai = np.array(
            [
                parameter_init.at["wp0_coeff0", "params_init"],
                parameter_init.at["wp0_coeff1", "params_init"],
                parameter_init.at["wp0_coeff2", "params_init"],
            ]
        )
        bi = np.array(
            [
                parameter_init.at["wp_coeff0", "params_init"],
                parameter_init.at["wp_coeff1", "params_init"],
                parameter_init.at["wp_coeff2", "params_init"],
            ]
        )
        ci = np.array(
            [
                parameter_init.at["t_prop0_coeff0", "params_init"],
                parameter_init.at["t_prop0_coeff1", "params_init"],
                parameter_init.at["t_prop0_coeff2", "params_init"],
            ]
        )
        di = np.array(
            [
                parameter_init.at["t_prop_coeff0", "params_init"],
                parameter_init.at["t_prop_coeff1", "params_init"],
                parameter_init.at["t_prop_coeff2", "params_init"],
            ]
        )
        xP_nd = parameter_init.at["xp_prop_nd", "params_init"]
        kt_coeff = np.array(
            [
                parameter_init.at["kt_coeff0", "params_init"],
                parameter_init.at["kt_coeff1", "params_init"],
                parameter_init.at["kt_coeff2", "params_init"],
            ]
        )
        kt2nd_coeff = np.array(
            [
                parameter_init.at["kt2nd_coeff0", "params_init"],
                parameter_init.at["kt2nd_coeff1", "params_init"],
                parameter_init.at["kt2nd_coeff2", "params_init"],
            ]
        )
        AAi = np.array(
            [
                parameter_init.at["ai_coeff_prop0", "params_init"],
                parameter_init.at["ai_coeff_prop1", "params_init"],
                parameter_init.at["ai_coeff_prop2", "params_init"],
                parameter_init.at["ai_coeff_prop3", "params_init"],
                parameter_init.at["ai_coeff_prop4", "params_init"],
                parameter_init.at["ai_coeff_prop5", "params_init"],
                parameter_init.at["ai_coeff_prop6", "params_init"],
                parameter_init.at["ai_coeff_prop7", "params_init"],
            ]
        )
        BBi = np.array(
            [
                parameter_init.at["bi_coeff_prop0", "params_init"],
                parameter_init.at["bi_coeff_prop1", "params_init"],
                parameter_init.at["bi_coeff_prop2", "params_init"],
                parameter_init.at["bi_coeff_prop3", "params_init"],
                parameter_init.at["bi_coeff_prop4", "params_init"],
                parameter_init.at["bi_coeff_prop5", "params_init"],
                parameter_init.at["bi_coeff_prop6", "params_init"],
                parameter_init.at["bi_coeff_prop7", "params_init"],
            ]
        )
        CCi = np.array(
            [
                parameter_init.at["ci_coeff_prop0", "params_init"],
                parameter_init.at["ci_coeff_prop1", "params_init"],
                parameter_init.at["ci_coeff_prop2", "params_init"],
                parameter_init.at["ci_coeff_prop3", "params_init"],
            ]
        )
        Jmin = parameter_init.at["Jmin", "params_init"]
        alpha_p = parameter_init.at["alpha_prop", "params_init"]
        # Rudder
        lr_rudder_nd = parameter_init.at["lr_rudder_nd", "params_init"]
        kx_rudder = parameter_init.at["kx_rudder", "params_init"]
        cpr_rudder = parameter_init.at["cpr_rudder", "params_init"]
        gammaN = parameter_init.at["gammaN_rudder", "params_init"]
        gammaP = parameter_init.at["gammaP_rudder", "params_init"]
        coeff_urudder_zero = parameter_init.at["coeff_urudder_zero", "params_init"]
        ei = np.array(
            [
                parameter_init.at["ei_coeff_vec0", "params_init"],
                parameter_init.at["ei_coeff_vec1", "params_init"],
                parameter_init.at["ei_coeff_vec2", "params_init"],
            ]
        )
        fi = np.array(
            [
                parameter_init.at["fi_coeff_vec0", "params_init"],
                parameter_init.at["fi_coeff_vec1", "params_init"],
                parameter_init.at["fi_coeff_vec2", "params_init"],
            ]
        )
        gi = np.array(
            [
                parameter_init.at["gi_coeff_vec0", "params_init"],
                parameter_init.at["gi_coeff_vec1", "params_init"],
                parameter_init.at["gi_coeff_vec2", "params_init"],
            ]
        )
        asi = np.array(
            [
                parameter_init.at["asi_coeff_vec0", "params_init"],
                parameter_init.at["asi_coeff_vec1", "params_init"],
                parameter_init.at["asi_coeff_vec2", "params_init"],
                parameter_init.at["asi_coeff_vec3", "params_init"],
            ]
        )
        api = np.array(
            [
                parameter_init.at["api_coeff_vec0", "params_init"],
                parameter_init.at["api_coeff_vec1", "params_init"],
                parameter_init.at["api_coeff_vec2", "params_init"],
                parameter_init.at["api_coeff_vec3", "params_init"],
            ]
        )
        bsi = np.array(
            [
                parameter_init.at["bsi_coeff_vec0", "params_init"],
                parameter_init.at["bsi_coeff_vec1", "params_init"],
            ]
        )
        bpi = np.array(
            [
                parameter_init.at["bpi_coeff_vec0", "params_init"],
                parameter_init.at["bpi_coeff_vec1", "params_init"],
            ]
        )
        dsi = np.array(
            [
                parameter_init.at["dsi_coeff_vec0", "params_init"],
                parameter_init.at["dsi_coeff_vec1", "params_init"],
                parameter_init.at["dsi_coeff_vec2", "params_init"],
            ]
        )
        dpi = np.array(
            [
                parameter_init.at["dpi_coeff_vec0", "params_init"],
                parameter_init.at["dpi_coeff_vec1", "params_init"],
                parameter_init.at["dpi_coeff_vec2", "params_init"],
            ]
        )
        esi = np.array(
            [
                parameter_init.at["esi_coeff_vec0", "params_init"],
                parameter_init.at["esi_coeff_vec1", "params_init"],
                parameter_init.at["esi_coeff_vec2", "params_init"],
            ]
        )
        epi = np.array(
            [
                parameter_init.at["epi_coeff_vec0", "params_init"],
                parameter_init.at["epi_coeff_vec1", "params_init"],
                parameter_init.at["epi_coeff_vec2", "params_init"],
            ]
        )
        gsi = np.array(
            [
                parameter_init.at["gsi_coeff_vec0", "params_init"],
                parameter_init.at["gsi_coeff_vec1", "params_init"],
                parameter_init.at["gsi_coeff_vec2", "params_init"],
            ]
        )
        gpi = np.array(
            [
                parameter_init.at["gpi_coeff_vec0", "params_init"],
                parameter_init.at["gpi_coeff_vec1", "params_init"],
                parameter_init.at["gpi_coeff_vec2", "params_init"],
            ]
        )
        hsi = np.array(
            [
                parameter_init.at["hsi_coeff_vec0", "params_init"],
                parameter_init.at["hsi_coeff_vec1", "params_init"],
                parameter_init.at["hsi_coeff_vec2", "params_init"],
            ]
        )
        hpi = np.array(
            [
                parameter_init.at["hpi_coeff_vec0", "params_init"],
                parameter_init.at["hpi_coeff_vec1", "params_init"],
                parameter_init.at["hpi_coeff_vec2", "params_init"],
            ]
        )
        ksi = np.array(
            [
                parameter_init.at["ksi_coeff_vec0", "params_init"],
                parameter_init.at["ksi_coeff_vec1", "params_init"],
                parameter_init.at["ksi_coeff_vec2", "params_init"],
            ]
        )
        kpi = np.array(
            [
                parameter_init.at["kpi_coeff_vec0", "params_init"],
                parameter_init.at["kpi_coeff_vec1", "params_init"],
                parameter_init.at["kpi_coeff_vec2", "params_init"],
            ]
        )
        rsi = np.array(
            [
                parameter_init.at["rsi_coeff_vec0", "params_init"],
                parameter_init.at["rsi_coeff_vec1", "params_init"],
                parameter_init.at["rsi_coeff_vec2", "params_init"],
                parameter_init.at["rsi_coeff_vec3", "params_init"],
                parameter_init.at["rsi_coeff_vec4", "params_init"],
                parameter_init.at["rsi_coeff_vec5", "params_init"],
                parameter_init.at["rsi_coeff_vec6", "params_init"],
                parameter_init.at["rsi_coeff_vec7", "params_init"],
                parameter_init.at["rsi_coeff_vec8", "params_init"],
                parameter_init.at["rsi_coeff_vec9", "params_init"],
                parameter_init.at["rsi_coeff_vec10", "params_init"],
                parameter_init.at["rsi_coeff_vec11", "params_init"],
                parameter_init.at["rsi_coeff_vec12", "params_init"],
                parameter_init.at["rsi_coeff_vec13", "params_init"],
                parameter_init.at["rsi_coeff_vec14", "params_init"],
                parameter_init.at["rsi_coeff_vec15", "params_init"],
            ]
        )
        rpi = np.array(
            [
                parameter_init.at["rpi_coeff_vec0", "params_init"],
                parameter_init.at["rpi_coeff_vec1", "params_init"],
                parameter_init.at["rpi_coeff_vec2", "params_init"],
                parameter_init.at["rpi_coeff_vec3", "params_init"],
                parameter_init.at["rpi_coeff_vec4", "params_init"],
                parameter_init.at["rpi_coeff_vec5", "params_init"],
                parameter_init.at["rpi_coeff_vec6", "params_init"],
                parameter_init.at["rpi_coeff_vec7", "params_init"],
                parameter_init.at["rpi_coeff_vec8", "params_init"],
                parameter_init.at["rpi_coeff_vec9", "params_init"],
                parameter_init.at["rpi_coeff_vec10", "params_init"],
                parameter_init.at["rpi_coeff_vec11", "params_init"],
                parameter_init.at["rpi_coeff_vec12", "params_init"],
                parameter_init.at["rpi_coeff_vec13", "params_init"],
                parameter_init.at["rpi_coeff_vec14", "params_init"],
                parameter_init.at["rpi_coeff_vec15", "params_init"],
            ]
        )
        Csi = np.array(
            [
                parameter_init.at["Csi_coeff_vec0", "params_init"],
                parameter_init.at["Csi_coeff_vec1", "params_init"],
                parameter_init.at["Csi_coeff_vec2", "params_init"],
                parameter_init.at["Csi_coeff_vec3", "params_init"],
            ]
        )
        Cpi = np.array(
            [
                parameter_init.at["Cpi_coeff_vec0", "params_init"],
                parameter_init.at["Cpi_coeff_vec1", "params_init"],
                parameter_init.at["Cpi_coeff_vec2", "params_init"],
                parameter_init.at["Cpi_coeff_vec3", "params_init"],
            ]
        )
        # Side Thruster
        KT_bow_forward = parameter_init.at["KT_bow_forward", "params_init"]
        KT_bow_reverse = parameter_init.at["KT_bow_reverse", "params_init"]
        aY_bow = np.array(
            [
                parameter_init.at["aY_bow_coeff0", "params_init"],
                parameter_init.at["aY_bow_coeff1", "params_init"],
                parameter_init.at["aY_bow_coeff2", "params_init"],
            ]
        )
        aN_bow = np.array(
            [
                parameter_init.at["aN_bow_coeff0", "params_init"],
                parameter_init.at["aN_bow_coeff1", "params_init"],
                parameter_init.at["aN_bow_coeff2", "params_init"],
            ]
        )
        BRps = parameter_init.at["BRps", "params_init"]
        KT_stern_forward = parameter_init.at["KT_stern_forward", "params_init"]
        KT_stern_reverse = parameter_init.at["KT_stern_reverse", "params_init"]
        aY_stern = np.array(
            [
                parameter_init.at["aY_stern_coeff0", "params_init"],
                parameter_init.at["aY_stern_coeff1", "params_init"],
                parameter_init.at["aY_stern_coeff2", "params_init"],
            ]
        )
        aN_stern = np.array(
            [
                parameter_init.at["aN_stern_coeff0", "params_init"],
                parameter_init.at["aN_stern_coeff1", "params_init"],
                parameter_init.at["aN_stern_coeff2", "params_init"],
            ]
        )
        SRps = parameter_init.at["SRps", "params_init"]
        Thruster_speed_max = parameter_init.at["Thruster_speed_max", "params_init"]
        #
        param1 = np.array(
            [
                MassX_nd,
                MassY_nd,
                IJzz_nd,
                Xuu_nd,
                Xvr_nd,
                Yv_nd,
                Yr_nd,
                Nv_nd,
                Nr_nd,
                CD,
                C_rY,
                C_rN,
                X_0F_nd,
                X_0A_nd,
            ]
        )
        param2 = np.concatenate(
            [
                ai,
                bi,
                ci,
                di,
            ]
        )
        param3 = np.array([xP_nd])
        param4 = np.concatenate(
            [
                kt_coeff,
                kt2nd_coeff,
                AAi,
                BBi,
                CCi,
            ],
            axis=0,
        )
        param5 = np.array(
            [
                Jmin,
                alpha_p,
                lr_rudder_nd,
                kx_rudder,
                cpr_rudder,
                gammaN,
                gammaP,
                coeff_urudder_zero,
            ]
        )
        param6 = np.concatenate(
            [
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
                Csi,
                Cpi,
            ],
            axis=0,
        )
        param7 = np.array(
            [
                KT_bow_forward,
                KT_bow_reverse,
            ]
        )
        param8 = np.concatenate([aY_bow, aN_bow], axis=0)
        param9 = np.array(
            [
                BRps,
                KT_stern_forward,
                KT_stern_reverse,
            ]
        )
        param10 = np.concatenate(
            [
                aY_stern,
                aN_stern,
            ],
            axis=0,
        )
        param11 = np.array([SRps, Thruster_speed_max])
        self.mmg_params_vector = np.concatenate(
            [
                param1,
                param2,
                param3,
                param4,
                param5,
                param6,
                param7,
                param8,
                param9,
                param10,
                param11,
            ]
        )

    def set_switch(self):
        switch = pd.read_csv(
            f"{__dir__}/input_csv/model_switch_cement.csv", header=0, index_col=0
        )
        # switch computation model
        # 1: Hachii 2: Yasukawa for thrust computation on 2,3,4th q.
        switch_prop_x = switch.at["switch_prop_x", "value"]
        # 0: YP=0 at n>0, 1: use exp.polynomial for 2nd q
        switch_prop_y = switch.at["switch_prop_y", "value"]
        # 0: NP=0 at n>0, 1: use exp.polynomial for 2nd q
        switch_prop_n = switch.at["switch_prop_n", "value"]
        # 0: YR & NR are 0 at n<0, 1: compute force at n<0
        switch_rudder = switch.at["switch_rudder", "value"]
        # model for rudder normal force, 1: Fujii, 2: lindenburg
        switch_fn_rudder = switch.at["switch_fn", "value"]
        # model for rudder normal force, 1: CFD, 2: Kang
        switch_cn_rudder = switch.at["switch_cn", "value"]
        # model for effective inflow angle of rudder, 1: standard MMG by Yoshimura, 2: reversed model by Yasukawa on u<0
        switch_ur_rudder = switch.at["switch_ur", "value"]
        # 0: off, 1: on
        switch_wind = switch.at["switch_wind", "value"]
        # 1: uniform wind, 2: irregular wind, 3: instantaneous wind measured by exp (converted to true), 4: use exp relavite wind
        switch_windtype = switch.at["switch_windtype", "value"]
        #
        self.switch = np.array(
            [
                switch_prop_x,
                switch_prop_y,
                switch_prop_n,
                switch_rudder,
                switch_fn_rudder,
                switch_cn_rudder,
                switch_ur_rudder,
                switch_wind,
                switch_windtype,
            ]
        )

    def ode_rhs(self, x, u, w):
        [delta_rudder_p, delta_rudder_s, n_prop, n_bt] = u
        [Wind_velocity, Wind_Direction] = w
        #
        dx = f2py_mmg_takaoki_3m.takaoki_vr_vector_input.mmg_lowspeed_vr_model(
            x,
            delta_rudder_p,
            delta_rudder_s,
            n_prop,
            n_bt,
            0.0,
            Wind_Direction,
            Wind_velocity,
            self.switch,
            self.pp_vector,
            self.mmg_params_vector,
        )
        return dx
