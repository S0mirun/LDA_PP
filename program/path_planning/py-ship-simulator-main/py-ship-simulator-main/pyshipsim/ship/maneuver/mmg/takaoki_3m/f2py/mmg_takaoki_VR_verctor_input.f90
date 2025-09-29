! MMG model for sigle rudder, single propeller ship "Esso Osaka"
! with artificial Bow thruster and Stern thruster
!
module Takaoki_VR_vector_input
    implicit none
    
    double precision, parameter :: pi = 4.0d0*datan(1.0d0);
    double precision, parameter :: grav = 9.80665d0;

contains

    subroutine MMG_LowSpeed_VR_model(&
        time_derivative, state, delta_rudder_p, delta_rudder_s, &
        n_prop, n_bt, n_st, Wind_Direction, Wind_velocity, &
        switch, pp_vector, mmg_params_vector)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! switch = (
        !     switch_prop_x, switch_prop_y, switch_prop_n, switch_rudder,
        !     switch_fn_rudder, switch_cn_rudder, switch_ur_rudder, switch_wind, switch_windtype,
        ! )
        ! pp_vector = (
        !     rho_fresh, rho_air,
        !     lpp, B, draft, Mass_nd, xG_nd,
        !     dia_prop, pitchr,
        !     area_rudder, lambda_rudder, x_location_rudder, maxtickness_rudder,
        !     area_projected_trans, area_projected_lateral, area_projected_lateral_superstructure,
        !     lcw, lcbr, hbr, hc_AL, breadth_wind, swayforce_to_cg,
        !     D_bthrust, D_sthrust, x_bowthrust_loc, x_stnthrust_loc
        ! )
        ! mmg_params_vector = (
        !     MassX_nd, MassY_nd, IJzz_nd,
        !     Xuu_nd, Xvr_nd, Yv_nd, Yr_nd, Nv_nd, Nr_nd,
        !     CD, C_rY, C_rN, X_0F_nd, X_0A_nd,
        !     ai, bi, ci,  di,
        !     xP_nd, kt_coeff, kt2nd_coeff,
        !     AAi, BBi, CCi,
        !     Jmin, alpha_p,
        !     lr_rudder_nd, kx_rudder,
        !     cpr_rudder, gammaN, gammaP, coeff_urudder_zero,
        !     ei, fi, gi,
        !     asi, api, bsi, bpi,
        !     dsi, dpi, esi, epi, gsi, gpi,
        !     hsi, hpi, ksi, kpi,
        !     rsi, rpi, Csi, Cpi,
        !     KT_bow_forward, KT_bow_reverse,
        !     aY_bow, aN_bow, BRps,
        !     KT_stern_forward, KT_stern_reverse,
        !     aY_stern, aN_stern,
        !     SRps, Thruster_speed_max
        ! )
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        implicit none
 
        double precision, intent(in) :: state(6), delta_rudder_p, delta_rudder_s, n_prop, n_bt, n_st, Wind_Direction, Wind_velocity
        integer, intent(in) :: switch(9)
        double precision, intent(in) :: pp_vector(26), mmg_params_vector(171)
        double precision, intent(out) :: time_derivative(6)

        !!! local variables
        integer :: i,j
        integer :: i_sw, i_pp, i_para
        integer :: i_max

        ! Default Swich Prameter
        integer :: switch_prop_x
        integer :: switch_prop_y
        integer :: switch_prop_n
        integer :: switch_rudder
        integer :: switch_fn_rudder
        integer :: switch_cn_rudder
        integer :: switch_ur_rudder
        integer :: switch_wind
        integer :: switch_windtype

        ! Grobal Variables
        ! This variable depends on the ship.
        double precision :: rho_fresh, rho_air; 
        double precision :: lpp, breadth, draft, Mass_nd, xG_nd; ! Dimension 
        double precision :: dia_prop, pitchr; ! Propeller
        double precision :: area_rudder, lambda_rudder, x_location_rudder, maxtickness_rudder; ! Propeller
        double precision :: area_projected_trans, area_projected_lateral, area_projected_lateral_superstructure; ! Wind
        double precision :: lcw, lcbr, hbr, hc_AL, breadth_wind, swayforce_to_cg; ! Wind
        double precision :: MassX_nd, MassY_nd, IJzz_nd; ! Added mass  
        double precision :: Xuu_nd, Xvr_nd, Yv_nd, Yr_nd, Nv_nd, Nr_nd; ! Hull  
        double precision :: CD, C_rY, C_rN, X_0F_nd, X_0A_nd; ! Hull
        double precision :: ai(3), bi(3), ci(3), di(3); ! Proppeller    
        double precision :: xP_nd, kt_coeff(3), kt2nd_coeff(3);
        double precision :: AAi(8), BBi(8), CCi(4);
        double precision :: Jmin, alpha_p;                                                                   
        double precision :: lr_rudder_nd, kx_rudder; 
        double precision :: cpr_rudder, gammaN, gammaP, coeff_urudder_zero;
        double precision :: ei(3), fi(3), gi(3);
        double precision :: asi(4), api(4), bsi(2), bpi(2);
        double precision :: dsi(3), dpi(3), esi(3), epi(3), gsi(3), gpi(3);
        double precision :: hsi(3), hpi(3), ksi(3), kpi(3);
        double precision :: rsi(16), rpi(16), Csi(4), Cpi(4);
        double precision :: D_bthrust, D_sthrust, x_bowthrust_loc, x_stnthrust_loc;
        double precision :: KT_bow_forward, KT_bow_reverse;
        double precision :: aY_bow(3), aN_bow(3), BRps;
        double precision :: KT_stern_forward, KT_stern_reverse;
        double precision :: aY_stern(3), aN_stern(3);
        double precision :: SRps, Thruster_speed_max;
        ! 
        double precision :: u_velo, vm_velo, r_angvelo, psi, Xpos, Ypos
        double precision :: u_nd, v_nd, r_nd;
        double precision :: Fr_number, U_ship, beta, beta_hat, beta_t;
        double precision :: vp_nd;
        double precision :: D1_ad1, D1_ad2, D2_ad1, D2_ad2, Mass, MassX, MassY, IJzz, xG;
        double precision :: X, Y, N
        double precision :: XH, YH, NH;
        double precision :: XH_nd, YH_nd, NH_nd, YHN_nd, NHN_nd;
        double precision :: Y_ad, N_ad
        double precision :: x0, x1, comp0, comp1, comp2, comp3;
        !
        double precision :: XP, YP, NP
        double precision :: pitch, t_prop, t_prop0, wP0, tau_prop, CP_nd, kt_coeff0, kt_coeff1, kt_coeff2
        double precision :: J_prop, KT, Js, Jsyn, Jsyn0, wP, uprop, one_minus_wprop, one_minus_t_prop;
        double precision :: XR, YR, NR;
        double precision :: kappa_rudder, one_minus_wrudder, Ep, fa, uR, vR, urpr1, urpr2, ursq, UUR, aR, FN;
        double precision :: u_dot, vm_dot, r_dot, psi_dot, X_dot, Y_dot, delta_dot;
        double precision :: TempA1, TempA2, TempA3, TempB1, TempB2, TempB3
        !
        double precision :: t_rudder, ah_rudder_cal, ah_rudder, xh_rudder, xh_rudder_nd_cal, xh_rudder_nd;
        double precision :: slip_ratio_cal, slip_ratio;
        double precision :: cs1, cp1, cs2, cp2;
        double precision :: delta_rudder_s0, delta_rudder_p0, delta_rudder_s1, delta_rudder_p1, delta_rudder_s2, delta_rudder_p2;
        double precision :: delta_rudder_s_p, delta_rudder_p_s;
        double precision :: one_minus_wrudder_s, one_minus_wrudder_p;
        double precision :: hight_rudder, eta_rudder, epsilon_rudder_s, epsilon_rudder_p, kappa_rudder_s, kappa_rudder_p;
        double precision :: uR_p_s, uR_p_p;
        double precision :: ganma_rudder_s_cal1, ganma_rudder_s_cal2, ganma_rudder_s_cal3, ganma_rudder_s_cal4;
        double precision :: ganma_rudder_p_cal1, ganma_rudder_p_cal2, ganma_rudder_p_cal3, ganma_rudder_p_cal4;
        double precision :: ganma_rudder_s1, ganma_rudder_s2, ganma_rudder_s3, ganma_rudder_s4;
        double precision :: ganma_rudder_p1, ganma_rudder_p2, ganma_rudder_p3, ganma_rudder_p4;
        double precision :: ganma_rudder_s, ganma_rudder_p;
        double precision :: lR_s, lR_p, fs1, fp1, fs2, fp2, fs3, fp3;
        double precision :: uR_decrease_ratio_s_cal, uR_decrease_ratio_p_cal;
        double precision :: uR_decrease_ratio_s, uR_decrease_ratio_p;
        double precision :: vR_s, vR_p, uR_s, uR_p, u_rudder_s, u_rudder_p, resultant_U_rudder_s, resultant_U_rudder_p;
        double precision :: aR_s, aR_p, UUR_s, UUR_p;
        double precision :: aR_s_deg, aR_p_deg;
        double precision :: delta_rudder_s_deg, delta_rudder_p_deg;
        double precision :: delta_rudder_s0_deg, delta_rudder_p0_deg;
        double precision :: delta_rudder_s1_deg, delta_rudder_p1_deg;
        double precision :: delta_rudder_s_p_deg, delta_rudder_p_s_deg, delta_rudder_s2_deg, delta_rudder_p2_deg;
        double precision :: Cns, Cnp, FNs, FNp;
        !
        double precision :: XBT, YBT, NBT, XST, YST, NST;
        double precision :: BowThrust, BowMoment, SternThrust, SternMoment;
    
        !   Valiables for wind forces and moments
        double precision :: XA, YA, NA;
        double precision :: Windcoef(10), AT, AL, AOD, HC, SBW, Lz;
        double precision :: Relative_Wind_Direction, Relative_Wind_Velocity, Angle_of_attack;
        double precision :: CXwind, CYwind, CNwind, CKwind, FXwind, FYwind, FNwind, FKwind;
        double precision :: KK1, KK2, KK3, KK5;
        
        ! read State Variables and control
        Xpos      = state(1);
        u_velo    = state(2);
        Ypos      = state(3);
        vm_velo   = state(4); ! v at midship (v -->> vm_velo 4/16/2018 Nishikawa)
        psi       = state(5);
        r_angvelo = state(6);

        ! 
        i_sw = 1
        switch_prop_x = switch(i_sw); i_sw = i_sw + 1;
        switch_prop_y = switch(i_sw); i_sw = i_sw + 1;
        switch_prop_n = switch(i_sw); i_sw = i_sw + 1;
        switch_rudder = switch(i_sw); i_sw = i_sw + 1;
        switch_fn_rudder = switch(i_sw); i_sw = i_sw + 1;
        switch_cn_rudder = switch(i_sw); i_sw = i_sw + 1;
        switch_ur_rudder = switch(i_sw); i_sw = i_sw + 1;
        switch_wind = switch(i_sw); i_sw = i_sw + 1;
        switch_windtype = switch(i_sw); i_sw = i_sw + 1;
        !
        i_pp = 1; i_para = 1;
        rho_fresh = pp_vector(i_pp); i_pp = i_pp + 1;
        rho_air = pp_vector(i_pp); i_pp = i_pp + 1;
        lpp = pp_vector(i_pp); i_pp = i_pp + 1;
        breadth = pp_vector(i_pp); i_pp = i_pp + 1;
        draft = pp_vector(i_pp); i_pp = i_pp + 1;
        Mass_nd = pp_vector(i_pp); i_pp = i_pp + 1;
        xG_nd = pp_vector(i_pp); i_pp = i_pp + 1;
        dia_prop = pp_vector(i_pp); i_pp = i_pp + 1;
        pitchr = pp_vector(i_pp); i_pp = i_pp + 1;
        area_rudder = pp_vector(i_pp); i_pp = i_pp + 1;
        lambda_rudder = pp_vector(i_pp); i_pp = i_pp + 1;
        x_location_rudder = pp_vector(i_pp); i_pp = i_pp + 1;
        maxtickness_rudder = pp_vector(i_pp); i_pp = i_pp + 1;
        area_projected_trans = pp_vector(i_pp); i_pp = i_pp + 1;
        area_projected_lateral = pp_vector(i_pp); i_pp = i_pp + 1;
        area_projected_lateral_superstructure = pp_vector(i_pp); i_pp = i_pp + 1;
        lcw = pp_vector(i_pp); i_pp = i_pp + 1;
        lcbr = pp_vector(i_pp); i_pp = i_pp + 1;
        hbr = pp_vector(i_pp); i_pp = i_pp + 1;
        hc_AL = pp_vector(i_pp); i_pp = i_pp + 1;
        breadth_wind = pp_vector(i_pp); i_pp = i_pp + 1;
        swayforce_to_cg = pp_vector(i_pp); i_pp = i_pp + 1;
        MassX_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        MassY_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        IJzz_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        Xuu_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        Xvr_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        Yv_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        Yr_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        Nv_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        Nr_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        CD = mmg_params_vector(i_para); i_para = i_para + 1;
        C_rY = mmg_params_vector(i_para); i_para = i_para + 1;
        C_rN = mmg_params_vector(i_para); i_para = i_para + 1;
        X_0F_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        X_0A_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        ai = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        bi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        ci = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        di = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        xP_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        kt_coeff = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        kt2nd_coeff = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        AAi = mmg_params_vector(i_para:i_para+7); i_para = i_para + 8;
        BBi = mmg_params_vector(i_para:i_para+7); i_para = i_para + 8;
        CCi = mmg_params_vector(i_para:i_para+3); i_para = i_para + 4;
        Jmin = mmg_params_vector(i_para); i_para = i_para + 1;
        alpha_p = mmg_params_vector(i_para); i_para = i_para + 1;
        lr_rudder_nd = mmg_params_vector(i_para); i_para = i_para + 1;
        kx_rudder = mmg_params_vector(i_para); i_para = i_para + 1;
        cpr_rudder = mmg_params_vector(i_para); i_para = i_para + 1;
        gammaN = mmg_params_vector(i_para); i_para = i_para + 1;
        gammaP = mmg_params_vector(i_para); i_para = i_para + 1;
        coeff_urudder_zero = mmg_params_vector(i_para); i_para = i_para + 1;
        ei = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        fi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        gi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        asi = mmg_params_vector(i_para:i_para+3); i_para = i_para + 4;
        api = mmg_params_vector(i_para:i_para+3); i_para = i_para + 4;
        bsi = mmg_params_vector(i_para:i_para+1); i_para = i_para + 2;
        bpi = mmg_params_vector(i_para:i_para+1); i_para = i_para + 2;
        dsi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        dpi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        esi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        epi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        gsi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        gpi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        hsi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        hpi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        ksi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        kpi = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        rsi = mmg_params_vector(i_para:i_para+15); i_para = i_para + 16;
        rpi = mmg_params_vector(i_para:i_para+15); i_para = i_para + 16;
        Csi = mmg_params_vector(i_para:i_para+3); i_para = i_para + 4;
        Cpi = mmg_params_vector(i_para:i_para+3); i_para = i_para + 4;
        D_bthrust = pp_vector(i_pp); i_pp = i_pp + 1;
        D_sthrust = pp_vector(i_pp); i_pp = i_pp + 1;
        x_bowthrust_loc = pp_vector(i_pp); i_pp = i_pp + 1;
        x_stnthrust_loc = pp_vector(i_pp); i_pp = i_pp + 1;
        KT_bow_forward = mmg_params_vector(i_para); i_para = i_para + 1;
        KT_bow_reverse = mmg_params_vector(i_para); i_para = i_para + 1;
        aY_bow = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        aN_bow = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        BRps = mmg_params_vector(i_para); i_para = i_para + 1;
        KT_stern_forward = mmg_params_vector(i_para); i_para = i_para + 1;
        KT_stern_reverse = mmg_params_vector(i_para); i_para = i_para + 1;
        aY_stern = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        aN_stern = mmg_params_vector(i_para:i_para+2); i_para = i_para + 3;
        SRps = mmg_params_vector(i_para); i_para = i_para + 1;
        Thruster_speed_max = mmg_params_vector(i_para); i_para = i_para + 1;


        !!!!!!!!!!!!!
        !!! main !!!!
        !!!!!!!!!!!!!

        ! Forward velocity   
        U_ship = sqrt(u_velo ** 2.0d0 + vm_velo ** 2.0d0);
        beta_hat = atan2(vm_velo, u_velo);
        beta     = atan2(-vm_velo, u_velo);
        
        ! variables to add dimenstion
        D1_ad2 = 0.5d0 * rho_fresh * lpp ** 2.0d0 * draft;
        D2_ad2 = 0.5d0 * rho_fresh * lpp ** 4.0d0 * draft;
        Mass  = Mass_nd * D1_ad2;
        MassX = MassX_nd * D1_ad2;
        MassY = MassY_nd * D1_ad2;
        IJzz  = IJzz_nd * D2_ad2;
        xG    = xG_nd * lpp;
        !
        if(abs(U_ship)< 1.0d-5) then
           v_nd = 0.0000001d0;
           r_nd = 0.0000001d0;
           U_ship = 0.0000001d0;
        else
           v_nd = vm_velo / U_ship;  ! v -->> vm_velo (definition change 2018/4/5)
           r_nd = r_angvelo * lpp / U_ship;
        end if
        
        !!!!!!!!!!!!!!!!!!!!!
        !!! Force of Hull
        !!!!!!!!!!!!!!!!!!!!!
          
        !integration parameters
        i_max = 200;
        Y_ad = 0.0d0;
        N_ad = 0.0d0;
        
        do i = 1, i_max;
           x0 = -0.5d0 + dble(i - 1) / dble(i_max);
           x1 = -0.5d0 + dble(i)  /dble(i_max);
           comp0 =vm_velo + C_rY * r_angvelo * lpp * x0;
           comp1 =vm_velo + C_rY * r_angvelo * lpp * x1;
           comp2 =vm_velo + C_rN * r_angvelo * lpp * x0;
           comp3 =vm_velo + C_rN * r_angvelo * lpp * x1;
           Y_ad = Y_ad + 0.5d0 * (abs(comp0) * comp0 + abs(comp1) * comp1) / dble(i_max);
           N_ad = N_ad + 0.5d0 * (abs(comp2) * comp2 * x0 + abs(comp3) * comp3 * x1) / dble(i_max);
        end do
        YHN_nd = - CD * Y_ad;
        NHN_nd = - CD * N_ad;
        
        XH = 0.5d0 * rho_fresh * lpp      * draft * &
                ((X_0F_nd + (X_0A_nd - X_0F_nd) * (abs(beta_hat) / pi)) * u_velo * U_ship + Xvr_nd * vm_velo * r_angvelo * lpp);
        YH = 0.5d0 * rho_fresh * lpp      * draft *(Yv_nd * vm_velo * abs(u_velo) + Yr_nd * r_angvelo * lpp * u_velo + YHN_nd);
        NH = 0.5d0 * rho_fresh * lpp ** 2 * draft * (Nv_nd * vm_velo * u_velo + Nr_nd * r_angvelo * lpp * abs(u_velo) + NHN_nd);
    
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of Propeller !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !
        !   Propeller model of Kang
        !
        U_ship    = sqrt(u_velo**2 + vm_velo**2)
        if(U_ship < 1.0e-5)then;
            v_nd   = 1.0e-7;
            r_nd   = 1.0e-7;
            U_ship = 1.0e-7;
        else
            v_nd   = vm_velo / U_ship;
            r_nd   = r_angvelo * lpp / U_ship;
            U_ship = U_ship;
        end if;
        !
        Jsyn  = - 0.35d0;
        Jsyn0 = - 0.06d0;
        !Jst   = -0.6
        !
        if(abs(n_prop) < epsilon(1.0d0)) then
           Js = 1.0d+4
        elseif(abs(u_velo) < epsilon(1.0d0)) then
           Js = epsilon(1.0d0) 
        else
           Js = u_velo / (dia_prop * n_prop)
        end if
        !                                                        
        wP0 = 1.0d0 - (ai(1) + ai(2) * Js + ai(3) * Js ** 2);       
        vp_nd = v_nd - xP_nd * r_nd             
        wP = wP0 - bi(1) *(bi(2) *vp_nd +(vp_nd + bi(3)* vp_nd * abs(vp_nd)) ** 2)
        !
        if(u_velo > 0.0d0)then;
            one_minus_wprop = 1.0d0 - wP;
        else;
            one_minus_wprop = 1.0d0; !u = up at 2nd and 4th quad.
        end if;
        !
        if(one_minus_wprop > 1.0d0) then
            one_minus_wprop = 1.0d0
        else;
            one_minus_wprop = one_minus_wprop;
        end if;
        !
        if(one_minus_wprop <= 0.0d0) then
            one_minus_wprop = epsilon(1.0d0);
        else
            one_minus_wprop = one_minus_wprop;
        end if
        !
        if(u_velo >= 0.0d0) then;
            J_prop = Js * one_minus_wprop;
        else;
            J_prop = Js;
        end if;
        !
        t_prop0 = 1.0d0 - (ci(1) + ci(2) * J_prop + ci(3) * J_prop ** 2);
        t_prop = t_prop0 - di(1) * (di(2) * vp_nd +(vp_nd + di(3) * vp_nd* abs(vp_nd)) ** 2)
        !
        if(u_velo > 0.0d0)then;
            one_minus_t_prop = 1.0d0 - t_prop;
        else
            one_minus_t_prop = 1.0d0
        end if;
        !
        if(one_minus_t_prop > 1.0d0)then;
            one_minus_t_prop = 1.0d0;
        else
            one_minus_t_prop = one_minus_t_prop;        
        end if;
        !
        if(one_minus_t_prop <= 0.0d0)then;
            one_minus_t_prop = epsilon(1.0d0);
        else
            one_minus_t_prop = one_minus_t_prop; 
        end if;
        !
        if(switch_prop_x == 1)then;
            ! Hachii's model for effective thrust 
            if(J_prop < 0.0d0)then;
                KT = kt2nd_coeff(1) + kt2nd_coeff(2) * J_prop + kt2nd_coeff(3) * J_prop ** 2;
            else
                KT = kt_coeff(1) + kt_coeff(2) * J_prop + kt_coeff(3) * J_prop ** 2;
            end if;
            !
            if(n_prop >= 0.0d0)then;
                XP = rho_fresh * dia_prop ** 4 * n_prop ** 2 * (one_minus_t_prop) * KT;
            else if(Js >= CCi(4))then;
                XP = rho_fresh * dia_prop ** 4 * n_prop ** 2 * (CCi(2) + CCi(3) * Js)
            else;
                XP = rho_fresh * dia_prop ** 4 * n_prop ** 2 * CCi(1);
            end if;
            !
        elseif(switch_prop_x == 2)then;
            ! Yasukawa's model for effective thrust
            beta_t = 0.4811d0 * pitchr**2 - 1.1116d0 * pitchr -0.1404d0;
            !
            if(n_prop > 0.0d0)then;
                if(J_prop > Jmin)then;
                    KT = kt_coeff(1) + kt_coeff(2) * J_prop + kt_coeff(3) * J_prop ** 2;
                else;
                    KT = alpha_p * (kt_coeff(2)* (J_prop-Jmin) - kt_coeff(3) * (J_prop - Jmin) ** 2) + &
                        kt_coeff(1) + kt_coeff(2) * Jmin + kt_coeff(3) * Jmin ** 2;
                end if;
            else
                if(J_prop > Jmin)then;
                    KT = beta_t * ( kt_coeff(1) + kt_coeff(2) * J_prop + kt_coeff(3) * J_prop ** 2);
                else;
                    KT = beta_t * ( alpha_p * ( kt_coeff(2)* (J_prop-Jmin) - kt_coeff(3) * (J_prop - Jmin)** 2) &
                        + kt_coeff(1) + kt_coeff(2) * Jmin + kt_coeff(3) * Jmin**2);
                end if;
            end if;
            !
            if(n_prop >= 0.0d0)then;
                XP = rho_fresh * dia_prop ** 4 * n_prop ** 2 * (one_minus_t_prop) * KT;
            else;
                XP = rho_fresh * dia_prop ** 4 * n_prop ** 2 * KT
            end if;
        end if;
        
        if(n_prop < 0)then
            write(6, *)'Do not reverse propeller !!';   
        end if;
        ! Yp and Np               
        ! Ueno's experiment data of training vessel(Seiunmaru) for 2nd quadrant (u <0, n>0)
        ! Hachii's model for 3rd(u>0, n<0) & 4th quadrant (u<0, n<0)
        pitch  = dia_prop * pitchr
        if(switch_prop_y == 1)then;
            if(n_prop >= 0.0d0)then;
                if(u_velo >=0)then;
                    YP = 0.0d0;
                else;
                    YP = 0.5d0 * rho_fresh * lpp * draft * (n_prop * pitch)**2 * (AAi(6) * Js ** 2 + AAi(7) * Js + AAi(8));
                end if;
            else;
                if(Js < Jsyn)then;
                    YP = 0.5d0 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * (AAi(3) + AAi(4) * Js);
                else;
                    if(Jsyn0 < Js)then;
                        YP = 0.5d0 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * AAi(5);
                    else
                        YP = 0.5d0 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * (AAi(1) + AAi(2) * Js);
                    end if;
                end if;
            end if;
        else if(switch_prop_y == 0)then;
            if(n_prop >= 0.0d0)then;
                YP = 0.0d0;
            else
                if(Js < Jsyn)then;
                    YP = 0.5 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * (AAi(3) + AAi(4) * Js);
                else
                    if(Jsyn0 < Js)then;
                        YP = 0.5 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * AAi(5);
                    else
                        YP = 0.5 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * (AAi(1) + AAi(2) * Js);
                    end if
                endif;
            end if;
        end if;
        !
        !
        !
        if(switch_prop_n == 1)then;
            if(n_prop >= 0.0d0)then;
                if(u_velo >= 0)then;
                    NP = 0.0d0;
                else
                    NP = 0.5 * rho_fresh * lpp**2 * draft * (n_prop * pitch)**2 * (BBi(6) * Js**2 + BBi(7) * Js + BBi(8));
                end if;
            else
                if(Js < Jsyn)then;
                    NP = 0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (BBi(3) + BBi(4) * Js);
                else
                    if(Jsyn0 < Js)then;
                        NP = 0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * BBi(5);
                    else
                        NP = 0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (BBi(1) + BBi(2) * Js)
                    end if;
                end if;
            end if
            !
        elseif(switch_prop_n == 0)then;
            if(n_prop >= 0.0d0)then;
                NP = 0.0d0;
            else
                if(Js < Jsyn)then;
                    NP = 0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (BBi(3) + BBi(4) * Js);
                else
                    if(Jsyn0 < Js)then;
                        NP = 0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * BBi(5);
                    else
                        NP = 0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (BBi(1) + BBi(2) * Js)
                    end if;
                end if;
            end if;
        end if;
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of Rudder    !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ! interaction between ship and rudder
        t_rudder              = ei(1) + ei(2) * J_prop + ei(3) * J_prop ** 2
        ah_rudder_cal         = fi(1) + fi(2) * J_prop + fi(3) * J_prop ** 2
        xh_rudder_nd_cal      = gi(1) + gi(2) * J_prop + gi(3) * J_prop ** 2
        !
        if(ah_rudder_cal > 0.60556d0)then;
            ah_rudder = 0.60556d0;
        else
            if(ah_rudder_cal < 0.016d0)then;
                ah_rudder = 0.016d0
            else
                ah_rudder = ah_rudder_cal;
            end if;
        end if
        !
        if(xh_rudder_nd_cal > -0.159d0)then;
            xh_rudder_nd = -0.159d0;
        else
            if(ah_rudder_cal < -0.4344d0)then;
                xh_rudder_nd = -0.4344d0;
            else;
                xh_rudder_nd = ah_rudder_cal;
            end if;
        end if;
        !                    
        xh_rudder         = xh_rudder_nd * lpp;
        uprop             = u_velo * one_minus_wprop;
        pitch             = dia_prop * pitchr;
        !
        ! hydrodynamic neutral angle of starboard and port rudder
        if(abs(n_prop) < epsilon(1.0d0))then;
            slip_ratio_cal = 1.2d0;
        else
            slip_ratio_cal = 1.0d0 - uprop / (n_prop * pitch);
        end if;
        !
        if(slip_ratio_cal > 1.2d0)then;
            slip_ratio = 1.2d0;
        else;
            if(slip_ratio_cal < 0.0d0)then
                slip_ratio = 0.0d0;
            else
                slip_ratio = slip_ratio_cal;
            end if;
        end if;
        !
        delta_rudder_s0   = bsi(1) + bsi(2) * slip_ratio;
        delta_rudder_p0   = bpi(1) + bpi(2) * slip_ratio;
        ! effective rudder angle 1 considering delta_rudder0
        delta_rudder_s1   = delta_rudder_s - delta_rudder_s0;
        delta_rudder_p1   = delta_rudder_p - delta_rudder_p0;
        ! variation of inflow rudder angle due to interaction between starboard and port rudder
        cs1 = dsi(1) + dsi(2) * slip_ratio + dsi(3) * slip_ratio ** 2;
        cp1 = dpi(1) + dpi(2) * slip_ratio + dpi(3) * slip_ratio ** 2;
        cs2 = esi(1) + esi(2) * slip_ratio + esi(3) * slip_ratio ** 2;
        cp2 = epi(1) + epi(2) * slip_ratio + epi(3) * slip_ratio ** 2;
        !
        if(delta_rudder_s1 > 0.0d0)then;
            delta_rudder_s_p  = (cs1 * delta_rudder_p + cs2 * delta_rudder_p ** 2) * abs(cos(delta_rudder_s1));
        else;
            delta_rudder_s_p  = cs1 * delta_rudder_p + cs2 * delta_rudder_p**2;
        end if;
        !
        if(delta_rudder_p1 < 0.0d0)then;
            delta_rudder_p_s  = (cp1 * delta_rudder_s + cp2 * delta_rudder_s ** 2) * abs(cos(delta_rudder_p1));
        else;
            delta_rudder_p_s  = cp1 * delta_rudder_s + cp2 * delta_rudder_s ** 2;
        end if;
        ! effective rudder angle 2 considering delta_rudder1 and delta_rudder_(s_p)(p_s)
        delta_rudder_s2   = delta_rudder_s1 - delta_rudder_s_p;
        delta_rudder_p2   = delta_rudder_p1 - delta_rudder_p_s;
        ! effective wake fraction for starboard and port rudder
        one_minus_wrudder_s   = asi(1) + asi(2) * delta_rudder_s + asi(3) * delta_rudder_s ** 2 + asi(4) * delta_rudder_s ** 3;
        one_minus_wrudder_p   = api(1) + api(2) * delta_rudder_p + api(3) * delta_rudder_p ** 2 + api(4) * delta_rudder_p ** 3;
        ! rudder inflow velocity due to propeller for starboard and port rudder
        hight_rudder = sqrt(area_rudder * lambda_rudder);
        eta_rudder = dia_prop / hight_rudder;
        !
        epsilon_rudder_s = one_minus_wrudder_s / one_minus_wprop;
        epsilon_rudder_p = one_minus_wrudder_p / one_minus_wprop;
        !
        kappa_rudder_s = kx_rudder / epsilon_rudder_s;
        kappa_rudder_p = kx_rudder / epsilon_rudder_p;
        !
        if(n_prop > epsilon(1.0d0))then;
            KT = XP / (rho_fresh * dia_prop**4 * n_prop**2 * (one_minus_t_prop));
        else;
            if(abs(n_prop) > epsilon(1.0d0))then;
                KT = XP / (rho_fresh * dia_prop**4 * n_prop**2);
            else
                KT = 0.0d0;
            end if;
        end if;
        ! urpr1_s = u_velo * one_minus_wrudder_s  + n_prop * dia_prop * kx_rudder_reverse * np.sqrt(8 * np.abs(KT)/ np.pi)
        ! urpr1_p = u_velo * one_minus_wrudder_p  + n_prop * dia_prop * kx_rudder_reverse * np.sqrt(8 * np.abs(KT)/ np.pi)
        ! urpr2_s = u_velo * one_minus_wrudder_s
        ! urpr2_p = u_velo * one_minus_wrudder_p
        ! ursq_s  = eta_rudder * np.sign(urpr1_s) * urpr1_s**2 + (1- eta_rudder) * urpr2_s**2 + cpr_rudder * u_velo**2   
        ! ursq_p  = eta_rudder * np.sign(urpr1_p) * urpr1_p**2 + (1- eta_rudder) * urpr2_p**2 + cpr_rudder * u_velo**2              
        !
        if(switch_ur_rudder == 1)then;
            !
            if(n_prop >= 0.0d0 .and. KT > 0.0d0)then;
                uR_p_s = epsilon_rudder_s * sqrt(eta_rudder * (uprop+ kappa_rudder_s * (sqrt(uprop ** 2 &
                    + 8.0d0 * KT * n_prop**2 * dia_prop**2 / pi) - uprop))**2 + (1.0d0 - eta_rudder) * uprop ** 2); ! <= normarl mmg model for low speed (Yoshimura's)
            else;
                uR_p_s = 0.0d0;
            end if;
            !
            if(n_prop >= 0.0d0 .and. KT > 0.0d0)then;
                uR_p_p = epsilon_rudder_p * sqrt(eta_rudder * (uprop+ kappa_rudder_p * (sqrt(uprop ** 2 &
                    + 8.0d0 * KT * n_prop**2 * dia_prop**2 / pi) - uprop))**2 + (1.0d0 - eta_rudder) * uprop ** 2); ! <= normarl mmg model for low speed (Yoshimura's)
            else;
                uR_p_p = 0.0d0;
            end if;           
        ! elif switch_ur_rudder ==2:
        !     uR_star_s =  uprop * epsilon_rudder_s * (eta_rudder *  kappa_rudder_s * 
        !                                         (np.sign(u_velo)*np.sqrt(1 + 8.0 * KT / (np.pi * J_prop**2)) - 1.0) + 1.0) 
        !     uR_star_p =  uprop * epsilon_rudder_p * (eta_rudder *  kappa_rudder_p * 
        !                                         (np.sign(u_velo)*np.sqrt(1 + 8.0 * KT / (np.pi * J_prop**2)) - 1.0) + 1.0)                                    
        !     uR_twostar_s = 0.7 * np.pi * n_prop * dia_prop * coeff_urudder_zero
        !     uR_twostar_p = 0.7 * np.pi * n_prop * dia_prop * coeff_urudder_zero
        !     uR_p_s = np.where((n_prop >= 0)&(KT > 0), 
        !             np.where( u_velo>=0,
        !                      epsilon_rudder_s * np.sqrt(eta_rudder * (uprop+ kappa_rudder_s * 
        !                     (np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2/ (np.pi )) - uprop))**2 + (1- eta_rudder) * uprop**2), # <= 1st q, normarl mmg model for low speed (Yoshimura's)
        !                     np.where((uR_twostar_s-uR_star_s) * np.sign(u_velo) < 0.0,  # <= 2nd q, Yasukawa's model for backward
        !                             uR_star_s,
        !                             uR_twostar_s)),
        !             np.where( u_velo >= 0,
        !                     np.sign(ursq_s) * np.sqrt(np.abs(ursq_s)),  # <= kitagawa's model for n<0 (3rd q)
        !                     u_velo # <= uR=u at 4th quadrant
        !                 )
        !         )### note:J_prop=Js at backward !!!!)
        !     uR_p_p = np.where((n_prop >= 0)&(KT > 0), 
        !             np.where( u_velo>=0,
        !                      epsilon_rudder_p * np.sqrt(eta_rudder * (uprop+ kappa_rudder_p * 
        !                     (np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2/ (np.pi )) - uprop))**2 + (1- eta_rudder) * uprop**2), # <= 1st q, normarl mmg model for low speed (Yoshimura's)
        !                     np.where((uR_twostar_p-uR_star_p) * np.sign(u_velo) < 0.0,  # <= 2nd q, Yasukawa's model for backward
        !                             uR_star_p,
        !                             uR_twostar_p)),
        !             np.where( u_velo >= 0,
        !                     np.sign(ursq_p) * np.sqrt(np.abs(ursq_p)),  # <= kitagawa's model for n<0 (3rd q)
        !                     u_velo # <= uR=u at 4th quadrant
        !                 )
        !         )### note:J_prop=Js at backward !!!!)    
        else;
            write(6, *)'switch_ur error';
            return;
        end if;
        ! flow straightening coefficient of sway velocity for starboard and port rudder
        ganma_rudder_s_cal1 = (rsi(1)  + rsi(2)  * abs(beta) + (rsi(3)  + rsi(4)  * abs(beta)) *delta_rudder_s2 ) * abs(beta);
        ganma_rudder_s_cal2 = (rsi(5)  + rsi(6)  * abs(beta) + (rsi(7)  + rsi(8)  * abs(beta)) *delta_rudder_s2 ) * abs(beta);
        ganma_rudder_s_cal3 = (rsi(9)  + rsi(10) * abs(beta) + (rsi(11) + rsi(12) * abs(beta)) *delta_rudder_s2 ) * abs(beta);
        ganma_rudder_s_cal4 = (rsi(13) + rsi(14) * abs(beta) + (rsi(15) + rsi(16) * abs(beta)) *delta_rudder_s2 ) * abs(beta);
        ganma_rudder_p_cal1 = (rpi(1)  + rpi(2)  * abs(beta) + (rpi(3)  + rpi(4)  * abs(beta)) *delta_rudder_p2 ) * abs(beta);
        ganma_rudder_p_cal2 = (rpi(5)  + rpi(6)  * abs(beta) + (rpi(7)  + rpi(8)  * abs(beta)) *delta_rudder_p2 ) * abs(beta);
        ganma_rudder_p_cal3 = (rpi(9)  + rpi(10) * abs(beta) + (rpi(11) + rpi(12) * abs(beta)) *delta_rudder_p2 ) * abs(beta);
        ganma_rudder_p_cal4 = (rpi(13) + rpi(14) * abs(beta) + (rpi(15) + rpi(16) * abs(beta)) *delta_rudder_p2 ) * abs(beta);
        !
        if(ganma_rudder_s_cal1 < 0.0d0)then;
            ganma_rudder_s1 = 0.0d0;
        else;
            if(ganma_rudder_s_cal1 > 0.8d0)then;
                ganma_rudder_s1 = 0.8d0;
            else
                ganma_rudder_s1 = ganma_rudder_s_cal1;
            end if;
        end if
        !
        if(ganma_rudder_s_cal2 < 0.0d0)then;
            ganma_rudder_s2 = 0.0d0;
        else;
            if(ganma_rudder_s_cal2 > 0.8d0)then;
                ganma_rudder_s2 = 0.8d0;
            else
                ganma_rudder_s2 = ganma_rudder_s_cal2;
            end if;
        end if
        !
        if(ganma_rudder_s_cal3 < 0.0d0)then;
            ganma_rudder_s3 = 0.0d0;
        else;
            if(ganma_rudder_s_cal3 > 0.8d0)then;
                ganma_rudder_s3 = 0.8d0;
            else
                ganma_rudder_s3 = ganma_rudder_s_cal3;
            end if;
        end if
        !
        if(ganma_rudder_s_cal4 < 0.0d0)then;
            ganma_rudder_s4 = 0.0d0
        else
            if(ganma_rudder_s_cal4 > 1.0d0)then;
                ganma_rudder_s4 = 1.0d0;
            else
                ganma_rudder_s4 = ganma_rudder_s_cal4;
            end if;
        end if;
        !
        if(ganma_rudder_p_cal1 < 0.0)then;
            ganma_rudder_p1 = 0.0d0;
        else;
            if(ganma_rudder_p_cal1 > 1.0)then;
                ganma_rudder_p1 = 1.0d0;
            else;
                ganma_rudder_p1 = ganma_rudder_p_cal1;
            end if;
        end if;
        !
        if(ganma_rudder_p_cal2 < 0.0d0)then;
            ganma_rudder_p2 = 0.0d0;
        else;
            if(ganma_rudder_p_cal2 > 0.8d0)then;
                ganma_rudder_p2 = 0.8d0;
            else
                ganma_rudder_p2 = ganma_rudder_p_cal2;
            end if;
        end if;
        !
        if(ganma_rudder_p_cal3 < 0.0)then;
            ganma_rudder_p3 = 0.0d0;
        else;
            if(ganma_rudder_p_cal3 > 0.8)then;
                ganma_rudder_p3 = 0.8d0;
            else
                ganma_rudder_p3 = ganma_rudder_p_cal3;
            end if;
        end if;
        if(ganma_rudder_p_cal4 < 0.0)then;
            ganma_rudder_p4 = 0.0d0;
        else;
            if(ganma_rudder_p_cal4 > 0.8)then;
                ganma_rudder_p4 = 0.8d0;
            else;
                ganma_rudder_p4 = ganma_rudder_p_cal4;
            end if;
        end if;
        !
        if(beta >= 0)then;
            if(delta_rudder_s2 < 0)then;
                ganma_rudder_s = ganma_rudder_s1;
            else;
                ganma_rudder_s = ganma_rudder_s3;
            end if;
        else;
            if(delta_rudder_s2 < 0)then;
                ganma_rudder_s = ganma_rudder_s2;
            else;
                ganma_rudder_s = ganma_rudder_s4;
            end if;
        end if;
        !
        if(beta >= 0)then;
            if(delta_rudder_p2 <= 0)then;
                ganma_rudder_p = ganma_rudder_p1;
            else;
                ganma_rudder_p = ganma_rudder_p3;
            end if;
        else
            if(delta_rudder_p2 <= 0)then;
                ganma_rudder_p = ganma_rudder_p2;
            else;
                ganma_rudder_p = ganma_rudder_p4;
            end if;
        end if;
        ! flow straightening coefficient of yaw rate for starboard and port rudder
        if(r_angvelo >= 0.0d0)then;
            if(delta_rudder_s2 >= 0.0d0)then;
                lR_s = Csi(1) * x_location_rudder;
            else;
                lR_s = Csi(2) * x_location_rudder;
            end if;
        else;
            if(delta_rudder_s2 >= 0.0d0)then;
                lR_s = Csi(3) * x_location_rudder
            else;
                lR_s = Csi(4) * x_location_rudder;
            end if;
        end if;
        !
        if(r_angvelo >= 0.0d0)then;
            if(delta_rudder_p2 >= 0.0d0)then;
                lR_p = Cpi(1) * x_location_rudder;
            else;
                lR_p = Cpi(2) * x_location_rudder;
            end if;
        else;
            if(delta_rudder_p2 >= 0.0d0)then;
                lR_p = Cpi(3) * x_location_rudder;
            else;
                lR_p = Cpi(4) * x_location_rudder;
            end if;
        end if;
        ! decrement ratio of inflow velocity for starboard and port rudder 
        fs1 = gsi(1) + gsi(2) * slip_ratio + gsi(3) * slip_ratio ** 2;
        fp1 = gpi(1) + gpi(2) * slip_ratio + gpi(3) * slip_ratio ** 2;
        fs2 = hsi(1) + hsi(2) * slip_ratio + hsi(3) * slip_ratio ** 2;
        fp2 = hpi(1) + hpi(2) * slip_ratio + hpi(3) * slip_ratio ** 2;
        fs3 = ksi(1) + ksi(2) * slip_ratio + ksi(3) * slip_ratio ** 2;
        fp3 = kpi(1) + kpi(2) * slip_ratio + kpi(3) * slip_ratio ** 2;
        !
        if(delta_rudder_s2 > 0)then;
            uR_decrease_ratio_s_cal = fs1 + fs2 * delta_rudder_s2 + fs3 * delta_rudder_s2 ** 2;
        else; 
            uR_decrease_ratio_s_cal = 1.0d0
        end if;
        !
        if(delta_rudder_p2 < 0)then;
            uR_decrease_ratio_p_cal = fp1 + fp2 * delta_rudder_p2 + fp3 * delta_rudder_p2 ** 2;
        else;
            uR_decrease_ratio_p_cal = 1.0d0;
        end if;
        !
        if(uR_decrease_ratio_s_cal > 1.0d0)then;
            uR_decrease_ratio_s = 1.0d0;
        else
            uR_decrease_ratio_s = uR_decrease_ratio_s_cal;
        end if;
        !
        if(uR_decrease_ratio_p_cal > 1.0d0)then;
            uR_decrease_ratio_p = 1.0d0;
        else;
            uR_decrease_ratio_p = uR_decrease_ratio_p_cal;
        end if;
        ! effective rudder inflow velocity for starboard and port rudder
        vR_s = -ganma_rudder_s * vm_velo - lR_s *r_angvelo;
        vR_p = -ganma_rudder_p * vm_velo - lR_p *r_angvelo;
        !
        uR_s = uR_decrease_ratio_s * uR_p_s;
        uR_p = uR_decrease_ratio_p * uR_p_p;
        !
        u_rudder_s = uR_s;
        u_rudder_p = uR_p;
        !
        resultant_U_rudder_s = sqrt(uR_s**2 + vR_s**2);
        resultant_U_rudder_p = sqrt(uR_p**2 + vR_p**2);
        !
        aR_s = delta_rudder_s2 - atan2(vR_s, uR_s);
        aR_p = delta_rudder_p2 - atan2(vR_p, uR_p);
        !
        UUR_s = sqrt(uR_s**2 + vR_s**2);
        UUR_p = sqrt(uR_p**2 + vR_p**2);
        ! data to watch [deg.] Do not use in code!! 
        aR_s_deg               = aR_s             * 180.0d0/ pi;
        aR_p_deg               = aR_p             * 180.0d0/ pi;
        delta_rudder_s_deg     = delta_rudder_s   * 180.0d0/ pi;
        delta_rudder_p_deg     = delta_rudder_p   * 180.0d0/ pi;
        delta_rudder_s0_deg    = delta_rudder_s0  * 180.0d0/ pi;
        delta_rudder_p0_deg    = delta_rudder_p0  * 180.0d0/ pi;
        delta_rudder_s1_deg    = delta_rudder_s1  * 180.0d0/ pi;
        delta_rudder_p1_deg    = delta_rudder_p1  * 180.0d0/ pi;
        delta_rudder_s_p_deg   = delta_rudder_s_p * 180.0d0/ pi;
        delta_rudder_p_s_deg   = delta_rudder_p_s * 180.0d0/ pi;
        delta_rudder_s2_deg    = delta_rudder_s2  * 180.0d0/ pi;
        delta_rudder_p2_deg    = delta_rudder_p2  * 180.0d0/ pi;
        ! calculating cn from csv by spline
        !VecTwin_Cns   = pd.read_csv('input_csv/VecTwin_Cns.csv', header=0,index_col=0)
        !VecTwin_Cnp   = pd.read_csv('input_csv/VecTwin_Cnp.csv', header=0,index_col=0)

        !Cns_curve = interp1d(VecTwin_Cns.loc[:,'rudderangle_s'], VecTwin_Cns.loc[:,'cns'], kind="cubic")
        !Cnp_curve = interp1d(VecTwin_Cnp.loc[:,'rudderangle_p'], VecTwin_Cnp.loc[:,'cnp'], kind="cubic")
    
        if(switch_cn_rudder == 0)then;
        !    Cns = Cns_curve(aR_s) ! CFD result spline
        !    Cnp = Cnp_curve(aR_p) ! CFD result spline
        elseif(switch_cn_rudder == 1)then;
            ! Kang's cns model
            if(aR_s > 0.654498469d0)then;
                Cns = -0.8449d0 * aR_s ** 2 + 2.0621d0 * aR_s + 0.6387d0;
            else;
                if(aR_s < -0.654498469d0)then;
                    Cns = 0.8449d0 * aR_s ** 2 + 2.0621d0 * aR_s - 0.6387d0;
                else;
                    Cns = 2.6252d0 * sin(aR_s);
                end if;
            end if;
            ! Kang's cnp model
            if(aR_p > 0.654498469d0)then;
                Cnp = -0.8449d0 * aR_p ** 2 + 2.0621d0 * aR_p + 0.6387d0;
            else;
                if(aR_p < -0.654498469d0)then;
                    Cnp = 0.8449d0 * aR_p ** 2+ 2.0621 * aR_p - 0.6387d0;
                else
                    Cnp = 2.6252d0 * sin(aR_p);
                end if;
            end if;
        end if; 
        FNs = 0.5 * rho_fresh * area_rudder * Cns * UUR_s ** 2; 
        FNp = 0.5 * rho_fresh * area_rudder * Cnp * UUR_p ** 2 ;
        !
        ! calculating rudder force
        XR = - (1.0d0 - t_rudder)                   * (FNs * sin(delta_rudder_s) + FNp * sin(delta_rudder_p))
        YR = - (1.0d0 + ah_rudder)                  * (FNs * cos(delta_rudder_s) + FNp * cos(delta_rudder_p))
        NR = - (x_location_rudder + ah_rudder * xh_rudder) * (FNs * cos(delta_rudder_s) + FNp * cos(delta_rudder_p))
        !
        !## compute each componet of rudder force
        ! if switch_rudder == 1:
        !     XR = - (1-t_rudder)                         * (FNs * np.sin(delta_rudder_s) + FNp *np.sin(delta_rudder_p))
        !     YR = - (1+ah_rudder)                        * (FNs * np.cos(delta_rudder_s) + FNp *np.cos(delta_rudder_p))
        !     NR = - (x_location_rudder + ah_rudder * xh_rudder) * (FNs * np.cos(delta_rudder_s) + FNp * np.sin(delta_rudder_p))
        !
        ! ### original rudder model (XR=YR=NR=0 at n<0) 
        ! elif switch_rudder == 0:
        !     XR = np.where(n_prop > 0,  - (1-t_rudder)                         * (FNs * np.sin(delta_rudder_s) + FNp *np.sin(delta_rudder_p)), np.array(0.0))
        !     YR = np.where(n_prop > 0,  - (1+ah_rudder)                        * (FNs * np.cos(delta_rudder_s) + FNp *np.cos(delta_rudder_p)), np.array(0.0))
        !     NR = np.where(n_prop > 0,  - (x_location_rudder + ah_rudder * xh_rudder) * (FNs * np.cos(delta_rudder_s) + FNp * np.sin(delta_rudder_p)), np.array(0.0))
        !
        !rudderforce = [ XR, YR, NR, aR_s,aR_p, resultant_U_rudder_s,resultant_U_rudder_p,u_rudder_s, u_rudder_p, FNs, FNp,
        !                aR_s_deg, aR_p_deg, delta_rudder_s_deg, delta_rudder_p_deg, delta_rudder_s2_deg, delta_rudder_p2_deg,
        !                Cns, Cnp, delta_rudder_s1_deg,delta_rudder_p1_deg, delta_rudder_s_p_deg, delta_rudder_p_s_deg, uprop, 
        !                uR_p_s, uR_p_p, uR_decrease_ratio_s, uR_decrease_ratio_p, slip_ratio]
        !rudderforce = np.array( rudderforce )
        !rudderforce = rudderforce.reshape(1,-1)
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of Bow and stern thruster    !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !
        Fr_number = abs(u_velo/sqrt(grav*lpp))
        ! Bow
        if(n_bt >= 0.0d0)then;
            BowThrust = rho_fresh * D_bthrust ** 4 * n_bt ** 2 * KT_bow_forward
            BowMoment = BowThrust * x_bowthrust_loc
        elseif(n_bt < 0.0d0)then;
            BowThrust = rho_fresh * D_bthrust ** 4 * n_bt ** 2 * KT_bow_reverse
            BowMoment = BowThrust * x_bowthrust_loc
        end if;
        ! Stern
        if(n_st >= 0.0d0)then;
            SternThrust = rho_fresh * D_sthrust ** 4 * n_st ** 2 * KT_stern_forward
            SternMoment = SternThrust * x_stnthrust_loc
        elseif(n_st < 0.0d0)then;
            SternThrust = rho_fresh * D_sthrust ** 4 * n_st ** 2 * KT_stern_reverse
            SternMoment = SternThrust * x_stnthrust_loc
        end if;
        XBT = 0.0d0;
        YBT = (1.0d0 + aY_bow(1)+ aY_bow(2) * Fr_number +aY_bow(3) * Fr_number**2) * BowThrust;
        NBT = (1.0d0 + aN_bow(1)+ aN_bow(2) * Fr_number +aN_bow(3) * Fr_number**2) * BowMoment;
        XST = 0.0d0;
        YST = (1.0d0 + aY_stern(1)+ aY_stern(2) * Fr_number +aY_stern(3) * Fr_number ** 2) * SternThrust;
        NST = (1.0d0 + aN_stern(1)+ aN_stern(2) * Fr_number +aN_stern(3) * Fr_number ** 2) * SternMoment;
        !
        ! limitation of u_velocity
        if(abs(u_velo) >= Thruster_speed_max)then;
            XBT = 0.0d0;
            YBT = 0.0d0;
            NBT = 0.0d0;
            XST = 0.0d0;
            YST = 0.0d0;
            NST = 0.0d0;
        end if;
        

        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of wind      !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        AT = area_projected_trans; ! AT
        AL = area_projected_lateral; ! AL
        AOD = area_projected_lateral_superstructure; ! AOD 
        LCW = lcw; ! C or LCW, midship to center of AL
        LCBR = lcbr; ! CBR, midship to center of AOD(superstructure or bridge)
        HBR = hbr; ! Height from free surface to top of the superstructure (bridge) (m) 
        HC = hc_AL; ! Hc, hight of center of lateral projected area
        SBW = breadth_wind; ! ship breadth fro wind force computation 
        Lz = swayforce_to_cg; ! Lz --- Acting position of sway force from center of gravity (it is necessary to calculate roll moment)
        !   Save as Windcoef
        Windcoef(1) = AT;   Windcoef(2) = AL;  Windcoef(3) = AOD; Windcoef(4) = LCW;
        Windcoef(5) = LCBR; Windcoef(6) = HBR; Windcoef(7) = HC;  Windcoef(8) = SBW;
        !
        
        call Relative_Speed_and_Direction_Wind_or_Current(Wind_Direction, Wind_Velocity, psi, U_ship, beta_hat,&
        &                                                 Relative_Wind_Direction, Relative_Wind_Velocity)
        !   
        Angle_of_attack = 2.0d0 * pi - Relative_Wind_Direction
        
        !  Calculation of wind force and moment by Fujiwara's regresison
        call WindForce(Relative_Wind_Velocity, Angle_of_attack, &
        &              CXwind, CYwind, CNwind, CKwind,&
        &              FXwind, FYwind, FNwind, FKwind,&
        &              lpp, breadth, Lz, Windcoef)

        XA = FXwind 
        YA = FYwind
        NA = FNwind
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Sum of all forces and moments
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        X = XH + XP + XR + XA
        Y = YH + YP + YR + YA + YBT + YST
        N = NH + NP + NR + NA + NBT + NST
    
        TempA1 = Mass + MassY
        TempA2 = xG * Mass
        TempA3 = Y - (Mass + MassX) * u_velo * r_angvelo
        TempB1 = IJzz + xG ** 2 * Mass
        TempB2 = xG * Mass
        TempB3 = N - xG * Mass * u_velo * r_angvelo
        u_dot  = (X + (Mass + MassY) * vm_velo * r_angvelo + xG * Mass * r_angvelo **2.0d0) / (Mass + MassX)
        vm_dot = (TempA3 * TempB1 - TempA2 * TempB3) / (TempA1 * TempB1 - TempA2 * TempB2)
        r_dot  = (TempA3 * TempA2 - TempB3 * TempA1) / (TempA2 * TempB2 - TempA1 * TempB1)
        !
        X_dot     = u_velo * cos(psi) - vm_velo * sin(psi)
        Y_dot     = u_velo * sin(psi) + vm_velo * cos(psi)
        psi_dot   = r_angvelo
        !
        time_derivative(1) = X_dot
        time_derivative(2) = u_dot
        time_derivative(3) = Y_dot
        time_derivative(4) = vm_dot
        time_derivative(5) = psi_dot
        time_derivative(6) = r_dot
        !
    end subroutine MMG_LowSpeed_VR_model
      
    subroutine Relative_Speed_and_Direction_Wind_or_Current(&
        zeta, UT, psi, U_ship,beta_hat, &
        Relative_Direction, Relative_Velocity)
        !
        !   Calculation of relative direction and speed
        !   zeta (Input:rad) : Direction of TRUE Wind or Current, earth fix coordinate(!! rad !!)
        !   UT (Input :m/s) : Speed of Wind or Current
        !   psi  (Input:deg) : Direction of Ship
        !   U_ship   (Input:m/s) : Speed of Ship
        !  beta_hat(input: rad): drift angle of ship (!! rad !!) (-pi, pi)
        !   Relative_Direction (Output: rad): Relative direction of Wind or Current (!! rad !!) 
        !   Relative_Velocity  (Output: m/s): Relative speed of Wind or Current
        !
        implicit none
        double precision, intent(in) :: zeta, UT, psi, U_ship,beta_hat
        double precision, intent(out) :: Relative_Direction,Relative_Velocity
        double precision :: chi,chi_beta, gamma_beta, RV
        !
        !  compute chi_beta(angle between ship traveling direction and true wind)
        chi = mod(3.0d0*pi - zeta + psi, 2.0d0*pi)
        chi_beta = chi+beta_hat
        if(chi_beta <0.0d0)then
            chi_beta = chi_beta + 2.0d0*pi 
        elseif(chi_beta >2.0d0*pi) then
            chi_beta = chi_beta - 2.0d0*pi        
        endif

        !  Calculation of relative direction and speed 
        Relative_Velocity = sqrt(UT * UT + U_ship * U_ship - 2.0D0 * UT * U_ship * cos(chi_beta))
        RV = (U_ship * U_ship + Relative_Velocity * Relative_Velocity - UT * UT) / 2.0D0 / U_ship / Relative_Velocity 

        if(abs(RV) > 1.0d0)then
            RV = dsign(1.0d0, RV) !   To avoid the value larger than 1.0 due to numerical error. 
        endif
        ! compute gamma_beta(angle between ship traveling direction and apparent wind)
        ! use if sentence because acos range is (0 pi) and gamma_beta is (0 2pi)
        if(chi_beta <= pi)then
            gamma_beta = acos(RV) 
        else
            gamma_beta = 2.0d0 * pi - acos(RV)
        endif

        ! compute relative direction(gamma_a) from gamma_beta
        Relative_Direction = gamma_beta + beta_hat

        if(abs(U_ship) < 1.0d-5)then
            Relative_Direction = mod(3.0d0*pi - chi, 2.0d0*pi)
        elseif(abs(UT) < 1.0d-5)then
            Relative_Direction = beta_hat
        endif
        
        if(Relative_Direction > 2.0d0*pi) then
            Relative_Direction = Relative_Direction - 2.0d0*pi
        elseif(Relative_Direction <0.0d0) then
            Relative_Direction = Relative_Direction + 2.0d0*pi 
        endif
    
    end subroutine Relative_Speed_and_Direction_Wind_or_Current

    subroutine WindForce(&
        V, psi, CXwind, CYwind, CNwind, CKwind,&
        FXwind, FYwind, FNwind, FKwind,&
        lpp, breadth, Lz, Windcoef)
        !
        !   Estimation of Wind Forces and Moments acting on Ships by Fujiwara's method
        !  Journal of the Society of Naval Architects of Japan, Vol.183, pp.77-90, 1998
        !
        !       <<<<<       Potsitive definition of X wind force is backward       >>>>>
        !       <<<<<  Potsitive definition of psi is oppsite from usual MNVR case >>>>>
        !  
        !  ---------------------------------  Input  ---------------------------------
        !
        !  V ---- Wind velocity (m/s)
        !  psi -- Angle of attack (Potsitive definition of psi is oppsite from usual MNVR case) (rad)
        !  AT --- Transverse Projected area (m^2)
        !  AL --- Lateral Projected area (m^2)
        !  AOD -- Lateral Projected area of superstructure Ass and LNG tanks,
        !         container etc. on the deck (m^2)
        !         Here, Ass is defined as lateral Projected area of superstructure (m^2)
        !  C ---- Distance from midship section to center of lateral projected area (m)
        !  CBR -- Distance from midship section to center of the Ass (m)
        !  HBR -- Height from free surface to top of the superstructure (bridge) (m) 
        !  HC --- Height to center of lateral projected area (m)
        !
        !  -------------  Output (q = 0.5 * rhoa * U_ship ** 2, HL = AL / L)  -------------
        !
        !  CXwind --- CXwind = FXwind / (q * AT)
        !  CYwind --- CYwind = FYwind / (q * AL)
        !  CNwind --- CNwind = FNwind / (q * lpp * AT)
        !  CKwind --- CKwind = FKwind / (q * AL * HL)
        !  FXwind --- X directional wind force  acting on the ship hull
        !  FYwind --- Y directional wind force  acting on the ship hull
        !  FNwind --- N directional wind force  acting on the ship hull
        !  FKwind --- K directional wind moment acting on the ship hull  
        implicit none
        integer :: i
        double precision, intent(in) :: V, psi
        double precision, intent(in) :: lpp, breadth, Lz
        double precision, intent(out) :: CXwind, CYwind, CNwind, CKwind, FXwind, FYwind, FNwind, FKwind
        double precision :: rhoA, Gam
        double precision :: Windcoef(10)
        double precision :: AT, AL, AOD, LC, LCBR, HBR, HC, SBW
        double precision :: X00, X01, X02, X03
        double precision :: X10, X11, X12, X13, X14, X15, X16, X17
        double precision :: X30, X31, X32, X33, X34, X35, X36, X37
        double precision :: X50, X51, X52, X53
        double precision :: Y10, Y11, Y12, Y13, Y14, Y15
        double precision :: Y30, Y31, Y32, Y33, Y34, Y35, Y36
        double precision :: Y50, Y51, Y52, Y53, Y54, Y55, Y56
        double precision :: N10, N11, N12, N13, N14, N15, N16, N17, N18
        double precision :: N20, N21, N22, N23, N24, N25, N26, N27, N28
        double precision :: N30, N31, N32, N33
        double precision :: K10, K11, K12, K13, K14, K15, K16, K17, K18
        double precision :: K20, K21, K22, K23, K24, K25, K26, K27
        double precision :: K30, K31, K32, K33, K34, K35, K36
        double precision :: K50, K51, K52, K53, K54, K55
        double precision :: XX0, XX1, XX3, XX5
        double precision :: YY1, YY3, YY5
        double precision :: NN1, NN2, NN3
        double precision :: KK1, KK2, KK3,KK5
        !
        !	Difinition of physical values
        rhoA = 1.220d0 / grav
        Gam = 0.57721566490153286060d0
        !
        !   Wind coefficients
        AT   = Windcoef(1); AL  = Windcoef(2); AOD = Windcoef(3); LC  = Windcoef(4);
        LCBR = Windcoef(5); HBR = Windcoef(6); HC  = Windcoef(7); SBW = Windcoef(8);
        !   Coefficients of Fujiwara's regression
        !   X-directional Coefficients
        X00 = -0.330D0;  X01 =  0.293D0;  X02 =  0.0193D0; X03 =  0.6820D0;
        X10 = -1.353D0;  X11 =  1.700D0;  X12 =  2.8700D0; X13 = -0.4630D0;
        X14 = -0.570D0;  X15 = -6.640D0;  X16 = -0.0123D0; X17 =  0.0202D0;
        X30 =  0.830D0;  X31 = -0.413D0;  X32 = -0.0827D0; X33 = -0.5630D0;
        X34 =  0.804D0;  X35 = -5.670D0;  X36 =  0.0401D0; X37 = -0.1320D0;
        X50 =  0.0372D0; X51 = -0.0075D0; X52 = -0.1030D0; X53 =  0.0921D0;
        !   Y-directional Coefficients
        Y10 =  0.684d0;  Y11 =   0.717d0;  Y12 = -3.2200d0; Y13 =  0.0281d0; Y14 =  0.0661d0; Y15 =  0.2980d0;
        Y30 = -0.400d0;  Y31 =   0.282d0;  Y32 =  0.3070d0; Y33 =  0.0519d0; Y34 =  0.0526d0; Y35 = -0.0814d0; Y36 =  0.0582d0;
        Y50 =  0.122d0;  Y51 =  -0.166d0;  Y52 = -0.0054d0; Y53 = -0.0481d0; Y54 = -0.0136d0; Y55 =  0.0864d0; Y56 = -0.0297d0;
        !   N-directional Coefficients
        N10 =  0.2990d0; N11 =   1.710d0;  N12 =  0.183d0;  N13 = -1.09d0;   N14 = -0.0442d0; N15 = -0.289d0
        N16 =  4.24d0;  N17 = -0.0646d0; N18 =  0.0306d0;
        N20 =  0.1170d0; N21 =   0.123d0;  N22 = -0.323d0;  N23 =  0.0041d0; N24 = -0.166d0;  N25 = -0.0109d0;
        N26 =  0.174d0; N27 =  0.214d0;  N28 = -1.06d0
        N30 =  0.0230d0; N31 =   0.0385d0; N32 = -0.0339d0; N33 =  0.0023d0; 
        !   K-directional Coefficients
        K10 =  3.63d0;   K11 = -30.7d0;    K12 = 16.8d0;    K13 =  3.270d0;  K14 = -3.03d0;   K15 =  0.552d0;
        K16 = -3.03d0;   K17 = 1.82d0;   K18 = -0.224d0; 
        K20 = -0.480d0;  K21 =   0.166d0;  K22 =  0.318d0;  K23 =  0.132d0;  K24 = -0.148d0;
        K25 =  0.408d0;  K26 = -0.0394d0; K27 = 0.0041d0; 
        K30 =  0.164d0;  K31 =  -0.170d0;  K32 =  0.0803d0; K33 =  4.920d0;  K34 = -1.780d0;  K35 =  0.0404d0; K36 = -0.739d0; 
        K50 =  0.449d0;  K51 =  -0.148d0;  K52 = -0.0049d0; K53 = -0.396d0;  K54 = -0.0109d0; K55 = -0.0726d0;
        !  
        XX0 = X00 + X01 * (SBW * HBR / AT) + X02 * (LC / HC) + X03 * (AOD / lpp / lpp)
        XX1 = X10 + X11 * (AL / lpp / SBW) + X12 * (lpp * HC / AL) + X13 * (lpp * HBR / AL) + X14 * (AOD / AL) &
            + X15 * (AT / lpp / SBW) + X16 * (lpp * lpp / AT) + X17 * (lpp / HC)
        XX3 = X30 + X31 * (AL / lpp / HBR) + X32 * (AL / AT) + X33 * (lpp * HC / AL) + X34 * (AOD / AL) + X35 * (AOD / lpp / lpp) &
            + X36 * (LC / HC) + X37 * (LCBR / lpp)
        XX5 = X50 + X51 * (AL / AOD) + X52 * (LCBR / lpp) + X53 * (AL / lpp / SBW)
        !
        YY1 = Y10 + Y11 * (LCBR / lpp) + Y12 * (LC / lpp) + Y13 * (AL / AOD) + Y14 * (LC / HC) + Y15 * (AT / (SBW * HBR))
        YY3 = Y30 + Y31 * (AL / (lpp * SBW)) + Y32 * (lpp * HC / AL) + Y33 * (LCBR / lpp) + Y34 * (SBW / HBR) &
            & + Y35 * (AOD / AL) + Y36 * (AT / (SBW * HBR))
        YY5 = Y50 + Y51 * (AL / (lpp * SBW)) + Y52 * (lpp / HBR) + Y53 * (LCBR / lpp) + Y54 * (SBW ** 2 / AT) & 
            & + Y55 * (LC / lpp) + Y56 * (LC * HC / AL)
        !
        NN1 = N10 + N11 * (LC / lpp) + N12 * (lpp * HC / AL) + N13 * (AT / AL) + N14 * (LC / HC) &
            + N15 * (AL / (lpp * SBW)) + N16 * (AT / lpp ** 2) + N17 * (SBW ** 2 / AT) + N18 * (LCBR / lpp)
        NN2 = N20 + N21 * (LCBR / lpp) + N22 * (LC / lpp) + N23 * (AL / AOD) + N24 * (AT / SBW ** 2) &
            + N25 * (lpp / HBR) + N26 * (AT / (SBW * HBR)) + N27 * (AL / (lpp * SBW)) + N28 * (AL / lpp ** 2)
        NN3 = N30 + N31  *(LCBR / lpp) + N32 * (AT / (SBW * HBR)) + N33 * (AL / AT)
        !
        KK1 =K10 + K11 * (HBR / lpp) + K12 * (AT / (lpp * SBW)) + K13 * (lpp * HC / AL) + K14 * (LC / lpp) &
            + K15 * (LCBR / lpp) + K16 * (SBW / HBR) + K17 * (SBW ** 2 / AT) + K18 * (lpp / SBW)
        KK2 =K20 + K21 * (SBW / HBR) + K22 * (AT / SBW ** 2) + K23 * (AL / (lpp * HC)) + K24 * (LCBR / lpp) &
            + K25 * (HBR * LC / AL) + K26 * (lpp / SBW) + K27 * (lpp ** 2 / AL)
        KK3 =K30 + K31 * (SBW ** 2 / AT) + K32 * (LCBR / lpp) + K33 * (HC / lpp) + K34 * (AT / (lpp * SBW)) &
            + K35 * (lpp * SBW / AL) + K36 * (AOD / lpp ** 2)
        KK5 =K50 + K51 * (AL / (lpp * HC)) + K52 * (AL / AOD) + K53 * (AT / AL) + K54 * (lpp / SBW) + K55 * (AL / (lpp * SBW))
    

        !
        !  Cal of non-dimentionalized coefficients
        CXwind = XX0 + XX1 * cos(psi) + XX3 * cos(3.0D0 * psi) + XX5 * cos(5.0D0 * psi)
        CYwind = YY1 * sin(psi) + YY3 * sin(3.0D0 * psi) + YY5 * sin(5.0D0 * psi) ! YY5 is corrected (XX5 -->> YY5) 05/13/2019
        CNwind = NN1 * sin(psi) + NN2 * sin(2.0D0 * psi) + NN3 * sin(3.0D0 * psi)
        CKwind = KK1 * sin(psi) + KK2 * sin(2.0D0 * psi) + KK3 * sin(3.0D0 * psi) + KK5 * sin(5.0D0 * psi)
        !  Dimentionalization
        FXwind = CXwind * (0.5D0 * rhoA * V * V) * AT
        FYwind = CYwind * (0.5D0 * rhoA * V * V) * AL
        FNwind = CNwind * (0.5D0 * rhoA * V * V) * lpp *AL
        FKwind = CKwind * (0.5D0 * rhoA * V * V) * AL * (AL / lpp)
        !  Convert K morment around G
        FKwind = FKwind + FYwind * Lz
        CKwind = FKwind / ((0.5D0 * rhoA * V * V) * AL * (AL / lpp))
        !
        return
    end subroutine WindForce

end module Takaoki_VR_vector_input