'''
MMG model for PCC(with CPP&VecTwin)
In this code, propeller and rudder forces are modeled by regressing the results of the towing tank experiment.
Thus, this MMG model can be applied for the simulation of maneuvering around hover rudder angle(-80~-40[deg]/40~80[deg])
'''
from platform import libc_ver
import numpy as np

from .relative import true2Apparent
from .wind import windCoeff, irregularWind, windForce
from .hullforce import hullForceYoshimuraSimpson
from .propellerforce_cpp_noborio import get_propellerforce4
from .rudderforce_vectwin_modified import get_rudderforceVecTwin
from .SideThrusterForce import SideThrusterForce_2ndOrder


class MmgModel():

    def __init__(self, init_const, init_parameter, model_switch, coef):
        """ instantiate MmgModel
        
        input principal particulas, model setting of computation, initial value of parameters.
        Args:
            init_const (dataflame, float): set of principle parameters etc.
            init_parameters (dataflame, float): set of initial parameteres (hydro derivatives, etc)
            model_switch (dataflame, int): set of model setting of computation
        Returns:
            none
        """
        # white model 
        ### Principal Particulars ###
        self.lpp                = init_const.at['lpp','value'] 
        self.B                  = init_const.at['breadth','value'] 
        self.draft              = init_const.at['draft','value'] 
        self.Mass_nd            = init_const.at['mass_nd','value'] 
        self.xG_nd              = init_const.at['x_lcg','value'] 
        #self.rho_fresh          = init_const.at['rho_fresh','value'] /9.80665
        self.rho_fresh          = init_const.at['rho_fresh','value']
        # Propeller 
        self.dia_prop           = init_const.at['dia_prop','value'] 
        self.pitchr             = init_const.at['pitchr','value']  
        # Rudder 
        self.area_rudder        = init_const.at['area_rudder','value'] 
        self.lambda_rudder      = init_const.at['lambda_rudder','value'] 
        self.x_location_rudder  = init_const.at['x_location_rudder','value']
        self.maxtickness_rudder = init_const.at['maxtickness_rudder','value'] 
        
        # parameters for wind force computation
        #self.rho_air                               = init_const.at['rho_air','value'] /9.80665 # density of air
        self.rho_air                               = init_const.at['rho_air','value'] # density of air
        self.area_projected_trans                  = init_const.at['area_projected_trans','value']  # AT
        self.area_projected_lateral                = init_const.at['area_projected_lateral','value']  # AL
        self.area_projected_lateral_superstructure = init_const.at['AOD','value']  # AOD 
        self.lcw                                   = init_const.at['LCW','value']  # C or LCW, midship to center of AL
        self.lcbr                                  = init_const.at['LCBR','value']  # CBR, midship to center of AOD(superstructure or bridge)
        self.hbr                                   = init_const.at['HBR','value'] #  Height from free surface to top of the superstructure (bridge) (m) 
        self.hc_AL                                 = init_const.at['HC','value']  # Hc, hight of center of lateral projected area
        self.breadth_wind                          = init_const.at['SBW','value']  # ship breadth fro wind force computation 
        self.swayforce_to_cg                       = init_const.at['Lz','value']  # Lz --- Acting position of sway force from center of gravity (it is necessary to calculate roll moment)
        
        ### Parameter Init ###
        self.MassX_nd      = init_parameter.at['massx_nd','params_init'] 
        self.MassY_nd      = init_parameter.at['massy_nd','params_init'] 
        self.IJzz_nd       = init_parameter.at['IzzJzz_nd','params_init'] 
        # Hull
        self.Xuu_nd        = init_parameter.at['xuu_nd','params_init'] 
        self.Xvr_nd        = init_parameter.at['xvr_nd','params_init'] 
        self.Yv_nd         = init_parameter.at['yv_nd','params_init'] 
        self.Yr_nd         = init_parameter.at['yr_nd','params_init']
        self.Nv_nd         = init_parameter.at['nv_nd','params_init'] 
        self.Nr_nd         = init_parameter.at['nr_nd','params_init'] 
        self.CD            = init_parameter.at['coeff_drag_sway','params_init']#0.500
        self.C_rY          = init_parameter.at['cry_cross_flow','params_init'] #1.00
        self.C_rN          = init_parameter.at['crn_cross_flow','params_init'] #0.50
        self.X_0F_nd       = self.Xuu_nd
        self.X_0A_nd       = init_parameter.at['coeff_drag_aft','params_init']


        # Propeller
        #self.t_prop       = init_parameter.at['t_prop','params_init']  
        #self.wP0          = init_parameter.at['w_prop_zero','params_init']  
        self.ai               = np.array([init_parameter.at['wp0_coeff0','params_init'], init_parameter.at['wp0_coeff1','params_init'], 
                                           init_parameter.at['wp0_coeff2','params_init'] ]) 
        self.bi               = np.array([init_parameter.at['wp_coeff0','params_init'], init_parameter.at['wp_coeff1','params_init'], 
                                           init_parameter.at['wp_coeff2','params_init'] ])                                
        self.ci               = np.array([init_parameter.at['t_prop0_coeff0','params_init'], init_parameter.at['t_prop0_coeff1','params_init'], 
                                           init_parameter.at['t_prop0_coeff2','params_init'] ])      
        self.di               = np.array([init_parameter.at['t_prop_coeff0','params_init'], init_parameter.at['t_prop_coeff1','params_init'], 
                                        init_parameter.at['t_prop_coeff2','params_init'] ])                                                                
        #self.tau           = init_parameter.at['tau_prop','params_init'] 
        #self.CP_nd         = init_parameter.at['coeff_cp_prop','params_init'] 
        self.xP_nd            = init_parameter.at['xp_prop_nd','params_init']
        # self.kt_coeff         = np.array([init_parameter.at['kt_coeff0','params_init'], init_parameter.at['kt_coeff1','params_init'], 
        #                                 init_parameter.at['kt_coeff2','params_init']])                                         
        self.kt2nd_coeff      = np.array([init_parameter.at['kt2nd_coeff0','params_init'], init_parameter.at['kt2nd_coeff1','params_init'], 
                                        init_parameter.at['kt2nd_coeff2','params_init']])                                 
        self.AAi               = np.array([init_parameter.at['ai_coeff_prop0','params_init'], init_parameter.at['ai_coeff_prop1','params_init'], 
                                           init_parameter.at['ai_coeff_prop2','params_init'], init_parameter.at['ai_coeff_prop3','params_init'],
                                           init_parameter.at['ai_coeff_prop4','params_init'], init_parameter.at['ai_coeff_prop5','params_init'],
                                           init_parameter.at['ai_coeff_prop6','params_init'], init_parameter.at['ai_coeff_prop7','params_init']]) 
        self.BBi               = np.array([init_parameter.at['bi_coeff_prop0','params_init'], init_parameter.at['bi_coeff_prop1','params_init'], 
                                           init_parameter.at['bi_coeff_prop2','params_init'], init_parameter.at['bi_coeff_prop3','params_init'],
                                           init_parameter.at['bi_coeff_prop4','params_init'], init_parameter.at['bi_coeff_prop5','params_init'],
                                           init_parameter.at['bi_coeff_prop6','params_init'], init_parameter.at['bi_coeff_prop7','params_init']])
        self.CCi               = np.array([init_parameter.at['ci_coeff_prop0','params_init'], init_parameter.at['ci_coeff_prop1','params_init'], 
                                           init_parameter.at['ci_coeff_prop2','params_init'], init_parameter.at['ci_coeff_prop3','params_init']] )
        self.Jmin             = init_parameter.at['Jmin','params_init']  # -0.5 coeff. from exp
        self.alpha_p          = init_parameter.at['alpha_prop','params_init']  # 1.5  coeff. from exp                                                                   
        
        startstep = 0  
        # set control variable
        self.degrees_propmodel= np.array([init_parameter.at['theta_dim_x','params_init'], init_parameter.at['theta_dim_y','params_init'], 
                                          init_parameter.at['theta_dim_N','params_init'], init_parameter.at['J_dim_x','params_init'],
                                          init_parameter.at['J_dim_y','params_init'], init_parameter.at['J_dim_N','params_init']])          
        self.kt_coeff_x       = coef.loc[coef.index[startstep:],'coef_Kx'].values
        self.kt_coeff_y       = coef.loc[coef.index[startstep:],'coef_Ky'].values
        self.kt_coeff_N       = coef.loc[coef.index[startstep:],'coef_Kn'].values

        # Rudder
        #self.t_rudder           = init_parameter.at['t_rudder','params_init'] 
        #self.ah_rudder          = init_parameter.at['ah_rudder','params_init']  
        #self.xh_rudder_nd       = init_parameter.at['xh_rudder_nd','params_init'] 
        self.lr_rudder_nd       = init_parameter.at['lr_rudder_nd','params_init']
        self.kx_rudder          = init_parameter.at['kx_rudder','params_init']
        #self.kx_rudder_reverse  = init_parameter.at['kx_rudder_reverse','params_init']
        #self.epsilon_rudder     = init_parameter.at['epsilon_rudder','params_init']
        self.cpr_rudder         = init_parameter.at['cpr_rudder','params_init'] 
        self.gammaN             = init_parameter.at['gammaN_rudder','params_init']
        self.gammaP             = init_parameter.at['gammaP_rudder','params_init']
        self.coeff_urudder_zero = init_parameter.at['coeff_urudder_zero','params_init']
        self.ei            = np.array([init_parameter.at['ei_coeff_vec0','params_init'], init_parameter.at['ei_coeff_vec1','params_init'], 
                                        init_parameter.at['ei_coeff_vec2','params_init'] ]) 
        self.fi            = np.array([init_parameter.at['fi_coeff_vec0','params_init'], init_parameter.at['fi_coeff_vec1','params_init'], 
                                        init_parameter.at['fi_coeff_vec2','params_init'] ])
        self.gi            = np.array([init_parameter.at['gi_coeff_vec0','params_init'], init_parameter.at['gi_coeff_vec1','params_init'], 
                                        init_parameter.at['gi_coeff_vec2','params_init'] ])
        self.asi           = np.array([init_parameter.at['asi_coeff_vec0','params_init'], init_parameter.at['asi_coeff_vec1','params_init'], 
                                        init_parameter.at['asi_coeff_vec2','params_init'], init_parameter.at['asi_coeff_vec3','params_init']])
        self.api           = np.array([init_parameter.at['api_coeff_vec0','params_init'], init_parameter.at['api_coeff_vec1','params_init'], 
                                        init_parameter.at['api_coeff_vec2','params_init'], init_parameter.at['api_coeff_vec3','params_init']])  
        self.bsi           = np.array([init_parameter.at['bsi_coeff_vec0','params_init'], init_parameter.at['bsi_coeff_vec1','params_init']])
        self.bpi           = np.array([init_parameter.at['bpi_coeff_vec0','params_init'], init_parameter.at['bpi_coeff_vec1','params_init']])
        self.dsi           = np.array([init_parameter.at['dsi_coeff_vec0','params_init'], init_parameter.at['dsi_coeff_vec1','params_init'], 
                                        init_parameter.at['dsi_coeff_vec2','params_init'] ])
        self.dpi           = np.array([init_parameter.at['dpi_coeff_vec0','params_init'], init_parameter.at['dpi_coeff_vec1','params_init'], 
                                        init_parameter.at['dpi_coeff_vec2','params_init'] ])
        self.esi           = np.array([init_parameter.at['esi_coeff_vec0','params_init'], init_parameter.at['esi_coeff_vec1','params_init'], 
                                        init_parameter.at['esi_coeff_vec2','params_init'] ])
        self.epi           = np.array([init_parameter.at['epi_coeff_vec0','params_init'], init_parameter.at['epi_coeff_vec1','params_init'], 
                                        init_parameter.at['epi_coeff_vec2','params_init'] ])
        self.gsi           = np.array([init_parameter.at['gsi_coeff_vec0','params_init'], init_parameter.at['gsi_coeff_vec1','params_init'], 
                                        init_parameter.at['gsi_coeff_vec2','params_init'] ])
        self.gpi           = np.array([init_parameter.at['gpi_coeff_vec0','params_init'], init_parameter.at['gpi_coeff_vec1','params_init'], 
                                        init_parameter.at['gpi_coeff_vec2','params_init'] ])
        self.hsi           = np.array([init_parameter.at['hsi_coeff_vec0','params_init'], init_parameter.at['hsi_coeff_vec1','params_init'], 
                                        init_parameter.at['hsi_coeff_vec2','params_init'] ])
        self.hpi           = np.array([init_parameter.at['hpi_coeff_vec0','params_init'], init_parameter.at['hpi_coeff_vec1','params_init'], 
                                        init_parameter.at['hpi_coeff_vec2','params_init'] ])
        self.ksi           = np.array([init_parameter.at['ksi_coeff_vec0','params_init'], init_parameter.at['ksi_coeff_vec1','params_init'], 
                                        init_parameter.at['ksi_coeff_vec2','params_init'] ])
        self.kpi           = np.array([init_parameter.at['kpi_coeff_vec0','params_init'], init_parameter.at['kpi_coeff_vec1','params_init'], 
                                        init_parameter.at['kpi_coeff_vec2','params_init'] ])
        self.rsi           = np.array([init_parameter.at['rsi_coeff_vec0','params_init'], init_parameter.at['rsi_coeff_vec1','params_init'], 
                                        init_parameter.at['rsi_coeff_vec2','params_init'], init_parameter.at['rsi_coeff_vec3','params_init'],
                                        init_parameter.at['rsi_coeff_vec4','params_init'], init_parameter.at['rsi_coeff_vec5','params_init'], 
                                        init_parameter.at['rsi_coeff_vec6','params_init'], init_parameter.at['rsi_coeff_vec7','params_init'],
                                        init_parameter.at['rsi_coeff_vec8','params_init'], init_parameter.at['rsi_coeff_vec9','params_init'], 
                                        init_parameter.at['rsi_coeff_vec10','params_init'], init_parameter.at['rsi_coeff_vec11','params_init'],
                                        init_parameter.at['rsi_coeff_vec12','params_init'], init_parameter.at['rsi_coeff_vec13','params_init'], 
                                        init_parameter.at['rsi_coeff_vec14','params_init'], init_parameter.at['rsi_coeff_vec15','params_init']])                                
        self.rpi           = np.array([init_parameter.at['rpi_coeff_vec0','params_init'], init_parameter.at['rpi_coeff_vec1','params_init'], 
                                        init_parameter.at['rpi_coeff_vec2','params_init'], init_parameter.at['rpi_coeff_vec3','params_init'],
                                        init_parameter.at['rpi_coeff_vec4','params_init'], init_parameter.at['rpi_coeff_vec5','params_init'], 
                                        init_parameter.at['rpi_coeff_vec6','params_init'], init_parameter.at['rpi_coeff_vec7','params_init'],
                                        init_parameter.at['rpi_coeff_vec8','params_init'], init_parameter.at['rpi_coeff_vec9','params_init'], 
                                        init_parameter.at['rpi_coeff_vec10','params_init'], init_parameter.at['rpi_coeff_vec11','params_init'],
                                        init_parameter.at['rpi_coeff_vec12','params_init'], init_parameter.at['rpi_coeff_vec13','params_init'], 
                                        init_parameter.at['rpi_coeff_vec14','params_init'], init_parameter.at['rpi_coeff_vec15','params_init']])                                                                                                                                                                                                                                                                                                 
        self.Csi           = np.array([init_parameter.at['Csi_coeff_vec0','params_init'], init_parameter.at['Csi_coeff_vec1','params_init'], 
                                        init_parameter.at['Csi_coeff_vec2','params_init'], init_parameter.at['Csi_coeff_vec3','params_init']])
        self.Cpi           = np.array([init_parameter.at['Cpi_coeff_vec0','params_init'], init_parameter.at['Cpi_coeff_vec1','params_init'], 
                                        init_parameter.at['Cpi_coeff_vec2','params_init'], init_parameter.at['Cpi_coeff_vec3','params_init']])
        # Side Thruster
        self.D_bthrust           = init_const.at['D_bthrust','value']
        self.D_sthrust           = init_const.at['D_sthrust','value']
        self.x_bowthrust_loc     = init_const.at['x_bowthrust_loc','value']
        self.x_stnthrust_loc     = init_const.at['x_stnthrust_loc','value']
        self.KT_bow_forward      = init_parameter.at['KT_bow_forward','params_init']
        self.KT_bow_reverse      = init_parameter.at['KT_bow_reverse','params_init']
        self.aY_bow              = np.array([init_parameter.at['aY_bow_coeff0','params_init'], init_parameter.at['aY_bow_coeff1','params_init'], 
                                        init_parameter.at['aY_bow_coeff2','params_init'] ]) 
        self.aN_bow              = np.array([init_parameter.at['aN_bow_coeff0','params_init'], init_parameter.at['aN_bow_coeff1','params_init'], 
                                        init_parameter.at['aN_bow_coeff2','params_init'] ])                                 
        self.BRps                = init_parameter.at['BRps','params_init']
        self.KT_stern_forward    = init_parameter.at['KT_stern_forward','params_init']
        self.KT_stern_reverse    = init_parameter.at['KT_stern_reverse','params_init']
        self.aY_stern              = np.array([init_parameter.at['aY_stern_coeff0','params_init'], init_parameter.at['aY_stern_coeff1','params_init'], 
                                        init_parameter.at['aY_stern_coeff2','params_init'] ]) 
        self.aN_stern              = np.array([init_parameter.at['aN_stern_coeff0','params_init'], init_parameter.at['aN_stern_coeff1','params_init'], 
                                        init_parameter.at['aN_stern_coeff2','params_init'] ])    
        self.SRps                = init_parameter.at['SRps','params_init']
        self.Thruster_speed_max  = init_parameter.at['Thruster_speed_max','params_init']

         ### switch computation model
        self.switch_prop_x     = model_switch.at['switch_prop_x','value'] # 1: Hachii 2: Yasukawa for thrust computation on 2,3,4th q.
        self.switch_prop_y     = model_switch.at['switch_prop_y','value'] # 0: YP=0 at n>0, 1: use exp.polynomial for 2nd q 
        self.switch_prop_n     = model_switch.at['switch_prop_n','value'] # 0: NP=0 at n>0, 1: use exp.polynomial for 2nd q
        self.switch_rudder     = model_switch.at['switch_rudder','value'] # 0: YR & NR are 0 at n<0, 1: compute force at n<0
        self.switch_fn_rudder  = model_switch.at['switch_fn','value']     # model for rudder normal force, 1: Fujii, 2: lindenburg
        self.switch_cn_rudder  = model_switch.at['switch_cn','value']     # model for rudder normal force, 1: CFD, 2: Kang
        self.switch_ur_rudder  = model_switch.at['switch_ur','value']     # model for effective inflow angle of rudder, 1: standard MMG by Yoshimura, 2: reversed model by Yasukawa on u<0
        self.switch_wind       = model_switch.at['switch_wind','value'] # 0: off, 1: on
        self.switch_windtype   = model_switch.at['switch_windtype','value'] # 1: uniform wind, 2: irregular wind, 3: instantaneous wind measured by exp (converted to true), 4: use exp relavite wind

        # fig_path = os.path.join('4valid/draw_traj/output/', file_name + "_output_force.csv" )
        # self.f = open( fig_path, mode='w' ) 
        # self.writer = csv.writer(self.f)
        # self.writer.writerow(["XH","YH","NH","XP","YP","NP","XR","YR","NR",
        #                       "XST","YST","NST","XA","YA","NA","X","Y","N",
        #                       "Rudder_L","Rudder_R","prop_rev","pitch_angle"])
        
        ### wind force coefficients
        if 'XX0' in init_parameter.index :
            self.XX0 = init_parameter.at['XX0','params_init'] 
            self.XX1 = init_parameter.at['XX1','params_init'] 
            self.XX3 = init_parameter.at['XX3','params_init'] 
            self.XX5 = init_parameter.at['XX5','params_init'] 
            self.YY1 = init_parameter.at['YY1','params_init'] 
            self.YY3 = init_parameter.at['YY3','params_init']
            self.YY5 = init_parameter.at['YY5','params_init']
            self.NN1 = init_parameter.at['NN1','params_init']  
            self.NN2 = init_parameter.at['NN2','params_init']  
            self.NN3 = init_parameter.at['NN3','params_init']
            self.KK1 = 0
            self.KK2 = 0
            self.KK3 = 0
            self.KK5 = 0  

        else:
            self.windcoeff =windCoeff(self.lpp, self.area_projected_trans, self.area_projected_lateral, self.area_projected_lateral_superstructure,
                                            self.lcw, self.lcbr, self.hbr, self.hc_AL, self.breadth_wind, self.swayforce_to_cg)
            self.XX0 = self.windcoeff[0]
            self.XX1 = self.windcoeff[1]
            self.XX3 = self.windcoeff[2]
            self.XX5 = self.windcoeff[3]
            self.YY1 = self.windcoeff[4]
            self.YY3 = self.windcoeff[5]
            self.YY5 = self.windcoeff[6]
            self.NN1 = self.windcoeff[7]
            self.NN2 = self.windcoeff[8]
            self.NN3 = self.windcoeff[9]
            self.KK1 = self.windcoeff[10]
            self.KK2 = self.windcoeff[11]
            self.KK3 = self.windcoeff[12]
            self.KK5 = self.windcoeff[13]

    def setParams(self, parameter):
        """ update parameters (hydro derivatives etc) of MMG model
        
        Args:
            parameter(dataflame, float): set of parameters to update
        
        Returns:
            none
        """
        self.MassX_nd      = parameter.at['massx_nd','value'] 
        self.MassY_nd      = parameter.at['massy_nd','value'] 
        self.IJzz_nd       = parameter.at['IzzJzz_nd','value'] 
        # Hull
        self.Xuu_nd        = parameter.at['xuu_nd','value'] 
        self.Xvr_nd        = parameter.at['xvr_nd','value'] 
        self.Yv_nd         = parameter.at['yv_nd','value'] 
        self.Yr_nd         = parameter.at['yr_nd','value']
        self.Nv_nd         = parameter.at['nv_nd','value'] 
        self.Nr_nd         = parameter.at['nr_nd','value'] 
        self.CD            = parameter.at['coeff_drag_sway','value']#0.500
        self.C_rY          = parameter.at['cry_cross_flow','value'] #1.00
        self.C_rN          = parameter.at['crn_cross_flow','value'] #0.50
        self.X_0F_nd       = self.Xuu_nd
        self.X_0A_nd       = parameter.at['coeff_drag_aft','value']
        # Propeller
        self.t_prop             = parameter.at['t_prop','value']  
        self.wP0           = parameter.at['w_prop_zero','value']  
        self.tau           = parameter.at['tau_prop','value'] 
        self.CP_nd         = parameter.at['coeff_cp_prop','value'] 
        self.xP_nd         = parameter.at['xp_prop_nd','value'] 
        self.kt_coeff      = np.array([parameter.at['kt_coeff0','value'], parameter.at['kt_coeff1','value'], 
                                        parameter.at['kt_coeff2','value']]) 
        self.AAi            = np.array([parameter.at['ai_coeff_prop0','value'], parameter.at['ai_coeff_prop1','value'], 
                                        parameter.at['ai_coeff_prop2','value'], parameter.at['ai_coeff_prop3','value'],
                                        parameter.at['ai_coeff_prop4','value'], parameter.at['ai_coeff_prop5','value'],
                                        parameter.at['ai_coeff_prop6','value'], parameter.at['ai_coeff_prop7','value']]) 
        self.BBi            = np.array([parameter.at['bi_coeff_prop0','value'], parameter.at['bi_coeff_prop1','value'], 
                                        parameter.at['bi_coeff_prop2','value'], parameter.at['bi_coeff_prop3','value'],
                                        parameter.at['bi_coeff_prop4','value'], parameter.at['bi_coeff_prop5','value'],
                                        parameter.at['bi_coeff_prop6','value'], parameter.at['bi_coeff_prop7','value']])
        self.CCi            = np.array([parameter.at['ci_coeff_prop0','value'], parameter.at['ci_coeff_prop1','value'], 
                                        parameter.at['ci_coeff_prop2','value'], parameter.at['ci_coeff_prop3','value']]) 
        self.Jmin          = parameter.at['Jmin','value']  # -0.5 coeff. from exp
        self.alpha_p       = parameter.at['alpha_prop','value']  # 1.5  coeff. from exp   
        
        # Rudder
        self.t_rudder           = parameter.at['t_rudder','value'] 
        self.ah_rudder          = parameter.at['ah_rudder','value']
        self.xh_rudder_nd       = parameter.at['xh_rudder_nd','value'] 
        self.lr_rudder_nd       = parameter.at['lr_rudder_nd','value']
        self.kx_rudder          = parameter.at['kx_rudder','value']
        self.kx_rudder_reverse  = parameter.at['kx_rudder_reverse','value']
        #self.epsilon_rudder     = parameter.at['epsilon_rudder','value']
        self.cpr_rudder         = parameter.at['cpr_rudder','value'] 
        self.gammaN             = parameter.at['gammaN_rudder','value']
        self.gammaP             = parameter.at['gammaP_rudder','value']
        
    ##### Equation of motion #####
    def hydroForceLs(self, physical_time, x, n_prop, pitch_ang_prop, delta_rudder_s , delta_rudder_p ,n_bow_thruster,  n_stern_thruster, wind_velo_true, wind_dir_true,save_intermid):
        """ return accel. of ship by mmg model for low speed
        
        return accelerations of ship by mmg model for low speed from state value (position, speed, heading) and control (propeller rev., rudder angle).
        process several timestep's data at once.(use np.ndarray)
        Args:
            physical_time (ndarray, float) : physical time of sim (for irregular wind comp.) shape =[number_of_data, 1]
            x (ndarray, float) : set of state values. comtain data for several timestep (small batch).
                                x.shape = [number_of_timestep, number_of_stateval]
                                x = [x_position, u_velo, y_position, vm_velo, psi_hat, r_angvelo]
            n_prop (ndarray, float) : set of propeller revolution. (rps). n_prop.shape = [number_of_timestep]
            pitch_prop (ndarray, float) : set of propeller pitch. (!! rad !!). 
            n_bow_thruster (ndarray, float) : set of thruster revolution. (rps). 
            n_stern_thruster (ndarray, float) : set of tthruster revolution. (rps). 
            delta_rudder_s(ndarray, float): set of angle of starboard side rudder (!! rad !!). [-pi, pi], delta_rudder.shape = [number_of_timestep]
            delta_rudder_p(ndarray, float): set of angle of port side rudder (!! rad !!). [-pi, pi], delta_rudder.shape = [number_of_timestep]
            wind_velo_true (ndarray, float) : true wind velocity (velocity to ground.) if don't calculate wind force, put  np.zeros.
                                if use numerical wind (uniform of irregular), put average true wind for all timestep.
                                if use experimental result measured by Anemoeter, put instant value converted to midship from experimental result.
                                unit = (m/s). shape = [number_of_timestep]
            wind_dir_true (ndarray, float) : true wind direction (direction to ground). related to the earth fix coordinate.
                                            !! please convert to earth-fix coordinate x-direction you are using, in some case, north is not zero.  
                                            treat as same as wind_velo_true.
                                            unit = (rad), [0, 2pi]. shape = [number_of_timestep], clock-wise.
        Retruns:
            retrun (float, ndarray): time derivative of state value driven by MMG model. return.shape= [number_of_timestep, number_of_stateval]
                                    [X_dot, u_dot, Y_dot, vm_dot, psi_dot(=r_angvelo), r_dot]
        
        Note:
            Detail of state values:
            x_position_mid : mid ship position of ship on earth-fix coordinate: x_hat - y_hat system(= X-Y system).
            u_velo : ship fix velocity of ship heading direction.
            y_position_mid : mid ship potision of ship on earth-fix coordinate: x_hat - y_hat system(= X-Y system).
            vm_velo: ship fix velocity of ship breadth direction.
            psi_hat : heading of ship. psi_hat = 0 at x_hat direction.
            r_angvelo: angular velocity of ship heading (Yaw).
            
            Detail of time derivative of state values:
            x_dot : time derivative of x_position_mid. Earth fix velocity of ship, drive from coor. conversion of u_velo.
            u_dot : time derivative of u_velo. ship fix acceleration of ship heading direction.
            y_dot : time derivative of y_position_mid. Earth fix velocity of ship, drive from coor. conversion of vm_velo.
            u_dot : time derivative of vm_velo. ship fix acceleration of ship breadth direction.
            psi_dot: time derivative of psi_hat. which equal to r_angvelo.
            r_dot : time derivative of r_angvelo.            
        """        
        ### read State Variables ###
        x_position_mid = np.copy(x[:, 0:1])
        u_velo         = np.copy(x[:, 1:2])
        y_position_mid = np.copy(x[:, 2:3])
        vm_velo        = np.copy(x[:, 3:4])
        psi_hat        = np.copy(x[:, 4:5]) # psi_hat obtained by MMG (!! rad !!)
        r_angvelo      = np.copy(x[:, 5:6])

        ### main ###
        # Forward velocity
        U_ship    = np.sqrt(u_velo**2 + vm_velo**2)
        beta_hat  = np.arctan2(vm_velo, u_velo)
        beta      = np.arctan2(-vm_velo, u_velo)
        
        
        # add dimension
        Dim_add_M  = 0.5 * self.rho_fresh * self.lpp**2 * self.draft
        Dim_add_I  = 0.5 * self.rho_fresh * self.lpp**4 * self.draft #/ 9.80665
        Dim_add_uv = 0.5 * self.rho_fresh * self.lpp    * self.draft * U_ship**2
        Dim_add_r  = 0.5 * self.rho_fresh * self.lpp**2 * self.draft * U_ship**2
        Mass       = self.Mass_nd  * Dim_add_M
        MassX      = self.MassX_nd * Dim_add_M
        MassY      = self.MassY_nd * Dim_add_M
        IJzz       = self.IJzz_nd  * Dim_add_I
        xG         = self.xG_nd    * self.lpp

        # avoid zero divide at approx. U_ship = 0
        u_nd   = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-8 * np.ones(u_velo.shape),  u_velo  / U_ship        )
        v_nd   = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-8 * np.ones(vm_velo.shape), vm_velo / U_ship        )
        r_nd   = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-8 * np.ones(r_angvelo.shape),  r_angvelo * self.lpp / U_ship)
        U_ship = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-8 * np.ones(U_ship.shape),  U_ship             )
        
        ##########################
        ### Force of Hull ###
        ##########################
        hull_force = hullForceYoshimuraSimpson(U_ship, u_velo, vm_velo, r_angvelo, beta_hat, self.lpp, self.draft, 
                                                                   self.rho_fresh, self.X_0F_nd, self.X_0A_nd,self.Xvr_nd, self.Yv_nd, 
                                                                   self.Yr_nd, self.Nv_nd, self.Nr_nd, self.C_rY, self.C_rN, self.CD)

        XH = hull_force[:, 0:1] #/ 9.80665
        YH = hull_force[:, 1:2] #/ 9.80665
        NH = hull_force[:, 2:3] #/ 9.80665

        ##########################
        ### Force of Propeller ###
        ##########################

        XP = 0.0
        NP = 0.0
        YP = 0.0

        ##########################
        ### Force of Ruder ###
        ##########################

        # print(pitch_ang_prop)
        pitch_ang_prop = np.rad2deg(pitch_ang_prop)
        delta_rudder_p = np.rad2deg(delta_rudder_p)
        delta_rudder_s = np.rad2deg(delta_rudder_s)

        V_coef = np.array([4.82458691e-01, -8.01652940e-03, -1.94593141e-02, 1.53615177e-03,
                           1.48563746e-05, -1.74941853e-03, -2.47304074e-04, -2.12339043e-06,
                          -1.70487254e-04, -6.65162582e-07,  8.77495562e-09, -9.27218377e-03,
                          -5.10057104e-04,  1.00164671e-05, -3.36487299e-03,  1.51873270e-04,
                          -1.77100085e-06,  3.07293291e-04, -8.94825796e-06,  6.94006503e-08,
                          -1.40876676e+00,  1.83897910e-02, -4.09427038e-01, -8.81973398e-03,
                          -6.11287262e-05,  5.04699395e-02,  1.07439772e-03,  7.44404433e-06,
                          -8.92557220e-04, -1.75425692e-05, -1.16751685e-07, -1.05895970e+00,
                           1.77064555e-02, -9.96437531e-05,  1.06329721e-01, -1.78314618e-03,
                           1.01778750e-05, -1.54509584e-03,  2.50982659e-05, -1.42137587e-07,
                           1.83514313e+00, -2.02918733e-02,  7.67405327e-01,  1.60847067e-02,
                           1.08678137e-04, -9.27472410e-02, -1.92770177e-03, -1.30525644e-05,
                           1.68414568e-03,  3.23087639e-05,  2.09478428e-07,  1.60214230e+00,
                          -2.70231508e-02,  1.53350921e-04, -1.61438842e-01,  2.73877696e-03,
                          -1.58080394e-05,  2.28987035e-03, -3.77225107e-05,  2.17382124e-07])

        Fitcp = np.array([ -4.22943469, 15.34867271, -20.50352279 ])

        if n_prop > 0: 
            XR = V_coef[0]*pitch_ang_prop + V_coef[1]*(pitch_ang_prop**2) + V_coef[2]*delta_rudder_p + V_coef[3]*(delta_rudder_p**2) + V_coef[4]*(delta_rudder_p**3) + \
                 V_coef[5]*pitch_ang_prop*delta_rudder_p + V_coef[6]*pitch_ang_prop*(delta_rudder_p**2) + V_coef[7]*pitch_ang_prop*(delta_rudder_p**3) + \
                 V_coef[8]*(pitch_ang_prop**2)*delta_rudder_p + V_coef[9]*(pitch_ang_prop**2)*(delta_rudder_p**2) + V_coef[10]*(pitch_ang_prop**2)*(delta_rudder_p**3) + \
                 V_coef[11]*delta_rudder_s + V_coef[12]*(delta_rudder_s**2) + V_coef[13]*(delta_rudder_s**3) + \
                 V_coef[14]*pitch_ang_prop*delta_rudder_s + V_coef[15]*pitch_ang_prop*(delta_rudder_s**2) + V_coef[16]*pitch_ang_prop*(delta_rudder_s**3) + \
                 V_coef[17]*(pitch_ang_prop**2)*delta_rudder_s + V_coef[18]*(pitch_ang_prop**2)*(delta_rudder_s**2) + V_coef[19]*(pitch_ang_prop**2)*(delta_rudder_s**3) + Fitcp[0]
            YR = V_coef[20]*pitch_ang_prop + V_coef[21]*(pitch_ang_prop**2) + V_coef[22]*delta_rudder_p + V_coef[23]*(delta_rudder_p**2) + V_coef[24]*(delta_rudder_p**3) + \
                 V_coef[25]*pitch_ang_prop*delta_rudder_p + V_coef[26]*pitch_ang_prop*(delta_rudder_p**2) + V_coef[27]*pitch_ang_prop*(delta_rudder_p**3) + \
                 V_coef[28]*(pitch_ang_prop**2)*delta_rudder_p + V_coef[29]*(pitch_ang_prop**2)*(delta_rudder_p**2) + V_coef[30]*(pitch_ang_prop**2)*(delta_rudder_p**3) + \
                 V_coef[31]*delta_rudder_s + V_coef[32]*(delta_rudder_s**2) + V_coef[33]*(delta_rudder_s**3) + \
                 V_coef[34]*pitch_ang_prop*delta_rudder_s + V_coef[35]*pitch_ang_prop*(delta_rudder_s**2) + V_coef[36]*pitch_ang_prop*(delta_rudder_s**3) + \
                 V_coef[37]*(pitch_ang_prop**2)*delta_rudder_s + V_coef[38]*(pitch_ang_prop**2)*(delta_rudder_s**2) + V_coef[39]*(pitch_ang_prop**2)*(delta_rudder_s**3) + Fitcp[1]
            NR = V_coef[40]*pitch_ang_prop + V_coef[41]*(pitch_ang_prop**2) + V_coef[42]*delta_rudder_p + V_coef[43]*(delta_rudder_p**2) + V_coef[44]*(delta_rudder_p**3) + \
                 V_coef[45]*pitch_ang_prop*delta_rudder_p + V_coef[46]*pitch_ang_prop*(delta_rudder_p**2) + V_coef[47]*pitch_ang_prop*(delta_rudder_p**3) + \
                 V_coef[48]*(pitch_ang_prop**2)*delta_rudder_p + V_coef[49]*(pitch_ang_prop**2)*(delta_rudder_p**2) + V_coef[50]*(pitch_ang_prop**2)*(delta_rudder_p**3) + \
                 V_coef[51]*delta_rudder_s + V_coef[52]*(delta_rudder_s**2) + V_coef[53]*(delta_rudder_s**3) + \
                 V_coef[54]*pitch_ang_prop*delta_rudder_s + V_coef[55]*pitch_ang_prop*(delta_rudder_s**2) + V_coef[56]*pitch_ang_prop*(delta_rudder_s**3) + \
                 V_coef[57]*(pitch_ang_prop**2)*delta_rudder_s + V_coef[58]*(pitch_ang_prop**2)*(delta_rudder_s**2) + V_coef[59]*(pitch_ang_prop**2)*(delta_rudder_s**3) + Fitcp[2]
        
        else:
            XR = 0.0
            YR = 0.0
            NR = 0.0

        # print(XR)

        ##############################
        ### Force of Side Thruster ###
        ##############################
        n_stern_thruster = np.zeros(np.shape(n_bow_thruster))
        side_thruster_force = SideThrusterForce_2ndOrder(self.rho_fresh, u_velo, self.lpp, self.D_bthrust, self.D_sthrust, 
                                                                            n_bow_thruster, n_stern_thruster, self.aY_bow, self.aY_stern, self.aN_bow,
                                                                             self.aN_stern, self.KT_bow_forward, self.KT_bow_reverse, self.KT_stern_forward, 
                                                                             self.KT_stern_reverse, self.x_bowthrust_loc, self.x_stnthrust_loc, self.Thruster_speed_max,)
        XST = side_thruster_force[:, 0]
        YST = side_thruster_force[:, 1]
        NST = side_thruster_force[:, 2]

        ##########################
        ### Force of wind ###
        ##########################
        XA = np.zeros((x_position_mid.size, 1))
        YA = np.zeros((x_position_mid.size, 1))
        NA = np.zeros((x_position_mid.size, 1))
        if self.switch_wind == 0: # no wind force
            pass
        ### compute instant relative(apparent) wind dirction and speed
        elif self.switch_wind == 1:
            
            if self.switch_windtype == 1: #uniform
                wind_velo_true_instant = np.copy(wind_velo_true)
                wind_dir_true_instant  = np.copy(wind_dir_true)
                ### compute relative wind on midship
                wind_relative = true2Apparent(wind_velo_true_instant, U_ship, wind_dir_true_instant, beta_hat, psi_hat)
                wind_velo_relative = np.copy(wind_relative[:,0:1])
                angle_of_attack = 2.0*np.pi-np.copy(wind_relative[:,1:2])    
            elif self.switch_windtype == 2: # irregular wind
                wind_velo_true_instant = irregularWind(wind_velo_true, physical_time)
                wind_dir_true_instant  = np.copy(wind_dir_true)
                ### compute relative wind on midship
                wind_relative = true2Apparent(wind_velo_true_instant, U_ship, wind_dir_true_instant, beta_hat, psi_hat)
                wind_velo_relative = np.copy(wind_relative[:,0:1])
                angle_of_attack = 2.0*np.pi-np.copy(wind_relative[:,1:2])
            elif self.switch_windtype == 3: # measured wind by anemometer on model ship
                ### !!!! caution !!!!!
                ### conevert to true wind(wind_velo_true, wind_dir_true) from anemometer output 
                ### before use this part
                wind_velo_true_instant = np.copy(wind_velo_true)
                wind_dir_true_instant  = np.copy(wind_dir_true)
                ### compute relative wind on midship
                wind_relative = true2Apparent(wind_velo_true_instant, U_ship, wind_dir_true_instant, beta_hat, psi_hat)
                wind_velo_relative = np.copy(wind_relative[:,0:1])
                angle_of_attack = 2.0*np.pi-np.copy(wind_relative[:,1:2])
            elif self.switch_windtype == 4: # use apparent wind on model, not recommend to use, for debug
                wind_velo_relative = np.copy(wind_velo_true)
                angle_of_attack = 2.0*np.pi-np.copy(wind_dir_true)
        
            ### compute wind force by relative wind 
            wind_force = windForce(wind_velo_relative, angle_of_attack, self.area_projected_trans,
                                    self.area_projected_lateral, self.swayforce_to_cg,
                                    self.rho_air, self.lpp, self.XX0, self.XX1, self.XX3, self.XX5,
                                    self.YY1, self.YY3, self.YY5, self.NN1, self.NN2, self.NN3,
                                    self.KK1, self.KK2, self.KK3, self.KK5)
            XA = wind_force[:, 0:1]
            YA = wind_force[:, 1:2]
            NA = wind_force[:, 2:3]

        ######################
        ###    Summation of every force and moment
        ######################
        X = XH + XP + XR + XST + XA
        Y = YH + YP + YR + YST + YA
        N = NH + NP + NR + NST + NA

        AA1 = Mass + MassY
        AA2 = xG * Mass
        AA3 = Y - (Mass+MassX) * u_velo * r_angvelo
        BB1 = IJzz + xG**2 * Mass
        BB2 = xG * Mass
        BB3 = N - xG * Mass * u_velo * r_angvelo

        u_dot  = (X + (Mass+MassY)*vm_velo*r_angvelo 
                + xG*Mass*r_angvelo**2) / (Mass+MassX)
        vm_dot = (AA3*BB1 - AA2*BB3) / (AA1*BB1 - AA2*BB2)
        r_dot  = (AA3*AA2 - BB3*AA1) / (AA2*BB2 - AA1*BB1)
        x_dot = u_velo * np.cos(psi_hat) - vm_velo * np.sin(psi_hat) 
        y_dot = u_velo * np.sin(psi_hat) + vm_velo * np.cos(psi_hat)

        myvr  = MassY*r_angvelo*vm_velo
        
        #### set intermideate variables as attribute for postprocess####
        if(save_intermid):
            self.XH                   = XH
            self.YH                   = YH
            self.NH                   = NH

            self.XP                   = XP
            self.YP                   = YP
            self.NP                   = NP

            self.XR                   = XR
            self.YR                   = YR
            self.NR                   = NR

            self.XA                   = XA
            self.YA                   = YA
            self.NA                   = NA

            self.X                    = X
            self.Y                    = Y
            self.N                    = N

            self.aR_s                 = aR_s
            self.aR_p                 = aR_p

            self.resultant_U_rudder_s = resultant_U_rudder_s
            self.resultant_U_rudder_p = resultant_U_rudder_p

            self.u_rudder_s           = u_rudder_s
            self.u_rudder_p           = u_rudder_p

            

            #self.FNs                  = FNs
            #self.FNp                  = FNp
            #self.aR_s_deg             = aR_s_deg
            #self.aR_p_deg             = aR_p_deg
            #self.delta_rudder_s_deg   = delta_rudder_s_deg
            #self.delta_rudder_p_deg   = delta_rudder_p_deg
            #self.delta_rudder_s2_deg  = delta_rudder_s2_deg
            #self.delta_rudder_p2_deg  = delta_rudder_p2_deg
            #self.delta_rudder_s_p_deg = delta_rudder_s_p_deg
            #self.delta_rudder_p_s_deg = delta_rudder_p_s_deg
            #self.delta_rudder_s1_deg  = delta_rudder_s1_deg
            #self.delta_rudder_p1_deg  = delta_rudder_p1_deg
            #self.Cns                  = Cns
            #self.Cnp                  = Cnp
            #self.uprop                = uprop
            #self.uR_p_s               = uR_p_s
            #self.uR_p_p               = uR_p_p
            #self.uR_decrease_ratio_s  = uR_decrease_ratio_s
            #self.uR_decrease_ratio_p  = uR_decrease_ratio_p
            #self.slip_ratio           = slip_ratio
  
        return np.concatenate([x_dot, u_dot, y_dot, vm_dot, r_angvelo, r_dot], axis=1)
