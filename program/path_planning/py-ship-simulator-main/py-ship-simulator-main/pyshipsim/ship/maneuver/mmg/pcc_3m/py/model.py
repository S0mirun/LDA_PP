import os
import numpy as np
import pandas as pd

from .MMG_1p2r_rudder_from_exp import MmgModel


__dir__ = os.path.dirname(__file__)


class PCCMMG(object):
    def __init__(self):
        #
        principal_particulars = pd.read_csv(
            f"{pcc_dir}/input_csv/principal_particulars_PCC.csv",
            header=0,
            index_col=0,
        )
        parameter_init = pd.read_csv(
            f"{pcc_dir}/input_csv/MMG_params_PCC.csv", header=0, index_col=0
        )
        switch = pd.read_csv(
            f"{pcc_dir}/input_csv/model_switch_PCC.csv", header=0, index_col=0
        )
        coef = pd.read_csv(f"{pcc_dir}/input_csv/coef.csv", header=0, index_col=0)
        self.mmg = MmgModel(principal_particulars, parameter_init, switch, coef)
        #
        self.t = 0.0

    def ode_rhs(self, x, u, w):
        [delta_rudder_p, delta_rudder_s, n_prop, pitch_ang, n_bt] = u
        [Wind_velocity, Wind_Direction] = w
        #
        x = np.array(x)[np.newaxis, :]
        delta_rudder_p = np.array([delta_rudder_p])
        delta_rudder_s = np.array([delta_rudder_s])
        n_prop = np.array([n_prop])
        pitch_ang_prop = np.array([pitch_ang])
        n_bt = np.array([n_bt])
        n_st = np.array([0.0])
        Wind_velocity = np.array([Wind_velocity])
        Wind_Direction = np.array([Wind_Direction])
        #
        dx = self.mmg.hydroForceLs(
            self.t,
            x,
            n_prop,
            pitch_ang_prop,
            delta_rudder_s,
            delta_rudder_p,
            n_bt,
            n_st,
            Wind_velocity,
            Wind_Direction,
            False,
        )
        return dx[0, :]
