import os
import numpy as np


def Rmatrix(psi):
    R = np.array([
        [np.cos(psi), -np.sin(psi), 0.0],
        [np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return R


class VecTwinPositioningSystem(object):
    def __init__(self, n=10, init_t=0.0):
        # parameter
        rho = 998
        x_fR = -1.5694
        x_B = 1.3564
        D = 0.1001
        K_T = 0.16684
        C_B = rho*(D**4)*K_T
        self.delta_hover_p = -78.60
        self.delta_hover_s = 73.38
        self.V = np.array([
            [0.0236, -0.0296, 0.0],
            [0.0192, 0.0123, 0.0],
            [0.0, 0.0, C_B],
        ])
        self.T = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, x_fR, x_B],
        ])
        self.V_inv = np.linalg.inv(self.V)
        self.T_inv = np.linalg.inv(self.T)
        self.Z_inv = self.V_inv @ self.T_inv
        self.K_P = np.array([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0]
        ])
        self.K_I = np.array([
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.001]
        ])
        self.K_D = np.array([
            [25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
            [0.0, 0.0, 30.0]
        ])
        self.f_req_min = np.array([-1.5, -1.0, -1.7])
        self.f_req_max = np.array([0.8, 1.0, 1.5])
        # initialize
        self.n = n
        self.pre_t = init_t
        self.int_e = np.zeros(3)
        #
        self.gamma = 1.0

    def get_action(self, t, eta, nu, eta0):
        psi = eta[2]
        # calc
        e0 = eta0 - eta
        e = Rmatrix(psi).T @ e0
        #
        delta_t = t - self.pre_t
        self.int_e = self.gamma * self.int_e + delta_t * e
        self.pre_t = t
        #
        f_req = np.clip(
            self.K_P @ e + self.K_I @ self.int_e - self.K_D @ nu,
            self.f_req_min,
            self.f_req_max
        )
        u_tilde = self.Z_inv @ f_req
        delta_p_tilde, delta_s_tilde, n_B_tilde = u_tilde
        u = np.array([
            (delta_p_tilde+self.delta_hover_p)*np.pi/180,
            (delta_s_tilde+self.delta_hover_s)*np.pi/180,
            self.n,
            np.sign(n_B_tilde)*np.sqrt(np.abs(n_B_tilde))
        ])
        return u
