import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ....utils.apparent import polarTrue2xyApparent


cur_dir = Path(__file__).parents[0]
parent_dir = Path(__file__).parents[1]


DIM_LIST = [8, 256, 256, 256, 256, 3]
ACTIVATION = "tanh"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


def try_read_csv(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = None
        print(f"{path} is not found.")
    return df


class NNModel(object):
    def __init__(self, tag=""):
        self.model = Dynamics(
            dim_list=DIM_LIST,
            activation=ACTIVATION,
            device=device,
            tag=tag,
        )
        self.model.read_stats()
        self.model.load_parameter(tag="best", device=device)
        self.n_prop = 10.0

    def ode_rhs(self, x, u, w):
        x_dot = np.empty_like(x)
        #
        U_T, gamma_T = w[0], w[1]
        u_velo, vm_velo, psi = x[1], x[3], x[4]
        wA_x, wA_y = polarTrue2xyApparent(U_T, gamma_T, u_velo, vm_velo, psi)
        #
        nu = x[[1, 3, 5]]
        z = np.array([u[0], u[1], u[3], wA_x, wA_y])
        nu_ = self.model.as_tensor(nu).unsqueeze(0)
        z_ = self.model.as_tensor(z).unsqueeze(0)
        nu_dot_ = self.model.ode_rhs(nu_, z_)[0, :]
        nu_dot = self.model.as_array(nu_dot_)
        #
        x_dot = np.array(
            [
                x[1] * np.cos(x[4]) - x[3] * np.sin(x[4]),
                nu_dot[0],
                x[1] * np.sin(x[4]) + x[3] * np.cos(x[4]),
                nu_dot[1],
                x[5],
                nu_dot[2],
            ]
        )
        return x_dot


class Dynamics(nn.Module):
    def __init__(
        self,
        dim_list,
        activation="tanh",
        nu_stats=None,
        z_stats=None,
        nu_dot_stats=None,
        device="cpu",
        tag="",
    ):
        super().__init__()
        self.device = device
        self.tag = tag
        self.input_dim = dim_list[0]
        self.output_dim = dim_list[-1]
        # standization
        if nu_stats is not None:
            self.mu_nu = torch.tensor(nu_stats[0]).float().to(device)
            self.sigma_nu = torch.tensor(nu_stats[1]).float().to(device)
        if z_stats is not None:
            self.mu_z = torch.tensor(z_stats[0]).float().to(device)
            self.sigma_z = torch.tensor(z_stats[1]).float().to(device)
        if nu_dot_stats is not None:
            self.mu_nu_dot = torch.tensor(nu_dot_stats[0]).float().to(device)
            self.sigma_nu_dot = torch.tensor(nu_dot_stats[1]).float().to(device)
        # layers
        self.layers = nn.ModuleList(
            [nn.Linear(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)]
        )
        if activation == "tanh":
            self.act = torch.tanh
        else:
            self.act = torch.relu
        #
        self.to(device)

    def forward(self, t, nu):
        z = self.interp(t, self.t, self.z)
        nu_dot = self.ode_rhs(nu, z)
        return nu_dot

    def set_z(self, t, z):
        self.t = t
        self.z = z

    def interp(self, t_interp, t, y):
        lt, ut = t[:-1], t[1:]
        ly, uy = y[:-1], y[1:]
        if t[-1] <= t_interp:
            y_interp = y[-1]
        else:
            i = torch.where((lt <= t_interp) * (t_interp < ut))[0]
            if len(i) == 0:
                print("!!! Warning : interpolation !!!")
            i = i[0]
            rate_t = (t_interp - lt[i]) / (ut[i] - lt[i])
            y_interp = rate_t * (uy[i] - ly[i]) + ly[i]
        return y_interp

    def ode_rhs(self, nu, z):
        # input
        if hasattr(self, "mu_nu"):
            nu = (nu - self.mu_nu) / self.sigma_nu
        if hasattr(self, "mu_z"):
            z = (z - self.mu_z) / self.sigma_z
        x = torch.cat([nu, z], dim=1)
        # network
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        # output
        if hasattr(self, "mu_nu_dot"):
            nu_dot = x * self.sigma_nu_dot + self.mu_nu_dot
        else:
            nu_dot = x
        return nu_dot

    def as_tensor(self, x):
        x = torch.tensor(x).float().to(self.device)
        return x

    def as_array(self, x):
        return x.detach().cpu().numpy()

    def save_stats(self):
        path = f"{parent_dir}/params/{self.tag}/test.txt"
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        #
        if hasattr(self, "mu_nu"):
            mu_nu = self.as_array(self.mu_nu)
            sigma_nu = self.as_array(self.sigma_nu)
            nu_stats = np.concatenate(
                [mu_nu[:, np.newaxis], sigma_nu[:, np.newaxis]], axis=1
            )
            nu_stats_df = pd.DataFrame(nu_stats, columns=["mu", "sigma"])
            nu_stats_df.to_csv(f"{parent_dir}/params/{self.tag}/nu_stats.csv")
        #
        if hasattr(self, "mu_z"):
            mu_z = self.as_array(self.mu_z)
            sigma_z = self.as_array(self.sigma_z)
            z_stats = np.concatenate(
                [mu_z[:, np.newaxis], sigma_z[:, np.newaxis]], axis=1
            )
            z_stats_df = pd.DataFrame(z_stats, columns=["mu", "sigma"])
            z_stats_df.to_csv(f"{parent_dir}/params/{self.tag}/z_stats.csv")
        #
        if hasattr(self, "mu_nu_dot"):
            mu_nu_dot = self.as_array(self.mu_nu_dot)
            sigma_nu_dot = self.as_array(self.sigma_nu_dot)
            nu_dot_stats = np.concatenate(
                [mu_nu_dot[:, np.newaxis], sigma_nu_dot[:, np.newaxis]], axis=1
            )
            nu_dot_stats_df = pd.DataFrame(nu_dot_stats, columns=["mu", "sigma"])
            nu_dot_stats_df.to_csv(f"{parent_dir}/params/{self.tag}/nu_dot_stats.csv")

    def read_stats(self):
        nu_stats_df = try_read_csv(f"{parent_dir}/params/{self.tag}/nu_stats.csv")
        if nu_stats_df is not None:
            self.mu_nu = self.as_tensor(nu_stats_df["mu"])
            self.sigma_nu = self.as_tensor(nu_stats_df["sigma"])
        #
        z_stats_df = try_read_csv(f"{parent_dir}/params/{self.tag}/z_stats.csv")
        if z_stats_df is not None:
            self.mu_z = self.as_tensor(z_stats_df["mu"])
            self.sigma_z = self.as_tensor(z_stats_df["sigma"])
        #
        nu_dot_stats_df = try_read_csv(
            f"{parent_dir}/params/{self.tag}/nu_dot_stats.csv"
        )
        if nu_dot_stats_df is not None:
            self.mu_nu_dot = self.as_tensor(nu_dot_stats_df["mu"])
            self.sigma_nu_dot = self.as_tensor(nu_dot_stats_df["sigma"])

    def save_parameter(self, tag="best"):
        path = f"{parent_dir}/params/{self.tag}/{tag}.pth"
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load_parameter(self, tag="best", device="cpu"):
        path = f"{parent_dir}/params/{self.tag}/{tag}.pth"
        self.load_state_dict(torch.load(path, torch.device(device)))
