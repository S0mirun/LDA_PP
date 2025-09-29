import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
from torchdiffeq import odeint


from .model import Dynamics, DIM_LIST, ACTIVATION


cur_dir = Path(__file__).parents[0]


class TakaokiNNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        traj_list: list[pd.DataFrame],
        t_col: list[str],
        nu_col: list[str],
        z_col: list[str],
        w_col: list[str],
        eta_col: list[str],
        dt: float = 1.0,
        T: int = 100,
        S: int = 10,
        T_fil: int = 100,
        N_fil: int = 2,
    ):
        self.traj_list = traj_list
        self.dt = dt
        self.T = T
        self.S = S
        self.T_fil = T_fil
        self.N_fil = N_fil
        #
        self.preprocess(t_col, nu_col, z_col, w_col, eta_col)

    def preprocess(self, t_col, nu_col, z_col, w_col, eta_col):
        self.t = []
        self.nu = []
        self.z = []
        self.nu_sm = []
        self.nu_dot = []
        self.eta = []
        self.w = []
        self.nu_ = []
        self.z_ = []
        self.nu_dot_ = []
        #
        for traj in self.traj_list:
            # t
            if t_col in traj.columns.to_list():
                traj.set_index(t_col)
            t = traj.index.to_numpy()
            Dt = traj.index.to_series().diff().dropna(how="all", axis=0).to_numpy()
            # nu
            nu = traj[nu_col].to_numpy()
            Dnu = traj[nu_col].diff().dropna(how="all", axis=0).to_numpy()
            # eta
            eta = traj[eta_col].to_numpy()
            # z
            if w_col[0] in z_col:
                z_col_ = z_col.copy()
                U_A = traj[w_col[0]]
                gamma_A = traj[w_col[1]]
                traj["w_x"] = U_A * np.cos(gamma_A)
                traj["w_y"] = U_A * np.sin(gamma_A)
                z_col_.remove(w_col[0])
                z_col_.remove(w_col[1])
                z_col_.append("w_x")
                z_col_.append("w_y")
            else:
                z_col_ = z_col
            z = traj[z_col_].to_numpy()
            w = traj[w_col].to_numpy()
            # nu_dot
            nu_dot = Dnu / Dt[:, np.newaxis]
            # nu_smooth
            nu_smooth = np.empty_like(nu)
            for i, col in enumerate(nu_col):
                if i in [2]:
                    nu_smooth[:, i] = nu[:, i]
                else:
                    nu_smooth[:, i] = savgol_filter(nu[:, i], self.T_fil, self.N_fil)
            #
            Tn = len(t)
            di = int(self.dt / np.mean(Dt) + 0.5)
            #
            for phi in range(0, Tn - self.T * di, self.S * di):
                start_i = phi
                end_i = phi + self.T * di
                self.t.append(t[start_i:end_i:di, np.newaxis])
                self.nu.append(nu[start_i:end_i:di, :])
                self.z.append(z[start_i:end_i:di, :])
                self.nu_sm.append(nu_smooth[start_i:end_i:di, :])
                self.nu_dot.append(nu_dot[start_i:end_i:di, :])
                self.eta.append(eta[start_i:end_i:di, :])
                self.w.append(w[start_i:end_i:di, :])
            self.nu_.append(nu)
            self.z_.append(z)
            self.nu_dot_.append(nu_dot)
        self.t = np.array(self.t)
        self.nu = np.array(self.nu)
        self.z = np.array(self.z)
        self.nu_sm = np.array(self.nu_sm)
        self.nu_dot = np.array(self.nu_dot)
        self.eta = np.array(self.eta)
        self.w = np.array(self.w)
        self.nu_ = np.vstack(self.nu_)
        self.z_ = np.vstack(self.z_)
        self.nu_dot_ = np.vstack(self.nu_dot_)

    def get_nu_stats(self):
        return (self.nu_.mean(axis=0), self.nu_.std(axis=0))

    def get_z_stats(self):
        return (self.z_.mean(axis=0), self.z_.std(axis=0))

    def get_nu_dot_stats(self):
        return (self.nu_dot_.mean(axis=0), self.nu_dot_.std(axis=0))

    def __len__(self):
        return len(self.nu)

    def __getitem__(self, index):
        return (
            self.t[index],
            self.nu[index],
            self.z[index],
            self.nu_sm[index],
            self.nu_dot[index],
            self.eta[index],
            self.w[index],
        )


class TakaokiNNTrain(object):
    def __init__(
        self,
        data_dict,
        nu_stats=None,
        z_stats=None,
        nu_dot_stats=None,
        sigma_obs=[0.01, 0.01, 0.1 * np.pi / 180],
        lr=0.0001,
        betas=(0.9, 0.999),
        l2_lambda=1e-2,
        device="cpu",
        writer=None,
        tag="",
    ):
        self.device = device
        self.data_dict = data_dict
        self.sigma_obs = sigma_obs
        self.writer = writer
        self.tag = tag
        #
        self.model = Dynamics(
            dim_list=DIM_LIST,
            activation=ACTIVATION,
            nu_stats=nu_stats,
            z_stats=z_stats,
            nu_dot_stats=nu_dot_stats,
            device=device,
            tag=tag,
        )
        self.model.save_stats()
        #
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.criteria = nn.GaussianNLLLoss(reduction="sum")
        self.l2_lambda = l2_lambda

    def fitting(self, max_epochs=1e4, max_iterations=1e6):
        epoch = 0
        self.iteration = 0
        best_loss = np.inf
        best_epoch = 0
        while True:
            ### Init ###
            t_epoch_start = time.time()
            epoch += 1
            ### Train ###
            self.model.train()
            train_loss = self.train_epoch(self.data_dict["train_dr"])
            ### Eval ###
            self.model.eval()
            with torch.no_grad():
                valid_loss = self.eval_epoch(self.data_dict["valid_dr"])
            # saves
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                self.model.save_parameter(tag="best")
            self.model.save_parameter(tag="last")
            # write
            if self.writer is not None:
                self.write_board(epoch, train_loss, valid_loss)
            # print
            t_epoch_finish = time.time()
            print(
                f"Epoch {epoch}"
                + f" || Iteration: {self.iteration}"
                + f" || Time: {t_epoch_finish - t_epoch_start:.4f} sec."
                + f" || Loss: (Train) {train_loss:.4f}, (Valid) {valid_loss:.4f}"
            )
            ### End ###
            if self.iteration >= max_iterations:
                break
            if epoch >= max_epochs:
                break
        return best_epoch

    def train_epoch(self, data_loader):
        loss_epoch, count_epoch = 0.0, 0
        for batch_idx, (t, nu, z, nu_smooth, _, _, _) in enumerate(data_loader):
            batch_size = nu.size(0)
            self.iteration += 1
            #
            t = self.model.as_tensor(t[0, :, 0])
            nu = self.model.as_tensor(nu)
            z = self.model.as_tensor(z)
            nu0 = self.model.as_tensor(nu_smooth[:, 0, :])
            # sim
            self.model.set_z(t, z.permute(1, 0, 2))
            mu = (
                odeint(self.model, nu0, t, method="euler")
                .to(self.device)
                .permute(1, 0, 2)
            )
            var = torch.ones_like(nu) * self.model.as_tensor(self.sigma_obs) ** 2
            # loss
            loss_ = self.criteria(mu, nu, var) / batch_size
            reg = self.l2_lambda * self._L2(self.model)
            loss = loss_ + reg
            #
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e6)
            self.optimizer.step()
            #
            print(
                f"\r (Training) Mini Batch: {batch_idx:>4}||Loss: {loss.item():8.2f}",
                end="",
            )
            #
            loss_epoch += loss.item() * batch_size
            count_epoch += batch_size
        loss_epoch /= count_epoch
        print("\n" + "\033[1A", end="")
        return loss_epoch

    def eval_epoch(self, data_loader):
        loss_epoch, count_epoch = 0.0, 0
        for batch_idx, (t, nu, z, nu_smooth, _, _, _) in enumerate(data_loader):
            batch_size = nu.size(0)
            #
            t = self.model.as_tensor(t[0, :, 0])
            nu = self.model.as_tensor(nu)
            z = self.model.as_tensor(z)
            nu0 = self.model.as_tensor(nu_smooth[:, 0, :])
            # sim
            self.model.set_z(t, z.permute(1, 0, 2))
            mu = (
                odeint(self.model, nu0, t, method="euler")
                .to(self.device)
                .permute(1, 0, 2)
            )
            var = torch.ones_like(nu) * self.model.as_tensor(self.sigma_obs) ** 2
            # loss
            loss = self.criteria(mu, nu, var) / batch_size
            loss_epoch += loss.item() * batch_size
            count_epoch += batch_size
        loss_epoch /= count_epoch
        return loss_epoch

    def _L2(self, model):
        L2 = 0.0
        for param in model.parameters():
            L2 += param.norm(2).sum()
        return L2

    def write_board(self, epoch, train_loss, valid_loss):
        self.writer.add_scalar("Train Loss by Eopch", train_loss, epoch)
        self.writer.add_scalar("Valid Loss by Eopch", valid_loss, epoch)
