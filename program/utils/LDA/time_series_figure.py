
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from utils.LDA.visualization import *
from utils.LDA.ship_geometry import ship_shape_poly


class TimeSeries:
    def __init__(
        self, df, dt, label, L=None, B=None,
        color=None, line_style=None, line_width=None,
    ):
        self.df = df
        self.label = label
        self.L = L
        self.B = B
        self.color = color
        self.line_style = line_style
        self.line_width = line_width
        self.dt = dt

def make_traj_fig(
        ts_list,
        ship_plot_step_period, alpha_ship_shape,
        fig_size,
        title=None, legend_flag=False,
):
    #
    set_rcParams()
    fig = plt.figure(figsize = fig_size)
    ax = fig.add_subplot(1, 1, 1)
    if title is not None:
        ax.set_title(title)
    # ax setting
    ax.set_xlabel(r"$y_{\mathrm{E}} ~ \mathrm{[m]}$")
    ax.set_ylabel(r"$x_{\mathrm{E}} ~ \mathrm{[m]}$")
    L_max = max(ts.L for ts in ts_list)
    B_max = max(ts.B for ts in ts_list)
    h_ax_range, v_ax_range =  \
        drawing_range_2D(ts_list, "p_y [m]", "p_x [m]", L_max, B_max, True)
    ax.set_xlim(h_ax_range[0], h_ax_range[1])
    ax.set_ylim(v_ax_range[0], v_ax_range[1])
    ax.set_aspect('equal')
    # plot
    for ts in ts_list:
        plot_traj(
            ax, ts, "p_x [m]", "p_y [m]", "gyro deg [rad]",
            ship_plot_step_period, alpha_ship_shape, ts.L, ts.B
        )
    #
    fig.align_labels()
    fig.tight_layout()
    # legend
    if legend_flag:
        ax.legend()
    
def make_ts_fig(ts_list, fig_size,):
    #
    set_rcParams()
    fig = plt.figure(figsize = fig_size)
    #
    ax_px = fig.add_subplot(2, 3, 1)
    ax_py = fig.add_subplot(2, 3, 2)
    ax_gyro_deg = fig.add_subplot(2, 3, 3)
    ax_speed = fig.add_subplot(2, 3, 4)
    ax_beta = fig.add_subplot(2, 3, 5)
    ax_r = fig.add_subplot(2, 3, 6)
    # time
    [time_min, time_max] = get_time_range(ts_list, "t [s]",)
    # px
    ax_px.set_ylabel("$p_{x}$ [m]")
    v_ax_range = drawing_range(ts_list, "p_x [m]", 0.05)
    ax_px.set_xlim(time_min, time_max)
    ax_px.set_ylim(v_ax_range[0], v_ax_range[1])
    for ts in ts_list:
        plot_ts(
            ax=ax_px, ts=ts,
            df_label_t="t [s]", df_label_v="p_x [m]",
            fig_size=fig_size,
        )
    # py
    ax_py.set_ylabel("$p_{y}$ [m]")
    v_ax_range = drawing_range(ts_list, "p_y [m]", 0.05)
    ax_py.set_xlim(time_min, time_max)
    ax_py.set_ylim(v_ax_range[0], v_ax_range[1])
    for ts in ts_list:
        plot_ts(
            ax=ax_py, ts=ts,
            df_label_t="t [s]", df_label_v="p_y [m]",
            fig_size=fig_size,
        )
    # gyro deg
    ax_gyro_deg.set_ylabel("$gyro_deg$ [deg]")
    v_ax_range = drawing_range(ts_list, "gyro deg [deg]", 0.05)
    ax_gyro_deg.set_xlim(time_min, time_max)
    ax_gyro_deg.set_ylim(v_ax_range[0], v_ax_range[1])
    for ts in ts_list:
        plot_ts(
            ax=ax_gyro_deg, ts=ts,
            df_label_t="t [s]", df_label_v="gyro deg [deg]",
            fig_size=fig_size,
        )
    # speed
    # ax_speed.set_xlabel("$t$ [s]")
    ax_speed.set_ylabel("$U$ [m/s]")
    v_ax_range = drawing_range(ts_list, "U [m/s]", 0.05)
    ax_speed.set_xlim(time_min, time_max)
    ax_speed.set_ylim(v_ax_range[0], v_ax_range[1])
    for ts in ts_list:
        plot_ts(
            ax=ax_speed, ts=ts,
            df_label_t="t [s]", df_label_v="U [m/s]",
            fig_size=fig_size,
        )
    # ax_speed.legend()
    # beta
    # ax_beta.set_xlabel("$t$ [s]")
    ax_beta.set_ylabel(r"$\beta$ [deg]")
    v_ax_range = drawing_range(ts_list, "beta [deg]", 0.05)
    ax_beta.set_xlim(time_min, time_max)
    ax_beta.set_ylim(v_ax_range[0], v_ax_range[1])
    for ts in ts_list:
        plot_ts(
            ax=ax_beta, ts=ts,
            df_label_t="t [s]", df_label_v="beta [deg]",
            fig_size=fig_size,
        )
    # ax_beta.legend()
    fig.align_labels()
    fig.tight_layout()

def make_traj_and_velo_fig(ts_list, ship_plot_step_period, alpha_ship_shape, fig_size,):
    #
    set_rcParams()
    fig = plt.figure(figsize = fig_size)
    #
    ax_traj = fig.add_subplot(1, 2, 1)
    ax_u = fig.add_subplot(3, 2, 2)
    ax_v = fig.add_subplot(3, 2, 4)
    ax_r = fig.add_subplot(3, 2, 6)
    # time
    [time_min, time_max] = get_time_range(ts_list, "t [s]",)
    # traj
    ax_traj.set_xlabel(r"$y_{\mathrm{E}} ~ \mathrm{[m]}$")
    ax_traj.set_ylabel(r"$x_{\mathrm{E}} ~ \mathrm{[m]}$")
    L_max = max(ts.L for ts in ts_list)
    B_max = max(ts.B for ts in ts_list)
    h_ax_range, v_ax_range =  \
        drawing_range_2D(ts_list, "p_y [m]", "p_x [m]", L_max, B_max, True)
    ax_traj.set_xlim(h_ax_range[0], h_ax_range[1])
    ax_traj.set_ylim(v_ax_range[0], v_ax_range[1])
    ax_traj.set_aspect('equal')
    for ts in ts_list:
        plot_traj(
            ax_traj, ts, "p_x [m]", "p_y [m]", "gyro deg [rad]",
            ship_plot_step_period, alpha_ship_shape, ts.L, ts.B
        )
    # u
    # ax_speed.set_xlabel("$t$ [s]")
    ax_u.set_ylabel("$u$ [m/s]")
    v_ax_range = drawing_range(ts_list, "u [m/s]", 0.05)
    ax_u.set_xlim(time_min, time_max)
    ax_u.set_ylim(v_ax_range[0], v_ax_range[1])
    for ts in ts_list:
        plot_ts(
            ax=ax_u, ts=ts,
            df_label_t="t [s]", df_label_v="u [m/s]",
            fig_size=fig_size,
        )
    # ax_u.legend()
    # vm
    # ax_beta.set_xlabel("$t$ [s]")
    ax_v.set_ylabel("$v$ [m/s]")
    v_ax_range = drawing_range(ts_list, "vm [m/s]", 0.05)
    ax_v.set_xlim(time_min, time_max)
    ax_v.set_ylim(v_ax_range[0], v_ax_range[1])
    for ts in ts_list:
        plot_ts(
            ax=ax_v, ts=ts,
            df_label_t="t [s]", df_label_v="vm [m/s]",
            fig_size=fig_size,
        )
    # ax_v.legend()
    #
    fig.align_labels()
    fig.tight_layout()

def plot_traj(
        ax, ts,
        x_column_label, y_column_label, gyro_deg_column_label,
        ship_plot_step_period, alpha_ship_shape, Lpp, B,
):
    #
    ax.plot(
        ts.df[y_column_label],
        ts.df[x_column_label],
        color = ts.color,
        linestyle = "dashed",
        linewidth = 0.7,
    )
    #
    for j in range(len(ts.df)):
        if ( j % ship_plot_step_period == 0 ) or ( j == len(ts.df) - 1 ):
            #
            p = ts.df.iloc[
                j,
                [
                    ts.df.columns.get_loc(x_column_label),
                    ts.df.columns.get_loc(y_column_label),
                    ts.df.columns.get_loc(gyro_deg_column_label),
                ]
            ]
            if j == 0:
                ax.add_patch(
                    plt.Polygon(
                        ship_shape_poly(p, Lpp, B, scale=1.0,),
                        fill=True, alpha=alpha_ship_shape,
                        color=ts.color, linewidth=0.3,
                        label = ts.label
                    )
                )
            else:
                ax.add_patch(
                    plt.Polygon(
                        ship_shape_poly(p, Lpp, B, scale=1.0,),
                        fill=True, alpha=alpha_ship_shape,
                        color=ts.color, linewidth=0.3,
                    )
                )
            ax.add_patch(
                plt.Polygon(
                    ship_shape_poly(p, Lpp, B, scale=1.0,),
                    fill=False, color=ts.color, linewidth=0.3,
                )
            )

def plot_ts(
        ax, ts,
        df_label_t, df_label_v,
        fig_size, label=None,):
    ax.plot(
        ts.df[df_label_t],
        ts.df[df_label_v],
        color = ts.color if ts.color is not None  \
            else Colors.black,
        linestyle = ts.line_style if ts.line_style is not None  \
            else "solid",
        linewidth = ts.line_width if ts.line_width is not None  \
            else 0.2*max(fig_size),
        label = label,
    )

def plot_interval(ax, lower_lim, upper_lim, color,):
    # mask
    ax.add_patch(
        plt.Polygon(
            (
                [-1e10, lower_lim],
                [1e10, lower_lim],
                [-1e10, upper_lim],
                [1e10, upper_lim],
            ),
            fill=True, color=color, alpha=0.05,
            linewidth=1.0, ec=None,
        )
    )
    # limit line
    limit_line_style = (0, (5, 2))
    ax.axhline(
        y = lower_lim,
        color = color,
        linewidth = 1.0,
        ls = limit_line_style,
    )
    ax.axhline(
        y = upper_lim,
        color = color,
        linewidth = 1.0,
        ls = limit_line_style,
    )

def drawing_range(ts_list, label_item, margin_rate):
    #
    data_concatenated = pd.concat(pd.concat([ts.df[label_item],]) for ts in ts_list)
    data_min, data_max = min(data_concatenated), max(data_concatenated)
    width_original = data_max - data_min
    margin = margin_rate * width_original / (1.0 - 2.0 * margin_rate)
    v_ax_min = data_min - margin
    v_ax_max = data_max + margin
    #
    v_ax_range = [v_ax_min, v_ax_max]
    return v_ax_range

def get_time_range(ts_list, label_time):
    mins = []
    maxs = []
    for ts in ts_list:
        mins.append(min(ts.df[label_time]))
        maxs.append(max(ts.df[label_time]))
    time_min = min(mins)
    time_max = max(maxs)
    range = [time_min, time_max]
    return range

def drawing_range_2D(ts_list, label_h, label_v, L, B, square_flag):
    data_h_concatenated = pd.concat(pd.concat([ts.df[label_h],]) for ts in ts_list)
    data_v_concatenated = pd.concat(pd.concat([ts.df[label_v],]) for ts in ts_list)
    ship_shape_margin = np.sqrt(np.square(L) + np.square(B))
    h_ax_range, v_ax_range = calc_fig_range_2D(
        data_h_concatenated, data_v_concatenated,
        ship_shape_margin, square_flag
    )
    return h_ax_range, v_ax_range
