
import os
import pandas as pd

from utils.time_series_figure import  \
    TimeSeries, make_traj_fig, make_ts_fig
from utils.visualization import Colors, save_fig
from utils.read_ts_info_table import read_ship_shape


#
DIM_VEC = 2
N_CODE = 25
#
PERIOD = 1
L_DOC = 100
DELTA_TS_SHIFT = 10
NAME = ""
#
IS_DRAW_TRAJ = True
IS_DRAW_STATE = True


def main():
    #
    vq_id = f"dim_vec_{str(DIM_VEC)}_n_code_{str(N_CODE)}"
    sgmnt_id = f"PERIOD_{PERIOD}_L_DOC_{L_DOC}_DELTA_{DELTA_TS_SHIFT}"
    target_dir = f"./outputs/{vq_id}/sgmntd/{sgmnt_id}/"
    #
    df_path = f"{target_dir}csv/{NAME}.csv"
    df = pd.read_csv(df_path, index_col=0)
    #
    No_str = NAME[:3]
    No = int(No_str)
    L, B = read_ship_shape(No)
    ts = TimeSeries(
        df=df,
        dt=PERIOD,
        label=None,
        L=L,
        B=B,
        color=Colors.black,
        line_style="solid",
        line_width=0.1,
    )
    if IS_DRAW_TRAJ:
        make_traj_fig(
            ts_list=[ts],
            ship_plot_step_period=int(L_DOC/10), alpha_ship_shape=0.1,
            fig_size=(5, 5), legend_flag=False,
        )
        save_fig(f"{target_dir}/fig/traj/", f"{NAME}_traj")
    if IS_DRAW_STATE:
        make_ts_fig(ts_list=[ts], fig_size=(8, 5,))
        save_fig(f"{target_dir}/fig/state/", f"{NAME}_state")


if __name__ == "__main__":
    main()
    #
    print("\nDone\n")
