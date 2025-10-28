import os

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from test import df_to_xy, prepare, draw_base_map, draw_bui


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
RAW_DATAS_DIR = f"{DIR}/../../raw_datas"
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
bui_path = f"{RAW_DATAS_DIR}/内航船-要素/水深-Yokkaichi/水深-ブイ等/伊勢湾ブイ関係.xlsx"
#
BUI_SHEET = [
    # '伊良湖-シーバース東',
    '伊良湖-シーバース西側'
]

def read_bui():
    raw_df = pd.read_excel(
        bui_path,
        sheet_name=None
    )
    for name in BUI_SHEET:
        df = raw_df[name].iloc[[6,7,8,9,10,11,15], :]
        print(df)
        df["latitude [deg]"] = df["Unnamed: 5"]
        df["longitude [deg]"] = df["Unnamed: 6"]
        conv_df = df_to_xy(df)
        df["p_x [m]"] = conv_df[:, 0]
        df["p_y [m]"] = conv_df[:, 1]
    return df


def _bernstein_matrix(n: int, t: np.ndarray) -> np.ndarray:
    """
    n 次ベジェのバーンスタイン基底行列 B(t) を返す。
    形状: (len(t), n+1)
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)             # (M,1)
    k = np.arange(n + 1, dtype=int)                           # (n+1,)
    coeff = np.array([math.comb(n, i) for i in k], float)      # (n+1,)
    # ブロードキャストで (M, n+1) を作る
    T = t ** k                                                # (M, n+1)
    U = (1.0 - t) ** (n - k)                                  # (M, n+1)
    return coeff * T * U                                       # (M, n+1)


def bezier_curve_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    num: int = 400,
) -> np.ndarray:
    """
    DataFrame の制御点（x_col, y_col）から単一のベジェ曲線をサンプリングして返す。
    戻り値: 形状 (num, 2) の numpy 配列（曲線上の点列）
    """
    P = df[[x_col, y_col]].astype(float).to_numpy()  # (N,2) float に強制変換
    if P.shape[0] < 2:
        raise ValueError("制御点は最低2点必要です。")
    n = P.shape[0] - 1
    t = np.linspace(0.0, 1.0, num)
    B = _bernstein_matrix(n, t)                      # (num, n+1)
    C = B @ P                                        # (num, 2)
    return C


def plot_bezier_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    num: int = 400,
    show_ctrl: bool = True,
    ax=None,
):
    """
    単一のベジェ曲線をプロット。
    """
    C = bezier_curve_from_df(df, x_col=x_col, y_col=y_col, num=num)
    P = df[[x_col, y_col]].astype(float).to_numpy()

    if show_ctrl:
        ax.plot(P[:, 0], P[:, 1], "o--", linewidth=1, markersize=4, alpha=0.8, label="control polygon")

    ax.plot(C[:, 0], C[:, 1], linewidth=2, label="bezier")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Bézier curve")


def plot_piecewise_cubic_bezier_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    num_per_seg: int = 200,
    step: int = 3,
    show_ctrl: bool = True,
    ax=None,
):
    """
    区分的（cubic）ベジェを描く。
    4点ずつを1セグメント（3次）として描画し、セグメント間の開始点を step で進める。
    - step=3 だと 0..3, 3..6, 6..9,... と C0 連結（端点共有）
    - step=1 だと 0..3, 1..4, 2..5,... とスライド（見た目は滑らかだが端点共有ではない）
    """
    P = df[[x_col, y_col]].astype(float).to_numpy()
    if P.shape[0] < 4:
        raise ValueError("区分的ベジェには最低4点が必要です。")

    t = np.linspace(0.0, 1.0, num_per_seg)
    B3 = _bernstein_matrix(3, t)  # (M,4)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    if show_ctrl:
        ax.plot(P[:, 0], P[:, 1], "o--", linewidth=1, markersize=4, alpha=0.6, label="control polygon")

    first = True
    for i in range(0, len(P) - 3, step):
        Q = P[i:i + 4]                  # (4,2)
        C = B3 @ Q                      # (M,2)
        lbl = "cubic bezier segments" if first else None
        ax.plot(C[:, 0], C[:, 1], linewidth=2, label=lbl)
        first = False

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Piecewise cubic Bézier")
    if show_ctrl:
        ax.legend()
    return ax



if __name__ == "__main__":
    top_df, coast_df = prepare()
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_base_map(ax, top_df, coast_df,
                    apply_port_extra=False, apply_coast_extra=True, x_const=-6000.0)
    draw_bui(ax)
    # 単一ベジェ
    df = read_bui()
    plot_bezier_from_df(df, x_col="p_x [m]", y_col="p_y [m]"
                            , num=400, show_ctrl=True, ax=ax)

    plt.show()
