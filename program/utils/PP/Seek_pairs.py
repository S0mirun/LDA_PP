import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def pair_points_min_distance_df(
    df,
    x_col="x",
    y_col="y",
    colour_col="COLOUR",
    colour_a=3,
    colour_b=4,
):
    """
    COLOUR==colour_a の点と COLOUR==colour_b の点のみを対象に、
    距離総和が最小になるようにペアリングする。
    """
    df_a = df.loc[df[colour_col] == colour_a].copy()
    df_b = df.loc[df[colour_col] == colour_b].copy()

    idx_a = df_a.index.to_list()
    idx_b = df_b.index.to_list()

    coords_a = df_a[[x_col, y_col]].to_numpy(dtype=float)
    coords_b = df_b[[x_col, y_col]].to_numpy(dtype=float)

    rows = []
    total_distance = 0.0

    # 両方空
    if len(df_a) == 0 and len(df_b) == 0:
        return pd.DataFrame(), 0.0

    # colour_a 側が空
    if len(df_a) == 0:
        for i, idx in enumerate(idx_b):
            row = {
                "type": "leftover",
                f"idx{colour_a}": None,
                f"idx{colour_b}": idx,
                f"x{colour_a}": np.nan,
                f"y{colour_a}": np.nan,
                f"x{colour_b}": df_b.iloc[i][x_col],
                f"y{colour_b}": df_b.iloc[i][y_col],
                "distance": np.nan,
            }
            rows.append(row)
        return pd.DataFrame(rows), 0.0

    # colour_b 側が空
    if len(df_b) == 0:
        for i, idx in enumerate(idx_a):
            row = {
                "type": "leftover",
                f"idx{colour_a}": idx,
                f"idx{colour_b}": None,
                f"x{colour_a}": df_a.iloc[i][x_col],
                f"y{colour_a}": df_a.iloc[i][y_col],
                f"x{colour_b}": np.nan,
                f"y{colour_b}": np.nan,
                "distance": np.nan,
            }
            rows.append(row)
        return pd.DataFrame(rows), 0.0

    # 距離行列
    dist_matrix = np.linalg.norm(
        coords_a[:, None, :] - coords_b[None, :, :],
        axis=2
    )

    # 最小コスト割当
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matched_a = set()
    matched_b = set()

    # ペア
    for ia, ib in zip(row_ind, col_ind):
        d = float(dist_matrix[ia, ib])
        total_distance += d
        matched_a.add(ia)
        matched_b.add(ib)

        row = {
            "type": "pair",
            f"idx{colour_a}": idx_a[ia],
            f"idx{colour_b}": idx_b[ib],
            f"x{colour_a}": df_a.iloc[ia][x_col],
            f"y{colour_a}": df_a.iloc[ia][y_col],
            f"x{colour_b}": df_b.iloc[ib][x_col],
            f"y{colour_b}": df_b.iloc[ib][y_col],
            "distance": d,
        }
        rows.append(row)

    # colour_a 側 leftover
    for ia in range(len(df_a)):
        if ia not in matched_a:
            row = {
                "type": "leftover",
                f"idx{colour_a}": idx_a[ia],
                f"idx{colour_b}": None,
                f"x{colour_a}": df_a.iloc[ia][x_col],
                f"y{colour_a}": df_a.iloc[ia][y_col],
                f"x{colour_b}": np.nan,
                f"y{colour_b}": np.nan,
                "distance": np.nan,
            }
            rows.append(row)

    # colour_b 側 leftover
    for ib in range(len(df_b)):
        if ib not in matched_b:
            row = {
                "type": "leftover",
                f"idx{colour_a}": None,
                f"idx{colour_b}": idx_b[ib],
                f"x{colour_a}": np.nan,
                f"y{colour_a}": np.nan,
                f"x{colour_b}": df_b.iloc[ib][x_col],
                f"y{colour_b}": df_b.iloc[ib][y_col],
                "distance": np.nan,
            }
            rows.append(row)

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df = result_df.sort_values(
            by=["type", "distance"],
            ascending=[True, True],
            na_position="last"
        ).reset_index(drop=True)

    return result_df, total_distance