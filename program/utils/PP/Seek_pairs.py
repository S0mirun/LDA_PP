import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def pair_points_min_distance_df(
    df,
    x_col="x",
    y_col="y",
    colour_col="COLOUR",
    colour_a="3",
    colour_b="4",
    max_distance=1000.0,
):
    """
    COLOUR==colour_a と COLOUR==colour_b の点だけを対象に、
    d <= max_distance の組だけを許してペアを作る。

    最適化の優先順位:
    1. 成立ペア数を最大化
    2. その条件下で距離総和を最小化
    """

    df = df.copy()
    df[colour_col] = df[colour_col].astype(str)
    colour_a = str(colour_a)
    colour_b = str(colour_b)

    df_a = df.loc[df[colour_col] == colour_a].copy()
    df_b = df.loc[df[colour_col] == colour_b].copy()

    idx_a = df_a.index.to_list()
    idx_b = df_b.index.to_list()

    n_a = len(df_a)
    n_b = len(df_b)

    rows = []
    total_distance = 0.0

    if n_a == 0 and n_b == 0:
        return pd.DataFrame(), 0.0

    if n_a == 0:
        for ib in range(n_b):
            rows.append({
                "type": "leftover",
                f"idx{colour_a}": None,
                f"idx{colour_b}": idx_b[ib],
                f"x{colour_a}": np.nan,
                f"y{colour_a}": np.nan,
                f"x{colour_b}": df_b.iloc[ib][x_col],
                f"y{colour_b}": df_b.iloc[ib][y_col],
                "name1": None,
                "name2": df_b.iloc[ib]["NOBJNAM"] if "NOBJNAM" in df_b.columns else None,
                "distance": np.nan,
            })
        return pd.DataFrame(rows), 0.0

    if n_b == 0:
        for ia in range(n_a):
            rows.append({
                "type": "leftover",
                f"idx{colour_a}": idx_a[ia],
                f"idx{colour_b}": None,
                f"x{colour_a}": df_a.iloc[ia][x_col],
                f"y{colour_a}": df_a.iloc[ia][y_col],
                f"x{colour_b}": np.nan,
                f"y{colour_b}": np.nan,
                "name1": df_a.iloc[ia]["NOBJNAM"] if "NOBJNAM" in df_a.columns else None,
                "name2": None,
                "distance": np.nan,
            })
        return pd.DataFrame(rows), 0.0

    coords_a = df_a[[x_col, y_col]].to_numpy(dtype=float)
    coords_b = df_b[[x_col, y_col]].to_numpy(dtype=float)

    dist_matrix = np.linalg.norm(
        coords_a[:, None, :] - coords_b[None, :, :],
        axis=2
    )

    # leftover の罰則
    # これを正にすると、できるだけ leftover を減らす
    penalty = float(max_distance) + 1.0

    # 禁止辺の十分大きいコスト
    forbidden = 3.0 * penalty

    # 正方行列に拡張
    # [ real A      | dummy for B leftover ]
    # [ dummy for A | dummy-dummy          ]
    size = n_a + n_b
    cost = np.full((size, size), forbidden, dtype=float)

    # 左上: 実点どうし
    for ia in range(n_a):
        for ib in range(n_b):
            d = dist_matrix[ia, ib]
            if d <= max_distance:
                cost[ia, ib] = d
            else:
                cost[ia, ib] = forbidden

    # 右上: A を leftover にする
    for ia in range(n_a):
        cost[ia, n_b + ia] = penalty

    # 左下: B を leftover にする
    for ib in range(n_b):
        cost[n_a + ib, ib] = penalty

    # 右下: dummy-dummy
    # ここは 0 でよい
    for ib in range(n_b):
        for ia in range(n_a):
            cost[n_a + ib, n_b + ia] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_a = set()
    matched_b = set()

    for r, c in zip(row_ind, col_ind):
        # real A - real B の成立ペア
        if r < n_a and c < n_b:
            d = dist_matrix[r, c]
            if d <= max_distance:
                matched_a.add(r)
                matched_b.add(c)
                total_distance += float(d)

                rows.append({
                    "type": "pair",
                    f"idx{colour_a}": idx_a[r],
                    f"idx{colour_b}": idx_b[c],
                    f"x{colour_a}": df_a.iloc[r][x_col],
                    f"y{colour_a}": df_a.iloc[r][y_col],
                    f"x{colour_b}": df_b.iloc[c][x_col],
                    f"y{colour_b}": df_b.iloc[c][y_col],
                    "name1": df_a.iloc[r]["NOBJNAM"] if "NOBJNAM" in df_a.columns else None,
                    "name2": df_b.iloc[c]["NOBJNAM"] if "NOBJNAM" in df_b.columns else None,
                    "distance": float(d),
                })

    # unmatched A
    for ia in range(n_a):
        if ia not in matched_a:
            rows.append({
                "type": "leftover",
                f"idx{colour_a}": idx_a[ia],
                f"idx{colour_b}": None,
                f"x{colour_a}": df_a.iloc[ia][x_col],
                f"y{colour_a}": df_a.iloc[ia][y_col],
                f"x{colour_b}": np.nan,
                f"y{colour_b}": np.nan,
                "name1": df_a.iloc[ia]["NOBJNAM"] if "NOBJNAM" in df_a.columns else None,
                "name2": None,
                "distance": np.nan,
            })

    # unmatched B
    for ib in range(n_b):
        if ib not in matched_b:
            rows.append({
                "type": "leftover",
                f"idx{colour_a}": None,
                f"idx{colour_b}": idx_b[ib],
                f"x{colour_a}": np.nan,
                f"y{colour_a}": np.nan,
                f"x{colour_b}": df_b.iloc[ib][x_col],
                f"y{colour_b}": df_b.iloc[ib][y_col],
                "name1": None,
                "name2": df_b.iloc[ib]["NOBJNAM"] if "NOBJNAM" in df_b.columns else None,
                "distance": np.nan,
            })

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df["type_order"] = result_df["type"].map({"pair": 0, "leftover": 1})
        result_df = result_df.sort_values(
            by=["type_order", "distance"],
            ascending=[True, True],
            na_position="last"
        ).drop(columns="type_order").reset_index(drop=True)

    return result_df, total_distance

    
if __name__ == '__main__':
    import scipy
    print("SciPy:", scipy.__version__)

    from scipy.optimize import linear_sum_assignment
    print("linear_sum_assignment: OK")