# -*- coding: utf-8 -*-
"""
pair_points_min_distance_df の max_distance を変化させたときに
マッチング結果がどう変わるかを、全ての港について一括で調べるスクリプト。

出力（各港ごとに ./outputs/analyze_max_distance/ 以下に保存）:
  - {port}_summary.csv   : max_distance ごとのペア数・残り数・距離の集計
  - {port}_pairing.csv   : max_distance ごとにどの点同士が組まれたか
  - {port}_plot.png      : ペア数・平均距離の推移グラフ
  - max_distance_stable_range.csv
        : 各港について、max_distance=1000 のときのペアの組み合わせと
          一致し続ける max_distance の範囲（下限・上限）
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.PP.Seek_pairs import pair_points_min_distance_df
from utils.PP.dictionary_of_port import dictionary
from path_planning.PathPlanning2 import convert

dirname = os.path.splitext(os.path.basename(__file__))[0]
savedir = os.path.join("./outputs", dirname)
os.makedirs(savedir, exist_ok=True)

X_COL = "x [m]"
Y_COL = "y [m]"
COLOUR_COL = "COLOUR"
COLOUR_A = "3"
COLOUR_B = "4"

MAX_DISTANCE_VALUES = np.arange(200, 3200, 200)
REFERENCE_MAX_DISTANCE = 1000


def load_port_df(port):
    df_buoy = pd.read_csv(f"./outputs/data/buoy/{port['name']}.csv")
    port_csv = f"./raw_datas/tmp/coordinates_of_port/_{port['name']}.csv"
    df_buoy["x [m]"], df_buoy["y [m]"] = convert(df_buoy, port_csv, "latitude", "longitude")
    return df_buoy


def run_sweep(df_buoy):
    summary_records = []
    pairing_records = []
    pairing_sets = {}

    for max_distance in MAX_DISTANCE_VALUES:
        result_df, total_distance = pair_points_min_distance_df(
            df_buoy,
            x_col=X_COL,
            y_col=Y_COL,
            colour_col=COLOUR_COL,
            colour_a=COLOUR_A,
            colour_b=COLOUR_B,
            max_distance=max_distance,
        )

        is_pair = result_df["type"] == "pair"
        n_pairs = int(is_pair.sum())
        n_leftover = int((result_df["type"] == "leftover").sum())
        mean_distance = result_df.loc[is_pair, "distance"].mean()

        summary_records.append({
            "max_distance": max_distance,
            "n_pairs": n_pairs,
            "n_leftover": n_leftover,
            "total_distance": total_distance,
            "mean_pair_distance": mean_distance,
        })

        pairs_now = set()
        for _, row in result_df.loc[is_pair].iterrows():
            idx_a, idx_b = row[f"idx{COLOUR_A}"], row[f"idx{COLOUR_B}"]
            pairs_now.add((idx_a, idx_b))
            pairing_records.append({
                "max_distance": max_distance,
                f"idx{COLOUR_A}": idx_a,
                f"idx{COLOUR_B}": idx_b,
                "distance": row["distance"],
            })
        pairing_sets[max_distance] = pairs_now

    return pd.DataFrame(summary_records), pd.DataFrame(pairing_records), pairing_sets


def stable_range_around_reference(pairing_sets, reference_max_distance):
    values = sorted(pairing_sets.keys())
    if reference_max_distance not in pairing_sets:
        return np.nan, np.nan

    ref_idx = values.index(reference_max_distance)
    ref_set = pairing_sets[reference_max_distance]

    lo = ref_idx
    while lo - 1 >= 0 and pairing_sets[values[lo - 1]] == ref_set:
        lo -= 1

    hi = ref_idx
    while hi + 1 < len(values) and pairing_sets[values[hi + 1]] == ref_set:
        hi += 1

    return values[lo], values[hi]


ports = dictionary()
port_items = ports.items() if hasattr(ports, "items") else enumerate(ports)

stable_range_records = []

for port_number, port in port_items:
    df_buoy = load_port_df(port)

    if COLOUR_COL not in df_buoy.columns:
        print(f"[{port['name']}] {COLOUR_COL} 列が無いためスキップします。")
        continue

    summary_df, pairing_df, pairing_sets = run_sweep(df_buoy)

    min_stable, max_stable = stable_range_around_reference(pairing_sets, REFERENCE_MAX_DISTANCE)
    stable_range_records.append({
        "port": port["name"],
        "min_max_distance": min_stable,
        "max_max_distance": max_stable,
    })

    summary_df.to_csv(os.path.join(savedir, f"{port['name']}_summary.csv"), index=False)
    pairing_df.to_csv(os.path.join(savedir, f"{port['name']}_pairing.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(summary_df["max_distance"], summary_df["n_pairs"], marker="o", label="pairs")
    axes[0].plot(summary_df["max_distance"], summary_df["n_leftover"], marker="o", label="leftover")
    axes[0].set_xlabel("max_distance")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{port['name']}: matching count vs max_distance")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(summary_df["max_distance"], summary_df["mean_pair_distance"], marker="o", color="tab:orange")
    axes[1].set_xlabel("max_distance")
    axes[1].set_ylabel("mean pair distance")
    axes[1].set_title(f"{port['name']}: mean pair distance vs max_distance")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"{port['name']}_plot.png"), dpi=150)
    plt.close(fig)

    print(f"[{port['name']}] stable range around max_distance={REFERENCE_MAX_DISTANCE}: "
          f"{min_stable} - {max_stable}")

stable_range_df = pd.DataFrame(stable_range_records)
stable_range_df.to_csv(os.path.join(savedir, "max_distance_stable_range.csv"), index=False)
print(stable_range_df.to_string(index=False))
