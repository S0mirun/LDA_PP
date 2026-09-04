# -*- coding: utf-8 -*-
"""
異色標識(COLOUR=3 と 4)どうしの最近傍距離を全港について集計し、
100mごとのビンで分布（件数・割合・累積割合）を求めるスクリプト。

出力（./outputs/nearest_neighbor_distance/ 以下）:
  - nearest_distances.csv       : 港・標識ごとの最近傍異色標識距離（生データ）
  - nearest_distance_histogram.csv : 100mビンごとの件数・割合・累積割合（全港合算）
  - nearest_distance_histogram.png : 上記のヒストグラム
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

BIN_WIDTH = 100  # [m]


def load_port_df(port):
    df_buoy = pd.read_csv(f"./outputs/data/buoy/{port['name']}.csv")
    port_csv = f"./raw_datas/tmp/coordinates_of_port/_{port['name']}.csv"
    df_buoy["x [m]"], df_buoy["y [m]"] = convert(df_buoy, port_csv, "latitude", "longitude")
    return df_buoy


def get_name(df_part, idx):
    if "NOBJNAM" in df_part.columns:
        return df_part.loc[idx, "NOBJNAM"]
    return idx


def nearest_opposite_distances(df_buoy, port_name):
    """各標識について、異色の標識のうち最も近いものとの距離を求める。"""
    df = df_buoy.copy()
    df[COLOUR_COL] = df[COLOUR_COL].astype(str)

    df_a = df.loc[df[COLOUR_COL] == COLOUR_A]
    df_b = df.loc[df[COLOUR_COL] == COLOUR_B]

    if len(df_a) == 0 or len(df_b) == 0:
        return []

    coords_a = df_a[[X_COL, Y_COL]].to_numpy(dtype=float)
    coords_b = df_b[[X_COL, Y_COL]].to_numpy(dtype=float)

    dist_matrix = cdist(coords_a, coords_b)

    records = []
    for i, idx in enumerate(df_a.index):
        j = int(dist_matrix[i].argmin())
        nearest_idx = df_b.index[j]
        records.append({
            "port": port_name,
            "colour": COLOUR_A,
            "name": get_name(df_a, idx),
            "nearest_colour": COLOUR_B,
            "nearest_name": get_name(df_b, nearest_idx),
            "distance": dist_matrix[i, j],
        })
    for j, idx in enumerate(df_b.index):
        i = int(dist_matrix[:, j].argmin())
        nearest_idx = df_a.index[i]
        records.append({
            "port": port_name,
            "colour": COLOUR_B,
            "name": get_name(df_b, idx),
            "nearest_colour": COLOUR_A,
            "nearest_name": get_name(df_a, nearest_idx),
            "distance": dist_matrix[i, j],
        })

    return records


ports = dictionary()
port_items = ports.items() if hasattr(ports, "items") else enumerate(ports)

all_records = []

for port_number, port in port_items:
    df_buoy = load_port_df(port)

    if COLOUR_COL not in df_buoy.columns:
        print(f"[{port['name']}] {COLOUR_COL} 列が無いためスキップします。")
        continue

    all_records.extend(nearest_opposite_distances(df_buoy, port["name"]))

distance_df = pd.DataFrame(all_records).sort_values("distance").reset_index(drop=True)
distance_df.to_csv(os.path.join(savedir, "nearest_distances.csv"), index=False, encoding="utf-8-sig")


def summarize_and_plot(distances, title, file_prefix):
    distances = np.asarray(distances)

    max_bin_edge = (int(distances.max() // BIN_WIDTH) + 1) * BIN_WIDTH
    bin_edges = np.arange(0, max_bin_edge + BIN_WIDTH, BIN_WIDTH)

    counts, _ = np.histogram(distances, bins=bin_edges)
    histogram_df = pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "bin_end": bin_edges[1:],
        "count": counts,
    })
    histogram_df["percentage"] = histogram_df["count"] / len(distances) * 100
    histogram_df["cumulative_percentage"] = histogram_df["percentage"].cumsum()

    histogram_df.to_csv(
        os.path.join(savedir, f"{file_prefix}_histogram.csv"), index=False, encoding="utf-8-sig"
    )

    print(f"\n=== {title} (n={len(distances)}) ===")
    print(histogram_df.to_string(index=False))

    for threshold_pct in (90, 95, 99):
        row = histogram_df.loc[histogram_df["cumulative_percentage"] >= threshold_pct].iloc[0]
        print(f"累積 {threshold_pct}% は {int(row['bin_end'])} m 以下")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(histogram_df["bin_start"], histogram_df["count"], width=BIN_WIDTH * 0.9, align="edge")
    ax.set_xlabel("nearest opposite-colour distance [m]")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"{file_prefix}_histogram.png"), dpi=150)
    plt.savefig(os.path.join(savedir, f"{file_prefix}_histogram.pdf"), dpi=150)
    plt.close(fig)

    return histogram_df


# --- 全ての標識 -------------------------------------------------------
summarize_and_plot(
    distance_df["distance"],
    title="Distribution of nearest opposite-colour marker distance (all markers)",
    file_prefix="nearest_distance_all",
)

# --- 名前に「第」を含む標識のみ（対をなす側面標識に限定）--------------
NAME_FILTER_SUBSTR = "第"
distance_df_dai = distance_df.loc[
    distance_df["name"].astype(str).str.contains(NAME_FILTER_SUBSTR, na=False)
]
distance_df_dai.to_csv(
    os.path.join(savedir, "nearest_distances_dai_only.csv"), index=False, encoding="utf-8-sig"
)

if len(distance_df_dai) > 0:
    summarize_and_plot(
        distance_df_dai["distance"],
        title=f'Distribution of nearest opposite-colour marker distance',
        file_prefix="nearest_distance_dai_only",
    )
else:
    print(f'\n"{NAME_FILTER_SUBSTR}" を含む標識が見つかりませんでした。')
