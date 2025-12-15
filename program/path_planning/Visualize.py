"""
国交省から引っ張ってきたデータを可視化するファイル。
事前にGetCoastLineを動かしておくこと。
"""
import os

import pandas as pd
import matplotlib.pyplot as plt


# 可視化したい CSV ファイル名
DIR = os.path.dirname(__file__)
RAW_DATAS = f"{DIR}/../../raw_datas"
port_file = f"{RAW_DATAS}/tmp/coordinates_of_port/_Osaka_port45.csv"


def visualize(csv_file, num, save_dir, ax_all=None):
    df = pd.read_csv(csv_file)
    if not {"curve_id", "lat", "lon"}.issubset(df.columns):
        raise ValueError("CSV に 'curve_id', 'lat', 'lon' の列が必要です")

    fig, ax = plt.subplots(figsize=(8, 8))

    # curve_id ごとに線を描画
    for curve_id, g in df.groupby("curve_id"):
        x = g["lon"].values
        y = g["lat"].values

        ax.plot(
            x,
            y,
            linewidth=0.8,
            alpha=0.9,
            label=str(curve_id),
        )
        if ax_all is not None:
            ax_all.plot(
                x,
                y,
                linewidth=0.3,
                alpha=0.5,
            )
    if os.path.exists(port_file):
        port_df = pd.read_csv(port_file)
        y0 = port_df["Latitude"].iloc[0]
        x0 = port_df["Longitude"].iloc[0]
        delta = 0.05
        ax.set_xlim(x0 - delta, x0 + delta)
        ax.set_ylim(y0 - delta, y0 + delta)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Curves from C23-06_{num}-g.csv")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    fig.savefig(f"{save_dir}/C23-06_{num}-g_curves.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    NUM = [f"{i:02d}" for i in range(1, 48)]
    fig_all, ax_all = plt.subplots(figsize=(8, 8))

    for num in NUM:
        csv_file = os.path.abspath(
            f"{RAW_DATAS}/国土交通省/C23-06_{num}_GML/C23-06_{num}-g.csv"
        )
        if not os.path.exists(csv_file):
            continue

        save_dir = os.path.dirname(csv_file)
        visualize(csv_file, num, save_dir, ax_all=ax_all)
        print(f"{num} finished")

    # --- まとめ図の体裁を整える ---
    if os.path.exists(port_file):
        port_df = pd.read_csv(port_file)
        y0 = port_df["Latitude"].iloc[0]
        x0 = port_df["Longitude"].iloc[0]
        delta = 0.05
        ax_all.set_xlim(x0 - delta, x0 + delta)
        ax_all.set_ylim(y0 - delta, y0 + delta)

    ax_all.set_xlabel("Longitude [deg]")
    ax_all.set_ylabel("Latitude [deg]")
    ax_all.set_title("All curves")
    ax_all.grid(True)
    ax_all.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig_all.savefig(f"{RAW_DATAS}/国土交通省/All_Curves.png", dpi=300)
    plt.close(fig_all)
