import os

import pandas as pd
import matplotlib.pyplot as plt


# 可視化したい CSV ファイル名
DIR = os.path.dirname(__file__)
RAW_DATAS = f"{DIR}/../../raw_datas"
num = "24"
csv_file = os.path.abspath(f"{RAW_DATAS}/C23-06_{num}_GML/C23-06_{num}-g.csv")
SAVE_DIR = os.path.dirname(csv_file)


def visualize():
    df = pd.read_csv(csv_file)
    if not {"curve_id", "lat", "lon"}.issubset(df.columns):
        raise ValueError("CSV に 'curve_id', 'lat', 'lon' の列が必要です")

    fig, ax = plt.subplots(figsize=(8, 8))

    # curve_id ごとに線を描画
    for curve_id, g in df.groupby("curve_id"):
        ax.plot(
            g["lon"].values,
            g["lat"].values,
            linewidth=0.8,
            alpha=0.9,
            label=str(curve_id),
        )

    ax.set_xlim(136.5, 136.8)
    ax.set_ylim(34.8, 35.1)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Curves from C23-06_{num}-g.csv")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # 凡例が多すぎる場合はコメントアウトしてもよい
    # ax.legend(loc="best", fontsize=6)

    plt.tight_layout()

    fig.savefig(f"{SAVE_DIR}/C23-06_{num}-g_curves.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    visualize()