"""
分類された要素の出現割合を最適化用コストに使える形の辞書に格納し、
辞書を .py にエクスポートする。9.5kts 以上は 9.5 に集約（キーは数値文字列）、
角度は 0 / 5..55 / 60(≧60) を持つ。積み重ね棒グラフも保存する。
"""

import os
import os.path as osp
from datetime import datetime
from pprint import pformat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# paths
DIR = osp.dirname(__file__)
dirname = osp.splitext(osp.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
MODULE_OUT = osp.normpath(osp.join(DIR, "../utils/PP/Filtered_Dict.py"))
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(osp.join(SAVE_DIR, "fig"), exist_ok=True)
os.makedirs(osp.join(SAVE_DIR, "csv"), exist_ok=True)

# settings
FIGURE_PLOT_SWITCH = True
CSV_PATH = f"{DIR}/../../outputs/ClassifyElements/all_ports_classified_elements_fixed.csv"

def load_filtered_df(csv_path: str, selected_elements):
    """
    CSVを読み込み、指定要素のみを抽出したDataFrameを返す。
    引数:
        csv_path          : 入力CSVのパス
        selected_elements : 抽出対象の element のリスト
    戻り値:
        pandas.DataFrame（抽出済み）
    """
    df = pd.read_csv(csv_path)
    return df[df["element"].isin(selected_elements)].copy()

def build_bins(speed_min=1.5, speed_max=9.5, interval=1.0,
               angle_min=5, angle_max=60, angle_interval=5):
    """
    速度ビン（下端配列）と角度カテゴリを定義する。
    速度は [speed_min, speed_max) を interval 刻み、speed_max 以上は speed_max に集約。
    角度は 0 / 5..55 / 60(≧60) を用意する。
    戻り値:
        speed_bins(list[float]), angle_cats(list[int])
    """
    speed_bins = [round(speed_min + i * interval, 1)
                  for i in range(int((speed_max - speed_min) / interval))]
    angle_cats = [0] + list(range(angle_min, angle_max, angle_interval)) + [60]
    return speed_bins, angle_cats

def bin_speed_to_key(speed: float, speed_bins, speed_max: float) -> str:
    """
    実数速度を速度ビンに割り付け、キー（数値文字列）を返す。
    speed_max 以上は speed_max の文字列に集約する（例: '9.5'）。
    """
    for b in speed_bins:
        if b <= speed < b + (speed_bins[1] - speed_bins[0]):
            return f"{b:.1f}"
    return f"{speed_max:.1f}"

def bin_angle_to_key(diff_psi_abs: float) -> str:
    """
    絶対回頭角を角度カテゴリに割り付け、キー（数値文字列）を返す。
    0 は「保針（element==1 or 3）」、5..55 は5刻み、60は 60 以上。
    """
    if diff_psi_abs < 5:
        return "0"
    for a in range(5, 60, 5):
        if a <= diff_psi_abs < a + 5:
            return str(a)
    return "60"

def build_new_filtered_dict(filtered_df: pd.DataFrame,
                            selected_elements=(1, 3, 5),
                            speed_min=1.5, speed_max=9.5, interval=1.0,
                            angle_min=5, angle_max=60, angle_interval=5,
                            threshold_percent=1, normalization_standard=100):
    """
    フィルタ済みDFから最終的な new_filtered_dict を構築する。
    仕様:
      - 速度キーは数値文字列。9.5kts以上は '9.5' に集約。
      - 角度キーは '0' と '5','10',..,'55','60'(≧60) の数値文字列。
      - 各速度キーで合計が normalization_standard(=100) になるよう整数配分する。
      - 要素 1,3 は角度0とみなす。
      - 各速度ビンのサンプル数が全体の threshold_percent(%) 未満なら除外。
    戻り値:
      new_filtered_dict: dict[str, dict[str, int]]
    """
    speed_bins, angle_cats = build_bins(speed_min, speed_max, interval,
                                        angle_min, angle_max, angle_interval)

    dict_by_speed = {f"{b:.1f}": [] for b in speed_bins}
    dict_by_speed[f"{speed_max:.1f}"] = []

    # assign to bins (no duplication)
    for _, row in filtered_df.iterrows():
        spd = float(row["knot"])
        key_speed = bin_speed_to_key(spd, speed_bins, speed_max)
        dict_by_speed[key_speed].append(row)

    # filter by threshold
    total_len = len(filtered_df)
    threshold = max(1, round(total_len * (threshold_percent / 100.0)))
    kept = {k: v for k, v in dict_by_speed.items() if len(v) >= threshold}

    # count & normalize to 100
    new_filtered_dict = {}
    for s_key, rows in kept.items():
        # initialize counts for all angle categories
        counts = {str(a): 0 for a in angle_cats}
        for row in rows:
            if int(row["element"]) in (1, 3):
                a_key = "0"
            else:
                a_key = bin_angle_to_key(abs(float(row["diff_psi_raw"])))
            counts[a_key] = counts.get(a_key, 0) + 1

        coef = max(1.0, len(rows) / float(normalization_standard))
        tmp = []
        total = 0
        for a_key in counts:
            val = counts[a_key] / coef
            integ = int(val)
            frac = val - integ
            tmp.append((a_key, integ, frac))
            total += integ

        tmp.sort(key=lambda x: x[2], reverse=True)
        i = 0
        while total < normalization_standard and i < len(tmp):
            a_key, integ, frac = tmp[i]
            tmp[i] = (a_key, integ + 1, frac)
            total += 1
            i += 1

        new_filtered_dict[s_key] = {a_key: integ for a_key, integ, _ in tmp}
        if "60" not in new_filtered_dict[s_key]:
            new_filtered_dict[s_key]["60"] = 0

    return new_filtered_dict

def export_dict_module(d: dict, module_path: str, func_name: str = "new_filtered_dict", pretty: bool = True):
    """
    new_filtered_dict を返す関数を含む .py を生成する。
    例:
        from utils.PP.Filtered_Dict import new_filtered_dict
        D = new_filtered_dict()
    備考:
        program/utils および program/utils/PP に __init__.py を自動生成して import 可能にする。
    """
    from datetime import datetime
    from pprint import pformat
    os.makedirs(osp.dirname(module_path), exist_ok=True)

    # ensure packages: program/utils と program/utils/PP
    pkg_pp = osp.dirname(module_path)          # .../program/utils/PP
    pkg_utils = osp.dirname(pkg_pp)            # .../program/utils
    for p in [pkg_utils, pkg_pp]:
        initf = osp.join(p, "__init__.py")
        if not osp.exists(initf):
            with open(initf, "w", encoding="utf-8") as fw:
                fw.write("")

    payload = pformat(d, width=100) if pretty else repr(d)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(module_path, "w", encoding="utf-8") as f:
        f.write(f'"""自動生成ファイル: {stamp}\n')
        f.write('new_filtered_dict() がコスト用辞書を返す。\n')
        f.write('このファイルはスクリプトから上書き生成されます。\n')
        f.write('"""\n\n')
        f.write(f"def {func_name}():\n")
        f.write('    """コスト計算用の辞書を返す。"""\n')
        if "\n" not in payload:
            f.write(f"    return {payload}\n")
        else:
            f.write("    return (\n")
            for line in payload.splitlines():
                f.write("        " + line + "\n")
            f.write("    )\n")


def save_dictionary_csv(d: dict, path: str):
    """
    new_filtered_dict を縦長形式のCSVに保存する。
    列: speed_key, angle_key, value
    """
    rows = []
    for s_key, sub in d.items():
        for a_key, val in sub.items():
            rows.append({"speed_key": s_key, "angle_key": a_key, "value": int(val)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def plot_stacked_bar(d: dict, interval=1.0, speed_max=9.5, angle_interval=5):
    """
    new_filtered_dict をもとに積み重ね棒グラフを描画・保存する。
    X: 速度ビン（'1.5','2.5',...,'9.5'）
    Y: 角度カテゴリ（0 は 0-5 deg、60 は 60+）
    """
    plt.rcParams["font.family"] = "Times New Roman"

    speed_keys = sorted(d.keys(), key=lambda x: float(x))
    x = np.arange(len(speed_keys))

    speed_labels = []
    for sk in speed_keys:
        v = float(sk)
        if v < speed_max:
            speed_labels.append(f"{v:.1f}-{v+interval:.1f}")
        else:
            speed_labels.append(">=9.5")

    angles_plot = [0] + list(range(5, 30, angle_interval)) + [30]  # 0,5..25, 30+(表示用)
    cmap = plt.colormaps.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8, 12))
    bottom = np.zeros(len(speed_keys))

    legend_labels = []
    series = []
    for idx, ang in enumerate(angles_plot):
        if ang == 0:
            lab = "0-5 degrees"
            vals = [d[sk].get("0", 0) for sk in speed_keys]
        elif ang == 30:
            lab = "30 degrees or more"
            vals = []
            for sk in speed_keys:
                s = 0
                for a in list(range(30, 60, 5)) + [60]:
                    s += d[sk].get(str(a), 0)
                vals.append(s)
        else:
            lab = f"{ang}-{ang+angle_interval} degrees"
            vals = [d[sk].get(str(ang), 0) for sk in speed_keys]

        bars = ax.bar(x, vals, bottom=bottom, color=cmap(idx))
        bottom += np.array(vals)
        legend_labels.append(lab)
        series.append(bars)

    ax.set_xlabel("Speed Range [kts]", fontsize=16, labelpad=10)
    ax.set_ylabel("Percentage of Turning Angles", fontsize=16, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(speed_labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=14)

    num_cols = min(3, len(legend_labels))
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=cmap(i)) for i in range(len(legend_labels))],
              legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              fontsize=16, ncol=num_cols, frameon=True)

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    out_png = osp.join(SAVE_DIR, "fig", "stacked_normalized_histogram_.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    """
    1) CSV読み込み→辞書構築
    2) 辞書をCSV保存
    3) 積み重ね棒グラフPNG保存
    4) import可能な .py を自動生成（new_filtered_dict）
    """
    selected = [1, 3, 5]
    filtered_df = load_filtered_df(CSV_PATH, selected_elements=selected)

    d = build_new_filtered_dict(
        filtered_df,
        selected_elements=selected,
        speed_min=1.5, speed_max=9.5, interval=1.0,
        angle_min=5, angle_max=60, angle_interval=5,
        threshold_percent=1, normalization_standard=100
    )

    csv_out = osp.join(SAVE_DIR, "csv", "new_filtered_dict_long.csv")
    save_dictionary_csv(d, csv_out)

    if FIGURE_PLOT_SWITCH:
        plot_stacked_bar(d, interval=1.0, speed_max=9.5, angle_interval=5)

    module_path = osp.join(DIR, "../../../utils/PP/MakeDictionary_and_StackedBarGraph.py")
    export_dict_module(
        d,
        module_path=MODULE_OUT,
        func_name="new_filtered_dict",
        pretty=True
    )

    print(f"Dict CSV  : {csv_out}")
    print(f"PNG Figure: {osp.join(SAVE_DIR, 'fig', 'stacked_normalized_histogram_.png')}")
    print(f"Module    : {module_path}")
    print("\nDone\n")
