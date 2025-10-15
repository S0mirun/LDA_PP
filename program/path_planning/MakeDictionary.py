"""
要約:
- 分類済み要素から速度×角度の出現割合をコスト用辞書に整形し .py に書き出す
- 同時に n-gram (2,3,4) の確率可視化と確率辞書も計算し、両方の関数を Filtered_Dict.py に書き出す
出力:
- utils/PP/Filtered_Dict.py  (new_filtered_dict(), get_transition_probs() を返す)
- outputs/<script_name>/{fig,csv} 下に各種PNG/CSV
"""

import glob
import os
import os.path as osp
from datetime import datetime
from pprint import pformat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy.lib.stride_tricks import sliding_window_view

# ===== CONFIG =====
CSV_PATH = "../../outputs/ClassifyElements/all_ports_classified_elements_fixed.csv"  # input for cost dict
FILTERED_DICT_MODULE_OUT = "../utils/PP/Filtered_Dict.py"  # exported module
NGRAM_SEQ_GLOB = "../../outputs/ClassifyElements/*/*.csv"  # sequences for n-gram

SPEED_MIN = 1.5
SPEED_MAX = 9.5
SPEED_INTERVAL = 1.0
ANGLE_MIN = 5
ANGLE_MAX = 60
ANGLE_INTERVAL = 5
THRESHOLD_PERCENT = 1
NORMALIZATION_STANDARD = 100  # integer percent sum=100

N_STATES = 6
TOPK_TRIGRAM = 18
TOPK_FOURGRAM = 24
ALPHA_SMOOTH = 0.0

FIGURE_PLOT_SWITCH = True  # save figures
# ===================

# paths
DIR = osp.dirname(__file__)
dirname = osp.splitext(osp.basename(__file__))[0]
SAVE_DIR = osp.normpath(osp.join(DIR, f"../../outputs/{dirname}"))
os.makedirs(osp.join(SAVE_DIR, "fig"), exist_ok=True)
os.makedirs(osp.join(SAVE_DIR, "csv"), exist_ok=True)

# colormap
white_red = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#FF0000"])


# ---------- utils for integer percent ----------
def _largest_remainder_percent(row_probs: np.ndarray, total: int = 100) -> np.ndarray:
    """
    確率ベクトルを整数パーセントに丸め、合計 total（既定100）にする。
    行の合計が0のときは全0。
    """
    s = float(np.nansum(row_probs))
    if s <= 0.0:
        return np.zeros_like(row_probs, dtype=int)
    x = np.clip(row_probs, 0.0, None) / s * total
    base = np.floor(x).astype(int)
    rem = x - base
    need = total - int(base.sum())
    if need > 0:
        idx = np.argsort(rem)[::-1][:need]
        base[idx] += 1
    return base.astype(int)

def _matrix_to_percent_ints(P: np.ndarray, total: int = 100) -> np.ndarray:
    """
    行ごとに整数パーセントへ変換（合計=total、ゼロ行は全0）。
    """
    out = np.zeros_like(P, dtype=int)
    for i in range(P.shape[0]):
        out[i, :] = _largest_remainder_percent(P[i, :], total=total)
    return out


# --------- Part A: コスト用辞書（速度=floatキー、角度=intキー、整数%合計=100） ---------
def load_filtered_df(csv_path: str, selected_elements):
    """
    CSVを読み込み、指定要素のみ抽出したDataFrameを返す。
    """
    path = csv_path if osp.isabs(csv_path) else osp.normpath(osp.join(DIR, csv_path))
    df = pd.read_csv(path)
    return df[df["element"].isin(selected_elements)].copy()

def build_bins(speed_min=1.5, speed_max=9.5, interval=1.0,
               angle_min=5, angle_max=60, angle_interval=5):
    """
    速度ビン下端配列と角度カテゴリ（0,5..55,60+）を返す。
    """
    speed_bins = [round(speed_min + i * interval, 1)
                  for i in range(int((speed_max - speed_min) / interval))]
    angle_cats = [0] + list(range(angle_min, angle_max, angle_interval)) + [60]
    return speed_bins, angle_cats

def bin_speed_to_key(speed: float, speed_bins, speed_max: float) -> float:
    """
    実数速度を速度ビンに割当て、キー（float）を返す。speed_max 以上は speed_max。
    """
    step = speed_bins[1] - speed_bins[0] if len(speed_bins) >= 2 else SPEED_INTERVAL
    for b in speed_bins:
        if b <= speed < b + step:
            return float(b)
    return float(speed_max)

def bin_angle_to_key(diff_psi_abs: float) -> int:
    """
    絶対回頭角を 0 / 5..55 / 60 に割当て、キー（int）を返す。
    """
    if diff_psi_abs < 5:
        return 0
    for a in range(5, 60, 5):
        if a <= diff_psi_abs < a + 5:
            return int(a)
    return 60

def build_new_filtered_dict(filtered_df: pd.DataFrame,
                            selected_elements=(1, 3, 5),
                            speed_min=1.5, speed_max=9.5, interval=1.0,
                            angle_min=5, angle_max=60, angle_interval=5,
                            threshold_percent=1, normalization_standard=100):
    """
    フィルタ済みDFから new_filtered_dict を構築（キーは数値: 速度=float, 角度=int）。
    各速度ビンで整数%に正規化し、合計=normalization_standard（既定100）。
    要素1,3は角度0、速度>=speed_maxは speed_max に集約。
    戻り値: dict[float, dict[int, int]]
    """
    speed_bins, angle_cats = build_bins(speed_min, speed_max, interval,
                                        angle_min, angle_max, angle_interval)
    buckets = {float(b): [] for b in speed_bins}
    buckets[float(speed_max)] = []

    # assign rows to speed bins
    for _, row in filtered_df.iterrows():
        spd = float(row["knot"])
        key_speed = bin_speed_to_key(spd, speed_bins, speed_max)
        buckets[key_speed].append(row)

    # drop sparse bins
    total_len = len(filtered_df)
    threshold = max(1, round(total_len * (threshold_percent / 100.0)))
    kept = {k: v for k, v in buckets.items() if len(v) >= threshold}

    new_filtered: dict[float, dict[int, int]] = {}
    for s_key, rows in kept.items():
        # count angles as raw counts
        counts = {int(a): 0 for a in angle_cats}
        for row in rows:
            if int(row["element"]) in (1, 3):
                a_key = 0
            else:
                a_key = bin_angle_to_key(abs(float(row["diff_psi_raw"])))
            counts[a_key] = counts.get(a_key, 0) + 1

        # normalize to integer percent sum=normalization_standard
        coef = max(1.0, len(rows) / float(normalization_standard))
        tmp = []
        total = 0
        for a_key in counts:
            val = counts[a_key] / coef
            integ = int(val)
            frac = val - integ
            tmp.append((int(a_key), int(integ), float(frac)))
            total += int(integ)

        tmp.sort(key=lambda x: x[2], reverse=True)
        i = 0
        while total < normalization_standard and i < len(tmp):
            a_key, integ, frac = tmp[i]
            tmp[i] = (a_key, integ + 1, frac)
            total += 1
            i += 1

        new_filtered[float(s_key)] = {int(a_key): int(integ) for a_key, integ, _ in tmp}
        if 60 not in new_filtered[float(s_key)]:
            new_filtered[float(s_key)][60] = 0

    return new_filtered

def save_dictionary_csv(d: dict, path: str):
    """
    new_filtered_dict を縦長CSVとして保存（speed_key: float, angle_key: int, value: int）。
    """
    rows = []
    for s_key, sub in d.items():
        for a_key, val in sub.items():
            rows.append({"speed_key": float(s_key), "angle_key": int(a_key), "value": int(val)})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

def plot_stacked_bar(d: dict, interval=1.0, speed_max=9.5, angle_interval=5):
    """
    new_filtered_dict から積み重ね棒を描画・保存。
    """
    plt.rcParams["font.family"] = "Times New Roman"
    speed_keys = sorted(d.keys(), key=float)
    x = np.arange(len(speed_keys))
    labels = []
    for sk in speed_keys:
        v = float(sk)
        labels.append(f"{v:.1f}-{v+interval:.1f}" if v < speed_max else f">={speed_max:.1f}")

    angles_plot = [0] + list(range(5, 30, angle_interval)) + [30]  # view grouping
    cmap = plt.colormaps.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 12))
    bottom = np.zeros(len(speed_keys))
    legends = []
    for idx, ang in enumerate(angles_plot):
        if ang == 0:
            lab = "0-5°"
            vals = [d[sk].get(0, 0) for sk in speed_keys]
        elif ang == 30:
            lab = "30°+"
            vals = []
            for sk in speed_keys:
                s = 0
                for a in list(range(30, 60, 5)) + [60]:
                    s += d[sk].get(int(a), 0)
                vals.append(s)
        else:
            lab = f"{ang}-{ang+angle_interval}°"
            vals = [d[sk].get(int(ang), 0) for sk in speed_keys]
        ax.bar(x, vals, bottom=bottom, color=cmap(idx))
        bottom += np.array(vals)
        legends.append(lab)

    ax.set_xlabel("Speed Range [kts]", fontsize=16, labelpad=10)
    ax.set_ylabel("Percentage of Turning Angles", fontsize=16, labelpad=10)
    ax.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=14)
    ncol = min(3, len(legends))
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=cmap(i)) for i in range(len(legends))],
              legends, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              fontsize=16, ncol=ncol, frameon=True)
    plt.tight_layout(rect=[0, 0.3, 1, 1])
    out_png = osp.join(SAVE_DIR, "fig", "stacked_normalized_histogram_.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_png


# --------- Part B: n-gram 可視化 & 確率辞書（各行=100の整数%） ---------
def _save_fig(fig, order, filename="heatmap.png"):
    """
    図を保存して閉じる。
    """
    out_dir = osp.join(SAVE_DIR, "fig")
    os.makedirs(out_dir, exist_ok=True)
    root, ext = osp.splitext(filename)
    ext = ext if ext else ".png"
    out_name = f"order-{order}_{root}{ext}"
    fig.savefig(osp.join(out_dir, out_name), dpi=150)
    plt.close(fig)
    return osp.join(out_dir, out_name)

def _save_csv(df: pd.DataFrame, order, filename):
    """
    DataFrameをCSV保存する。
    """
    out_dir = osp.join(SAVE_DIR, "csv")
    os.makedirs(out_dir, exist_ok=True)
    root, ext = osp.splitext(filename)
    ext = ext if ext else ".csv"
    out_name = f"order-{order}_{root}{ext}"
    out_path = osp.join(out_dir, out_name)
    df.to_csv(out_path, encoding="utf-8-sig", index=True)
    return out_path

def ngram_counts_multi(seqs, N, order=2):
    """
    複数系列に対して n-gram（長さ=order）の総カウント行列を返す（形状=(N,)*order）。
    """
    shape = (N,) * order
    C = np.zeros(shape, dtype=np.int64)
    base = (N ** np.arange(order-1, -1, -1)).astype(np.int64)
    size_flat = N ** order
    for s in seqs:
        a = np.asarray(s, dtype=np.int64)
        if a.size < order:
            continue
        W = sliding_window_view(a, order)
        v = (W >= 0).all(axis=1) & (W < N).all(axis=1)
        if not np.any(v):
            continue
        idx_flat = (W[v] * base).sum(axis=1)
        C += np.bincount(idx_flat, minlength=size_flat).reshape(shape)
    return C

def counts_to_conditional_probs(counts, alpha=0.0):
    """
    カウント配列を条件付き確率（最後軸正規化）に変換。alpha>0 で加法的平滑化。
    """
    c = counts.astype(float)
    if alpha > 0:
        c += alpha
    denom = c.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.divide(c, denom, where=(denom > 0))
    P[np.isnan(P)] = 0.0
    return P

def plot_bigram_heatmap_probs(counts_mat, labels=None, title=None,
                              annotate=True, fmt=".1%", filename="heatmap_probs.png",
                              show_row_totals=False):
    """
    2要素（bigram）の確率ヒートマップを描画・保存（セル注記可）。返り値は確率行列P。
    """
    P = counts_to_conditional_probs(counts_mat)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(P, cmap=white_red, vmin=0.0, vmax=1.0)
    n = P.shape[0]
    labs = labels or list(range(n))
    if show_row_totals:
        totals = counts_mat.sum(axis=1).astype(int)
        yticks = [f"{labs[i]} [n={totals[i]}]" for i in range(n)]
    else:
        yticks = labs
    ax.set_xticks(np.arange(n), labels=labs)
    ax.set_yticks(np.arange(n), labels=yticks)
    ax.set_xlabel("N+1-th element")
    ax.set_ylabel("N-th element")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    ax.set_title(title or "N→N+1 transition (probabilities)")
    if annotate:
        thresh = 0.6
        for i in range(n):
            for j in range(n):
                v = P[i, j]
                ax.text(j, i, format(v, fmt),
                        ha="center", va="center",
                        fontsize=10, color=("white" if v >= thresh else "black"))
    fig.tight_layout()
    path = _save_fig(fig, order=2, filename=filename)
    return P, path

def plot_ngram_context_heatmap_probs(M: np.ndarray, row_labels, col_labels,
                                     order: int, title=None, annotate=False,
                                     fmt=".1%", filename="context_heatmap_probs.png",
                                     row_counts: np.ndarray | None = None):
    """
    上位コンテキスト×次状態の確率ヒートマップを描画・保存。
    M: 行=コンテキスト（top-kで抽出済み）, 列=次状態（0..N-1）の確率行列
    row_labels: 行ラベル（例: "a→b", "c→d" / "a→b→c"）
    col_labels: 列ラベル（例: list(range(N))）
    order: n-gramの次数（3 または 4）
    """
    fig_h = max(4, 0.35 * len(row_labels))  # 行数に応じて縦サイズ自動調整
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(M, cmap=white_red, vmin=0.0, vmax=1.0, aspect="auto")
    if row_counts is not None:
        ylabels = [f"{lab} [n={int(row_counts[i])}]" for i, lab in enumerate(row_labels)]
    else:
        ylabels = row_labels

    n_cols = len(col_labels)
    ax.set_xticks(np.arange(n_cols), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=ylabels)
    ax.set_xlabel("N+1-th element")
    ax.set_ylabel("Context (previous elements)")
    ax.set_title(title or f"Top contexts → next (order={order})")
    plt.colorbar(im, ax=ax)

    if annotate and len(row_labels) <= 24:  # 行数が少ないときだけ注記
        thresh = 0.6
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, format(v, fmt),
                        ha="center", va="center",
                        fontsize=9, color=("white" if v >= thresh else "black"))

    fig.tight_layout()
    return _save_fig(fig, order=order, filename=filename)


def topk_context_matrix(counts, topk=20, as_prob=True):
    """
    高次 n-gram の文脈（最後の軸以外）をフラット化し、上位topk行のみ抽出。
    """
    order = counts.ndim
    N = counts.shape[0]
    ctx_dims = order - 1
    C = counts.reshape(N**ctx_dims, N)
    totals = C.sum(axis=1)
    idx = np.argsort(totals)[::-1][:topk]
    M = C[idx, :].astype(float)
    if as_prob:
        M = counts_to_conditional_probs(M)
    return M, idx

def _decode_row_indices(idx, N, ctx_dims, labels=None):
    """
    フラット化行インデックスを元の文脈系列に復元しラベル化して返す。
    """
    base = labels or list(range(N))
    out = []
    for r in idx:
        ctx, x = [], int(r)
        for _ in range(ctx_dims):
            ctx.append(x % N); x //= N
        ctx = tuple(base[k] for k in ctx[::-1])
        out.append(ctx)
    return out

def _matrix_to_int_dict_rows(M: np.ndarray, row_labels, col_labels):
    """
    行列M（確率）を各行ごと整数%にして辞書へ。各行合計は100（ゼロ行は合計0）。
    """
    out = {}
    for i, rlab in enumerate(row_labels):
        ints = _largest_remainder_percent(M[i, :], total=100)
        row = {col_labels[j]: int(ints[j]) for j in range(len(col_labels))}
        out[rlab] = row
    return out

def build_transition_probs(seqs, N=6, labels=None, topk3=18, topk4=24, alpha=0.0):
    """
    複数系列から n-gram 確率辞書（2,3,4）を構築（各行は整数%で合計100。ゼロ行は全0）。
    """
    labels = labels or list(range(N))

    # bigram
    C2 = ngram_counts_multi(seqs, N, order=2)
    P2 = counts_to_conditional_probs(C2, alpha=alpha)
    P2_int = _matrix_to_percent_ints(P2, total=100)
    d2 = {labels[i]: {labels[j]: int(P2_int[i, j]) for j in range(N)} for i in range(N)}

    # trigram
    C3 = ngram_counts_multi(seqs, N, order=3)
    M3, idx3 = topk_context_matrix(C3, topk=topk3, as_prob=True)
    ctx3 = _decode_row_indices(idx3, N=N, ctx_dims=2, labels=labels)
    d3 = _matrix_to_int_dict_rows(M3, ctx3, labels)

    # fourgram
    C4 = ngram_counts_multi(seqs, N, order=4)
    M4, idx4 = topk_context_matrix(C4, topk=topk4, as_prob=True)
    ctx4 = _decode_row_indices(idx4, N=N, ctx_dims=3, labels=labels)
    d4 = _matrix_to_int_dict_rows(M4, ctx4, labels)

    return {
        "order_2": d2,
        f"order_3_top{topk3}": d3,
        f"order_4_top{topk4}": d4,
    }


# --------- Export (Filtered_Dict.py に2つのdefを書き出す) ---------
def export_combined_module(cost_dict: dict, probs_dict: dict, module_path: str,
                           func_cost: str = "new_filtered_dict",
                           func_probs: str = "get_transition_probs",
                           pretty: bool = True):
    """
    cost辞書とn-gram確率辞書を返す2関数を1つの .py に書き出す（__init__.py も自動生成）。
    """
    path_abs = module_path if osp.isabs(module_path) else osp.normpath(osp.join(DIR, module_path))
    os.makedirs(osp.dirname(path_abs), exist_ok=True)

    # make packages
    pkg_pp = osp.dirname(path_abs)
    pkg_utils = osp.dirname(pkg_pp)
    for p in [pkg_utils, pkg_pp]:
        initf = osp.join(p, "__init__.py")
        if not osp.exists(initf):
            with open(initf, "w", encoding="utf-8") as fw:
                fw.write("")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload_cost = pformat(cost_dict, width=100) if pretty else repr(cost_dict)
    payload_probs = pformat(probs_dict, width=100) if pretty else repr(probs_dict)

    with open(path_abs, "w", encoding="utf-8") as f:
        f.write(f'"""自動生成ファイル: {stamp}\n')
        f.write('new_filtered_dict() はコスト用辞書（キー: 速度=float, 角度=int。各速度ビン合計=100の整数%）、\n')
        f.write('get_transition_probs() はn-gram確率辞書（各行の合計=100の整数%。ゼロ行は全0）を返す。\n')
        f.write('このファイルはスクリプトから上書き生成されます。\n')
        f.write('"""\n\n')
        # cost dict function
        f.write(f"def {func_cost}():\n")
        f.write('    """コスト計算用の辞書（キー: 速度=float, 角度=int。値は整数%、各速度ビンで合計=100）を返す。"""\n')
        if "\n" not in payload_cost:
            f.write(f"    return {payload_cost}\n\n")
        else:
            f.write("    return (\n")
            for line in payload_cost.splitlines():
                f.write("        " + line + "\n")
            f.write("    )\n\n")
        # probs dict function
        f.write(f"def {func_probs}():\n")
        f.write('    """n-gram の確率辞書（各行の合計=100の整数%。ゼロ行は全0）を返す。"""\n')
        if "\n" not in payload_probs:
            f.write(f"    return {payload_probs}\n")
        else:
            f.write("    return (\n")
            for line in payload_probs.splitlines():
                f.write("        " + line + "\n")
            f.write("    )\n")


# --------- script mode ---------
if __name__ == "__main__":
    """
    1) コスト辞書の構築・CSV保存・積み重ね棒の保存
    2) n-gram可視化（bigram/trigram/fourgram）とCSV保存
    3) Filtered_Dict.py に new_filtered_dict() / get_transition_probs() を同居で書き出し
    """
    # A) cost dict (normalize to 100)
    selected = [1, 3, 5]
    filtered_df = load_filtered_df(CSV_PATH, selected_elements=selected)
    d_cost = build_new_filtered_dict(
        filtered_df,
        selected_elements=selected,
        speed_min=SPEED_MIN, speed_max=SPEED_MAX, interval=SPEED_INTERVAL,
        angle_min=ANGLE_MIN, angle_max=ANGLE_MAX, angle_interval=ANGLE_INTERVAL,
        threshold_percent=THRESHOLD_PERCENT, normalization_standard=NORMALIZATION_STANDARD
    )
    csv_out_path = osp.join(SAVE_DIR, "csv", "new_filtered_dict_long.csv")
    save_dictionary_csv(d_cost, csv_out_path)
    fig_out_path = None
    if FIGURE_PLOT_SWITCH:
        fig_out_path = plot_stacked_bar(
            d_cost, interval=SPEED_INTERVAL, speed_max=SPEED_MAX, angle_interval=ANGLE_INTERVAL
        )

    # B) n-gram
    seqs = []
    for path in glob.glob(osp.normpath(osp.join(DIR, NGRAM_SEQ_GLOB))):
        try:
            df = pd.read_csv(path, encoding="shift-jis")
        except UnicodeDecodeError:
            df = pd.read_csv(path)
        if "element" in df.columns:
            seqs.append(df["element"].to_numpy())

    labels = list(range(N_STATES))

    # bigram → PNG & CSV（CSVはfloat確率）
    C2 = ngram_counts_multi(seqs, N_STATES, order=2)
    P2, bigram_png = plot_bigram_heatmap_probs(
        C2,
        labels=labels,
        title="N→N+1 transition (probabilities)",
        annotate=True,
        fmt=".1%",
        filename="heatmap_probs.png",
        show_row_totals=True
    )
    bigram_csv = _save_csv(pd.DataFrame(P2, index=labels, columns=labels), order=2, filename="probs.csv")

    # trigram → ヒートマップPNG & CSV（CSVはfloat確率）
    C3 = ngram_counts_multi(seqs, N_STATES, order=3)
    M3, idx3 = topk_context_matrix(C3, topk=TOPK_TRIGRAM, as_prob=True)
    ctx3 = _decode_row_indices(idx3, N=N_STATES, ctx_dims=2, labels=labels)
    rows3 = [{"context": "→".join(map(str, ctx)), "next": j, "prob": float(M3[i, j])}
             for i, ctx in enumerate(ctx3) for j in range(M3.shape[1])]
    trigram_csv = _save_csv(pd.DataFrame(rows3), order=3, filename=f"probs_top{TOPK_TRIGRAM}.csv")
    totals3 = C3.reshape(N_STATES**2, N_STATES).sum(axis=1)[idx3].astype(int)
    trigram_png = plot_ngram_context_heatmap_probs(
        M3,
        row_labels=["→".join(map(str, ctx)) for ctx in ctx3],
        col_labels=labels,
        order=3,
        title=f"Top-{TOPK_TRIGRAM} trigram contexts → next (probabilities)",
        annotate=True,
        filename=f"heatmap_probs_top{TOPK_TRIGRAM}.png",
        row_counts=totals3
    )


    # fourgram → ヒートマップPNG & CSV（CSVはfloat確率）
    C4 = ngram_counts_multi(seqs, N_STATES, order=4)
    M4, idx4 = topk_context_matrix(C4, topk=TOPK_FOURGRAM, as_prob=True)
    ctx4 = _decode_row_indices(idx4, N=N_STATES, ctx_dims=3, labels=labels)
    rows4 = [{"context": "→".join(map(str, ctx)), "next": j, "prob": float(M4[i, j])}
             for i, ctx in enumerate(ctx4) for j in range(M4.shape[1])]
    fourgram_csv = _save_csv(pd.DataFrame(rows4), order=4, filename=f"probs_top{TOPK_FOURGRAM}.csv")
    totals4 = C4.reshape(N_STATES**3, N_STATES).sum(axis=1)[idx4].astype(int)
    fourgram_png = plot_ngram_context_heatmap_probs(
        M4,
        row_labels=["→".join(map(str, ctx)) for ctx in ctx4],
        col_labels=labels,
        order=4,
        title=f"Top-{TOPK_FOURGRAM} fourgram contexts → next (probabilities)",
        annotate=True,
        filename=f"heatmap_probs_top{TOPK_FOURGRAM}.png",
        row_counts=totals4
    )


    # build int-% probs dict
    probs = build_transition_probs(
        seqs, N=N_STATES, labels=labels, topk3=TOPK_TRIGRAM, topk4=TOPK_FOURGRAM, alpha=ALPHA_SMOOTH
    )

    # export both defs into one module
    export_combined_module(
        cost_dict=d_cost,
        probs_dict=probs,
        module_path=FILTERED_DICT_MODULE_OUT,
        func_cost="new_filtered_dict",
        func_probs="get_transition_probs",
        pretty=True
    )

    print("\n=== Outputs ===")
    print(f"Dict CSV        : {csv_out_path}")
    if fig_out_path:
        print(f"Stacked PNG     : {fig_out_path}")
    print(f"Bigram PNG      : {bigram_png}")
    print(f"Bigram CSV      : {bigram_csv}")
    print(f"Trigram PNG     : {trigram_png}")
    print(f"Trigram CSV     : {trigram_csv}")
    print(f"Fourgram PNG    : {fourgram_png}")
    print(f"Fourgram CSV    : {fourgram_csv}")
    print(f"Module (both)   : {osp.normpath(osp.join(DIR, FILTERED_DICT_MODULE_OUT))}")
    print("\nDone\n")
