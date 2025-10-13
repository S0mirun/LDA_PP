import glob
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from numpy.lib.stride_tricks import sliding_window_view

# paths
DIR = osp.dirname(__file__)
dirname = osp.splitext(osp.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(osp.join(SAVE_DIR, "fig"), exist_ok=True)

# colormap: white -> red
white_red = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#FF0000"])

N = 6
seqs = []

def _save_fig(fig, order):
    """
    図を保存して閉じる。
    保存先: {SAVE_DIR}/fig//heatmap_order_<order>.png
    引数:
        fig   : matplotlib.figure.Figure
        order : n-gram の長さ (2=bigram, 3=trigram, ...)
    副作用:
        ディレクトリを作成し、PNGを書き出して図をcloseする。
    """
    out_dir = osp.join(SAVE_DIR, "fig")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(osp.join(out_dir, f"heatmap_order_{order}.png"), dpi=150)
    plt.close(fig)

def ngram_counts_multi(seqs, N, order=2):
    """
    複数系列に対して n-gram（長さ=order）の総カウント行列を計算する。
    前提:
        各系列は 0..N-1 の整数IDからなる1次元配列。
    引数:
        seqs : 1次元配列のリスト（各系列）
        N    : 状態数
        order: n-gram の n（2=bigram, 3=trigram, ...）
    戻り値:
        形状 (N,)*order の ndarray（各 n-gram の出現回数）。
    備考:
        ウィンドウ内に範囲外IDが含まれる場合はそのウィンドウを無視する。
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
    カウント配列を条件付き確率（最後の軸で正規化）に変換する。
    引数:
        counts: 任意次元（最後の軸が「次の要素」）のカウント配列
        alpha : 加法的平滑化係数（0で無効）
    戻り値:
        counts と同形状の確率配列（各行が合計1、分母0は0とする）。
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
                              annotate=True, fmt=".1%"):
    """
    2要素（bigram）の確率ヒートマップを描画・保存する（セルに確率注記可）。
    引数:
        counts_mat : 形状 (N,N) のカウント行列（行=現状態, 列=次状態）
        labels     : 軸ラベル（長さNのリスト）。Noneなら0..N-1
        title      : タイトル文字列。Noneなら既定
        annotate   : Trueで各セルに確率を描画
        fmt        : 注記の書式（".1%" など）
    出力:
        {SAVE_DIR}/fig/order-2/heatmap.png に保存（表示はしない）。
    """
    P = counts_to_conditional_probs(counts_mat)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(P, cmap=white_red, vmin=0.0, vmax=1.0)
    n = P.shape[0]
    labs = labels or list(range(n))
    ax.set_xticks(np.arange(n), labels=labs)
    ax.set_yticks(np.arange(n), labels=labs)
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
    _save_fig(fig, order=2)

def plot_context_next_heatmap(counts, labels=None, topk=20, as_prob=True,
                              title=None, annotate=False, fmt=".1%"):
    """
    高次 n-gram（order>=3）について、文脈（最後の軸以外）→次要素のヒートマップを描画・保存する。
    手順:
        文脈をフラット化して行方向に並べ、総出現数上位 topk の文脈のみ表示。
        as_prob=True の場合は各行で正規化して確率を描画。
    引数:
        counts  : 形状 (N, N, ..., N)（最後の軸が次要素）のカウント配列
        labels  : 要素ラベル（長さN）。Noneなら0..N-1
        topk    : 表示する文脈の上位件数
        as_prob : Trueで確率化、Falseでカウントのまま
        title   : タイトル。Noneなら既定
        annotate: Trueでセルに値（確率/カウント）を注記
        fmt     : 注記の書式（".1%" など）
    出力:
        {SAVE_DIR}/fig/order-<order>/heatmap.png に保存（表示はしない）。
    備考:
        行数が多い場合、注記は視認性低下のため既定で無効。
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
    def decode_row(r):
        ctx, x = [], int(r)
        for _ in range(ctx_dims):
            ctx.append(x % N); x //= N
        ctx = ctx[::-1]
        base = labels or list(range(N))
        return "→".join(str(base[k]) for k in ctx)
    row_labels = [decode_row(r) for r in idx]
    col_labels = labels or list(range(N))
    h = max(3.5, 0.35 * len(row_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8, h))
    im = ax.imshow(M, cmap=white_red,
                   vmin=(0.0 if as_prob else None),
                   vmax=(1.0 if as_prob else None),
                   aspect="auto")
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xticks(np.arange(N), labels=col_labels)
    ax.set_xlabel("Next element (N+1)")
    ax.set_ylabel(f"Context (length {ctx_dims})")
    plt.colorbar(im, ax=ax)
    ax.set_title(title or f"N-gram context → next ({'probabilities' if as_prob else 'counts'}, top-{topk})")
    if annotate:
        thresh = 0.6 if as_prob else np.nanpercentile(M, 85)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, format(v, fmt),
                        ha="center", va="center",
                        fontsize=9, color=("white" if v >= thresh else "black"))
    fig.tight_layout()
    _save_fig(fig, order=order)

# load sequences (expects 0..N-1 integers in "element")
for path in glob.glob(f"{DIR}/../../outputs/ClassifyElements/*/*.csv"):
    df = pd.read_csv(path, encoding="shift-jis")
    seqs.append(df["element"].to_numpy())

# bigram (probabilities + annotations)
C2 = ngram_counts_multi(seqs, N, order=2)
plot_bigram_heatmap_probs(
    C2, labels=list(range(N)),
    title="N→N+1 transition (probabilities)",
    annotate=True, fmt=".1%"
)
print("\nbigram saved\n")

# trigram (probabilities, top-k)
C3 = ngram_counts_multi(seqs, N, order=3)
plot_context_next_heatmap(
    C3, labels=list(range(N)), topk=18,
    as_prob=True, title="Trigram: context (len 2) → next (probabilities, top-18)",
    annotate=True, fmt=".1%"
)
print("\ntrigram saved\n")

# 4-gram (probabilities, top-k)
C4 = ngram_counts_multi(seqs, N, order=4)
plot_context_next_heatmap(
    C4, labels=list(range(N)), topk=24,
    as_prob=True, title="4-gram: context (len 3) → next (probabilities, top-24)",
    annotate=True, fmt=".1%"
)
print("\n4-gram saved\n")
#
print("\nDone\n")