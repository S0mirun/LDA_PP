import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from numpy.lib.stride_tricks import sliding_window_view


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)
#
white_red = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#FF0000"])
#
seqs = []
N = 6

def ngram_counts_multi(seqs, N, order=2):
    """
    seqs: 1次元の整数ID列(0..N-1)のリスト（各シーケンス）
    N:    状態数
    order: n-gram の n（2=bigram, 3=trigram, ...）
    return: 形状 (N,)*order のカウント配列
    """
    shape = (N,)*order
    C = np.zeros(shape, dtype=np.int64)
    base = (N ** np.arange(order-1, -1, -1)).astype(np.int64)  # N進法の重み
    size_flat = N**order

    for s in seqs:
        a = np.asarray(s, dtype=np.int64)
        if a.size < order:
            continue
        W = sliding_window_view(a, order)           # [len-order+1, order]
        v = (W >= 0).all(axis=1) & (W < N).all(axis=1)
        if not np.any(v):
            continue
        idx_flat = (W[v] * base).sum(axis=1)        # 各ウィンドウを一意な整数にエンコード
        b = np.bincount(idx_flat, minlength=size_flat)
        C += b.reshape(shape)

    return C

# ---- 1) n-gramカウント（シーケンス: 整数IDの1次元配列） ----
def transition_counts(sequence, n_states, order=2):
    seq = np.asarray(sequence, dtype=int)
    assert order >= 2
    shape = (n_states,) * order
    counts = np.zeros(shape, dtype=np.int64)
    for t in range(len(seq) - order + 1):
        idx = tuple(seq[t + i] for i in range(order))
        counts[idx] += 1
    return counts  # 2要素→(N,N), 3要素→(N,N,N), 4要素→(N,N,N,N)

# ---- 2) 条件付き確率に変換（行=文脈、列=次要素） ----
def counts_to_conditional_probs(counts, alpha=0.0):
    # 最後の軸を「次の要素」とみなして正規化
    c = counts.astype(float)
    if alpha > 0:
        c = c + alpha
    denom = c.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.divide(c, denom, where=(denom > 0))
    P[np.isnan(P)] = 0.0
    return P

# ---- 3) bigram用 ヒートマップ（N×N） ----
def plot_bigram_heatmap(mat, labels=None, use_log=False, title="N→N+1 要素遷移"):
    data = np.array(mat, dtype=float)
    if use_log:
        data = np.where(data > 0, data, np.nanmin(data[data>0]) if np.any(data>0) else 1e-12)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(
        data,
        cmap=white_red,
        norm=LogNorm(vmin=np.nanmin(data[data>0])) if use_log and np.any(data>0) else None
    )
    n = data.shape[0]
    ax.set_xticks(np.arange(n), labels=(labels or range(n)))
    ax.set_yticks(np.arange(n), labels=(labels or range(n)))
    ax.set_xlabel("N+1 番目")
    ax.set_ylabel("N 番目")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ---- 4) 高次（3要素以上）: 文脈を行に畳み込んで Top-K をヒートマップ表示 ----
def plot_context_next_heatmap(counts, labels=None, topk=20, as_prob=True, use_log=False, title=None):
    # counts 形状: (N, N, ..., N) で最後の軸が「次の要素」
    order = counts.ndim
    N = counts.shape[0]
    ctx_dims = order - 1
    # 行 = 文脈（ctx_dims 個）をフラット化、列 = 次要素
    C = counts.reshape(N**ctx_dims, N)
    totals = C.sum(axis=1)
    # 出現上位の文脈だけ抽出
    idx = np.argsort(totals)[::-1][:topk]
    C_top = C[idx, :]
    # 確率化（各行で正規化）
    M = counts_to_conditional_probs(C_top, alpha=0.0) if as_prob else C_top.astype(float)

    # 行ラベルを整備
    def decode_row(r):
        ctx = []
        x = r
        for _ in range(ctx_dims):
            ctx.append(x % N)
            x //= N
        ctx = ctx[::-1]
        if labels:
            return "→".join(str(labels[k]) for k in ctx)
        else:
            return "→".join(str(k) for k in ctx)
    row_labels = [decode_row(int(r)) for r in idx]
    col_labels = labels or list(range(N))

    # ヒートマップ
    data = M
    if use_log:
        data = np.where(data > 0, data, np.nanmin(data[data>0]) if np.any(data>0) else 1e-12)

    h = max(3.5, 0.35*len(row_labels) + 1.5)  # 行数に応じて高さ伸縮
    fig, ax = plt.subplots(figsize=(8, h))
    im = ax.imshow(
        data,
        cmap=white_red,
        norm=LogNorm(vmin=np.nanmin(data[data>0])) if use_log and np.any(data>0) else None,
        aspect="auto"
    )
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xticks(np.arange(N), labels=col_labels, rotation=0)
    ax.set_xlabel("次の要素 (N+1)")
    ax.set_ylabel(f"文脈 (長さ {ctx_dims})")
    plt.colorbar(im, ax=ax)
    if title is None:
        what = "確率" if as_prob else "回数"
        title = f"N-gram 文脈→次要素ヒートマップ（{what}, top-{topk} 文脈）"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

for path in glob.glob(f"{DIR}/../../outputs/ClassifyElements/*/*.csv"):
    raw_df = pd.read_csv(
        path,
        encoding='shift-jis'
    )
    #
    seq = raw_df["element"].to_numpy()
    seqs.append(seq)
# 2要素（bigram）
C2 = ngram_counts_multi(seqs, N, order=2)   # 形状 (6,6)：総カウント
plot_bigram_heatmap(C2, labels=list(range(N)), use_log=False,
                    title="N→N+1 要素遷移（回数）")

# 3要素（trigram）
C3 = ngram_counts_multi(seqs, N, order=3)   # 形状 (6,6,6)
plot_context_next_heatmap(C3, labels=list(range(N)), topk=18,
                          as_prob=True, use_log=False,
                          title="Trigram: 文脈(長さ2) → 次の要素（確率, top-18）")


# 4要素（4-gram）
C4 = ngram_counts_multi(seqs, N, order=4)   # 形状 (6,6,6,6)
plot_context_next_heatmap(C4, labels=list(range(N)), topk=24,
                          as_prob=True, use_log=False,
                          title="4-gram: 文脈(長さ3) → 次の要素（確率, top-24）")

