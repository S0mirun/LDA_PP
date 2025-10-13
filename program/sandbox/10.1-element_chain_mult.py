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
os.makedirs(osp.join(SAVE_DIR, "csv"), exist_ok=True)

# colormap: white -> red
white_red = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#FF0000"])

N = 6

def _save_fig(fig, order, filename="heatmap.png"):
    """
    図を保存して閉じる。
    保存先: {SAVE_DIR}/fig
    """
    out_dir = osp.join(SAVE_DIR, "fig")
    os.makedirs(out_dir, exist_ok=True)
    root, ext = osp.splitext(filename)
    ext = ext if ext else ".png"
    out_name = f"order-{order}_{root}{ext}"
    fig.savefig(osp.join(out_dir, out_name), dpi=150)
    plt.close(fig)

def _save_csv(df: pd.DataFrame, order, filename):
    """
    DataFrameをCSV保存する。
    保存先: {SAVE_DIR}/csv
    """
    out_dir = osp.join(SAVE_DIR, "csv")
    os.makedirs(out_dir, exist_ok=True)
    root, ext = osp.splitext(filename)
    ext = ext if ext else ".csv"
    out_name = f"order-{order}_{root}{ext}"
    df.to_csv(osp.join(out_dir, out_name), encoding="utf-8-sig", index=True)

def ngram_counts_multi(seqs, N, order=2):
    """
    複数系列に対して n-gram（長さ=order）の総カウント行列を計算する。
    各系列は 0..N-1 の整数IDからなる1次元配列を前提とする。
    戻り値は形状 (N,)*order のndarray（各 n-gram の出現回数）。
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
    alpha>0 の場合は加法的平滑化（ラプラス平滑化）を行う。
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
                              annotate=True, fmt=".1%", filename="heatmap_probs.png"):
    """
    2要素（bigram）の確率ヒートマップを描画・保存する（セルに確率注記可）。
    保存先: {SAVE_DIR}/fig/order-2/{filename}
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
    _save_fig(fig, order=2, filename=filename)
    return P  # for export

def topk_context_matrix(counts, topk=20, as_prob=True):
    """
    高次 n-gram の文脈（最後の軸以外）をフラット化し、総出現数上位 topk の行のみ抽出する。
    as_prob=True の場合、各行を正規化して確率を返す。
    戻り値: (M, idx) ただし Mは [topk, N] 行列、idxは元の行インデックス。
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
    フラット化された文脈行インデックス（idx）を元の文脈系列に復元し、ラベル化して返す。
    戻り値はラベル列（各要素はタプル）。
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

def probs_to_dict_order2(P, labels=None, round_ndigits=6):
    """
    2要素（bigram）の確率行列を入れ子辞書に変換する。
    戻り値: {row_label: {col_label: prob}}
    """
    n = P.shape[0]
    labs = labels or list(range(n))
    d = {}
    for i in range(n):
        d_row = {}
        for j in range(n):
            d_row[labs[j]] = round(float(P[i, j]), round_ndigits)
        d[labs[i]] = d_row
    return d

def probs_to_dict_higher(M, idx, labels=None, round_ndigits=6):
    """
    高次（order>=3）の確率行列Mと行インデックスidxを入れ子辞書に変換する。
    戻り値: {context_tuple: {next_label: prob}}
    """
    n = M.shape[1]
    labs = labels or list(range(n))
    # ctx_dims を推定（len(context) は不明なので後で渡す想定が自然だが、ここでは idx→decodeを外で済ませる代わりに簡便化）
    # ここでは呼び出し側で decode 済みの context_labels を渡す方針にする
    raise NotImplementedError("Use probs_to_dict_higher_with_contexts")

def probs_to_dict_higher_with_contexts(M, context_labels, labels=None, round_ndigits=6):
    """
    上位文脈の確率行列Mと各行の文脈ラベル(context_labels)から入れ子辞書を作る。
    戻り値: {context_tuple: {next_label: prob}}
    """
    n = M.shape[1]
    labs = labels or list(range(n))
    d = {}
    for i, ctx in enumerate(context_labels):
        row = {}
        for j in range(n):
            row[labs[j]] = round(float(M[i, j]), round_ndigits)
        d[ctx] = row
    return d

def export_probs_module(probs_dict, path, func_name="get_transition_probs"):
    """
    確率辞書を返す関数を持つ .py モジュールを書き出す。
    例: from transition_probs import get_transition_probs
        probs = get_transition_probs()
    """
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# auto-generated\n")
        f.write(f"def {func_name}():\n")
        f.write("    return ")
        f.write(repr(probs_dict))
        f.write("\n")

def plot_context_next_heatmap(counts, labels=None, topk=20, as_prob=True,
                              title=None, annotate=False, fmt=".1%", filename="heatmap_probs_topk.png"):
    """
    高次 n-gram の文脈→次要素ヒートマップを描画・保存する（Top-K文脈のみ）。
    保存先: {SAVE_DIR}/fig/order-<order>/{filename}
    戻り値: (M, idx) ただし Mは確率行列[topk, N]、idxは元行インデックス。
    """
    order = counts.ndim
    N = counts.shape[0]
    ctx_dims = order - 1
    M, idx = topk_context_matrix(counts, topk=topk, as_prob=as_prob)
    # row labels
    row_labels = _decode_row_indices(idx, N=N, ctx_dims=ctx_dims, labels=labels)
    col_labels = labels or list(range(N))
    # plot
    h = max(3.5, 0.35 * len(row_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8, h))
    im = ax.imshow(M, cmap=white_red, vmin=(0.0 if as_prob else None), vmax=(1.0 if as_prob else None), aspect="auto")
    ax.set_yticks(np.arange(len(row_labels)), labels=["→".join(map(str, r)) for r in row_labels])
    ax.set_xticks(np.arange(N), labels=col_labels)
    ax.set_xlabel("Next element (N+1)")
    ax.set_ylabel(f"Context (length {ctx_dims})")
    plt.colorbar(im, ax=ax)
    ax.set_title(title or f"N-gram context → next (probabilities, top-{topk})")
    if annotate:
        thresh = 0.6 if as_prob else np.nanpercentile(M, 85)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, format(v, fmt),
                        ha="center", va="center",
                        fontsize=9, color=("white" if v >= thresh else "black"))
    fig.tight_layout()
    _save_fig(fig, order=order, filename=filename)
    return M, idx, row_labels, col_labels

# --------- public API (import 用) ---------
def build_transition_probs(seqs, N=6, labels=None, topk3=18, topk4=24, alpha=0.0):
    """
    複数系列から n-gram 確率辞書を構築する（2,3,4-gram）。
    戻り値: {'order_2': {...}, 'order_3_topk': {...}, 'order_4_topk': {...}}
    """
    labels = labels or list(range(N))
    # bigram
    C2 = ngram_counts_multi(seqs, N, order=2)
    P2 = counts_to_conditional_probs(C2, alpha=alpha)
    d2 = probs_to_dict_order2(P2, labels=labels)

    # trigram
    C3 = ngram_counts_multi(seqs, N, order=3)
    M3, idx3 = topk_context_matrix(C3, topk=topk3, as_prob=True)
    ctx3 = _decode_row_indices(idx3, N=N, ctx_dims=2, labels=labels)
    d3 = probs_to_dict_higher_with_contexts(M3, ctx3, labels=labels)

    # fourgram
    C4 = ngram_counts_multi(seqs, N, order=4)
    M4, idx4 = topk_context_matrix(C4, topk=topk4, as_prob=True)
    ctx4 = _decode_row_indices(idx4, N=N, ctx_dims=3, labels=labels)
    d4 = probs_to_dict_higher_with_contexts(M4, ctx4, labels=labels)

    return {
        "order_2": d2,
        f"order_3_top{topk3}": d3,
        f"order_4_top{topk4}": d4,
    }

# --------- script mode ---------
if __name__ == "__main__":
    """
    CSVを読み込み、確率CSVとヒートマップPNGを保存し、確率辞書を.pyモジュールとして書き出す。
    """
    seqs = []
    for path in glob.glob(f"{DIR}/../../outputs/ClassifyElements/*/*.csv"):
        df = pd.read_csv(path, encoding="shift-jis")
        seqs.append(df["element"].to_numpy())

    labels = list(range(N))

    # bigram: heatmap + CSV
    C2 = ngram_counts_multi(seqs, N, order=2)
    P2 = plot_bigram_heatmap_probs(
        C2, labels=labels, title="N→N+1 transition (probabilities)",
        annotate=True, fmt=".1%", filename="heatmap_probs.png"
    )
    df_p2 = pd.DataFrame(P2, index=labels, columns=labels)
    _save_csv(df_p2, order=2, filename="probs.csv")

    # trigram: heatmap(top-k) + CSV(flat)
    C3 = ngram_counts_multi(seqs, N, order=3)
    M3, idx3, row_labels_3, col_labels_3 = plot_context_next_heatmap(
        C3, labels=labels, topk=18, as_prob=True,
        title="Trigram: context (len 2) → next (probabilities, top-18)",
        annotate=False, filename="heatmap_probs_top18.png"
    )
    rows = []
    for i, ctx in enumerate(row_labels_3):
        for j, nxt in enumerate(col_labels_3):
            rows.append({"context": "→".join(map(str, ctx)), "next": nxt, "prob": float(M3[i, j])})
    df_p3 = pd.DataFrame(rows)
    _save_csv(df_p3, order=3, filename="probs_top18.csv")

    # fourgram: heatmap(top-k) + CSV(flat)
    C4 = ngram_counts_multi(seqs, N, order=4)
    M4, idx4, row_labels_4, col_labels_4 = plot_context_next_heatmap(
        C4, labels=labels, topk=24, as_prob=True,
        title="4-gram: context (len 3) → next (probabilities, top-24)",
        annotate=False, filename="heatmap_probs_top24.png"
    )
    rows = []
    for i, ctx in enumerate(row_labels_4):
        for j, nxt in enumerate(col_labels_4):
            rows.append({"context": "→".join(map(str, ctx)), "next": nxt, "prob": float(M4[i, j])})
    df_p4 = pd.DataFrame(rows)
    _save_csv(df_p4, order=4, filename="probs_top24.csv")

    # export module for import
    probs = build_transition_probs(seqs, N=N, labels=labels, topk3=18, topk4=24, alpha=0.0)
    export_probs_module(
        probs,
        path=osp.join(SAVE_DIR, "transition_probs.py"),
        func_name="get_transition_probs"
    )
    print("\nDone\n")
