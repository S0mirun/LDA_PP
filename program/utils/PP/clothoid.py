# -*- coding: utf-8 -*-
"""
clothoid.py
===========

クロソイド曲線導入方針（クロソイト_曲線導入方針_PathPlanning2.pdf）13.1節
「第1段階：単一屈曲点の幾何検証」に対応するモジュール。

対象コード（PathPlanning2.py）の `_find_best_fillet_arc(pt1, pt2, pt3, ...)` と
同じ入力形式（連続する3点 pt1, pt2, pt3）を想定し、円弧の代わりに

    進入直線 -> 曲率増加クロソイド -> 曲率減少クロソイド -> 退出直線

という bi-clothoid（クロソイド対）を生成する。

方針書 6.2 節にある通り、モジュール内部では
    - 点は (x, y) 順
    - theta = 0 は x軸正方向
    - 反時計回り(CCW)を正
という標準座標系のみを用いる。既存コードの (ver, hor) や船首方位 psi
（北0・時計回り正）との変換は、このモジュールの外（呼び出し側 / PathPlanning2.py
への組込み時＝方針書 6.3 節）で行う。したがって pt1, pt2, pt3 は
「平面上の3点」として渡せばよく、どちらの軸を第1成分にするかは呼び出し側の
一貫性にのみ依存する（本スクリプトの検証では素直に (x, y) として扱う）。

本ファイルは方針書の以下の節に対応する:
    3   節 各屈曲点で使用する幾何情報   -> compute_corner_geometry()
    4   節 最適化変数の選定             -> Lin, Lout (または Lc, eta) から kappa_peak を導出
    5   節 クロソイド対の配置方法       -> solve_trim_distances(), build_biclothoid()
    7   節 必要な制約条件               -> check_constraints()
    13.1 節 単一屈曲点の幾何検証        -> verify_biclothoid() + __main__ のテストケース
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid


# ----------------------------------------------------------------------
# 基本ベクトル演算 (PDF 3節)
# ----------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    """単位ベクトルを返す。"""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero-length vector has no direction")
    return v / n


def cross2d(u: np.ndarray, v: np.ndarray) -> float:
    """2次元外積 (スカラー)。"""
    return float(u[0] * v[1] - u[1] * v[0])


def signed_turn_angle(t_in: np.ndarray, t_out: np.ndarray) -> float:
    """
    PDF 3節: dpsi = atan2(t_in x t_out, t_in . t_out)
    左旋回(CCW)が正、右旋回(CW)が負。
    """
    return float(np.arctan2(cross2d(t_in, t_out), np.dot(t_in, t_out)))


def heading_of(v: np.ndarray) -> float:
    """標準座標系(x軸=0, CCW正)でのベクトルの方位角。"""
    return float(np.arctan2(v[1], v[0]))


def rot(theta: float) -> np.ndarray:
    """2x2 回転行列 (CCW正)。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------
# 屈曲点の幾何情報 (PDF 3節)
# ----------------------------------------------------------------------

@dataclass
class CornerGeometry:
    pt1: np.ndarray
    pt2: np.ndarray
    pt3: np.ndarray
    t_in: np.ndarray     # 進入方向 単位ベクトル
    t_out: np.ndarray    # 退出方向 単位ベクトル
    L1: float            # |pt2 - pt1|
    L2: float            # |pt3 - pt2|
    dpsi: float          # 符号付き変針角 [rad]


def compute_corner_geometry(pt1, pt2, pt3) -> CornerGeometry:
    pt1 = np.asarray(pt1, dtype=float)
    pt2 = np.asarray(pt2, dtype=float)
    pt3 = np.asarray(pt3, dtype=float)

    t_in = unit(pt2 - pt1)
    t_out = unit(pt3 - pt2)

    return CornerGeometry(
        pt1=pt1, pt2=pt2, pt3=pt3,
        t_in=t_in, t_out=t_out,
        L1=float(np.linalg.norm(pt2 - pt1)),
        L2=float(np.linalg.norm(pt3 - pt2)),
        dpsi=signed_turn_angle(t_in, t_out),
    )


# ----------------------------------------------------------------------
# クロソイド1区間の積分 (弧長パラメータ表示)
#     theta(s) = theta0 + kappa0*s + 0.5*sigma*s^2
#     kappa(s) = kappa0 + sigma*s
#     x(s) = x0 + ∫cos(theta(u))du,  y(s) = y0 + ∫sin(theta(u))du
# ----------------------------------------------------------------------

def _integrate_segment(x0, y0, theta0, kappa0, sigma, length, n):
    """
    局所座標(x0,y0)・局所方位theta0・曲率kappa0を起点として、
    曲率変化率sigmaで弧長lengthだけクロソイド区間を積分する。
    戻り値: s, x, y, theta, kappa (各 shape=(n,))
    """
    s = np.linspace(0.0, length, n)
    theta = theta0 + kappa0 * s + 0.5 * sigma * s ** 2
    kappa = kappa0 + sigma * s

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = x0 + cumulative_trapezoid(cos_t, s, initial=0.0)
    y = y0 + cumulative_trapezoid(sin_t, s, initial=0.0)

    return s, x, y, theta, kappa


# ----------------------------------------------------------------------
# クロソイド対 (bi-clothoid) の生成結果
# ----------------------------------------------------------------------

@dataclass
class BiClothoidResult:
    entry_point: np.ndarray      # Qin
    exit_point: np.ndarray       # Qout
    din: float                   # 進入直線からのトリム距離
    dout: float                  # 退出直線からのトリム距離
    kappa_peak: float
    sigma_in: float
    sigma_out: float
    p: np.ndarray                 # (N,2) 座標列 (Qin ... Qout)
    psi: np.ndarray               # (N,) 方位 (標準座標系, x軸=0, CCW正)
    kappa: np.ndarray             # (N,) 曲率
    sigma: np.ndarray             # (N,) 曲率変化率 (区間毎の値)
    s: np.ndarray                 # (N,) 累積弧長 (クロソイド開始点=0)
    geom: CornerGeometry
    Lin: float
    Lout: float


def solve_trim_distances(t_in, t_out, delta_p_c):
    """
    PDF 5節:
        [t_in  t_out] [din; dout] = delta_p_c
    を解いて din, dout を求める。
    """
    A = np.column_stack([t_in, t_out])  # 2x2
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        raise ValueError(
            "進入方向と退出方向がほぼ平行のため、din/doutを一意に決定できません"
            f" (det={det:.3e})。変針角が極端に小さい場合はクロソイド不要。"
        )
    din, dout = np.linalg.solve(A, delta_p_c)
    return float(din), float(dout)


def build_biclothoid(pt1, pt2, pt3, Lin: float, Lout: float, n_per_segment: int = 200) -> BiClothoidResult:
    """
    PDF 4, 5節に基づき、屈曲点(pt1, pt2, pt3)に対して
    進入クロソイド長Lin・退出クロソイド長Loutのbi-clothoidを生成する。

    手順:
      1. 変針角 dpsi を計算 (3節)
      2. kappa_peak = 2*dpsi / (Lin+Lout) (4節)
      3. 局所座標(原点・方位0・曲率0)を起点にクロソイド対を積分し、
         終端での局所変位・局所方位変化を得る
      4. 局所変位を進入方向tinの向きだけ回転させ、大域座標での相対変位 delta_p_c を得る
      5. delta_p_c = din*t_in + dout*t_out を解いて din, dout を求める (5節)
      6. Qin = pt2 - din*t_in, Qout = pt2 + dout*t_out を計算し、
         局所クロソイド座標列を回転・並行移動して大域座標に変換する
    """
    if Lin <= 0.0 or Lout <= 0.0:
        raise ValueError("Lin, Lout は正の値である必要があります")

    geom = compute_corner_geometry(pt1, pt2, pt3)
    dpsi = geom.dpsi

    if abs(dpsi) < 1e-9:
        raise ValueError("変針角がほぼ0のため、クロソイドを配置する必要がありません")

    kappa_peak = 2.0 * dpsi / (Lin + Lout)
    sigma_in = kappa_peak / Lin
    sigma_out = -kappa_peak / Lout

    # --- 局所座標での積分 (原点, 方位0, 曲率0を起点) -----------------
    s_a, x_a, y_a, theta_a, kappa_a = _integrate_segment(
        x0=0.0, y0=0.0, theta0=0.0, kappa0=0.0,
        sigma=sigma_in, length=Lin, n=n_per_segment,
    )

    s_b, x_b, y_b, theta_b, kappa_b = _integrate_segment(
        x0=x_a[-1], y0=y_a[-1], theta0=theta_a[-1], kappa0=kappa_a[-1],
        sigma=sigma_out, length=Lout, n=n_per_segment,
    )

    # 局所座標での終端変位・終端方位変化
    local_end = np.array([x_b[-1], y_b[-1]])
    theta_end_local = theta_b[-1]

    # --- 大域座標への回転 (進入方向 t_in の方位だけ回転) --------------
    theta_in_global = heading_of(geom.t_in)
    delta_p_c = rot(theta_in_global) @ local_end

    # --- トリム距離 din, dout を解く (5節) ---------------------------
    din, dout = solve_trim_distances(geom.t_in, geom.t_out, delta_p_c)

    entry_point = geom.pt2 - din * geom.t_in
    exit_point = geom.pt2 + dout * geom.t_out

    # --- 局所座標列を大域座標へ変換 -----------------------------------
    x_local = np.concatenate([x_a, x_b[1:]])
    y_local = np.concatenate([y_a, y_b[1:]])
    theta_local = np.concatenate([theta_a, theta_b[1:]])
    kappa_all = np.concatenate([kappa_a, kappa_b[1:]])
    s_all = np.concatenate([s_a, s_a[-1] + s_b[1:]])

    local_pts = np.column_stack([x_local, y_local])
    global_pts = (rot(theta_in_global) @ local_pts.T).T + entry_point
    psi_global = theta_local + theta_in_global

    sigma_all = np.where(s_all <= Lin, sigma_in, sigma_out)

    return BiClothoidResult(
        entry_point=entry_point,
        exit_point=exit_point,
        din=din,
        dout=dout,
        kappa_peak=kappa_peak,
        sigma_in=sigma_in,
        sigma_out=sigma_out,
        p=global_pts,
        psi=psi_global,
        kappa=kappa_all,
        sigma=sigma_all,
        s=s_all,
        geom=geom,
        Lin=Lin,
        Lout=Lout,
    )


# ----------------------------------------------------------------------
# 制約条件チェック (PDF 7節)
# ----------------------------------------------------------------------

def check_constraints(result: BiClothoidResult, min_turn_radius: float | None = None,
                       sigma_max: float | None = None) -> dict:
    """
    PDF 7節の制約を判定する。
      7.1 最大曲率      |kappa_peak| <= 1/min_turn_radius
      7.2 最大曲率変化率 |sigma_in|, |sigma_out| <= sigma_max
      7.3 線分長とクロソイドの重複  0 <= din < L1,  0 <= dout < L2
    """
    checks = {}

    if min_turn_radius is not None:
        kappa_max = 1.0 / min_turn_radius
        checks["max_curvature_ok"] = abs(result.kappa_peak) <= kappa_max
        checks["kappa_max_allowed"] = kappa_max
    else:
        checks["max_curvature_ok"] = None

    if sigma_max is not None:
        checks["sigma_in_ok"] = abs(result.sigma_in) <= sigma_max
        checks["sigma_out_ok"] = abs(result.sigma_out) <= sigma_max
    else:
        checks["sigma_in_ok"] = None
        checks["sigma_out_ok"] = None

    checks["din_in_range"] = 0.0 <= result.din < result.geom.L1
    checks["dout_in_range"] = 0.0 <= result.dout < result.geom.L2

    return checks


# ----------------------------------------------------------------------
# 13.1節 単一屈曲点の幾何検証
# ----------------------------------------------------------------------

def verify_biclothoid(result: BiClothoidResult, tol: float = 1e-6) -> dict:
    """
    PDF 13.1節「確認項目」を数値的に検証する。
      1. 開始点(entry_point)が進入線 pt1->pt2 上にある
      2. 終了点(exit_point)が退出線 pt2->pt3 上にある
      3. 始端・終端の接線方向が各直線と一致する
      4. 曲率が 0 -> kappa_peak -> 0 となる
      5. 左右旋回の符号が正しい (kappa_peak の符号 == dpsi の符号)
    """
    geom = result.geom
    report = {}

    # 1. entry_point が pt1->pt2 の直線上にあるか (点と直線の距離)
    def point_to_line_dist(pt, line_pt, line_dir):
        rel = pt - line_pt
        proj = np.dot(rel, line_dir) * line_dir
        return float(np.linalg.norm(rel - proj))

    report["entry_on_line1"] = point_to_line_dist(result.entry_point, geom.pt1, geom.t_in) < tol
    # 2. exit_point が pt2->pt3 の直線上にあるか
    report["exit_on_line2"] = point_to_line_dist(result.exit_point, geom.pt2, geom.t_out) < tol

    # 3. 始端・終端の接線方向
    #    始端はクロソイド構成上厳密に t_in と一致する(回転の基準そのもの)。
    #    終端は theta_in_global + dpsi が t_out の方位と一致するかで判定する。
    theta_in_global = heading_of(geom.t_in)
    theta_out_expected = heading_of(geom.t_out)
    theta_out_actual = result.psi[-1]
    # 角度差を[-pi, pi]に正規化
    d = (theta_out_actual - theta_out_expected + np.pi) % (2 * np.pi) - np.pi
    report["start_tangent_matches_t_in"] = abs(result.psi[0] - theta_in_global) < 1e-8
    report["end_tangent_matches_t_out"] = abs(d) < 1e-6

    # 4. 曲率プロファイル: 端点で0、ピークで|kappa_peak|
    report["kappa_starts_at_zero"] = abs(result.kappa[0]) < 1e-8
    report["kappa_ends_at_zero"] = abs(result.kappa[-1]) < 1e-8
    peak_idx = np.argmax(np.abs(result.kappa))
    report["kappa_peak_reached"] = abs(abs(result.kappa[peak_idx]) - abs(result.kappa_peak)) < 1e-6

    # 5. 左右旋回の符号
    report["turn_sign_correct"] = (
        np.sign(result.kappa_peak) == np.sign(geom.dpsi) or abs(geom.dpsi) < 1e-9
    )

    report["all_pass"] = all(
        v for k, v in report.items() if isinstance(v, (bool, np.bool_))
    )
    return report


# ----------------------------------------------------------------------
# デモ / 検証スクリプト本体
#   PDF 13.1節に列挙された4パターンをすべて確認する:
#     - 左旋回と右旋回
#     - 小さな変針角
#     - 90度前後の変針角
#     - 進入線・退出線の長さが異なる場合
# ----------------------------------------------------------------------

def _print_case(title, pt1, pt2, pt3, Lin, Lout, min_turn_radius=None, sigma_max=None):
    print(f"\n===== {title} =====")
    geom = compute_corner_geometry(pt1, pt2, pt3)
    print(f"  pt1={pt1}, pt2={pt2}, pt3={pt3}")
    print(f"  L1={geom.L1:.2f}, L2={geom.L2:.2f}, "
          f"dpsi={np.degrees(geom.dpsi):+.2f} deg")

    result = build_biclothoid(pt1, pt2, pt3, Lin, Lout)
    print(f"  Lin={Lin:.2f}, Lout={Lout:.2f}, "
          f"kappa_peak={result.kappa_peak:+.5f} (1/m), "
          f"R_min_equiv={1.0/abs(result.kappa_peak):.1f} m")
    print(f"  sigma_in={result.sigma_in:+.6f}, sigma_out={result.sigma_out:+.6f}")
    print(f"  din={result.din:.3f} (<{geom.L1:.2f}), dout={result.dout:.3f} (<{geom.L2:.2f})")
    print(f"  entry_point={result.entry_point}, exit_point={result.exit_point}")

    report = verify_biclothoid(result)
    for k, v in report.items():
        mark = "OK" if v else "NG"
        print(f"    [{mark}] {k}: {v}")

    cons = check_constraints(result, min_turn_radius=min_turn_radius, sigma_max=sigma_max)
    print(f"  constraints: {cons}")

    return result


def _plot_cases(results, save_path="clothoid_stage1_verification.png"):
    import matplotlib.pyplot as plt
    import matplotlib

    for jp_font in ("Noto Sans CJK JP", "IPAexGothic", "TakaoGothic"):
        if any(jp_font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = jp_font
            break

    n = len(results)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, result) in zip(axes, results):
        geom = result.geom
        # original polyline
        polyline = np.vstack([geom.pt1, geom.pt2, geom.pt3])
        ax.plot(polyline[:, 0], polyline[:, 1], "--", color="gray", lw=1.0, label="original polyline")

        # entry line (pt1 -> entry_point), exit line (exit_point -> pt3)
        ax.plot([geom.pt1[0], result.entry_point[0]], [geom.pt1[1], result.entry_point[1]],
                color="tab:blue", lw=2.0, label="entry line")
        ax.plot([result.exit_point[0], geom.pt3[0]], [result.exit_point[1], geom.pt3[1]],
                color="tab:green", lw=2.0, label="exit line")

        # bi-clothoid
        ax.plot(result.p[:, 0], result.p[:, 1], color="tab:red", lw=2.5, label="bi-clothoid")

        ax.scatter(*result.entry_point, color="k", zorder=5, s=25)
        ax.scatter(*result.exit_point, color="k", zorder=5, s=25)
        ax.scatter(*geom.pt2, color="orange", marker="x", zorder=5, s=40, label="corner pt2")

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\n[plot] saved verification figure -> {save_path}")


if __name__ == "__main__":
    results = []

    # 1) 左旋回 (小さめの変針角, 対称)
    r = _print_case(
        "左回頭",
        pt1=[0.0, 0.0], pt2=[100.0, 0.0], pt3=[30.0, 50.0],
        Lin=25.0, Lout=25.0,
        min_turn_radius=150.0, sigma_max=0.01,
    )
    results.append(("回頭", r))

    # 2) 右旋回 (90度前後)
    r = _print_case(
        "右旋回・90度前後",
        pt1=[0.0, 0.0], pt2=[100.0, 0.0], pt3=[100.0, -90.0],
        Lin=30.0, Lout=30.0,
        min_turn_radius=150.0, sigma_max=0.02,
    )
    results.append(("右旋回・90度前後", r))

    # 3) 左旋回・非対称 (進入線・退出線の長さが異なる, Lin != Lout)
    r = _print_case(
        "左旋回・非対称(Lin != Lout)",
        pt1=[0.0, 0.0], pt2=[80.0, 0.0], pt3=[80.0, 60.0],
        Lin=15.0, Lout=35.0,
        min_turn_radius=150.0, sigma_max=0.02,
    )
    results.append(("左旋回・非対称(Lin != Lout)", r))

    # 4) 極端に短い進入・退出線 (7.3節: din, doutが線分長を超えないかの確認)
    r = _print_case(
        "短い直線区間 (制約7.3の確認)",
        pt1=[0.0, 0.0], pt2=[40.0, 0.0], pt3=[70.0, 40.0],
        Lin=18.0, Lout=18.0,
        min_turn_radius=150.0, sigma_max=0.02,
    )
    results.append(("短い直線区間 (制約7.3の確認)", r))

    try:
        _plot_cases(results)
    except Exception as e:  # pragma: no cover
        print(f"[plot] skipped ({e})")

    n_pass = sum(1 for _, r in results if verify_biclothoid(r)["all_pass"])
    print(f"\n===== 検証結果: {n_pass}/{len(results)} ケースが全チェックに合格 =====")
