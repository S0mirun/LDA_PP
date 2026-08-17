import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
import numpy as np

from utils.LDA.ship_geometry import *


LEGEND_TRAFFIC_LANE = Patch(facecolor='magenta', alpha=0.25, edgecolor='none', label='Traffic Lane')

LEGEND_BUOY = Line2D([0], [0], marker='o', color='none',
                     markerfacecolor='orange', markeredgecolor='none', markersize=1.5, label='Buoy')

LEGEND_BUOY_LINE = Line2D([0], [0], color='orange', linestyle="-", linewidth=2, label='Buoy line')

LEGEND_CANDIDATE_LINES = Line2D([0], [0], color='red', linestyle="-", linewidth=2, label='Candidate Line Set')

LEGEND_CAPTAIN_ROUTE = Line2D([0], [0], color='gray', alpha=0.3,
                              linewidth=2, marker='D', markersize=1.5, label='Captain Route')

LEGEND_PLANNED_PATH = Line2D([0], [0], color='blue', linestyle='--',
                             marker='o', markersize=1.5, linewidth=2.5, label='Planned Path')

LEGEND_SHIP_SHAPE = Line2D([0], [0], color='black', linewidth=1.2, label='Ship shape (1 min interval)')

BBOX_PAD_IN = 0.03  # 保存画像の外周に残す最小余白（インチ）。小さいほど余白が減る。


def _axes_only_width_in(fig, ax):
    """
    legend を除いた ax 自体（目盛りラベル等を含む）の描画幅を inch 単位で返す。
    legend の横幅をこの値以下に収めるための基準として使う。
    """
    legend = ax.get_legend()
    was_visible = legend is not None and legend.get_visible()
    if legend is not None:
        legend.set_visible(False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_in = ax.get_tightbbox(renderer=renderer).transformed(fig.dpi_scale_trans.inverted())

    if legend is not None:
        legend.set_visible(was_visible)
    return bbox_in.x1 - bbox_in.x0


def _place_legend_fit_width(fig, ax, legends, fontsize, ncol_max, anchor_y=-0.02, min_ncol=1):
    """
    legends を ax の下部中央に配置する。
    legend の横幅が ax 本体の横幅を超える場合は、ncol_max から min_ncol まで
    ncol を自動的に減らしながら再配置し、「legend 幅 <= ax 幅」になる
    （または min_ncol まで減らしても収まらない場合はそこで諦める）
    ncol を採用する。

    既存の legend があれば作り直す前に取り除く。
    """
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    if not legends:
        return None

    max_width_in = _axes_only_width_in(fig, ax)
    legend_kwargs = dict(loc='upper center', bbox_to_anchor=(0.5, anchor_y),
                         fontsize=fontsize, frameon=True, framealpha=0.9, edgecolor='black')

    ncol_start = max(min_ncol, min(ncol_max, len(legends)))
    legend = None
    for ncol in range(ncol_start, min_ncol - 1, -1):
        if legend is not None:
            legend.remove()
        legend = ax.legend(handles=legends, bbox_transform=ax.transAxes, ncol=ncol, **legend_kwargs)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width_in = legend.get_window_extent(renderer=renderer) \
                         .transformed(fig.dpi_scale_trans.inverted()).width
        if width_in <= max_width_in or ncol == min_ncol:
            break
    return legend


def _compute_save_bbox(fig, ax, pad_in=BBOX_PAD_IN):
    """
    ax（+ 現在表示されている legend や text 等の子artist）の
    実際の描画結果をそのまま tight に囲む bbox を返す。
    legend が ax 幅を超えないように配置されている前提なので、
    横幅は常に ax 本体の幅で決まり、複数の図を並べたときの
    メイン画像サイズが揃う。
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_in = ax.get_tightbbox(renderer=renderer).transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(bbox_in.x0 - pad_in, bbox_in.y0 - pad_in,
                              bbox_in.x1 + pad_in, bbox_in.y1 + pad_in)


BUOY_SCATTER_KWARGS = dict(color='orange', s=20, zorder=2)

BUOY_COLOR_LIST = ["white", "black", "red", "green", "blue", "yellow"]
BUOY_COLOR_SCATTER_KWARGS = dict(s=10, zorder=3)

START_END_SCATTER_KWARGS = dict(c="black", s=10, zorder=11)
START_END_ANNOTATE_FONTSIZE = 25

SHIP_SHAPE_KWARGS = dict(facecolor='none', edgecolor='black', linewidth=1.2, alpha=0.9, zorder=9)

OPTIMIZATION_LINE_KWARGS = dict(color="red", lw=1.5, zorder=6)
OPTIMIZATION_SHIP_SHAPE_KWARGS = dict(facecolor="red", edgecolor="red", linewidth=1.0, alpha=0.5, zorder=6)


def save_fig(fig, ax, save_dir, name, legends, handles, pdf=False, pdf_dir=None,
             fontsize=12, ncol_max=4):
    handles.extend(legends)

    _place_legend_fit_width(fig, ax, legends, fontsize=fontsize, ncol_max=ncol_max)

    bbox = _compute_save_bbox(fig, ax)
    fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=400, bbox_inches=bbox)
    if pdf and pdf_dir is not None:
        fig.savefig(os.path.join(pdf_dir, f"{name}.pdf"), dpi=400, bbox_inches=bbox)

    if handles:
        for h in list(handles):
            try:
                if h is not None and h.axes is not None:
                    h.remove()
            except ValueError:
                pass
        handles.clear()


def draw_ship_shapes(ax, ship_shapes):
    """
    ship_shapes ((N,2)座標配列のリスト。1要素=1隻分の船体多角形) を ax 上に描画する。
    戻り値は生成された Patch のリスト（呼び出し側で handles による後片付けに使う）。
    """
    patches = []
    for hull_xy in ship_shapes:
        patch = ax.fill(hull_xy[:, 0], hull_xy[:, 1], **SHIP_SHAPE_KWARGS)[0]
        patches.append(patch)
    return patches


def save_lines(fig, ax, lines, handles, save_dir, name, legends, pdf=False, pdf_dir=None):
    """
    lines (Line インスタンスのリスト) を描画してから save_fig で保存する。
    """
    line_kwargs = dict(color="red", linestyle='-')

    for ln in lines:
        pts = np.vstack([ln.fixed_pt, ln.end_pt])
        h, = ax.plot(pts[:, 1], pts[:, 0], **line_kwargs)
        handles.append(h)

    save_fig(fig, ax, save_dir, name, legends, handles, pdf=pdf, pdf_dir=pdf_dir)


def save_pts(fig, ax, pts, pp_start, pp_end, handles, save_dir, name, legends,
             pdf=False, pdf_dir=None, pt_size=20):
    """
    経路点 pts (+ start/end) を散布図・線として描画してから save_fig で保存する。
    """
    scatter_kwargs = dict(c="blue", s=pt_size, zorder=10)
    line_kwargs = dict(c="blue", ls="--", alpha=0.5, zorder=10)

    full_pts = np.vstack([pp_start, pts, pp_end])
    h1 = ax.scatter(full_pts[:, 1], full_pts[:, 0], **scatter_kwargs)
    h2, = ax.plot(full_pts[:, 1], full_pts[:, 0], **line_kwargs)
    handles.extend([h1, h2])

    save_fig(fig, ax, save_dir, name, legends, handles, pdf=pdf, pdf_dir=pdf_dir)

def save_optimization_fig(fig, ax, pts, phis, boundary_pt, boundary_phi, goal, goal_phi,
                           handles, save_dir, name, legends, L, B, pdf=False, pdf_dir=None):
    """
    最適化の途中経過(restartごとの最良解)や最終結果を描画してから保存する。
    """
    full_pts = np.vstack([boundary_pt, pts, goal])
    full_phis = np.concatenate([[boundary_phi], np.asarray(phis, dtype=float), [goal_phi]])

    h_line, = ax.plot(full_pts[:, 1], full_pts[:, 0], **OPTIMIZATION_LINE_KWARGS)
    handles.append(h_line)

    for (ver, hor), psi in zip(full_pts, full_phis):
        hull = np.asarray(ship_shape_poly((ver, hor, psi), L=L, B=B))
        patch = ax.fill(hull[:, 0], hull[:, 1], **OPTIMIZATION_SHIP_SHAPE_KWARGS)[0]
        handles.append(patch)

    save_fig(fig, ax, save_dir, name, legends, handles, pdf=pdf, pdf_dir=pdf_dir)


def make_config_text(approach_algo_name, supplement_mode_name, redraw_by_AI):
    """
    最終結果図の下部に表示する設定サマリーのテキストを作る。
    """
    ai_str = "ON" if redraw_by_AI else "OFF"
    return f"Approach: {approach_algo_name}   /   Supplement: {supplement_mode_name}   /   AI redraw: {ai_str}"


def setup_result_legends():
    """
    最終結果図 (save_result_fig) 専用の凡例ハンドルを作る。
    """
    return [
        Line2D([0], [0], label="Buoy", color='none',
               marker='o', markersize=1.5, markerfacecolor='orange', markeredgecolor='orange'),
        Line2D([0], [0], label="Way points", color='none',
               marker='X', markersize=1.5, markerfacecolor="#8A2BE2", markeredgecolor="#8A2BE2", markeredgewidth=0.8),
        Line2D([0], [0], label="Captain's route", color='gray',
               marker='D', markersize=1.5, ls='-', lw=1.0, alpha=0.5),
        Line2D([0], [0], label="Generated Path", color='blue',
               marker='o', markersize=1.5, ls='--', lw=1.0, alpha=0.5),
        LEGEND_SHIP_SHAPE,
    ]


def save_result_fig(fig, ax, save_dir_path, file_name, pp_start, pp_end, result_pts, way_points,
                     approach_algo_name, supplement_mode_name, redraw_by_AI,
                     fontsize=10, ncol_max=3):
    """
    最終的な経路結果を1枚の図として保存する。
    """
    scatter_kwargs = dict(c="blue", s=5, zorder=10)
    line_kwargs = dict(c="blue", ls="--", alpha=0.5, zorder=10)
    wp_scatter_kwargs = dict(c="#8A2BE2", marker="X", edgecolors="#8A2BE2", linewidths=0.8, s=20, zorder=10)
    text_pos = (0.5, -0.01)
    text_kwargs = dict(ha='center', va='top', fontsize=12)

    SAVE_DIR = f"{save_dir_path}/results"

    full_pts = np.vstack([pp_start, result_pts, pp_end])
    ax.scatter(full_pts[:, 1], full_pts[:, 0], **scatter_kwargs)
    ax.plot(full_pts[:, 1], full_pts[:, 0], **line_kwargs)

    ax.scatter(way_points[:, 1], way_points[:, 0], **wp_scatter_kwargs)

    config_text = make_config_text(approach_algo_name, supplement_mode_name, redraw_by_AI)
    ax.text(text_pos[0], text_pos[1], config_text, transform=ax.transAxes, **text_kwargs)

    legends = setup_result_legends()
    # config_text は legend よりさらに下にあるので、legend の anchor を少し高めにしておく
    _place_legend_fit_width(fig, ax, legends, fontsize=fontsize, ncol_max=ncol_max, anchor_y=-0.03)

    bbox = _compute_save_bbox(fig, ax)
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, f"{file_name}.png"), dpi=400, bbox_inches=bbox)