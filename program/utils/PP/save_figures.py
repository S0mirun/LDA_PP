"""
描画結果の保存を担当するモジュール。

PathPlanning.py から import して使用する。
図のサイズ・軸範囲など figure 自体のセットアップは PathPlanning.py 側の
setup_figure() の仕様に従う（本モジュールでは新たに figure を作成しない）。

各図の legend・点群の描画スタイル・text・fontsize・savename などの
「見た目」に関する要素はこのファイル内で一元管理する。
PathPlanning.py 側は、ここで定義した定数・関数を名前で参照するだけでよい。
"""

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


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


BUOY_SCATTER_KWARGS = dict(color='orange', s=20, zorder=2)

BUOY_COLOR_LIST = ["white", "black", "red", "green", "blue", "yellow"]
BUOY_COLOR_SCATTER_KWARGS = dict(s=10, zorder=3)

START_END_SCATTER_KWARGS = dict(c="black", s=10, zorder=11)
START_END_ANNOTATE_FONTSIZE = 25

SHIP_SHAPE_KWARGS = dict(facecolor='none', edgecolor='black', linewidth=1.2, alpha=0.9, zorder=9)


def save_fig(fig, ax, save_dir, name, legends, handles, pdf=False, pdf_dir=None):
    """
    現在の legends/handles を使って凡例を描画し、png（必要なら pdf も）に保存する。
    保存後、handles に登録された描画物（デバッグ用の一時的な線や点）は figure から取り除く。

    legend は項目数が増えても経路の描画と重ならないよう、図の外(下)に配置する。
    """
    legend_kwargs = dict(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4,
                          fontsize=12, frameon=True, framealpha=0.9, edgecolor='black')

    handles.extend(legends)
    ax.legend(handles=legends, bbox_transform=ax.transAxes, **legend_kwargs)

    fig.savefig(os.path.join(save_dir, f"{name}.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    if pdf and pdf_dir is not None:
        fig.savefig(os.path.join(pdf_dir, f"{name}.pdf"),
                    dpi=400, bbox_inches="tight", pad_inches=0.05)

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
                     approach_algo_name, supplement_mode_name, redraw_by_AI):
    """
    最終的な経路結果を1枚の図として保存する。
    船型は _draw_captain_path 側で毎回描画されるため、ここでは扱わない。
    """
    scatter_kwargs = dict(c="blue", s=5, zorder=10)
    line_kwargs = dict(c="blue", ls="--", alpha=0.5, zorder=10)
    wp_scatter_kwargs = dict(c="#8A2BE2", marker="X", edgecolors="#8A2BE2", linewidths=0.8, s=20, zorder=10)
    text_pos = (0.5, -0.01)
    text_kwargs = dict(ha='center', va='top', fontsize=12)
    legend_kwargs = dict(loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=3,
                          fontsize=10, frameon=True, fancybox=False, edgecolor='black')

    SAVE_DIR = f"{save_dir_path}/results"

    full_pts = np.vstack([pp_start, result_pts, pp_end])
    ax.scatter(full_pts[:, 1], full_pts[:, 0], **scatter_kwargs)
    ax.plot(full_pts[:, 1], full_pts[:, 0], **line_kwargs)

    ax.scatter(way_points[:, 1], way_points[:, 0], **wp_scatter_kwargs)

    config_text = make_config_text(approach_algo_name, supplement_mode_name, redraw_by_AI)
    ax.text(text_pos[0], text_pos[1], config_text, transform=ax.transAxes, **text_kwargs)

    legends = setup_result_legends()
    ax.legend(handles=legends, bbox_transform=ax.transAxes, **legend_kwargs)

    plt.subplots_adjust(bottom=0.10)
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, f"{file_name}.png"),
                dpi=400, bbox_inches="tight", pad_inches=0.05)
