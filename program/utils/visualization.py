
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def set_rcParams():
    
    plt.rcParams["figure.autolayout"] = False   # Whether to use automatic layout adjustment
    plt.rcParams["figure.subplot.left"] = 0.14  # Blank
    plt.rcParams["figure.subplot.bottom"] = 0.14# Blank
    plt.rcParams["figure.subplot.right"] =0.90  # Blank
    plt.rcParams["figure.subplot.top"] = 0.90   # Blank
    plt.rcParams["figure.subplot.wspace"] = 0.20# Horizontal spacing of figure
    plt.rcParams["figure.subplot.hspace"] = 0.20# Vertical spacing of figure

    plt.rcParams["font.family"] = "serif"       # Font
    
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 15  # Basic size of font
    # plt.rcParams["font.size"] = 20  # Basic size of font
    plt.rcParams["mathtext.cal"] = "serif"      # Font for TeX
    plt.rcParams["mathtext.rm"] = "serif"       # Font for TeX
    plt.rcParams["mathtext.it"] = "serif:italic"# Font for TeX
    plt.rcParams["mathtext.bf"] = "serif:bold"  # Font for TeX
    plt.rcParams["mathtext.fontset"] = "cm"     # Font for TeX

    plt.rcParams["xtick.direction"] = "in"      # Scale line orientation, inner "in" or outer "out" or both "inout"
    plt.rcParams["ytick.direction"] = "in"      # Scale line orientation, inner "in" or outer "out" or both "inout"
    plt.rcParams["xtick.top"] = True            # Whether to draw a scale line at the top
    plt.rcParams["xtick.bottom"] = True         # Whether to draw a scale line at the bottom
    plt.rcParams["ytick.left"] = True           # Whether to draw a scale line at the left
    plt.rcParams["ytick.right"] = True          # Whether to draw a scale line at the right
    plt.rcParams["xtick.major.size"] = 2.0      # x-axis main scale line length
    plt.rcParams["ytick.major.size"] = 2.0      # y-axis main scale line length
    plt.rcParams["xtick.major.width"] = 0.5     # x-axis main scale line width
    plt.rcParams["ytick.major.width"] = 0.5     # y-axis main scale line width
    plt.rcParams["xtick.minor.visible"] = False # whether to draw x-axis minor tick marks
    plt.rcParams["ytick.minor.visible"] = False # whether to draw y-axis minor tick marks
    plt.rcParams["xtick.minor.size"] = 1.0      # x-axis minor scale line length
    plt.rcParams["ytick.minor.size"] = 1.0      # y-axis minor scale line length
    plt.rcParams["xtick.minor.width"] = 0.3     # x-axis minor scale line width
    plt.rcParams["ytick.minor.width"] = 0.3     # y-axis minor scale line width
    # plt.rcParams["xtick.labelsize"] = 10  # Font size of graph x-scale
    # plt.rcParams["ytick.labelsize"] = 10  # Font size of graph y-scale
    plt.rcParams["xtick.labelsize"] = 15  # Font size of graph x-scale
    plt.rcParams["ytick.labelsize"] = 15  # Font size of graph y-scale

    plt.rcParams["axes.labelsize"] = 20  # Axis label font size
    # plt.rcParams["axes.labelsize"] = 20  # Axis label font size
    plt.rcParams["axes.linewidth"] = 0.5  # Line thickness around the graph
    plt.rcParams["axes.grid"] = True  # Whether to show the grid

    plt.rcParams["grid.color"] = "black"  # Grid color
    plt.rcParams["grid.linewidth"] = 0.05  # Grid width

    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams["legend.loc"] = "best"         # Legend position
    plt.rcParams["legend.frameon"] = True       # Whether to surround the legend
    plt.rcParams["legend.framealpha"] = 1.0     # Transmittance alpha
    plt.rcParams["legend.facecolor"] = "white"  # Background color
    plt.rcParams["legend.edgecolor"] = "black"  # Enclosure color
    plt.rcParams["legend.fancybox"] = False     # Set to True to round the four corners of the enclosure

class Colors:
    # accent
    red = [255 / 330, 75 / 330, 0.0]
    yellow = [255 / 496, 241 / 496, 0.0]
    green = [3 / 300, 175 / 300, 122 / 300]
    blue = [0 / 345, 90 / 345, 255 / 345]
    skyblue = [77 / 528, 196 / 528, 255 / 528]
    pink = [255/513, 128/513, 130/513]
    orange = [246/416, 170/416, 0]
    purple = [153/306, 0, 153/306]
    brown = [128/192, 64/192, 0]
    # monotone
    light_gray = [200.0/603.0, 200.0/603.0, 203.0/603.0]
    black = [0.0, 0.0, 0.0]
    # other
    facecolor_ppt = "#F8F8F8"  # facecolor in PowerPoint

def save_fig(dir, name):
    os.makedirs(f"{dir}png", exist_ok = True)
    plt.savefig(f"{dir}{name}.pdf", dpi = 100)
    plt.savefig(f"{dir}png/{name}.png", dpi = 100)
    print(f'\nFigure saved: \"{name}\"\n')
    plt.close()

def calc_ax_range(data, margin_rate):
    width = max(data) - min(data)
    margin = margin_rate * width / (1.0 - 2.0 * margin_rate)
    ax_min = min(data) - margin
    ax_max = max(data) + margin
    return ax_min, ax_max

def calc_fig_range_2D(data_h, data_v, margin, square_flag):
    #
    if square_flag:
        width = max(
            max(data_h) - min(data_h),
            max(data_v) - min(data_v),
        )
        h_ax_mid = 0.5 * (max(data_h) + min(data_h))
        v_ax_mid = 0.5 * (max(data_v) + min(data_v))
        h_ax_min = h_ax_mid - 0.5 * width - margin
        h_ax_max = h_ax_mid + 0.5 * width + margin
        v_ax_min = v_ax_mid - 0.5 * width - margin
        v_ax_max = v_ax_mid + 0.5 * width + margin
    else:
        h_ax_min = min(data_h) - margin
        h_ax_max = max(data_h) + margin
        v_ax_min = min(data_v) - margin
        v_ax_max = max(data_v) + margin
    #
    h_ax_range = [h_ax_min, h_ax_max]
    v_ax_range = [v_ax_min, v_ax_max]
    return h_ax_range, v_ax_range
