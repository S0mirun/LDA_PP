import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ----
plt.rcParams["font.family"] = "Times New Roman"

INPUT_FILE = "output/elements/20250120/all_ports_classified_elements_fixed.csv"
ELEMENTS = {1: "Going Straight", 2: "Moving Sideways", 3: "Drift", 4: "Pivot Turning", 5: "Turning"}
COLORS = {1: "lightgreen", 2: "lightblue", 3: "#F0B27A", 4: "plum", 5: "lightcoral"}

# speed bins: exclude < 0.5 knots
SPEED_BINS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, np.inf]
SPEED_LABELS = ["0.5-1.5", "1.5-2.5", "2.5-3.5", "3.5-4.5", "4.5-5.5",
                "5.5-6.5", "6.5-7.5", "7.5-8.5", "8.5-9.5", ">=9.5"]

# ---- I/O paths ----
today = datetime.now().strftime("%Y%m%d")
OUT_DIR = f"output/elements/{today}"
OUT_FILE = os.path.join(OUT_DIR, "element_proportion_stacked_bar_chart_with_percent(excluding_STOP).png")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load ----
df = pd.read_csv(INPUT_FILE)

# ---- Preprocess ----
# keep >= 0.5 knot and bin speeds
df = df.loc[df["knot"] >= 0.5].copy()
df["speed_category"] = pd.cut(
    df["knot"],
    bins=SPEED_BINS,
    labels=SPEED_LABELS,
    right=False,
    include_lowest=True,
    ordered=True
)

# ---- Aggregate to % per speed bin ----
counts = (
    df.groupby("speed_category", observed=False)["element"]
      .value_counts()
      .unstack(fill_value=0)
)

# ensure columns for elements 1..5 exist and are ordered
for e in range(1, 6):
    if e not in counts.columns:
        counts[e] = 0
counts = counts[[1, 2, 3, 4, 5]]

proportions = counts.div(counts.sum(axis=1), axis=0).fillna(0.0) * 100.0
proportions = proportions.reindex(SPEED_LABELS)  # keep x-order

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 12))  # (width, height)

bottom = np.zeros(len(SPEED_LABELS), dtype=float)
for e in [1, 2, 3, 4, 5]:
    ax.bar(
        SPEED_LABELS,
        proportions[e].to_numpy(),
        label=ELEMENTS[e],
        color=COLORS[e],
        bottom=bottom
    )
    bottom += proportions[e].to_numpy()

# axes
ax.set_xlabel("Speed range [kts]", fontsize=16, labelpad=10)
ax.set_ylabel("Percentage of Maneuvers [%]", fontsize=16, labelpad=10)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis="both", labelsize=14)
ax.set_xticklabels(SPEED_LABELS, rotation=45, ha="right")

# legend below the plot (≤ 3 columns)
legend_cols = min(3, len(ELEMENTS))
ax.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=COLORS[e]) for e in [1, 2, 3, 4, 5]],
    [ELEMENTS[e] for e in [1, 2, 3, 4, 5]],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    fontsize=14,
    ncol=legend_cols,
    frameon=True
)

plt.tight_layout(rect=[0, 0.3, 1, 1])

# ---- Save ----
plt.savefig(OUT_FILE, bbox_inches="tight", dpi=150)
print(f"Stacked bar chart saved to '{OUT_FILE}'")
