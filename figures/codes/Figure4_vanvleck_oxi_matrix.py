# Van Vleck Q2-Q3 plane of all 15 systems in ONE figure, with the oxidation state of the
# transition metal.
#
# Layout: lower triangle of a 5x5 transition-metal matrix, the same as Figure2.py.
#   diagonal cell (i, i)        -> single-TM system  Na_x (TM_i) O2
#   off-diagonal cell (i, j<i)  -> mixed-TM system   Na_x (TM_j)0.5 (TM_i)0.5 O2
#   The row TM is written on the left and the column TM at the bottom.
# The empty upper-right triangle holds the legend in the Fe row: one box with the colors
# (oxidation state) and the markers (element) side by side, separated by a dotted line. The block
# above it (rows Ti and Mn, columns Fe to Ni, about 5.6 x 3.4 in on the 10.0 x 9.8 in page) is
# left free for the schematic of the Q2 and Q3 modes.
#
# Every panel is drawn as in Figure4_vanvleck_oxi_polar_each.py (same points, colors, markers
# and grid):
#   polar plot of the Q2-Q3 plane, +Q3 points to the right and +Q2 points up
#   radius = rho (in A), rings every 0.1 A, the same scale in all panels; the values of the
#   rings at 0.2 and 0.4 A are written without background just outside these rings
#   a thin spoke every 30 degrees, black dotted lines along 0, 120 and 240 degrees
#   color = charge, marker = element, gray = charge not assigned (n/a in Figure 3)
#
# Input   data/singleTM_vanvleck_oxi_states.csv, data/doubleTM_vanvleck_oxi_states.csv
#         (written by Figure4_vanvleck_oxi.py)
# Output  ../Figure4_vanvleck_oxi_matrix.pdf              with the legend
#         ../Figure4_vanvleck_oxi_matrix_no_legend.pdf    the same page without the legend
#
# Run     conda run -n research python Figure4_vanvleck_oxi_matrix.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from Figure4_vanvleck_oxi_polar_each import (TM_ORDER, ELEM_MARKERS, CHARGE_COLORS, NA_COLOR,
                                             draw_points, draw_grid)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"                            # .../data
OUT_PATH = SCRIPT_DIR.parent / "Figure4_vanvleck_oxi_matrix.pdf"
OUT_PATH_NO_LEGEND = SCRIPT_DIR.parent / "Figure4_vanvleck_oxi_matrix_no_legend.pdf"

PANEL = 2.4             # in, figure size per cell as in Figure2.py (a panel itself is 1.8 in wide)
GAP = 0.06              # space between neighboring panels, in units of the panel size
SIZE = 70               # marker area
EDGE = 0.8              # black outline of the markers
FS_TM = 20              # TM labels along the matrix, as in Figure2.py
FS_RHO = 11             # rho values written between the rings
FS_LEGEND = 17

# Cell of the matrix (row, column) whose lower right corner is the lower right corner of the
# legend box. Rows and columns are counted from 0 in the order Ti, Mn, Fe, Co, Ni.
LEGEND_ROW, LEGEND_COL = 2, 4


def main():
    single = pd.read_csv(DATA_DIR / "singleTM_vanvleck_oxi_states.csv")
    double = pd.read_csv(DATA_DIR / "doubleTM_vanvleck_oxi_states.csv")
    table = pd.concat([single, double], ignore_index=True)

    n = len(TM_ORDER)
    fig = plt.figure(figsize=(PANEL * n, PANEL * n))
    grid = fig.add_gridspec(n, n, wspace=GAP, hspace=GAP)

    # ---- the 15 panels -------------------------------------------------------------------------
    for row in range(n):
        for col in range(row + 1):                          # lower triangle only
            if row == col:
                system = TM_ORDER[row]                      # diagonal: single TM
            else:
                system = f"{TM_ORDER[col]}_{TM_ORDER[row]}" # off-diagonal: pair, e.g. Ti_Mn
            ax = fig.add_subplot(grid[row, col], projection="polar")
            draw_points(ax, table[table.system == system], size=SIZE, edge=EDGE)
            # no angle labels and no name of the radius: they belong to the schematic
            draw_grid(ax, fontsize=FS_RHO, rho_background=False)

            # TM labels along the matrix: column TM under the bottom row, row TM left of the
            # first column
            if row == n - 1:
                ax.text(0.5, -0.07, TM_ORDER[col], transform=ax.transAxes,
                        ha="center", va="top", fontsize=FS_TM)
            if col == 0:
                ax.text(-0.07, 0.5, TM_ORDER[row], transform=ax.transAxes,
                        ha="right", va="center", fontsize=FS_TM)
            print(f"  row {row}, column {col}: {system}")

    # ---- legends in the empty upper-right triangle --------------------------------------------
    charge_handles = [Patch(facecolor=color, edgecolor="black", linewidth=0.8, label=f"{charge}+")
                      for charge, color in CHARGE_COLORS.items()]
    charge_handles.append(Patch(facecolor=NA_COLOR, edgecolor="0.45", linewidth=0.6, label="n/a"))
    element_handles = [Line2D([], [], linestyle="", marker=ELEM_MARKERS[element], markersize=12,
                              markerfacecolor="white", markeredgecolor="black",
                              markeredgewidth=1.0, label=element)
                       for element in TM_ORDER]

    # One box for both groups. Each group is a legend without a frame, the element group at the
    # right and the oxidation-state group left of it. The box and the dotted line are drawn after
    # the size of the two groups is known.
    corner = grid[LEGEND_ROW, LEGEND_COL].get_position(fig)             # in figure coordinates
    group = dict(loc="lower right", borderaxespad=0, frameon=False, borderpad=0.6,
                 labelspacing=0.4, handlelength=1.4, handletextpad=0.6,
                 fontsize=FS_LEGEND, title_fontsize=FS_LEGEND)
    size_of = lambda legend: legend.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    elements = fig.legend(handles=element_handles, title="Element",
                          bbox_to_anchor=(corner.x1, corner.y0), **group)
    fig.canvas.draw()
    right = size_of(elements)
    charges = fig.legend(handles=charge_handles, title="Oxidation state",
                         bbox_to_anchor=(right.x0, corner.y0), **group)
    fig.canvas.draw()
    left = size_of(charges)

    # rounded white box with a soft shadow, as the legend of Figure2.py
    x0, x1 = left.x0, right.x1
    y0, y1 = min(left.y0, right.y0), max(left.y1, right.y1)
    point = 1 / 72 / fig.get_figwidth()                     # one point in figure units (square figure)
    rounded = dict(boxstyle=f"round,pad=0,rounding_size={0.5 * FS_LEGEND * point}",
                   transform=fig.transFigure)
    shadow = FancyBboxPatch((x0 + 3 * point, y0 - 3 * point), x1 - x0, y1 - y0, zorder=1,
                            facecolor="0.45", edgecolor="none", alpha=0.45, **rounded)
    box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, zorder=2,
                         facecolor="white", edgecolor="0.2", linewidth=1.6, **rounded)
    # dotted line between the colors and the markers, as the divider in the legend of Figure2.py
    divider = Line2D([right.x0, right.x0], [y0 + 12 * point, y1 - 12 * point], zorder=3,
                     transform=fig.transFigure, color="gray", linewidth=1.1,
                     linestyle=(0, (1, 1.6)))
    legend_parts = [elements, charges, shadow, box, divider]
    for part in (shadow, box, divider):
        fig.add_artist(part)

    # ---- two files on the same page: with the legend, and with the legend hidden ---------------
    # The page is the one that fits the figure with the legend (0.1 in margin), so the panels
    # are at the same place in both files.
    page = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)     # in inches
    fig.savefig(OUT_PATH, bbox_inches=page)
    print(f"saved {OUT_PATH}")
    for part in legend_parts:
        part.set_visible(False)
    fig.savefig(OUT_PATH_NO_LEGEND, bbox_inches=page)
    print(f"saved {OUT_PATH_NO_LEGEND}")
    plt.close(fig)


if __name__ == "__main__":
    main()
