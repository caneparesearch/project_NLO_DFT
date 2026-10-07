# Empty polar plot of Figure 5: one panel without points.
# It is the base of the schematic of the Q2 and Q3 modes, on which the changes of the axis
# lengths are drawn by hand around the circle.
#
# The panel is drawn as the panels of Figure4_vanvleck_oxi_matrix.py and at the same size:
#   rings every 0.1 A up to 0.5 A (frame at 0.55 A), the rings at 0.2 and 0.4 A darker
#   a thin spoke every 30 degrees, black dotted lines along 0, 120 and 240 degrees
#   +Q3 points to the right and +Q2 points up
#
# Output  ../Figure4_vanvleck_oxi_empty_polar.pdf              with the rho values 0.2 and 0.4 (small)
#                                                               and one example point that shows
#                                                               what rho and phi are
#         ../Figure4_vanvleck_oxi_empty_polar_no_values.pdf    rings and lines only
#
# Run     conda run -n research python Figure4_vanvleck_oxi_empty_polar.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Figure4_vanvleck_oxi_polar_each import draw_grid

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent

PANEL_WIDTH = 1.775     # in, diameter of one panel of Figure 5
FS_RHO = 8              # rho values written between the rings (11 in Figure 5)
FS_MARK = 13            # the symbols rho and phi of the example

EXAMPLE_RHO = 0.37      # A, radius of the example point
EXAMPLE_PHI = 45        # degrees, angle of the example point
EXAMPLE_COLOR = "0.85"  # light gray fill of the example point
ARC_RADIUS = 0.16       # A, radius of the arc that shows phi


def draw_example(ax):
    """One example point with its radius rho (line from the origin) and its angle phi (arc)."""
    phi = np.radians(EXAMPLE_PHI)

    # ---- rho: line from the origin to the point
    ax.plot([0, phi], [0, EXAMPLE_RHO], color="black", linewidth=1.5, zorder=3)
    ax.plot([phi], [EXAMPLE_RHO], marker="o", markersize=6.5, markerfacecolor=EXAMPLE_COLOR,
            markeredgecolor="black", markeredgewidth=1.3, zorder=4)
    # the symbol sits beside the line (9 points to its left), between two rings;
    # no background, the grid stays visible behind it
    ax.annotate(r"$\rho$", xy=(phi, 0.70 * EXAMPLE_RHO), xytext=(-9 * np.sin(phi), 9 * np.cos(phi)),
                textcoords="offset points", ha="center", va="center", fontsize=FS_MARK, zorder=5)

    # ---- phi: arc from the +Q3 direction (0 degrees) to the line, with an arrow head at its end
    arc = np.linspace(0, phi, 60)
    ax.plot(arc[:-6], np.full(len(arc) - 6, ARC_RADIUS), color="black", linewidth=1.2, zorder=3)
    ax.annotate("", xy=(arc[-1], ARC_RADIUS), xytext=(arc[-8], ARC_RADIUS),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.2,
                                mutation_scale=8, shrinkA=0, shrinkB=0), zorder=3)
    # the symbol sits outside the middle of the arc (12 points away from it), between two rings
    ax.annotate(r"$\varphi$", xy=(phi / 2, ARC_RADIUS),
                xytext=(12 * np.cos(phi / 2), 12 * np.sin(phi / 2)), textcoords="offset points",
                ha="center", va="center", fontsize=FS_MARK, zorder=5)


def main():
    for with_values, name in ((True, "Figure4_vanvleck_oxi_empty_polar.pdf"),
                              (False, "Figure4_vanvleck_oxi_empty_polar_no_values.pdf")):
        fig = plt.figure(figsize=(PANEL_WIDTH / 0.8, PANEL_WIDTH / 0.8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")     # the circle is 0.8 of the page
        if with_values:
            draw_example(ax)
        draw_grid(ax, fontsize=FS_RHO, write_rho=with_values, rho_background=False)
        fig.savefig(OUT_DIR / name, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"saved {OUT_DIR / name}")


if __name__ == "__main__":
    main()
