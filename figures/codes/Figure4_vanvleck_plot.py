# Single overlaid polar (Q2, Q3) plot of the Van Vleck modes computed by
# Figure4_vanvleck.py: every TM octahedron of every stable hull point (single +
# double TM) as one point at angle phi, radius rho = sqrt(Q2^2 + Q3^2).
#
# Encoding:
#   color + marker shape = TM element (paper-wide ELEM_COLORS/ELEM_MARKERS,
#     extended with V/Cr), filled = single-TM system, open = double-TM system.
#   Spokes at 0/120/240 deg are elongations, 60/180/300 deg compressions.
# The x,y,z frame is common per structure and its labelling is fixed by the
# bond geometry (Figure4_vanvleck.py): z = long axis of the most elongated
# octahedron of the cell, so the dominant elongation of every structure sits
# on the 0 deg spoke and a second elongation direction, if present, on 120 deg.
#
# Input:  ../van_vleck/van_vleck_modes.csv
# Output: ../Figure4_vanvleck_polar.pdf

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "van_vleck" / "van_vleck_modes.csv"
OUT_PDF = SCRIPT_DIR.parent / "Figure4_vanvleck_polar.pdf"

# paper-wide element colors/markers (Figure4_setting.py); V and Cr excluded
ELEM_COLORS = {"Ti": "#8c564b", "Mn": "#9467bd", "Fe": "#ff7f0e",
               "Co": "#e377c2", "Ni": "#2ca02c"}
ELEM_MARKERS = {"Ti": "o", "Mn": "s", "Fe": "^", "Co": "D", "Ni": "v"}
ELEM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]

RHO_SPLIT = 0.1     # A: points below drawn underneath (the undistorted blob)
RMAX = 0.55
ALPHA = 0.55
SIZE = 55

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():
    df = pd.read_csv(CSV_PATH).dropna(subset=["rho", "phi_deg"])
    df = df[df.element.isin(ELEM_ORDER)]
    print(f"{len(df)} octahedra from {CSV_PATH.name}")

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="polar")

    for el in ELEM_ORDER:
        for ds, filled in (("doubleTM", False), ("singleTM", True)):
            sub = df[(df.element == el) & (df.dataset == ds)]
            if sub.empty:
                continue
            # translucent fills but opaque edges: alpha lives in the RGBA colors
            style = dict(marker=ELEM_MARKERS[el], s=SIZE)
            if filled:
                style.update(facecolor=to_rgba(ELEM_COLORS[el], ALPHA),
                             edgecolor="black", linewidths=0.7)
            else:
                style.update(facecolor="none",
                             edgecolor=to_rgba(ELEM_COLORS[el], 0.85),
                             linewidths=1.4)
            for lo, z in ((sub[sub.rho < RHO_SPLIT], 2),
                          (sub[sub.rho >= RHO_SPLIT], 3)):
                if not lo.empty:
                    ax.scatter(np.radians(lo.phi_deg), lo.rho, zorder=z, **style)

    # axes cosmetics: elongation/compression spokes, quiet radial grid.
    # Uniform one-word spoke labels; the Q3 sign only on the 0/180 spokes.
    ax.set_thetagrids(range(0, 360, 60),
                      labels=["elong.\n($+Q_3$)", "comp.", "elong.",
                              "comp.\n($-Q_3$)", "elong.", "comp."])
    ax.set_rmax(RMAX)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5])
    # built-in r labels draw below the data (axis artist); use own texts instead,
    # white-boxed and above the markers, with the axis label outside the rim on
    # the same spoke so the three read as one radial axis
    ax.set_yticklabels([])
    for rv in (0.2, 0.4):
        ax.text(np.radians(165), rv, f"{rv:.1f}", ha="center", va="center",
                fontsize=12, zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.text(np.radians(165), RMAX * 1.12, r"$\rho$ ($\mathrm{\AA}$)",
            ha="center", va="center", fontsize=13)
    ax.tick_params(labelsize=12, pad=6)
    ax.tick_params(axis="x", labelcolor="0.2")
    ax.grid(color="0.85", linewidth=0.7)
    ax.spines["polar"].set_color("0.6")

    handles = [Line2D([], [], linestyle="", marker=ELEM_MARKERS[el],
                      markerfacecolor=ELEM_COLORS[el], markeredgecolor="black",
                      markeredgewidth=0.7, markersize=9, label=el)
               for el in ELEM_ORDER if el in set(df.element)]
    handles += [Line2D([], [], linestyle="", marker="o", markerfacecolor="0.6",
                       markeredgecolor="black", markeredgewidth=0.7,
                       markersize=9, label="single TM"),
                Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
                       markeredgecolor="0.3", markeredgewidth=1.4,
                       markersize=9, label="double TM")]
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.25, 0.5),
                    frameon=True, fancybox=False, edgecolor="black",
                    facecolor="white", framealpha=1.0, borderpad=0.8,
                    labelspacing=0.4, handletextpad=0.4)
    leg.get_frame().set_linewidth(1.3)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
