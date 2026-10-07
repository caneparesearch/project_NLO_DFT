# Van Vleck Q2-Q3 plane of every system, with the oxidation state of the transition metal.
# 15 SEPARATE PDFs (5 single TM + 10 mixed TM), one point per TMO6 octahedron of every
# stable (hull) structure.
#
# How to read one figure
#   The polar plot is the Q2-Q3 plane: Q3 = rho cos(phi) is the horizontal direction and
#   Q2 = rho sin(phi) the vertical one.
#   radius = rho (size of the distortion, in Angstrom), angle = phi
#   color  = charge of the transition metal (2+ blue, 3+ gold, 4+ red, 5+ purple)
#   marker = element (the same markers as in the other figures of the paper)
#   gray   = octahedra of the structures whose charges could not be assigned (n/a in Figure 3)
#   Grid as in Figure 5 of Steele et al., J. Am. Chem. Soc. 2026, 148, 22062: a thin gray spoke
#   every 30 degrees and a black dotted line along 0, 120 and 240 degrees, the directions of a
#   pure elongation along z, y and x.
#
# Input   data/singleTM_vanvleck_oxi_states.csv, data/doubleTM_vanvleck_oxi_states.csv
#         (written by Figure4_vanvleck_oxi.py)
# Output  ../van_vleck/Figure4_vanvleck_oxi_polar_<system>.pdf   (15 files)
#
# Run     conda run -n research python Figure4_vanvleck_oxi_polar_each.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"                            # .../data
OUT_DIR = SCRIPT_DIR.parent / "van_vleck"

TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]
ELEM_MARKERS = {"Ti": "o", "Mn": "s", "Fe": "^", "Co": "D", "Ni": "v"}
# Low charge blue, high charge red, as the arrows of Figure 3 (red = oxidation).
# The four colors stay apart for color-blind readers (checked for every pair).
CHARGE_COLORS = {2: "#2c7bb6", 3: "#e69f00", 4: "#d7301f", 5: "#7b3294"}
NA_COLOR = "0.85"                                           # not assigned (n/a)

RMAX = 0.55             # A, the same rho scale in all figures
RINGS = [0.1, 0.2, 0.3, 0.4, 0.5]   # A
RHO_LABELS = [0.2, 0.4]             # A, rings that are drawn darker and carry their rho value
RHO_ANGLE = 300                     # degrees, direction of the rho values: the one with the fewest points
RHO_ANGLE_FREE = 285                # degrees, direction of the rho values that have no background
RHO_SHIFT = 0.045                   # A, these values are written between their ring and the next one
                                    # (a bit less than half of the ring spacing, because the two ends
                                    # of the straight text come close to the next ring)
RHO_ROTATION = RHO_ANGLE_FREE - 270 # degrees, turns these values parallel to their ring
ELONGATION = [0, 120, 240]          # degrees, pure elongation along z, y and x
SIZE = 60               # marker area
EDGE = 0.8              # black outline of the markers

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def ion_label(element, charge):
    """Mn, 3 -> Mn3+ with the charge as superscript."""
    return element + r"$^{" + f"{charge}" + r"+}$"


def draw_points(ax, table, size=SIZE, edge=EDGE):
    """Draw the octahedra of one system on a polar axes (also used by Figure4_vanvleck_oxi_matrix.py).

    Returns the ions that were drawn, e.g. [("Mn", 4), ("Ni", 2), ("Ni", 3)], and whether the
    system has octahedra without an assigned charge.
    """
    # ---- octahedra without an assigned charge, drawn first so that they stay in the background
    not_assigned = table[table.charge.isna()]
    for element in TM_ORDER:
        points = not_assigned[not_assigned.element == element]
        if not points.empty:
            ax.scatter(np.radians(points.phi_deg), points.rho_A, marker=ELEM_MARKERS[element],
                       s=0.6 * size, color=NA_COLOR, edgecolor="0.45", linewidths=0.6, zorder=2)

    # ---- octahedra with an assigned charge: one group of points per ion
    assigned = table[table.charge.notna()]
    # the ions with few octahedra are drawn on top of the ions with many, so that they stay visible
    n_octahedra = assigned.groupby(["element", "charge"]).size().sort_values(ascending=False)
    layer = {ion: 3 + 0.01 * rank for rank, ion in enumerate(n_octahedra.index)}
    ions = []
    for element in TM_ORDER:
        for charge in sorted(CHARGE_COLORS):
            points = assigned[(assigned.element == element) & (assigned.charge == charge)]
            if points.empty:
                continue
            ax.scatter(np.radians(points.phi_deg), points.rho_A, marker=ELEM_MARKERS[element],
                       s=size, color=CHARGE_COLORS[charge], edgecolor="black", linewidths=edge,
                       zorder=layer[(element, charge)])
            ions.append((element, charge))
    return ions, not not_assigned.empty


def draw_grid(ax, fontsize, write_rho=True, rho_background=True):
    """Rings, spokes and rho values of one polar panel (also used by Figure4_vanvleck_oxi_matrix.py
    and Figure4_vanvleck_oxi_empty_polar.py).

    rho_background=True   the value sits on its ring, on a white box that interrupts the ring
    rho_background=False  the value has no background and sits just outside its ring, between
                          two rings and two spokes, so that no dark line crosses it; it is
                          turned by RHO_ROTATION, so that it runs parallel to its ring
    Call it after draw_points, since plotting rescales the radius.
    """
    ax.set_axisbelow(True)                                  # the grid stays under everything else
    ax.set_thetagrids(range(0, 360, 30), labels=[])         # a thin spoke every 30 degrees
    ax.set_rticks(RINGS)
    ax.set_yticklabels([])
    ax.grid(color="0.82", linewidth=0.7)

    full_turn = np.linspace(0, 2 * np.pi, 361)
    for rho in RHO_LABELS:                                  # labeled rings: darker, with their value
        ax.plot(full_turn, np.full_like(full_turn, rho), color="0.45", linewidth=0.9, zorder=1.0)
        if write_rho and rho_background:
            ax.text(np.radians(RHO_ANGLE), rho, f"{rho:.1f}", ha="center", va="center",
                    fontsize=fontsize, zorder=1.8,          # under the points, so that it hides none
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
        elif write_rho:
            ax.text(np.radians(RHO_ANGLE_FREE), rho + RHO_SHIFT, f"{rho:.1f}", ha="center",
                    va="center_baseline", rotation=RHO_ROTATION, rotation_mode="anchor",
                    fontsize=fontsize, zorder=1.8)
    for angle in ELONGATION:                                # black dotted lines
        ax.plot([np.radians(angle)] * 2, [0, RMAX], color="black", linewidth=1.3,
                linestyle=(0, (1.2, 2.0)), zorder=1.2)

    ax.set_rlim(0, RMAX)
    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(1.4)


def plot_one(table, title, out_pdf):
    """One polar figure for one system."""
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="polar")
    ions, has_not_assigned = draw_points(ax, table)
    draw_grid(ax, fontsize=15)

    # ---- one legend entry per ion
    legend_handles = [Line2D([], [], linestyle="", marker=ELEM_MARKERS[element], markersize=9,
                             markerfacecolor=CHARGE_COLORS[charge], markeredgecolor="black",
                             markeredgewidth=EDGE, label=ion_label(element, charge))
                      for element, charge in ions]
    if has_not_assigned:
        legend_handles.append(Line2D([], [], linestyle="", marker="o", markersize=7,
                                     markerfacecolor=NA_COLOR, markeredgecolor="0.45",
                                     markeredgewidth=0.6, label="n/a"))

    # ---- labels around the circle: the direction every 60 degrees, nothing on the spokes between
    ax.set_xticklabels(["elong.\n($+Q_3$)", "", "comp.", "", "elong.", "",
                        "comp.\n($-Q_3$)", "", "elong.", "", "comp.", ""])
    ax.tick_params(labelsize=14, pad=5)
    for angle, label in zip(range(0, 360, 30), ax.get_xticklabels()):
        label.set_horizontalalignment("left" if angle in (0, 60, 300) else "right")
    # the Q2 direction of the Q2-Q3 plane (the Q3 direction is part of the labels at 0 and 180)
    ax.text(np.radians(90), RMAX * 1.05, r"$+Q_2$", ha="center", va="bottom", fontsize=14)
    ax.text(np.radians(270), RMAX * 1.05, r"$-Q_2$", ha="center", va="top", fontsize=14)
    # name and unit of the radius, at the end of the row of rho values
    ax.text(np.radians(RHO_ANGLE), 0.495, r"$\rho$ ($\mathrm{\AA}$)", ha="center", va="center",
            fontsize=15, zorder=1.8,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
    ax.set_title(title, fontsize=18, y=1.13)

    # ---- legend of the oxidation states: rounded white box with a soft shadow, as in Figure2.py
    legend = ax.legend(handles=legend_handles, title="Oxidation state",
                       loc="center left", bbox_to_anchor=(1.26, 0.5),
                       frameon=True, fancybox=True, facecolor="white", edgecolor="0.2",
                       framealpha=1.0, shadow=dict(ox=3, oy=-3, fc="0.45", ec="none", alpha=0.45),
                       borderpad=0.6, labelspacing=0.5, handletextpad=0.3, title_fontsize=14)
    legend.get_frame().set_linewidth(1.6)
    legend.get_frame().set_boxstyle("round", pad=0, rounding_size=0.5)

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    single = pd.read_csv(DATA_DIR / "singleTM_vanvleck_oxi_states.csv")
    double = pd.read_csv(DATA_DIR / "doubleTM_vanvleck_oxi_states.csv")
    table = pd.concat([single, double], ignore_index=True)

    systems = list(TM_ORDER)                                                  # Ti, Mn, Fe, Co, Ni
    systems += [f"{a}_{b}" for i, a in enumerate(TM_ORDER) for b in TM_ORDER[i + 1:]]   # Ti_Mn, ...

    for system in systems:
        one_system = table[table.system == system]
        out_pdf = OUT_DIR / f"Figure4_vanvleck_oxi_polar_{system}.pdf"
        plot_one(one_system, system.replace("_", "–"), out_pdf)
        print(f"  {out_pdf.name}: {len(one_system)} octahedra, {one_system.charge.isna().sum()} n/a, "
              f"ions: {' '.join(sorted(one_system.ion.dropna().unique()))}")
    print(f"wrote {len(systems)} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
