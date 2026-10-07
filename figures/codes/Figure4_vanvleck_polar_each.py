# Per-system Van Vleck polar figures: 15 SEPARATE PDFs (5 single TM + 10
# double TM over Ti/Mn/Fe/Co/Ni), one point per TM octahedron of every stable
# hull point, from the CSV written by Figure4_vanvleck.py.
#
# Encoding (differs from Figure4_vanvleck_plot.py, where color = element):
#   color   = Na content x, binned into 5 discrete classes (legend):
#             x=0 white, low red, mid gold, high blue, x=1 black
#   marker  = element (paper ELEM_MARKERS), which separates the two TMs
#             within a double-TM figure (legend shown there)
#   radius  = rho, angle = phi; all figures share the same rho scale.
#
# Input:  ../van_vleck/van_vleck_modes.csv
# Output: ../van_vleck/Figure4_vanvleck_polar_<system>.pdf  (15 files)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "van_vleck" / "van_vleck_modes.csv"
OUT_DIR = SCRIPT_DIR.parent / "van_vleck"

TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]
ELEM_MARKERS = {"Ti": "o", "Mn": "s", "Fe": "^", "Co": "D", "Ni": "v"}

# Na-content bins: (label, fill color); assigned by na_bin() below
NA_BINS = [
    (r"$x = 0$",              "white"),
    (r"$0 < x \leq 1/3$",     "#d7301f"),
    (r"$1/3 < x \leq 2/3$",   "#fdb863"),
    (r"$2/3 < x < 1$",        "#2c7bb6"),
    (r"$x = 1$",              "black"),
]
EPS = 1e-6


def na_bin(x):
    if x < EPS:
        return 0
    if x > 1 - EPS:
        return 4
    if x <= 1 / 3 + EPS:
        return 1
    if x <= 2 / 3 + EPS:
        return 2
    return 3


RHO_SPLIT = 0.1     # A: points below drawn underneath (the undistorted blob)
RMAX = 0.55
SIZE = 45

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


def plot_one(sub, title, out_pdf):
    """One standalone polar figure for one system."""
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="polar")

    sub = sub.assign(nabin=sub.xNa.map(na_bin))
    elements = [el for el in TM_ORDER if el in set(sub.element)]
    for el in elements:
        d = sub[sub.element == el]
        for lo, z in ((d[d.rho < RHO_SPLIT], 2), (d[d.rho >= RHO_SPLIT], 3)):
            if not lo.empty:
                ax.scatter(np.radians(lo.phi_deg), lo.rho,
                           c=[NA_BINS[b][1] for b in lo.nabin],
                           marker=ELEM_MARKERS[el], s=SIZE,
                           edgecolor="black", linewidths=0.6, zorder=z)

    # styling after plotting: scatter autoscales the r limit
    ax.set_thetagrids(range(0, 360, 60),
                      labels=["elong.\n($+Q_3$)", "comp.", "elong.",
                              "comp.\n($-Q_3$)", "elong.", "comp."])
    ax.set_rmax(RMAX)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5])
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
    ax.set_title(title, fontsize=16, y=1.14)

    # Na-content legend (replaces the colorbar) to the right of the axes
    na_handles = [Line2D([], [], linestyle="", marker="o",
                         markerfacecolor=col, markeredgecolor="black",
                         markeredgewidth=0.6, markersize=9, label=lab)
                  for lab, col in NA_BINS]
    na_leg = ax.legend(handles=na_handles, title=r"$x$ in Na$_x$TMO$_2$",
                       loc="center left", bbox_to_anchor=(1.08, 0.5),
                       frameon=True, fancybox=False, edgecolor="black",
                       facecolor="white", framealpha=1.0, borderpad=0.6,
                       labelspacing=0.5, handletextpad=0.4)
    na_leg.get_frame().set_linewidth(1.3)
    na_leg.get_title().set_fontsize(13)
    ax.add_artist(na_leg)

    if len(elements) > 1:   # double TM: marker shapes carry the element
        handles = [Line2D([], [], linestyle="", marker=ELEM_MARKERS[el],
                          markerfacecolor="0.75", markeredgecolor="black",
                          markeredgewidth=0.6, markersize=9, label=el)
                   for el in elements]
        leg = ax.legend(handles=handles, loc="upper left",
                        bbox_to_anchor=(-0.18, 1.12), frameon=True,
                        fancybox=False, edgecolor="black", facecolor="white",
                        framealpha=1.0, borderpad=0.6, labelspacing=0.4,
                        handletextpad=0.4)
        leg.get_frame().set_linewidth(1.3)

    fig.savefig(out_pdf, bbox_inches="tight", bbox_extra_artists=[na_leg])
    plt.close(fig)


def main():
    df = pd.read_csv(CSV_PATH).dropna(subset=["rho", "phi_deg"])
    df = df[df.element.isin(TM_ORDER)]
    print(f"{len(df)} octahedra from {CSV_PATH.name}")

    systems = [("singleTM", el, el) for el in TM_ORDER]
    systems += [("doubleTM", f"{TM_ORDER[c]}_{TM_ORDER[r]}",
                 f"{TM_ORDER[c]}-{TM_ORDER[r]}")
                for r in range(len(TM_ORDER)) for c in range(r)]

    for dataset, system, title in systems:
        sub = df[(df.dataset == dataset) & (df.system == system)]
        out = OUT_DIR / f"Figure4_vanvleck_polar_{system}.pdf"
        plot_one(sub, title, out)
        print(f"  {out.name}: {len(sub)} octahedra")
    print(f"wrote {len(systems)} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
