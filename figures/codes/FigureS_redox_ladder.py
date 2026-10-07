# Draft supporting figure: redox-couple ladder, ONE-COLUMN version.
# (The wide version with the page size of Figure 3 is FigureS_redox_ladder_wide.py.)
# For every system, the average hull voltage over the first half (x = 1 -> 0.5) and the
# second half (x = 0.5 -> 0) of the charge, with the transition metal that is oxidized in
# that half (from the moment-based oxidation states of Figure 3).
# Data: data/singleTM_hull_points.csv, data/doubleTM_hull_points.csv,
#       data/doubleTM_stable_oxi_states.csv.   E_Na = -1.31 eV/atom (bcc Na).
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
OUT_PATH = SCRIPT_DIR.parent / "FigureS_redox_ladder.pdf"
E_Na_ref = -1.31
# One journal column of Advanced Energy Materials is 8.5 cm (1000 px at 300 dpi). The figure is drawn
# at twice that width, so at print size the annotations are 7 pt, the tick labels 8 pt, the axis
# title 9 pt and the legend 6 pt.
FIGSIZE = (6.7, 7.07)
M_LEFT, M_RIGHT, M_BOT, M_TOP = 0.93, 0.13, 0.76, 0.10   # margins around the plot frame, in
GAP = 1.0                            # extra rows between the single-TM and the mixed-TM block
PAD_TOP, PAD_BOT = 1.75, 1.0         # rows above the first and below the last row (0.35 in per row)
MS_O, MS_D, MEW = 14, 12, 1.6        # circle / diamond size and black outline width
FS_ANN = 14                          # redox-couple annotations
FS_TICK, FS_LABEL = 16, 18           # tick labels (as Figure2.py) and axis label
LW_BOX = 1.4                         # axes box and major ticks; minor ticks 0.75 of it
MINOR_STEP = 0.1                     # V, minor ticks on the voltage axis
XLIM = (0.0, 5.6)                    # V; room for the annotations beside the dots and the legends on the left

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 11, "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42, "ps.fonttype": 42,
})
# same element colors as Figure 4 (Figure4_setting.ELEM_COLORS)
ELEM_COLORS = {"Ti": "#8c564b", "Mn": "#9467bd", "Fe": "#ff7f0e", "Co": "#e377c2", "Ni": "#2ca02c"}
INK, MUTED, BAND = "#1a1a1a", "#8a8a8a", "#ececec"

hs = pd.read_csv(DATA_DIR / "singleTM_hull_points.csv")
hd = pd.read_csv(DATA_DIR / "doubleTM_hull_points.csv")
ox = pd.read_csv(DATA_DIR / "doubleTM_stable_oxi_states.csv")


def vavg(g, xa, xb):
    g = g.sort_values("xNa")
    e = lambda x: float(np.interp(x, g.xNa.values, g.E_TMO2_eV.values))
    return -((e(xa) - e(xb)) / (xa - xb) - E_Na_ref)


def state(pair, x):
    o = ox[(ox.tm_pair == pair) & (ox.charge_neutral == True)].copy()
    o["dx"] = (o.xNa - x).abs()
    r = o.sort_values("dx").iloc[0]
    return r.tm1, float(r.avg_ox_tm1), r.tm2, float(r.avg_ox_tm2)


def couple(tm, a, b):
    lo, hi = int(np.floor(a + 1e-6)), int(np.ceil(b - 1e-6))
    body = f"{lo}{{+}}/{hi}{{+}}" if hi - lo == 1 else f"{lo}{{+}}" + r"\rightarrow" + f"{hi}{{+}}"
    return tm + r"$^{" + body + r"}$"


rows = []
for tm in ["Ti", "Mn", "Fe", "Co", "Ni"]:                      # Figure 2 order
    g = hs[hs.tm == tm]
    rows.append((tm, vavg(g, 1, .5), tm, couple(tm, 3, 4), vavg(g, .5, 0), tm, couple(tm, 3, 4)))
rows.append(None)                                               # spacer
for pair in ["Ti_Mn", "Ti_Fe", "Ti_Co", "Ti_Ni", "Mn_Fe", "Mn_Co", "Mn_Ni", "Fe_Co", "Fe_Ni", "Co_Ni"]:
    g = hd[hd.tm_pair == pair]
    t1, a1, t2, a2 = state(pair, 1.0); _, b1, _, b2 = state(pair, 0.5); _, c1, _, c2 = state(pair, 0.0)
    halves = []
    for (p, q) in (((a1, a2), (b1, b2)), ((b1, b2), (c1, c2))):
        d1, d2 = q[0] - p[0], q[1] - p[1]
        halves.append((t1, couple(t1, p[0], q[0])) if d1 >= d2 else (t2, couple(t2, p[1], q[1])))
    rows.append((pair.replace("_", "–"), vavg(g, 1, .5), halves[0][0], halves[0][1],
                 vavg(g, .5, 0), halves[1][0], halves[1][1]))

fig = plt.figure(figsize=FIGSIZE)
AX = [M_LEFT / FIGSIZE[0], M_BOT / FIGSIZE[1],
      (FIGSIZE[0] - M_LEFT - M_RIGHT) / FIGSIZE[0], (FIGSIZE[1] - M_BOT - M_TOP) / FIGSIZE[1]]
ax = fig.add_axes(AX)
DX = 0.194 * (XLIM[1] - XLIM[0]) / (AX[2] * FIGSIZE[0])  # dot-to-annotation distance, 0.194 in, in V
ys, labels = [], []
y = 0
ni23 = [r[1] for r in rows if r and r[0] in ("Mn–Ni", "Ti–Ni")]
ax.axvspan(min(ni23), max(ni23), color=BAND, zorder=0, lw=0)
for r in rows:
    if r is None:                                       # dotted separator between the two blocks
        y_sep = y + 0.5 - GAP / 2
        ax.axhline(y_sep, color="0.3", lw=1.2, ls=(0, (1, 2.2)), zorder=0.5)
        y -= GAP
        continue
    name, v1, m1, c1, v2, m2, c2 = r
    ax.plot([v1, v2], [y, y], color="black", lw=1.5, zorder=1, solid_capstyle="round")
    ax.plot(v1, y, marker="o", ms=MS_O, mfc=ELEM_COLORS[m1], mec="black", mew=MEW, zorder=3)
    ax.plot(v2, y, marker="D", ms=MS_D, mfc=ELEM_COLORS[m2], mec="black", mew=MEW, zorder=3)
    ax.text(v1 - DX, y, c1, ha="right", va="center", fontsize=FS_ANN, color=INK)
    ax.text(v2 + DX, y, c2, ha="left", va="center", fontsize=FS_ANN, color=INK)
    ys.append(y); labels.append(name); y -= 1
ax.set_yticks(ys); ax.set_yticklabels(labels)
YLIM = (min(ys) - PAD_BOT, max(ys) + PAD_TOP)
ax.set_ylim(*YLIM)
ax.set_xlim(*XLIM)
ax.set_xlabel(r"Average voltage (V vs. Na/Na$^+$)", fontsize=FS_LABEL)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(MINOR_STEP))
ax.tick_params(axis="both", labelsize=FS_TICK, width=LW_BOX)
ax.tick_params(axis="x", which="both", top=True)
ax.tick_params(axis="x", which="minor", length=3, width=0.75 * LW_BOX)
for loc, tick in zip(ax.xaxis.get_minorticklocs(), ax.xaxis.get_minor_ticks()):
    if abs(loc % 1 - 0.5) < 1e-6:                       # half-volt minor ticks a bit longer
        tick.tick1line.set_markersize(4.5)
        tick.tick2line.set_markersize(4.5)
for sp in ax.spines.values():
    sp.set_linewidth(LW_BOX)
ax.text(np.mean(ni23), max(ys) + 1.0, r"Ni$^{2{+}/3{+}}$ redox couple", ha="center", va="center",
        fontsize=FS_ANN, color=INK)
h1 = [Line2D([0], [0], marker="o", lw=0, ms=MS_O, mfc="#d0d0d0", mec="black", mew=MEW,
             label="first half\n" + r"($x$ = 1$\rightarrow$0.5)"),
      Line2D([0], [0], marker="D", lw=0, ms=MS_D, mfc="#d0d0d0", mec="black", mew=MEW,
             label="second half\n" + r"($x$ = 0.5$\rightarrow$0)")]
# row-wise reading order (Ti Mn / Fe Co / Ni) in a legend that fills its columns first
h2 = [Patch(facecolor=ELEM_COLORS[k], edgecolor="none", label=k) for k in ("Ti", "Fe", "Ni", "Mn", "Co")]
# Legends inside the plot, in the two pockets that the annotations leave free on the left: beside the
# single-TM rows and in the lower left corner. Rounded white box with a soft drop shadow, as the legend
# of Figure2.py.
LEG_X = AX[0] + 0.15 / FIGSIZE[0]                       # 0.15 in right of the frame, clear of the row ticks
fy = lambda row: AX[1] + AX[3] * (row - YLIM[0]) / (YLIM[1] - YLIM[0])   # row coordinate -> figure fraction
LEG_BOX = dict(borderaxespad=0, frameon=True, fancybox=True,
               facecolor="white", edgecolor="0.2", framealpha=1.0,
               shadow=dict(ox=3, oy=-3, fc="0.45", ec="none", alpha=0.45),
               borderpad=0.6, fontsize=12, title_fontsize=12, alignment="left")
leg1 = fig.legend(handles=h1, title="Part of charge", loc="center left",
                  bbox_to_anchor=(LEG_X, fy((-0.31 + y_sep) / 2)),      # between the Ti annotation and the separator
                  handlelength=1.3, handletextpad=0.5, labelspacing=0.8, **LEG_BOX)
leg2 = fig.legend(handles=h2, title="Metal oxidized", ncols=2, loc="lower left",
                  bbox_to_anchor=(LEG_X, AX[1] + 0.18 / FIGSIZE[1]),
                  handlelength=1.3, handletextpad=0.6, labelspacing=0.45, columnspacing=1.2, **LEG_BOX)
for leg in (leg1, leg2):
    leg.get_frame().set_linewidth(1.6)
    leg.get_frame().set_boxstyle("round", pad=0, rounding_size=0.5)
fig.savefig(OUT_PATH)
if len(sys.argv) > 1:                                   # optional raster preview
    fig.savefig(sys.argv[1], dpi=130)
print("saved", OUT_PATH)
for r in rows:
    if r: print(f"{r[0]:6s} {r[1]:.2f} {r[3]:22s} {r[4]:.2f} {r[6]}")
