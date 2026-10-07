# Schematic energy diagram of the redox couples relative to the Fermi level of Na metal (Na/Na+),
# after Figure 2b of Goodenough and Kim, Chem. Mater. 2010, 22, 587.
# Companion panel of FigureS_redox_ladder.py (same page size, colors and fonts).
# The depth of each couple below the Na level is e x (average hull voltage over the first half of the
# charge, x = 1 -> 0.5), averaged over the compounds in which that couple is active, and is drawn TO SCALE.
# The TM 4s and O 2p bands are schematic: their positions are not computed here.
# Data: data/singleTM_hull_points.csv, data/doubleTM_hull_points.csv.   E_Na = -1.31 eV/atom (bcc Na).
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, FancyArrowPatch, Polygon
from matplotlib.path import Path as MPath

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
OUT_PATH = SCRIPT_DIR.parent / "FigureS_redox_band_schematic.pdf"
E_Na_ref = -1.31
FIGSIZE = (6.7, 7.07)                # same page as the one-column redox ladder
FS_LABEL, FS_VALUE, FS_AXIS, FS_BAND = 14, 13, 18, 15

plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42, "ps.fonttype": 42})
ELEM_COLORS = {"Ti": "#8c564b", "Mn": "#9467bd", "Fe": "#ff7f0e", "Co": "#e377c2", "Ni": "#2ca02c"}
GRAY, FILLED = "#a6a6a6", "#e3e3e3"

hs = pd.read_csv(DATA_DIR / "singleTM_hull_points.csv")
hd = pd.read_csv(DATA_DIR / "doubleTM_hull_points.csv")


def v_first_half(name):
    g = hs[hs.tm == name] if "_" not in name else hd[hd.tm_pair == name]
    g = g.sort_values("xNa")
    e = lambda x: float(np.interp(x, g.xNa.values, g.E_TMO2_eV.values))
    return -((e(1.0) - e(0.5)) / 0.5 - E_Na_ref)


# couple -> compounds whose first half of the charge uses that couple (see the redox ladder)
COUPLES = [
    (r"Ti$^{3{+}/4{+}}$", ELEM_COLORS["Ti"], ["Ti"]),
    (r"(Mn, Fe, Co)$^{2{+}/3{+}}$", GRAY, ["Ti_Mn", "Ti_Fe", "Ti_Co"]),
    (r"Mn$^{3{+}/4{+}}$", ELEM_COLORS["Mn"], ["Mn", "Mn_Fe", "Mn_Co"]),
    (r"Co$^{3{+}/4{+}}$", ELEM_COLORS["Co"], ["Co", "Fe_Co"]),
    (r"Ni$^{2{+}/3{+}}$", ELEM_COLORS["Ni"], ["Mn_Ni", "Ti_Ni"]),
    (r"Fe$^{3{+}/4{+}}$", ELEM_COLORS["Fe"], ["Fe"]),
]
levels = []
for label, color, comps in COUPLES:
    v = [v_first_half(c) for c in comps]
    levels.append((label, color, float(np.mean(v)), min(v), max(v)))

# ---- geometry, in inches
S = 1.18                      # inches per eV
Y_NA = 5.98                   # Fermi level of Na metal
X_AX = 2.30                   # energy axis
Y_BOT, Y_TIP = 0.62, 6.92     # ends of the energy axis
L_BAND, T_BAND = 2.30, 0.24   # length and thickness of a redox-couple band
DX_ARROW = 0.31               # spacing of the depth arrows
E_O2P = 3.55                  # top of the O 2p band (schematic), eV below the Na level
y_of = lambda e: Y_NA - S * e

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIGSIZE[0]); ax.set_ylim(0, FIGSIZE[1]); ax.axis("off")

# filled O 2p band (below) and empty TM 4s band (above), quarter ellipses that start on the axis
t = np.linspace(0, np.pi / 2, 120)
a_o, b_o = 2.25, y_of(E_O2P) - Y_BOT
xo, yo = X_AX + a_o * np.sin(t), y_of(E_O2P) - b_o + b_o * np.cos(t)
ax.add_patch(Polygon(np.column_stack([np.r_[xo, X_AX], np.r_[yo, Y_BOT]]), closed=True, fc=FILLED, ec="none", zorder=1))
ax.plot(xo, yo, color="black", lw=1.6, zorder=2, solid_capstyle="round")
a_s, b_s = 1.45, 0.82
ax.plot(X_AX + a_s * np.sin(t), Y_NA + b_s - b_s * np.cos(t), color="black", lw=1.6, zorder=2, solid_capstyle="round")
ax.text(X_AX + a_s + 0.15, Y_NA + b_s - 0.14, r"TM: 4s$^{0}$", fontsize=FS_BAND, ha="left", va="center")
ax.text(X_AX + 0.25, Y_BOT + 0.42, r"O: 2p$^{6}$", fontsize=FS_BAND, ha="left", va="center")

# Fermi level of Na metal
x_left = X_AX - len(levels) * DX_ARROW - 0.05
ax.plot([x_left, FIGSIZE[0] - 0.12], [Y_NA, Y_NA], color="black", lw=1.5, ls=(0, (5, 3)), zorder=3)
ax.text(FIGSIZE[0] - 0.12, Y_NA + 0.07, r"Fermi level of Na metal (Na/Na$^{+}$)", fontsize=FS_VALUE, ha="right", va="bottom")

# redox couples
def band(yc, color):
    h = T_BAND / 2
    verts = [(X_AX, yc + h), (X_AX + 0.55 * L_BAND, yc + 0.78 * h), (X_AX + L_BAND, yc + 0.34 * h), (X_AX + L_BAND, yc),
             (X_AX + L_BAND, yc - 0.34 * h), (X_AX + 0.55 * L_BAND, yc - 0.78 * h), (X_AX, yc - h), (X_AX, yc + h)]
    codes = [MPath.MOVETO] + [MPath.CURVE4] * 6 + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="black", lw=1.2, zorder=4, joinstyle="round"))

for k, (label, color, e, lo, hi) in enumerate(levels):
    yc = y_of(e); band(yc, color)
    ax.text(X_AX + L_BAND + 0.10, yc, label, fontsize=FS_LABEL, ha="left", va="center")
    xa = X_AX - (k + 1) * DX_ARROW
    ax.add_patch(FancyArrowPatch((xa, Y_NA), (xa, yc), arrowstyle="<|-|>", mutation_scale=11, lw=1.2, color="black",
                                 shrinkA=0, shrinkB=0, zorder=5))
    ax.plot([xa, X_AX], [yc, yc], color="black", lw=1.0, ls=(0, (1, 2)), zorder=3)
    ax.text(xa - 0.045, (Y_NA + yc) / 2, f"{e:.1f} eV", fontsize=FS_VALUE, rotation=90, ha="right", va="center")

# axes: energy (up) and density of states (right)
ax.add_patch(FancyArrowPatch((X_AX, Y_BOT), (X_AX, Y_TIP), arrowstyle="-|>", mutation_scale=20, lw=2.2, color="black",
                             shrinkA=0, shrinkB=0, zorder=6))
ax.text(X_AX - 0.16, Y_TIP - 0.10, r"$E$", fontsize=FS_AXIS, ha="right", va="center")
ax.add_patch(FancyArrowPatch((X_AX + 0.90, Y_BOT - 0.30), (X_AX + 1.75, Y_BOT - 0.30), arrowstyle="-|>", mutation_scale=14,
                             lw=1.3, color="black", shrinkA=0, shrinkB=0, zorder=6))
ax.text(X_AX + 0.80, Y_BOT - 0.30, r"$N(E)$", fontsize=FS_LABEL, ha="right", va="center")

fig.savefig(OUT_PATH)
if len(sys.argv) > 1:                                   # optional raster preview
    fig.savefig(sys.argv[1], dpi=130)
print("saved", OUT_PATH)
for label, color, e, lo, hi in levels:
    print(f"{label:34s} mean {e:.2f}  range {lo:.2f}-{hi:.2f} V")
