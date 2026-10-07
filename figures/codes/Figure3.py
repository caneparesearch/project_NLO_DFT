# Figure 3: Table of average transition-metal oxidation states in mixed (double-TM)
# layered oxides Na_x TM1_0.5 TM2_0.5 O2.
#
# Layout: 10 TM-pair systems, each a 2-row block (TM1 on top, TM2 below) -> 20 rows.
#   - Leftmost merged cell: TM-mix label TM1_0.5 TM2_0.5 (spans the 2 rows).
#   - Next column: the TM symbol for each sub-row.
#   - Remaining columns: x in Na_x TM1 TM2 O2, shown as fractions (union of stable
#     compositions across all pairs).
#
# Each data cell:
#   - filled by color according to E_hull vs Materials Project (doubleTM_hull_mp_ehull.csv),
#   - drawn with a solid black border if that (pair, x) is on the hull (doubleTM_hull_points.csv),
#   - annotated with the average oxidation state (doubleTM_stable_oxi_states.csv): avg_ox_tm1 in
#     the TM1 row, avg_ox_tm2 in the TM2 row. If a stable point fails the charge-neutrality
#     check it has no oxidation-state info -> "n/a". Compositions not stable for a pair are blank.
#
# Color legend (E_hull vs MP) is drawn to the right of the table.

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# -----------------------------------------------------------------------------
# PARAMETERS  (edit these)
# -----------------------------------------------------------------------------
TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]   # defines pair order (i<j): TMi_TMj

CMAP = plt.cm.YlOrRd          # E_hull color (colorblind-safe sequential): pale yellow = on MP hull (0), dark red = high
EHULL_VMAX = 100              # meV/atom; top of color scale (values above are clamped)

# cell geometry (axis units)
W_FORMULA = 2.0               # leftmost merged composition-label column
W_TM = 1.2                    # TM-symbol column
W_DATA = 1.0                  # each x column
H_HEADER = 1.1                # header row (fraction labels)
H_ROW = 1.0                   # each data sub-row
SX, SY = 0.58, 0.32           # inches per axis unit (x, y)

# redox arrows (per element, comparing displayed oxidation states across x)
ARROW_RED = "#d62728"         # oxidation (oxidation state increases with x)
ARROW_BLUE = "#1f77b4"        # reduction (oxidation state decreases with x)
ARROW_LW = 2.4
ARROW_SCALE = 20            # arrowhead size
ARROW_STYLE = "->"          # open (unfilled) V-head arrow
ARROW_LS = (0, (1.5, 1.5))  # dotted shaft
ARROW_HEAD_LEN = 0.3        # axis units reserved for the solid head (shaft dotted up to here)

OXI_FMT = "{:.2f}"            # average oxidation-state number format

# Compositions reached by an electrochemical (de)sodiation process in experiment:
# a "*" is drawn on every stable cell with x >= the listed threshold (up to x = 1).
# Mn_Ni: 0.0 -> all concentrations. Only pairs listed here get asterisks.
ASTERISK_FROM = {
    "Mn_Fe": 0.5,
    "Fe_Co": 1/3,
    "Co_Ni": 0.4,
    "Fe_Ni": 3/7,
    "Mn_Ni": 0.0,
    "Ti_Ni": 0.5,
}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"          # .../data
OUT_PATH = SCRIPT_DIR.parent / "Figure3_oxidation_table.pdf"

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,        # embed TrueType fonts (journal-compliant, no Type 3)
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def xkey(x):
    return round(float(x), 6)


def frac_label(x):
    """Format a Na content as a fraction string, e.g. 0.4285.. -> '3/7'."""
    if abs(x) < 1e-9:
        return "0"
    if abs(x - 1.0) < 1e-9:
        return "1"
    fr = Fraction(x).limit_denominator(20)
    return f"{fr.numerator}/{fr.denominator}"


def formula_label(tm1, tm2):
    return rf"$\mathrm{{{tm1}_{{0.5}}{tm2}_{{0.5}}}}$"


def text_color(rgba):
    """Black or white text depending on background luminance.

    Threshold 0.36 maximizes the worst-case WCAG contrast over the YlOrRd
    ramp (4.6:1; mid-orange reads better with black than white)."""
    r, g, b = rgba[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.36 else "white"


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
df_e = pd.read_csv(DATA_DIR / "doubleTM_hull_mp_ehull.csv")      # ehull vs MP
df_h = pd.read_csv(DATA_DIR / "doubleTM_hull_points.csv")        # hull membership
df_o = pd.read_csv(DATA_DIR / "doubleTM_stable_oxi_states.csv")  # avg oxidation states

# pair order: lower triangle of TM_ORDER (i < j) -> "TMi_TMj"
pairs = [f"{TM_ORDER[i]}_{TM_ORDER[j]}"
         for i in range(len(TM_ORDER)) for j in range(i + 1, len(TM_ORDER))]

# x columns: union of stable compositions across all pairs, sorted
x_cols = sorted({xkey(x) for x in df_e["xNa"]})

ehull_map = {(p, xkey(x)): float(e) * 1000.0   # eV/atom -> meV/atom
             for p, x, e in zip(df_e["tm_pair"], df_e["xNa"], df_e["ehull"])}
hull_set = {(p, xkey(x)) for p, x in zip(df_h["tm_pair"], df_h["xNa"])}
oxi_map = {(row["tm_pair"], xkey(row["xNa"])): (row["avg_ox_tm1"], row["avg_ox_tm2"])
           for _, row in df_o.iterrows()}

norm = Normalize(vmin=0.0, vmax=EHULL_VMAX)

# -----------------------------------------------------------------------------
# FIGURE GEOMETRY
# -----------------------------------------------------------------------------
n_pairs = len(pairs)
n_rows = 2 * n_pairs
n_data = len(x_cols)

x_tm0 = W_FORMULA
x_data0 = W_FORMULA + W_TM
total_w = x_data0 + n_data * W_DATA
total_h = H_HEADER + n_rows * H_ROW

y_body_top = total_h - H_HEADER            # top edge of first data row


def row_top(ri):
    return y_body_top - ri * H_ROW


def col_left(j):
    return x_data0 + j * W_DATA


table_w_in = total_w * SX
table_h_in = total_h * SY
cbar_region_in = 2.4
figw = table_w_in + cbar_region_in
figh = table_h_in
tf = 1.0

fig = plt.figure(figsize=(figw, figh))
ax = fig.add_axes([0.0, 0.0, table_w_in / figw, 1.0])
ax.set_xlim(-0.05, total_w + 0.05)
ax.set_ylim(-0.05, total_h + 0.05)
ax.axis("off")

GRID_EDGE = "#d9d9d9"
EMPTY_FACE = "white"

# -----------------------------------------------------------------------------
# DRAW CELLS
# -----------------------------------------------------------------------------
# header band + fraction labels over each data column
ax.add_patch(Rectangle((0, y_body_top), total_w, H_HEADER,
                       facecolor="#efefef", edgecolor="none", zorder=0))
ax.text(W_FORMULA / 2, y_body_top + H_HEADER / 2, "TM mix",
        ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(x_tm0 + W_TM / 2, y_body_top + H_HEADER / 2, r"Na$_\mathrm{x}$ =",
        ha="center", va="center", fontsize=12, fontweight="bold")
for j, x in enumerate(x_cols):
    ax.text(col_left(j) + W_DATA / 2, y_body_top + H_HEADER / 2, frac_label(x),
            ha="center", va="center", fontsize=11.5)

for p, pair in enumerate(pairs):
    tm1, tm2 = pair.split("_")
    ri_top = 2 * p
    top = row_top(ri_top)
    # merged composition-label cell (spans the 2 rows)
    ax.add_patch(Rectangle((0, top - 2 * H_ROW), W_FORMULA, 2 * H_ROW,
                           facecolor="#f6f6f6", edgecolor=GRID_EDGE, lw=0.6, zorder=1))
    ax.text(W_FORMULA / 2, top - H_ROW, formula_label(tm1, tm2),
            ha="center", va="center", fontsize=14, zorder=2)

    for s, tm in enumerate((tm1, tm2)):
        ri = ri_top + s
        rtop = row_top(ri)
        rbot = rtop - H_ROW
        # TM-symbol cell
        ax.add_patch(Rectangle((x_tm0, rbot), W_TM, H_ROW,
                               facecolor="white", edgecolor=GRID_EDGE, lw=0.6, zorder=1))
        ax.text(x_tm0 + W_TM / 2, rtop - H_ROW / 2, tm,
                ha="center", va="center", fontsize=11.5, zorder=2)

        # data cells
        for j, x in enumerate(x_cols):
            left = col_left(j)
            key = (pair, x)
            if key in ehull_map:
                rgba = CMAP(norm(ehull_map[key]))
                on_hull = key in hull_set
                ax.add_patch(Rectangle((left, rbot), W_DATA, H_ROW,
                                       facecolor=rgba,
                                       edgecolor="black" if on_hull else GRID_EDGE,
                                       lw=1.4 if on_hull else 0.6, zorder=2))
                val = oxi_map.get(key, (np.nan, np.nan))[s]
                txt = "n/a" if pd.isna(val) else OXI_FMT.format(float(val))
                tcol = text_color(rgba)
                ax.text(left + W_DATA / 2, rtop - H_ROW / 2, txt,
                        ha="center", va="center", fontsize=12,
                        fontweight="bold", color=tcol, zorder=5)
                # asterisk: composition reached electrochemically in experiment
                if pair in ASTERISK_FROM and x >= ASTERISK_FROM[pair] - 1e-6:
                    ax.text(left + W_DATA * 0.90, rtop - H_ROW * 0.30, "*",
                            ha="center", va="center", fontsize=10, fontweight="bold",
                            color=tcol, zorder=6)
            else:
                ax.add_patch(Rectangle((left, rbot), W_DATA, H_ROW,
                                       facecolor=EMPTY_FACE, edgecolor=GRID_EDGE,
                                       lw=0.6, zorder=1))

# -----------------------------------------------------------------------------
# REDOX ARROWS  (per element, comparing oxidation states across x)
#   Walk each metal's cells left->right, skipping blank/n/a cells. Between each
#   pair of consecutive numbers:
#     - no arrow if the value is unchanged,
#     - no arrow if the two boxes are directly adjacent (no gap to span),
#     - otherwise draw an arrow across the gap, from box outline to box outline
#       (never overlapping the numbers): blue = reduction (value decreases with x),
#       red = oxidation (value increases with x); arrowhead toward the more-oxidized end.
# -----------------------------------------------------------------------------
for p, pair in enumerate(pairs):
    for s in range(2):
        ri = 2 * p + s
        yc = row_top(ri) - H_ROW / 2
        seq = [(j, round(float(oxi_map[(pair, x)][s]), 2))
               for j, x in enumerate(x_cols)
               if (pair, x) in oxi_map and not pd.isna(oxi_map[(pair, x)][s])]
        for (ja, va), (jb, vb) in zip(seq, seq[1:]):
            if va == vb or jb == ja + 1:             # unchanged, or no gap -> no arrow
                continue
            color = ARROW_RED if vb > va else ARROW_BLUE
            # endpoints at box outlines; all arrows point left (color encodes ox/red)
            x_head, x_tail = col_left(ja) + W_DATA, col_left(jb)
            # dotted shaft (no head), stopping where the solid head starts
            ax.annotate("", xy=(x_head + ARROW_HEAD_LEN, yc), xytext=(x_tail, yc),
                        arrowprops=dict(arrowstyle="-", color=color, lw=ARROW_LW,
                                        linestyle=ARROW_LS, shrinkA=0, shrinkB=0),
                        zorder=4.5)
            # solid open V-head
            ax.annotate("", xy=(x_head, yc), xytext=(x_head + ARROW_HEAD_LEN, yc),
                        arrowprops=dict(arrowstyle=ARROW_STYLE, color=color, lw=ARROW_LW,
                                        shrinkA=0, shrinkB=0, mutation_scale=ARROW_SCALE,
                                        capstyle="round", joinstyle="round"),
                        zorder=4.5)

# -----------------------------------------------------------------------------
# STRUCTURAL LINES
# -----------------------------------------------------------------------------
ax.add_patch(Rectangle((0, 0), total_w, total_h, fill=False,
                       edgecolor="black", lw=1.6, zorder=4))           # outer border
ax.plot([0, total_w], [y_body_top, y_body_top], color="black", lw=1.4, zorder=4)  # header sep
for xv in (W_FORMULA, x_data0):                                        # section dividers
    ax.plot([xv, xv], [0, total_h], color="black", lw=1.2, zorder=4)
for p in range(n_pairs + 1):                                           # pair-group separators
    yv = y_body_top - 2 * p * H_ROW
    ax.plot([0, total_w], [yv, yv], color="black", lw=1.0, zorder=4)
for j in range(1, n_data):                                            # vertical lines between x columns
    xv = col_left(j)
    ax.plot([xv, xv], [0, total_h], color="#9e9e9e", lw=0.7, zorder=4)

# -----------------------------------------------------------------------------
# COLORBAR (E_hull vs MP) to the right of the table
# -----------------------------------------------------------------------------
cax = fig.add_axes([(table_w_in + 0.50) / figw, 0.15 * tf, 0.030, 0.70 * tf])
sm = ScalarMappable(norm=norm, cmap=CMAP)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Energy above the hull (meV/atom)", fontsize=24)
cb.ax.tick_params(labelsize=20)
cb.outline.set_edgecolor("black")
cb.outline.set_linewidth(1.4)

fig.savefig(OUT_PATH, dpi=500, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
