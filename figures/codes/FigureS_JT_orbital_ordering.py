# Schematic of the rule of Radin and Van der Ven (Chem. Mater. 2018, 30, 607) for the ordering of the
# Jahn-Teller elongations in a TMO2 layer:
#       every Jahn-Teller octahedron has two long and four short TM-O bonds (the two long ones are opposite),
#       and an ordering is stable only when EVERY OXYGEN TAKES ONE LONG AND TWO SHORT BONDS.
#
# The TMO2 layer is drawn from above, undistorted:
#       the transition metals form a triangular lattice,
#       one oxygen sits above the centre of every upward triangle and one below the centre of every
#       downward triangle, so each oxygen is bonded to three transition metals,
#       the three O-TM-O axes of an octahedron point, in this view, along 30, 150 and 270 degrees.
# In each panel every row of octahedra (rows run horizontally) is elongated along one axis:
#       (a) collinear    all rows along the same axis           -> every oxygen takes one long bond
#       (b) zig-zag      rows alternate between two axes        -> every oxygen takes one long bond
#       (c) rule broken  rows alternate between two other axes  -> oxygens take two long bonds or none
#
# Output  ../FigureS_JT_orbital_ordering.pdf
# Run     conda run -n research python FigureS_JT_orbital_ordering.py
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR.parent / "FigureS_JT_orbital_ordering.pdf"

plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42, "ps.fonttype": 42})

# ---- the lattice (lengths in units of the TM-TM distance)
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])
# in-plane direction from a transition metal to its oxygen above the layer, for the axes 1, 2, 3;
# the oxygen below the layer on the same axis lies in the opposite direction
AXIS_TO_UPPER_O = {1: (A1 + A2) / 3, 2: (-2 * A1 + A2) / 3, 3: (A1 - 2 * A2) / 3}
ROWS, COLUMNS = 4, 4                 # transition metals drawn in each panel

# ---- the three orderings: elongation axis of the rows, from the bottom row upward
PANELS = [
    ("(a) Collinear", [1, 1, 1, 1]),
    ("(b) Zig-zag", [1, 2, 1, 2]),
    ("(c) Rule broken", [1, 3, 1, 3]),
]

# ---- look
R_TM, R_O = 0.105, 0.150
LONG = dict(color="black", linewidth=3.4, solid_capstyle="round")
SHORT = dict(color="0.62", linewidth=1.0)
COLOR_TM = "0.30"
COLOR_TWO_LONG = "#d7301f"           # oxygen with two long bonds
COLOR_NO_LONG = "#2c7bb6"            # oxygen without a long bond
FS_TITLE, FS_LEGEND = 15, 12


def draw_panel(ax, title, axis_of_row):
    metals = {(i, j): (i - j // 2) * A1 + j * A2 for j in range(ROWS) for i in range(COLUMNS)}
    # shift every second row back, so that the patch is a rectangle and not a rhombus
    long_bonds = {}                  # oxygen position -> number of long bonds
    bonds = []                       # (metal position, oxygen position, above the layer?, long?)
    for (i, j), metal in metals.items():
        for axis, to_upper in AXIS_TO_UPPER_O.items():
            for above, oxygen in ((True, metal + to_upper), (False, metal - to_upper)):
                is_long = axis == axis_of_row[j]
                key = (round(oxygen[0], 3), round(oxygen[1], 3), above)
                long_bonds[key] = long_bonds.get(key, 0) + int(is_long)
                bonds.append((metal, oxygen, above, is_long, key))
    # an oxygen at the edge of the patch has lost neighbors, so it is drawn only if all three are there
    n_bonds = {}
    for _, _, _, _, key in bonds:
        n_bonds[key] = n_bonds.get(key, 0) + 1
    complete = {key for key, n in n_bonds.items() if n == 3}

    for metal, oxygen, above, is_long, key in bonds:
        if key in complete:
            ax.plot([metal[0], oxygen[0]], [metal[1], oxygen[1]], zorder=2 if is_long else 1,
                    **(LONG if is_long else SHORT))
    for x, y, above in complete:
        n_long = long_bonds[(x, y, above)]
        face = {0: COLOR_NO_LONG, 1: "white"}.get(n_long, COLOR_TWO_LONG)
        ax.add_patch(Circle((x, y), R_O, facecolor=face, edgecolor="black", linewidth=1.2,
                            linestyle="-" if above else (0, (2.2, 1.6)), zorder=3))
    used = {tuple(np.round(metal, 3)) for metal, _, _, _, key in bonds if key in complete}
    for metal in metals.values():
        if tuple(np.round(metal, 3)) in used:
            ax.add_patch(Circle(metal, R_TM, facecolor=COLOR_TM, edgecolor="black", linewidth=1.0, zorder=4))

    xs = [k[0] for k in complete] + [m[0] for m in metals.values()]
    ys = [k[1] for k in complete] + [m[1] for m in metals.values()]
    ax.set_xlim(min(xs) - 0.25, max(xs) + 0.25)
    ax.set_ylim(min(ys) - 0.25, max(ys) + 0.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=FS_TITLE, pad=6)
    counts = sorted({long_bonds[k] for k in complete})
    return counts


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.23, wspace=0.03)
    for ax, (title, axis_of_row) in zip(axes, PANELS):
        counts = draw_panel(ax, title, axis_of_row)
        print(f"{title}: long bonds per oxygen {counts}")

    def atom(color, label, radius, below=False):
        """Legend entry drawn as the atoms of the panels (a dashed outline for an oxygen below the layer)."""
        return Circle((0, 0), radius, facecolor=color, edgecolor="black", linewidth=1.2,
                      linestyle=(0, (2.2, 1.6)) if below else "-", label=label)

    def atom_in_legend(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        radius = 0.62 * fontsize * orig_handle.get_radius() / R_O
        return Circle((width / 2 - xdescent, height / 2 - ydescent), radius)

    first_row = [atom(COLOR_TM, "TM", R_TM), atom("white", "O above the TM layer", R_O),
                 atom("white", "O below the TM layer", R_O, below=True),
                 Line2D([], [], label="long TM\u2013O bond", **LONG)]
    second_row = [Line2D([], [], label="short TM\u2013O bond", **SHORT),
                  atom(COLOR_TWO_LONG, "O with two long bonds", R_O),
                  atom(COLOR_NO_LONG, "O without a long bond", R_O)]
    # a legend fills its columns first: interleave the two rows, so that they are read row by row
    handles = [first_row[0], second_row[0], first_row[1], second_row[1], first_row[2], second_row[2], first_row[3]]
    fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=FS_LEGEND, frameon=False,
               handletextpad=0.5, columnspacing=1.6, labelspacing=0.6, bbox_to_anchor=(0.5, 0.0),
               handler_map={Circle: HandlerPatch(patch_func=atom_in_legend)})
    fig.savefig(OUT_PATH)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
