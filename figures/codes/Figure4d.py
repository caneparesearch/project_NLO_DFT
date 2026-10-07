# Figure 4d: all stable compounds, ratio vs d(O-Na-O), by phase and Na content.
#   x = d(O-Na-O)/d(O-TM-O), y = d(O-Na-O); marker = phase, color = Na content x.
# Shared data/helpers live in Figure4_setting.py.

from Figure4_setting import *          # noqa: F401,F403  (params, cells, helpers)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

LEGEND_FONT = 17           # pt, text in the legend box (tick labels: 19 pt)
LEGEND_BOX_LINE = 1.6      # pt, outline of the legend box
# legend box of Figure2.py: rounded white box with a soft drop shadow
LEGEND_BOX = dict(frameon=True, fancybox=True, facecolor="white", edgecolor="0.2", framealpha=1.0,
                  shadow=dict(ox=3, oy=-3, fc="0.45", ec="none", alpha=0.45))


def make_scatter(out_path):
    R, Y, X, PH = [], [], [], []
    for _, pts in cells.values():
        if pts is None:
            continue
        R += list(pts["d_na"] / pts["d_tm"])
        Y += list(pts["d_na"])
        X += list(pts["x"])
        PH += list(pts["phase"])
    R, Y, X, PH = (np.array(v) for v in (R, Y, X, PH))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.axhline(3.4, color="0.4", linestyle=(0, (1, 1.5)), linewidth=1.2, zorder=2)
    ax.axvline(1.6, color="0.4", linestyle=(0, (1, 1.5)), linewidth=1.2, zorder=2)
    # distinct shapes + per-phase sizes (triangles read smaller, so bump them up)
    markers = {"O3": ("o", 165), "P3": ("D", 165), "O1": ("^", 225)}
    zorders = {"P3": 3, "O1": 3, "O3": 4}   # O3 circles drawn last -> on top of P3
    sc = None
    for kind, (mk, sz) in markers.items():
        m = PH == kind
        if m.any():
            sc = ax.scatter(R[m], Y[m], c=X[m], cmap="inferno", vmin=-0.25, vmax=1,
                            marker=mk, s=sz, edgecolors="black", linewidths=0.9,
                            alpha=0.95, zorder=zorders[kind])

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.set_ylim(0, 1)              # show only the physical x >= 0 portion
    cbar.set_label(r"x in Na$_{\mathrm{x}}$TM O$_{\mathrm{2}}$", fontsize=AXIS_LABEL_SIZE)
    cbar.outline.set_linewidth(0.8)

    ax.set_xlim(1.2, 1.9)
    ax.set_ylim(2.4, 3.7)
    ax.set_xticks([1.4, 1.6, 1.8])     # drop the 1.2 edge tick
    ax.set_xlabel(r"d(O–Na–O) / d(O–TM–O)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"d(O–Na–O) (Å)", fontsize=AXIS_LABEL_SIZE)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", labelsize=19, length=6)
    ax.tick_params(which="minor", length=3)
    ax.tick_params(which="both", top=True, right=True)      # major and minor ticks on all four sides
    cbar.ax.tick_params(labelsize=19)
    ax.grid(True, which="major", color="0.9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    phase_marks = [Line2D([], [], linestyle="none", marker=markers[k][0], markersize=11,
                          markerfacecolor="0.7", markeredgecolor="black",
                          markeredgewidth=0.7, label=f"{k} phase")
                   for k in markers if (PH == k).any()]
    leg = ax.legend(handles=phase_marks, loc="lower right", fontsize=LEGEND_FONT,
                    borderpad=0.5, labelspacing=0.4, handletextpad=0.3,
                    handlelength=1.0,            # narrow box, it stays right of the dotted line at 1.6
                    borderaxespad=0.8,           # clear of the ticks on the bottom and right side
                    **LEGEND_BOX)
    leg.get_frame().set_linewidth(LEGEND_BOX_LINE)
    leg.get_frame().set_boxstyle("round", pad=0, rounding_size=0.5)

    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_scatter(OUT_SCATTER)
