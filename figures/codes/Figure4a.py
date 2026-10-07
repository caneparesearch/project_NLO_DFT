# Figure 4a: slab thicknesses on the 5x5 TM matrix.
#   left axis  -> d(O-Na-O), the Na interlayer thickness        [range 2.5-4.0 A]
#   right axis -> d(O-TM-O), the TM-O slab thickness (diagonal)  [range 1.8-2.4 A]
# Shared data/helpers live in Figure4_setting.py.

from Figure4_setting import *          # noqa: F401,F403  (params, cells, helpers)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

NA_YLIM, NA_TICKS = (2.5, 4.0), [2.5, 3.0, 3.5, 4.0]   # left: d(O-Na-O)
# d(O-TM-O) data spans 1.89-2.28 A, so 1.8-2.4 is used (a 1.3-1.5 range shows no data)
TM_YLIM, TM_TICKS = (1.8, 2.4), [1.8, 2.0, 2.2, 2.4]   # right: d(O-TM-O)
MS = 8.5                                               # marker size (match Fig 4b)


def make_slab(out_path):
    fig, axes = plt.subplots(n, n, figsize=(PANEL_W * n, PANEL_H * n),
                             gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    for r in range(n):
        for c in range(n):
            ax = axes[r][c]
            if c > r:                       # empty upper-right triangle
                ax.axis("off")
                continue
            pts = cells[(r, c)][1]
            ax_r = None
            if pts is not None:
                ax_r = ax.twinx()                       # right axis: d(O-TM-O)
                shade_phases(ax_r, pts["regions"])      # shading on the back axes
                ax_r.plot(pts["x"], pts["d_tm"], color=COL_TM, marker="s", markersize=MS,
                          markeredgecolor="black", markeredgewidth=0.7, lw=1.8, zorder=3)
                ax.plot(pts["x"], pts["d_na"], color=COL_NA, marker="o", markersize=MS,
                        markeredgecolor="black", markeredgewidth=0.7, lw=1.8, zorder=4)
                ax.set_zorder(ax_r.get_zorder() + 1)    # d(O-Na-O) above d(O-TM-O)/shading
                ax.patch.set_visible(False)

            ax.set_xlim(0, 1)
            ax.set_ylim(*NA_YLIM)
            ax.set_xticks(X_TICKS)
            ax.set_yticks(NA_TICKS)
            ax.xaxis.set_minor_locator(MultipleLocator(0.125))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            ax.tick_params(labelbottom=(r == n - 1), labelleft=(c == 0))
            ax.tick_params(which="minor", length=3)
            if r == n - 1:
                ax.set_xticklabels([f"{t:g}" if (t < 0.99 or c == n - 1) else "" for t in X_TICKS])
            if c == 0:
                # hide the lowest tick (2.5) except on the last row, to avoid seam overlap
                ax.set_yticklabels([f"{t:g}" if (t > 2.6 or r == n - 1) else "" for t in NA_TICKS])
            if r == n - 1:
                ax.set_xlabel(TM_ORDER[c], fontsize=20, labelpad=6)
            if c == 0:
                ax.set_ylabel(TM_ORDER[r], fontsize=20, labelpad=6)

            # right axis d(O-TM-O): ticks/labels only on the diagonal panel of each row
            if ax_r is not None:
                ax_r.set_ylim(*TM_YLIM)
                ax_r.set_yticks(TM_TICKS)
                ax_r.yaxis.set_minor_locator(MultipleLocator(0.1))
                ax_r.tick_params(labelright=(c == r), right=(c == r))
                ax_r.tick_params(which="minor", length=3, right=(c == r))
                if c == r:
                    # hide the lowest tick (1.8) except on the bottom-right panel
                    ax_r.set_yticklabels([f"{t:g}" if (t > 1.85 or r == n - 1) else ""
                                          for t in TM_TICKS])
                else:
                    ax_r.set_yticklabels([])

    fig.supxlabel(r"x in Na$_{\mathrm{x}}$TM1$_{\mathrm{0.5}}$TM2$_{\mathrm{0.5}}$O$_{\mathrm{2}}$",
                  fontsize=AXIS_LABEL_SIZE, y=0.02)
    fig.supylabel(r"d(O–Na–O)  (Å)", fontsize=AXIS_LABEL_SIZE, x=0.02)
    fig.text(0.985, 0.5, r"d(O–TM–O)  (Å)", rotation=90,
             va="center", ha="center", fontsize=AXIS_LABEL_SIZE)

    legend_ax = fig.add_subplot(axes[0][0].get_gridspec()[0:2, 2:n])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=COL_NA, marker="o", markeredgecolor="black",
               markeredgewidth=0.7, lw=1.8, label=r"d(O–Na–O)   Na interlayer"),
        Line2D([0], [0], color=COL_TM, marker="s", markeredgecolor="black",
               markeredgewidth=0.7, lw=1.8, label=r"d(O–TM–O)   TM slab"),
    ] + phase_handles()
    style_legend(legend_ax, handles, fontsize=15.5)

    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_slab(OUT_SLAB)
