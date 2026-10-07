# Figure 4b: mean axial/equatorial TM-O ratio (Jahn-Teller) on the 5x5 TM matrix.
#   one line per element, error bars = octahedron-to-octahedron std within each cell.
# Shared data/helpers live in Figure4_setting.py.

from Figure4_setting import *          # noqa: F401,F403  (params, cells, helpers)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def make_jt(out_path):
    fig, axes = plt.subplots(n, n, figsize=(PANEL_W * n, PANEL_H * n),
                             gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    for r in range(n):
        for c in range(n):
            ax = axes[r][c]
            if c > r:                       # empty upper-right triangle
                ax.axis("off")
                continue
            pts = cells[(r, c)][1]
            if pts is not None:
                shade_phases(ax, pts["regions"])
                ax.axhline(1.0, color="0.4", lw=0.9, linestyle=(0, (1, 1.5)), zorder=2)
                for el, arr in pts["jt"].items():
                    col = ELEM_COLORS.get(el, "0.3")
                    ax.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], color=col,
                                marker=ELEM_MARKERS.get(el, "^"), markersize=8.5,
                                markeredgecolor="black", markeredgewidth=0.6, lw=1.8,
                                capsize=4.0, elinewidth=0.9, capthick=2.2, zorder=4)

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(*JT_YLIM)
            ax.set_xticks(X_TICKS)
            ax.set_yticks(JT_TICKS)
            ax.xaxis.set_minor_locator(MultipleLocator(0.125))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.tick_params(labelbottom=(r == n - 1), labelleft=(c == 0))
            ax.tick_params(which="minor", length=3)
            if r == n - 1:                # -0.05..1.05 margin leaves room to show '1'
                ax.set_xticklabels([f"{t:g}" for t in X_TICKS])
            if c == 0:                    # show 1.0 on every panel
                ax.set_yticklabels([f"{t:.1f}" if (t > 0.9 or r == n - 1) else ""
                                    for t in JT_TICKS])
            if r == n - 1:
                ax.set_xlabel(TM_ORDER[c], fontsize=20, labelpad=6)
            if c == 0:
                ax.set_ylabel(TM_ORDER[r], fontsize=20, labelpad=6)

    fig.supxlabel(r"x in Na$_{\mathrm{x}}$TM1$_{\mathrm{0.5}}$TM2$_{\mathrm{0.5}}$O$_{\mathrm{2}}$",
                  fontsize=AXIS_LABEL_SIZE, y=0.02)
    fig.supylabel(r"Mean axial/equatorial TM–O ratio", fontsize=AXIS_LABEL_SIZE, x=0.02)

    legend_ax = fig.add_subplot(axes[0][0].get_gridspec()[0:2, 2:n])
    legend_ax.axis("off")
    handles = [Line2D([0], [0], color=ELEM_COLORS[el], marker=ELEM_MARKERS[el],
                      markeredgecolor="black", markeredgewidth=0.5, lw=1.8,
                      label=f"{el}–O") for el in TM_ORDER]
    handles += [Line2D([0], [0], color="0.4", lw=0.9, linestyle=(0, (1, 1.5)),
                       label="ratio = 1 (ideal octahedron)")]
    handles += phase_handles()
    style_legend(legend_ax, handles, fontsize=14.5)

    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_jt(OUT_JT)
