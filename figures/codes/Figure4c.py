# Figure 4c: every system's d(O-Na-O)/d(O-TM-O) ratio vs x in one panel.
#   single TM -> bold solid (element color/marker); double TM -> thin dashed (tab10).
# Shared data/helpers live in Figure4_setting.py.

from Figure4_setting import *          # noqa: F401,F403  (params, cells, helpers)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def make_combined(out_path):
    fig, ax = plt.subplots(figsize=(6.9, 6.0))

    # light reference grid behind everything
    ax.grid(True, which="major", color="0.88", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_color("0.6")
        ax.spines[side].set_linewidth(0.8)

    double_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
    single_handles, double_handles = [], []

    pairs = [(r, c) for r in range(n) for c in range(r)]   # double TM: thin dashed, secondary
    cmap = plt.get_cmap("tab10")
    for k, (r, c) in enumerate(pairs):
        pts = cells[(r, c)][1]
        if pts is None:
            continue
        label = f"{TM_ORDER[c]}–{TM_ORDER[r]}"
        (h,) = ax.plot(pts["x"], pts["d_na"] / pts["d_tm"], color=cmap(k % 10),
                       marker=double_markers[k % len(double_markers)], markersize=6,
                       markeredgecolor="white", markeredgewidth=0.4, lw=1.3,
                       linestyle=(0, (5, 1.8)), alpha=0.85, zorder=3, label=label)
        double_handles.append(h)

    for i, el in enumerate(TM_ORDER):                       # single TM: bold solid, primary
        pts = cells[(i, i)][1]
        if pts is None:
            continue
        (h,) = ax.plot(pts["x"], pts["d_na"] / pts["d_tm"], color=ELEM_COLORS[el],
                       marker=ELEM_MARKERS[el], markersize=9, markeredgecolor="black",
                       markeredgewidth=0.7, lw=2.8, zorder=5, label=el)
        single_handles.append(h)

    ax.set_xlim(0, 1)
    ax.set_ylim(1.25, 1.9)
    ax.set_xticks(X_TICKS)
    ax.xaxis.set_minor_locator(MultipleLocator(0.125))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.set_xlabel(r"x in Na$_{\mathrm{x}}$TM1$_{\mathrm{0.5}}$TM2$_{\mathrm{0.5}}$O$_{\mathrm{2}}$",
                  fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"d(O–Na–O) / d(O–TM–O)", fontsize=AXIS_LABEL_SIZE)

    # one unified legend, single column, with bold section headers
    spacer = Line2D([], [], linestyle="none", marker="none")
    hdr = lambda: Line2D([], [], linestyle="none", marker="none")
    handles = [hdr()] + single_handles + [spacer, hdr()] + double_handles
    labels = (["Single TM"] + [h.get_label() for h in single_handles]
              + ["", "Double TM"] + [h.get_label() for h in double_handles])

    leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    ncol=1, frameon=True, fancybox=False, edgecolor="black",
                    facecolor="white", framealpha=1.0, borderpad=1.1, labelspacing=0.42,
                    handlelength=2.4, handletextpad=0.8, fontsize=12.5)
    leg.get_frame().set_linewidth(1.2)
    for t in leg.get_texts():
        if t.get_text() in ("Single TM", "Double TM"):
            t.set_fontweight("bold")
            t.set_fontsize(13.5)

    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_combined(OUT_RATIO_ALL)
