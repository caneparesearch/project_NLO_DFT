# Code to generate Supplementary Figures 5 and 6, XRD patterns of the stable
# (convex-hull) structures of every Na layered oxide.
#
# Generates one broken-axis XRD figure per system: the 5 single-TM systems
# Na_x(TM)O2 (TM = Ti, Mn, Fe, Co, Ni) and the 10 equimolar mixed-TM pairs
# Na_x(TM1)0.5(TM2)0.5O2, saved as individual PDFs in ../xrd_patterns/.
#
# The pattern generation follows notebook/dft_structure_and_thermo_analysis.ipynb
# (single-TM and double-TM XRD cells) verbatim; only the output directory and the
# save DPI (raised to 500) differ from the notebook.

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"                 # .../data
OUT_DIR = SCRIPT_DIR.parent / "xrd_patterns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

single_csv = DATA_DIR / "singleTM_hull_points.csv"
double_csv = DATA_DIR / "doubleTM_hull_points.csv"
structures_json = DATA_DIR / "structures.json"

dpi = 500

# Broken-axis windows
x1_min, x1_max = 10, 21
x2_min, x2_max = 29, 50

# Gaussian broadening
use_gaussian_broadening = True
sigma_deg = 0.10
x_step_deg = 0.02

# Number of top smeared peaks to annotate with their (hkl) labels.
N_PEAKS_TO_ANNOTATE = 3
# When a peak is taller than this threshold (in I a.u.), the (hkl) label is
# shifted slightly to the left so it doesn't sit on top of the peak.
TALL_PEAK_THRESHOLD = 20.0
TALL_PEAK_X_OFFSET = 0.6   # 2-theta degrees to shift left for tall peaks

# ---- Fixed inch-based layout (all positions derive from these) ------------
SUBPLOT_H_IN = 0.9   # height of one row, in inches
DATA_W_IN = 5.00   # combined width of (left panel + gap + right panel)
LEFT_MARGIN_IN = 1.20   # space for y tick labels + y-axis title
RIGHT_MARGIN_IN = 1.55   # space for the side legend (xNa, phase, ordering)
TOP_MARGIN_IN = 0.12
BOTTOM_MARGIN_IN = 0.95   # space for x tick labels + x-axis title
YLABEL_OFFSET_IN = 0.92   # 'I (a.u.)' distance left of the data area
XLABEL_OFFSET_IN = 0.65   # x-axis title distance below the data area
# ---------------------------------------------------------------------------

# Major xticks to use on EVERY row (not just the bottom one), so the dashed
# grid lines render at all of these positions in every panel.
XTICKS_LEFT = [10, 15, 20]
XTICKS_RIGHT = [30, 35, 40, 45, 50]


# -----------------------------------------------------------------------------
# HELPERS (verbatim from the notebook XRD cells)
# -----------------------------------------------------------------------------
def short_phase(phase_name):
    return str(phase_name).replace("_orderings", "")


def pretty_ordering(name):
    """JLK_zeta_w_vac -> 'JLK zeta' (underscores -> spaces; no Greek conversion)."""
    s = str(name).replace("_w_vac", "")
    return s.replace("_", " ")


def format_spacegroup(sg):
    """Render a Hermann-Mauguin spacegroup symbol with bars/subscripts via
    mathtext, e.g. 'R-3m' -> 'R$\\bar{3}$m', 'P6_3/mmc' -> 'P6$_{3}$/mmc'."""
    s = str(sg)
    s = re.sub(r"-([0-9])", r"$\\bar{\1}$", s)
    s = re.sub(r"([0-9])_([0-9]+)", r"\1$_{\2}$", s)
    return s


def format_hkl(hkl_tuple):
    """Render a Miller (or Miller-Bravais) tuple with no spaces between
    indices and bar notation for negatives, e.g. (1, 0, -1, 1) -> '(10\\bar{1}1)'."""
    parts = []
    for v in hkl_tuple:
        iv = int(round(float(v)))
        if iv < 0:
            parts.append(rf"\bar{{{abs(iv)}}}")
        else:
            parts.append(str(iv))
    return r"$(" + "".join(parts) + r")$"


def gaussian_broadened_profile(x_peaks, y_peaks, x_grid, sigma):
    profile = np.zeros_like(x_grid, dtype=float)
    if sigma <= 0:
        return profile
    for xp, yp in zip(x_peaks, y_peaks):
        profile += yp * np.exp(-0.5 * ((x_grid - xp) / sigma) ** 2)

    max_stick = float(np.max(y_peaks)) if len(y_peaks) else 0.0
    max_prof = float(np.max(profile)) if len(profile) else 0.0
    if max_stick > 0 and max_prof > 0:
        profile *= max_stick / max_prof
    return profile


def top_smeared_peaks(x_grid, y_grid, mask, n_top):
    """Find local maxima of the smeared profile within the boolean ``mask``,
    sorted by descending height. Returns at most ``n_top`` (x, y) tuples."""
    x_vis = x_grid[mask]
    y_vis = y_grid[mask]
    if x_vis.size == 0:
        return []
    peak_idx, _ = find_peaks(y_vis)
    if peak_idx.size == 0:
        return [(float(x_vis[int(np.argmax(y_vis))]), float(np.max(y_vis)))]
    order = peak_idx[np.argsort(y_vis[peak_idx])[::-1]]
    return [(float(x_vis[i]), float(y_vis[i])) for i in order[:n_top]]


def render_xrd_figure(pair_hull_df, all_structures, calc):
    """Build a (n x 2) broken-axis XRD figure with fixed-inch row size.
    Returns (fig, axes). Caller decides whether to show or save."""
    n = len(pair_hull_df)

    fig_w_in = LEFT_MARGIN_IN + DATA_W_IN + RIGHT_MARGIN_IN
    fig_h_in = TOP_MARGIN_IN + n * SUBPLOT_H_IN + BOTTOM_MARGIN_IN

    fig, axes = plt.subplots(
        n, 2,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,
        gridspec_kw={"hspace": 0.0, "wspace": 0.035, "width_ratios": [1.12, 1.88]},
    )
    if n == 1:
        axes = np.array([axes])

    fig_left = LEFT_MARGIN_IN / fig_w_in
    fig_right = 1.0 - RIGHT_MARGIN_IN / fig_w_in
    fig_bottom = BOTTOM_MARGIN_IN / fig_h_in
    fig_top = 1.0 - TOP_MARGIN_IN / fig_h_in

    for idx, (_, row) in enumerate(pair_hull_df.iterrows()):
        ax_l = axes[idx, 0]
        ax_r = axes[idx, 1]

        structure = Structure.from_dict(all_structures[row["key"]])
        pattern = calc.get_pattern(structure, two_theta_range=(0, 90))
        x = np.array(pattern.x, dtype=float)
        y = np.array(pattern.y, dtype=float)

        m1 = (x >= x1_min) & (x <= x1_max)
        m2 = (x >= x2_min) & (x <= x2_max)

        if use_gaussian_broadening:
            x_grid = np.arange(0.0, 90.0 + x_step_deg, x_step_deg)
            y_grid = gaussian_broadened_profile(x, y, x_grid, sigma=sigma_deg)
            g1 = (x_grid >= x1_min) & (x_grid <= x1_max)
            g2 = (x_grid >= x2_min) & (x_grid <= x2_max)

            ax_l.plot(x_grid[g1], y_grid[g1], color="#111827", linewidth=1.25)
            ax_r.plot(x_grid[g2], y_grid[g2], color="#111827", linewidth=1.25)
            ax_l.vlines(x[m1], 0, y[m1], color="red", linewidth=0.6, alpha=1.0)
            ax_r.vlines(x[m2], 0, y[m2], color="red", linewidth=0.6, alpha=1.0)
        else:
            ax_l.vlines(x[m1], 0, y[m1], color="#111827", linewidth=1.2)
            ax_r.vlines(x[m2], 0, y[m2], color="#111827", linewidth=1.2)

        yticks_this = [0, 20, 40, 60] if idx == 0 else [0, 20, 40]
        for ax in (ax_l, ax_r):
            ax.set_ylim(0, 60)
            ax.set_yticks(yticks_this)
            ax.tick_params(axis="both", labelsize=15.0, length=4.2, width=0.95)
            ax.grid(axis="x", linestyle="--", alpha=0.3, color="#9ca3af")
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_linewidth(1.0)
            ax.spines["left"].set_linewidth(1.0)
            ax.spines["bottom"].set_linewidth(1.0)
            ax.spines["right"].set_linewidth(1.0)

        ax_l.set_xlim(x1_min, x1_max)
        ax_r.set_xlim(x2_min, x2_max)
        ax_l.set_xticks(XTICKS_LEFT)
        ax_r.set_xticks(XTICKS_RIGHT)
        ax_l.set_ylabel("")
        ax_r.tick_params(axis="y", left=False, labelleft=False)
        ax_r.spines["left"].set_visible(False)
        ax_l.spines["right"].set_visible(False)

        # ---- Annotate top-N smeared peaks with their (hkl) from the stick data ----
        if use_gaussian_broadening and len(x) > 0:
            visible_mask = g1 | g2
            top_peaks = top_smeared_peaks(x_grid, y_grid, visible_mask, N_PEAKS_TO_ANNOTATE)
            for peak_x, peak_y in top_peaks:
                stick_i = int(np.argmin(np.abs(x - peak_x)))
                hkl_tuple = pattern.hkls[stick_i][0]["hkl"]
                hkl_str = format_hkl(hkl_tuple)

                if x1_min <= peak_x <= x1_max:
                    target_ax = ax_l
                elif x2_min <= peak_x <= x2_max:
                    target_ax = ax_r
                else:
                    continue

                # Rotated label. For tall peaks (intensity > threshold) the
                # label is shifted slightly to the left of the peak so it does
                # not sit on top of the peak; otherwise it goes just above.
                if peak_y > TALL_PEAK_THRESHOLD:
                    x_anchor = peak_x - TALL_PEAK_X_OFFSET
                else:
                    x_anchor = peak_x
                y_anchor = min(peak_y + 1.5, 40.0)

                target_ax.text(
                    x_anchor, y_anchor, hkl_str,
                    ha="center", va="bottom",
                    rotation=90, rotation_mode="anchor",
                    fontsize=11.0, color="#0b1222",
                )

        phase_label = short_phase(row["phase"])
        ord_label = pretty_ordering(row["ordering"])
        spg_symbol = SpacegroupAnalyzer(structure).get_space_group_symbol()
        spg_label = format_spacegroup(spg_symbol)
        primary = rf"$\mathit{{x}}_{{\mathrm{{Na}}}} = {row['xNa']:.3g}$"
        secondary = rf"{phase_label} $\cdot$ {ord_label}"
        tertiary = rf"SG: {spg_label}"

        ax_r.annotate(
            "",
            xy=(1.025, 0.08), xycoords="axes fraction",
            xytext=(1.025, 0.92), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-", color="#0f172a", linewidth=2.2),
        )
        ax_r.text(1.05, 0.74, primary, transform=ax_r.transAxes,
                  ha="left", va="center", fontsize=12.6, color="#0f172a")
        ax_r.text(1.05, 0.50, secondary, transform=ax_r.transAxes,
                  ha="left", va="center", fontsize=10.6, color="#475569")
        ax_r.text(1.05, 0.26, tertiary, transform=ax_r.transAxes,
                  ha="left", va="center", fontsize=10.0, color="#475569")

    d_slope = 1.5
    mark_kw = dict(marker=[(-1, -d_slope), (1, d_slope)], markersize=9,
                   linestyle="none", color="k", mec="k", mew=1.1, clip_on=False)
    for idx in range(n):
        ax_l = axes[idx, 0]
        ax_r = axes[idx, 1]
        ax_l.plot([1], [0], transform=ax_l.transAxes, **mark_kw)
        ax_r.plot([0], [0], transform=ax_r.transAxes, **mark_kw)
        if idx > 0:
            ax_l.plot([1], [1], transform=ax_l.transAxes, **mark_kw)
            ax_r.plot([0], [1], transform=ax_r.transAxes, **mark_kw)
        if idx < n - 1:
            ax_l.tick_params(axis="x", labelbottom=False)
            ax_r.tick_params(axis="x", labelbottom=False)

    xlabel_x = 0.5 * (fig_left + fig_right)
    xlabel_y = (BOTTOM_MARGIN_IN - XLABEL_OFFSET_IN) / fig_h_in
    ylabel_x = (LEFT_MARGIN_IN - YLABEL_OFFSET_IN) / fig_w_in
    ylabel_y = 0.5 * (fig_bottom + fig_top)

    fig.supxlabel(r"2$\theta$ (deg), Cu K$\alpha$", fontsize=22.0,
                  x=xlabel_x, y=xlabel_y)
    fig.supylabel("Intensity (a.u.)", fontsize=22.0,
                  x=ylabel_x, y=ylabel_y)

    fig.subplots_adjust(left=fig_left, right=fig_right,
                        top=fig_top, bottom=fig_bottom,
                        hspace=0.0, wspace=0.035)
    return fig, axes


# -----------------------------------------------------------------------------
# HULL-POINT SELECTION (verbatim from the notebook batch cell)
# -----------------------------------------------------------------------------
def build_single_hull(single_hull, target_tm, all_structures):
    """Filter singleTM_hull_points by `tm`, sort by hull_order, and build the
    structure key (no `tm_pair`/`elec` segments)."""
    sub = single_hull[single_hull["tm"].astype(str) == target_tm].copy()
    sub["hull_order"] = pd.to_numeric(sub["hull_order"], errors="coerce")
    sub["xNa"] = pd.to_numeric(sub["xNa"], errors="coerce")
    sub = sub.sort_values(["hull_order"]).reset_index(drop=True)
    if sub.empty:
        return sub
    sub["key"] = sub.apply(
        lambda r: f"{r['phase']}-{r['ordering']}-{r['tm']}",
        axis=1,
    )
    missing = [k for k in sub["key"] if k not in all_structures]
    if missing:
        print(f"[{target_tm}] missing {len(missing)} keys (skipped):")
        for k in missing:
            print("   ", k)
    sub = sub[sub["key"].isin(all_structures)].copy().reset_index(drop=True)
    return sub


def build_pair_hull(double_hull, target_pair, all_structures):
    """Filter doubleTM_hull_points by `tm_pair`, sort by (hull_order, elec), and
    build the structure key including the `tm_pair`/`elec` segments."""
    sub = double_hull[double_hull["tm_pair"].astype(str) == target_pair].copy()
    sub["hull_order"] = pd.to_numeric(sub["hull_order"], errors="coerce")
    sub["xNa"] = pd.to_numeric(sub["xNa"], errors="coerce")
    sub["elec"] = sub["elec"].astype(str)
    sub = sub.sort_values(["hull_order", "elec"]).reset_index(drop=True)
    if sub.empty:
        return sub
    sub["key"] = sub.apply(
        lambda r: f"{r['phase']}-{r['ordering']}-{r['tm_pair']}-{r['elec']}",
        axis=1,
    )
    missing = [k for k in sub["key"] if k not in all_structures]
    if missing:
        print(f"[{target_pair}] missing {len(missing)} keys (skipped):")
        for k in missing:
            print("   ", k)
    sub = sub[sub["key"].isin(all_structures)].copy().reset_index(drop=True)
    return sub


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
with structures_json.open() as f:
    all_structures = json.load(f)

calc = XRDCalculator(wavelength="CuKa")

single_hull = pd.read_csv(single_csv)
double_hull = pd.read_csv(double_csv)

saved = []

# ---------------- singleTM (Figure S5) ----------------
all_tms = sorted(single_hull["tm"].dropna().astype(str).unique())
print(f"Generating XRD figures for {len(all_tms)} singleTM systems into {OUT_DIR}")
for tm in all_tms:
    sub = build_single_hull(single_hull, tm, all_structures)
    if sub.empty:
        print(f"[skip] {tm}: no usable hull points")
        continue
    fig, _ = render_xrd_figure(sub, all_structures, calc)
    out_path = OUT_DIR / f"xrd_{tm}.pdf"
    # Save without bbox_inches='tight' so the on-disk figure preserves the exact
    # inch-based size computed by render_xrd_figure (consistent across systems
    # even though they have different row counts).
    fig.savefig(out_path)
    plt.close(fig)
    saved.append(out_path.name)
    print(f"[saved] {tm}: {out_path}  (n={len(sub)})")

# ---------------- doubleTM (Figure S6) ----------------
all_pairs = sorted(double_hull["tm_pair"].dropna().astype(str).unique())
print(f"\nGenerating XRD figures for {len(all_pairs)} doubleTM pairs into {OUT_DIR}")
for pair in all_pairs:
    sub = build_pair_hull(double_hull, pair, all_structures)
    if sub.empty:
        print(f"[skip] {pair}: no usable hull points")
        continue
    fig, _ = render_xrd_figure(sub, all_structures, calc)
    out_path = OUT_DIR / f"xrd_{pair}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    saved.append(out_path.name)
    print(f"[saved] {pair}: {out_path}  (n={len(sub)})")

if use_gaussian_broadening:
    fwhm = 2.3548 * sigma_deg
    print(f"\nGaussian broadening: sigma={sigma_deg:.3f} deg, "
          f"FWHM~{fwhm:.3f} deg, step={x_step_deg:.3f} deg")

print(f"\nSaved {len(saved)} XRD figures (dpi={dpi}) to {OUT_DIR}:")
for s in saved:
    print("  ", s)
