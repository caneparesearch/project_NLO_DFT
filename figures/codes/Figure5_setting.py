# Shared setting module for the Figure 5 series (Figure5a/b.py import this and call
# make_pareto). Holds parameters, data loading, the Pareto computation, and plotting.
#
# Figure 5: volume change (%) vs gravimetric energy density (mWh/g).
#
# For each of the 15 systems (5 single-TM + 10 double-TM pairs), start from the
# fully sodiated stable structure (x=1) and desodiate through each stable (convex-
# hull) point. At every stable point we plot:
#   x = volume change (%) relative to the fully sodiated cell, per TMO2 formula unit
#   y = cumulative gravimetric energy density (mWh/g) obtained by integrating the
#       convex-hull voltage profile from x=1 down to that point.
# Energy density is normalized to the fully sodiated formula mass (Na1 TMO2), the
# same convention used for the theoretical capacity in Figure 2.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial import ConvexHull
from pymatgen.core.composition import Composition

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]
E_Na_ref = -1.31            # Na reference energy (eV), from Materials Project
F_C_PER_MOL = 96485.33212   # Faraday constant
N_A = 6.02214076e23         # Avogadro constant

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
OUT_GRAV = SCRIPT_DIR.parent / "Figure5a_pareto_gravimetric.pdf"
OUT_VOL = SCRIPT_DIR.parent / "Figure5b_pareto_volumetric.pdf"

NA_MAP = {
    "AJT_a": 1/7, "AJT_b": 1/4, "AJT_c": 1/4, "AJT_d": 1/3, "AJT_e": 1/3,
    "AJT_f": 2/5, "AJT_g": 1/2, "AJT_h": 3/5, "AJT_i": 5/8, "AJT_j": 5/8,
    "AJT_k": 2/3, "AJT_l": 2/3, "AJT_m": 2/3, "AJT_n": 3/4, "AJT_o": 3/4,
    "AJT_p": 5/6, "AJT_q": 7/8, "AJT_r": 8/9,
    "JLK_b": 3/8, "JLK_c": 7/10, "JLK_theta_4": 4/5, "JLK_theta_5": 5/6,
    "JLK_delta": 2/3, "JLK_eta": 4/7, "JLK_zeta": 1/2, "JLK_zeta2": 2/5,
    "JLK_zeta3": 3/7, "JLK_zeta6": 6/11,
    "CD_a": 1/2, "JMAS_a": 1/3, "JMAS_b": 2/5, "JMAS_e": 1, "XC_a": 1/3,
    "XC_b": 1/2, "JMAS_c": 1/2, "JMAS_d": 2/3, "XL_a": 5/8,
}

plt.rcParams.update({
    "font.size": 17,
    "axes.labelsize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 15,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# HELPERS  (identical conventions to Figure 2)
# -----------------------------------------------------------------------------
def short_name(ordering):
    s = str(ordering).replace("_w_vac", "")
    if s.startswith(("O1_", "O3_", "P3_")):
        rest = s.split("_", 1)[1]
        if rest in {"full", "empty"}:
            return rest
    return s


def energy_per_TMO2(row, tm_list):
    formula, natoms = row["formula"], float(row["natoms"])
    if pd.isna(formula) or natoms <= 0:
        return np.nan
    comp = Composition(formula)
    if comp.num_atoms <= 0:
        return np.nan
    scale = natoms / comp.num_atoms
    n_units = sum(comp[tm] for tm in tm_list) * scale
    if n_units <= 0:
        return np.nan
    if not pd.isna(row.get("energy_eV")):
        e_tot = float(row["energy_eV"])
    elif not pd.isna(row.get("energy_per_atom_eV")):
        e_tot = float(row["energy_per_atom_eV"]) * natoms
    else:
        return np.nan
    return e_tot / n_units


def volume_per_fu(row, tm_list):
    formula, natoms, vol = row["formula"], float(row["natoms"]), row.get("volume_A3")
    if pd.isna(formula) or natoms <= 0 or pd.isna(vol):
        return np.nan
    comp = Composition(formula)
    if comp.num_atoms <= 0:
        return np.nan
    scale = natoms / comp.num_atoms
    n_units = sum(comp[tm] for tm in tm_list) * scale
    if n_units <= 0:
        return np.nan
    return float(vol) / n_units


def fu_mass(tm_list):
    n = len(tm_list)
    frac = "".join(f"{tm}{1.0/n}" for tm in tm_list)
    return Composition(f"Na1{frac}O2").weight


def lower_hull_indices(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) == 0:
        return []
    uniq = [np.where(np.isclose(xs, x0))[0][np.argmin(ys[np.where(np.isclose(xs, x0))[0]])]
            for x0 in sorted(set(xs))]
    xu, yu = xs[uniq], ys[uniq]
    if len(xu) <= 2:
        return sorted(uniq, key=lambda i: xs[i])
    verts = sorted(ConvexHull(np.column_stack([xu, yu])).vertices, key=lambda i: (xu[i], yu[i]))
    lower = []
    for i in verts:
        while len(lower) >= 2:
            j, k = lower[-2], lower[-1]
            cross = (xu[k] - xu[j]) * (yu[i] - yu[j]) - (yu[k] - yu[j]) * (xu[i] - xu[j])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(i)
    return sorted([uniq[i] for i in lower], key=lambda i: xs[i])


def load_clean(path):
    df = pd.read_csv(path)
    mask = pd.Series(True, index=df.index)
    if "status" in df:
        mask &= df["status"] == "OK"
    if "n_custodian_errors" in df:
        mask &= pd.to_numeric(df["n_custodian_errors"], errors="coerce").fillna(0) == 0
    if "converged" in df:
        c = df["converged"]
        mask &= (c.astype(str).str.lower() == "true") if c.dtype == object else c
    df = df[mask].copy()
    for col in ("natoms", "energy_eV", "energy_per_atom_eV", "volume_A3"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_xna(df, ordering_short):
    df = df.copy()
    df["ordering_short"] = ordering_short
    df["xNa"] = df["ordering_short"].map(NA_MAP)
    df.loc[df["ordering_short"] == "full", "xNa"] = 1.0
    df.loc[df["ordering_short"] == "empty", "xNa"] = 0.0
    return df


def trajectory(df_sub, cfg_col, tm_list):
    """Return (volume_change_%, energy_density_mWh_g) at each stable point,
    ordered from fully sodiated (origin) toward full desodiation."""
    pts = (df_sub.groupby([cfg_col, "phase", "xNa"], as_index=False)
                 .agg({"E_TMO2_eV": "mean", "vol_fu_A3": "mean"}))
    if len(pts) < 2:
        return None
    xs = pts["xNa"].to_numpy(float)
    es = pts["E_TMO2_eV"].to_numpy(float)
    vs = pts["vol_fu_A3"].to_numpy(float)
    idx = lower_hull_indices(xs, es)
    if len(idx) < 2:
        return None
    idx = sorted(idx, key=lambda i: xs[i])           # ascending x
    x_h, e_h, v_h = xs[idx], es[idx], vs[idx]
    ref = len(x_h) - 1                                # most sodiated stable point (max x)

    # cumulative energy (eV per f.u.) integrating V dx from x=1 down to each point
    ev = np.zeros(len(x_h))
    for i in range(ref - 1, -1, -1):
        mu = (e_h[i + 1] - e_h[i]) / (x_h[i + 1] - x_h[i])
        V = -(mu - E_Na_ref)
        ev[i] = ev[i + 1] + V * (x_h[i + 1] - x_h[i])

    K = F_C_PER_MOL / 3.6 / fu_mass(tm_list)         # mAh/g per unit x
    energy_density = K * ev                           # mWh/g
    vol_change = (v_h - v_h[ref]) / v_h[ref] * 100.0  # %

    order = np.argsort(-x_h)                          # sodiated -> desodiated
    vc, ed, xo = vol_change[order], energy_density[order], x_h[order]
    keep = xo > 1e-9                                  # drop fully desodiated (x=0): unrealistic
    vc, ed, xo = vc[keep], ed[keep], xo[keep]
    # x metric = the MAXIMUM strain experienced along the path up to each point
    # (running max of |volume change|), monotonically non-decreasing on desodiation
    strain = np.maximum.accumulate(np.abs(vc))
    # fully sodiated (pristine) crystal density, for volumetric energy density
    rho = fu_mass(tm_list) / (v_h[ref] * 1e-24 * N_A)   # g/cm^3
    return strain, ed, xo, rho


def label_single(tm):
    return tm


def label_double(t1, t2):
    return f"{t1}-{t2}"


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
df_s = load_clean(DATA_DIR / "singleTM_tighter_conv.csv")
df_s["E_TMO2_eV"] = df_s.apply(lambda r: energy_per_TMO2(r, [r["tm"]]), axis=1)
df_s["vol_fu_A3"] = df_s.apply(lambda r: volume_per_fu(r, [r["tm"]]), axis=1)
df_s = add_xna(df_s, df_s["meta_ordering"].fillna(df_s["ordering"]).apply(short_name))
df_s["cfg"] = df_s["ordering_short"]
df_s = df_s.dropna(subset=["xNa", "E_TMO2_eV", "vol_fu_A3"])

df_d = load_clean(DATA_DIR / "doubleTM_dft_tighter.csv")
df_d["E_TMO2_eV"] = df_d.apply(lambda r: energy_per_TMO2(r, str(r["tm_pair"]).split("_")), axis=1)
df_d["vol_fu_A3"] = df_d.apply(lambda r: volume_per_fu(r, str(r["tm_pair"]).split("_")), axis=1)
df_d = add_xna(df_d, df_d["ordering"].apply(short_name))
df_d["cfg"] = df_d["ordering_short"].astype(str) + "__" + df_d["elec"].fillna("e?").astype(str)
df_d = df_d.dropna(subset=["xNa", "E_TMO2_eV", "vol_fu_A3"])

# -----------------------------------------------------------------------------
# BUILD THE 15 CURVES  (5 single-TM, then 10 pairs)
# -----------------------------------------------------------------------------
singles = []   # (label, tm_list, traj)
for tm in TM_ORDER:
    tr = trajectory(df_s[df_s["tm"] == tm], "cfg", [tm])
    if tr is not None:
        singles.append((label_single(tm), [tm], tr))

doubles = []
for i in range(len(TM_ORDER)):
    for j in range(i + 1, len(TM_ORDER)):
        t1, t2 = TM_ORDER[i], TM_ORDER[j]
        key = f"{t1}_{t2}"
        tr = trajectory(df_d[df_d["tm_pair"] == key], "cfg", [t1, t2])
        if tr is not None:
            doubles.append((label_double(t1, t2), [t1, t2], tr))

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
colors = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)   # >= 15
markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "d", "H", "8"]

XLIM = (-0.3, 9)           # absolute volume change (%): 0 strain (best) at left
YLIM = (0, 800)            # gravimetric (Wh/kg)
YLIM_VOL = (0, 3200)       # volumetric (Wh/L)

# ticks: major and minor ticks on all four sides of the plot. Every major step holds three minor
# ticks, and the one halfway between two major ticks is drawn longer.
X_MAJOR = 2                # % between two labeled ticks of the strain axis
Y_MAJOR = 100              # Wh/kg between two labeled ticks (gravimetric)
Y_MAJOR_VOL = 500          # Wh/L between two labeled ticks (volumetric)
MINOR_PER_MAJOR = 4        # a major step is divided into four, which gives three minor ticks
TICK_MINOR = 3.0           # pt, length of a minor tick (major ticks: 6 pt, see rcParams)
TICK_HALFWAY = 4.5         # pt, length of the minor tick halfway between two major ticks


# -----------------------------------------------------------------------------
# PARETO PLOT  (single + double together; winners highlighted/labeled)
#   volumetric=False -> gravimetric energy density (Wh/kg)
#   volumetric=True  -> volumetric energy density (Wh/L), using each system's
#                       fully sodiated crystal density
# -----------------------------------------------------------------------------
def make_pareto(out_path, volumetric=False, figsize=(9.5, 8.0), label_size=None, square=True):
    # per-system energy metric and strain
    curves = []   # (label, is_single, strain, energy)
    for lab, tml, (strain, ed, xh, rho), is_single in (
            [(l, t, tr, True) for l, t, tr in singles]
            + [(l, t, tr, False) for l, t, tr in doubles]):
        energy = ed * rho if volumetric else ed     # Wh/kg * g/cm^3 (=kg/L) -> Wh/L
        curves.append((lab, is_single, strain, energy))

    # all (strain, energy) points; "good" = high energy AND low peak strain
    pts = [(c, e, si) for si, (_, _, strain, energy) in enumerate(curves)
           for c, e in zip(strain, energy)]

    def dominated(c, e):
        return any((c2 <= c and e2 >= e) and (c2 < c or e2 > e) for c2, e2, _ in pts)

    nd = [(c, e, si) for (c, e, si) in pts if not dominated(c, e)]
    # a system "wins" if its achievable endpoint (most-desodiated retained point) is Pareto-optimal
    winners = {si for si, (_, _, strain, energy) in enumerate(curves)
               if not dominated(strain[-1], energy[-1])}
    front = sorted({(round(c, 4), round(e, 4)) for c, e, _ in nd}, key=lambda t: t[1])
    fv, fe = [p[0] for p in front], [p[1] for p in front]
    front_pts = set(front)

    ylim = YLIM_VOL if volumetric else YLIM
    ylabel = ("Vol. energy density (Wh L$^{-1}$)" if volumetric
              else "Grav. energy density (Wh kg$^{-1}$)")
    yspan = ylim[1] - ylim[0]

    fig, ax = plt.subplots(figsize=figsize)
    wpalette = list(plt.cm.tab10.colors)
    wcolor = {si: wpalette[i % len(wpalette)] for i, si in enumerate(sorted(winners))}

    for si, (lab, is_single, strain, energy) in enumerate(curves):
        mk = "o" if is_single else "D"
        if si in winners:
            col, lw, ms, al, z = wcolor[si], 2.6, 11, 0.95, 4
        else:
            col, lw, ms, al, z = "0.72", 1.3, 6, 0.85, 2
        ax.plot(strain, energy, color=col, marker=mk, markersize=ms, markeredgecolor="black",
                markeredgewidth=0.7, linewidth=lw, alpha=al, zorder=z)
        # thicker outline on markers that lie on the Pareto front
        on_front = [(c, e) for c, e in zip(strain, energy)
                    if (round(c, 4), round(e, 4)) in front_pts]
        if on_front:
            ax.plot([p[0] for p in on_front], [p[1] for p in on_front], linestyle="none",
                    marker=mk, markersize=ms, markerfacecolor=col, markeredgecolor="black",
                    markeredgewidth=2.4, alpha=al, zorder=5.5)

    ax.plot(fv, fe, color="black", lw=2.2, linestyle=(0, (5, 3)), zorder=5)

    # ---- labels for all systems; collisions split about x-midpoint (thresholds scale with y-range)
    DX = 0.65
    DY = 0.0275 * yspan
    OFFY, OFFY_WIN = 0.006 * yspan, 0.0225 * yspan   # winners sit a bit higher
    pos = [[curves[si][2][-1], curves[si][3][-1] + (OFFY_WIN if si in winners else OFFY)]
           for si in range(len(curves))]
    for _ in range(200):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if abs(pos[i][0] - pos[j][0]) < DX and abs(pos[i][1] - pos[j][1]) < DY:
                    mid = (pos[i][0] + pos[j][0]) / 2
                    lo, hi = (i, j) if pos[i][0] <= pos[j][0] else (j, i)
                    pos[lo][0] = mid - DX / 2
                    pos[hi][0] = mid + DX / 2
                    moved = True
        if not moved:
            break
    for si in range(len(curves)):
        lx, ly = pos[si]
        if si in winners:
            ax.text(lx, ly, curves[si][0], ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=wcolor[si], zorder=6)
        else:
            ax.text(lx, ly, curves[si][0], ha="center", va="bottom", fontsize=8.5,
                    color="0.45", zorder=6)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    if square:
        ax.set_box_aspect(1)

    # ---- major and minor ticks on all four sides, the halfway minor tick longer
    y_major = Y_MAJOR_VOL if volumetric else Y_MAJOR
    for axis, major in ((ax.xaxis, X_MAJOR), (ax.yaxis, y_major)):
        axis.set_major_locator(MultipleLocator(major))
        axis.set_minor_locator(MultipleLocator(major / MINOR_PER_MAJOR))
    ax.tick_params(which="both", top=True, right=True)
    ax.tick_params(which="minor", length=TICK_MINOR)
    for axis, major in ((ax.xaxis, X_MAJOR), (ax.yaxis, y_major)):
        for position, tick in zip(axis.get_minorticklocs(), axis.get_minor_ticks()):
            if abs(position % major - major / 2) < 1e-6 * major:      # halfway between two major ticks
                tick.tick1line.set_markersize(TICK_HALFWAY)             # bottom or left side
                tick.tick2line.set_markersize(TICK_HALFWAY)             # top or right side

    lab_kw = {"fontsize": label_size} if label_size else {}
    ax.set_xlabel("Peak volumetric strain (%)", **lab_kw)
    ax.set_ylabel(ylabel, **lab_kw)
    ax.grid(True, color="0.92", lw=0.7, zorder=0)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", lw=0, markerfacecolor="0.7", markeredgecolor="black",
                  markersize=10, label="single TM"),
           Line2D([0], [0], marker="D", lw=0, markerfacecolor="0.7", markeredgecolor="black",
                  markersize=10, label="double TM"),
           Line2D([0], [0], color="black", lw=2.2, linestyle=(0, (5, 3)), label="Pareto front")]
    leg = ax.legend(handles=leg, loc="lower right", frameon=True, fancybox=False,
                    edgecolor="black", facecolor="white", framealpha=1.0, fontsize=13,
                    borderpad=1.0, labelspacing=0.5, handlelength=2.3, handletextpad=0.9,
                    borderaxespad=1.0)        # 13 pt from the frame, clear of the ticks (default 0.5)
    leg.get_frame().set_linewidth(1.3)

    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(winners)} Pareto-front systems)")
