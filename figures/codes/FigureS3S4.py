# Code to generate Supplementary Figure 3, Convex hull of all Na layered oxides.
#
# Generates all 15 convex-hull plots in one run: the 5 single-TM systems
# Na_x(TM)O2 (TM = Ti, Mn, Fe, Co, Ni) and the 10 equimolar mixed-TM pairs
# Na_x(TM1)0.5(TM2)0.5O2, saved as individual PDFs in ../convex_hulls/.
#
# The hull construction follows notebook/dft_convexhull_plot.ipynb (single TM) and
# notebook/dft_convexhull_doubleTM.ipynb (double TM) verbatim; only the plot fonts
# are switched to the Figure 2-5 house style (inward ticks, TrueType embedding).

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
TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]   # the 5 single-TM systems to plot

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"             # .../data
OUT_DIR = SCRIPT_DIR.parent / "convex_hulls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_ORDERINGS = set()

# Figure 2-5 house style (replaces the notebook's larger fonts)
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Ordering -> Na content (x), shared by single- and double-TM data.
NA_MAP = {
    "AJT_a": 1/7, "AJT_b": 1/4, "AJT_c": 1/4, "AJT_d": 1/3, "AJT_e": 1/3,
    "AJT_f": 2/5, "AJT_g": 1/2, "AJT_h": 3/5, "AJT_i": 5/8, "AJT_j": 5/8,
    "AJT_k": 2/3, "AJT_l": 2/3, "AJT_m": 2/3, "AJT_n": 3/4, "AJT_o": 3/4,
    "AJT_p": 5/6, "AJT_q": 7/8, "AJT_r": 8/9,
    "JLK_b": 3/8, "JLK_c": 7/10, "JLK_theta4": 4/5, "JLK_theta5": 5/6,
    "JLK_delta": 2/3, "JLK_eta": 4/7, "JLK_zeta": 1/2, "JLK_zeta2": 2/5,
    "JLK_zeta3": 3/7, "JLK_zeta6": 6/11,
    "CD_a": 1/2, "JMAS_a": 1/3, "JMAS_b": 2/5, "JMAS_e": 1, "XC_a": 1/3,
    "XC_b": 1/2, "JMAS_c": 1/2, "JMAS_d": 2/3, "XL_a": 5/8,
}


# -----------------------------------------------------------------------------
# HELPERS (verbatim from the notebooks)
# -----------------------------------------------------------------------------
def energy_per_TMO2_single(row):
    """Energy per TMO2 unit (eV) for a single-TM row."""
    comp_norm = Composition(row["formula"])
    n_norm = comp_norm.num_atoms
    natoms = row["natoms"]
    if n_norm == 0 or natoms == 0:
        return np.nan
    scale = natoms / n_norm
    n_units = comp_norm[row["tm"]] * scale
    if n_units == 0:
        return np.nan
    if "energy_eV" in row and not pd.isna(row["energy_eV"]):
        e_tot = row["energy_eV"]
    elif "energy_per_atom_eV" in row and not pd.isna(row["energy_per_atom_eV"]):
        e_tot = row["energy_per_atom_eV"] * natoms
    else:
        return np.nan
    return e_tot / n_units


def energy_per_TMO2_double(row):
    """Energy per TMO2 unit (eV) for a double-TM row."""
    natoms = float(row["natoms"])
    tm_list = [x for x in str(row["tm_pair"]).split("_") if x]
    if not tm_list or pd.isna(row["formula"]) or pd.isna(natoms) or natoms <= 0:
        return np.nan
    comp_norm = Composition(row["formula"])
    n_norm = comp_norm.num_atoms
    if n_norm <= 0:
        return np.nan
    scale = natoms / n_norm
    n_units = sum(comp_norm[tm] for tm in tm_list) * scale
    if n_units <= 0:
        return np.nan
    if "energy_eV" in row and not pd.isna(row["energy_eV"]):
        e_tot = float(row["energy_eV"])
    elif "energy_per_atom_eV" in row and not pd.isna(row["energy_per_atom_eV"]):
        e_tot = float(row["energy_per_atom_eV"]) * natoms
    else:
        return np.nan
    return e_tot / n_units


def lower_hull_indices_scipy(xs, ys):
    """Indices of the points on the lower convex hull (energy vs composition)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0:
        return []
    unique_indices = []
    for x0 in sorted(set(xs)):
        idxs = np.where(np.isclose(xs, x0))[0]
        unique_indices.append(idxs[np.argmin(ys[idxs])])
    xs_u = xs[unique_indices]
    ys_u = ys[unique_indices]
    if len(xs_u) <= 2:
        return sorted(unique_indices, key=lambda i: xs[i])
    pts = np.column_stack([xs_u, ys_u])
    verts_sorted = sorted(ConvexHull(pts).vertices, key=lambda i: (xs_u[i], ys_u[i]))
    lower_u = []
    for i in verts_sorted:
        while len(lower_u) >= 2:
            j, k = lower_u[-2], lower_u[-1]
            x1, y1 = xs_u[j], ys_u[j]
            x2, y2 = xs_u[k], ys_u[k]
            x3, y3 = xs_u[i], ys_u[i]
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross <= 0:
                lower_u.pop()
            else:
                break
        lower_u.append(i)
    return sorted([unique_indices[i] for i in lower_u], key=lambda i: xs[i])


def _canonical(s):
    """Normalize ordering names for consistency (e.g. JLK_theta_4 -> JLK_theta4)."""
    return s.replace("JLK_theta_4", "JLK_theta4").replace("JLK_theta_5", "JLK_theta5")


def short_name_single(ordering):
    return _canonical(str(ordering).replace("_w_vac", ""))


def short_name_double(ordering):
    s = str(ordering).replace("_w_vac", "")
    if s.startswith(("O1_", "O3_", "P3_")):
        rest = s.split("_", 1)[1]
        if rest in {"full", "empty"}:
            return rest
    return _canonical(s)


def clean(df):
    """Keep clean, converged runs (notebook convention)."""
    mask = pd.Series(True, index=df.index)
    if "status" in df.columns:
        mask &= df["status"] == "OK"
    if "n_custodian_errors" in df.columns:
        mask &= pd.to_numeric(df["n_custodian_errors"], errors="coerce").fillna(0) == 0
    if "converged" in df.columns:
        c = df["converged"]
        mask &= (c.astype(str).str.lower() == "true") if c.dtype == object else c
    return df[mask].copy()


# -----------------------------------------------------------------------------
# PLOTTING (shared by single and double TM, matching the notebooks)
# -----------------------------------------------------------------------------
def plot_hull(df_pts, xlabel, ylabel, outpath):
    xs = df_pts["xNa"].to_numpy(dtype=float)
    ys = df_pts["E_TMO2_eV"].to_numpy(dtype=float)
    hull_idx = lower_hull_indices_scipy(xs, ys)
    if not hull_idx:
        return False
    hull_mask = np.zeros(len(df_pts), dtype=bool)
    hull_mask[hull_idx] = True

    # subtract the tie-line between hull endpoints so they sit at dE = 0
    if len(hull_idx) >= 2:
        iL, iR = hull_idx[0], hull_idx[-1]
        xL, yL, xR, yR = xs[iL], ys[iL], xs[iR], ys[iR]
        ys_rel = ys - (yL + (yR - yL) * (xs - xL) / (xR - xL)) if xR != xL else ys - ys.min()
    else:
        ys_rel = ys - ys.min()
    df_pts = df_pts.copy()
    df_pts["E_rel_TMO2_eV"] = ys_rel

    # fixed axis range (eV/TMO2) shared by all systems
    y_bot, y_top = -0.7, 0.1

    fig, ax = plt.subplots(figsize=(6, 4.5))
    is_o3 = df_pts["phase"].astype(str).str.contains("O3")
    is_p3 = df_pts["phase"].astype(str).str.contains("P3")
    is_o1 = df_pts["phase"].astype(str).str.contains("O1")

    for phase_mask, marker in [(is_o3, "o"), (is_p3, "s"), (is_o1, "^")]:
        m_non = phase_mask & (~hull_mask)
        m_hul = phase_mask & hull_mask
        if m_non.any():
            ax.scatter(df_pts.loc[m_non, "xNa"], df_pts.loc[m_non, "E_rel_TMO2_eV"],
                       marker=marker, s=130, facecolors="none", edgecolors="red", linewidths=1.7)
        if m_hul.any():
            ax.scatter(df_pts.loc[m_hul, "xNa"], df_pts.loc[m_hul, "E_rel_TMO2_eV"],
                       marker=marker, s=170, facecolors="none", edgecolors="black", linewidths=2.2)

    if len(hull_idx) > 1:
        xs_arr = df_pts["xNa"].to_numpy()
        e_arr = df_pts["E_rel_TMO2_eV"].to_numpy()
        hx = [xs_arr[i] for i in hull_idx]
        hy = [e_arr[i] for i in hull_idx]
        ax.plot(hx, hy, color="black", linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(y_bot, y_top)
    ax.set_box_aspect(1)               # equal x and y axis lengths (square plot box)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which="minor", length=3)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


# -----------------------------------------------------------------------------
# SINGLE-TM HULLS
# -----------------------------------------------------------------------------
df_s = clean(pd.read_csv(DATA_DIR / "singleTM_tighter_conv.csv"))
df_s["E_TMO2_eV"] = df_s.apply(energy_per_TMO2_single, axis=1)
df_s = df_s.dropna(subset=["E_TMO2_eV"]).copy()
ordering_raw = df_s["meta_ordering"].fillna(df_s["ordering"])
df_s["ordering_short"] = ordering_raw.apply(short_name_single)
if EXCLUDE_ORDERINGS:
    df_s = df_s[~df_s["ordering_short"].isin(EXCLUDE_ORDERINGS)].copy()
df_s["xNa"] = df_s["ordering_short"].map(NA_MAP)
df_s.loc[df_s["ordering"] == "full", "xNa"] = 1.0
df_s.loc[df_s["ordering"] == "empty", "xNa"] = 0.0
df_s = df_s.dropna(subset=["xNa"]).copy()

saved = []
for tm in TM_ORDER:
    df_tm = df_s[df_s["tm"] == tm]
    if df_tm.empty:
        continue
    df_pts = (df_tm.groupby(["ordering_short", "phase", "xNa"], as_index=False)
                   .agg({"E_TMO2_eV": "mean"})).reset_index(drop=True)
    out = OUT_DIR / f"convexhull_{tm}.pdf"
    if plot_hull(df_pts, rf"x in Na$_{{\mathrm{{x}}}}${tm}O$_2$",
                 "Formation Energy (eV/TMO$_2$)", out):
        saved.append(out.name)

# -----------------------------------------------------------------------------
# DOUBLE-TM HULLS
# -----------------------------------------------------------------------------
df_d = clean(pd.read_csv(DATA_DIR / "doubleTM_dft_tighter.csv"))
df_d["natoms"] = pd.to_numeric(df_d["natoms"], errors="coerce")
df_d["energy_eV"] = pd.to_numeric(df_d.get("energy_eV"), errors="coerce")
if "energy_per_atom_eV" in df_d.columns:
    df_d["energy_per_atom_eV"] = pd.to_numeric(df_d["energy_per_atom_eV"], errors="coerce")
df_d["ordering_short"] = df_d["ordering"].apply(short_name_double)
if EXCLUDE_ORDERINGS:
    df_d = df_d[~df_d["ordering_short"].isin(EXCLUDE_ORDERINGS)].copy()
df_d["xNa"] = df_d["ordering_short"].map(NA_MAP)
df_d.loc[df_d["ordering_short"] == "full", "xNa"] = 1.0
df_d.loc[df_d["ordering_short"] == "empty", "xNa"] = 0.0
df_d = df_d.dropna(subset=["xNa"]).copy()
df_d["E_TMO2_eV"] = df_d.apply(energy_per_TMO2_double, axis=1)
df_d = df_d.dropna(subset=["E_TMO2_eV"]).copy()
df_d["ordering_elec"] = (df_d["ordering_short"].astype(str) + "__"
                         + df_d["elec"].fillna("elec_unknown").astype(str))

# pair order: lower triangle of TM_ORDER (i < j)
pairs = [f"{TM_ORDER[i]}_{TM_ORDER[j]}"
         for i in range(len(TM_ORDER)) for j in range(i + 1, len(TM_ORDER))]
for pair in pairs:
    df_tm = df_d[df_d["tm_pair"] == pair]
    if df_tm.empty:
        continue
    df_pts = (df_tm.groupby(["ordering_elec", "phase", "xNa"], as_index=False)
                   .agg({"E_TMO2_eV": "mean"})).reset_index(drop=True)
    if len(df_pts) < 2:
        continue
    tm1, tm2 = pair.split("_")
    out = OUT_DIR / f"convexhull_{pair}.pdf"
    if plot_hull(df_pts,
                 f"x in Na$_x${tm1}$_{{0.5}}${tm2}$_{{0.5}}$O$_2$",
                 "Formation Energy (eV/TMO$_2$)", out):
        saved.append(out.name)

print(f"Saved {len(saved)} convex-hull plots to {OUT_DIR}:")
for s in saved:
    print("  ", s)
