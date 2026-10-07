# Shared setting module for the Figure 4 series (Figure4a/b/c/d.py import this).
# Importing it loads structures.json and builds the per-system `cells` once.
#
# For each system the ground-state (lowest-energy) config is selected at every Na
# content x, then three structural metrics are computed from the relaxed structure:
#   - d(O-Na-O): z-thickness of the Na (alkali) interlayer  -> left axis, blue line
#   - d(O-TM-O): z-thickness of the TM-O slab               -> left axis, black line
#   - TM-O bond lengths split into equatorial (4 short) and axial (2 long) per
#     element -> right axis, shaded band whose WIDTH is the Jahn-Teller distortion.
#
# Layout mirrors Figure2.py: lower triangle of TM_ORDER.
#   diagonal (i, i)      -> single-TM   Na_x (TM_i) O2
#   off-diagonal (i, j<i)-> double-TM   Na_x (TM_j)0.5 (TM_i)0.5 O2
# The empty upper-right triangle holds the shared legend.
#
# Shared ticks: x-labels on the bottom row only, slab labels (left axis) on the
# left column only, bond labels (right axis) on the diagonal panel of each row.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull
from pymatgen.core import Structure
from pymatgen.core.composition import Composition

# -----------------------------------------------------------------------------
# PARAMETERS  (edit these)
# -----------------------------------------------------------------------------
TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]   # rows/cols of the triangle, top->bottom

SLAB_YLIM = (1.6, 3.8)        # left axis (Fig4a): slab thickness (Angstrom)
SLAB_TICKS = [2.0, 2.5, 3.0, 3.5]
RATIO_YLIM = (1.2, 2.0)       # right axis (Fig4a): d(O-Na-O)/d(O-TM-O) ratio (-)
RATIO_TICKS = [1.2, 1.4, 1.6, 1.8, 2.0]   # 1.2 labeled only on the bottom-right panel
JT_YLIM = (1.0, 1.25)         # left axis (Fig4b): mean axial/equatorial TM-O ratio (-)
JT_TICKS = [1.0, 1.1, 1.2]
X_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]        # '1.0' shown only on the rightmost panel

PANEL_W, PANEL_H = 2.4, 2.4   # inches per panel (square)

COL_NA = "#6a3d9a"            # d(O-Na-O) line (purple: avoids the blue P3 shading)
COL_TM = "black"              # d(O-TM-O) line
COL_RATIO = "#1b9e77"         # d(O-Na-O)/d(O-TM-O) ratio line (teal)
# axial/equatorial ratio line, one color + one marker per TM (distinct from slab lines)
ELEM_COLORS = {"Ti": "#8c564b", "Mn": "#9467bd", "Fe": "#ff7f0e",
               "Co": "#e377c2", "Ni": "#2ca02c"}
ELEM_MARKERS = {"Ti": "o", "Mn": "s", "Fe": "^", "Co": "D", "Ni": "v"}

# stable-phase background shading (same palette as Figure2.py)
PHASE_COLORS = {"O3": "#ff4d4f", "P3": "#1d8cff", "O1": "#2ca02c"}
PHASE_ALPHA = {"O3": 0.22, "P3": 0.22, "O1": 0.30}
PHASE_CHANGE_EDGE_WIDTH = 0.03

Z_PLANE_TOL = 0.6             # Angstrom: cluster O atoms into planes within this z-tol

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"              # .../data
STRUCT_PATH = DATA_DIR / "structures.json"
OUT_SLAB = SCRIPT_DIR.parent / "Figure4a_slab_thickness.pdf"
OUT_JT = SCRIPT_DIR.parent / "Figure4b_TMO_ratio.pdf"
OUT_RATIO_ALL = SCRIPT_DIR.parent / "Figure4c_ratio_all.pdf"
OUT_SCATTER = SCRIPT_DIR.parent / "Figure4d_ratio_scatter.pdf"

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

# Ordering -> Na content (x). Shared by single- and double-TM data (from Figure2.py).
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


# -----------------------------------------------------------------------------
# HELPERS  (data loading / energy -- shared logic with Figure2.py)
# -----------------------------------------------------------------------------
def short_name(ordering: str) -> str:
    """Normalize an ordering label and collapse endmember aliases to full/empty."""
    s = str(ordering).replace("_w_vac", "")
    if s.startswith(("O1_", "O3_", "P3_")):
        rest = s.split("_", 1)[1]
        if rest in {"full", "empty"}:
            return rest
    return s


def energy_per_TMO2(row, tm_list):
    """Total energy normalized to one TMO2 formula unit (eV)."""
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


def load_clean(path: Path) -> pd.DataFrame:
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


def add_xna(df: pd.DataFrame, ordering_short: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["ordering_short"] = ordering_short
    df["xNa"] = df["ordering_short"].map(NA_MAP)
    df.loc[df["ordering_short"] == "full", "xNa"] = 1.0
    df.loc[df["ordering_short"] == "empty", "xNa"] = 0.0
    return df


def lower_hull_indices(xs, ys):
    """Indices of points on the lower convex hull (energy vs composition). From Fig2."""
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


def phase_kind(phase_str: str) -> str:
    s = str(phase_str)
    for k in ("O3", "P3", "O1"):
        if k in s:
            return k
    return "other"


def merge_intervals(intervals, tol=1e-9):
    """Merge touching/overlapping (x0, x1) intervals to avoid white seams. From Fig2."""
    iv = sorted((min(a, b), max(a, b)) for a, b in intervals)
    if not iv:
        return []
    merged = [list(iv[0])]
    for x0, x1 in iv[1:]:
        if x0 <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [tuple(m) for m in merged]


def phase_regions(xs, phases):
    """Stable-phase x-intervals along the hull, with thin edges at phase changes."""
    regions = {"O3": [], "P3": [], "O1": []}
    kinds = [phase_kind(p) for p in phases]
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        if x2 <= x1:
            continue
        kl, kr = kinds[i], kinds[i + 1]
        if kl == kr and kl in regions:
            regions[kl].append((x1, x2))
        else:
            edge = min(PHASE_CHANGE_EDGE_WIDTH, 0.5 * (x2 - x1))
            if kl in regions:
                regions[kl].append((x1, x1 + edge))
            if kr in regions:
                regions[kr].append((x2 - edge, x2))
    return regions


def struct_key(relpath: str) -> str:
    """structures.json key = relpath with '/' replaced by '-' (verified)."""
    return str(relpath).replace("/", "-")


# -----------------------------------------------------------------------------
# HELPERS  (geometry from a relaxed structure)
# -----------------------------------------------------------------------------
def _planes_frac(fc, tol_frac):
    """Cluster fractional c-coords into plane centers in [0,1), wrap-aware.

    Rotate the sorted coords past the largest circular gap so no plane straddles
    the 0/1 boundary, cluster linearly, then fold centers back into [0,1).
    """
    f = np.sort(np.asarray(fc, float) % 1.0)
    gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))   # n gaps incl. wrap
    roll = (np.argmax(gaps) + 1) % len(f)               # start just after biggest gap
    f2 = np.roll(f, -roll)
    f2 = np.where(f2 < f2[0], f2 + 1.0, f2)             # make monotonic increasing
    groups = [[f2[0]]]
    for z in f2[1:]:
        if z - groups[-1][-1] <= tol_frac:
            groups[-1].append(z)
        else:
            groups.append([z])
    return np.sort(np.array([np.mean(g) % 1.0 for g in groups]))


def slab_thicknesses(s: Structure):
    """Mean TM-slab and Na-interlayer thickness along the layer normal (Angstrom).

    Tilt-invariant: planes are found in fractional c (the stacking coordinate),
    and a fractional gap is converted to a perpendicular thickness via the (001)
    interplanar spacing d001 = V / |a x b|. Each O-plane gap is a TM-O slab or a
    Na interlayer, classified by which cation type lies in the gap.
    """
    L = s.lattice
    d001 = L.volume / np.linalg.norm(np.cross(L.matrix[0], L.matrix[1]))
    tol_frac = Z_PLANE_TOL / d001          # ~0.6 A plane-clustering tolerance
    sp = [str(x) for x in s.species]
    fc = s.frac_coords[:, 2] % 1.0
    o_idx = [i for i in range(len(s)) if sp[i] == "O"]
    tm_idx = [i for i in range(len(s)) if sp[i] not in ("O", "Na")]
    na_idx = [i for i in range(len(s)) if sp[i] == "Na"]
    planes = _planes_frac(fc[o_idx], tol_frac)
    n = len(planes)
    tm_slabs, na_slabs = [], []
    for k in range(n):
        z0 = planes[k]
        gapf = (planes[(k + 1) % n] - z0) % 1.0
        in_tm = sum(any(z0 - 1e-6 < fc[i] + m < z0 + gapf + 1e-6 for m in (0, 1, -1))
                    for i in tm_idx)
        in_na = sum(any(z0 - 1e-6 < fc[i] + m < z0 + gapf + 1e-6 for m in (0, 1, -1))
                    for i in na_idx)
        if in_tm > 0 and in_na > 0:
            return np.nan, np.nan      # mixed interlayer -> not cleanly layered
        if in_tm > 0:
            tm_slabs.append(gapf * d001)
        else:
            na_slabs.append(gapf * d001)
    d_tm = float(np.mean(tm_slabs)) if tm_slabs else np.nan
    d_na = float(np.mean(na_slabs)) if na_slabs else np.nan
    return d_na, d_tm


def jt_ratio(s: Structure):
    """Per-element Jahn-Teller ratio, averaged over that element's octahedra.

    For each TM octahedron: ratio = mean(2 longest TM-O bonds) / mean(4 shortest).
    Per-octahedron ratios are averaged within each element (equal site weight), so
    1.0 = regular octahedron and larger = stronger tetragonal (JT) elongation.
    Returns {element: mean ratio}.
    """
    sp = [str(x) for x in s.species]
    acc = {}   # element -> list of per-octahedron ratios
    for i in range(len(s)):
        el = sp[i]
        if el in ("O", "Na"):
            continue
        # all O within a cutoff, including periodic images -> the true octahedron
        d = sorted(nb.nn_distance for nb in s.get_neighbors(s[i], 3.2)
                   if nb.specie.symbol == "O")
        if len(d) < 6:
            continue
        d6 = np.array(d[:6])
        acc.setdefault(el, []).append(np.mean(d6[4:]) / np.mean(d6[:4]))   # ax / eq
    # per element: (mean ratio, std across that element's octahedra in this cell)
    return {el: (float(np.mean(v)), float(np.std(v))) for el, v in acc.items()}


def system_points(df_sub, tm_list, structures):
    """Ground-state structural metrics vs x for one system.

    Returns dict with sorted arrays x, d_na, d_tm and jt = {element: (x, ratio) array}
    of the per-element axial/equatorial ratio. At each x the lowest-energy config is used.
    """
    df_sub = df_sub.dropna(subset=["xNa", "E_TMO2_eV"]).copy()
    if df_sub.empty:
        return None
    df_sub["xr"] = df_sub["xNa"].round(4)
    best = df_sub.loc[df_sub.groupby("xr")["E_TMO2_eV"].idxmin()].sort_values("xr")
    # keep only thermodynamically stable (lower convex-hull) compositions
    hull = lower_hull_indices(best["xr"].to_numpy(), best["E_TMO2_eV"].to_numpy())
    if len(hull) < 1:
        return None
    best = best.iloc[hull].sort_values("xr")
    regions = phase_regions(best["xr"].to_numpy(float), best["phase"].tolist())

    x, d_na, d_tm, phase = [], [], [], []
    jt = {el: [] for el in tm_list}
    for _, row in best.iterrows():
        key = struct_key(row["relpath"])
        if key not in structures:
            continue
        s = Structure.from_dict(structures[key])
        dn, dt = slab_thicknesses(s)
        if np.isnan(dn) or np.isnan(dt):
            print(f"  skip (non-layered): {key}  x={row['xr']:.3f}")
            continue
        x.append(row["xr"]); d_na.append(dn); d_tm.append(dt)
        phase.append(phase_kind(row["phase"]))
        ratios = jt_ratio(s)
        for el in tm_list:
            if el in ratios:
                jt[el].append((row["xr"], ratios[el][0], ratios[el][1]))
    if not x:
        return None
    jt = {el: np.array(v) for el, v in jt.items() if v}
    return dict(x=np.array(x), d_na=np.array(d_na), d_tm=np.array(d_tm), jt=jt,
                phase=np.array(phase), regions=regions)


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
structures = json.load(open(STRUCT_PATH))

df_s = load_clean(DATA_DIR / "singleTM_tighter_conv.csv")
df_s["E_TMO2_eV"] = df_s.apply(lambda r: energy_per_TMO2(r, [r["tm"]]), axis=1)
df_s = add_xna(df_s, df_s["meta_ordering"].fillna(df_s["ordering"]).apply(short_name))

df_d = load_clean(DATA_DIR / "doubleTM_dft_tighter.csv")
df_d["E_TMO2_eV"] = df_d.apply(lambda r: energy_per_TMO2(r, str(r["tm_pair"]).split("_")), axis=1)
df_d = add_xna(df_d, df_d["ordering"].apply(short_name))

# -----------------------------------------------------------------------------
# FIGURE
# -----------------------------------------------------------------------------
n = len(TM_ORDER)
AXIS_LABEL_SIZE = 23

# precompute ground-state structural metrics for every lower-triangle system once
cells = {}
for r in range(n):
    for c in range(r + 1):
        if r == c:
            tm_list = [TM_ORDER[r]]
            df_sub = df_s[df_s["tm"] == TM_ORDER[r]]
        else:
            tm_list = [TM_ORDER[c], TM_ORDER[r]]
            df_sub = df_d[df_d["tm_pair"] == f"{TM_ORDER[c]}_{TM_ORDER[r]}"]
        cells[(r, c)] = (tm_list, system_points(df_sub, tm_list, structures)
                         if not df_sub.empty else None)


def shade_phases(ax, regions):
    """Tint each panel by the stable phase over composition (same as Figure2.py)."""
    for kind, regs in regions.items():
        for x0, x1 in merge_intervals(regs):
            ax.axvspan(x0, x1, facecolor=PHASE_COLORS[kind], alpha=PHASE_ALPHA[kind],
                       edgecolor="none", linewidth=0, antialiased=False, zorder=0)


def phase_handles():
    """Legend patches for the stable-phase shading."""
    return [Patch(facecolor=PHASE_COLORS[k], alpha=PHASE_ALPHA[k], edgecolor="none",
                  label=f"{k} stable") for k in ("O3", "P3", "O1")]


def style_legend(legend_ax, handles, **kw):
    """Clean white box with a thin black border, in the Figure2.py style."""
    leg = legend_ax.legend(handles=handles, loc="center", frameon=True, fancybox=False,
                           edgecolor="black", facecolor="white", framealpha=1.0,
                           borderpad=1.0, labelspacing=0.35, handlelength=2.3,
                           handletextpad=0.9, **kw)
    leg.get_frame().set_linewidth(1.3)
    return leg
