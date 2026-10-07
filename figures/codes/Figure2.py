# Figure 2: Composite of DFT voltage curves + volume / c-axis change, referenced to experiment.
#
# Layout: lower triangle of a 5x5 transition-metal matrix (TM_ORDER).
#   - diagonal cell (i, i)      -> single-TM system  Na_x (TM_i) O2
#   - off-diagonal cell (i, j<i) -> double-TM system Na_x (TM_j)0.5 (TM_i)0.5 O2
# The empty upper-right triangle holds the shared legend.
#
# Each panel: black DFT voltage step curve (left axis)
#             + volume (gray) and c-axis length per TM layer (purple) as % change relative to
#               the x = 1 hull point (right twin)
#             + red dashed experimental curve where available.
# Shared ticks: x-labels only on the bottom row, voltage labels only on the left
#               column, right-axis labels only on the diagonal (rightmost panel of a row).

#

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator
from scipy.spatial import ConvexHull
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]

E_Na_ref = -1.31
PHASE_CHANGE_EDGE_WIDTH = 0.03

VOLT_YLIM = (0.0, 5.5)
VOLT_TICKS = [0, 1, 2, 3, 4, 5]
REL_YLIM = (-25.0, 12.5)          # right axis: % change of volume and c relative to x = 1
REL_TICKS = [-20, -10, 0, 10]
X_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0] 

PANEL_W, PANEL_H = 2.4, 2.4
PHASE_COLORS = {"O3": "#ff4d4f", "P3": "#1d8cff", "O1": "#2ca02c"}
PHASE_ALPHA = 0.20
PHASE_ALPHA_BY_KIND = {"O3": 0.5, "P3": 0.5, "O1": 0.6}
C_COLOR = "#6a3d9a"               # c-axis series
C_MARKER_ALPHA = 0.5              # face alpha of the c-axis markers

SHOW_REFS = False

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
EXP_DIR = DATA_DIR / "experiments"
OUT_PATH = SCRIPT_DIR.parent / "Figure2_voltage_caxis.pdf"

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

# Ordering -> Na content (x). Shared by single- and double-TM data.
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

# Experimental references, keyed by single-TM symbol or double-TM "TMa_TMb".
EXP_SINGLE = {
    "Ti": {"file": "EES2015_NTO.csv",      "label": "Wu et al. EES 2015",        "is_capacity": True},
    "Mn": {"file": "AFM2018_NMO_charge.csv","label": "Chen et al. AFM 2018",      "is_capacity": False},
    "Fe": {"file": "AEM2018_NFO.csv",      "label": "Kubota et al. AEM 2018",    "is_capacity": False},
    "Co": {"file": "AFM2016_NCO.csv",      "label": "Kubota et al. AFM 2016",    "is_capacity": True},
    "Ni": {"file": "ChemRxiv_NNO.csv",     "label": "Steele et al. JACS 2026","is_capacity": False},
}
EXP_DOUBLE = {
    "Mn_Fe": {"file": "NM2012_NMF.csv",  "label": "Yabuuchi et al. Nat. Mater. 2012", "is_capacity": True},
    "Fe_Ni": {"file": "JMCA2017_NFN.csv","label": "Vassilaras et al. JMCA 2017",      "is_capacity": False},
    "Co_Ni": {"file": "JMCA2017_NCN.csv","label": "Vassilaras et al. JMCA 2017",      "is_capacity": False},
    "Mn_Ni": {"file": "IC2012_NMN.csv",  "label": "Komaba et al. Inorg. Chem. 2012",  "is_capacity": False},
    "Ti_Ni": {"file": "CM2016_NTN.csv",  "label": "Nanba et al. Chem. Mater. 2016",   "is_capacity": False},
    "Fe_Co": {"file": "AFM2016_NFC.csv", "label": "Kubota et al. AFM 2016",           "is_capacity": False},
}

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def short_name(ordering: str) -> str:
    """Normalize an ordering label and collapse endmember aliases to full/empty."""
    s = str(ordering).replace("_w_vac", "")
    if s.startswith(("O1_", "O3_", "P3_")):
        rest = s.split("_", 1)[1]
        if rest in {"full", "empty"}:
            return rest
    return s


def merge_intervals(intervals, tol=1e-9):
    """Merge touching/overlapping (x0, x1) intervals to avoid white seams."""
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


def phase_kind(phase_str: str) -> str:
    s = str(phase_str)
    for k in ("O3", "P3", "O1"):
        if k in s:
            return k
    return "other"


def lattice_c_from_cell(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan
    try:
        d = cell if isinstance(cell, dict) else ast.literal_eval(str(cell))
        return float(Lattice.from_dict(d).c)
    except Exception:
        return np.nan


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


def volume_per_fu(row, tm_list):
    """Cell volume normalized to one TMO2 formula unit (Angstrom^3)."""
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


LATTICE_IJK = np.array([v for v in itertools.product(range(-4, 5), repeat=3) if any(v)])


def c_per_layer(struct, tm_list):
    """c-axis length per TM layer = mean TM-layer spacing along the stacking normal (Angstrom).

    Supercells hold 1-3 layers and the a-b plane is not the layer plane in all of them
    (e.g. XL_a), so the layer plane is rebuilt from the lattice: the two shortest translations
    lying in a TM layer span one layer cell of area A, the stacking period is V / A, and that
    period holds n TM layers.
    """
    tm = [s for s in struct if s.specie.symbol in tm_list]
    rel = np.array([s.coords - tm[0].coords for s in struct.get_sites_in_sphere(tm[0].coords, 14.0)
                    if s.specie.symbol in tm_list])
    r = np.linalg.norm(rel, axis=1)
    normal = np.linalg.svd(rel[(r > 0.1) & (r < 3.7)])[2][-1]        # from the 6 in-plane TM neighbours
    normal = np.linalg.svd(rel[np.abs(rel @ normal) < 1.0])[2][-1]   # refined on the whole TM layer
    T = LATTICE_IJK @ struct.lattice.matrix
    T = T[(np.linalg.norm(T, axis=1) < 30.0) & (np.abs(T @ normal) < 1.0)]   # in-plane translations
    T = T[np.argsort(np.linalg.norm(T, axis=1))]
    t2 = next(t for t in T if np.linalg.norm(np.cross(T[0], t)) > 1.0)
    area = np.cross(T[0], t2)                                        # layer cell; exact layer normal
    period = struct.volume / np.linalg.norm(area)
    h = np.sort((np.array([s.coords for s in tm]) @ area / np.linalg.norm(area)) % period)
    n_layers = np.sum(np.diff(np.append(h, h[0] + period)) > 2.0)
    return period / n_layers


def formula_label(tm_list):
    """Bold mathtext composition label, e.g. Na_x Ti_0.5 Ni_0.5 O_2 (upright subscripts)."""
    if len(tm_list) == 1:
        body = r"Na_{x}" + tm_list[0] + r"O_{2}"
    else:
        body = r"Na_{x}" + tm_list[0] + r"_{0.5}" + tm_list[1] + r"_{0.5}" + r"O_{2}"
    return r"$\mathbf{" + body + r"}$"


def theoretical_capacity_mAh_g(tm_list):
    """Capacity for 1 Na per TMO2 (mAh/g)."""
    n = len(tm_list)
    frac = "".join(f"{tm}{1.0/n}" for tm in tm_list)
    M = Composition(f"Na1{frac}O2").weight
    return 96485.33212 / 3.6 / M


def lower_hull_indices(xs, ys):
    """Indices of points on the lower convex hull (energy vs composition)."""
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


def robust_load_csv(path: Path) -> pd.DataFrame:
    """Load a 2-column experimental CSV, skipping metadata/header lines."""
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=[0, 1])
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "[#" or "source" in line.lower():
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=[0, 1])


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


def build_curve(df_sub, cfg_col):
    """Convex-hull voltage curve + % change of volume and c per layer from one system's rows."""
    pts = (df_sub.groupby([cfg_col, "phase", "xNa"], as_index=False)
                 .agg({"E_TMO2_eV": "mean", "vol_fu_A3": "mean", "c_layer_A": "mean"}))
    if len(pts) < 2:
        return None
    xs = pts["xNa"].to_numpy(float)
    ys = pts["E_TMO2_eV"].to_numpy(float)
    idx = lower_hull_indices(xs, ys)
    if len(idx) < 2:
        return None
    idx = sorted(idx, key=lambda i: xs[i])
    x_h = np.array([xs[i] for i in idx])
    e_h = np.array([ys[i] for i in idx])
    # volume per formula unit is intensive -> no layer-count scaling needed
    vol_plot = np.array([pts.iloc[i]["vol_fu_A3"] for i in idx])
    c_plot = np.array([pts.iloc[i]["c_layer_A"] for i in idx])
    # % change relative to the most sodiated hull point (x = 1)
    vol_plot = 100.0 * (vol_plot / vol_plot[-1] - 1.0)
    c_plot = 100.0 * (c_plot / c_plot[-1] - 1.0)
    ph_h = [str(pts.iloc[i]["phase"]) for i in idx]

    x_plot, v_plot = [], []
    regions = {"O3": [], "P3": [], "O1": []}
    for i in range(len(x_h) - 1):
        x1, x2 = x_h[i], x_h[i + 1]
        if np.isclose(x1, x2):
            continue
        mu = (e_h[i + 1] - e_h[i]) / (x2 - x1)
        V = -(mu - E_Na_ref)
        x_plot += [x1, x2]
        v_plot += [V, V]
        kl, kr = phase_kind(ph_h[i]), phase_kind(ph_h[i + 1])
        if kl == kr and kl in regions:
            regions[kl].append((x1, x2))
        else:
            edge = min(PHASE_CHANGE_EDGE_WIDTH, 0.5 * (x2 - x1))
            if kl in regions:
                regions[kl].append((x1, x1 + edge))
            if kr in regions:
                regions[kr].append((x2 - edge, x2))
    if len(x_plot) < 2:
        return None
    return dict(x_plot=x_plot, v_plot=v_plot, x_hull=x_h, vol_plot=vol_plot, c_plot=c_plot,
                regions=regions)


def load_experiment(cfg, tm_list):
    """Return (x, v) experimental arrays in composition units, or None."""
    if cfg is None:
        return None
    df_exp = robust_load_csv(EXP_DIR / cfg["file"])
    if df_exp.empty:
        return None
    raw_x = df_exp[0].to_numpy(float)
    raw_v = df_exp[1].to_numpy(float)
    C = theoretical_capacity_mAh_g(tm_list)
    max_x = np.nanmax(raw_x) if len(raw_x) else 0.0
    # Auto-detect: capacity values are large (mAh/g); composition values are <=1.
    if cfg["is_capacity"] and max_x < 5.0:
        x = raw_x
    elif (not cfg["is_capacity"]) and max_x > 5.0:
        x = 1.0 - raw_x / C
    elif cfg["is_capacity"]:
        x = 1.0 - raw_x / C
    else:
        x = raw_x
    ok = np.isfinite(x) & np.isfinite(raw_v)
    if not ok.any():
        return None
    order = np.argsort(x[ok])
    return x[ok][order], raw_v[ok][order]


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

# c-axis length per TM layer from the relaxed structures (key = relpath with "/" -> "-")
structs = {}
for name in ("singleTM_structures.json", "doubleTM_structures.json"):
    with open(DATA_DIR / name) as f:
        structs.update(json.load(f))
struct_of = lambda relpath: Structure.from_dict(structs[str(relpath).replace("/", "-")])
df_s = df_s[df_s["tm"].isin(TM_ORDER)].copy()     # the CSV also holds Cr and V (not plotted)
df_s["c_layer_A"] = df_s.apply(lambda r: c_per_layer(struct_of(r["relpath"]), [r["tm"]]), axis=1)
df_d["c_layer_A"] = df_d.apply(
    lambda r: c_per_layer(struct_of(r["relpath"]), str(r["tm_pair"]).split("_")), axis=1)

# -----------------------------------------------------------------------------
# FIGURE
# -----------------------------------------------------------------------------
n = len(TM_ORDER)
fig, axes = plt.subplots(n, n, figsize=(PANEL_W * n, PANEL_H * n),
                         gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ref_entries = []

for r in range(n):
    for c in range(n):
        ax = axes[r][c]
        if c > r:                     # empty upper-right triangle
            ax.axis("off")
            continue

        if r == c:                      # diagonal -> single TM
            tm_list = [TM_ORDER[r]]
            df_sub = df_s[df_s["tm"] == TM_ORDER[r]]
            exp_cfg = EXP_SINGLE.get(TM_ORDER[r])
        else:                           # off-diagonal -> pair
            key = f"{TM_ORDER[c]}_{TM_ORDER[r]}"
            tm_list = [TM_ORDER[c], TM_ORDER[r]]
            df_sub = df_d[df_d["tm_pair"] == key]
            exp_cfg = EXP_DOUBLE.get(key)

        curve = build_curve(df_sub, "cfg") if not df_sub.empty else None

        ax_c = ax.twinx()
        if curve is not None:
            for kind, regs in curve["regions"].items():
                for x0, x1 in merge_intervals(regs):
                    ax.axvspan(x0, x1, facecolor=PHASE_COLORS[kind],
                               alpha=PHASE_ALPHA_BY_KIND.get(kind, PHASE_ALPHA),
                               edgecolor="none", linewidth=0, antialiased=False, zorder=0)
            ax.plot(curve["x_plot"], curve["v_plot"], color="black", linewidth=2.0, zorder=3)
            ax_c.axhline(0, color="0.35", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
            valid = np.isfinite(curve["vol_plot"])
            if valid.any():
                ax_c.plot(curve["x_hull"][valid], curve["vol_plot"][valid], color="tab:gray",
                          marker="s", markersize=7, markeredgecolor="black", markeredgewidth=1.0,
                          linewidth=1.4, linestyle=(0, (3.5, 2.2)), alpha=0.9, zorder=2)
            ax_c.plot(curve["x_hull"], curve["c_plot"], color=C_COLOR, marker="o", markersize=6.5,
                      markerfacecolor=to_rgba(C_COLOR, C_MARKER_ALPHA), markeredgecolor="black",
                      markeredgewidth=1.0, linewidth=1.4, zorder=2.5)

        exp = load_experiment(exp_cfg, tm_list)
        if exp is not None:
            ax.plot(exp[0], exp[1], color="tab:red", linestyle=(0, (3, 2)),
                    linewidth=1.8, alpha=0.95, zorder=4)
            ref_entries.append(("_".join(tm_list), formula_label(tm_list), exp_cfg["label"]))

        # axes ranges + shared ticks
        ax.set_xlim(0, 1)
        ax.set_ylim(*VOLT_YLIM)
        ax_c.set_ylim(*REL_YLIM)
        ax.set_xticks(X_TICKS)
        ax.set_yticks(VOLT_TICKS)
        ax_c.set_yticks(REL_TICKS)
        ax.xaxis.set_minor_locator(MultipleLocator(0.125))   # half of x major (0.25)
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))     # half of voltage major (1)
        ax_c.yaxis.set_minor_locator(MultipleLocator(5))     # half of % change major (10)
        ax.tick_params(labelbottom=(r == n - 1), labelleft=(c == 0))
        ax.tick_params(which="both", top=True)
        ax.tick_params(which="minor", length=3)
        ax_c.tick_params(labelright=(c == r), right=(c == r))
        ax_c.tick_params(which="minor", length=3, right=(c == r))
        # x labels (bottom row): hide 1.0 except on the rightmost column
        if r == n - 1:
            ax.set_xticklabels([f"{t:g}" if (t < 0.99 or c == n - 1) else "" for t in X_TICKS])
        # voltage labels (left column): hide bottom '0' except on the last row to
        # avoid colliding with the '5' of the panel below
        if c == 0:
            ax.set_yticklabels([f"{t:g}" if (t > 0.01 or r == n - 1) else "" for t in VOLT_TICKS])
        # % change labels: show the full tick set on every right-axis (diagonal) panel
        if c == r:
            ax_c.set_yticklabels([f"{t:g}".replace("-", "\u2212") for t in REL_TICKS])
        else:
            ax_c.set_yticklabels([])

        # TM sub-labels along the matrix axes: column (left) TM under the bottom row,
        # row (right) TM to the left of the first column.
        if r == n - 1:
            ax.set_xlabel(TM_ORDER[c], fontsize=20, labelpad=6)
        if c == 0:
            ax.set_ylabel(TM_ORDER[r], fontsize=20, labelpad=10, rotation=0,
                          ha="right", va="center")

# reorder experimental references: singles first, then pairs (Ti-first ordering)
REF_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni",
             "Ti_Ni", "Mn_Fe", "Mn_Ni", "Fe_Co", "Fe_Ni", "Co_Ni"]
order_index = {k: i for i, k in enumerate(REF_ORDER)}
ref_entries.sort(key=lambda e: order_index.get(e[0], len(REF_ORDER)))
ref_entries = [(formula, cite) for _, formula, cite in ref_entries]

# shared axis labels
AXIS_LABEL_SIZE = 23
fig.supxlabel(r"x in Na$_{\mathrm{x}}$TM$_{\mathrm{0.5}}$TM'$_{\mathrm{0.5}}$O$_{\mathrm{2}}$",
              fontsize=AXIS_LABEL_SIZE, y=0.02)
fig.supylabel(r"Voltage (V vs Na/Na$^+$)", fontsize=AXIS_LABEL_SIZE, x=0.02)
fig.text(0.967, 0.5, "Volume and c-axis change upon desodiation (%)",
         rotation=90, va="center", ha="center", fontsize=AXIS_LABEL_SIZE)

# combined legend + experimental references, in the empty upper-right triangle
legend_ax = fig.add_subplot(axes[0][0].get_gridspec()[0:2, 2:n])
LEGEND_SHIFT_X = 0.02              # figure fraction; moves the box and its contents to the right
lpos = legend_ax.get_position()
legend_ax.set_position([lpos.x0 + LEGEND_SHIFT_X, lpos.y0, lpos.width, lpos.height])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis("off")

series = [
    ("line",  dict(color="black", lw=2.2),                               "Predicted voltages, this work"),
    ("line",  dict(color="tab:red", lw=2.0, linestyle=(0, (3, 2))),      "Exp. measured voltages"),
    ("sline", dict(color="tab:gray", lw=1.5, linestyle=(0, (3.5, 2.2))), "Volume change"),
    ("oline", dict(color=C_COLOR, lw=1.5),                               "c-axis length change"),
    ("patch", dict(facecolor=PHASE_COLORS["O3"], alpha=PHASE_ALPHA_BY_KIND["O3"]), "O3 stable"),
    ("patch", dict(facecolor=PHASE_COLORS["P3"], alpha=PHASE_ALPHA_BY_KIND["P3"]), "P3 stable"),
    ("patch", dict(facecolor=PHASE_COLORS["O1"], alpha=PHASE_ALPHA_BY_KIND["O1"]), "O1 stable"),
]

top, dy_item, dy_ref = 0.90, 0.11, 0.046
group_gap = 0.035                    # extra gap (holds the divider) between curve and phase entries
box_left, box_right = 0.111, 0.985   # box horizontal extent within legend_ax
gx0, gx1, tx = 0.149, 0.224, 0.247   # glyph start/end and text x
text_x = 0.149                       # left edge of header/reference text

# precompute row centers so the enclosing box fits the content exactly
y = top
series_y = []
for i, (kind, _, _) in enumerate(series):
    if kind == "patch" and i > 0 and series[i - 1][0] != "patch":
        y_div = y + (dy_item - group_gap) / 2   # midway between last curve and first phase entry
        y -= group_gap
    series_y.append(y)
    y -= dy_item

show_refs = SHOW_REFS and len(ref_entries) > 0
ref_y = []
if show_refs:
    y_header = y - dy_item * 0.55
    y_sep = (series_y[-1] + y_header) / 2   # equal gap: O1 item <-> sep <-> header
    yy = y_header - dy_ref * 1.4            # extra space below the header
    for _ in ref_entries:
        ref_y.append(yy)
        yy -= dy_ref
    box_bottom = yy + dy_ref * 0.05
else:
    box_bottom = y + dy_item - (0.985 - top)   # same padding below the last row as above the first

# enclosing box (behind everything): rounded corners + soft drop shadow.
# mutation_aspect = legend_ax width/height keeps the corners circular on the page.
box_aspect = (lpos.width * fig.get_figwidth()) / (lpos.height * fig.get_figheight())
box_style = dict(boxstyle="round,pad=0,rounding_size=0.03", mutation_aspect=box_aspect)
legend_ax.add_patch(FancyBboxPatch((box_left + 0.007, box_bottom - 0.007 * box_aspect),
                                   box_right - box_left, 0.985 - box_bottom,
                                   facecolor="0.45", edgecolor="none", alpha=0.45, zorder=0,
                                   **box_style))
legend_ax.add_patch(FancyBboxPatch((box_left, box_bottom), box_right - box_left, 0.985 - box_bottom,
                                   facecolor="white", edgecolor="0.2", linewidth=1.6, zorder=1,
                                   **box_style))

# standard legend entries
for (kind, st, label), yc in zip(series, series_y):
    if kind in ("line", "sline", "oline"):
        legend_ax.plot([gx0, gx1], [yc, yc], zorder=2, **st)
        if kind == "sline":      # volume: gray squares
            legend_ax.plot([(gx0 + gx1) / 2], [yc], marker="s", markersize=9,
                           markeredgecolor="black", markeredgewidth=1.0,
                           color=st["color"], zorder=3)
        elif kind == "oline":    # c axis: semi-transparent circles
            legend_ax.plot([(gx0 + gx1) / 2], [yc], marker="o", markersize=8.5,
                           markerfacecolor=to_rgba(st["color"], C_MARKER_ALPHA),
                           markeredgecolor="black", markeredgewidth=1.0, zorder=3)
    else:
        legend_ax.add_patch(Rectangle((gx0, yc - 0.024), gx1 - gx0, 0.048,
                                      facecolor=st["facecolor"], alpha=st["alpha"],
                                      edgecolor="none", zorder=2))
    legend_ax.text(tx, yc, label, va="center", ha="left", fontsize=19.5, zorder=2)

# dotted divider between the curve entries and the stable-phase shading entries
legend_ax.plot([box_left + 0.04, box_right - 0.04], [y_div, y_div], linestyle=(0, (1, 1.6)),
               color="gray", lw=1.1, zorder=2)

# dotted separator + experimental references
if show_refs:
    legend_ax.plot([box_left + 0.03, box_right - 0.03], [y_sep, y_sep], linestyle=(0, (1, 1.6)),
                   color="gray", lw=1.1, zorder=2)
    legend_ax.text(text_x, y_header, "Experimental references", va="center", ha="left",
                   fontsize=12.5, fontweight="bold", zorder=2)
    for (formula, cite), yc in zip(ref_entries, ref_y):
        legend_ax.text(text_x, yc, f"{formula}: {cite}", va="center", ha="left",
                       fontsize=10.4, zorder=2)

fig.savefig(OUT_PATH, dpi=500, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
