# Figure 4e: cationic (ionic) potential as a predictor of O3/P3 phase selection.
#
# Ionic radii (in nm) from Zhao et al., Sci. 2020 (aay9972), Table S2 (ref. 46).
# Key = "<element>_<charge>" plus "_hs"/"_ls" only when both spin states are listed;
# single-valued ions keep "<element>_<charge>". Ionic potential = charge / radius,
# so it is recomputable from the key (charge) and value (radius) — not stored here.
IONIC_RADII_NM = {
    # charge +1
    "Li_1": 0.0760, "Na_1": 0.1020, "K_1": 0.1380, "Cu_1": 0.0770, "Ag_1": 0.1150,
    # charge +2
    "Mg_2": 0.0720, "Ca_2": 0.1000,
    "Mn_2_hs": 0.0830, "Mn_2_ls": 0.0670,
    "Fe_2_hs": 0.0645, "Fe_2_ls": 0.0550,
    "Co_2_hs": 0.0745, "Co_2_ls": 0.0650,
    "Ni_2": 0.0690, "Cu_2": 0.0730, "Zn_2": 0.0740, "Sr_2": 0.1170, "Ba_2": 0.1350,
    # charge +3
    "B_3": 0.0270, "Al_3": 0.0535, "Sc_3": 0.0745, "Ti_3": 0.0670, "V_3": 0.0640,
    "Cr_3": 0.0615,
    "Mn_3_hs": 0.0645, "Mn_3_ls": 0.0580,
    "Fe_3_hs": 0.0645, "Fe_3_ls": 0.0550,
    "Co_3_hs": 0.0610, "Co_3_ls": 0.0545,
    "Ni_3_hs": 0.0600, "Ni_3_ls": 0.0560,
    "Cu_3": 0.0540,
    "Y_3": 0.0900, "Nb_3": 0.0720, "Mo_3": 0.0690, "Rh_3": 0.0665, "Sb_3": 0.0760,
    "La_3": 0.1032, "Lu_3": 0.0861, "Ir_3": 0.0680, "Bi_3": 0.1030,
    # charge +4
    "Si_4": 0.0400, "Ge_4": 0.0530, "Ti_4": 0.0605, "V_4": 0.0580, "Cr_4": 0.0550,
    "Mn_4": 0.0530, "Fe_4": 0.0585, "Co_4": 0.0530, "Ni_4": 0.0480, "Zr_4": 0.0720,
    "Nb_4": 0.0680, "Mo_4": 0.0650, "Ru_4": 0.0620, "Rh_4": 0.0600, "Pd_4": 0.0615,
    "Sn_4": 0.0690, "Ir_4": 0.0625, "Pt_4": 0.0625,
    # charge +5
    "V_5": 0.0540, "Nb_5": 0.0640, "Mo_5": 0.0610, "Ru_5": 0.0565, "Sb_5": 0.0600,
    "Hf_5": 0.0710, "Ta_5": 0.0640, "W_5": 0.0620, "Ir_5": 0.0570, "Bi_5": 0.0760,
    # charge +6
    "Mo_6": 0.0590, "Te_6": 0.0560, "W_6": 0.0600,
}

import ast
import math
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from matplotlib.ticker import MultipleLocator
from scipy.spatial import ConvexHull
from pymatgen.core.composition import Composition

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
TM_ORDER = ["Ti", "Mn", "Fe", "Co", "Ni"]

# spin state used to pick the ionic radius for ions that list both hs/ls (editable).
# standard layered-oxide convention: Mn/Fe high-spin, Co3+/Ni3+ low-spin, Co2+ high-spin.
SPIN = {("Mn", 2): "hs", ("Mn", 3): "hs",
        ("Fe", 2): "hs", ("Fe", 3): "hs",
        ("Co", 2): "hs", ("Co", 3): "ls",
        ("Ni", 3): "ls"}

# O3/P3 marker colors = the phase-shading palette used in the Fig 2 / Fig 4 backgrounds.
PHASE_COLORS = {"O3": "#ff4d4f", "P3": "#1d8cff"}
# Set MARKER_ALPHA to the background alpha you used (e.g. 0.22) to reproduce the faded
# tint; it is pre-blended onto white so overlapping markers don't darken. 1.0 = solid.
MARKER_ALPHA = 1.0


def _on_white(color, a):
    """Opaque equivalent of `color` drawn at alpha `a` over a white background."""
    r, g, b, _ = to_rgba(color)
    return (a * r + (1 - a), a * g + (1 - a), a * b + (1 - a))


PHASE_RGB = {k: _on_white(c, MARKER_ALPHA) for k, c in PHASE_COLORS.items()}

LEGEND_FONT = 17           # pt, text in the legend box (tick labels: 19 pt)
LEGEND_BOX_LINE = 1.6      # pt, outline of the legend box
# legend box of Figure2.py: rounded white box with a soft drop shadow
LEGEND_BOX = dict(frameon=True, fancybox=True, facecolor="white", edgecolor="0.2", framealpha=1.0,
                  shadow=dict(ox=3, oy=-3, fc="0.45", ec="none", alpha=0.45))

R_O_NM = 0.140                          # O2- ionic radius -> Phi_O = 4/0.140 = 28.5714
R_NA_NM = IONIC_RADII_NM["Na_1"]        # 0.1020
PHI_O = 4.0 / R_O_NM                    # per formula unit (two O2-): 28.57143

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
OUT_PATH = SCRIPT_DIR.parent / "Figure4e_cationic_potential.pdf"

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
    "font.size": 13, "axes.labelsize": 23,           # match Figure 4d
    "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 14,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.direction": "in", "ytick.direction": "in",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# IONIC-POTENTIAL HELPERS
# -----------------------------------------------------------------------------
def radius_nm(el, charge):
    """Ionic radius (nm) for (element, integer charge), applying the SPIN choice.
    Returns None if the species is not in IONIC_RADII_NM (caller skips the point)."""
    base = f"{el}_{charge}"
    if base in IONIC_RADII_NM:
        return IONIC_RADII_NM[base]
    spin = SPIN.get((el, charge))
    if spin and f"{base}_{spin}" in IONIC_RADII_NM:
        return IONIC_RADII_NM[f"{base}_{spin}"]
    return None


def phi_tm(species):
    """Phi_TM = sum_i w_i * n_i / R_i over (element, charge, content) species.
    Returns None if any species radius is unavailable."""
    total = 0.0
    for el, q, w in species:
        r = radius_nm(el, q)
        if r is None:
            return None
        total += w * q / r
    return total


def phi_cation(species, x):
    """Zhao 2020 cationic potential = (Phi_TM * Phi_Na) / Phi_O (per formula unit)."""
    ptm = phi_tm(species)
    if ptm is None:
        return None, None
    p_na = x / R_NA_NM
    return ptm * p_na / PHI_O, p_na


def phase_kind(phase_str):
    s = str(phase_str)
    for k in ("O3", "P3", "O1"):
        if k in s:
            return k
    return "other"


def species_from_avg(el, content, avg):
    """Split an average oxidation `avg` of element `el` (total content) into the two
    adjacent integer charges so that sum(content*charge) is preserved."""
    lo = int(math.floor(avg + 1e-9))
    hi = int(math.ceil(avg - 1e-9))
    if hi <= lo:
        return [(el, lo, content)]
    f_hi = avg - lo
    return [(el, lo, content * (1.0 - f_hi)), (el, hi, content * f_hi)]


def guess_species(tm1, tm2, x):
    """Best-guess TM species when DFT charges are unavailable: pymatgen
    Composition.oxi_state_guesses() (ICSD-probability ranked) on Na_x TM1.5 TM2.5 O2,
    then split each TM's guessed average charge into integer species. None if no guess."""
    fr = Fraction(x).limit_denominator(24)
    m = 2 * (fr.denominator or 1)               # scale so Na/TM/O counts are integers
    counts = {"Na": round(x * m), tm1: round(0.5 * m), tm2: round(0.5 * m), "O": 2 * m}
    counts = {k: v for k, v in counts.items() if v > 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guesses = Composition(counts).oxi_state_guesses()
    if not guesses:
        return None
    g = guesses[0]
    species = []
    for el in (tm1, tm2):
        if el not in g:
            return None
        species += species_from_avg(el, 0.5, g[el])
    return species


# -----------------------------------------------------------------------------
# SINGLE-TM hull-stable points (charge = 4 - x by neutrality; no DFT oxi file)
# -----------------------------------------------------------------------------
def short_name(ordering):
    s = str(ordering).replace("_w_vac", "")
    if s.startswith(("O1_", "O3_", "P3_")):
        rest = s.split("_", 1)[1]
        if rest in {"full", "empty"}:
            return rest
    return s


def energy_per_TMO2(row, tm):
    formula, natoms = row["formula"], float(row["natoms"])
    if pd.isna(formula) or natoms <= 0:
        return np.nan
    comp = Composition(formula)
    if comp.num_atoms <= 0 or comp[tm] <= 0:
        return np.nan
    scale = natoms / comp.num_atoms
    if not pd.isna(row.get("energy_eV")):
        e_tot = float(row["energy_eV"])
    elif not pd.isna(row.get("energy_per_atom_eV")):
        e_tot = float(row["energy_per_atom_eV"]) * natoms
    else:
        return np.nan
    return e_tot / (comp[tm] * scale)


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


def single_tm_stable_points():
    """Yield (tm, x, phase) for each single-TM hull-stable composition."""
    df = pd.read_csv(DATA_DIR / "singleTM_tighter_conv.csv")
    if "status" in df:
        df = df[df["status"] == "OK"]
    if "converged" in df:
        df = df[df["converged"].astype(str).str.lower() == "true"]
    for col in ("natoms", "energy_eV", "energy_per_atom_eV"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.copy()
    df["E"] = df.apply(lambda r: energy_per_TMO2(r, r["tm"]), axis=1)
    osn = df["meta_ordering"].fillna(df["ordering"]).apply(short_name)
    df["xNa"] = osn.map(NA_MAP)
    df.loc[osn == "full", "xNa"] = 1.0
    df.loc[osn == "empty", "xNa"] = 0.0
    df = df.dropna(subset=["xNa", "E"])
    for tm in TM_ORDER:
        sub = df[df["tm"] == tm].copy()
        if sub.empty:
            continue
        sub["xr"] = sub["xNa"].round(4)
        best = sub.loc[sub.groupby("xr")["E"].idxmin()].sort_values("xr")
        hull = lower_hull_indices(best["xr"].to_numpy(), best["E"].to_numpy())
        for _, row in best.iloc[hull].iterrows():
            yield tm, float(row["xr"]), phase_kind(row["phase"])


# -----------------------------------------------------------------------------
# COLLECT POINTS  (phi_cation, phi_na, phase, label)
# -----------------------------------------------------------------------------
points = []
skipped = []
guessed = []

# --- double-TM: per-site integer charges from DFT; oxi_state_guesses fallback ---
dfo = pd.read_csv(DATA_DIR / "doubleTM_stable_oxi_states.csv")
seen = set()
for _, row in dfo.iterrows():
    ph = phase_kind(row["phase"])
    if ph not in ("O3", "P3"):
        continue
    x = float(row["xNa"])
    key = (row["tm_pair"], round(x, 4))
    if key in seen:
        continue
    tm1, tm2 = row["tm1"], row["tm2"]

    species = None
    if str(row["charge_neutral"]).strip().lower() == "true":
        try:
            c1 = [int(q) for q in ast.literal_eval(row["ox_tm1_list"])]
            c2 = [int(q) for q in ast.literal_eval(row["ox_tm2_list"])]
            if c1 and c2:
                ntot = len(c1) + len(c2)
                species = []
                for el, cl in ((tm1, c1), (tm2, c2)):
                    for q in sorted(set(cl)):
                        species.append((el, q, cl.count(q) / ntot))
        except Exception:
            species = None
    is_guess = False
    if species is None:                       # no DFT charges -> best guess
        species = guess_species(tm1, tm2, x)
        is_guess = species is not None
    if species is None:
        skipped.append(f"{row['tm_pair']} x={x:.3f} (no DFT charge; guess failed)")
        continue

    pcat, pna = phi_cation(species, x)
    if pcat is None:
        skipped.append(f"{row['tm_pair']} x={x:.3f} (radius missing)")
        continue
    points.append((pcat, pna, ph, "double"))
    seen.add(key)
    if is_guess:
        guessed.append(f"{row['tm_pair']} x={x:.3f}")

# --- single-TM: charge = 4 - x, split into adjacent integer charges ---
for tm, x, ph in single_tm_stable_points():
    if ph not in ("O3", "P3"):
        continue
    species = species_from_avg(tm, 1.0, 4.0 - x)   # charge = 4 - x by neutrality
    pcat, pna = phi_cation(species, x)
    if pcat is None:
        skipped.append(f"{tm} x={x:.3f} (radius missing)")
        continue
    points.append((pcat, pna, ph, "single"))

points = [p for p in points if p[1] > 1e-9]   # drop meaningless x=0 (origin) points
print(f"plotted {len(points)} points; {len(guessed)} via oxi_state_guesses; skipped {len(skipped)}")
for s in guessed:
    print(f"  guessed charges: {s}")
for s in skipped:
    print(f"  skip: {s}")

# -----------------------------------------------------------------------------
# FIGURE
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 6.0))
KIND_MARKER = {"single": "^", "double": "v"}   # shape = TM type; color = phase
for ph, col in PHASE_RGB.items():
    for kind, mk in KIND_MARKER.items():
        sub = [p for p in points if p[2] == ph and p[3] == kind]
        if not sub:
            continue
        xs = [p[0] for p in sub]
        ys = [p[1] for p in sub]
        ax.scatter(xs, ys, marker=mk, s=150, facecolors="none", edgecolors=col,
                   linewidths=1.3, zorder=3)

# O3/P3 boundary line, spanning the full x-range
xline = np.array([0.0, 20.0])
ax.plot(xline, 0.4065 * xline + 1.572, color="black", lw=1.5, linestyle="--", zorder=2)

ax.set_xlim(0, 20)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_box_aspect(1)               # square plot box (equal x/y axis lengths)
ax.set_xlabel(r"$\Phi_{\mathrm{cation}}$")
ax.set_ylabel(r"$\overline{\Phi_{\mathrm{Na}}}$")
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.tick_params(which="major", length=6)
ax.tick_params(which="minor", length=3)
ax.tick_params(which="both", top=True, right=True)      # major and minor ticks on all four sides
ax.grid(True, color="0.9", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

handles = [Patch(facecolor=PHASE_RGB["O3"], edgecolor="none", label="O3"),
           Patch(facecolor=PHASE_RGB["P3"], edgecolor="none", label="P3"),
           Line2D([], [], marker="^", markerfacecolor="none", markeredgecolor="0.35",
                  markeredgewidth=1.3, linestyle="none", markersize=12, label="single TM"),
           Line2D([], [], marker="v", markerfacecolor="none", markeredgecolor="0.35",
                  markeredgewidth=1.3, linestyle="none", markersize=12, label="mixed TM")]
leg = ax.legend(handles=handles, loc="upper left", fontsize=LEGEND_FONT,
                borderpad=0.7, labelspacing=0.4, handletextpad=0.5,
                borderaxespad=1.0,               # clear of the ticks on the top and left side
                **LEGEND_BOX)
leg.get_frame().set_linewidth(LEGEND_BOX_LINE)
leg.get_frame().set_boxstyle("round", pad=0, rounding_size=0.5)

fig.savefig(OUT_PATH, dpi=500, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
