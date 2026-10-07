# Van Vleck octahedral distortion modes for every stable hull point.
#
# For each row of singleTM_hull_points.csv / doubleTM_hull_points.csv the relaxed
# structure is taken from singleTM_structures.json / doubleTM_structures.json
# (key = "<phase>-<ordering>-<tm>[-<elec>]") and, for every TM octahedron in the
# cell, the Van Vleck modes Q1-Q6, rho = sqrt(Q2^2+Q3^2), phi (JT phase angle),
# and the angular shear fraction eta are computed with VanVleckCalculator
# (https://github.com/lnaglecocco/VanVleckCalculator).
#
# Axes convention (all octahedra in one cell share a common x,y,z frame):
#   The package default rotates each octahedron independently, which makes
#   Q2/Q3/phi incomparable between sites. Here the three opposite-ligand pair
#   directions of every TM octahedron (in the crystal Cartesian frame) are
#   averaged and orthonormalized into ONE axes set per structure, which is then
#   passed to every octahedron via specified_axes with automatic_rotation=False.
#   The frame is therefore fixed to the crystal axes and identical for all
#   octahedra of a given structure.
#   Which averaged direction is called x, y, z is fixed by the bond geometry so
#   that phi is also comparable BETWEEN structures (details in common_axes):
#     z = long axis of the most elongated octahedron in the cell (the
#         "reference", flagged by frame_ref=1) -> it sits at phi ~ 0 (+Q3);
#         if no octahedron is elongation-type, the short axis of the most
#         compressed one -> phi ~ 180;
#     y = the strongest second elongation direction, if any (-> phi ~ 120);
#         otherwise a second compression direction (-> phi ~ 300); otherwise
#         the reference's second-longest pair;
#     x = the remaining direction (elongation along x = phi ~ 240).
#   Package sign convention, checked numerically on the data: with d = pair
#   length minus its mean, Q3 = (2dz - dx - dy)/(2 sqrt3) (NaMnO2: z-pair
#   4.76 A vs 3.93 A gives Q3 = +0.484) and Q2 = (dy - dx)/2, the OPPOSITE
#   sign of the textbook (dx - dy)/2. Hence elongation along z/y/x is phi =
#   0/120/240 and compression along z/y/x is phi = 180/300/60.
#
# Sites whose O coordination is not 6 (e.g. desodiated CrO2 relaxes to an
# 8-coordinate Cr environment) get NaN modes; see the n_ligands column.
#
# Output: ../van_vleck/van_vleck_modes.csv, one row per TM octahedron.
# No plotting here.

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"                    # .../data
# VanVleckCalculator (Nagle-Cocco & Dutton 2024, https://github.com/lnaglecocco/VanVleckCalculator)
# is not on PyPI: clone it next to this repository, or point VVC_CODE at its "code" folder.
VVC_CODE = Path(os.environ.get("VVC_CODE", SCRIPT_DIR.parents[2] / "VanVleckCalculator" / "code"))
sys.path.insert(1, str(VVC_CODE))
from van_vleck_calculator import Octahedron

OUT_DIR = SCRIPT_DIR.parent / "van_vleck"
OUT_CSV = OUT_DIR / "van_vleck_modes.csv"

LIGAND_MAX_DIST = 2.7   # A: TM-O cutoff (JT-elongated Mn3+/Fe4+ axial ~2.3 A fit)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def tm_octahedra(s: Structure):
    """[(site_index, element, n_O_neighbours, Octahedron|None), ...] per TM site.

    The O coordination is counted from the structure itself: Octahedron would
    silently keep the 6 nearest ligands even for a non-octahedral site (e.g.
    8-coordinate Cr in desodiated CrO2), so those sites get None instead.
    """
    octs = []
    for i, site in enumerate(s):
        el = site.specie.symbol
        if el in ("Na", "O"):
            continue
        n_o = sum(1 for nb in s.get_neighbors(site, LIGAND_MAX_DIST)
                  if nb.specie.symbol == "O")
        o = (Octahedron(site, s, possible_ligands=["O"],
                        ligands_max_distance=LIGAND_MAX_DIST) if n_o == 6 else None)
        octs.append((i, el, n_o, o))
    return octs


def common_axes(octs):
    """Common orthonormal axes (3x3, rows = x,y,z) and the reference octahedron.

    Returns (axes, k_ref) where k_ref indexes `octs` at the octahedron whose
    distinguished axis defines z.

    Directions: each octahedron's three opposite-ligand pair directions
    (crystal Cartesian frame) are matched to the first octahedron's by largest
    |dot|, sign-aligned, averaged, and projected onto the nearest rotation
    matrix by SVD.

    Labelling (which averaged direction is x, y, z) from the O-O distances
    across each octahedron ("pair lengths" d1 >= d2 >= d3):
      e_plus  = d1 - (d2 + d3)/2   axial elongation  (one long pair)
      e_minus = (d1 + d2)/2 - d3   axial compression (one short pair)
      An octahedron is elongation-type if e_plus >= e_minus (distinguished
      axis = longest pair, strength e_plus), else compression-type
      (distinguished axis = shortest pair, strength e_minus).
      z = distinguished axis of the strongest elongation-type octahedron
          (the reference) -> phi ~ 0 (+Q3). If no octahedron is
          elongation-type, the strongest compression-type one -> phi ~ 180.
      y = the "secondary" axis: distinguished axis of the strongest remaining
          elongation-type octahedron whose axis is not z (second elongation
          direction -> phi ~ 120); failing that, of the strongest
          compression-type one with axis not z (-> phi ~ 300); failing that,
          the reference's second-longest pair (reference lands in [0, 30]).
      x = the remaining direction; its sign is chosen to make the frame
          right-handed (axis signs do not affect Q2/Q3).
    Package convention: Q2 = (dy - dx)/2, so elongation along y is phi = 120
    and along x phi = 240 (see module header).
    """
    ref, acc = None, np.zeros((3, 3))
    per_oct = []    # (strength, elongation-type?, axis row, other rows, k)
    for k, (_, _, _, o) in enumerate(octs):
        if o is None:
            continue
        try:
            pairs = o.find_ligand_opposites()
        except Exception as e:
            print(f"    pair finding failed on one octahedron: {e}")
            continue
        vecs, lens = [], []
        for p in pairs:
            v = np.asarray(p[0].coords) - np.asarray(p[1].coords)
            d = float(np.linalg.norm(v))
            vecs.append(v / d)
            lens.append(d)
        if ref is None:
            ref = np.array(vecs)
        rows = []
        for v in vecs:
            dots = ref @ v
            j = int(np.argmax(np.abs(dots)))
            acc[j] += np.sign(dots[j]) * v
            rows.append(j)
        if len(set(rows)) == 3:       # cleanly matched to three distinct axes
            i1, i2, i3 = np.argsort(lens)[::-1]             # longest first
            d1, d2, d3 = lens[i1], lens[i2], lens[i3]
            e_plus, e_minus = d1 - 0.5 * (d2 + d3), 0.5 * (d1 + d2) - d3
            if e_plus >= e_minus:
                per_oct.append((e_plus, True, rows[i1], [rows[i2], rows[i3]], k))
            else:
                per_oct.append((e_minus, False, rows[i3], [rows[i1], rows[i2]], k))
    if ref is None:
        raise RuntimeError("no octahedron yielded opposite-ligand pairs")
    U, _, Vt = np.linalg.svd(acc)
    R = U @ Vt

    # label the rows (see docstring): z = primary axis, y = secondary axis
    strength = lambda t: t[0]
    elong = [t for t in per_oct if t[1]]
    comp = [t for t in per_oct if not t[1]]
    reference = max(elong, key=strength) if elong else max(comp, key=strength)
    z_row = reference[2]
    sec_e = [t for t in elong if t[2] != z_row]
    sec_c = [t for t in comp if t[2] != z_row]
    if sec_e:
        y_row = max(sec_e, key=strength)[2]
    elif sec_c:
        y_row = max(sec_c, key=strength)[2]
    else:
        y_row = reference[3][0]
    x_row = ({0, 1, 2} - {y_row, z_row}).pop()
    R = R[[x_row, y_row, z_row]]
    if np.linalg.det(R) < 0:          # keep a right-handed frame
        R[0] *= -1.0
    return [[float(x) for x in row] for row in R], reference[4]


def phi_deg(q2, q3):
    """JT phase angle, package convention: phi = atan2(Q2, Q3) mod 2pi."""
    return float(np.degrees(np.mod(np.arctan2(q2, q3), 2.0 * np.pi)))


def max_off_axis(pairs):
    """Max distance (A) of any ligand from its assigned axis after rotation.

    `pairs` has shape (3,2,3): 3 axis-assigned pairs x 2 sites x xyz. QC metric:
    large values mean the common frame fits this octahedron poorly.
    """
    dev = 0.0
    for ax, pair in enumerate(pairs):
        for c in pair:
            c = np.asarray(c, float)
            dev = max(dev, float(np.sqrt(sum(c[j] ** 2 for j in range(3) if j != ax))))
    return dev


def hull_structures():
    """Yield one dict per stable hull point (single + double TM)."""
    entries = []
    hs = pd.read_csv(DATA_DIR / "singleTM_hull_points.csv")
    ss = json.load(open(DATA_DIR / "singleTM_structures.json"))
    for _, r in hs.iterrows():
        key = f"{r['phase']}-{r['ordering']}-{r['tm']}"
        entries.append(dict(dataset="singleTM", system=r["tm"], phase=r["phase"],
                            ordering=r["ordering"], elec="", xNa=r["xNa"],
                            hull_order=r["hull_order"], formula=r["formula"],
                            key=key, struct=Structure.from_dict(ss[key])))
    hd = pd.read_csv(DATA_DIR / "doubleTM_hull_points.csv")
    sd = json.load(open(DATA_DIR / "doubleTM_structures.json"))
    for _, r in hd.iterrows():
        key = f"{r['phase']}-{r['ordering']}-{r['tm_pair']}-{r['elec']}"
        entries.append(dict(dataset="doubleTM", system=r["tm_pair"], phase=r["phase"],
                            ordering=r["ordering"], elec=r["elec"], xNa=r["xNa"],
                            hull_order=r["hull_order"], formula=r["formula"],
                            key=key, struct=Structure.from_dict(sd[key])))
    return entries


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    entries = hull_structures()
    print(f"{len(entries)} hull structures "
          f"({sum(e['dataset'] == 'singleTM' for e in entries)} single, "
          f"{sum(e['dataset'] == 'doubleTM' for e in entries)} double)")

    rows = []
    for e in entries:
        s = e["struct"]
        try:
            octs = tm_octahedra(s)
            axes, k_ref = (common_axes(octs)
                           if any(o is not None for *_, o in octs) else (None, None))
        except Exception as exc:
            print(f"  SKIP {e['key']}: {exc}")
            continue
        n_ok = 0
        for k, (i, el, n_o, o) in enumerate(octs):
            row = {c: e[c] for c in ("dataset", "system", "phase", "ordering",
                                     "elec", "xNa", "hull_order", "formula", "key")}
            row.update(site_index=i, element=el, n_ligands=n_o,
                       frame_ref=int(k == k_ref))
            nan_cols = dict(Q1=np.nan, Q2=np.nan, Q3=np.nan, Q4=np.nan, Q5=np.nan,
                            Q6=np.nan, rho=np.nan, phi_deg=np.nan, eta=np.nan,
                            max_off_axis_A=np.nan)
            if o is None:
                print(f"  NOT OCTAHEDRAL {e['key']} site {i} ({el}): "
                      f"{n_o} O ligands")
                row.update(nan_cols)
                rows.append(row)
                continue
            try:
                modes, pairs = o.calculate_van_vleck_distortion_modes(
                    specified_axes=axes, automatic_rotation=False,
                    output_pairs=True, suppress_warnings=True)
                eta = o.calculate_shear_fraction_angular_distortion(
                    specified_axes=axes, automatic_rotation=False,
                    suppress_warnings=True)
                q1, q2, q3, q4, q5, q6 = (float(m) for m in modes)
                row.update(Q1=q1, Q2=q2, Q3=q3, Q4=q4, Q5=q5, Q6=q6,
                           rho=float(np.hypot(q2, q3)), phi_deg=phi_deg(q2, q3),
                           eta=float(eta), max_off_axis_A=max_off_axis(pairs))
                n_ok += 1
            except Exception as exc:
                print(f"  FAIL {e['key']} site {i} ({el}): {exc}")
                row.update(nan_cols)
            rows.append(row)
        print(f"  {e['key']}: {n_ok}/{len(octs)} octahedra")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    n_bad = int(df["Q1"].isna().sum())
    print(f"\nwrote {OUT_CSV}  ({len(df)} octahedra, {n_bad} failed)")
    if len(df):
        print(df.groupby("dataset")[["rho", "eta", "max_off_axis_A"]]
                .agg(["mean", "max"]).round(4))


if __name__ == "__main__":
    main()
