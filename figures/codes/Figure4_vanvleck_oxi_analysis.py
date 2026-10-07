# Numbers quoted in the section "Local Jahn-Teller distortions of the TMO6 octahedra" (Figure 5).
#
# For every TMO6 octahedron of the stable (hull) structures this script adds, to the table of
# charges and Van Vleck modes, what the text needs:
#   STEP 1  the six TM-O bond lengths and the shape of the octahedron
#           (elongated = one long axis, compressed = one short axis, regular)
#   STEP 2  the neighbors: how many of the six TM neighbors are elongated along the same direction
#   STEP 3  per structure: number of elongation directions, the ratio |sum of (Q2, Q3)| / sum of rho,
#           and how many long bonds each oxygen takes (rule of Radin et al. 2018)
#   STEP 4  prints the numbers in the order in which they appear in the text
#
# Input   data/singleTM_vanvleck_oxi_states.csv, data/doubleTM_vanvleck_oxi_states.csv
#         (written by Figure4_vanvleck_oxi.py)
#         data/singleTM_structures.json, data/doubleTM_structures.json
# Output  ../van_vleck/vanvleck_oxi_analysis_octahedra.csv     one row per octahedron
#         ../van_vleck/vanvleck_oxi_analysis_structures.csv    one row per structure
#
# Run     conda run -n research python Figure4_vanvleck_oxi_analysis.py      (about 1 min)

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"                            # .../data
OUT_DIR = SCRIPT_DIR.parent / "van_vleck"

RHO_DISTORTED = 0.10        # A, an octahedron with rho above this value is called distorted
TM_O_MAX = 2.7              # A, TM-O bond
TM_TM_MAX = 3.6             # A, in-plane TM-TM neighbor
REGULAR = 0.04              # A, below this difference of the axis lengths the octahedron is regular


# =============================================================================================
# STEP 1  bonds and shape of one octahedron
# =============================================================================================
def describe_octahedron(structure, site_index, neighbors):
    """Bond lengths, shape and elongation (or compression) axis of the octahedron around one TM."""
    center = structure[site_index]
    oxygens = sorted([n for n in neighbors[site_index]
                      if n.specie.symbol == "O" and n.nn_distance < TM_O_MAX],
                     key=lambda n: n.nn_distance)
    assert len(oxygens) == 6
    bond = np.array([n.nn_distance for n in oxygens])
    vector = [n.coords - center.coords for n in oxygens]

    # the three pairs of opposite oxygens: the pairing with the largest sum of O-TM-O angles
    best_angle, pairs = -1.0, None
    rest0 = [1, 2, 3, 4, 5]
    for a in rest0:
        rest1 = [k for k in rest0 if k != a]
        for b in rest1[1:]:
            last = [k for k in rest1[1:] if k != b]
            trial = [(0, a), (rest1[0], b), (last[0], last[1])]
            angle = sum(np.degrees(np.arccos(np.clip(
                np.dot(vector[p], vector[q]) / (bond[p] * bond[q]), -1, 1))) for p, q in trial)
            if angle > best_angle:
                best_angle, pairs = angle, trial

    axis_length = np.array([bond[p] + bond[q] for p, q in pairs])       # O-TM-O length of each axis
    longest, middle, shortest = np.argsort(axis_length)[::-1]
    elongation = axis_length[longest] - (axis_length[middle] + axis_length[shortest]) / 2
    compression = (axis_length[longest] + axis_length[middle]) / 2 - axis_length[shortest]
    if max(elongation, compression) < REGULAR:
        shape, special = "regular", pairs[longest]
    elif elongation >= compression:
        shape, special = "elongated", pairs[longest]
    else:
        shape, special = "compressed", pairs[shortest]
    axis = vector[special[0]] - vector[special[1]]
    axis = axis / np.linalg.norm(axis)

    two_longest = set(np.argsort(bond)[-2:])
    tm_neighbors = sorted([n for n in neighbors[site_index]
                           if n.specie.symbol not in ("Na", "O") and n.nn_distance < TM_TM_MAX],
                          key=lambda n: n.nn_distance)[:6]
    return {
        "bonds_A": " ".join(f"{d:.3f}" for d in bond),
        "short4_A": float(np.sort(bond)[:4].mean()),
        "long2_A": float(np.sort(bond)[-2:].mean()),
        "shape": shape,
        "long_bonds_opposite": any(set(pair) == two_longest for pair in pairs),
        "axis_x": axis[0], "axis_y": axis[1], "axis_z": axis[2],
        "long_oxygens": " ".join(str(oxygens[k].index) for k in np.argsort(bond)[-2:]),
        "tm_neighbors": " ".join(str(n.index) for n in tm_neighbors),
    }


def build_tables():
    table = pd.concat([pd.read_csv(DATA_DIR / "singleTM_vanvleck_oxi_states.csv").assign(kind="single"),
                       pd.read_csv(DATA_DIR / "doubleTM_vanvleck_oxi_states.csv").assign(kind="mixed")],
                      ignore_index=True)
    structures = {name: json.load(open(DATA_DIR / name))
                  for name in ("singleTM_structures.json", "doubleTM_structures.json")}

    octahedra, per_structure = [], []
    for (file_name, key), rows in table.groupby(["structure_file", "structure_key"], sort=False):
        structure = Structure.from_dict(structures[file_name][key])
        neighbors = structure.get_all_neighbors(3.7)
        rows = rows.copy()
        described = [describe_octahedron(structure, int(i), neighbors) for i in rows.site_index]
        for column in described[0]:
            rows[column] = [d[column] for d in described]
        rows["distorted"] = rows.rho_A >= RHO_DISTORTED
        rows["jt_elongated"] = rows.distorted & (rows["shape"] == "elongated")

        # ---- STEP 2: neighbors that are elongated along the same direction -------------------
        by_site = rows.set_index("site_index")
        n_parallel = []
        for _, row in rows.iterrows():
            mine = np.array([row.axis_x, row.axis_y, row.axis_z])
            count = 0
            for k in [int(v) for v in row.tm_neighbors.split()]:
                other = by_site.loc[k]
                if other.jt_elongated:
                    theirs = np.array([other.axis_x, other.axis_y, other.axis_z])
                    count += abs(np.dot(mine, theirs)) > 0.9
            n_parallel.append(int(count))
        rows["n_parallel_neighbors"] = n_parallel
        octahedra.append(rows)

        # ---- STEP 3: the structure as a whole -------------------------------------------------
        elongated = rows[rows.jt_elongated]
        directions = set((np.round(elongated.phi_deg / 60).astype(int) % 6) * 60)
        long_bonds = Counter(int(o) for cell in elongated.long_oxygens for o in cell.split())
        oxygens = [i for i, s in enumerate(structure) if s.specie.symbol == "O"]
        taken = Counter(long_bonds.get(i, 0) for i in oxygens)
        per_structure.append({
            "kind": rows.kind.iloc[0], "system": rows.system.iloc[0], "xNa": rows.xNa.iloc[0],
            "structure_key": key, "n_octahedra": len(rows), "n_distorted": int(rows.distorted.sum()),
            "n_elongated": len(elongated), "n_elongation_directions": len(directions),
            # 1 = all octahedra are distorted in the same way, 0 = the distortions cancel
            "ratio_sum": float(np.hypot(rows.Q2_A.sum(), rows.Q3_A.sum()) / rows.rho_A.sum())
                         if rows.rho_A.sum() > 1e-3 else np.nan,
            "oxygens_with_0_long_bonds": taken.get(0, 0), "oxygens_with_1_long_bond": taken.get(1, 0),
            "oxygens_with_2_long_bonds": taken.get(2, 0) + taken.get(3, 0),
        })
        print(f"  {key}")
    return pd.concat(octahedra, ignore_index=True), pd.DataFrame(per_structure)


# =============================================================================================
# STEP 4  the numbers of the text
# =============================================================================================
def print_numbers(o, s):
    ion = lambda name: o[o.ion == name]
    big = o[o.distorted]
    deviation = lambda rows: (rows.phi_deg - 60 * np.round(rows.phi_deg / 60)).abs().mean()
    has_distorted = s[s.n_distorted > 0]

    print("\n1. Which ions distort")
    for name in ("Mn3+", "Ni3+", "Mn4+", "Ni4+", "Co3+", "Co4+"):
        print(f"   {name}: rho from {ion(name).rho_A.min():.2f} to {ion(name).rho_A.max():.2f} A, "
              f"mean {ion(name).rho_A.mean():.2f}")
    print(f"   octahedra with rho >= {RHO_DISTORTED} A: {len(big)}, elongated {int(big.jt_elongated.sum())}, "
          f"compressed {int((big['shape'] == 'compressed').sum())}")
    print(f"   elongated ones with the two long bonds opposite to each other: "
          f"{int(big[big.jt_elongated].long_bonds_opposite.sum())}")
    print(f"   deviation of phi from a pure tetragonal distortion: Mn3+ {deviation(big[big.ion == 'Mn3+']):.1f} deg, "
          f"Ni3+ {deviation(big[big.ion == 'Ni3+']):.1f} deg")
    for key, name in (("O3_orderings-JMAS_e-Mn", "Mn3+"), ("O3_orderings-JMAS_e-Ni", "Ni3+"),
                      ("O3_orderings-JMAS_e-Ti_Co-elec_1", "Co2+")):
        row = o[(o.structure_key == key) & (o.ion == name)].iloc[0]
        print(f"   {key}: {name} bonds {row.short4_A:.2f} A (4x) and {row.long2_A:.2f} A (2x), rho {row.rho_A:.2f}")

    print("\n2. Fe4+ and Ti4+")
    fe = ion("Fe4+")
    for kind in ("single", "mixed"):
        g = fe[fe.kind == kind]; d = g[g.distorted]
        print(f"   Fe4+ {kind}: mean rho {g.rho_A.mean():.2f} A; with rho >= 0.1: "
              f"{int(d.jt_elongated.sum())} elongated, {int((d['shape'] == 'compressed').sum())} compressed")
    print(f"   Fe4+ in NaxFeO2 at x = 0 and 0.5: rho "
          f"{fe[(fe.system == 'Fe') & fe.xNa.isin([0.0, 0.5])].rho_A.round(2).unique()}")
    ti = o[(o.system == "Ti_Mn") & (o.ion == "Ti4+")].sort_values("rho_A").iloc[-1]
    print(f"   Ti4+ in NaxTi0.5Mn0.5O2: largest rho {ti.rho_A:.2f} A, {ti['shape']}, phi {ti.phi_deg:.0f} deg")

    print("\n3. Charge disproportionation")
    for key in ("P3_orderings-JMAS_d-Ni", "P3_orderings-JMAS_d-Ti_Ni-elec_1", "O3_orderings-AJT_d_w_vac-Fe"):
        g = o[o.structure_key == key]
        print(f"   {key}: {g.ion.value_counts().to_dict()}, largest rho {g.rho_A.max():.3f} A")
    fe = o[o.structure_key == "O3_orderings-AJT_d_w_vac-Fe"]
    for name, g in fe.groupby("ion"):
        mean_bond = np.mean([float(v) for bonds in g.bonds_A for v in bonds.split()])
        print(f"   NaxFeO2 at x = 1/3, {name}: {len(g)} octahedra, moment {g.magmom_muB.abs().mean():.2f} muB, "
              f"mean Fe-O bond {mean_bond:.2f} A, rho {g.rho_A.max():.2f} A")

    print("\n4. Direction of the elongation")
    single = has_distorted[has_distorted.system.isin(["Mn", "Ni"])]
    print(f"   NaxMnO2 and NaxNiO2: number of elongation directions per structure "
          f"{sorted(single.n_elongation_directions.unique())}")
    with_elongated = s[s.n_elongated > 0]
    print(f"   structures with elongated octahedra: {len(with_elongated)}, of which no oxygen takes two long "
          f"bonds: {int((with_elongated.oxygens_with_2_long_bonds == 0).sum())}")
    mn = o[o.system == "Mn"]
    print(f"   NaxMnO2: Mn3+ rho >= {mn[mn.ion == 'Mn3+'].rho_A.min():.2f} A, "
          f"Mn4+ rho <= {mn[mn.ion == 'Mn4+'].rho_A.max():.2f} A")

    print("\n5. Magnitude against the number of neighbors elongated along the same direction")
    print("   (elongated octahedra only: a compressed octahedron has no elongation direction)")
    for name, system in (("Mn3+", "Mn"), ("Ni3+", "Ni")):
        elongated = ion(name)[ion(name).jt_elongated]
        g = elongated.groupby("n_parallel_neighbors").rho_A.mean()
        x13 = o[(o.system == system) & (o.ion == name) & ((o.xNa - 1 / 3).abs() < 1e-6)].rho_A.mean()
        x1 = o[(o.system == system) & (o.ion == name) & (o.xNa == 1.0)].rho_A.mean()
        print(f"   {name}: rho {g[0]:.2f} A with no such neighbor, {g[6]:.2f} A with six, "
              f"cooperative part {100 * (g[6] - g[0]) / g[6]:.0f} %; "
              f"in the single-TM oxide {x13:.2f} A at x = 1/3 and {x1:.2f} A at x = 1")
        for n in (0, 6):                 # the structures behind the two values
            for (sys_name, x, phase, ordering), h in elongated[elongated.n_parallel_neighbors == n].groupby(
                    ["system", "xNa", "phase", "ordering"]):
                print(f"      {n} neighbors: {sys_name:6s} x = {x:.3f}  {phase[:2]}  {ordering.replace('_w_vac', ''):12s} "
                      f"{len(h):2d} octahedra, rho {h.rho_A.mean():.2f} A")

    print("\n6. Mixing")
    mn3 = ion("Mn3+")
    print(f"   Mn3+ mean rho: NaxMnO2 {mn3[mn3.kind == 'single'].rho_A.mean():.2f} A; per structure of Mn-Fe and "
          f"Mn-Co {mn3[mn3.system.isin(['Mn_Fe', 'Mn_Co'])].groupby('structure_key').rho_A.mean().round(2).tolist()}")
    mixed = has_distorted[has_distorted.kind == "mixed"]
    print(f"   ratio |sum (Q2, Q3)| / sum rho: NaxMnO2 and NaxNiO2 {single.ratio_sum.min():.2f} to "
          f"{single.ratio_sum.max():.2f}; mixtures mean {mixed.ratio_sum.mean():.2f}, smallest "
          f"{mixed.ratio_sum.min():.2f} ({mixed.sort_values('ratio_sum').iloc[0].structure_key})")
    print(f"   structures with two or three elongation directions: "
          f"{with_elongated[with_elongated.n_elongation_directions > 1].groupby('system').size().to_dict()}")
    # how many of the distorted octahedra are pure elongations, i.e. within 10 degrees of a dotted line of Figure 5
    to_line = np.minimum.reduce([np.abs((big.phi_deg - line + 180) % 360 - 180) for line in (0, 120, 240)])
    on_line = big[to_line <= 10]
    for system in ("Mn", "Ni", "Fe_Co", "Fe_Ni", "Co_Ni"):
        middle = (big.system == system) if system in ("Mn", "Ni") else (big.system == system) & (big.xNa > 0) & (big.xNa < 1)
        n_all, n_on = int(middle.sum()), int(middle[on_line.index].sum())
        print(f"   {system:6s} distorted octahedra{'' if system in ('Mn', 'Ni') else ' at 0 < x < 1'}: {n_all}, within 10 deg of a "
              f"dotted line: {n_on} ({100 * n_on / n_all:.0f} %), mean rho {big[middle].rho_A.mean():.2f} A")
    for name in ("Mn3+", "Ni3+", "Fe4+"):
        for kind in ("single", "mixed"):
            g = big[(big.ion == name) & (big.kind == kind)]
            print(f"   {name} {kind}: distorted {len(g)}, within 10 deg of a dotted line {int(g.index.isin(on_line.index).sum())}")

    print("\n7. Experiment")
    print(f"   NaMnO2, bonds 1.94 and 2.39 A (Ma et al. 2011): rho = {2 / np.sqrt(3) * (2.39 - 1.94):.2f} A")
    print(f"   NaNiO2, bond splitting 0.243 A (Nagle-Cocco et al. 2024): rho = {2 / np.sqrt(3) * 0.243:.2f} A")
    for key, name in (("O3_orderings-JMAS_e-Mn", "NaMnO2"), ("O3_orderings-JMAS_e-Ni", "NaNiO2")):
        row = o[o.structure_key == key].iloc[0]
        print(f"   {name}, this work: bonds {row.short4_A:.2f} A (4x) and {row.long2_A:.2f} A (2x), "
              f"difference {row.long2_A - row.short4_A:.2f} A, rho {row.rho_A:.2f} A")
    ni = o[(o.system == "Mn_Ni") & (o.ion == "Ni3+") & (o.xNa == 0.5)]
    print(f"   Na0.5Mn0.5Ni0.5O2, this work: Ni3+ bonds {ni.short4_A.mean():.2f} A (4x) and {ni.long2_A.mean():.2f} A (2x); "
          f"EXAFS 1.92 A (4x) and 2.04 A (2x) (Komaba et al. 2012)")


def main():
    octahedra, per_structure = build_tables()
    octahedra.to_csv(OUT_DIR / "vanvleck_oxi_analysis_octahedra.csv", index=False)
    per_structure.to_csv(OUT_DIR / "vanvleck_oxi_analysis_structures.csv", index=False)
    print(f"wrote {len(octahedra)} octahedra and {len(per_structure)} structures to {OUT_DIR}")
    print_numbers(octahedra, per_structure)


if __name__ == "__main__":
    main()
