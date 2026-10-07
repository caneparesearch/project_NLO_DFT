# Charge and Van Vleck modes (Q2, Q3) of every TMO6 octahedron.
#
# For every stable (hull) structure of the 5 single-TM and the 10 mixed-TM systems:
#   STEP 1  read the relaxed structure and the magnetic moment of every TM atom
#   STEP 2  assign a charge to every TM atom from its magnetic moment
#   STEP 3  calculate the Van Vleck modes Q2 and Q3 of the octahedron around every TM atom
#   STEP 4  write one row per TM atom to a csv file
#
# Where the methods come from
#   Charge assignment : notebook/dft_oxidation_analysis.ipynb, the code cell that writes
#                       data/doubleTM_stable_oxi_states.csv (functions oxidation_candidates and
#                       solve_atomwise_pair). The moment windows below are copied from it.
#   Van Vleck modes   : Figure4_vanvleck.py (tm_octahedra, common_axes), which uses the
#                       VanVleckCalculator package. Same axes convention, so the numbers are
#                       the same as in van_vleck/van_vleck_modes.csv.
#
# Input   data/singleTM_hull_points.csv   data/doubleTM_hull_points.csv    (which structures are stable)
#         data/singleTM_structures.json   data/doubleTM_structures.json    (relaxed structures)
#         data/singleTM_magmom.csv        data/doubleTM_magmom.csv         (magnetic moments)
# Output  data/singleTM_vanvleck_oxi_states.csv
#         data/doubleTM_vanvleck_oxi_states.csv
#
# Run     conda run -n research python Figure4_vanvleck_oxi.py      (about 10 s)

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from Figure4_vanvleck import tm_octahedra, common_axes, phi_deg

DATA_DIR = Path(__file__).resolve().parents[2] / "data"              # .../data
TM_LIST = ["Ti", "Mn", "Fe", "Co", "Ni"]
INF = float("inf")


# =============================================================================================
# STEP 2 (rules)  magnetic moment -> possible charges
#   A window (lower, upper, charges) applies when lower < |moment| < upper (moment in Bohr magneton).
#   Two charges in one window means that the moment alone cannot tell them apart,
#   and charge neutrality of the cell decides.
# =============================================================================================
# Copied from the notebook. Used for all systems.
WINDOWS_NOTEBOOK = {
    "Ti": [(-INF, 0.5, [4])],
    "Mn": [(1.8, 2.2, [3]), (2.9, 3.3, [4]), (3.5, 4.0, [3]), (4.2, INF, [2])],
    "Fe": [(-INF, 0.2, [2]), (0.9, 1.3, [3]), (1.9, 2.4, [4]), (3.5, 4.1, [2, 4]), (4.1, INF, [3])],
    "Co": [(-INF, 0.5, [3]), (0.5, 1.5, [2, 4]), (2.3, 3.5, [2])],
    "Ni": [(-INF, 0.7, [4]), (0.7, 1.5, [3]), (1.5, 1.9, [2]), (1.9, 2.3, [3])],
}
# NOT in the notebook. The notebook treats the mixed-TM systems only, and two moments of the
# single-TM systems fall outside its windows. These windows are used for the single-TM systems only,
# so the mixed-TM charges stay the same as in data/doubleTM_stable_oxi_states.csv (Figure 3).
WINDOWS_ADDED_SINGLE_TM = {
    "Ti": [(0.5, INF, [3])],      # NaTiO2: 0.65
    "Fe": [(2.4, 3.5, [5])],      # Na1/3FeO2 and Na3/8FeO2: 3.40 to 3.46
}


def possible_charges(element, moment, single_tm):
    """Charges that the magnetic moment of one atom allows, e.g. [3] or [2, 4] or [] (no window)."""
    windows = list(WINDOWS_NOTEBOOK[element])
    if single_tm:
        windows += WINDOWS_ADDED_SINGLE_TM.get(element, [])
    for lower, upper, charges in windows:
        if lower < abs(moment) < upper:
            return charges
    return []


def assign_charges(elements, moments, n_Na, n_O, single_tm):
    """Charge of every TM atom of one cell.

    elements, moments : one entry per TM atom
    Returns (charges, possible, note). charges is None when the cell cannot be assigned,
    and note tells why. Such a structure is "n/a" in Figure 3.
    """
    possible = [possible_charges(el, m, single_tm) for el, m in zip(elements, moments)]
    if any(len(p) == 0 for p in possible):
        return None, possible, "n/a: a moment is outside the windows"

    # charge neutrality of the cell: Na(+1) * n_Na + (sum of the TM charges) + O(-2) * n_O = 0
    required_tm_charge = 2 * n_O - n_Na

    # Atoms with one possible charge are fixed. The undecided atoms of one element (for example all
    # Co atoms that can be 2+ or 4+) take the same charge, and every combination is tried.
    # (Every element has at most one window with two charges.)
    options = {}                                        # for example {"Co": [2, 4], "Fe": [2, 4]}
    for el, p in zip(elements, possible):
        if len(p) > 1:
            options[el] = p
    neutral = []
    for choice in itertools.product(*options.values()):
        chosen = dict(zip(options, choice))             # for example {"Co": 4, "Fe": 2}
        trial = [p[0] if len(p) == 1 else chosen[el] for el, p in zip(elements, possible)]
        if sum(trial) == required_tm_charge:
            neutral.append(trial)

    if len(neutral) == 1:
        return neutral[0], possible, "assigned"
    if len(neutral) == 0:
        return None, possible, "n/a: not charge neutral"
    return None, possible, "n/a: more than one neutral assignment"


# =============================================================================================
# STEP 3  Van Vleck modes
# =============================================================================================
def van_vleck_q2_q3(structure):
    """{site_index: (Q2, Q3)} in Angstrom for every TM octahedron of one structure.

    All octahedra of one structure share one x, y, z frame (common_axes of Figure4_vanvleck.py):
    z is the long axis of the most elongated octahedron of the structure.
    """
    octahedra = tm_octahedra(structure)             # [(site_index, element, n_O, Octahedron), ...]
    axes, _ = common_axes(octahedra)
    modes = {}
    for site_index, _, _, octahedron in octahedra:
        q = octahedron.calculate_van_vleck_distortion_modes(
            specified_axes=axes, automatic_rotation=False, suppress_warnings=True)
        modes[site_index] = (float(q[1]), float(q[2]))      # q = [Q1, Q2, Q3, Q4, Q5, Q6]
    return modes


# =============================================================================================
# STEP 1 to 4 for one set of systems (single TM or mixed TM)
# =============================================================================================
def build_table(kind):
    """One row per TM atom of every hull structure. kind = "singleTM" or "doubleTM"."""
    single_tm = (kind == "singleTM")
    system_column = "tm" if single_tm else "tm_pair"
    structure_file = f"{kind}_structures.json"

    hull = pd.read_csv(DATA_DIR / f"{kind}_hull_points.csv")
    magmom = pd.read_csv(DATA_DIR / f"{kind}_magmom.csv")
    structures = json.load(open(DATA_DIR / structure_file))
    if single_tm:
        hull = hull[hull.tm.isin(TM_LIST)]
        hull = hull.assign(elec="")
        magmom = magmom.assign(elec="")
    else:
        magmom = magmom.rename(columns={"tm_ordering": "elec"})

    rows = []
    for _, point in hull.iterrows():
        system = point[system_column]

        # ---- STEP 1: structure and magnetic moments ------------------------------------------
        key = "-".join(x for x in (point.phase, point.ordering, system, point.elec) if x)
        structure = Structure.from_dict(structures[key])
        same_run = ((magmom.phase == point.phase) & (magmom.ordering == point.ordering)
                    & (magmom[system_column] == system) & (magmom.elec == point.elec))
        # {"Mn": [...], "Ni": [...]}: moments of each element, in the order of the atoms in the structure
        moments_of = ast.literal_eval(magmom[same_run].tm_magmom_dict.iloc[0])

        tm_sites, elements, moments = [], [], []
        n_seen = {el: 0 for el in moments_of}
        for site_index, site in enumerate(structure):
            el = site.specie.symbol
            if el in moments_of:
                tm_sites.append(site_index)
                elements.append(el)
                moments.append(moments_of[el][n_seen[el]])
                n_seen[el] += 1
        assert n_seen == {el: len(m) for el, m in moments_of.items()}, f"atom count differs: {key}"

        # ---- STEP 2: charge of every TM atom ----------------------------------------------------
        n_Na = structure.composition["Na"]
        n_O = structure.composition["O"]
        charges, possible, note = assign_charges(elements, moments, n_Na, n_O, single_tm)

        # ---- STEP 3: Van Vleck modes of every TM octahedron -------------------------------------
        q2_q3 = van_vleck_q2_q3(structure)

        # ---- STEP 4: one row per TM atom --------------------------------------------------------
        for k, site_index in enumerate(tm_sites):
            q2, q3 = q2_q3[site_index]
            charge = charges[k] if charges else None
            rows.append({
                "system": system,
                "xNa": point.xNa,
                "phase": point.phase,
                "ordering": point.ordering,
                "elec": point.elec,
                "formula": point.formula,
                "structure_file": structure_file,
                "structure_key": key,
                "atom_number": site_index + 1,            # n-th atom of the structure (POSCAR, VESTA)
                "site_index": site_index,                 # the same atom counted from 0 (pymatgen)
                "element": elements[k],
                "magmom_muB": moments[k],
                "possible_charges": "|".join(str(c) for c in possible[k]),
                "charge": charge,
                "ion": f"{elements[k]}{charge}+" if charge else "",      # empty = not assigned
                "note": note,
                "Q2_A": q2,
                "Q3_A": q3,
                "rho_A": float(np.hypot(q2, q3)),         # size of the distortion
                "phi_deg": phi_deg(q2, q3),               # 0 = elongation along z
            })
        print(f"  {key:50s} x = {point.xNa:.3f}   {len(tm_sites):3d} octahedra   {note}")

    table = pd.DataFrame(rows)
    table["charge"] = table["charge"].astype("Int64")       # integer column that may have empty cells
    table = table.round({"Q2_A": 4, "Q3_A": 4, "rho_A": 4, "phi_deg": 2})
    table["phi_deg"] = table["phi_deg"] % 360               # 359.999 is rounded to 360.00, which is 0
    if single_tm:
        table = table.drop(columns="elec")                  # only the mixed-TM runs have elec_0 / elec_1
    return table


def check_against_notebook(table):
    """The mixed-TM charges must be the ones of the notebook (data/doubleTM_stable_oxi_states.csv)."""
    notebook = pd.read_csv(DATA_DIR / "doubleTM_stable_oxi_states.csv")
    n_same = 0
    for _, ref in notebook.iterrows():
        key = f"{ref.phase}-{ref.ordering}-{ref.tm_pair}-{ref.elec}"
        mine = table[table.structure_key == key]
        for tm, ref_list in ((ref.tm1, ref.ox_tm1_list), (ref.tm2, ref.ox_tm2_list)):
            ref_charges = ast.literal_eval(ref_list) if ref.charge_neutral else []
            my_charges = mine[mine.element == tm].charge.dropna().tolist()
            assert my_charges == ref_charges, f"charges differ from the notebook: {key} {tm}"
        n_same += 1
    print(f"check: {n_same} of {len(notebook)} mixed-TM structures have the charges of the notebook")


def main():
    for kind in ("singleTM", "doubleTM"):
        print(f"{kind}")
        table = build_table(kind)
        if kind == "doubleTM":
            check_against_notebook(table)
        out_csv = DATA_DIR / f"{kind}_vanvleck_oxi_states.csv"
        table.to_csv(out_csv, index=False)
        n_structures = table.structure_key.nunique()
        n_assigned = table[table.note == "assigned"].structure_key.nunique()
        print(f"wrote {out_csv}: {len(table)} octahedra of {n_structures} structures, "
              f"{n_assigned} structures assigned, {n_structures - n_assigned} n/a\n")


if __name__ == "__main__":
    main()
