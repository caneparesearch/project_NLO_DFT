#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a simple Co -> TM sweep for full O3 phase/empty O3 phase/empty O1 phase 
with MACE pre-relaxation and prepare VASP inputs for follow-up DFT runs.

This script mirrors the ordering_staging -> ordering_staging_output ->
dft_singleTM flow but without Na-vacancy ordering. For each selected phase and
transition metal, it:

1) loads the reference POSCAR (default: data/dft_priliminary/O1_NaCoO2_POSCAR
   and O3_NaCoO2_POSCAR),
2) replaces all Co atoms with the target TM,
3) relaxes the structure with MACE (by default include cell relaxation)
4) writes a summary.csv under the MACE output root.
5) writes MPRelaxSet-style VASP inputs under the DFT root. (include DFT-D3-BJ)

Example
-------
python full_na_phase_pipeline.py \
    --model ./data/mace-omat-0-medium.model \
    --mace-root ./ordering_staging_output \
    --dft-root ./dft_singleTM
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from ase.constraints import ExpCellFilter
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from mace.calculators.mace import MACECalculator
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet

TMS: List[str] = ["Co", "Cr", "Mn", "Ni", "Fe", "Ti", "V"]
# FMAX and STEPS are for MACE relaxation.
FMAX = 0.05
STEPS = 1000
WRITE_TRAJ = True

# Since there is no available staging.
GROUP_LABEL = "group_000"

# Where to save.
PHASE_DIR_MAP = {
    "O1": "O1_orderings",
    "O3": "O3_orderings",
    "P3": "P3_orderings",
}

# End points consideration.
NA_CASES = {
    "O1": [("empty", "empty")],                   # only empty O1
    "O3": [("full", "full"), ("empty", "empty")], # full + empty O3
    "P3": [("full", "full"), ("empty", "empty")], # full + empty P3
}

USER_INCAR = {
    "ENCUT": 520,
    "IVDW": 12,  # Grimme D3-BJ
    "ISPIN": 2,  # spin-polarized (FM init)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MACE pre-relax for endpoints of O1/O3/P3."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a MACE model file.",
    )
    parser.add_argument(
        "--poscar-root",
        type=Path,
        default=Path("../data/dft_priliminary"),
        help="Directory containing O1_NaCoO2_POSCAR and O3_NaCoO2_POSCAR.",
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        default=("O1", "O3", "P3"),
        help="Phases to process (default: (O1, O3, P3)).",
    )
    parser.add_argument(
        "--mace-root",
        type=Path,
        default=Path("../ordering_staging_output"),
        help="Root directory for MACE outputs and summary.csv.",
    )
    parser.add_argument(
        "--relax-cell",
        action="store_true",
        help="Relax both atoms and cell using ExpCellFilter.",
    )
    return parser.parse_args()


def device_auto() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def poscar_map(poscar_root: Path) -> Dict[str, Path]:
    mapping = {
        "O1": poscar_root / "O1_NaCoO2_POSCAR",
        "O3": poscar_root / "O3_NaCoO2_POSCAR",
        "P3": poscar_root / "P3_NaCoO2_POSCAR",
    }
    missing = [p for p in mapping.values() if p is not None and not p.is_file()]
    if missing:
        missing_list = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing POSCAR files:\n{missing_list}")
    return mapping


def load_atoms(path: Path):
    atoms = read(path)
    atoms.pbc = True
    return atoms


def replace_co_with_tm(atoms, tm: str):
    """Return a copy of atoms with all Co replaced by tm."""
    syms = atoms.get_chemical_symbols()
    new_syms = [tm if s == "Co" else s for s in syms]
    atoms.set_chemical_symbols(new_syms)
    return atoms


def remove_na(atoms):
    """Return a copy of atoms with all Na removed."""
    symbols = atoms.get_chemical_symbols()
    keep_idx = [i for i, s in enumerate(symbols) if s != "Na"]
    if len(keep_idx) == len(symbols):
        # Nothing to remove
        return atoms
    new_atoms = atoms[keep_idx]
    new_atoms.pbc = atoms.pbc
    new_atoms.set_cell(atoms.get_cell())
    return new_atoms


def make_calc(model_path: Path, device: str) -> MACECalculator:
    return MACECalculator(model_path=str(model_path), device=device)


def relax_structure(atoms, calc, out_dir: Path, relax_cell: bool = False):
    atoms.set_calculator(calc)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "opt.log"
    traj_file = out_dir / "relax.traj" if WRITE_TRAJ else None

    dyn_obj = ExpCellFilter(atoms) if relax_cell else atoms
    optimizer = QuasiNewton(dyn_obj, logfile=str(log_file))

    if traj_file is not None:
        traj = Trajectory(str(traj_file), "w", atoms)
        optimizer.attach(traj)

    optimizer.run(fmax=FMAX, steps=STEPS)

    forces = atoms.get_forces()
    fmax_final = float(np.max(np.linalg.norm(forces, axis=1)))
    steps_taken = getattr(optimizer, "nsteps", None)
    return fmax_final, steps_taken


def write_poscar(path: Path, atoms) -> None:
    write(path, atoms, format="vasp", vasp5=True, direct=True)


def run_case(
    logical_phase: str,
    phase_dir: str,
    ordering: str,
    group: str,
    na_state: str,
    poscar_path: Path,
    tm: str,
    mace_root: Path,
    calc,
    relax_cell: bool = False,
) -> dict:
    """
    Run a single TM + Na-state case and return a result row.

    - logical_phase: 'O1', 'O3', or 'P3'
    - phase_dir: directory name under mace_root (e.g. 'O3_orderings')
    - ordering: ordering label ('full' or 'empty')
    - group: synthetic group name ('group_000')
    - na_state: 'full' or 'empty'
    """
    t0 = time.time()

    out_dir = mace_root / phase_dir / ordering / group / tm
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = load_atoms(poscar_path)

    # For empty phases, strip Na before TM substitution
    if na_state == "empty":
        atoms = remove_na(atoms)

    atoms = replace_co_with_tm(atoms, tm)

    # Save input structure
    write_poscar(out_dir / "POSCAR_in.vasp", atoms)

    result = {
        "phase": phase_dir,       # must match directory name for DFT script
        "ordering": ordering,
        "group": group,
        "tm": tm,
        "relax_cell": relax_cell,
        "natoms": len(atoms),
        "energy_eV": None,
        "energy_per_atom_eV": None,
        "volume_A3": None,
        "fmax_final_eV_per_A": None,
        "n_steps": None,
        "runtime_s": 0.0,
        "status": "OK",
        "out_dir": str(out_dir),
        # (logical_phase/na_state only in-memory; not written to CSV
        # to keep schema identical with ordering_calculation_mace_cli.py)
    }

    try:
        fmax_final, steps_taken = relax_structure(
            atoms, calc, out_dir, relax_cell=relax_cell
        )
        energy = float(atoms.get_potential_energy())
        volume = float(atoms.get_volume())
        result.update(
            energy_eV=energy,
            energy_per_atom_eV=energy / len(atoms),
            volume_A3=volume,
            fmax_final_eV_per_A=fmax_final,
            n_steps=steps_taken,
        )
        write_poscar(out_dir / "final.POSCAR", atoms)
    except Exception as exc:
        result["status"] = f"FAIL: {type(exc).__name__}: {exc}"

    result["runtime_s"] = time.time() - t0
    return result



def write_summary(rows: Iterable[dict], csv_path: Path) -> None:
    fieldnames = [
        "phase",
        "ordering",
        "group",
        "tm",
        "relax_cell",
        "natoms",
        "energy_eV",
        "energy_per_atom_eV",
        "volume_A3",
        "fmax_final_eV_per_A",
        "n_steps",
        "runtime_s",
        "status",
        "out_dir",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_to_summary(new_rows: Iterable[dict], csv_path: Path) -> None:
    """
    Append new_rows to an existing summary.csv (if present),
    otherwise create it from scratch.
    """
    all_rows: list[dict] = []

    if csv_path.is_file():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    all_rows.extend(new_rows)
    write_summary(all_rows, csv_path)


def main():
    args = parse_args()
    mace_out_root = args.mace_root
    mace_out_root.mkdir(parents=True, exist_ok=True)

    poscars = poscar_map(args.poscar_root)

    # Only keep phases that we have POSCARs for
    requested_phases = [p for p in args.phases if p in poscars]
    if not requested_phases:
        raise SystemExit(
            f"No valid phases requested. Requested={args.phases}, "
            f"available={list(poscars.keys())}"
        )

    device = device_auto()
    print(f"[INFO] Using device: {device}")
    calc = make_calc(args.model, device)

    rows: list[dict] = []

    for logical_phase in requested_phases:
        poscar_path = poscars[logical_phase]
        phase_dir = PHASE_DIR_MAP.get(logical_phase, logical_phase)
        cases = NA_CASES.get(logical_phase, [("full", "full")])

        print(f"\n[PHASE] {logical_phase} (dir={phase_dir}): {poscar_path}")
        for na_state, ordering_label in cases:
            print(f"  [Na={na_state}] ordering={ordering_label}")
            for tm in TMS:
                print(f"    - TM={tm} ... ", end="", flush=True)
                res = run_case(
                    logical_phase=logical_phase,
                    phase_dir=phase_dir,
                    ordering=ordering_label,
                    group=GROUP_LABEL,
                    na_state=na_state,
                    poscar_path=poscar_path,
                    tm=tm,
                    mace_root=mace_out_root,
                    calc=calc,
                    relax_cell=args.relax_cell,
                )
                rows.append(res)
                print(res["status"])

    summary_path = mace_out_root / "summary.csv"
    append_to_summary(rows, summary_path)
    print(f"\n[INFO] Appended {len(rows)} rows to summary: {summary_path}")



if __name__ == "__main__":
    main()
