#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yun
"""
Minimal MACE runner over ordering_staging groups.

Assumptions
-----------
- Should have directory structure like this:
  ordering_staging/
    O3_orderings|P3_orderings/
      <ordering_name>/
        group_000/ POSCAR (or POSCAR*)
        group_001/ POSCAR ...

- For generating ordering_staging/, 
  see notebook/Staging_layers_noTMMix.ipynb
        
This script will, for each group:
  1) Load POSCAR
  2) Remove all K (K = vacancy marker)
  3) For each TM in TMS, replace Co -> TM
  4) Relax atoms (and optionally the cell) with MACE
  5) Write final.POSCAR
  6) Append a row to summary.csv

Usage
-----
python ordering_calculation_mace_cli.py \
    --staging-root /path/to/ordering_staging/ \
    --output-root /path/to/mace_results/ \
    --model /path/to/mace_model.pth \
    --relax-cell
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.constraints import ExpCellFilter
from mace.calculators.mace import MACECalculator

# ---------- Fixed settings (edit here if needed) ----------
TMS = ["Ti", "V", "Cr", "Mn", "Fe", "Ni", "Co"]   # Co will be replaced by each of these
FMAX = 0.05                                       # eV/Å
STEPS = 1000
WRITE_TRAJ = True                                 # set False to disable relax.traj
# ---------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Run MACE over ordering_staging groups (simple).")
    p.add_argument("--staging-root", type=Path, required=True,
                   help="Path to ordering_staging/ containing O3_orderings and P3_orderings.")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Output dir for results and summary.csv.")
    p.add_argument("--model", type=Path, required=True,
                   help="Path to MACE model.")
    p.add_argument("--relax-cell", action="store_true",
                   help="Relax cell + atoms using ExpCellFilter.")
    return p.parse_args()


def device_auto() -> str:
    """Pick CUDA if available, otherwise CPU."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def find_group_poscars(staging_root: Path):
    """Yield (poscar_path, phase, ordering, group)."""
    for phase in ("O3_orderings", "P3_orderings"):
        phase_dir = staging_root / phase
        if not phase_dir.exists():
            continue
        for ordering_dir in sorted(phase_dir.iterdir()):
            if not ordering_dir.is_dir():
                continue
            ordering = ordering_dir.name
            for gdir in sorted(ordering_dir.glob("group_*")):
                if not gdir.is_dir():
                    continue
                group = gdir.name
                poscar = gdir / "POSCAR"
                if not poscar.exists():
                    cands = sorted(gdir.glob("POSCAR*"))
                    if not cands:
                        continue
                    poscar = cands[0]
                yield poscar, phase, ordering, group


def load_atoms(path: Path):
    atoms = read(path)  # ASE infers VASP
    atoms.pbc = True
    return atoms


def remove_K(atoms):
    """Treat K as vacancy: drop all K atoms."""
    keep = [s != "K" for s in atoms.get_chemical_symbols()]
    return atoms[keep]


def replace_Co_with_TM(atoms, tm: str):
    """Replace all Co atoms by the chosen TM."""
    syms = atoms.get_chemical_symbols()
    new_syms = [tm if s == "Co" else s for s in syms]
    atoms.set_chemical_symbols(new_syms)
    return atoms


def make_calc(model_path: Path, device: str) -> MACECalculator:
    """Build a MACE calculator on the requested device."""
    return MACECalculator(model_path=str(model_path), device=device)


def relax_structure(atoms, calc, out_dir: Path, relax_cell: bool = False):
    """
    Relax atoms (and optionally the cell) with QuasiNewton.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to relax. Will be modified in-place.
    calc : ASE calculator
        MACE calculator.
    out_dir : Path
        Directory for logs / trajectory.
    relax_cell : bool
        If True, use ExpCellFilter to relax the cell as well.

    Returns
    -------
    fmax_final : float
        Final max force (eV/Å).
    steps_taken : int
        Number of optimizer steps.
    """
    atoms.set_calculator(calc)

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "opt.log"
    traj_file = out_dir / "relax.traj" if WRITE_TRAJ else None

    # Wrap in ExpCellFilter if we also relax the cell.
    if relax_cell:
        dyn_obj = ExpCellFilter(atoms)
    else:
        dyn_obj = atoms

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


def run_case(poscar_path: Path, phase: str, ordering: str, group: str,
             tm: str, out_root: Path, calc, relax_cell: bool = False) -> dict:
    """
    Run a single TM replacement + relaxation case and return a result dict.
    """
    t0 = time.time()
    out_dir = out_root / phase / ordering / group / tm
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare input atoms (K->vac, Co->TM)
    atoms = load_atoms(poscar_path)
    atoms = remove_K(atoms)
    atoms = replace_Co_with_TM(atoms, tm)

    # Save input structure
    write_poscar(out_dir / "POSCAR_in.vasp", atoms)

    result = {
        "phase": phase,
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
    }

    try:
        fmax_final, steps_taken = relax_structure(atoms, calc, out_dir, relax_cell=relax_cell)
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

    # Write a per-case JSON for quick inspection (best-effort).
    try:
        import json
        with open(out_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    return result


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = device_auto()
    print(f"[INFO] Using device: {device}")
    calc = make_calc(args.model, device)

    tasks = list(find_group_poscars(args.staging_root))
    if not tasks:
        print("No POSCARs found under staging root.", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Found {len(tasks)} groups.")

    rows: list[dict] = []

    for poscar_path, phase, ordering, group in tasks:
        print(f"\n[GROUP] {phase}/{ordering}/{group}")
        for tm in TMS:
            print(f"  - TM={tm} ... ", end="", flush=True)
            res = run_case(
                poscar_path=poscar_path,
                phase=phase,
                ordering=ordering,
                group=group,
                tm=tm,
                out_root=args.output_root,
                calc=calc,
                relax_cell=args.relax_cell,
            )
            rows.append(res)
            print(res["status"])

    # --- Clean CSV summary ---
    csv_path = args.output_root / "summary.csv"
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

    print(f"\n[INFO] Wrote summary: {csv_path}")


if __name__ == "__main__":
    main()
