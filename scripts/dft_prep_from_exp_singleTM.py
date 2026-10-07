#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare DFT inputs for experimental O3/P3 orderings with a single-TM sweep.

For each specified ordering POSCAR (.vasp), this script:
1) reads the structure;
2) replaces all TM sites (everything except Na/O) with each target TM;
3) writes MPRelaxSet-based VASP inputs under:
      <out_root>/<phase>/<ordering>/<tm>/
4) writes a SLURM job script (job.sh);
5) optionally submits each job with sbatch.

DFT setup is intentionally aligned with dft_prep_from_mace_single_TM.py:
  - MPRelaxSet defaults for POTCAR
  - INCAR overrides:
      ENCUT = 680
      EDIFF = 1e-6 eV/atom (converted to total EDIFF by natoms)
      EDIFFG = -0.02
      KSPACING = 0.3
      ISPIN = 2
      IVDW = 12
  - KPOINTS file is intentionally not written (use KSPACING from INCAR)
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet

DEFAULT_TMS: List[str] = ["Co", "Cr", "Fe", "Mn", "Ni", "Ti", "V"]

DEFAULT_PHASE_ORDERINGS: Dict[str, List[str]] = {
    "O3_orderings": ["JMAS_a", "JMAS_b", "JMAS_e", "XC_a", "XC_b", "CD_a", "XL_a"],
    "P3_orderings": ["JMAS_c", "JMAS_d"],
}

USER_EDIFF_PER_ATOM = 1e-6
# Previous setting kept for reference:
# "ENCUT": 520, "IVDW": 12, "ISPIN": 2
USER_INCAR_BASE = {
    "ENCUT": 680,
    "EDIFFG": -0.02,
    "KSPACING": 0.3,
    "IVDW": 12,  # Grimme D3-BJ
    "ISPIN": 2,  # spin-polarized (FM init)
}


def make_job_name(phase: str, ordering: str, tm: str) -> str:
    base = f"vasp_{phase}_{ordering}_{tm}"
    return re.sub(r"[^A-Za-z0-9_]", "_", base)


def write_slurm_script(dft_dir: Path, phase: str, ordering: str, tm: str) -> Path:
    job_name = make_job_name(phase, ordering, tm)
    script_path = dft_dir / "job.sh"
    script = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=48
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module load intel-oneapi/2022.3.1
module load vasp-canepa/6.5.0
mpirun vasp_std > vasp.out
"""
    script_path.write_text(script)
    return script_path


def submit_job(dft_dir: Path, script_path: Path) -> None:
    res = subprocess.run(
        ["sbatch", script_path.name],
        cwd=dft_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    print(f"    sbatch -> returncode {res.returncode}")
    if res.stdout:
        print(f"    stdout: {res.stdout.strip()}")
    if res.stderr:
        print(f"    stderr: {res.stderr.strip()}")


def replace_tm_sites(structure: Structure, target_tm: str) -> Structure:
    """
    Replace every non-Na/non-O site with target_tm.
    Assumes these structures are single-TM hosts.
    """
    new_struct = structure.copy()
    replaced = 0
    for idx, site in enumerate(new_struct.sites):
        symbol = site.specie.symbol
        if symbol not in {"Na", "O"}:
            new_struct.replace(idx, target_tm)
            replaced += 1
    if replaced == 0:
        raise ValueError("No TM sites found (only Na/O present).")
    return new_struct


def resolve_input_file(data_root: Path, phase: str, ordering: str) -> Path:
    poscar = data_root / phase / f"{ordering}.vasp"
    if not poscar.is_file():
        raise FileNotFoundError(f"Missing input file: {poscar}")
    return poscar


def write_case(
    source_path: Path,
    out_root: Path,
    phase: str,
    ordering: str,
    tm: str,
    dry_run: bool = False,
    submit: bool = False,
) -> Path:
    dft_dir = out_root / phase / ordering / tm
    if dry_run:
        print(f"[DRY] {source_path} -> {dft_dir}")
        return dft_dir

    dft_dir.mkdir(parents=True, exist_ok=True)

    base_struct = Structure.from_file(str(source_path))
    tm_struct = replace_tm_sites(base_struct, tm)

    user_incar = dict(USER_INCAR_BASE)
    user_incar["EDIFF"] = USER_EDIFF_PER_ATOM * len(tm_struct)

    # Ni-only POTCAR override (easy to disable by commenting this block):
    user_potcar_settings = None
    # if tm == "Ni":
    #     user_potcar_settings = {"Ni": "Ni_sv_GW"}

    vset = MPRelaxSet(
        structure=tm_struct,
        user_incar_settings=user_incar,
        user_potcar_settings=user_potcar_settings,
        user_potcar_functional="PBE_54_W_HASH",
    )
    vset.poscar.write_file(dft_dir / "POSCAR")
    vset.incar.write_file(dft_dir / "INCAR")
    vset.potcar.write_file(dft_dir / "POTCAR")
    # Intentionally do not write KPOINTS: VASP should use KSPACING from INCAR.
    kpoints_path = dft_dir / "KPOINTS"
    if kpoints_path.exists():
        kpoints_path.unlink()

    (dft_dir / "POSCAR_exp_source").write_text(source_path.read_text())
    meta_path = dft_dir / "EXP_metadata.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"phase: {phase}",
                f"ordering: {ordering}",
                f"tm: {tm}",
                f"exp_input_file: {source_path}",
                "tm_replacement_rule: replace all non-Na/non-O species",
                "",
            ]
        )
    )

    script_path = write_slurm_script(dft_dir, phase=phase, ordering=ordering, tm=tm)
    if submit:
        submit_job(dft_dir, script_path)

    return dft_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DFT single-TM runs from experimental orderings in data/O3_orderings "
            "and data/P3_orderings."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory containing O3_orderings/ and P3_orderings/.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dft_singleTM",
        help="Root directory where DFT input trees will be written.",
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        default=list(DEFAULT_PHASE_ORDERINGS.keys()),
        help="Phase directories to include (default: O3_orderings P3_orderings).",
    )
    parser.add_argument(
        "--orderings",
        nargs="*",
        default=None,
        help=(
            "Optional subset of ordering names to include "
            "(e.g. JMAS_a XC_a JMAS_c). If omitted, use defaults per phase."
        ),
    )
    parser.add_argument(
        "--tms",
        nargs="*",
        default=DEFAULT_TMS,
        help="Target TM symbols (default: Co Cr Fe Mn Ni Ti V).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print actions, do not write files or submit jobs.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit each written directory with sbatch job.sh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    selected = []
    orderings_filter = set(args.orderings) if args.orderings else None
    for phase in args.phases:
        if phase not in DEFAULT_PHASE_ORDERINGS:
            print(f"[WARN] Unknown phase skipped: {phase}")
            continue
        phase_orderings = DEFAULT_PHASE_ORDERINGS[phase]
        if orderings_filter is not None:
            phase_orderings = [o for o in phase_orderings if o in orderings_filter]
        for ordering in phase_orderings:
            selected.append((phase, ordering))

    if not selected:
        raise SystemExit("No (phase, ordering) selected after filters.")

    print(f"Preparing {len(selected)} orderings x {len(args.tms)} TMs")
    print(f"Out root: {out_root}")
    if args.submit:
        print("Submission: enabled")

    total = 0
    for phase, ordering in selected:
        source = resolve_input_file(args.data_root, phase, ordering)
        for tm in args.tms:
            dft_dir = write_case(
                source_path=source,
                out_root=out_root,
                phase=phase,
                ordering=ordering,
                tm=tm,
                dry_run=args.dry_run,
                submit=args.submit,
            )
            print(f"[OK] ({phase}, {ordering}, {tm}) -> {dft_dir}")
            total += 1

    print(f"Done. Prepared {total} DFT directories.")


if __name__ == "__main__":
    main()
