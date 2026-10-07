#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yun
"""
Prepare DFT inputs for double-TM workflow from POSCAR_lowestEconfig.vasp files.

Input tree (example):
  ordering_doubleTM/
    O1_orderings|O3_orderings|P3_orderings/
      <ordering>/
        <TM1_TM2>/
          elec_*/
            POSCAR_lowestEconfig.vasp

Output tree (mirrors input hierarchy under --out-root):
  dft_doubleTM/
    O1_orderings|O3_orderings|P3_orderings/
      <ordering>/
        <TM1_TM2>/
          elec_*/
            POSCAR
            INCAR
            POTCAR
            POSCAR_MACE_selected
            MACE_metadata.txt
            job.sh

INCAR/POTCAR generation matches dft_prep_from_mace_singleTM.py:
  - MPRelaxSet with user_incar_settings:
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
from typing import List, Tuple

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet


PHASE_DIRS = ("O1_orderings", "O3_orderings", "P3_orderings")


def _sorted_dirs_by_suffix(paths: List[Path], prefix: str) -> List[Path]:
    def _key(p: Path):
        name = p.name
        if not name.startswith(prefix):
            return (1, name)
        tail = name[len(prefix) :]
        try:
            return (0, int(tail))
        except ValueError:
            return (0, tail)

    return sorted(paths, key=_key)


def make_job_name(rel_parts: Tuple[str, ...]) -> str:
    base = "_".join(rel_parts)
    return re.sub(r"[^A-Za-z0-9_]", "_", base)


def write_slurm_script(
    rel_parts: Tuple[str, ...],
    dft_dir: Path,
    submit: bool = False,
    dry_run: bool = False,
) -> Path:
    job_name = make_job_name(rel_parts)
    script_path = dft_dir / "job.sh"

    script = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 2
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=28
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module load intel-oneapi/2022.3.1
module load vasp-canepa/6.5.0
mpirun vasp_std > vasp.out
vasp_exit=$?

if [ "$vasp_exit" -eq 0 ]; then
    rm -f CHG CHGCAR
fi

exit "$vasp_exit"
"""

    if dry_run:
        print(f"[DRY] Would write SLURM script at: {script_path}")
    else:
        script_path.write_text(script)

    if submit and not dry_run:
        try:
            res = subprocess.run(
                f"cd {dft_dir} && sbatch {script_path.name}",
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )
            print(f"    sbatch -> returncode {res.returncode}")
            if res.stdout:
                print(f"    stdout: {res.stdout.strip()}")
            if res.stderr:
                print(f"    stderr: {res.stderr.strip()}")
        except Exception as exc:
            print(f"    [WARN] Failed to submit job in {dft_dir}: {exc}")

    return script_path


def discover_lowest_poscars(
    root: Path,
    poscar_name: str,
    phases: List[str] | None = None,
    orderings: List[str] | None = None,
    tm_pairs: List[str] | None = None,
) -> List[Path]:
    out: List[Path] = []
    phase_list = phases if phases else list(PHASE_DIRS)
    ordering_set = set(orderings) if orderings else None
    tm_pair_set = set(tm_pairs) if tm_pairs else None

    for phase in phase_list:
        phase_dir = root / phase
        if not phase_dir.exists():
            continue
        for ordering_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
            if ordering_set is not None and ordering_dir.name not in ordering_set:
                continue
            for tm_pair_dir in sorted([p for p in ordering_dir.iterdir() if p.is_dir()]):
                if tm_pair_set is not None and tm_pair_dir.name not in tm_pair_set:
                    continue
                elec_dirs = [p for p in tm_pair_dir.glob("elec_*") if p.is_dir()]
                elec_dirs = _sorted_dirs_by_suffix(elec_dirs, "elec_")
                for elec_dir in elec_dirs:
                    p = elec_dir / poscar_name
                    if p.is_file():
                        out.append(p)
    return out


def parse_elec_filter(raw: List[str] | None) -> set[int] | None:
    """
    Parse elec filter tokens from CLI.
    Supports:
      --elec 0
      --elec 4 5
      --elec 4,5
      --elec 0,2 4
    """
    if not raw:
        return None
    out: set[int] = set()
    for token in raw:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            out.add(int(part))
    return out


def remove_k_sites(struct: Structure) -> Structure:
    """Return a copy of structure with all K sites removed."""
    s = struct.copy()
    k_idx = [i for i, site in enumerate(s) if site.specie.symbol == "K"]
    if k_idx:
        s.remove_sites(k_idx)
    return s


def write_one_case(
    poscar_path: Path,
    in_root: Path,
    out_root: Path,
    dry_run: bool,
    submit: bool,
    overwrite: bool,
) -> Path:
    rel = poscar_path.parent.relative_to(in_root)  # phase/ordering/tm_pair/elec_x
    dft_dir = out_root / rel
    if not dry_run:
        dft_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[DRY] DFT dir: {dft_dir}")
        print(f"[DRY] Using POSCAR: {poscar_path}")
        return dft_dir

    # Skip if existing VASP input set is already present and overwrite is off.
    # Previous skip check (kept for reference) included KPOINTS:
    # ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "job.sh")
    if not overwrite and all((dft_dir / fn).exists() for fn in ("POSCAR", "INCAR", "POTCAR", "job.sh")):
        print(f"[SKIP] Existing VASP set: {dft_dir}")
        return dft_dir

    struct = Structure.from_file(str(poscar_path))
    struct = remove_k_sites(struct)
    # Previous setting kept for reference:
    # "ENCUT": 520, IVDW: 12, ISPIN: 2
    user_incar = {
        "ENCUT": 680,
        "EDIFF": 1e-6 * len(struct),  # 1e-6 eV/atom -> total EDIFF
        "EDIFFG": -0.02,
        "KSPACING": 0.3,
        "IVDW": 12,
        "ISPIN": 2,
        "NELM": 150,
        "NSW": 200,
    }

    # Ni-only POTCAR override (easy to disable by commenting this block):
    user_potcar_settings = None
    # if "Ni" in rel.parts[2].split("_"):
    #     user_potcar_settings = {"Ni": "Ni_sv_GW"}
    #     print("Changing Ni POTCAR to Ni_sv_GW")

    vset = MPRelaxSet(
        structure=struct,
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

    selected_copy = dft_dir / "POSCAR_MACE_selected"
    if overwrite or not selected_copy.exists():
        struct.to(filename=str(selected_copy))

    meta_path = dft_dir / "MACE_metadata.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"source_poscar: {poscar_path}",
                f"input_root: {in_root}",
                f"relative_case_dir: {rel}",
            ]
        )
        + "\n"
    )

    rel_parts = tuple(rel.parts)
    write_slurm_script(rel_parts=rel_parts, dft_dir=dft_dir, submit=submit, dry_run=dry_run)
    return dft_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Mirror ordering_doubleTM hierarchy into dft_doubleTM and prepare VASP "
            "inputs from POSCAR_lowestEconfig.vasp."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("../ordering_doubleTM"),
        help="Input root containing O1/O3/P3_orderings tree.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("../dft_doubleTM"),
        help="Output root for DFT VASP input sets.",
    )
    p.add_argument(
        "--poscar-name",
        default="POSCAR_lowestEconfig.vasp",
        help="POSCAR filename to search under each elec_*.",
    )
    p.add_argument(
        "--elec",
        nargs="*",
        default=None,
        help=(
            "Optional elec index filter. Examples: --elec 0, --elec 4 5, --elec 4,5. "
            "Only matching elec_* will be processed."
        ),
    )
    p.add_argument(
        "--phases",
        nargs="*",
        default=None,
        help="Optional phase filters (e.g. O1_orderings O3_orderings P3_orderings).",
    )
    p.add_argument(
        "--orderings",
        nargs="*",
        default=None,
        help="Optional ordering filters (e.g. AJT_a_w_vac XC_a).",
    )
    p.add_argument(
        "--tms",
        "--tm-pairs",
        dest="tm_pairs",
        nargs="*",
        default=None,
        help="Optional TM-pair filters (e.g. Co_Ni Mn_Fe).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, do not write files or submit jobs.",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Submit generated jobs with sbatch.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    args = p.parse_args()

    in_root = args.root.resolve()
    out_root = args.out_root.resolve()
    elec_filter = parse_elec_filter(args.elec)

    if not in_root.exists():
        raise SystemExit(f"Input root not found: {in_root}")
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    poscars = discover_lowest_poscars(
        in_root,
        args.poscar_name,
        phases=args.phases,
        orderings=args.orderings,
        tm_pairs=args.tm_pairs,
    )
    if elec_filter is not None:
        filtered: List[Path] = []
        for pth in poscars:
            elec_name = pth.parent.name  # elec_*
            if not elec_name.startswith("elec_"):
                continue
            try:
                idx = int(elec_name.split("_", 1)[1])
            except Exception:
                continue
            if idx in elec_filter:
                filtered.append(pth)
        poscars = filtered

    if not poscars:
        raise SystemExit(
            f"No '{args.poscar_name}' found under {in_root} in expected elec_* layout."
        )

    print(f"Found {len(poscars)} selected POSCARs.")
    print("First few:")
    for pth in poscars[:5]:
        print(f"  - {pth}")

    for poscar_path in poscars:
        dft_dir = write_one_case(
            poscar_path=poscar_path,
            in_root=in_root,
            out_root=out_root,
            dry_run=args.dry_run,
            submit=args.submit,
            overwrite=args.overwrite,
        )
        print(f"[OK] {poscar_path.parent} -> {dft_dir}")

    action = "launched" if args.submit and not args.dry_run else "prepared"
    print(f"[SUMMARY] DFT calculations {action}: {len(poscars)}")


if __name__ == "__main__":
    main()
