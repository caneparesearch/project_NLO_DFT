#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate static single-point VASP runs from converged relaxations.

Behavior:
  - Searches ROOT for directories containing CONTCAR
  - Can start from original relaxation directories or from 2nd_relax dirs
  - Creates a "static" subfolder under each selected source run dir
  - Copies:
      * POSCAR from source CONTCAR
      * POTCAR (from source run dir)
      * job.sh  (from source run dir)
  - Writes:
      * INCAR (tightened for static run)
      * KPOINTS (denser Gamma-centered mesh)
  - Optionally submits with sbatch
Usage
-----
python src/dft_generate_static_run.py [ROOT] [--no-submit]
                                     [--phases ...]
                                     [--tms ...]
                                     [--orderings ...]
                                     [--source]
                                     [--skip-convergence-check]

Notes
-----
- Requires prior relaxations.
- By default, only prepares static runs from relaxations that look converged
- Can chose which relaxations to prepare from with --source
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Optional, Tuple

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet

R2SCAN_DIRNAME = "r2scan"
STATIC_DIRNAME = "static"
FIRST_STATIC_DIRNAME = "static_from_1st"
MACE_DIRNAME = "mace"
SECOND_RELAX_DIRNAME = "2nd_relax"
EXCLUDED_DIRS = {STATIC_DIRNAME, R2SCAN_DIRNAME, MACE_DIRNAME, FIRST_STATIC_DIRNAME}
DEFAULT_KPPA = 6000
DEFAULT_EDIFF_PER_ATOM = 1e-7

PHASE_SYNONYMS = {
    "O1": ["O1", "O1_orderings", "O1_ordering"],
    "O3": ["O3", "O3_orderings", "O3_ordering"],
    "P3": ["P3", "P3_orderings", "P3_ordering"],
}


def normalize_phase_filters(phases):
    if phases is None:
        return None
    allowed = set()
    for p in phases:
        name = str(p).strip()
        if not name:
            continue
        if name in PHASE_SYNONYMS:
            allowed.update(PHASE_SYNONYMS[name])
        else:
            allowed.add(name)
    return allowed or None


def parse_elec_filters(elec_args):
    if elec_args is None:
        return None
    selected = set()
    for raw in elec_args:
        for tok in str(raw).split(","):
            value = tok.strip()
            if not value:
                continue
            if value.startswith("elec_"):
                value = value[5:]
            if not value.isdigit():
                raise ValueError(f"Invalid --elec token: {tok!r}")
            selected.add(int(value))
    return selected or None


def extract_phase_ordering_tm(source_dir: Path, root: Path) -> Optional[Tuple[str, str, str]]:
    """
    Expected layouts:
      root/<phase>/<ordering>/<tm>/
      root/<phase>/<ordering>/<tm>/2nd_relax/
    """
    try:
        rel = source_dir.relative_to(root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 3:
        return None

    return parts[0], parts[1], parts[2]


def extract_elec_index(source_dir: Path, root: Path) -> Optional[int]:
    try:
        rel = source_dir.relative_to(root)
    except ValueError:
        return None

    for part in rel.parts:
        if part.startswith("elec_"):
            idx = part[5:]
            if idx.isdigit():
                return int(idx)
    return None


def find_source_runs(root: Path, source: str) -> Iterable[Path]:
    for contcar in root.rglob("CONTCAR"):
        run_dir = contcar.parent
        if source == "2nd_relax":
            if run_dir.name == SECOND_RELAX_DIRNAME:
                yield run_dir
            continue

        # source == "relax"
        # TODO Need to check.
        if run_dir.name in EXCLUDED_DIRS or run_dir.name == SECOND_RELAX_DIRNAME:
            continue
        yield run_dir


def copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


def is_relaxation_converged(run_dir: Path) -> Tuple[bool, str]:
    """
    Prefer vasprun.xml convergence flags, then fallback to OUTCAR tail check.
    """
    vasprun_path = run_dir / "vasprun.xml"
    if vasprun_path.is_file():
        try:
            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
            elec = bool(vr.converged_electronic)
            ionic = bool(vr.converged_ionic)
            if elec and ionic:
                return True, "vasprun.xml converged (electronic + ionic)"
            return False, f"vasprun.xml not converged (electronic={elec}, ionic={ionic})"
        except Exception as exc:
            return False, f"failed to parse vasprun.xml: {type(exc).__name__}: {exc}"

    outcar_path = run_dir / "OUTCAR"
    if not outcar_path.is_file():
        return False, "missing both vasprun.xml and OUTCAR"

    try:
        # VASP writes this phrase near successful ionic completion.
        tail = "\n".join(outcar_path.read_text(errors="ignore").splitlines()[-500:])
        marker = "reached required accuracy - stopping structural energy minimisation"
        if marker in tail.lower():
            return True, "OUTCAR indicates ionic convergence"
        return False, "OUTCAR tail lacks convergence marker"
    except Exception as exc:
        return False, f"failed to read OUTCAR: {type(exc).__name__}: {exc}"


def write_incar(path: Path, structure: Structure) -> None:
    ediff_total = DEFAULT_EDIFF_PER_ATOM * len(structure)
    user_incar = {
        "ENCUT": 520,
        "IVDW": 12,
        "ISPIN": 2,
        "EDIFF": ediff_total,
        "ISMEAR": -5,
        "SIGMA": 0.05,
        "IBRION": -1,
        "NSW": 0,
        "ISIF": 2,
        "NELM": 5000,
        "LREAL": "Auto",
        "ALGO": "Normal",
        "LWAVE": False,
        "LCHARG": True,
        "PREC": "Accurate",
    }
    incar = MPRelaxSet(structure, user_incar_settings=user_incar).incar
    incar.write_file(path)


def write_kpoints(path: Path, structure: Structure) -> None:
    kpoints = Kpoints.automatic_density(structure, kppa=DEFAULT_KPPA, force_gamma=True)
    kpoints.write_file(path)


def prepare_static_directory(source_dir: Path) -> Path:
    # If source is .../<tm>/2nd_relax, write static at .../<tm>/static
    # (one level above 2nd_relax). Otherwise keep .../<tm>/static.
    if source_dir.name == SECOND_RELAX_DIRNAME:
        static_dir = source_dir.parent / STATIC_DIRNAME
    else:
        static_dir = source_dir / STATIC_DIRNAME
    static_dir.mkdir(parents=True, exist_ok=True)

    contcar = source_dir / "CONTCAR"
    structure = Structure.from_file(contcar)

    shutil.copy2(contcar, static_dir / "POSCAR")
    copy_if_present(source_dir / "POTCAR", static_dir / "POTCAR")
    copy_if_present(source_dir / "job.sh", static_dir / "job.sh")

    write_incar(static_dir / "INCAR", structure)
    write_kpoints(static_dir / "KPOINTS", structure)
    return static_dir


def submit_job(static_dir: Path) -> bool:
    script = static_dir / "job.sh"
    if not script.exists():
        print(f"[skip] No job.sh in {static_dir}")
        return False
    try:
        subprocess.run(["sbatch", script.name], cwd=static_dir, check=True)
        print(f"[submitted] {static_dir}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[error] job submission failed in {static_dir}: {exc}")
        return False


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "dft_singleTM",
        type=Path,
        help="Root directory containing relax calculations (default: ./dft_singleTM)",
    )
    parser.add_argument(
        "--source",
        choices=["relax", "2nd_relax"],
        default="relax",
        help="Choose whether static inputs are made from original relax dirs or 2nd_relax dirs.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Prepare inputs without launching job.sh",
    )
    parser.add_argument(
        "--skip-convergence-check",
        action="store_true",
        help="Allow preparing static runs even if source relaxation does not look converged.",
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        default=None,
        help="Optional phase filters (e.g. O3_orderings P3_orderings or O3 P3).",
    )
    parser.add_argument(
        "--tms",
        nargs="*",
        default=None,
        help="Optional TM filters (e.g. Co Ni Mn).",
    )
    parser.add_argument(
        "--orderings",
        nargs="*",
        default=None,
        help="Optional ordering filters (e.g. full empty AJT_a_w_vac).",
    )
    parser.add_argument(
        "--elec",
        nargs="*",
        default=None,
        help="Optional elec_* filters by index (e.g. 0 1 or 0,1 or elec_0).",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    root: Path = args.root

    if not root.exists():
        print(f"[warning] Root directory does not exist: {root}")
        return

    phase_filters = normalize_phase_filters(args.phases)
    ordering_filters = set(args.orderings) if args.orderings else None
    tm_filters = set(args.tms) if args.tms else None
    try:
        elec_filters = parse_elec_filters(args.elec)
    except ValueError as exc:
        print(f"[error] {exc}")
        return

    all_runs = list(find_source_runs(root, source=args.source))
    if not all_runs:
        print(f"[warning] No source CONTCAR runs found under {root} (source={args.source})")
        return

    filtered_runs = []
    for source_dir in all_runs:
        parsed = extract_phase_ordering_tm(source_dir, root)
        if parsed is None:
            continue
        phase, ordering, tm = parsed
        if phase_filters is not None and phase not in phase_filters:
            continue
        if ordering_filters is not None and ordering not in ordering_filters:
            continue
        if tm_filters is not None and tm not in tm_filters:
            continue
        if elec_filters is not None:
            elec_idx = extract_elec_index(source_dir, root)
            if elec_idx is None or elec_idx not in elec_filters:
                continue
        filtered_runs.append(source_dir)

    if not filtered_runs:
        print(
            "[warning] No CONTCAR directories left after filtering.\n"
            f"  ROOT       = {root}\n"
            f"  source     = {args.source}\n"
            f"  phases     = {args.phases}\n"
            f"  orderings  = {args.orderings}\n"
            f"  tms        = {args.tms}\n"
            f"  elec       = {args.elec}"
        )
        return

    print(
        f"[info] Preparing static runs for {len(filtered_runs)} directories "
        f"(from {len(all_runs)} source runs with CONTCAR)."
    )

    n_skip_unconv = 0
    for source_dir in filtered_runs:
        if not args.skip_convergence_check:
            ok, reason = is_relaxation_converged(source_dir)
            if not ok:
                print(f"[skip] unconverged source: {source_dir} ({reason})")
                n_skip_unconv += 1
                continue

        static_dir = prepare_static_directory(source_dir)
        if not args.no_submit:
            submit_job(static_dir)
        else:
            print(f"[prepared] {static_dir}")

    if not args.skip_convergence_check:
        print(f"[info] Convergence-check skips: {n_skip_unconv}")


if __name__ == "__main__":
    main()
