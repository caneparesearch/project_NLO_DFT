#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yun
"""
Collect DFT results from double-TM VASP runs, run custodian handlers in check/fix mode,
and write a summary CSV.

Expected relax tree:
  dft_doubleTM/
    <phase>/<ordering>/<tm_pair>/elec_*/INCAR, vasprun.xml, ...

Sub-runs (optional):
  dft_doubleTM/<phase>/<ordering>/<tm_pair>/elec_*/static/...
  dft_doubleTM/<phase>/<ordering>/<tm_pair>/elec_*/r2scan/...
  dft_doubleTM/<phase>/<ordering>/<tm_pair>/elec_*/2nd_relax/...

Compared with singleTM collector:
  - No MACE metadata parsing.
  - Adds --elec filter (e.g. --elec 0, --elec 4 5, --elec 4,5).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Outcar, Vasprun

from custodian.vasp.handlers import (
    AliasingErrorHandler,
    LrfCommutatorHandler,
    NonConvergingErrorHandler,
    PositiveEnergyErrorHandler,
    PotimErrorHandler,
    StdErrHandler,
    UnconvergedErrorHandler,
    VaspErrorHandler,
)
from custodian.utils import tracked_lru_cache

R2SCAN_DIRNAME = "r2scan"
STATIC_DIRNAME = "static"
MACE_DIRNAME = "mace"
SECOND_RELAX_DIRNAME = "2nd_relax"
EXCLUDED_DIRS = {STATIC_DIRNAME, R2SCAN_DIRNAME, MACE_DIRNAME, SECOND_RELAX_DIRNAME}


@contextmanager
def chdir(path: Path):
    old = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def parse_elec_filter(raw: Optional[List[str]]) -> Optional[set[int]]:
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


def find_vasp_run_dirs(
    dft_root: Path,
    run_type: str = "relax",
    phases: Optional[List[str]] = None,
    orderings: Optional[List[str]] = None,
    tm_pairs: Optional[List[str]] = None,
    elec_filter: Optional[set[int]] = None,
) -> List[Path]:
    """
    Find VASP run directories containing INCAR + vasprun.xml with doubleTM layout.
    """
    allowed = {"relax", "static", "r2scan", "2nd_relax"}
    if run_type not in allowed:
        raise ValueError(f"Invalid run_type={run_type!r}; choose one of {sorted(allowed)}")

    phase_set = set(phases) if phases else None
    ordering_set = set(orderings) if orderings else None
    pair_set = set(tm_pairs) if tm_pairs else None

    run_dirs: List[Path] = []
    for root, _, files in os.walk(dft_root):
        files_set = set(files)
        if "INCAR" not in files_set or "vasprun.xml" not in files_set:
            continue

        run_dir = Path(root)
        try:
            rel_parts = run_dir.relative_to(dft_root).parts
        except ValueError:
            continue

        if run_type == "static":
            if STATIC_DIRNAME not in rel_parts:
                continue
        elif run_type == "r2scan":
            if R2SCAN_DIRNAME not in rel_parts:
                continue
        elif run_type == "2nd_relax":
            if SECOND_RELAX_DIRNAME not in rel_parts:
                continue
        else:
            # Original relax must be exactly phase/ordering/tm_pair/elec_*
            if len(rel_parts) != 4:
                continue
            if any(name in rel_parts for name in EXCLUDED_DIRS):
                continue

        # For both relax and sub-runs, infer from first 4 components.
        if len(rel_parts) < 4:
            continue
        phase, ordering, tm_pair, elec = rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3]
        if not elec.startswith("elec_"):
            continue
        try:
            elec_i = int(elec.split("_", 1)[1])
        except Exception:
            continue

        if phase_set is not None and phase not in phase_set:
            continue
        if ordering_set is not None and ordering not in ordering_set:
            continue
        if pair_set is not None and tm_pair not in pair_set:
            continue
        if elec_filter is not None and elec_i not in elec_filter:
            continue

        run_dirs.append(run_dir)

    return sorted(run_dirs)


def run_handlers_on_dir(run_dir: Path, apply_fixes: bool = False) -> Dict[str, object]:
    ve_subset = list(VaspErrorHandler.error_msgs.keys())
    if "bravais" in ve_subset:
        ve_subset.remove("bravais")
    vasp_err_handler = VaspErrorHandler(errors_subset_to_catch=ve_subset)

    handlers = [
        vasp_err_handler,
        UnconvergedErrorHandler(),
        NonConvergingErrorHandler(),
        AliasingErrorHandler(),
        PositiveEnergyErrorHandler(),
        PotimErrorHandler(),
        LrfCommutatorHandler(),
        StdErrHandler(),
    ]

    records: List[Dict] = []
    corrections: List[Dict] = []
    # Use an absolute run path for handler I/O to avoid cache collisions on
    # repeated relative paths such as "./vasprun.xml" across different run dirs.
    abs_run_dir = str(run_dir.resolve())
    tracked_lru_cache.tracked_cache_clear()
    for h in handlers:
        hname = h.__class__.__name__
        rec: Dict[str, object] = {
            "handler": hname,
            "error_detected": False,
            "errors": None,
            "handler_error": None,
            "correction": None,
        }
        try:
            has_error = bool(h.check(directory=abs_run_dir))
        except TypeError:
            # Backward compatibility for handlers that don't accept
            # a `directory` kwarg.
            with chdir(run_dir):
                has_error = bool(h.check())
        except Exception as exc:
            rec["handler_error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            continue

        if has_error:
            rec["error_detected"] = True
            rec["errors"] = getattr(h, "errors", None)
            if apply_fixes:
                try:
                    corr = h.correct(directory=abs_run_dir)
                    rec["correction"] = corr
                    corrections.append({"handler": hname, "correction": corr})
                except TypeError:
                    try:
                        # Backward compatibility for handlers that don't accept
                        # a `directory` kwarg.
                        with chdir(run_dir):
                            corr = h.correct()
                        rec["correction"] = corr
                        corrections.append({"handler": hname, "correction": corr})
                    except Exception as exc:
                        rec["handler_error"] = f"correct() failed: {type(exc).__name__}: {exc}"
                except Exception as exc:
                    rec["handler_error"] = f"correct() failed: {type(exc).__name__}: {exc}"
        records.append(rec)
    tracked_lru_cache.tracked_cache_clear()

    triggered = [r for r in records if r["error_detected"]]
    n_errors = len(triggered)
    handler_names = sorted({r["handler"] for r in triggered})
    handler_str = ";".join(handler_names)

    error_type_list: List[str] = []
    for r in triggered:
        errs = r.get("errors")
        if isinstance(errs, (list, tuple)):
            for e in errs:
                error_type_list.append(f"{r['handler']}:{e}")
        elif errs:
            error_type_list.append(f"{r['handler']}:{errs}")
    has_unconverged = "UnconvergedErrorHandler" in handler_names
    corrections_applied = bool(corrections) if apply_fixes else False
    print(
        f"[handler] {run_dir} -> "
        f"has_unconverged={has_unconverged}, "
        f"corrections_applied={corrections_applied}"
    )

    return {
        "n_custodian_errors": n_errors,
        "custodian_handlers": handler_str,
        "custodian_error_types": ";".join(error_type_list),
        "custodian_corrections": json.dumps(corrections) if corrections else "",
    }


def read_incar_nsw(run_dir: Path) -> Optional[int]:
    """Read NSW from INCAR if available/parsible."""
    incar_path = run_dir / "INCAR"
    if not incar_path.is_file():
        return None
    try:
        incar = Incar.from_file(str(incar_path))
        nsw = incar.get("NSW", None)
        return int(nsw) if nsw is not None else None
    except Exception:
        return None


def apply_nonconvergence_fix(
    run_dir: Path,
    nsw_target_min: int = 180,
    nsw_increment: int = 80,
) -> Dict[str, object]:
    """
    Apply a minimal fix for ionic non-convergence by increasing NSW in INCAR.
    Returns summary fields for logging into the CSV row.
    """
    out = {
        "nonconv_fix_applied": False,
        "nonconv_fix_reason": "",
        "incar_nsw_new": None,
        "nonconv_fix_error": "",
    }

    incar_path = run_dir / "INCAR"
    if not incar_path.is_file():
        out["nonconv_fix_error"] = "INCAR not found"
        return out

    actions: List[str] = []

    try:
        incar = Incar.from_file(str(incar_path))
        old_nsw = incar.get("NSW", None)
        old_nsw_i = int(old_nsw) if old_nsw is not None else None
        new_nsw = nsw_target_min if old_nsw_i is None else max(old_nsw_i + nsw_increment, nsw_target_min)
        old_ibrion = incar.get("IBRION", None)
        old_ibrion_i = int(old_ibrion) if old_ibrion is not None else None

        if old_nsw_i is None or old_nsw_i < new_nsw:
            incar["NSW"] = int(new_nsw)
            out["incar_nsw_new"] = int(new_nsw)
            actions.append(f"NSW: {old_nsw_i} -> {new_nsw}")

        if old_ibrion_i == 2:
            incar["IBRION"] = 1
            actions.append("IBRION: 2 -> 1")

        if actions:
            incar.write_file(str(incar_path))
            out["nonconv_fix_applied"] = True

        contcar_path = run_dir / "CONTCAR"
        poscar_path = run_dir / "POSCAR"
        if contcar_path.is_file():
            shutil.copy2(contcar_path, poscar_path)
            out["nonconv_fix_applied"] = True
            actions.append("CONTCAR -> POSCAR")

        if actions:
            out["nonconv_fix_reason"] = "; ".join(actions)
        else:
            out["nonconv_fix_reason"] = "No changes needed"
    except Exception as exc:
        out["nonconv_fix_error"] = f"{type(exc).__name__}: {exc}"

    return out


def submit_job(run_dir: Path, job_script_name: str) -> None:
    script_path = run_dir / job_script_name
    if not script_path.is_file():
        print(f"[WARN] {job_script_name} not found in {run_dir}")
        return

    print(f"[SUBMIT] Launching {job_script_name} in {run_dir}")
    with chdir(run_dir):
        res = subprocess.run(
            ["sbatch", job_script_name],
            check=False,
            capture_output=True,
            text=True,
        )
    if res.returncode != 0:
        print(f"         sbatch failed with return code {res.returncode}")
        if res.stderr:
            print(f"         stderr: {res.stderr.strip()}")
    elif res.stdout:
        print(f"         {res.stdout.strip()}")


def summarize_run(
    run_dir: Path,
    dft_root: Path,
    apply_fixes: bool = False,
    submit_fixed: bool = False,
    job_script_name: str = "job.sh",
) -> Dict:
    row: Dict = {}
    row["run_dir"] = str(run_dir)
    rel = run_dir.relative_to(dft_root)
    parts = list(rel.parts)
    row["relpath"] = str(rel)

    # Infer phase/ordering/tm_pair/elec from first four components
    row["phase"] = parts[0] if len(parts) >= 1 else None
    row["ordering"] = parts[1] if len(parts) >= 2 else None
    row["tm_pair"] = parts[2] if len(parts) >= 3 else None
    row["elec"] = parts[3] if len(parts) >= 4 else None

    row["converged_electronic"] = None
    row["converged_ionic"] = None
    row["converged"] = False
    row["nsteps_ionic"] = None
    row["energy_eV"] = None
    row["energy_per_atom_eV"] = None
    row["volume_A3"] = None
    row["lattice"] = None
    row["natoms"] = None
    row["formula"] = None
    row["efermi_eV"] = None
    row["total_mag"] = None
    row["status"] = "OK"
    row["error_msg"] = None
    row["n_custodian_errors"] = 0
    row["custodian_handlers"] = ""
    row["custodian_error_types"] = ""
    row["custodian_corrections"] = ""
    row["incar_nsw"] = None
    row["ionic_hit_nsw_limit"] = False
    row["nonconv_fix_applied"] = False
    row["nonconv_fix_reason"] = ""
    row["incar_nsw_new"] = None
    row["nonconv_fix_error"] = ""

    vasprun_path = run_dir / "vasprun.xml"
    outcar_path = run_dir / "OUTCAR"
    parse_ok = False

    if not vasprun_path.is_file():
        row["status"] = "NO_VASPRUN"
        row["error_msg"] = "vasprun.xml not found"
    else:
        try:
            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
            parse_ok = True
            row["converged_electronic"] = bool(vr.converged_electronic)
            row["converged_ionic"] = bool(vr.converged_ionic)
            row["converged"] = bool(vr.converged_electronic and vr.converged_ionic)

            final_struct: Structure = vr.final_structure
            row["natoms"] = len(final_struct)
            row["volume_A3"] = float(final_struct.volume)
            row["lattice"] = final_struct.lattice.as_dict()
            row["formula"] = final_struct.composition.reduced_formula
            row["energy_eV"] = float(vr.final_energy)
            row["energy_per_atom_eV"] = row["energy_eV"] / row["natoms"] if row["natoms"] else None
            if vr.ionic_steps is not None:
                row["nsteps_ionic"] = len(vr.ionic_steps)
            try:
                row["efermi_eV"] = float(vr.efermi)
            except Exception:
                row["efermi_eV"] = None
        except Exception as exc:
            row["status"] = "PARSE_ERROR_VASPRUN"
            row["error_msg"] = f"{type(exc).__name__}: {exc}"

    if outcar_path.is_file():
        try:
            oc = Outcar(str(outcar_path))
            row["total_mag"] = getattr(oc, "total_mag", None)
        except Exception as exc:
            if row["error_msg"]:
                row["error_msg"] += f" | OUTCAR: {type(exc).__name__}: {exc}"
            else:
                row["error_msg"] = f"OUTCAR: {type(exc).__name__}: {exc}"

    row.update(run_handlers_on_dir(run_dir, apply_fixes=apply_fixes))

    # Check whether ionic loop appears to have hit the NSW cap.
    row["incar_nsw"] = read_incar_nsw(run_dir)
    if (
        parse_ok
        and row["converged_ionic"] is False
        and row["nsteps_ionic"] is not None
        and row["incar_nsw"] is not None
    ):
        row["ionic_hit_nsw_limit"] = bool(int(row["nsteps_ionic"]) >= int(row["incar_nsw"]))

    if row["status"] == "OK":
        if parse_ok:
            if not row["converged"]:
                row["status"] = "NOT_CONVERGED"
            elif row["n_custodian_errors"] > 0:
                row["status"] = "HANDLER_ERROR"
        elif row["n_custodian_errors"] > 0:
            row["status"] = "HANDLER_ERROR"

    # Optional extra fixer for ionic non-convergence even when handlers do not trigger.
    if apply_fixes and parse_ok and row["converged_ionic"] is False:
        nonconv_info = apply_nonconvergence_fix(run_dir)
        if row["ionic_hit_nsw_limit"]:
            reason_prefix = "ionic_not_converged_hit_nsw_limit"
        else:
            reason_prefix = "ionic_not_converged"
        if nonconv_info["nonconv_fix_applied"]:
            nonconv_info["nonconv_fix_reason"] = (
                f"{reason_prefix} | {nonconv_info['nonconv_fix_reason']}"
            )
        elif not nonconv_info["nonconv_fix_reason"]:
            nonconv_info["nonconv_fix_reason"] = reason_prefix
        row.update(nonconv_info)
        print(
            f"[nsw_fix] {run_dir} -> "
            f"applied={row['nonconv_fix_applied']}, "
            f"reason={row['nonconv_fix_reason'] or row['nonconv_fix_error']}"
        )

    had_handler_corrections = bool(row.get("custodian_corrections"))
    if apply_fixes and submit_fixed and (had_handler_corrections or row["nonconv_fix_applied"]):
        submit_job(run_dir, job_script_name)

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Collect DFT results from double-TM VASP runs into a CSV summary."
    )
    parser.add_argument("--dft-root", type=Path, required=True, help="Root directory (e.g. ./dft_doubleTM).")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--only-converged", action="store_true", help="Include only fully converged runs.")
    parser.add_argument("--apply-fixes", action="store_true", help="Apply custodian corrections.")
    parser.add_argument(
        "--submit-fixed",
        action="store_true",
        help="Submit runs where corrections were applied (requires --apply-fixes).",
    )
    parser.add_argument("--job-script-name", type=str, default="job.sh", help="SLURM script to sbatch.")
    parser.add_argument(
        "--run-type",
        choices=["relax", "static", "r2scan", "2nd_relax"],
        default="relax",
        help="Which run directories to collect.",
    )
    parser.add_argument("--phases", nargs="*", default=None, help="Optional phase filters.")
    parser.add_argument("--orderings", nargs="*", default=None, help="Optional ordering filters.")
    parser.add_argument("--tm-pairs", nargs="*", default=None, help="Optional tm_pair filters (e.g. Co_Ni).")
    parser.add_argument(
        "--elec",
        nargs="*",
        default=None,
        help="Optional elec filter. Examples: --elec 0, --elec 4 5, --elec 4,5",
    )

    args = parser.parse_args()
    if args.submit_fixed and not args.apply_fixes:
        raise SystemExit("--submit-fixed requires --apply-fixes.")

    dft_root: Path = args.dft_root
    if not dft_root.is_dir():
        raise SystemExit(f"dft_root is not a directory: {dft_root}")

    elec_filter = parse_elec_filter(args.elec)
    run_dirs = find_vasp_run_dirs(
        dft_root=dft_root,
        run_type=args.run_type,
        phases=args.phases,
        orderings=args.orderings,
        tm_pairs=args.tm_pairs,
        elec_filter=elec_filter,
    )
    if not run_dirs:
        raise SystemExit(
            f"No VASP run directories (INCAR + vasprun.xml) found under {dft_root} "
            f"for run_type={args.run_type!r}"
        )

    print(f"Found {len(run_dirs)} VASP run directories under {dft_root}")
    rows: List[Dict] = []
    for rd in run_dirs:
        rows.append(
            summarize_run(
                rd,
                dft_root=dft_root,
                apply_fixes=args.apply_fixes,
                submit_fixed=args.submit_fixed,
                job_script_name=args.job_script_name,
            )
        )

    df = pd.DataFrame(rows)
    if args.only_converged:
        before = len(df)
        df = df[df["converged"] == True].copy()
        print(f"Filtered to converged runs: {before} -> {len(df)}")

    sort_cols = [c for c in ["phase", "ordering", "tm_pair", "elec"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote summary CSV to: {args.output_csv}")

    preview_cols = [
        c
        for c in [
            "phase",
            "ordering",
            "tm_pair",
            "elec",
            "run_dir",
            "converged",
            "n_custodian_errors",
            "status",
            "energy_per_atom_eV",
        ]
        if c in df.columns
    ]
    print("\nPreview:")
    print(df[preview_cols].head().to_string(index=False))

    ok_mask = (
        (df["converged"] == True)
        & (df["n_custodian_errors"] == 0)
        & (df["status"] == "OK")
    )
    print(f"\nTotal clean, converged runs (no handler errors): {int(ok_mask.sum())} / {len(df)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
