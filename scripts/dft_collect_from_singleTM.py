#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yun
"""
Collect DFT results from a directory tree of single-TM VASP runs,
check convergence and common VASP errors (via custodian), 
and write a summary CSV with data + metadata.

Assumptions
-----------
- Should have directory structure like this:
  dft_singleTM/
    O3_orderings/
      AJT_a_w_vac/
        Co/
          INCAR, POSCAR, POTCAR, KPOINTS, vasprun.xml, OUTCAR, ...
          MACE_metadata.txt
        Ni/
          ...
    P3_orderings/
      ...

This script will:
  1) For each run, parse:
      - convergence flags (ionic/electronic)
      - final energy, energy/atom, volume, composition
      - total magnetization (if OUTCAR present)
      - phase / ordering / tm from path and/or MACE_metadata.txt
  2) Run custodian error handloers in check-only mode and attach:
      - which handlers triggered
      - human readable suggestions.
  3) Write a CSV summary to the save path.

Usage
-----
# Just collect + diagnose:
python dft_collect_from_single_TM.py \
    --dft-root ./dft_singleTM \
    --output-csv ./data/singleTM_dft_summary.csv

# Only collect converged runs in CSV:
python dft_collect_from_single_TM.py \
    --dft-root ./dft_singleTM \
    --output-csv ./data/singleTM_dft_summary_converged.csv \
    --only-converged

# Diagnose + apply custodian corrections to inputs (no submission):
python dft_collect_from_single_TM.py \
    --dft-root ./dft_singleTM \
    --output-csv ./data/singleTM_dft_summary_fixed.csv \
    --apply-fixes

# Diagnose + apply corrections + resubmit fixed runs:
python dft_collect_from_single_TM.py \
    --dft-root ./dft_singleTM \
    --output-csv ./data/singleTM_dft_summary_fixed_submitted.csv \
    --apply-fixes \
    --submit-fixed \
    --job-script-name job.sh
"""

from __future__ import annotations

import os, json, argparse, subprocess, shutil
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Vasprun, Outcar

from custodian.vasp.handlers import (
    VaspErrorHandler,
    UnconvergedErrorHandler,
    NonConvergingErrorHandler,
    MeshSymmetryErrorHandler,
    AliasingErrorHandler,
    PositiveEnergyErrorHandler,
    PotimErrorHandler,
    LrfCommutatorHandler,
    StdErrHandler,
)
from custodian.utils import tracked_lru_cache

R2SCAN_DIRNAME = "r2scan"
STATIC_DIRNAME = "static"
MACE_DIRNAME = "mace"
SECOND_RELAX_DIRNAME = "2nd_relax"
EXCLUDED_DIRS = {STATIC_DIRNAME, 
                 R2SCAN_DIRNAME, 
                 MACE_DIRNAME, 
                 SECOND_RELAX_DIRNAME}

# ----------------- Crawling ----------------- #

@contextmanager
def chdir(path: Path):
    """Temporarily change directory to path."""
    old = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def find_vasp_run_dirs(
    dft_root: Path,
    run_type: str = "relax",
    phases: Optional[List[str]] = None,
    orderings: Optional[List[str]] = None,
    tms: Optional[List[str]] = None,
) -> List[Path]:
    """
    Recursively find VASP run directories under dft_root.

    A directory is considered a run if it contains both INCAR and vasprun.xml.

    run_type:
      - "relax"  : collect only original relaxation directories under
                   <phase>/<ordering>/<tm> (exclude static/r2scan/mace sub-runs).
      - "static" : collect runs under paths containing a "static" directory name.
      - "r2scan" : collect runs under paths containing an "r2scan" directory name.
      - "2nd_relax" : collect runs under paths containing a "2nd_relax" directory name.
    phases/orderings/tms:
      Optional filters applied to inferred <phase>/<ordering>/<tm> from the
      first three path components. These filters are combined with run_type.
    """
    allowed = {"relax", "static", "r2scan", "2nd_relax"}
    if run_type not in allowed:
        raise ValueError(f"Invalid run_type={run_type!r}; choose one of {sorted(allowed)}")

    phase_set = set(phases) if phases else None
    ordering_set = set(orderings) if orderings else None
    tm_set = set(tms) if tms else None

    run_dirs: List[Path] = []
    for root, dirs, files in os.walk(dft_root):
        files_set = set(files)
        if "INCAR" in files_set and "vasprun.xml" in files_set:
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
                # Keep only original relax directories (phase/ordering/tm)
                if len(rel_parts) != 3:
                    continue
                if any(name in rel_parts for name in EXCLUDED_DIRS):
                    continue

            # Apply optional phase/ordering/tm filters.
            # For sub-runs (static/r2scan/2nd_relax), we still infer these from
            # the first three components: <phase>/<ordering>/<tm>/...
            if len(rel_parts) < 3:
                continue
            phase, ordering, tm = rel_parts[0], rel_parts[1], rel_parts[2]
            if phase_set is not None and phase not in phase_set:
                continue
            if ordering_set is not None and ordering not in ordering_set:
                continue
            if tm_set is not None and tm not in tm_set:
                continue

            run_dirs.append(run_dir)

    return sorted(run_dirs)


# ----------------- Parsing, Data Retriving, Error Handling ----------------- #

def parse_mace_metadata(run_dir: Path) -> Dict[str, Optional[str]]:
    """
    Parse MACE_metadata.txt if present in the run directory.
    Returns a dict of metadata entries.
    TODO This is a bit redundant
    """
    meta_file = run_dir / "MACE_metadata.txt"
    meta: Dict[str, Optional[str]] = {
        "meta_phase": None,
        "meta_ordering": None,
        "meta_tm": None,
        "meta_group_min": None,
        "meta_MACE_root": None,
        "meta_MACE_out_dir": None,
        "meta_MACE_energy_per_atom": None,
    }

    if not meta_file.is_file():
        return meta

    try:
        with meta_file.open() as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key == "phase":
                    meta["meta_phase"] = val
                elif key == "ordering":
                    meta["meta_ordering"] = val
                elif key == "tm":
                    meta["meta_tm"] = val
                elif key == "group_min":
                    meta["meta_group_min"] = val
                elif key == "MACE_root":
                    meta["meta_MACE_root"] = val
                elif key == "MACE_out_dir":
                    meta["meta_MACE_out_dir"] = val
                elif key == "energy_per_atom_eV":
                    meta["meta_MACE_energy_per_atom"] = val
    except Exception:
        pass

    return meta


def run_handlers_on_dir(run_dir: Path, apply_fixes: bool = False) -> Dict[str, object]:
    """
    Run a set of custodian VASP error handlers on run_dir.

    - By default (apply_fixes=False), this only *checks* and records which
      handlers triggered and what error labels they detected.
    - If apply_fixes=True, it additionally calls handler.correct(), which
      modifies input files according to custodian's logic. The corrections
      returned by correct() are recorded.

    Returns a dict with aggregated info:
      - n_custodian_errors: int
      - custodian_handlers: "Handler1;Handler2;..."
      - custodian_error_types: "Handler1:err_a;Handler1:err_b;Handler2:err_c;..."
      - custodian_corrections: JSON string of corrections (if apply_fixes)

    Note: Add MeshSymmetryErrorHandler if want to check mesh/symmetry issues.
    Otherwise, follow the basic rule of VASP runs - run one more SCF.
    Or don't build Custon VaspErrorHandler.
    """
    # Custom VaspErrorHandler that ignores the "bravais" error
    ve_subset = list(VaspErrorHandler.error_msgs.keys())
    if "bravais" in ve_subset:
        ve_subset.remove("bravais")  # ignore "Inconsistent Bravais lattice" errors

    vasp_err_handler = VaspErrorHandler(errors_subset_to_catch=ve_subset)

    # Handlers chosen to be meaningful for post-run analysis
    handlers = [
        vasp_err_handler,               # generic VASP errors in vasp.out
        UnconvergedErrorHandler(),        # convergence issues from vasprun.xml
        NonConvergingErrorHandler(),      # repeated non-convergence from OSZICAR
        AliasingErrorHandler(),           # FFT aliasing warnings
        PositiveEnergyErrorHandler(),     # positive absolute energy
        PotimErrorHandler(),              # too-large POTIM / huge dE
        LrfCommutatorHandler(),           # LRF_COMMUTATOR errors (stderr)
        StdErrHandler(),                  # common stderr errors (OOM, etc.)
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
            # Handler itself failed (e.g. OSZICAR missing) – note but don't treat as error
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

    # Aggregate info for the DataFrame row
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
    error_types_str = ";".join(error_type_list)

    corrections_str = json.dumps(corrections) if corrections else ""
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
        "custodian_error_types": error_types_str,
        "custodian_corrections": corrections_str,
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
    nsw_target_min: int = 300,
    nsw_increment: int = 200,
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
    """
    Submit a SLURM job in run_dir using: sbatch <job_script_name>.
    Just prints what it did; does not record anything in the data row.
    """
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
    else:
        if res.stdout:
            print(f"         {res.stdout.strip()}")


def summarize_run(
    run_dir: Path,
    dft_root: Path,
    apply_fixes: bool = False,
    submit_fixed: bool = False,
    job_script_name: str = "job.sh",
) -> Dict:
    """
    Summarize a single VASP run directory.

    Returns a dict with:
      - path info
      - phase / ordering / tm guesses from path
      - MACE metadata (if present)
      - convergence flags (if vasprun.xml parses)
      - energies, volume, composition (if vasprun.xml parses)
      - total magnetization (if OUTCAR present)
      - custodian error info (handlers + error labels + corrections)
      - job submission info (if submit_fixed & corrections applied)
      - status / error info if parsing fails

    IMPORTANT:
      - Even if vasprun.xml is missing or Vasprun parsing fails,
        we STILL run custodian handlers on the directory so that
        errors like "BZINTS: Tetrahedron method fails" are captured.
    """
    row: Dict = {}
    row["run_dir"] = str(run_dir)

    # Relative path pieces (e.g. O3_orderings / AJT_a_w_vac / Co)
    rel = run_dir.relative_to(dft_root)
    parts = list(rel.parts)
    row["relpath"] = str(rel)

    # Try to infer phase, ordering, tm from path
    row["phase"] = parts[0] if len(parts) >= 1 else None
    row["ordering"] = parts[1] if len(parts) >= 2 else None
    row["tm"] = parts[2] if len(parts) >= 3 else None

    # Attach MACE metadata if present
    meta = parse_mace_metadata(run_dir)
    row.update(meta)

    # Default flags
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

    # Custodian-related columns (defaults)
    row["n_custodian_errors"] = 0
    row["custodian_handlers"] = ""
    row["custodian_error_types"] = ""
    row["custodian_corrections"] = ""
    # My Catcher for ionic convergence.
    row["incar_nsw"] = None
    row["ionic_hit_nsw_limit"] = False
    row["nonconv_fix_applied"] = False
    row["nonconv_fix_reason"] = ""
    row["incar_nsw_new"] = None
    row["nonconv_fix_error"] = ""

    vasprun_path = run_dir / "vasprun.xml"
    outcar_path = run_dir / "OUTCAR"

    parse_ok = False  # track whether Vasprun parsed successfully

    # Try to parse vasprun.xml, but DO NOT return early if it fails
    if not vasprun_path.is_file():
        row["status"] = "NO_VASPRUN"
        row["error_msg"] = "vasprun.xml not found"
    else:
        try:
            vr = Vasprun(
                str(vasprun_path),
                parse_dos=False,
                parse_eigen=False,
            )
            parse_ok = True

            row["converged_electronic"] = bool(vr.converged_electronic)
            row["converged_ionic"] = bool(vr.converged_ionic)
            row["converged"] = bool(vr.converged_electronic and vr.converged_ionic)

            final_struct: Structure = vr.final_structure
            row["natoms"] = len(final_struct)
            row["volume_A3"] = float(final_struct.volume)
            row["lattice"] = final_struct.lattice.as_dict()
            row["formula"] = final_struct.composition.reduced_formula

            # Energy & e/atom
            row["energy_eV"] = float(vr.final_energy)
            if row["natoms"]:
                row["energy_per_atom_eV"] = row["energy_eV"] / row["natoms"]
            else:
                row["energy_per_atom_eV"] = None

            # Ionic steps count
            if vr.ionic_steps is not None:
                row["nsteps_ionic"] = len(vr.ionic_steps)

            # Fermi level
            try:
                row["efermi_eV"] = float(vr.efermi)
            except Exception:
                row["efermi_eV"] = None

        except Exception as exc:
            row["status"] = "PARSE_ERROR_VASPRUN"
            row["error_msg"] = f"{type(exc).__name__}: {exc}"

    # Parse OUTCAR for total magnetization (if present)
    # This can still work even if vasprun.xml is broken.
    if outcar_path.is_file():
        try:
            oc = Outcar(str(outcar_path))
            row["total_mag"] = getattr(oc, "total_mag", None)
        except Exception as exc:
            if row["error_msg"]:
                row["error_msg"] += f" | OUTCAR: {type(exc).__name__}: {exc}"
            else:
                row["error_msg"] = f"OUTCAR: {type(exc).__name__}: {exc}"

    # --- HERE IS THE IMPORTANT PART: always run custodian handlers ---
    handler_info = run_handlers_on_dir(run_dir, apply_fixes=apply_fixes)
    row.update(handler_info)

    # Check whether ionic loop appears to have hit the NSW cap.
    row["incar_nsw"] = read_incar_nsw(run_dir)
    if (
        parse_ok
        and row["converged_ionic"] is False
        and row["nsteps_ionic"] is not None
        and row["incar_nsw"] is not None
    ):
        row["ionic_hit_nsw_limit"] = bool(int(row["nsteps_ionic"]) >= int(row["incar_nsw"]))

    # Update status based on convergence + handler errors
    # Only override status if it is still "OK"
    if row["status"] == "OK":
        if parse_ok:
            if not row["converged"]:
                row["status"] = "NOT_CONVERGED"
            elif row["n_custodian_errors"] > 0:
                row["status"] = "HANDLER_ERROR"
        else:
            # Parsed nothing but no explicit earlier status set (rare)
            if row["n_custodian_errors"] > 0:
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

    # If requested, submit jobs for runs where we actually applied fixes
    had_handler_corrections = bool(row.get("custodian_corrections"))
    if apply_fixes and submit_fixed and (had_handler_corrections or row["nonconv_fix_applied"]):
        submit_job(run_dir, job_script_name)

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Collect DFT results from single-TM VASP runs into a CSV summary (with custodian error checks)."
    )
    parser.add_argument(
        "--dft-root",
        type=Path,
        required=True,
        help="Root directory of DFT runs (e.g. ./dft_singleTM).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to write the summary CSV (e.g. ./data/singleTM_dft_summary.csv).",
    )
    parser.add_argument(
        "--only-converged",
        action="store_true",
        help="If set, include only runs where both ionic and electronic steps converged.",
    )
    parser.add_argument(
        "--apply-fixes",
        action="store_true",
        help="If set, apply custodian handler corrections (correct()) to input files in problematic runs.",
    )
    parser.add_argument(
        "--submit-fixed",
        action="store_true",
        help="If set (and --apply-fixes is also set), submit jobs in runs where handler corrections were applied.",
    )
    parser.add_argument(
        "--job-script-name",
        type=str,
        default="job.sh",
        help="Name of the SLURM job script in each run dir to submit with sbatch (default: job.sh).",
    )
    parser.add_argument(
        "--run-type",
        choices=["relax", "static", "r2scan", "2nd_relax"],
        default="relax",
        help=(
            "Which run directories to collect. "
            "'relax' (default): original relax dirs only; "
            "'static': under static subdirs; "
            "'r2scan': under r2scan subdirs; "
            "'2nd_relax': under 2nd_relax subdirs."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        default=None,
        help=(
            "Optional phase directory names to include "
            "(e.g. O3_orderings P3_orderings)."
        ),
    )
    parser.add_argument(
        "--orderings",
        nargs="*",
        default=None,
        help=(
            "Optional ordering directory names to include "
            "(e.g. JMAS_a XC_a JMAS_c)."
        ),
    )
    parser.add_argument(
        "--tms",
        nargs="*",
        default=None,
        help=(
            "Optional TM directory names to include "
            "(e.g. Co Cr Fe Mn Ni Ti V)."
        ),
    )

    args = parser.parse_args()

    run_type = args.run_type

    if args.submit_fixed and not args.apply_fixes:
        raise SystemExit("--submit-fixed requires --apply-fixes (otherwise nothing is changed before submission).")

    dft_root: Path = args.dft_root
    if not dft_root.is_dir():
        raise SystemExit(f"dft_root is not a directory: {dft_root}")

    run_dirs = find_vasp_run_dirs(
        dft_root,
        run_type=run_type,
        phases=args.phases,
        orderings=args.orderings,
        tms=args.tms,
    )
    if not run_dirs:
        raise SystemExit(
            f"No VASP run directories (INCAR + vasprun.xml) found under {dft_root} "
            f"for run_type={run_type!r}"
        )

    print(f"Found {len(run_dirs)} VASP run directories under {dft_root}")

    rows: List[Dict] = []
    for rd in run_dirs:
        row = summarize_run(
            rd,
            dft_root=dft_root,
            apply_fixes=args.apply_fixes,
            submit_fixed=args.submit_fixed,
            job_script_name=args.job_script_name,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    if args.only_converged:
        before = len(df)
        df = df[df["converged"] == True].copy()
        after = len(df)
        print(f"Filtered to converged runs: {before} -> {after}")

    # Sort a bit for readability
    sort_cols = [c for c in ["meta_phase", "meta_ordering", "meta_tm", "phase", "ordering", "tm"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Ensure output directory exists
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote summary CSV to: {args.output_csv}")

    # Show a short preview
    preview_cols = [
        c
        for c in [
            "meta_phase",
            "meta_ordering",
            "meta_tm",
            "phase",
            "ordering",
            "tm",
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

    # Run-by-run summary
    print("\n=== Run-by-run summary ===")
    for _, row in df.iterrows():
        label = f"{row.get('phase', '')}/{row.get('ordering', '')}/{row.get('tm', '')}".strip("/")
        converged = bool(row.get("converged"))
        n_err = int(row.get("n_custodian_errors", 0))

        if converged and n_err == 0 and row.get("status") == "OK":
            epa = row.get("energy_per_atom_eV")
            epa_str = f"{epa:.6f}" if epa is not None else "NA"
            print(f"[OK] {label} @ {row['run_dir']}  E/atom = {epa_str} eV")
        else:
            print(f"[PROBLEM] {label} @ {row['run_dir']}")
            print(f"  status: {row['status']}")
            if not converged:
                print(f"  converged: {converged}")
            if n_err > 0:
                print(f"  handlers: {row['custodian_handlers']}")
                if row["custodian_error_types"]:
                    print(f"  error_types: {row['custodian_error_types']}")
            if row.get("job_submitted"):
                print(f"  job_submitted: {row['job_submitted']}  (rc={row['submit_returncode']})")
                if row.get("submit_stdout"):
                    print(f"    sbatch stdout: {row['submit_stdout']}")
                if row.get("submit_stderr"):
                    print(f"    sbatch stderr: {row['submit_stderr']}")

    # Final summary of clean runs
    ok_mask = (
        (df["converged"] == True) &
        (df["n_custodian_errors"] == 0) &
        (df["status"] == "OK")
    )
    n_ok = int(ok_mask.sum())
    n_total = len(df)
    print(f"\nTotal clean, converged runs (no handler errors): {n_ok} / {n_total}")

    print("\nDone.")


if __name__ == "__main__":
    main()
