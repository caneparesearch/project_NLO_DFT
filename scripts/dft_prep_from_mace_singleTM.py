#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yun
"""
Select lowest-energy MACE-relaxed staging per ordering 
and build VASP DFT inputs (MPRelaxSet, but little tweaked).

Assumptions
-----------
- Should have directory structure like this:
      <mace_root>/
        summary.csv
        O3_orderings/ or O3/
        P3_orderings/ or P3/
- Each relaxed calculation should contains a file have string 
  "final" and "POSCAR" in its name (e.g. final.POSCAR).

- summary.csv has at least the columns
  phase, ordering, group, tm, energy_per_atom_eV, out_dir, status

This script will, for each (phase, ordering, tm) triple:
  1) find the group with the lowest energy_per_atom_eV among all groups;
  2) read the corresponding final POSCAR;
  3) write an MPRelaxSet-style VASP input set (POSCAR/INCAR/POTCAR)
     in  <out_root>/<phase>/<ordering>/<tm>/;
  4) write a SLURM job script (job.sh) in the same directory;

The VASP inputs:
  - use default functional for POTCARs as configured in ~/.pmgrc.yaml
    (set PMG_DEFAULT_FUNCTIONAL=PBE_54 there if you want PBE_54);
  - are spin-polarized with ferromagnetic initialization;
  - include DFT-D3(BJ) via IVDW = 12.

Usage
-----
python prepare_dft_from_mace_minima.py \
    --summary /path/to/mace_root \
    --out-root /path/to/dft_inputs \
    --phases O3_orderings P3_orderings
    --submit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import re
import subprocess
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet


def load_and_filter_summary(
    csv_path: Path,
    phases: List[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load summary.csv, filter to OK rows and optional phases, and pick energy column."""

    df = pd.read_csv(csv_path)

    if "status" in df.columns:
        df_ok = df[df["status"] == "OK"].copy()
    else:
        df_ok = df.copy()

    if phases is not None and len(phases) > 0 and "phase" in df_ok.columns:
        df_ok = df_ok[df_ok["phase"].isin(phases)].copy()

    if "energy_per_atom_eV" in df_ok.columns:
        energy_col = "energy_per_atom_eV"
    else:
        raise KeyError(
            "'energy_per_atom_eV' not found in in summary.csv."
        )

    # Drop rows with NaN energies
    df_ok = df_ok.dropna(subset=[energy_col]).copy()

    return df_ok, energy_col


def select_minima_per_group(df_ok: pd.DataFrame, energy_col: str) -> pd.DataFrame:
    """
    Using pandas groupby to select the row index with minimal energy
    in each (phase, ordering, tm) group.
    """

    for col in ["phase", "ordering", "tm"]:
        if col not in df_ok.columns:
            raise KeyError(f"Column '{col}' not found in summary.csv.")

    # Index of minima per group
    idx_min = (
        df_ok.groupby(["phase", "ordering", "tm"])[energy_col]
        .idxmin()
        .dropna()
        .astype(int)
    )

    df_min = df_ok.loc[idx_min].copy().reset_index(drop=True)
    return df_min


def find_relaxed_poscar(out_dir_str: str) -> Path:
    """
    Find a relaxed POSCAR in the MACE output directory.

    Priority:
      1) exact match 'final.POSCAR'
      2) any file whose name contains both 'final' and 'POSCAR'
         (case-insensitive).
    """
    mace_out_dir = Path(out_dir_str)

    exact = mace_out_dir / "final.POSCAR"
    if exact.is_file():
        return exact

    for p in mace_out_dir.iterdir():
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if "final" in name_lower and "poscar" in name_lower:
            return p

    raise FileNotFoundError(
        f"Could not find relaxed POSCAR in {mace_out_dir} "
        "(no 'final.POSCAR' and no file with both 'final' and 'POSCAR' in name)."
    )


def make_job_name(phase: str, ordering: str, tm: str) -> str:
    """
    Make a SLURM-safe, reasonably short job name like 'vasp_O3_AJT_a_Co'.
    Non-alphanumeric characters are replaced by '_'.
    """
    base = f"vasp_{phase}_{ordering}_{tm}"
    return re.sub(r"[^A-Za-z0-9_]", "_", base)


def write_slurm_script(
    phase: str,
    ordering: str,
    tm: str,
    dft_dir: Path,
    submit: bool = False,
    dry_run: bool = False,
) -> Path:

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

    if dry_run:
        print(f"[DRY] Would write SLURM script at: {script_path}")
    else:
        script_path.write_text(script)

    if submit and not dry_run:
        try:
            # Change to the directory that contains the script, then run sbatch
            script_basename = script_path.name  # e.g. "job.sh" or "run_vasp.slurm"
            cmd = f"cd {dft_dir} && sbatch {script_basename}"
            res = subprocess.run(
                cmd,
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


def write_vasp_inputs_for_row(
    row: pd.Series,
    mace_root: Path,
    out_root: Path,
    energy_col: str,
    dry_run: bool = False,
    submit: bool = False,
) -> Path:
    """
    For one minima row, write MPRelaxSet-based VASP inputs into
    <out_root>/<phase>/<ordering>/<tm>/, plus a SLURM script.
    Optionally submit the job with sbatch.

    MACE output directory is reconstructed as:
      <mace_root>/<phase>/<ordering>/<group>/<tm>/
    """

    phase = str(row["phase"]).strip()
    ordering = str(row["ordering"]).strip()
    tm = str(row["tm"]).strip()
    group = str(row["group"]).strip()

    # Reconstruct where the MACE run lives:
    mace_out_dir = mace_root / phase / ordering / group / tm
    poscar_path = find_relaxed_poscar(mace_out_dir)

    # DFT directory: <out_root>/<phase>/<ordering>/<tm>/
    dft_dir = out_root / phase / ordering / tm
    dft_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[DRY] DFT dir: {dft_dir}")
        print(f"[DRY] Using MACE POSCAR from: {poscar_path}")
        return dft_dir

    # Load structure from MACE-relaxed POSCAR
    struct = Structure.from_file(str(poscar_path))

    # MPRelaxSet-style inputs with user overrides.
    # Previous settings kept for reference:
    # "ENCUT": 520, "IVDW": 12, "ISPIN": 2
    user_incar = {
        "ENCUT": 680,
        "EDIFF": 1e-6 * len(struct),  # 1e-6 eV/atom -> total EDIFF
        "EDIFFG": -0.02,
        "KSPACING": 0.3,
        "IVDW": 12,  # Grimme D3-BJ
        "ISPIN": 2,  # spin-polarized (FM init)
    }

    # Ni-only POTCAR override (easy to disable by commenting this block):
    user_potcar_settings = None
    # if tm == "Ni":
    #     user_potcar_settings = {"Ni": "Ni_sv_GW"}

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

    # Keep a record of the MACE-relaxed structure
    mace_copy_path = dft_dir / "POSCAR_MACE_relaxed"
    if not mace_copy_path.exists():
        struct.to(filename=str(mace_copy_path))

    # Minimal metadata for traceability
    meta_path = dft_dir / "MACE_metadata.txt"
    with meta_path.open("w") as f:
        f.write(f"phase: {phase}\n")
        f.write(f"ordering: {ordering}\n")
        f.write(f"tm: {tm}\n")
        f.write(f"group_min: {group}\n")
        f.write(f"MACE_root: {mace_root}\n")
        f.write(f"MACE_out_dir: {mace_out_dir}\n")
        if energy_col in row:
            f.write(f"{energy_col}: {row[energy_col]:.8f}\n")

    # Write SLURM job script (and optionally submit)
    write_slurm_script(
        phase=phase,
        ordering=ordering,
        tm=tm,
        dft_dir=dft_dir,
        submit=submit,
        dry_run=dry_run,
    )

    return dft_dir


# ----------------- CLI entrypoint ----------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pick lowest-energy MACE group per (phase, ordering, tm), reconstruct "
            "MACE output dirs from a root, make DFT inputs, and optionally submit "
            "SLURM jobs."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help=(
            "Path to ordering_staging_output ROOT directory (containing summary.csv "
            "and MACE outputs, e.g. ../ordering_staging_output). "
            "The script will read <summary>/summary.csv."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Root directory where DFT input trees will be written.",
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        default=None,
        help="Optional subset of phases to include, e.g. O3_orderings P3_orderings "
             "or O3 P3. Leave empty to include all.",
    )
    parser.add_argument(
        "--tms",
        nargs="*",
        default=None,
        help=(
            "Optional subset of TM symbols to include (e.g. Co Ni Mn). "
            "If omitted, use all TMs present in summary.csv."
        ),
    )
    parser.add_argument(
        "--orderings",
        nargs="*",
        default=None,
        help=(
            "Optional subset of ordering labels to include "
            "(e.g. full empty AJT_a_w_vac). If omitted, use all."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, do not write any files or submit jobs.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="After creating inputs + SLURM scripts, submit all jobs with sbatch.",
    )

    args = parser.parse_args()

    # Interpret --summary as a ROOT directory
    if args.summary.is_dir():
        mace_root = args.summary
        summary_csv = mace_root / "summary.csv"
    else:
        # If user accidentally passes the CSV directly, still handle it:
        summary_csv = args.summary
        mace_root = args.summary.parent

    if not summary_csv.is_file():
        raise SystemExit(f"summary.csv not found at: {summary_csv}")

    df_ok, energy_col = load_and_filter_summary(
        csv_path=summary_csv,
        phases=args.phases,
    )

    if df_ok.empty:
        raise SystemExit("No usable rows after filtering summary.csv.")

    df_min = select_minima_per_group(df_ok, energy_col=energy_col)

    # Optional extra filters on TM and ordering
    if args.tms is not None and len(args.tms) > 0:
        df_min = df_min[df_min["tm"].isin(args.tms)].copy()

    if args.orderings is not None and len(args.orderings) > 0:
        df_min = df_min[df_min["ordering"].isin(args.orderings)].copy()

    if df_min.empty:
        raise SystemExit("No minima left after applying TM/ordering filters.")

    print(f"Found {len(df_min)} minima (phase, ordering, tm) to set up.")
    print("First few minima:")
    print(
        df_min[["phase", "ordering", "tm", "group", energy_col]]
        .head()
        .to_string(index=False)
    )

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # For bookkeeping: add reconstructed MACE directory to df_min
    def _mace_dir_for_row(r):
        return str(
            mace_root
            / str(r["phase"]).strip()
            / str(r["ordering"]).strip()
            / str(r["group"]).strip()
            / str(r["tm"]).strip()
        )

    df_min["mace_out_dir"] = df_min.apply(_mace_dir_for_row, axis=1)

    minima_csv = out_root / "minima_summary.csv"
    df_min.to_csv(minima_csv, index=False)
    print(f"\n[INFO] Wrote minima summary: {minima_csv}")

    # Write VASP inputs + SLURM job scripts (and optionally submit)
    for _, row in df_min.iterrows():
        key = (row["phase"], row["ordering"], row["tm"])
        dft_dir = write_vasp_inputs_for_row(
            row=row,
            mace_root=mace_root,
            out_root=out_root,
            energy_col=energy_col,
            dry_run=args.dry_run,
            submit=args.submit,
        )
        print(f"[OK] {key} -> {dft_dir}")


if __name__ == "__main__":
    main()
