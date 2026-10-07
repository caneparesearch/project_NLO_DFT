#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yun
"""
Collect transition-metal magnetic moments from single-TM VASP runs.

Crawling/filter logic is aligned with dft_collect_from_singleTM.py.

Expected relax tree:
  dft_singleTM/
    <phase>/<ordering>/<tm>/INCAR, vasprun.xml, OUTCAR, ...

Sub-runs (optional):
  dft_singleTM/<phase>/<ordering>/<tm>/static/...
  dft_singleTM/<phase>/<ordering>/<tm>/r2scan/...
  dft_singleTM/<phase>/<ordering>/<tm>/2nd_relax/...

Output is intentionally minimal:
  - phase
  - ordering
  - tm
  - tm_magmom_dict (e.g. {'Ni': [...]} or {'Co': [...]})
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

R2SCAN_DIRNAME = "r2scan"
STATIC_DIRNAME = "static"
MACE_DIRNAME = "mace"
SECOND_RELAX_DIRNAME = "2nd_relax"
EXCLUDED_DIRS = {STATIC_DIRNAME, R2SCAN_DIRNAME, MACE_DIRNAME, SECOND_RELAX_DIRNAME}


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
            # Keep only original relax directories (phase/ordering/tm)
            if len(rel_parts) != 3:
                continue
            if any(name in rel_parts for name in EXCLUDED_DIRS):
                continue

        # For sub-runs too, infer <phase>/<ordering>/<tm> from first 3 parts.
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


def load_structure_for_site_labels(run_dir: Path) -> tuple[Optional[Structure], str, Optional[str]]:
    """
    Load a structure with site order aligned to OUTCAR magnetic table as best effort.
    Priority: CONTCAR -> POSCAR -> vasprun.xml(final_structure).
    """
    for name in ("CONTCAR", "POSCAR"):
        p = run_dir / name
        if p.is_file():
            try:
                return Structure.from_file(str(p)), name, None
            except Exception as exc:
                return None, name, f"{type(exc).__name__}: {exc}"

    vr_path = run_dir / "vasprun.xml"
    if vr_path.is_file():
        try:
            vr = Vasprun(str(vr_path), parse_dos=False, parse_eigen=False)
            return vr.final_structure, "vasprun.xml(final_structure)", None
        except Exception as exc:
            return None, "vasprun.xml", f"{type(exc).__name__}: {exc}"

    return None, "none", "No CONTCAR/POSCAR/vasprun.xml available for site labels"


def summarize_run(run_dir: Path, dft_root: Path) -> Optional[Dict[str, object]]:
    row: Dict[str, object] = {}
    rel = run_dir.relative_to(dft_root)
    parts = list(rel.parts)
    row["phase"] = parts[0] if len(parts) >= 1 else None
    row["ordering"] = parts[1] if len(parts) >= 2 else None
    row["tm"] = parts[2] if len(parts) >= 3 else None

    outcar_path = run_dir / "OUTCAR"
    if not outcar_path.is_file():
        print(f"[skip] {run_dir}: OUTCAR not found")
        return None

    try:
        oc = Outcar(str(outcar_path))
    except Exception as exc:
        print(f"[skip] {run_dir}: failed to parse OUTCAR ({type(exc).__name__}: {exc})")
        return None

    mags = getattr(oc, "magnetization", None)
    if not mags:
        print(f"[skip] {run_dir}: no site magnetization table in OUTCAR")
        return None

    struct, src, struct_err = load_structure_for_site_labels(run_dir)
    if struct is None:
        print(f"[skip] {run_dir}: no structure for site labels from {src} ({struct_err})")
        return None

    n_use = min(len(struct), len(mags))
    if len(struct) != len(mags):
        print(
            f"[warn] {run_dir}: "
            f"Structure sites={len(struct)} but OUTCAR mag rows={len(mags)}; "
            f"using first {n_use} rows."
        )

    elem_to_vals: Dict[str, List[float]] = {}
    for i in range(n_use):
        site = struct[i]
        mag_row = mags[i]
        try:
            is_tm = bool(site.specie.is_transition_metal)
        except Exception:
            is_tm = False
        if not is_tm:
            continue

        element = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
        m = mag_row.get("tot", None) if isinstance(mag_row, dict) else None
        if m is None:
            continue
        elem_to_vals.setdefault(element, []).append(float(m))

    if not elem_to_vals:
        print(f"[skip] {run_dir}: no transition-metal site magnetic moments parsed")
        return None

    ordered = {k: elem_to_vals[k] for k in sorted(elem_to_vals.keys())}
    row["tm_magmom_dict"] = ordered
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Collect transition-metal magnetic moments from single-TM OUTCAR files."
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
        help="Path to write the summary CSV.",
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
        help="Optional phase directory names to include (e.g. O3_orderings P3_orderings).",
    )
    parser.add_argument(
        "--orderings",
        nargs="*",
        default=None,
        help="Optional ordering directory names to include.",
    )
    parser.add_argument(
        "--tms",
        nargs="*",
        default=None,
        help="Optional TM directory names to include (e.g. Co Cr Fe Mn Ni Ti V).",
    )

    args = parser.parse_args()
    dft_root: Path = args.dft_root
    if not dft_root.is_dir():
        raise SystemExit(f"dft_root is not a directory: {dft_root}")

    run_dirs = find_vasp_run_dirs(
        dft_root=dft_root,
        run_type=args.run_type,
        phases=args.phases,
        orderings=args.orderings,
        tms=args.tms,
    )
    if not run_dirs:
        raise SystemExit(
            f"No VASP run directories (INCAR + vasprun.xml) found under {dft_root} "
            f"for run_type={args.run_type!r}"
        )

    print(f"Found {len(run_dirs)} VASP run directories under {dft_root}")
    rows = [r for r in (summarize_run(rd, dft_root=dft_root) for rd in run_dirs) if r is not None]
    if not rows:
        raise SystemExit("No valid magnetic-moment rows collected (all runs were skipped).")

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["phase", "ordering", "tm"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote magmom summary CSV to: {args.output_csv}")

    preview_cols = [
        c
        for c in [
            "phase",
            "ordering",
            "tm",
            "tm_magmom_dict",
        ]
        if c in df.columns
    ]
    print("\nPreview:")
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
