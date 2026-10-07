#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect single-TM CONTCAR structures and export one key-value JSON file.

Key format:
  phase-ordering-tm

Value format:
  pymatgen Structure.as_dict() for the parsed structure.

Crawling/filter logic follows dft_collect_from_singleTM.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pymatgen.core.structure import Structure

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
    """Find VASP run directories containing INCAR + vasprun.xml with singleTM layout."""
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


def load_structure(
    run_dir: Path,
    primary_name: str = "CONTCAR",
    fallback_names: Optional[List[str]] = None,
) -> Tuple[Optional[Structure], Optional[str], Optional[str]]:
    """Load structure from run_dir using primary file then optional fallbacks."""
    candidates = [primary_name] + (fallback_names or [])
    errors: List[str] = []

    for name in candidates:
        p = run_dir / name
        if not p.is_file():
            continue
        try:
            return Structure.from_file(str(p)), name, None
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if errors:
        return None, None, "; ".join(errors)
    return None, None, f"No structure file found in {run_dir} (tried: {', '.join(candidates)})"


def make_key_from_run(run_dir: Path, dft_root: Path) -> Optional[str]:
    parts = run_dir.relative_to(dft_root).parts
    if len(parts) < 3:
        return None
    phase, ordering, tm = parts[0], parts[1], parts[2]
    return f"{phase}-{ordering}-{tm}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect single-TM structures and write one key-value JSON mapping."
    )
    parser.add_argument(
        "--dft-root",
        type=Path,
        required=True,
        help="Root directory of DFT runs (e.g. ./dft_singleTM).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON path for {phase-ordering-tm: structure_dict}.",
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
    parser.add_argument(
        "--structure-file",
        default="CONTCAR",
        help="Primary structure file name to convert (default: CONTCAR).",
    )
    parser.add_argument(
        "--fallback-files",
        nargs="*",
        default=[],
        help="Optional fallback file names if primary file is missing/unreadable.",
    )
    args = parser.parse_args()

    dft_root = args.dft_root
    output_json = args.output_json

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
    payload: Dict[str, Dict] = {}
    key_source: Dict[str, str] = {}

    for run_dir in run_dirs:
        key = make_key_from_run(run_dir, dft_root=dft_root)
        if key is None:
            print(f"[skip] {run_dir}: cannot infer key from path")
            continue

        structure, used_file, err = load_structure(
            run_dir=run_dir,
            primary_name=args.structure_file,
            fallback_names=args.fallback_files,
        )
        if structure is None or used_file is None:
            print(f"[skip] {run_dir}: {err}")
            continue

        if key in payload:
            raise SystemExit(
                "Duplicate key detected. "
                f"key={key!r}, previous_run={key_source[key]!r}, new_run={str(run_dir)!r}. "
                "Please narrow filters or run_type to avoid collisions."
            )

        payload[key] = structure.as_dict()
        key_source[key] = str(run_dir)
        print(f"[ok] {key} ({used_file})")

    if not payload:
        raise SystemExit("No structures were exported (all runs skipped).")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {len(payload)} structures to: {output_json}")
    print("Preview keys:")
    for k in list(payload.keys())[:10]:
        print(f"  {k}")


if __name__ == "__main__":
    main()
