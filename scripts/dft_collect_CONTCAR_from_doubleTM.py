#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect double-TM CONTCAR structures and export one key-value JSON file.

Key format:
  phase-ordering-tm_pair-elec

Value format:
  pymatgen Structure.as_dict() for the parsed structure.

Crawling/filter logic follows dft_collect_from_doubleTM.py.
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
    """Find VASP run directories containing INCAR + vasprun.xml with doubleTM layout."""
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
    if len(parts) < 4:
        return None
    phase, ordering, tm_pair, elec = parts[0], parts[1], parts[2], parts[3]
    return f"{phase}-{ordering}-{tm_pair}-{elec}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect double-TM structures and write one key-value JSON mapping."
    )
    parser.add_argument("--dft-root", type=Path, required=True, help="Root directory (e.g. ./dft_doubleTM).")
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON path for {phase-ordering-tm_pair-elec: structure_dict}.",
    )
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
