#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yun
"""
Collect transition-metal magnetic moments from double-TM VASP runs.

Crawling/filter logic is aligned with dft_collect_from_doubleTM.py.

Expected relax tree:
  dft_doubleTM/
    <phase>/<ordering>/<tm_pair>/elec_*/INCAR, vasprun.xml, OUTCAR, ...

Sub-runs (optional):
  dft_doubleTM/<phase>/<ordering>/<tm_pair>/elec_*/static/...
  dft_doubleTM/<phase>/<ordering>/<tm_pair>/elec_*/r2scan/...
  dft_doubleTM/<phase>/<ordering>/<tm_pair>/elec_*/2nd_relax/...
Output is intentionally minimal:
  - phase
  - ordering
  - tm_pair
  - tm_ordering (elec_0 / elec_1 / ...)
  - tm_magmom_dict (e.g. {'Ni': [...], 'Co': [...]})
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
    row["tm_pair"] = parts[2] if len(parts) >= 3 else None
    row["tm_ordering"] = parts[3] if len(parts) >= 4 else None

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
        m = float(m)

        elem_to_vals.setdefault(element, []).append(m)

    if not elem_to_vals:
        print(f"[skip] {run_dir}: no transition-metal site magnetic moments parsed")
        return None

    ordered = {k: elem_to_vals[k] for k in sorted(elem_to_vals.keys())}
    row["tm_magmom_dict"] = ordered
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Collect transition-metal magnetic moments from double-TM OUTCAR files."
    )
    parser.add_argument("--dft-root", type=Path, required=True, help="Root directory (e.g. ./dft_doubleTM).")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path.")
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
    rows = [r for r in (summarize_run(rd, dft_root=dft_root) for rd in run_dirs) if r is not None]
    if not rows:
        raise SystemExit("No valid magnetic-moment rows collected (all runs were skipped).")
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["phase", "ordering", "tm_pair", "tm_ordering"] if c in df.columns]
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
            "tm_pair",
            "tm_ordering",
            "tm_magmom_dict",
        ]
        if c in df.columns
    ]
    print("\nPreview:")
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
