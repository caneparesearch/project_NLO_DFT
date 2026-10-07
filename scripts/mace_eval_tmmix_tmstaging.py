#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLANATION:
Evaluate TM-staging translation groups with a pretrained MACE model (single-point only).

Expected tree (under --root):
  O1_orderings|O3_orderings|P3_orderings/
    <ordering_name>/
      <TM1_TM2>/
        elec_*/
          TM_staging/
            group_*/POSCAR

For each elec_* directory:
  1) evaluate all group_* POSCAR energies with MACE static calculation,
  2) select lowest-energy group,
  3) copy that group's POSCAR -> elec_*/POSCAR_TMstaging,
  4) write per-elec JSON summary.

During crawl, prints quick counts:
  - # of elec_* under each (phase/ordering/TM pair)
  - # of group_* under each elec_*

USAGE:
# quick crawl only (no MACE run)
python src/mace_eval_tmmix_tmstaging.py --root ordering_doubleTM --dry-run

# full evaluation
python src/mace_eval_tmmix_tmstaging.py \
  --root ordering_doubleTM \
  --model /path/to/your_pretrained.model \
  --device auto
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional


PHASE_DIRS = ("O1_orderings", "O3_orderings", "P3_orderings")


@dataclass
class GroupTask:
    group_dir: Path
    poscar_path: Path


@dataclass
class ElecTask:
    phase: str
    ordering: str
    tm_pair: str
    elec_dir: Path
    groups: List[GroupTask]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static MACE evaluation for TM-mix translation groups under ordering_doubleTM."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("../ordering_doubleTM"),
        help="Root containing O1_orderings/O3_orderings/P3_orderings.",
    )
    p.add_argument(
        "--model", 
        type=Path, 
        default=Path("../data/mace-omat-0-medium.model"), 
        help="Path to pretrained MACE model."
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: auto|cpu|cuda. Default: auto.",
    )
    p.add_argument(
        "--summary-name",
        default="TM_staging_mace_summary.json",
        help="Per-elec JSON summary filename.",
    )
    p.add_argument(
        "--best-poscar-name",
        default="POSCAR_TMstaging",
        help="Filename for lowest-energy POSCAR in each elec_* directory.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if summary and POSCAR_TMstaging already exist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only crawl and print counts; do not run MACE.",
    )
    return p.parse_args()


def auto_device(user_device: str) -> str:
    d = user_device.strip().lower()
    if d != "auto":
        return d
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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


def find_group_poscar(group_dir: Path) -> Optional[Path]:
    p0 = group_dir / "POSCAR"
    if p0.exists():
        return p0
    cands = sorted(group_dir.glob("POSCAR*"))
    return cands[0] if cands else None


def build_tasks(root: Path) -> List[ElecTask]:
    """
    Crawl the directory and get the list of dataclass, to make MACE run easier.
    Return List of ElecTask, will be used in the main function.
    """
    tasks: List[ElecTask] = []

    for phase in PHASE_DIRS:
        phase_dir = root / phase
        if not phase_dir.exists():
            continue

        for ordering_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
            ordering = ordering_dir.name
            for tm_pair_dir in sorted([p for p in ordering_dir.iterdir() if p.is_dir()]):
                tm_pair = tm_pair_dir.name
                elec_dirs = [p for p in tm_pair_dir.glob("elec_*") if p.is_dir()]
                elec_dirs = _sorted_dirs_by_suffix(elec_dirs, "elec_")

                print(
                    f"[COUNT] {phase}/{ordering}/{tm_pair}: elec_dirs={len(elec_dirs)}",
                    flush=True,
                )

                for elec_dir in elec_dirs:
                    # Primary layout: elec_*/TM_staging/group_*
                    tm_staging_dir = elec_dir / "TM_staging"
                    if tm_staging_dir.is_dir():
                        group_dirs = [p for p in tm_staging_dir.glob("group_*") if p.is_dir()]
                    else:
                        # Backward-compatible fallback: elec_*/group_*
                        group_dirs = [p for p in elec_dir.glob("group_*") if p.is_dir()]
                    group_dirs = _sorted_dirs_by_suffix(group_dirs, "group_")
                    print(
                        f"[COUNT]   {elec_dir.name}: groups={len(group_dirs)}",
                        flush=True,
                    )

                    groups: List[GroupTask] = []
                    for gdir in group_dirs:
                        poscar = find_group_poscar(gdir)
                        if poscar is None:
                            print(
                                f"[WARN] Missing POSCAR in {gdir}",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        groups.append(GroupTask(group_dir=gdir, poscar_path=poscar))

                    tasks.append(
                        ElecTask(
                            phase=phase,
                            ordering=ordering,
                            tm_pair=tm_pair,
                            elec_dir=elec_dir,
                            groups=groups,
                        )
                    )

    return tasks


def _swap_symbols_for_energy(atoms: Any, source: str = "K", target: str = "Na") -> List[int]:
    """
    Temporarily replace one species with another on the same sites.
    Returns the modified atom indices so the caller can restore them after evaluation.
    """
    swapped_indices: List[int] = []
    symbols = atoms.get_chemical_symbols()
    for i, symbol in enumerate(symbols):
        if symbol == source:
            atoms[i].symbol = target
            swapped_indices.append(i)
    return swapped_indices


def _restore_swapped_symbols(
    atoms: Any,
    swapped_indices: List[int],
    restored_symbol: str = "K",
) -> None:
    for i in swapped_indices:
        atoms[i].symbol = restored_symbol


def evaluate_elec_task(
    task: ElecTask,
    calc: Any,
    model_path: Path,
    summary_name: str,
    best_poscar_name: str,
    overwrite: bool,
) -> None:
    """
    Evaluate one ElecTask.
    ElecTask corresponds to one elec_* directory, 
    which means one TM-mix ordering in given TM-pair in certain ordering.
    """
    summary_path = task.elec_dir / summary_name
    best_poscar_path = task.elec_dir / best_poscar_name

    if not overwrite and summary_path.exists() and best_poscar_path.exists():
        print(f"[SKIP] {task.elec_dir} (existing outputs)", flush=True)
        return

    t0 = time.time()
    records = []

    # Go over GroupTask
    for gt in task.groups:
        row = {
            "group": gt.group_dir.name,
            "group_dir": str(gt.group_dir),
            "poscar": str(gt.poscar_path),
            "natoms": None,
            "energy_eV": None,
            "energy_per_atom_eV": None,
            "status": "OK",
        }

        try:
            from ase.io import read

            atoms = read(gt.poscar_path)
            atoms.pbc = True
            atoms.calc = calc
            swapped_indices = _swap_symbols_for_energy(atoms, source="K", target="Na")
            try:
                e = float(atoms.get_potential_energy())
            finally:
                _restore_swapped_symbols(atoms, swapped_indices, restored_symbol="K")
            n = len(atoms)

            row["natoms"] = int(n)
            row["energy_eV"] = e
            row["energy_per_atom_eV"] = e / float(n) if n > 0 else None
        except Exception as exc:
            row["status"] = f"FAIL: {type(exc).__name__}: {exc}"

        records.append(row)

    ok_rows = [r for r in records if r["status"] == "OK" and r["energy_eV"] is not None]

    best = None
    if ok_rows:
        best = min(ok_rows, key=lambda r: float(r["energy_eV"]))
        shutil.copy2(best["poscar"], best_poscar_path)

    payload = {
        "phase": task.phase,
        "ordering": task.ordering,
        "tm_pair": task.tm_pair,
        "elec_dir": str(task.elec_dir),
        "n_groups_discovered": len(task.groups),
        "n_groups_evaluated_ok": len(ok_rows),
        "model": str(model_path),
        "best_group": best["group"] if best else None,
        "best_energy_eV": best["energy_eV"] if best else None,
        "best_energy_per_atom_eV": best["energy_per_atom_eV"] if best else None,
        "best_poscar": str(best_poscar_path) if best else None,
        "runtime_s": time.time() - t0,
        "groups": records,
    }

    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    if best is None:
        print(f"[DONE] {task.elec_dir} no successful group evaluations", flush=True)
    else:
        print(
            f"[DONE] {task.elec_dir} best={best['group']} E={best['energy_eV']:.6f} eV",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    print(f"[INFO] root={root}", flush=True)
    tasks = build_tasks(root)
    print(f"[INFO] discovered elec_* directories: {len(tasks)}", flush=True)

    if args.dry_run:
        print("[INFO] dry-run requested; skipping MACE evaluation.", flush=True)
        return

    if args.model is None:
        raise ValueError("--model is required unless --dry-run is used.")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = auto_device(args.device)
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] model={args.model}", flush=True)

    from mace.calculators.mace import MACECalculator

    calc = MACECalculator(model_path=str(args.model), device=device)

    for task in tasks:
        evaluate_elec_task(
            task=task,
            calc=calc,
            model_path=args.model,
            summary_name=args.summary_name,
            best_poscar_name=args.best_poscar_name,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
