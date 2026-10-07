#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Na-vacancy staging translations from POSCAR_TMstaging in each elec_* directory.

For each elec_*:
  - input:  elec_*/POSCAR_TMstaging
  - output: elec_*/Na_staging/trans_*/POSCAR

Rules:
  - If input POSCAR_TMstaging is missing: skip.
  - If no K exists in any Na/K layer: write trans_0 with the input structure.
  - Otherwise:
      1) identify Na/K layers by z,
      2) pick one Na-K mixed layer as template,
      3) find translational vectors template -> each other Na/K layer,
      4) enumerate translation combos and apply template Na/K pattern,
      5) write every translated structure under trans_* (no symmetry grouping).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar


PHASE_DIRS = ("O1_orderings", "O3_orderings", "P3_orderings")
ALKALI = {"Na", "K"}

Z_LAYER_TOL = 0.02
XY_DECIMALS = 2
XY_MATCH_BUFFER = 0.05


@dataclass
class ElecPath:
    phase: str
    ordering: str
    tm_pair: str
    elec_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Na_staging groups from POSCAR_TMstaging under ordering_doubleTM."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("../ordering_doubleTM"),
        help="Root containing O1_orderings/O3_orderings/P3_orderings.",
    )
    p.add_argument(
        "--input-poscar-name",
        default="POSCAR_TMstaging",
        help="Input POSCAR filename under each elec_*.",
    )
    p.add_argument(
        "--na-staging-dir-name",
        default="Na_staging",
        help="Output directory name under each elec_*.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Na_staging outputs.",
    )
    return p.parse_args()


def frac01(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def key2(xy: np.ndarray, decimals: int = XY_DECIMALS) -> Tuple[float, float]:
    v = frac01(np.asarray(xy))[:2]
    v = np.mod(np.round(v, decimals=decimals), 1.0)
    return tuple(v.astype(float))


def _sorted_dirs_by_suffix(paths: List[Path], prefix: str) -> List[Path]:
    def _key(p: Path):
        tail = p.name[len(prefix) :] if p.name.startswith(prefix) else p.name
        try:
            return (0, int(tail))
        except Exception:
            return (1, str(tail))

    return sorted(paths, key=_key)


def collect_elec_dirs(root: Path) -> List[ElecPath]:
    out: List[ElecPath] = []
    for phase in PHASE_DIRS:
        phase_dir = root / phase
        if not phase_dir.exists():
            continue
        for ordering_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
            for tm_pair_dir in sorted([p for p in ordering_dir.iterdir() if p.is_dir()]):
                elec_dirs = [p for p in tm_pair_dir.glob("elec_*") if p.is_dir()]
                elec_dirs = _sorted_dirs_by_suffix(elec_dirs, "elec_")
                for elec_dir in elec_dirs:
                    out.append(
                        ElecPath(
                            phase=phase,
                            ordering=ordering_dir.name,
                            tm_pair=tm_pair_dir.name,
                            elec_dir=elec_dir,
                        )
                    )
    return out


def split_layers_by_frac_z(structure: Structure, z_tol: float = Z_LAYER_TOL) -> List[List[int]]:
    alkali_idx = [i for i, site in enumerate(structure) if site.specie.symbol in ALKALI]
    if not alkali_idx:
        return []

    zs = sorted(
        [(float(structure[i].frac_coords[2] % 1.0), i) for i in alkali_idx],
        key=lambda t: t[0],
    )
    layers: List[List[int]] = []
    cur = [zs[0][1]]
    z0 = zs[0][0]

    for z, ii in zs[1:]:
        if abs(z - z0) <= z_tol:
            cur.append(ii)
        else:
            layers.append(cur)
            cur = [ii]
            z0 = z
    layers.append(cur)

    if len(layers) > 1 and abs((zs[0][0] + 1.0) - zs[-1][0]) <= z_tol:
        layers[0] = layers[-1] + layers[0]
        layers = layers[:-1]

    return layers


def layer_xy_sets(
    struct: Structure, indices: Sequence[int]
) -> Tuple[np.ndarray, Dict[Tuple[float, float], List[int]]]:
    xys = []
    mapping: Dict[Tuple[float, float], List[int]] = {}
    for i in indices:
        xy = frac01(struct[i].frac_coords)[:2]
        xys.append(xy)
        mapping.setdefault(key2(xy), []).append(i)
    return np.asarray(xys), mapping


def translations_that_map(P: np.ndarray, Q: np.ndarray) -> List[np.ndarray]:
    if len(P) != len(Q):
        raise RuntimeError("Layer site counts differ; cannot map by translation.")
    if len(P) == 0:
        return []

    p0 = P[0]
    vectors: List[np.ndarray] = []
    seen = set()

    def _can_match(shifted: np.ndarray, target: np.ndarray, buf: float) -> bool:
        used = [False] * len(shifted)
        for p in shifted:
            best_j = -1
            best_norm = float("inf")
            for j, q in enumerate(target):
                if used[j]:
                    continue
                d = np.abs(p - q)
                d = np.minimum(d, 1.0 - d)
                if d[0] <= buf and d[1] <= buf:
                    dn = float(np.linalg.norm(d))
                    if dn < best_norm:
                        best_norm = dn
                        best_j = j
            if best_j < 0:
                return False
            used[best_j] = True
        return True

    for q in Q:
        v = frac01(q - p0)
        shifted = frac01(P + v)
        if _can_match(shifted, Q, XY_MATCH_BUFFER):
            k = key2(v)
            if k not in seen:
                seen.add(k)
                vectors.append(v)

    return vectors


def apply_pattern_to_target_layer(
    struct: Structure,
    src_xy: np.ndarray,
    pattern: Dict[Tuple[float, float], str],
    target_xy_map: Dict[Tuple[float, float], List[int]],
    v2d: np.ndarray,
) -> None:
    target_entries: List[Tuple[int, np.ndarray]] = []
    for q_key, idxs in target_xy_map.items():
        if len(idxs) != 1:
            raise RuntimeError(f"Ambiguous mapping at key={q_key}: {idxs}")
        target_entries.append((idxs[0], np.array(q_key, dtype=float)))

    used = [False] * len(target_entries)
    assignments: List[Tuple[int, str]] = []

    for xy in src_xy:
        sp = pattern[key2(xy)]
        shifted = frac01(xy + v2d)
        best_j = -1
        best_norm = float("inf")

        for j, (_, qxy) in enumerate(target_entries):
            if used[j]:
                continue
            d = np.abs(shifted - qxy)
            d = np.minimum(d, 1.0 - d)
            if d[0] <= XY_MATCH_BUFFER and d[1] <= XY_MATCH_BUFFER:
                dn = float(np.linalg.norm(d))
                if dn < best_norm:
                    best_norm = dn
                    best_j = j

        if best_j < 0:
            raise RuntimeError(f"No target site for translated key={key2(shifted)}")

        used[best_j] = True
        assignments.append((target_entries[best_j][0], sp))

    for idx, sp in assignments:
        struct[idx] = (sp, struct[idx].frac_coords)


def write_single_trans0(struct: Structure, na_staging_dir: Path, reason: str, source_poscar: Path) -> None:
    t0 = na_staging_dir / "trans_0"
    t0.mkdir(parents=True, exist_ok=True)
    Poscar(struct.get_sorted_structure()).write_file(t0 / "POSCAR")
    with open(t0 / "trans_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "reason": reason,
                "source_poscar": str(source_poscar),
                "trans_index": 0,
            },
            f,
            indent=2,
        )


def process_one_elec(ep: ElecPath, args: argparse.Namespace) -> str:
    in_poscar = ep.elec_dir / args.input_poscar_name
    if not in_poscar.exists():
        return f"[SKIP] {ep.elec_dir} missing {args.input_poscar_name}"

    na_staging_dir = ep.elec_dir / args.na_staging_dir_name
    summary_path = na_staging_dir / "na_staging_summary.json"
    if na_staging_dir.exists() and summary_path.exists() and not args.overwrite:
        return f"[SKIP] {ep.elec_dir} existing {args.na_staging_dir_name}"
    if args.overwrite and na_staging_dir.exists():
        for old in na_staging_dir.glob("trans_*"):
            if old.is_dir():
                shutil.rmtree(old)

    na_staging_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    warnings: List[str] = []

    def _run() -> dict:
        struct = Structure.from_file(str(in_poscar))
        layers = split_layers_by_frac_z(struct)
        if not layers:
            write_single_trans0(struct, na_staging_dir, "No Na/K layer in structure.", in_poscar)
            return {"num_translations": 1, "k_total": 0}

        k_total = sum(1 for site in struct if site.specie.symbol == "K")
        if k_total == 0:
            write_single_trans0(struct, na_staging_dir, "No K in structure.", in_poscar)
            return {"num_translations": 1, "k_total": 0}

        mixed_layers = []
        for li, layer in enumerate(layers):
            symbols = [struct[i].specie.symbol for i in layer]
            has_na = any(s == "Na" for s in symbols)
            has_k = any(s == "K" for s in symbols)
            if has_na and has_k:
                mixed_layers.append(li)

        if not mixed_layers:
            write_single_trans0(struct, na_staging_dir, "No Na-K mixed layer found.", in_poscar)
            return {"num_translations": 1, "k_total": k_total}

        if len(mixed_layers) > 1:
            raise RuntimeError(f"Multiple Na-K mixed layers found: {mixed_layers}")
        template_li = mixed_layers[0]
        template_layer = layers[template_li]
        target_layers = [layers[i] for i in range(len(layers)) if i != template_li]

        p_xy, p_map = layer_xy_sets(struct, template_layer)
        pattern: Dict[Tuple[float, float], str] = {}
        for key, idxs in p_map.items():
            if len(idxs) != 1:
                raise RuntimeError(f"Template layer has duplicate xy key={key}: {idxs}")
            sp = struct[idxs[0]].specie.symbol
            if sp not in ALKALI:
                raise RuntimeError(f"Template layer has non-alkali species: {sp}")
            pattern[key] = sp

        target_data = []
        for tl in target_layers:
            q_xy, q_map = layer_xy_sets(struct, tl)
            vlist = translations_that_map(p_xy, q_xy)
            if not vlist:
                raise RuntimeError("No translation vectors found from template to a target layer.")
            target_data.append((q_map, vlist))

        n_trans = 0
        if not target_data:
            trans_dir = na_staging_dir / "trans_0"
            trans_dir.mkdir(parents=True, exist_ok=True)
            Poscar(struct.get_sorted_structure()).write_file(trans_dir / "POSCAR")
            with open(trans_dir / "trans_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_poscar": str(in_poscar),
                        "template_layer_index": template_li,
                        "n_layers": len(layers),
                        "n_mixed_layers": len(mixed_layers),
                        "k_total": k_total,
                        "trans_index": 0,
                        "translations": [],
                    },
                    f,
                    indent=2,
                )
            n_trans = 1
        else:
            vec_lists = [item[1] for item in target_data]
            for combo_i, combo in enumerate(product(*vec_lists)):
                s_new = struct.copy()
                trans_meta = []
                for (q_map, _), v in zip(target_data, combo):
                    apply_pattern_to_target_layer(s_new, p_xy, pattern, q_map, v)
                    trans_meta.append([float(v[0]), float(v[1])])
                trans_dir = na_staging_dir / f"trans_{combo_i}"
                trans_dir.mkdir(parents=True, exist_ok=True)
                Poscar(s_new.get_sorted_structure()).write_file(trans_dir / "POSCAR")
                with open(trans_dir / "trans_meta.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "source_poscar": str(in_poscar),
                            "template_layer_index": template_li,
                            "n_layers": len(layers),
                            "n_mixed_layers": len(mixed_layers),
                            "k_total": k_total,
                            "trans_index": combo_i,
                            "translations": trans_meta,
                        },
                        f,
                        indent=2,
                    )
                n_trans += 1

        # Sanity check:
        # For 3-layer staging, expected translation count is often N^2 where
        # N = (Na + K) count in the template layer.
        n_alkali_template = len(template_layer)
        expected_square = n_alkali_template * n_alkali_template
        if n_trans != expected_square:
            warnings.append(
                f"Translation count mismatch: generated={n_trans}, expected_N2={expected_square}, "
                f"N_template(Na+K)={n_alkali_template}"
            )
        return {
            "num_translations": int(n_trans),
            "k_total": int(k_total),
        }

    try:
        run_info = _run()
    except Exception as exc:
        with open(na_staging_dir / "error.txt", "w", encoding="utf-8") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
        return f"[FAIL] {ep.elec_dir} {type(exc).__name__}: {exc}"

    if warnings:
        with open(na_staging_dir / "warnings.txt", "w", encoding="utf-8") as f:
            for w in warnings:
                f.write(w + "\n")
        for w in warnings:
            print(f"[WARN] {ep.elec_dir} {w}", flush=True)

    payload = {
        "phase": ep.phase,
        "ordering": ep.ordering,
        "tm_pair": ep.tm_pair,
        "elec_dir": str(ep.elec_dir),
        "input_poscar": str(in_poscar),
        "output_dir": str(na_staging_dir),
        "runtime_s": float(time.time() - t0),
        "warnings": warnings,
        **run_info,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return (
        f"[DONE] {ep.elec_dir} translations={run_info['num_translations']}"
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    tasks = collect_elec_dirs(root)
    print(f"[INFO] root={root}")
    print(f"[INFO] discovered elec_*={len(tasks)}")

    n_done = 0
    n_skip = 0
    n_fail = 0
    for ep in tasks:
        msg = process_one_elec(ep, args)
        print(msg, flush=True)
        if msg.startswith("[DONE]"):
            n_done += 1
        elif msg.startswith("[SKIP]"):
            n_skip += 1
        else:
            n_fail += 1

    print(
        f"[SUMMARY] done={n_done} skip={n_skip} fail={n_fail} total={len(tasks)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
