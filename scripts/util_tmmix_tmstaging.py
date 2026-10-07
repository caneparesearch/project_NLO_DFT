#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yunyeong Choi
# Logic: 
# Parallel execution model (SLURM + multiprocessing):
#
# 1) Inter-rank parallelism (across SLURM tasks)
#    - rank    = SLURM_PROCID   (this process ID among all tasks)
#    - n_ranks = SLURM_NTASKS   (total number of tasks launched by srun)
#    - We shard input files by index: file i is handled by rank (i % n_ranks).
#    - This creates local_poscars for each rank and avoids duplicate work/output collisions.
#
# 2) Intra-rank parallelism (within one task)
#    - workers = local multiprocessing process count (default: SLURM_CPUS_PER_TASK).
#    - Each rank processes its local_poscars with mp.Pool(workers).
#
# Resource rule:
#    ntasks-per-node * cpus-per-task must not exceed physical CPUs per node.
#    Example on 48-CPU node:
#      valid: (48,1), (24,2), (12,4)
#      invalid: (48,2) -> requests 96 CPUs/node.
#
# Practical recommendation:
#    - If cpus-per-task=1, keep workers=1.
#    - If cpus-per-task=k, use workers=k (or k-1 if memory is tight).


import argparse
import json
import multiprocessing as mp
import os
import signal
from collections import Counter
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp import Poscar
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
)
PHASE_DIRS = ["O1_orderings", "O3_orderings", "P3_orderings"]
# TODO: Na-staging after TM ordering is implemented.
# TODO: Carefully check XL*.vasp.
PATTERNS = [
    "AJT*_w_vac.vasp",
    "JLK*_w_vac.vasp",
    "JMAS*.vasp",
    "XC*.vasp",
    "XL*.vasp",
    "CD*.vasp",
    "O3*.vasp",
    "P3*.vasp",
    "O1*.vasp",
]
TM_EXCLUDE = {"Na", "K", "O"}
Z_LAYER_TOL = 0.02
Z_LAYER_TOL_CANDIDATES = [0.02, 0.01, 0.005, 0.002]
XY_DECIMALS = 2
XY_MATCH_BUFFER = 0.05
TOP_ELEC = 5
MAX_RANKED = 200
SM_LATTICE_TOL = 0.2
SM_SITE_TOL = 0.3
SM_ANGLE_TOL = 5.0


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """
    Simple wall-time limit using SIGALRM (Linux/HPC friendly).
    """
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate double-TM orderings by: electrostatic ordering on the first TM "
            "layer, translational staging to other TM layers, then StructureMatcher grouping."
        )
    )
    p.add_argument("--tm-a", default="Ti")
    p.add_argument("--tm-b", default="Mn")
    p.add_argument("--input-root", type=Path, default=Path("../data"))
    p.add_argument("--output-root", type=Path, default=Path("../ordering_doubleTM"))
    p.add_argument("--max-ranked", type=int, default=MAX_RANKED)
    p.add_argument("--top-elec", type=int, default=TOP_ELEC)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Local multiprocessing workers per rank/process.",
    )
    p.add_argument(
        "--large-tm-threshold",
        type=int,
        default=42,
        help="Apply timeout when total TM sites >= this value.",
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=3600,
        help="Timeout (seconds) for large-TM structures in step3.",
    )
    p.add_argument(
        "--timeout-start-elec",
        type=int,
        default=2,
        help="Start timeout policy from this elec index (e.g., 2 -> elec_2,3,4...).",
    )
    p.add_argument(
        "--only",
        default="",
        help="Optional substring filter for input filename/path (debug convenience).",
    )
    p.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated execution steps. "
            "Use from: step1,step2,step3 or all. "
            "step1=first-layer TM-mix only, "
            "step2=export translation vectors for debug, "
            "step3=apply translations + group structures."
        ),
    )
    p.add_argument(
        "--debug-print-target-layer-data",
        action="store_true",
        help="DEBUG ONLY: print target_layer_data to stdout.",
    )
    p.add_argument(
        "--vectors-json-name",
        default="target_layer_data.json",
        help="Filename used to save/load step2 translation-vector data.",
    )
    return p.parse_args()


def frac01(x: np.ndarray) -> np.ndarray:
    """
    Wrap fractional coordinates into [0,1).
    """
    return x - np.floor(x)


def key2(xy: np.ndarray, decimals: int = XY_DECIMALS) -> Tuple[float, float]:
    """
    Hashable key for 2D fractional coords under PBC (rounded).
    """
    v = frac01(np.asarray(xy))[:2]
    # NOTE:
    # A coordinate very close to 1.0 (e.g., 0.99995) can round to exactly 1.00.
    # Re-wrap after rounding so hash keys stay in [0,1) under PBC.
    v = np.mod(np.round(v, decimals=decimals), 1.0)
    return tuple(v.astype(float))


def tm_layers_by_frac_z(structure: Structure, tm_symbol: str, z_tol: float = Z_LAYER_TOL) -> List[List[int]]:
    """
    Split TM layers by fractional z coordinate, with a tolerance to account for small deviations.
    # TODO: This does not work for the non-conventional cells, such as XL_a.
    # TODO: Need a more robust method to identify layers.
    return List[List[int]], index of transition metals in each layer.
    """
    tm_idx = [i for i, site in enumerate(structure) if site.specie.symbol == tm_symbol]
    if not tm_idx:
        return []

    zs = sorted(
        [(float(structure[i].frac_coords[2] % 1.0), i) for i in tm_idx],
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


def tm_layers_by_frac_z_all_tm(structure: Structure, z_tol: float = Z_LAYER_TOL) -> List[List[int]]:
    """
    Split TM layers by fractional z using all TM species (exclude Na/K/O).
    Useful after TM mixing where first layer no longer has only host_tm.
    """
    tm_idx = [i for i, site in enumerate(structure) if site.specie.symbol not in TM_EXCLUDE]
    if not tm_idx:
        return []

    zs = sorted(
        [(float(structure[i].frac_coords[2] % 1.0), i) for i in tm_idx],
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


def ensure_even_first_layer(structure: Structure, first_layer_size: int) -> Tuple[Structure, Optional[str]]:
    """
    Make cell size doubled if there are odd numbers of TM in the first layer.
    """
    if first_layer_size % 2 == 0:
        return structure, None

    axis = "a" if float(structure.lattice.a) <= float(structure.lattice.b) else "b"
    scale = [[2, 0, 0], [0, 1, 0], [0, 0, 1]] if axis == "a" else [[1, 0, 0], [0, 2, 0], [0, 0, 1]]

    s2 = structure.copy()
    s2.make_supercell(scale)
    return s2, axis


def collect_poscars(input_root: Path) -> List[Path]:
    """
    Collect all POSCAR files from the input root directory (../data).
    """
    if input_root is None or not input_root.exists():
        input_root = Path("../data")
    files: List[Path] = []
    seen = set()
    for phase in PHASE_DIRS:
        phase_dir = input_root / phase
        if not phase_dir.exists():
            continue
        for pat in PATTERNS:
            for p in phase_dir.glob(pat):
                if "_conventional" in p.stem:
                    continue
                if "_wrongdirection" in p.stem:
                    continue
                if "_rotate" in p.stem:
                    continue
                if p.is_file() and p not in seen:
                    seen.add(p)
                    files.append(p)
    return sorted(files)


def detect_host_tm(structure: Structure) -> str:
    """
    Check what kind of TM are in the given structure.
    This is used since there are non-Co version from exp orderings.
    """
    tm_species = [site.specie.symbol for site in structure if site.specie.symbol not in TM_EXCLUDE]
    if not tm_species:
        raise RuntimeError("No transition-metal species found.")
    return Counter(tm_species).most_common(1)[0][0]


def _ewald_total_energy(s: Structure) -> Optional[float]:
    # TODO: May Need Manual Oxidation State Assignment
    try:
        ss = s.copy()
        ss.add_oxidation_state_by_guess()
        return float(EwaldSummation(ss).total_energy)
    except Exception:
        return None


def ordered_first_layer_candidates(
    structure: Structure,
    first_layer: Sequence[int],
    tm_a: str,
    tm_b: str,
    max_ranked: int,
    top_elec: int,
) -> List[Tuple[Structure, Optional[float]]]:
    # TODO: Need Manual Oxidation State Assignment
    # Return ordered structure with electrostatic energy.
    dis = structure.copy()
    for i in first_layer:
        dis.replace(i, {tm_a: 0.5, tm_b: 0.5})

    dis.add_oxidation_state_by_guess()
    trans = OrderDisorderedStructureTransformation(algo=0)
    ranked_list = trans.apply_transformation(dis, return_ranked_list=max_ranked)

    out: List[Tuple[Structure, Optional[float]]] = []
    for item in ranked_list[:top_elec]:
        s_ord = item.get("structure") if isinstance(item, dict) else item
        e = item.get("energy") if isinstance(item, dict) else None
        s_clean = s_ord.copy()
        try:
            s_clean.remove_oxidation_states()
        except Exception:
            pass
        out.append((s_clean, e))
    return out


def layer_xy_sets(struct: Structure, indices: Sequence[int]) -> Tuple[np.ndarray, Dict[Tuple[float, float], List[int]]]:
    """
    For a given layer (site indices), return:
      - the same indices,
      - Nx2 array of fractional (x,y),
      - dict mapping hashed xy key -> list of site indices 
    """
    xys = []
    mapping: Dict[Tuple[float, float], List[int]] = {}
    for i in indices:
        xy = frac01(struct[i].frac_coords)[:2]
        xys.append(xy)
        mapping.setdefault(key2(xy), []).append(i)
    return np.array(xys), mapping


def _has_duplicate_xy(struct: Structure, layer_indices: Sequence[int]) -> bool:
    """
    Sanity check if there are duplicate xy sites in the given layer, which would make the translation mapping ambiguous.
    """
    _, layer_map = layer_xy_sets(struct, layer_indices)
    return any(len(v) > 1 for v in layer_map.values())


def _pick_tm_layers(struct: Structure, host_tm: Optional[str] = None) -> List[List[int]]:
    """
    Loose Z layer tolerance iteratively to find TM layers without duplicate xy sites.
    """
    for tol in Z_LAYER_TOL_CANDIDATES:
        if host_tm is None:
            layers = tm_layers_by_frac_z_all_tm(struct, z_tol=tol)
        else:
            layers = tm_layers_by_frac_z(struct, tm_symbol=host_tm, z_tol=tol)
        if not layers:
            continue
        if all(not _has_duplicate_xy(struct, layer) for layer in layers):
            return layers
    if host_tm is None:
        return tm_layers_by_frac_z_all_tm(struct, z_tol=Z_LAYER_TOL_CANDIDATES[-1])
    return tm_layers_by_frac_z(struct, tm_symbol=host_tm, z_tol=Z_LAYER_TOL_CANDIDATES[-1])


def translations_that_map(P: np.ndarray, Q: np.ndarray) -> List[np.ndarray]:
    """
    Core Function.
    Find all translation vectors v (2D fractional) such that 
    (P + v) % 1 is a permutation of Q within rounding tolerance.
    Strategy: Try v = q - p0 for every q in Q using an arbitrary p0 in P, then verify full-set membership.
    """
    if len(P) != len(Q):
        raise RuntimeError("TM layer site counts differ; cannot map by translation.")
    if len(P) == 0:
        return []

    p0 = P[0]
    vectors: List[np.ndarray] = []
    seen = set()

    def _can_bijectively_match_with_buffer(shifted: np.ndarray, target: np.ndarray, buf: float) -> bool:
        """
        Buffered 1:1 matching under PBC.
        Two xy points are considered the same if each component differs by <= buf
        after minimum-image wrapping.
        """
        n = len(shifted)
        used = [False] * n
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
        ok = _can_bijectively_match_with_buffer(shifted, Q, XY_MATCH_BUFFER)
        if ok:
            k = key2(v)
            if k not in seen:
                seen.add(k)
                vectors.append(v)

    return vectors


def parse_steps(step_arg: str) -> set:
    raw = [x.strip().lower() for x in step_arg.split(",") if x.strip()]
    if not raw or "all" in raw:
        return {"step1", "step2", "step3"}
    valid = {"step1", "step2", "step3"}
    bad = [x for x in raw if x not in valid]
    if bad:
        raise ValueError(f"Invalid --steps value(s): {bad}. Use step1,step2,step3,all")
    return set(raw)


def canonical_tm_pair(tm_a: str, tm_b: str) -> Tuple[str, str]:
    """
    Keep TM pair ordering consistent for directory naming:
    lower atomic number first.
    """
    a = tm_a.strip()
    b = tm_b.strip()
    if not a or not b:
        raise ValueError("tm_a/tm_b must be non-empty")
    if a == b:
        return a, b

    za = Element(a).Z
    zb = Element(b).Z
    if za < zb:
        return a, b
    if zb < za:
        return b, a
    return tuple(sorted([a, b]))


def build_translation_debug_data(struct: Structure, layers: Sequence[Sequence[int]]) -> List[dict]:
    """
    Build serializable translation-vector data from first TM layer to each target TM layer.
    This is intentionally simple and explicit for debug readability.
    """
    if len(layers) < 2:
        raise RuntimeError("Need at least 2 TM layers for translation staging.")

    first_layer = layers[0]
    p_xy_ref, _ = layer_xy_sets(struct, first_layer)
    out = []
    for li, target in enumerate(layers[1:], start=1):
        q_xy, _ = layer_xy_sets(struct, target)
        vlist = translations_that_map(p_xy_ref, q_xy)
        if not vlist:
            raise RuntimeError(f"no translation map first-layer -> layer {li}")
        n_tm_target = len(target)
        n_vectors = len(vlist)
        if n_vectors != n_tm_target:
            print(
                f"[WARNING][step2] target_layer={li} "
                f"vector_count={n_vectors} != tm_sites_in_layer={n_tm_target}"
            )
        out.append(
            {
                "target_layer": int(li),
                "tm_sites_in_layer": int(n_tm_target),
                "vector_count": int(n_vectors),
                "vectors_frac_xy": [[float(v[0]), float(v[1])] for v in vlist],
            }
        )
    return out


def runtime_target_layer_data_from_debug(
    struct: Structure,
    layers: Sequence[Sequence[int]],
    debug_data: Sequence[dict],
) -> List[Tuple[int, Dict[Tuple[float, float], List[int]], List[np.ndarray]]]:
    """
    Convert debug json data from step 2 back into runtime format:
      (target_layer_index, target_xy_map, list_of_vectors_as_np_arrays)
    If one wants to reuse the step2 output for step3.
    """
    out = []
    for item in debug_data:
        li = int(item["target_layer"])
        if li >= len(layers):
            raise RuntimeError(f"target_layer index {li} is out of range for current structure.")
        _, q_map = layer_xy_sets(struct, layers[li])
        vectors = [np.array(v, dtype=float) for v in item["vectors_frac_xy"]]
        out.append((li, q_map, vectors))
    return out


def build_pattern(struct: Structure, layer_indices: Sequence[int]) -> Dict[Tuple[float, float], str]:
    """
    Build Transition Metal pattern dict for the given layer indices.
    Return dict mapping hashed xy key -> TM symbol.
    """
    pattern: Dict[Tuple[float, float], str] = {}
    _, layer_map = layer_xy_sets(struct, layer_indices)
    for xy_key, idxs in layer_map.items():
        if len(idxs) != 1:
            raise RuntimeError(f"Duplicated sites at xy={xy_key}: {idxs}")
        pattern[xy_key] = struct[idxs[0]].specie.symbol
    return pattern


def apply_pattern_to_target_layer(
    struct: Structure,
    src_xy: np.ndarray,
    pattern: Dict[Tuple[float, float], str],
    target_xy_map: Dict[Tuple[float, float], List[int]],
    v2d: np.ndarray,
) -> None:
    # Build unique target entries once and perform buffered 1:1 matching under PBC.
    target_entries: List[Tuple[int, np.ndarray]] = []
    for q_key, idxs in target_xy_map.items():
        if len(idxs) != 1:
            raise RuntimeError(f"Ambiguous translated mapping at key={q_key}: {idxs}")
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
            tgt_key = key2(shifted)
            raise RuntimeError(
                f"No target site for translated key={tgt_key} "
                f"within XY_MATCH_BUFFER={XY_MATCH_BUFFER}"
            )

        used[best_j] = True
        tgt_idx = target_entries[best_j][0]
        assignments.append((tgt_idx, sp))

    for idx, sp in assignments:
        struct[idx] = (sp, struct[idx].frac_coords)


def group_and_write(
    structures: Sequence[Structure],
    metas: Sequence[dict],
    out_dir: Path,
    base_meta: dict,
    ignore_k_in_grouping: bool = True,
) -> int:
    """
    Group the given structures by StructureMatcher and write out each group with metadata.
    """
    sm = StructureMatcher(
        ltol=SM_LATTICE_TOL,
        stol=SM_SITE_TOL,
        angle_tol=SM_ANGLE_TOL,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )

    def _normalized_for_grouping(s: Structure) -> Tuple[Structure, List[int]]:
        # NOTE:
        # For grouping only, treat K as Na so Na/K vacancy labeling does not split
        # otherwise equivalent staging patterns.
        s_norm = s.copy()
        k_indices = []
        if ignore_k_in_grouping:
            for i, site in enumerate(s_norm):
                if site.specie.symbol == "K":
                    k_indices.append(i)
                    s_norm[i] = ("Na", site.frac_coords)
        return s_norm, k_indices

    def _restore_k_positions(s: Structure, k_indices: Sequence[int]) -> Structure:
        # Restore K labels after grouping (output must preserve original K positions).
        out = s.copy()
        for i in k_indices:
            out[i] = ("K", out[i].frac_coords)
        return out

    if ignore_k_in_grouping:
        norm_and_k = [_normalized_for_grouping(s) for s in structures]
        grouping_structs = [x[0] for x in norm_and_k]
        k_index_list = [x[1] for x in norm_and_k]
    else:
        grouping_structs = list(structures)
        k_index_list = [[] for _ in structures]

    groups = sm.group_structures(grouping_structs)
    idx_map = {id(s): i for i, s in enumerate(grouping_structs)}

    for gi, group in enumerate(groups):
        gdir = out_dir / f"group_{gi}"
        gdir.mkdir(parents=True, exist_ok=True)

        rep = group[0]
        rep_i = idx_map[id(rep)]
        rep_struct = _restore_k_positions(structures[rep_i], k_index_list[rep_i])
        Poscar(rep_struct).write_file(gdir / "POSCAR")

        group_meta = dict(base_meta)
        group_meta["representative"] = metas[rep_i]
        group_meta["members"] = [metas[idx_map[id(m)]] for m in group]
        group_meta["group_size"] = len(group)

        with open(gdir / "group_meta.json", "w", encoding="utf-8") as f:
            json.dump(group_meta, f, indent=2)

    return len(groups)


def process_one(poscar_path: Path, args: argparse.Namespace) -> str:
    struct0 = Poscar.from_file(str(poscar_path)).structure
    host_tm = detect_host_tm(struct0)
    tm_a, tm_b = canonical_tm_pair(args.tm_a, args.tm_b)
    phase = poscar_path.parent.name
    stem = poscar_path.stem
    pair_dirname = f"{tm_a}_{tm_b}"
    base_out = args.output_root / phase / stem / pair_dirname
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"[{phase}/{stem}/{pair_dirname}] start")

    def _status(message: str) -> str:
        with open(base_out / "status.txt", "w", encoding="utf-8") as f:
            f.write(message + "\n")
        return message

    layers0 = _pick_tm_layers(struct0, host_tm)
    if not layers0:
        return _status(f"[SKIP] {poscar_path}: no TM layers for {host_tm}")

    struct, doubled_axis = ensure_even_first_layer(struct0, len(layers0[0]))
    layers = _pick_tm_layers(struct, host_tm)
    if len(layers) < 2:
        return _status(f"[SKIP] {poscar_path}: need >=2 TM layers")

    first_layer = layers[0]
    candidates = ordered_first_layer_candidates(
        structure=struct,
        first_layer=first_layer,
        tm_a=tm_a,
        tm_b=tm_b,
        max_ranked=args.max_ranked,
        top_elec=args.top_elec,
    )

    if not candidates:
        return _status(f"[FAIL] {poscar_path}: no valid first-layer electrostatic orderings")

    steps = parse_steps(args.steps)
    print(
        f"[{phase}/{stem}/{pair_dirname}] steps={sorted(list(steps))} "
        f"candidates={len(candidates)} tm_layers={len(layers)}"
    )

    # Step 1: generate first-layer-only TM-mixed structures and save under elec_*.
    if "step1" in steps:
        print(f"[{phase}/{stem}/{pair_dirname}] step1: start")
        for elec_i, (s_elec, energy) in enumerate(candidates):
            elec_dir = base_out / f"elec_{elec_i}"
            elec_dir.mkdir(parents=True, exist_ok=True)
            if (elec_dir / "POSCAR_step1").exists() and not args.overwrite:
                print(f"[{phase}/{stem}/{pair_dirname}] step1: skip elec_{elec_i} (exists)")
                continue
            print(f"[{phase}/{stem}/{pair_dirname}] step1: writing elec_{elec_i}")
            Poscar(s_elec.get_sorted_structure()).write_file(elec_dir / "POSCAR_step1")
            step1_meta = {
                "source_poscar": str(poscar_path),
                "host_tm": host_tm,
                "tm_pair": [tm_a, tm_b],
                "elec_index": elec_i,
                "electrostatic_energy": float(energy) if energy is not None else _ewald_total_energy(s_elec),
                "supercell_doubled_axis": doubled_axis,
                "num_tm_layers": len(layers),
                "notes": "Step1 output: first TM layer is mixed; no inter-layer translation applied yet.",
            }
            with open(elec_dir / "step1_meta.json", "w", encoding="utf-8") as f:
                json.dump(step1_meta, f, indent=2)
        print(f"[{phase}/{stem}/{pair_dirname}] step1: done")

    vectors_json = base_out / args.vectors_json_name
    target_layer_data_debug: List[dict] = []

    # Step 2: export translation vectors for debug.
    if "step2" in steps:
        print(f"[{phase}/{stem}/{pair_dirname}] step2: start")
        try:
            target_layer_data_debug = build_translation_debug_data(struct, layers)
        except Exception as exc:
            return _status(f"[FAIL] {poscar_path}: {exc}")
        with open(vectors_json, "w", encoding="utf-8") as f:
            json.dump(target_layer_data_debug, f, indent=2)
        print(
            f"[{phase}/{stem}/{pair_dirname}] step2: done "
            f"target_layers={len(target_layer_data_debug)} "
            f"vector_counts={[len(x['vectors_frac_xy']) for x in target_layer_data_debug]}"
        )
        if args.debug_print_target_layer_data:
            # DEBUG NOTE: this can be large, intentionally printed as requested.
            print(target_layer_data_debug)

    # Step 3: apply translations (prefer vectors from step2 output), then group.
    if "step3" not in steps:
        return (
            f"[OK] {phase}/{stem}: "
            f"step1={'yes' if 'step1' in steps else 'no'} "
            f"step2={'yes' if 'step2' in steps else 'no'} step3=no"
        )

    if vectors_json.exists():
        with open(vectors_json, "r", encoding="utf-8") as f:
            target_layer_data_debug = json.load(f)
    elif "step2" not in steps:
        # If user runs only step3, we still build vectors in-memory as a fallback.
        target_layer_data_debug = build_translation_debug_data(struct, layers)

    if not target_layer_data_debug:
        return _status(f"[FAIL] {poscar_path}: no translation vector data for step3")

    print(f"[{phase}/{stem}/{pair_dirname}] step3: start")
    n_tm_total = sum(1 for site in struct if site.specie.symbol not in TM_EXCLUDE)
    use_timeout = n_tm_total >= args.large_tm_threshold and args.timeout_sec > 0
    if use_timeout:
        print(
            f"[{phase}/{stem}/{pair_dirname}] step3: timeout policy active "
            f"(n_tm={n_tm_total} >= {args.large_tm_threshold}, "
            f"timeout={args.timeout_sec}s, start_elec={args.timeout_start_elec})"
        )
    n_written_groups = 0
    last_layer_maps: List[int] = []
    for elec_i, (s_elec, energy) in enumerate(candidates):
        elec_dir = base_out / f"elec_{elec_i}"
        elec_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{phase}/{stem}/{pair_dirname}] step3: start elec_{elec_i}")

        def _run_one_elec() -> Tuple[int, List[int], int]:
            # IMPORTANT:
            # We must rebuild TM layer indices on each candidate structure because
            # ordering/transformation can change site indexing.
            layers_elec = _pick_tm_layers(s_elec, host_tm=None)
            if len(layers_elec) < 2:
                raise RuntimeError(f"{poscar_path}: candidate elec_{elec_i} has <2 TM layers")

            first_layer_elec = layers_elec[0]
            pattern = build_pattern(s_elec, first_layer_elec)
            p_xy, _ = layer_xy_sets(s_elec, first_layer_elec)
            target_layer_data = runtime_target_layer_data_from_debug(
                s_elec,
                layers_elec,
                target_layer_data_debug,
            )
            if args.debug_print_target_layer_data:
                # DEBUG NOTE: this includes q_map and can be very verbose.
                print(f"[DEBUG] elec_{elec_i} target_layer_data:", target_layer_data)

            vec_lists = [item[2] for item in target_layer_data]
            combos = list(product(*vec_lists))
            layer_map_sizes = [len(x[2]) for x in target_layer_data]
            print(
                f"[{phase}/{stem}/{pair_dirname}] step3: elec_{elec_i} "
                f"layer_maps={layer_map_sizes} combos={len(combos)}"
            )
            staged_structs = []
            staged_metas = []

            for combo_idx, combo in enumerate(combos):
                s_new = s_elec.copy()
                trans_meta = []
                for (layer_idx, q_map, _), v in zip(target_layer_data, combo):
                    apply_pattern_to_target_layer(s_new, p_xy, pattern, q_map, v)
                    trans_meta.append(
                        {
                            "target_layer": int(layer_idx),
                            "vector_frac_xy": [float(v[0]), float(v[1])],
                        }
                    )

                staged_structs.append(s_new.get_sorted_structure())
                staged_metas.append({"combo_index": combo_idx, "translations": trans_meta})

            base_meta = {
                "source_poscar": str(poscar_path),
                "host_tm": host_tm,
                "tm_pair": [tm_a, tm_b],
                "elec_index": elec_i,
                "electrostatic_energy": float(energy) if energy is not None else _ewald_total_energy(s_elec),
                "supercell_doubled_axis": doubled_axis,
                "num_tm_layers": len(layers),
                "num_staging_candidates": len(staged_structs),
            }

            staging_dir = elec_dir / "TM_staging"
            n_groups_local = group_and_write(staged_structs, staged_metas, staging_dir, base_meta)
            return n_groups_local, layer_map_sizes, len(combos)

        timeout_active_for_elec = use_timeout and elec_i >= args.timeout_start_elec
        try:
            if timeout_active_for_elec:
                with time_limit(args.timeout_sec):
                    n_groups, layer_map_sizes, _ = _run_one_elec()
            else:
                n_groups, layer_map_sizes, _ = _run_one_elec()
        except TimeoutError:
            msg = (
                f"[WARNING] Timeout at elec_{elec_i} after {args.timeout_sec}s; "
                f"skip remaining elec>= {elec_i} for this structure."
            )
            print(f"[{phase}/{stem}/{pair_dirname}] {msg}")
            with open(elec_dir / "timeout.txt", "w", encoding="utf-8") as f:
                f.write(msg + "\n")
            for remain_i in range(elec_i + 1, len(candidates)):
                remain_dir = base_out / f"elec_{remain_i}"
                remain_dir.mkdir(parents=True, exist_ok=True)
                with open(remain_dir / "skipped_due_to_timeout.txt", "w", encoding="utf-8") as f:
                    f.write(
                        f"Skipped because earlier elec_{elec_i} timed out "
                        f"({args.timeout_sec}s limit).\n"
                    )
            break

        last_layer_maps = layer_map_sizes
        n_written_groups += n_groups
        print(f"[{phase}/{stem}/{pair_dirname}] step3: done elec_{elec_i} groups={n_groups}")
    print(f"[{phase}/{stem}/{pair_dirname}] step3: done total_groups={n_written_groups}")

    return (
        f"[OK] {phase}/{stem}: step1={'yes' if 'step1' in steps else 'no'} "
        f"step2={'yes' if 'step2' in steps else 'no'} step3=yes "
        f"elec={len(candidates)} layer_maps={last_layer_maps} "
        f"groups_written={n_written_groups}"
    )


def process_one_safe(task: Tuple[Path, argparse.Namespace]) -> str:
    """
    Wrapper for multiprocessing safety:
    keeps per-file error handling behavior in both serial and parallel modes.
    """
    poscar_path, args = task
    try:
        return process_one(poscar_path, args)
    except Exception as exc:
        phase = poscar_path.parent.name
        stem = poscar_path.stem
        tm_a, tm_b = canonical_tm_pair(args.tm_a, args.tm_b)
        out_dir = args.output_root / phase / stem / f"{tm_a}_{tm_b}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "status.txt", "w", encoding="utf-8") as f:
            f.write(f"[ERROR] {poscar_path}: {exc}\n")
        return f"[ERROR] {poscar_path}: {exc}"


def main() -> None:
    args = parse_args()
    # Validate early so typo in steps fails fast.
    parse_steps(args.steps)

    for phase in PHASE_DIRS:
        (args.output_root / phase).mkdir(parents=True, exist_ok=True)

    poscars = collect_poscars(args.input_root)
    if args.only:
        poscars = [p for p in poscars if args.only in str(p)]
    if not poscars:
        print(f"No input POSCARs found under {args.input_root}")
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    n_ranks = int(os.environ.get("SLURM_NTASKS", "1"))
    local_poscars = [p for i, p in enumerate(poscars) if i % n_ranks == rank]
    workers = max(1, int(args.workers))

    print(
        f"Found {len(poscars)} files. Output root: {args.output_root} | "
        f"rank={rank}/{n_ranks} local_files={len(local_poscars)} workers={workers}"
    )
    if not local_poscars:
        print("No local files assigned to this rank.")
        return

    tasks = [(p, args) for p in local_poscars]
    if workers == 1 or len(tasks) == 1:
        for i, t in enumerate(tasks, start=1):
            msg = process_one_safe(t)
            print(f"[local {i}/{len(tasks)}] {msg}")
    else:
        chunksize = max(1, len(tasks) // (workers * 4))
        with mp.Pool(processes=workers) as pool:
            for i, msg in enumerate(pool.imap_unordered(process_one_safe, tasks, chunksize=chunksize), start=1):
                print(f"[local {i}/{len(tasks)}] {msg}")


if __name__ == "__main__":
    main()
