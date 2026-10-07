#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Na-O local-relationship checker for experimental ordering structures.

What it does:
1) For each Na, finds the 6 nearest O atoms.
2) Builds vectors: (Na_frac - O_frac) with PBC wrapping to [-0.5, 0.5).
3) Classifies Na sites into groups with mutually consistent 6-vector sets.

Outputs:
- Console summary
- JSON report (default: <input_stem>_na_o_vectors.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pymatgen.core import Structure


def wrap_centered(frac_vec: np.ndarray) -> np.ndarray:
    """Wrap fractional vector into [-0.5, 0.5)."""
    return (np.asarray(frac_vec, dtype=float) + 0.5) % 1.0 - 0.5


def vector_distance_frac(vf1: np.ndarray, vf2: np.ndarray) -> float:
    """Distance between two fractional vectors."""
    # Wrap centered to get the minimum-image distance in fractional space.
    d = wrap_centered(np.asarray(vf1, dtype=float) - np.asarray(vf2, dtype=float))
    return float(np.linalg.norm(d))


def _sort_vectors_for_matching(vectors_frac: List[np.ndarray]) -> List[np.ndarray]:
    return sorted(vectors_frac, key=lambda v: float(np.linalg.norm(v)))


def match_vector_sets(
    ref_vectors_frac: List[np.ndarray],
    cand_vectors_frac: List[np.ndarray],
    tol_frac: float,
) -> Tuple[bool, float, float, List[Dict[str, object]]]:
    """
    Compare two vector sets (order-insensitive) with greedy minimum-distance matching.

    Returns:
      (is_match, max_dev_frac, mean_dev_frac, pair_matches)
    where pair_matches contains per-pair mapping/deviation info.
    """
    if len(ref_vectors_frac) != len(cand_vectors_frac):
        return False, float("inf"), float("inf"), []

    if not ref_vectors_frac:
        return True, 0.0, 0.0, []

    ref_sorted = _sort_vectors_for_matching(ref_vectors_frac)
    # To catch original indices for reporting.
    cand_remaining = [
        {"orig_idx": i, "vec": np.asarray(v, dtype=float)}
        for i, v in enumerate(cand_vectors_frac)
    ]
    deviations: List[float] = []
    pair_matches: List[Dict[str, object]] = []

    for ref_idx, rv in enumerate(ref_sorted):
        best_index = -1
        best_distance = float("inf")
        for j, c in enumerate(cand_remaining):
            d = vector_distance_frac(rv, c["vec"])
            if d < best_distance:
                best_distance = d
                best_index = j
        if best_index < 0:
            return False, float("inf"), float("inf"), pair_matches
        deviations.append(best_distance)
        chosen = cand_remaining.pop(best_index)
        pair_matches.append(
            {
                "ref_sorted_index": int(ref_idx),
                "ref_vector_frac": rv.tolist(),
                "cand_original_index": int(chosen["orig_idx"]),
                "cand_vector_frac": chosen["vec"].tolist(),
                "deviation_frac": float(best_distance),
            }
        )

    max_dev = max(deviations)
    mean_dev = float(np.mean(deviations))
    return max_dev <= tol_frac, float(max_dev), mean_dev, pair_matches


def classify_na_vector_groups(
    na_records: List[Dict[str, object]],
    vector_sets_frac: List[List[np.ndarray]],
    tol_frac: float,
) -> List[Dict[str, object]]:
    """
    Group Na sites by mutual vector-set consistency.

    Returns:
      groups_for_json, a compact grouped view for reporting.
    """
    if len(na_records) != len(vector_sets_frac):
        raise RuntimeError(
            f"Internal size mismatch: na_records={len(na_records)} vs vector_sets={len(vector_sets_frac)}"
        )

    group_refs: List[List[np.ndarray]] = []
    group_members: List[List[int]] = []

    for na_pos, vset in enumerate(vector_sets_frac):
        best_group = -1
        best_max_dev = float("inf")

        for gidx, gref in enumerate(group_refs):
            ok, max_dev, _mean_dev, _pair_matches = match_vector_sets(
                ref_vectors_frac=gref,
                cand_vectors_frac=vset,
                tol_frac=tol_frac,
            )
            if ok and max_dev < best_max_dev:
                best_max_dev = max_dev
                best_group = gidx

        if best_group < 0:
            group_refs.append(vset)
            group_members.append([na_pos])
        else:
            group_members[best_group].append(na_pos)

    groups_for_json: List[Dict[str, object]] = []
    for gidx, members in enumerate(group_members, start=1):
        na_members: List[Dict[str, object]] = []
        for na_pos in members:
            rec = na_records[na_pos]
            near_o_rows = [
                {
                    "o_index": int(o_rec["o_index"]),
                    "vector_frac": [float(x) for x in o_rec["vector_frac"]],
                }
                for o_rec in rec["near_o"]
            ]
            na_members.append(
                {
                    "na_index": int(rec["na_index"]),
                    "near_o": near_o_rows,
                }
            )
        groups_for_json.append(
            {
                "group_id": int(gidx),
                "n_na": int(len(members)),
                "na_members": na_members,
            }
        )

    return groups_for_json


def nearest_o_vectors_for_na(
    struct: Structure,
    na_idx: int,
    o_indices: List[int],
    n_nearest_o: int,
) -> Tuple[List[Dict[str, object]], float]:
    """
    For one Na site, return vectors to the n_nearest_o oxygens.

    Vector convention follows user request: Na_frac - O_frac.
    Returns:
        (near_rows, dmax)
    """
    na_frac = np.asarray(struct[na_idx].frac_coords, dtype=float)

    rows: List[Dict[str, object]] = []
    lattice = struct.lattice
    for o_idx in o_indices:
        o_frac = np.asarray(struct[o_idx].frac_coords, dtype=float)
        # True minimum-image Cartesian distance + image vector from pymatgen.
        dcart, jimage = lattice.get_distance_and_image(na_frac, o_frac)
        o_frac_img = o_frac + np.asarray(jimage, dtype=float)
        vfrac = na_frac - o_frac_img
        rows.append(
            {
                "o_index": int(o_idx),
                "o_species": struct[o_idx].specie.symbol,
                "vector_frac": vfrac,
                "distance_cart": float(dcart),
                "o_image": [int(jimage[0]), int(jimage[1]), int(jimage[2])],
            }
        )

    if not rows:
        return [], float("inf")

    rows.sort(key=lambda r: float(r["distance_cart"]))
    near = rows[:n_nearest_o]
    dmax = float(near[-1]["distance_cart"]) if near else float("inf")
    return near, dmax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map Na-O neighbor vectors and classify Na sites into consistent vector-set groups."
    )
    p.add_argument(
        "--input", 
        type=Path, 
        required=True, 
        help="Input structure.")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON report path (default: <input_stem>_na_o_vectors.json).",
    )
    p.add_argument(
        "--n-nearest-o",
        type=int,
        default=6,
        help="Number of nearest O neighbors per Na (default: 6).",
    )
    p.add_argument(
        "--consistency-tol-frac",
        type=float,
        default=0.02,
        help="Tolerance in fractional units for vector-set consistency across Na sites.",
    )
    p.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Rounding decimals used only for JSON readability.",
    )
    return p.parse_args()


def _round_vec(v: np.ndarray, nd: int) -> List[float]:
    arr = np.asarray(v, dtype=float)
    return [round(float(arr[0]), nd), round(float(arr[1]), nd), round(float(arr[2]), nd)]


def main() -> None:
    args = parse_args()
    in_path = args.input.resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else in_path.with_name(f"{in_path.stem}_na_o_vectors.json")
    )

    struct = Structure.from_file(str(in_path))

    na_indices = [i for i, s in enumerate(struct) if s.specie.symbol == "Na"]
    o_indices = [i for i, s in enumerate(struct) if s.specie.symbol == "O"]

    if not na_indices:
        raise RuntimeError("No Na sites found.")
    if not o_indices:
        raise RuntimeError("No O sites found.")
    if args.n_nearest_o <= 0:
        raise ValueError("--n-nearest-o must be positive.")
    if len(o_indices) < args.n_nearest_o:
        raise RuntimeError(
            f"Not enough O sites: found {len(o_indices)} but requested {args.n_nearest_o} nearest O."
        )

    # Build 6-nearest-O vector sets for each Na
    na_records: List[Dict[str, object]] = []
    vector_sets_frac: List[List[np.ndarray]] = []

    for na_idx in na_indices:
        near_rows, dmax = nearest_o_vectors_for_na(
            struct=struct,
            na_idx=na_idx,
            o_indices=o_indices,
            n_nearest_o=args.n_nearest_o,
        )

        if len(near_rows) != args.n_nearest_o:
            raise RuntimeError(
                f"Na index {na_idx}: expected {args.n_nearest_o} nearest O but got {len(near_rows)}."
            )

        vset = [np.asarray(r["vector_frac"], dtype=float) for r in near_rows]
        vector_sets_frac.append(vset)

        na_records.append(
            {
                "na_index": int(na_idx),
                "na_frac": _round_vec(struct[na_idx].frac_coords, args.round_decimals),
                "n_near_o": int(len(near_rows)),
                "dmax_near_o_A": round(float(dmax), args.round_decimals),
                "near_o": [
                    {
                        "o_index": int(r["o_index"]),
                        "distance_cart": round(float(r["distance_cart"]), args.round_decimals),
                        "o_image": [int(x) for x in r["o_image"]],
                        "vector_frac": _round_vec(r["vector_frac"], args.round_decimals),
                    }
                    for r in near_rows
                ],
            }
        )

    # Debug print: vector sets for all Na
    # TODO Delete later after fully developed and tested.
    print("[DEBUG] Na-O vector sets (Na_frac - O_frac, wrapped):")
    for rec in na_records:
        print(
            f"[DEBUG] Na index={rec['na_index']} frac={rec['na_frac']} "
            f"n_near_o={rec['n_near_o']}"
        )
        for j, orec in enumerate(rec["near_o"]):
            print(
                f"  [v{j}] O index={orec['o_index']} "
                f"dist_cart={orec['distance_cart']} "
                f"o_image={orec['o_image']} "
                f"vector_frac={orec['vector_frac']}"
            )

    groups = classify_na_vector_groups(
        na_records=na_records,
        vector_sets_frac=vector_sets_frac,
        tol_frac=args.consistency_tol_frac,
    )
    group_sizes = [int(g["n_na"]) for g in groups]

    report = {
        "input": str(in_path),
        "n_sites_total": int(len(struct)),
        "n_na": int(len(na_indices)),
        "n_o": int(len(o_indices)),
        "settings": {
            "n_nearest_o": int(args.n_nearest_o),
            "consistency_tol_frac": float(args.consistency_tol_frac),
            "round_decimals": int(args.round_decimals),
        },
        "grouping": {
            "n_groups": int(len(groups)),
            "groups": groups,
        },
        "na_records": na_records,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] input={in_path}")
    print(f"[OK] wrote report={out_json}")
    print(
        "[SUMMARY] "
        f"Na={len(na_indices)} O={len(o_indices)} "
        f"n_nearest_o={args.n_nearest_o} "
        f"n_groups={len(groups)} "
        f"group_sizes={group_sizes}"
    )


if __name__ == "__main__":
    main()
