#!/usr/bin/env python3
"""Create stratified subsets for paid teacher prompt distillation."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_BUCKET_RATIOS = {
    "normal": 0.40,
    "hard_candidate": 0.25,
    "no_history": 0.15,
    "no_geo": 0.10,
    "short_context": 0.10,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified sample teacher distillation inputs and labels.")
    p.add_argument("--city", default="NewYork")
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("retrieval_assets/NewYork/distill"))
    p.add_argument("--name", default=None, help="Output suffix. Defaults to {split}_{size}.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def align_rows(inputs: List[Dict[str, Any]], labels: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    label_by_id = {str(row["sample_id"]): row for row in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("Duplicate sample_id found in labels")

    aligned = []
    seen = set()
    for row in inputs:
        sid = str(row.get("sample_id"))
        if not sid:
            raise ValueError("Input row without sample_id")
        if sid in seen:
            raise ValueError(f"Duplicate input sample_id: {sid}")
        seen.add(sid)
        label = label_by_id.get(sid)
        if not label:
            raise ValueError(f"Missing label for sample_id: {sid}")
        aligned.append((row, label))
    return aligned


def classify_bucket(inp: Dict[str, Any], label: Dict[str, Any]) -> str:
    evidence = inp.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}
    target = str(label.get("target_poi_id"))
    candidates = {str(x) for x in inp.get("candidate_poi_ids") or []}
    has_geo = bool(geo.get("nearby_pois") or [])
    has_history = int(pref.get("history_count") or 0) > 0
    current_len = len(seq.get("current_trajectory") or [])

    if target not in candidates:
        return "hard_candidate"
    if not has_history:
        return "no_history"
    if not has_geo:
        return "no_geo"
    if current_len <= 1:
        return "short_context"
    return "normal"


def target_counts(size: int) -> Dict[str, int]:
    counts = {bucket: int(size * ratio) for bucket, ratio in DEFAULT_BUCKET_RATIOS.items()}
    missing = size - sum(counts.values())
    order = ["normal", "hard_candidate", "no_history", "no_geo", "short_context"]
    for bucket in order[:missing]:
        counts[bucket] += 1
    return counts


def sample_stratified(
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    size: int,
    seed: int,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for pair in aligned:
        buckets[classify_bucket(*pair)].append(pair)
    for rows in buckets.values():
        rng.shuffle(rows)

    desired = target_counts(size)
    selected: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    selected_ids = set()
    bucket_selected = Counter()

    for bucket, want in desired.items():
        for pair in buckets.get(bucket, [])[:want]:
            sid = str(pair[0]["sample_id"])
            if sid not in selected_ids:
                selected.append(pair)
                selected_ids.add(sid)
                bucket_selected[bucket] += 1

    if len(selected) < size:
        leftovers = []
        for bucket in sorted(buckets):
            for pair in buckets[bucket]:
                if str(pair[0]["sample_id"]) not in selected_ids:
                    leftovers.append(pair)
        rng.shuffle(leftovers)
        for pair in leftovers[: size - len(selected)]:
            sid = str(pair[0]["sample_id"])
            selected.append(pair)
            selected_ids.add(sid)
            bucket_selected[classify_bucket(*pair)] += 1

    if len(selected) < size:
        raise ValueError(f"Requested {size} samples, only {len(selected)} available")

    rng.shuffle(selected)
    available = {bucket: len(rows) for bucket, rows in sorted(buckets.items())}
    stats = {
        "requested_size": size,
        "actual_size": len(selected),
        "seed": seed,
        "desired_bucket_counts": desired,
        "available_bucket_counts": available,
        "selected_bucket_counts": dict(bucket_selected),
    }
    return selected, stats


def main() -> None:
    args = parse_args()
    suffix = args.name or f"{args.split}_{args.size}"
    out_inputs = args.output_dir / f"teacher_distill_inputs_{suffix}.jsonl"
    out_labels = args.output_dir / f"teacher_distill_labels_{suffix}.jsonl"
    out_stats = args.output_dir / f"teacher_distill_sample_{suffix}.stats.json"
    for path in [out_inputs, out_labels, out_stats]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")

    inputs = read_jsonl(args.input)
    labels = read_jsonl(args.labels)
    aligned = align_rows(inputs, labels)
    selected, stats = sample_stratified(aligned, args.size, args.seed)

    write_jsonl(out_inputs, (inp for inp, _ in selected))
    write_jsonl(out_labels, (label for _, label in selected))
    stats.update(
        {
            "city": args.city,
            "split": args.split,
            "input": str(args.input),
            "labels": str(args.labels),
            "output_inputs": str(out_inputs),
            "output_labels": str(out_labels),
        }
    )
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
