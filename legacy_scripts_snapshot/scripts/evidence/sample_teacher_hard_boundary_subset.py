#!/usr/bin/env python3
"""Sample hard/boundary teacher distillation inputs using deterministic evidence stats.

This is used to add low-cost, high-value teacher samples after an initial
random/stratified distillation set. Labels are aligned and written for later
training joins, but labels are not used for hard/boundary sampling.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


QUALITY_ORDER = ("good", "partial", "weak")
DEFAULT_RATIOS = {"good": 0.20, "partial": 0.45, "weak": 0.35}
DEFAULT_HARD_FACTORS = (
    "short_context",
    "no_transition_candidates",
    "sparse_transition",
    "dispersed_transition",
    "no_geo",
    "sparse_geo",
    "no_history",
    "sparse_history",
    "dispersed_history",
    "many_candidates",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample hard/boundary teacher distillation inputs.")
    p.add_argument("--city", default="NewYork")
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--input", type=Path, required=True, help="Full teacher_distill_inputs_*.jsonl.")
    p.add_argument("--labels", type=Path, required=True, help="Full teacher_distill_labels_*.jsonl.")
    p.add_argument(
        "--exclude-inputs",
        type=Path,
        action="append",
        default=[],
        help="Existing teacher input JSONL whose sample_ids should be excluded. Can be repeated.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("retrieval_assets/NewYork/distill_decision"))
    p.add_argument("--name", default=None, help="Output suffix. Defaults to {split}_hard_{size}.")
    p.add_argument("--good-ratio", type=float, default=DEFAULT_RATIOS["good"])
    p.add_argument("--partial-ratio", type=float, default=DEFAULT_RATIOS["partial"])
    p.add_argument("--weak-ratio", type=float, default=DEFAULT_RATIOS["weak"])
    p.add_argument(
        "--hard-factors",
        default=",".join(DEFAULT_HARD_FACTORS),
        help="Comma-separated runtime hard factors used for priority sampling.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def sample_id(row: Dict[str, Any]) -> str:
    sid = str(row.get("sample_id") or "")
    if not sid:
        raise ValueError("Row without sample_id")
    return sid


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_quality(row: Dict[str, Any]) -> str:
    evidence = row.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}

    current_len = len(seq.get("current_trajectory") or [])
    transition_count = len(seq.get("transition_candidates") or [])
    nearby_count = len(geo.get("nearby_pois") or [])
    history_count = safe_int(pref.get("history_count"))
    top_cat_count = len(pref.get("top_categories") or [])
    revisit_count = len(pref.get("revisited_pois") or [])
    candidate_count = len(row.get("candidate_poi_ids") or [])

    score = 0
    if current_len >= 3:
        score += 2
    elif current_len >= 2:
        score += 1
    if transition_count >= 8:
        score += 2
    elif transition_count > 0:
        score += 1
    if nearby_count >= 8:
        score += 2
    elif nearby_count > 0:
        score += 1
    if history_count >= 10:
        score += 2
    elif history_count > 0:
        score += 1
    if top_cat_count > 0 or revisit_count > 0:
        score += 1
    if candidate_count >= 16:
        score += 1

    if score >= 7:
        return "good"
    if score >= 4:
        return "partial"
    return "weak"


def top_count(items: Sequence[Dict[str, Any]], key: str = "count") -> int:
    counts = [safe_int(item.get(key)) for item in items]
    return max(counts) if counts else 0


def infer_runtime_hard_factors(row: Dict[str, Any]) -> List[str]:
    evidence = row.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}

    trajectory = seq.get("current_trajectory") or []
    transitions = seq.get("transition_candidates") or []
    nearby = geo.get("nearby_pois") or []
    top_categories = pref.get("top_categories") or []
    history_count = safe_int(pref.get("history_count"))
    candidate_count = len(row.get("candidate_poi_ids") or [])
    factors: List[str] = []

    if len(trajectory) <= 1:
        factors.append("short_context")
    if not transitions:
        factors.append("no_transition_candidates")
    elif len(transitions) <= 3:
        factors.append("sparse_transition")
    if transitions and top_count(transitions) <= 2:
        factors.append("dispersed_transition")

    if not nearby:
        factors.append("no_geo")
    elif len(nearby) <= 3:
        factors.append("sparse_geo")

    if history_count <= 0:
        factors.append("no_history")
    elif history_count < 5:
        factors.append("sparse_history")
    if history_count > 0 and top_count(top_categories) <= 2:
        factors.append("dispersed_history")

    if candidate_count >= 48:
        factors.append("many_candidates")
    return factors


def target_counts(size: int, ratios: Dict[str, float]) -> Dict[str, int]:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Quality ratios must sum to a positive value")
    normalized = {key: ratios[key] / total for key in QUALITY_ORDER}
    counts = {key: int(size * normalized[key]) for key in QUALITY_ORDER}
    missing = size - sum(counts.values())
    for key in QUALITY_ORDER[:missing]:
        counts[key] += 1
    return counts


def load_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    labels = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in labels:
        sid = sample_id(row)
        if sid in by_id:
            raise ValueError(f"{path} duplicate sample_id: {sid}")
        by_id[sid] = row
    return by_id


def load_excluded(paths: Sequence[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            excluded.add(sample_id(row))
    return excluded


def sample_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(args.seed)
    hard_factor_set = {x.strip() for x in args.hard_factors.split(",") if x.strip()}
    if not hard_factor_set:
        raise ValueError("--hard-factors must contain at least one factor")

    labels = load_labels(args.labels)
    excluded = load_excluded(args.exclude_inputs)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    available_quality = Counter()
    available_hard = Counter()
    skipped_missing_label = 0
    skipped_excluded = 0

    for source_row in read_jsonl(args.input):
        sid = sample_id(source_row)
        if sid in excluded:
            skipped_excluded += 1
            continue
        if sid not in labels:
            skipped_missing_label += 1
            continue
        row = dict(source_row)
        quality = infer_quality(row)
        factors = infer_runtime_hard_factors(row)
        hard_score = sum(1 for factor in factors if factor in hard_factor_set)
        row["_sampling_quality"] = quality
        row["_sampling_hard_factors"] = factors
        row["_sampling_hard_score"] = hard_score
        available_quality[quality] += 1
        available_hard.update(factors)
        buckets[quality].append(row)

    for rows in buckets.values():
        rng.shuffle(rows)
        rows.sort(key=lambda item: item["_sampling_hard_score"], reverse=True)

    desired = target_counts(
        args.size,
        {
            "good": args.good_ratio,
            "partial": args.partial_ratio,
            "weak": args.weak_ratio,
        },
    )
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_quality = Counter()
    selected_hard = Counter()

    def add(row: Dict[str, Any]) -> bool:
        sid = sample_id(row)
        if sid in selected_ids:
            return False
        selected.append(row)
        selected_ids.add(sid)
        selected_quality[row["_sampling_quality"]] += 1
        selected_hard.update(row["_sampling_hard_factors"])
        return True

    for quality, want in desired.items():
        for row in buckets.get(quality, []):
            if selected_quality[quality] >= want:
                break
            add(row)

    if len(selected) < args.size:
        leftovers = [
            row
            for quality in QUALITY_ORDER
            for row in buckets.get(quality, [])
            if sample_id(row) not in selected_ids
        ]
        rng.shuffle(leftovers)
        leftovers.sort(key=lambda item: item["_sampling_hard_score"], reverse=True)
        for row in leftovers:
            if len(selected) >= args.size:
                break
            add(row)

    if len(selected) < args.size:
        raise ValueError(f"Requested {args.size} samples, only {len(selected)} available")

    rng.shuffle(selected)
    clean_inputs: List[Dict[str, Any]] = []
    clean_labels: List[Dict[str, Any]] = []
    for row in selected:
        sid = sample_id(row)
        row = dict(row)
        row.pop("_sampling_quality", None)
        row.pop("_sampling_hard_factors", None)
        row.pop("_sampling_hard_score", None)
        clean_inputs.append(row)
        clean_labels.append(labels[sid])

    stats = {
        "requested_size": args.size,
        "actual_size": len(clean_inputs),
        "seed": args.seed,
        "quality_ratios": {
            "good": args.good_ratio,
            "partial": args.partial_ratio,
            "weak": args.weak_ratio,
        },
        "desired_quality_counts": desired,
        "available_quality_counts_after_exclusion": dict(available_quality),
        "selected_quality_counts": dict(selected_quality),
        "available_hard_factor_counts_after_exclusion": dict(available_hard),
        "selected_hard_factor_counts": dict(selected_hard),
        "excluded_sample_ids": len(excluded),
        "skipped_excluded": skipped_excluded,
        "skipped_missing_label": skipped_missing_label,
    }
    return clean_inputs, clean_labels, stats


def main() -> None:
    args = parse_args()
    suffix = args.name or f"{args.split}_hard_{args.size}"
    out_inputs = args.output_dir / f"teacher_distill_inputs_{suffix}.jsonl"
    out_labels = args.output_dir / f"teacher_distill_labels_{suffix}.jsonl"
    out_stats = args.output_dir / f"teacher_distill_sample_{suffix}.stats.json"
    for path in [out_inputs, out_labels, out_stats]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")

    inputs, labels, stats = sample_rows(args)
    write_jsonl(out_inputs, inputs)
    write_jsonl(out_labels, labels)
    stats.update(
        {
            "city": args.city,
            "split": args.split,
            "input": str(args.input),
            "labels": str(args.labels),
            "exclude_inputs": [str(path) for path in args.exclude_inputs],
            "output_inputs": str(out_inputs),
            "output_labels": str(out_labels),
        }
    )
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
