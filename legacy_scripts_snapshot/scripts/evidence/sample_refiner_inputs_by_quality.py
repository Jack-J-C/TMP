#!/usr/bin/env python3
"""Sample label-free refiner inputs by evidence quality and hard factors.

This script is for LoRA-A batch inference cost control. It samples from full
teacher_distill_inputs_*.jsonl files and writes a smaller JSONL that can be
passed directly to generate_refined_prompts.py.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


QUALITY_ORDER = ("good", "partial", "weak")
DEFAULT_RATIOS = {"good": 0.60, "partial": 0.30, "weak": 0.10}
DEFAULT_HARD_FACTORS = (
    "short_context",
    "no_transition_candidates",
    "no_geo",
    "no_history",
    "many_candidates",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality-stratified sample LoRA-A refiner inputs.")
    p.add_argument("--input", type=Path, required=True, help="Full teacher_distill_inputs_*.jsonl.")
    p.add_argument("--output", type=Path, required=True, help="Sampled input JSONL for LoRA-A generation.")
    p.add_argument("--size", type=int, required=True)
    p.add_argument(
        "--teacher-outputs",
        type=Path,
        default=None,
        help="Deprecated and disabled. Sampling quality is always inferred from deterministic evidence statistics.",
    )
    p.add_argument("--labels", type=Path, default=None, help="Optional labels, used only for stats.")
    p.add_argument("--good-ratio", type=float, default=DEFAULT_RATIOS["good"])
    p.add_argument("--partial-ratio", type=float, default=DEFAULT_RATIOS["partial"])
    p.add_argument("--weak-ratio", type=float, default=DEFAULT_RATIOS["weak"])
    p.add_argument(
        "--hard-ratio",
        type=float,
        default=0.0,
        help=(
            "Within each quality bucket, reserve this fraction for runtime hard-factor samples. "
            "0 keeps the original quality-only sampling behavior."
        ),
    )
    p.add_argument(
        "--hard-factors",
        type=str,
        default=",".join(DEFAULT_HARD_FACTORS),
        help="Comma-separated runtime difficulty factors used for hard-focused sampling.",
    )
    p.add_argument(
        "--include-label-hard-factors",
        action="store_true",
        help=(
            "Allow label-derived hard factors such as target_not_in_candidates to affect sampling. "
            "Use only for train/val hard mining, never for test-time routing features."
        ),
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


def load_label_map(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def normalize_quality(value: Any) -> str:
    quality = str(value or "").strip().lower()
    if quality in QUALITY_ORDER:
        return quality
    return "partial"


def parse_factor_list(value: str) -> Tuple[str, ...]:
    factors = tuple(x.strip() for x in value.split(",") if x.strip())
    if not factors:
        raise ValueError("--hard-factors must contain at least one factor when --hard-ratio > 0")
    return factors


def infer_quality(row: Dict[str, Any]) -> str:
    evidence = row.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}

    current_len = len(seq.get("current_trajectory") or [])
    transition_count = len(seq.get("transition_candidates") or [])
    nearby_count = len(geo.get("nearby_pois") or [])
    history_count = int(pref.get("history_count") or 0)
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


def infer_runtime_difficulty_factors(row: Dict[str, Any]) -> List[str]:
    evidence = row.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}

    factors: List[str] = []
    if len(seq.get("current_trajectory") or []) <= 1:
        factors.append("short_context")
    if not (seq.get("transition_candidates") or []):
        factors.append("no_transition_candidates")
    if not (geo.get("nearby_pois") or []):
        factors.append("no_geo")
    if int(pref.get("history_count") or 0) <= 0:
        factors.append("no_history")
    if len(row.get("candidate_poi_ids") or []) >= 48:
        factors.append("many_candidates")
    return factors


def infer_label_difficulty_factors(row: Dict[str, Any], label: Dict[str, Any] | None = None) -> List[str]:
    factors: List[str] = []
    if label:
        target = str(label.get("target_poi_id") or "")
        candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
        if target and target not in candidates:
            factors.append("target_not_in_candidates")
    return factors


def target_counts(size: int, ratios: Dict[str, float]) -> Dict[str, int]:
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("Quality ratios must sum to a positive value")
    normalized = {key: value / total_ratio for key, value in ratios.items()}
    counts = {key: int(size * normalized[key]) for key in QUALITY_ORDER}
    missing = size - sum(counts.values())
    for key in QUALITY_ORDER[:missing]:
        counts[key] += 1
    return counts


def sample_by_quality(
    rows: List[Dict[str, Any]],
    label_by_id: Dict[str, Dict[str, Any]],
    size: int,
    ratios: Dict[str, float],
    hard_ratio: float,
    hard_factors: Tuple[str, ...],
    include_label_hard_factors: bool,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not 0.0 <= hard_ratio <= 1.0:
        raise ValueError("--hard-ratio must be in [0, 1]")
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    runtime_factors_all = Counter()
    label_factors_all = Counter()
    target_in_candidates = Counter()
    hard_factor_set = set(hard_factors)

    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("Input row without sample_id")
        quality = infer_quality(row)
        row = dict(row)
        row["evidence_quality"] = quality
        label = label_by_id.get(sample_id)
        runtime_factors = infer_runtime_difficulty_factors(row)
        label_factors = infer_label_difficulty_factors(row, label)
        row["difficulty_factors"] = runtime_factors
        if label_factors:
            row["label_difficulty_factors"] = label_factors
        runtime_factors_all.update(runtime_factors)
        label_factors_all.update(label_factors)
        sampling_factors = set(runtime_factors)
        if include_label_hard_factors:
            sampling_factors.update(label_factors)
        row["_hard_focus"] = bool(sampling_factors & hard_factor_set)
        if label:
            target = str(label.get("target_poi_id") or "")
            candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
            target_in_candidates["yes" if target in candidates else "no"] += 1
        buckets[quality].append(row)

    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    desired = target_counts(size, ratios)
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    selected_counts = Counter()
    selected_hard_counts = Counter()

    def add_row(row: Dict[str, Any]) -> bool:
        sample_id = str(row["sample_id"])
        if sample_id in selected_ids:
            return False
        selected.append(row)
        selected_ids.add(sample_id)
        quality = row["evidence_quality"]
        selected_counts[quality] += 1
        if row.get("_hard_focus"):
            selected_hard_counts[quality] += 1
        return True

    for quality in QUALITY_ORDER:
        want = desired[quality]
        bucket = buckets.get(quality, [])
        hard_want = int(round(want * hard_ratio))
        if hard_want > 0:
            for row in [x for x in bucket if x.get("_hard_focus")][:hard_want]:
                add_row(row)
        for row in bucket:
            if selected_counts[quality] >= want:
                break
            add_row(row)

    if len(selected) < size:
        leftovers: List[Dict[str, Any]] = []
        for quality in QUALITY_ORDER:
            leftovers.extend(row for row in buckets.get(quality, []) if str(row["sample_id"]) not in selected_ids)
        rng.shuffle(leftovers)
        for row in leftovers[: size - len(selected)]:
            add_row(row)

    if len(selected) < size:
        raise ValueError(f"Requested {size} samples, only {len(selected)} available")

    rng.shuffle(selected)
    selected_runtime_factors = Counter()
    selected_label_factors = Counter()
    for row in selected:
        selected_runtime_factors.update(row.get("difficulty_factors") or [])
        selected_label_factors.update(row.get("label_difficulty_factors") or [])

    stats = {
        "requested_size": size,
        "actual_size": len(selected),
        "seed": seed,
        "ratios": ratios,
        "hard_ratio": hard_ratio,
        "hard_factors": list(hard_factors),
        "include_label_hard_factors": include_label_hard_factors,
        "desired_quality_counts": desired,
        "available_quality_counts": {quality: len(buckets.get(quality, [])) for quality in QUALITY_ORDER},
        "selected_quality_counts": {quality: selected_counts.get(quality, 0) for quality in QUALITY_ORDER},
        "available_hard_focus_counts": {
            quality: sum(1 for row in buckets.get(quality, []) if row.get("_hard_focus")) for quality in QUALITY_ORDER
        },
        "selected_hard_focus_counts": {quality: selected_hard_counts.get(quality, 0) for quality in QUALITY_ORDER},
        "quality_source": "heuristic_only",
        "runtime_difficulty_factor_counts_all_inputs": dict(runtime_factors_all),
        "label_difficulty_factor_counts_all_inputs": dict(label_factors_all),
        "runtime_difficulty_factor_counts_selected": dict(selected_runtime_factors),
        "label_difficulty_factor_counts_selected": dict(selected_label_factors),
        "target_in_candidates_counts_all_inputs": dict(target_in_candidates),
    }
    for row in selected:
        row.pop("_hard_focus", None)
    return selected, stats


def main() -> None:
    args = parse_args()
    if args.teacher_outputs is not None:
        raise ValueError(
            "--teacher-outputs is disabled for sampling. "
            "Use deterministic evidence statistics to avoid teacher hallucination in difficulty labels."
        )
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    for path in [args.output, stats_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")

    ratios = {
        "good": args.good_ratio,
        "partial": args.partial_ratio,
        "weak": args.weak_ratio,
    }
    hard_factors = parse_factor_list(args.hard_factors)
    rows = read_jsonl(args.input)
    label_by_id = load_label_map(args.labels)
    selected, stats = sample_by_quality(
        rows,
        label_by_id,
        args.size,
        ratios,
        args.hard_ratio,
        hard_factors,
        args.include_label_hard_factors,
        args.seed,
    )

    write_jsonl(args.output, selected)
    stats.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "teacher_outputs": None,
            "labels": str(args.labels) if args.labels else None,
        }
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
