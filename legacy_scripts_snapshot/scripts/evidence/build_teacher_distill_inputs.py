#!/usr/bin/env python3
"""Build label-free teacher distillation inputs from raw evidence JSONL files.

The output inputs are safe to send to a teacher model: target/answer fields are
removed recursively. Labels are written to a separate file for later training
or validation joins.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


LABEL_KEYS = {
    "answer",
    "gold",
    "gold_answer",
    "label",
    "target",
    "target_category",
    "target_poi",
    "target_poi_id",
    "true_next",
    "true_next_poi",
    "true_next_poi_id",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create label-free teacher distillation inputs from sequence/geo/preference evidence."
    )
    p.add_argument("--cities", nargs="+", default=["NewYork"])
    p.add_argument("--evidence-root", type=Path, default=Path("retrieval_assets"))
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--candidate-limit", type=int, default=64)
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
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_label_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_label_fields(item)
            for key, item in value.items()
            if key.lower() not in LABEL_KEYS
        }
    if isinstance(value, list):
        return [strip_label_fields(item) for item in value]
    return value


def ensure_aligned(split: str, seq_rows: List[Dict[str, Any]], geo_rows: List[Dict[str, Any]], pref_rows: List[Dict[str, Any]]) -> None:
    counts = {"sequence": len(seq_rows), "geo": len(geo_rows), "preference": len(pref_rows)}
    if len(set(counts.values())) != 1:
        raise ValueError(f"{split} row count mismatch: {counts}")

    seen = set()
    for idx, (seq, geo, pref) in enumerate(zip(seq_rows, geo_rows, pref_rows), 1):
        sid = seq.get("sample_id")
        if not sid:
            raise ValueError(f"{split}:{idx} missing sequence sample_id")
        if sid in seen:
            raise ValueError(f"{split}:{idx} duplicate sample_id {sid}")
        seen.add(sid)
        if geo.get("sample_id") != sid or pref.get("sample_id") != sid:
            raise ValueError(
                f"{split}:{idx} sample_id mismatch: sequence={sid}, geo={geo.get('sample_id')}, "
                f"preference={pref.get('sample_id')}"
            )


def append_unique(values: List[str], value: Any, limit: int) -> None:
    if value is None:
        return
    text = str(value)
    if not text or text in values:
        return
    if len(values) < limit:
        values.append(text)


def collect_candidate_pois(seq: Dict[str, Any], geo: Dict[str, Any], pref: Dict[str, Any], limit: int) -> List[str]:
    candidates: List[str] = []
    for item in seq.get("transition_candidates") or []:
        append_unique(candidates, item.get("poi_id"), limit)
    for item in geo.get("nearby_pois") or []:
        append_unique(candidates, item.get("poi_id"), limit)
    for item in pref.get("revisited_pois") or []:
        append_unique(candidates, item.get("poi_id"), limit)
    for item in seq.get("current_trajectory") or []:
        append_unique(candidates, item.get("poi_id"), limit)
    return candidates


def build_teacher_instruction(candidate_limit: int) -> str:
    return (
        "You are preparing evidence for a next-POI prediction model. "
        "Read the short-term sequence evidence, geographic evidence, and long-term preference evidence. "
        "Write a concise, label-free reasoning prompt that helps another model predict the next POI. "
        "Do not claim that any candidate is the correct answer. Do not mention gold labels, targets, or answers. "
        f"Use at most {candidate_limit} candidate POIs if candidates are useful."
    )


def contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in LABEL_KEYS:
                return True
            if contains_forbidden_label_key(item):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_label_key(item) for item in value)
    return False


def build_split(city: str, split: str, evidence_dir: Path, candidate_limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    seq_rows = read_jsonl(evidence_dir / f"sequence_evidence_{split}.jsonl")
    geo_rows = read_jsonl(evidence_dir / f"geo_evidence_{split}.jsonl")
    pref_rows = read_jsonl(evidence_dir / f"preference_evidence_{split}.jsonl")
    ensure_aligned(split, seq_rows, geo_rows, pref_rows)

    input_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    stats = Counter()
    candidate_sizes: List[int] = []

    instruction = build_teacher_instruction(candidate_limit)
    for seq, geo, pref in zip(seq_rows, geo_rows, pref_rows):
        sample_id = seq["sample_id"]
        target_poi_id = seq.get("target_poi_id")
        target_category = seq.get("target_category")
        if not target_poi_id:
            raise ValueError(f"{split}:{sample_id} missing target_poi_id in sequence evidence")

        clean_seq = strip_label_fields(seq)
        clean_geo = strip_label_fields(geo)
        clean_pref = strip_label_fields(pref)
        candidates = collect_candidate_pois(clean_seq, clean_geo, clean_pref, candidate_limit)
        candidate_sizes.append(len(candidates))

        row = {
            "sample_id": sample_id,
            "city": city,
            "split": split,
            "user_id": seq.get("user_id"),
            "trajectory_id": seq.get("trajectory_id"),
            "instruction": instruction,
            "evidence": {
                "sequence": clean_seq,
                "geo": clean_geo,
                "preference": clean_pref,
            },
            "candidate_poi_ids": candidates,
        }
        if contains_forbidden_label_key(row):
            raise ValueError(f"{split}:{sample_id} sanitized teacher input still contains a label key")

        input_rows.append(row)
        label_rows.append(
            {
                "sample_id": sample_id,
                "city": city,
                "split": split,
                "user_id": seq.get("user_id"),
                "trajectory_id": seq.get("trajectory_id"),
                "target_poi_id": target_poi_id,
                "target_category": target_category,
            }
        )

        if not candidates:
            stats["empty_candidates"] += 1
        if target_poi_id in candidates:
            stats["target_in_candidates"] += 1
        if not (geo.get("nearby_pois") or []):
            stats["empty_geo"] += 1
        if int(pref.get("history_count") or 0) == 0:
            stats["empty_preference_history"] += 1
        if not (seq.get("transition_candidates") or []):
            stats["empty_transition_candidates"] += 1

    total = len(input_rows)
    split_stats = {
        "samples": total,
        "avg_candidates": round(sum(candidate_sizes) / total, 4) if total else 0.0,
        "max_candidates": max(candidate_sizes) if candidate_sizes else 0,
        "empty_candidates": int(stats["empty_candidates"]),
        "target_in_candidates": int(stats["target_in_candidates"]),
        "target_in_candidates_ratio": round(stats["target_in_candidates"] / total, 6) if total else 0.0,
        "empty_geo": int(stats["empty_geo"]),
        "empty_preference_history": int(stats["empty_preference_history"]),
        "empty_transition_candidates": int(stats["empty_transition_candidates"]),
        "label_keys_removed": sorted(LABEL_KEYS),
    }
    return input_rows, label_rows, split_stats


def main() -> None:
    args = parse_args()
    all_stats: Dict[str, Any] = {}

    for city in args.cities:
        evidence_dir = args.evidence_root / city / "evidence"
        if not evidence_dir.exists():
            raise FileNotFoundError(f"Evidence directory does not exist: {evidence_dir}")

        city_stats: Dict[str, Any] = {}
        for split in args.splits:
            input_path = evidence_dir / f"teacher_distill_inputs_{split}.jsonl"
            label_path = evidence_dir / f"teacher_distill_labels_{split}.jsonl"
            if not args.overwrite and (input_path.exists() or label_path.exists()):
                raise FileExistsError(f"Output exists for {city}/{split}; pass --overwrite to replace it")

            input_rows, label_rows, split_stats = build_split(city, split, evidence_dir, args.candidate_limit)
            write_jsonl(input_path, input_rows)
            write_jsonl(label_path, label_rows)
            city_stats[split] = {
                **split_stats,
                "input_path": str(input_path),
                "label_path": str(label_path),
            }
            print(json.dumps({"city": city, "split": split, **city_stats[split]}, ensure_ascii=False))

        stats_path = evidence_dir / "teacher_distill_stats.json"
        stats_path.write_text(json.dumps({"city": city, "splits": city_stats}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_stats[city] = city_stats

    summary_path = args.evidence_root / "teacher_distill_summary.json"
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
