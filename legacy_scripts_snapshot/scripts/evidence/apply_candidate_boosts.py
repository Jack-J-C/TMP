#!/usr/bin/env python3
"""Merge LoRA-C supplemental candidates into label-free teacher inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply supplemental candidates to teacher_distill_inputs JSONL.")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--boosts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None, help="Optional labels for recall reporting only.")
    p.add_argument("--candidate-limit", type=int, default=96)
    p.add_argument("--supplemental-limit", type=int, default=32)
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_boosts(path: Path, supplemental_limit: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} contains boost row without sample_id")
        values = []
        for poi in row.get("supplemental_poi_ids") or []:
            text = str(poi)
            if text.startswith("v") and text[1:].isdigit() and text not in values:
                values.append(text)
            if len(values) >= supplemental_limit:
                break
        out[sid] = values
    return out


def load_labels(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    labels: Dict[str, str] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        target = str(row.get("target_poi_id") or "")
        if sid and target:
            labels[sid] = target
    return labels


def merge_candidates(original: List[Any], supplemental: List[str], candidate_limit: int) -> List[str]:
    merged: List[str] = []
    for value in [*original, *supplemental]:
        text = str(value)
        if text and text not in merged:
            merged.append(text)
        if len(merged) >= candidate_limit:
            break
    return merged


def main() -> None:
    args = parse_args()
    inputs = read_jsonl(args.inputs)
    boosts = load_boosts(args.boosts, args.supplemental_limit)
    labels = load_labels(args.labels)

    output_rows: List[Dict[str, Any]] = []
    original_hits = 0
    boosted_hits = 0
    rows_with_boosts = 0
    added_total = 0
    for row in inputs:
        sid = str(row.get("sample_id") or "")
        original = [str(x) for x in row.get("candidate_poi_ids") or []]
        supplemental = boosts.get(sid, [])
        merged = merge_candidates(original, supplemental, args.candidate_limit)
        new_row = dict(row)
        new_row["candidate_poi_ids"] = merged
        new_row["candidate_boost"] = {
            "source": str(args.boosts),
            "supplemental_poi_ids": [poi for poi in supplemental if poi not in set(original)],
            "original_candidate_count": len(original),
            "boosted_candidate_count": len(merged),
        }
        output_rows.append(new_row)
        added = len(merged) - len(original)
        added_total += max(0, added)
        rows_with_boosts += int(added > 0)
        target = labels.get(sid)
        if target:
            original_hits += int(target in set(original))
            boosted_hits += int(target in set(merged))

    written = write_jsonl(args.output, output_rows, args.overwrite)
    report = {
        "input_rows": len(inputs),
        "written": written,
        "boost_rows": len(boosts),
        "rows_with_added_candidates": rows_with_boosts,
        "avg_added_candidates": round(added_total / len(inputs), 4) if inputs else 0.0,
        "candidate_limit": args.candidate_limit,
    }
    if labels:
        report.update(
            {
                "original_target_in_candidates": original_hits,
                "boosted_target_in_candidates": boosted_hits,
                "original_target_in_candidates_ratio": round(original_hits / len(inputs), 6) if inputs else 0.0,
                "boosted_target_in_candidates_ratio": round(boosted_hits / len(inputs), 6) if inputs else 0.0,
                "absolute_recall_gain": round((boosted_hits - original_hits) / len(inputs), 6) if inputs else 0.0,
            }
        )
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
