#!/usr/bin/env python3
"""Evaluate offline RAW/REFINED routing from paired generation files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RAW, REFINED, and confidence-based selector outputs.")
    p.add_argument("--raw-generations", type=Path, required=True)
    p.add_argument("--refined-generations", type=Path, required=True)
    p.add_argument("--refined-prompts", type=Path, default=None, help="Optional LoRA-A outputs with refiner_confidence.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confidence-threshold", choices=["high"], default="high")
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


def by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} row without sample_id")
        if sid in out:
            raise ValueError(f"{path} duplicate sample_id: {sid}")
        out[sid] = row
    return out


def confidence_for(row: Dict[str, Any], refined_prompt_by_id: Dict[str, Dict[str, Any]]) -> str:
    sid = str(row.get("sample_id") or "")
    metadata = row.get("metadata") or {}
    value = metadata.get("refiner_confidence")
    if value:
        return str(value).lower()
    prompt_row = refined_prompt_by_id.get(sid)
    if prompt_row:
        return str(prompt_row.get("refiner_confidence") or "unknown").lower()
    return "unknown"


def summarize(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    parse_ok = sum(1 for row in rows if row.get("parse_ok"))
    in_candidate = [row for row in rows if row.get("target_in_candidates")]
    correct_in = sum(1 for row in in_candidate if row["correct"])
    by_quality: Dict[str, Dict[str, Any]] = {}
    for quality in sorted({str(row.get("evidence_quality") or "unknown") for row in rows}):
        group = [row for row in rows if str(row.get("evidence_quality") or "unknown") == quality]
        by_quality[quality] = {
            "rows": len(group),
            "top1": round(sum(1 for row in group if row["correct"]) / len(group), 6) if group else 0.0,
        }
    return {
        f"{prefix}_rows": total,
        f"{prefix}_top1": round(correct / total, 6) if total else 0.0,
        f"{prefix}_parse_rate": round(parse_ok / total, 6) if total else 0.0,
        f"{prefix}_top1_when_target_in_candidates": round(correct_in / len(in_candidate), 6) if in_candidate else 0.0,
        f"{prefix}_by_evidence_quality": by_quality,
    }


def main() -> None:
    args = parse_args()
    raw_by_id = by_id(args.raw_generations)
    refined_by_id = by_id(args.refined_generations)
    prompt_by_id = by_id(args.refined_prompts) if args.refined_prompts else {}
    ids = sorted(set(raw_by_id) & set(refined_by_id))
    if not ids:
        raise ValueError("No paired sample_id between RAW and REFINED generations")

    rows = []
    selector_rows = []
    counts = Counter()
    confidence_counts = Counter()
    for sid in ids:
        raw = raw_by_id[sid]
        refined = refined_by_id[sid]
        conf = confidence_for(refined, prompt_by_id)
        confidence_counts[conf] += 1
        raw_correct = bool(raw.get("correct"))
        refined_correct = bool(refined.get("correct"))
        if raw_correct and refined_correct:
            counts["both_correct"] += 1
        elif raw_correct and not refined_correct:
            counts["refined_hurt"] += 1
        elif not raw_correct and refined_correct:
            counts["refined_fix"] += 1
        else:
            counts["both_wrong"] += 1
        use_refined = conf == args.confidence_threshold
        chosen = refined if use_refined else raw
        selector_rows.append(
            {
                **chosen,
                "selector_route": "REFINED" if use_refined else "RAW",
                "refiner_confidence": conf,
                "raw_pred_poi_id": raw.get("pred_poi_id"),
                "refined_pred_poi_id": refined.get("pred_poi_id"),
                "raw_correct": raw_correct,
                "refined_correct": refined_correct,
            }
        )
        rows.append({"raw": raw, "refined": refined, "confidence": conf})

    raw_rows = [raw_by_id[sid] for sid in ids]
    refined_rows = [refined_by_id[sid] for sid in ids]
    report: Dict[str, Any] = {
        "paired_rows": len(ids),
        "confidence_threshold": args.confidence_threshold,
        "confidence_counts": dict(confidence_counts),
        "paired_counts": dict(counts),
    }
    report.update(summarize(raw_rows, "raw"))
    report.update(summarize(refined_rows, "refined"))
    report.update(summarize(selector_rows, "selector"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
