#!/usr/bin/env python3
"""Mine Stage3 hard samples from Stage2 evaluation outputs.

The output is a label-free refiner input JSONL. It should be passed to
generate_refined_prompts.py so LoRA-A only regenerates prompts for samples that
the Stage2 model actually struggled with.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine hard samples for Stage3 hard-focused finetuning.")
    p.add_argument("--inputs", type=Path, required=True, help="Full or subset teacher_distill_inputs_*.jsonl.")
    p.add_argument("--eval-outputs", type=Path, required=True, help="Stage2 eval JSONL from evaluate_poi_lora.py.")
    p.add_argument("--output", type=Path, required=True, help="Hard refiner inputs JSONL.")
    p.add_argument(
        "--exclude-refined",
        type=Path,
        action="append",
        default=[],
        help="Optional existing refiner_outputs_*.jsonl to skip already refined sample_ids.",
    )
    p.add_argument("--rank-threshold", type=int, default=20, help="Select samples with candidate_rank greater than this.")
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-heuristic-hard", action="store_true", help="Also select correct samples with hard metadata.")
    p.add_argument("--hard-qualities", nargs="*", default=["partial", "weak"])
    p.add_argument(
        "--hard-factors",
        nargs="*",
        default=["short_context", "no_history", "no_geo", "no_transition_candidates", "many_candidates"],
    )
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


def load_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{path} row without sample_id")
        if sample_id in by_id:
            raise ValueError(f"{path} duplicate sample_id: {sample_id}")
        by_id[sample_id] = row
    return by_id


def load_refined_ids(paths: List[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                ids.add(sample_id)
    return ids


def is_false(value: Any) -> bool:
    return value is False or str(value).strip().lower() in {"false", "0", "no"}


def hard_reasons(row: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []
    pred = str(row.get("pred_poi_id") or "")
    rank = row.get("candidate_rank")
    quality = str(row.get("evidence_quality") or "").lower()
    factors = {str(x) for x in row.get("difficulty_factors") or []}

    if is_false(row.get("correct")):
        reasons.append("stage2_incorrect")
    if not pred:
        reasons.append("parse_fail")
    if rank is not None:
        try:
            if int(rank) > args.rank_threshold:
                reasons.append("poor_candidate_rank")
        except (TypeError, ValueError):
            pass

    if args.include_heuristic_hard:
        if quality in set(args.hard_qualities):
            reasons.append("hard_quality")
        if factors & set(args.hard_factors):
            reasons.append("hard_factor")

    return reasons


def main() -> None:
    args = parse_args()
    input_by_id = load_by_id(args.inputs)
    refined_ids = load_refined_ids(args.exclude_refined)

    selected: List[Dict[str, Any]] = []
    stats = Counter()
    seen = set()
    for ev in read_jsonl(args.eval_outputs):
        sample_id = str(ev.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        if sample_id in refined_ids:
            stats["excluded_existing_refined"] += 1
            continue
        inp = input_by_id.get(sample_id)
        if inp is None:
            stats["eval_without_input"] += 1
            continue
        reasons = hard_reasons(ev, args)
        if not reasons:
            continue
        row = dict(inp)
        row["hard_mining"] = {
            "source_eval": str(args.eval_outputs),
            "reasons": reasons,
            "stage2_pred_poi_id": ev.get("pred_poi_id"),
            "stage2_candidate_rank": ev.get("candidate_rank"),
            "stage2_evidence_quality": ev.get("evidence_quality"),
            "stage2_difficulty_factors": ev.get("difficulty_factors") or [],
        }
        selected.append(row)
        stats["selected"] += 1
        for reason in reasons:
            stats[f"reason_{reason}"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    if args.max_samples >= 0:
        selected = selected[: args.max_samples]

    written = write_jsonl(args.output, selected, args.overwrite)
    report = {
        "inputs": str(args.inputs),
        "eval_outputs": str(args.eval_outputs),
        "output": str(args.output),
        "written": written,
        "max_samples": args.max_samples,
        "rank_threshold": args.rank_threshold,
        "exclude_refined": [str(x) for x in args.exclude_refined],
        "counts": dict(stats),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
