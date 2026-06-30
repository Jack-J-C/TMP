#!/usr/bin/env python3
"""Build LoRA-C SFT data for supplemental candidate generation.

LoRA-C learns a narrow task:
    label-free evidence + original candidates -> supplemental_poi_ids

The user prompt never contains target labels. Labels are used only to build the
assistant supervision target. For samples whose target is already in the
original candidates, the default target is an empty supplemental list. For
missed samples, the target POI is emitted as the first supplemental candidate.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = "You are a candidate expansion model for next-POI prediction. Output only JSON."
OUTPUT_SCHEMA = '{"supplemental_poi_ids":["<poi_id>"]}'
FORBIDDEN_PROMPT_TOKENS = (
    "target_poi_id",
    "target_category",
    "answer",
    "gold",
    "ground truth",
    "correct answer",
)
POI_RE = re.compile(r"^v\d+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LoRA-C supplemental-candidate SFT JSONL.")
    p.add_argument("--inputs", type=Path, required=True, help="Label-free teacher_distill_inputs_*.jsonl")
    p.add_argument("--labels", type=Path, required=True, help="teacher_distill_labels_*.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-limit", type=int, default=64, help="Number of original candidates shown to LoRA-C.")
    p.add_argument("--transition-limit", type=int, default=20)
    p.add_argument("--geo-limit", type=int, default=20)
    p.add_argument("--top-category-limit", type=int, default=8)
    p.add_argument("--revisited-limit", type=int, default=20)
    p.add_argument("--miss-oversample", type=int, default=2, help="Duplicate target-missed rows this many times.")
    p.add_argument(
        "--only-missed",
        action="store_true",
        help="Only keep samples whose target is not in the original candidate list.",
    )
    p.add_argument(
        "--include-hit-target",
        action="store_true",
        help="Also train hit rows to output the target. Default trains hit rows to output an empty list.",
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
            raise ValueError(f"{path} contains row without sample_id")
        if sample_id in by_id:
            raise ValueError(f"{path} duplicate sample_id: {sample_id}")
        by_id[sample_id] = row
    return by_id


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def weekday_name(value: Any) -> str:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    idx = safe_int(value, -1)
    return names[idx] if 0 <= idx < len(names) else "unknown"


def assert_prompt_safe(prompt: str, sample_id: str) -> None:
    lower = prompt.lower()
    for token in FORBIDDEN_PROMPT_TOKENS:
        if token in lower:
            raise ValueError(f"{sample_id} LoRA-C prompt contains forbidden label token: {token}")


def fmt_count(value: Any) -> str:
    return str(safe_int(value))


def build_prompt(row: Dict[str, Any], args: argparse.Namespace) -> str:
    evidence = row.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}
    trajectory = seq.get("current_trajectory") or []
    last = trajectory[-1] if trajectory else {}
    candidates = [str(x) for x in row.get("candidate_poi_ids") or []][: args.candidate_limit]

    lines: List[str] = [
        "Task:",
        "Propose supplemental POI candidates that may be missing from the original candidate list.",
        "Use only the evidence below. Do not explain.",
        "Return an empty list if no supplemental candidate is needed.",
        "",
        "Output format:",
        OUTPUT_SCHEMA,
        "",
        "Current trajectory:",
    ]
    for idx, item in enumerate(trajectory, 1):
        lines.append(
            " | ".join(
                [
                    f"{idx}. poi={item.get('poi_id')}",
                    f"category={item.get('category') or 'unknown'}",
                    f"weekday={weekday_name(item.get('dow'))}",
                    f"slot={safe_int(item.get('slot'))}",
                ]
            )
        )
    if not trajectory:
        lines.append("- empty trajectory")

    lines.extend(
        [
            "",
            "Short-term pattern:",
            f"trajectory_len={len(trajectory)}",
            f"last_category={last.get('category') or 'unknown'}",
            f"summary={seq.get('sequence_summary') or 'No sequence summary available.'}",
            "",
            "Transition evidence:",
        ]
    )
    transitions = seq.get("transition_candidates") or []
    if transitions:
        for item in transitions[: args.transition_limit]:
            lines.append(
                "- "
                f"poi={item.get('poi_id')} | "
                f"category={item.get('category') or 'unknown'} | "
                f"source={item.get('source') or 'unknown'} | "
                f"count={fmt_count(item.get('count'))}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "Geographic evidence near the last POI:"])
    nearby = geo.get("nearby_pois") or []
    if nearby:
        for item in nearby[: args.geo_limit]:
            try:
                distance_text = f"{float(item.get('distance_km')):.3f}"
            except (TypeError, ValueError):
                distance_text = "unknown"
            lines.append(
                "- "
                f"poi={item.get('poi_id')} | "
                f"category={item.get('category') or 'unknown'} | "
                f"distance_km={distance_text}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Long-term preference evidence:",
            f"history_count={safe_int(pref.get('history_count'))}",
            "top_categories:",
        ]
    )
    top_categories = pref.get("top_categories") or []
    if top_categories:
        for item in top_categories[: args.top_category_limit]:
            lines.append(f"- category={item.get('category') or 'unknown'} | count={fmt_count(item.get('count'))}")
    else:
        lines.append("- none")

    lines.append("revisited_pois:")
    revisited = pref.get("revisited_pois") or []
    if revisited:
        for item in revisited[: args.revisited_limit]:
            lines.append(f"- poi={item.get('poi_id')} | count={fmt_count(item.get('count'))}")
    else:
        lines.append("- none")

    lines.extend(["", "Original candidate POIs:", ", ".join(candidates) if candidates else "none"])
    return "\n".join(lines)


def target_supplemental(row: Dict[str, Any], label: Dict[str, Any], include_hit_target: bool) -> List[str]:
    target = str(label.get("target_poi_id") or "")
    if not POI_RE.fullmatch(target):
        raise ValueError(f"{row.get('sample_id')} invalid target_poi_id: {target!r}")
    candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
    if target in candidates and not include_hit_target:
        return []
    return [target]


def build_records(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    inputs = read_jsonl(args.inputs)
    labels = load_by_id(args.labels)
    records: List[Dict[str, Any]] = []
    stats = Counter()

    for row in inputs:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{args.inputs} contains row without sample_id")
        label = labels.get(sample_id)
        if label is None:
            raise ValueError(f"Missing label for {sample_id}")
        target = str(label.get("target_poi_id") or "")
        original_candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
        original_hit = target in original_candidates
        if args.only_missed and original_hit:
            stats["skipped_original_hit"] += 1
            continue
        prompt = build_prompt(row, args)
        assert_prompt_safe(prompt, sample_id)
        supplemental = target_supplemental(row, label, args.include_hit_target)
        target_in_candidates = not supplemental
        repeat = max(1, args.miss_oversample if not target_in_candidates else 1)
        assistant = json.dumps({"supplemental_poi_ids": supplemental}, ensure_ascii=False, separators=(",", ":"))
        for repeat_idx in range(repeat):
            record_id = sample_id if repeat_idx == 0 else f"{sample_id}#boost{repeat_idx}"
            records.append(
                {
                    "sample_id": record_id,
                    "source_sample_id": sample_id,
                    "split": row.get("split"),
                    "city": row.get("city"),
                    "task": "candidate_booster",
                    "target_in_original_candidates": target_in_candidates,
                    "original_candidate_count": len(row.get("candidate_poi_ids") or []),
                    "input_prompt": prompt,
                    "candidate_poi_ids": [str(x) for x in row.get("candidate_poi_ids") or []],
                    "target": {
                        "poi_id": str(label.get("target_poi_id") or ""),
                        "category": label.get("target_category"),
                    },
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                }
            )
        stats["input_rows"] += 1
        stats["output_rows"] += repeat
        stats["target_in_original_candidates"] += int(target_in_candidates)
        stats["target_missed_original_candidates"] += int(not target_in_candidates)
        stats["miss_oversampled_extra_rows"] += repeat - 1

    summary = {
        "input_rows": stats["input_rows"],
        "skipped_original_hit": stats["skipped_original_hit"],
        "output_rows": stats["output_rows"],
        "target_in_original_candidates": stats["target_in_original_candidates"],
        "target_missed_original_candidates": stats["target_missed_original_candidates"],
        "target_missed_ratio": round(stats["target_missed_original_candidates"] / stats["input_rows"], 6)
        if stats["input_rows"]
        else 0.0,
        "miss_oversampled_extra_rows": stats["miss_oversampled_extra_rows"],
        "miss_oversample": args.miss_oversample,
        "include_hit_target": args.include_hit_target,
        "only_missed": args.only_missed,
    }
    return records, summary


def main() -> None:
    args = parse_args()
    records, summary = build_records(args)
    written = write_jsonl(args.output, records, args.overwrite)
    summary["written"] = written
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
