#!/usr/bin/env python3
"""Build Stage1/Stage2/Stage3 SFT JSONL for the 8B next-POI predictor.

Stage1 uses RAW prompts only. Later stages can reuse the same schema by
joining LoRA-A refined prompts and setting route=REFINED.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROMPT_VERSION = "raw_v2_no_user_no_date"
SYSTEM_PROMPT = "You are a next-POI predictor. Output only JSON."
OUTPUT_FORMAT_BLOCK = 'Output format:\n{"next_poi_id":"<poi_id>"}'
REFINED_OUTPUT_FORMAT_BLOCK = OUTPUT_FORMAT_BLOCK + "\nReturn only this JSON object and no extra text."
FORBIDDEN_PROMPT_TOKENS = (
    "user_id",
    "sample_id",
    "trajectory_id",
    "target_poi_id",
    "target_category",
    "target_in_candidates",
    "label_difficulty_factors",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build POI SFT data for raw/refined next-POI prediction.")
    p.add_argument("--inputs", type=Path, required=True, help="teacher_distill_inputs_*.jsonl")
    p.add_argument("--labels", type=Path, required=True, help="teacher_distill_labels_*.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--refined-prompts", type=Path, default=None, help="Optional refiner_outputs_*.jsonl.")
    p.add_argument("--route", choices=["RAW", "REFINED"], default="RAW")
    p.add_argument("--candidate-limit", type=int, default=64)
    p.add_argument("--transition-limit", type=int, default=16)
    p.add_argument("--geo-limit", type=int, default=16)
    p.add_argument("--revisited-limit", type=int, default=10)
    p.add_argument("--top-category-limit", type=int, default=8)
    p.add_argument("--require-refined", action="store_true", help="For REFINED route, fail if any row lacks refined prompt.")
    p.add_argument("--max-refined-chars", type=int, default=1600)
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


def parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def weekday_name(value: Any) -> str:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    idx = safe_int(value, -1)
    return names[idx] if 0 <= idx < len(names) else "unknown"


def delta_minutes(prev: Dict[str, Any] | None, cur: Dict[str, Any]) -> int | None:
    if prev is None:
        return None
    prev_time = parse_time(prev.get("time"))
    cur_time = parse_time(cur.get("time"))
    if prev_time is None or cur_time is None:
        return None
    return int(round((cur_time - prev_time).total_seconds() / 60.0))


def evidence_quality(row: Dict[str, Any]) -> str:
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


def runtime_difficulty_factors(row: Dict[str, Any]) -> List[str]:
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
    if safe_int(pref.get("history_count")) <= 0:
        factors.append("no_history")
    if len(row.get("candidate_poi_ids") or []) >= 48:
        factors.append("many_candidates")
    return factors


def label_difficulty_factors(row: Dict[str, Any], label: Dict[str, Any]) -> List[str]:
    target = str(label.get("target_poi_id") or "")
    candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
    return ["target_not_in_candidates"] if target and target not in candidates else []


def poi_category_map(row: Dict[str, Any]) -> Dict[str, str]:
    evidence = row.get("evidence") or {}
    mapping: Dict[str, str] = {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}
    for item in seq.get("current_trajectory") or []:
        if item.get("poi_id"):
            mapping[str(item["poi_id"])] = str(item.get("category") or "unknown")
    for item in seq.get("transition_candidates") or []:
        if item.get("poi_id"):
            mapping[str(item["poi_id"])] = str(item.get("category") or "unknown")
    for item in geo.get("nearby_pois") or []:
        if item.get("poi_id"):
            mapping[str(item["poi_id"])] = str(item.get("category") or "unknown")
    for item in pref.get("revisited_pois") or []:
        if item.get("poi_id") and item.get("category"):
            mapping[str(item["poi_id"])] = str(item.get("category") or "unknown")
    return mapping


def fmt_count(value: Any) -> str:
    return str(safe_int(value))


def build_raw_prompt(row: Dict[str, Any], args: argparse.Namespace) -> str:
    evidence = row.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}
    trajectory = seq.get("current_trajectory") or []
    last = trajectory[-1] if trajectory else {}
    lines: List[str] = [
        "[ROUTE=RAW]",
        "",
        "Task:",
        f"Predict the next POI after the current trajectory in {row.get('city') or 'the city'}.",
        "The candidate list is useful evidence but is not guaranteed to contain the answer.",
        "Return exactly one POI id in JSON format.",
        "",
        "Constraints:",
        "The user is anonymized.",
        "Do not rely on stable user/sample/trajectory identifiers, absolute calendar date, or raw coordinates.",
        "Use only movement sequence, time slot, transition evidence, geographic evidence, and aggregated historical behavior.",
        "",
        "Current trajectory:",
    ]
    for idx, item in enumerate(trajectory, 1):
        delta = delta_minutes(trajectory[idx - 2] if idx > 1 else None, item)
        parts = [
            f"{idx}. poi={item.get('poi_id')}",
            f"category={item.get('category') or 'unknown'}",
            f"weekday={weekday_name(item.get('dow'))}",
            f"slot={safe_int(item.get('slot'))}",
        ]
        if delta is not None:
            parts.append(f"delta_minutes={delta}")
        lines.append(" | ".join(parts))
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
            distance = item.get("distance_km")
            try:
                distance_text = f"{float(distance):.3f}"
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

    mapping = poi_category_map(row)
    lines.append("revisited_pois:")
    revisited = pref.get("revisited_pois") or []
    if revisited:
        for item in revisited[: args.revisited_limit]:
            poi_id = str(item.get("poi_id") or "")
            category = mapping.get(poi_id, "unknown")
            lines.append(f"- poi={poi_id} | category={category} | count={fmt_count(item.get('count'))}")
    else:
        lines.append("- none")

    candidates = [str(x) for x in row.get("candidate_poi_ids") or []][: args.candidate_limit]
    lines.extend(["", "Candidate POIs:", ", ".join(candidates) if candidates else "none"])
    lines.extend(["", *OUTPUT_FORMAT_BLOCK.splitlines()])
    return "\n".join(lines)


def build_refined_prompt(raw_prompt: str, refined: str, max_chars: int) -> str:
    clean = " ".join(str(refined or "").split())
    if max_chars > 0:
        clean = clean[:max_chars]
    raw_body = raw_prompt.removeprefix("[ROUTE=RAW]").lstrip()
    if raw_body.endswith(OUTPUT_FORMAT_BLOCK):
        raw_body = raw_body[: -len(OUTPUT_FORMAT_BLOCK)].rstrip()
    return (
        f"[ROUTE=REFINED]\n{raw_body}"
        f"\n\n[Refined Evidence]\n{clean}"
        f"\n\n{REFINED_OUTPUT_FORMAT_BLOCK}"
    )


def assert_prompt_safe(prompt: str, sample_id: str) -> None:
    lower = prompt.lower()
    for token in FORBIDDEN_PROMPT_TOKENS:
        if token.lower() in lower:
            raise ValueError(f"{sample_id} prompt contains forbidden token: {token}")
    if "[Refined Evidence]" in prompt:
        refined_idx = prompt.find("[Refined Evidence]")
        output_idx = prompt.rfind("Output format:")
        if output_idx < refined_idx:
            raise ValueError(f"{sample_id} refined prompt must place Output format after Refined Evidence")
        if prompt.count("Output format:") != 1:
            raise ValueError(f"{sample_id} refined prompt must contain exactly one Output format block")


def make_record(
    row: Dict[str, Any],
    label: Dict[str, Any],
    route: str,
    input_prompt: str,
    refined_prompt: str | None,
) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    candidates = [str(x) for x in row.get("candidate_poi_ids") or []]
    metadata = {
        "evidence_quality": evidence_quality(row),
        "difficulty_factors": runtime_difficulty_factors(row),
        "label_difficulty_factors": label_difficulty_factors(row, label),
        "target_in_candidates": str(label.get("target_poi_id") or "") in set(candidates),
        "trajectory_len": len(((row.get("evidence") or {}).get("sequence") or {}).get("current_trajectory") or []),
        "candidate_count": len(candidates),
        "transition_candidate_count": len(((row.get("evidence") or {}).get("sequence") or {}).get("transition_candidates") or []),
        "geo_nearby_count": len(((row.get("evidence") or {}).get("geo") or {}).get("nearby_pois") or []),
        "history_count": safe_int(((row.get("evidence") or {}).get("preference") or {}).get("history_count")),
    }
    assistant = json.dumps({"next_poi_id": str(label["target_poi_id"])}, ensure_ascii=False, separators=(",", ":"))
    return {
        "sample_id": sample_id,
        "split": row.get("split"),
        "city": row.get("city"),
        "route": route,
        "prompt_version": PROMPT_VERSION,
        "input_prompt": input_prompt,
        "refined_prompt": refined_prompt,
        "target": {
            "poi_id": str(label["target_poi_id"]),
            "category": label.get("target_category"),
        },
        "candidate_poi_ids": candidates,
        "metadata": metadata,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": assistant},
        ],
    }


def load_refined(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    refined: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        text = str(row.get("distilled_prompt") or "").strip()
        if sample_id and text:
            if sample_id in refined:
                raise ValueError(f"{path} duplicate sample_id: {sample_id}")
            refined[sample_id] = row
    return refined


def assert_same_identity(row: Dict[str, Any], other: Dict[str, Any], sample_id: str, source: str) -> None:
    for key in ("trajectory_id", "user_id", "split", "city"):
        left = row.get(key)
        right = other.get(key)
        if left is None or right is None:
            continue
        if str(left) != str(right):
            raise ValueError(
                f"{sample_id} {source} {key} mismatch: input={left!r}, {source}={right!r}"
            )


def build_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    inputs = read_jsonl(args.inputs)
    labels = load_by_id(args.labels)
    refined_by_id = load_refined(args.refined_prompts)
    records: List[Dict[str, Any]] = []
    stats = Counter()

    for row in inputs:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("Input row without sample_id")
        label = labels.get(sample_id)
        if label is None:
            raise ValueError(f"Missing label for {sample_id}")
        assert_same_identity(row, label, sample_id, "label")
        raw_prompt = build_raw_prompt(row, args)
        if args.route == "RAW":
            input_prompt = raw_prompt
            refined_prompt = None
        else:
            refined_row = refined_by_id.get(sample_id)
            if refined_row is None:
                if args.require_refined:
                    raise ValueError(f"Missing refined prompt for {sample_id}")
                stats["skipped_missing_refined"] += 1
                continue
            assert_same_identity(row, refined_row, sample_id, "refined")
            refined_prompt = str(refined_row.get("distilled_prompt") or "").strip()
            input_prompt = build_refined_prompt(raw_prompt, refined_prompt, args.max_refined_chars)
        assert_prompt_safe(input_prompt, sample_id)
        record = make_record(row, label, args.route, input_prompt, refined_prompt)
        records.append(record)
        stats[f"route_{args.route.lower()}"] += 1
        stats[f"quality_{record['metadata']['evidence_quality']}"] += 1
        if record["metadata"]["target_in_candidates"]:
            stats["target_in_candidates"] += 1
        for factor in record["metadata"]["difficulty_factors"]:
            stats[f"difficulty_{factor}"] += 1
        for factor in record["metadata"]["label_difficulty_factors"]:
            stats[f"label_difficulty_{factor}"] += 1

    missing_labels = set(labels) - {str(row.get("sample_id") or "") for row in inputs}
    if missing_labels:
        stats["labels_without_input"] = len(missing_labels)
    summary = {
        "input_rows": len(inputs),
        "output_rows": len(records),
        "route": args.route,
        "prompt_version": PROMPT_VERSION,
        "counts": dict(stats),
    }
    return records, summary


def main() -> None:
    args = parse_args()
    records, summary = build_records(args)
    written = write_jsonl(args.output, records, args.overwrite)
    summary["written"] = written
    summary_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
