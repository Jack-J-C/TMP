#!/usr/bin/env python3
"""Build LoRA-C SFT data with an explicit expansion candidate pool.

LoRA-C should not invent unseen POI ids. This builder first creates a
train-derived expansion index, then asks the model to select supplemental POIs
only from a visible Expansion candidate pool.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SYSTEM_PROMPT = (
    "You are a candidate expansion model for next-POI prediction. "
    "Output only JSON."
)
OUTPUT_SCHEMA = '{"supplemental_poi_ids":["<poi_id>"]}'
POI_RE = re.compile(r"^v\d+$")
FORBIDDEN_PROMPT_TOKENS = (
    "target_poi_id",
    "target_category",
    "answer",
    "gold",
    "ground truth",
    "correct answer",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pool-based LoRA-C SFT JSONL.")
    p.add_argument("--inputs", type=Path, required=True, help="Split inputs to convert.")
    p.add_argument("--labels", type=Path, required=True, help="Labels for split supervision.")
    p.add_argument("--index-inputs", type=Path, required=True, help="Train inputs used to build expansion indexes.")
    p.add_argument("--index-labels", type=Path, required=True, help="Train labels used to build expansion indexes.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-limit", type=int, default=64)
    p.add_argument("--pool-limit", type=int, default=64)
    p.add_argument("--transition-limit", type=int, default=20)
    p.add_argument("--geo-limit", type=int, default=20)
    p.add_argument("--top-category-limit", type=int, default=8)
    p.add_argument("--revisited-limit", type=int, default=20)
    p.add_argument("--last-poi-pool", type=int, default=32)
    p.add_argument("--last-category-pool", type=int, default=48)
    p.add_argument("--category-pool", type=int, default=8)
    p.add_argument("--global-pool", type=int, default=16)
    p.add_argument("--only-missed", action="store_true")
    p.add_argument(
        "--require-target-in-pool",
        action="store_true",
        help="Skip rows whose target is not naturally present in the expansion pool.",
    )
    p.add_argument(
        "--force-target-in-pool",
        action="store_true",
        help="Append target to pool for supervision when natural expansion misses it. Use for diagnostics only.",
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
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
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
            raise ValueError(f"{sample_id} prompt contains forbidden label token: {token}")


def fmt_count(value: Any) -> str:
    return str(safe_int(value))


def evidence_parts(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    evidence = row.get("evidence") or {}
    return evidence.get("sequence") or {}, evidence.get("geo") or {}, evidence.get("preference") or {}


def visible_categories(row: Dict[str, Any]) -> List[str]:
    seq, geo, pref = evidence_parts(row)
    cats: List[str] = []
    for item in seq.get("current_trajectory") or []:
        cats.append(str(item.get("category") or ""))
    for item in seq.get("transition_candidates") or []:
        cats.append(str(item.get("category") or ""))
    for item in geo.get("nearby_pois") or []:
        cats.append(str(item.get("category") or ""))
    for item in pref.get("top_categories") or []:
        cats.append(str(item.get("category") or ""))
    out: List[str] = []
    for cat in cats:
        if cat and cat != "unknown" and cat not in out:
            out.append(cat)
    return out


def add_poi_category(mapping: Dict[str, str], poi: Any, category: Any) -> None:
    poi_id = str(poi or "")
    cat = str(category or "")
    if POI_RE.fullmatch(poi_id) and cat and cat != "unknown":
        mapping.setdefault(poi_id, cat)


class ExpansionIndex:
    def __init__(self) -> None:
        self.last_poi_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.last_category_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.category_popular: Dict[str, Counter[str]] = defaultdict(Counter)
        self.global_popular: Counter[str] = Counter()
        self.poi_category: Dict[str, str] = {}

    @staticmethod
    def from_rows(inputs: List[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> "ExpansionIndex":
        index = ExpansionIndex()
        for row in inputs:
            sample_id = str(row.get("sample_id") or "")
            label = labels.get(sample_id)
            if label is None:
                continue
            target = str(label.get("target_poi_id") or "")
            target_category = str(label.get("target_category") or "")
            if not POI_RE.fullmatch(target):
                continue
            seq, geo, pref = evidence_parts(row)
            trajectory = seq.get("current_trajectory") or []
            last = trajectory[-1] if trajectory else {}
            last_poi = str(last.get("poi_id") or "")
            last_category = str(last.get("category") or "")
            if POI_RE.fullmatch(last_poi):
                index.last_poi_next[last_poi][target] += 1
            if last_category and last_category != "unknown":
                index.last_category_next[last_category][target] += 1
            if target_category and target_category != "unknown":
                index.category_popular[target_category][target] += 1
                add_poi_category(index.poi_category, target, target_category)
            index.global_popular[target] += 1
            for item in trajectory:
                add_poi_category(index.poi_category, item.get("poi_id"), item.get("category"))
            for item in seq.get("transition_candidates") or []:
                add_poi_category(index.poi_category, item.get("poi_id"), item.get("category"))
            for item in geo.get("nearby_pois") or []:
                add_poi_category(index.poi_category, item.get("poi_id"), item.get("category"))
            for item in pref.get("revisited_pois") or []:
                add_poi_category(index.poi_category, item.get("poi_id"), item.get("category"))
        return index

    def category_of(self, poi: str) -> str:
        return self.poi_category.get(poi, "unknown")


def add_pool_item(
    pool: List[Dict[str, Any]],
    seen: set[str],
    poi: str,
    category: str,
    source: str,
    score: int,
    original_candidates: set[str],
    pool_limit: int,
) -> None:
    if len(pool) >= pool_limit:
        return
    if not POI_RE.fullmatch(poi) or poi in original_candidates or poi in seen:
        return
    seen.add(poi)
    pool.append({"poi_id": poi, "category": category or "unknown", "source": source, "score": int(score)})


def build_expansion_pool(row: Dict[str, Any], index: ExpansionIndex, args: argparse.Namespace) -> List[Dict[str, Any]]:
    seq, _, _ = evidence_parts(row)
    trajectory = seq.get("current_trajectory") or []
    last = trajectory[-1] if trajectory else {}
    last_poi = str(last.get("poi_id") or "")
    last_category = str(last.get("category") or "")
    original_candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
    pool: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for poi, score in index.last_poi_next.get(last_poi, Counter()).most_common(args.last_poi_pool):
        add_pool_item(pool, seen, poi, index.category_of(poi), "train_same_last_poi", score, original_candidates, args.pool_limit)
    for poi, score in index.last_category_next.get(last_category, Counter()).most_common(args.last_category_pool):
        add_pool_item(
            pool,
            seen,
            poi,
            index.category_of(poi),
            "train_same_last_category",
            score,
            original_candidates,
            args.pool_limit,
        )
    for category in visible_categories(row):
        for poi, score in index.category_popular.get(category, Counter()).most_common(args.category_pool):
            add_pool_item(
                pool,
                seen,
                poi,
                index.category_of(poi),
                f"train_category_popular:{category}",
                score,
                original_candidates,
                args.pool_limit,
            )
    for poi, score in index.global_popular.most_common(args.global_pool):
        add_pool_item(pool, seen, poi, index.category_of(poi), "train_global_popular", score, original_candidates, args.pool_limit)
    return pool


def build_prompt(row: Dict[str, Any], pool: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    seq, geo, pref = evidence_parts(row)
    trajectory = seq.get("current_trajectory") or []
    last = trajectory[-1] if trajectory else {}
    candidates = [str(x) for x in row.get("candidate_poi_ids") or []][: args.candidate_limit]

    lines: List[str] = [
        "Task:",
        "Select supplemental POI candidates from the Expansion candidate pool.",
        "Do not invent POI ids. Do not output next_poi_id.",
        "Only output POI ids that appear in the Expansion candidate pool.",
        "Return an empty list if no supplemental candidate is useful.",
        "",
        "Output format:",
        OUTPUT_SCHEMA,
        "",
        "Current trajectory:",
    ]
    for idx, item in enumerate(trajectory, 1):
        parts = [
            f"{idx}. poi={item.get('poi_id')}",
            f"category={item.get('category') or 'unknown'}",
            f"weekday={weekday_name(item.get('dow'))}",
            f"slot={safe_int(item.get('slot'))}",
        ]
        if idx > 1 and item.get("delta_minutes") is not None:
            parts.append(f"delta_minutes={safe_int(item.get('delta_minutes'))}")
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
    lines.extend(["", "Expansion candidate pool:"])
    if pool:
        for item in pool:
            lines.append(
                "- "
                f"poi={item['poi_id']} | "
                f"category={item['category']} | "
                f"source={item['source']} | "
                f"score={item['score']}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def build_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    inputs = read_jsonl(args.inputs)
    labels = load_by_id(args.labels)
    index = ExpansionIndex.from_rows(read_jsonl(args.index_inputs), load_by_id(args.index_labels))
    records: List[Dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for row in inputs:
        sample_id = str(row.get("sample_id") or "")
        label = labels.get(sample_id)
        if label is None:
            raise ValueError(f"Missing label for {sample_id}")
        target = str(label.get("target_poi_id") or "")
        target_category = label.get("target_category")
        if not POI_RE.fullmatch(target):
            raise ValueError(f"{sample_id} invalid target_poi_id: {target!r}")

        original_candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
        original_hit = target in original_candidates
        stats["input_rows"] += 1
        stats["original_hit"] += int(original_hit)
        if args.only_missed and original_hit:
            stats["skipped_original_hit"] += 1
            continue

        pool = build_expansion_pool(row, index, args)
        natural_pool_ids = {str(item["poi_id"]) for item in pool}
        natural_hit = target in natural_pool_ids
        stats["natural_pool_hit"] += int(natural_hit)
        if args.require_target_in_pool and not natural_hit:
            stats["skipped_target_not_in_pool"] += 1
            continue
        if args.force_target_in_pool and not original_hit and not natural_hit:
            add_pool_item(
                pool,
                natural_pool_ids,
                target,
                str(target_category or "unknown"),
                "oracle_training_target",
                1,
                original_candidates,
                args.pool_limit,
            )
            stats["forced_target_into_pool"] += 1

        final_pool_ids = {str(item["poi_id"]) for item in pool}
        supplemental = [target] if (not original_hit and target in final_pool_ids) else []
        if not supplemental:
            stats["empty_target"] += 1
        prompt = build_prompt(row, pool, args)
        assert_prompt_safe(prompt, sample_id)
        assistant = json.dumps({"supplemental_poi_ids": supplemental}, ensure_ascii=False, separators=(",", ":"))

        records.append(
            {
                "sample_id": sample_id,
                "source_sample_id": sample_id,
                "split": row.get("split"),
                "city": row.get("city"),
                "task": "candidate_booster_pool",
                "target_in_original_candidates": original_hit,
                "target_in_expansion_pool": target in final_pool_ids,
                "natural_target_in_expansion_pool": natural_hit,
                "original_candidate_count": len(original_candidates),
                "expansion_pool_count": len(pool),
                "input_prompt": prompt,
                "candidate_poi_ids": [str(x) for x in row.get("candidate_poi_ids") or []],
                "expansion_candidate_poi_ids": [str(item["poi_id"]) for item in pool],
                "target": {"poi_id": target, "category": target_category},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
        stats["output_rows"] += 1
        stats["non_empty_targets"] += int(bool(supplemental))
        stats["pool_items_total"] += len(pool)

    denom = max(1, stats["input_rows"] - stats["skipped_original_hit"])
    summary = {
        "input_rows": stats["input_rows"],
        "output_rows": stats["output_rows"],
        "only_missed": args.only_missed,
        "skipped_original_hit": stats["skipped_original_hit"],
        "original_hit": stats["original_hit"],
        "natural_pool_hit": stats["natural_pool_hit"],
        "natural_pool_hit_ratio_after_filter": round(stats["natural_pool_hit"] / denom, 6),
        "skipped_target_not_in_pool": stats["skipped_target_not_in_pool"],
        "forced_target_into_pool": stats["forced_target_into_pool"],
        "non_empty_targets": stats["non_empty_targets"],
        "non_empty_target_ratio": round(stats["non_empty_targets"] / stats["output_rows"], 6)
        if stats["output_rows"]
        else 0.0,
        "avg_pool_items": round(stats["pool_items_total"] / stats["output_rows"], 4) if stats["output_rows"] else 0.0,
        "pool_limit": args.pool_limit,
        "require_target_in_pool": args.require_target_in_pool,
        "force_target_in_pool": args.force_target_in_pool,
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
