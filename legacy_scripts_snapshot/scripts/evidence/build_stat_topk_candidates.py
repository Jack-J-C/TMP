#!/usr/bin/env python3
"""Build statistical top-K POI candidates and report target recall.

The script builds train-derived transition/category/global indexes, then scores
each sample's candidate POIs from sequence, geo, history, and aggregate train
statistics. For train rows, it applies leave-one-out subtraction so the current
sample's target contribution is not directly injected from aggregate indexes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build statistical top-K candidates.")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--index-inputs", type=Path, required=True)
    p.add_argument("--index-labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--transition-limit", type=int, default=80)
    p.add_argument("--geo-limit", type=int, default=60)
    p.add_argument("--history-limit", type=int, default=50)
    p.add_argument("--category-limit", type=int, default=80)
    p.add_argument("--global-limit", type=int, default=100)
    p.add_argument("--knn-limit", type=int, default=100)
    p.add_argument("--graph-limit", type=int, default=100)
    p.add_argument("--co-visit-limit", type=int, default=100)
    p.add_argument("--disable-v2-sources", action="store_true")
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
            raise ValueError(f"{path} row without sample_id")
        if sample_id in by_id:
            raise ValueError(f"{path} duplicate sample_id: {sample_id}")
        by_id[sample_id] = row
    return by_id


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evidence_parts(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    evidence = row.get("evidence") or {}
    return evidence.get("sequence") or {}, evidence.get("geo") or {}, evidence.get("preference") or {}


def poi_ok(value: Any) -> bool:
    text = str(value or "")
    return text.startswith("v") and text[1:].isdigit()


def add_category(mapping: Dict[str, str], poi: Any, category: Any) -> None:
    poi_id = str(poi or "")
    cat = str(category or "")
    if poi_ok(poi_id) and cat and cat != "unknown":
        mapping.setdefault(poi_id, cat)


def visible_categories(row: Dict[str, Any]) -> List[str]:
    seq, geo, pref = evidence_parts(row)
    values: List[str] = []
    for item in seq.get("current_trajectory") or []:
        values.append(str(item.get("category") or ""))
    for item in seq.get("transition_candidates") or []:
        values.append(str(item.get("category") or ""))
    for item in geo.get("nearby_pois") or []:
        values.append(str(item.get("category") or ""))
    for item in pref.get("top_categories") or []:
        values.append(str(item.get("category") or ""))
    out: List[str] = []
    for value in values:
        if value and value != "unknown" and value not in out:
            out.append(value)
    return out


class StatIndex:
    def __init__(self) -> None:
        self.last_poi_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.last_category_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.last2_poi_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.last2_category_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.traj_signature_next: Dict[Tuple[Any, ...], Counter[str]] = defaultdict(Counter)
        self.category_popular: Dict[str, Counter[str]] = defaultdict(Counter)
        self.slot_popular: Dict[int, Counter[str]] = defaultdict(Counter)
        self.weekday_slot_popular: Dict[Tuple[int, int], Counter[str]] = defaultdict(Counter)
        self.user_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.user_category_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.co_visit: Dict[str, Counter[str]] = defaultdict(Counter)
        self.global_popular: Counter[str] = Counter()
        self.poi_category: Dict[str, str] = {}

    @staticmethod
    def from_rows(rows: List[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> "StatIndex":
        idx = StatIndex()
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            label = labels.get(sample_id)
            if label is None:
                continue
            target = str(label.get("target_poi_id") or "")
            target_category = str(label.get("target_category") or "")
            if not poi_ok(target):
                continue
            seq, geo, pref = evidence_parts(row)
            trajectory = seq.get("current_trajectory") or []
            last = trajectory[-1] if trajectory else {}
            last_poi = str(last.get("poi_id") or "")
            last_category = str(last.get("category") or "")
            last_slot = safe_int(last.get("slot"), -1)
            last_dow = safe_int(last.get("dow"), -1)
            user_id = str(row.get("user_id") or "")
            if poi_ok(last_poi):
                idx.last_poi_next[last_poi][target] += 1
            if last_category and last_category != "unknown":
                idx.last_category_next[last_category][target] += 1
            if len(trajectory) >= 2:
                prev = trajectory[-2]
                prev_poi = str(prev.get("poi_id") or "")
                prev_category = str(prev.get("category") or "")
                if poi_ok(prev_poi) and poi_ok(last_poi):
                    idx.last2_poi_next[(prev_poi, last_poi)][target] += 1
                if prev_category and last_category and prev_category != "unknown" and last_category != "unknown":
                    idx.last2_category_next[(prev_category, last_category)][target] += 1
            if last_category and last_category != "unknown":
                idx.traj_signature_next[(last_category, last_dow, last_slot)][target] += 1
                idx.traj_signature_next[(last_category, last_slot)][target] += 1
            if last_slot >= 0:
                idx.slot_popular[last_slot][target] += 1
            if last_dow >= 0 and last_slot >= 0:
                idx.weekday_slot_popular[(last_dow, last_slot)][target] += 1
            if user_id:
                idx.user_next[user_id][target] += 1
                if target_category and target_category != "unknown":
                    idx.user_category_next[(user_id, target_category)][target] += 1
            if target_category and target_category != "unknown":
                idx.category_popular[target_category][target] += 1
                add_category(idx.poi_category, target, target_category)
            idx.global_popular[target] += 1
            for item in trajectory:
                add_category(idx.poi_category, item.get("poi_id"), item.get("category"))
            for item in seq.get("transition_candidates") or []:
                add_category(idx.poi_category, item.get("poi_id"), item.get("category"))
            for item in geo.get("nearby_pois") or []:
                add_category(idx.poi_category, item.get("poi_id"), item.get("category"))
            for item in pref.get("revisited_pois") or []:
                add_category(idx.poi_category, item.get("poi_id"), item.get("category"))
            related = [str(item.get("poi_id") or "") for item in trajectory]
            related.extend(str(item.get("poi_id") or "") for item in pref.get("revisited_pois") or [])
            related = [poi for poi in dict.fromkeys(related) if poi_ok(poi)]
            for poi in related:
                idx.co_visit[poi][target] += 1
        return idx

    def category_of(self, poi_id: str) -> str:
        return self.poi_category.get(poi_id, "unknown")


def add_score(
    scores: Dict[str, float],
    sources: Dict[str, List[str]],
    poi_id: Any,
    amount: float,
    source: str,
) -> None:
    poi = str(poi_id or "")
    if not poi_ok(poi):
        return
    scores[poi] = scores.get(poi, 0.0) + float(amount)
    if source not in sources[poi]:
        sources[poi].append(source)


def adjusted_counter(counter: Counter[str], target: str, subtract: bool) -> Counter[str]:
    out = counter.copy()
    if subtract and target and out.get(target, 0) > 0:
        out[target] -= 1
        if out[target] <= 0:
            del out[target]
    return out


def build_candidates(
    row: Dict[str, Any],
    label: Dict[str, Any],
    index: StatIndex,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    seq, geo, pref = evidence_parts(row)
    trajectory = seq.get("current_trajectory") or []
    last = trajectory[-1] if trajectory else {}
    last_poi = str(last.get("poi_id") or "")
    last_category = str(last.get("category") or "")
    last_slot = safe_int(last.get("slot"), -1)
    last_dow = safe_int(last.get("dow"), -1)
    user_id = str(row.get("user_id") or "")
    target = str(label.get("target_poi_id") or "")
    subtract = str(row.get("split") or "") == "train"

    scores: Dict[str, float] = {}
    sources: Dict[str, List[str]] = defaultdict(list)

    # Preserve strong local evidence from precomputed rows, but allow top-100 to
    # include more aggregate candidates after these evidence-backed POIs.
    for rank, poi in enumerate(row.get("candidate_poi_ids") or []):
        add_score(scores, sources, poi, 850.0 - rank * 2.0, "original_candidate")
    for rank, item in enumerate(seq.get("transition_candidates") or []):
        count = max(1, safe_int(item.get("count"), 1))
        source = str(item.get("source") or "transition")
        add_score(scores, sources, item.get("poi_id"), 1200.0 + count * 12.0 - rank, source)
    for rank, item in enumerate(geo.get("nearby_pois") or []):
        try:
            dist = float(item.get("distance_km"))
        except (TypeError, ValueError):
            dist = 1.0
        add_score(scores, sources, item.get("poi_id"), 750.0 + max(0.0, 1.0 - dist) * 80.0 - rank, "geo_nearby")
    for rank, item in enumerate(pref.get("revisited_pois") or []):
        count = max(1, safe_int(item.get("count"), 1))
        add_score(scores, sources, item.get("poi_id"), 900.0 + count * 18.0 - rank, "user_revisited")

    last_poi_counter = adjusted_counter(index.last_poi_next.get(last_poi, Counter()), target, subtract)
    for rank, (poi, count) in enumerate(last_poi_counter.most_common(args.transition_limit)):
        add_score(scores, sources, poi, 1100.0 + count * 10.0 - rank, "train_same_last_poi")

    last_cat_counter = adjusted_counter(index.last_category_next.get(last_category, Counter()), target, subtract)
    for rank, (poi, count) in enumerate(last_cat_counter.most_common(args.transition_limit)):
        add_score(scores, sources, poi, 760.0 + count * 5.0 - rank, "train_same_last_category")

    if not args.disable_v2_sources:
        if len(trajectory) >= 2:
            prev = trajectory[-2]
            prev_poi = str(prev.get("poi_id") or "")
            prev_category = str(prev.get("category") or "")
            c = adjusted_counter(index.last2_poi_next.get((prev_poi, last_poi), Counter()), target, subtract)
            for rank, (poi, count) in enumerate(c.most_common(args.knn_limit)):
                add_score(scores, sources, poi, 980.0 + count * 12.0 - rank, "train_last2_poi")
            c = adjusted_counter(
                index.last2_category_next.get((prev_category, last_category), Counter()), target, subtract
            )
            for rank, (poi, count) in enumerate(c.most_common(args.knn_limit)):
                add_score(scores, sources, poi, 700.0 + count * 6.0 - rank, "train_last2_category")

        for key, source, base in [
            ((last_category, last_dow, last_slot), "train_signature_category_dow_slot", 680.0),
            ((last_category, last_slot), "train_signature_category_slot", 620.0),
        ]:
            c = adjusted_counter(index.traj_signature_next.get(key, Counter()), target, subtract)
            for rank, (poi, count) in enumerate(c.most_common(args.knn_limit)):
                add_score(scores, sources, poi, base + count * 4.0 - rank, source)

        if user_id:
            c = adjusted_counter(index.user_next.get(user_id, Counter()), target, subtract)
            for rank, (poi, count) in enumerate(c.most_common(args.history_limit)):
                add_score(scores, sources, poi, 940.0 + count * 14.0 - rank, "train_user_next")
            for category in visible_categories(row):
                c = adjusted_counter(index.user_category_next.get((user_id, category), Counter()), target, subtract)
                for rank, (poi, count) in enumerate(c.most_common(args.history_limit)):
                    add_score(scores, sources, poi, 820.0 + count * 12.0 - rank, f"train_user_category:{category}")

        if last_slot >= 0:
            c = adjusted_counter(index.slot_popular.get(last_slot, Counter()), target, subtract)
            for rank, (poi, count) in enumerate(c.most_common(args.global_limit)):
                add_score(scores, sources, poi, 240.0 + count * 1.5 - rank * 0.25, "train_slot")
        if last_dow >= 0 and last_slot >= 0:
            c = adjusted_counter(index.weekday_slot_popular.get((last_dow, last_slot), Counter()), target, subtract)
            for rank, (poi, count) in enumerate(c.most_common(args.global_limit)):
                add_score(scores, sources, poi, 280.0 + count * 2.0 - rank * 0.25, "train_weekday_slot")

        graph_seeds = [last_poi]
        graph_seeds.extend(str(item.get("poi_id") or "") for item in pref.get("revisited_pois") or [])
        for seed in [poi for poi in dict.fromkeys(graph_seeds) if poi_ok(poi)]:
            c = adjusted_counter(index.co_visit.get(seed, Counter()), target, subtract)
            for rank, (poi, count) in enumerate(c.most_common(args.co_visit_limit)):
                add_score(scores, sources, poi, 520.0 + count * 6.0 - rank * 0.5, "train_covisit_graph")

    for category in visible_categories(row):
        cat_counter = adjusted_counter(index.category_popular.get(category, Counter()), target, subtract)
        for rank, (poi, count) in enumerate(cat_counter.most_common(args.category_limit)):
            add_score(scores, sources, poi, 420.0 + count * 3.0 - rank * 0.5, f"train_category:{category}")

    global_counter = adjusted_counter(index.global_popular, target, subtract)
    for rank, (poi, count) in enumerate(global_counter.most_common(args.global_limit)):
        add_score(scores, sources, poi, 160.0 + count * 1.5 - rank * 0.25, "train_global")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top = [poi for poi, _ in ranked[: args.top_k]]
    details = {
        poi: {
            "score": round(scores[poi], 4),
            "category": index.category_of(poi),
            "sources": sources[poi],
        }
        for poi in top
    }
    return top, details


def percentile(values: List[int], q: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def main() -> None:
    args = parse_args()
    labels = load_by_id(args.labels)
    index = StatIndex.from_rows(read_jsonl(args.index_inputs), load_by_id(args.index_labels))
    rows = read_jsonl(args.inputs)

    output_rows: List[Dict[str, Any]] = []
    hit_counts = Counter()
    lengths: List[int] = []
    ranks: List[int] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        label = labels.get(sample_id)
        if label is None:
            raise ValueError(f"Missing label for {sample_id}")
        target = str(label.get("target_poi_id") or "")
        candidates, details = build_candidates(row, label, index, args)
        lengths.append(len(candidates))
        rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
        if rank is not None:
            ranks.append(rank)
        hit_counts["rows"] += 1
        hit_counts["hit@100"] += int(rank is not None and rank <= 100)
        hit_counts["hit@50"] += int(rank is not None and rank <= 50)
        hit_counts["hit@30"] += int(rank is not None and rank <= 30)
        hit_counts["hit@25"] += int(rank is not None and rank <= 25)
        hit_counts["hit@10"] += int(rank is not None and rank <= 10)
        hit_counts["original_hit"] += int(target in {str(x) for x in row.get("candidate_poi_ids") or []})
        output_rows.append(
            {
                "sample_id": sample_id,
                "split": row.get("split"),
                "city": row.get("city"),
                "target": {
                    "poi_id": target,
                    "category": label.get("target_category"),
                },
                "candidate_poi_ids": candidates,
                "candidate_details": details,
                "target_rank": rank,
                "target_in_topk": rank is not None,
                "original_candidate_count": len(row.get("candidate_poi_ids") or []),
                "original_target_in_candidates": target in {str(x) for x in row.get("candidate_poi_ids") or []},
            }
        )

    written = write_jsonl(args.output, output_rows, args.overwrite)
    n = hit_counts["rows"]
    report = {
        "inputs": str(args.inputs),
        "index_inputs": str(args.index_inputs),
        "output": str(args.output),
        "written": written,
        "top_k": args.top_k,
        "rows": n,
        "original_hit": hit_counts["original_hit"],
        "original_hit_ratio": round(hit_counts["original_hit"] / n, 6) if n else 0.0,
        "hit@10": hit_counts["hit@10"],
        "hit@10_ratio": round(hit_counts["hit@10"] / n, 6) if n else 0.0,
        "hit@25": hit_counts["hit@25"],
        "hit@25_ratio": round(hit_counts["hit@25"] / n, 6) if n else 0.0,
        "hit@30": hit_counts["hit@30"],
        "hit@30_ratio": round(hit_counts["hit@30"] / n, 6) if n else 0.0,
        "hit@50": hit_counts["hit@50"],
        "hit@50_ratio": round(hit_counts["hit@50"] / n, 6) if n else 0.0,
        "hit@100": hit_counts["hit@100"],
        "hit@100_ratio": round(hit_counts["hit@100"] / n, 6) if n else 0.0,
        "candidate_count": {
            "min": min(lengths) if lengths else None,
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "max": max(lengths) if lengths else None,
            "mean": round(sum(lengths) / len(lengths), 4) if lengths else 0.0,
        },
        "target_rank_when_hit": {
            "min": min(ranks) if ranks else None,
            "p50": percentile(ranks, 0.50),
            "p90": percentile(ranks, 0.90),
            "max": max(ranks) if ranks else None,
            "mean": round(sum(ranks) / len(ranks), 4) if ranks else 0.0,
        },
        "leave_one_out_for_train": True,
    }
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".stats.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
