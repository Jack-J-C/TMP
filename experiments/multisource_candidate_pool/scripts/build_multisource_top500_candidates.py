#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
GRAPHRAG_SCRIPT = ROOT / "src/graphrag/build_graphrag_topk_candidates.py"
spec = importlib.util.spec_from_file_location("graphrag_builder", GRAPHRAG_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {GRAPHRAG_SCRIPT}")
graphrag = importlib.util.module_from_spec(spec)
sys.modules["graphrag_builder"] = graphrag
spec.loader.exec_module(graphrag)


SOURCE_WEIGHTS = {
    "graphrag": 1.00,
    "evidence_transition": 1.20,
    "history": 1.05,
    "graph_user": 0.95,
    "semantic_edge": 0.85,
    "graph_transition": 0.80,
    "semantic_popular": 0.60,
    "category_popular": 0.55,
    "evidence_geo": 0.50,
    "legacy_candidate": 0.45,
    "time": 0.35,
    "global": 0.20,
}

SOURCE_QUOTAS = {
    "graphrag": 300,
    "evidence_transition": 80,
    "history": 60,
    "graph_user": 80,
    "semantic_edge": 120,
    "graph_transition": 100,
    "semantic_popular": 80,
    "category_popular": 60,
    "evidence_geo": 50,
    "legacy_candidate": 60,
    "time": 40,
    "global": 30,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a multi-source fused TopK candidate pool.")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--index-inputs", type=Path, required=True)
    p.add_argument("--index-labels", type=Path, required=True)
    p.add_argument("--semantic-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--graphrag-pool-k", type=int, default=900)
    p.add_argument("--rrf-c", type=float, default=60.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def poi_ok(value: Any) -> bool:
    return graphrag.poi_ok(value)


def counter_ranked(counter: Counter[str], target: str, subtract: bool, limit: int) -> list[str]:
    return [
        poi
        for poi, _ in graphrag.adjusted(counter, target, subtract).most_common(limit)
        if poi_ok(poi)
    ]


def unique(values: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        poi = str(value or "")
        if poi_ok(poi) and poi not in seen:
            seen.add(poi)
            out.append(poi)
            if len(out) >= limit:
                break
    return out


def add_ranked(
    source_lists: dict[str, list[str]],
    source: str,
    values: list[str],
    limit: int,
) -> None:
    ranked = unique(values, limit)
    if not ranked:
        return
    source_lists[source].extend(poi for poi in ranked if poi not in source_lists[source])


def semantic_context(semantic_map: dict[str, dict[str, Any]], row: dict[str, Any]) -> tuple[list[str], list[str]]:
    return graphrag.semantic_neighbors(semantic_map, row)


def graph_args(top_k: int) -> SimpleNamespace:
    return SimpleNamespace(
        top_k=top_k,
        transition_limit=180,
        graph_limit=220,
        category_limit=180,
        time_limit=140,
        global_limit=160,
        semantic_edge_limit=220,
        disable_semantic_id_edges=False,
    )


def build_source_lists(
    row: dict[str, Any],
    label: dict[str, Any],
    index: Any,
    semantic_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    seq, geo, pref = graphrag.evidence_parts(row)
    trajectory = seq.get("current_trajectory") or []
    last = trajectory[-1] if trajectory else {}
    last_poi = str(last.get("poi_id") or "")
    last_cat = str(last.get("category") or "")
    last_semantic_id = graphrag.semantic_id(semantic_map, last_poi)
    last_category_token = graphrag.semantic_category_token(semantic_map, last_poi)
    last_geo_cell = graphrag.semantic_geo_cell(semantic_map, last_poi)
    last_slot = graphrag.safe_int(last.get("slot"), -1)
    last_dow = graphrag.safe_int(last.get("dow"), -1)
    user_id = str(row.get("user_id") or "")
    target = str(label.get("target_poi_id") or "")
    subtract = str(row.get("split") or "") == "train"
    source_lists: dict[str, list[str]] = defaultdict(list)
    source_details: dict[str, dict[str, Any]] = {}

    graph_candidates, graph_details = graphrag.retrieve_candidates(
        row,
        label,
        index,
        semantic_map,
        graph_args(args.graphrag_pool_k),
    )
    add_ranked(source_lists, "graphrag", graph_candidates, args.graphrag_pool_k)
    for rank, poi in enumerate(graph_candidates, 1):
        detail = graph_details.get(poi) or {}
        source_details.setdefault(poi, {})["graph_rank"] = rank
        source_details[poi]["graph_score"] = detail.get("score")
        source_details[poi]["graph_sources"] = detail.get("sources") or []

    add_ranked(source_lists, "legacy_candidate", [str(x) for x in row.get("candidate_poi_ids") or []], 300)
    add_ranked(
        source_lists,
        "evidence_transition",
        [str(item.get("poi_id") or "") for item in seq.get("transition_candidates") or []],
        220,
    )
    add_ranked(
        source_lists,
        "history",
        [
            str(item.get("poi_id") or "")
            for item in sorted(
                pref.get("revisited_pois") or [],
                key=lambda item: (-graphrag.safe_int(item.get("count"), 1), str(item.get("poi_id") or "")),
            )
        ],
        180,
    )
    add_ranked(
        source_lists,
        "evidence_geo",
        [
            str(item.get("poi_id") or "")
            for item in sorted(
                geo.get("nearby_pois") or [],
                key=lambda item: (float(item.get("distance_km") or 999.0), str(item.get("poi_id") or "")),
            )
        ],
        160,
    )

    graph_transition: list[str] = []
    graph_transition.extend(counter_ranked(index.poi_next[last_poi], target, subtract, 220))
    graph_transition.extend(counter_ranked(index.category_next[last_cat], target, subtract, 180))
    if len(trajectory) >= 2:
        prev = trajectory[-2]
        graph_transition.extend(
            counter_ranked(index.last2_poi_next[(str(prev.get("poi_id") or ""), last_poi)], target, subtract, 220)
        )
        graph_transition.extend(
            counter_ranked(index.last2_category_next[(str(prev.get("category") or ""), last_cat)], target, subtract, 180)
        )
    add_ranked(source_lists, "graph_transition", graph_transition, 360)

    if user_id:
        user_values: list[str] = []
        user_values.extend(counter_ranked(index.user_next[user_id], target, subtract, 240))
        for cat in graphrag.visible_categories(row):
            user_values.extend(counter_ranked(index.user_category_next[(user_id, cat)], target, subtract, 160))
        user_values.extend(counter_ranked(index.user_semantic_category_next[(user_id, last_category_token)], target, subtract, 160))
        user_values.extend(counter_ranked(index.user_semantic_cell_next[(user_id, last_geo_cell)], target, subtract, 160))
        add_ranked(source_lists, "graph_user", user_values, 360)

    semantic_values: list[str] = []
    semantic_values.extend(counter_ranked(index.semantic_id_next[last_semantic_id], target, subtract, 220))
    semantic_values.extend(counter_ranked(index.semantic_category_next[last_category_token], target, subtract, 220))
    semantic_values.extend(counter_ranked(index.semantic_cell_next[last_geo_cell], target, subtract, 220))
    if len(trajectory) >= 2:
        prev_semantic_id = graphrag.semantic_id(semantic_map, trajectory[-2].get("poi_id"))
        semantic_values.extend(counter_ranked(index.last2_semantic_id_next[(prev_semantic_id, last_semantic_id)], target, subtract, 220))
    for seed in [last_poi, *[str(x.get("poi_id") or "") for x in pref.get("revisited_pois") or []]]:
        semantic_values.extend(counter_ranked(index.semantic_covisit[graphrag.semantic_id(semantic_map, seed)], target, subtract, 160))
    add_ranked(source_lists, "semantic_edge", semantic_values, 420)

    category_values: list[str] = []
    for cat in graphrag.visible_categories(row):
        category_values.extend(counter_ranked(index.category_popular[cat], target, subtract, 180))
    add_ranked(source_lists, "category_popular", category_values, 260)

    semantic_popular: list[str] = []
    cat_tokens, cells = semantic_context(semantic_map, row)
    for token in cat_tokens:
        semantic_popular.extend(counter_ranked(index.semantic_category_popular[token], target, subtract, 180))
    for cell in cells:
        semantic_popular.extend(counter_ranked(index.semantic_cell_popular[cell], target, subtract, 180))
    add_ranked(source_lists, "semantic_popular", semantic_popular, 320)

    time_values: list[str] = []
    if last_slot >= 0:
        time_values.extend(counter_ranked(index.slot_popular[last_slot], target, subtract, 180))
    if last_dow >= 0 and last_slot >= 0:
        time_values.extend(counter_ranked(index.weekday_slot_popular[(last_dow, last_slot)], target, subtract, 180))
    add_ranked(source_lists, "time", time_values, 220)
    add_ranked(source_lists, "global", counter_ranked(index.global_popular, target, subtract, 300), 300)

    return dict(source_lists), source_details


def fuse_sources(
    source_lists: dict[str, list[str]],
    source_details: dict[str, dict[str, Any]],
    semantic_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    fused_scores: dict[str, float] = defaultdict(float)
    sources: dict[str, list[str]] = defaultdict(list)
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for source, values in source_lists.items():
        weight = SOURCE_WEIGHTS.get(source, 0.5)
        for rank, poi in enumerate(values, 1):
            fused_scores[poi] += weight / (args.rrf_c + rank)
            if source not in sources[poi]:
                sources[poi].append(source)
            source_ranks[poi][source] = rank

    selected: list[str] = []
    seen: set[str] = set()
    for source, quota in SOURCE_QUOTAS.items():
        for poi in source_lists.get(source, [])[:quota]:
            if poi in seen:
                continue
            seen.add(poi)
            selected.append(poi)
            if len(selected) >= args.top_k:
                break
        if len(selected) >= args.top_k:
            break

    ranked_fill = sorted(fused_scores, key=lambda poi: (-fused_scores[poi], poi))
    for poi in ranked_fill:
        if poi in seen:
            continue
        seen.add(poi)
        selected.append(poi)
        if len(selected) >= args.top_k:
            break

    details: dict[str, dict[str, Any]] = {}
    for rank, poi in enumerate(selected, 1):
        sem = semantic_map.get(poi) or {}
        extra = source_details.get(poi) or {}
        details[poi] = {
            "semantic_id": sem.get("semantic_id"),
            "category": sem.get("category"),
            "score": round(float(fused_scores.get(poi, 0.0)), 8),
            "sources": sources.get(poi, []),
            "source_ranks": source_ranks.get(poi, {}),
            "multi_source_rank": rank,
            **extra,
        }
    return selected, details


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    labels = graphrag.load_by_id(args.labels)
    semantic_map = graphrag.load_semantic_map(args.semantic_map)
    index = graphrag.GraphIndex.from_rows(
        graphrag.read_jsonl(args.index_inputs),
        graphrag.load_by_id(args.index_labels),
        semantic_map,
    )
    rows = graphrag.read_jsonl(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    ranks: list[int] = []
    source_hit_counts: Counter[str] = Counter()
    source_presence: Counter[str] = Counter()
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            sid = str(row.get("sample_id") or "")
            label = labels.get(sid)
            if label is None:
                raise ValueError(f"Missing label for {sid}")
            target = str(label.get("target_poi_id") or "")
            source_lists, source_details = build_source_lists(row, label, index, semantic_map, args)
            candidates, details = fuse_sources(source_lists, source_details, semantic_map, args)
            rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
            counts["rows"] += 1
            for k in [50, 100, 150, 200, 300, 400, 500]:
                counts[f"hit@{k}"] += int(rank is not None and rank <= k)
            if rank is not None:
                ranks.append(rank)
                for source in details[target].get("sources") or []:
                    source_hit_counts[source] += 1
            for source, values in source_lists.items():
                if values:
                    source_presence[source] += 1
            target_sem = semantic_map.get(target) or {}
            f.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "split": row.get("split"),
                        "city": row.get("city"),
                        "target": {
                            "poi_id": target,
                            "semantic_id": target_sem.get("semantic_id"),
                            "category": label.get("target_category"),
                        },
                        "candidate_poi_ids": candidates,
                        "candidate_semantic_ids": [(semantic_map.get(poi) or {}).get("semantic_id") for poi in candidates],
                        "candidate_details": details,
                        "target_rank": rank,
                        "target_in_topk": rank is not None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = counts["rows"]
    report = {
        "output": str(args.output),
        "rows": n,
        "top_k": args.top_k,
        "graphrag_pool_k": args.graphrag_pool_k,
        "rrf_c": args.rrf_c,
        "source_weights": SOURCE_WEIGHTS,
        "source_quotas": SOURCE_QUOTAS,
        **{f"hit@{k}_ratio": round(counts[f"hit@{k}"] / n, 6) if n else 0.0 for k in [50, 100, 150, 200, 300, 400, 500]},
        "target_rank_when_hit": {
            "p50": percentile(ranks, 0.5),
            "p90": percentile(ranks, 0.9),
            "mean": round(sum(ranks) / len(ranks), 4) if ranks else 0.0,
            "max": max(ranks) if ranks else None,
        },
        "source_presence_rows": dict(source_presence),
        "source_hit_counts": dict(source_hit_counts),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
