#!/usr/bin/env python3
"""GraphRAG-style Top-K candidate retrieval with semantic IDs.

This is the minimum closed-loop candidate generator for the double_LLM branch:
semantic POI IDs + train-derived graph retrieval + recall evaluation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GraphRAG Top-K POI candidates.")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--index-inputs", type=Path, required=True)
    p.add_argument("--index-labels", type=Path, required=True)
    p.add_argument("--semantic-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--transition-limit", type=int, default=120)
    p.add_argument("--graph-limit", type=int, default=160)
    p.add_argument("--category-limit", type=int, default=120)
    p.add_argument("--time-limit", type=int, default=100)
    p.add_argument("--global-limit", type=int, default=100)
    p.add_argument(
        "--semantic-edge-limit",
        type=int,
        default=160,
        help="Candidates per explicit Semantic-ID graph edge.",
    )
    p.add_argument(
        "--disable-semantic-id-edges",
        action="store_true",
        help="Disable explicit Semantic-ID node edges for ablation.",
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


def open_jsonl_writer(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def load_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} row without sample_id")
        out[sid] = row
    return out


def load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[str(row["poi_id"])] = row
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def poi_ok(value: Any) -> bool:
    text = str(value or "")
    return text.startswith("v") and text[1:].isdigit()


def evidence_parts(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    evidence = row.get("evidence") or {}
    return evidence.get("sequence") or {}, evidence.get("geo") or {}, evidence.get("preference") or {}


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


def semantic_id(semantic_map: Dict[str, Dict[str, Any]], poi_id: Any) -> str:
    sem = semantic_map.get(str(poi_id or "")) or {}
    return str(sem.get("semantic_id") or "")


def semantic_category_token(semantic_map: Dict[str, Dict[str, Any]], poi_id: Any) -> str:
    sem = semantic_map.get(str(poi_id or "")) or {}
    return str(sem.get("category_token") or "")


def semantic_geo_cell(semantic_map: Dict[str, Dict[str, Any]], poi_id: Any) -> str:
    sem = semantic_map.get(str(poi_id or "")) or {}
    return str(sem.get("geo_cell") or "")


class GraphIndex:
    def __init__(self) -> None:
        self.poi_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.category_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.last2_poi_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.last2_category_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.user_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.user_category_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.category_popular: Dict[str, Counter[str]] = defaultdict(Counter)
        self.semantic_category_popular: Dict[str, Counter[str]] = defaultdict(Counter)
        self.semantic_cell_popular: Dict[str, Counter[str]] = defaultdict(Counter)
        self.semantic_id_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.last2_semantic_id_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.semantic_category_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.semantic_cell_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.user_semantic_category_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.user_semantic_cell_next: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.semantic_covisit: Dict[str, Counter[str]] = defaultdict(Counter)
        self.slot_popular: Dict[int, Counter[str]] = defaultdict(Counter)
        self.weekday_slot_popular: Dict[Tuple[int, int], Counter[str]] = defaultdict(Counter)
        self.co_visit: Dict[str, Counter[str]] = defaultdict(Counter)
        self.global_popular: Counter[str] = Counter()

    @staticmethod
    def from_rows(
        rows: List[Dict[str, Any]],
        labels: Dict[str, Dict[str, Any]],
        semantic_map: Dict[str, Dict[str, Any]],
    ) -> "GraphIndex":
        idx = GraphIndex()
        for row in rows:
            label = labels.get(str(row.get("sample_id") or ""))
            if label is None:
                continue
            target = str(label.get("target_poi_id") or "")
            target_category = str(label.get("target_category") or "")
            if not poi_ok(target):
                continue
            sem = semantic_map.get(target) or {}
            target_semantic_id = str(sem.get("semantic_id") or "")
            seq, _, pref = evidence_parts(row)
            trajectory = seq.get("current_trajectory") or []
            last = trajectory[-1] if trajectory else {}
            last_poi = str(last.get("poi_id") or "")
            last_cat = str(last.get("category") or "")
            last_semantic_id = semantic_id(semantic_map, last_poi)
            last_category_token = semantic_category_token(semantic_map, last_poi)
            last_geo_cell = semantic_geo_cell(semantic_map, last_poi)
            last_slot = safe_int(last.get("slot"), -1)
            last_dow = safe_int(last.get("dow"), -1)
            user_id = str(row.get("user_id") or "")
            if poi_ok(last_poi):
                idx.poi_next[last_poi][target] += 1
            if last_semantic_id:
                idx.semantic_id_next[last_semantic_id][target] += 1
            if last_category_token:
                idx.semantic_category_next[last_category_token][target] += 1
            if last_geo_cell:
                idx.semantic_cell_next[last_geo_cell][target] += 1
            if last_cat and last_cat != "unknown":
                idx.category_next[last_cat][target] += 1
            if len(trajectory) >= 2:
                prev = trajectory[-2]
                prev_poi = str(prev.get("poi_id") or "")
                prev_cat = str(prev.get("category") or "")
                prev_semantic_id = semantic_id(semantic_map, prev_poi)
                if poi_ok(prev_poi) and poi_ok(last_poi):
                    idx.last2_poi_next[(prev_poi, last_poi)][target] += 1
                if prev_semantic_id and last_semantic_id:
                    idx.last2_semantic_id_next[(prev_semantic_id, last_semantic_id)][target] += 1
                if prev_cat and last_cat and prev_cat != "unknown" and last_cat != "unknown":
                    idx.last2_category_next[(prev_cat, last_cat)][target] += 1
            if user_id:
                idx.user_next[user_id][target] += 1
                if target_category and target_category != "unknown":
                    idx.user_category_next[(user_id, target_category)][target] += 1
                if last_category_token:
                    idx.user_semantic_category_next[(user_id, last_category_token)][target] += 1
                if last_geo_cell:
                    idx.user_semantic_cell_next[(user_id, last_geo_cell)][target] += 1
            if target_category and target_category != "unknown":
                idx.category_popular[target_category][target] += 1
            if sem.get("category_token"):
                idx.semantic_category_popular[str(sem["category_token"])][target] += 1
            if sem.get("geo_cell"):
                idx.semantic_cell_popular[str(sem["geo_cell"])][target] += 1
            if last_slot >= 0:
                idx.slot_popular[last_slot][target] += 1
            if last_dow >= 0 and last_slot >= 0:
                idx.weekday_slot_popular[(last_dow, last_slot)][target] += 1
            idx.global_popular[target] += 1
            related = [str(item.get("poi_id") or "") for item in trajectory]
            related.extend(str(item.get("poi_id") or "") for item in pref.get("revisited_pois") or [])
            for poi in [x for x in dict.fromkeys(related) if poi_ok(x)]:
                idx.co_visit[poi][target] += 1
                sid = semantic_id(semantic_map, poi)
                if sid:
                    idx.semantic_covisit[sid][target] += 1
        return idx


def adjusted(counter: Counter[str], target: str, subtract: bool) -> Counter[str]:
    out = counter.copy()
    if subtract and target and out.get(target, 0) > 0:
        out[target] -= 1
        if out[target] <= 0:
            del out[target]
    return out


def add_score(scores: Dict[str, float], sources: Dict[str, List[str]], poi: Any, score: float, source: str) -> None:
    poi_id = str(poi or "")
    if not poi_ok(poi_id):
        return
    scores[poi_id] = scores.get(poi_id, 0.0) + float(score)
    if source not in sources[poi_id]:
        sources[poi_id].append(source)


def add_counter(
    scores: Dict[str, float],
    sources: Dict[str, List[str]],
    counter: Counter[str],
    target: str,
    subtract: bool,
    limit: int,
    base: float,
    weight: float,
    source: str,
) -> None:
    if not counter:
        return
    for rank, (poi, count) in enumerate(adjusted(counter, target, subtract).most_common(limit)):
        add_score(scores, sources, poi, base + count * weight - rank * 0.35, source)


def semantic_neighbors(semantic_map: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    seq, geo, pref = evidence_parts(row)
    categories = visible_categories(row)
    cells: List[str] = []
    for item in [*(seq.get("current_trajectory") or []), *(geo.get("nearby_pois") or [])]:
        sem = semantic_map.get(str(item.get("poi_id") or "")) or {}
        if sem.get("geo_cell") and sem["geo_cell"] not in cells:
            cells.append(str(sem["geo_cell"]))
    for item in pref.get("revisited_pois") or []:
        sem = semantic_map.get(str(item.get("poi_id") or "")) or {}
        if sem.get("geo_cell") and sem["geo_cell"] not in cells:
            cells.append(str(sem["geo_cell"]))
    cat_tokens: List[str] = []
    for poi, sem in semantic_map.items():
        # Cheap category token lookup for visible categories without building an inverse map.
        if sem.get("category") in categories:
            token = str(sem.get("category_token") or "")
            if token and token not in cat_tokens:
                cat_tokens.append(token)
    return cat_tokens, cells


def retrieve_candidates(
    row: Dict[str, Any],
    label: Dict[str, Any],
    index: GraphIndex,
    semantic_map: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    seq, geo, pref = evidence_parts(row)
    trajectory = seq.get("current_trajectory") or []
    last = trajectory[-1] if trajectory else {}
    last_poi = str(last.get("poi_id") or "")
    last_cat = str(last.get("category") or "")
    last_semantic_id = semantic_id(semantic_map, last_poi)
    last_category_token = semantic_category_token(semantic_map, last_poi)
    last_geo_cell = semantic_geo_cell(semantic_map, last_poi)
    last_slot = safe_int(last.get("slot"), -1)
    last_dow = safe_int(last.get("dow"), -1)
    user_id = str(row.get("user_id") or "")
    target = str(label.get("target_poi_id") or "")
    subtract = str(row.get("split") or "") == "train"
    scores: Dict[str, float] = {}
    sources: Dict[str, List[str]] = defaultdict(list)

    for rank, poi in enumerate(row.get("candidate_poi_ids") or []):
        add_score(scores, sources, poi, 700.0 - rank, "legacy_candidate")
    for rank, item in enumerate(seq.get("transition_candidates") or []):
        add_score(
            scores,
            sources,
            item.get("poi_id"),
            980.0 + safe_int(item.get("count"), 1) * 10.0 - rank,
            str(item.get("source") or "evidence_transition"),
        )
    for rank, item in enumerate(geo.get("nearby_pois") or []):
        try:
            dist = float(item.get("distance_km"))
        except (TypeError, ValueError):
            dist = 1.0
        add_score(scores, sources, item.get("poi_id"), 640.0 + max(0.0, 1.0 - dist) * 80.0 - rank, "evidence_geo")
    for rank, item in enumerate(pref.get("revisited_pois") or []):
        add_score(scores, sources, item.get("poi_id"), 900.0 + safe_int(item.get("count"), 1) * 12.0 - rank, "history")

    add_counter(scores, sources, index.poi_next[last_poi], target, subtract, args.transition_limit, 1050, 10, "graph_last_poi")
    add_counter(scores, sources, index.category_next[last_cat], target, subtract, args.transition_limit, 760, 5, "graph_last_category")
    if not args.disable_semantic_id_edges:
        add_counter(
            scores,
            sources,
            index.semantic_id_next[last_semantic_id],
            target,
            subtract,
            args.semantic_edge_limit,
            760,
            7,
            "semantic_edge:last_semantic_id",
        )
        add_counter(
            scores,
            sources,
            index.semantic_category_next[last_category_token],
            target,
            subtract,
            args.semantic_edge_limit,
            460,
            3,
            "semantic_edge:last_category_token",
        )
        add_counter(
            scores,
            sources,
            index.semantic_cell_next[last_geo_cell],
            target,
            subtract,
            args.semantic_edge_limit,
            380,
            2,
            "semantic_edge:last_geo_cell",
        )
    if len(trajectory) >= 2:
        prev = trajectory[-2]
        prev_semantic_id = semantic_id(semantic_map, prev.get("poi_id"))
        add_counter(
            scores,
            sources,
            index.last2_poi_next[(str(prev.get("poi_id") or ""), last_poi)],
            target,
            subtract,
            args.transition_limit,
            980,
            12,
            "graph_last2_poi",
        )
        if not args.disable_semantic_id_edges:
            add_counter(
                scores,
                sources,
                index.last2_semantic_id_next[(prev_semantic_id, last_semantic_id)],
                target,
                subtract,
                args.semantic_edge_limit,
                700,
                8,
                "semantic_edge:last2_semantic_id",
            )
        add_counter(
            scores,
            sources,
            index.last2_category_next[(str(prev.get("category") or ""), last_cat)],
            target,
            subtract,
            args.transition_limit,
            720,
            6,
            "graph_last2_category",
        )
    if user_id:
        add_counter(scores, sources, index.user_next[user_id], target, subtract, args.graph_limit, 920, 12, "graph_user")
        for cat in visible_categories(row):
            add_counter(
                scores,
                sources,
                index.user_category_next[(user_id, cat)],
                target,
                subtract,
                args.graph_limit,
                820,
                10,
                f"graph_user_category:{cat}",
            )
        if not args.disable_semantic_id_edges:
            add_counter(
                scores,
                sources,
                index.user_semantic_category_next[(user_id, last_category_token)],
                target,
                subtract,
                args.semantic_edge_limit,
                520,
                5,
                "semantic_edge:user_last_category_token",
            )
            add_counter(
                scores,
                sources,
                index.user_semantic_cell_next[(user_id, last_geo_cell)],
                target,
                subtract,
                args.semantic_edge_limit,
                460,
                4,
                "semantic_edge:user_last_geo_cell",
            )
    for seed in [last_poi, *[str(x.get("poi_id") or "") for x in pref.get("revisited_pois") or []]]:
        add_counter(scores, sources, index.co_visit[seed], target, subtract, args.graph_limit, 560, 5, "graph_covisit")
        if not args.disable_semantic_id_edges:
            add_counter(
                scores,
                sources,
                index.semantic_covisit[semantic_id(semantic_map, seed)],
                target,
                subtract,
                args.semantic_edge_limit,
                420,
                3,
                "semantic_edge:semantic_covisit",
            )
    for cat in visible_categories(row):
        add_counter(scores, sources, index.category_popular[cat], target, subtract, args.category_limit, 410, 3, f"semantic_category:{cat}")
    cat_tokens, cells = semantic_neighbors(semantic_map, row)
    for token in cat_tokens:
        add_counter(scores, sources, index.semantic_category_popular[token], target, subtract, args.category_limit, 360, 2.5, f"semantic_token:{token}")
    for cell in cells:
        add_counter(scores, sources, index.semantic_cell_popular[cell], target, subtract, args.category_limit, 330, 2.0, f"semantic_cell:{cell}")
    if last_slot >= 0:
        add_counter(scores, sources, index.slot_popular[last_slot], target, subtract, args.time_limit, 250, 1.5, "time_slot")
    if last_dow >= 0 and last_slot >= 0:
        add_counter(
            scores,
            sources,
            index.weekday_slot_popular[(last_dow, last_slot)],
            target,
            subtract,
            args.time_limit,
            290,
            2.0,
            "time_weekday_slot",
        )
    add_counter(scores, sources, index.global_popular, target, subtract, args.global_limit, 120, 1.5, "global")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    candidates = [poi for poi, _ in ranked[: args.top_k]]
    details = {}
    for poi in candidates:
        sem = semantic_map.get(poi) or {}
        details[poi] = {
            "semantic_id": sem.get("semantic_id"),
            "category": sem.get("category"),
            "score": round(scores[poi], 4),
            "sources": sources[poi],
        }
    return candidates, details


def percentile(values: List[int], q: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def main() -> None:
    args = parse_args()
    labels = load_by_id(args.labels)
    semantic_map = load_semantic_map(args.semantic_map)
    index = GraphIndex.from_rows(read_jsonl(args.index_inputs), load_by_id(args.index_labels), semantic_map)
    rows = read_jsonl(args.inputs)
    counts = Counter()
    ranks: List[int] = []
    written = 0
    with open_jsonl_writer(args.output, args.overwrite) as f:
        for row in rows:
            sid = str(row.get("sample_id") or "")
            label = labels.get(sid)
            if label is None:
                raise ValueError(f"Missing label for {sid}")
            target = str(label.get("target_poi_id") or "")
            candidates, details = retrieve_candidates(row, label, index, semantic_map, args)
            rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
            counts["rows"] += 1
            for k in [1, 5, 10, 20, 25, 30, 50, 100]:
                counts[f"hit@{k}"] += int(rank is not None and rank <= k)
            counts["legacy_hit"] += int(target in {str(x) for x in row.get("candidate_poi_ids") or []})
            if rank is not None:
                ranks.append(rank)
            target_sem = semantic_map.get(target) or {}
            out_row = {
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
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
    n = counts["rows"]
    report = {
        "output": str(args.output),
        "written": written,
        "rows": n,
        "top_k": args.top_k,
        "legacy_hit_ratio": round(counts["legacy_hit"] / n, 6) if n else 0.0,
        **{f"hit@{k}_ratio": round(counts[f"hit@{k}"] / n, 6) if n else 0.0 for k in [1, 5, 10, 20, 25, 30, 50, 100]},
        "target_rank_when_hit": {
            "p50": percentile(ranks, 0.5),
            "p90": percentile(ranks, 0.9),
            "mean": round(sum(ranks) / len(ranks), 4) if ranks else 0.0,
            "max": max(ranks) if ranks else None,
        },
    }
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".stats.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
