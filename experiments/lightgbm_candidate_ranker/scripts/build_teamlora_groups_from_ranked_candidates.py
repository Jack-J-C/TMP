#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TeamLoRA group JSONL from joined parquet + ranked candidate JSONL.")
    p.add_argument("--joined", type=Path, required=True)
    p.add_argument("--ranked-candidates", type=Path, required=True)
    p.add_argument("--semantic-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_semantic_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["poi_id"]): row for row in read_jsonl(path)}


def clean_text(value: Any, max_chars: int | None = None) -> str:
    text = str(value or "").strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[TRUNCATED]"
    return text


def semantic_text(sem: dict[str, Any] | None, poi_id: str) -> str:
    if not sem:
        return f"{poi_id}|SEM_UNK"
    semantic_id = str(sem.get("semantic_id") or "SEM_UNK")
    category = str(sem.get("category") or sem.get("category_token") or "CAT_UNK")
    geo_cell = str(sem.get("geo_cell") or "GEO_UNK")
    lat = sem.get("latitude")
    lon = sem.get("longitude")
    coord = "coord=UNK" if lat is None or lon is None else f"coord={float(lat):.4f},{float(lon):.4f}"
    return f"{poi_id}|semantic_id={semantic_id}|category={category}|geo={geo_cell}|{coord}"


def build_graph_text(candidates: list[str], ranked_row: dict[str, Any], semantic_map: dict[str, dict[str, Any]]) -> str:
    scores = ranked_row.get("candidate_scores") or {}
    details = ranked_row.get("candidate_details") or {}
    lines = ["XGBoost-reranked GraphRAG top100 candidates:"]
    for rank, poi in enumerate(candidates, 1):
        sem = semantic_map.get(poi) or {}
        detail = details.get(poi) or {}
        sources = detail.get("sources") or detail.get("graph_sources") or []
        score = scores.get(poi, detail.get("xgb_score", detail.get("score", 0.0)))
        sem_id = str(sem.get("semantic_id") or "SEM_UNK")
        cat = str(sem.get("category_token") or sem.get("category") or "CAT_UNK")
        cell = str(sem.get("geo_cell") or "GEO_UNK")
        lines.append(f"{rank}. {poi}|{cat}|{cell}|semantic_id={sem_id}|score={float(score):.8f}|src={','.join(map(str, sources)) or 'xgboost_ranker'}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    semantic_map = load_semantic_map(args.semantic_map)
    joined_by_id = {str(row.get("sample_id") or ""): row for row in pq.read_table(args.joined).to_pylist()}
    ranked_rows = read_jsonl(args.ranked_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"rows": 0, "written": 0, "missing_joined": 0, "target_in_candidates": 0, "candidate_count": 0}
    with args.output.open("w", encoding="utf-8") as f:
        for ranked in ranked_rows:
            stats["rows"] += 1
            sid = str(ranked.get("sample_id") or "")
            joined = joined_by_id.get(sid)
            if joined is None:
                stats["missing_joined"] += 1
                continue
            candidates = [str(x) for x in ranked.get("candidate_poi_ids") or []][: args.top_k]
            target = str((ranked.get("target") or {}).get("poi_id") or joined.get("target_poi_id") or "")
            scores = ranked.get("candidate_scores") or {}
            details = ranked.get("candidate_details") or {}
            graph_text = build_graph_text(candidates, ranked, semantic_map)
            candidate_rows = []
            for rank, poi in enumerate(candidates, 1):
                sem = semantic_map.get(poi) or {}
                detail = details.get(poi) or {}
                sources = detail.get("sources") or detail.get("graph_sources") or ["xgboost_ranker"]
                score = float(scores.get(poi, detail.get("xgb_score", detail.get("score", 0.0))) or 0.0)
                candidate_rows.append(
                    {
                        "poi_id": poi,
                        "rank": rank,
                        "score": score,
                        "sources": sources,
                        "semantic_id": str(sem.get("semantic_id") or "SEM_UNK"),
                        "category": str(sem.get("category") or sem.get("category_token") or "CAT_UNK"),
                        "geo_cell": str(sem.get("geo_cell") or "GEO_UNK"),
                        "candidate_text": "\n".join(
                            [
                                f"Candidate POI: {semantic_text(sem, poi)}",
                                f"Graph evidence: rank={rank}; score={score:.8f}; sources={','.join(map(str, sources)) or 'xgboost_ranker'}",
                            ]
                        ),
                        "label": 1 if poi == target else 0,
                    }
                )
            group = {
                "sample_id": sid,
                "split": str(joined.get("split") or ranked.get("split") or ""),
                "city": str(joined.get("city") or ranked.get("city") or "NewYork"),
                "target_poi_id": target,
                "target_category": str((ranked.get("target") or {}).get("category") or joined.get("target_category") or ""),
                "target_in_candidates": any(x["label"] == 1 for x in candidate_rows),
                "target_rank": next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None),
                "refined_confidence": joined.get("refined_confidence"),
                "refined_useful": joined.get("refined_useful"),
                "pref_text": clean_text(joined.get("raw_text") or joined.get("input_text"), max_chars=6000),
                "refine_text": clean_text(joined.get("refined_text"), max_chars=2000),
                "graph_text": clean_text(graph_text, max_chars=9000),
                "candidates": candidate_rows,
            }
            stats["written"] += 1
            stats["target_in_candidates"] += int(bool(group["target_in_candidates"]))
            stats["candidate_count"] += len(candidate_rows)
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
    stats["target_in_candidates_ratio"] = round(stats["target_in_candidates"] / stats["written"], 6) if stats["written"] else 0.0
    stats["avg_candidates"] = round(stats["candidate_count"] / stats["written"], 3) if stats["written"] else 0.0
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
