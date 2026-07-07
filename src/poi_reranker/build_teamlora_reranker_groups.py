#!/usr/bin/env python3
"""Build grouped TopK samples for candidate-wise TeamLoRA reranking."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build grouped candidate reranker JSONL from joined TopK parquet.")
    p.add_argument("--joined", type=Path, required=True)
    p.add_argument("--semantic-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc


def load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["poi_id"]): row for row in read_jsonl(path)}


def parse_graph_text_details(graph_text: str) -> Dict[str, Dict[str, Any]]:
    details: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(r"^\s*(\d+)\.\s+(v\d+)\|.*?\|score=([^|]+)\|src=(.+?)\s*$")
    for line in graph_text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rank, poi_id, score_text, sources_text = match.groups()
        try:
            score = float(score_text)
        except ValueError:
            score = 0.0
        details[poi_id] = {
            "rank": int(rank),
            "score": score,
            "sources": [x.strip() for x in sources_text.split(",") if x.strip()],
        }
    return details


def semantic_text(row: Dict[str, Any] | None, poi_id: str) -> str:
    if not row:
        return f"{poi_id}|SEM_UNK"
    semantic_id = str(row.get("semantic_id") or "SEM_UNK")
    category = str(row.get("category") or row.get("category_token") or "CAT_UNK")
    geo_cell = str(row.get("geo_cell") or "GEO_UNK")
    lat = row.get("latitude")
    lon = row.get("longitude")
    coord = "coord=UNK" if lat is None or lon is None else f"coord={float(lat):.4f},{float(lon):.4f}"
    return f"{poi_id}|semantic_id={semantic_id}|category={category}|geo={geo_cell}|{coord}"


def poi_hypothesis_text(row: Dict[str, Any] | None, poi_id: str) -> str:
    if not row:
        return f"id={poi_id}; semantic_id=SEM_UNK"
    semantic_id = str(row.get("semantic_id") or "SEM_UNK")
    category = str(row.get("category") or row.get("category_token") or "CAT_UNK")
    geo_cell = str(row.get("geo_cell") or "GEO_UNK")
    lat = row.get("latitude")
    lon = row.get("longitude")
    coord = "UNK" if lat is None or lon is None else f"{float(lat):.4f},{float(lon):.4f}"
    return f"id={poi_id}; semantic_id={semantic_id}; category={category}; geo_cell={geo_cell}; coord={coord}"


def clean_text(value: Any, max_chars: int | None = None) -> str:
    text = str(value or "").strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[TRUNCATED]"
    return text


def build_group(row: Dict[str, Any], semantic_map: Dict[str, Dict[str, Any]], top_k: int) -> Dict[str, Any] | None:
    candidates = [str(x) for x in (row.get("graph_candidate_poi_ids") or [])][:top_k]
    if not candidates:
        return None
    ranks = list(row.get("graph_candidate_ranks") or range(1, len(candidates) + 1))[: len(candidates)]
    scores = list(row.get("graph_candidate_scores") or [0.0] * len(candidates))[: len(candidates)]
    if len(ranks) != len(candidates):
        ranks = list(range(1, len(candidates) + 1))
    if len(scores) != len(candidates):
        scores = [0.0] * len(candidates)
    text_details = parse_graph_text_details(str(row.get("graph_text") or ""))
    target = str(row.get("target_poi_id") or "")
    candidate_rows: List[Dict[str, Any]] = []
    for idx, poi_id in enumerate(candidates):
        detail = text_details.get(poi_id) or {}
        rank = int(detail.get("rank") or ranks[idx])
        try:
            score = float(detail.get("score") if detail.get("score") is not None else scores[idx])
        except (TypeError, ValueError):
            score = 0.0
        sources = detail.get("sources") or []
        sem = semantic_map.get(poi_id)
        candidate_rows.append(
            {
                "poi_id": poi_id,
                "rank": rank,
                "score": score,
                "sources": sources,
                "semantic_id": str((sem or {}).get("semantic_id") or "SEM_UNK"),
                "category": str((sem or {}).get("category") or (sem or {}).get("category_token") or "CAT_UNK"),
                "geo_cell": str((sem or {}).get("geo_cell") or "GEO_UNK"),
                "candidate_text": "\n".join(
                    [
                        f"Candidate POI: {semantic_text(sem, poi_id)}",
                        f"Graph evidence: rank={rank}; score={score:.4f}; sources={','.join(sources) or 'unknown'}",
                    ]
                ),
                "candidate_hypothesis_text": poi_hypothesis_text(sem, poi_id),
                "label": 1 if poi_id == target else 0,
            }
        )
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "split": str(row.get("split") or ""),
        "city": str(row.get("city") or "NewYork"),
        "target_poi_id": target,
        "target_category": str(row.get("target_category") or ""),
        "target_in_candidates": any(x["label"] == 1 for x in candidate_rows),
        "target_rank": row.get("graph_target_rank"),
        "refined_confidence": row.get("refined_confidence"),
        "refined_useful": row.get("refined_useful"),
        "pref_text": clean_text(row.get("raw_text"), max_chars=6000),
        "refine_text": clean_text(row.get("refined_text"), max_chars=2000),
        "user_semantic_profile": clean_text(row.get("user_semantic_profile"), max_chars=3000),
        "graph_text": clean_text(row.get("graph_text"), max_chars=9000),
        "candidates": candidate_rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    semantic_map = load_semantic_map(args.semantic_map)
    table = pq.read_table(args.joined)
    rows = table.to_pylist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"rows": 0, "written": 0, "target_in_candidates": 0, "candidate_count": 0}
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            stats["rows"] += 1
            group = build_group(row, semantic_map, args.top_k)
            if group is None:
                continue
            stats["written"] += 1
            stats["target_in_candidates"] += int(bool(group["target_in_candidates"]))
            stats["candidate_count"] += len(group["candidates"])
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
    stats["target_in_candidates_ratio"] = round(stats["target_in_candidates"] / stats["written"], 6) if stats["written"] else 0.0
    stats["avg_candidates"] = round(stats["candidate_count"] / stats["written"], 3) if stats["written"] else 0.0
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
