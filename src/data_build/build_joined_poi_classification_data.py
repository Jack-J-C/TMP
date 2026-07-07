#!/usr/bin/env python3
"""Build joined train/val data for POI classification.

This pre-joins:
  semantic raw prompt + full refined prompt + GraphRAG TopK evidence

The output is a compact Parquet file by default, so training reads one file per
split instead of repeatedly scanning three large JSONL sources.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build joined POI classification train/val data.")
    p.add_argument("--base-dir", type=Path, default=Path("retrieval_assets/NewYork"))
    p.add_argument("--output-dir", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification"))
    p.add_argument("--semantic-map", type=Path, default=None)
    p.add_argument("--graph-top-k", type=int, default=30)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    p.add_argument("--format", choices=["parquet", "jsonl", "both"], default="parquet")
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


def load_by_sample_id(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} row without sample_id")
        if sid in out:
            raise ValueError(f"{path} duplicate sample_id: {sid}")
        out[sid] = row
    return out


def load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["poi_id"]): row for row in read_jsonl(path)}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def norm_category_token(value: Any) -> str:
    text = str(value or "CAT_UNK").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or "CAT_UNK"


def category_token_lookup(semantic_map: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in semantic_map.values():
        category = str(row.get("category") or "").strip()
        token = str(row.get("category_token") or "").strip()
        if category and token:
            out.setdefault(category, token)
    return out


def common_slots_from_summary(summary: Any) -> List[str]:
    text = str(summary or "")
    match = re.search(r"Common local time slots are ([^.]+)", text)
    if not match:
        return []
    slots: List[str] = []
    for part in re.split(r"[,，]\s*", match.group(1).strip()):
        part = part.strip()
        if not part:
            continue
        if part.lower().startswith("s"):
            part = part[1:]
        if part.isdigit():
            slots.append(f"s{int(part)}")
    return slots


def build_user_semantic_profile(
    pref: Dict[str, Any] | None,
    semantic_map: Dict[str, Dict[str, Any]],
    category_tokens: Dict[str, str],
    max_categories: int = 4,
    max_revisited: int = 4,
    max_geo: int = 3,
    max_slots: int = 3,
) -> str:
    pref = pref or {}
    history_count = safe_int(pref.get("history_count"))
    lines: List[str] = [
        f"history_count={history_count}",
        "",
        "Long-term category affinity:",
    ]

    top_categories = list(pref.get("top_categories") or [])[:max_categories]
    if top_categories:
        for idx, item in enumerate(top_categories, 1):
            category = str(item.get("category") or item.get("value") or "Unknown")
            count = safe_int(item.get("count"))
            token = category_tokens.get(category) or norm_category_token(category)
            lines.append(f"{idx}. {token}|category={category}|visits={count}")
    else:
        lines.append("- none")

    lines.extend(["", "Revisit affinity:"])
    revisited = list(pref.get("revisited_pois") or [])[:max_revisited]
    geo_counts: Counter[str] = Counter()
    geo_categories: Dict[str, Counter[str]] = {}
    if revisited:
        for idx, item in enumerate(revisited, 1):
            poi_id = str(item.get("poi_id") or item.get("poi") or "")
            count = safe_int(item.get("count"))
            sem = semantic_map.get(poi_id) or {}
            semantic_id = str(sem.get("semantic_id") or "SEM_UNK")
            category = str(sem.get("category") or sem.get("category_token") or "CAT_UNK")
            geo_cell = str(sem.get("geo_cell") or "GEO_UNK")
            lines.append(f"{idx}. {poi_id}|semantic_id={semantic_id}|category={category}|geo={geo_cell}|visits={count}")
            if geo_cell and geo_cell != "GEO_UNK":
                geo_counts[geo_cell] += max(count, 1)
                geo_categories.setdefault(geo_cell, Counter())[category] += max(count, 1)
    else:
        lines.append("- none")

    lines.extend(["", "Geo routine:"])
    if geo_counts:
        for idx, (geo_cell, count) in enumerate(geo_counts.most_common(max_geo), 1):
            cats = ",".join(cat for cat, _ in geo_categories.get(geo_cell, Counter()).most_common(3)) or "unknown"
            lines.append(f"{idx}. geo={geo_cell}|visits={int(count)}|categories={cats}")
    else:
        lines.append("- none")

    lines.extend(["", "Temporal routine:"])
    slots = common_slots_from_summary(pref.get("preference_summary"))[:max_slots]
    if slots:
        for idx, slot in enumerate(slots, 1):
            lines.append(f"{idx}. slot={slot}")
    else:
        lines.append("- none_available")

    return "\n".join(lines)


def compact_geo(row: Dict[str, Any]) -> Tuple[str, bool]:
    if row.get("geo_cell") == "GEO_UNK" or row.get("latitude") is None or row.get("longitude") is None:
        return "UNK", True
    try:
        return f"{float(row['latitude']):.2f},{float(row['longitude']):.2f}", False
    except (TypeError, ValueError):
        return "UNK", True


def semantic_code(semantic_map: Dict[str, Dict[str, Any]], poi_id: str) -> str:
    row = semantic_map.get(str(poi_id))
    if row is None:
        return "SEM_UNK"
    category = str(row.get("category_token") or "CAT_UNK")
    local_id = str(row.get("semantic_id") or "").split("::")[-1] or "P0000"
    geo, missing = compact_geo(row)
    code = f"{category}@{geo}#{local_id}"
    return f"{code}|geo=unk" if missing else code


def strip_generation_instructions(prompt: str) -> str:
    lines: List[str] = []
    skip_output_json = False
    for line in prompt.splitlines():
        text = line.strip()
        if text.startswith("Output only"):
            continue
        if text == "Output format:":
            skip_output_json = True
            continue
        if skip_output_json:
            skip_output_json = False
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def graph_view_text(row: Dict[str, Any], semantic_map: Dict[str, Dict[str, Any]], top_k: int) -> str:
    details = row.get("candidate_details") or {}
    candidates = [str(x) for x in row.get("candidate_poi_ids") or []][:top_k]
    lines = [f"GraphRAG top{len(candidates)} candidates:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for rank, poi in enumerate(candidates, 1):
        detail = details.get(poi) or {}
        score = detail.get("score")
        try:
            score_text = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_text = "na"
        sources = ",".join(str(x) for x in (detail.get("sources") or [])[:4]) or "unknown"
        lines.append(f"{rank}. {poi}|{semantic_code(semantic_map, poi)}|score={score_text}|src={sources}")
    return "\n".join(lines)


def graph_candidate_features(row: Dict[str, Any], top_k: int) -> tuple[List[str], List[int], List[float]]:
    details = row.get("candidate_details") or {}
    candidates = [str(x) for x in row.get("candidate_poi_ids") or []][:top_k]
    ranks: List[int] = []
    scores: List[float] = []
    for rank, poi in enumerate(candidates, 1):
        detail = details.get(poi) or {}
        raw_score = detail.get("score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        ranks.append(rank)
        scores.append(score)
    return candidates, ranks, scores


def split_paths(base_dir: Path, split: str) -> Dict[str, Path]:
    return {
        "raw": base_dir / "semantic_poi_sft" / f"stage1_{split}_raw_semantic.jsonl",
        "preference": base_dir / "evidence" / f"preference_evidence_{split}.jsonl",
        "refined": base_dir / "refined_prompts_decision" / f"lora_a_decision_v2_{split}_full_outputs.jsonl",
        "graph": base_dir / "double_llm" / f"graphrag_semantic_edges_v2_top100_{split}_candidates.jsonl",
    }


def build_split(base_dir: Path, split: str, semantic_map: Dict[str, Dict[str, Any]], graph_top_k: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = split_paths(base_dir, split)
    category_tokens = category_token_lookup(semantic_map)
    preference_by_id = load_by_sample_id(paths["preference"])
    refined_by_id = load_by_sample_id(paths["refined"])
    graph_by_id = load_by_sample_id(paths["graph"])
    rows: List[Dict[str, Any]] = []
    counts = Counter()
    for raw in read_jsonl(paths["raw"]):
        sid = str(raw.get("sample_id") or "")
        preference = preference_by_id.get(sid)
        refined = refined_by_id.get(sid)
        graph = graph_by_id.get(sid)
        if preference is None:
            counts["missing_preference"] += 1
            continue
        if refined is None:
            counts["missing_refined"] += 1
            continue
        if graph is None:
            counts["missing_graph"] += 1
            continue
        target_poi = str(((raw.get("target") or {}).get("poi_id")) or "")
        graph_target = str(((graph.get("target") or {}).get("poi_id")) or "")
        if graph_target and graph_target != target_poi:
            counts["graph_target_mismatch"] += 1
            continue
        raw_text = strip_generation_instructions(str(raw.get("input_prompt") or ""))
        refined_text = str(refined.get("distilled_prompt") or "").strip()
        user_semantic_profile = build_user_semantic_profile(preference, semantic_map, category_tokens)
        graph_text = graph_view_text(graph, semantic_map, graph_top_k)
        graph_candidate_ids, graph_candidate_ranks, graph_candidate_scores = graph_candidate_features(graph, graph_top_k)
        target_rank = int(graph.get("target_rank") or 0)
        target_in_selected_topk = target_rank > 0 and target_rank <= graph_top_k
        input_text = "\n\n".join(
            [
                "[VIEW=RAW_SEM]",
                raw_text,
                "[VIEW=REFINED source=full]",
                refined_text,
                "[VIEW=USER_SEMANTIC_PROFILE source=preference_evidence]",
                user_semantic_profile,
                "[VIEW=GRAPH_RAG]",
                graph_text,
            ]
        )
        rows.append(
            {
                "sample_id": sid,
                "split": split,
                "city": str(raw.get("city") or ""),
                "input_text": input_text,
                "raw_text": raw_text,
                "refined_text": refined_text,
                "user_semantic_profile": user_semantic_profile,
                "graph_text": graph_text,
                "graph_candidate_poi_ids": graph_candidate_ids,
                "graph_candidate_ranks": graph_candidate_ranks,
                "graph_candidate_scores": graph_candidate_scores,
                "target_poi_id": target_poi,
                "target_category": str((raw.get("target") or {}).get("category") or ""),
                "graph_target_in_topk": target_in_selected_topk,
                "graph_target_rank": target_rank,
                "refined_confidence": str(refined.get("refiner_confidence") or ""),
                "refined_useful": str(refined.get("useful_for_refinement") or ""),
            }
        )
        counts["rows"] += 1
        counts[f"refined_confidence_{refined.get('refiner_confidence')}"] += 1
        counts[f"refined_useful_{refined.get('useful_for_refinement')}"] += 1
        counts["graph_target_in_topk"] += int(target_in_selected_topk)
    stats = {
        "split": split,
        "paths": {k: str(v) for k, v in paths.items()},
        "graph_top_k": graph_top_k,
        "counts": dict(counts),
        "graph_target_in_topk_ratio": round(counts["graph_target_in_topk"] / counts["rows"], 6) if counts["rows"] else 0.0,
    }
    return rows, stats


def write_jsonl(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", compression_level=7)


def main() -> None:
    args = parse_args()
    semantic_path = args.semantic_map or (args.base_dir / "double_llm" / "semantic_poi_ids.jsonl")
    semantic_map = load_semantic_map(semantic_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "semantic_map": str(semantic_path),
        "graph_top_k": args.graph_top_k,
        "format": args.format,
        "splits": {},
    }
    for split in args.splits:
        rows, stats = build_split(args.base_dir, split, semantic_map, args.graph_top_k)
        if args.format in {"parquet", "both"}:
            out = args.output_dir / f"{split}_joined_top{args.graph_top_k}.parquet"
            write_parquet(out, rows, args.overwrite)
            stats["parquet_output"] = str(out)
            stats["parquet_bytes"] = out.stat().st_size
        if args.format in {"jsonl", "both"}:
            out = args.output_dir / f"{split}_joined_top{args.graph_top_k}.jsonl"
            write_jsonl(out, rows, args.overwrite)
            stats["jsonl_output"] = str(out)
            stats["jsonl_bytes"] = out.stat().st_size
        stats_path = args.output_dir / f"{split}_joined_top{args.graph_top_k}.stats.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["splits"][split] = stats
    (args.output_dir / f"build_summary_top{args.graph_top_k}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
