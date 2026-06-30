#!/usr/bin/env python3
"""Build C1 SFT data for GraphRAG Top100 -> Top50 candidate compression.

C1 is trained as a lightweight candidate compressor/reranker. The prompt is
label-free; target fields are only used offline to construct supervised output
and report recall ceilings.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build C1 GraphRAG Top100-to-Top50 SFT data.")
    p.add_argument("--inputs", type=Path, required=True, help="Evidence input JSONL.")
    p.add_argument("--graphrag-candidates", type=Path, required=True, help="GraphRAG TopK candidate JSONL.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--semantic-map", type=Path, default=None)
    p.add_argument("--top-in", type=int, default=100)
    p.add_argument("--top-out", type=int, default=50)
    p.add_argument(
        "--target-policy",
        choices=["ensure_top_out", "promote_first", "graph_order"],
        default="ensure_top_out",
        help=(
            "ensure_top_out: keep GraphRAG order, but insert target at the last TopOut slot if target is in TopIn; "
            "promote_first: put target first when target is in TopIn; "
            "graph_order: use original GraphRAG TopOut only."
        ),
    )
    p.add_argument("--include-missing-target", action="store_true", help="Also train rows where target is not in TopIn.")
    p.add_argument("--max-trajectory", type=int, default=8)
    p.add_argument("--max-top-categories", type=int, default=8)
    p.add_argument("--max-revisited", type=int, default=12)
    p.add_argument("--max-sources", type=int, default=5)
    p.add_argument(
        "--output-fields",
        choices=["poi_ids", "poi_and_semantic_ids"],
        default="poi_ids",
        help="Assistant output schema. poi_ids is recommended for fast, robust generation.",
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
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} row without sample_id")
        out[sid] = row
    return out


def load_semantic_map(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[str(row["poi_id"])] = row
    return out


def compact_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def evidence_parts(row: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    evidence = row.get("evidence") or {}
    return evidence.get("sequence") or {}, evidence.get("geo") or {}, evidence.get("preference") or {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_output_candidates(
    candidates: List[str],
    target: str,
    top_in: int,
    top_out: int,
    policy: str,
    include_missing_target: bool,
) -> List[str] | None:
    candidates = [str(x) for x in candidates[:top_in] if x]
    target_in = target in candidates
    if not target_in and not include_missing_target:
        return None
    base = candidates[:top_out]
    if not target_in or policy == "graph_order":
        return base
    if policy == "promote_first":
        return [target] + [poi for poi in candidates if poi != target][: max(0, top_out - 1)]
    if target in base:
        return base
    return base[: max(0, top_out - 1)] + [target]


def format_sources(sources: Any, max_sources: int) -> str:
    if not isinstance(sources, list):
        return "unknown"
    cleaned = [compact_text(x, "") for x in sources if compact_text(x, "")]
    return ",".join(cleaned[:max_sources]) if cleaned else "unknown"


def format_prompt(
    evidence_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    semantic_map: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    seq, geo, pref = evidence_parts(evidence_row)
    lines: List[str] = [
        "Task: Compress the GraphRAG Top100 candidate POIs into exactly Top50 likely next POIs.",
        "Use trajectory, user preference, geographic context, semantic POI IDs, graph scores, and graph evidence sources.",
        "Return JSON only with candidate_poi_ids. Do not output candidate_semantic_ids. Do not add explanations.",
        "",
        f"sample_id: {compact_text(evidence_row.get('sample_id'))}",
        f"city: {compact_text(evidence_row.get('city'))}",
        f"user_id: {compact_text(evidence_row.get('user_id'))}",
    ]
    if seq.get("sequence_summary"):
        lines.append(f"sequence_summary: {seq['sequence_summary']}")
    if geo.get("geo_summary"):
        lines.append(f"geo_summary: {geo['geo_summary']}")
    if pref.get("preference_summary"):
        lines.append(f"preference_summary: {pref['preference_summary']}")

    trajectory = list(seq.get("current_trajectory") or [])[-args.max_trajectory :]
    lines.append("")
    lines.append("current_trajectory_recent:")
    if trajectory:
        for idx, item in enumerate(trajectory, 1):
            poi_id = compact_text(item.get("poi_id"))
            sem = semantic_map.get(poi_id) or {}
            sem_id = sem.get("semantic_id") or "NA"
            lines.append(
                f"{idx}. time={compact_text(item.get('time'))} poi_id={poi_id} "
                f"semantic_id={sem_id} category={compact_text(item.get('category'))} "
                f"dow={compact_text(item.get('dow'))} slot={compact_text(item.get('slot'))}"
            )
    else:
        lines.append("none")

    lines.append("")
    lines.append("user_top_categories:")
    top_categories = list(pref.get("top_categories") or [])[: args.max_top_categories]
    if top_categories:
        lines.append(
            "; ".join(
                f"{compact_text(x.get('category'))}:{compact_text(x.get('count'))}" for x in top_categories
            )
        )
    else:
        lines.append("none")

    lines.append("")
    lines.append("revisited_pois:")
    revisited = list(pref.get("revisited_pois") or [])[: args.max_revisited]
    if revisited:
        parts = []
        for item in revisited:
            poi_id = compact_text(item.get("poi_id"))
            sem = semantic_map.get(poi_id) or {}
            parts.append(f"{poi_id}({sem.get('semantic_id') or 'NA'}, count={compact_text(item.get('count'))})")
        lines.append("; ".join(parts))
    else:
        lines.append("none")

    details = candidate_row.get("candidate_details") or {}
    candidates = list(candidate_row.get("candidate_poi_ids") or [])[: args.top_in]
    lines.append("")
    lines.append("graphrag_top100_candidates:")
    for rank, poi_id in enumerate(candidates, 1):
        detail = details.get(str(poi_id)) or {}
        sem_id = detail.get("semantic_id") or (semantic_map.get(str(poi_id)) or {}).get("semantic_id") or "NA"
        category = detail.get("category") or (semantic_map.get(str(poi_id)) or {}).get("category") or "unknown"
        score = safe_float(detail.get("score"))
        sources = format_sources(detail.get("sources"), args.max_sources)
        lines.append(
            f"{rank:03d}. poi_id={poi_id} semantic_id={sem_id} "
            f"category={category} score={score:.3f} sources={sources}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.top_out > args.top_in:
        raise ValueError("--top-out must be <= --top-in")
    evidence_by_id = load_by_id(args.inputs)
    semantic_map = load_semantic_map(args.semantic_map)
    out_rows: List[Dict[str, Any]] = []
    counts = Counter()
    target_ranks: List[int] = []
    for cand in read_jsonl(args.graphrag_candidates):
        sid = str(cand.get("sample_id") or "")
        evidence_row = evidence_by_id.get(sid)
        if evidence_row is None:
            raise ValueError(f"Missing evidence row for {sid}")
        target = str((cand.get("target") or {}).get("poi_id") or "")
        input_candidates = [str(x) for x in cand.get("candidate_poi_ids") or []]
        target_rank = cand.get("target_rank")
        if isinstance(target_rank, int):
            target_ranks.append(target_rank)
        output_pois = build_output_candidates(
            input_candidates,
            target,
            args.top_in,
            args.top_out,
            args.target_policy,
            args.include_missing_target,
        )
        counts["input_rows"] += 1
        counts["target_in_top_in"] += int(target in input_candidates[: args.top_in])
        counts["target_in_graph_top_out"] += int(target in input_candidates[: args.top_out])
        if output_pois is None:
            counts["skipped_missing_target"] += 1
            continue
        counts["written_rows"] += 1
        counts["target_in_output"] += int(target in output_pois)
        assistant_obj: Dict[str, Any] = {"candidate_poi_ids": output_pois}
        if args.output_fields == "poi_and_semantic_ids":
            assistant_obj["candidate_semantic_ids"] = [
                ((cand.get("candidate_details") or {}).get(poi) or {}).get("semantic_id")
                or (semantic_map.get(poi) or {}).get("semantic_id")
                for poi in output_pois
            ]
        assistant = json.dumps(assistant_obj, ensure_ascii=False, separators=(",", ":"))
        out_rows.append(
            {
                "sample_id": sid,
                "split": evidence_row.get("split"),
                "city": evidence_row.get("city"),
                "task": "c1_graphrag_top100_to_top50",
                "top_in": args.top_in,
                "top_out": args.top_out,
                "target_policy": args.target_policy,
                "output_fields": args.output_fields,
                "offline_target_rank": target_rank,
                "offline_target_in_input": target in input_candidates[: args.top_in],
                "offline_target_in_output": target in output_pois,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are C1, a Semantic-ID and GraphRAG candidate compressor for next-POI prediction. "
                            "Select exactly 50 POI candidates from the provided GraphRAG Top100 list. "
                            "Output valid compact JSON only."
                        ),
                    },
                    {"role": "user", "content": format_prompt(evidence_row, cand, semantic_map, args)},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )

    written = write_jsonl(args.output, out_rows, args.overwrite)
    n = counts["input_rows"]
    w = counts["written_rows"]
    report = {
        "output": str(args.output),
        "written": written,
        "input_rows": n,
        "written_rows": w,
        "skipped_missing_target": counts["skipped_missing_target"],
        "top_in": args.top_in,
        "top_out": args.top_out,
        "target_policy": args.target_policy,
        "include_missing_target": args.include_missing_target,
        "target_in_top_in_ratio": round(counts["target_in_top_in"] / n, 6) if n else 0.0,
        "target_in_graph_top_out_ratio": round(counts["target_in_graph_top_out"] / n, 6) if n else 0.0,
        "target_in_output_ratio_written": round(counts["target_in_output"] / w, 6) if w else 0.0,
        "target_rank_when_hit": {
            "mean": round(sum(target_ranks) / len(target_ranks), 4) if target_ranks else 0.0,
            "max": max(target_ranks) if target_ranks else None,
        },
    }
    report_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
