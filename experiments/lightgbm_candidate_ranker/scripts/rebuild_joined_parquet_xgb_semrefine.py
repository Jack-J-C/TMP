#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild joined Top100 parquet with XGBoost candidates and semantic-code refined evidence.")
    p.add_argument("--joined", type=Path, required=True)
    p.add_argument("--ranked-candidates", type=Path, required=True)
    p.add_argument("--semantic-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stats-output", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--matched-candidate-limit", type=int, default=5)
    p.add_argument("--compact-refine", action="store_true")
    p.add_argument("--omit-full-graph-text", action="store_true")
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


def load_semantic_map(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_poi = {}
    category_to_token = {}
    for row in read_jsonl(path):
        poi = str(row["poi_id"])
        by_poi[poi] = row
        category = str(row.get("category") or "")
        token = str(row.get("category_token") or "")
        if category and token:
            category_to_token[category.lower()] = token
            category_to_token[token.lower()] = token
            category_to_token[category.replace(" ", "_").lower()] = token
    return by_poi, category_to_token


def category_token(category: str, category_to_token: dict[str, str]) -> str:
    raw = str(category or "").strip()
    if not raw:
        return "CAT_UNK"
    return category_to_token.get(raw.lower()) or re.sub(r"[^A-Z0-9_]+", "", raw.upper().replace(" ", "_")) or "CAT_UNK"


def parse_refined(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "verdict": None,
        "confidence": None,
        "hypotheses": [],
        "weak": "",
        "final_hint": "",
    }
    m = re.search(r"useful_for_refinement=([^;.\s]+)", text)
    if m:
        out["verdict"] = m.group(1)
    m = re.search(r"confidence=([^;.\s]+)", text)
    if m:
        out["confidence"] = m.group(1)
    hyp_pattern = re.compile(
        r"(?:^|\s)(\d+)\.\s*category=([^;]+);\s*evidence=([^;]+);\s*reason=(.*?)(?=(?:\s+\d+\.\s*category=)|(?:\s+Weak or noisy cues:)|(?:\s+Final hint:)|$)",
        re.DOTALL,
    )
    for match in hyp_pattern.finditer(text):
        idx, category, evidence, reason = match.groups()
        out["hypotheses"].append(
            {
                "idx": int(idx),
                "category": category.strip(),
                "evidence": evidence.strip(),
                "reason": " ".join(reason.strip().split()),
            }
        )
    m = re.search(r"Weak or noisy cues:\s*(.*?)(?=\s+Final hint:|$)", text, re.DOTALL)
    if m:
        out["weak"] = " ".join(m.group(1).strip().split())
    m = re.search(r"Final hint:\s*(.*)$", text, re.DOTALL)
    if m:
        out["final_hint"] = " ".join(m.group(1).strip().split())
    return out


def matching_candidates(
    candidates: list[str],
    details: dict[str, Any],
    semantic_map: dict[str, dict[str, Any]],
    token: str,
    limit: int = 5,
) -> list[str]:
    matches = []
    for rank, poi in enumerate(candidates, 1):
        sem = semantic_map.get(poi) or {}
        sem_token = str(sem.get("category_token") or "")
        if sem_token != token:
            continue
        detail = details.get(poi) or {}
        sid = sem.get("semantic_id") or detail.get("semantic_id") or "SEM_UNK"
        score = detail.get("xgb_score", detail.get("score", ""))
        matches.append(f"r{rank}:{poi}:{sid}:score={score}")
        if len(matches) >= limit:
            break
    return matches


def build_refine_sem_text(
    original_text: str,
    candidates: list[str],
    details: dict[str, Any],
    semantic_map: dict[str, dict[str, Any]],
    category_to_token: dict[str, str],
    matched_candidate_limit: int = 5,
    compact: bool = False,
) -> str:
    parsed = parse_refined(original_text)
    lines = [
        "[REFINE_SEM]",
        f"verdict={parsed.get('verdict') or 'unk'} confidence={parsed.get('confidence') or 'unk'}",
    ]
    if not compact:
        lines.append("Priority hypotheses use category-level evidence; matched candidates are examples from the current XGBoost Top100, not gold labels.")
    if parsed["hypotheses"]:
        for hyp in parsed["hypotheses"]:
            token = category_token(hyp["category"], category_to_token)
            matches = matching_candidates(candidates, details, semantic_map, token, limit=max(0, matched_candidate_limit))
            if compact:
                reason = hyp["reason"][:160].rstrip()
                lines.append(
                    f"H{hyp['idx']} CAT={token} EVIDENCE={hyp['evidence']} "
                    f"MATCH={';'.join(matches) if matches else 'none'} REASON={reason}"
                )
            else:
                lines.append(
                    f"H{hyp['idx']} CATEGORY_TOKEN={token} EVIDENCE={hyp['evidence']} "
                    f"MATCHED_CANDIDATES={';'.join(matches) if matches else 'none'} "
                    f"REASON={hyp['reason']}"
                )
    else:
        lines.append("H_NONE CAT=CAT_UNK EVIDENCE=unk MATCH=none REASON=parse_failed" if compact else "H_NONE CATEGORY_TOKEN=CAT_UNK EVIDENCE=unk MATCHED_CANDIDATES=none REASON=parse_failed")
    if parsed.get("weak"):
        weak_tokens = []
        for name in re.split(r",| and |;|\(|\)", parsed["weak"]):
            name = name.strip()
            if not name or len(name) > 40:
                continue
            token = category_token(name, category_to_token)
            if token != "CAT_UNK" and token not in weak_tokens:
                weak_tokens.append(token)
        if compact:
            lines.append(f"WEAK_CUES TOKENS={','.join(weak_tokens[:8]) if weak_tokens else 'unk'}")
        else:
            lines.append(f"WEAK_CUES TOKENS={','.join(weak_tokens[:12]) if weak_tokens else 'unk'} TEXT={parsed['weak']}")
    if parsed.get("final_hint"):
        final_hint = parsed["final_hint"][:220].rstrip() if compact else parsed["final_hint"]
        lines.append(f"FINAL_HINT {final_hint}")
    return "\n".join(lines)


def build_graph_text(
    candidates: list[str],
    ranked_row: dict[str, Any],
    semantic_map: dict[str, dict[str, Any]],
    omit_full: bool = False,
) -> str:
    if omit_full:
        return "\n".join(
            [
                "[GRAPH_PRIOR]",
                "XGBoost-reranked GraphRAG Top100 rank/score are stored in graph_candidate_* columns.",
                "Full candidate list text is omitted for dual-expert training.",
            ]
        )
    scores = ranked_row.get("candidate_scores") or {}
    details = ranked_row.get("candidate_details") or {}
    lines = ["XGBoost-reranked GraphRAG top100 candidates:"]
    for rank, poi in enumerate(candidates, 1):
        sem = semantic_map.get(poi) or {}
        detail = details.get(poi) or {}
        sources = detail.get("sources") or detail.get("graph_sources") or ["xgboost_ranker"]
        score = scores.get(poi, detail.get("xgb_score", detail.get("score", 0.0)))
        token = str(sem.get("category_token") or sem.get("category") or "CAT_UNK")
        cell = str(sem.get("geo_cell") or "GEO_UNK")
        sid = str(sem.get("semantic_id") or detail.get("semantic_id") or "SEM_UNK")
        lines.append(
            f"{rank}. {poi}|{token}|{cell}|semantic_id={sid}|score={float(score):.8f}|src={','.join(map(str, sources))}"
        )
    return "\n".join(lines)


def candidate_sources(candidates: list[str], details: dict[str, Any]) -> list[list[str]]:
    out: list[list[str]] = []
    for poi in candidates:
        detail = details.get(poi) or {}
        sources = detail.get("sources") or detail.get("graph_sources") or ["xgboost_ranker"]
        out.append([str(x) for x in sources if str(x)])
    return out


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    semantic_map, category_to_token = load_semantic_map(args.semantic_map)
    ranked_by_id = {str(row.get("sample_id") or ""): row for row in read_jsonl(args.ranked_candidates)}
    df = pd.read_parquet(args.joined)
    rows = []
    stats = {
        "source": str(args.joined),
        "ranked_candidates": str(args.ranked_candidates),
        "output": str(args.output),
        "rows": 0,
        "missing_ranked": 0,
        "target_in_topk": 0,
        "parsed_hypotheses": 0,
        "avg_candidates": 0.0,
        "compact_refine": bool(args.compact_refine),
        "matched_candidate_limit": int(args.matched_candidate_limit),
        "omit_full_graph_text": bool(args.omit_full_graph_text),
        "include_graph_candidate_sources": True,
    }
    total_candidates = 0
    for row in df.to_dict("records"):
        stats["rows"] += 1
        sid = str(row.get("sample_id") or "")
        ranked = ranked_by_id.get(sid)
        if ranked is None:
            stats["missing_ranked"] += 1
            rows.append(row)
            continue
        candidates = [str(x) for x in ranked.get("candidate_poi_ids") or []][: args.top_k]
        details = ranked.get("candidate_details") or {}
        scores = ranked.get("candidate_scores") or {}
        target = str(row.get("target_poi_id") or (ranked.get("target") or {}).get("poi_id") or "")
        ranks = list(range(1, len(candidates) + 1))
        candidate_scores = [float(scores.get(poi, (details.get(poi) or {}).get("xgb_score", 0.0)) or 0.0) for poi in candidates]
        sources = candidate_sources(candidates, details)
        target_rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
        refined_text = str(row.get("refined_text") or "")
        parsed = parse_refined(refined_text)
        stats["parsed_hypotheses"] += len(parsed["hypotheses"])
        row["refined_text"] = build_refine_sem_text(
            refined_text,
            candidates,
            details,
            semantic_map,
            category_to_token,
            matched_candidate_limit=args.matched_candidate_limit,
            compact=args.compact_refine,
        )
        row["graph_text"] = build_graph_text(candidates, ranked, semantic_map, omit_full=args.omit_full_graph_text)
        row["graph_candidate_poi_ids"] = candidates
        row["graph_candidate_ranks"] = ranks
        row["graph_candidate_scores"] = candidate_scores
        row["graph_candidate_sources"] = sources
        row["graph_target_in_topk"] = target_rank is not None
        row["graph_target_rank"] = target_rank
        stats["target_in_topk"] += int(target_rank is not None)
        total_candidates += len(candidates)
        rows.append(row)
    columns = list(df.columns)
    for extra in ("graph_candidate_sources",):
        if extra not in columns:
            columns.append(extra)
    out_df = pd.DataFrame(rows, columns=columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    stats["target_in_topk_ratio"] = round(stats["target_in_topk"] / max(1, stats["rows"]), 6)
    stats["avg_candidates"] = round(total_candidates / max(1, stats["rows"] - stats["missing_ranked"]), 3)
    stats["avg_hypotheses_per_row"] = round(stats["parsed_hypotheses"] / max(1, stats["rows"] - stats["missing_ranked"]), 3)
    stats_output = args.stats_output or args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
