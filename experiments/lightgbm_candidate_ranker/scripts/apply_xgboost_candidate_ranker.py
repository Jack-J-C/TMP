#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HELPER_SCRIPT = ROOT / "experiments/lightgbm_candidate_ranker/scripts/train_sklearn_candidate_ranker.py"
spec = importlib.util.spec_from_file_location("ranker_helpers", HELPER_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {HELPER_SCRIPT}")
helpers = importlib.util.module_from_spec(spec)
sys.modules["ranker_helpers"] = helpers
spec.loader.exec_module(helpers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply a trained XGBoost candidate ranker to TopK candidate JSONL.")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--top-in", type=int, default=500)
    p.add_argument("--top-out", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.03)
    p.add_argument("--graph-prior", choices=["rank_log", "rank_inv", "rank_linear"], default="rank_log")
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


def graph_prior(rank: int, top_in: int, mode: str) -> float:
    rank = max(1, int(rank))
    if mode == "rank_inv":
        return 1.0 / rank
    if mode == "rank_linear":
        return 1.0 - ((rank - 1) / max(top_in - 1, 1))
    if mode == "rank_log":
        return 1.0 / math.log1p(rank)
    raise ValueError(mode)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    vocab = bundle["vocab"]
    rows = read_jsonl(args.candidates)

    features: list[list[float]] = []
    groups: list[dict[str, Any]] = []
    for row in rows:
        candidates = [str(x) for x in row.get("candidate_poi_ids") or []][: args.top_in]
        details = row.get("candidate_details") or {}
        target = str((row.get("target") or {}).get("poi_id") or "")
        start = len(features)
        for idx, poi in enumerate(candidates):
            features.append(helpers.feature_vector(idx + 1, details.get(poi) or {}, vocab))
        groups.append({"row": row, "candidates": candidates, "details": details, "target": target, "start": start, "end": len(features)})

    x_all = np.asarray(features, dtype=np.float32)
    probs = model.predict_proba(x_all)[:, 1] if len(x_all) else np.asarray([], dtype=np.float32)

    out_rows: list[dict[str, Any]] = []
    source_counts = {k: 0 for k in [50, 100, 150, 200, 300, 500]}
    rerank_counts = {k: 0 for k in [50, 100, 150, 200, 300, 500]}
    for group in groups:
        row = group["row"]
        candidates = group["candidates"]
        details = group["details"]
        target = group["target"]
        source_rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
        learned = probs[group["start"] : group["end"]].astype(np.float32)
        learned_norm = helpers.normalize_group_scores(learned)
        priors = np.asarray([graph_prior(idx + 1, args.top_in, args.graph_prior) for idx, _ in enumerate(candidates)], dtype=np.float32)
        final_scores = priors + float(args.alpha) * learned_norm
        ranked_pairs = sorted(zip(candidates, final_scores), key=lambda item: (-float(item[1]), item[0]))
        ranked_candidates = [poi for poi, _ in ranked_pairs]
        rerank_rank = next((idx for idx, poi in enumerate(ranked_candidates, 1) if poi == target), None)
        for k in source_counts:
            source_counts[k] += int(source_rank is not None and source_rank <= k)
            rerank_counts[k] += int(rerank_rank is not None and rerank_rank <= k)
        selected = ranked_pairs[: args.top_out]
        selected_details: dict[str, dict[str, Any]] = {}
        for idx, (poi, score) in enumerate(selected, 1):
            detail = dict(details.get(poi) or {})
            detail["xgb_rank"] = idx
            detail["xgb_score"] = round(float(score), 8)
            detail["source_rank"] = next((rank for rank, src_poi in enumerate(candidates, 1) if src_poi == poi), None)
            selected_details[poi] = detail
        out_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "city": row.get("city"),
                "target": row.get("target"),
                "source_target_rank": source_rank,
                "reranked_target_rank": rerank_rank,
                "candidate_poi_ids": [poi for poi, _ in selected],
                "candidate_scores": {poi: round(float(score), 8) for poi, score in selected},
                "candidate_details": selected_details,
                "target_rank": rerank_rank if rerank_rank is not None and rerank_rank <= args.top_out else None,
                "target_in_topk": bool(rerank_rank is not None and rerank_rank <= args.top_out),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(out_rows)
    report = {
        "candidates": str(args.candidates),
        "model": str(args.model),
        "output": str(args.output),
        "rows": n,
        "top_in": args.top_in,
        "top_out": args.top_out,
        "alpha": args.alpha,
        "graph_prior": args.graph_prior,
        "source_coverage": {f"hit@{k}": round(source_counts[k] / n, 6) if n else 0.0 for k in sorted(source_counts)},
        "reranked_coverage": {f"hit@{k}": round(rerank_counts[k] / n, 6) if n else 0.0 for k in sorted(rerank_counts)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
