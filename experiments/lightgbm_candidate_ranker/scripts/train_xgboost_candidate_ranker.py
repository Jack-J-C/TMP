#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[3]
SKLEARN_SCRIPT = ROOT / "experiments/lightgbm_candidate_ranker/scripts/train_sklearn_candidate_ranker.py"
spec = importlib.util.spec_from_file_location("sklearn_ranker_helpers", SKLEARN_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {SKLEARN_SCRIPT}")
helpers = importlib.util.module_from_spec(spec)
sys.modules["sklearn_ranker_helpers"] = helpers
spec.loader.exec_module(helpers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a GPU XGBoost candidate ranker.")
    p.add_argument("--train-candidates", type=Path, required=True)
    p.add_argument("--val-candidates", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--report-output", type=Path, required=True)
    p.add_argument("--pred-output", type=Path, default=None)
    p.add_argument("--top-in", type=int, default=500)
    p.add_argument("--top-outs", nargs="+", type=int, default=[50, 100, 150, 200, 300, 500])
    p.add_argument("--negatives-per-positive", type=int, default=80)
    p.add_argument("--hard-top-n", type=int, default=60)
    p.add_argument("--near-target-window", type=int, default=30)
    p.add_argument("--max-train-groups", type=int, default=None)
    p.add_argument("--max-val-groups", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=1.0)
    p.add_argument("--reg-lambda", type=float, default=0.05)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="Default uses negatives / positives from the sampled train matrix.",
    )
    p.add_argument(
        "--residual-alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
    )
    p.add_argument("--graph-prior", choices=["rank_log", "rank_inv", "rank_linear"], default="rank_log")
    return p.parse_args()


def graph_prior(rank: int, top_in: int, mode: str) -> float:
    rank = max(1, int(rank))
    if mode == "rank_inv":
        return 1.0 / rank
    if mode == "rank_linear":
        return 1.0 - ((rank - 1) / max(top_in - 1, 1))
    if mode == "rank_log":
        return 1.0 / math.log1p(rank)
    raise ValueError(mode)


def batch_rank_val_rows(
    model: XGBClassifier,
    rows: list[dict[str, Any]],
    vocab: dict[str, dict[str, int]],
    top_in: int,
    residual_alphas: list[float],
    graph_prior_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features: list[list[float]] = []
    groups: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in rows:
        candidates = [str(x) for x in row.get("candidate_poi_ids") or []][:top_in]
        details = row.get("candidate_details") or {}
        target = str((row.get("target") or {}).get("poi_id") or "")
        start = len(features)
        for idx, poi in enumerate(candidates):
            features.append(helpers.feature_vector(idx + 1, details.get(poi) or {}, vocab))
            labels.append(int(poi == target))
        groups.append(
            {
                "row": row,
                "candidates": candidates,
                "details": details,
                "target": target,
                "start": start,
                "end": len(features),
            }
        )

    x_val = np.asarray(features, dtype=np.float32)
    probs = model.predict_proba(x_val)[:, 1] if len(x_val) else np.asarray([], dtype=np.float32)
    out_rows: list[dict[str, Any]] = []
    for group in groups:
        row = group["row"]
        candidates = group["candidates"]
        details = group["details"]
        target = group["target"]
        learned_scores = probs[group["start"] : group["end"]]
        learned_norm = helpers.normalize_group_scores(learned_scores.astype(np.float32))
        graph_scores = np.asarray(
            [graph_prior(idx + 1, top_in, graph_prior_mode) for idx, _ in enumerate(candidates)],
            dtype=np.float32,
        )
        ranked_by_alpha: dict[str, list[tuple[str, float]]] = {}
        for alpha in residual_alphas:
            final_scores = graph_scores + float(alpha) * learned_norm
            ranked_by_alpha[str(alpha)] = sorted(zip(candidates, final_scores), key=lambda item: (-float(item[1]), item[0]))
        source_rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
        learned_ranks = {
            str(alpha): next((idx for idx, (poi, _) in enumerate(ranked_by_alpha[str(alpha)], 1) if poi == target), None)
            for alpha in residual_alphas
        }
        last_alpha = str(residual_alphas[-1])
        out_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "target": row.get("target"),
                "source_target_rank": source_rank,
                "learned_target_rank": learned_ranks.get(last_alpha),
                "residual_target_ranks": learned_ranks,
                "candidate_poi_ids": [poi for poi, _ in ranked_by_alpha[last_alpha]],
                "candidate_scores": {poi: round(float(score), 8) for poi, score in ranked_by_alpha[last_alpha][:200]},
            }
        )

    metric: dict[str, Any] = {}
    if len(set(labels)) > 1:
        metric["pair_auc"] = float(roc_auc_score(labels, probs))
        metric["average_precision"] = float(average_precision_score(labels, probs))
    return out_rows, metric


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    train_rows = helpers.read_jsonl(args.train_candidates, args.max_train_groups)
    val_rows = helpers.read_jsonl(args.val_candidates, args.max_val_groups)
    vocab = helpers.build_vocab(train_rows, args.top_in)
    x_train, y_train, train_summary = helpers.build_train_matrix(train_rows, vocab, args)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training data must contain both positive and negative rows.")
    counts = Counter(y_train.tolist())
    scale_pos_weight = args.scale_pos_weight
    if scale_pos_weight is None:
        scale_pos_weight = float(counts[0] / max(counts[1], 1))

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=args.device,
        random_state=args.seed,
        n_jobs=0,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train)
    ranked_rows, pair_metrics = batch_rank_val_rows(
        model,
        val_rows,
        vocab,
        args.top_in,
        args.residual_alphas,
        args.graph_prior,
    )
    report = {
        "train_candidates": str(args.train_candidates),
        "val_candidates": str(args.val_candidates),
        "top_in": args.top_in,
        "top_outs": args.top_outs,
        "residual_alphas": args.residual_alphas,
        "graph_prior": args.graph_prior,
        "feature_count": int(x_train.shape[1]),
        "train_summary": train_summary,
        "model": {
            "type": "xgboost.XGBClassifier",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "min_child_weight": args.min_child_weight,
            "device": args.device,
            "scale_pos_weight": scale_pos_weight,
        },
        "coverage": helpers.coverage(ranked_rows, args.top_outs, args.residual_alphas),
        "pair_metrics": pair_metrics,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("wb") as f:
        pickle.dump({"model": model, "vocab": vocab, "source_keys": helpers.SOURCE_KEYS}, f)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.pred_output:
        write_jsonl(args.pred_output, ranked_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
