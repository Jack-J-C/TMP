#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


SOURCE_KEYS = [
    "legacy_candidate",
    "same_last_poi",
    "same_last_category",
    "evidence_transition",
    "evidence_geo",
    "history",
    "graph_last_poi",
    "graph_last_category",
    "graph_last2_poi",
    "graph_last2_category",
    "graph_user",
    "graph_covisit",
    "semantic_edge",
    "semantic_category",
    "semantic_token",
    "semantic_cell",
    "time_slot",
    "time_weekday_slot",
    "global",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU candidate compressor for GraphRAG TopK.")
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--val-candidates", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--pred-output", type=Path, default=None)
    parser.add_argument("--top-in", type=int, default=500)
    parser.add_argument("--top-outs", nargs="+", type=int, default=[50, 100, 150, 200])
    parser.add_argument("--negatives-per-positive", type=int, default=80)
    parser.add_argument("--hard-top-n", type=int, default=60)
    parser.add_argument("--near-target-window", type=int, default=30)
    parser.add_argument("--max-train-groups", type=int, default=None)
    parser.add_argument("--max-val-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument(
        "--residual-alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        help="Evaluate final_score = graph_prior + alpha * learned_score. alpha=0 is GraphRAG original order.",
    )
    parser.add_argument(
        "--graph-prior",
        choices=["rank_log", "rank_inv", "rank_linear"],
        default="rank_log",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def category_token(value: Any) -> str:
    text = str(value or "")
    text = text.upper().replace(" ", "_")
    return re.sub(r"[^A-Z0-9_]+", "", text)


def source_match(sources: list[str], key: str) -> float:
    return float(any(key in str(src) for src in sources))


def parse_semantic_id(semantic_id: Any) -> tuple[str, str, int]:
    text = str(semantic_id or "")
    parts = text.split("::")
    cat = parts[1] if len(parts) > 1 else ""
    cell = parts[2] if len(parts) > 2 else ""
    local = parts[3] if len(parts) > 3 else ""
    local_id = 0
    if local.startswith("P"):
        try:
            local_id = int(local[1:])
        except ValueError:
            local_id = 0
    return cat, cell, local_id


def build_vocab(rows: list[dict[str, Any]], top_in: int, max_categories: int = 128, max_cells: int = 256) -> dict[str, dict[str, int]]:
    cats: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    for row in rows:
        details = row.get("candidate_details") or {}
        for poi in (row.get("candidate_poi_ids") or [])[:top_in]:
            detail = details.get(str(poi)) or {}
            cat, cell, _ = parse_semantic_id(detail.get("semantic_id"))
            cat = cat or category_token(detail.get("category"))
            if cat:
                cats[cat] += 1
            if cell:
                cells[cell] += 1
    cat_vocab = {cat: idx + 1 for idx, (cat, _) in enumerate(cats.most_common(max_categories))}
    cell_vocab = {cell: idx + 1 for idx, (cell, _) in enumerate(cells.most_common(max_cells))}
    return {"category": cat_vocab, "cell": cell_vocab}


def feature_vector(rank: int, detail: dict[str, Any], vocab: dict[str, dict[str, int]]) -> list[float]:
    score = float(detail.get("score") or 0.0)
    sources = [str(x) for x in detail.get("sources") or []]
    cat, cell, local_id = parse_semantic_id(detail.get("semantic_id"))
    cat = cat or category_token(detail.get("category"))
    cat_idx = vocab["category"].get(cat, 0)
    cell_idx = vocab["cell"].get(cell, 0)
    source_count = len(sources)
    return [
        float(rank),
        1.0 / math.log1p(max(rank, 1)),
        1.0 / max(rank, 1),
        math.log1p(max(score, 0.0)),
        score / 10000.0,
        float(source_count),
        float(cat_idx),
        float(cell_idx),
        float(local_id),
        *[source_match(sources, key) for key in SOURCE_KEYS],
    ]


def graph_prior(rank: int, score: float, top_in: int, mode: str) -> float:
    rank = max(1, int(rank))
    if mode == "rank_inv":
        return 1.0 / rank
    if mode == "rank_linear":
        return 1.0 - ((rank - 1) / max(top_in - 1, 1))
    if mode == "rank_log":
        return 1.0 / math.log1p(rank)
    raise ValueError(f"Unsupported graph prior: {mode}")


def normalize_group_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        return values * 0.0
    return (values - mean) / std


def choose_training_negatives(
    candidates: list[str],
    target: str,
    target_rank: int,
    negatives_per_positive: int,
    hard_top_n: int,
    near_target_window: int,
    rng: random.Random,
) -> list[int]:
    candidate_count = len(candidates)
    selected: set[int] = set()
    for idx in range(min(candidate_count, hard_top_n)):
        if candidates[idx] != target:
            selected.add(idx)
    if target_rank > 0:
        center = target_rank - 1
        start = max(0, center - near_target_window)
        end = min(candidate_count, center + near_target_window + 1)
        for idx in range(start, end):
            if candidates[idx] != target:
                selected.add(idx)
    remaining = [idx for idx, poi in enumerate(candidates) if poi != target and idx not in selected]
    rng.shuffle(remaining)
    for idx in remaining[: max(0, negatives_per_positive - len(selected))]:
        selected.add(idx)
    return sorted(selected)


def build_train_matrix(
    rows: list[dict[str, Any]],
    vocab: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(args.seed)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    counts = Counter()
    for row in rows:
        candidates = [str(x) for x in row.get("candidate_poi_ids") or []][: args.top_in]
        target = str((row.get("target") or {}).get("poi_id") or "")
        details = row.get("candidate_details") or {}
        target_rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
        counts["groups"] += 1
        if target_rank is None:
            counts["missed_groups"] += 1
            continue
        counts["hit_groups"] += 1
        target_idx = target_rank - 1
        selected = {target_idx}
        selected.update(
            choose_training_negatives(
                candidates,
                target,
                target_rank,
                args.negatives_per_positive,
                args.hard_top_n,
                args.near_target_window,
                rng,
            )
        )
        for idx in sorted(selected):
            poi = candidates[idx]
            x_rows.append(feature_vector(idx + 1, details.get(poi) or {}, vocab))
            y_rows.append(int(poi == target))
    counts["rows"] = len(y_rows)
    counts["positives"] = int(sum(y_rows))
    counts["negatives"] = int(len(y_rows) - sum(y_rows))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), dict(counts)


def rank_val_rows(
    model: HistGradientBoostingClassifier,
    rows: list[dict[str, Any]],
    vocab: dict[str, dict[str, int]],
    top_in: int,
    residual_alphas: list[float],
    graph_prior_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows = []
    labels = []
    probs = []
    for row in rows:
        candidates = [str(x) for x in row.get("candidate_poi_ids") or []][:top_in]
        details = row.get("candidate_details") or {}
        features = np.asarray(
            [feature_vector(idx + 1, details.get(poi) or {}, vocab) for idx, poi in enumerate(candidates)],
            dtype=np.float32,
        )
        if len(features) == 0:
            learned_scores = np.asarray([], dtype=np.float32)
        else:
            learned_scores = model.predict_proba(features)[:, 1]
        graph_scores = np.asarray(
            [
                graph_prior(idx + 1, float((details.get(poi) or {}).get("score") or 0.0), top_in, graph_prior_mode)
                for idx, poi in enumerate(candidates)
            ],
            dtype=np.float32,
        )
        learned_norm = normalize_group_scores(learned_scores.astype(np.float32))
        ranked_by_alpha: dict[str, list[tuple[str, float]]] = {}
        for alpha in residual_alphas:
            final_scores = graph_scores + float(alpha) * learned_norm
            ranked_by_alpha[str(alpha)] = sorted(zip(candidates, final_scores), key=lambda item: (-float(item[1]), item[0]))
        target = str((row.get("target") or {}).get("poi_id") or "")
        source_rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
        learned_ranks = {
            str(alpha): next((idx for idx, (poi, _) in enumerate(ranked_by_alpha[str(alpha)], 1) if poi == target), None)
            for alpha in residual_alphas
        }
        learned_rank = learned_ranks.get(str(residual_alphas[-1]))
        out_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "target": row.get("target"),
                "source_target_rank": source_rank,
                "learned_target_rank": learned_rank,
                "residual_target_ranks": learned_ranks,
                "candidate_poi_ids": [poi for poi, _ in ranked_by_alpha[str(residual_alphas[-1])]],
                "candidate_scores": {poi: round(float(score), 8) for poi, score in ranked_by_alpha[str(residual_alphas[-1])][:200]},
            }
        )
        if source_rank is not None:
            for poi, score in zip(candidates, learned_scores):
                labels.append(int(poi == target))
                probs.append(float(score))
    metric = {}
    if len(set(labels)) > 1:
        metric["pair_auc"] = float(roc_auc_score(labels, probs))
        metric["average_precision"] = float(average_precision_score(labels, probs))
    return out_rows, metric


def coverage(rows: list[dict[str, Any]], top_outs: list[int], residual_alphas: list[float] | None = None) -> dict[str, Any]:
    total = len(rows)
    result: dict[str, Any] = {"rows": total}
    for prefix, key in [("graph", "source_target_rank"), ("learned", "learned_target_rank")]:
        for k in top_outs:
            result[f"{prefix}_hit@{k}"] = round(
                sum(1 for row in rows if row.get(key) is not None and int(row[key]) <= k) / total,
                6,
            ) if total else 0.0
    source_hit = [row for row in rows if row.get("source_target_rank") is not None]
    retained = [row for row in source_hit if row.get("learned_target_rank") is not None]
    result["source_hit@top_in"] = round(len(source_hit) / total, 6) if total else 0.0
    result["learned_retained_any_rank"] = round(len(retained) / total, 6) if total else 0.0
    changes = []
    for row in retained:
        changes.append(int(row["learned_target_rank"]) - int(row["source_target_rank"]))
    if changes:
        result["rank_delta_mean"] = round(float(np.mean(changes)), 4)
        result["rank_delta_improved"] = int(sum(1 for x in changes if x < 0))
        result["rank_delta_worse"] = int(sum(1 for x in changes if x > 0))
    if residual_alphas is not None:
        residual = {}
        for alpha in residual_alphas:
            alpha_key = str(alpha)
            alpha_metrics = {}
            for k in top_outs:
                alpha_metrics[f"hit@{k}"] = round(
                    sum(
                        1
                        for row in rows
                        if (row.get("residual_target_ranks") or {}).get(alpha_key) is not None
                        and int((row.get("residual_target_ranks") or {})[alpha_key]) <= k
                    )
                    / total,
                    6,
                ) if total else 0.0
            residual[alpha_key] = alpha_metrics
        result["residual_alphas"] = residual
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_candidates, args.max_train_groups)
    val_rows = read_jsonl(args.val_candidates, args.max_val_groups)
    vocab = build_vocab(train_rows, args.top_in)
    x_train, y_train, train_summary = build_train_matrix(train_rows, vocab, args)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training data must contain both positive and negative rows.")

    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(x_train, y_train)
    ranked_rows, pair_metrics = rank_val_rows(
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
            "type": "sklearn.HistGradientBoostingClassifier",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "n_iter": int(getattr(model, "n_iter_", -1)),
        },
        "coverage": coverage(ranked_rows, args.top_outs, args.residual_alphas),
        "pair_metrics": pair_metrics,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("wb") as f:
        pickle.dump({"model": model, "vocab": vocab, "source_keys": SOURCE_KEYS}, f)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.pred_output:
        write_jsonl(args.pred_output, ranked_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
