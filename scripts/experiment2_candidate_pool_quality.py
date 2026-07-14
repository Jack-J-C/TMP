#!/usr/bin/env python3
"""Experiment2 candidate pool quality probe.

This script studies whether a lightweight learnable pre-recall scorer can
compress a wider GraphRAG pool back to a fixed TopK candidate set with higher
target coverage. It does not train or load TeamLoRA.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


POI_POPULARITY: Counter[str] = Counter()
POI_POPULARITY_MAX_LOG = 1.0
TAIL_THRESHOLD = 2
HEAD_THRESHOLD = 10
TAIL_SCORE_BOOST = 0.0
TAIL_QUOTA = 0

SOURCE_FEATURES = [
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
    "graph_user_category",
    "graph_covisit",
    "semantic_edge_last_semantic_id",
    "semantic_edge_last_category_token",
    "semantic_edge_last_geo_cell",
    "semantic_edge_last2_semantic_id",
    "semantic_edge_user_last_category_token",
    "semantic_edge_user_last_geo_cell",
    "semantic_edge_semantic_covisit",
    "semantic_category",
    "semantic_token",
    "semantic_cell",
    "time_slot",
    "time_weekday_slot",
    "global",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate a fixed-TopK pre-recall candidate pool scorer.")
    p.add_argument("--train-candidates", type=Path, default=None, help="Wide GraphRAG candidate JSONL for fitting scorer.")
    p.add_argument(
        "--eval-candidates",
        nargs="+",
        required=True,
        help="Evaluation candidate files. Use name=path or path.",
    )
    p.add_argument("--top-k", type=int, default=100, help="Final candidate pool size.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--model-type", choices=["linear", "residual_mlp"], default="linear")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-candidates-dir", type=Path, default=None, help="Optional dir for rescored TopK JSONL outputs.")
    p.add_argument("--no-fit", action="store_true", help="Only evaluate graph-order baseline and initialized scorer.")
    p.add_argument(
        "--popularity-candidates",
        type=Path,
        default=None,
        help="Candidate JSONL used to count train target popularity. Defaults to --train-candidates.",
    )
    p.add_argument("--tail-threshold", type=int, default=2, help="Train target count <= this is tail.")
    p.add_argument("--head-threshold", type=int, default=10, help="Train target count > this is head; between is mid.")
    p.add_argument("--tail-loss-weight", type=float, default=2.0, help="Loss multiplier for tail target rows.")
    p.add_argument("--mid-loss-weight", type=float, default=1.25, help="Loss multiplier for mid-frequency target rows.")
    p.add_argument("--tail-score-boost", type=float, default=0.0, help="Inference score boost for tail candidate POIs.")
    p.add_argument("--tail-quota", type=int, default=0, help="Optional minimum number of tail POIs in final TopK.")
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_eval_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path)
    path = Path(spec)
    return path.stem, path


def target_poi(row: Dict[str, Any]) -> str:
    target = row.get("target") or {}
    return str(target.get("poi_id") or row.get("target_poi_id") or "")


def build_popularity(rows: Sequence[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        poi = target_poi(row)
        if poi:
            counts[poi] += 1
    return counts


def popularity_bucket(poi: str) -> str:
    count = POI_POPULARITY.get(str(poi), 0)
    if count <= TAIL_THRESHOLD:
        return "tail"
    if count <= HEAD_THRESHOLD:
        return "mid"
    return "head"


def popularity_features(poi: str) -> List[float]:
    count = POI_POPULARITY.get(str(poi), 0)
    log_pop = math.log1p(count) / max(POI_POPULARITY_MAX_LOG, 1e-6)
    inv_pop = 1.0 / math.sqrt(float(count) + 1.0)
    bucket = popularity_bucket(poi)
    return [
        log_pop,
        inv_pop,
        1.0 if bucket == "tail" else 0.0,
        1.0 if bucket == "mid" else 0.0,
        1.0 if bucket == "head" else 0.0,
    ]


def canonical_source(source: Any) -> str | None:
    text = str(source or "")
    if not text:
        return None
    text = text.replace(":", "_")
    if text.startswith("graph_user_category"):
        return "graph_user_category"
    if text.startswith("semantic_edge_last_semantic_id"):
        return "semantic_edge_last_semantic_id"
    if text.startswith("semantic_edge_last_category_token"):
        return "semantic_edge_last_category_token"
    if text.startswith("semantic_edge_last_geo_cell"):
        return "semantic_edge_last_geo_cell"
    if text.startswith("semantic_edge_last2_semantic_id"):
        return "semantic_edge_last2_semantic_id"
    if text.startswith("semantic_edge_user_last_category_token"):
        return "semantic_edge_user_last_category_token"
    if text.startswith("semantic_edge_user_last_geo_cell"):
        return "semantic_edge_user_last_geo_cell"
    if text.startswith("semantic_edge_semantic_covisit"):
        return "semantic_edge_semantic_covisit"
    if text.startswith("semantic_category"):
        return "semantic_category"
    if text.startswith("semantic_token"):
        return "semantic_token"
    if text.startswith("semantic_cell"):
        return "semantic_cell"
    if text.startswith("same_last_poi"):
        return "same_last_poi"
    if text.startswith("same_last_category"):
        return "same_last_category"
    if text.startswith("evidence_transition"):
        return "evidence_transition"
    if text in SOURCE_FEATURES:
        return text
    return None


def source_flags(sources: Sequence[Any]) -> List[float]:
    seen = {canonical_source(src) for src in sources}
    return [1.0 if name in seen else 0.0 for name in SOURCE_FEATURES]


def candidate_features(row: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[float]]:
    candidates = [str(x) for x in row.get("candidate_poi_ids") or [] if str(x)]
    details = row.get("candidate_details") or {}
    n = max(len(candidates), 1)
    rows: List[List[float]] = []
    graph_scores: List[float] = []
    for idx, poi in enumerate(candidates, 1):
        detail = details.get(poi) or {}
        try:
            graph_score = float(detail.get("score") or 0.0)
        except (TypeError, ValueError):
            graph_score = 0.0
        graph_scores.append(graph_score)
        sources = detail.get("sources") or []
        if not isinstance(sources, list):
            sources = [sources]
        rank_norm = 1.0 - float(idx - 1) / float(max(n - 1, 1))
        log_score = math.log1p(max(graph_score, 0.0)) / 12.0
        feats = [
            1.0,
            1.0 / math.log2(idx + 1.0),
            rank_norm,
            log_score,
            min(float(len(sources)), 20.0) / 20.0,
        ]
        feats.extend(popularity_features(poi))
        feats.extend(source_flags(sources))
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32), candidates, graph_scores


def feature_names() -> List[str]:
    return [
        "bias",
        "rank_prior",
        "rank_norm",
        "log_graph_score",
        "source_count",
        "train_target_pop_log",
        "train_target_pop_inv",
        "is_tail_poi",
        "is_mid_poi",
        "is_head_poi",
    ] + SOURCE_FEATURES


class LinearPoolScorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class ResidualMLPPoolScorer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x) + self.residual(x)).squeeze(-1)


def init_model(dim: int, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model_type == "residual_mlp":
        model: nn.Module = ResidualMLPPoolScorer(dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    else:
        model = LinearPoolScorer(dim).to(device)
    with torch.no_grad():
        model.linear.weight.zero_()
        names = feature_names()
        # GraphRAG initialization: start close to original graph ordering.
        for name, value in {
            "rank_prior": 4.0,
            "rank_norm": 2.0,
            "log_graph_score": 1.0,
            "source_count": 0.2,
            "is_tail_poi": 0.1,
        }.items():
            if name in names:
                model.linear.weight[0, names.index(name)] = value
    return model


def rows_with_target(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
    out: List[Tuple[Dict[str, Any], int]] = []
    for row in rows:
        candidates = [str(x) for x in row.get("candidate_poi_ids") or []]
        target = target_poi(row)
        if target and target in candidates:
            out.append((row, candidates.index(target)))
    return out


def fit_model(model: nn.Module, train_rows: Sequence[Dict[str, Any]], args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    random.seed(args.seed)
    usable = rows_with_target(train_rows)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        random.shuffle(usable)
        total_loss = 0.0
        total = 0
        hit100 = 0
        bucket_hits = Counter()
        bucket_total = Counter()
        for row, target_idx in usable:
            feats, _, _ = candidate_features(row)
            x = torch.from_numpy(feats).to(device)
            y = torch.tensor([target_idx], dtype=torch.long, device=device)
            scores = model(x).unsqueeze(0)
            bucket = popularity_bucket(target_poi(row))
            weight = args.tail_loss_weight if bucket == "tail" else args.mid_loss_weight if bucket == "mid" else 1.0
            loss = F.cross_entropy(scores, y) * float(weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            total += 1
            bucket_total[bucket] += 1
            with torch.no_grad():
                rank = int((scores.squeeze(0) > scores.squeeze(0)[target_idx]).sum().item()) + 1
                hit100 += int(rank <= args.top_k)
                bucket_hits[bucket] += int(rank <= args.top_k)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(total, 1),
                f"train_hit@{args.top_k}": hit100 / max(total, 1),
                "train_hit_by_popularity": {
                    bucket: bucket_hits[bucket] / max(bucket_total[bucket], 1)
                    for bucket in ["tail", "mid", "head"]
                    if bucket_total[bucket]
                },
                "usable_train_rows": total,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    return {"train_rows": len(train_rows), "usable_train_rows": len(usable), "history": history}


def adjusted_scores(candidates: Sequence[str], scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).copy()
    if TAIL_SCORE_BOOST:
        for idx, poi in enumerate(candidates):
            if popularity_bucket(poi) == "tail":
                arr[idx] += float(TAIL_SCORE_BOOST)
    return arr


def ranked_indices(candidates: Sequence[str], scores: Sequence[float], top_k: int | None = None) -> List[int]:
    adjusted = adjusted_scores(candidates, scores)
    order = sorted(range(len(candidates)), key=lambda i: (-float(adjusted[i]), candidates[i]))
    if top_k is None or TAIL_QUOTA <= 0:
        return order if top_k is None else order[:top_k]
    selected = order[:top_k]
    selected_set = set(selected)
    tail_selected = [idx for idx in selected if popularity_bucket(candidates[idx]) == "tail"]
    if len(tail_selected) >= TAIL_QUOTA:
        return selected
    tail_candidates = [idx for idx in order[top_k:] if popularity_bucket(candidates[idx]) == "tail"]
    need = min(TAIL_QUOTA - len(tail_selected), len(tail_candidates))
    if need <= 0:
        return selected
    # Replace the lowest-scored non-tail selected candidates first.
    replaceable = [idx for idx in reversed(selected) if popularity_bucket(candidates[idx]) != "tail"]
    for add_idx, drop_idx in zip(tail_candidates[:need], replaceable):
        selected_set.discard(drop_idx)
        selected_set.add(add_idx)
    return sorted(selected_set, key=lambda i: (-float(adjusted[i]), candidates[i]))[:top_k]


def rank_from_scores(candidates: Sequence[str], scores: Sequence[float], target: str) -> int | None:
    if not target or target not in set(candidates):
        return None
    order = ranked_indices(candidates, scores)
    for rank, idx in enumerate(order, 1):
        if candidates[idx] == target:
            return rank
    return None


def summarize_ranks(ranks: Sequence[int | None], top_k: int) -> Dict[str, Any]:
    total = len(ranks)
    present = [r for r in ranks if r is not None]
    arr = np.asarray(present, dtype=np.int64) if present else np.asarray([], dtype=np.int64)
    out: Dict[str, Any] = {
        "evaluated": total,
        "candidate_hit_all": len(present) / max(total, 1),
    }
    for k in [1, 5, 10, 20, 50, top_k]:
        key = f"hit@{k}"
        out[key] = float(np.mean(arr <= k)) * (len(present) / max(total, 1)) if len(arr) else 0.0
    if len(arr):
        out["mrr"] = float(np.sum(1.0 / arr) / max(total, 1))
        out["rank_when_hit"] = {
            "mean": float(np.mean(arr)),
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
            "max": int(np.max(arr)),
        }
    else:
        out["mrr"] = 0.0
        out["rank_when_hit"] = {"mean": 0.0, "p50": None, "p90": None, "max": None}
    return out


def summarize_by_bucket(rows: Sequence[Dict[str, Any]], ranks: Sequence[int | None], top_k: int) -> Dict[str, Any]:
    grouped: Dict[str, List[int | None]] = {"tail": [], "mid": [], "head": []}
    for row, rank in zip(rows, ranks):
        grouped[popularity_bucket(target_poi(row))].append(rank)
    return {bucket: summarize_ranks(values, top_k) for bucket, values in grouped.items() if values}


def score_rows(model: nn.Module, rows: Sequence[Dict[str, Any]], device: torch.device) -> List[np.ndarray]:
    all_scores: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            feats, _, _ = candidate_features(row)
            scores = model(torch.from_numpy(feats).to(device)).detach().cpu().numpy()
            all_scores.append(scores)
    return all_scores


def evaluate_rows(model: nn.Module, rows: Sequence[Dict[str, Any]], top_k: int, device: torch.device) -> Dict[str, Any]:
    learned_scores = score_rows(model, rows, device)
    baseline_ranks: List[int | None] = []
    learned_ranks: List[int | None] = []
    wide_ranks: List[int | None] = []
    candidate_sizes = []
    source_counts = Counter()
    rescued = 0
    lost = 0
    for row, scores in zip(rows, learned_scores):
        feats, candidates, graph_scores = candidate_features(row)
        del feats
        target = target_poi(row)
        candidate_sizes.append(len(candidates))
        for poi in candidates[:top_k]:
            for src in (row.get("candidate_details") or {}).get(poi, {}).get("sources", []) or []:
                canon = canonical_source(src)
                if canon:
                    source_counts[canon] += 1
        baseline_rank = next((idx for idx, poi in enumerate(candidates[:top_k], 1) if poi == target), None)
        learned_order = ranked_indices(candidates, scores, top_k)
        learned_rank = next((idx for idx, cand_idx in enumerate(learned_order, 1) if candidates[cand_idx] == target), None)
        wide_rank = next((idx for idx, poi in enumerate(candidates, 1) if poi == target), None)
        baseline_ranks.append(baseline_rank)
        learned_ranks.append(learned_rank)
        wide_ranks.append(wide_rank)
        rescued += int(baseline_rank is None and learned_rank is not None)
        lost += int(baseline_rank is not None and learned_rank is None)
    out = {
        "graph_order_topk": summarize_ranks(baseline_ranks, top_k),
        "learned_rescore_topk": summarize_ranks(learned_ranks, top_k),
        "wide_pool_oracle": summarize_ranks(wide_ranks, top_k),
        "graph_order_topk_by_popularity": summarize_by_bucket(rows, baseline_ranks, top_k),
        "learned_rescore_topk_by_popularity": summarize_by_bucket(rows, learned_ranks, top_k),
        "wide_pool_oracle_by_popularity": summarize_by_bucket(rows, wide_ranks, top_k),
        "rescued_by_rescore": rescued,
        "lost_by_rescore": lost,
        "candidate_size": {
            "mean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
            "min": int(np.min(candidate_sizes)) if candidate_sizes else 0,
            "max": int(np.max(candidate_sizes)) if candidate_sizes else 0,
        },
        "topk_source_counts_graph_order": dict(source_counts.most_common()),
    }
    out["delta_hit@topk"] = out["learned_rescore_topk"][f"hit@{top_k}"] - out["graph_order_topk"][f"hit@{top_k}"]
    return out


def rescored_rows(model: nn.Module, rows: Sequence[Dict[str, Any]], top_k: int, device: torch.device) -> Iterable[Dict[str, Any]]:
    learned_scores = score_rows(model, rows, device)
    for row, scores in zip(rows, learned_scores):
        candidates = [str(x) for x in row.get("candidate_poi_ids") or [] if str(x)]
        order = ranked_indices(candidates, scores, top_k)
        selected = [candidates[i] for i in order]
        old_details = row.get("candidate_details") or {}
        details: Dict[str, Any] = {}
        for new_rank, idx in enumerate(order, 1):
            poi = candidates[idx]
            detail = dict(old_details.get(poi) or {})
            detail["score"] = round(float(scores[idx]), 6)
            detail["original_graph_rank"] = idx + 1
            if "sources" in detail and isinstance(detail["sources"], list):
                if "learned_rescore" not in detail["sources"]:
                    detail["sources"] = list(detail["sources"]) + ["learned_rescore"]
            else:
                detail["sources"] = ["learned_rescore"]
            details[poi] = detail
        target = target_poi(row)
        target_rank = next((idx for idx, poi in enumerate(selected, 1) if poi == target), None)
        out = dict(row)
        out["candidate_poi_ids"] = selected
        out["candidate_semantic_ids"] = [details.get(poi, {}).get("semantic_id") for poi in selected]
        out["candidate_details"] = details
        out["target_rank"] = target_rank
        out["target_in_topk"] = target_rank is not None
        yield out


def main() -> None:
    global POI_POPULARITY, POI_POPULARITY_MAX_LOG, TAIL_THRESHOLD, HEAD_THRESHOLD, TAIL_SCORE_BOOST, TAIL_QUOTA
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    TAIL_THRESHOLD = int(args.tail_threshold)
    HEAD_THRESHOLD = int(args.head_threshold)
    TAIL_SCORE_BOOST = float(args.tail_score_boost)
    TAIL_QUOTA = int(args.tail_quota)

    popularity_path = args.popularity_candidates or args.train_candidates
    if popularity_path:
        POI_POPULARITY = build_popularity(read_jsonl(popularity_path))
        POI_POPULARITY_MAX_LOG = math.log1p(max(POI_POPULARITY.values() or [1]))

    eval_specs = [parse_eval_spec(spec) for spec in args.eval_candidates]
    first_rows = read_jsonl(eval_specs[0][1])
    if not first_rows:
        raise ValueError(f"No rows in {eval_specs[0][1]}")
    dim = candidate_features(first_rows[0])[0].shape[1]
    model = init_model(dim, args, device)
    train_summary: Dict[str, Any] | None = None
    if args.train_candidates and not args.no_fit:
        train_rows = read_jsonl(args.train_candidates)
        train_summary = fit_model(model, train_rows, args, device)

    weights = {
        name: float(value)
        for name, value in zip(feature_names(), model.linear.weight.detach().cpu().numpy().reshape(-1).tolist())
    }
    report: Dict[str, Any] = {
        "top_k": args.top_k,
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim if args.model_type == "residual_mlp" else None,
        "dropout": args.dropout if args.model_type == "residual_mlp" else None,
        "tail_threshold": TAIL_THRESHOLD,
        "head_threshold": HEAD_THRESHOLD,
        "tail_loss_weight": args.tail_loss_weight,
        "mid_loss_weight": args.mid_loss_weight,
        "tail_score_boost": TAIL_SCORE_BOOST,
        "tail_quota": TAIL_QUOTA,
        "popularity_source": str(popularity_path) if popularity_path else None,
        "popularity_summary": {
            "known_pois": len(POI_POPULARITY),
            "tail_pois": sum(1 for count in POI_POPULARITY.values() if count <= TAIL_THRESHOLD),
            "mid_pois": sum(1 for count in POI_POPULARITY.values() if TAIL_THRESHOLD < count <= HEAD_THRESHOLD),
            "head_pois": sum(1 for count in POI_POPULARITY.values() if count > HEAD_THRESHOLD),
            "max_target_count": max(POI_POPULARITY.values() or [0]),
        },
        "feature_names": feature_names(),
        "weights": weights,
        "train": train_summary,
        "eval": {},
    }
    for name, path in eval_specs:
        rows = read_jsonl(path)
        metrics = evaluate_rows(model, rows, args.top_k, device)
        metrics["path"] = str(path)
        report["eval"][name] = metrics
        print(json.dumps({name: metrics}, ensure_ascii=False, indent=2), flush=True)
        if args.output_candidates_dir:
            out_path = args.output_candidates_dir / f"{name}_learned_top{args.top_k}_candidates.jsonl"
            count = write_jsonl(out_path, rescored_rows(model, rows, args.top_k, device))
            report["eval"][name]["rescored_candidates"] = str(out_path)
            report["eval"][name]["rescored_rows"] = count

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
