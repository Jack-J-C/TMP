#!/usr/bin/env python3
"""Train an anonymous TeamLoRA mobility encoder with online Top500 compression.

This version keeps the fixed candidate pool concept, but trains a lightweight
linear compressor on Top500 candidates jointly with the LLM mobility encoder
and a DCNv2 fusion head.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from build_teamlora_reranker_groups import load_semantic_map
from train_anonymous_mobility_encoder import (
    MobilityAligner,
    RoutedTeamLoRALinear,
    TARGET_MODULES,
    clean_text,
    coerce_path_defaults,
    encode_texts,
    gated_similar_user_profile,
    normalize_config_keys,
    parse_profile_signal_sets,
    parse_recent_context,
    parse_step_list,
    read_jsonl,
    score_group_loss,
    set_seed,
    source_family_statistics,
    stable_bucket,
    strip_preference_section,
    user_semantic_profile_text,
)


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

POOL_FEATURE_DIM = 5 + 5 + len(SOURCE_FEATURES)
FAMILY_NAMES = ("transition", "geo", "history", "semantic", "time", "other")
BASE_RELATION_DIM = 16
ENHANCED_RELATION_DIM = 28
EVENT_NUMERIC_DIM = 16
LLAMA_TARGET_MODULES = TARGET_MODULES
BERT_TARGET_MODULES = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None, help="Optional YAML config file. CLI args override config values.")
    config_args, _ = bootstrap.parse_known_args()

    p = argparse.ArgumentParser(description="Train an anonymous TeamLoRA mobility encoder with online pool compression.")
    p.add_argument("--config", type=Path, default=None, help="Optional YAML config file. CLI args override config values.")
    p.add_argument("--train-joined", type=Path, default=Path("retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/train_joined_top100.parquet"))
    p.add_argument("--val-joined", type=Path, default=Path("retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/val_joined_top100.parquet"))
    p.add_argument("--test-joined", type=Path, default=Path("retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/test_joined_top100.parquet"))
    p.add_argument("--train-candidates", type=Path, default=Path("retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_train_Candidates.jsonl"))
    p.add_argument("--val-candidates", type=Path, default=Path("retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_val_Candidates.jsonl"))
    p.add_argument("--test-candidates", type=Path, default=Path("retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_test_Candidates.jsonl"))
    p.add_argument("--train-teacher-candidates", type=Path, default=None)
    p.add_argument("--val-teacher-candidates", type=Path, default=None)
    p.add_argument("--test-teacher-candidates", type=Path, default=None)
    p.add_argument("--semantic-map", type=Path, default=Path("retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl"))
    p.add_argument("--base-model", type=Path, default=Path("models/Llama-3.2-1B-Instruct"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--pool-k", type=int, default=500, help="Input wide candidate pool size.")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-length", type=int, default=1280)
    p.add_argument("--input-template", choices=["semantic_profile_mobility_v1", "semantic_profile_simuser_mobility_v1"], default="semantic_profile_simuser_mobility_v1")
    p.add_argument("--structured-mobility-input", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enhanced-behavior-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-trajectory-events", type=int, default=16)
    p.add_argument("--train-hit-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--val-hit-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--test-hit-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--batch-groups", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-num", type=int, default=3)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-preset", choices=["auto", "llama", "bert"], default="auto")
    p.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--text-proj-dim", type=int, default=256)
    p.add_argument("--anchor-proj-dim", type=int, default=256)
    p.add_argument("--anchor-hidden-dim", type=int, default=256)
    p.add_argument("--relation-hidden-dim", type=int, default=128)
    p.add_argument("--relation-proj-dim", type=int, default=128)
    p.add_argument("--category-bucket-size", type=int, default=256)
    p.add_argument("--geo-bucket-size", type=int, default=1024)
    p.add_argument("--semantic-bucket-size", type=int, default=4096)
    p.add_argument("--alignment-mode", choices=["linear", "low_rank", "dcnv2"], default="dcnv2")
    p.add_argument("--alignment-dropout", type=float, default=0.1)
    p.add_argument("--candidate-token-attention", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--din-history-attention", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pool-hidden-dim", type=int, default=0, help="Optional hidden dim for compressor; 0 keeps it linear.")
    p.add_argument("--pool-dropout", type=float, default=0.0)
    p.add_argument("--pool-alpha-init", type=float, default=1.0)
    p.add_argument("--fusion-alpha-init", type=float, default=0.3)
    p.add_argument("--keep-margin", type=float, default=0.1)
    p.add_argument("--pool-loss-weight", type=float, default=1.0)
    p.add_argument("--keep-loss-weight", type=float, default=1.0)
    p.add_argument("--final-loss-weight", type=float, default=1.0)
    p.add_argument("--fixed-pool-prior", choices=["none", "rank_log", "rank_linear"], default="none")
    p.add_argument("--selection-mode", choices=["score", "original"], default="score")
    p.add_argument("--final-score-mode", choices=["fused", "dcn_only"], default="fused")
    p.add_argument("--train-negatives", type=int, default=0)
    p.add_argument("--hard-negatives", type=int, default=0)
    p.add_argument("--raat-mode", choices=["none", "target_mask_2view"], default="none")
    p.add_argument("--raat-loss-weight", type=float, default=0.0)
    p.add_argument("--teacher-loss-weight", type=float, default=0.0)
    p.add_argument("--teacher-top-k", type=int, default=20)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--eval-candidate-batch-size", type=int, default=32)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--eval-at-steps", default=None)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-groups", type=int, default=None)
    p.add_argument("--max-val-groups", type=int, default=None)
    p.add_argument("--max-test-groups", type=int, default=None)
    p.add_argument("--eval-candidate-limit", type=int, default=None)
    p.add_argument("--eval-final-only", action=argparse.BooleanOptionalAction, default=False)

    if config_args.config is not None:
        config = normalize_config_keys(load_yaml_config(config_args.config))
        valid_keys = {action.dest for action in p._actions}
        unknown = sorted(set(config) - valid_keys)
        if unknown:
            p.error(f"Unknown config key(s) in {config_args.config}: {', '.join(unknown)}")
        p.set_defaults(**config)

    args = p.parse_args()
    coerce_path_defaults(p, args)
    if args.output_dir is None:
        p.error("--output-dir is required, either on the command line or in --config")
    return args


def load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required for --config. Install pyyaml or pass CLI args directly.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {path}")
    return dict(data)


def read_joined_parquet(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    return rows if limit is None else rows[:limit]


def read_candidate_jsonl_map(path: Path, limit: int | None = None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(read_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        sample_id = str(row.get("sample_id") or "")
        if sample_id:
            out[sample_id] = row
    return out


def target_from_candidate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    target = row.get("target") or {}
    return {
        "poi_id": str(target.get("poi_id") or row.get("target_poi_id") or ""),
        "semantic_id": str(target.get("semantic_id") or ""),
        "category": str(target.get("category") or row.get("target_category") or ""),
    }


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


def popularity_bucket(poi: str, pop: Dict[str, int], tail_threshold: int = 2, head_threshold: int = 10) -> str:
    count = pop.get(str(poi), 0)
    if count <= tail_threshold:
        return "tail"
    if count <= head_threshold:
        return "mid"
    return "head"


def popularity_features(poi: str, pop: Dict[str, int], tail_threshold: int = 2, head_threshold: int = 10) -> List[float]:
    count = pop.get(str(poi), 0)
    max_log = max(1.0, math.log1p(max(pop.values()) if pop else 1))
    bucket = popularity_bucket(poi, pop, tail_threshold=tail_threshold, head_threshold=head_threshold)
    return [
        math.log1p(count) / max_log,
        1.0 / math.sqrt(float(count) + 1.0),
        1.0 if bucket == "tail" else 0.0,
        1.0 if bucket == "mid" else 0.0,
        1.0 if bucket == "head" else 0.0,
    ]


def build_popularity(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        target = target_from_candidate_row(row)
        poi = target["poi_id"]
        if poi:
            counts[poi] = counts.get(poi, 0) + 1
    return counts


def candidate_pool_features(row: Dict[str, Any], popularity: Dict[str, int]) -> Tuple[np.ndarray, List[str], List[float]]:
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
        feats.extend(popularity_features(poi, popularity))
        feats.extend(source_flags(sources))
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32), candidates, graph_scores


def fixed_pool_prior_scores(candidates: Sequence[Dict[str, Any]], mode: str) -> torch.Tensor:
    n = max(1, len(candidates))
    values: List[float] = []
    for idx, _candidate in enumerate(candidates):
        rank = float(idx + 1)
        if mode == "rank_log":
            values.append(1.0 / math.log2(rank + 1.0))
        elif mode == "rank_linear":
            values.append(1.0 - float(idx) / float(max(n - 1, 1)))
        else:
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def slot_index(slot: Any) -> int:
    text = str(slot or "").strip().lower()
    if text.startswith("s"):
        text = text[1:]
    try:
        return max(0, min(47, int(text)))
    except ValueError:
        return 0


def weekday_index(weekday: Any) -> int:
    mapping = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    return mapping.get(str(weekday or "").strip().lower()[:3], 0)


def cyclic_pair(index: int, period: int) -> Tuple[float, float]:
    angle = 2.0 * math.pi * float(index % period) / float(max(1, period))
    return math.sin(angle), math.cos(angle)


def geo_cell_xy(geo_cell: Any) -> Tuple[float | None, float | None]:
    text = str(geo_cell or "")
    match = re_geo_cell_xy(text)
    if match is None:
        return None, None
    try:
        lat = float(match.group(1).replace("_", "."))
        lon = float(match.group(2).replace("_", "."))
        return lat, lon
    except ValueError:
        return None, None


def re_geo_cell_xy(text: str) -> Any:
    import re

    return re.search(r"LAT([0-9_]+)_LON([0-9_]+)", text.upper())


def geo_distance_km(a: Any, b: Any) -> float | None:
    lat1, lon1 = geo_cell_xy(a)
    lat2, lon2 = geo_cell_xy(b)
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * math.cos(mean_lat)
    return math.sqrt(dlat * dlat + dlon * dlon)


def norm_distance(value: float | None, max_km: float = 20.0) -> float:
    if value is None:
        return 1.0
    return min(math.log1p(max(value, 0.0)) / math.log1p(max_km), 1.0)


def build_relation_context(group: Dict[str, Any], include_similar_profile: bool) -> Dict[str, Any]:
    raw_text = group.get("raw_text") or group.get("pref_text") or ""
    return {
        "trajectory": parse_recent_context(raw_text),
        "user_profile": parse_profile_signal_sets(group.get("user_semantic_profile") or ""),
        "similar_profile": parse_profile_signal_sets(gated_similar_user_profile(group) if include_similar_profile else ""),
    }


def candidate_intrinsic_features(candidate: Dict[str, Any], popularity: Dict[str, int]) -> List[float]:
    rank = max(1.0, float(candidate.get("rank") or 999.0))
    score = max(0.0, float(candidate.get("score") or 0.0))
    sources = [str(x or "") for x in (candidate.get("sources") or [])]
    family_stats = source_family_statistics(sources)
    return [
        1.0 / math.log1p(rank),
        math.log1p(score) / 10.0,
        min(rank, 100.0) / 100.0,
        math.log1p(len(sources)) / math.log1p(16.0),
        *family_stats,
        *popularity_features(str(candidate.get("poi_id") or ""), popularity),
    ]


def recency_weight(length: int, idx: int) -> float:
    if length <= 0:
        return 0.0
    distance_from_end = max(0, length - 1 - idx)
    return 1.0 / float(distance_from_end + 1)


def enhanced_candidate_relation_features(candidate: Dict[str, Any], relation_context: Dict[str, Any]) -> List[float]:
    ctx = relation_context.get("trajectory") or {}
    trajectory = list(ctx.get("trajectory") or [])
    n = max(1, len(trajectory))
    last = ctx.get("last") or {}
    last2 = ctx.get("last2") or {}
    candidate_category = str(candidate.get("category") or "").strip()
    candidate_geo_cell = str(candidate.get("geo_cell") or "").strip()
    candidate_poi_id = str(candidate.get("poi_id") or "").strip()

    poi_hits = [idx for idx, item in enumerate(trajectory) if str(item.get("poi_id") or "").strip() == candidate_poi_id]
    category_hits = [idx for idx, item in enumerate(trajectory) if str(item.get("category") or "").strip() == candidate_category]
    geo_hits = [idx for idx, item in enumerate(trajectory) if str(item.get("geo_cell") or "").strip() == candidate_geo_cell]
    pair_seen = any(
        str(prev.get("category") or "").strip() == str(last.get("category") or "").strip()
        and str(cur.get("category") or "").strip() == candidate_category
        for prev, cur in zip(trajectory, trajectory[1:])
    )

    poi_recency = max((recency_weight(len(trajectory), idx) for idx in poi_hits), default=0.0)
    category_recency = max((recency_weight(len(trajectory), idx) for idx in category_hits), default=0.0)
    geo_recency = max((recency_weight(len(trajectory), idx) for idx in geo_hits), default=0.0)
    category_seen = bool(category_hits)
    geo_seen = bool(geo_hits)
    poi_seen = bool(poi_hits)
    last_category = str(last.get("category") or "").strip()
    last_geo = str(last.get("geo_cell") or "").strip()
    last2_category = str(last2.get("category") or "").strip()
    last2_geo = str(last2.get("geo_cell") or "").strip()
    dist_last = geo_distance_km(candidate_geo_cell, last_geo)
    dist_last2 = geo_distance_km(candidate_geo_cell, last2_geo)
    recent_dists = [geo_distance_km(candidate_geo_cell, item.get("geo_cell")) for item in trajectory[-8:]]
    recent_dists = [x for x in recent_dists if x is not None]
    min_recent_dist = min(recent_dists) if recent_dists else None
    last_pair_match = bool(last_category) and candidate_category == last_category and candidate_geo_cell == last_geo
    recent_count = max(1.0, float(min(len(trajectory), 8)))

    return [
        float(poi_seen),
        float(category_seen),
        float(geo_seen),
        poi_recency,
        category_recency,
        geo_recency,
        min(float(len(poi_hits)), 8.0) / 8.0,
        min(float(len(category_hits)), 8.0) / 8.0,
        min(float(len(geo_hits)), 8.0) / 8.0,
        float(candidate_category == last_category and candidate_geo_cell != last_geo),
        float(candidate_geo_cell == last_geo and candidate_category != last_category),
        float((not poi_seen) and (category_seen or geo_seen)),
        float(pair_seen),
        float(bool(last_category) and candidate_category != last_category),
        float(bool(last_geo) and candidate_geo_cell != last_geo),
        float(bool(last2_category) and candidate_category == last2_category and candidate_category != last_category),
        norm_distance(dist_last),
        float(dist_last is not None and dist_last <= 0.5),
        float(dist_last is not None and dist_last <= 2.0),
        float(dist_last is not None and dist_last <= 10.0),
        norm_distance(dist_last2),
        norm_distance(min_recent_dist),
        float(min_recent_dist is not None and min_recent_dist <= 1.0),
        float(min_recent_dist is not None and min_recent_dist <= 5.0),
        float(len(category_hits)) / recent_count,
        float(len(geo_hits)) / recent_count,
        float(last_pair_match),
        float((not poi_seen) and (dist_last is not None and dist_last <= 2.0) and category_seen),
    ]


def candidate_relation_features(
    candidate: Dict[str, Any],
    relation_context: Dict[str, Any],
    enhanced: bool = False,
) -> List[float]:
    ctx = relation_context.get("trajectory") or {}
    last = ctx.get("last") or {}
    last2 = ctx.get("last2") or {}
    recent_categories = list(ctx.get("recent_categories") or [])
    recent_geo_cells = list(ctx.get("recent_geo_cells") or [])
    current_slot = str(ctx.get("current_slot") or "").strip()
    candidate_category = str(candidate.get("category") or "").strip()
    candidate_geo_cell = str(candidate.get("geo_cell") or "").strip()
    candidate_poi_id = str(candidate.get("poi_id") or "").strip()

    user_profile = relation_context.get("user_profile") or {}
    similar_profile = relation_context.get("similar_profile") or {}

    recent_category_hits = sum(1 for item in recent_categories if item == candidate_category)
    recent_geo_hits = sum(1 for item in recent_geo_cells if item == candidate_geo_cell)
    temporal_match = float(bool(current_slot) and current_slot in user_profile["slots"])
    similar_temporal_match = float(bool(current_slot) and current_slot in similar_profile["slots"])

    base = [
        float(candidate_poi_id == str(last.get("poi_id") or "")),
        float(candidate_category == str(last.get("category") or "")),
        float(candidate_poi_id == str(last2.get("poi_id") or "")),
        float(candidate_category == str(last2.get("category") or "")),
        float(recent_category_hits) / max(1.0, float(len(recent_categories))),
        float(candidate_geo_cell == str(last.get("geo_cell") or "")),
        float(recent_geo_hits) / max(1.0, float(len(recent_geo_cells))),
        float(candidate_category in user_profile["categories"]),
        float(f"{candidate_category}||{candidate_geo_cell}" in user_profile["revisit_pairs"]),
        float(candidate_geo_cell in user_profile["geo_cells"]),
        temporal_match,
        float(candidate_category in similar_profile["categories"]),
        float(f"{candidate_category}||{candidate_geo_cell}" in similar_profile["revisit_pairs"]),
        float(candidate_geo_cell in similar_profile["geo_cells"]),
        similar_temporal_match,
        float(temporal_match or similar_temporal_match),
    ]
    if enhanced:
        base.extend(enhanced_candidate_relation_features(candidate, relation_context))
    return base


def gap_bucket(delta_min: Any) -> str:
    try:
        value = int(delta_min or 0)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        return "gap_zero"
    if value <= 10:
        return "gap_tiny"
    if value <= 60:
        return "gap_short"
    if value <= 240:
        return "gap_mid"
    return "gap_long"


def event_feature_tensors(
    relation_context: Dict[str, Any],
    max_events: int,
    category_bucket_size: int,
    geo_bucket_size: int,
    semantic_bucket_size: int,
) -> Dict[str, torch.Tensor]:
    trajectory = list((relation_context.get("trajectory") or {}).get("trajectory") or [])
    max_events = max(1, int(max_events))
    selected = trajectory[-max_events:]
    pad = max_events - len(selected)
    poi_ids: List[int] = [0] * pad
    category_ids: List[int] = [0] * pad
    geo_ids: List[int] = [0] * pad
    slot_ids: List[int] = [0] * pad
    weekday_ids: List[int] = [0] * pad
    numeric_rows: List[List[float]] = [[0.0] * EVENT_NUMERIC_DIM for _ in range(pad)]
    mask: List[bool] = [False] * pad

    n = max(1, len(selected))
    category_counts: Dict[str, int] = {}
    geo_counts: Dict[str, int] = {}
    poi_counts: Dict[str, int] = {}
    for item in selected:
        poi = str(item.get("poi_id") or "")
        category = str(item.get("category") or "")
        geo = str(item.get("geo_cell") or "")
        poi_counts[poi] = poi_counts.get(poi, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        geo_counts[geo] = geo_counts.get(geo, 0) + 1
    for idx, item in enumerate(selected):
        distance_from_end = len(selected) - 1 - idx
        try:
            delta = int(item.get("delta_min") or 0)
        except (TypeError, ValueError):
            delta = 0
        poi = str(item.get("poi_id") or "")
        category = str(item.get("category") or "")
        geo = str(item.get("geo_cell") or "")
        slot = str(item.get("slot") or "")
        weekday = str(item.get("weekday") or "")
        prev = selected[idx - 1] if idx > 0 else {}
        prev_geo = str(prev.get("geo_cell") or "")
        prev_category = str(prev.get("category") or "")
        step_dist = geo_distance_km(geo, prev_geo) if idx > 0 else None
        slot_sin, slot_cos = cyclic_pair(slot_index(slot), 48)
        weekday_sin, weekday_cos = cyclic_pair(weekday_index(weekday), 7)
        poi_ids.append(stable_bucket(poi, semantic_bucket_size))
        category_ids.append(stable_bucket(category, category_bucket_size))
        geo_ids.append(stable_bucket(geo, geo_bucket_size))
        slot_ids.append(stable_bucket(slot, 128))
        weekday_ids.append(stable_bucket(weekday, 16))
        numeric_rows.append(
            [
                1.0 / float(distance_from_end + 1),
                float(idx + 1) / float(n),
                min(math.log1p(max(delta, 0)) / math.log1p(1440.0), 1.0),
                float(delta <= 0),
                float(0 < delta <= 30),
                float(delta >= 360),
                float(distance_from_end == 0),
                float(distance_from_end == 1),
                slot_sin,
                slot_cos,
                weekday_sin,
                weekday_cos,
                norm_distance(step_dist),
                float(bool(prev_geo) and geo == prev_geo),
                float(bool(prev_category) and category == prev_category),
                min(math.log1p(max(poi_counts.get(poi, 0), category_counts.get(category, 0), geo_counts.get(geo, 0))) / math.log1p(8.0), 1.0),
            ]
        )
        mask.append(True)

    return {
        "event_poi_ids": torch.tensor(poi_ids, dtype=torch.long),
        "event_category_ids": torch.tensor(category_ids, dtype=torch.long),
        "event_geo_ids": torch.tensor(geo_ids, dtype=torch.long),
        "event_slot_ids": torch.tensor(slot_ids, dtype=torch.long),
        "event_weekday_ids": torch.tensor(weekday_ids, dtype=torch.long),
        "event_numeric": torch.tensor(numeric_rows, dtype=torch.float32),
        "event_mask": torch.tensor(mask, dtype=torch.bool),
    }


def format_structured_mobility_text(
    group: Dict[str, Any],
    include_similar_profile: bool,
    max_events: int,
    enhanced: bool = False,
) -> str:
    ctx = parse_recent_context(group.get("raw_text") or group.get("pref_text") or "")
    trajectory = list(ctx.get("trajectory") or [])[-max(1, int(max_events)) :]
    event_lines: List[str] = []
    category_counts: Dict[str, int] = {}
    geo_counts: Dict[str, int] = {}
    poi_counts: Dict[str, int] = {}
    for item in trajectory:
        category = str(item.get("category") or "CAT_UNK")
        geo = str(item.get("geo_cell") or "GEO_UNK")
        poi = str(item.get("poi_id") or "POI_UNK")
        category_counts[category] = category_counts.get(category, 0) + 1
        geo_counts[geo] = geo_counts.get(geo, 0) + 1
        poi_counts[poi] = poi_counts.get(poi, 0) + 1
    for idx, item in enumerate(trajectory):
        category = str(item.get("category") or "CAT_UNK")
        geo = str(item.get("geo_cell") or "GEO_UNK")
        poi = str(item.get("poi_id") or "POI_UNK")
        delta = item.get("delta_min") or 0
        reverse_pos = len(trajectory) - idx
        tokens = [
            f"EVT[-{reverse_pos}]",
            f"POI={poi}",
            f"CAT={category}",
            f"GEO={geo}",
            f"WEEKDAY={item.get('weekday') or 'UNK'}",
            f"SLOT={item.get('slot') or 'UNK'}",
            f"GAP={gap_bucket(delta)}",
        ]
        if enhanced:
            prev_category = str(trajectory[idx - 1].get("category") or "CAT_START") if idx > 0 else "CAT_START"
            prev_geo = str(trajectory[idx - 1].get("geo_cell") or "GEO_START") if idx > 0 else "GEO_START"
            tokens.extend(
                [
                    f"PREV_CAT={prev_category}",
                    f"CAT_TRANS={prev_category}->{category}",
                    f"GEO_MOVE={'stay' if prev_geo == geo else 'move'}",
                    f"POI_FREQ={min(poi_counts.get(poi, 0), 4)}",
                    f"CAT_FREQ={min(category_counts.get(category, 0), 4)}",
                    f"GEO_FREQ={min(geo_counts.get(geo, 0), 4)}",
                ]
            )
        event_lines.append(" ".join(tokens))
    if not event_lines:
        event_lines = ["none"]
    parts = [
        "[TASK=MOBILITY_ENCODING]",
        "Encode structured trajectory events for candidate-conditioned POI alignment.",
        "[TRAJECTORY_EVENTS]",
        "\n".join(event_lines),
        "[USER_SEMANTIC_PROFILE]",
        user_semantic_profile_text(group),
    ]
    if include_similar_profile:
        similar_profile = gated_similar_user_profile(group)
        if similar_profile:
            parts.extend(["[SIMILAR_USER_SEMANTIC_PROFILE]", similar_profile])
    parts.extend(["[MOBILITY_EMBED]"])
    return "\n\n".join(parts)


def format_mobility_text(
    group: Dict[str, Any],
    include_similar_profile: bool,
    structured: bool = False,
    max_events: int = 16,
    enhanced: bool = False,
) -> str:
    if structured:
        return format_structured_mobility_text(group, include_similar_profile, max_events, enhanced=enhanced)
    parts = [
        "[TASK=MOBILITY_ENCODING]",
        "Encode the user's mobility trajectory for fixed-pool candidate alignment.",
        "[RECENT_CONTEXT]",
        strip_preference_section(group.get("pref_text")),
        "[USER_SEMANTIC_PROFILE]",
        user_semantic_profile_text(group),
    ]
    if include_similar_profile:
        similar_profile = gated_similar_user_profile(group)
        if similar_profile:
            parts.extend(["[SIMILAR_USER_SEMANTIC_PROFILE]", similar_profile])
    parts.extend(["[MOBILITY_EMBED]"])
    return "\n\n".join(parts)


def build_online_group(
    joined_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    semantic_map: Dict[str, Dict[str, Any]],
    pool_k: int,
    teacher_row: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    target = target_from_candidate_row(candidate_row)
    candidates = [str(x) for x in (candidate_row.get("candidate_poi_ids") or [])][:pool_k]
    if not candidates:
        return None
    details = candidate_row.get("candidate_details") or {}
    candidate_rows: List[Dict[str, Any]] = []
    for idx, poi_id in enumerate(candidates):
        detail = details.get(poi_id) or {}
        sem = semantic_map.get(poi_id) or {}
        rank = int(detail.get("rank") or idx + 1)
        try:
            score = float(detail.get("score") if detail.get("score") is not None else 0.0)
        except (TypeError, ValueError):
            score = 0.0
        sources = detail.get("sources") or []
        if not isinstance(sources, list):
            sources = [sources]
        semantic_id = str(detail.get("semantic_id") or sem.get("semantic_id") or "SEM_UNK")
        category = str(detail.get("category") or sem.get("category") or sem.get("category_token") or "CAT_UNK")
        geo_cell = str(detail.get("geo_cell") or sem.get("geo_cell") or "GEO_UNK")
        candidate_rows.append(
            {
                "poi_id": poi_id,
                "rank": rank,
                "score": score,
                "sources": sources,
                "semantic_id": semantic_id,
                "category": category,
                "geo_cell": geo_cell,
                "label": 1 if poi_id == target["poi_id"] else 0,
            }
        )

    group = {
        "sample_id": str(joined_row.get("sample_id") or candidate_row.get("sample_id") or ""),
        "split": str(joined_row.get("split") or candidate_row.get("split") or ""),
        "city": str(joined_row.get("city") or candidate_row.get("city") or "NYC"),
        "target_poi_id": target["poi_id"],
        "target_category": target["category"] or str(joined_row.get("target_category") or ""),
        "target_in_candidates": any(x["label"] == 1 for x in candidate_rows),
        "target_rank": candidate_row.get("target_rank"),
        "pref_text": clean_text(joined_row.get("raw_text"), max_chars=6000),
        "refine_text": clean_text(joined_row.get("refined_text"), max_chars=2000),
        "user_semantic_profile": clean_text(joined_row.get("user_semantic_profile"), max_chars=3000),
        "similar_user_semantic_profile": clean_text(joined_row.get("similar_user_semantic_profile"), max_chars=1600),
        "similar_user_count": int(joined_row.get("similar_user_count") or 0),
        "similar_user_profile_has_signal": bool(joined_row.get("similar_user_profile_has_signal")),
        "user_profile_insufficient": bool(joined_row.get("user_profile_insufficient")),
        "raw_text": clean_text(joined_row.get("raw_text"), max_chars=6000),
        "candidates": candidate_rows,
        "candidate_poi_ids": candidates,
        "candidate_semantic_ids": [x["semantic_id"] for x in candidate_rows],
        "candidate_details": {
            x["poi_id"]: {k: x[k] for k in ("rank", "score", "sources", "semantic_id", "category", "geo_cell")}
            for x in candidate_rows
        },
    }
    if teacher_row is not None:
        teacher_ids = [str(x) for x in (teacher_row.get("candidate_poi_ids") or []) if str(x)]
        teacher_details = teacher_row.get("candidate_details") or {}
        teacher_scores: Dict[str, float] = {}
        for idx, poi_id in enumerate(teacher_ids):
            detail = teacher_details.get(poi_id) or {}
            try:
                score = float(detail.get("score") if detail.get("score") is not None else 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score == 0.0:
                score = 1.0 / math.log2(float(idx + 2))
            teacher_scores[poi_id] = score
        group["teacher_candidate_poi_ids"] = teacher_ids
        group["teacher_scores"] = teacher_scores
    return group


@dataclass
class OnlineMobilityDataset(Dataset):
    groups: List[Dict[str, Any]]
    include_similar_profile: bool
    eval_candidate_limit: int | None
    category_bucket_size: int
    geo_bucket_size: int
    semantic_bucket_size: int
    popularity: Dict[str, int]
    structured_mobility_input: bool
    enhanced_behavior_features: bool
    max_trajectory_events: int
    fixed_pool_prior: str
    teacher_top_k: int

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self.groups[idx]
        candidates = list(group.get("candidates") or [])
        if self.eval_candidate_limit is not None:
            candidates = candidates[: max(1, int(self.eval_candidate_limit))]
        target_pos = next((i for i, c in enumerate(candidates) if int(c.get("label") or 0) == 1), -1)
        relation_context = build_relation_context(group, include_similar_profile=self.include_similar_profile)
        pool_arr, _, _ = candidate_pool_features(group, self.popularity)
        pool_arr = pool_arr[: len(candidates)]
        return {
            "sample_id": group.get("sample_id"),
            "target_pos": target_pos,
            "target_in_candidates": bool(group.get("target_in_candidates")),
            "input_text": format_mobility_text(
                group,
                include_similar_profile=self.include_similar_profile,
                structured=self.structured_mobility_input,
                max_events=self.max_trajectory_events,
                enhanced=self.enhanced_behavior_features,
            ),
            "pool_features": torch.from_numpy(np.asarray(pool_arr, dtype=np.float32)),
            "pool_prior_scores": fixed_pool_prior_scores(candidates, self.fixed_pool_prior),
            "teacher_scores": torch.tensor(
                [float((group.get("teacher_scores") or {}).get(str(c.get("poi_id") or ""), 0.0)) for c in candidates],
                dtype=torch.float32,
            ),
            "teacher_top_indices": torch.tensor(
                [
                    idx
                    for idx, c in enumerate(candidates)
                    if str(c.get("poi_id") or "") in set((group.get("teacher_candidate_poi_ids") or [])[: max(0, int(self.teacher_top_k))])
                ],
                dtype=torch.long,
            ),
            "candidate_numeric": torch.tensor([candidate_intrinsic_features(c, self.popularity) for c in candidates], dtype=torch.float32),
            "candidate_relation": torch.tensor(
                [candidate_relation_features(c, relation_context, enhanced=self.enhanced_behavior_features) for c in candidates],
                dtype=torch.float32,
            ),
            "candidate_category_ids": torch.tensor([stable_bucket(c.get("category"), self.category_bucket_size) for c in candidates], dtype=torch.long),
            "candidate_geo_ids": torch.tensor([stable_bucket(c.get("geo_cell"), self.geo_bucket_size) for c in candidates], dtype=torch.long),
            "candidate_semantic_ids": torch.tensor([stable_bucket(c.get("semantic_id"), self.semantic_bucket_size) for c in candidates], dtype=torch.long),
            "candidate_poi_ids": [c.get("poi_id") for c in candidates],
            **event_feature_tensors(
                relation_context,
                max_events=self.max_trajectory_events,
                category_bucket_size=self.category_bucket_size,
                geo_bucket_size=self.geo_bucket_size,
                semantic_bucket_size=self.semantic_bucket_size,
            ),
        }


def collate_groups(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"groups": list(features)}


def load_groups_from_files(
    joined_path: Path,
    candidate_path: Path,
    semantic_map_path: Path,
    pool_k: int,
    limit: int | None = None,
    teacher_path: Path | None = None,
) -> List[Dict[str, Any]]:
    semantic_map = load_semantic_map(semantic_map_path)
    joined_rows = read_joined_parquet(joined_path, limit=limit)
    candidate_rows = read_candidate_jsonl_map(candidate_path, limit=limit)
    teacher_rows = read_candidate_jsonl_map(teacher_path, limit=limit) if teacher_path is not None else {}
    groups: List[Dict[str, Any]] = []
    for joined_row in joined_rows:
        sample_id = str(joined_row.get("sample_id") or "")
        candidate_row = candidate_rows.get(sample_id)
        if candidate_row is None:
            continue
        group = build_online_group(joined_row, candidate_row, semantic_map, pool_k=pool_k, teacher_row=teacher_rows.get(sample_id))
        if group is not None:
            groups.append(group)
    return groups


def select_topk_indices(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, scores.size(-1))
    return torch.topk(scores, k=top_k, dim=-1).indices


def select_candidate_indices(scores: torch.Tensor, top_k: int, mode: str) -> torch.Tensor:
    top_k = min(top_k, scores.size(-1))
    if mode == "original":
        return torch.arange(top_k, device=scores.device, dtype=torch.long).unsqueeze(0).expand(scores.size(0), -1)
    return select_topk_indices(scores, top_k)


def sampled_train_indices(
    target_pos: int,
    num_candidates: int,
    hard_negatives: int,
    train_negatives: int,
    teacher_indices: Sequence[int] | None = None,
) -> List[int] | None:
    if target_pos < 0 or target_pos >= num_candidates:
        return None
    if train_negatives <= 0 or train_negatives >= num_candidates - 1:
        return list(range(num_candidates))
    selected_set = {target_pos}
    hard: List[int] = []
    for raw_idx in teacher_indices or []:
        idx = int(raw_idx)
        if 0 <= idx < num_candidates and idx != target_pos and idx not in selected_set:
            hard.append(idx)
            selected_set.add(idx)
        if len(hard) >= max(0, hard_negatives):
            break
    for idx in range(num_candidates):
        if idx == target_pos or idx in selected_set:
            continue
        hard.append(idx)
        selected_set.add(idx)
        if len(hard) >= max(0, hard_negatives):
            break
    remaining = [idx for idx in range(num_candidates) if idx not in selected_set]
    random_need = max(0, train_negatives - len(hard))
    if random_need > 0 and remaining:
        hard.extend(random.sample(remaining, k=min(random_need, len(remaining))))
    selected = [target_pos] + hard[:train_negatives]
    return selected


def remap_target_position(indices: torch.Tensor, target_pos: int) -> int:
    match = (indices == int(target_pos)).nonzero(as_tuple=False)
    return int(match[0].item()) if match.numel() else -1


def target_mask_candidate_view(
    numeric: torch.Tensor,
    relation: torch.Tensor,
    target_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masked_numeric = numeric.clone()
    masked_relation = relation.clone()
    if 0 <= target_pos < masked_numeric.size(1):
        masked_numeric[:, target_pos, :] = 0.0
        masked_relation[:, target_pos, :] = 0.0
    return masked_numeric, masked_relation


def teacher_kl_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, temperature: float) -> torch.Tensor | None:
    positive = teacher_scores > 0
    if int(positive.sum().item()) < 2:
        return None
    temp = max(float(temperature), 1.0e-6)
    masked_teacher = teacher_scores.float().masked_fill(~positive, torch.finfo(torch.float32).min)
    teacher_prob = F.softmax(masked_teacher / temp, dim=-1)
    student_log_prob = F.log_softmax(student_scores.float() / temp, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temp * temp)


def gather_along_candidates(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view = indices
    while view.ndim < tensor.ndim:
        view = view.unsqueeze(-1)
    expand_shape = list(view.shape)
    expand_shape[-1] = tensor.shape[-1] if tensor.ndim > 2 else 1
    view = view.expand(*view.shape[:-1], *tensor.shape[2:])
    return torch.gather(tensor, 1, view)


class PoolCompressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
        else:
            self.net = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def resolve_lora_targets(model: nn.Module, preset: str) -> List[str]:
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    if preset == "llama" or (preset == "auto" and "llama" in model_type):
        return list(LLAMA_TARGET_MODULES)
    if preset == "bert" or (preset == "auto" and "bert" in model_type):
        return list(BERT_TARGET_MODULES)
    if preset == "auto":
        raise ValueError(f"Cannot infer LoRA target preset from model_type={model_type!r}; pass --lora-target-preset explicitly.")
    raise ValueError(f"Unsupported LoRA target preset: {preset}")


def inject_routed_lora(model: nn.Module, target_modules: Sequence[str], r: int, alpha: int, dropout: float, lora_num: int) -> int:
    replaced = 0
    for key in list(dict(model.named_modules()).keys()):
        if not any(key.endswith(target) for target in target_modules):
            continue
        target = model.get_submodule(key)
        if not isinstance(target, nn.Linear):
            continue
        parent_name = ".".join(key.split(".")[:-1])
        child_name = key.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        module = RoutedTeamLoRALinear(target, r=r, alpha=alpha, dropout=dropout, lora_num=lora_num)
        module.to(target.weight.device, dtype=target.weight.dtype)
        setattr(parent, child_name, module)
        replaced += 1
    if replaced == 0:
        raise ValueError(f"No LoRA target modules replaced: {list(target_modules)}")
    return replaced


def unfreeze_last_encoder_layers(model: nn.Module, n_layers: int) -> List[str]:
    if n_layers <= 0:
        return []
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = list(model.encoder.layer)
        prefix = "encoder.layer"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
        prefix = "model.layers"
    else:
        raise ValueError("Cannot locate transformer encoder layers for unfreezing.")
    n_layers = min(int(n_layers), len(layers))
    unfrozen: List[str] = []
    start = len(layers) - n_layers
    for idx in range(start, len(layers)):
        layer = layers[idx]
        for param in layer.parameters():
            param.requires_grad = True
        unfrozen.append(f"{prefix}.{idx}")
    return unfrozen


def make_optimizer(model: nn.Module, lr: float, encoder_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    encoder_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("aligner.base_model.") and "lora_" not in name:
            encoder_params.append(param)
        else:
            other_params.append(param)
    groups: List[Dict[str, Any]] = []
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


class OnlineMobilityCompressor(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        pool_dim: int,
        numeric_dim: int,
        relation_dim: int,
        text_proj_dim: int,
        anchor_proj_dim: int,
        anchor_hidden_dim: int,
        relation_hidden_dim: int,
        relation_proj_dim: int,
        category_bucket_size: int,
        geo_bucket_size: int,
        semantic_bucket_size: int,
        alignment_mode: str,
        dropout: float,
        pool_hidden_dim: int,
        pool_dropout: float,
        candidate_token_attention: bool,
        din_history_attention: bool,
        pool_alpha_init: float,
        fusion_alpha_init: float,
    ):
        super().__init__()
        self.pool_scorer = PoolCompressor(pool_dim, hidden_dim=pool_hidden_dim, dropout=pool_dropout)
        self.aligner = MobilityAligner(
            base_model=base_model,
            hidden_size=hidden_size,
            numeric_dim=numeric_dim,
            relation_dim=relation_dim,
            text_proj_dim=text_proj_dim,
            anchor_proj_dim=anchor_proj_dim,
            anchor_hidden_dim=anchor_hidden_dim,
            relation_hidden_dim=relation_hidden_dim,
            relation_proj_dim=relation_proj_dim,
            category_bucket_size=category_bucket_size,
            geo_bucket_size=geo_bucket_size,
            semantic_bucket_size=semantic_bucket_size,
            alignment_mode=alignment_mode,
            dropout=dropout,
        )
        self.candidate_token_attention = bool(candidate_token_attention)
        self.din_history_attention = bool(din_history_attention)
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_size, text_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_proj_dim, anchor_proj_dim),
        )
        event_emb_dim = max(8, anchor_proj_dim // 8)
        self.event_poi_emb = nn.Embedding(semantic_bucket_size, event_emb_dim)
        self.event_category_emb = nn.Embedding(category_bucket_size, event_emb_dim)
        self.event_geo_emb = nn.Embedding(geo_bucket_size, event_emb_dim)
        self.event_slot_emb = nn.Embedding(128, event_emb_dim)
        self.event_weekday_emb = nn.Embedding(16, event_emb_dim)
        self.event_numeric_proj = nn.Sequential(
            nn.Linear(EVENT_NUMERIC_DIM, event_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_proj = nn.Sequential(
            nn.Linear(event_emb_dim * 6, anchor_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(anchor_proj_dim, anchor_proj_dim),
        )
        self.din_attention = nn.Sequential(
            nn.Linear(anchor_proj_dim * 4, anchor_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(anchor_proj_dim, 1),
        )
        self.history_gate = nn.Sequential(
            nn.Linear(anchor_proj_dim * 3, anchor_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(anchor_proj_dim, 1),
            nn.Sigmoid(),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(anchor_proj_dim * 3, anchor_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(anchor_proj_dim, 1),
            nn.Sigmoid(),
        )
        self.pool_alpha = nn.Parameter(torch.tensor(float(pool_alpha_init), dtype=torch.float32))
        self.fusion_alpha = nn.Parameter(torch.tensor(float(fusion_alpha_init), dtype=torch.float32))

    def gradient_checkpointing_enable(self) -> None:
        self.aligner.gradient_checkpointing_enable()

    def encode_selected(
        self,
        text_batch: Dict[str, torch.Tensor],
        candidate_numeric: torch.Tensor,
        candidate_relation: torch.Tensor,
        candidate_category_ids: torch.Tensor,
        candidate_geo_ids: torch.Tensor,
        candidate_semantic_ids: torch.Tensor,
        event_numeric: torch.Tensor | None = None,
        event_poi_ids: torch.Tensor | None = None,
        event_category_ids: torch.Tensor | None = None,
        event_geo_ids: torch.Tensor | None = None,
        event_slot_ids: torch.Tensor | None = None,
        event_weekday_ids: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user, token_states, token_mask = self.encode_user_context(text_batch)
        anchors = self.aligner.encode_anchors(
            candidate_numeric,
            candidate_relation,
            candidate_category_ids,
            candidate_geo_ids,
            candidate_semantic_ids,
        )
        anchors = F.normalize(anchors, dim=-1)
        if self.candidate_token_attention:
            pair_user = self.candidate_conditioned_user(
                user,
                token_states,
                token_mask,
                anchors,
                event_numeric=event_numeric,
                event_poi_ids=event_poi_ids,
                event_category_ids=event_category_ids,
                event_geo_ids=event_geo_ids,
                event_slot_ids=event_slot_ids,
                event_weekday_ids=event_weekday_ids,
                event_mask=event_mask,
            )
        else:
            pair_user = user.unsqueeze(1).expand_as(anchors)
            if self.din_history_attention and event_numeric is not None:
                event_states = self.encode_events(
                    event_numeric,
                    event_poi_ids,
                    event_category_ids,
                    event_geo_ids,
                    event_slot_ids,
                    event_weekday_ids,
                )
                event_context = self.candidate_history_attention(anchors, event_states, event_mask)
                gate = self.history_gate(torch.cat([pair_user, event_context, anchors], dim=-1))
                pair_user = F.normalize(gate * event_context + (1.0 - gate) * pair_user, dim=-1)
        return self.score_user_anchor(pair_user, anchors)

    def encode_user_context(self, text_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.aligner.base_model(
            input_ids=text_batch["input_ids"],
            attention_mask=text_batch["attention_mask"],
            return_dict=True,
        )
        hidden = outputs.last_hidden_state.float()
        mask = text_batch["attention_mask"].bool()
        last_idx = text_batch["attention_mask"].long().sum(dim=1).clamp(min=1) - 1
        last_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        user = F.normalize(self.aligner.text_proj(last_hidden), dim=-1)
        tokens = F.normalize(self.token_proj(hidden), dim=-1)
        return user, tokens, mask

    def candidate_conditioned_user(
        self,
        user: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        anchors: torch.Tensor,
        event_numeric: torch.Tensor | None = None,
        event_poi_ids: torch.Tensor | None = None,
        event_category_ids: torch.Tensor | None = None,
        event_geo_ids: torch.Tensor | None = None,
        event_slot_ids: torch.Tensor | None = None,
        event_weekday_ids: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = torch.einsum("bnd,bld->bnl", anchors, token_states) / math.sqrt(float(anchors.size(-1)))
        logits = logits.masked_fill(~token_mask.unsqueeze(1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bnl,bld->bnd", weights, token_states)
        user_expand = user.unsqueeze(1).expand_as(anchors)
        gate = self.context_gate(torch.cat([user_expand, context, anchors], dim=-1))
        pair_user = F.normalize(gate * context + (1.0 - gate) * user_expand, dim=-1)
        if not self.din_history_attention or event_numeric is None:
            return pair_user
        event_states = self.encode_events(
            event_numeric,
            event_poi_ids,
            event_category_ids,
            event_geo_ids,
            event_slot_ids,
            event_weekday_ids,
        )
        event_context = self.candidate_history_attention(anchors, event_states, event_mask)
        history_gate = self.history_gate(torch.cat([pair_user, event_context, anchors], dim=-1))
        return F.normalize(history_gate * event_context + (1.0 - history_gate) * pair_user, dim=-1)

    def encode_events(
        self,
        event_numeric: torch.Tensor,
        event_poi_ids: torch.Tensor | None,
        event_category_ids: torch.Tensor | None,
        event_geo_ids: torch.Tensor | None,
        event_slot_ids: torch.Tensor | None,
        event_weekday_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if any(x is None for x in (event_poi_ids, event_category_ids, event_geo_ids, event_slot_ids, event_weekday_ids)):
            raise ValueError("DIN history attention requires all event id tensors.")
        parts = [
            self.event_numeric_proj(event_numeric.float()),
            self.event_poi_emb(event_poi_ids),
            self.event_category_emb(event_category_ids),
            self.event_geo_emb(event_geo_ids),
            self.event_slot_emb(event_slot_ids),
            self.event_weekday_emb(event_weekday_ids),
        ]
        return F.normalize(self.event_proj(torch.cat(parts, dim=-1).float()), dim=-1)

    def candidate_history_attention(
        self,
        anchors: torch.Tensor,
        event_states: torch.Tensor,
        event_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, num_candidates, dim = anchors.shape
        num_events = event_states.size(1)
        anchor_expand = anchors.unsqueeze(2).expand(bsz, num_candidates, num_events, dim)
        event_expand = event_states.unsqueeze(1).expand(bsz, num_candidates, num_events, dim)
        pair = torch.cat(
            [anchor_expand, event_expand, anchor_expand - event_expand, anchor_expand * event_expand],
            dim=-1,
        )
        logits = self.din_attention(pair).squeeze(-1)
        if event_mask is not None:
            mask = event_mask.bool().unsqueeze(1)
            logits = logits.masked_fill(~mask, -1.0e4)
            weights = torch.softmax(logits, dim=-1) * mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        else:
            weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bnl,bld->bnd", weights, event_states)
        return F.normalize(context, dim=-1)

    def score_user_anchor(self, pair_user: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        if self.aligner.alignment_mode == "linear":
            return torch.sum(pair_user * anchors, dim=-1)
        if self.aligner.alignment_mode == "low_rank":
            pair = torch.cat([pair_user, anchors, pair_user * anchors], dim=-1)
            return self.aligner.pair_scorer(pair).squeeze(-1)
        pair = torch.cat([pair_user, anchors], dim=-1)
        cross_out = self.aligner.cross_network(pair)
        deep_out = self.aligner.deep_tower(pair)
        score_input = torch.cat([pair, cross_out, deep_out], dim=-1)
        return self.aligner.final_scorer(score_input).squeeze(-1)

    def fuse_scores(self, pool_scores: torch.Tensor, dcn_scores: torch.Tensor) -> torch.Tensor:
        return self.pool_alpha * pool_scores + self.fusion_alpha * dcn_scores


@torch.no_grad()
def evaluate(model: OnlineMobilityCompressor, tokenizer: Any, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    ranks: List[int] = []
    missing = 0
    candidate_batch_size = max(1, int(args.eval_candidate_batch_size))
    for batch in tqdm(loader, desc="eval", leave=False):
        groups = batch["groups"]
        for group in groups:
            target_pos = int(group["target_pos"])
            if target_pos < 0:
                missing += 1
                continue
            pool_features = group["pool_features"].to(device)
            pool_scores = model.pool_scorer(pool_features.float())
            base_scores = group["pool_prior_scores"].to(device) if args.fixed_pool_prior != "none" else pool_scores
            selected_idx = select_candidate_indices(base_scores.unsqueeze(0), args.top_k, args.selection_mode)[0]
            selected_idx_cpu = selected_idx.detach().cpu()
            selected_target_pos = next((i for i, idx in enumerate(selected_idx_cpu.tolist()) if int(idx) == target_pos), -1)
            if selected_target_pos < 0:
                missing += 1
                continue

            numeric = group["candidate_numeric"][selected_idx_cpu].unsqueeze(0).to(device)
            relation = group["candidate_relation"][selected_idx_cpu].unsqueeze(0).to(device)
            cat = group["candidate_category_ids"][selected_idx_cpu].unsqueeze(0).to(device)
            geo = group["candidate_geo_ids"][selected_idx_cpu].unsqueeze(0).to(device)
            sem = group["candidate_semantic_ids"][selected_idx_cpu].unsqueeze(0).to(device)
            event_numeric = group["event_numeric"].unsqueeze(0).to(device)
            event_poi_ids = group["event_poi_ids"].unsqueeze(0).to(device)
            event_category_ids = group["event_category_ids"].unsqueeze(0).to(device)
            event_geo_ids = group["event_geo_ids"].unsqueeze(0).to(device)
            event_slot_ids = group["event_slot_ids"].unsqueeze(0).to(device)
            event_weekday_ids = group["event_weekday_ids"].unsqueeze(0).to(device)
            event_mask = group["event_mask"].unsqueeze(0).to(device)
            text = encode_texts(tokenizer, [group["input_text"]], args.max_length, device)
            dcn_scores = model.encode_selected(
                text,
                numeric,
                relation,
                cat,
                geo,
                sem,
                event_numeric=event_numeric,
                event_poi_ids=event_poi_ids,
                event_category_ids=event_category_ids,
                event_geo_ids=event_geo_ids,
                event_slot_ids=event_slot_ids,
                event_weekday_ids=event_weekday_ids,
                event_mask=event_mask,
            )[0]

            selected_base_scores = base_scores[selected_idx]
            final_scores = dcn_scores if args.final_score_mode == "dcn_only" else model.fuse_scores(selected_base_scores, dcn_scores)
            target_score = final_scores[selected_target_pos]
            rank = int(1 + (final_scores > target_score).sum().item())
            ranks.append(rank)

    total = len(ranks) + missing
    if not ranks:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "top20": 0.0,
            "ndcg1": 0.0,
            "ndcg5": 0.0,
            "ndcg10": 0.0,
            "mrr": 0.0,
            "conditional_top1": 0.0,
            "conditional_top5": 0.0,
            "conditional_top10": 0.0,
            "conditional_top20": 0.0,
            "conditional_ndcg1": 0.0,
            "conditional_ndcg5": 0.0,
            "conditional_ndcg10": 0.0,
            "conditional_mrr": 0.0,
            "candidate_hit": 0.0,
            "evaluated": float(total),
            "missed": float(missing),
        }
    arr = np.asarray(ranks, dtype=np.int64)
    candidate_hit = float(len(ranks) / total) if total else 0.0
    ndcg1 = np.where(arr <= 1, 1.0 / np.log2(arr + 1.0), 0.0)
    ndcg5 = np.where(arr <= 5, 1.0 / np.log2(arr + 1.0), 0.0)
    ndcg10 = np.where(arr <= 10, 1.0 / np.log2(arr + 1.0), 0.0)
    conditional = {
        "conditional_top1": float(np.mean(arr <= 1)),
        "conditional_top5": float(np.mean(arr <= 5)),
        "conditional_top10": float(np.mean(arr <= 10)),
        "conditional_top20": float(np.mean(arr <= 20)),
        "conditional_ndcg1": float(np.mean(ndcg1)),
        "conditional_ndcg5": float(np.mean(ndcg5)),
        "conditional_ndcg10": float(np.mean(ndcg10)),
        "conditional_mrr": float(np.mean(1.0 / arr)),
    }
    return {
        "top1": conditional["conditional_top1"] * candidate_hit,
        "top5": conditional["conditional_top5"] * candidate_hit,
        "top10": conditional["conditional_top10"] * candidate_hit,
        "top20": conditional["conditional_top20"] * candidate_hit,
        "ndcg1": conditional["conditional_ndcg1"] * candidate_hit,
        "ndcg5": conditional["conditional_ndcg5"] * candidate_hit,
        "ndcg10": conditional["conditional_ndcg10"] * candidate_hit,
        "mrr": conditional["conditional_mrr"] * candidate_hit,
        **conditional,
        "candidate_hit": candidate_hit,
        "evaluated": float(total),
        "missed": float(missing),
    }


def train_step(
    model: OnlineMobilityCompressor,
    tokenizer: Any,
    groups: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    scaler_enabled: bool,
) -> Tuple[torch.Tensor | None, Dict[str, float]]:
    texts = [g["input_text"] for g in groups]
    text_batch = encode_texts(tokenizer, texts, args.max_length, device)
    pool_features = torch.stack([g["pool_features"] for g in groups], dim=0).to(device)
    prior_scores = torch.stack([g["pool_prior_scores"] for g in groups], dim=0).to(device)
    numeric = torch.stack([g["candidate_numeric"] for g in groups], dim=0).to(device)
    relation = torch.stack([g["candidate_relation"] for g in groups], dim=0).to(device)
    category_ids = torch.stack([g["candidate_category_ids"] for g in groups], dim=0).to(device)
    geo_ids = torch.stack([g["candidate_geo_ids"] for g in groups], dim=0).to(device)
    semantic_ids = torch.stack([g["candidate_semantic_ids"] for g in groups], dim=0).to(device)
    event_numeric = torch.stack([g["event_numeric"] for g in groups], dim=0).to(device)
    event_poi_ids = torch.stack([g["event_poi_ids"] for g in groups], dim=0).to(device)
    event_category_ids = torch.stack([g["event_category_ids"] for g in groups], dim=0).to(device)
    event_geo_ids = torch.stack([g["event_geo_ids"] for g in groups], dim=0).to(device)
    event_slot_ids = torch.stack([g["event_slot_ids"] for g in groups], dim=0).to(device)
    event_weekday_ids = torch.stack([g["event_weekday_ids"] for g in groups], dim=0).to(device)
    event_mask = torch.stack([g["event_mask"] for g in groups], dim=0).to(device)
    target_positions = [int(g["target_pos"]) for g in groups]

    pool_scores = model.pool_scorer(pool_features.float())
    if args.pool_loss_weight > 0:
        pool_loss = score_group_loss(pool_scores, target_positions)
    else:
        pool_loss = torch.tensor(0.0, device=device)
    if pool_loss is None:
        return None, {}
    base_scores = prior_scores if args.fixed_pool_prior != "none" else pool_scores

    valid_rows = [idx for idx, pos in enumerate(target_positions) if pos >= 0]
    keep_loss = torch.tensor(0.0, device=device)
    final_loss = torch.tensor(0.0, device=device)
    total_keep = 0
    total_final = 0
    selected_idx = select_topk_indices(base_scores, args.top_k)
    selected_base_scores = torch.gather(base_scores, 1, selected_idx)
    selected_numeric = gather_candidate_tensor(numeric, selected_idx)
    selected_relation = gather_candidate_tensor(relation, selected_idx)
    selected_category_ids = gather_candidate_tensor(category_ids, selected_idx)
    selected_geo_ids = gather_candidate_tensor(geo_ids, selected_idx)
    selected_semantic_ids = gather_candidate_tensor(semantic_ids, selected_idx)
    dcn_scores = model.encode_selected(
        text_batch,
        selected_numeric,
        selected_relation,
        selected_category_ids,
        selected_geo_ids,
        selected_semantic_ids,
        event_numeric=event_numeric,
        event_poi_ids=event_poi_ids,
        event_category_ids=event_category_ids,
        event_geo_ids=event_geo_ids,
        event_slot_ids=event_slot_ids,
        event_weekday_ids=event_weekday_ids,
        event_mask=event_mask,
    )
    final_scores = model.fuse_scores(selected_base_scores, dcn_scores)
    final_target_positions: List[int] = []

    for row_idx in valid_rows:
        target_pos = target_positions[row_idx]
        selected_row = selected_idx[row_idx]
        selected_pos = (selected_row == target_pos).nonzero(as_tuple=False)
        if selected_pos.numel() > 0:
            final_target_positions.append(int(selected_pos[0].item()))
            total_final += 1
        cutoff_pos = min(args.top_k - 1, pool_scores.size(-1) - 1)
        if args.keep_loss_weight > 0:
            target_score = base_scores[row_idx, target_pos]
            cutoff_score = torch.sort(base_scores[row_idx], descending=True).values[cutoff_pos]
            keep_loss = keep_loss + F.relu(torch.tensor(args.keep_margin, device=device) - (target_score - cutoff_score))
            total_keep += 1

    if final_target_positions:
        final_rows = [idx for idx, pos in enumerate(target_positions) if pos >= 0 and pos in selected_idx[idx].tolist()]
        if final_rows:
            row_scores = final_scores[final_rows]
            row_targets = torch.tensor(final_target_positions, device=device, dtype=torch.long)
            final_loss = F.cross_entropy(row_scores, row_targets)
    elif args.pool_loss_weight <= 0 and args.keep_loss_weight <= 0:
        return None, {}

    keep_loss = keep_loss / max(total_keep, 1)
    total_loss = args.pool_loss_weight * pool_loss + args.keep_loss_weight * keep_loss + args.final_loss_weight * final_loss
    stats = {
        "pool_loss": float(pool_loss.detach().cpu()),
        "keep_loss": float(keep_loss.detach().cpu()),
        "final_loss": float(final_loss.detach().cpu()),
        "selected_rows": float(total_final),
    }
    return total_loss, stats


def gather_candidate_tensor(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return torch.gather(tensor, 1, indices)
    idx = indices.unsqueeze(-1)
    while idx.ndim < tensor.ndim:
        idx = idx.unsqueeze(-1)
    idx = idx.expand(*indices.shape, *tensor.shape[2:])
    return torch.gather(tensor, 1, idx)


def main() -> None:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_groups = load_groups_from_files(
        args.train_joined,
        args.train_candidates,
        args.semantic_map,
        args.pool_k,
        args.max_train_groups,
        teacher_path=args.train_teacher_candidates,
    )
    val_groups = load_groups_from_files(
        args.val_joined,
        args.val_candidates,
        args.semantic_map,
        args.pool_k,
        args.max_val_groups,
        teacher_path=args.val_teacher_candidates,
    )
    test_groups = load_groups_from_files(
        args.test_joined,
        args.test_candidates,
        args.semantic_map,
        args.pool_k,
        args.max_test_groups,
        teacher_path=args.test_teacher_candidates,
    )
    raw_train_groups = len(train_groups)
    raw_val_groups = len(val_groups)
    raw_test_groups = len(test_groups)
    if args.train_hit_only:
        train_groups = [group for group in train_groups if group.get("target_in_candidates")]
    if args.val_hit_only:
        val_groups = [group for group in val_groups if group.get("target_in_candidates")]
    if args.test_hit_only:
        test_groups = [group for group in test_groups if group.get("target_in_candidates")]

    data_summary = {
        "raw_train_groups": raw_train_groups,
        "raw_val_groups": raw_val_groups,
        "raw_test_groups": raw_test_groups,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_hit": sum(1 for x in train_groups if x.get("target_in_candidates")) / max(1, len(train_groups)),
        "val_hit": sum(1 for x in val_groups if x.get("target_in_candidates")) / max(1, len(val_groups)),
        "test_hit": sum(1 for x in test_groups if x.get("target_in_candidates")) / max(1, len(test_groups)),
    }
    (args.output_dir / "data_summary.json").write_text(json.dumps(data_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data_summary, ensure_ascii=False))

    popularity = build_popularity(train_groups)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModel.from_pretrained(args.base_model, **model_kwargs)
    base_model.config.use_cache = False
    for param in base_model.parameters():
        param.requires_grad = False
    lora_target_modules = resolve_lora_targets(base_model, args.lora_target_preset)
    replaced = inject_routed_lora(
        base_model,
        target_modules=lora_target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_num=args.lora_num,
    )
    relation_dim = BASE_RELATION_DIM + (ENHANCED_RELATION_DIM if args.enhanced_behavior_features else 0)
    model = OnlineMobilityCompressor(
        base_model=base_model,
        hidden_size=int(base_model.config.hidden_size),
        pool_dim=POOL_FEATURE_DIM,
        numeric_dim=12 + 5,  # intrinsic features plus popularity features
        relation_dim=relation_dim,
        text_proj_dim=args.text_proj_dim,
        anchor_proj_dim=args.anchor_proj_dim,
        anchor_hidden_dim=args.anchor_hidden_dim,
        relation_hidden_dim=args.relation_hidden_dim,
        relation_proj_dim=args.relation_proj_dim,
        category_bucket_size=args.category_bucket_size,
        geo_bucket_size=args.geo_bucket_size,
        semantic_bucket_size=args.semantic_bucket_size,
        alignment_mode=args.alignment_mode,
        dropout=args.alignment_dropout,
        pool_hidden_dim=args.pool_hidden_dim,
        pool_dropout=args.pool_dropout,
        candidate_token_attention=args.candidate_token_attention,
        din_history_attention=args.din_history_attention,
        pool_alpha_init=args.pool_alpha_init,
        fusion_alpha_init=args.fusion_alpha_init,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    unfrozen_layers = unfreeze_last_encoder_layers(model.aligner.base_model, args.unfreeze_last_n_layers)
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith("aligner.base_model."):
            param.data = param.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({"lora_replaced_modules": replaced, "trainable_params": trainable, "total_params": total}, ensure_ascii=False))
    teamlora_variant = "anonymous_online_pool_fusion_dinlite" if args.din_history_attention else "anonymous_online_pool_fusion"
    print(
        json.dumps(
            {
                "teamlora_variant": teamlora_variant,
                "lora_num": int(args.lora_num),
                "lora_target_preset": str(args.lora_target_preset),
                "lora_target_modules": list(lora_target_modules),
                "unfreeze_last_n_layers": int(args.unfreeze_last_n_layers),
                "unfrozen_encoder_layers": list(unfrozen_layers),
                "encoder_lr": float(args.encoder_lr),
                "alignment_mode": str(args.alignment_mode),
                "candidate_token_attention": bool(args.candidate_token_attention),
                "din_history_attention": bool(args.din_history_attention),
                "structured_mobility_input": bool(args.structured_mobility_input),
                "enhanced_behavior_features": bool(args.enhanced_behavior_features),
                "relation_dim": int(relation_dim),
                "event_numeric_dim": int(EVENT_NUMERIC_DIM),
                "pool_hidden_dim": int(args.pool_hidden_dim),
                "pool_alpha_init": float(args.pool_alpha_init),
                "fusion_alpha_init": float(args.fusion_alpha_init),
                "keep_margin": float(args.keep_margin),
                "pool_loss_weight": float(args.pool_loss_weight),
                "fixed_pool_prior": str(args.fixed_pool_prior),
                "selection_mode": str(args.selection_mode),
                "final_score_mode": str(args.final_score_mode),
                "train_negatives": int(args.train_negatives),
                "hard_negatives": int(args.hard_negatives),
                "raat_mode": str(args.raat_mode),
                "raat_loss_weight": float(args.raat_loss_weight),
                "teacher_loss_weight": float(args.teacher_loss_weight),
                "teacher_top_k": int(args.teacher_top_k),
                "teacher_temperature": float(args.teacher_temperature),
                "pool_feature_dim": int(POOL_FEATURE_DIM),
            },
            ensure_ascii=False,
        )
    )

    include_similar_profile = args.input_template == "semantic_profile_simuser_mobility_v1"
    train_ds = OnlineMobilityDataset(
        train_groups,
        include_similar_profile=include_similar_profile,
        eval_candidate_limit=None,
        category_bucket_size=args.category_bucket_size,
        geo_bucket_size=args.geo_bucket_size,
        semantic_bucket_size=args.semantic_bucket_size,
        popularity=popularity,
        structured_mobility_input=args.structured_mobility_input,
        enhanced_behavior_features=args.enhanced_behavior_features,
        max_trajectory_events=args.max_trajectory_events,
        fixed_pool_prior=args.fixed_pool_prior,
        teacher_top_k=args.teacher_top_k,
    )
    val_ds = OnlineMobilityDataset(
        val_groups,
        include_similar_profile=include_similar_profile,
        eval_candidate_limit=args.eval_candidate_limit,
        category_bucket_size=args.category_bucket_size,
        geo_bucket_size=args.geo_bucket_size,
        semantic_bucket_size=args.semantic_bucket_size,
        popularity=popularity,
        structured_mobility_input=args.structured_mobility_input,
        enhanced_behavior_features=args.enhanced_behavior_features,
        max_trajectory_events=args.max_trajectory_events,
        fixed_pool_prior=args.fixed_pool_prior,
        teacher_top_k=args.teacher_top_k,
    )
    test_ds = OnlineMobilityDataset(
        test_groups,
        include_similar_profile=include_similar_profile,
        eval_candidate_limit=args.eval_candidate_limit,
        category_bucket_size=args.category_bucket_size,
        geo_bucket_size=args.geo_bucket_size,
        semantic_bucket_size=args.semantic_bucket_size,
        popularity=popularity,
        structured_mobility_input=args.structured_mobility_input,
        enhanced_behavior_features=args.enhanced_behavior_features,
        max_trajectory_events=args.max_trajectory_events,
        fixed_pool_prior=args.fixed_pool_prior,
        teacher_top_k=args.teacher_top_k,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_groups, shuffle=True, collate_fn=collate_groups, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_groups, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_groups, num_workers=args.num_workers)

    optimizer = make_optimizer(model, lr=args.lr, encoder_lr=args.encoder_lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    explicit_eval_steps = sorted({step for step in parse_step_list(args.eval_at_steps) if 0 < step <= total_steps})
    if explicit_eval_steps:
        args.eval_at_steps = explicit_eval_steps
        print(json.dumps({"eval_at_steps": explicit_eval_steps, "eval_mode": "explicit"}, ensure_ascii=False))

    metadata = vars(args).copy()
    metadata.update(
        {
            "teamlora_variant": teamlora_variant,
            "target_modules": lora_target_modules,
            "relation_dim": relation_dim,
            "event_numeric_dim": EVENT_NUMERIC_DIM,
            "data_summary": data_summary,
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")

    scaler_enabled = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    global_step = 0
    micro_step = 0
    best_mrr = -1.0
    running_loss: List[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="train")
    while global_step < total_steps:
        for batch in train_loader:
            groups = batch["groups"]
            with torch.cuda.amp.autocast(enabled=scaler_enabled, dtype=torch.float16):
                loss_parts: List[torch.Tensor] = []
                stats_sum = {"pool_loss": 0.0, "keep_loss": 0.0, "final_loss": 0.0}
                for group in groups:
                    text_batch = encode_texts(tokenizer, [group["input_text"]], args.max_length, device)
                    pool_features = group["pool_features"].unsqueeze(0).to(device)
                    prior_scores = group["pool_prior_scores"].unsqueeze(0).to(device)
                    teacher_scores = group["teacher_scores"].unsqueeze(0).to(device)
                    teacher_top_indices = group["teacher_top_indices"].tolist()
                    numeric = group["candidate_numeric"].unsqueeze(0).to(device)
                    relation = group["candidate_relation"].unsqueeze(0).to(device)
                    category_ids = group["candidate_category_ids"].unsqueeze(0).to(device)
                    geo_ids = group["candidate_geo_ids"].unsqueeze(0).to(device)
                    semantic_ids = group["candidate_semantic_ids"].unsqueeze(0).to(device)
                    event_numeric = group["event_numeric"].unsqueeze(0).to(device)
                    event_poi_ids = group["event_poi_ids"].unsqueeze(0).to(device)
                    event_category_ids = group["event_category_ids"].unsqueeze(0).to(device)
                    event_geo_ids = group["event_geo_ids"].unsqueeze(0).to(device)
                    event_slot_ids = group["event_slot_ids"].unsqueeze(0).to(device)
                    event_weekday_ids = group["event_weekday_ids"].unsqueeze(0).to(device)
                    event_mask = group["event_mask"].unsqueeze(0).to(device)
                    target_positions = [int(group["target_pos"])]
                    pool_scores = model.pool_scorer(pool_features.float())
                    if args.pool_loss_weight > 0:
                        pool_loss = score_group_loss(pool_scores, target_positions)
                    else:
                        pool_loss = torch.tensor(0.0, device=device)
                    if pool_loss is None:
                        continue
                    base_scores = prior_scores if args.fixed_pool_prior != "none" else pool_scores
                    if args.train_negatives > 0:
                        sampled = sampled_train_indices(
                            int(group["target_pos"]),
                            int(base_scores.size(-1)),
                            hard_negatives=args.hard_negatives,
                            train_negatives=args.train_negatives,
                            teacher_indices=teacher_top_indices,
                        )
                        if sampled is None:
                            continue
                        selected_idx = torch.tensor([sampled], device=device, dtype=torch.long)
                    else:
                        selected_idx = select_candidate_indices(base_scores, args.top_k, args.selection_mode)
                    selected_base_scores = torch.gather(base_scores, 1, selected_idx)
                    selected_teacher_scores = torch.gather(teacher_scores, 1, selected_idx)
                    selected_numeric = gather_candidate_tensor(numeric, selected_idx)
                    selected_relation = gather_candidate_tensor(relation, selected_idx)
                    selected_category_ids = gather_candidate_tensor(category_ids, selected_idx)
                    selected_geo_ids = gather_candidate_tensor(geo_ids, selected_idx)
                    selected_semantic_ids = gather_candidate_tensor(semantic_ids, selected_idx)
                    dcn_scores = model.encode_selected(
                        text_batch,
                        selected_numeric,
                        selected_relation,
                        selected_category_ids,
                        selected_geo_ids,
                        selected_semantic_ids,
                        event_numeric=event_numeric,
                        event_poi_ids=event_poi_ids,
                        event_category_ids=event_category_ids,
                        event_geo_ids=event_geo_ids,
                        event_slot_ids=event_slot_ids,
                        event_weekday_ids=event_weekday_ids,
                        event_mask=event_mask,
                    )
                    final_scores = dcn_scores if args.final_score_mode == "dcn_only" else model.fuse_scores(selected_base_scores, dcn_scores)
                    selected_target_pos = remap_target_position(selected_idx[0], int(group["target_pos"]))
                    if selected_target_pos >= 0:
                        final_loss = F.cross_entropy(final_scores, torch.tensor([selected_target_pos], device=device))
                        if args.teacher_loss_weight > 0:
                            distill = teacher_kl_loss(final_scores[0], selected_teacher_scores[0], args.teacher_temperature)
                            if distill is not None:
                                final_loss = final_loss + args.teacher_loss_weight * distill
                        if args.raat_mode == "target_mask_2view" and args.raat_loss_weight > 0:
                            masked_numeric, masked_relation = target_mask_candidate_view(
                                selected_numeric,
                                selected_relation,
                                selected_target_pos,
                            )
                            masked_dcn_scores = model.encode_selected(
                                text_batch,
                                masked_numeric,
                                masked_relation,
                                selected_category_ids,
                                selected_geo_ids,
                                selected_semantic_ids,
                                event_numeric=event_numeric,
                                event_poi_ids=event_poi_ids,
                                event_category_ids=event_category_ids,
                                event_geo_ids=event_geo_ids,
                                event_slot_ids=event_slot_ids,
                                event_weekday_ids=event_weekday_ids,
                                event_mask=event_mask,
                            )
                            masked_scores = (
                                masked_dcn_scores
                                if args.final_score_mode == "dcn_only"
                                else model.fuse_scores(selected_base_scores, masked_dcn_scores)
                            )
                            final_loss = final_loss + args.raat_loss_weight * F.cross_entropy(
                                masked_scores,
                                torch.tensor([selected_target_pos], device=device),
                            )
                    else:
                        if args.pool_loss_weight <= 0 and args.keep_loss_weight <= 0:
                            continue
                        final_loss = torch.tensor(0.0, device=device)
                    cutoff_pos = min(args.top_k - 1, pool_scores.size(-1) - 1)
                    if args.keep_loss_weight > 0:
                        target_score = base_scores[0, group["target_pos"]]
                        cutoff_score = torch.sort(base_scores[0], descending=True).values[cutoff_pos]
                        keep_loss = F.relu(torch.tensor(args.keep_margin, device=device) - (target_score - cutoff_score))
                    else:
                        keep_loss = torch.tensor(0.0, device=device)
                    loss = args.pool_loss_weight * pool_loss + args.keep_loss_weight * keep_loss + args.final_loss_weight * final_loss
                    loss = loss / max(1, args.grad_accum)
                    loss_parts.append(loss)
                    stats_sum["pool_loss"] += float(pool_loss.detach().cpu())
                    stats_sum["keep_loss"] += float(keep_loss.detach().cpu())
                    stats_sum["final_loss"] += float(final_loss.detach().cpu())
                if not loss_parts:
                    continue
                loss = torch.stack(loss_parts).sum()
            scaler.scale(loss).backward()
            running_loss.append(float(loss.detach().cpu().item() * max(1, args.grad_accum)))
            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)
            if global_step % args.logging_steps == 0:
                avg_loss = sum(running_loss[-args.logging_steps :]) / max(1, min(len(running_loss), args.logging_steps))
                print(json.dumps({"step": global_step, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]}, ensure_ascii=False))
            if explicit_eval_steps:
                should_eval = global_step in explicit_eval_steps
            else:
                should_eval = (global_step % args.eval_steps == 0 or global_step == total_steps) and (
                    not args.eval_final_only or global_step == total_steps
                )
            if should_eval:
                model_state = {
                    "step": global_step,
                    "checkpoint_type": "latest_before_eval",
                }
                model_path = args.output_dir / "latest"
                model_path.mkdir(parents=True, exist_ok=True)
                torch.save({name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}, model_path / "trainable_model.bin")
                (model_path / "metadata.json").write_text(json.dumps(metadata | model_state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
                val_metrics = evaluate(model, tokenizer, val_loader, args, device)
                test_metrics = evaluate(model, tokenizer, test_loader, args, device)
                metrics = {"step": global_step, "val": val_metrics, "test": test_metrics}
                print(json.dumps(metrics, ensure_ascii=False))
                with (args.output_dir / "eval_history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                if val_metrics["mrr"] > best_mrr:
                    best_mrr = val_metrics["mrr"]
                    best_path = args.output_dir / "best"
                    best_path.mkdir(parents=True, exist_ok=True)
                    torch.save({name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}, best_path / "trainable_model.bin")
                    (best_path / "metadata.json").write_text(
                        json.dumps(metadata | {"best_step": global_step, "best_metrics": metrics}, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8",
                    )
                model.train()
            if global_step % args.save_steps == 0 and not should_eval:
                latest_path = args.output_dir / "latest"
                latest_path.mkdir(parents=True, exist_ok=True)
                torch.save({name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}, latest_path / "trainable_model.bin")
                (latest_path / "metadata.json").write_text(
                    json.dumps(metadata | {"step": global_step, "checkpoint_type": "latest"}, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
    progress.close()
    final_val_metrics = evaluate(model, tokenizer, val_loader, args, device)
    final_test_metrics = evaluate(model, tokenizer, test_loader, args, device)
    final_metrics = {"step": global_step, "val": final_val_metrics, "test": final_test_metrics}
    (args.output_dir / "final_eval.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
