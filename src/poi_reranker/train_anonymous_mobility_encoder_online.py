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
    p.add_argument("--semantic-map", type=Path, default=Path("retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl"))
    p.add_argument("--base-model", type=Path, default=Path("models/Llama-3.2-1B-Instruct"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--pool-k", type=int, default=500, help="Input wide candidate pool size.")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-length", type=int, default=1280)
    p.add_argument("--input-template", choices=["semantic_profile_mobility_v1", "semantic_profile_simuser_mobility_v1"], default="semantic_profile_simuser_mobility_v1")
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
    p.add_argument("--pool-hidden-dim", type=int, default=0, help="Optional hidden dim for compressor; 0 keeps it linear.")
    p.add_argument("--pool-dropout", type=float, default=0.0)
    p.add_argument("--fusion-alpha-init", type=float, default=0.3)
    p.add_argument("--keep-margin", type=float, default=0.1)
    p.add_argument("--keep-loss-weight", type=float, default=1.0)
    p.add_argument("--final-loss-weight", type=float, default=1.0)
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


def candidate_relation_features(candidate: Dict[str, Any], relation_context: Dict[str, Any]) -> List[float]:
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

    return [
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


def format_mobility_text(group: Dict[str, Any], include_similar_profile: bool) -> str:
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
            "input_text": format_mobility_text(group, include_similar_profile=self.include_similar_profile),
            "pool_features": torch.from_numpy(np.asarray(pool_arr, dtype=np.float32)),
            "candidate_numeric": torch.tensor([candidate_intrinsic_features(c, self.popularity) for c in candidates], dtype=torch.float32),
            "candidate_relation": torch.tensor(
                [candidate_relation_features(c, relation_context) for c in candidates],
                dtype=torch.float32,
            ),
            "candidate_category_ids": torch.tensor([stable_bucket(c.get("category"), self.category_bucket_size) for c in candidates], dtype=torch.long),
            "candidate_geo_ids": torch.tensor([stable_bucket(c.get("geo_cell"), self.geo_bucket_size) for c in candidates], dtype=torch.long),
            "candidate_semantic_ids": torch.tensor([stable_bucket(c.get("semantic_id"), self.semantic_bucket_size) for c in candidates], dtype=torch.long),
            "candidate_poi_ids": [c.get("poi_id") for c in candidates],
        }


def collate_groups(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"groups": list(features)}


def load_groups_from_files(
    joined_path: Path,
    candidate_path: Path,
    semantic_map_path: Path,
    pool_k: int,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    semantic_map = load_semantic_map(semantic_map_path)
    joined_rows = read_joined_parquet(joined_path, limit=limit)
    candidate_rows = read_candidate_jsonl_map(candidate_path, limit=limit)
    groups: List[Dict[str, Any]] = []
    for joined_row in joined_rows:
        sample_id = str(joined_row.get("sample_id") or "")
        candidate_row = candidate_rows.get(sample_id)
        if candidate_row is None:
            continue
        group = build_online_group(joined_row, candidate_row, semantic_map, pool_k=pool_k)
        if group is not None:
            groups.append(group)
    return groups


def select_topk_indices(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, scores.size(-1))
    return torch.topk(scores, k=top_k, dim=-1).indices


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
    ) -> torch.Tensor:
        return self.aligner(
            text_batch,
            candidate_numeric,
            candidate_relation,
            candidate_category_ids,
            candidate_geo_ids,
            candidate_semantic_ids,
        )


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
            selected_idx = select_topk_indices(pool_scores, args.top_k)
            selected_idx_cpu = selected_idx.detach().cpu()
            selected_target_pos = next((i for i, idx in enumerate(selected_idx_cpu.tolist()) if int(idx) == target_pos), -1)
            if selected_target_pos < 0:
                missing += 1
                continue

            if candidate_batch_size >= len(selected_idx):
                numeric = group["candidate_numeric"][selected_idx_cpu].unsqueeze(0).to(device)
                relation = group["candidate_relation"][selected_idx_cpu].unsqueeze(0).to(device)
                cat = group["candidate_category_ids"][selected_idx_cpu].unsqueeze(0).to(device)
                geo = group["candidate_geo_ids"][selected_idx_cpu].unsqueeze(0).to(device)
                sem = group["candidate_semantic_ids"][selected_idx_cpu].unsqueeze(0).to(device)
                text = encode_texts(tokenizer, [group["input_text"]], args.max_length, device)
                dcn_scores = model.encode_selected(text, numeric, relation, cat, geo, sem)[0]
            else:
                text = encode_texts(tokenizer, [group["input_text"]], args.max_length, device)
                user = model.aligner.text_proj(model.aligner.encode_text(text["input_ids"], text["attention_mask"]))
                user = F.normalize(user, dim=-1)
                numeric = group["candidate_numeric"][selected_idx_cpu].to(device)
                relation = group["candidate_relation"][selected_idx_cpu].to(device)
                cat = group["candidate_category_ids"][selected_idx_cpu].to(device)
                geo = group["candidate_geo_ids"][selected_idx_cpu].to(device)
                sem = group["candidate_semantic_ids"][selected_idx_cpu].to(device)
                chunks: List[torch.Tensor] = []
                for start in range(0, numeric.size(0), candidate_batch_size):
                    end = min(numeric.size(0), start + candidate_batch_size)
                    anchors = model.aligner.encode_anchors(numeric[start:end], relation[start:end], cat[start:end], geo[start:end], sem[start:end])
                    anchors = F.normalize(anchors, dim=-1)
                    if model.aligner.alignment_mode == "linear":
                        chunks.append(torch.einsum("bd,nd->n", user, anchors))
                    elif model.aligner.alignment_mode == "low_rank":
                        user_expand = user.expand(anchors.size(0), -1)
                        pair = torch.cat([user_expand, anchors, user_expand * anchors], dim=-1)
                        chunks.append(model.aligner.pair_scorer(pair).squeeze(-1))
                    else:
                        user_expand = user.expand(anchors.size(0), -1)
                        pair = torch.cat([user_expand, anchors], dim=-1)
                        cross_out = model.aligner.cross_network(pair)
                        deep_out = model.aligner.deep_tower(pair)
                        score_input = torch.cat([pair, cross_out, deep_out], dim=-1)
                        chunks.append(model.aligner.final_scorer(score_input).squeeze(-1))
                dcn_scores = torch.cat(chunks, dim=0)

            selected_pool_scores = pool_scores[selected_idx]
            final_scores = selected_pool_scores + model.fusion_alpha * dcn_scores
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
    numeric = torch.stack([g["candidate_numeric"] for g in groups], dim=0).to(device)
    relation = torch.stack([g["candidate_relation"] for g in groups], dim=0).to(device)
    category_ids = torch.stack([g["candidate_category_ids"] for g in groups], dim=0).to(device)
    geo_ids = torch.stack([g["candidate_geo_ids"] for g in groups], dim=0).to(device)
    semantic_ids = torch.stack([g["candidate_semantic_ids"] for g in groups], dim=0).to(device)
    target_positions = [int(g["target_pos"]) for g in groups]

    pool_scores = model.pool_scorer(pool_features.float())
    pool_loss = score_group_loss(pool_scores, target_positions)
    if pool_loss is None:
        return None, {}

    valid_rows = [idx for idx, pos in enumerate(target_positions) if pos >= 0]
    keep_loss = torch.tensor(0.0, device=device)
    final_loss = torch.tensor(0.0, device=device)
    total_keep = 0
    total_final = 0
    selected_idx = select_topk_indices(pool_scores, args.top_k)
    selected_pool_scores = torch.gather(pool_scores, 1, selected_idx)
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
    )
    final_scores = selected_pool_scores + model.fusion_alpha * dcn_scores
    final_target_positions: List[int] = []

    for row_idx in valid_rows:
        target_pos = target_positions[row_idx]
        selected_row = selected_idx[row_idx]
        selected_pos = (selected_row == target_pos).nonzero(as_tuple=False)
        if selected_pos.numel() > 0:
            final_target_positions.append(int(selected_pos[0].item()))
            total_final += 1
        cutoff_pos = min(args.top_k - 1, pool_scores.size(-1) - 1)
        target_score = pool_scores[row_idx, target_pos]
        cutoff_score = torch.sort(pool_scores[row_idx], descending=True).values[cutoff_pos]
        keep_loss = keep_loss + F.relu(torch.tensor(args.keep_margin, device=device) - (target_score - cutoff_score))
        total_keep += 1

    if final_target_positions:
        final_rows = [idx for idx, pos in enumerate(target_positions) if pos >= 0 and pos in selected_idx[idx].tolist()]
        if final_rows:
            row_scores = final_scores[final_rows]
            row_targets = torch.tensor(final_target_positions, device=device, dtype=torch.long)
            final_loss = F.cross_entropy(row_scores, row_targets)

    keep_loss = keep_loss / max(total_keep, 1)
    total_loss = pool_loss + args.keep_loss_weight * keep_loss + args.final_loss_weight * final_loss
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

    train_groups = load_groups_from_files(args.train_joined, args.train_candidates, args.semantic_map, args.pool_k, args.max_train_groups)
    val_groups = load_groups_from_files(args.val_joined, args.val_candidates, args.semantic_map, args.pool_k, args.max_val_groups)
    test_groups = load_groups_from_files(args.test_joined, args.test_candidates, args.semantic_map, args.pool_k, args.max_test_groups)
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
    replaced = 0
    from train_anonymous_mobility_encoder import inject_routed_teamlora

    replaced = inject_routed_teamlora(base_model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, lora_num=args.lora_num)
    model = OnlineMobilityCompressor(
        base_model=base_model,
        hidden_size=int(base_model.config.hidden_size),
        pool_dim=POOL_FEATURE_DIM,
        numeric_dim=12 + 5,  # intrinsic features plus popularity features
        relation_dim=16,
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
        fusion_alpha_init=args.fusion_alpha_init,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({"lora_replaced_modules": replaced, "trainable_params": trainable, "total_params": total}, ensure_ascii=False))
    print(
        json.dumps(
            {
                "teamlora_variant": "anonymous_online_pool_fusion",
                "lora_num": int(args.lora_num),
                "alignment_mode": str(args.alignment_mode),
                "pool_hidden_dim": int(args.pool_hidden_dim),
                "fusion_alpha_init": float(args.fusion_alpha_init),
                "keep_margin": float(args.keep_margin),
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
    )
    val_ds = OnlineMobilityDataset(
        val_groups,
        include_similar_profile=include_similar_profile,
        eval_candidate_limit=args.eval_candidate_limit,
        category_bucket_size=args.category_bucket_size,
        geo_bucket_size=args.geo_bucket_size,
        semantic_bucket_size=args.semantic_bucket_size,
        popularity=popularity,
    )
    test_ds = OnlineMobilityDataset(
        test_groups,
        include_similar_profile=include_similar_profile,
        eval_candidate_limit=args.eval_candidate_limit,
        category_bucket_size=args.category_bucket_size,
        geo_bucket_size=args.geo_bucket_size,
        semantic_bucket_size=args.semantic_bucket_size,
        popularity=popularity,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_groups, shuffle=True, collate_fn=collate_groups, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_groups, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_groups, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    explicit_eval_steps = sorted({step for step in parse_step_list(args.eval_at_steps) if 0 < step <= total_steps})
    if explicit_eval_steps:
        args.eval_at_steps = explicit_eval_steps
        print(json.dumps({"eval_at_steps": explicit_eval_steps, "eval_mode": "explicit"}, ensure_ascii=False))

    metadata = vars(args).copy()
    metadata.update({"teamlora_variant": "anonymous_online_pool_fusion", "target_modules": TARGET_MODULES, "data_summary": data_summary})
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
                    numeric = group["candidate_numeric"].unsqueeze(0).to(device)
                    relation = group["candidate_relation"].unsqueeze(0).to(device)
                    category_ids = group["candidate_category_ids"].unsqueeze(0).to(device)
                    geo_ids = group["candidate_geo_ids"].unsqueeze(0).to(device)
                    semantic_ids = group["candidate_semantic_ids"].unsqueeze(0).to(device)
                    target_positions = [int(group["target_pos"])]
                    pool_scores = model.pool_scorer(pool_features.float())
                    pool_loss = score_group_loss(pool_scores, target_positions)
                    if pool_loss is None:
                        continue
                    selected_idx = select_topk_indices(pool_scores, args.top_k)
                    selected_pool_scores = torch.gather(pool_scores, 1, selected_idx)
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
                    )
                    final_scores = selected_pool_scores + model.fusion_alpha * dcn_scores
                    selected_target_pos = next((i for i, idx in enumerate(selected_idx[0].tolist()) if idx == group["target_pos"]), -1)
                    if selected_target_pos >= 0:
                        final_loss = F.cross_entropy(final_scores, torch.tensor([selected_target_pos], device=device))
                    else:
                        final_loss = torch.tensor(0.0, device=device)
                    target_score = pool_scores[0, group["target_pos"]]
                    cutoff_pos = min(args.top_k - 1, pool_scores.size(-1) - 1)
                    cutoff_score = torch.sort(pool_scores[0], descending=True).values[cutoff_pos]
                    keep_loss = F.relu(torch.tensor(args.keep_margin, device=device) - (target_score - cutoff_score))
                    loss = pool_loss + args.keep_loss_weight * keep_loss + args.final_loss_weight * final_loss
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
