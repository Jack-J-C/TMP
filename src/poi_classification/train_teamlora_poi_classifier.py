#!/usr/bin/env python3
"""End-to-end POI classification with Llama + TeamLoRA + MLP head.

This is the first non-generative version:
  multi-view evidence text -> Llama hidden state -> MLP -> full POI logits.

The JSON output format is handled only as post-processing. Training labels are
integer POI class ids, not assistant tokens.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TeamLoRA POI classifier.")
    p.add_argument("--base-model", type=Path, default=Path("/mnt/data/yyl/TMP/models/Llama-3.2-1B-Instruct"))
    p.add_argument("--semantic-map", type=Path, default=Path("retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl"))
    p.add_argument("--train-joined", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet"))
    p.add_argument("--val-joined", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet"))
    p.add_argument("--train-raw", type=Path, default=Path("retrieval_assets/NewYork/semantic_poi_sft/stage1_train_raw_semantic.jsonl"))
    p.add_argument("--val-raw", type=Path, default=Path("retrieval_assets/NewYork/semantic_poi_sft/stage1_val_raw_semantic.jsonl"))
    p.add_argument(
        "--train-refined",
        type=Path,
        default=Path("retrieval_assets/NewYork/refined_prompts_cleaned/lora_a_decision_v2_train_full_outputs_high_medium_clean.jsonl"),
    )
    p.add_argument(
        "--val-refined",
        type=Path,
        default=Path("retrieval_assets/NewYork/refined_prompts_cleaned/lora_a_decision_v2_val_full_outputs_high_medium_clean.jsonl"),
    )
    p.add_argument(
        "--train-refined-fallback",
        type=Path,
        default=Path("retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_train_full_outputs.jsonl"),
        help="Full refined outputs used when cleaned refined rows are missing.",
    )
    p.add_argument(
        "--val-refined-fallback",
        type=Path,
        default=Path("retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_val_full_outputs.jsonl"),
        help="Full refined outputs used when cleaned refined rows are missing.",
    )
    p.add_argument(
        "--train-graph",
        type=Path,
        default=Path("retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_train_candidates.jsonl"),
    )
    p.add_argument(
        "--val-graph",
        type=Path,
        default=Path("retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_val_candidates.jsonl"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("models/poi-teamlora-classifier-llama32-1b-v1"))
    p.add_argument("--graph-top-k", type=int, default=100)
    p.add_argument("--graph-prior-alpha", type=float, default=1.0)
    p.add_argument("--graph-prior-mode", choices=["none", "rank", "score"], default="rank")
    p.add_argument("--mlp-logit-scale", type=float, default=1.0, help="Scale classifier logits before adding GraphRAG prior.")
    p.add_argument("--graph-prior-dropout", type=float, default=0.0, help="Train-time probability of removing all GraphRAG prior candidates.")
    p.add_argument(
        "--graph-prior-random-cutoffs",
        type=str,
        default="",
        help="Comma-separated train-time candidate cutoffs sampled per row, e.g. '10,30,100'. Empty keeps all candidates.",
    )
    p.add_argument("--graph-candidate-drop-prob", type=float, default=0.0, help="Train-time per-candidate drop probability for RAAT-style partial candidates.")
    p.add_argument("--graph-target-mask-prob", type=float, default=0.0, help="Train-time probability of removing the target from GraphRAG candidates.")
    p.add_argument("--graph-irrelevant-mix-prob", type=float, default=0.0, help="Train-time probability of replacing some candidates with irrelevant POIs.")
    p.add_argument("--graph-rank-noise-prob", type=float, default=0.0, help="Train-time probability of shuffling candidate ranks/scores.")
    p.add_argument("--graph-noise-loss-weight", type=float, default=0.0, help="Auxiliary weight for 4-way GraphRAG noise-type classification.")
    p.add_argument(
        "--raat-mode",
        choices=["none", "target_mask_2view"],
        default="none",
        help="Optional RAAT hardest-selection mode. target_mask_2view uses max CE over clean and target-masked GraphRAG views.",
    )
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--max-train-samples", type=int, default=None, help="Optional smoke-test limit.")
    p.add_argument("--max-val-samples", type=int, default=None, help="Optional smoke-test limit.")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-num", type=int, default=3)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--classifier-dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["last", "mean", "last_mean"], default="last_mean")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="If > 0, override epochs and stop after this many optimizer steps.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--load-best-model-at-end", action="store_true")
    p.add_argument("--metric-for-best-model", type=str, default="mrr")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device-map", default=None)
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust-remote-code", action="store_true")
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


def load_graph_by_sample_id(path: Path, top_k: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} row without sample_id")
        if sid in out:
            raise ValueError(f"{path} duplicate sample_id: {sid}")
        candidates = [str(x) for x in row.get("candidate_poi_ids") or []][:top_k]
        details = row.get("candidate_details") or {}
        out[sid] = {
            "sample_id": sid,
            "target": row.get("target") or {},
            "target_in_topk": row.get("target_in_topk"),
            "candidate_poi_ids": candidates,
            "candidate_details": {poi: details.get(poi) or {} for poi in candidates},
        }
    return out


def load_refined_with_fallback(cleaned_path: Path, fallback_path: Path | None) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    cleaned = load_by_sample_id(cleaned_path)
    source = {sid: "cleaned" for sid in cleaned}
    fallback_added = 0
    fallback = {}
    if fallback_path is not None and fallback_path.exists():
        fallback = load_by_sample_id(fallback_path)
        for sid, row in fallback.items():
            if sid not in cleaned:
                cleaned[sid] = row
                source[sid] = "fallback_full"
                fallback_added += 1
    return cleaned, source, {
        "cleaned_rows": sum(1 for value in source.values() if value == "cleaned"),
        "fallback_rows_available": len(fallback),
        "fallback_rows_added": fallback_added,
    }


def load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["poi_id"]): row for row in read_jsonl(path)}


def load_joined_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    raise ValueError(f"Unsupported joined data format: {path}")


def build_poi_vocab(
    semantic_map: Dict[str, Dict[str, Any]],
    raw_paths: Sequence[Path] = (),
    joined_rows: Sequence[Dict[str, Any]] = (),
) -> tuple[Dict[str, int], Dict[int, str]]:
    poi_ids = set(semantic_map)
    for row in joined_rows:
        target = str(row.get("target_poi_id") or "")
        if target:
            poi_ids.add(target)
    for path in raw_paths:
        for row in read_jsonl(path):
            target = ((row.get("target") or {}).get("poi_id")) or ""
            if target:
                poi_ids.add(str(target))
    ordered = sorted(poi_ids, key=lambda x: (len(x), x))
    poi2idx = {poi: idx for idx, poi in enumerate(ordered)}
    idx2poi = {idx: poi for poi, idx in poi2idx.items()}
    return poi2idx, idx2poi


def prepare_joined_rows(rows: List[Dict[str, Any]], poi2idx: Dict[str, int]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    counts = Counter()
    for row in rows:
        target = str(row.get("target_poi_id") or "")
        if target not in poi2idx:
            counts["target_not_in_vocab"] += 1
            continue
        graph_candidate_poi_ids = [str(x) for x in (row.get("graph_candidate_poi_ids") or [])]
        if not graph_candidate_poi_ids:
            graph_candidate_poi_ids = parse_graph_candidate_ids(str(row.get("graph_text") or ""))
        graph_candidate_ranks = [int(x) for x in (row.get("graph_candidate_ranks") or [])]
        if len(graph_candidate_ranks) != len(graph_candidate_poi_ids):
            graph_candidate_ranks = list(range(1, len(graph_candidate_poi_ids) + 1))
        graph_candidate_scores = []
        for value in row.get("graph_candidate_scores") or []:
            try:
                graph_candidate_scores.append(float(value))
            except (TypeError, ValueError):
                graph_candidate_scores.append(0.0)
        if len(graph_candidate_scores) != len(graph_candidate_poi_ids):
            graph_candidate_scores = [0.0] * len(graph_candidate_poi_ids)
        graph_candidate_indices = [poi2idx[x] for x in graph_candidate_poi_ids if x in poi2idx]
        graph_candidate_ranks = [rank for poi, rank in zip(graph_candidate_poi_ids, graph_candidate_ranks) if poi in poi2idx]
        graph_candidate_scores = [score for poi, score in zip(graph_candidate_poi_ids, graph_candidate_scores) if poi in poi2idx]
        out.append(
            {
                "sample_id": str(row.get("sample_id") or ""),
                "input_text": str(row.get("input_text") or ""),
                "target_poi_id": target,
                "target_poi_idx": poi2idx[target],
                "graph_candidate_indices": graph_candidate_indices,
                "graph_candidate_ranks": graph_candidate_ranks,
                "graph_candidate_scores": graph_candidate_scores,
                "graph_target_in_candidates": poi2idx[target] in set(graph_candidate_indices),
            }
        )
        counts["rows"] += 1
        if row.get("graph_target_in_topk"):
            counts["graph_target_in_topk"] += 1
        if row.get("refined_confidence"):
            counts[f"refined_confidence_{row.get('refined_confidence')}"] += 1
        if row.get("refined_useful"):
            counts[f"refined_useful_{row.get('refined_useful')}"] += 1
    stats = dict(counts)
    stats["graph_target_in_topk_ratio"] = round(counts["graph_target_in_topk"] / counts["rows"], 6) if counts["rows"] else 0.0
    return out, stats


def parse_graph_candidate_ids(graph_text: str) -> List[str]:
    out: List[str] = []
    for line in graph_text.splitlines():
        match = re.match(r"\s*\d+\.\s+(v\d+)\|", line)
        if match:
            out.append(match.group(1))
    return out


def compact_geo(row: Dict[str, Any]) -> tuple[str, bool]:
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


def build_joined_rows(
    raw_path: Path,
    refined_path: Path,
    refined_fallback_path: Path | None,
    graph_path: Path,
    semantic_map: Dict[str, Dict[str, Any]],
    poi2idx: Dict[str, int],
    graph_top_k: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    refined_by_id, refined_source, refined_stats = load_refined_with_fallback(refined_path, refined_fallback_path)
    graph_by_id = load_graph_by_sample_id(graph_path, graph_top_k)
    rows: List[Dict[str, Any]] = []
    counts = Counter()
    for raw in read_jsonl(raw_path):
        sid = str(raw.get("sample_id") or "")
        refined = refined_by_id.get(sid)
        graph = graph_by_id.get(sid)
        if refined is None:
            counts["missing_refined"] += 1
            continue
        if graph is None:
            counts["missing_graph"] += 1
            continue
        target_poi = str(((raw.get("target") or {}).get("poi_id")) or "")
        if not target_poi:
            counts["missing_target"] += 1
            continue
        if target_poi not in poi2idx:
            counts["target_not_in_vocab"] += 1
            continue
        graph_target = str(((graph.get("target") or {}).get("poi_id")) or "")
        if graph_target and graph_target != target_poi:
            counts["graph_target_mismatch"] += 1
            continue
        raw_text = strip_generation_instructions(str(raw.get("input_prompt") or ""))
        refined_text = str(refined.get("distilled_prompt") or "").strip() or "none"
        refined_header = f"[VIEW=REFINED source={refined_source.get(sid, 'unknown')}]"
        input_text = "\n\n".join(
            [
                "[VIEW=RAW_SEM]",
                raw_text,
                refined_header,
                refined_text,
                "[VIEW=GRAPH_RAG]",
                graph_view_text(graph, semantic_map, graph_top_k),
            ]
        )
        rows.append(
            {
                "sample_id": sid,
                "input_text": input_text,
                "target_poi_id": target_poi,
                "target_poi_idx": poi2idx[target_poi],
                "graph_candidate_indices": [poi2idx[poi] for poi in graph.get("candidate_poi_ids", []) if poi in poi2idx],
                "graph_candidate_ranks": [rank for rank, poi in enumerate(graph.get("candidate_poi_ids", []), 1) if poi in poi2idx],
                "graph_candidate_scores": [
                    float(((graph.get("candidate_details") or {}).get(poi) or {}).get("score") or 0.0)
                    for poi in graph.get("candidate_poi_ids", [])
                    if poi in poi2idx
                ],
            }
        )
        counts["rows"] += 1
        if graph.get("target_in_topk"):
            counts["graph_target_in_topk"] += 1
        counts[f"refined_source_{refined_source.get(sid, 'unknown')}"] += 1
    stats = dict(counts)
    stats["refined_loading"] = refined_stats
    stats["graph_target_in_topk_ratio"] = round(counts["graph_target_in_topk"] / counts["rows"], 6) if counts["rows"] else 0.0
    return rows, stats


@dataclass
class POIClassificationDataset(Dataset):
    rows: List[Dict[str, Any]]
    tokenizer: Any
    max_length: int
    all_poi_indices: Sequence[int] = ()
    graph_prior_dropout: float = 0.0
    graph_prior_random_cutoffs: Sequence[int] = ()
    graph_candidate_drop_prob: float = 0.0
    graph_target_mask_prob: float = 0.0
    graph_irrelevant_mix_prob: float = 0.0
    graph_rank_noise_prob: float = 0.0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["input_text"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = int(row["target_poi_idx"])
        target_idx = int(row["target_poi_idx"])
        graph_candidate_indices = list(row.get("graph_candidate_indices") or [])
        graph_candidate_ranks = list(row.get("graph_candidate_ranks") or [])
        graph_candidate_scores = list(row.get("graph_candidate_scores") or [])
        graph_noise_label = 0
        if self.graph_prior_dropout > 0 and torch.rand(()).item() < self.graph_prior_dropout:
            graph_candidate_indices = []
            graph_candidate_ranks = []
            graph_candidate_scores = []
            graph_noise_label = 1
        elif self.graph_prior_random_cutoffs:
            cutoff_idx = int(torch.randint(len(self.graph_prior_random_cutoffs), ()).item())
            cutoff = max(0, int(self.graph_prior_random_cutoffs[cutoff_idx]))
            if cutoff < len(graph_candidate_indices):
                graph_noise_label = max(graph_noise_label, 1)
            graph_candidate_indices = graph_candidate_indices[:cutoff]
            graph_candidate_ranks = graph_candidate_ranks[:cutoff]
            graph_candidate_scores = graph_candidate_scores[:cutoff]
        if graph_candidate_indices and self.graph_candidate_drop_prob > 0:
            kept = [
                (idx, rank, score)
                for idx, rank, score in zip(graph_candidate_indices, graph_candidate_ranks, graph_candidate_scores)
                if torch.rand(()).item() >= self.graph_candidate_drop_prob
            ]
            if len(kept) != len(graph_candidate_indices):
                graph_noise_label = max(graph_noise_label, 1)
            graph_candidate_indices = [x[0] for x in kept]
            graph_candidate_ranks = [x[1] for x in kept]
            graph_candidate_scores = [x[2] for x in kept]
        if graph_candidate_indices and self.graph_target_mask_prob > 0 and target_idx in graph_candidate_indices and torch.rand(()).item() < self.graph_target_mask_prob:
            kept = [
                (idx, rank, score)
                for idx, rank, score in zip(graph_candidate_indices, graph_candidate_ranks, graph_candidate_scores)
                if idx != target_idx
            ]
            graph_candidate_indices = [x[0] for x in kept]
            graph_candidate_ranks = [x[1] for x in kept]
            graph_candidate_scores = [x[2] for x in kept]
            graph_noise_label = 2
        if graph_candidate_indices and self.graph_irrelevant_mix_prob > 0 and self.all_poi_indices and torch.rand(()).item() < self.graph_irrelevant_mix_prob:
            graph_candidate_indices, graph_candidate_ranks, graph_candidate_scores = self.mix_irrelevant_candidates(
                graph_candidate_indices,
                graph_candidate_ranks,
                graph_candidate_scores,
                target_idx,
            )
            graph_noise_label = 3
        if len(graph_candidate_indices) > 1 and self.graph_rank_noise_prob > 0 and torch.rand(()).item() < self.graph_rank_noise_prob:
            perm = torch.randperm(len(graph_candidate_indices)).tolist()
            graph_candidate_indices = [graph_candidate_indices[i] for i in perm]
            graph_candidate_ranks = list(range(1, len(graph_candidate_indices) + 1))
            graph_candidate_scores = [graph_candidate_scores[i] for i in perm]
            graph_noise_label = max(graph_noise_label, 1)
        enc["graph_candidate_indices"] = graph_candidate_indices
        enc["graph_candidate_ranks"] = graph_candidate_ranks
        enc["graph_candidate_scores"] = graph_candidate_scores
        enc["graph_noise_labels"] = int(graph_noise_label)
        return enc

    def mix_irrelevant_candidates(
        self,
        indices: List[int],
        ranks: List[int],
        scores: List[float],
        target_idx: int,
    ) -> tuple[List[int], List[int], List[float]]:
        if not self.all_poi_indices:
            return indices, ranks, scores
        replace_n = max(1, min(len(indices), int(round(len(indices) * 0.1))))
        candidate_set = set(indices)
        mixed_indices = list(indices)
        mixed_scores = list(scores)
        for _ in range(replace_n):
            pos = int(torch.randint(len(mixed_indices), ()).item())
            for _attempt in range(20):
                new_idx = int(self.all_poi_indices[int(torch.randint(len(self.all_poi_indices), ()).item())])
                if new_idx != target_idx and new_idx not in candidate_set:
                    break
            mixed_indices[pos] = new_idx
            mixed_scores[pos] = 0.0
            candidate_set.add(new_idx)
        return mixed_indices, list(range(1, len(mixed_indices) + 1)), mixed_scores


@dataclass
class DataCollatorForPOIClassification:
    tokenizer: Any
    raat_mode: str = "none"

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([int(x.pop("labels")) for x in features], dtype=torch.long)
        graph_noise_labels = torch.tensor([int(x.pop("graph_noise_labels", 0)) for x in features], dtype=torch.long)
        graph_indices = [x.pop("graph_candidate_indices", []) for x in features]
        graph_ranks = [x.pop("graph_candidate_ranks", []) for x in features]
        graph_scores = [x.pop("graph_candidate_scores", []) for x in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["graph_noise_labels"] = graph_noise_labels
        max_graph = max((len(x) for x in graph_indices), default=0)
        if max_graph > 0:
            padded_indices = torch.full((len(features), max_graph), -1, dtype=torch.long)
            padded_ranks = torch.zeros((len(features), max_graph), dtype=torch.float32)
            padded_scores = torch.zeros((len(features), max_graph), dtype=torch.float32)
            mask = torch.zeros((len(features), max_graph), dtype=torch.bool)
            for row_idx, (indices, ranks, scores) in enumerate(zip(graph_indices, graph_ranks, graph_scores)):
                n = min(len(indices), max_graph)
                if n == 0:
                    continue
                padded_indices[row_idx, :n] = torch.tensor(indices[:n], dtype=torch.long)
                padded_ranks[row_idx, :n] = torch.tensor(ranks[:n], dtype=torch.float32)
                padded_scores[row_idx, :n] = torch.tensor(scores[:n], dtype=torch.float32)
                mask[row_idx, :n] = True
            batch["graph_candidate_indices"] = padded_indices
            batch["graph_candidate_ranks"] = padded_ranks
            batch["graph_candidate_scores"] = padded_scores
            batch["graph_candidate_mask"] = mask
            if self.raat_mode == "target_mask_2view":
                raat_mask = mask.clone()
                raat_labels = labels.view(-1, 1)
                raat_mask = raat_mask & padded_indices.ne(raat_labels)
                batch["raat_graph_candidate_indices"] = padded_indices
                batch["raat_graph_candidate_ranks"] = padded_ranks
                batch["raat_graph_candidate_scores"] = padded_scores
                batch["raat_graph_candidate_mask"] = raat_mask
        return batch


class TeamLoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int,
        lora_num: int,
        lora_dropout: float,
        bias: bool,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.r = int(r)
        self.lora_num = int(lora_num)
        self.scaling = float(lora_alpha) / float(r) * float(lora_num)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lora_route = nn.Linear(in_features, lora_num, bias=False)
        self.lora_A = nn.Linear(in_features, r * lora_num, bias=False)
        self.lora_B = nn.ModuleList([nn.Linear(r, out_features, bias=False) for _ in range(lora_num)])
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
        for layer in self.lora_B:
            nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        route = F.softmax(self.lora_route(x), dim=-1).to(result.dtype)
        a = self.lora_dropout(self.lora_A(x) * self.scaling)
        route = torch.repeat_interleave(route, repeats=self.r, dim=-1)
        b_weight = torch.cat([layer.weight for layer in self.lora_B], dim=-1).t()
        return result + (a * route) @ b_weight


def inject_teamlora(model: nn.Module, r: int, alpha: int, lora_num: int, dropout: float) -> int:
    replaced = 0
    key_list = [name for name, _ in model.named_modules()]
    for key in key_list:
        if not any(key.endswith(target) for target in TARGET_MODULES):
            continue
        target = model.get_submodule(key)
        if not isinstance(target, nn.Linear):
            continue
        parent_name = ".".join(key.split(".")[:-1])
        child_name = key.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        new_module = TeamLoRALinear(
            target.in_features,
            target.out_features,
            r=r,
            lora_alpha=alpha,
            lora_num=lora_num,
            lora_dropout=dropout,
            bias=target.bias is not None,
        )
        new_module.weight = target.weight
        if target.bias is not None:
            new_module.bias = target.bias
        new_module.to(target.weight.device, dtype=target.weight.dtype)
        setattr(parent, child_name, new_module)
        replaced += 1
    if replaced == 0:
        raise ValueError(f"No target modules replaced. Targets={TARGET_MODULES}")
    return replaced


class LlamaTeamLoRAPOIClassifier(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_pois: int,
        classifier_dropout: float,
        pooling: str,
        graph_prior_alpha: float,
        graph_prior_mode: str,
        mlp_logit_scale: float,
        graph_noise_loss_weight: float,
        raat_mode: str,
    ):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.graph_prior_alpha = float(graph_prior_alpha)
        self.graph_prior_mode = graph_prior_mode
        self.mlp_logit_scale = float(mlp_logit_scale)
        self.graph_noise_loss_weight = float(graph_noise_loss_weight)
        self.raat_mode = raat_mode
        classifier_input_size = hidden_size * 2 if pooling == "last_mean" else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, num_pois),
        )
        self.graph_noise_head = nn.Linear(classifier_input_size, 4)
        # Prevent Trainer from wrapping the whole 1B model in DataParallel and
        # replicating it onto smaller/busy GPUs. Use CUDA_VISIBLE_DEVICES or
        # torchrun/deepspeed explicitly for multi-GPU training.
        self.is_parallelizable = True
        self.model_parallel = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Dict[str, Any] | None = None) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.base_model.gradient_checkpointing_enable()
            else:
                self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        graph_candidate_indices: torch.Tensor | None = None,
        graph_candidate_ranks: torch.Tensor | None = None,
        graph_candidate_scores: torch.Tensor | None = None,
        graph_candidate_mask: torch.Tensor | None = None,
        raat_graph_candidate_indices: torch.Tensor | None = None,
        raat_graph_candidate_ranks: torch.Tensor | None = None,
        raat_graph_candidate_scores: torch.Tensor | None = None,
        raat_graph_candidate_mask: torch.Tensor | None = None,
        graph_noise_labels: torch.Tensor | None = None,
        **_: Any,
    ):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        last_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        last_pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        if self.pooling == "last":
            pooled = last_pooled
        else:
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = mean_pooled if self.pooling == "mean" else torch.cat([last_pooled, mean_pooled], dim=-1)
        mlp_logits = self.classifier(pooled)
        logits = mlp_logits * self.mlp_logit_scale
        if self.graph_prior_alpha != 0 and self.graph_prior_mode != "none" and graph_candidate_indices is not None:
            logits = logits + self.build_graph_prior(
                logits,
                graph_candidate_indices,
                graph_candidate_ranks,
                graph_candidate_scores,
                graph_candidate_mask,
            )
        loss = None
        if labels is not None:
            if self.raat_mode == "target_mask_2view" and raat_graph_candidate_indices is not None:
                raat_logits = mlp_logits * self.mlp_logit_scale
                if self.graph_prior_alpha != 0 and self.graph_prior_mode != "none":
                    raat_logits = raat_logits + self.build_graph_prior(
                        raat_logits,
                        raat_graph_candidate_indices,
                        raat_graph_candidate_ranks,
                        raat_graph_candidate_scores,
                        raat_graph_candidate_mask,
                    )
                clean_loss = F.cross_entropy(logits.float(), labels, reduction="none")
                raat_loss = F.cross_entropy(raat_logits.float(), labels, reduction="none")
                loss = torch.maximum(clean_loss, raat_loss).mean()
            else:
                loss = F.cross_entropy(logits.float(), labels)
        graph_noise_logits = self.graph_noise_head(pooled)
        if loss is not None and self.graph_noise_loss_weight > 0 and graph_noise_labels is not None:
            noise_loss = F.cross_entropy(graph_noise_logits.float(), graph_noise_labels.to(graph_noise_logits.device))
            loss = loss + self.graph_noise_loss_weight * noise_loss
        return {"loss": loss, "logits": logits}

    def build_graph_prior(
        self,
        logits: torch.Tensor,
        graph_candidate_indices: torch.Tensor,
        graph_candidate_ranks: torch.Tensor | None,
        graph_candidate_scores: torch.Tensor | None,
        graph_candidate_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        indices = graph_candidate_indices.to(logits.device)
        valid = indices.ge(0)
        if graph_candidate_mask is not None:
            valid = valid & graph_candidate_mask.to(logits.device)
        safe_indices = indices.clamp(min=0)
        if self.graph_prior_mode == "score" and graph_candidate_scores is not None:
            raw = graph_candidate_scores.to(logits.device, dtype=logits.dtype)
            raw = raw.masked_fill(~valid, 0)
            denom = raw.max(dim=1, keepdim=True).values.clamp(min=1.0)
            values = raw / denom
        else:
            ranks = graph_candidate_ranks.to(logits.device, dtype=logits.dtype) if graph_candidate_ranks is not None else torch.ones_like(safe_indices, dtype=logits.dtype)
            values = 1.0 / ranks.clamp(min=1.0).log1p()
            values = values.masked_fill(~valid, 0)
        prior = torch.zeros_like(logits)
        scatter_values = (values * self.graph_prior_alpha).to(dtype=prior.dtype)
        prior.scatter_add_(1, safe_indices, scatter_values)
        return prior

    def save_trainable(self, output_dir: Path, metadata: Dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad}
        torch.save(state, output_dir / "trainable_model.bin")
        (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class TrainableOnlyTrainer(Trainer):
    def __init__(self, *args: Any, save_metadata: Dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_metadata = save_metadata or {}

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        out = Path(output_dir or self.args.output_dir)
        if hasattr(self.model, "save_trainable"):
            self.model.save_trainable(out, self.save_metadata)
        processing_class = getattr(self, "processing_class", None)
        tokenizer = getattr(self, "tokenizer", None)
        if processing_class is not None:
            processing_class.save_pretrained(out)
        elif tokenizer is not None:
            tokenizer.save_pretrained(out)


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
        "label_names": ["labels", "graph_noise_labels"],
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "steps"
    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = "steps"
    if "lr_scheduler_type" in sig.parameters:
        kwargs["lr_scheduler_type"] = "cosine"
    if "optim" in sig.parameters:
        kwargs["optim"] = "adamw_torch"
    return TrainingArguments(**kwargs)


def make_metrics_fn():
    def compute_metrics(eval_pred):
        logits = np.asarray(eval_pred.predictions)
        label_ids = eval_pred.label_ids
        if isinstance(label_ids, (tuple, list)):
            label_ids = label_ids[0]
        labels = np.asarray(label_ids).astype(np.int64)
        if labels.ndim > 1 and labels.shape[0] <= 4:
            labels = labels[0]
        labels = labels.reshape(-1)
        if logits.ndim > 2:
            logits = logits.reshape(-1, logits.shape[-1])
        target_scores = logits[np.arange(len(labels)), labels]
        ranks = 1 + (logits > target_scores[:, None]).sum(axis=1)
        return {
            "top1": float(np.mean(ranks <= 1)),
            "top5": float(np.mean(ranks <= 5)),
            "top10": float(np.mean(ranks <= 10)),
            "top20": float(np.mean(ranks <= 20)),
            "mrr": float(np.mean(1.0 / ranks)),
        }

    return compute_metrics


def print_trainable_parameters(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({"trainable_params": trainable, "total_params": total, "trainable_ratio": trainable / total}, ensure_ascii=False))


def keep_trainable_params_fp32(model: nn.Module) -> None:
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def parse_cutoffs(value: str) -> List[int]:
    if not value.strip():
        return []
    cutoffs: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        cutoff = int(item)
        if cutoff < 0:
            raise ValueError(f"graph prior cutoff must be non-negative, got {cutoff}")
        cutoffs.append(cutoff)
    return cutoffs


def main() -> None:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args = parse_args()
    if args.raat_mode != "none":
        noisy_graph_args = {
            "graph_prior_dropout": args.graph_prior_dropout,
            "graph_prior_random_cutoffs": args.graph_prior_random_cutoffs,
            "graph_candidate_drop_prob": args.graph_candidate_drop_prob,
            "graph_target_mask_prob": args.graph_target_mask_prob,
            "graph_irrelevant_mix_prob": args.graph_irrelevant_mix_prob,
            "graph_rank_noise_prob": args.graph_rank_noise_prob,
        }
        active_noisy_args = {key: value for key, value in noisy_graph_args.items() if value not in (0, 0.0, "", None)}
        if active_noisy_args:
            print(
                json.dumps(
                    {
                        "warning": "RAAT mode is enabled; legacy random graph noise args are also active.",
                        "active_legacy_noise_args": active_noisy_args,
                    },
                    ensure_ascii=False,
                )
            )
    graph_prior_random_cutoffs = parse_cutoffs(args.graph_prior_random_cutoffs)

    semantic_map = load_semantic_map(args.semantic_map)
    if args.train_joined.exists() and args.val_joined.exists():
        raw_train_joined = load_joined_rows(args.train_joined)
        raw_val_joined = load_joined_rows(args.val_joined)
        poi2idx, idx2poi = build_poi_vocab(semantic_map, joined_rows=[*raw_train_joined, *raw_val_joined])
        train_rows, train_stats = prepare_joined_rows(raw_train_joined, poi2idx)
        val_rows, val_stats = prepare_joined_rows(raw_val_joined, poi2idx)
        train_stats["source"] = str(args.train_joined)
        val_stats["source"] = str(args.val_joined)
    else:
        poi2idx, idx2poi = build_poi_vocab(semantic_map, raw_paths=[args.train_raw, args.val_raw])
        train_rows, train_stats = build_joined_rows(
            args.train_raw, args.train_refined, args.train_refined_fallback, args.train_graph, semantic_map, poi2idx, args.graph_top_k
        )
        val_rows, val_stats = build_joined_rows(
            args.val_raw, args.val_refined, args.val_refined_fallback, args.val_graph, semantic_map, poi2idx, args.graph_top_k
        )
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
        train_stats["limited_rows"] = len(train_rows)
    if args.max_val_samples is not None:
        val_rows = val_rows[: args.max_val_samples]
        val_stats["limited_rows"] = len(val_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = {
        "num_pois": len(poi2idx),
        "graph_top_k": args.graph_top_k,
        "train": train_stats,
        "val": val_stats,
    }
    (args.output_dir / "data_summary.json").write_text(json.dumps(data_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "poi_vocab.json").write_text(
        json.dumps({"poi2idx": poi2idx, "idx2poi": {str(k): v for k, v in idx2poi.items()}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(data_summary, ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModel.from_pretrained(args.base_model, **model_kwargs)
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
    for param in base_model.parameters():
        param.requires_grad = False
    replaced = inject_teamlora(base_model, args.lora_r, args.lora_alpha, args.lora_num, args.lora_dropout)
    model = LlamaTeamLoRAPOIClassifier(
        base_model=base_model,
        hidden_size=int(base_model.config.hidden_size),
        num_pois=len(poi2idx),
        classifier_dropout=args.classifier_dropout,
        pooling=args.pooling,
        graph_prior_alpha=args.graph_prior_alpha,
        graph_prior_mode=args.graph_prior_mode,
        mlp_logit_scale=args.mlp_logit_scale,
        graph_noise_loss_weight=args.graph_noise_loss_weight,
        raat_mode=args.raat_mode,
    )
    keep_trainable_params_fp32(model)
    print(json.dumps({"teamlora_replaced_modules": replaced}, ensure_ascii=False))
    print_trainable_parameters(model)

    train_ds = POIClassificationDataset(
        train_rows,
        tokenizer,
        args.max_length,
        all_poi_indices=list(range(len(poi2idx))),
        graph_prior_dropout=args.graph_prior_dropout,
        graph_prior_random_cutoffs=graph_prior_random_cutoffs,
        graph_candidate_drop_prob=args.graph_candidate_drop_prob,
        graph_target_mask_prob=args.graph_target_mask_prob,
        graph_irrelevant_mix_prob=args.graph_irrelevant_mix_prob,
        graph_rank_noise_prob=args.graph_rank_noise_prob,
    )
    val_ds = POIClassificationDataset(val_rows, tokenizer, args.max_length)
    collator = DataCollatorForPOIClassification(tokenizer, raat_mode=args.raat_mode)
    save_metadata = {
        "base_model": str(args.base_model),
        "num_pois": len(poi2idx),
        "target_modules": TARGET_MODULES,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_num": args.lora_num,
        "lora_dropout": args.lora_dropout,
        "classifier_dropout": args.classifier_dropout,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "graph_top_k": args.graph_top_k,
        "graph_prior_alpha": args.graph_prior_alpha,
        "graph_prior_mode": args.graph_prior_mode,
        "mlp_logit_scale": args.mlp_logit_scale,
        "graph_prior_dropout": args.graph_prior_dropout,
        "graph_prior_random_cutoffs": graph_prior_random_cutoffs,
        "graph_candidate_drop_prob": args.graph_candidate_drop_prob,
        "graph_target_mask_prob": args.graph_target_mask_prob,
        "graph_irrelevant_mix_prob": args.graph_irrelevant_mix_prob,
        "graph_rank_noise_prob": args.graph_rank_noise_prob,
        "graph_noise_loss_weight": args.graph_noise_loss_weight,
        "raat_mode": args.raat_mode,
    }
    trainer = TrainableOnlyTrainer(
        **{
            "model": model,
            "args": make_training_args(args),
            "train_dataset": train_ds,
            "eval_dataset": val_ds,
            "data_collator": collator,
            "compute_metrics": make_metrics_fn(),
            "save_metadata": save_metadata,
            ("processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"): tokenizer,
        }
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
