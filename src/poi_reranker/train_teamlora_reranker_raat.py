#!/usr/bin/env python3
"""Train a 3-expert TeamLoRA candidate-wise POI reranker with 2-view RAAT.

The model ranks GraphRAG TopK candidates. Training samples a small candidate
list per query; evaluation ranks the full candidate list from each group.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from build_teamlora_reranker_groups import build_group, load_semantic_map


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXPERTS = ("pref", "graph", "refine")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 3-expert TeamLoRA candidate-wise reranker with 2-view RAAT.")
    p.add_argument("--train-groups", type=Path, default=None, help="Optional prebuilt grouped TopK JSONL.")
    p.add_argument("--val-groups", type=Path, default=None, help="Optional prebuilt grouped TopK JSONL.")
    p.add_argument("--train-joined", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet"))
    p.add_argument("--val-joined", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet"))
    p.add_argument("--semantic-map", type=Path, default=Path("retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl"))
    p.add_argument("--base-model", type=Path, default=Path("/mnt/data/yyl/TMP/models/Llama-3.2-1B-Instruct"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--expert-mode", choices=["anonymous", "named"], default="anonymous")
    p.add_argument("--input-template", choices=["legacy", "semantic_profile_v1"], default="legacy")
    p.add_argument("--train-negatives", type=int, default=31)
    p.add_argument("--hard-negatives", type=int, default=16, help="Prefer negatives from top ranks.")
    p.add_argument("--eval-top-k", type=int, default=100)
    p.add_argument("--train-hit-only", action=argparse.BooleanOptionalAction, default=True, help="Train only groups whose TopK contains the target.")
    p.add_argument("--val-hit-only", action=argparse.BooleanOptionalAction, default=False, help="Evaluate only val groups whose TopK contains the target.")
    p.add_argument("--batch-groups", type=int, default=1, help="Number of query groups per batch.")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--scorer-dropout", type=float, default=0.1)
    p.add_argument("--graph-feature-dim", type=int, default=32)
    p.add_argument("--use-graph-prior", action="store_true", help="Use GraphRAG rank prior and train TeamLoRA as residual correction.")
    p.add_argument("--graph-prior-type", choices=["rank_log", "rank_invlog"], default="rank_log")
    p.add_argument("--residual-alpha-init", type=float, default=0.1)
    p.add_argument("--residual-l2", type=float, default=0.0, help="Optional residual score L2 penalty.")
    p.add_argument("--residual-bound-mode", choices=["none", "tanh", "clamp"], default="none")
    p.add_argument("--residual-bound-value", type=float, default=0.3)
    p.add_argument("--raat-mode", choices=["none", "target_mask_2view", "graph_3view"], default="target_mask_2view")
    p.add_argument("--target-demote-rank-min", type=int, default=50)
    p.add_argument("--target-demote-score-scale", type=float, default=0.2)
    p.add_argument("--hardneg-promote-topn", type=int, default=3)
    p.add_argument("--hardneg-score-boost", type=float, default=2.0)
    p.add_argument("--source-dropout-prob", type=float, default=0.0)
    p.add_argument("--rank-noise-prob", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-groups", type=int, default=None)
    p.add_argument("--max-val-groups", type=int, default=None)
    p.add_argument("--eval-candidate-limit", type=int, default=None, help="Optional fast eval limit per group. Final Top100 reports should leave this unset.")
    p.add_argument("--eval-candidate-batch-size", type=int, default=None, help="Score eval candidates in chunks to reduce peak memory.")
    p.add_argument("--eval-final-only", action="store_true", help="Skip intermediate eval and evaluate only at final step.")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_bits(sources: Sequence[str]) -> List[float]:
    text = " ".join(str(x) for x in sources).lower()
    return [
        float("transition" in text or "last_poi" in text or "last_category" in text),
        float("geo" in text or "near" in text),
        float("history" in text or "user" in text or "legacy" in text),
        float("semantic" in text or "edge" in text),
    ]


def graph_features(
    candidate: Dict[str, Any],
    masked: bool = False,
    rank_override: float | None = None,
    score_scale: float = 1.0,
    score_boost: float = 1.0,
) -> List[float]:
    rank = max(1.0, float(rank_override if rank_override is not None else candidate.get("rank") or 999.0))
    score = max(0.0, float(candidate.get("score") or 0.0) * float(score_scale) * float(score_boost))
    bits = [0.0, 0.0, 0.0, 0.0] if masked else source_bits(candidate.get("sources") or [])
    return [
        1.0 / math.log1p(rank),
        math.log1p(score) / 10.0,
        min(rank, 100.0) / 100.0,
        float(masked),
        *bits,
    ]


def format_pref_text(group: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "[EXPERT=PREF]",
            "Judge whether this candidate matches the user's trajectory, temporal pattern, and historical preference.",
            str(group.get("pref_text") or ""),
            str(candidate.get("candidate_text") or ""),
        ]
    )


def format_graph_text(group: Dict[str, Any], candidate: Dict[str, Any], masked: bool = False) -> str:
    graph_line = "Graph evidence is target-masked for adversarial robustness." if masked else str(candidate.get("candidate_text") or "")
    return "\n\n".join(
        [
            "[EXPERT=GRAPH]",
            "Judge whether graph retrieval evidence supports this candidate.",
            graph_line,
            "Full candidate context:",
            str(group.get("graph_text") or ""),
        ]
    )


def format_refine_text(group: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "[EXPERT=REFINE]",
            "Judge whether the refined evidence points to this candidate.",
            str(group.get("refine_text") or ""),
            str(candidate.get("candidate_text") or ""),
        ]
    )


def strip_preference_section(text: Any) -> str:
    lines = str(text or "").splitlines()
    out: List[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Preference:":
            skipping = True
            continue
        if skipping and stripped in {"Trajectory:", "Transitions:", "Nearby:"}:
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).strip()


def format_semantic_profile_text(group: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    hypothesis = str(candidate.get("candidate_hypothesis_text") or "").strip()
    if not hypothesis:
        hypothesis = (
            f"id={candidate.get('poi_id')}; semantic_id={candidate.get('semantic_id')}; "
            f"category={candidate.get('category')}; geo_cell={candidate.get('geo_cell')}"
        )
    profile = str(group.get("user_semantic_profile") or "").strip()
    if not profile:
        profile = "Long-term category affinity:\n- none\n\nRevisit affinity:\n- none\n\nGeo routine:\n- none\n\nTemporal routine:\n- none_available"
    return "\n\n".join(
        [
            "[TASK=CANDIDATE_RERANK]",
            "Estimate whether the POI hypothesis matches the user's next check-in.",
            "[RECENT_CONTEXT]",
            strip_preference_section(group.get("pref_text")),
            "[USER_SEMANTIC_PROFILE]",
            profile,
            "[POI_HYPOTHESIS]",
            hypothesis,
        ]
    )


def format_anonymous_text(
    group: Dict[str, Any],
    candidate: Dict[str, Any],
    view: str = "clean",
    input_template: str = "legacy",
) -> str:
    if input_template == "semantic_profile_v1":
        return format_semantic_profile_text(group, candidate)
    if view == "target_mask":
        graph_evidence = "Graph evidence is target-masked for adversarial robustness."
    elif view == "target_demote":
        graph_evidence = f"Graph evidence is demoted for robustness. Candidate identity: {candidate.get('poi_id')}; semantic_id={candidate.get('semantic_id')}; category={candidate.get('category')}; geo={candidate.get('geo_cell')}."
    elif view == "hardneg_promote":
        graph_evidence = f"Graph evidence is stress-tested with promoted hard negatives. {candidate.get('candidate_text') or ''}"
    else:
        graph_evidence = str(candidate.get("candidate_text") or "")
    return "\n\n".join(
        [
            "[TASK=CANDIDATE_RERANK]",
            "Score whether the candidate POI is the next location.",
            "[QUERY]",
            str(group.get("pref_text") or ""),
            "[REFINED_EVIDENCE]",
            str(group.get("refine_text") or ""),
            "[CANDIDATE]",
            graph_evidence,
        ]
    )


@dataclass
class RerankerGroupDataset(Dataset):
    groups: List[Dict[str, Any]]
    train: bool
    train_negatives: int
    hard_negatives: int
    expert_mode: str = "anonymous"
    target_demote_rank_min: int = 50
    target_demote_score_scale: float = 0.2
    hardneg_promote_topn: int = 3
    hardneg_score_boost: float = 2.0
    eval_candidate_limit: int | None = None
    input_template: str = "legacy"

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self.groups[idx]
        candidates = list(group.get("candidates") or [])
        pos = [c for c in candidates if int(c.get("label") or 0) == 1]
        neg = [c for c in candidates if int(c.get("label") or 0) != 1]
        if self.train:
            selected: List[Dict[str, Any]] = []
            if pos:
                selected.append(pos[0])
            neg_n = max(1, self.train_negatives)
            hard_pool = neg[: max(self.hard_negatives, 0)]
            sampled: List[Dict[str, Any]] = []
            if hard_pool:
                sampled.extend(random.sample(hard_pool, k=min(len(hard_pool), min(neg_n, len(hard_pool)))))
            remaining = [c for c in neg if c not in sampled]
            if len(sampled) < neg_n and remaining:
                sampled.extend(random.sample(remaining, k=min(len(remaining), neg_n - len(sampled))))
            selected.extend(sampled)
            random.shuffle(selected)
        else:
            selected = candidates
            if self.eval_candidate_limit is not None:
                selected = selected[: max(1, int(self.eval_candidate_limit))]
        labels = [int(c.get("label") or 0) for c in selected]
        if not any(labels):
            # No positive in TopK. Keep the group for eval denominator, but skip
            # training by marking target position unavailable.
            target_pos = -1
        else:
            target_pos = labels.index(1)
        hardneg_ids = {
            c.get("poi_id")
            for c in sorted(
                [c for c in selected if int(c.get("label") or 0) != 1],
                key=lambda item: float(item.get("rank") or 999.0),
            )[: max(0, self.hardneg_promote_topn)]
        }
        if self.expert_mode == "named":
            pref_texts = [format_pref_text(group, c) for c in selected]
            graph_texts = [format_graph_text(group, c, masked=False) for c in selected]
            graph_texts_masked = [format_graph_text(group, c, masked=(int(c.get("label") or 0) == 1)) for c in selected]
            graph_texts_demote = [format_graph_text(group, c, masked=(int(c.get("label") or 0) == 1)) for c in selected]
            graph_texts_hardneg = [format_graph_text(group, c, masked=False) for c in selected]
            refine_texts = [format_refine_text(group, c) for c in selected]
        else:
            pref_texts = [format_anonymous_text(group, c, view="clean", input_template=self.input_template) for c in selected]
            graph_texts = [format_anonymous_text(group, c, view="clean", input_template=self.input_template) for c in selected]
            graph_texts_masked = [
                format_anonymous_text(
                    group,
                    c,
                    view="target_mask" if int(c.get("label") or 0) == 1 else "clean",
                    input_template=self.input_template,
                )
                for c in selected
            ]
            graph_texts_demote = [
                format_anonymous_text(
                    group,
                    c,
                    view="target_demote" if int(c.get("label") or 0) == 1 else "clean",
                    input_template=self.input_template,
                )
                for c in selected
            ]
            graph_texts_hardneg = [
                format_anonymous_text(
                    group,
                    c,
                    view="hardneg_promote" if c.get("poi_id") in hardneg_ids else "clean",
                    input_template=self.input_template,
                )
                for c in selected
            ]
            refine_texts = [format_anonymous_text(group, c, view="clean", input_template=self.input_template) for c in selected]
        demote_features = []
        hardneg_features = []
        for candidate in selected:
            is_target = int(candidate.get("label") or 0) == 1
            is_hardneg = candidate.get("poi_id") in hardneg_ids
            demote_features.append(
                graph_features(
                    candidate,
                    masked=is_target,
                    rank_override=max(float(candidate.get("rank") or 999.0), float(self.target_demote_rank_min)) if is_target else None,
                    score_scale=self.target_demote_score_scale if is_target else 1.0,
                )
            )
            hardneg_features.append(
                graph_features(
                    candidate,
                    masked=False,
                    rank_override=1.0 if is_hardneg else None,
                    score_boost=self.hardneg_score_boost if is_hardneg else 1.0,
                )
            )
        return {
            "sample_id": group.get("sample_id"),
            "target_pos": target_pos,
            "target_in_candidates": bool(group.get("target_in_candidates")),
            "pref_texts": pref_texts,
            "graph_texts": graph_texts,
            "graph_texts_masked": graph_texts_masked,
            "graph_texts_demote": graph_texts_demote,
            "graph_texts_hardneg": graph_texts_hardneg,
            "refine_texts": refine_texts,
            "graph_features": [graph_features(c, masked=False) for c in selected],
            "graph_features_masked": [graph_features(c, masked=(int(c.get("label") or 0) == 1)) for c in selected],
            "graph_features_demote": demote_features,
            "graph_features_hardneg": hardneg_features,
            "candidate_poi_ids": [c.get("poi_id") for c in selected],
        }


def collate_groups(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"groups": list(features)}


class MultiExpertLoRALinear(nn.Linear):
    def __init__(self, source: nn.Linear, r: int, alpha: int, dropout: float, experts: Sequence[str]):
        super().__init__(source.in_features, source.out_features, bias=source.bias is not None)
        self.weight = source.weight
        if source.bias is not None:
            self.bias = source.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.active_expert = experts[0]
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.ModuleDict({name: nn.Linear(source.in_features, r, bias=False) for name in experts})
        self.lora_B = nn.ModuleDict({name: nn.Linear(r, source.out_features, bias=False) for name in experts})
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        for name in self.lora_A:
            nn.init.kaiming_uniform_(self.lora_A[name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name].weight)

    def set_active_expert(self, name: str) -> None:
        self.active_expert = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        lora_a = self.lora_A[self.active_expert]
        lora_b = self.lora_B[self.active_expert]
        lora_input = x.to(dtype=lora_a.weight.dtype)
        a = self.dropout(lora_a(lora_input))
        delta = lora_b(a) * self.scaling
        return result + delta.to(dtype=result.dtype)


def inject_multi_expert_lora(model: nn.Module, r: int, alpha: int, dropout: float) -> int:
    replaced = 0
    for key in list(dict(model.named_modules()).keys()):
        if not any(key.endswith(target) for target in TARGET_MODULES):
            continue
        target = model.get_submodule(key)
        if not isinstance(target, nn.Linear):
            continue
        parent_name = ".".join(key.split(".")[:-1])
        child_name = key.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        module = MultiExpertLoRALinear(target, r=r, alpha=alpha, dropout=dropout, experts=EXPERTS)
        module.to(target.weight.device, dtype=target.weight.dtype)
        setattr(parent, child_name, module)
        replaced += 1
    if replaced == 0:
        raise ValueError(f"No LoRA target modules replaced: {TARGET_MODULES}")
    return replaced


def set_active_expert(model: nn.Module, expert: str) -> None:
    for module in model.modules():
        if isinstance(module, MultiExpertLoRALinear):
            module.set_active_expert(expert)


class TeamLoRAReranker(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        graph_feature_size: int,
        graph_feature_dim: int,
        dropout: float,
        use_graph_prior: bool = False,
        graph_prior_type: str = "rank_log",
        residual_alpha_init: float = 0.1,
        residual_bound_mode: str = "none",
        residual_bound_value: float = 0.3,
    ):
        super().__init__()
        self.base_model = base_model
        self.use_graph_prior = bool(use_graph_prior)
        self.graph_prior_type = str(graph_prior_type)
        self.residual_bound_mode = str(residual_bound_mode)
        self.residual_bound_value = float(residual_bound_value)
        if self.use_graph_prior:
            init = max(float(residual_alpha_init), 1e-6)
            self.residual_alpha_param = nn.Parameter(torch.tensor(math.log(math.expm1(init)), dtype=torch.float32))
        self.graph_feature_proj = nn.Sequential(
            nn.Linear(graph_feature_size, graph_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3 + graph_feature_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size + graph_feature_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, expert: str) -> torch.Tensor:
        set_active_expert(self.base_model, expert)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        last_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

    def forward(
        self,
        pref_batch: Dict[str, torch.Tensor],
        graph_batch: Dict[str, torch.Tensor],
        refine_batch: Dict[str, torch.Tensor],
        graph_features_tensor: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pref = self.encode(pref_batch["input_ids"], pref_batch["attention_mask"], "pref")
        h_graph = self.encode(graph_batch["input_ids"], graph_batch["attention_mask"], "graph")
        h_refine = self.encode(refine_batch["input_ids"], refine_batch["attention_mask"], "refine")
        gf = self.graph_feature_proj(graph_features_tensor.to(h_pref.device, dtype=torch.float32))
        h_pref = h_pref.float()
        h_graph = h_graph.float()
        h_refine = h_refine.float()
        gate_logits = self.gate(torch.cat([h_pref, h_graph, h_refine, gf], dim=-1))
        gate = F.softmax(gate_logits, dim=-1)
        stacked = torch.stack([h_pref, h_graph, h_refine], dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        residual = self.scorer(torch.cat([fused, gf], dim=-1)).squeeze(-1)
        if not self.use_graph_prior:
            return residual
        prior = self.graph_prior(graph_features_tensor).to(residual.device, dtype=residual.dtype)
        alpha = F.softplus(self.residual_alpha_param).to(residual.device, dtype=residual.dtype)
        correction = self.bound_residual_correction(alpha * residual)
        scores = prior + correction
        if return_parts:
            return scores, residual, prior, alpha
        return scores

    def bound_residual_correction(self, correction: torch.Tensor) -> torch.Tensor:
        if self.residual_bound_mode == "none":
            return correction
        bound = max(float(self.residual_bound_value), 1e-6)
        if self.residual_bound_mode == "tanh":
            return correction.new_tensor(bound) * torch.tanh(correction / bound)
        if self.residual_bound_mode == "clamp":
            return correction.clamp(min=-bound, max=bound)
        raise ValueError(f"Unsupported residual_bound_mode: {self.residual_bound_mode}")

    def graph_prior(self, graph_features_tensor: torch.Tensor) -> torch.Tensor:
        features = graph_features_tensor.float()
        if self.graph_prior_type == "rank_invlog":
            return features[:, 0]
        rank_norm = features[:, 2].clamp(min=0.01, max=1.0)
        rank = rank_norm * 100.0
        return -torch.log(rank.clamp(min=1.0))

    def save_trainable(self, output_dir: Path, metadata: Dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad}
        torch.save(state, output_dir / "trainable_model.bin")
        (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def flatten_group_texts(groups: Sequence[Dict[str, Any]], key: str) -> tuple[List[str], List[int]]:
    texts: List[str] = []
    sizes: List[int] = []
    for group in groups:
        values = list(group[key])
        texts.extend(values)
        sizes.append(len(values))
    return texts, sizes


def encode_texts(tokenizer: Any, texts: Sequence[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def group_loss(scores: torch.Tensor, sizes: Sequence[int], target_positions: Sequence[int]) -> torch.Tensor | None:
    losses: List[torch.Tensor] = []
    offset = 0
    for size, target_pos in zip(sizes, target_positions):
        group_scores = scores[offset : offset + size]
        offset += size
        if target_pos < 0:
            continue
        losses.append(F.cross_entropy(group_scores.view(1, -1).float(), torch.tensor([target_pos], device=scores.device)))
    if not losses:
        return None
    return torch.stack(losses).mean()


def make_batch(
    tokenizer: Any,
    groups: Sequence[Dict[str, Any]],
    max_length: int,
    device: torch.device,
    masked: bool = False,
    view: str = "clean",
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[int], List[int]]:
    pref_texts, sizes = flatten_group_texts(groups, "pref_texts")
    if view == "target_mask" or masked:
        graph_key = "graph_texts_masked"
        feat_key = "graph_features_masked"
    elif view == "target_demote":
        graph_key = "graph_texts_demote"
        feat_key = "graph_features_demote"
    elif view == "hardneg_promote":
        graph_key = "graph_texts_hardneg"
        feat_key = "graph_features_hardneg"
    else:
        graph_key = "graph_texts"
        feat_key = "graph_features"
    graph_texts, _ = flatten_group_texts(groups, graph_key)
    refine_texts, _ = flatten_group_texts(groups, "refine_texts")
    graph_features_flat = [feat for group in groups for feat in group[feat_key]]
    target_positions = [int(group["target_pos"]) for group in groups]
    return (
        encode_texts(tokenizer, pref_texts, max_length, device),
        encode_texts(tokenizer, graph_texts, max_length, device),
        encode_texts(tokenizer, refine_texts, max_length, device),
        torch.tensor(graph_features_flat, dtype=torch.float32, device=device),
        sizes,
        target_positions,
    )


def slice_eval_group(group: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    sliced = dict(group)
    keys = [
        "pref_texts",
        "graph_texts",
        "graph_texts_masked",
        "graph_texts_demote",
        "graph_texts_hardneg",
        "refine_texts",
        "graph_features",
        "graph_features_masked",
        "graph_features_demote",
        "graph_features_hardneg",
        "candidate_poi_ids",
    ]
    for key in keys:
        if key in group:
            sliced[key] = list(group[key])[start:end]
    sliced["target_pos"] = -1
    return sliced


def score_eval_group(
    model: TeamLoRAReranker,
    tokenizer: Any,
    group: Dict[str, Any],
    max_length: int,
    device: torch.device,
    candidate_batch_size: int | None,
) -> torch.Tensor:
    size = len(group["pref_texts"])
    if candidate_batch_size is None or candidate_batch_size <= 0 or candidate_batch_size >= size:
        pref, graph, refine, gf, _, _ = make_batch(tokenizer, [group], max_length, device, masked=False)
        return model(pref, graph, refine, gf).detach()

    chunks: List[torch.Tensor] = []
    for start in range(0, size, candidate_batch_size):
        chunk_group = slice_eval_group(group, start, min(size, start + candidate_batch_size))
        pref, graph, refine, gf, _, _ = make_batch(tokenizer, [chunk_group], max_length, device, masked=False)
        chunks.append(model(pref, graph, refine, gf).detach())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def evaluate(model: TeamLoRAReranker, tokenizer: Any, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    ranks: List[int] = []
    missing = 0
    candidate_batch_size = getattr(args, "eval_candidate_batch_size", None)
    for batch in tqdm(loader, desc="eval", leave=False):
        groups = batch["groups"]
        for group in groups:
            target_pos = int(group["target_pos"])
            if target_pos < 0:
                missing += 1
                continue
            group_scores = score_eval_group(model, tokenizer, group, args.max_length, device, candidate_batch_size)
            target_score = group_scores[target_pos]
            rank = int(1 + (group_scores > target_score).sum().item())
            ranks.append(rank)
    total = len(ranks) + missing
    if not ranks:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "top20": 0.0,
            "mrr": 0.0,
            "conditional_top1": 0.0,
            "conditional_top5": 0.0,
            "conditional_top10": 0.0,
            "conditional_top20": 0.0,
            "conditional_mrr": 0.0,
            "candidate_hit": 0.0,
            "evaluated": float(total),
            "missed": float(missing),
        }
    arr = np.asarray(ranks)
    candidate_hit = float(len(ranks) / total) if total else 0.0
    conditional = {
        "conditional_top1": float(np.mean(arr <= 1)),
        "conditional_top5": float(np.mean(arr <= 5)),
        "conditional_top10": float(np.mean(arr <= 10)),
        "conditional_top20": float(np.mean(arr <= 20)),
        "conditional_mrr": float(np.mean(1.0 / arr)),
    }
    metrics = {
        "top1": conditional["conditional_top1"] * candidate_hit,
        "top5": conditional["conditional_top5"] * candidate_hit,
        "top10": conditional["conditional_top10"] * candidate_hit,
        "top20": conditional["conditional_top20"] * candidate_hit,
        "mrr": conditional["conditional_mrr"] * candidate_hit,
        **conditional,
        "candidate_hit": candidate_hit,
        "evaluated": float(total),
        "missed": float(missing),
    }
    if getattr(model, "use_graph_prior", False):
        metrics["residual_alpha"] = float(F.softplus(model.residual_alpha_param).detach().cpu().item())
        metrics["graph_prior_type"] = str(model.graph_prior_type)
        metrics["residual_bound_mode"] = str(model.residual_bound_mode)
        metrics["residual_bound_value"] = float(model.residual_bound_value)
    return metrics


def load_groups(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    groups = list(read_jsonl(path))
    if limit is not None:
        groups = groups[:limit]
    return groups


def load_groups_from_joined(path: Path, semantic_map_path: Path, top_k: int, limit: int | None = None) -> List[Dict[str, Any]]:
    path = Path(path)
    semantic_map_path = Path(semantic_map_path)
    semantic_map = load_semantic_map(semantic_map_path)
    table = pq.read_table(path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    groups: List[Dict[str, Any]] = []
    for row in rows:
        group = build_group(row, semantic_map, top_k)
        if group is not None:
            groups.append(group)
    return groups


def main() -> None:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.train_groups is not None:
        train_groups = load_groups(args.train_groups, args.max_train_groups)
    else:
        train_groups = load_groups_from_joined(args.train_joined, args.semantic_map, args.top_k, args.max_train_groups)
    if args.val_groups is not None:
        val_groups = load_groups(args.val_groups, args.max_val_groups)
    else:
        val_groups = load_groups_from_joined(args.val_joined, args.semantic_map, args.top_k, args.max_val_groups)
    raw_train_groups = len(train_groups)
    if args.train_hit_only:
        train_groups = [group for group in train_groups if group.get("target_in_candidates")]
    raw_val_groups = len(val_groups)
    if args.val_hit_only:
        val_groups = [group for group in val_groups if group.get("target_in_candidates")]
    data_summary = {
        "raw_train_groups": raw_train_groups,
        "raw_val_groups": raw_val_groups,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_hit": sum(1 for x in train_groups if x.get("target_in_candidates")) / max(1, len(train_groups)),
        "val_hit": sum(1 for x in val_groups if x.get("target_in_candidates")) / max(1, len(val_groups)),
    }
    (args.output_dir / "data_summary.json").write_text(json.dumps(data_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data_summary, ensure_ascii=False, indent=2))

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
    replaced = inject_multi_expert_lora(base_model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model = TeamLoRAReranker(
        base_model=base_model,
        hidden_size=int(base_model.config.hidden_size),
        graph_feature_size=8,
        graph_feature_dim=args.graph_feature_dim,
        dropout=args.scorer_dropout,
        use_graph_prior=args.use_graph_prior,
        graph_prior_type=args.graph_prior_type,
        residual_alpha_init=args.residual_alpha_init,
        residual_bound_mode=args.residual_bound_mode,
        residual_bound_value=args.residual_bound_value,
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
                "model_graph_prior": bool(model.use_graph_prior),
                "model_graph_prior_type": str(model.graph_prior_type),
                "model_residual_bound_mode": str(model.residual_bound_mode),
                "model_residual_bound_value": float(model.residual_bound_value),
            },
            ensure_ascii=False,
        )
    )

    train_ds = RerankerGroupDataset(
        train_groups,
        train=True,
        train_negatives=args.train_negatives,
        hard_negatives=args.hard_negatives,
        expert_mode=args.expert_mode,
        target_demote_rank_min=args.target_demote_rank_min,
        target_demote_score_scale=args.target_demote_score_scale,
        hardneg_promote_topn=args.hardneg_promote_topn,
        hardneg_score_boost=args.hardneg_score_boost,
        input_template=args.input_template,
    )
    val_ds = RerankerGroupDataset(
        val_groups,
        train=False,
        train_negatives=args.train_negatives,
        hard_negatives=args.hard_negatives,
        expert_mode=args.expert_mode,
        target_demote_rank_min=args.target_demote_rank_min,
        target_demote_score_scale=args.target_demote_score_scale,
        hardneg_promote_topn=args.hardneg_promote_topn,
        hardneg_score_boost=args.hardneg_score_boost,
        eval_candidate_limit=args.eval_candidate_limit,
        input_template=args.input_template,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_groups, shuffle=True, collate_fn=collate_groups, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_groups, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    metadata = vars(args).copy()
    metadata.update({"experts": list(EXPERTS), "target_modules": TARGET_MODULES, "data_summary": data_summary})
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")

    scaler_enabled = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    global_step = 0
    best_mrr = -1.0
    running_loss: List[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="train")
    while global_step < total_steps:
        for batch_idx, batch in enumerate(train_loader):
            groups = batch["groups"]
            pref, graph, refine, gf, sizes, target_positions = make_batch(tokenizer, groups, args.max_length, device, masked=False)
            with torch.cuda.amp.autocast(enabled=scaler_enabled, dtype=torch.float16):
                output = model(pref, graph, refine, gf, return_parts=bool(args.use_graph_prior and args.residual_l2 > 0))
                if isinstance(output, tuple):
                    scores, residual, _, _ = output
                    residual_penalty = residual.float().pow(2).mean() * float(args.residual_l2)
                else:
                    scores = output
                    residual_penalty = None
                clean_loss = group_loss(scores, sizes, target_positions)
                if clean_loss is None:
                    continue
                loss = clean_loss
                if args.raat_mode == "target_mask_2view":
                    pref_m, graph_m, refine_m, gf_m, sizes_m, target_positions_m = make_batch(tokenizer, groups, args.max_length, device, view="target_mask")
                    scores_m = model(pref_m, graph_m, refine_m, gf_m)
                    hard_loss = group_loss(scores_m, sizes_m, target_positions_m)
                    if hard_loss is not None:
                        loss = torch.maximum(clean_loss, hard_loss)
                elif args.raat_mode == "graph_3view":
                    losses = [clean_loss]
                    for view in ("target_demote", "hardneg_promote"):
                        pref_v, graph_v, refine_v, gf_v, sizes_v, target_positions_v = make_batch(tokenizer, groups, args.max_length, device, view=view)
                        scores_v = model(pref_v, graph_v, refine_v, gf_v)
                        view_loss = group_loss(scores_v, sizes_v, target_positions_v)
                        if view_loss is not None:
                            losses.append(view_loss)
                    loss = torch.stack(losses).max()
                if residual_penalty is not None:
                    loss = loss + residual_penalty
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()
            running_loss.append(float(loss.detach().cpu().item() * max(1, args.grad_accum)))
            if (batch_idx + 1) % args.grad_accum != 0:
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
            should_eval = (global_step % args.eval_steps == 0 or global_step == total_steps) and (
                not args.eval_final_only or global_step == total_steps
            )
            if should_eval:
                model.save_trainable(args.output_dir / "latest", metadata | {"step": global_step, "checkpoint_type": "latest_before_eval"})
                metrics = evaluate(model, tokenizer, val_loader, args, device)
                metrics["step"] = global_step
                print(json.dumps(metrics, ensure_ascii=False))
                with (args.output_dir / "eval_history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                if metrics["mrr"] > best_mrr:
                    best_mrr = metrics["mrr"]
                    model.save_trainable(args.output_dir / "best", metadata | {"best_step": global_step, "best_metrics": metrics})
                model.train()
            if global_step % args.save_steps == 0 and not should_eval:
                model.save_trainable(args.output_dir / "latest", metadata | {"step": global_step, "checkpoint_type": "latest"})
            if global_step >= total_steps:
                break
    progress.close()
    model.save_trainable(args.output_dir / "latest", metadata | {"final_step": global_step, "checkpoint_type": "latest_final"})


if __name__ == "__main__":
    main()
