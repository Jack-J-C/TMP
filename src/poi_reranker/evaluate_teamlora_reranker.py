#!/usr/bin/env python3
"""Standalone evaluator for TeamLoRA candidate-wise POI reranker checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from train_teamlora_reranker_raat import (
    RerankerGroupDataset,
    TeamLoRAReranker,
    collate_groups,
    evaluate,
    inject_multi_expert_lora,
    load_groups,
    load_groups_from_joined,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved TeamLoRA reranker checkpoint.")
    p.add_argument("--model-dir", type=Path, required=True, help="Training output dir containing metadata.json and checkpoint subdirs.")
    p.add_argument("--checkpoint", default="best", help="Checkpoint subdir name, e.g. best or latest.")
    p.add_argument("--checkpoint-path", type=Path, default=None, help="Optional direct checkpoint directory override.")
    p.add_argument("--base-model", type=Path, default=None)
    p.add_argument("--val-groups", type=Path, default=None)
    p.add_argument("--val-joined", type=Path, default=None)
    p.add_argument("--semantic-map", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--expert-mode", choices=["anonymous", "named"], default=None)
    p.add_argument("--input-template", choices=["legacy", "semantic_profile_v1"], default=None)
    p.add_argument("--val-hit-only", action="store_true")
    p.add_argument("--max-val-groups", type=int, default=None)
    p.add_argument("--eval-candidate-limit", type=int, default=None)
    p.add_argument("--eval-candidate-batch-size", type=int, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)
    return p.parse_args()


def load_metadata(model_dir: Path, checkpoint_dir: Path) -> Dict[str, Any]:
    ckpt_meta = checkpoint_dir / "metadata.json"
    root_meta = model_dir / "metadata.json"
    if ckpt_meta.exists():
        return json.loads(ckpt_meta.read_text(encoding="utf-8"))
    if root_meta.exists():
        return json.loads(root_meta.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No metadata.json found in {checkpoint_dir} or {model_dir}")


def metadata_value(args: argparse.Namespace, metadata: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return metadata.get(key, default)


def main() -> None:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args = parse_args()
    checkpoint_dir = args.checkpoint_path or (args.model_dir / args.checkpoint)
    metadata = load_metadata(args.model_dir, checkpoint_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    base_model_path = Path(metadata_value(args, metadata, "base_model"))
    val_groups_path = metadata_value(args, metadata, "val_groups")
    val_joined_path = metadata_value(args, metadata, "val_joined")
    semantic_map_path = metadata_value(args, metadata, "semantic_map")
    top_k = int(metadata_value(args, metadata, "top_k", 100))
    max_length = int(metadata_value(args, metadata, "max_length", 512))
    expert_mode = str(metadata_value(args, metadata, "expert_mode", "anonymous"))
    input_template = str(metadata_value(args, metadata, "input_template", "legacy"))
    attn_impl = args.attn_implementation or metadata.get("attn_implementation")

    if val_groups_path:
        val_groups = load_groups(Path(val_groups_path), args.max_val_groups)
    else:
        if val_joined_path is None or semantic_map_path is None:
            raise ValueError("Need --val-joined and --semantic-map, or metadata containing val_joined/semantic_map.")
        val_groups = load_groups_from_joined(Path(val_joined_path), Path(semantic_map_path), top_k, args.max_val_groups)
    raw_val_groups = len(val_groups)
    if args.val_hit_only:
        val_groups = [group for group in val_groups if group.get("target_in_candidates")]

    tokenizer_dir = args.model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 or metadata.get("bf16") else torch.float16 if args.fp16 or metadata.get("fp16") else None
    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    base_model = AutoModel.from_pretrained(base_model_path, **model_kwargs)
    base_model.config.use_cache = False
    for param in base_model.parameters():
        param.requires_grad = False
    inject_multi_expert_lora(
        base_model,
        r=int(metadata.get("lora_r", 8)),
        alpha=int(metadata.get("lora_alpha", 16)),
        dropout=float(metadata.get("lora_dropout", 0.05)),
    )
    model = TeamLoRAReranker(
        base_model=base_model,
        hidden_size=int(base_model.config.hidden_size),
        graph_feature_size=8,
        graph_feature_dim=int(metadata.get("graph_feature_dim", 32)),
        dropout=float(metadata.get("scorer_dropout", 0.1)),
        use_graph_prior=bool(metadata.get("use_graph_prior", False)),
        graph_prior_type=str(metadata.get("graph_prior_type", "rank_log")),
        residual_alpha_init=float(metadata.get("residual_alpha_init", 0.1)),
        residual_bound_mode=str(metadata.get("residual_bound_mode", "none")),
        residual_bound_value=float(metadata.get("residual_bound_value", 0.3)),
    ).to(device)
    state_path = checkpoint_dir / "trainable_model.bin"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu")
    load_result = model.load_state_dict(state, strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading checkpoint: {load_result.unexpected_keys[:10]}")

    val_ds = RerankerGroupDataset(
        val_groups,
        train=False,
        train_negatives=int(metadata.get("train_negatives", 15)),
        hard_negatives=int(metadata.get("hard_negatives", 12)),
        expert_mode=expert_mode,
        target_demote_rank_min=int(metadata.get("target_demote_rank_min", 50)),
        target_demote_score_scale=float(metadata.get("target_demote_score_scale", 0.2)),
        hardneg_promote_topn=int(metadata.get("hardneg_promote_topn", 3)),
        hardneg_score_boost=float(metadata.get("hardneg_score_boost", 2.0)),
        eval_candidate_limit=args.eval_candidate_limit,
        input_template=input_template,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_groups, num_workers=0)
    eval_args = argparse.Namespace(max_length=max_length, eval_candidate_batch_size=args.eval_candidate_batch_size)
    metrics = evaluate(model, tokenizer, val_loader, eval_args, device)
    metrics.update(
        {
            "model_dir": str(args.model_dir),
            "checkpoint": str(checkpoint_dir),
            "raw_val_groups": raw_val_groups,
            "val_groups": len(val_groups),
            "val_hit_only": bool(args.val_hit_only),
            "eval_candidate_limit": args.eval_candidate_limit,
        }
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
