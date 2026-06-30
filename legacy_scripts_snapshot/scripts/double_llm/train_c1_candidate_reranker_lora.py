#!/usr/bin/env python3
"""LoRA/QLoRA SFT for C1 GraphRAG candidate compressor.

Expected assistant target:
    {"candidate_poi_ids":["v..."]}
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def patch_torch_set_submodule() -> None:
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def set_submodule(self: torch.nn.Module, target: str, module: torch.nn.Module) -> None:
        if not target:
            raise ValueError("Cannot set the root module")
        atoms = target.split(".")
        parent = self.get_submodule(".".join(atoms[:-1])) if len(atoms) > 1 else self
        if not hasattr(parent, atoms[-1]):
            raise AttributeError(f"{parent._get_name()} has no child module {atoms[-1]}")
        setattr(parent, atoms[-1], module)

    torch.nn.Module.set_submodule = set_submodule  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train C1 candidate compressor with LoRA/QLoRA.")
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--val-data", type=Path, required=True)
    p.add_argument("--base-model", type=Path, default=Path("/mnt/data/yyl/LLMMRA/models/Llama-3.2-1B-Instruct"))
    p.add_argument("--output-dir", type=Path, default=Path("models/c1-graphrag-top100-to-top50-lora-llama32-1b"))
    p.add_argument("--init-lora-path", type=Path, default=None)
    p.add_argument("--max-source-length", type=int, default=4096)
    p.add_argument("--max-target-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device-map", default=None)
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust-remote-code", action="store_true")
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


@dataclass
class C1Dataset(Dataset):
    rows: List[Dict[str, Any]]
    tokenizer: Any
    max_source_length: int
    max_target_length: int

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.rows[idx]
        messages = row.get("messages") or []
        if len(messages) != 3:
            raise ValueError(f"{row.get('sample_id')} messages must contain system/user/assistant")
        target = str(messages[2].get("content") or "").strip()
        try:
            parsed = json.loads(target)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{row.get('sample_id')} target is not valid JSON") from exc
        if "candidate_poi_ids" not in parsed:
            raise ValueError(f"{row.get('sample_id')} target must contain candidate_poi_ids")

        source = self.tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        target = target + self.tokenizer.eos_token
        source_ids = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_length,
            truncation=True,
        )["input_ids"]
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_target_length,
            truncation=True,
        )["input_ids"]
        input_ids = source_ids + target_ids
        labels = [-100] * len(source_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
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
        kwargs["optim"] = "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch"
    return TrainingArguments(**kwargs)


def make_trainer_kwargs(tokenizer: Any) -> Dict[str, Any]:
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in sig.parameters:
        return {"tokenizer": tokenizer}
    return {}


def main() -> None:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args = parse_args()
    if args.load_in_4bit:
        patch_torch_set_submodule()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    print(json.dumps({"train_rows": len(train_rows), "val_rows": len(val_rows)}, ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    device_map: Any = args.device_map
    if local_rank >= 0 and device_map is None:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing or args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.init_lora_path is not None:
        print(json.dumps({"init_lora_path": str(args.init_lora_path)}, ensure_ascii=False))
        model = PeftModel.from_pretrained(model, args.init_lora_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = C1Dataset(train_rows, tokenizer, args.max_source_length, args.max_target_length)
    val_ds = C1Dataset(val_rows, tokenizer, args.max_source_length, args.max_target_length)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        **make_trainer_kwargs(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
