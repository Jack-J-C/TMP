#!/usr/bin/env python3
"""Fine-tune a lightweight Llama prompt refiner LoRA.

The model learns:
    label-free teacher_distill_input -> teacher distilled_prompt

It does not train next-POI prediction and never reads target labels.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.evidence.refiner_prompt_styles import REFINER_STYLES, build_user_prompt, system_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a LoRA prompt refiner from teacher distilled prompts.")
    p.add_argument("--train-inputs", type=Path, required=True)
    p.add_argument("--train-outputs", type=Path, required=True)
    p.add_argument("--val-inputs", type=Path, required=True)
    p.add_argument("--val-outputs", type=Path, required=True)
    p.add_argument("--refiner-style", choices=REFINER_STYLES, default="summary")
    p.add_argument("--base-model", type=Path, default=Path("/mnt/data/yyl/LLMMRA/models/Llama-3.2-1B-Instruct"))
    p.add_argument("--output-dir", type=Path, default=Path("models/prompt-refiner-lora-llama32-1b"))
    p.add_argument("--max-source-length", type=int, default=1536)
    p.add_argument("--max-target-length", type=int, default=384)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device-map", default=None, help="Optional transformers device_map, e.g. auto.")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


def load_pairs(input_path: Path, output_path: Path, refiner_style: str) -> List[Dict[str, str]]:
    inputs = read_jsonl(input_path)
    outputs = read_jsonl(output_path)
    input_by_id = {str(row["sample_id"]): row for row in inputs}
    if len(input_by_id) != len(inputs):
        raise ValueError(f"Duplicate sample_id in {input_path}")

    pairs: List[Dict[str, str]] = []
    seen = set()
    for row in outputs:
        sid = str(row.get("sample_id"))
        if not sid:
            raise ValueError(f"Output without sample_id in {output_path}")
        if sid in seen:
            raise ValueError(f"Duplicate output sample_id: {sid}")
        seen.add(sid)
        inp = input_by_id.get(sid)
        if inp is None:
            raise ValueError(f"Missing input for output sample_id: {sid}")
        distilled = str(row.get("distilled_prompt") or "").strip()
        if len(distilled) < 80:
            raise ValueError(f"distilled_prompt too short for {sid}")
        pairs.append(
            {
                "sample_id": sid,
                "user_prompt": build_user_prompt(inp, refiner_style),
                "target": distilled,
            }
        )

    missing = set(input_by_id) - seen
    if missing:
        raise ValueError(f"{len(missing)} inputs have no teacher output; first={sorted(missing)[:5]}")
    return pairs


@dataclass
class PromptRefinerDataset(Dataset):
    pairs: List[Dict[str, str]]
    tokenizer: Any
    refiner_style: str
    max_source_length: int
    max_target_length: int

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.pairs[idx]
        messages = [
            {"role": "system", "content": system_prompt(self.refiner_style)},
            {"role": "user", "content": item["user_prompt"]},
        ]
        source = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target = item["target"] + self.tokenizer.eos_token

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
        kwargs["optim"] = "adamw_torch"
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

    train_pairs = load_pairs(args.train_inputs, args.train_outputs, args.refiner_style)
    val_pairs = load_pairs(args.val_inputs, args.val_outputs, args.refiner_style)
    print(
        json.dumps(
            {"train_pairs": len(train_pairs), "val_pairs": len(val_pairs), "refiner_style": args.refiner_style},
            ensure_ascii=False,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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

    train_ds = PromptRefinerDataset(
        train_pairs, tokenizer, args.refiner_style, args.max_source_length, args.max_target_length
    )
    val_ds = PromptRefinerDataset(val_pairs, tokenizer, args.refiner_style, args.max_source_length, args.max_target_length)
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
