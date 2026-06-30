#!/usr/bin/env python3
"""Generate supplemental candidate POIs with a trained LoRA-C adapter."""
from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


POI_RE = re.compile(r"v\d+")
CANDIDATE_TASK_SYSTEM = "You are a candidate expansion model for next-POI prediction. Output only JSON."
CANDIDATE_OUTPUT_SCHEMA = '{"supplemental_poi_ids":["<poi_id>"]}'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate supplemental candidate POIs with LoRA-C.")
    p.add_argument("--data", type=Path, required=True, help="LoRA-C JSONL with messages or input_prompt.")
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--lora-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-source-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-supplemental", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-progress", action="store_true")
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(rows: List[Dict[str, Any]], offset: int, limit: int) -> List[Dict[str, Any]]:
    rows = rows[offset:]
    if limit >= 0:
        rows = rows[:limit]
    return rows


def iter_batches(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def extract_candidate_body(prompt: str) -> str:
    """Convert a POI-prediction prompt into the LoRA-C candidate-booster prompt body."""
    start = prompt.find("Current trajectory:")
    if start < 0:
        return prompt

    end_candidates = prompt.find("\n\n[Refined Evidence]", start)
    end_output = prompt.find("\n\nOutput format:", start)
    ends = [idx for idx in [end_candidates, end_output] if idx >= 0]
    end = min(ends) if ends else len(prompt)
    body = prompt[start:end].strip()
    return body.replace("\nCandidate POIs:\n", "\nOriginal candidate POIs:\n")


def build_candidate_prompt(row: Dict[str, Any]) -> str:
    raw_prompt = str(row.get("input_prompt") or "")
    messages = row.get("messages") or []
    if not raw_prompt and len(messages) >= 2:
        raw_prompt = str(messages[1].get("content") or "")
    body = extract_candidate_body(raw_prompt)
    return "\n".join(
        [
            "Task:",
            "Propose supplemental POI candidates that may be missing from the original candidate list.",
            "Use only the evidence below. Do not explain.",
            "Never output next_poi_id.",
            "Return only valid JSON with the field supplemental_poi_ids.",
            "",
            "Output format:",
            CANDIDATE_OUTPUT_SCHEMA,
            "",
            body,
        ]
    )


def source_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages") or []
    if len(messages) >= 2 and "supplemental_poi_ids" in str(messages[1].get("content") or ""):
        return messages[:2]
    return [
        {"role": "system", "content": CANDIDATE_TASK_SYSTEM},
        {"role": "user", "content": build_candidate_prompt(row)},
    ]


def parse_supplemental(text: str, original_candidates: set[str], max_items: int) -> List[str]:
    values: List[str] = []
    try:
        obj = json.loads(text)
        raw_values = obj.get("supplemental_poi_ids") or []
        if isinstance(raw_values, str):
            raw_values = POI_RE.findall(raw_values)
        for value in raw_values:
            poi = str(value)
            if POI_RE.fullmatch(poi) and poi not in original_candidates and poi not in values:
                values.append(poi)
            if len(values) >= max_items:
                return values
    except json.JSONDecodeError:
        pass
    for poi in POI_RE.findall(text):
        if poi not in original_candidates and poi not in values:
            values.append(poi)
        if len(values) >= max_items:
            break
    return values


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if args.report and args.report.exists() and not args.overwrite:
        raise FileExistsError(f"{args.report} exists; pass --overwrite")

    rows = select_rows(read_jsonl(args.data), args.offset, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.to(args.device)
    model.eval()

    outputs: List[Dict[str, Any]] = []
    batches = list(iter_batches(rows, args.batch_size))
    progress: Iterable[List[Dict[str, Any]]] = batches
    if not args.no_progress and tqdm is not None:
        progress = tqdm(batches, total=len(batches), desc="Generating candidate boosts", unit="batch")

    autocast_ctx = torch.autocast("cuda", dtype=dtype) if args.device.startswith("cuda") and dtype else nullcontext()
    for batch in progress:
        prompts = [
            tokenizer.apply_chat_template(source_messages(row), tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        ).to(args.device)
        with torch.no_grad(), autocast_ctx:
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        for row, generated_ids in zip(batch, generated):
            text = tokenizer.decode(generated_ids[prompt_width:], skip_special_tokens=True).strip()
            original_candidates = {str(x) for x in row.get("candidate_poi_ids") or []}
            supplemental = parse_supplemental(text, original_candidates, args.max_supplemental)
            outputs.append(
                {
                    "sample_id": row.get("source_sample_id") or row.get("sample_id"),
                    "split": row.get("split"),
                    "city": row.get("city"),
                    "supplemental_poi_ids": supplemental,
                    "raw_generation": text,
                    "parse_ok": bool(supplemental) or '"supplemental_poi_ids"' in text,
                    "original_candidate_count": len(original_candidates),
                    "supplemental_count": len(supplemental),
                }
            )

    write_jsonl(args.output, outputs, args.overwrite)
    report = {
        "rows": len(outputs),
        "parse_rate": round(sum(1 for row in outputs if row["parse_ok"]) / len(outputs), 6) if outputs else 0.0,
        "non_empty_rate": round(sum(1 for row in outputs if row["supplemental_poi_ids"]) / len(outputs), 6)
        if outputs
        else 0.0,
        "avg_supplemental_count": round(
            sum(len(row["supplemental_poi_ids"]) for row in outputs) / len(outputs), 4
        )
        if outputs
        else 0.0,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
