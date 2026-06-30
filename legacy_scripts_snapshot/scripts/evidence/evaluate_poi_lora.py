#!/usr/bin/env python3
"""Evaluate next-POI LoRA by constrained JSON generation."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


POI_RE = re.compile(r"v\d+")


def patch_torch_set_submodule() -> None:
    """Backport nn.Module.set_submodule for bitsandbytes quantization."""
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
    p = argparse.ArgumentParser(description="Evaluate next-POI LoRA.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base-model", type=Path, default=Path("/mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct"))
    p.add_argument("--lora-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-source-length", type=int, default=3072)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--rank-candidates", action="store_true", help="Also rank candidate POIs by target JSON NLL.")
    p.add_argument("--rank-batch-size", type=int, default=16)
    p.add_argument("--rank-top-k", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit base model loading for evaluation.")
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


def source_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages") or []
    if len(messages) >= 2:
        return messages[:2]
    return [
        {"role": "system", "content": "You are a next-POI predictor. Output only JSON."},
        {"role": "user", "content": str(row.get("input_prompt") or "")},
    ]


def extract_poi(text: str) -> str:
    try:
        obj = json.loads(text)
        value = str(obj.get("next_poi_id") or "")
        if POI_RE.fullmatch(value):
            return value
    except json.JSONDecodeError:
        pass
    match = POI_RE.search(text)
    return match.group(0) if match else ""


def build_report(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(outputs)
    correct = sum(1 for row in outputs if row["pred_poi_id"] == row["target_poi_id"])
    parse_ok = sum(1 for row in outputs if row["pred_poi_id"])
    target_in_candidates = sum(1 for row in outputs if row.get("target_in_candidates"))
    correct_when_target_in = sum(
        1 for row in outputs if row.get("target_in_candidates") and row["pred_poi_id"] == row["target_poi_id"]
    )
    by_quality: Dict[str, Counter] = {}
    for row in outputs:
        quality = str(row.get("evidence_quality") or "unknown")
        if quality not in by_quality:
            by_quality[quality] = Counter()
        by_quality[quality]["total"] += 1
        by_quality[quality]["correct"] += int(row["pred_poi_id"] == row["target_poi_id"])
    report = {
        "rows": total,
        "parse_rate": round(parse_ok / total, 6) if total else 0.0,
        "top1": round(correct / total, 6) if total else 0.0,
        "target_in_candidates_ratio": round(target_in_candidates / total, 6) if total else 0.0,
        "top1_when_target_in_candidates": round(correct_when_target_in / target_in_candidates, 6)
        if target_in_candidates
        else 0.0,
        "by_evidence_quality": {
            key: {
                "rows": val["total"],
                "top1": round(val["correct"] / val["total"], 6) if val["total"] else 0.0,
            }
            for key, val in sorted(by_quality.items())
        },
    }
    ranked = [row for row in outputs if row.get("candidate_rank") is not None]
    if ranked:
        def hit_at(k: int) -> float:
            return sum(1 for row in ranked if int(row["candidate_rank"]) <= k) / len(ranked)

        def rr(row: Dict[str, Any]) -> float:
            return 1.0 / int(row["candidate_rank"]) if row.get("candidate_rank") else 0.0

        def ndcg_at(k: int) -> float:
            vals = []
            for row in ranked:
                rank = int(row["candidate_rank"])
                vals.append(1.0 / torch.log2(torch.tensor(rank + 1.0)).item() if rank <= k else 0.0)
            return sum(vals) / len(vals) if vals else 0.0

        report["candidate_ranking"] = {
            "rows": len(ranked),
            "hit1": round(hit_at(1), 6),
            "hit5": round(hit_at(5), 6),
            "hit10": round(hit_at(10), 6),
            "hit20": round(hit_at(20), 6),
            "mrr": round(sum(rr(row) for row in ranked) / len(ranked), 6),
            "ndcg5": round(ndcg_at(5), 6),
            "ndcg10": round(ndcg_at(10), 6),
            "ndcg20": round(ndcg_at(20), 6),
        }
    return report


def candidate_target(candidate: str, eos_token: str) -> str:
    return json.dumps({"next_poi_id": str(candidate)}, ensure_ascii=False, separators=(",", ":")) + eos_token


def score_candidate_batch(
    model: Any,
    tokenizer: Any,
    source_text: str,
    candidates: List[str],
    device: str,
    max_source_length: int,
    rank_batch_size: int,
) -> Dict[str, float]:
    source_ids = tokenizer(
        source_text,
        add_special_tokens=False,
        max_length=max_source_length,
        truncation=True,
    )["input_ids"]
    scores: Dict[str, float] = {}
    for start in range(0, len(candidates), rank_batch_size):
        batch_candidates = candidates[start : start + rank_batch_size]
        encoded_rows = []
        label_rows = []
        for candidate in batch_candidates:
            target_ids = tokenizer(
                candidate_target(candidate, tokenizer.eos_token),
                add_special_tokens=False,
            )["input_ids"]
            input_ids = source_ids + target_ids
            labels = [-100] * len(source_ids) + target_ids
            encoded_rows.append(input_ids)
            label_rows.append(labels)
        max_len = max(len(x) for x in encoded_rows)
        pad_id = tokenizer.pad_token_id
        input_ids_tensor = torch.tensor(
            [row + [pad_id] * (max_len - len(row)) for row in encoded_rows],
            dtype=torch.long,
            device=device,
        )
        labels_tensor = torch.tensor(
            [row + [-100] * (max_len - len(row)) for row in label_rows],
            dtype=torch.long,
            device=device,
        )
        attention_mask = (input_ids_tensor != pad_id).long()
        with torch.no_grad():
            logits = model(input_ids=input_ids_tensor, attention_mask=attention_mask).logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels_tensor[:, 1:].contiguous()
        losses = torch.nn.functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shifted_labels.shape)
        token_counts = (shifted_labels != -100).sum(dim=1).clamp_min(1)
        seq_scores = losses.sum(dim=1) / token_counts
        for candidate, score in zip(batch_candidates, seq_scores.detach().cpu().tolist()):
            scores[candidate] = float(score)
    return scores


def main() -> None:
    args = parse_args()
    if args.load_in_4bit:
        patch_torch_set_submodule()
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
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.to(args.device)
    model.eval()

    outputs: List[Dict[str, Any]] = []
    batches = list(iter_batches(rows, args.batch_size))
    progress = batches
    if not args.no_progress and tqdm is not None:
        progress = tqdm(batches, total=len(batches), desc="Evaluating POI LoRA", unit="batch")

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
        with torch.no_grad():
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
            pred = extract_poi(text)
            target = str((row.get("target") or {}).get("poi_id") or "")
            metadata = row.get("metadata") or {}
            candidate_rank = None
            candidate_top = []
            if args.rank_candidates:
                candidates = [str(x) for x in row.get("candidate_poi_ids") or []]
                if target in set(candidates):
                    source_text = tokenizer.apply_chat_template(
                        source_messages(row), tokenize=False, add_generation_prompt=True
                    )
                    scores = score_candidate_batch(
                        model,
                        tokenizer,
                        source_text,
                        candidates,
                        args.device,
                        args.max_source_length,
                        args.rank_batch_size,
                    )
                    ranked = sorted(scores.items(), key=lambda x: x[1])
                    candidate_top = [poi for poi, _ in ranked[: args.rank_top_k]]
                    candidate_rank = next((idx for idx, (poi, _) in enumerate(ranked, 1) if poi == target), None)
            outputs.append(
                {
                    "sample_id": row.get("sample_id"),
                    "split": row.get("split"),
                    "city": row.get("city"),
                    "route": row.get("route"),
                    "target_poi_id": target,
                    "pred_poi_id": pred,
                    "correct": pred == target,
                    "raw_generation": text,
                    "parse_ok": bool(pred),
                    "target_in_candidates": bool(metadata.get("target_in_candidates")),
                    "evidence_quality": metadata.get("evidence_quality"),
                    "difficulty_factors": metadata.get("difficulty_factors") or [],
                    "metadata": metadata,
                    "candidate_count": len(row.get("candidate_poi_ids") or []),
                    "candidate_rank": candidate_rank,
                    "candidate_top": candidate_top,
                }
            )

    write_jsonl(args.output, outputs, args.overwrite)
    report = build_report(outputs)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
