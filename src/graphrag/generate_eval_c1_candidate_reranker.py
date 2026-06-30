#!/usr/bin/env python3
"""Generate and evaluate C1 Top50 candidate outputs.

The preferred C1 output is compact JSON:
    {"candidate_poi_ids":["v..."]}
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


POI_RE = re.compile(r"^v\d+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/evaluate C1 Top50 candidates.")
    p.add_argument("--input-data", type=Path, required=True, help="C1 SFT JSONL with messages.")
    p.add_argument("--graphrag-candidates", type=Path, required=True, help="Reference GraphRAG Top100 JSONL with targets.")
    p.add_argument("--base-model", type=Path, default=Path("/mnt/data/yyl/LLMMRA/models/Llama-3.2-1B-Instruct"))
    p.add_argument("--lora-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--top-out", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-source-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--disable-tqdm", action="store_true")
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


def load_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if sid:
            out[sid] = row
    return out


def extract_json(text: str) -> tuple[Dict[str, Any] | None, str | None]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None, "missing_json_object"
    snippet = text[start : end + 1]
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    return obj, None


def extract_candidate_poi_ids(text: str) -> tuple[List[str], str | None]:
    parsed, error = extract_json(text)
    if parsed is not None:
        raw = parsed.get("candidate_poi_ids")
        if isinstance(raw, list):
            return [str(x) for x in raw], None
        return [], "missing_candidate_poi_ids"
    # Salvage truncated JSON when the candidate_poi_ids array was completed
    # before the object was cut off.
    key = '"candidate_poi_ids"'
    start = text.find(key)
    if start < 0:
        return [], error
    bracket_start = text.find("[", start)
    bracket_end = text.find("]", bracket_start)
    if bracket_start < 0 or bracket_end < bracket_start:
        return [], error
    snippet = text[bracket_start : bracket_end + 1]
    try:
        raw = json.loads(snippet)
    except json.JSONDecodeError:
        return [], error
    if not isinstance(raw, list):
        return [], error
    return [str(x) for x in raw], "salvaged_candidate_poi_ids"


def normalize_candidates(raw: Any, allowed: List[str], top_out: int) -> List[str]:
    allowed_set = set(allowed)
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            poi = str(item)
            if POI_RE.match(poi) and poi in allowed_set and poi not in out:
                out.append(poi)
            if len(out) >= top_out:
                break
    for poi in allowed:
        if len(out) >= top_out:
            break
        if poi not in out:
            out.append(poi)
    return out[:top_out]


def batched(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    rows = read_jsonl(args.input_data)
    if args.limit is not None:
        rows = rows[: args.limit]
    refs = load_by_id(args.graphrag_candidates)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    iterator = batched(rows, args.batch_size)
    progress = iterator if args.disable_tqdm else tqdm(iterator, total=(len(rows) + args.batch_size - 1) // args.batch_size)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 and torch.cuda.is_available() else nullcontext()
    with args.output.open("w", encoding="utf-8") as f, torch.inference_mode(), autocast_ctx:
        for batch in progress:
            prompts = [
                tokenizer.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
                add_special_tokens=False,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for row, seq, input_len in zip(batch, generated, encoded["attention_mask"].sum(dim=1).tolist()):
                sid = str(row.get("sample_id") or "")
                ref = refs.get(sid) or {}
                allowed = [str(x) for x in ref.get("candidate_poi_ids") or []]
                target = str((ref.get("target") or {}).get("poi_id") or "")
                text = tokenizer.decode(seq[int(input_len) :], skip_special_tokens=True).strip()
                raw_candidates, error = extract_candidate_poi_ids(text)
                predicted = normalize_candidates(raw_candidates, allowed, args.top_out)
                rank = next((idx for idx, poi in enumerate(predicted, 1) if poi == target), None)
                counts["rows"] += 1
                counts["parse_ok"] += int(error is None)
                counts["salvaged"] += int(error == "salvaged_candidate_poi_ids")
                counts["fallback_filled"] += int(len(predicted) == args.top_out)
                counts["changed_vs_graph_top50"] += int(predicted != allowed[: args.top_out])
                for k in [1, 5, 10, 20, args.top_out]:
                    counts[f"hit@{k}"] += int(rank is not None and rank <= k)
                f.write(
                    json.dumps(
                        {
                            "sample_id": sid,
                            "raw_generation": text,
                            "parse_error": error,
                            "candidate_poi_ids": predicted,
                            "target_poi_id": target,
                            "target_rank": rank,
                            "target_in_top_out": rank is not None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    n = counts["rows"]
    report = {
        "input_data": str(args.input_data),
        "lora_path": str(args.lora_path),
        "output": str(args.output),
        "rows": n,
        "top_out": args.top_out,
        "parse_ok_ratio": round(counts["parse_ok"] / n, 6) if n else 0.0,
        "salvaged_ratio": round(counts["salvaged"] / n, 6) if n else 0.0,
        "changed_vs_graph_top50_ratio": round(counts["changed_vs_graph_top50"] / n, 6) if n else 0.0,
        **{f"hit@{k}_ratio": round(counts[f"hit@{k}"] / n, 6) if n else 0.0 for k in [1, 5, 10, 20, args.top_out]},
    }
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
