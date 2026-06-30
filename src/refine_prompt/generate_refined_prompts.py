#!/usr/bin/env python3
"""Generate distilled evidence prompts with the local prompt-refiner LoRA."""
from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm should exist in the training env.
    tqdm = None

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.evidence.refiner_prompt_styles import REFINER_STYLES, build_user_prompt, system_prompt
from scripts.evidence.train_prompt_refiner_lora import read_jsonl


FORBIDDEN_PATTERNS = [
    re.compile(r"\btarget_poi_id\b", re.IGNORECASE),
    re.compile(r"\btarget_category\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]", re.IGNORECASE),
    re.compile(r"\bgold\s+answer\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+answer\b", re.IGNORECASE),
    re.compile(r"\bground\s+truth\b", re.IGNORECASE),
]
CONFIDENCE_RE = re.compile(r"\bconfidence\s*=\s*(high|medium|low)\b", re.IGNORECASE)
USEFUL_RE = re.compile(r"\buseful_for_refinement\s*=\s*(yes|no)\b", re.IGNORECASE)
FINAL_HINT_RE = re.compile(r"Final hint:", re.IGNORECASE)
EVIDENCE_SOURCE_RE = re.compile(r"\bevidence\s*=\s*(transition|geo|history|sequence|mixed)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate local LoRA-A refined evidence prompts.")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--teacher-outputs", type=Path, default=None)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--lora-path", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, default=None, help="Tokenizer path. Defaults to --base-model.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--refiner-style", choices=REFINER_STYLES, default="summary")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-source-length", type=int, default=1536)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from output.tmp or output by sample_id.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
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


def append_jsonl(f: Any, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def select_rows(rows: List[Dict[str, Any]], offset: int, limit: int) -> List[Dict[str, Any]]:
    selected = rows[offset:]
    if limit >= 0:
        selected = selected[:limit]
    return selected


def build_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    raise RuntimeError("build_messages requires args.refiner_style; use build_messages_for_style")


def build_messages_for_style(row: Dict[str, Any], refiner_style: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(refiner_style)},
        {"role": "user", "content": build_user_prompt(row, refiner_style)},
    ]


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^assistant\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
    text = re.sub(r"```(?:text|json)?", "", text)
    text = text.replace("```", "")
    return re.sub(r"\s+", " ", text).strip()


def parse_decision_signals(text: str) -> Dict[str, str]:
    confidence = CONFIDENCE_RE.search(text)
    useful = USEFUL_RE.search(text)
    return {
        "refiner_confidence": confidence.group(1).lower() if confidence else "unknown",
        "useful_for_refinement": useful.group(1).lower() if useful else "unknown",
    }


def _split_short_cues(text: str, limit: int = 3) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .;,")
    if not text:
        return "low-count or generic cues"
    pieces = [piece.strip(" .;") for piece in re.split(r"[,;]", text) if piece.strip(" .;")]
    if not pieces:
        pieces = [text]
    return ", ".join(pieces[:limit])


def repair_decision_prompt(text: str) -> str:
    """Stabilize local LoRA-A decision outputs without changing their main signals."""
    text = clean_generation(text)
    first_verdict = text.lower().find("evidence verdict:")
    if first_verdict > 0:
        text = text[first_verdict:].strip()
    second_verdict = text.lower().find("evidence verdict:", len("Evidence verdict:"))
    if second_verdict > 0:
        text = text[:second_verdict].strip()

    final_match = FINAL_HINT_RE.search(text)
    final_text = ""
    if final_match:
        final_text = text[final_match.start() :].strip()
        text = text[: final_match.start()].strip()
        next_verdict = final_text.lower().find("evidence verdict:", len("Final hint:"))
        if next_verdict > 0:
            final_text = final_text[:next_verdict].strip()

    weak_match = re.search(r"Weak or noisy cues:", text, flags=re.IGNORECASE)
    if weak_match:
        weak_prefix = text[: weak_match.end()]
        weak_body = text[weak_match.end() :].strip()
        next_verdict = weak_body.lower().find("evidence verdict:")
        if next_verdict >= 0:
            weak_body = weak_body[:next_verdict].strip()
        text = f"{weak_prefix} {_split_short_cues(weak_body)}."

    if not final_text:
        useful = parse_decision_signals(text).get("useful_for_refinement")
        source_match = EVIDENCE_SOURCE_RE.search(text)
        source = source_match.group(1).lower() if source_match else "raw"
        if useful == "no":
            final_text = "Final hint: prefer raw because the refined evidence is weak or incomplete."
        else:
            final_text = f"Final hint: prefer {source} because it is the clearest available evidence source."

    final_text = re.sub(r"\s+", " ", final_text).strip()
    if not final_text.endswith("."):
        final_text += "."
    return f"{text.rstrip()} {final_text}"


def has_leakage(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORBIDDEN_PATTERNS)


def load_teacher_outputs(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows if row.get("sample_id")}


def word_count(text: str) -> int:
    return len(text.split())


def build_report(outputs: List[Dict[str, Any]], teacher_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(row["distilled_prompt"]) for row in outputs]
    words = [word_count(row["distilled_prompt"]) for row in outputs]
    teacher_deltas = []
    for row in outputs:
        teacher = teacher_by_id.get(row["sample_id"])
        if teacher:
            teacher_deltas.append(len(row["distilled_prompt"]) - len(str(teacher.get("distilled_prompt") or "")))

    def avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "rows": len(outputs),
        "duplicate_ids": len(outputs) - len({row["sample_id"] for row in outputs}),
        "empty_outputs": sum(1 for row in outputs if not row["distilled_prompt"]),
        "short_outputs": sum(1 for row in outputs if len(row["distilled_prompt"]) < 80),
        "leakage_outputs": sum(1 for row in outputs if has_leakage(row["distilled_prompt"])),
        "chars_min": min(lengths) if lengths else 0,
        "chars_avg": avg(lengths),
        "chars_max": max(lengths) if lengths else 0,
        "words_min": min(words) if words else 0,
        "words_avg": avg(words),
        "words_max": max(words) if words else 0,
        "teacher_length_delta_avg": avg(teacher_deltas),
        "teacher_matched_rows": len(teacher_deltas),
    }


def iter_batches(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def dedupe_outputs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        deduped.append(row)
    return deduped


def main() -> None:
    args = parse_args()
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if tmp_output.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"{tmp_output} exists; pass --overwrite or --resume")
    if args.report and args.report.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"{args.report} exists; pass --overwrite")

    rows = select_rows(read_jsonl(args.inputs), args.offset, args.limit)
    teacher_by_id = load_teacher_outputs(args.teacher_outputs)
    existing_outputs: List[Dict[str, Any]] = []
    if args.resume:
        existing_outputs = read_jsonl_if_exists(tmp_output)
        if not existing_outputs:
            existing_outputs = read_jsonl_if_exists(args.output)
        existing_outputs = dedupe_outputs(existing_outputs)
        existing_ids = {str(row["sample_id"]) for row in existing_outputs}
        rows = [row for row in rows if str(row["sample_id"]) not in existing_ids]
        print(
            json.dumps(
                {
                    "resume_existing_rows": len(existing_outputs),
                    "remaining_rows": len(rows),
                    "tmp_path": str(tmp_output),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    tokenizer_path = args.tokenizer_path or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.to(args.device)
    model.eval()

    outputs: List[Dict[str, Any]] = list(existing_outputs)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0.0,
        "batch_size": args.batch_size,
        "max_source_length": args.max_source_length,
        "refiner_style": args.refiner_style,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    batches = list(iter_batches(rows, args.batch_size))
    progress = batches
    if not args.no_progress and tqdm is not None:
        progress = tqdm(batches, total=len(batches), desc="Generating refined prompts", unit="batch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_existing = args.resume and existing_outputs and not tmp_output.exists()
    mode = "a" if args.resume and tmp_output.exists() else "w"
    with tmp_output.open(mode, encoding="utf-8") as out_f:
        if write_existing:
            for row in existing_outputs:
                append_jsonl(out_f, row)
        for batch in progress:
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages_for_style(row, args.refiner_style), tokenize=False, add_generation_prompt=True
                )
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
                    do_sample=args.temperature > 0.0,
                    temperature=args.temperature if args.temperature > 0.0 else None,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = encoded["input_ids"].shape[1]
            for row, generated_ids in zip(batch, generated):
                sample_id = str(row["sample_id"])
                new_tokens = generated_ids[prompt_width:]
                distilled = clean_generation(tokenizer.decode(new_tokens, skip_special_tokens=True))
                if args.refiner_style == "decision":
                    distilled = repair_decision_prompt(distilled)
                output_row = {
                    "sample_id": sample_id,
                    "city": row.get("city"),
                    "split": row.get("split"),
                    "distilled_prompt": distilled,
                    "source_model": str(args.lora_path),
                    "generation_config": generation_config,
                    "refiner_style": args.refiner_style,
                }
                if args.refiner_style == "decision":
                    output_row.update(parse_decision_signals(distilled))
                outputs.append(output_row)
                append_jsonl(out_f, output_row)
    tmp_output.replace(args.output)

    report = build_report(outputs, teacher_by_id)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
