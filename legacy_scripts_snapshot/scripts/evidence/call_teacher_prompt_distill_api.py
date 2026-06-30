#!/usr/bin/env python3
"""Call an OpenAI-compatible teacher API to distill evidence into downstream prompts.

This script is intended for small-batch paid API experiments. It supports
concurrent requests, retries, and resume-by-sample-id. Labels are never read.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI

import sys

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.evidence.refiner_prompt_styles import REFINER_STYLES, build_teacher_messages


FORBIDDEN_PATTERNS = [
    re.compile(r"\bgold\s+answer\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+answer\b", re.IGNORECASE),
    re.compile(r"\btrue\s+next\b", re.IGNORECASE),
    re.compile(r"\btarget\s+(poi|venue|is)\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill label-free evidence into compact downstream prompts with a teacher API.")
    p.add_argument("--input", type=Path, required=True, help="teacher_distill_inputs_{split}.jsonl")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL with distilled_prompt per sample_id")
    p.add_argument("--provider", choices=["openai-compatible", "aliyun"], default="openai-compatible")
    p.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "deepseek-v4-flash"))
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--base-url-env", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument(
        "--enable-thinking",
        choices=["true", "false", "none"],
        default="none",
        help="Aliyun/DashScope non-standard parameter. Use false with response_format JSON mode.",
    )
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--no-response-format", action="store_true", help="Disable response_format=json_object for providers/models that reject it.")
    p.add_argument("--limit", type=int, default=20, help="Maximum number of new rows to process. Use -1 for all.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--sample", action="store_true", help="Randomly sample after offset/limit filtering.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-evidence-chars", type=int, default=12000)
    p.add_argument("--max-output-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--refiner-style", choices=REFINER_STYLES, default="summary")
    p.add_argument("--resume", action="store_true", help="Skip sample_ids already present in output.")
    p.add_argument("--dry-run", action="store_true", help="Print one request payload without calling the API.")
    return p.parse_args()


def resolve_api_config(args: argparse.Namespace) -> Tuple[str, str | None, str | None]:
    if args.provider == "aliyun":
        api_key_env = args.api_key_env or "DASHSCOPE_API_KEY"
        base_url = args.base_url or os.environ.get(args.base_url_env or "DASHSCOPE_BASE_URL")
        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        enable_thinking = "false" if args.enable_thinking == "none" else args.enable_thinking
    else:
        api_key_env = args.api_key_env or "DEEPSEEK_API_KEY"
        base_url = args.base_url or os.environ.get(args.base_url_env or "DEEPSEEK_BASE_URL")
        enable_thinking = None if args.enable_thinking == "none" else args.enable_thinking
    return api_key_env, base_url, enable_thinking


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
                raise ValueError(f"{path}:{line_no} is invalid JSON: {exc}") from exc
    return rows


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path}:{line_no} is invalid JSON; fix or remove the partial output before resume")
            sid = row.get("sample_id")
            if sid:
                done.add(str(sid))
    return done


def select_rows(rows: List[Dict[str, Any]], args: argparse.Namespace, done_ids: set[str]) -> List[Dict[str, Any]]:
    selected = rows[args.offset :]
    if args.resume:
        selected = [row for row in selected if str(row.get("sample_id")) not in done_ids]
    if args.sample:
        rng = random.Random(args.seed)
        rng.shuffle(selected)
    if args.limit >= 0:
        selected = selected[: args.limit]
    return selected


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate_output(sample_id: str, obj: Dict[str, Any]) -> Tuple[str, str]:
    if str(obj.get("sample_id")) != sample_id:
        raise ValueError(f"sample_id mismatch: expected {sample_id}, got {obj.get('sample_id')}")
    prompt = str(obj.get("distilled_prompt") or "").strip()
    if len(prompt) < 80:
        raise ValueError("distilled_prompt is too short")
    if any(pattern.search(prompt) for pattern in FORBIDDEN_PATTERNS):
        raise ValueError("distilled_prompt contains forbidden answer-leakage wording")
    quality = str(obj.get("evidence_quality") or "partial").strip().lower()
    if quality not in {"good", "partial", "weak"}:
        quality = "partial"
    return prompt, quality


def build_request_kwargs(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    _, _, enable_thinking = resolve_api_config(args)
    kwargs: Dict[str, Any] = {
        "model": args.model,
        "messages": build_teacher_messages(row, args.refiner_style, args.max_evidence_chars),
        "temperature": args.temperature,
        "max_tokens": args.max_output_tokens,
        "timeout": args.timeout,
    }
    if not args.no_response_format:
        kwargs["response_format"] = {"type": "json_object"}
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort
    if enable_thinking is not None:
        kwargs["extra_body"] = {"enable_thinking": enable_thinking == "true"}
    return kwargs


def call_one(client: OpenAI, row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(**build_request_kwargs(row, args))
            content = response.choices[0].message.content or ""
            obj = extract_json_object(content)
            prompt, quality = validate_output(sample_id, obj)
            usage = getattr(response, "usage", None)
            return {
                "sample_id": sample_id,
                "city": row.get("city"),
                "split": row.get("split"),
                "user_id": row.get("user_id"),
                "trajectory_id": row.get("trajectory_id"),
                "distilled_prompt": prompt,
                "evidence_quality": quality,
                "teacher_model": args.model,
                "refiner_style": args.refiner_style,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
            }
        except Exception as exc:  # noqa: BLE001 - preserve API failure details in output.
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)
    raise RuntimeError(f"{sample_id} failed after {args.max_retries} attempts: {last_error}")


def append_jsonl(path: Path, lock: threading.Lock, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    done_ids = load_done_ids(args.output) if args.resume else set()
    selected = select_rows(rows, args, done_ids)
    if not selected:
        print(json.dumps({"status": "nothing_to_do", "input": str(args.input), "output": str(args.output)}))
        return

    if args.dry_run:
        api_key_env, base_url, enable_thinking = resolve_api_config(args)
        request_kwargs = build_request_kwargs(selected[0], args)
        preview = {
            "selected": len(selected),
            "first_sample_id": selected[0].get("sample_id"),
            "provider": args.provider,
            "api_key_env": api_key_env,
            "base_url": base_url,
            "enable_thinking": enable_thinking,
            "request_kwargs": request_kwargs,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    api_key_env, base_url, _ = resolve_api_config(args)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
    client = OpenAI(api_key=api_key, base_url=base_url or None)

    lock = threading.Lock()
    ok = 0
    failed = 0
    start = time.time()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sid = {executor.submit(call_one, client, row, args): str(row["sample_id"]) for row in selected}
        for future in futures.as_completed(future_to_sid):
            sid = future_to_sid[future]
            try:
                result = future.result()
                append_jsonl(args.output, lock, result)
                ok += 1
                print(json.dumps({"status": "ok", "sample_id": sid, "ok": ok, "failed": failed}, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                err_row = {"sample_id": sid, "error": str(exc), "teacher_model": args.model}
                append_jsonl(args.output.with_suffix(args.output.suffix + ".errors"), lock, err_row)
                print(json.dumps({"status": "error", "sample_id": sid, "error": str(exc), "ok": ok, "failed": failed}, ensure_ascii=False))

    print(
        json.dumps(
            {
                "status": "done",
                "input": str(args.input),
                "output": str(args.output),
                "requested": len(selected),
                "ok": ok,
                "failed": failed,
                "seconds": round(time.time() - start, 2),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
