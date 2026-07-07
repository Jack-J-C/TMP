#!/usr/bin/env python3
"""Compress raw_text in joined POI parquet files.

The reranker uses a candidate-postfix template. Full raw_text is often long
enough to push [CANDIDATE] past short max_length settings, so this script keeps
the high-value semantic sections and truncates long evidence lists.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


SECTION_HEADINGS = {
    "Trajectory:",
    "Transitions:",
    "Nearby:",
    "Preference:",
    "top_categories:",
    "revisited_pois:",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress joined topK parquet raw_text while keeping schema stable.")
    p.add_argument("--input-dir", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification"))
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    p.add_argument("--backup-suffix", default="fullraw")
    p.add_argument("--trajectory-lines", type=int, default=8)
    p.add_argument("--transitions", type=int, default=4)
    p.add_argument("--nearby", type=int, default=4)
    p.add_argument("--top-categories", type=int, default=3)
    p.add_argument("--revisited-pois", type=int, default=3)
    p.add_argument("--tokenizer", type=Path, default=Path("models/Llama-3.2-1B-Instruct"))
    p.add_argument("--skip-token-stats", action="store_true")
    return p.parse_args()


def split_sections(raw_text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for line in str(raw_text or "").splitlines():
        stripped = line.strip()
        if stripped in SECTION_HEADINGS:
            current = stripped[:-1]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def first_nonempty(lines: Iterable[str], limit: int) -> List[str]:
    out: List[str] = []
    for line in lines:
        if not str(line).strip():
            continue
        out.append(str(line))
        if len(out) >= limit:
            break
    return out


def preference_lines(sections: Dict[str, List[str]], top_categories: int, revisited_pois: int) -> List[str]:
    pref = [line for line in sections.get("Preference", []) if str(line).strip()]
    out: List[str] = []
    for line in pref:
        stripped = str(line).strip()
        if stripped.startswith("hist="):
            out.append(stripped)
            break
    out.append("top_categories:")
    out.extend(first_nonempty(sections.get("top_categories", []), top_categories))
    out.append("revisited_pois:")
    out.extend(first_nonempty(sections.get("revisited_pois", []), revisited_pois))
    return out


def compress_raw_text(
    raw_text: str,
    trajectory_lines: int,
    transitions: int,
    nearby: int,
    top_categories: int,
    revisited_pois: int,
) -> str:
    sections = split_sections(raw_text)
    header = [line for line in sections.get("header", []) if str(line).strip()]
    if header and header[0].strip() == "[ROUTE=RAW_SEM]":
        header[0] = "[ROUTE=RAW_SEM_COMPACT]"

    chunks: List[str] = []
    chunks.extend(header)
    chunks.append("")
    chunks.append("Trajectory:")
    chunks.extend(first_nonempty(sections.get("Trajectory", []), trajectory_lines))
    chunks.append("")
    chunks.append("Transitions:")
    chunks.extend(first_nonempty(sections.get("Transitions", []), transitions))
    chunks.append("")
    chunks.append("Nearby:")
    chunks.extend(first_nonempty(sections.get("Nearby", []), nearby))
    chunks.append("")
    chunks.append("Preference:")
    chunks.extend(preference_lines(sections, top_categories, revisited_pois))
    return "\n".join(chunks).strip()


def rebuild_input_text(row: Dict[str, Any], compressed_raw: str) -> str:
    parts = [
        "[VIEW=RAW_SEM_COMPACT]",
        compressed_raw,
        "[VIEW=REFINED source=full]",
        str(row.get("refined_text") or "").strip(),
    ]
    user_semantic_profile = str(row.get("user_semantic_profile") or "").strip()
    if user_semantic_profile:
        parts.extend(["[VIEW=USER_SEMANTIC_PROFILE source=preference_evidence]", user_semantic_profile])
    parts.extend(["[VIEW=GRAPH_RAG]", str(row.get("graph_text") or "").strip()])
    return "\n\n".join(parts)


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def summarize(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "max": max(values),
    }


def maybe_load_tokenizer(path: Path, skip: bool) -> Any:
    if skip:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception as exc:  # pragma: no cover - stats are optional.
        print(f"[WARN] skip token stats: failed to load tokenizer from {path}: {exc}")
        return None


def token_len(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return 0
    return len(tokenizer(str(text or ""), add_special_tokens=False).input_ids)


def compress_split(args: argparse.Namespace, split: str, tokenizer: Any) -> Dict[str, Any]:
    main_path = args.input_dir / f"{split}_joined_top{args.top_k}.parquet"
    backup_path = args.input_dir / f"{split}_joined_top{args.top_k}_{args.backup_suffix}.parquet"
    main_stats = args.input_dir / f"{split}_joined_top{args.top_k}.stats.json"
    backup_stats = args.input_dir / f"{split}_joined_top{args.top_k}_{args.backup_suffix}.stats.json"

    if backup_path.exists():
        source_path = backup_path
    else:
        if not main_path.exists():
            raise FileNotFoundError(main_path)
        main_path.rename(backup_path)
        if main_stats.exists() and not backup_stats.exists():
            main_stats.rename(backup_stats)
        source_path = backup_path

    df = pd.read_parquet(source_path)
    raw_before: List[int] = []
    raw_after: List[int] = []
    candidate_start_after: List[int] = []

    new_rows: List[Dict[str, Any]] = []
    fixed_prefix = "[TASK=CANDIDATE_RERANK]\n\nScore whether the candidate POI is the next location.\n\n[QUERY]\n\n"
    mid_prefix = "\n\n[REFINED_EVIDENCE]\n\n"
    cand_prefix = "\n\n[CANDIDATE]"
    for row in df.to_dict(orient="records"):
        original_raw = str(row.get("raw_text") or "")
        compressed_raw = compress_raw_text(
            original_raw,
            trajectory_lines=args.trajectory_lines,
            transitions=args.transitions,
            nearby=args.nearby,
            top_categories=args.top_categories,
            revisited_pois=args.revisited_pois,
        )
        row["raw_text"] = compressed_raw
        row["input_text"] = rebuild_input_text(row, compressed_raw)
        new_rows.append(row)
        if tokenizer is not None:
            raw_before.append(token_len(tokenizer, original_raw))
            raw_after.append(token_len(tokenizer, compressed_raw))
            candidate_start_after.append(
                token_len(
                    tokenizer,
                    fixed_prefix + compressed_raw + mid_prefix + str(row.get("refined_text") or "").strip() + cand_prefix,
                )
            )

    out = pd.DataFrame(new_rows)
    out.to_parquet(main_path, index=False, compression="zstd", compression_level=7)
    stats = {
        "split": split,
        "source": str(source_path),
        "output": str(main_path),
        "rows": len(out),
        "policy": {
            "trajectory_lines": args.trajectory_lines,
            "transitions": args.transitions,
            "nearby": args.nearby,
            "top_categories": args.top_categories,
            "revisited_pois": args.revisited_pois,
        },
        "bytes": main_path.stat().st_size,
    }
    if tokenizer is not None:
        stats["token_stats"] = {
            "raw_text_before": summarize(raw_before),
            "raw_text_after": summarize(raw_after),
            "candidate_start_after": summarize(candidate_start_after),
        }
    main_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    args = parse_args()
    tokenizer = maybe_load_tokenizer(args.tokenizer, args.skip_token_stats)
    summary = {
        "top_k": args.top_k,
        "input_dir": str(args.input_dir),
        "backup_suffix": args.backup_suffix,
        "splits": {},
    }
    for split in args.splits:
        summary["splits"][split] = compress_split(args, split, tokenizer)
    out_path = args.input_dir / f"compression_summary_top{args.top_k}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
