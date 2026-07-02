#!/usr/bin/env python3
"""Build joined POI parquet files with fullraw backup and compact main files.

Pipeline contract:
  1. Build full raw joined parquet as *_fullraw.parquet.
  2. Build compact raw-context parquet as the main training file.

The main reranker scripts keep reading train/val_joined_topK.parquet, while the
full raw version remains available for audits or future rebuilding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from build_joined_poi_classification_data import build_split, load_semantic_map, write_parquet
from compress_joined_raw_context import compress_split, maybe_load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fullraw joined parquet and compact main parquet for POI reranking.")
    p.add_argument("--base-dir", type=Path, default=Path("retrieval_assets/NewYork"))
    p.add_argument("--output-dir", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification"))
    p.add_argument("--semantic-map", type=Path, default=None)
    p.add_argument("--graph-top-k", type=int, default=100)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    p.add_argument("--fullraw-suffix", default="fullraw")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--trajectory-lines", type=int, default=8)
    p.add_argument("--transitions", type=int, default=4)
    p.add_argument("--nearby", type=int, default=4)
    p.add_argument("--top-categories", type=int, default=3)
    p.add_argument("--revisited-pois", type=int, default=3)
    p.add_argument("--tokenizer", type=Path, default=Path("models/Llama-3.2-1B-Instruct"))
    p.add_argument("--skip-token-stats", action="store_true")
    return p.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_fullraw(args: argparse.Namespace, split: str, semantic_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows, stats = build_split(args.base_dir, split, semantic_map, args.graph_top_k)
    out = args.output_dir / f"{split}_joined_top{args.graph_top_k}_{args.fullraw_suffix}.parquet"
    write_parquet(out, rows, args.overwrite)
    stats.update(
        {
            "raw_context": "fullraw",
            "parquet_output": str(out),
            "parquet_bytes": out.stat().st_size,
        }
    )
    stats_path = args.output_dir / f"{split}_joined_top{args.graph_top_k}_{args.fullraw_suffix}.stats.json"
    write_json(stats_path, stats)
    return stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = args.semantic_map or (args.base_dir / "double_llm" / "semantic_poi_ids.jsonl")
    semantic_map = load_semantic_map(semantic_path)
    tokenizer = maybe_load_tokenizer(args.tokenizer, args.skip_token_stats)

    summary: Dict[str, Any] = {
        "semantic_map": str(semantic_path),
        "graph_top_k": args.graph_top_k,
        "output_dir": str(args.output_dir),
        "fullraw_suffix": args.fullraw_suffix,
        "compact_policy": {
            "trajectory_lines": args.trajectory_lines,
            "transitions": args.transitions,
            "nearby": args.nearby,
            "top_categories": args.top_categories,
            "revisited_pois": args.revisited_pois,
        },
        "splits": {},
    }

    compress_args = SimpleNamespace(
        input_dir=args.output_dir,
        top_k=args.graph_top_k,
        backup_suffix=args.fullraw_suffix,
        trajectory_lines=args.trajectory_lines,
        transitions=args.transitions,
        nearby=args.nearby,
        top_categories=args.top_categories,
        revisited_pois=args.revisited_pois,
    )

    for split in args.splits:
        fullraw_stats = build_fullraw(args, split, semantic_map)
        compact_stats = compress_split(compress_args, split, tokenizer)
        summary["splits"][split] = {
            "fullraw": fullraw_stats,
            "compact_main": compact_stats,
        }

    out_path = args.output_dir / f"build_pipeline_top{args.graph_top_k}.json"
    write_json(out_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
