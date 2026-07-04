#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GraphRAG candidate coverage from JSONL outputs.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[50, 100, 200, 300, 400, 500])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_ranks(path: Path) -> list[int | None]:
    ranks: list[int | None] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rank = row.get("target_rank")
            ranks.append(int(rank) if rank is not None else None)
    return ranks


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        ranks = read_ranks(path)
        hit_ranks = [rank for rank in ranks if rank is not None]
        row = {
            "file": str(path),
            "rows": len(ranks),
            "max_candidate_k": max((rank for rank in hit_ranks), default=0),
            "hit_ratio": round(len(hit_ranks) / len(ranks), 6) if ranks else 0.0,
            "rank_when_hit": {
                "mean": round(mean(hit_ranks), 4) if hit_ranks else 0.0,
                "p50": percentile(hit_ranks, 0.50),
                "p90": percentile(hit_ranks, 0.90),
                "p95": percentile(hit_ranks, 0.95),
                "max": max(hit_ranks) if hit_ranks else None,
            },
        }
        for k in args.ks:
            row[f"hit@{k}"] = round(
                sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks),
                6,
            ) if ranks else 0.0
        rows.append(row)

    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
