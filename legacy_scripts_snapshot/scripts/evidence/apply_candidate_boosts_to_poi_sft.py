#!/usr/bin/env python3
"""Apply LoRA-C supplemental candidates to POI SFT JSONL prompts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


CANDIDATE_BLOCK_RE = re.compile(r"\nCandidate POIs:\n(?P<candidates>.*?)(?=\n\nOutput format:)", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge candidate boosts into POI SFT rows and prompt text.")
    p.add_argument("--data", type=Path, required=True, help="stage*.jsonl POI SFT data.")
    p.add_argument("--boosts", type=Path, required=True, help="LoRA-C generated supplemental candidates.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-limit", type=int, default=96)
    p.add_argument("--supplemental-limit", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_boosts(path: Path, supplemental_limit: int) -> Dict[str, List[str]]:
    boosts: Dict[str, List[str]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} contains boost row without sample_id")
        vals: List[str] = []
        for value in row.get("supplemental_poi_ids") or []:
            poi = str(value)
            if poi.startswith("v") and poi[1:].isdigit() and poi not in vals:
                vals.append(poi)
            if len(vals) >= supplemental_limit:
                break
        boosts[sid] = vals
    return boosts


def merge_candidates(original: List[Any], supplemental: List[str], candidate_limit: int) -> List[str]:
    merged: List[str] = []
    for value in [*original, *supplemental]:
        poi = str(value)
        if poi and poi not in merged:
            merged.append(poi)
        if len(merged) >= candidate_limit:
            break
    return merged


def replace_candidate_block(prompt: str, candidates: List[str], sample_id: str) -> str:
    replacement = "\nCandidate POIs:\n" + ", ".join(candidates)
    if not CANDIDATE_BLOCK_RE.search(prompt):
        raise ValueError(f"{sample_id} prompt does not contain a replaceable Candidate POIs block")
    return CANDIDATE_BLOCK_RE.sub(replacement, prompt, count=1)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data)
    boosts = load_boosts(args.boosts, args.supplemental_limit)
    out_rows: List[Dict[str, Any]] = []
    original_hits = 0
    boosted_hits = 0
    rows_with_added = 0
    added_total = 0

    for row in rows:
        sid = str(row.get("sample_id") or "")
        original = [str(x) for x in row.get("candidate_poi_ids") or []]
        supplemental = [poi for poi in boosts.get(sid, []) if poi not in set(original)]
        merged = merge_candidates(original, supplemental, args.candidate_limit)
        added = max(0, len(merged) - len(original))
        rows_with_added += int(added > 0)
        added_total += added
        target = str((row.get("target") or {}).get("poi_id") or "")
        original_hits += int(bool(target) and target in set(original))
        boosted_hits += int(bool(target) and target in set(merged))

        new_row = dict(row)
        new_row["candidate_poi_ids"] = merged
        new_row["input_prompt"] = replace_candidate_block(str(row.get("input_prompt") or ""), merged, sid)
        messages = [dict(msg) for msg in row.get("messages") or []]
        if len(messages) >= 2:
            messages[1]["content"] = replace_candidate_block(str(messages[1].get("content") or ""), merged, sid)
        new_row["messages"] = messages
        metadata = dict(row.get("metadata") or {})
        metadata["candidate_count"] = len(merged)
        metadata["target_in_candidates"] = bool(target and target in set(merged))
        new_row["metadata"] = metadata
        new_row["candidate_boost"] = {
            "source": str(args.boosts),
            "supplemental_poi_ids": supplemental,
            "original_candidate_count": len(original),
            "boosted_candidate_count": len(merged),
        }
        out_rows.append(new_row)

    written = write_jsonl(args.output, out_rows, args.overwrite)
    report = {
        "input_rows": len(rows),
        "written": written,
        "boost_rows": len(boosts),
        "rows_with_added_candidates": rows_with_added,
        "avg_added_candidates": round(added_total / len(rows), 4) if rows else 0.0,
        "candidate_limit": args.candidate_limit,
        "original_target_in_candidates": original_hits,
        "boosted_target_in_candidates": boosted_hits,
        "original_target_in_candidates_ratio": round(original_hits / len(rows), 6) if rows else 0.0,
        "boosted_target_in_candidates_ratio": round(boosted_hits / len(rows), 6) if rows else 0.0,
        "absolute_recall_gain": round((boosted_hits - original_hits) / len(rows), 6) if rows else 0.0,
    }
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
