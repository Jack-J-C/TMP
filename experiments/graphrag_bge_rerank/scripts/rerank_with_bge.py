#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank GraphRAG candidates with BGE reranker.")
    parser.add_argument("--candidates", type=Path, required=True, help="GraphRAG large-K candidate JSONL.")
    parser.add_argument("--joined-parquet", type=Path, required=True, help="Joined parquet containing raw_text/refined_text.")
    parser.add_argument("--model-name-or-path", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--top-in", type=int, default=500)
    parser.add_argument("--top-out", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional quick probe limit.")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N rows.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-refined", action="store_true")
    parser.add_argument(
        "--query-mode",
        choices=["raw", "trajectory_transition", "trajectory_transition_preference"],
        default="raw",
        help="How much source context to expose to the BGE query encoder.",
    )
    parser.add_argument(
        "--passage-mode",
        choices=["basic", "graph"],
        default="basic",
        help="basic uses candidate identity/category only; graph also exposes GraphRAG score/sources.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_joined(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(path, columns=["sample_id", "raw_text", "refined_text"])
    return {str(row["sample_id"]): row for row in df.to_dict("records")}


def extract_raw_sections(raw_text: str, section_names: list[str]) -> str:
    lines = raw_text.splitlines()
    wanted = {f"{name}:" for name in section_names}
    stop_headers = {"Trajectory:", "Transitions:", "Nearby:", "Preference:"}
    out: list[str] = []
    keep = False
    for line in lines:
        stripped = line.strip()
        if stripped in stop_headers:
            keep = stripped in wanted
            if keep:
                out.extend(["", stripped])
            continue
        if keep:
            out.append(line)
    return "\n".join(x for x in out if x.strip()).strip()


def build_query(joined_row: dict[str, Any], include_refined: bool, query_mode: str) -> str:
    raw_text = str(joined_row.get("raw_text") or "")
    if query_mode == "trajectory_transition":
        query = extract_raw_sections(raw_text, ["Trajectory", "Transitions"])
    elif query_mode == "trajectory_transition_preference":
        query = extract_raw_sections(raw_text, ["Trajectory", "Transitions", "Preference"])
    else:
        query = raw_text

    parts = [query or raw_text]
    if include_refined:
        refined = str(joined_row.get("refined_text") or "")
        if refined:
            parts.extend(["", "Refined evidence:", refined])
    return "\n".join(parts)


def build_passage(poi_id: str, details: dict[str, Any], passage_mode: str) -> str:
    sources = ",".join(str(x) for x in details.get("sources") or [])
    basic = (
        f"Candidate POI: {poi_id}\n"
        f"semantic_id: {details.get('semantic_id')}\n"
        f"category: {details.get('category')}"
    )
    if passage_mode == "basic":
        return basic
    return (
        f"{basic}\n"
        f"graph_score: {details.get('score')}\n"
        f"graph_sources: {sources}"
    )


@torch.inference_mode()
def score_pairs(
    tokenizer,
    model,
    pairs: list[tuple[str, str]],
    batch_size: int,
    max_length: int,
    device: str,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        features = tokenizer(
            [q for q, _ in batch],
            [p for _, p in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(**features, return_dict=True).logits
        if logits.ndim == 2 and logits.shape[-1] == 1:
            batch_scores = logits[:, 0]
        elif logits.ndim == 2:
            batch_scores = logits[:, -1]
        else:
            batch_scores = logits.reshape(-1)
        scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())
    return scores


def summarize(rows: list[dict[str, Any]], top_out: int) -> dict[str, Any]:
    total = len(rows)
    hit = 0
    ranks = []
    for row in rows:
        target = str((row.get("target") or {}).get("poi_id") or "")
        ranked = [str(x) for x in row.get("bge_candidate_poi_ids") or []]
        rank = next((idx for idx, poi in enumerate(ranked[:top_out], 1) if poi == target), None)
        if rank is not None:
            hit += 1
            ranks.append(rank)
    return {
        "rows": total,
        "top_out": top_out,
        f"hit@{top_out}": round(hit / total, 6) if total else 0.0,
        "rank_when_hit": {
            "mean": round(sum(ranks) / len(ranks), 4) if ranks else 0.0,
            "max": max(ranks) if ranks else None,
        },
    }


def main() -> None:
    args = parse_args()
    joined = load_joined(args.joined_parquet)
    rows = read_jsonl(args.candidates)
    if args.max_rows is not None:
        rows = rows[: max(0, int(args.max_rows))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    model.to(args.device)
    model.eval()

    out_rows = []
    start_time = time.time()
    for row_idx, row in enumerate(rows, 1):
        sid = str(row.get("sample_id") or "")
        joined_row = joined.get(sid)
        if joined_row is None:
            raise ValueError(f"Missing joined row for sample_id={sid}")
        query = build_query(joined_row, args.include_refined, args.query_mode)
        candidates = [str(x) for x in row.get("candidate_poi_ids") or []][: args.top_in]
        details = row.get("candidate_details") or {}
        pairs = [(query, build_passage(poi, details.get(poi) or {}, args.passage_mode)) for poi in candidates]
        bge_scores = score_pairs(
            tokenizer=tokenizer,
            model=model,
            pairs=pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        ranked = sorted(zip(candidates, bge_scores), key=lambda item: (-item[1], item[0]))
        out_row = {
            "sample_id": sid,
            "target": row.get("target"),
            "source_candidate_count": len(candidates),
            "source_target_rank": row.get("target_rank"),
            "bge_candidate_poi_ids": [poi for poi, _ in ranked[: args.top_out]],
            "bge_scores": {poi: round(score, 6) for poi, score in ranked[: args.top_out]},
        }
        target = str((row.get("target") or {}).get("poi_id") or "")
        out_row["bge_target_rank"] = next(
            (idx for idx, poi in enumerate(out_row["bge_candidate_poi_ids"], 1) if poi == target),
            None,
        )
        out_rows.append(out_row)
        if row_idx == 1 or row_idx % max(1, args.log_every) == 0:
            elapsed = max(1e-6, time.time() - start_time)
            rows_per_sec = row_idx / elapsed
            eta = (len(rows) - row_idx) / rows_per_sec if rows_per_sec > 0 else 0.0
            print(
                f"processed={row_idx}/{len(rows)} "
                f"pairs={row_idx * len(candidates)}/{len(rows) * len(candidates)} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )

    write_jsonl(args.output, out_rows)
    report = {
        "candidates": str(args.candidates),
        "joined_parquet": str(args.joined_parquet),
        "model_name_or_path": args.model_name_or_path,
        "top_in": args.top_in,
        "top_out": args.top_out,
        "include_refined": args.include_refined,
        "query_mode": args.query_mode,
        "passage_mode": args.passage_mode,
        **summarize(out_rows, args.top_out),
    }
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
