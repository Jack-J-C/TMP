#!/usr/bin/env python3
"""Validate decision-style refined prompts before Stage2b training."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


POI_ID_RE = re.compile(r"\bv\d+\b")
ABS_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s*\d{4}\b"
    r"|\b20\d{2}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
COORD_RE = re.compile(r"\b(latitude|longitude|coordinates?|lat|lon|lng)\b", re.IGNORECASE)
FORBIDDEN_RE = re.compile(
    r"\b(user_id|sample_id|trajectory_id|target|label|answer|ground truth|correct answer|gold answer)\b",
    re.IGNORECASE,
)
REQUIRED_PATTERNS = {
    "evidence_verdict": re.compile(r"Evidence verdict:", re.IGNORECASE),
    "priority_hypotheses": re.compile(r"Priority hypotheses:", re.IGNORECASE),
    "weak_or_noisy_cues": re.compile(r"Weak or noisy cues:", re.IGNORECASE),
    "final_hint": re.compile(r"Final hint:", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate decision-style refined prompt JSONL.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-clean", type=Path, default=None)
    p.add_argument("--output-removed", type=Path, default=None)
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=220)
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            row = dict(row)
            row.pop("_line_no", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_text(row: Dict[str, Any]) -> str:
    return str(row.get("distilled_prompt") or "").strip()


def validate_row(row: Dict[str, Any], min_words: int, max_words: int) -> List[str]:
    text = prompt_text(row)
    reasons: List[str] = []
    words = text.split()
    if len(words) < min_words:
        reasons.append("too_short")
    if len(words) > max_words:
        reasons.append("too_long")
    for name, pattern in REQUIRED_PATTERNS.items():
        if not pattern.search(text):
            reasons.append(f"missing_{name}")
    if POI_ID_RE.search(text):
        reasons.append("poi_id")
    if ABS_DATE_RE.search(text):
        reasons.append("absolute_date")
    if COORD_RE.search(text):
        reasons.append("coord_word")
    if FORBIDDEN_RE.search(text):
        reasons.append("forbidden_word")
    return reasons


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    clean = []
    removed = []
    reason_counts: Dict[str, int] = {}
    seen = set()
    duplicate_ids = 0
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if sid in seen:
            duplicate_ids += 1
            reasons = ["duplicate_id"]
        else:
            seen.add(sid)
            reasons = validate_row(row, args.min_words, args.max_words)
        if reasons:
            removed.append({**row, "remove_reasons": reasons})
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            clean.append(row)

    if args.output_clean:
        write_jsonl(args.output_clean, clean)
    if args.output_removed:
        write_jsonl(args.output_removed, removed)

    report = {
        "input": str(args.input),
        "rows": len(rows),
        "clean_rows": len(clean),
        "removed_rows": len(removed),
        "duplicate_ids": duplicate_ids,
        "reason_counts": reason_counts,
        "output_clean": str(args.output_clean) if args.output_clean else None,
        "output_removed": str(args.output_removed) if args.output_removed else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
