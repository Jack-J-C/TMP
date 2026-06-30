#!/usr/bin/env python3
"""Build semantic-ID optimized RAW POI SFT data.

The source files are the existing poi_sft/stage1_*_raw.jsonl records. This
script keeps the same supervision target and message schema, but rewrites the
user prompt into a more compact semantic-id format:

    poi=v6525 | category=Coffee Shop -> v6525|sid=NYC::COFFEE_SHOP::...

POIs with missing coordinates are preserved with GEO_UNK/geo=unk instead of
being dropped, because missing geo is a real signal in this dataset.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SYSTEM_PROMPT = "You are a next-POI predictor. Output only JSON."
OUTPUT_FORMAT_BLOCK = 'Output format:\n{"next_poi_id":"<poi_id>"}'
SEMANTIC_PROMPT_VERSION = "raw_semantic_id_v1"

POI_RE = re.compile(r"poi=(v\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build semantic-ID optimized RAW POI SFT data.")
    p.add_argument(
        "--semantic-map",
        type=Path,
        default=Path("retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl"),
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("retrieval_assets/NewYork/poi_sft"),
        help="Directory containing stage1_{train,val,test}_raw.jsonl.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("retrieval_assets/NewYork/semantic_poi_sft"),
    )
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Deprecated for RAW semantic prompts. Candidate sets are intentionally omitted.",
    )
    p.add_argument(
        "--candidate-semantic-limit",
        type=int,
        default=0,
        help="Deprecated for RAW semantic prompts. Candidate sets are intentionally omitted.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc


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


def load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        poi_id = str(row.get("poi_id") or "")
        if poi_id:
            out[poi_id] = row
    return out


def compact_geo(row: Dict[str, Any]) -> Tuple[str, bool]:
    if row.get("geo_cell") == "GEO_UNK" or row.get("latitude") is None or row.get("longitude") is None:
        return "UNK", True
    try:
        return f"{float(row['latitude']):.2f},{float(row['longitude']):.2f}", False
    except (TypeError, ValueError):
        return "UNK", True


def semantic_code(semantic_map: Dict[str, Dict[str, Any]], poi_id: str, stats: Counter[str]) -> str:
    row = semantic_map.get(poi_id)
    if row is None:
        stats["poi_missing_semantic_id"] += 1
        return "SEM_UNK"
    category = str(row.get("category_token") or "CAT_UNK")
    local_id = str(row.get("semantic_id") or "").split("::")[-1] or "P0000"
    geo, missing_geo = compact_geo(row)
    code = f"{category}@{geo}#{local_id}"
    if missing_geo:
        stats["poi_geo_unk_mentions"] += 1
        return f"{code}|geo=unk"
    return code


def parse_fields(line: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for part in line.split("|"):
        text = part.strip()
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def format_trajectory_line(line: str, semantic_map: Dict[str, Dict[str, Any]], stats: Counter[str]) -> str | None:
    match = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
    if not match:
        return None
    idx, rest = match.groups()
    fields = parse_fields(rest)
    poi = fields.get("poi")
    if not poi:
        return None
    parts = [f"{idx}. {poi}", semantic_code(semantic_map, poi, stats)]
    if fields.get("weekday"):
        parts.append(fields["weekday"])
    if fields.get("slot"):
        parts.append(f"s{fields['slot']}")
    if fields.get("delta_minutes"):
        parts.append(f"+{fields['delta_minutes']}m")
    stats["trajectory_lines"] += 1
    return "|".join(parts)


def format_transition_line(line: str, semantic_map: Dict[str, Dict[str, Any]], stats: Counter[str]) -> str | None:
    text = line.strip()
    if not text.startswith("- "):
        return None
    fields = parse_fields(text[2:])
    poi = fields.get("poi")
    if not poi:
        return None
    parts = [f"- {poi}", semantic_code(semantic_map, poi, stats)]
    if fields.get("source"):
        parts.append(f"src={fields['source']}")
    if fields.get("count"):
        parts.append(f"c{fields['count']}")
    stats["transition_lines"] += 1
    return "|".join(parts)


def format_nearby_line(line: str, semantic_map: Dict[str, Dict[str, Any]], stats: Counter[str]) -> str | None:
    text = line.strip()
    if not text.startswith("- "):
        return None
    fields = parse_fields(text[2:])
    poi = fields.get("poi")
    if not poi:
        return None
    parts = [f"- {poi}", semantic_code(semantic_map, poi, stats)]
    if fields.get("distance_km"):
        parts.append(f"d={fields['distance_km']}km")
    stats["nearby_lines"] += 1
    return "|".join(parts)


def format_category_line(line: str) -> str | None:
    text = line.strip()
    if not text.startswith("- "):
        return None
    fields = parse_fields(text[2:])
    category = fields.get("category")
    count = fields.get("count")
    if not category:
        return None
    return f"- {category}|c{count}" if count else f"- {category}"


def format_revisited_line(line: str, semantic_map: Dict[str, Dict[str, Any]], stats: Counter[str]) -> str | None:
    text = line.strip()
    if not text.startswith("- "):
        return None
    fields = parse_fields(text[2:])
    poi = fields.get("poi")
    if not poi:
        return None
    parts = [f"- {poi}", semantic_code(semantic_map, poi, stats)]
    if fields.get("count"):
        parts.append(f"c{fields['count']}")
    stats["revisited_lines"] += 1
    return "|".join(parts)


def section_for_line(line: str, current: str) -> str:
    text = line.strip()
    if text == "Current trajectory:":
        return "trajectory"
    if text == "Transition evidence:":
        return "transition"
    if text == "Geographic evidence near the last POI:":
        return "nearby"
    if text == "Short-term pattern:":
        return "short_pattern"
    if text == "Long-term preference evidence:":
        return "preference"
    if text == "top_categories:":
        return "top_categories"
    if text == "revisited_pois:":
        return "revisited"
    if text == "Candidate POIs:":
        return "candidates"
    if not text or text.endswith(":") or text.startswith("Output format:"):
        if current != "candidates":
            return ""
    return current


def rewrite_prompt(
    row: Dict[str, Any],
    semantic_map: Dict[str, Dict[str, Any]],
    candidate_limit: int,
    candidate_semantic_limit: int,
    stats: Counter[str],
) -> str:
    original = str(row.get("input_prompt") or "")
    lines = original.splitlines()
    out: List[str] = [
        "[ROUTE=RAW_SEM]",
        "",
        f"Task: predict the next POI in {row.get('city') or 'the city'}.",
        "SemCode=CATEGORY@lat,lon#local; @UNK means missing coordinates.",
        'Output only {"next_poi_id":"<poi_id>"} using original POI id.',
    ]
    section = ""
    skip_next_candidate_line = False
    saw_output = False

    for line in lines:
        text = line.strip()
        if text.startswith("[ROUTE="):
            continue
        if text in {"Task:", "Constraints:"}:
            section = ""
            continue
        if text.startswith("Predict the next POI") or text.startswith("The candidate list"):
            continue
        if text.startswith("Return exactly one POI id"):
            continue
        if text.startswith("The user is anonymized.") or text.startswith("Do not rely"):
            continue
        if text.startswith("Use only movement sequence"):
            continue
        if text == "Output format:":
            saw_output = True
            break

        new_section = section_for_line(line, section)
        if new_section != section:
            section = new_section
            if text == "Current trajectory:":
                out.extend(["", "Trajectory:"])
                continue
            if text == "Transition evidence:":
                out.extend(["", "Transitions:"])
                continue
            if text == "Geographic evidence near the last POI:":
                out.extend(["", "Nearby:"])
                continue
            if text == "Short-term pattern:":
                continue
            if text == "Long-term preference evidence:":
                out.extend(["", "Preference:"])
                continue
            if text == "Candidate POIs:":
                stats["candidate_sections_omitted"] += 1
                skip_next_candidate_line = True
                continue

        if skip_next_candidate_line:
            skip_next_candidate_line = False
            continue

        if section == "short_pattern":
            continue
        if section == "preference":
            if text.startswith("history_count="):
                out.append(f"hist={text.split('=', 1)[1]}")
            continue
        if section == "trajectory":
            formatted = format_trajectory_line(line, semantic_map, stats)
            out.append(formatted if formatted is not None else line)
        elif section == "transition":
            formatted = format_transition_line(line, semantic_map, stats)
            out.append(formatted if formatted is not None else line)
        elif section == "nearby":
            formatted = format_nearby_line(line, semantic_map, stats)
            out.append(formatted if formatted is not None else line)
        elif section == "top_categories":
            formatted = format_category_line(line)
            out.append(formatted if formatted is not None else line)
        elif section == "revisited":
            formatted = format_revisited_line(line, semantic_map, stats)
            out.append(formatted if formatted is not None else line)
        elif section != "candidates":
            out.append(line)

    if not saw_output:
        stats["prompt_without_output_block"] += 1
    out.extend(["", *OUTPUT_FORMAT_BLOCK.splitlines()])
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out).strip())


def rewrite_record(
    row: Dict[str, Any],
    semantic_map: Dict[str, Dict[str, Any]],
    candidate_limit: int,
    candidate_semantic_limit: int,
    stats: Counter[str],
) -> Dict[str, Any]:
    prompt = rewrite_prompt(row, semantic_map, candidate_limit, candidate_semantic_limit, stats)
    new_row = dict(row)
    new_row["route"] = "RAW_SEM"
    new_row["prompt_version"] = SEMANTIC_PROMPT_VERSION
    new_row["input_prompt"] = prompt
    new_row["refined_prompt"] = None
    messages = list(row.get("messages") or [])
    assistant = messages[2] if len(messages) >= 3 else {"role": "assistant", "content": row.get("target", {})}
    new_row["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        assistant,
    ]
    stats["rows"] += 1
    stats["input_chars"] += len(str(row.get("input_prompt") or ""))
    stats["output_chars"] += len(prompt)
    if "@UNK" in prompt or "geo=unk" in prompt:
        stats["rows_with_geo_unk"] += 1
    return new_row


def build_split(args: argparse.Namespace, split: str, semantic_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    input_path = args.input_dir / f"stage1_{split}_raw.jsonl"
    output_path = args.output_dir / f"stage1_{split}_raw_semantic.jsonl"
    stats: Counter[str] = Counter()

    def rows() -> Iterable[Dict[str, Any]]:
        for row in read_jsonl(input_path):
            yield rewrite_record(row, semantic_map, args.candidate_limit, args.candidate_semantic_limit, stats)

    written = write_jsonl(output_path, rows(), args.overwrite)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "written": written,
        "prompt_version": SEMANTIC_PROMPT_VERSION,
        "candidate_limit": args.candidate_limit,
        "candidate_semantic_limit": args.candidate_semantic_limit,
        "counts": dict(stats),
        "avg_input_chars": round(stats["input_chars"] / written, 2) if written else 0.0,
        "avg_output_chars": round(stats["output_chars"] / written, 2) if written else 0.0,
        "char_ratio": round(stats["output_chars"] / stats["input_chars"], 6) if stats["input_chars"] else 0.0,
        "rows_with_geo_unk_ratio": round(stats["rows_with_geo_unk"] / written, 6) if written else 0.0,
    }
    output_path.with_suffix(output_path.suffix + ".stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def semantic_map_stats(semantic_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    for row in semantic_map.values():
        counts["total"] += 1
        if row.get("geo_cell") == "GEO_UNK" or row.get("latitude") is None or row.get("longitude") is None:
            counts["missing_geo"] += 1
        else:
            counts["has_geo"] += 1
    return {
        "total": counts["total"],
        "has_geo": counts["has_geo"],
        "missing_geo": counts["missing_geo"],
        "missing_geo_ratio": round(counts["missing_geo"] / counts["total"], 6) if counts["total"] else 0.0,
    }


def main() -> None:
    args = parse_args()
    semantic_map = load_semantic_map(args.semantic_map)
    summaries = {
        "semantic_map": str(args.semantic_map),
        "semantic_map_stats": semantic_map_stats(semantic_map),
        "splits": {},
    }
    for split in args.splits:
        summaries["splits"][split] = build_split(args, split, semantic_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
