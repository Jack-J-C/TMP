#!/usr/bin/env python3
"""Build stable semantic IDs for POIs.

Semantic IDs encode category and coarse geo cells so small LLMs do not need to
learn arbitrary opaque POI ids only. The mapping is deterministic and derived
from label-free train/evidence POI metadata.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


POI_RE = re.compile(r"^v\d+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build semantic POI ID mapping.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--city-code", default="NYC")
    p.add_argument("--lat-precision", type=int, default=2)
    p.add_argument("--lon-precision", type=int, default=2)
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


def norm_token(text: Any, default: str = "UNK") -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value.upper() if value else default


def poi_ok(value: Any) -> bool:
    text = str(value or "")
    return bool(POI_RE.fullmatch(text))


def collect_item(info: Dict[str, Dict[str, Any]], item: Dict[str, Any]) -> None:
    poi = str(item.get("poi_id") or "")
    if not poi_ok(poi):
        return
    row = info.setdefault(
        poi,
        {
            "poi_id": poi,
            "category_counts": Counter(),
            "latitudes": [],
            "longitudes": [],
        },
    )
    category = str(item.get("category") or "")
    if category and category != "unknown":
        row["category_counts"][category] += 1
    if item.get("coord_available", True):
        try:
            row["latitudes"].append(float(item.get("latitude")))
            row["longitudes"].append(float(item.get("longitude")))
        except (TypeError, ValueError):
            pass


def collect_from_row(info: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> None:
    evidence = row.get("evidence") or {}
    seq = evidence.get("sequence") or {}
    geo = evidence.get("geo") or {}
    pref = evidence.get("preference") or {}
    for item in seq.get("current_trajectory") or []:
        collect_item(info, item)
    for item in seq.get("transition_candidates") or []:
        collect_item(info, item)
    for item in geo.get("nearby_pois") or []:
        collect_item(info, item)
    for item in pref.get("revisited_pois") or []:
        collect_item(info, item)


def mean(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


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


def main() -> None:
    args = parse_args()
    info: Dict[str, Dict[str, Any]] = {}
    for path in args.inputs:
        for row in read_jsonl(path):
            collect_from_row(info, row)

    groups: Dict[tuple[str, str], List[str]] = defaultdict(list)
    rows: Dict[str, Dict[str, Any]] = {}
    for poi, item in info.items():
        category = item["category_counts"].most_common(1)[0][0] if item["category_counts"] else "unknown"
        lat = mean(item["latitudes"])
        lon = mean(item["longitudes"])
        if lat is None or lon is None:
            cell = "GEO_UNK"
        else:
            cell = f"LAT{round(lat, args.lat_precision):.{args.lat_precision}f}_LON{round(lon, args.lon_precision):.{args.lon_precision}f}"
            cell = norm_token(cell)
        cat_token = norm_token(category)
        groups[(cat_token, cell)].append(poi)
        rows[poi] = {
            "poi_id": poi,
            "category": category,
            "category_token": cat_token,
            "geo_cell": cell,
            "latitude": lat,
            "longitude": lon,
        }

    output_rows: List[Dict[str, Any]] = []
    for key, pois in groups.items():
        for idx, poi in enumerate(sorted(pois), 1):
            row = rows[poi]
            row["semantic_id"] = f"{args.city_code}::{row['category_token']}::{row['geo_cell']}::P{idx:04d}"
            output_rows.append(row)
    output_rows.sort(key=lambda x: x["poi_id"])
    written = write_jsonl(args.output, output_rows, args.overwrite)
    stats = {
        "output": str(args.output),
        "written": written,
        "categories": len({row["category_token"] for row in output_rows}),
        "geo_cells": len({row["geo_cell"] for row in output_rows}),
        "city_code": args.city_code,
    }
    args.output.with_suffix(args.output.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
