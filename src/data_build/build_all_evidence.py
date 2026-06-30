#!/usr/bin/env python3
"""Build LLM4POI-style evidence JSONL files from pre-split Massive-STEPS CSVs.

This script is intentionally data-preparation only. It does not train or load
any model. It keeps the train/validation/test split supplied by TCPP and builds
natural-language evidence files for downstream LLM prediction.
"""
from __future__ import annotations

import sys
import os as _os

_PROJ_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import argparse
import json
import math
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


@dataclass(frozen=True)
class CitySpec:
    city: str
    prefix: str
    input_dir: Path
    train_file: str
    val_file: str
    test_file: str
    timezone_minutes: int


@dataclass(frozen=True)
class GeoIndex:
    poi_ids: np.ndarray
    categories: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    tree: BallTree | None


DEFAULT_CITIES = {
    "NewYork": CitySpec(
        city="NewYork",
        prefix="NY",
        input_dir=Path("/mnt/data/yyl/TCPP/Massive-STEPS-New-York"),
        train_file="new_york_checkins_train.csv",
        val_file="new_york_checkins_validation.csv",
        test_file="new_york_checkins_test.csv",
        timezone_minutes=-240,
    ),
    "Moscow": CitySpec(
        city="Moscow",
        prefix="MO",
        input_dir=Path("/mnt/data/yyl/TCPP/Massive-STEPS-Moscow"),
        train_file="moscow_checkins_train.csv",
        val_file="moscow_checkins_validation.csv",
        test_file="moscow_checkins_test.csv",
        timezone_minutes=180,
    ),
    "SaoPaulo": CitySpec(
        city="SaoPaulo",
        prefix="SP",
        input_dir=Path("/mnt/data/yyl/TCPP/Massive-STEPS-Sao-Paulo"),
        train_file="sao_paulo_checkins_train.csv",
        val_file="sao_paulo_checkins_validation.csv",
        test_file="sao_paulo_checkins_test.csv",
        timezone_minutes=-180,
    ),
}


STANDARD_COLUMNS = [
    "user_id",
    "POI_id",
    "POI_catid",
    "POI_catid_code",
    "POI_catname",
    "latitude",
    "longitude",
    "coord_available",
    "timezone",
    "UTC_time",
    "local_time",
    "day_of_week",
    "norm_in_day_time",
    "trajectory_id",
    "norm_day_shift",
    "norm_relative_time",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build three evidence JSONL files and merged LLM evidence JSONL.")
    p.add_argument("--cities", nargs="+", default=["NewYork", "Moscow", "SaoPaulo"], choices=sorted(DEFAULT_CITIES))
    p.add_argument("--tcpp-root", type=Path, default=Path("/mnt/data/yyl/TCPP"))
    p.add_argument("--dataset-dir", type=Path, default=Path(_os.path.join(_PROJ_ROOT, "dataset")))
    p.add_argument("--output-root", type=Path, default=Path(_os.path.join(_PROJ_ROOT, "retrieval_assets")))
    p.add_argument("--poi-threshold", type=int, default=9)
    p.add_argument("--user-threshold", type=int, default=9)
    p.add_argument("--min-traj-len", type=int, default=2)
    p.add_argument("--history-limit", type=int, default=50)
    p.add_argument("--current-limit", type=int, default=50)
    p.add_argument("--transition-top-k", type=int, default=20)
    p.add_argument("--geo-top-k", type=int, default=20)
    p.add_argument("--candidate-limit", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_city_spec(base: CitySpec, tcpp_root: Path) -> CitySpec:
    """Resolve Massive-STEPS input paths from the user-supplied TCPP root."""
    dir_name = base.input_dir.name
    input_dir = tcpp_root / dir_name
    return replace(base, input_dir=input_dir)


def _normalize_poi_id(value) -> str:
    text = str(value)
    return text if text.startswith("v") else f"v{text}"


def _coord_fields(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid = lat.notna() & lon.notna()
    return lat.fillna(0.0).astype(float), lon.fillna(0.0).astype(float), valid.astype(int)


def _format_time(dt: pd.Series) -> pd.Series:
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def load_split(path: Path, city: CitySpec, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "trail_id",
        "user_id",
        "venue_id",
        "venue_category",
        "venue_category_id",
        "venue_category_id_code",
        "latitude",
        "longitude",
        "timestamp",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    local_dt = pd.to_datetime(df["timestamp"], errors="coerce")
    if local_dt.isna().any():
        raise ValueError(f"{path} has {int(local_dt.isna().sum())} unparseable timestamps")
    utc_dt = local_dt - pd.Timedelta(minutes=city.timezone_minutes)
    lat, lon, coord_available = _coord_fields(df)
    minutes = local_dt.dt.hour * 60 + local_dt.dt.minute
    relative_hours = (local_dt - local_dt.min()).dt.total_seconds() / 3600.0

    out = pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str),
            "POI_id": df["venue_id"].map(_normalize_poi_id),
            "POI_catid": df["venue_category_id"].fillna("Unknown").astype(str),
            "POI_catid_code": pd.to_numeric(df["venue_category_id_code"], errors="coerce").fillna(-1).astype(int),
            "POI_catname": df["venue_category"].fillna("Unknown").astype(str),
            "latitude": lat,
            "longitude": lon,
            "coord_available": coord_available,
            "timezone": int(city.timezone_minutes),
            "UTC_time": utc_dt.dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "local_time": _format_time(local_dt),
            "day_of_week": local_dt.dt.dayofweek.astype(int),
            "norm_in_day_time": minutes / 1440.0,
            "trajectory_id": df["trail_id"].astype(str),
            "norm_day_shift": 0.0,
            "norm_relative_time": relative_hours,
            "_local_dt": local_dt,
            "_split": split,
        }
    )
    return out.sort_values(["user_id", "_local_dt", "trajectory_id"], kind="mergesort")


def filter_splits(
    splits: Dict[str, pd.DataFrame],
    poi_threshold: int,
    user_threshold: int,
    min_traj_len: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    train = splits["train"]
    stats = {}
    poi_counts = train.groupby("POI_id").size()
    user_counts = train.groupby("user_id").size()
    train_pois = set(poi_counts[poi_counts >= poi_threshold].index) if poi_threshold > 0 else set(poi_counts.index)
    train_users = set(user_counts[user_counts >= user_threshold].index) if user_threshold > 0 else set(user_counts.index)
    stats["train_pois_total"] = int(len(poi_counts))
    stats["train_users_total"] = int(len(user_counts))
    stats["allowed_train_pois"] = int(len(train_pois))
    stats["allowed_train_users"] = int(len(train_users))

    filtered = {}
    for split, df in splits.items():
        before = len(df)
        cur = df[df["POI_id"].isin(train_pois) & df["user_id"].isin(train_users)].copy()
        after_vocab = len(cur)
        if min_traj_len > 1:
            sizes = cur.groupby(["user_id", "trajectory_id"]).size()
            valid_trajs = set(sizes[sizes >= min_traj_len].index)
            row_keys = cur[["user_id", "trajectory_id"]].apply(tuple, axis=1)
            cur = cur[row_keys.isin(valid_trajs)].copy()
        cur = cur.sort_values(["user_id", "_local_dt", "trajectory_id"], kind="mergesort")
        traj_count = cur[["user_id", "trajectory_id"]].drop_duplicates().shape[0]
        filtered[split] = cur
        stats[split] = {
            "rows_before": int(before),
            "rows_after_vocab_filter": int(after_vocab),
            "rows_after_traj_filter": int(len(cur)),
            "users": int(cur["user_id"].nunique()),
            "pois": int(cur["POI_id"].nunique()),
            "trajectories": int(traj_count),
        }
    return filtered, stats


def build_graph_x(train_df: pd.DataFrame) -> pd.DataFrame:
    graph = (
        train_df.groupby("POI_id", sort=False)
        .agg(
            checkin_cnt=("POI_id", "size"),
            poi_catid=("POI_catid", "first"),
            poi_catid_code=("POI_catid_code", "first"),
            poi_catname=("POI_catname", "first"),
            coord_available=("coord_available", "max"),
        )
        .reset_index()
        .rename(columns={"POI_id": "node_name/poi_id"})
    )
    coords = []
    for poi_id, group in train_df.groupby("POI_id", sort=False):
        valid = group[group["coord_available"].astype(int) == 1]
        row = valid.iloc[0] if len(valid) else group.iloc[0]
        coords.append(
            {
                "node_name/poi_id": poi_id,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
        )
    graph = graph.merge(pd.DataFrame(coords), on="node_name/poi_id", how="left")
    return graph[
        [
            "node_name/poi_id",
            "checkin_cnt",
            "poi_catid",
            "poi_catid_code",
            "poi_catname",
            "latitude",
            "longitude",
            "coord_available",
        ]
    ]


def build_geo_index(graph_x: pd.DataFrame) -> GeoIndex:
    valid = graph_x[graph_x["coord_available"].astype(int) == 1].copy()
    latitudes = valid["latitude"].astype(float).to_numpy()
    longitudes = valid["longitude"].astype(float).to_numpy()
    coords_rad = np.radians(np.column_stack([latitudes, longitudes])) if len(valid) else np.empty((0, 2))
    return GeoIndex(
        poi_ids=valid["node_name/poi_id"].astype(str).to_numpy(),
        categories=valid["poi_catname"].astype(str).to_numpy(),
        latitudes=latitudes,
        longitudes=longitudes,
        tree=BallTree(coords_rad, metric="haversine") if len(valid) else None,
    )


def build_poi_category_map(graph_x: pd.DataFrame) -> Dict[str, str]:
    return {
        str(row["node_name/poi_id"]): str(row["poi_catname"])
        for _, row in graph_x.iterrows()
    }


def build_transition_index(train_df: pd.DataFrame) -> Dict[str, Dict[str, Counter]]:
    poi_to_next: Dict[str, Counter] = defaultdict(Counter)
    cat_to_next: Dict[str, Counter] = defaultdict(Counter)
    for _, group in train_df.groupby(["user_id", "trajectory_id"], sort=False):
        group = group.sort_values("_local_dt", kind="mergesort")
        records = group[["POI_id", "POI_catname"]].to_dict("records")
        for prev, nxt in zip(records, records[1:]):
            poi_to_next[str(prev["POI_id"])][str(nxt["POI_id"])] += 1
            cat_to_next[str(prev["POI_catname"])][str(nxt["POI_id"])] += 1
    return {"poi_to_next": poi_to_next, "cat_to_next": cat_to_next}


def transition_candidates(
    sample: dict,
    transition_index: Dict[str, Dict[str, Counter]],
    poi_category: Dict[str, str],
    limit: int,
) -> List[dict]:
    cur = sample["current_trajectory"]
    if not cur:
        return []
    last = cur[-1]
    target_id = str(sample["target"]["poi_id"])
    split = sample["split"]
    counters = [
        ("same_last_poi", transition_index["poi_to_next"].get(str(last["poi_id"]), Counter())),
        ("same_last_category", transition_index["cat_to_next"].get(str(last["category"]), Counter())),
    ]
    rows = []
    seen = set()
    for source, counter in counters:
        adjusted = counter.copy()
        if split == "train" and adjusted.get(target_id, 0) > 0:
            adjusted[target_id] -= 1
            if adjusted[target_id] <= 0:
                del adjusted[target_id]
        for poi_id, count in adjusted.most_common(limit):
            if poi_id in seen:
                continue
            seen.add(poi_id)
            rows.append(
                {
                    "poi_id": str(poi_id),
                    "category": poi_category.get(str(poi_id), "Unknown"),
                    "count": int(count),
                    "source": source,
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def build_user_history(train_df: pd.DataFrame) -> Dict[str, dict]:
    history = {}
    for user_id, group in train_df.sort_values("_local_dt").groupby("user_id", sort=False):
        entries = []
        times = []
        for _, row in group.iterrows():
            item = {
                "time": str(row["local_time"]),
                "datetime": row["_local_dt"],
                "poi_id": str(row["POI_id"]),
                "category": str(row["POI_catname"]),
                "dow": int(row["day_of_week"]),
                "slot": int(float(row["norm_in_day_time"]) * 48),
                "coord_available": bool(int(row["coord_available"])),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
            entries.append(item)
            times.append(row["_local_dt"])
        history[str(user_id)] = {"entries": entries, "times": times}
    return history


def history_before(history_map: Dict[str, dict], user_id: str, cutoff, limit: int) -> List[dict]:
    user_hist = history_map.get(str(user_id))
    if not user_hist:
        return []
    idx = bisect_left(user_hist["times"], cutoff)
    return user_hist["entries"][:idx][-limit:]


def trajectory_samples(
    df: pd.DataFrame,
    split: str,
    city: str,
    history_map: Dict[str, dict],
    history_limit: int,
    current_limit: int,
) -> List[dict]:
    rows = []
    for (user_key, traj_id), group in df.groupby(["user_id", "trajectory_id"], sort=False):
        group = group.sort_values("_local_dt", kind="mergesort")
        if len(group) < 2:
            continue
        user_id = str(group["user_id"].iloc[0])
        target = group.iloc[-1]
        current = group.iloc[:-1].tail(current_limit)
        cutoff = current["_local_dt"].iloc[0]
        history = history_before(history_map, user_id, cutoff, history_limit)
        sample_id = f"{city}-{split}-{len(rows):06d}"
        current_entries = [
            {
                "time": str(row.local_time),
                "poi_id": str(row.POI_id),
                "category": str(row.POI_catname),
                "dow": int(row.day_of_week),
                "slot": int(float(row.norm_in_day_time) * 48),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "coord_available": bool(int(row.coord_available)),
            }
            for row in current.itertuples(index=False)
        ]
        rows.append(
            {
                "sample_id": sample_id,
                "city": city,
                "split": split,
                "user_id": user_id,
                "trajectory_id": str(traj_id),
                "history": history,
                "current_trajectory": current_entries,
                "target": {
                    "time": str(target["local_time"]),
                    "poi_id": str(target["POI_id"]),
                    "category": str(target["POI_catname"]),
                    "latitude": float(target["latitude"]),
                    "longitude": float(target["longitude"]),
                    "coord_available": bool(int(target["coord_available"])),
                    "dow": int(target["day_of_week"]),
                    "slot": int(float(target["norm_in_day_time"]) * 48),
                },
            }
        )
    return rows


def top_counter(entries: Iterable[dict], key: str, limit: int) -> List[dict]:
    counts = Counter(str(e.get(key, "Unknown")) for e in entries)
    return [{"value": value, "count": int(count)} for value, count in counts.most_common(limit)]


def sequence_summary(sample: dict) -> str:
    cur = sample["current_trajectory"]
    if not cur:
        return "The current trajectory has no usable prefix records."
    cats = [e["category"] for e in cur]
    start = cur[0]
    end = cur[-1]
    top_cats = ", ".join([x["value"] for x in top_counter(cur, "category", 3)])
    return (
        f"The current trajectory contains {len(cur)} check-ins, moving from {start['category']} "
        f"to {end['category']}. Frequent categories in this trajectory are {top_cats}."
    )


def preference_summary(sample: dict) -> str:
    hist = sample["history"]
    if not hist:
        return "No same-user training history is available before this trajectory."
    top_cats = top_counter(hist, "category", 4)
    cat_text = ", ".join([f"{x['value']} ({x['count']})" for x in top_cats])
    slots = Counter(int(e["slot"]) for e in hist)
    common_slots = ", ".join([str(slot) for slot, _ in slots.most_common(4)])
    return (
        f"The same-user history has {len(hist)} check-ins. Frequent categories are {cat_text}. "
        f"Common local time slots are {common_slots}."
    )


def geo_evidence(sample: dict, geo_index: GeoIndex, geo_top_k: int) -> Tuple[str, List[dict]]:
    cur = sample["current_trajectory"]
    if not cur:
        return "No current POI is available for geographic evidence.", []
    last = cur[-1]
    if not last["coord_available"]:
        return "The last current POI has missing/private coordinates, so geographic evidence is limited.", []
    if len(geo_index.poi_ids) == 0 or geo_index.tree is None:
        return "No train POI with valid coordinates is available.", []
    query_k = min(len(geo_index.poi_ids), geo_top_k + 1)
    query = np.radians(np.array([[float(last["latitude"]), float(last["longitude"])]]))
    dist_rad, ind = geo_index.tree.query(query, k=query_k)
    distances = dist_rad[0] * 6371.0088
    indices = ind[0]
    candidates = [
        {
            "poi_id": str(geo_index.poi_ids[i]),
            "category": str(geo_index.categories[i]),
            "distance_km": round(float(dist), 4),
            "coord_available": True,
        }
        for i, dist in zip(indices, distances)
        if str(geo_index.poi_ids[i]) != last["poi_id"] and math.isfinite(float(dist))
    ][:geo_top_k]
    if not candidates:
        return "No nearby POI with valid coordinates is available.", []
    cat_counts = Counter(c["category"] for c in candidates)
    cat_text = ", ".join([cat for cat, _ in cat_counts.most_common(4)])
    summary = (
        f"The last current POI has valid coordinates. The nearest {len(candidates)} train POIs "
        f"are mainly in categories: {cat_text}."
    )
    return summary, candidates


def geo_cache_key(sample: dict, geo_top_k: int) -> Tuple:
    cur = sample["current_trajectory"]
    if not cur:
        return ("", False, 0.0, 0.0, geo_top_k)
    last = cur[-1]
    return (
        str(last["poi_id"]),
        bool(last["coord_available"]),
        round(float(last["latitude"]), 7),
        round(float(last["longitude"]), 7),
        int(geo_top_k),
    )


def build_prompt(seq_ev: dict, geo_ev: dict, pref_ev: dict) -> str:
    lines = [
        "Predict the next POI for the user.",
        "Use the sequence, geographic, and long-term preference evidence.",
        "Return exactly one POI id.",
        "",
        "[Sequence Evidence]",
        seq_ev["sequence_summary"],
        "Current trajectory:",
    ]
    for item in seq_ev["current_trajectory"]:
        lines.append(f"- {item['time']} | {item['poi_id']} | {item['category']} | dow={item['dow']} | slot={item['slot']}")
    if seq_ev["transition_candidates"]:
        lines.append("Historical transition candidates from train:")
        for item in seq_ev["transition_candidates"][:10]:
            lines.append(
                f"- {item['poi_id']} | {item['category']} | count={item['count']} | source={item['source']}"
            )
    lines.extend(["", "[Geographic Evidence]", geo_ev["geo_summary"]])
    for item in geo_ev["nearby_pois"][:10]:
        lines.append(f"- {item['poi_id']} | {item['category']} | distance_km={item['distance_km']}")
    lines.extend(["", "[Preference Evidence]", pref_ev["preference_summary"]])
    for item in pref_ev["top_categories"]:
        lines.append(f"- category={item['category']} count={item['count']}")
    for item in pref_ev["revisited_pois"][:10]:
        lines.append(f"- revisited_poi={item['poi_id']} count={item['count']}")
    return "\n".join(lines)


def evidence_for_sample(
    sample: dict,
    geo_index: GeoIndex,
    transition_index: Dict[str, Dict[str, Counter]],
    poi_category: Dict[str, str],
    transition_top_k: int,
    geo_top_k: int,
    candidate_limit: int,
    geo_cache: Dict[Tuple, Tuple[str, List[dict]]],
) -> Tuple[dict, dict, dict, dict]:
    seq_ev = {
        "sample_id": sample["sample_id"],
        "city": sample["city"],
        "split": sample["split"],
        "user_id": sample["user_id"],
        "trajectory_id": sample["trajectory_id"],
        "target_poi_id": sample["target"]["poi_id"],
        "target_category": sample["target"]["category"],
        "sequence_summary": sequence_summary(sample),
        "current_trajectory": sample["current_trajectory"],
        "transition_candidates": transition_candidates(sample, transition_index, poi_category, transition_top_k),
    }
    cache_key = geo_cache_key(sample, geo_top_k)
    if cache_key not in geo_cache:
        geo_cache[cache_key] = geo_evidence(sample, geo_index, geo_top_k)
    geo_summary, nearby = geo_cache[cache_key]
    geo_ev = {
        "sample_id": sample["sample_id"],
        "last_poi_id": sample["current_trajectory"][-1]["poi_id"] if sample["current_trajectory"] else "",
        "geo_summary": geo_summary,
        "nearby_pois": nearby,
    }
    hist = sample["history"]
    top_categories = [{"category": x["value"], "count": x["count"]} for x in top_counter(hist, "category", 8)]
    revisited = [{"poi_id": x["value"], "count": x["count"]} for x in top_counter(hist, "poi_id", 20)]
    pref_ev = {
        "sample_id": sample["sample_id"],
        "user_id": sample["user_id"],
        "history_count": len(hist),
        "preference_summary": preference_summary(sample),
        "top_categories": top_categories,
        "revisited_pois": revisited,
    }
    candidate_ids = []
    for item in seq_ev["transition_candidates"]:
        candidate_ids.append(item["poi_id"])
    for item in nearby:
        candidate_ids.append(item["poi_id"])
    for item in revisited:
        candidate_ids.append(item["poi_id"])
    for item in sample["current_trajectory"]:
        candidate_ids.append(item["poi_id"])
    candidate_ids = list(dict.fromkeys(candidate_ids))[:candidate_limit]
    merged = {
        "sample_id": sample["sample_id"],
        "city": sample["city"],
        "split": sample["split"],
        "user_id": sample["user_id"],
        "trajectory_id": sample["trajectory_id"],
        "target_poi_id": sample["target"]["poi_id"],
        "target_category": sample["target"]["category"],
        "candidate_poi_ids": candidate_ids,
        "sequence_evidence": seq_ev,
        "geo_evidence": geo_ev,
        "preference_evidence": pref_ev,
        "prompt": build_prompt(seq_ev, geo_ev, pref_ev),
        "answer": sample["target"]["poi_id"],
    }
    return seq_ev, geo_ev, pref_ev, merged


def write_jsonl(path: Path, rows: Iterable[dict], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_standard_dataset(city_dir: Path, city: CitySpec, splits: Dict[str, pd.DataFrame], graph_x: pd.DataFrame, stats: Dict, overwrite: bool) -> None:
    city_dir.mkdir(parents=True, exist_ok=True)
    graph_x.to_csv(city_dir / "graph_X.csv", index=False)
    for split, df in splits.items():
        out_name = f"{city.prefix}_{split}.csv" if split != "val" else f"{city.prefix}_val.csv"
        clean = df[STANDARD_COLUMNS].copy()
        clean.to_csv(city_dir / out_name, index=False)
    metadata = {
        "city": city.city,
        "source": str(city.input_dir),
        "split_source": "TCPP pre-split train/validation/test files",
        "protocol": "LLM4POI-style last-point trajectory prediction",
        "filtering": stats,
    }
    with (city_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def process_city(city: CitySpec, args: argparse.Namespace) -> Dict:
    start_time = time.time()
    print(f"[{city.city}] loading TCPP splits from {city.input_dir}", flush=True)
    split_paths = {
        "train": city.input_dir / city.train_file,
        "val": city.input_dir / city.val_file,
        "test": city.input_dir / city.test_file,
    }
    for split, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{city.city} {split} file not found: {path}")
    raw = {split: load_split(path, city, split) for split, path in split_paths.items()}
    print(f"[{city.city}] loaded splits in {time.time() - start_time:.1f}s", flush=True)
    filtered, filter_stats = filter_splits(raw, args.poi_threshold, args.user_threshold, args.min_traj_len)
    print(f"[{city.city}] filtered splits in {time.time() - start_time:.1f}s", flush=True)
    graph_x = build_graph_x(filtered["train"])
    geo_index = build_geo_index(graph_x)
    poi_category = build_poi_category_map(graph_x)
    transition_index = build_transition_index(filtered["train"])
    history_map = build_user_history(filtered["train"])
    print(f"[{city.city}] built graph/history in {time.time() - start_time:.1f}s", flush=True)

    dataset_city_dir = args.dataset_dir / city.city
    write_standard_dataset(dataset_city_dir, city, filtered, graph_x, filter_stats, args.overwrite)
    print(f"[{city.city}] wrote standard CSV files in {time.time() - start_time:.1f}s", flush=True)

    out_dir = args.output_root / city.city / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    city_stats = {
        "city": city.city,
        "filtering": filter_stats,
        "evidence": {},
    }

    for split, df in filtered.items():
        split_start = time.time()
        print(f"[{city.city}] building {split} evidence...", flush=True)
        samples = trajectory_samples(df, split, city.city, history_map, args.history_limit, args.current_limit)
        seq_rows, geo_rows, pref_rows, merged_rows = [], [], [], []
        geo_cache: Dict[Tuple, Tuple[str, List[dict]]] = {}
        for sample in samples:
            seq_ev, geo_ev, pref_ev, merged = evidence_for_sample(
                sample,
                geo_index,
                transition_index,
                poi_category,
                args.transition_top_k,
                args.geo_top_k,
                args.candidate_limit,
                geo_cache,
            )
            seq_rows.append(seq_ev)
            geo_rows.append(geo_ev)
            pref_rows.append(pref_ev)
            merged_rows.append(merged)
        counts = {
            "sequence": write_jsonl(out_dir / f"sequence_evidence_{split}.jsonl", seq_rows, args.overwrite),
            "geo": write_jsonl(out_dir / f"geo_evidence_{split}.jsonl", geo_rows, args.overwrite),
            "preference": write_jsonl(out_dir / f"preference_evidence_{split}.jsonl", pref_rows, args.overwrite),
            "stage2": write_jsonl(out_dir / f"stage2_llm_evidence_{split}.jsonl", merged_rows, args.overwrite),
        }
        target_in_candidates = sum(1 for row in merged_rows if row["target_poi_id"] in set(row["candidate_poi_ids"]))
        city_stats["evidence"][split] = {
            "samples": len(samples),
            "files": counts,
            "target_candidate_recall": float(target_in_candidates / len(samples)) if samples else 0.0,
            "avg_candidates": float(np.mean([len(r["candidate_poi_ids"]) for r in merged_rows])) if merged_rows else 0.0,
            "avg_history_count": float(np.mean([len(s["history"]) for s in samples])) if samples else 0.0,
            "avg_current_len": float(np.mean([len(s["current_trajectory"]) for s in samples])) if samples else 0.0,
        }
        print(
            f"[{city.city}] wrote {split} evidence: {len(samples)} samples "
            f"in {time.time() - split_start:.1f}s",
            flush=True,
        )

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(city_stats, f, ensure_ascii=False, indent=2)
    return city_stats


def main() -> None:
    args = parse_args()
    all_stats = {}
    for city_name in args.cities:
        city = resolve_city_spec(DEFAULT_CITIES[city_name], args.tcpp_root)
        stats = process_city(city, args)
        all_stats[city_name] = stats
        print(json.dumps({
            "city": city_name,
            "train_samples": stats["evidence"]["train"]["samples"],
            "val_samples": stats["evidence"]["val"]["samples"],
            "test_samples": stats["evidence"]["test"]["samples"],
            "train_target_candidate_recall": stats["evidence"]["train"]["target_candidate_recall"],
        }, ensure_ascii=False, indent=2))
    summary_path = args.output_root / "evidence_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
