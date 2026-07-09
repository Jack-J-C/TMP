#!/usr/bin/env python3
"""Convert CLSPRec raw check-ins to TMP standard CSV files.

The output schema matches TMP's dataset/NewYork CSV layer, so downstream TMP
evidence and GraphRAG builders can own the actual sample construction.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


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


@dataclass(frozen=True)
class CitySpec:
    city: str
    timezone_minutes: int


CITY_SPECS = {
    "NYC": CitySpec("NYC", -240),
    "PHO": CitySpec("PHO", -420),
    "SIN": CitySpec("SIN", 480),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CLSPRec raw_data CSVs into TMP standard split CSVs.")
    p.add_argument("--clsprec-raw-dir", type=Path, default=Path("/mnt/data/users/yyl/CLSPRec/raw_data"))
    p.add_argument("--output-dir", type=Path, default=Path("/mnt/data/users/yyl/TMP/raw_data"))
    p.add_argument("--cities", nargs="+", default=["NYC", "PHO", "SIN"], choices=sorted(CITY_SPECS))
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--filter-policy",
        choices=["none", "clsprec-static7"],
        default="none",
        help="Optional CLSPRec-style validity filter before TMP splitting.",
    )
    p.add_argument("--min-seq-len", type=int, default=3)
    p.add_argument("--min-seq-num", type=int, default=3)
    p.add_argument("--min-short-term-len", type=int, default=5)
    p.add_argument("--pre-seq-window-days", type=int, default=7)
    p.add_argument("--min-long-term-count", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def category_token(value: str) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or "UNKNOWN"


def split_user_trajectories(traj_meta: pd.DataFrame, train_ratio: float, val_ratio: float) -> Dict[str, str]:
    split_by_traj: Dict[str, str] = {}
    for _, group in traj_meta.sort_values("start_time").groupby("user_id", sort=False):
        trajs = group["trajectory_id"].tolist()
        n = len(trajs)
        if n == 1:
            train_n, val_n = 1, 0
        elif n == 2:
            train_n, val_n = 1, 0
        elif n < 10:
            train_n, val_n = max(1, n - 2), 1
        else:
            train_n = max(1, int(n * train_ratio))
            val_n = max(1, int(n * val_ratio))
            if train_n + val_n >= n:
                val_n = max(0, n - train_n - 1)
        for idx, traj_id in enumerate(trajs):
            if idx < train_n:
                split = "train"
            elif idx < train_n + val_n:
                split = "val"
            else:
                split = "test"
            split_by_traj[str(traj_id)] = split
    return split_by_traj


def build_graph_x(train_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for poi_id, group in train_df.groupby("POI_id", sort=False):
        valid = group[group["coord_available"].astype(int) == 1]
        row = valid.iloc[0] if len(valid) else group.iloc[0]
        rows.append(
            {
                "node_name/poi_id": poi_id,
                "checkin_cnt": int(len(group)),
                "poi_catid": str(row["POI_catid"]),
                "poi_catid_code": int(row["POI_catid_code"]),
                "poi_catname": str(row["POI_catname"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "coord_available": int(row["coord_available"]),
            }
        )
    return pd.DataFrame(rows)


def apply_clsprec_static7_filter(tmp: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Keep TMP daily trajectories that satisfy CLSPRec's static-day sample validity rules."""
    traj_meta = (
        tmp.groupby(["user_id", "trajectory_id"], sort=False)
        .agg(start_time=("_local_dt", "min"), rows=("POI_id", "size"))
        .reset_index()
    )
    traj_meta["date"] = traj_meta["start_time"].dt.date
    valid_daily = traj_meta[traj_meta["rows"] >= args.min_seq_len].copy()
    valid_counts = valid_daily.groupby("user_id").size()
    valid_users = set(valid_counts[valid_counts >= args.min_seq_num].index)
    valid_daily = valid_daily[valid_daily["user_id"].isin(valid_users)].copy()

    eligible_trajs: set[str] = set()
    eligible_users: set[str] = set()
    for user_id, group in valid_daily.sort_values("start_time").groupby("user_id", sort=False):
        previous = []
        for row in group.itertuples(index=False):
            if int(row.rows) >= args.min_short_term_len:
                start_date = row.date - pd.Timedelta(days=args.pre_seq_window_days)
                long_terms = [prev for prev in previous if start_date <= prev.date <= row.date]
                if len(long_terms) >= args.min_long_term_count:
                    eligible_trajs.add(str(row.trajectory_id))
                    eligible_users.add(str(user_id))
            previous.append(row)

    filtered = tmp[tmp["trajectory_id"].isin(eligible_trajs)].copy()
    stats = {
        "policy": "clsprec-static7",
        "description": (
            "TMP-format daily trajectories are filtered using CLSPRec's validity rules, "
            "without copying CLSPRec pickle preprocessing, feature reindexing, or random sample split."
        ),
        "rules": {
            "daily_trajectory_unit": "same user_id + same local date",
            "min_seq_len": int(args.min_seq_len),
            "min_seq_num": int(args.min_seq_num),
            "min_short_term_len": int(args.min_short_term_len),
            "pre_seq_window_days": int(args.pre_seq_window_days),
            "min_long_term_count": int(args.min_long_term_count),
            "consecutive_duplicate_removal": "not applied; CLSPRec code comments mention it, but the released implementation keeps all visits",
            "poi_frequency_filter": "not applied",
            "user_checkin_frequency_filter": "not applied",
        },
        "before": {
            "rows": int(len(tmp)),
            "users": int(tmp["user_id"].nunique()),
            "pois": int(tmp["POI_id"].nunique()),
            "trajectories": int(tmp["trajectory_id"].nunique()),
        },
        "valid_daily_after_min_seq_len": int((traj_meta["rows"] >= args.min_seq_len).sum()),
        "users_after_min_seq_num": int(len(valid_users)),
        "valid_daily_after_user_filter": int(len(valid_daily)),
        "eligible_target_trajectories": int(len(eligible_trajs)),
        "eligible_target_users": int(len(eligible_users)),
        "after": {
            "rows": int(len(filtered)),
            "users": int(filtered["user_id"].nunique()),
            "pois": int(filtered["POI_id"].nunique()),
            "trajectories": int(filtered["trajectory_id"].nunique()),
        },
        "tmp_split_note": "Eligible target trajectories are split by TMP per-user chronological train/val/test ratios.",
    }
    return filtered, stats


def write_filtering_rules(output_dir: Path, args: argparse.Namespace, summary: dict) -> None:
    if args.filter_policy != "clsprec-static7":
        return
    lines = [
        "# CLSPRec-style Filtering Rules",
        "",
        "This directory stores CLSPRec raw check-ins converted to TMP standard CSVs.",
        "Only the CLSPRec filtering rules are reused; CLSPRec pickle preprocessing, feature reindexing, and random sample split are not copied.",
        "",
        "Rules applied before TMP splitting:",
        "",
        f"- Daily trajectory unit: same `user_id` and same local date.",
        f"- Keep daily trajectories with length >= `{args.min_seq_len}`.",
        f"- Keep users with at least `{args.min_seq_num}` valid daily trajectories.",
        f"- A target/current daily trajectory is eligible when its length >= `{args.min_short_term_len}`.",
        f"- The eligible target/current trajectory must have at least `{args.min_long_term_count}` previous valid daily trajectories within the previous `{args.pre_seq_window_days}` days.",
        "- No explicit POI frequency threshold is applied.",
        "- No explicit user check-in frequency threshold is applied.",
        "- Consecutive duplicate visit removal is not applied, matching the released CLSPRec implementation behavior.",
        "",
        "TMP-specific handling:",
        "",
        "- Eligible target trajectories are still written as TMP standard CSV rows.",
        "- Train/val/test split uses TMP per-user chronological ratios from this converter, not CLSPRec's random shuffled sample split.",
        "",
        "Per-city summary:",
        "",
        "| city | rows after filter | users | POIs | eligible trajectories | train traj | val traj | test traj |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for city, stats in summary.items():
        filt = stats.get("filtering", {})
        after = filt.get("after", {})
        splits = stats.get("splits", {})
        lines.append(
            f"| {city} | {after.get('rows', 0)} | {after.get('users', 0)} | {after.get('pois', 0)} | "
            f"{filt.get('eligible_target_trajectories', 0)} | "
            f"{splits.get('train', {}).get('trajectories', 0)} | "
            f"{splits.get('val', {}).get('trajectories', 0)} | "
            f"{splits.get('test', {}).get('trajectories', 0)} |"
        )
    (output_dir / "FILTERING_RULES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_city(spec: CitySpec, args: argparse.Namespace) -> dict:
    src = args.clsprec_raw_dir / f"{spec.city}_checkin_with_active_regionId.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = args.output_dir / spec.city
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"{spec.city}_{split}.csv" for split in ("train", "val", "test")]
    expected += [out_dir / "graph_X.csv", out_dir / "metadata.json"]
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError(f"{out_dir} already has converted files; pass --overwrite")

    raw = pd.read_csv(src)
    raw = raw.copy()
    raw["Local_Time_True"] = pd.to_datetime(raw["Local_Time_True"], errors="coerce")
    if raw["Local_Time_True"].isna().any():
        raise ValueError(f"{src} has unparseable Local_Time_True rows: {int(raw['Local_Time_True'].isna().sum())}")
    raw = raw.sort_values(["UserId", "Local_Time_True", "VenueId"], kind="mergesort").reset_index(drop=True)

    venue_values = sorted(str(x) for x in raw["VenueId"].astype(str).unique())
    venue_to_tmp = {venue: f"v{idx}" for idx, venue in enumerate(venue_values)}
    cat_values = sorted(str(x) for x in raw["Category"].astype(str).unique())
    cat_to_code = {cat: idx for idx, cat in enumerate(cat_values)}

    local_dt = raw["Local_Time_True"]
    timezone = pd.to_numeric(raw.get("TimeZone"), errors="coerce").fillna(spec.timezone_minutes).astype(int)
    utc_dt = local_dt - pd.to_timedelta(timezone, unit="m")
    lat = pd.to_numeric(raw["Latitude"], errors="coerce")
    lon = pd.to_numeric(raw["Longitude"], errors="coerce")
    coord_available = (lat.notna() & lon.notna()).astype(int)
    lat = lat.fillna(0.0).astype(float)
    lon = lon.fillna(0.0).astype(float)
    minutes = local_dt.dt.hour * 60 + local_dt.dt.minute
    city_min_time = local_dt.min()
    local_date = local_dt.dt.strftime("%Y%m%d")

    tmp = pd.DataFrame(
        {
            "user_id": raw["UserId"].astype(str),
            "POI_id": raw["VenueId"].astype(str).map(venue_to_tmp),
            "POI_catid": raw["L1_Category"].astype(str),
            "POI_catid_code": raw["Category"].astype(str).map(cat_to_code).astype(int),
            "POI_catname": raw["Category"].astype(str),
            "latitude": lat,
            "longitude": lon,
            "coord_available": coord_available,
            "timezone": timezone,
            "UTC_time": utc_dt.dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "local_time": local_dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": local_dt.dt.dayofweek.astype(int),
            "norm_in_day_time": minutes / 1440.0,
            "trajectory_id": spec.city + "_" + raw["UserId"].astype(str) + "_" + local_date,
            "norm_day_shift": (local_dt.dt.normalize() - city_min_time.normalize()).dt.days.astype(float),
            "norm_relative_time": (local_dt - city_min_time).dt.total_seconds() / 3600.0,
            "_local_dt": local_dt,
        }
    )
    tmp = tmp.sort_values(["user_id", "_local_dt", "trajectory_id"], kind="mergesort").reset_index(drop=True)

    filtering_stats = {"policy": "none"}
    if args.filter_policy == "clsprec-static7":
        tmp, filtering_stats = apply_clsprec_static7_filter(tmp, args)

    traj_meta = (
        tmp.groupby(["user_id", "trajectory_id"], sort=False)
        .agg(start_time=("_local_dt", "min"), rows=("POI_id", "size"))
        .reset_index()
    )
    split_by_traj = split_user_trajectories(traj_meta, args.train_ratio, args.val_ratio)
    tmp["_split"] = tmp["trajectory_id"].map(split_by_traj)

    stats = {
        "city": spec.city,
        "source": str(src),
        "rows": int(len(tmp)),
        "users": int(tmp["user_id"].nunique()),
        "pois": int(tmp["POI_id"].nunique()),
        "categories": int(tmp["POI_catname"].nunique()),
        "trajectories": int(tmp["trajectory_id"].nunique()),
        "split_policy": {
            "unit": "per-user daily trajectory",
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": round(1.0 - float(args.train_ratio) - float(args.val_ratio), 6),
        },
        "splits": {},
        "poi_id_mapping": "POI_id uses deterministic v{index} over sorted raw VenueId values.",
        "category_fields": {
            "POI_catid": "CLSPRec L1_Category",
            "POI_catid_code": "deterministic integer code over fine Category",
            "POI_catname": "CLSPRec fine Category",
        },
        "filtering": filtering_stats,
    }
    for split in ("train", "val", "test"):
        split_df = tmp[tmp["_split"] == split].copy()
        clean = split_df[STANDARD_COLUMNS]
        clean.to_csv(out_dir / f"{spec.city}_{split}.csv", index=False)
        traj_sizes = split_df.groupby("trajectory_id").size() if len(split_df) else pd.Series(dtype=int)
        stats["splits"][split] = {
            "rows": int(len(split_df)),
            "users": int(split_df["user_id"].nunique()),
            "pois": int(split_df["POI_id"].nunique()),
            "trajectories": int(split_df["trajectory_id"].nunique()),
            "trajectories_len_ge2": int((traj_sizes >= 2).sum()) if len(traj_sizes) else 0,
            "trajectories_len_ge5": int((traj_sizes >= 5).sum()) if len(traj_sizes) else 0,
        }

    graph_x = build_graph_x(tmp[tmp["_split"] == "train"])
    graph_x.to_csv(out_dir / "graph_X.csv", index=False)
    stats["graph_x"] = {
        "rows": int(len(graph_x)),
        "train_pois": int(graph_x["node_name/poi_id"].nunique()) if len(graph_x) else 0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for city in args.cities:
        stats = convert_city(CITY_SPECS[city], args)
        summary[city] = stats
        print(json.dumps({"city": city, "splits": stats["splits"], "graph_x": stats["graph_x"]}, ensure_ascii=False, indent=2))
    (args.output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_filtering_rules(args.output_dir, args, summary)
    print(f"wrote {args.output_dir / 'conversion_summary.json'}")


if __name__ == "__main__":
    main()
