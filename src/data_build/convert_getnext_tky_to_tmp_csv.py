#!/usr/bin/env python3
"""Convert GETNext TKY CSVs to TMP standard CSVs with CLSPRec-style filtering."""
from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert GETNext TKY data into TMP standard split CSVs.")
    p.add_argument("--getnext-dir", type=Path, default=Path("/mnt/data/users/yyl/GETNext/dataset/TKY"))
    p.add_argument("--output-dir", type=Path, default=Path("/mnt/data/users/yyl/TMP/raw_data/TKY"))
    p.add_argument("--city", default="TKY")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--min-seq-len", type=int, default=3)
    p.add_argument("--min-seq-num", type=int, default=3)
    p.add_argument("--min-short-term-len", type=int, default=5)
    p.add_argument("--pre-seq-window-days", type=int, default=7)
    p.add_argument("--min-long-term-count", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def split_user_trajectories(traj_meta: pd.DataFrame, train_ratio: float, val_ratio: float) -> Dict[str, str]:
    split_by_traj: Dict[str, str] = {}
    for _, group in traj_meta.sort_values("start_time").groupby("user_id", sort=False):
        trajs = group["trajectory_id"].astype(str).tolist()
        n = len(trajs)
        if n <= 2:
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
            split_by_traj[traj_id] = split
    return split_by_traj


def summarize(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()) if len(df) else 0,
        "pois": int(df["POI_id"].nunique()) if len(df) else 0,
        "trajectories": int(df["trajectory_id"].nunique()) if len(df) else 0,
    }


def apply_clsprec_filter(tmp: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
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
    return filtered, {
        "policy": "clsprec-static7",
        "rules": {
            "daily_trajectory_unit": "same user_id + same local date, rebuilt from GETNext local_time",
            "min_seq_len": int(args.min_seq_len),
            "min_seq_num": int(args.min_seq_num),
            "min_short_term_len": int(args.min_short_term_len),
            "pre_seq_window_days": int(args.pre_seq_window_days),
            "min_long_term_count": int(args.min_long_term_count),
            "poi_frequency_filter": "not applied",
            "user_checkin_frequency_filter": "not applied",
            "getnext_original_split": "not reused; TMP chronological per-user split is rebuilt after filtering",
        },
        "before": summarize(tmp),
        "valid_daily_after_min_seq_len": int((traj_meta["rows"] >= args.min_seq_len).sum()),
        "users_after_min_seq_num": int(len(valid_users)),
        "valid_daily_after_user_filter": int(len(valid_daily)),
        "eligible_target_trajectories": int(len(eligible_trajs)),
        "eligible_target_users": int(len(eligible_users)),
        "after": summarize(filtered),
    }


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


def write_filtering_rules(output_dir: Path, stats: dict) -> None:
    f = stats["filtering"]
    splits = stats["splits"]
    after = f["after"]
    lines = [
        "# GETNext TKY Filtering Rules",
        "",
        "Source directory: `/mnt/data/users/yyl/GETNext/dataset/TKY`.",
        "",
        "Only CLSPRec-style trajectory validity filtering is reused.",
        "GETNext's original train/val/test split and graph files are not reused.",
        "",
        "Rules:",
        "",
        "- Daily trajectory unit: same `user_id` and same local date, rebuilt from `local_time`.",
        "- Keep daily trajectories with length >= `3`.",
        "- Keep users with at least `3` valid daily trajectories.",
        "- A target/current daily trajectory is eligible when its length >= `5`.",
        "- The eligible target/current trajectory must have at least `2` previous valid daily trajectories within the previous `7` days.",
        "- No explicit POI frequency threshold is applied.",
        "- No explicit user check-in frequency threshold is applied.",
        "- Train/val/test split uses TMP per-user chronological ratios after filtering.",
        "",
        "Summary:",
        "",
        "| city | rows after filter | users | POIs | eligible trajectories | train traj | val traj | test traj |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {stats['city']} | {after['rows']} | {after['users']} | {after['pois']} | "
            f"{f['eligible_target_trajectories']} | {splits['train']['trajectories']} | "
            f"{splits['val']['trajectories']} | {splits['test']['trajectories']} |"
        ),
    ]
    (output_dir / "FILTERING_RULES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = [args.output_dir / f"{args.city}_{split}.csv" for split in ("train", "val", "test")]
    expected += [args.output_dir / "graph_X.csv", args.output_dir / "metadata.json"]
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError(f"{args.output_dir} already has converted files; pass --overwrite")

    frames = []
    for split in ("train", "val", "test"):
        path = args.getnext_dir / f"{args.city}_{split}.csv"
        df = pd.read_csv(path)
        df["_source_split"] = split
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    local_dt = pd.to_datetime(raw["local_time"], errors="coerce")
    if local_dt.isna().any():
        raise ValueError(f"GETNext TKY has {int(local_dt.isna().sum())} unparseable local_time rows")
    raw["_local_dt"] = local_dt
    raw = raw.sort_values(["user_id", "_local_dt", "POI_id"], kind="mergesort").reset_index(drop=True)
    local_dt = raw["_local_dt"]
    city_min_time = local_dt.min()

    raw_pois = sorted(str(x) for x in raw["POI_id"].astype(str).unique())
    poi_map = {poi: f"v{idx}" for idx, poi in enumerate(raw_pois)}
    lat = pd.to_numeric(raw["latitude"], errors="coerce")
    lon = pd.to_numeric(raw["longitude"], errors="coerce")
    coord_available = (lat.notna() & lon.notna()).astype(int)

    tmp = pd.DataFrame(
        {
            "user_id": raw["user_id"].astype(str),
            "POI_id": raw["POI_id"].astype(str).map(poi_map),
            "POI_catid": raw["POI_catid"].fillna("Unknown").astype(str),
            "POI_catid_code": pd.to_numeric(raw["POI_catid_code"], errors="coerce").fillna(-1).astype(int),
            "POI_catname": raw["POI_catname"].fillna("Unknown").astype(str),
            "latitude": lat.fillna(0.0).astype(float),
            "longitude": lon.fillna(0.0).astype(float),
            "coord_available": coord_available,
            "timezone": pd.to_numeric(raw["timezone"], errors="coerce").fillna(540).astype(int),
            "UTC_time": pd.to_datetime(raw["UTC_time"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "local_time": local_dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": local_dt.dt.dayofweek.astype(int),
            "norm_in_day_time": (local_dt.dt.hour * 60 + local_dt.dt.minute) / 1440.0,
            "trajectory_id": args.city + "_" + raw["user_id"].astype(str) + "_" + local_dt.dt.strftime("%Y%m%d"),
            "norm_day_shift": (local_dt.dt.normalize() - city_min_time.normalize()).dt.days.astype(float),
            "norm_relative_time": (local_dt - city_min_time).dt.total_seconds() / 3600.0,
            "_local_dt": local_dt,
        }
    )
    tmp = tmp.sort_values(["user_id", "_local_dt", "trajectory_id"], kind="mergesort").reset_index(drop=True)
    tmp, filtering_stats = apply_clsprec_filter(tmp, args)

    traj_meta = (
        tmp.groupby(["user_id", "trajectory_id"], sort=False)
        .agg(start_time=("_local_dt", "min"), rows=("POI_id", "size"))
        .reset_index()
    )
    split_by_traj = split_user_trajectories(traj_meta, args.train_ratio, args.val_ratio)
    tmp["_split"] = tmp["trajectory_id"].map(split_by_traj)

    stats = {
        "city": args.city,
        "source": str(args.getnext_dir),
        **summarize(tmp),
        "categories": int(tmp["POI_catname"].nunique()) if len(tmp) else 0,
        "split_policy": {
            "unit": "per-user daily trajectory",
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": round(1.0 - float(args.train_ratio) - float(args.val_ratio), 6),
        },
        "poi_id_mapping": "POI_id uses deterministic v{index} over sorted raw GETNext POI_id values.",
        "filtering": filtering_stats,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_df = tmp[tmp["_split"] == split].copy()
        split_df[STANDARD_COLUMNS].to_csv(args.output_dir / f"{args.city}_{split}.csv", index=False)
        traj_sizes = split_df.groupby("trajectory_id").size() if len(split_df) else pd.Series(dtype=int)
        stats["splits"][split] = {
            "rows": int(len(split_df)),
            "users": int(split_df["user_id"].nunique()) if len(split_df) else 0,
            "pois": int(split_df["POI_id"].nunique()) if len(split_df) else 0,
            "trajectories": int(split_df["trajectory_id"].nunique()) if len(split_df) else 0,
            "trajectories_len_ge2": int((traj_sizes >= 2).sum()) if len(traj_sizes) else 0,
            "trajectories_len_ge5": int((traj_sizes >= 5).sum()) if len(traj_sizes) else 0,
        }
    graph_x = build_graph_x(tmp[tmp["_split"] == "train"])
    graph_x.to_csv(args.output_dir / "graph_X.csv", index=False)
    stats["graph_x"] = {
        "rows": int(len(graph_x)),
        "train_pois": int(graph_x["node_name/poi_id"].nunique()) if len(graph_x) else 0,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "conversion_summary.json").write_text(json.dumps({args.city: stats}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_filtering_rules(args.output_dir, stats)
    return stats


def main() -> None:
    args = parse_args()
    stats = convert(args)
    print(json.dumps({"city": args.city, "splits": stats["splits"], "graph_x": stats["graph_x"]}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_dir / 'conversion_summary.json'}")


if __name__ == "__main__":
    main()
