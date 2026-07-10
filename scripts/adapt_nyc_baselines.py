#!/usr/bin/env python3
"""Adapt TMP NYC splits to CLSPRec and GETNext data formats.

This script changes only the data layer. It keeps the baseline model code and
architectures untouched by writing files in their native input formats.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


MIN_SEQ_LEN = 3
MIN_SHORT_TERM_LEN = 5
MIN_LONG_TERM_COUNT = 2
PRE_SEQ_WINDOW_DAYS = 7


@dataclass
class Trajectory:
    trajectory_id: str
    split: str
    user_id: int
    date: Any
    rows: pd.DataFrame


@dataclass
class ClsprecBuildConfig:
    min_short_term_len: int
    min_long_term_count: int
    pre_seq_window_days: int | None
    max_history_count: int | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adapt TMP NYC data to CLSPRec and GETNext baseline formats.")
    p.add_argument("--tmp-city-dir", type=Path, default=Path("dataset_clsprec/NYC"))
    p.add_argument("--city", default="NYC")
    p.add_argument("--clsprec-out", type=Path, default=Path("/mnt/data/users/yyl/CLSPRec/processed_data/tmp_nyc"))
    p.add_argument("--getnext-out", type=Path, default=Path("/mnt/data/users/yyl/GETNext/dataset/TMP_NYC"))
    p.add_argument("--targets", nargs="+", choices=["getnext", "clsprec"], default=["getnext", "clsprec"])
    p.add_argument("--clsprec-min-short-term-len", type=int, default=MIN_SHORT_TERM_LEN)
    p.add_argument("--clsprec-min-long-term-count", type=int, default=MIN_LONG_TERM_COUNT)
    p.add_argument(
        "--clsprec-pre-seq-window-days",
        type=int,
        default=PRE_SEQ_WINDOW_DAYS,
        help="Use <=0 to allow all previous trajectories before optional max-history truncation.",
    )
    p.add_argument("--clsprec-max-history-count", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_split(tmp_city_dir: Path, city: str, split: str) -> pd.DataFrame:
    suffix = "val" if split == "valid" else split
    path = tmp_city_dir / f"{city}_{suffix}.csv"
    df = pd.read_csv(path)
    df["split"] = split
    df["local_dt"] = pd.to_datetime(df["local_time"])
    df["local_date"] = df["local_dt"].dt.date
    df = df.sort_values(["user_id", "local_dt", "trajectory_id"]).reset_index(drop=True)
    return df


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_getnext_trajectory_ids(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """GETNext parses user_id as trajectory_id.split('_')[0]."""
    out = df.copy()

    def normalize(row: pd.Series) -> str:
        user = str(row["user_id"])
        traj = str(row["trajectory_id"])
        if traj.split("_", 1)[0] == user:
            return traj
        city_prefix = f"{city}_"
        if traj.startswith(city_prefix):
            stripped = traj[len(city_prefix):]
            if stripped.split("_", 1)[0] == user:
                return stripped
        return f"{user}_{traj}"

    out["trajectory_id"] = out.apply(normalize, axis=1)
    return out


def write_getnext_data(tmp_city_dir: Path, city: str, out_dir: Path, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    ensure_clean_dir(out_dir, overwrite=True)
    name_map = {"train": "train", "valid": "val", "test": "test"}
    getnext_splits = {split: normalize_getnext_trajectory_ids(df, city) for split, df in splits.items()}
    for split, df in splits.items():
        getnext_splits[split].drop(columns=["split", "local_dt", "local_date"], errors="ignore").to_csv(
            out_dir / f"{city}_{name_map[split]}.csv", index=False
        )

    graph_x_src = tmp_city_dir / "graph_X.csv"
    graph_x = pd.read_csv(graph_x_src)
    graph_x.to_csv(out_dir / "graph_X.csv", index=False)

    poi_ids = graph_x["node_name/poi_id"].astype(str).tolist()
    poi_to_idx = {poi: idx for idx, poi in enumerate(poi_ids)}
    adj = np.zeros((len(poi_ids), len(poi_ids)), dtype=np.float32)
    train_df = getnext_splits["train"]
    for _, traj_df in train_df.groupby("trajectory_id", sort=False):
        seq = [str(x) for x in traj_df.sort_values("local_dt")["POI_id"].tolist()]
        for src, dst in zip(seq[:-1], seq[1:]):
            if src in poi_to_idx and dst in poi_to_idx:
                adj[poi_to_idx[src], poi_to_idx[dst]] += 1.0
    np.savetxt(out_dir / "graph_A.csv", adj, delimiter=",", fmt="%.0f")

    return {
        "out_dir": str(out_dir),
        "graph_nodes": len(poi_ids),
        "graph_edges": int(np.count_nonzero(adj)),
        "graph_edge_weight_sum": int(adj.sum()),
        "trajectory_id_rule": "trajectory_id starts with user_id because GETNext uses trajectory_id.split('_')[0].",
        "files": sorted(p.name for p in out_dir.iterdir()),
    }


def build_mappings(train_df: pd.DataFrame, graph_x: pd.DataFrame, all_df: pd.DataFrame) -> Dict[str, Any]:
    train_pois = set(train_df["POI_id"].astype(str))
    poi_values = [str(x) for x in graph_x["node_name/poi_id"].tolist() if str(x) in train_pois]
    seen = set(poi_values)
    for poi in train_df["POI_id"].astype(str):
        if poi not in seen:
            poi_values.append(poi)
            seen.add(poi)

    cat_values = sorted(train_df["POI_catid_code"].dropna().astype(int).unique().tolist())
    user_values = sorted(train_df["user_id"].dropna().astype(int).unique().tolist())
    hour_values = list(range(24))
    day_values = [False, True]
    date_values = sorted(all_df["local_date"].unique().tolist())
    return {
        "POI": np.asarray(poi_values, dtype=object),
        "cat": np.asarray(cat_values, dtype=object),
        "user": np.asarray(user_values, dtype=object),
        "hour": np.asarray(hour_values, dtype=object),
        "day": np.asarray(day_values, dtype=object),
        "date": np.asarray(date_values, dtype=object),
    }


def index_maps(meta: Dict[str, np.ndarray]) -> Dict[str, Dict[Any, int]]:
    return {name: {value: idx for idx, value in enumerate(values.tolist())} for name, values in meta.items()}


def make_feature_sequence(rows: pd.DataFrame, maps: Dict[str, Dict[Any, int]]) -> List[List[int]]:
    rows = rows.sort_values("local_dt")
    poi = [maps["POI"][str(x)] for x in rows["POI_id"].astype(str)]
    cat = [maps["cat"][int(x)] for x in rows["POI_catid_code"]]
    user = [maps["user"][int(x)] for x in rows["user_id"]]
    hour = [maps["hour"][int(x)] for x in rows["local_dt"].dt.hour]
    day = [maps["day"][bool(x)] for x in (rows["local_dt"].dt.dayofweek > 4)]
    date = [maps["date"][x] for x in rows["local_date"]]
    return [poi, cat, user, hour, day, date]


def valid_trajectory_groups(all_df: pd.DataFrame, maps: Dict[str, Dict[Any, int]]) -> List[Trajectory]:
    valid: List[Trajectory] = []
    for traj_id, traj_df in all_df.groupby("trajectory_id", sort=False):
        traj_df = traj_df.sort_values("local_dt")
        if len(traj_df) < MIN_SEQ_LEN:
            continue
        if not set(traj_df["POI_id"].astype(str)).issubset(maps["POI"]):
            continue
        if not set(traj_df["user_id"].astype(int)).issubset(maps["user"]):
            continue
        valid.append(
            Trajectory(
                trajectory_id=str(traj_id),
                split=str(traj_df["split"].iloc[0]),
                user_id=int(traj_df["user_id"].iloc[0]),
                date=traj_df["local_date"].iloc[0],
                rows=traj_df,
            )
        )
    valid.sort(key=lambda x: (x.user_id, x.date, x.trajectory_id))
    return valid


def build_clsprec_samples(
    all_df: pd.DataFrame,
    meta: Dict[str, np.ndarray],
    config: ClsprecBuildConfig,
) -> tuple[Dict[str, List[Any]], Dict[str, Any]]:
    maps = index_maps(meta)
    trajectories = valid_trajectory_groups(all_df, maps)
    by_user: Dict[int, List[Trajectory]] = {}
    for traj in trajectories:
        by_user.setdefault(traj.user_id, []).append(traj)

    samples = {"train": [], "valid": [], "test": []}
    skipped = {"short_current": 0, "insufficient_history": 0}
    history_lengths = []
    for user_trajs in by_user.values():
        for idx, current in enumerate(user_trajs):
            if len(current.rows) < config.min_short_term_len:
                skipped["short_current"] += 1
                continue
            if config.pre_seq_window_days is None:
                history = list(user_trajs[:idx])
            else:
                current_ts = pd.Timestamp(current.date)
                lower = current_ts - pd.Timedelta(days=config.pre_seq_window_days)
                history = [
                    prev
                    for prev in user_trajs[:idx]
                    if lower <= pd.Timestamp(prev.date) <= current_ts
                ]
            if len(history) < config.min_long_term_count:
                skipped["insufficient_history"] += 1
                continue
            if config.max_history_count is not None and len(history) > config.max_history_count:
                history = history[-config.max_history_count:]
            sample = [make_feature_sequence(prev.rows, maps) for prev in history]
            sample.append(make_feature_sequence(current.rows, maps))
            if current.split in samples:
                samples[current.split].append(sample)
                history_lengths.append(len(history))

    stats = {
        "valid_daily_trajectories": len(trajectories),
        "users": len(by_user),
        "samples": {k: len(v) for k, v in samples.items()},
        "skipped": skipped,
        "build_config": {
            "min_short_term_len": config.min_short_term_len,
            "min_long_term_count": config.min_long_term_count,
            "pre_seq_window_days": config.pre_seq_window_days,
            "max_history_count": config.max_history_count,
        },
        "history_lengths": {
            "min": int(min(history_lengths)) if history_lengths else 0,
            "mean": float(np.mean(history_lengths)) if history_lengths else 0.0,
            "max": int(max(history_lengths)) if history_lengths else 0,
        },
    }
    return samples, stats


def write_clsprec_data(
    out_dir: Path,
    city: str,
    splits: Dict[str, pd.DataFrame],
    graph_x: pd.DataFrame,
    config: ClsprecBuildConfig,
) -> Dict[str, Any]:
    ensure_clean_dir(out_dir, overwrite=True)
    all_df = pd.concat([splits["train"], splits["valid"], splits["test"]], ignore_index=True)
    meta = build_mappings(splits["train"], graph_x, all_df)
    samples, stats = build_clsprec_samples(all_df, meta, config)
    train_valid = samples["train"] + samples["valid"]

    outputs = {
        "train": samples["train"],
        "valid": samples["valid"],
        "test": samples["test"],
        "train_valid": train_valid,
        "meta": {k: v for k, v in meta.items() if k != "date"},
    }
    for suffix, data in outputs.items():
        with (out_dir / f"{city}_{suffix}").open("wb") as f:
            pickle.dump(data, f)

    stats["meta_sizes"] = {k: int(len(v)) for k, v in outputs["meta"].items()}
    stats["out_dir"] = str(out_dir)
    stats["files"] = sorted(p.name for p in out_dir.iterdir())
    return stats


def main() -> None:
    args = parse_args()
    splits = {
        "train": read_split(args.tmp_city_dir, args.city, "train"),
        "valid": read_split(args.tmp_city_dir, args.city, "valid"),
        "test": read_split(args.tmp_city_dir, args.city, "test"),
    }
    graph_x = pd.read_csv(args.tmp_city_dir / "graph_X.csv")

    getnext_stats = None
    clsprec_stats = None
    if "getnext" in args.targets:
        getnext_stats = write_getnext_data(args.tmp_city_dir, args.city, args.getnext_out, splits)
    if "clsprec" in args.targets:
        clsprec_config = ClsprecBuildConfig(
            min_short_term_len=args.clsprec_min_short_term_len,
            min_long_term_count=args.clsprec_min_long_term_count,
            pre_seq_window_days=args.clsprec_pre_seq_window_days if args.clsprec_pre_seq_window_days > 0 else None,
            max_history_count=args.clsprec_max_history_count,
        )
        clsprec_stats = write_clsprec_data(args.clsprec_out, args.city, splits, graph_x, clsprec_config)

    summary = {
        "city": args.city,
        "source": str(args.tmp_city_dir),
        "targets": args.targets,
        "split_rows": {k: int(len(v)) for k, v in splits.items()},
        "split_trajectories": {k: int(v["trajectory_id"].nunique()) for k, v in splits.items()},
        "fairness_note": "Uses TMP NYC train/val/test split; no baseline model architecture changes.",
    }
    outputs = []
    if getnext_stats is not None:
        summary["getnext"] = getnext_stats
        outputs.append(args.getnext_out / "adaptation_summary.json")
    if clsprec_stats is not None:
        summary["clsprec"] = clsprec_stats
        outputs.append(args.clsprec_out / "adaptation_summary.json")
    for out in outputs:
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
