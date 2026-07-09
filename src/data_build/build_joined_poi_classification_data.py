#!/usr/bin/env python3
"""Build joined train/val data for POI classification.

This pre-joins:
  semantic raw prompt + full refined prompt + GraphRAG TopK evidence

The output is a compact Parquet file by default, so training reads one file per
split instead of repeatedly scanning three large JSONL sources.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build joined POI classification train/val data.")
    p.add_argument("--base-dir", type=Path, default=Path("retrieval_assets/NewYork"))
    p.add_argument("--output-dir", type=Path, default=Path("retrieval_assets/NewYork/joined_poi_classification"))
    p.add_argument("--semantic-map", type=Path, default=None)
    p.add_argument("--graph-top-k", type=int, default=30)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    p.add_argument("--format", choices=["parquet", "jsonl", "both"], default="parquet")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--similar-profile", action="store_true", help="Build compact similar-user semantic profiles for every row.")
    p.add_argument("--similar-profile-model", type=Path, default=Path("models/bge-m3"))
    p.add_argument("--similar-profile-k", type=int, default=20)
    p.add_argument("--similar-profile-min-sim", type=float, default=0.35)
    p.add_argument("--similar-profile-batch-size", type=int, default=64)
    p.add_argument("--similar-profile-search-batch-size", type=int, default=256)
    p.add_argument("--similar-profile-max-length", type=int, default=256)
    p.add_argument("--similar-profile-device", default="cuda")
    p.add_argument(
        "--similar-profile-query-source",
        choices=["user_profile", "recent_profile"],
        default="recent_profile",
        help="BGE query text for neighbor retrieval. Output remains a compact long-term profile aggregate.",
    )
    p.add_argument("--similar-profile-local-files-only", action="store_true")
    p.add_argument(
        "--allow-missing-refined",
        action="store_true",
        help="Allow missing refined prompt files and keep refined_text empty.",
    )
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


def load_by_sample_id(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        if not sid:
            raise ValueError(f"{path} row without sample_id")
        if sid in out:
            raise ValueError(f"{path} duplicate sample_id: {sid}")
        out[sid] = row
    return out


def load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["poi_id"]): row for row in read_jsonl(path)}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def norm_category_token(value: Any) -> str:
    text = str(value or "CAT_UNK").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or "CAT_UNK"


def category_token_lookup(semantic_map: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in semantic_map.values():
        category = str(row.get("category") or "").strip()
        token = str(row.get("category_token") or "").strip()
        if category and token:
            out.setdefault(category, token)
    return out


def common_slots_from_summary(summary: Any) -> List[str]:
    text = str(summary or "")
    match = re.search(r"Common local time slots are ([^.]+)", text)
    if not match:
        return []
    slots: List[str] = []
    for part in re.split(r"[,，]\s*", match.group(1).strip()):
        part = part.strip()
        if not part:
            continue
        if part.lower().startswith("s"):
            part = part[1:]
        if part.isdigit():
            slots.append(f"s{int(part)}")
    return slots


def build_user_semantic_profile(
    pref: Dict[str, Any] | None,
    semantic_map: Dict[str, Dict[str, Any]],
    category_tokens: Dict[str, str],
    max_categories: int = 4,
    max_revisited: int = 4,
    max_geo: int = 3,
    max_slots: int = 3,
) -> str:
    pref = pref or {}
    history_count = safe_int(pref.get("history_count"))
    lines: List[str] = [
        f"history_count={history_count}",
        "",
        "Long-term category affinity:",
    ]

    top_categories = list(pref.get("top_categories") or [])[:max_categories]
    if top_categories:
        for idx, item in enumerate(top_categories, 1):
            category = str(item.get("category") or item.get("value") or "Unknown")
            count = safe_int(item.get("count"))
            token = category_tokens.get(category) or norm_category_token(category)
            lines.append(f"{idx}. {token}|category={category}|visits={count}")
    else:
        lines.append("- none")

    lines.extend(["", "Revisit affinity:"])
    revisited = list(pref.get("revisited_pois") or [])[:max_revisited]
    geo_counts: Counter[str] = Counter()
    geo_categories: Dict[str, Counter[str]] = {}
    if revisited:
        for idx, item in enumerate(revisited, 1):
            poi_id = str(item.get("poi_id") or item.get("poi") or "")
            count = safe_int(item.get("count"))
            sem = semantic_map.get(poi_id) or {}
            semantic_id = str(sem.get("semantic_id") or "SEM_UNK")
            category = str(sem.get("category") or sem.get("category_token") or "CAT_UNK")
            geo_cell = str(sem.get("geo_cell") or "GEO_UNK")
            lines.append(f"{idx}. {poi_id}|semantic_id={semantic_id}|category={category}|geo={geo_cell}|visits={count}")
            if geo_cell and geo_cell != "GEO_UNK":
                geo_counts[geo_cell] += max(count, 1)
                geo_categories.setdefault(geo_cell, Counter())[category] += max(count, 1)
    else:
        lines.append("- none")

    lines.extend(["", "Geo routine:"])
    if geo_counts:
        for idx, (geo_cell, count) in enumerate(geo_counts.most_common(max_geo), 1):
            cats = ",".join(cat for cat, _ in geo_categories.get(geo_cell, Counter()).most_common(3)) or "unknown"
            lines.append(f"{idx}. geo={geo_cell}|visits={int(count)}|categories={cats}")
    else:
        lines.append("- none")

    lines.extend(["", "Temporal routine:"])
    slots = common_slots_from_summary(pref.get("preference_summary"))[:max_slots]
    if slots:
        for idx, slot in enumerate(slots, 1):
            lines.append(f"{idx}. slot={slot}")
    else:
        lines.append("- none_available")

    return "\n".join(lines)


def parse_profile_items(profile: str) -> Dict[str, List[str]]:
    sections = {
        "Long-term category affinity": [],
        "Revisit affinity": [],
        "Geo routine": [],
        "Temporal routine": [],
    }
    current: str | None = None
    for line in str(profile or "").splitlines():
        stripped = line.strip()
        if stripped.endswith(":"):
            name = stripped[:-1]
            current = name if name in sections else None
            continue
        if current is None:
            continue
        if stripped.startswith(("-", "history_count=")) or not stripped:
            continue
        if re.match(r"^\d+\.\s+", stripped):
            sections[current].append(re.sub(r"^\d+\.\s+", "", stripped))
    return sections


def own_profile_insufficient(profile: str) -> bool:
    history_match = re.search(r"\bhistory_count=(\d+)", str(profile or ""))
    history_count = safe_int(history_match.group(1) if history_match else 0)
    sections = parse_profile_items(profile)
    return (
        history_count <= 3
        or len(sections["Long-term category affinity"]) <= 1
        or len(sections["Revisit affinity"]) == 0
        or len(sections["Geo routine"]) == 0
    )


def similar_profile_has_signal(profile: str) -> bool:
    sections = parse_profile_items(profile)
    non_empty = sum(1 for values in sections.values() if values)
    return non_empty >= 2


def raw_recent_context(raw_text: str, max_lines: int = 24) -> str:
    keep_headers = {"Trajectory:", "Transitions:", "Nearby:"}
    stop_headers = {"Trajectory:", "Transitions:", "Nearby:", "Preference:"}
    out: List[str] = []
    keep = False
    per_section = 0
    for line in str(raw_text or "").splitlines():
        stripped = line.strip()
        if stripped in stop_headers:
            keep = stripped in keep_headers
            per_section = 0
            if keep:
                out.append(stripped)
            continue
        if keep and stripped:
            out.append(stripped)
            per_section += 1
            if per_section >= max_lines:
                keep = False
    return "\n".join(out)


def similar_query_text(row: Dict[str, Any], source: str) -> str:
    profile = str(row.get("user_semantic_profile") or "")
    if source == "user_profile":
        return profile
    recent = raw_recent_context(str(row.get("raw_text") or ""))
    return "\n\n".join(x for x in [recent, "[USER_SEMANTIC_PROFILE]", profile] if x.strip())


def parse_visit_item(item: str) -> Tuple[str, int]:
    visits_match = re.search(r"\bvisits=(\d+)", item)
    support = safe_int(visits_match.group(1) if visits_match else 1, default=1)
    return item, max(1, support)


def parse_key_value_fields(item: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    parts = [part.strip() for part in str(item or "").split("|") if part.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def similar_revisit_weak_key(item: str) -> str:
    fields = parse_key_value_fields(item)
    category = fields.get("category") or "CAT_UNK"
    geo = fields.get("geo") or "GEO_UNK"
    return f"category={category}|geo={geo}"


def aggregate_similar_user_profile(
    row: Dict[str, Any],
    neighbors: List[Tuple[Dict[str, Any], float]],
    max_categories: int = 2,
    max_revisited: int = 1,
    max_geo: int = 1,
    max_slots: int = 1,
) -> str:
    category_counts: Counter[str] = Counter()
    revisit_counts: Counter[str] = Counter()
    geo_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    used_users = set()
    used_neighbors = 0

    for neighbor, sim in neighbors:
        if sim <= 0:
            continue
        used_neighbors += 1
        user_id = str(neighbor.get("user_id") or "")
        if user_id:
            used_users.add(user_id)
        sections = parse_profile_items(str(neighbor.get("user_semantic_profile") or ""))
        weight = max(0.01, float(sim))
        for item in sections["Long-term category affinity"]:
            key, visits = parse_visit_item(item)
            category_counts[key] += weight * visits
        for item in sections["Revisit affinity"]:
            _, visits = parse_visit_item(item)
            key = similar_revisit_weak_key(item)
            revisit_counts[key] += weight * visits
        for item in sections["Geo routine"]:
            key, visits = parse_visit_item(item)
            geo_counts[key] += weight * visits
        for item in sections["Temporal routine"]:
            slot_counts[item] += weight

    lines = [
        f"source=bge_m3_user_knn;k={len(neighbors)};support_users={len(used_users)};support_neighbors={used_neighbors}",
        "",
        "Long-term category affinity:",
    ]
    if category_counts:
        for idx, (item, score) in enumerate(category_counts.most_common(max_categories), 1):
            lines.append(f"{idx}. {item}|support={score:.2f}")
    else:
        lines.append("- none")

    lines.extend(["", "Revisit affinity:"])
    if revisit_counts:
        for idx, (item, score) in enumerate(revisit_counts.most_common(max_revisited), 1):
            lines.append(f"{idx}. {item}|support={score:.2f}")
    else:
        lines.append("- none")

    lines.extend(["", "Geo routine:"])
    if geo_counts:
        for idx, (item, score) in enumerate(geo_counts.most_common(max_geo), 1):
            lines.append(f"{idx}. {item}|support={score:.2f}")
    else:
        lines.append("- none")

    lines.extend(["", "Temporal routine:"])
    if slot_counts:
        for idx, (item, score) in enumerate(slot_counts.most_common(max_slots), 1):
            lines.append(f"{idx}. {item}|support={score:.2f}")
    else:
        lines.append("- none_available")
    return "\n".join(lines)


def encode_bge_texts(args: argparse.Namespace, texts: List[str]) -> "Any":
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = args.similar_profile_device if torch.cuda.is_available() or args.similar_profile_device == "cpu" else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.similar_profile_model, local_files_only=args.similar_profile_local_files_only)
    model = AutoModel.from_pretrained(args.similar_profile_model, local_files_only=args.similar_profile_local_files_only)
    model.to(device)
    model.eval()
    chunks = []
    print(f"[similar_profile] encode texts={len(texts)} device={device}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(texts), max(1, args.similar_profile_batch_size)):
            batch_texts = texts[start : start + max(1, args.similar_profile_batch_size)]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.similar_profile_max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_dict=True)
            hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = F.normalize(pooled.float(), p=2, dim=1)
            chunks.append(pooled.cpu())
            if start == 0 or start + len(batch_texts) >= len(texts) or (start // max(1, args.similar_profile_batch_size)) % 50 == 0:
                print(f"[similar_profile] encoded={start + len(batch_texts)}/{len(texts)}", flush=True)
    return torch.cat(chunks, dim=0)


def augment_similar_user_profiles(rows_by_split: Dict[str, List[Dict[str, Any]]], args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    train_rows = rows_by_split.get("train") or []
    if not train_rows:
        raise ValueError("--similar-profile requires the train split to build a non-leaking neighbor bank.")
    bank_rows = train_rows
    bank_texts = [similar_query_text(row, args.similar_profile_query_source) for row in bank_rows]
    bank_emb = encode_bge_texts(args, bank_texts)

    stats: Dict[str, Any] = {
        "enabled": True,
        "model": str(args.similar_profile_model),
        "query_source": args.similar_profile_query_source,
        "k": args.similar_profile_k,
        "min_sim": args.similar_profile_min_sim,
        "bank_rows": len(bank_rows),
        "splits": {},
    }
    bank_user_ids = [str(row.get("user_id") or "") for row in bank_rows]
    bank_sample_ids = [str(row.get("sample_id") or "") for row in bank_rows]
    search_device = args.similar_profile_device if torch.cuda.is_available() or args.similar_profile_device == "cpu" else "cpu"
    bank_emb_device = bank_emb.to(search_device)
    topk_extra = min(len(bank_rows), max(args.similar_profile_k * 10, args.similar_profile_k + 10))

    for split, rows in rows_by_split.items():
        query_texts = [similar_query_text(row, args.similar_profile_query_source) for row in rows]
        query_emb = bank_emb if rows is bank_rows else encode_bge_texts(args, query_texts)
        split_counts = Counter()
        search_batch = max(1, int(args.similar_profile_search_batch_size))
        print(f"[similar_profile] search split={split} rows={len(rows)} bank={len(bank_rows)} device={search_device}", flush=True)
        for start in range(0, len(rows), search_batch):
            end = min(len(rows), start + search_batch)
            q = query_emb[start:end].to(search_device)
            scores = torch.matmul(q, bank_emb_device.T)
            values_batch, indices_batch = torch.topk(scores, k=topk_extra, dim=1)
            values_batch = values_batch.cpu()
            indices_batch = indices_batch.cpu()
            for local_idx, row in enumerate(rows[start:end]):
                own_user = str(row.get("user_id") or "")
                own_sample = str(row.get("sample_id") or "")
                ranked: List[Tuple[float, int]] = []
                for sim_tensor, bank_idx_tensor in zip(values_batch[local_idx], indices_batch[local_idx]):
                    bank_idx = int(bank_idx_tensor.item())
                    sim = float(sim_tensor.item())
                    if sim < args.similar_profile_min_sim:
                        continue
                    if bank_sample_ids[bank_idx] == own_sample:
                        continue
                    if own_user and bank_user_ids[bank_idx] == own_user:
                        continue
                    if not similar_profile_has_signal(str(bank_rows[bank_idx].get("user_semantic_profile") or "")):
                        continue
                    ranked.append((sim, bank_idx))
                    if len(ranked) >= args.similar_profile_k:
                        break
                neighbors = [(bank_rows[bank_idx], sim) for sim, bank_idx in ranked]
                profile = aggregate_similar_user_profile(row, neighbors) if neighbors else ""
                row["similar_user_semantic_profile"] = profile
                row["similar_user_count"] = len(neighbors)
                row["similar_user_profile_has_signal"] = similar_profile_has_signal(profile)
                row["user_profile_insufficient"] = own_profile_insufficient(str(row.get("user_semantic_profile") or ""))
                split_counts["rows"] += 1
                split_counts["with_neighbors"] += int(bool(neighbors))
                split_counts["has_signal"] += int(bool(row["similar_user_profile_has_signal"]))
                split_counts["own_profile_insufficient"] += int(bool(row["user_profile_insufficient"]))
            if start == 0 or end >= len(rows) or (start // search_batch) % 20 == 0:
                print(f"[similar_profile] searched split={split} rows={end}/{len(rows)}", flush=True)
        stats["splits"][split] = dict(split_counts)
    return stats


def compact_geo(row: Dict[str, Any]) -> Tuple[str, bool]:
    if row.get("geo_cell") == "GEO_UNK" or row.get("latitude") is None or row.get("longitude") is None:
        return "UNK", True
    try:
        return f"{float(row['latitude']):.2f},{float(row['longitude']):.2f}", False
    except (TypeError, ValueError):
        return "UNK", True


def semantic_code(semantic_map: Dict[str, Dict[str, Any]], poi_id: str) -> str:
    row = semantic_map.get(str(poi_id))
    if row is None:
        return "SEM_UNK"
    category = str(row.get("category_token") or "CAT_UNK")
    local_id = str(row.get("semantic_id") or "").split("::")[-1] or "P0000"
    geo, missing = compact_geo(row)
    code = f"{category}@{geo}#{local_id}"
    return f"{code}|geo=unk" if missing else code


def strip_generation_instructions(prompt: str) -> str:
    lines: List[str] = []
    skip_output_json = False
    for line in prompt.splitlines():
        text = line.strip()
        if text.startswith("Output only"):
            continue
        if text == "Output format:":
            skip_output_json = True
            continue
        if skip_output_json:
            skip_output_json = False
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def graph_view_text(row: Dict[str, Any], semantic_map: Dict[str, Dict[str, Any]], top_k: int) -> str:
    details = row.get("candidate_details") or {}
    candidates = [str(x) for x in row.get("candidate_poi_ids") or []][:top_k]
    lines = [f"GraphRAG top{len(candidates)} candidates:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for rank, poi in enumerate(candidates, 1):
        detail = details.get(poi) or {}
        score = detail.get("score")
        try:
            score_text = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_text = "na"
        sources = ",".join(str(x) for x in (detail.get("sources") or [])[:4]) or "unknown"
        lines.append(f"{rank}. {poi}|{semantic_code(semantic_map, poi)}|score={score_text}|src={sources}")
    return "\n".join(lines)


def graph_candidate_features(row: Dict[str, Any], top_k: int) -> tuple[List[str], List[int], List[float]]:
    details = row.get("candidate_details") or {}
    candidates = [str(x) for x in row.get("candidate_poi_ids") or []][:top_k]
    ranks: List[int] = []
    scores: List[float] = []
    for rank, poi in enumerate(candidates, 1):
        detail = details.get(poi) or {}
        raw_score = detail.get("score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        ranks.append(rank)
        scores.append(score)
    return candidates, ranks, scores


def build_input_text(row: Dict[str, Any]) -> str:
    parts = [
        "[VIEW=RAW_SEM]",
        str(row.get("raw_text") or "").strip(),
        "[VIEW=REFINED source=full]",
        str(row.get("refined_text") or "").strip(),
        "[VIEW=USER_SEMANTIC_PROFILE source=preference_evidence]",
        str(row.get("user_semantic_profile") or "").strip(),
    ]
    similar_profile = str(row.get("similar_user_semantic_profile") or "").strip()
    if similar_profile:
        parts.extend(["[VIEW=SIMILAR_USER_SEMANTIC_PROFILE source=bge_m3_user_knn]", similar_profile])
    parts.extend(["[VIEW=GRAPH_RAG]", str(row.get("graph_text") or "").strip()])
    return "\n\n".join(parts)


def split_paths(base_dir: Path, split: str) -> Dict[str, Path]:
    return {
        "raw": base_dir / "semantic_poi_sft" / f"stage1_{split}_raw_semantic.jsonl",
        "preference": base_dir / "evidence" / f"preference_evidence_{split}.jsonl",
        "refined": base_dir / "refined_prompts_decision" / f"lora_a_decision_v2_{split}_full_outputs.jsonl",
        "graph": base_dir / "double_llm" / f"graphrag_semantic_edges_v2_top100_{split}_candidates.jsonl",
    }


def build_split(
    base_dir: Path,
    split: str,
    semantic_map: Dict[str, Dict[str, Any]],
    graph_top_k: int,
    allow_missing_refined: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = split_paths(base_dir, split)
    category_tokens = category_token_lookup(semantic_map)
    preference_by_id = load_by_sample_id(paths["preference"])
    refined_file_missing = not paths["refined"].exists()
    if refined_file_missing and not allow_missing_refined:
        raise FileNotFoundError(paths["refined"])
    refined_by_id = load_by_sample_id(paths["refined"]) if paths["refined"].exists() else {}
    graph_by_id = load_by_sample_id(paths["graph"])
    rows: List[Dict[str, Any]] = []
    counts = Counter()
    for raw in read_jsonl(paths["raw"]):
        sid = str(raw.get("sample_id") or "")
        preference = preference_by_id.get(sid)
        refined = refined_by_id.get(sid)
        graph = graph_by_id.get(sid)
        if preference is None:
            counts["missing_preference"] += 1
            continue
        if refined is None and not allow_missing_refined:
            counts["missing_refined"] += 1
            continue
        if graph is None:
            counts["missing_graph"] += 1
            continue
        target_poi = str(((raw.get("target") or {}).get("poi_id")) or "")
        graph_target = str(((graph.get("target") or {}).get("poi_id")) or "")
        if graph_target and graph_target != target_poi:
            counts["graph_target_mismatch"] += 1
            continue
        raw_text = strip_generation_instructions(str(raw.get("input_prompt") or ""))
        refined = refined or {}
        refined_text = str(refined.get("distilled_prompt") or "").strip()
        user_semantic_profile = build_user_semantic_profile(preference, semantic_map, category_tokens)
        graph_text = graph_view_text(graph, semantic_map, graph_top_k)
        graph_candidate_ids, graph_candidate_ranks, graph_candidate_scores = graph_candidate_features(graph, graph_top_k)
        target_rank = int(graph.get("target_rank") or 0)
        target_in_selected_topk = target_rank > 0 and target_rank <= graph_top_k
        row = (
            {
                "sample_id": sid,
                "split": split,
                "city": str(raw.get("city") or ""),
                "user_id": str(preference.get("user_id") or raw.get("user_id") or ""),
                "input_text": "",
                "raw_text": raw_text,
                "refined_text": refined_text,
                "user_semantic_profile": user_semantic_profile,
                "similar_user_semantic_profile": "",
                "similar_user_count": 0,
                "similar_user_profile_has_signal": False,
                "user_profile_insufficient": own_profile_insufficient(user_semantic_profile),
                "graph_text": graph_text,
                "graph_candidate_poi_ids": graph_candidate_ids,
                "graph_candidate_ranks": graph_candidate_ranks,
                "graph_candidate_scores": graph_candidate_scores,
                "target_poi_id": target_poi,
                "target_category": str((raw.get("target") or {}).get("category") or ""),
                "graph_target_in_topk": target_in_selected_topk,
                "graph_target_rank": target_rank,
                "refined_confidence": str(refined.get("refiner_confidence") or ""),
                "refined_useful": str(refined.get("useful_for_refinement") or ""),
            }
        )
        row["input_text"] = build_input_text(row)
        rows.append(row)
        counts["rows"] += 1
        counts[f"refined_confidence_{refined.get('refiner_confidence')}"] += 1
        counts[f"refined_useful_{refined.get('useful_for_refinement')}"] += 1
        counts["graph_target_in_topk"] += int(target_in_selected_topk)
    stats = {
        "split": split,
        "paths": {k: str(v) for k, v in paths.items()},
        "graph_top_k": graph_top_k,
        "counts": dict(counts),
        "allow_missing_refined": bool(allow_missing_refined),
        "refined_file_missing": bool(refined_file_missing),
        "graph_target_in_topk_ratio": round(counts["graph_target_in_topk"] / counts["rows"], 6) if counts["rows"] else 0.0,
    }
    return rows, stats


def write_jsonl(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", compression_level=7)


def main() -> None:
    args = parse_args()
    semantic_path = args.semantic_map or (args.base_dir / "double_llm" / "semantic_poi_ids.jsonl")
    semantic_map = load_semantic_map(semantic_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "semantic_map": str(semantic_path),
        "graph_top_k": args.graph_top_k,
        "format": args.format,
        "similar_profile": {"enabled": bool(args.similar_profile)},
        "splits": {},
    }
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in args.splits:
        rows, stats = build_split(args.base_dir, split, semantic_map, args.graph_top_k, args.allow_missing_refined)
        rows_by_split[split] = rows
        summary["splits"][split] = stats
    if args.similar_profile:
        summary["similar_profile"] = augment_similar_user_profiles(rows_by_split, args)
        for rows in rows_by_split.values():
            for row in rows:
                row["input_text"] = build_input_text(row)
    for split in args.splits:
        rows = rows_by_split[split]
        stats = summary["splits"][split]
        if args.format in {"parquet", "both"}:
            out = args.output_dir / f"{split}_joined_top{args.graph_top_k}.parquet"
            write_parquet(out, rows, args.overwrite)
            stats["parquet_output"] = str(out)
            stats["parquet_bytes"] = out.stat().st_size
        if args.format in {"jsonl", "both"}:
            out = args.output_dir / f"{split}_joined_top{args.graph_top_k}.jsonl"
            write_jsonl(out, rows, args.overwrite)
            stats["jsonl_output"] = str(out)
            stats["jsonl_bytes"] = out.stat().st_size
        stats_path = args.output_dir / f"{split}_joined_top{args.graph_top_k}.stats.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"build_summary_top{args.graph_top_k}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
