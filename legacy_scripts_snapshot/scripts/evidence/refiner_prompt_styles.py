"""Prompt styles for LoRA-A evidence refinement."""
from __future__ import annotations

import json
from typing import Any, Dict, List


REFINER_STYLES = ("summary", "decision")

SUMMARY_SYSTEM_PROMPT = (
    "You are an evidence refiner for next-POI prediction. "
    "Turn structured sequence, geographic, and preference evidence into a concise, label-free prompt. "
    "Do not predict the next POI. Do not mention gold labels, targets, answers, or ground truth."
)

DECISION_SYSTEM_PROMPT = (
    "You are a decision-style evidence refiner for next-POI prediction. "
    "Convert structured evidence into a compact, label-free decision aid for a downstream model. "
    "Do not predict the exact next POI. Do not mention gold labels, targets, answers, or ground truth. "
    "Do not output POI ids, absolute dates, coordinate-related words or values, user ids, sample ids, "
    "or trajectory ids. If evidence is too sparse or conflicting, mark the refinement as not useful."
)


def compact_evidence(row: Dict[str, Any], max_chars: int | None = None) -> str:
    payload = {
        "sample_id": row.get("sample_id"),
        "city": row.get("city"),
        "split": row.get("split"),
        "user_id": row.get("user_id"),
        "trajectory_id": row.get("trajectory_id"),
        "evidence": row.get("evidence"),
        "candidate_poi_ids": row.get("candidate_poi_ids"),
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n...[TRUNCATED]"
    return text


def system_prompt(style: str) -> str:
    if style == "summary":
        return SUMMARY_SYSTEM_PROMPT
    if style == "decision":
        return DECISION_SYSTEM_PROMPT
    raise ValueError(f"Unknown refiner style: {style}")


def build_user_prompt(row: Dict[str, Any], style: str, max_evidence_chars: int | None = None) -> str:
    evidence = compact_evidence(row, max_evidence_chars)
    if style == "summary":
        return (
            "Create a compact, label-free downstream prediction prompt from the evidence below.\n"
            "Return only the distilled prompt text. Do not output JSON.\n\n"
            f"Evidence JSON:\n{evidence}"
        )
    if style == "decision":
        return (
            "Create a decision-style, label-free refinement prompt from the evidence below.\n"
            "Return only the refined prompt text. Do not output JSON.\n\n"
            "Required format:\n"
            "Evidence verdict: useful_for_refinement=<yes|no>; confidence=<high|medium|low>.\n"
            "Priority hypotheses:\n"
            "1. category=<category>; evidence=<transition|geo|history|sequence|mixed>; reason=<short reason>.\n"
            "2. category=<category>; evidence=<transition|geo|history|sequence|mixed>; reason=<short reason>.\n"
            "3. category=<category>; evidence=<transition|geo|history|sequence|mixed>; reason=<short reason>.\n"
            "Weak or noisy cues: <at most 3 short cues that should not dominate>.\n"
            "Final hint: prefer <transition|geo|history|sequence|raw> because <one short reason>.\n\n"
            "Constraints:\n"
            "- Do not include any POI id such as v123.\n"
            "- Do not include user_id, sample_id, trajectory_id, target, label, answer, or ground truth.\n"
            "- Do not include absolute dates.\n"
            "- Do not use coordinate-related words such as coordinate, coordinates, latitude, longitude, "
            "lat, lon, or lng; say geographic evidence is unavailable or weak instead.\n"
            "- Set useful_for_refinement=no when the evidence is too sparse, contradictory, generic, "
            "or no source clearly improves over the raw trajectory prompt.\n"
            "- Set confidence=low when useful_for_refinement=no or when all hypotheses are weak.\n"
            "- Keep Weak or noisy cues short: at most 3 comma-separated cues, no long category lists.\n"
            "- Always finish with exactly one Final hint sentence.\n"
            "- Avoid generic uncertainty-only conclusions; provide priority and evidence source.\n"
            "- Keep the output under 180 words.\n\n"
            f"Evidence JSON:\n{evidence}"
        )
    raise ValueError(f"Unknown refiner style: {style}")


def build_teacher_messages(row: Dict[str, Any], style: str, max_evidence_chars: int) -> List[Dict[str, str]]:
    if style == "summary":
        user = (
            "Create a compact, label-free downstream prediction prompt from the evidence below.\n\n"
            "Output JSON schema:\n"
            "{\n"
            '  "sample_id": "<same sample_id>",\n'
            '  "distilled_prompt": "<120-220 words, useful for predicting the next POI, no answer leakage>",\n'
            '  "evidence_quality": "good|partial|weak"\n'
            "}\n\n"
            "The prompt should summarize:\n"
            "1. short-term movement and time pattern,\n"
            "2. useful transition candidates without calling them correct,\n"
            "3. nearby geographic options when available,\n"
            "4. long-term user preferences when available,\n"
            "5. uncertainty when evidence is weak.\n\n"
            f"Evidence JSON:\n{compact_evidence(row, max_evidence_chars)}"
        )
    elif style == "decision":
        user = (
            "Create a decision-style, label-free downstream prediction prompt from the evidence below.\n\n"
            "Output JSON schema:\n"
            "{\n"
            '  "sample_id": "<same sample_id>",\n'
            '  "distilled_prompt": "<decision-style prompt under 180 words>",\n'
            '  "evidence_quality": "good|partial|weak"\n'
            "}\n\n"
            "The distilled_prompt must follow this format exactly:\n"
            "Evidence verdict: useful_for_refinement=<yes|no>; confidence=<high|medium|low>.\n"
            "Priority hypotheses:\n"
            "1. category=<category>; evidence=<transition|geo|history|sequence|mixed>; reason=<short reason>.\n"
            "2. category=<category>; evidence=<transition|geo|history|sequence|mixed>; reason=<short reason>.\n"
            "3. category=<category>; evidence=<transition|geo|history|sequence|mixed>; reason=<short reason>.\n"
            "Weak or noisy cues: <at most 3 short cues that should not dominate>.\n"
            "Final hint: prefer <transition|geo|history|sequence|raw> because <one short reason>.\n\n"
            "Hard constraints:\n"
            "- Do not predict the exact next POI or identify any candidate as the answer.\n"
            "- Do not include POI ids such as v123.\n"
            "- Do not include user_id, sample_id, trajectory_id, target, label, answer, or ground truth.\n"
            "- Do not include absolute dates.\n"
            "- Do not use coordinate-related words such as coordinate, coordinates, latitude, longitude, "
            "lat, lon, or lng; say geographic evidence is unavailable or weak instead.\n"
            "- Use useful_for_refinement=yes only when at least one evidence source gives a concrete "
            "decision advantage over the raw trajectory prompt.\n"
            "- Use useful_for_refinement=no when evidence is too sparse, contradictory, generic, dominated "
            "by low-count cues, or no source clearly improves over the raw prompt.\n"
            "- Use confidence=low when useful_for_refinement=no or when all priority hypotheses are weak.\n"
            "- Keep Weak or noisy cues short: at most 3 comma-separated cues, no long category lists.\n"
            "- Always finish with exactly one Final hint sentence.\n"
            "- Avoid generic uncertainty-only conclusions; always prioritize evidence sources.\n\n"
            f"Evidence JSON:\n{compact_evidence(row, max_evidence_chars)}"
        )
    else:
        raise ValueError(f"Unknown refiner style: {style}")
    return [{"role": "system", "content": system_prompt(style)}, {"role": "user", "content": user}]
