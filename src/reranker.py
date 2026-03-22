"""LLM reranker — Claude Sonnet re-scores retrieval candidates for relevance."""
from __future__ import annotations

import json
import re

from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    RERANKER_MODEL,
    TOP_K_RERANK,
)
from src.providers.prompts import RERANK_PROMPT


def _format_candidate(idx: int, segment: dict) -> str:
    """Format a single candidate segment for the reranker prompt."""
    scene = segment.get("scene_description", "")
    transcript = segment.get("transcript", "")
    time_of_day = segment.get("time_of_day", "")
    max_volume = segment.get("max_volume", "")

    return (
        f"--- Segment {idx + 1} (id: {segment['id']}) ---\n"
        f"File: {segment.get('file_name', '')} "
        f"[{segment.get('start_s', 0):.0f}s - {segment.get('end_s', 0):.0f}s] "
        f"| time: {time_of_day} | volume: {max_volume}\n"
        f"Scene: {scene}\n"
        f"Transcript: {transcript}"
    )


def _parse_rerank_response(text: str, candidates: list[dict]) -> list[dict]:
    """Parse reranker JSON response, falling back to original order on failure."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    try:
        scored = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                scored = json.loads(match.group())
            except json.JSONDecodeError:
                return candidates
        else:
            return candidates

    if not isinstance(scored, list):
        return candidates

    # Build lookup of candidate data by ID
    candidate_map = {c["id"]: c for c in candidates}

    results = []
    for item in scored:
        seg_id = item.get("id")
        if seg_id and seg_id in candidate_map:
            results.append({
                **candidate_map[seg_id],
                "rerank_score": item.get("score", 0),
                "rerank_reason": item.get("reason", ""),
            })

    # Append any candidates the reranker missed (shouldn't happen, but safe)
    seen = {r["id"] for r in results}
    for c in candidates:
        if c["id"] not in seen:
            results.append({**c, "rerank_score": 0, "rerank_reason": ""})

    return results


def rerank(
    query: str,
    candidates: list[dict],
    *,
    top_k: int = TOP_K_RERANK,
) -> list[dict]:
    """Re-score candidates using Claude Sonnet.

    Takes the top candidates from RRF fusion and asks the LLM to judge
    relevance based on the actual scene descriptions and transcripts.

    Returns candidates sorted by rerank_score descending, trimmed to top_k.
    """
    # Only send top_k candidates to the reranker
    to_rerank = candidates[:top_k]

    if not to_rerank:
        return []

    formatted = "\n\n".join(
        _format_candidate(i, seg) for i, seg in enumerate(to_rerank)
    )

    prompt = RERANK_PROMPT.format(
        query=query,
        n=len(to_rerank),
        candidates=formatted,
    )

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    response = client.chat.completions.create(
        model=RERANKER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    return _parse_rerank_response(raw, to_rerank)
