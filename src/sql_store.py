"""Supabase Postgres client — segment storage + full-text search via REST API."""
from __future__ import annotations

import json

import httpx

from src.config import SUPABASE_URL, SUPABASE_KEY


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _rest_url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def _rpc_url(fn: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/rpc/{fn}"


# --- Write ---

def upsert_segment(segment: dict) -> None:
    """Upsert a single segment row.

    Expected keys: id, file_name, start_s, end_s, scene_description,
    transcript, prosody_json, time_of_day, max_volume.
    """
    headers = _headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    resp = httpx.post(
        _rest_url("segments"),
        headers=headers,
        json=segment,
        timeout=30,
    )
    resp.raise_for_status()


def upsert_segments(segments: list[dict]) -> None:
    """Upsert multiple segment rows in a single request."""
    headers = _headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    resp = httpx.post(
        _rest_url("segments"),
        headers=headers,
        json=segments,
        timeout=60,
    )
    resp.raise_for_status()


# --- Read ---

def get_segment(segment_id: str) -> dict | None:
    """Fetch a single segment by ID."""
    resp = httpx.get(
        _rest_url("segments"),
        headers=_headers(),
        params={"id": f"eq.{segment_id}", "limit": "1"},
        timeout=15,
    )
    resp.raise_for_status()
    rows = resp.json()
    return rows[0] if rows else None


def get_segments_by_file(file_name: str) -> list[dict]:
    """Fetch all segments for a given video file, ordered by start time."""
    resp = httpx.get(
        _rest_url("segments"),
        headers=_headers(),
        params={
            "file_name": f"eq.{file_name}",
            "order": "start_s.asc",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# --- Full-text search ---

def search_fts(
    query: str,
    *,
    time_of_day: str | None = None,
    max_volume: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Full-text search via the search_segments RPC function.

    Returns rows with a `rank` field, sorted by relevance descending.
    """
    params = {
        "query": query,
        "match_limit": limit,
    }
    if time_of_day is not None:
        params["filter_time_of_day"] = time_of_day
    if max_volume is not None:
        params["filter_max_volume"] = max_volume

    # search_segments defined in Supabase SQL functions
    resp = httpx.post(
        _rpc_url("search_segments"),
        headers=_headers(),
        json=params,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# --- Delete ---

def delete_segment(segment_id: str) -> None:
    """Delete a single segment by ID."""
    resp = httpx.delete(
        _rest_url("segments"),
        headers=_headers(),
        params={"id": f"eq.{segment_id}"},
        timeout=15,
    )
    resp.raise_for_status()


def delete_segments_by_file(file_name: str) -> None:
    """Delete all segments for a given video file."""
    resp = httpx.delete(
        _rest_url("segments"),
        headers=_headers(),
        params={"file_name": f"eq.{file_name}"},
        timeout=15,
    )
    resp.raise_for_status()
