"""Ingestion orchestrator — segment → analyze → embed → store."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import httpx

from src.config import SEGMENTS_DIR
from src.providers import analyze_segment, get_embedding_provider
from src.providers.base import AnalysisResult
from src import sql_store, vector_store

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubled each retry

# Errors worth retrying (rate limits, timeouts, server errors)
_RETRYABLE = (
    httpx.HTTPStatusError,
    httpx.TimeoutException,
    httpx.ConnectError,
    ConnectionError,
    TimeoutError,
)


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, _RETRYABLE)


def _segment_id(file_name: str, index: int) -> str:
    """Deterministic segment ID shared across Supabase and Pinecone."""
    stem = Path(file_name).stem
    return f"{stem}_seg_{index:03d}"


def _convert_timestamps(text: str, offset_s: float) -> str:
    """Convert relative timestamps [MM:SS] to absolute timestamps in the source video.

    e.g. with offset_s=1390, [00:18] → [23:28]
    """
    def _replace(m: re.Match) -> str:
        mins, secs = int(m.group(1)), int(m.group(2))
        abs_s = offset_s + mins * 60 + secs
        abs_min = int(abs_s // 60)
        abs_sec = int(abs_s % 60)
        return f"[{abs_min:02d}:{abs_sec:02d}]"

    return re.sub(r"\[(\d{1,2}):(\d{2})\]", _replace, text)


def _build_supabase_row(
    segment_id: str,
    seg_meta: dict,
    analysis: AnalysisResult,
) -> dict:
    """Build the dict for Supabase upsert from segment metadata + analysis."""
    return {
        "id": segment_id,
        "file_name": seg_meta["file_name"],
        "start_s": seg_meta["start_s"],
        "end_s": seg_meta["end_s"],
        "scene_description": analysis.scene.scene_description,
        "transcript": analysis.transcript.transcript,
        "prosody_json": asdict(analysis.prosody),
        "time_of_day": analysis.scene.time_of_day,
        "max_volume": analysis.prosody.max_volume,
    }


def _build_pinecone_metadata(seg_meta: dict, analysis: AnalysisResult) -> dict:
    """Build the metadata dict for Pinecone upsert."""
    return {
        "file_name": seg_meta["file_name"],
        "start_s": seg_meta["start_s"],
        "end_s": seg_meta["end_s"],
        "max_volume": analysis.prosody.max_volume,
        "time_of_day": analysis.scene.time_of_day,
    }


def ingest_segment(seg_meta: dict, force: bool = False) -> str | None:
    """Ingest a single segment: analyze → convert timestamps → embed → store.

    Skips if already ingested (unless force=True). Retries on transient failures.

    seg_meta: dict from segmenter with keys:
        file_name, segment_path, segment_index, start_s, end_s

    Returns the segment ID, or None if skipped.
    """
    seg_id = _segment_id(seg_meta["file_name"], seg_meta["segment_index"])

    if not force and sql_store.get_segment(seg_id) is not None:
        return None

    video_bytes = Path(seg_meta["segment_path"]).read_bytes()

    for attempt in range(MAX_RETRIES):
        try:
            # 1. Analyze (scene + transcript + prosody)
            analysis = analyze_segment(video_bytes)

            # 2. Convert relative timestamps to absolute
            analysis.transcript.transcript = _convert_timestamps(
                analysis.transcript.transcript, seg_meta["start_s"]
            )
            analysis.scene.scene_description = _convert_timestamps(
                analysis.scene.scene_description, seg_meta["start_s"]
            )

            # 3. Embed (scene_description + transcript concatenated)
            embed_text = (
                analysis.scene.scene_description
                + "\n\n"
                + analysis.transcript.transcript
            )
            embedder = get_embedding_provider()
            vector = embedder.embed(embed_text)

            # 4. Store in both databases
            sql_store.upsert_segment(
                _build_supabase_row(seg_id, seg_meta, analysis)
            )
            vector_store.upsert_segment(
                seg_id, vector, _build_pinecone_metadata(seg_meta, analysis)
            )

            return seg_id

        except Exception as e:
            if _is_retryable(e) and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"  [RETRY] {seg_id} attempt {attempt + 1}/{MAX_RETRIES}: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def ingest_video(
    segment_metas: list[dict],
    *,
    max_workers: int = 4,
    force: bool = False,
) -> list[str]:
    """Ingest all segments for a video, with parallel API calls.

    Skips already-ingested segments unless force=True. Retries transient failures.

    segment_metas: list of dicts from segmenter.segment_video().
    Returns list of segment IDs that were newly ingested.
    """
    ingested: list[str] = []
    skipped: int = 0
    failed: list[tuple[int, str]] = []
    total = len(segment_metas)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(ingest_segment, meta, force): meta
            for meta in segment_metas
        }

        for future in as_completed(futures):
            meta = futures[future]
            idx = meta["segment_index"]
            try:
                seg_id = future.result()
                if seg_id is None:
                    skipped += 1
                else:
                    ingested.append(seg_id)
                    print(f"  [{len(ingested)}/{total}] {seg_id}")
            except Exception as e:
                failed.append((idx, str(e)))
                print(f"  [FAIL] segment {idx}: {e}")

    if skipped:
        print(f"  Skipped {skipped} already-ingested segment(s)")

    if failed:
        print(f"\n{len(failed)} segment(s) failed (after {MAX_RETRIES} retries):")
        for idx, err in sorted(failed):
            print(f"  segment {idx}: {err}")

    return ingested


def ingest_all_videos(max_workers: int = 4, force: bool = False) -> list[str]:
    """Ingest all segmented videos found in SEGMENTS_DIR.

    Discovers segment files on disk rather than re-segmenting.
    Returns list of all ingested segment IDs.
    """
    segments_root = Path(SEGMENTS_DIR)
    all_ingested: list[str] = []

    video_dirs = sorted(
        d for d in segments_root.iterdir()
        if d.is_dir() and any(d.glob("*.mp4"))
    )

    for video_dir in video_dirs:
        seg_files = sorted(video_dir.glob("*_seg_*.mp4"))
        if not seg_files:
            continue

        print(f"\nIngesting: {video_dir.name} ({len(seg_files)} segments)")

        segment_metas = []
        for seg_file in seg_files:
            # Parse index from filename: {stem}_seg_003.mp4 → 3
            match = re.search(r"_seg_(\d+)", seg_file.stem)
            if not match:
                continue
            idx = int(match.group(1))

            # Reconstruct metadata
            # file_name is the original video name (dir name + .mp4)
            segment_metas.append({
                "file_name": video_dir.name + ".mp4",
                "segment_path": str(seg_file),
                "segment_index": idx,
                "start_s": idx * (20 - 5),  # step size = SEGMENT_LENGTH - SEGMENT_OVERLAP
                "end_s": idx * (20 - 5) + 20,
            })

        ingested = ingest_video(segment_metas, max_workers=max_workers, force=force)
        all_ingested.extend(ingested)

    print(f"\nTotal ingested: {len(all_ingested)} segments")
    return all_ingested
