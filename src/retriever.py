"""Dual retrieval (FTS + vector) with Reciprocal Rank Fusion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.config import RRF_K, TOP_K_RETRIEVAL
from src.providers import get_embedding_provider
from src import sql_store, vector_store


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a single result at a given rank.

    score = 1 / (k + rank), where rank is 1-based.
    k controls how much top ranks dominate — higher k flattens the curve.
    """
    return 1.0 / (k + rank)


def _fuse_results(
    fts_results: list[dict],
    vector_results: list[dict],
) -> list[dict]:
    """Fuse FTS and vector results using RRF.

    Both inputs are ranked lists (best first). Output is a unified list
    sorted by combined RRF score, with full segment data from Supabase.

    FTS results have full segment data (from Supabase RPC).
    Vector results only have id, score, and light metadata.
    For vector-only hits, we fetch full data from Supabase.
    """
    scores: dict[str, float] = {}
    segment_data: dict[str, dict] = {}

    # Score FTS results (already ranked by ts_rank)
    for rank, row in enumerate(fts_results, start=1):
        seg_id = row["id"]
        scores[seg_id] = scores.get(seg_id, 0) + _rrf_score(rank)
        segment_data[seg_id] = row

    # Score vector results (already ranked by cosine similarity)
    for rank, hit in enumerate(vector_results, start=1):
        seg_id = hit["id"]
        scores[seg_id] = scores.get(seg_id, 0) + _rrf_score(rank)
        # Vector hits don't carry full text — fetch from Supabase if needed
        if seg_id not in segment_data:
            row = sql_store.get_segment(seg_id)
            if row:
                segment_data[seg_id] = row

    # Sort by combined RRF score
    ranked_ids = sorted(scores, key=lambda sid: scores[sid], reverse=True)

    results = []
    for seg_id in ranked_ids:
        data = segment_data.get(seg_id)
        if data is None:
            continue
        results.append({
            **data,
            "rrf_score": scores[seg_id],
        })

    return results


def retrieve(
    query: str,
    *,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """Run FTS and vector search in parallel, fuse with RRF.

    No hard filters — metadata like time_of_day or volume level is captured
    in the scene descriptions and embeddings, so it influences ranking
    naturally. The reranker makes the final relevance judgement.

    Returns ranked list of segment dicts with an added `rrf_score` field.
    """
    embedder = get_embedding_provider()
    query_vector = embedder.embed(query)

    # Run both searches in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        fts_future = pool.submit(
            sql_store.search_fts,
            query,
            limit=top_k,
        )
        vector_future = pool.submit(
            vector_store.search,
            query_vector,
            top_k=top_k,
        )

        fts_results = fts_future.result()
        vector_results = vector_future.result()

    return _fuse_results(fts_results, vector_results)[:top_k]
