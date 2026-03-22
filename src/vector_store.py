"""Pinecone client — vector storage + semantic similarity search."""
from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION


def _get_index():
    """Get or create the Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(PINECONE_INDEX_NAME)


# Module-level lazy singleton
_index = None


def _idx():
    global _index
    if _index is None:
        _index = _get_index()
    return _index


# --- Write ---

def upsert_segment(
    segment_id: str,
    vector: list[float],
    metadata: dict,
) -> None:
    """Upsert a single segment vector with metadata.

    metadata should include: file_name, start_s, end_s, max_volume, time_of_day.
    """
    _idx().upsert(vectors=[(segment_id, vector, metadata)])


def upsert_segments(
    ids: list[str],
    vectors: list[list[float]],
    metadatas: list[dict],
    batch_size: int = 100,
) -> None:
    """Upsert multiple segment vectors in batches."""
    records = list(zip(ids, vectors, metadatas))
    for i in range(0, len(records), batch_size):
        _idx().upsert(vectors=records[i : i + batch_size])


# --- Search ---

def search(
    query_vector: list[float],
    *,
    top_k: int = 50,
    filter: dict | None = None,
) -> list[dict]:
    """Semantic similarity search.

    Returns list of dicts with: id, score, metadata.
    filter example: {"time_of_day": "night", "max_volume": {"$in": ["loud", "shouting"]}}
    """
    results = _idx().query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
    )
    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
        }
        for match in results.matches
    ]


# --- Delete ---

def delete_segment(segment_id: str) -> None:
    """Delete a single segment vector."""
    _idx().delete(ids=[segment_id])


def delete_segments_by_file(file_name: str) -> None:
    """Delete all vectors for a given video file."""
    _idx().delete(filter={"file_name": file_name})
