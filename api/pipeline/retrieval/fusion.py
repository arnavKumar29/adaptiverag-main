"""
Reciprocal Rank Fusion (RRF) to merge dense + sparse retrieval results.
RRF_score(d) = sum(1 / (k + rank_i(d)))   where k = 60
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from api.pipeline.retrieval.dense import RetrievedChunk

RRF_K = 60  # standard constant


def reciprocal_rank_fusion(
    *result_lists: list[RetrievedChunk],
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """
    Merge multiple ranked result lists using RRF.
    Robust to score scale differences between dense (cosine 0-1)
    and sparse (BM25 unbounded, normalized).

    Args:
        *result_lists: One or more ranked lists of RetrievedChunk.
        top_k:         Maximum number of results to return.

    Returns:
        Merged, deduplicated list sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (RRF_K + rank)
            # Keep the chunk object (prefer higher-score version)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
            elif chunk.score > chunk_map[chunk.chunk_id].score:
                chunk_map[chunk.chunk_id] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    results: list[RetrievedChunk] = []
    for cid in sorted_ids[:top_k]:
        chunk = chunk_map[cid]
        chunk.score = rrf_scores[cid]   # replace original score with RRF score
        chunk.retriever = "hybrid"
        results.append(chunk)

    return results


async def hybrid_retrieve(
    query: str,
    top_k: int = 20,
    collection: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[RetrievedChunk]:
    """Run dense + sparse retrieval in parallel, then fuse with RRF."""
    import asyncio
    from api.pipeline.retrieval.dense import dense_retrieve
    from api.pipeline.retrieval.sparse import sparse_retrieve

    dense_results, sparse_results = await asyncio.gather(
        dense_retrieve(query, top_k=top_k, collection=collection, tags=tags),
        sparse_retrieve(query, top_k=top_k, collection=collection, tags=tags),
    )
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
