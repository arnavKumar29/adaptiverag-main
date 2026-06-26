"""
Cross-encoder reranker pipeline component.
Calls the reranker microservice and merges scores back onto chunks.
Also handles parent-chunk context expansion.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from api.core.config import get_settings
from api.pipeline.retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """
    Send (query, chunk) pairs to the reranker microservice.
    Returns top_k chunks re-sorted by cross-encoder score.
    """
    if not chunks:
        return []

    top_k = top_k or settings.reranker_top_k
    texts = [c.content for c in chunks]
    chunk_map = {c.content: c for c in chunks}

    payload = {"query": query, "texts": texts, "top_k": top_k}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.reranker_url}/rerank", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Reranker unavailable, returning unranked results: {e}")
        return chunks[:top_k]

    reranked: list[RetrievedChunk] = []
    for scored in data["results"]:
        original_chunk = chunk_map.get(scored["text"])
        if original_chunk:
            original_chunk.score = scored["score"]
            reranked.append(original_chunk)

    logger.debug(f"Reranked {len(chunks)} → top {len(reranked)} chunks")
    return reranked


async def expand_to_parents(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    For parent-child chunked documents, replace child chunks with their parent
    chunks to provide richer context to the LLM.
    Deduplicates by parent_id.
    """
    import weaviate
    from weaviate.auth import AuthApiKey
    from api.ingestion.indexer import get_parent_chunk

    wv_client = weaviate.Client(
        url=settings.weaviate_url,
        auth_client_secret=AuthApiKey(api_key=settings.weaviate_api_key),
    )

    seen_parents: set[str] = set()
    expanded: list[RetrievedChunk] = []

    for chunk in chunks:
        if chunk.parent_id and chunk.parent_id not in seen_parents:
            parent_data = await asyncio.to_thread(
                get_parent_chunk, wv_client, chunk.parent_id
            )
            if parent_data:
                seen_parents.add(chunk.parent_id)
                expanded.append(
                    RetrievedChunk(
                        chunk_id=parent_data.get("chunk_id", chunk.parent_id),
                        document_id=chunk.document_id,
                        content=parent_data.get("content", chunk.content),
                        parent_id=None,
                        source=chunk.source,
                        score=chunk.score,
                        retriever=chunk.retriever,
                    )
                )
            else:
                expanded.append(chunk)
        elif not chunk.parent_id:
            expanded.append(chunk)

    return expanded


import asyncio  # noqa: E402 (placed here to avoid circular import)
