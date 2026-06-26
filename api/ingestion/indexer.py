"""
Parallel indexer: writes embedded chunks to Weaviate (dense) and
OpenSearch (BM25) simultaneously.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import asyncio
import weaviate
from opensearchpy import AsyncOpenSearch
from weaviate.auth import AuthApiKey

from api.core.config import get_settings
from api.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()

WEAVIATE_CLASS = "DocumentChunk"
OS_INDEX = "document_chunks"


# ── Client factories ──────────────────────────────────────────────────────────
def get_weaviate_client() -> weaviate.Client:
    return weaviate.Client(
        url=settings.weaviate_url,
        auth_client_secret=AuthApiKey(api_key=settings.weaviate_api_key),
    )


def get_opensearch_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=False,
        verify_certs=False,
        timeout=30,
    )


# ── Weaviate indexing ─────────────────────────────────────────────────────────
def index_chunks_weaviate(
    client: weaviate.Client,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Batch-import chunks + vectors into Weaviate. Returns count indexed."""
    if not chunks or not embeddings:
        return 0

    # Only index child chunks (or all chunks if not parent-child strategy)
    indexable = [
        (c, e)
        for c, e in zip(chunks, embeddings)
        if c.parent_id is not None or all(ch.parent_id is None for ch in chunks)
    ]

    with client.batch as batch:
        batch.batch_size = 50
        batch.dynamic = True

        for chunk, embedding in indexable:
            props = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "parent_id": chunk.parent_id or "",
                "chunk_index": chunk.chunk_index,
                "embedding_model": settings.embedding_model_version,
                "chunk_strategy": chunk.strategy,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            batch.add_data_object(
                data_object=props,
                class_name=WEAVIATE_CLASS,
                uuid=chunk.chunk_id,
                vector=embedding,
            )

    logger.info(f"Weaviate: indexed {len(indexable)} chunks")
    return len(indexable)


# ── OpenSearch indexing ───────────────────────────────────────────────────────
async def index_chunks_opensearch(
    client: AsyncOpenSearch,
    chunks: list[Chunk],
    source: str = "",
    collection: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> int:
    """Bulk-index all chunks (both parent and child) into OpenSearch."""
    if not chunks:
        return 0

    actions = []
    for chunk in chunks:
        actions.append({"index": {"_index": OS_INDEX, "_id": chunk.chunk_id}})
        actions.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "parent_id": chunk.parent_id or "",
            "chunk_index": chunk.chunk_index,
            "chunk_strategy": chunk.strategy,
            "embedding_model": settings.embedding_model_version,
            "source": source,
            "collection": collection or "",
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    resp = await client.bulk(body=actions)
    errors = [item for item in resp.get("items", []) if "error" in item.get("index", {})]
    if errors:
        logger.warning(f"OpenSearch bulk errors: {len(errors)}")

    count = len(chunks) - len(errors)
    logger.info(f"OpenSearch: indexed {count} chunks")
    return count


# ── Parent chunk lookup (for context retrieval) ───────────────────────────────
def get_parent_chunk(
    client: weaviate.Client, parent_id: str
) -> Optional[dict]:
    """Fetch parent chunk content from Weaviate by chunk_id."""
    try:
        result = (
            client.query.get(WEAVIATE_CLASS, ["content", "chunk_id", "document_id"])
            .with_where({
                "path": ["chunk_id"],
                "operator": "Equal",
                "valueText": parent_id,
            })
            .with_limit(1)
            .do()
        )
        hits = result.get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
        return hits[0] if hits else None
    except Exception as e:
        logger.error(f"Parent chunk lookup failed: {e}")
        return None


# ── Main indexer ──────────────────────────────────────────────────────────────
async def index_document(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    source: str = "",
    collection: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Index chunks into both Weaviate and OpenSearch in parallel.
    Only child chunks are vectorized in Weaviate.
    All chunks go into OpenSearch for BM25.
    """
    weaviate_client = get_weaviate_client()
    os_client = get_opensearch_client()

    try:
        # Run both indexing operations concurrently
        weaviate_count, os_count = await asyncio.gather(
            asyncio.to_thread(
                index_chunks_weaviate, weaviate_client, chunks, embeddings
            ),
            index_chunks_opensearch(os_client, chunks, source, collection, tags),
        )
        return {
            "weaviate_indexed": weaviate_count,
            "opensearch_indexed": os_count,
            "total_chunks": len(chunks),
        }
    finally:
        await os_client.close()
