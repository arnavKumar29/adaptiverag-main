"""
Dense retrieval via Weaviate HNSW nearest-neighbour search.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import weaviate
from weaviate.auth import AuthApiKey

from api.core.config import get_settings
from api.ingestion.embedder import get_embedder

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    parent_id: Optional[str]
    source: str
    score: float
    retriever: str = "dense"


def _get_client() -> weaviate.Client:
    return weaviate.Client(
        url=settings.weaviate_url,
        auth_client_secret=AuthApiKey(api_key=settings.weaviate_api_key),
    )


async def dense_retrieve(
    query: str,
    top_k: int = 20,
    collection: Optional[str] = None,
    tags: Optional[list[str]] = None,
    embedding_model_version: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Embed the query then run Weaviate nearVector ANN search.
    Returns top_k chunks sorted by cosine similarity (descending).
    """
    embedder = get_embedder()
    query_vector = await embedder.embed_single(query)

    client = _get_client()

    # Build the query
    query_builder = (
        client.query.get(
            "DocumentChunk",
            ["chunk_id", "document_id", "content", "parent_id", "source"],
        )
        .with_near_vector({"vector": query_vector, "certainty": 0.5})
        .with_limit(top_k)
        .with_additional(["certainty", "id"])
    )

    # Optional filters
    filters: list[dict] = []
    if collection:
        filters.append({
            "path": ["collection"],
            "operator": "Equal",
            "valueText": collection,
        })
    if tags:
        filters.append({
            "path": ["tags"],
            "operator": "ContainsAny",
            "valueText": tags,
        })
    if embedding_model_version:
        filters.append({
            "path": ["embedding_model"],
            "operator": "Equal",
            "valueText": embedding_model_version,
        })

    if filters:
        where_filter = (
            {"operator": "And", "operands": filters}
            if len(filters) > 1
            else filters[0]
        )
        query_builder = query_builder.with_where(where_filter)

    try:
        result = query_builder.do()
        hits = (
            result.get("data", {})
            .get("Get", {})
            .get("DocumentChunk", [])
        )
    except Exception as e:
        logger.error(f"Dense retrieval error: {e}")
        return []

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        certainty = hit.get("_additional", {}).get("certainty", 0.0)
        chunks.append(
            RetrievedChunk(
                chunk_id=hit.get("chunk_id", ""),
                document_id=hit.get("document_id", ""),
                content=hit.get("content", ""),
                parent_id=hit.get("parent_id") or None,
                source=hit.get("source", ""),
                score=float(certainty),
                retriever="dense",
            )
        )

    logger.debug(f"Dense retrieval: {len(chunks)} results for query '{query[:60]}...'")
    return chunks
