"""
Sparse retrieval via OpenSearch BM25.
"""
from __future__ import annotations

import logging
from typing import Optional

from opensearchpy import AsyncOpenSearch

from api.core.config import get_settings
from api.pipeline.retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

OS_INDEX = "document_chunks"


def _get_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=False,
        verify_certs=False,
        timeout=30,
    )


async def sparse_retrieve(
    query: str,
    top_k: int = 20,
    collection: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[RetrievedChunk]:
    """
    OpenSearch BM25 multi_match query.
    No embedding needed — pure keyword/term-based retrieval.
    """
    client = _get_client()

    # Base BM25 query
    must_clause: dict = {
        "multi_match": {
            "query": query,
            "fields": ["content^2"],   # boost content field
            "type": "best_fields",
            "fuzziness": "AUTO",
        }
    }

    # Filters
    filter_clauses: list[dict] = []
    if collection:
        filter_clauses.append({"term": {"collection": collection}})
    if tags:
        filter_clauses.append({"terms": {"tags": tags}})

    os_query: dict = {
        "query": {
            "bool": {
                "must": [must_clause],
                "filter": filter_clauses,
            }
        },
        "size": top_k,
        "_source": ["chunk_id", "document_id", "content", "parent_id", "source"],
    }

    try:
        resp = await client.search(index=OS_INDEX, body=os_query)
        hits = resp.get("hits", {}).get("hits", [])
    except Exception as e:
        logger.error(f"Sparse retrieval error: {e}")
        return []
    finally:
        await client.close()

    max_score = max((h["_score"] for h in hits), default=1.0) or 1.0

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        src = hit.get("_source", {})
        chunks.append(
            RetrievedChunk(
                chunk_id=src.get("chunk_id", hit["_id"]),
                document_id=src.get("document_id", ""),
                content=src.get("content", ""),
                parent_id=src.get("parent_id") or None,
                source=src.get("source", ""),
                score=hit["_score"] / max_score,  # normalize to [0, 1]
                retriever="sparse",
            )
        )

    logger.debug(f"Sparse retrieval: {len(chunks)} results for '{query[:60]}...'")
    return chunks
