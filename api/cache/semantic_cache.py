"""
Two-layer semantic cache:
  Layer 1: Weaviate ANN search on past query embeddings (similarity > threshold)
  Layer 2: Redis stores full response JSON (TTL 1hr)
  + Redis mutex for cache stampede prevention
Design doc Section 14.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import redis.asyncio as aioredis
import weaviate
from weaviate.auth import AuthApiKey

from api.core.config import get_settings
from api.ingestion.embedder import get_embedder

logger = logging.getLogger(__name__)
settings = get_settings()

CACHE_CLASS = "CachedQuery"
CACHE_TTL = 3600        # 1 hour
LOCK_TTL = 10           # 10s mutex
STAMPEDE_WAIT = 8       # seconds to wait if lock held


class SemanticCache:
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._weaviate: Optional[weaviate.Client] = None

    def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    def _get_weaviate(self) -> weaviate.Client:
        if self._weaviate is None:
            self._weaviate = weaviate.Client(
                url=settings.weaviate_url,
                auth_client_secret=AuthApiKey(api_key=settings.weaviate_api_key),
            )
        return self._weaviate

    async def lookup(self, query: str) -> Optional[dict]:
        """
        Check semantic cache. Returns cached response dict or None.
        Uses cosine similarity > threshold to find semantically equivalent past queries.
        """
        redis = self._get_redis()
        wv = self._get_weaviate()
        embedder = get_embedder()

        # Embed the query
        query_vector = await embedder.embed_single(query)

        # Search Weaviate for similar past queries
        try:
            result = (
                wv.query.get(CACHE_CLASS, ["cache_key", "query_text"])
                .with_near_vector({
                    "vector": query_vector,
                    "certainty": settings.semantic_cache_threshold,
                })
                .with_limit(1)
                .with_additional(["certainty"])
                .do()
            )
            hits = (
                result.get("data", {}).get("Get", {}).get(CACHE_CLASS, [])
            )
        except Exception as e:
            logger.warning(f"Cache lookup Weaviate error: {e}")
            return None

        if not hits:
            return None

        cache_key = hits[0].get("cache_key")
        certainty = hits[0].get("_additional", {}).get("certainty", 0.0)
        logger.debug(f"Cache ANN hit: certainty={certainty:.3f}, key={cache_key}")

        # Fetch full response from Redis
        cached_json = await redis.get(f"semantic_cache:{cache_key}")
        if cached_json:
            return json.loads(cached_json)

        return None

    async def store(self, query: str, query_hash: str, response: dict) -> None:
        """Store query vector in Weaviate + full response in Redis."""
        redis = self._get_redis()
        wv = self._get_weaviate()
        embedder = get_embedder()

        try:
            query_vector = await embedder.embed_single(query)

            # Store in Weaviate
            from datetime import datetime, timezone
            wv.data_object.create(
                data_object={
                    "cache_key": query_hash,
                    "query_text": query[:500],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                class_name=CACHE_CLASS,
                vector=query_vector,
            )
        except Exception as e:
            logger.warning(f"Failed to store query vector in Weaviate: {e}")

        # Store response in Redis (TTL 1hr)
        await redis.setex(
            f"semantic_cache:{query_hash}",
            CACHE_TTL,
            json.dumps(response),
        )

    async def acquire_lock(self, query_hash: str) -> bool:
        """
        Redis mutex to prevent cache stampede.
        Returns True if lock acquired (this caller should compute + cache).
        Returns False if another caller holds the lock (wait for cache).
        """
        redis = self._get_redis()
        acquired = await redis.set(
            f"query_lock:{query_hash}", "1", nx=True, ex=LOCK_TTL
        )
        return bool(acquired)

    async def wait_for_cache(self, query_hash: str) -> Optional[dict]:
        """Wait up to STAMPEDE_WAIT seconds for cache to be populated."""
        redis = self._get_redis()
        deadline = time.time() + STAMPEDE_WAIT
        while time.time() < deadline:
            cached = await redis.get(f"semantic_cache:{query_hash}")
            if cached:
                return json.loads(cached)
            import asyncio
            await asyncio.sleep(0.5)
        return None

    async def invalidate_by_document(self, document_id: str) -> int:
        """
        Document-triggered invalidation: remove all cache entries
        that reference chunks from this document.
        (Simplified: scan Redis for entries mentioning document_id)
        """
        redis = self._get_redis()
        pattern = "semantic_cache:*"
        count = 0
        async for key in redis.scan_iter(pattern):
            val = await redis.get(key)
            if val and document_id in val:
                await redis.delete(key)
                count += 1
        logger.info(f"Invalidated {count} cache entries for document {document_id}")
        return count

    async def clear_all(self) -> int:
        """Clear all semantic cache entries from Redis."""
        redis = self._get_redis()
        keys = [k async for k in redis.scan_iter("semantic_cache:*")]
        if keys:
            await redis.delete(*keys)
        return len(keys)

    async def stats(self) -> dict:
        """Return cache statistics."""
        redis = self._get_redis()
        keys = [k async for k in redis.scan_iter("semantic_cache:*")]
        # Count hits in last hour from query_logs (simplified)
        return {
            "total_entries": len(keys),
            "cache_size_bytes": sum(
                await redis.memory_usage(k) or 0 for k in keys[:100]
            ),
        }


# Module singleton
_cache: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
