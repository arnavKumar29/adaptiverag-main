"""
HTTP client for the BGE-M3 embedding microservice.
Used by the ingestion pipeline and retrieval components.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from api.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbedderClient:
    """Async HTTP client wrapping the embedder microservice."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0):
        self.base_url = base_url or settings.embedder_url
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def embed(
        self,
        texts: list[str],
        model_version: Optional[str] = None,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        if not texts:
            return []

        payload = {
            "texts": texts,
            "model_version": model_version or settings.embedding_model_version,
            "normalize": normalize,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()

        logger.debug(
            f"Embedded {len(texts)} texts, "
            f"cache_hits={data.get('cached_count', 0)}, "
            f"dim={data.get('dim', 0)}"
        )
        return data["embeddings"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed([text])
        return results[0] if results else []

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            return resp.json()


# Module-level singleton
_client: Optional[EmbedderClient] = None


def get_embedder() -> EmbedderClient:
    global _client
    if _client is None:
        _client = EmbedderClient()
    return _client
