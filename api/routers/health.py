"""Health check endpoint — probes all backing services."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx
import redis.asyncio as aioredis
import weaviate
from fastapi import APIRouter
from opensearchpy import AsyncOpenSearch
from sqlalchemy import text
from weaviate.auth import AuthApiKey

from api.core.config import get_settings
from api.db.postgres import AsyncSessionLocal
from api.models.schemas import HealthResponse, ServiceHealth

router = APIRouter(tags=["health"])
settings = get_settings()
logger = logging.getLogger(__name__)


async def _check(name: str, coro) -> ServiceHealth:
    try:
        start = time.perf_counter()
        await coro
        ms = round((time.perf_counter() - start) * 1000, 1)
        return ServiceHealth(name=name, status="ok", latency_ms=ms)
    except Exception as e:
        return ServiceHealth(name=name, status="error", detail=str(e)[:100])


async def _check_postgres() -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(text("SELECT 1"))


async def _check_redis() -> None:
    r = aioredis.from_url(settings.redis_url)
    await r.ping()
    await r.aclose()


async def _check_weaviate() -> None:
    async with httpx.AsyncClient(timeout=5) as c:
        resp = await c.get(f"{settings.weaviate_url}/v1/.well-known/ready")
        resp.raise_for_status()


async def _check_opensearch() -> None:
    os = AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=False,
    )
    await os.cluster.health()
    await os.close()


async def _check_ollama() -> None:
    async with httpx.AsyncClient(timeout=5) as c:
        resp = await c.get(f"{settings.ollama_url}/api/tags")
        resp.raise_for_status()


async def _check_embedder() -> None:
    async with httpx.AsyncClient(timeout=5) as c:
        resp = await c.get(f"{settings.embedder_url}/health")
        resp.raise_for_status()

@router.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    checks = await asyncio.gather(
        _check("postgres", _check_postgres()),
        _check("redis", _check_redis()),
        _check("weaviate", _check_weaviate()),
        _check("opensearch", _check_opensearch()),
        _check("ollama", _check_ollama()),
        _check("embedder", _check_embedder()),
    )
    all_ok = all(s.status == "ok" for s in checks)
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        version="1.0.0",
        services=list(checks),
    )
