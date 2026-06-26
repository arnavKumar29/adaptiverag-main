"""
BGE-M3 Embedding microservice.
FastAPI app exposing POST /embed — runs inference with sentence-transformers.
Redis caches embeddings by SHA256(text + model_version) for 7 days.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
MODEL_VERSION = os.getenv("EMBEDDING_MODEL_VERSION", "bge-m3-v1")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
CACHE_TTL = 7 * 24 * 3600  # 7 days

# Global model (loaded once at startup)
_model: Optional[SentenceTransformer] = None
_redis: Optional[aioredis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _redis
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    _model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Model loaded. Embedding dim: {_model.get_sentence_embedding_dimension()}")
    _redis = aioredis.from_url(REDIS_URL, decode_responses=False)
    yield
    if _redis:
        await _redis.aclose()


app = FastAPI(title="BGE-M3 Embedding Service", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    texts: list[str]
    model_version: str = MODEL_VERSION
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model_version: str
    cached_count: int
    dim: int


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_key(text: str, model_version: str) -> str:
    digest = hashlib.sha256(f"{text}::{model_version}".encode()).hexdigest()
    return f"embedding_cache:{digest}"


async def _get_cached(text: str, model_version: str) -> Optional[np.ndarray]:
    if _redis is None:
        return None
    key = _cache_key(text, model_version)
    val = await _redis.get(key)
    if val:
        return pickle.loads(val)
    return None


async def _set_cached(text: str, model_version: str, emb: np.ndarray) -> None:
    if _redis is None:
        return
    key = _cache_key(text, model_version)
    await _redis.setex(key, CACHE_TTL, pickle.dumps(emb))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")

    embeddings: list[Optional[np.ndarray]] = [None] * len(req.texts)
    to_compute: list[tuple[int, str]] = []
    cached_count = 0

    # Check cache
    for i, text in enumerate(req.texts):
        cached = await _get_cached(text, req.model_version)
        if cached is not None:
            embeddings[i] = cached
            cached_count += 1
        else:
            to_compute.append((i, text))

    # Batch compute uncached
    if to_compute and _model is not None:
        indices, texts_batch = zip(*to_compute)
        # Process in batches of BATCH_SIZE
        computed: list[np.ndarray] = []
        for start in range(0, len(texts_batch), BATCH_SIZE):
            batch = list(texts_batch[start : start + BATCH_SIZE])
            vecs = _model.encode(
                batch,
                normalize_embeddings=req.normalize,
                show_progress_bar=False,
                batch_size=BATCH_SIZE,
            )
            computed.extend(vecs)

        # Store computed embeddings + cache
        for idx, (orig_i, text) in enumerate(to_compute):
            emb = computed[idx]
            embeddings[orig_i] = emb
            await _set_cached(text, req.model_version, emb)

    result = [e.tolist() for e in embeddings if e is not None]
    dim = len(result[0]) if result else 0

    return EmbedResponse(
        embeddings=result,
        model_version=req.model_version,
        cached_count=cached_count,
        dim=dim,
    )


@app.get("/health")
async def health():
    model_ok = _model is not None
    try:
        redis_ok = await _redis.ping() if _redis else False
    except Exception:
        redis_ok = False
    return {
        "status": "ok" if model_ok else "degraded",
        "model": EMBEDDING_MODEL,
        "model_loaded": model_ok,
        "redis": redis_ok,
    }


@app.get("/info")
async def info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model": EMBEDDING_MODEL,
        "version": MODEL_VERSION,
        "dim": _model.get_sentence_embedding_dimension(),
        "max_seq_length": _model.max_seq_length,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
