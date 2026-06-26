"""
BGE Reranker microservice.
FastAPI app exposing POST /rerank — cross-encoder scoring of (query, chunk) pairs.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_model: Optional[CrossEncoder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info(f"Loading reranker model: {RERANKER_MODEL}")
    _model = CrossEncoder(RERANKER_MODEL, max_length=512)
    logger.info("Reranker model loaded.")
    yield


app = FastAPI(title="BGE Reranker Service", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    texts: list[str]   # list of chunk contents
    top_k: int = 5


class ScoredText(BaseModel):
    text: str
    original_index: int
    score: float


class RerankResponse(BaseModel):
    results: list[ScoredText]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.texts:
        return RerankResponse(results=[])

    pairs = [[req.query, text] for text in req.texts]

    import asyncio
    scores = await asyncio.to_thread(
        _model.predict, pairs, show_progress_bar=False
    )

    scored = [
        ScoredText(text=text, original_index=i, score=float(score))
        for i, (text, score) in enumerate(zip(req.texts, scores))
    ]
    scored.sort(key=lambda x: x.score, reverse=True)

    return RerankResponse(results=scored[: req.top_k])


@app.get("/health")
async def health():
    return {"status": "ok" if _model is not None else "loading", "model": RERANKER_MODEL}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
