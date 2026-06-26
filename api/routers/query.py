"""
POST /api/query — full 12-step RAG pipeline.
Design doc Section 2.3.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache.semantic_cache import get_cache
from api.core.auth import require_auth
from api.core.config import get_settings
from api.core.telemetry import (
    get_tracer,
    rag_cache_hit_total,
    rag_query_latency,
    rag_query_total,
    rag_reranker_latency,
    rag_retrieval_latency,
)
from api.db.postgres import QueryLog, get_db
from api.eval.ragas_eval import evaluate_query
from api.models.schemas import QueryRequest, QueryResponse, SourceChunk, RAGASScores, QueryLogEntry
from api.pipeline.compressor import compress_context, format_context_with_sources
from api.pipeline.generator import generate_with_fallback
from api.pipeline.reranker import rerank_chunks
from api.pipeline.retrieval.dense import dense_retrieve
from api.pipeline.retrieval.fusion import hybrid_retrieve
from api.pipeline.retrieval.sparse import sparse_retrieve
from api.pipeline.router import QueryClass, Strategy, route_query

router = APIRouter(prefix="/api", tags=["query"])
settings = get_settings()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(require_auth),
) -> QueryResponse:
    """
    Execute the full RAG pipeline:
    1. JWT validated (by require_auth)
    2. OTel trace started
    3. Semantic cache lookup
    4. Query routing
    5. Retrieval (dense / sparse / hybrid)
    6. Reranking
    7. Context compression
    8. LLM generation
    9. Response construction
    10. Async RAGAS evaluation (background)
    11. Logging to Postgres
    12. Return response
    """
    tracer = get_tracer()
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4()).replace("-", "")[:16]

    with tracer.start_as_current_span("rag.query") as root_span:
        root_span.set_attribute("rag.query_text", req.query[:200])
        root_span.set_attribute("rag.trace_id", trace_id)

        # ── Step 3: Semantic cache lookup ────────────────────────────
        cache_hit = False
        query_hash = hashlib.sha256(req.query.encode()).hexdigest()[:32]

        if req.use_cache:
            with tracer.start_as_current_span("rag.cache_lookup"):
                cache = get_cache()

                # Stampede prevention
                lock_acquired = await cache.acquire_lock(query_hash)
                if not lock_acquired:
                    # Another request is computing — wait for cache
                    cached = await cache.wait_for_cache(query_hash)
                else:
                    cached = await cache.lookup(req.query)

                if cached:
                    cache_hit = True
                    rag_cache_hit_total.labels(result="hit").inc()
                    rag_query_total.labels(
                        strategy=cached.get("strategy_used", "cached"),
                        cache_hit="true",
                    ).inc()
                    elapsed = int((time.perf_counter() - start_time) * 1000)
                    cached.pop("cache_hit", None)
                    return QueryResponse(
                        **cached,
                        cache_hit=True,
                        trace_id=trace_id,
                        latency_ms=elapsed,
                    )
                else:
                    rag_cache_hit_total.labels(result="miss").inc()

        # ── Step 4: Query routing ────────────────────────────────────
        with tracer.start_as_current_span("rag.router") as router_span:
            strategy, query_class = await route_query(req.query, req.strategy)
            router_span.set_attribute("rag.strategy", strategy.value)
            router_span.set_attribute("rag.query_class", query_class.value)

        # ── Steps 5: Retrieval ───────────────────────────────────────
        retrieval_start = time.perf_counter()
        with tracer.start_as_current_span(f"rag.{strategy.value}_retrieve") as ret_span:
            top_k = req.top_k * 4  # over-retrieve, then rerank down

            if strategy == Strategy.DENSE:
                chunks = await dense_retrieve(
                    req.query, top_k=top_k,
                    collection=req.collection, tags=req.tags,
                )
            elif strategy == Strategy.SPARSE:
                chunks = await sparse_retrieve(
                    req.query, top_k=top_k,
                    collection=req.collection, tags=req.tags,
                )
            else:  # hybrid
                chunks = await hybrid_retrieve(
                    req.query, top_k=top_k,
                    collection=req.collection, tags=req.tags,
                )

            ret_span.set_attribute("rag.num_results", len(chunks))

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        rag_retrieval_latency.labels(retriever=strategy.value).observe(
            retrieval_ms / 1000
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please ingest some documents first.",
            )

        # ── Step 6: Reranking ────────────────────────────────────────
        if req.use_reranker:
            rerank_start = time.perf_counter()
            with tracer.start_as_current_span("rag.rerank") as rr_span:
                chunks = await rerank_chunks(req.query, chunks, top_k=req.top_k)
                rr_span.set_attribute("rag.reranked_count", len(chunks))
            rag_reranker_latency.observe(
                (time.perf_counter() - rerank_start)
            )
        else:
            chunks = chunks[: req.top_k]

        # ── Step 7: Context compression ──────────────────────────────
        with tracer.start_as_current_span("rag.compress") as comp_span:
            if req.use_compression:
                compressed = await compress_context(req.query, chunks)
                context = format_context_with_sources(chunks, compressed)
            else:
                context = format_context_with_sources(chunks)
            comp_span.set_attribute("rag.context_length", len(context))

        # ── Step 8: LLM generation ───────────────────────────────────
        with tracer.start_as_current_span("rag.generate") as gen_span:
            gen_result = await generate_with_fallback(req.query, context)
            gen_span.set_attribute("rag.model", gen_result.get("model", ""))
            gen_span.set_attribute("rag.prompt_tokens", gen_result.get("prompt_tokens", 0))
            gen_span.set_attribute("rag.completion_tokens", gen_result.get("completion_tokens", 0))

        answer = gen_result.get("text", "")
        model_used = gen_result.get("model", "")

        # ── Step 9: Build response ───────────────────────────────────
        total_ms = int((time.perf_counter() - start_time) * 1000)
        source_chunks = [
            SourceChunk(
                chunk_id=c.chunk_id,
                content=c.content[:500],  # truncate for response size
                document_id=c.document_id,
                source=c.source,
                score=round(c.score, 4),
            )
            for c in chunks
        ]

        response_dict = {
            "answer": answer,
            "sources": [s.model_dump() for s in source_chunks],
            "strategy_used": strategy.value,
            "query_class": query_class.value,
            "cache_hit": False,
            "model_used": model_used,
        }

        # ── Step 11: Log to Postgres (non-blocking) ──────────────────
        log_id = uuid.uuid4()
        background_tasks.add_task(
            _log_query,
            log_id=log_id,
            query=req.query,
            query_class=query_class.value,
            strategy=strategy.value,
            answer=answer,
            model=model_used,
            latency_ms=total_ms,
            cache_hit=False,
            trace_id=trace_id,
        )

        # ── Step 10: Async RAGAS evaluation ──────────────────────────
        contexts = [c.content for c in chunks]
        background_tasks.add_task(
            evaluate_query,
            query_log_id=log_id,
            query=req.query,
            answer=answer,
            contexts=contexts,
        )

        # ── Cache the result ─────────────────────────────────────────
        if req.use_cache:
            background_tasks.add_task(
                get_cache().store, req.query, query_hash, response_dict
            )

        # ── Prometheus ───────────────────────────────────────────────
        rag_query_total.labels(strategy=strategy.value, cache_hit="false").inc()
        rag_query_latency.labels(strategy=strategy.value).observe(total_ms / 1000)

        root_span.set_attribute("rag.latency_ms", total_ms)
        root_span.set_attribute("rag.strategy_used", strategy.value)

        return QueryResponse(
            **response_dict,
            trace_id=trace_id,
            latency_ms=total_ms,
        )


async def _log_query(
    log_id: uuid.UUID,
    query: str,
    query_class: str,
    strategy: str,
    answer: str,
    model: str,
    latency_ms: int,
    cache_hit: bool,
    trace_id: str,
) -> None:
    from api.db.postgres import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        log = QueryLog(
            id=log_id,
            query=query,
            query_class=query_class,
            strategy_used=strategy,
            answer=answer,
            model_used=model,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            trace_id=trace_id,
        )
        db.add(log)
        await db.commit()


@router.get("/queries", response_model=list[QueryLogEntry])
async def get_queries(limit: int = 50, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    from api.db.postgres import QueryLog
    stmt = select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [
        QueryLogEntry(
            id=log.id,
            query=log.query,
            query_class=log.query_class,
            strategy_used=log.strategy_used,
            answer=log.answer,
            model_used=log.model_used,
            latency_ms=log.latency_ms,
            cache_hit=log.cache_hit,
            created_at=log.created_at.isoformat() if log.created_at else ""
        ) for log in logs
    ]
