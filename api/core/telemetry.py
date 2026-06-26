"""
OpenTelemetry setup + Prometheus metrics registration.
All 9 trace spans + 7 Prometheus metrics from the design document.
"""
from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram

from api.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Prometheus Metrics ────────────────────────────────────────────────────────
# Design doc Section 15.2

rag_query_total = Counter(
    "rag_query_total",
    "Total queries processed",
    ["strategy", "cache_hit"],
)

rag_query_latency = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency",
    ["strategy"],
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
)

rag_cache_hit_total = Counter(
    "rag_cache_hit_total",
    "Semantic cache hits vs misses",
    ["result"],  # 'hit' or 'miss'
)

rag_ragas_faithfulness = Gauge(
    "rag_ragas_faithfulness",
    "Rolling mean RAGAS faithfulness score (last 100 queries)",
)

rag_retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval latency by type",
    ["retriever"],  # dense, sparse, hybrid
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

rag_reranker_latency = Histogram(
    "rag_reranker_latency_seconds",
    "Cross-encoder reranker latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0],
)

rag_llm_tokens_total = Counter(
    "rag_llm_tokens_total",
    "LLM tokens consumed",
    ["model", "token_type"],  # token_type: prompt or completion
)

rag_ingestion_total = Counter(
    "rag_ingestion_total",
    "Documents ingested",
    ["source_type", "status"],
)

# ── OpenTelemetry Tracer ──────────────────────────────────────────────────────

_tracer: trace.Tracer | None = None


def setup_telemetry() -> None:
    """Call once at application startup."""
    global _tracer

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": "1.0.0",
        "deployment.environment": settings.environment,
    })

    provider = TracerProvider(resource=resource)

    try:
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(f"OTel exporter → {settings.otel_exporter_otlp_endpoint}")
    except Exception as e:
        logger.warning(f"OTel exporter setup failed (traces won't be exported): {e}")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.otel_service_name)

    # Auto-instrument FastAPI and HTTPX
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

    logger.info("OpenTelemetry tracing configured.")


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(settings.otel_service_name)
    return _tracer
