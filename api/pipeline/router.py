"""
Query router — all four versions from the design document.
v1: Rule-based (heuristics)
v2: ML classifier (LogisticRegression, trained after 500+ logs)
v3: LLM-based (structured JSON output)
v4: Self-learning (Thompson Sampling multi-armed bandit)
Active version is selected by ROUTER_VERSION env var (default: v1).
"""
from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from typing import Optional

import redis.asyncio as aioredis

from api.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

ROUTER_VERSION = os.getenv("ROUTER_VERSION", "v1")
CACHE_TTL = 300  # 5 minutes


class Strategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QueryClass(str, Enum):
    CONCEPTUAL = "conceptual"
    KEYWORD = "keyword"
    MIXED = "mixed"


# ── Redis client ──────────────────────────────────────────────────────────────
_redis: Optional[aioredis.Redis] = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


# ── v1: Rule-based Router (design doc Section 7.1) ────────────────────────────
_SPARSE_SIGNALS = {
    "code", "function", "error", "version", "syntax", "class", "method",
    "import", "library", "api", "endpoint", "parameter", "argument",
    "exception", "traceback", "bug", "fix",
}
_DENSE_SIGNALS = {
    "explain", "why", "how", "concept", "mean", "understand", "describe",
    "overview", "summary", "difference", "compare", "relationship", "similar",
    "what is", "tell me", "elaborate",
}


def classify_query(query: str) -> QueryClass:
    tokens = set(query.lower().split())
    # Check multi-word signals
    ql = query.lower()
    has_dense = bool(tokens & _DENSE_SIGNALS) or any(s in ql for s in _DENSE_SIGNALS)
    has_sparse = bool(tokens & _SPARSE_SIGNALS)
    has_quoted = bool(re.search(r'"[^"]+"', query))

    if has_sparse or has_quoted:
        return QueryClass.KEYWORD
    if has_dense:
        return QueryClass.CONCEPTUAL
    return QueryClass.MIXED


def route_v1(query: str) -> tuple[Strategy, QueryClass]:
    """Rule-based router."""
    qclass = classify_query(query)
    strategy_map = {
        QueryClass.KEYWORD: Strategy.SPARSE,
        QueryClass.CONCEPTUAL: Strategy.DENSE,
        QueryClass.MIXED: Strategy.HYBRID,
    }
    return strategy_map[qclass], qclass


# ── v2: ML Classifier Router (design doc Section 7.2) ────────────────────────
def route_v2(query: str) -> tuple[Strategy, QueryClass]:
    """
    Logistic regression classifier. Falls back to v1 if model not trained yet.
    Model is retrained weekly via GitHub Actions and saved to models/router_clf.pkl.
    """
    import pickle
    from pathlib import Path

    model_path = Path("models/router_clf.pkl")
    if not model_path.exists():
        logger.debug("ML router model not found, falling back to v1")
        return route_v1(query)

    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        features = _extract_features(query)
        pred = clf.predict([features])[0]
        proba = clf.predict_proba([features])[0]
        confidence = max(proba)

        # Low confidence → fall back to v1
        if confidence < 0.6:
            return route_v1(query)

        strategy = Strategy(pred)
        qclass = classify_query(query)
        return strategy, qclass
    except Exception as e:
        logger.warning(f"ML router failed: {e}, falling back to v1")
        return route_v1(query)


def _extract_features(query: str) -> list[float]:
    """Extract 6 features for the ML router."""
    import re
    words = query.split()

    question_words = {"what", "why", "how", "when", "which", "who", "where"}
    has_question_word = float(
        any(w.lower() in question_words for w in words[:3])
    )
    has_quoted = float(bool(re.search(r'"[^"]+"', query)))

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query[:500])
        entity_count = float(len(doc.ents))
        sentence_count = float(len(list(doc.sents)))
    except Exception:
        entity_count = 0.0
        sentence_count = 1.0

    return [
        float(len(words)),                              # query length
        float(sum(len(w) for w in words) / max(len(words), 1)),  # avg word length
        has_question_word,
        has_quoted,
        entity_count,
        sentence_count,
    ]


# ── v3: LLM Router (design doc Section 7.3) ──────────────────────────────────
async def route_v3(query: str) -> tuple[Strategy, QueryClass]:
    """LLM-based router using structured JSON output. Fastest local model."""
    import httpx

    prompt = f"""You are a retrieval strategy classifier. Given a query, output ONLY valid JSON:
{{"strategy": "dense"|"sparse"|"hybrid", "query_class": "conceptual"|"keyword"|"mixed", "confidence": 0.0-1.0}}

Rules:
- dense: conceptual/semantic queries (why, how, explain, overview)
- sparse: exact keyword/code/name queries (function names, error codes, quoted strings)
- hybrid: mixed queries needing both

Query: {query}
JSON:"""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            text = resp.json().get("response", "")
            # Extract JSON from response
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                strategy = Strategy(data.get("strategy", "hybrid"))
                qclass = QueryClass(data.get("query_class", "mixed"))
                return strategy, qclass
    except Exception as e:
        logger.warning(f"LLM router failed: {e}, falling back to v1")

    return route_v1(query)


# ── v4: Self-learning Router (Thompson Sampling) ──────────────────────────────
async def route_v4(query: str) -> tuple[Strategy, QueryClass]:
    """
    Multi-armed bandit (Thompson Sampling) over router_weights in Redis/Postgres.
    Uses Beta(alpha, beta) distribution to balance exploration/exploitation.
    """
    import numpy as np
    from api.db.postgres import AsyncSessionLocal, RouterWeight
    from sqlalchemy import select

    qclass = classify_query(query)
    strategies = [Strategy.DENSE, Strategy.SPARSE, Strategy.HYBRID]

    # Load weights from Redis cache
    redis = _get_redis()
    cache_key = f"router_weights:{qclass.value}"
    cached = await redis.get(cache_key)

    if cached:
        weights = json.loads(cached)
    else:
        # Load from Postgres
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(RouterWeight).where(RouterWeight.query_class == qclass.value)
            )
            rows = result.scalars().all()
            weights = {
                r.strategy: {"alpha": r.alpha, "beta": r.beta}
                for r in rows
            }
        # Defaults if no data yet
        for s in strategies:
            if s.value not in weights:
                weights[s.value] = {"alpha": 1.0, "beta": 1.0}
        await redis.setex(cache_key, CACHE_TTL, json.dumps(weights))

    # Thompson sampling: sample from Beta distribution for each strategy
    samples = {}
    for s in strategies:
        w = weights.get(s.value, {"alpha": 1.0, "beta": 1.0})
        samples[s] = np.random.beta(w["alpha"], w["beta"])

    best_strategy = max(samples, key=lambda k: samples[k])
    return best_strategy, qclass


# ── Thompson Sampling weight update (called after RAGAS eval) ─────────────────
async def update_thompson_weights(
    query_class: str,
    strategy: str,
    faithfulness: float,
    threshold: float = 0.8,
) -> None:
    """Update alpha/beta in Postgres + invalidate Redis cache."""
    from api.db.postgres import AsyncSessionLocal, RouterWeight
    from sqlalchemy import select

    success = faithfulness > threshold

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(RouterWeight).where(
                RouterWeight.query_class == query_class,
                RouterWeight.strategy == strategy,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = RouterWeight(
                query_class=query_class,
                strategy=strategy,
                alpha=1.0,
                beta=1.0,
                sample_count=0,
            )
            db.add(row)
        if success:
            row.alpha = (row.alpha or 1.0) + 1.0
        else:
            row.beta = (row.beta or 1.0) + 1.0
        row.sample_count = (row.sample_count or 0) + 1
        await db.commit()

    # Invalidate Redis cache
    redis = _get_redis()
    await redis.delete(f"router_weights:{query_class}")


# ── Main dispatch ─────────────────────────────────────────────────────────────
async def route_query(
    query: str,
    override_strategy: Optional[str] = None,
) -> tuple[Strategy, QueryClass]:
    """
    Route a query to the appropriate retrieval strategy.
    override_strategy: if set and not 'auto', bypass routing.
    """
    if override_strategy and override_strategy != "auto":
        qclass = classify_query(query)
        return Strategy(override_strategy), qclass

    version_map = {
        "v1": lambda: (None, route_v1(query)),
        "v2": lambda: (None, route_v2(query)),
    }

    # Async versions need await
    if ROUTER_VERSION == "v3":
        return await route_v3(query)
    elif ROUTER_VERSION == "v4":
        return await route_v4(query)
    elif ROUTER_VERSION == "v2":
        return route_v2(query)
    else:
        return route_v1(query)
