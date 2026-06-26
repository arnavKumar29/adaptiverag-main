"""
Agent tool definitions for the LangGraph agentic RAG workflow.
These wrap existing pipeline modules for use within the agent graph.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from api.core.config import get_settings
from api.pipeline.retrieval.dense import dense_retrieve
from api.pipeline.retrieval.sparse import sparse_retrieve
from api.pipeline.retrieval.fusion import hybrid_retrieve
from api.pipeline.generator import generate_with_fallback
from api.pipeline.compressor import compress_context, format_context_with_sources

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Search Documents ──────────────────────────────────────────────────────────

async def search_documents(
    query: str,
    strategy: str = "hybrid",
    top_k: int = 10,
    collection: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """
    Retrieve documents using the specified strategy.
    Returns list of chunk dicts.
    """
    try:
        if strategy == "dense":
            chunks = await dense_retrieve(
                query, top_k=top_k, collection=collection, tags=tags,
            )
        elif strategy == "sparse":
            chunks = await sparse_retrieve(
                query, top_k=top_k, collection=collection, tags=tags,
            )
        else:  # hybrid
            chunks = await hybrid_retrieve(
                query, top_k=top_k, collection=collection, tags=tags,
            )

        return [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "content": c.content,
                "parent_id": c.parent_id,
                "source": c.source,
                "score": c.score,
                "retriever": c.retriever,
            }
            for c in chunks
        ]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


# ── Evaluate Retrieval Quality ────────────────────────────────────────────────

async def evaluate_retrieval_quality(
    query: str,
    chunks: list[dict],
) -> float:
    """
    Evaluate retrieval quality by checking:
    1. Number of chunks retrieved
    2. Average relevance score
    3. Content coverage (basic heuristic)

    Returns a quality score between 0.0 and 1.0.
    """
    if not chunks:
        return 0.0

    # Factor 1: Number of chunks (0-1, with 5+ being ideal)
    count_score = min(len(chunks) / 5.0, 1.0)

    # Factor 2: Average relevance score
    scores = [c.get("score", 0.0) for c in chunks]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    # Normalize — scores vary by retriever, assume >0.3 is decent
    score_quality = min(avg_score / 0.5, 1.0) if avg_score > 0 else 0.0

    # Factor 3: Content diversity — check that chunks aren't too similar
    contents = [c.get("content", "")[:100] for c in chunks[:5]]
    unique_starts = len(set(contents))
    diversity = unique_starts / max(len(contents), 1)

    # Factor 4: Query term coverage
    query_terms = set(query.lower().split())
    all_content = " ".join(c.get("content", "") for c in chunks[:5]).lower()
    covered = sum(1 for t in query_terms if t in all_content)
    coverage = covered / max(len(query_terms), 1)

    # Weighted combination
    quality = (
        count_score * 0.2
        + score_quality * 0.3
        + diversity * 0.2
        + coverage * 0.3
    )

    logger.debug(
        f"Quality eval: count={count_score:.2f} score={score_quality:.2f} "
        f"diversity={diversity:.2f} coverage={coverage:.2f} → {quality:.2f}"
    )
    return quality


# ── Refine Query ──────────────────────────────────────────────────────────────

async def refine_query(
    original_query: str,
    current_query: str,
    chunks: list[dict],
) -> str:
    """
    Use the LLM to generate a refined query based on initial results.
    Falls back to keyword expansion if LLM is unavailable.
    """
    # Gather context from current chunks for the LLM
    context_snippets = "\n".join(
        f"- {c.get('content', '')[:200]}" for c in chunks[:3]
    )

    prompt = f"""Given the original question and some partially relevant search results,
generate an improved search query that will find more specific and relevant documents.

Original question: {original_query}
Current search query: {current_query}

Partial results found:
{context_snippets}

Generate ONLY the improved search query (one line, no explanation):"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 100},
                },
            )
            if resp.status_code == 200:
                refined = resp.json().get("response", "").strip()
                # Sanitize — keep only the first line, remove quotes
                refined = refined.split("\n")[0].strip().strip('"\'')
                if refined and len(refined) > 5:
                    return refined
    except Exception as e:
        logger.warning(f"LLM query refinement failed: {e}")

    # Fallback: expand query with terms from top chunks
    logger.debug("Using keyword expansion fallback for query refinement")
    expansion_terms = set()
    for chunk in chunks[:2]:
        words = chunk.get("content", "").split()[:10]
        expansion_terms.update(w.lower().strip(".,;:!?") for w in words if len(w) > 4)

    # Add a few expansion terms
    current_terms = set(current_query.lower().split())
    new_terms = [t for t in expansion_terms if t not in current_terms][:3]
    if new_terms:
        return f"{original_query} {' '.join(new_terms)}"

    return original_query


# ── Generate Answer ───────────────────────────────────────────────────────────

async def generate_answer(
    query: str,
    chunks: list[dict],
) -> dict:
    """
    Generate an answer from retrieved chunks using the LLM pipeline.
    Wraps the existing generator module.
    """
    if not chunks:
        return {
            "text": "I couldn't find relevant documents to answer your question.",
            "model": "none",
        }

    # Build context string from chunks
    context_parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        context_parts.append(f"[{i}] Source: {source}\n{content}")

    context = "\n\n---\n\n".join(context_parts)

    result = await generate_with_fallback(query, context)
    return result
