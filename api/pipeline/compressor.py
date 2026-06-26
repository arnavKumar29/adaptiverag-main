"""
Context compression via sentence relevance filtering.
Strips irrelevant sentences to reduce token count before LLM generation.
~30ms overhead, ~40% token reduction (design doc Section 9).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from api.core.config import get_settings
from api.ingestion.embedder import get_embedder
from api.pipeline.retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        return re.split(r"(?<=[.!?])\s+", text)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def compress_context(
    query: str,
    chunks: list[RetrievedChunk],
    max_tokens: int = 3000,
    min_sentence_score: float = 0.3,
) -> str:
    """
    Sentence relevance filter (Section 9.2):
    1. Embed query and all sentences from all chunks
    2. Score sentences by cosine similarity to query
    3. Select highest-scoring sentences within token budget
    4. Return compressed context string

    Falls back to simple truncation if embedding fails.
    """
    if not chunks:
        return ""

    # Flatten all sentences from all chunks
    sentences: list[str] = []
    for chunk in chunks:
        sents = _split_sentences(chunk.content)
        sentences.extend([s.strip() for s in sents if s.strip()])

    if not sentences:
        return ""

    # Try embedding-based compression
    try:
        embedder = get_embedder()
        all_texts = [query] + sentences
        embeddings = await embedder.embed(all_texts)
        query_emb = embeddings[0]
        sent_embs = embeddings[1:]

        # Score and rank
        scored = [
            (sent, _cosine_similarity(query_emb, emb))
            for sent, emb in zip(sentences, sent_embs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select within token budget
        selected: list[str] = []
        token_count = 0

        for sent, score in scored:
            if score < min_sentence_score:
                break
            tokens = _count_tokens(sent)
            if token_count + tokens > max_tokens:
                break
            selected.append(sent)
            token_count += tokens

        if selected:
            logger.debug(
                f"Context compressed: {len(sentences)} sentences → {len(selected)}, "
                f"~{token_count} tokens"
            )
            return " ".join(selected)

    except Exception as e:
        logger.warning(f"Embedding-based compression failed, falling back: {e}")

    # Fallback: simple truncation by joining chunks until token budget
    result_parts: list[str] = []
    token_count = 0
    for chunk in chunks:
        tokens = _count_tokens(chunk.content)
        if token_count + tokens > max_tokens:
            break
        result_parts.append(chunk.content)
        token_count += tokens

    return "\n\n".join(result_parts)


def format_context_with_sources(
    chunks: list[RetrievedChunk],
    compressed_text: Optional[str] = None,
) -> str:
    """Format context with source citations for the LLM prompt."""
    if compressed_text:
        # Build source list from chunks
        sources = list({chunk.source for chunk in chunks if chunk.source})
        source_str = ", ".join(f"[{s}]" for s in sources[:5])
        return f"{compressed_text}\n\nSources: {source_str}"

    # Fall back to formatting each chunk with its source
    parts: list[str] = []
    for chunk in chunks:
        source_label = f"[Source: {chunk.source}]" if chunk.source else ""
        parts.append(f"{chunk.content} {source_label}".strip())
    return "\n\n---\n\n".join(parts)
