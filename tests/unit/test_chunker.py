"""Unit tests for chunking strategies — no external services needed."""
import pytest
from api.ingestion.chunker import (
    ChunkStrategy,
    chunk_document,
    chunk_fixed,
    chunk_parent_child,
    chunk_recursive,
    chunk_sentence,
    select_strategy,
)

SAMPLE_TEXT = """
The Adaptive RAG Engine is a production-grade document intelligence system.
It uses hybrid retrieval combining dense vector search with sparse BM25 keyword search.
The system evaluates every query using RAGAS metrics including faithfulness and relevancy.

Reciprocal Rank Fusion merges results from multiple retrievers robustly.
Cross-encoder reranking dramatically improves retrieval precision.
Context compression reduces token usage while preserving relevant information.

The query router adaptively selects the best retrieval strategy per query class.
Dense retrieval excels at semantic and conceptual queries.
Sparse retrieval excels at keyword-heavy and exact-match queries.
Hybrid retrieval handles mixed queries by combining both approaches.
""".strip()


def test_fixed_chunking_produces_chunks():
    chunks = chunk_fixed(SAMPLE_TEXT, "doc1", chunk_size=50, overlap=10)
    assert len(chunks) > 0
    assert all(c.content.strip() for c in chunks)
    assert all(c.document_id == "doc1" for c in chunks)


def test_sentence_chunking_groups_correctly():
    chunks = chunk_sentence(SAMPLE_TEXT, "doc1", sentences_per_chunk=2)
    assert len(chunks) > 0


def test_parent_child_creates_both():
    chunks = chunk_parent_child(SAMPLE_TEXT, "doc1", child_size=50, parent_size=200)
    parent_chunks = [c for c in chunks if c.parent_id is None]
    child_chunks = [c for c in chunks if c.parent_id is not None]
    assert len(parent_chunks) > 0
    assert len(child_chunks) > 0
    # Every child references a valid parent
    parent_ids = {p.chunk_id for p in parent_chunks}
    for child in child_chunks:
        assert child.parent_id in parent_ids


def test_recursive_chunking():
    chunks = chunk_recursive(SAMPLE_TEXT, "doc1", max_tokens=100)
    assert len(chunks) > 0
    assert all(len(c.content) > 10 for c in chunks)


def test_no_empty_chunks():
    for strategy in ChunkStrategy:
        chunks = chunk_document(SAMPLE_TEXT, "doc1", strategy=strategy)
        for c in chunks:
            assert c.content.strip(), f"Empty chunk found with strategy {strategy}"


@pytest.mark.parametrize("source_type,expected", [
    ("pdf", ChunkStrategy.PARENT_CHILD),
    ("docx", ChunkStrategy.PARENT_CHILD),
    ("md", ChunkStrategy.SENTENCE),
    ("url", ChunkStrategy.SENTENCE),
    ("txt", ChunkStrategy.RECURSIVE),
])
def test_strategy_selection(source_type, expected):
    assert select_strategy(source_type) == expected
