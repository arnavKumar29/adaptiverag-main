"""Unit tests for RRF fusion — no external services needed."""
import pytest
from api.pipeline.retrieval.dense import RetrievedChunk
from api.pipeline.retrieval.fusion import reciprocal_rank_fusion


def make_chunks(ids: list[str], retriever: str = "dense") -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=cid,
            document_id="doc1",
            content=f"Content of chunk {cid}",
            parent_id=None,
            source="test.pdf",
            score=1.0 / (i + 1),
            retriever=retriever,
        )
        for i, cid in enumerate(ids)
    ]


def test_rrf_deduplication():
    dense = make_chunks(["a", "b", "c", "d"], "dense")
    sparse = make_chunks(["c", "d", "e", "f"], "sparse")
    result = reciprocal_rank_fusion(dense, sparse, top_k=10)
    chunk_ids = [c.chunk_id for c in result]
    # No duplicates
    assert len(chunk_ids) == len(set(chunk_ids))


def test_rrf_overlap_chunks_score_higher():
    """Chunks appearing in both lists should score higher than single-list chunks."""
    dense = make_chunks(["overlap", "only_dense_1", "only_dense_2"], "dense")
    sparse = make_chunks(["overlap", "only_sparse_1", "only_sparse_2"], "sparse")
    result = reciprocal_rank_fusion(dense, sparse, top_k=10)
    scores = {c.chunk_id: c.score for c in result}
    # overlap should beat any single-list chunk
    assert scores["overlap"] > scores["only_dense_1"]
    assert scores["overlap"] > scores["only_sparse_1"]


def test_rrf_top_k_respected():
    dense = make_chunks([f"d{i}" for i in range(20)], "dense")
    sparse = make_chunks([f"s{i}" for i in range(20)], "sparse")
    result = reciprocal_rank_fusion(dense, sparse, top_k=5)
    assert len(result) <= 5


def test_rrf_single_list():
    """RRF with a single list should still work correctly."""
    chunks = make_chunks(["a", "b", "c"])
    result = reciprocal_rank_fusion(chunks, top_k=3)
    assert len(result) == 3
    # Should be sorted by RRF score (rank 1 → highest)
    assert result[0].chunk_id == "a"


def test_rrf_all_retriever_labels_set_to_hybrid():
    dense = make_chunks(["x"], "dense")
    sparse = make_chunks(["y"], "sparse")
    result = reciprocal_rank_fusion(dense, sparse)
    assert all(c.retriever == "hybrid" for c in result)
