"""Unit tests for the query router — no external services needed."""
import pytest
from api.pipeline.router import (
    QueryClass,
    Strategy,
    classify_query,
    route_v1,
)


@pytest.mark.parametrize("query,expected_class", [
    ("what is the syntax for a Python class?", QueryClass.KEYWORD),
    ("how does transformer attention work?", QueryClass.CONCEPTUAL),
    ("explain gradient descent", QueryClass.CONCEPTUAL),
    ("IndexError on line 42", QueryClass.KEYWORD),
    ("compare BERT and GPT architectures", QueryClass.CONCEPTUAL),
    ("what happened in the meeting", QueryClass.MIXED),
    ('"exact phrase" in the document', QueryClass.KEYWORD),
])
def test_classify_query(query, expected_class):
    result = classify_query(query)
    assert result == expected_class, f"Query '{query}' expected {expected_class}, got {result}"


@pytest.mark.parametrize("query,expected_strategy", [
    ("how does RAG work?", Strategy.DENSE),
    ("Python ImportError fix", Strategy.SPARSE),
    ("tell me about the project timeline", QueryClass.MIXED),  # hybrid
])
def test_route_v1(query, expected_strategy):
    if expected_strategy == QueryClass.MIXED:
        strategy, _ = route_v1(query)
        assert strategy == Strategy.HYBRID
    else:
        strategy, _ = route_v1(query)
        assert strategy == expected_strategy
