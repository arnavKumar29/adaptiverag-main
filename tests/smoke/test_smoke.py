"""Post-deploy smoke tests — run against the live API."""
import os
import time

import httpx
import pytest

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
JWT_TOKEN = os.getenv("SMOKE_JWT_TOKEN", "")


@pytest.fixture
def headers():
    if JWT_TOKEN:
        return {"Authorization": f"Bearer {JWT_TOKEN}"}
    return {}


def test_health_endpoint(headers):
    resp = httpx.get(f"{BASE_URL}/api/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "services" in data


def test_root_endpoint():
    resp = httpx.get(f"{BASE_URL}/", timeout=5)
    assert resp.status_code == 200
    assert "Adaptive RAG Engine" in resp.json()["name"]


def test_metrics_endpoint():
    resp = httpx.get(f"{BASE_URL}/metrics", timeout=5)
    assert resp.status_code == 200
    assert b"rag_query_total" in resp.content


def test_query_endpoint_requires_auth():
    resp = httpx.post(
        f"{BASE_URL}/api/query",
        json={"query": "test query"},
        timeout=10,
    )
    assert resp.status_code == 401
