"""
Adaptive RAG Engine — Admin Dashboard.
Built with FastAPI (pure Python HTML) + HTMX for live updates.
TailAdmin-inspired dark design.
"""
from __future__ import annotations

import logging
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from components.layout import page_shell, nav_link
from components.stats_cards import render_stats_cards
from components.query_log_table import render_query_log_table
from components.documents_table import render_documents_table
from components.eval_chart import render_eval_section
from components.chat_ui import render_chat_ui, render_chat_message

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")

app = FastAPI(title="RAG Dashboard", docs_url=None, redoc_url=None)

# ── Helpers ───────────────────────────────────────────────────────────────────

_token_cache: dict[str, str] = {}


async def _get_token() -> str:
    """Get a JWT token for API calls."""
    if "token" in _token_cache:
        return _token_cache["token"]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{API_URL}/api/token",
                json={"username": "admin", "password": JWT_SECRET[:8]},
            )
            if resp.status_code == 200:
                token = resp.json()["access_token"]
                _token_cache["token"] = token
                return token
    except Exception as e:
        logger.warning(f"Failed to get API token: {e}")
    return ""


async def _api_get(path: str) -> dict | list | None:
    """Make an authenticated GET request to the API."""
    token = await _get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_URL}{path}", headers=headers)
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.warning(f"API call failed ({path}): {e}")
    return None


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard overview page."""
    health = await _api_get("/api/health")
    status_ok = health and health.get("status") == "ok"

    status_badge = (
        '<span class="badge badge-ok">System Online</span>'
        if status_ok
        else '<span class="badge badge-warn">Degraded</span>'
    )

    content = f"""
    <div class="page-header">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1>Overview</h1>
                <p>Real-time monitoring for the Adaptive RAG Engine</p>
            </div>
            {status_badge}
        </div>
    </div>

    <div id="stats-cards" hx-get="/partials/stats" hx-trigger="load, every 30s" hx-swap="innerHTML">
        <div class="loading"><div class="spinner"></div><span>Loading metrics…</span></div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Recent Queries</span>
            <a href="/queries" style="font-size: 13px; color: var(--color-blue); text-decoration: none;">View all →</a>
        </div>
        <div id="query-log" hx-get="/partials/queries?limit=5" hx-trigger="load, every 30s" hx-swap="innerHTML">
            <div class="loading"><div class="spinner"></div><span>Loading queries…</span></div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Quality Metrics</span>
            <a href="/eval" style="font-size: 13px; color: var(--color-blue); text-decoration: none;">View all →</a>
        </div>
        <div id="eval-section" hx-get="/partials/eval" hx-trigger="load, every 60s" hx-swap="innerHTML">
            <div class="loading"><div class="spinner"></div><span>Loading evaluation data…</span></div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Service Health</span>
        </div>
        <div id="health-detail" hx-get="/partials/health" hx-trigger="load, every 15s" hx-swap="innerHTML">
            <div class="loading"><div class="spinner"></div><span>Checking services…</span></div>
        </div>
    </div>
    """
    return page_shell("Overview", content, active="overview")


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    """Documents management page."""
    content = """
    <div class="page-header">
        <h1>Documents</h1>
        <p>Manage ingested documents and their indexing status</p>
    </div>

    <div id="documents-table" hx-get="/partials/documents" hx-trigger="load, every 30s" hx-swap="innerHTML">
        <div class="loading"></div>
    </div>
    """
    return page_shell("Documents", content, active="documents")


@app.get("/queries", response_class=HTMLResponse)
async def queries_page(request: Request):
    """Full query log page."""
    content = """
    <div class="page-header">
        <h1>Query Log</h1>
        <p>Complete query execution history with strategy and performance data</p>
    </div>

    <div id="query-log-full" hx-get="/partials/queries?limit=50" hx-trigger="load, every 30s" hx-swap="innerHTML">
        <div class="loading"></div>
    </div>
    """
    return page_shell("Query Log", content, active="queries")


@app.get("/eval", response_class=HTMLResponse)
async def eval_page(request: Request):
    """Evaluation scores page."""
    content = """
    <div class="page-header">
        <h1>Evaluation</h1>
        <p>RAGAS quality scores, feedback loop status, and golden dataset benchmarks</p>
    </div>

    <div id="eval-full" hx-get="/partials/eval" hx-trigger="load, every 60s" hx-swap="innerHTML">
        <div class="loading"></div>
    </div>
    """
    return page_shell("Evaluation", content, active="eval")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat UI page."""
    return page_shell("Chat AI", render_chat_ui(), active="chat")


@app.post("/chat/send", response_class=HTMLResponse)
async def chat_send(request: Request):
    """Proxy for sending chat messages from HTMX form."""
    form = await request.form()
    query = form.get("query", "").strip()
    
    if not query:
        return ""
        
    token = await _get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{API_URL}/api/query",
                headers=headers,
                json={"query": query, "top_k": 5}
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No answer generated.")
                sources = data.get("sources", [])
                latency = data.get("latency_ms", 0)
                return render_chat_message("user", query) + render_chat_message("ai", answer, sources, latency)
            else:
                return render_chat_message("user", query) + render_chat_message("ai", f"Error: {resp.text}")
    except Exception as e:
        return render_chat_message("user", query) + render_chat_message("ai", f"Connection error: {e}")


@app.post("/ingest", response_class=HTMLResponse)
async def ingest_document(request: Request):
    """Proxy for uploading documents to the API."""
    form = await request.form()
    file_obj = form.get("file")
    collection = form.get("collection", "")
    
    if not file_obj or not hasattr(file_obj, "filename"):
        return "<div class='error-box'>[ ERROR ] No valid file provided.</div>"
        
    token = await _get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    content = await file_obj.read()
    files = {"file": (file_obj.filename, content, file_obj.content_type)}
    data = {"collection": collection} if collection else {}
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{API_URL}/api/ingest",
                headers=headers,
                files=files,
                data=data
            )
            if resp.status_code == 200:
                docs = await _api_get("/api/documents")
                return render_documents_table(docs if isinstance(docs, list) else [])
            else:
                return f"<div class='error-box'>[ ERROR ] API rejected upload: {resp.text}</div>"
    except Exception as e:
        return f"<div class='error-box'>[ ERROR ] Connection error: {e}</div>"


# ── HTMX Partials ─────────────────────────────────────────────────────────────

@app.get("/partials/stats", response_class=HTMLResponse)
async def partial_stats():
    """Stats cards partial — loaded via HTMX."""
    health = await _api_get("/api/health")
    return render_stats_cards(health)


@app.get("/partials/queries", response_class=HTMLResponse)
async def partial_queries(limit: int = 10):
    """Query log table partial."""
    logs = await _api_get(f"/api/queries?limit={limit}")
    return render_query_log_table(logs if isinstance(logs, list) else [], limit=limit)


@app.get("/partials/documents", response_class=HTMLResponse)
async def partial_documents():
    """Documents table partial."""
    docs = await _api_get("/api/documents")
    return render_documents_table(docs if isinstance(docs, list) else [])


@app.get("/partials/eval", response_class=HTMLResponse)
async def partial_eval():
    """Evaluation section partial."""
    return render_eval_section()


@app.get("/partials/health", response_class=HTMLResponse)
async def partial_health():
    """Health detail partial."""
    health = await _api_get("/api/health")
    if not health:
        return """
        <div class="error-box">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            Unable to reach the API — make sure the backend is running.
        </div>"""

    services = health.get("services", [])
    rows = ""
    for svc in services:
        is_ok = svc["status"] == "ok"
        dot_cls = "status-ok" if is_ok else "status-error"
        badge_cls = "badge-ok" if is_ok else "badge-error"
        latency = f'{svc.get("latency_ms", "—")} ms' if svc.get("latency_ms") else "—"
        detail = svc.get("detail", "—") or "—"

        rows += f"""
        <tr>
            <td style="color: var(--color-text); font-weight: 500;">
                <span class="status-dot {dot_cls}"></span>{svc['name']}
            </td>
            <td><span class="badge {badge_cls}">{svc['status']}</span></td>
            <td style="font-weight: 600; color: {'var(--color-green)' if is_ok else 'var(--color-red)'};">{latency}</td>
            <td class="detail-cell" style="font-size: 12px; color: var(--color-muted);" title="{detail}">{detail}</td>
        </tr>
        """

    total = len(services)
    ok = sum(1 for s in services if s.get("status") == "ok")

    return f"""
    <div class="card">
        <div class="card-header">
            <span class="card-title">Service Health</span>
            <span class="badge {'badge-ok' if ok == total else 'badge-warn'}">{ok}/{total} healthy</span>
        </div>
        <div style="overflow-x: auto;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Service</th><th>Status</th><th>Latency</th><th>Detail</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>
    """


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")
