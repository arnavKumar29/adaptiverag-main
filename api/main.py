"""
FastAPI application entry point.
Wires together all routers, middleware, telemetry, and lifespan events.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.core.config import get_settings
from api.core.telemetry import setup_telemetry
from api.db.postgres import create_tables
from api.routers import health, ingest, query

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Adaptive RAG Engine v1.0")
    setup_telemetry()

    # In dev, auto-create tables; in prod use Alembic migrations
    if settings.environment == "development":
        await create_tables()
        logger.info("Database tables ensured (dev mode)")

    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM: Ollama @ {settings.ollama_url} ({settings.ollama_model})")
    logger.info(f"Weaviate: {settings.weaviate_url}")
    logger.info(f"OpenSearch: {settings.opensearch_url}")
    yield
    logger.info("Shutting down Adaptive RAG Engine")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive RAG Engine",
    description=(
        "Production-grade document intelligence with adaptive retrieval, "
        "self-evaluation, and full observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)

# ── Prometheus metrics endpoint ───────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Token endpoint (dev convenience) ─────────────────────────────────────────
from api.core.auth import create_access_token
from api.models.schemas import TokenRequest, TokenResponse
from fastapi import HTTPException


@app.post("/api/token", response_model=TokenResponse, tags=["auth"])
async def get_token(req: TokenRequest) -> TokenResponse:
    """
    Dev convenience endpoint to get a JWT.
    In production, integrate with your identity provider.
    """
    # Simple hardcoded dev credentials — replace with real auth in prod
    if req.username != "admin" or req.password != settings.jwt_secret[:8]:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": req.username})
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_hours * 3600,
    )


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "Adaptive RAG Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "metrics": "/metrics",
    }
