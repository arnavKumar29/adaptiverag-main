# Adaptive RAG Engine 🔍

> Production-grade document intelligence with adaptive retrieval, self-evaluation, and full observability.

**Stack:** FastAPI · Weaviate · OpenSearch · PostgreSQL · Redis · FastHTML · LangGraph  
**Observability:** OpenTelemetry → Jaeger · Prometheus · Grafana  
**Deployment:** Docker Compose · Nginx · GitHub Actions · Hetzner VPS

---

## Architecture

```
Client → Nginx → FastAPI
                  ├── Semantic Cache (Redis + Weaviate)
                  ├── Query Router (rule-based → ML → self-learning)
                  ├── Retrieval (Weaviate dense + OpenSearch BM25 → RRF fusion)
                  ├── Reranker (BGE cross-encoder)
                  ├── Compressor (sentence relevance filter)
                  ├── Generator (Ollama qwen2.5:7b)
                  └── RAGAS Evaluator (async, background)
```

## Quick Start

```bash
# 1. Copy env file and configure secrets
cp .env.example .env
# Edit .env with your passwords

# 2. Start all services
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 3. Initialise Weaviate + OpenSearch schemas
python scripts/init_weaviate.py
python scripts/init_opensearch.py

# 4. Pull Ollama model
docker compose exec ollama ollama pull qwen2.5:7b

# 5. Get a JWT token
curl -X POST http://localhost:8000/api/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"<first 8 chars of JWT_SECRET>"}'

# 6. Ingest a document
curl -X POST http://localhost:8000/api/ingest \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/document.pdf"

# 7. Query
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main argument of the document?"}'
```

## Service Ports (dev)

| Service | Port | URL |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000/docs |
| BGE-M3 Embedder | 8001 | http://localhost:8001/info |
| BGE Reranker | 8002 | http://localhost:8002/health |
| Weaviate | 8080 | http://localhost:8080 |
| OpenSearch | 9200 | http://localhost:9200 |
| Ollama | 11434 | http://localhost:11434 |
| Jaeger UI | 16686 | http://localhost:16686 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 (admin/GRAFANA_PASSWORD) |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/query` | Execute RAG query |
| `POST` | `/api/ingest` | Upload document |
| `POST` | `/api/ingest/url` | Ingest URL |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Delete document |
| `GET` | `/api/health` | System health |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/api/token` | Get JWT (dev) |

## Query Routing

| Version | Method | When active |
|---|---|---|
| v1 | Rule-based heuristics | Default (always available) |
| v2 | LogisticRegression classifier | After 500+ query logs |
| v3 | LLM structured output | Set `ROUTER_VERSION=v3` |
| v4 | Thompson Sampling bandit | Set `ROUTER_VERSION=v4` |

## Running Tests

```bash
# Unit tests (no Docker needed)
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=api --cov-report=html

# Smoke tests (requires running stack)
API_BASE_URL=http://localhost:8000 pytest tests/smoke/ -v
```

## Project Structure

```
adaptiverag/
├── api/                   # FastAPI backend
│   ├── core/              # config, auth, telemetry
│   ├── pipeline/          # router, retrieval, reranker, compressor, generator
│   ├── ingestion/         # parser, chunker, embedder client, indexer
│   ├── eval/              # RAGAS evaluation, feedback loop
│   ├── cache/             # semantic cache
│   ├── agents/            # LangGraph agentic workflow (Phase 7)
│   ├── models/            # Pydantic schemas
│   ├── db/                # SQLAlchemy models + migrations
│   └── routers/           # FastAPI route handlers
├── embedder/              # BGE-M3 microservice
├── reranker/              # BGE reranker microservice
├── dashboard/             # FastHTML live dashboard (Phase 7)
├── scripts/               # Init scripts
├── tests/                 # unit / integration / smoke
├── infra/                 # nginx.conf, prometheus.yml, grafana/
└── .github/workflows/     # CI/CD
```
