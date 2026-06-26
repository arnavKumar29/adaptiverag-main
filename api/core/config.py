from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    # ── Auth ─────────────────────────────────────────────────
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24

    # ── Databases ────────────────────────────────────────────
    database_url: str = "postgresql://rag:changeme_postgres@localhost:5432/ragdb"
    redis_url: str = "redis://:changeme_redis@localhost:6379"

    # ── Weaviate ─────────────────────────────────────────────
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = "changeme_weaviate"

    # ── OpenSearch ───────────────────────────────────────────
    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str = "admin"
    opensearch_password: str = "Changeme_opensearch1!"

    # ── Embedding microservice ────────────────────────────────
    embedder_url: str = "http://localhost:8001"
    embedding_model: str = "BAAI/bge-m3"
    embedding_model_version: str = "bge-m3-v1"

    # ── Reranker microservice ─────────────────────────────────
    reranker_url: str = "http://localhost:8002"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ── LLM (Ollama) ─────────────────────────────────────────
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_fallback_model: str = "llama3.1:8b"

    # ── Observability ─────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "adaptive-rag-engine"

    # ── Pipeline tuning ──────────────────────────────────────
    max_file_size_mb: int = 50
    rate_limit_per_minute: int = 100
    semantic_cache_threshold: float = 0.95
    retrieval_top_k: int = 20
    reranker_top_k: int = 5
    max_context_tokens: int = 3000
    ragas_faithfulness_target: float = 0.80

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
