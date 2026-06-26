"""
SQLAlchemy models + async session factory.
Tables: documents, query_logs, ragas_scores, router_weights, api_keys
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from api.core.config import get_settings

settings = get_settings()

# ── Async engine ─────────────────────────────────────────────────────────────
# Convert postgresql:// → postgresql+asyncpg://
_db_url = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://", 1
)

engine = create_async_engine(
    _db_url,
    echo=settings.environment == "development",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Base ─────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Models ───────────────────────────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text, nullable=False)
    source_type = Column(
        String(10),
        CheckConstraint("source_type IN ('pdf','docx','txt','md','url')"),
    )
    embedding_model_version = Column(String(50), default="bge-m3-v1")
    status = Column(
        String(20),
        CheckConstraint("status IN ('pending','processing','indexed','failed')"),
        default="pending",
    )
    chunk_count = Column(Integer)
    error_message = Column(Text)
    collection = Column(String(100))
    tags = Column(Text)  # JSON array stored as text
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    query_logs = relationship("QueryLog", back_populates="document", lazy="noload")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    query_class = Column(String(50))  # conceptual / keyword / mixed
    strategy_used = Column(String(20))  # dense / sparse / hybrid
    answer = Column(Text)
    model_used = Column(String(100))
    latency_ms = Column(Integer)
    cache_hit = Column(Boolean, default=False)
    trace_id = Column(String(64))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="query_logs", lazy="noload")
    ragas_score = relationship(
        "RagasScore", back_populates="query_log", uselist=False, lazy="noload"
    )


class RagasScore(Base):
    __tablename__ = "ragas_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_log_id = Column(UUID(as_uuid=True), ForeignKey("query_logs.id"))
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    context_recall = Column(Float)
    context_precision = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    query_log = relationship("QueryLog", back_populates="ragas_score", lazy="noload")


class RouterWeight(Base):
    __tablename__ = "router_weights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_class = Column(String(50), nullable=False)
    strategy = Column(String(20), nullable=False)
    weight = Column(Float, default=0.33)
    alpha = Column(Float, default=1.0)   # Thompson sampling successes
    beta = Column(Float, default=1.0)    # Thompson sampling failures
    sample_count = Column(Integer, default=0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(64), unique=True, nullable=False)  # SHA256
    label = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ── Dependency ───────────────────────────────────────────────────────────────
async def get_db() -> AsyncSession:  # type: ignore[return]
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create all tables (used in dev). Use Alembic migrations in prod."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
