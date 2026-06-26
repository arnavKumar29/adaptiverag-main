"""
POST /api/ingest, POST /api/ingest/url — document ingestion endpoints.
Runs the 7-stage pipeline: validate → parse → clean → metadata → chunk → embed → index.
Design doc Section 3.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.auth import require_auth
from api.core.config import get_settings
from api.core.telemetry import rag_ingestion_total
from api.db.postgres import Document, get_db
from api.ingestion.chunker import chunk_document
from api.ingestion.embedder import get_embedder
from api.ingestion.indexer import index_document
from api.ingestion.parser import parse_document, parse_url
from api.models.schemas import DocumentStatus, IngestResponse, IngestURLRequest

router = APIRouter(prefix="/api", tags=["ingestion"])
settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
MAX_SIZE_BYTES = settings.max_file_size_mb * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: Optional[str] = Form(default=None),
    tags: Optional[str] = Form(default=None),  # comma-separated
    db: AsyncSession = Depends(get_db),
    _user: str = Depends(require_auth),
) -> IngestResponse:
    """Upload and ingest a document (PDF, DOCX, TXT, MD)."""

    # ── Stage 1: Validation ───────────────────────────────────────────────────
    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_file_size_mb}MB.",
        )

    # Create document record (status=pending)
    doc_id = uuid.uuid4()
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    doc = Document(
        id=doc_id,
        filename=file.filename,
        source_type=ext,
        status="pending",
        collection=collection,
        tags=str(tag_list) if tag_list else None,
    )
    db.add(doc)
    await db.commit()

    # Dispatch background pipeline
    background_tasks.add_task(
        _run_ingestion_pipeline,
        doc_id=str(doc_id),
        content=content,
        filename=file.filename,
        source_type=ext,
        collection=collection,
        tags=tag_list,
    )

    rag_ingestion_total.labels(source_type=ext, status="pending").inc()

    return IngestResponse(
        document_id=str(doc_id),
        filename=file.filename,
        status="pending",
        message="Document queued for ingestion. Check GET /api/documents for status.",
    )


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(
    req: IngestURLRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _user: str = Depends(require_auth),
) -> IngestResponse:
    """Fetch and ingest a URL."""
    doc_id = uuid.uuid4()
    filename = req.url.split("/")[-1][:100] or "webpage"

    doc = Document(
        id=doc_id,
        filename=filename,
        source_type="url",
        status="pending",
        collection=req.collection,
        tags=str(req.tags) if req.tags else None,
    )
    db.add(doc)
    await db.commit()

    background_tasks.add_task(
        _run_url_ingestion,
        doc_id=str(doc_id),
        url=req.url,
        depth=req.depth,
        collection=req.collection,
        tags=req.tags or [],
    )

    return IngestResponse(
        document_id=str(doc_id),
        filename=filename,
        status="pending",
        message="URL queued for ingestion.",
    )


@router.get("/documents", response_model=list[DocumentStatus])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    _user: str = Depends(require_auth),
) -> list[DocumentStatus]:
    """List all ingested documents."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [
        DocumentStatus(
            id=d.id,
            filename=d.filename,
            source_type=d.source_type or "unknown",
            status=d.status,
            chunk_count=d.chunk_count,
            embedding_model_version=d.embedding_model_version or "bge-m3-v1",
            collection=d.collection,
            created_at=str(d.created_at),
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    _user: str = Depends(require_auth),
) -> dict:
    """Delete a document and its chunks from all stores."""
    result = await db.execute(
        select(Document).where(Document.id == uuid.UUID(doc_id))
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from Weaviate + OpenSearch in background
    background = BackgroundTasks()
    background.add_task(_delete_document_chunks, doc_id)

    await db.delete(doc)
    await db.commit()

    return {"message": f"Document {doc_id} deleted", "status": "ok"}


# ── Background pipeline workers ───────────────────────────────────────────────

async def _run_ingestion_pipeline(
    doc_id: str,
    content: bytes,
    filename: str,
    source_type: str,
    collection: Optional[str],
    tags: list[str],
) -> None:
    from api.db.postgres import AsyncSessionLocal
    from sqlalchemy import update

    async with AsyncSessionLocal() as db:
        try:
            # Update status → processing
            await db.execute(
                update(Document)
                .where(Document.id == uuid.UUID(doc_id))
                .values(status="processing")
            )
            await db.commit()

            # Stage 3-5: Parse + clean + metadata
            parsed = parse_document(content, filename, source_type)

            # Stage 6: Chunk
            chunks = chunk_document(parsed.text, doc_id, source_type=source_type)
            logger.info(f"Document {doc_id}: {len(chunks)} chunks created")

            if not chunks:
                raise ValueError("No chunks produced — document may be empty or too short")

            # Stage 7: Embed (only child chunks for parent-child strategy)
            embedder = get_embedder()
            texts = [c.content for c in chunks]

            # Batch in groups of 32
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), 32):
                batch_embs = await embedder.embed(texts[i : i + 32])
                all_embeddings.extend(batch_embs)

            # Stage 7b: Index into Weaviate + OpenSearch
            index_result = await index_document(
                chunks=chunks,
                embeddings=all_embeddings,
                source=filename,
                collection=collection,
                tags=tags,
            )
            logger.info(f"Document {doc_id} indexed: {index_result}")

            # Update status → indexed
            await db.execute(
                update(Document)
                .where(Document.id == uuid.UUID(doc_id))
                .values(
                    status="indexed",
                    chunk_count=len(chunks),
                    embedding_model_version=settings.embedding_model_version,
                )
            )
            await db.commit()
            rag_ingestion_total.labels(source_type=source_type, status="indexed").inc()

        except Exception as e:
            logger.error(f"Ingestion failed for {doc_id}: {e}", exc_info=True)
            await db.execute(
                update(Document)
                .where(Document.id == uuid.UUID(doc_id))
                .values(status="failed", error_message=str(e)[:500])
            )
            await db.commit()
            rag_ingestion_total.labels(source_type=source_type, status="failed").inc()


async def _run_url_ingestion(
    doc_id: str,
    url: str,
    depth: int,
    collection: Optional[str],
    tags: list[str],
) -> None:
    try:
        parsed = parse_url(url, depth=depth)
        content = parsed.text.encode("utf-8")
        await _run_ingestion_pipeline(
            doc_id=doc_id,
            content=content,
            filename=url,
            source_type="url",
            collection=collection,
            tags=tags,
        )
    except Exception as e:
        logger.error(f"URL ingestion failed for {url}: {e}")


async def _delete_document_chunks(doc_id: str) -> None:
    """Remove document chunks from Weaviate and OpenSearch."""
    try:
        import weaviate
        from weaviate.auth import AuthApiKey
        from opensearchpy import AsyncOpenSearch

        wv = weaviate.Client(
            url=settings.weaviate_url,
            auth_client_secret=AuthApiKey(api_key=settings.weaviate_api_key),
        )
        wv.batch.delete_objects(
            class_name="DocumentChunk",
            where={"path": ["document_id"], "operator": "Equal", "valueText": doc_id},
        )

        os_client = AsyncOpenSearch(
            hosts=[settings.opensearch_url],
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=False,
        )
        await os_client.delete_by_query(
            index="document_chunks",
            body={"query": {"term": {"document_id": doc_id}}},
        )
        await os_client.close()
        logger.info(f"Deleted chunks for document {doc_id}")
    except Exception as e:
        logger.error(f"Chunk deletion failed for {doc_id}: {e}")
