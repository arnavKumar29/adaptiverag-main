"""
All five chunking strategies from the design document.
The recommended primary strategy is parent-child.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"
    RECURSIVE = "recursive"


@dataclass
class Chunk:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    parent_id: Optional[str] = None   # set for child chunks
    chunk_index: int = 0
    strategy: str = ChunkStrategy.PARENT_CHILD
    token_count: int = 0


# ── Token counting (tiktoken) ─────────────────────────────────────────────────
def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: rough estimate
        return len(text.split())


# ── 1. Fixed-size chunking ────────────────────────────────────────────────────
def chunk_fixed(
    text: str,
    document_id: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    """Split by token count with overlap. Good for legal/contract docs."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
    except ImportError:
        tokens = text.split()

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            chunk_text = enc.decode(tokens[start:end])
        except (ImportError, AttributeError):
            chunk_text = " ".join(tokens[start:end])

        chunks.append(
            Chunk(
                document_id=document_id,
                content=chunk_text.strip(),
                chunk_index=idx,
                strategy=ChunkStrategy.FIXED,
                token_count=end - start,
            )
        )
        start += chunk_size - overlap
        idx += 1

    return chunks


# ── 2. Sentence-aware chunking ────────────────────────────────────────────────
def chunk_sentence(
    text: str,
    document_id: str,
    sentences_per_chunk: int = 4,
) -> list[Chunk]:
    """Group sentences into chunks. Good for articles/blogs."""
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
    except ImportError:
        # Fallback: split on ". "
        sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[Chunk] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i : i + sentences_per_chunk]
        content = " ".join(group).strip()
        if content:
            chunks.append(
                Chunk(
                    document_id=document_id,
                    content=content,
                    chunk_index=len(chunks),
                    strategy=ChunkStrategy.SENTENCE,
                    token_count=_count_tokens(content),
                )
            )
    return chunks


# ── 3. Semantic chunking ──────────────────────────────────────────────────────
def chunk_semantic(
    text: str,
    document_id: str,
    threshold: float = 0.85,
) -> list[Chunk]:
    """
    Split on topic shifts detected via sentence embedding cosine similarity.
    Good for research papers.
    """
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
    except ImportError:
        sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) <= 3:
        return chunk_sentence(text, document_id)

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight model for splitting
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Find split points where similarity drops
        split_indices = [0]
        for i in range(1, len(embeddings)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim < threshold:
                split_indices.append(i)
        split_indices.append(len(sentences))

        chunks: list[Chunk] = []
        for j in range(len(split_indices) - 1):
            group = sentences[split_indices[j] : split_indices[j + 1]]
            content = " ".join(group).strip()
            if content:
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        content=content,
                        chunk_index=len(chunks),
                        strategy=ChunkStrategy.SEMANTIC,
                        token_count=_count_tokens(content),
                    )
                )
        return chunks
    except ImportError:
        # Fallback if sentence-transformers not available
        return chunk_sentence(text, document_id)


# ── 4. Parent-child chunking (PRIMARY STRATEGY) ───────────────────────────────
def chunk_parent_child(
    text: str,
    document_id: str,
    child_size: int = 256,
    parent_size: int = 1024,
    overlap: int = 32,
) -> list[Chunk]:
    """
    Creates parent chunks (1024 tokens) and child chunks (256 tokens).
    Child chunks are indexed for precise retrieval.
    Parent chunks are returned to the LLM for richer context.
    Returns all chunks (both parent and child) — caller should index
    child chunks in vector store and store parent chunks for context retrieval.
    """
    # First create parent chunks
    parents = chunk_fixed(text, document_id, chunk_size=parent_size, overlap=overlap)

    all_chunks: list[Chunk] = []
    child_idx = 0

    for parent in parents:
        parent.strategy = ChunkStrategy.PARENT_CHILD
        all_chunks.append(parent)

        # Split parent into child chunks
        children = chunk_fixed(
            parent.content, document_id, chunk_size=child_size, overlap=16
        )
        for child in children:
            child.strategy = ChunkStrategy.PARENT_CHILD
            child.parent_id = parent.chunk_id
            child.chunk_index = child_idx
            child_idx += 1
            all_chunks.append(child)

    return all_chunks


# ── 5. Recursive chunking ─────────────────────────────────────────────────────
def chunk_recursive(
    text: str,
    document_id: str,
    max_tokens: int = 512,
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    """
    Hierarchical splitting: \\n\\n → \\n → '. ' → ' '
    Good for mixed-format docs.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: list[str]) -> list[str]:
        if not seps or _count_tokens(t) <= max_tokens:
            return [t] if t.strip() else []
        sep = seps[0]
        parts = t.split(sep)
        results: list[str] = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if _count_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    results.append(current)
                if _count_tokens(part) > max_tokens:
                    results.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            results.append(current)
        return results

    pieces = _split(text, separators)
    return [
        Chunk(
            document_id=document_id,
            content=p.strip(),
            chunk_index=i,
            strategy=ChunkStrategy.RECURSIVE,
            token_count=_count_tokens(p),
        )
        for i, p in enumerate(pieces)
        if p.strip()
    ]


# ── Strategy selector ─────────────────────────────────────────────────────────
def select_strategy(source_type: str, page_count: int = 1) -> ChunkStrategy:
    """Auto-select chunking strategy based on document type."""
    mapping = {
        "pdf": ChunkStrategy.PARENT_CHILD,   # primary
        "docx": ChunkStrategy.PARENT_CHILD,
        "txt": ChunkStrategy.RECURSIVE,
        "md": ChunkStrategy.SENTENCE,
        "url": ChunkStrategy.SENTENCE,
    }
    return mapping.get(source_type, ChunkStrategy.PARENT_CHILD)


# ── Main entry point ──────────────────────────────────────────────────────────
def chunk_document(
    text: str,
    document_id: str,
    source_type: str = "pdf",
    strategy: Optional[ChunkStrategy] = None,
) -> list[Chunk]:
    """Chunk text using the appropriate strategy."""
    if strategy is None:
        strategy = select_strategy(source_type)

    dispatch = {
        ChunkStrategy.FIXED: lambda: chunk_fixed(text, document_id),
        ChunkStrategy.SENTENCE: lambda: chunk_sentence(text, document_id),
        ChunkStrategy.SEMANTIC: lambda: chunk_semantic(text, document_id),
        ChunkStrategy.PARENT_CHILD: lambda: chunk_parent_child(text, document_id),
        ChunkStrategy.RECURSIVE: lambda: chunk_recursive(text, document_id),
    }

    fn = dispatch.get(strategy, dispatch[ChunkStrategy.PARENT_CHILD])
    chunks = fn()

    # Filter out empty or tiny chunks
    return [c for c in chunks if len(c.content.strip()) > 50]
