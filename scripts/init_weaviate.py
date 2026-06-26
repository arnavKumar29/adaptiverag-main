"""
Weaviate schema initialisation script.
Run once after `docker compose up -d`:
    python scripts/init_weaviate.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import weaviate
from weaviate.auth import AuthApiKey
from api.core.config import get_settings

settings = get_settings()


def get_client() -> weaviate.Client:
    return weaviate.Client(
        url=settings.weaviate_url,
        auth_client_secret=AuthApiKey(api_key=settings.weaviate_api_key),
    )


DOCUMENT_CHUNK_CLASS = {
    "class": "DocumentChunk",
    "description": "A chunk of text from an ingested document",
    "vectorizer": "none",  # We supply our own vectors
    "properties": [
        {"name": "chunk_id", "dataType": ["text"], "description": "UUID of this chunk"},
        {"name": "document_id", "dataType": ["text"], "description": "UUID of parent document"},
        {"name": "content", "dataType": ["text"], "description": "Chunk text"},
        {"name": "parent_id", "dataType": ["text"], "description": "UUID of parent chunk (parent-child strategy)"},
        {"name": "chunk_index", "dataType": ["int"], "description": "Position in document"},
        {"name": "embedding_model", "dataType": ["text"], "description": "e.g. bge-m3-v1"},
        {"name": "source", "dataType": ["text"], "description": "Filename or URL"},
        {"name": "collection", "dataType": ["text"], "description": "User-defined collection"},
        {"name": "tags", "dataType": ["text[]"], "description": "User-defined tags"},
        {"name": "chunk_strategy", "dataType": ["text"], "description": "Chunking strategy used"},
        {"name": "created_at", "dataType": ["date"]},
    ],
    "vectorIndexConfig": {
        "distance": "cosine",
        "ef": 128,
        "maxConnections": 64,
        "efConstruction": 128,
    },
    "invertedIndexConfig": {
        "bm25": {"b": 0.75, "k1": 1.2},
    },
}

CACHED_QUERY_CLASS = {
    "class": "CachedQuery",
    "description": "Semantic cache of past query embeddings",
    "vectorizer": "none",
    "properties": [
        {"name": "cache_key", "dataType": ["text"], "description": "Redis cache key"},
        {"name": "query_text", "dataType": ["text"], "description": "Original query"},
        {"name": "created_at", "dataType": ["date"]},
    ],
    "vectorIndexConfig": {
        "distance": "cosine",
        "ef": 64,
        "maxConnections": 32,
    },
}


def init_schema(client: weaviate.Client) -> None:
    existing = {c["class"] for c in client.schema.get().get("classes", [])}

    for cls in [DOCUMENT_CHUNK_CLASS, CACHED_QUERY_CLASS]:
        name = cls["class"]
        if name in existing:
            print(f"  [skip] {name} already exists")
        else:
            client.schema.create_class(cls)
            print(f"  [ok]   {name} created")


if __name__ == "__main__":
    print("Connecting to Weaviate at", settings.weaviate_url)
    client = get_client()
    meta = client.get_meta()
    print(f"  Weaviate version: {meta['version']}")
    init_schema(client)
    print("Done.")
