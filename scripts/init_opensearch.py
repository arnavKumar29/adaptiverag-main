"""
OpenSearch index initialisation script.
Run once after `docker compose up -d`:
    python scripts/init_opensearch.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from opensearchpy import OpenSearch
from api.core.config import get_settings

settings = get_settings()


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=False,
        verify_certs=False,
        timeout=30,
    )


INDEX_NAME = "document_chunks"

INDEX_BODY = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "english_custom": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stop", "english_stemmer"],
                }
            },
            "filter": {
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
            },
        },
        "similarity": {
            "bm25_custom": {
                "type": "BM25",
                "k1": 1.2,
                "b": 0.75,
            }
        },
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "english_custom",
                "similarity": "bm25_custom",
            },
            "source": {"type": "keyword"},
            "collection": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_strategy": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    },
}


def init_index(client: OpenSearch) -> None:
    if client.indices.exists(index=INDEX_NAME):
        print(f"  [skip] Index '{INDEX_NAME}' already exists")
    else:
        resp = client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
        if resp.get("acknowledged"):
            print(f"  [ok]   Index '{INDEX_NAME}' created")
        else:
            print(f"  [err]  Unexpected response: {resp}")


if __name__ == "__main__":
    print("Connecting to OpenSearch at", settings.opensearch_url)
    client = get_client()
    info = client.info()
    print(f"  OpenSearch version: {info['version']['number']}")
    init_index(client)
    print("Done.")
