"""ChromaDB indexer for runtime experiment logging."""

import logging

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DB_DIR, MODELS

logger = logging.getLogger(__name__)

_encoder = None
_client = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(MODELS["embedding"])
    return _encoder


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _client


def get_collection(name: str):
    return _get_client().get_or_create_collection(name)


def add_document(collection_name: str, doc_id: str, text: str, metadata: dict | None = None):
    """Add a single document to a collection."""
    collection = get_collection(collection_name)
    encoder = _get_encoder()
    embedding = encoder.encode([text]).tolist()
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=embedding,
        metadatas=[metadata or {}],
    )
    logger.debug("Added doc %s to collection %s", doc_id, collection_name)
