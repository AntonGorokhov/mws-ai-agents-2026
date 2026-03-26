"""RAG retriever: query ChromaDB collections."""

import logging

from src.rag.indexer import get_collection, _get_encoder

logger = logging.getLogger(__name__)


def rag_query(question: str, collection_name: str = "ml_knowledge", top_k: int = 5) -> str:
    """Query a ChromaDB collection and return concatenated results."""
    try:
        collection = get_collection(collection_name)
        encoder = _get_encoder()
        embedding = encoder.encode([question]).tolist()

        results = collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, collection.count()) if collection.count() > 0 else 1,
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        docs = results["documents"][0]
        sources = [m.get("source", "unknown") for m in results["metadatas"][0]] if results["metadatas"] else []

        context_parts = []
        for i, doc in enumerate(docs):
            source = sources[i] if i < len(sources) else "unknown"
            context_parts.append(f"[Source: {source}]\n{doc}")

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        logger.warning("RAG query failed: %s", e)
        return ""
