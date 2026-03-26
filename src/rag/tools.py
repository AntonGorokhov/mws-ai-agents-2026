"""RAG tool functions for agents."""

from src.rag.retriever import rag_query


def search_ml_knowledge(question: str, top_k: int = 5) -> str:
    """Search the ML knowledge base."""
    return rag_query(question, "ml_knowledge", top_k)


def search_error_solutions(error_msg: str, top_k: int = 3) -> str:
    """Search known error solutions."""
    return rag_query(error_msg, "error_solutions", top_k)


def search_experiment_log(query: str, top_k: int = 3) -> str:
    """Search past experiment results."""
    return rag_query(query, "experiment_log", top_k)
