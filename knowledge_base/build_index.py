"""Build ChromaDB index from knowledge base source files."""

import glob
import pathlib
import sys

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.config import CHROMA_DB_DIR, KNOWLEDGE_SOURCES_DIR


def build_index():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Delete existing collection to rebuild
    try:
        client.delete_collection("ml_knowledge")
    except Exception:
        pass
    collection = client.create_collection("ml_knowledge")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". "],
    )

    pattern = str(KNOWLEDGE_SOURCES_DIR / "**" / "*.md")
    files = glob.glob(pattern, recursive=True)
    total_chunks = 0

    for filepath in files:
        text = pathlib.Path(filepath).read_text()
        topic = pathlib.Path(filepath).parent.name  # models / features / evaluation
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{pathlib.Path(filepath).stem}_{i}"
            embedding = encoder.encode([chunk]).tolist()
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=embedding,
                metadatas=[{"source": filepath, "topic": topic}],
            )
            total_chunks += 1

    print(f"Indexed {total_chunks} chunks from {len(files)} files into 'ml_knowledge'")

    # Create error_solutions collection (empty for now, populated at runtime)
    try:
        client.get_or_create_collection("error_solutions")
    except Exception:
        pass

    # Create experiment_log collection (populated at runtime)
    try:
        client.get_or_create_collection("experiment_log")
    except Exception:
        pass


if __name__ == "__main__":
    build_index()
