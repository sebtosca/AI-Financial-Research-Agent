import logging
from pathlib import Path

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DOCS_PATH,
    LOG_LEVEL,
    RAG_REBUILD_INDEX,
)

from .embeddings import build_embedding_model
from .loader import extract_documents, load_documents
from .splitter import split_documents
from .vectorstore import build_vectorstore


logger = logging.getLogger(__name__)


def build_rag_index() -> None:
    """Build and persist the configured Chroma index from local PDF documents."""
    index_path = Path(CHROMA_DB_DIR).expanduser()
    if (index_path / "chroma.sqlite3").exists() and not RAG_REBUILD_INDEX:
        logger.info("RAG index already exists; skipping rebuild | directory=%s", index_path)
        return

    pdf_files = list(DOCS_PATH.rglob("*.pdf")) if DOCS_PATH.exists() else []

    if not pdf_files:
        extract_documents()

    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = build_embedding_model()
    build_vectorstore(
        documents=chunks,
        embedding_model=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    logger.info(
        "RAG index ready | documents=%d | chunks=%d | directory=%s",
        len(documents),
        len(chunks),
        CHROMA_DB_DIR,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    build_rag_index()
