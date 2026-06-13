import logging
import time
from typing import Any, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
)

logger = logging.getLogger(__name__)


def build_vectorstore(
    documents: List[Document],
    embedding_model: Any,
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = CHROMA_DB_DIR,
) -> Chroma:
    """
    Build and persist a Chroma vector store.

    Args:
        documents: List of chunked documents.
        embedding_model: Embedding model instance.
        collection_name: Chroma collection name.
        persist_directory: Local persistence directory.

    Returns:
        Configured Chroma vector store.
    """

    start_time = time.perf_counter()

    if not documents:
        raise ValueError(
            "No documents provided to vector store"
        )

    if embedding_model is None:
        raise ValueError(
            "Embedding model is missing"
        )

    try:
        logger.info(
            (
                "Creating vector store | "
                "collection=%s | "
                "documents=%d | "
                "persist_directory=%s"
            ),
            collection_name,
            len(documents),
            persist_directory,
        )

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            (
                "Vector store created successfully | "
                "collection=%s | "
                "documents=%d | "
                "duration_ms=%.2f"
            ),
            collection_name,
            len(documents),
            duration_ms,
        )

        return vectorstore

    except Exception as e:
        logger.exception(
            (
                "Vector store creation failed | "
                "collection=%s"
            ),
            collection_name,
        )

        raise RuntimeError(
            f"Vector store creation failed: {e}"
        ) from e
