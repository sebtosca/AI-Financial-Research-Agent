import logging

from langchain_community.vectorstores import Chroma

from app.config import RETRIEVER_SEARCH_TYPE, RETRIEVER_TOP_K


logger = logging.getLogger(__name__)


def build_retriever(
    vectorstore: Chroma,
    top_k: int = RETRIEVER_TOP_K,
):
    """
    Create a retriever from a vector store.

    Args:
        vectorstore: Initialized Chroma vector store.
        top_k: Number of chunks to retrieve.

    Returns:
        Configured retriever.
    """

    if vectorstore is None:
        raise ValueError(
            "Vector store is missing"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    logger.info(
        (
            "Creating retriever | "
            "search_type=%s | "
            "top_k=%d"
        ),
        RETRIEVER_SEARCH_TYPE,
        top_k,
    )

    retriever = vectorstore.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": top_k},
    )

    logger.info(
        (
            "Retriever created successfully | "
            "search_type=%s | "
            "top_k=%d"
        ),
        RETRIEVER_SEARCH_TYPE,
        top_k,
    )

    return retriever
