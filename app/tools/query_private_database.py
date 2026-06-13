import logging
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import ToolException, tool
from langchain_openai import ChatOpenAI

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    PRIVATE_DATABASE_ERROR_MESSAGE,
    PRIVATE_DATABASE_MAX_CONTEXT_CHARS,
    PRIVATE_DATABASE_MAX_QUERY_CHARS,
    PRIVATE_DATABASE_MAX_RETRIES,
    PRIVATE_DATABASE_MODEL,
    PRIVATE_DATABASE_NO_RESULTS_MESSAGE,
    PRIVATE_DATABASE_SYSTEM_PROMPT,
    PRIVATE_DATABASE_TEMPERATURE,
    PRIVATE_DATABASE_REQUEST_TIMEOUT,
    REQUIRED_PRIVATE_DATABASE_ENV_VARS,
    validate_required_environment,
)
from app.rag.embeddings import build_embedding_model
from app.rag.retriever import build_retriever


logger = logging.getLogger(__name__)


def _validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Private database query cannot be empty")

    cleaned_query = query.strip()

    if len(cleaned_query) > PRIVATE_DATABASE_MAX_QUERY_CHARS:
        raise ValueError(
            "Private database query exceeds the configured character limit"
        )

    return cleaned_query


@lru_cache(maxsize=1)
def _get_retriever():
    database_path = Path(CHROMA_DB_DIR).expanduser()

    if not database_path.exists():
        raise FileNotFoundError(
            f"Chroma database directory does not exist: {database_path}"
        )

    logger.info(
        "Loading private vector store | collection=%s | directory=%s",
        CHROMA_COLLECTION_NAME,
        database_path,
    )

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(database_path),
        embedding_function=build_embedding_model(),
    )

    return build_retriever(vectorstore)


@lru_cache(maxsize=1)
def _get_model() -> ChatOpenAI:
    validate_required_environment(REQUIRED_PRIVATE_DATABASE_ENV_VARS)

    logger.info(
        "Initializing private database model | model=%s | temperature=%.2f",
        PRIVATE_DATABASE_MODEL,
        PRIVATE_DATABASE_TEMPERATURE,
    )

    return ChatOpenAI(
        model=PRIVATE_DATABASE_MODEL,
        temperature=PRIVATE_DATABASE_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        timeout=PRIVATE_DATABASE_REQUEST_TIMEOUT,
        max_retries=PRIVATE_DATABASE_MAX_RETRIES,
    )


def _document_source(document: Document, index: int) -> str:
    metadata = document.metadata or {}
    return str(
        metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("company")
        or f"document-{index}"
    )


def _format_context(documents: list[Document]) -> str:
    sections = []

    for index, document in enumerate(documents, start=1):
        source = _document_source(document, index)
        sections.append(f"[Source: {source}]\n{document.page_content.strip()}")

    context = "\n\n".join(sections)
    return context[:PRIVATE_DATABASE_MAX_CONTEXT_CHARS]


@tool
def query_private_database(query: str) -> str:
    """Query private analyst reports about company AI initiatives and plans."""
    try:
        cleaned_query = _validate_query(query)

        logger.info(
            "Querying private database | query_chars=%d",
            len(cleaned_query),
        )

        documents = _get_retriever().invoke(cleaned_query)

        if not documents:
            logger.warning("Private database returned no documents")
            return PRIVATE_DATABASE_NO_RESULTS_MESSAGE

        logger.info(
            "Private database retrieval completed | document_count=%d",
            len(documents),
        )

        messages = [
            SystemMessage(content=PRIVATE_DATABASE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Retrieved context:\n{_format_context(documents)}\n\n"
                    f"Question:\n{cleaned_query}"
                )
            ),
        ]
        response = _get_model().invoke(messages)
        answer = response.content

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Private database model returned an empty response")

        logger.info(
            "Private database query completed | answer_chars=%d",
            len(answer.strip()),
        )

        return answer.strip()

    except ToolException:
        raise
    except ValueError as exc:
        logger.warning("Invalid private database query | error=%s", exc)
        raise ToolException(str(exc)) from exc
    except Exception as exc:
        logger.exception("Private database query failed")
        raise ToolException(PRIVATE_DATABASE_ERROR_MESSAGE) from exc
