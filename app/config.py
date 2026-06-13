import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Got: {value}") from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float. Got: {value}") from exc


def _get_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def validate_required_environment(required_vars: Iterable[str]) -> None:
    missing = [name for name in required_vars if not os.getenv(name)]

    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(sorted(missing))
        )


# Application
APP_NAME: str = os.getenv("APP_NAME", "Financial Research Agent")
APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT: int = _get_int("APP_PORT", 8000)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
DEBUG: bool = _get_bool("DEBUG", False)

# OpenAI
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE: str | None = os.getenv("OPENAI_API_BASE") or None
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE: float = _get_float("OPENAI_TEMPERATURE", 0.0)
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# LangSmith
LANGCHAIN_TRACING_V2: bool = _get_bool("LANGCHAIN_TRACING_V2", False)
LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT: str = os.getenv(
    "LANGCHAIN_PROJECT",
    "financial-research-agent",
)

# Tavily
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS: int = _get_int("TAVILY_MAX_RESULTS", 5)
TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
TAVILY_INCLUDE_ANSWER: bool = _get_bool("TAVILY_INCLUDE_ANSWER", True)
TAVILY_INCLUDE_RAW_CONTENT: bool = _get_bool(
    "TAVILY_INCLUDE_RAW_CONTENT",
    False,
)
TAVILY_INCLUDE_IMAGES: bool = _get_bool("TAVILY_INCLUDE_IMAGES", False)

# RAG document loading
DOCS_PATH: Path = _get_path("DOCS_PATH", PROJECT_ROOT / "app" / "docs")
ZIP_FILE: Path = _get_path(
    "ZIP_FILE",
    DOCS_PATH / "Companies-AI-Initiatives.zip",
)

# RAG chunking
TEXT_SPLITTER_ENCODING: str = os.getenv(
    "TEXT_SPLITTER_ENCODING",
    "cl100k_base",
)
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 200)

# Vector store and retrieval
CHROMA_COLLECTION_NAME: str = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "ai_initiatives",
)
CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")
RETRIEVER_TOP_K: int = _get_int("RETRIEVER_TOP_K", 10)
RETRIEVER_SEARCH_TYPE: str = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity")

# Private database tool
PRIVATE_DATABASE_MODEL: str = os.getenv(
    "PRIVATE_DATABASE_MODEL",
    OPENAI_MODEL,
)
PRIVATE_DATABASE_TEMPERATURE: float = _get_float(
    "PRIVATE_DATABASE_TEMPERATURE",
    0.0,
)
PRIVATE_DATABASE_MAX_QUERY_CHARS: int = _get_int(
    "PRIVATE_DATABASE_MAX_QUERY_CHARS",
    2000,
)
PRIVATE_DATABASE_MAX_CONTEXT_CHARS: int = _get_int(
    "PRIVATE_DATABASE_MAX_CONTEXT_CHARS",
    20000,
)
PRIVATE_DATABASE_REQUEST_TIMEOUT: float = _get_float(
    "PRIVATE_DATABASE_REQUEST_TIMEOUT",
    30.0,
)
PRIVATE_DATABASE_MAX_RETRIES: int = _get_int(
    "PRIVATE_DATABASE_MAX_RETRIES",
    2,
)
PRIVATE_DATABASE_NO_RESULTS_MESSAGE: str = os.getenv(
    "PRIVATE_DATABASE_NO_RESULTS_MESSAGE",
    "I don't know - this information is not available in our analyst reports.",
)
PRIVATE_DATABASE_ERROR_MESSAGE: str = os.getenv(
    "PRIVATE_DATABASE_ERROR_MESSAGE",
    "The private analyst database is temporarily unavailable.",
)
PRIVATE_DATABASE_SYSTEM_PROMPT: str = os.getenv(
    "PRIVATE_DATABASE_SYSTEM_PROMPT",
    (
        "You answer questions about company AI initiatives using only the "
        "retrieved analyst-report context. Cite the source for every material "
        "claim. Do not use outside knowledge. If the context does not contain "
        "the answer, reply exactly with: "
        f"{PRIVATE_DATABASE_NO_RESULTS_MESSAGE}"
    ),
)

# Tool defaults
STOCK_HISTORY_DEFAULT_PERIOD: str = os.getenv(
    "STOCK_HISTORY_DEFAULT_PERIOD",
    "1y",
)

REQUIRED_AGENT_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY",)
REQUIRED_EMBEDDING_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY",)
REQUIRED_SENTIMENT_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY",)
REQUIRED_TAVILY_ENV_VARS: tuple[str, ...] = ("TAVILY_API_KEY",)
REQUIRED_PRIVATE_DATABASE_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY",)
