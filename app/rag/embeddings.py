import logging
from langchain_core.embeddings import Embeddings

from app.config import (
    EMBEDDING_MODEL,
    LOG_LEVEL,
    REQUIRED_EMBEDDING_ENV_VARS,
    validate_required_environment,
)
from app.providers import build_embedding_model_from_config

logger = logging.getLogger(__name__)

def validate_environment() -> None:
    validate_required_environment(REQUIRED_EMBEDDING_ENV_VARS)

def build_embedding_model() -> Embeddings:
    validate_environment()

    logger.info(
        "Initializing embedding model | model=%s",
        EMBEDDING_MODEL,
    )

    embedding_model = build_embedding_model_from_config()

    logger.info(
        "Embedding model initialized | model=%s | use_case=%s",
        EMBEDDING_MODEL,
        "semantic_similarity_search",
    )

    return embedding_model

if __name__ == "__main__":
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    embeddings = build_embedding_model()
