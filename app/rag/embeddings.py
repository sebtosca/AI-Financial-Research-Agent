import logging
from langchain_openai import OpenAIEmbeddings

from app.config import (
    EMBEDDING_MODEL,
    LOG_LEVEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    REQUIRED_EMBEDDING_ENV_VARS,
    validate_required_environment,
)

logger = logging.getLogger(__name__)

def validate_environment() -> None:
    validate_required_environment(REQUIRED_EMBEDDING_ENV_VARS)

def build_embedding_model() -> OpenAIEmbeddings:
    validate_environment()


    logger.info(
        "Initializing embedding model | model=%s",
        EMBEDDING_MODEL,
    )

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )

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
