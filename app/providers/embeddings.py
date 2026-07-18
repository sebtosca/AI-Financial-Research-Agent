import logging

from langchain_core.embeddings import Embeddings

from app.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GOOGLE_API_KEY,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    REQUIRED_EMBEDDING_ENV_VARS,
    REQUIRED_GOOGLE_ENV_VARS,
    validate_required_environment,
)

logger = logging.getLogger(__name__)


def _build_openai_embeddings(*, model: str, api_key: str | None, base_url: str | None) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key or OPENAI_API_KEY,
        base_url=base_url or OPENAI_API_BASE,
    )


def _build_google_embeddings(*, model: str, api_key: str | None, base_url: str | None) -> Embeddings:
    validate_required_environment(REQUIRED_GOOGLE_ENV_VARS)

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Google provider requested but langchain-google-genai is not "
            "installed. pip install -r requirements-optional.txt"
        ) from exc

    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key or GOOGLE_API_KEY)


def _build_anthropic_embeddings(*, model: str, api_key: str | None, base_url: str | None) -> Embeddings:
    raise NotImplementedError("Anthropic does not provide an embeddings API")


_EMBEDDING_BUILDERS = {
    "openai": _build_openai_embeddings,
    "anthropic": _build_anthropic_embeddings,
    "google": _build_google_embeddings,
}


def build_embedding_model(
    *,
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Embeddings:
    builder = _EMBEDDING_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown embedding provider: {provider!r}. "
            f"Supported providers: {sorted(_EMBEDDING_BUILDERS)}"
        )

    logger.info("Building embedding model | provider=%s | model=%s", provider, model)

    return builder(model=model, api_key=api_key, base_url=base_url)


def build_embedding_model_from_config() -> Embeddings:
    """Reproduce the original OpenAI-only embeddings construction unchanged."""

    validate_required_environment(REQUIRED_EMBEDDING_ENV_VARS)

    return build_embedding_model(
        provider=EMBEDDING_PROVIDER,
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
