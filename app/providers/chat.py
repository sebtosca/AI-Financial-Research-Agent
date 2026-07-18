import logging

from langchain_core.language_models import BaseChatModel

from app.config import (
    ANTHROPIC_API_BASE,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    REQUIRED_ANTHROPIC_ENV_VARS,
    REQUIRED_GOOGLE_ENV_VARS,
    ROUTING_TIER_MODEL,
    ROUTING_TIER_PROVIDER,
    validate_required_environment,
)

logger = logging.getLogger(__name__)


def _build_openai(
    *,
    model: str,
    temperature: float,
    timeout: float | None,
    max_retries: int | None,
    api_key: str | None,
    base_url: str | None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key or OPENAI_API_KEY,
        "base_url": base_url or OPENAI_API_BASE,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    return ChatOpenAI(**kwargs)


def _build_anthropic(
    *,
    model: str,
    temperature: float,
    timeout: float | None,
    max_retries: int | None,
    api_key: str | None,
    base_url: str | None,
) -> BaseChatModel:
    validate_required_environment(REQUIRED_ANTHROPIC_ENV_VARS)

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic provider requested but langchain-anthropic is not "
            "installed. pip install -r requirements-optional.txt"
        ) from exc

    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key or ANTHROPIC_API_KEY,
        "base_url": base_url or ANTHROPIC_API_BASE,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    return ChatAnthropic(**kwargs)


def _build_google(
    *,
    model: str,
    temperature: float,
    timeout: float | None,
    max_retries: int | None,
    api_key: str | None,
    base_url: str | None,
) -> BaseChatModel:
    validate_required_environment(REQUIRED_GOOGLE_ENV_VARS)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Google provider requested but langchain-google-genai is not "
            "installed. pip install -r requirements-optional.txt"
        ) from exc

    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "google_api_key": api_key or GOOGLE_API_KEY,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    return ChatGoogleGenerativeAI(**kwargs)


_CHAT_BUILDERS = {
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "google": _build_google,
}


def build_chat_model(
    *,
    provider: str,
    model: str,
    temperature: float = 0.0,
    timeout: float | None = None,
    max_retries: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Build a chat model for the given provider.

    Dispatches to the provider's LangChain integration, which already
    implements the shared BaseChatModel contract, so callers (tool
    binding, LangGraph) do not need any provider-specific handling.
    """

    builder = _CHAT_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown chat provider: {provider!r}. "
            f"Supported providers: {sorted(_CHAT_BUILDERS)}"
        )

    logger.info(
        "Building chat model | provider=%s | model=%s | temperature=%.2f",
        provider,
        model,
        temperature,
    )

    return builder(
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
        base_url=base_url,
    )


def get_default_chat_model() -> BaseChatModel:
    """Reproduce the original OpenAI-only model construction unchanged."""

    validate_required_environment(("OPENAI_API_KEY",))

    return build_chat_model(
        provider="openai",
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )


def get_chat_model_for_tier(tier: str) -> BaseChatModel:
    """Build a chat model for a named routing tier (e.g. 'fast', 'capable')."""

    tier_key = tier.upper()
    provider = ROUTING_TIER_PROVIDER.get(tier_key, "openai")
    model = ROUTING_TIER_MODEL.get(tier_key, OPENAI_MODEL)

    return build_chat_model(provider=provider, model=model, temperature=OPENAI_TEMPERATURE)
