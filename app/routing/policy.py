import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.config import (
    OPENAI_MODEL,
    ROUTING_ENABLED,
    ROUTING_TIER_MODEL,
    ROUTING_TIER_PROVIDER,
)
from app.providers import build_chat_model

logger = logging.getLogger(__name__)

ALL_TOOL_NAMES = (
    "get_stock_price",
    "get_stock_history",
    "search_financial_news",
    "analyze_sentiment",
)
RAG_TOOL_NAME = "query_private_database"


class ModelTier(str, Enum):
    FAST = "fast"
    CAPABLE = "capable"


@dataclass(frozen=True)
class RoutingDecision:
    model_tier: ModelTier
    provider: str
    model_name: str
    tool_names: tuple[str, ...]
    rag_engaged: bool
    matched_rules: tuple[str, ...]


class _RoutingClassification(BaseModel):
    """Structured output schema the routing classifier must return."""

    model_tier: Literal["fast", "capable"] = Field(
        description=(
            "'fast' for a narrow factual lookup (a single price/history "
            "check). 'capable' for comparison, ranking, or a full research/"
            "investment-analysis request."
        )
    )
    relevant_tools: list[
        Literal[
            "get_stock_price",
            "get_stock_history",
            "search_financial_news",
            "analyze_sentiment",
        ]
    ] = Field(description="Which general-purpose tools are relevant to answering this query.")
    needs_private_database: bool = Field(
        description=(
            "Whether internal analyst reports about company AI initiatives "
            "are relevant to this query."
        )
    )
    reasoning: str = Field(description="One sentence explaining the classification.")


_CLASSIFIER_SYSTEM_PROMPT = (
    "You are the routing policy for a financial research agent. Classify "
    "the user's query so the system can pick a model tier, tool subset, "
    "and data sources. Do not answer the query yourself -- only classify it."
)


def _model_for_tier(tier: ModelTier) -> tuple[str, str]:
    tier_key = tier.value.upper()
    provider = ROUTING_TIER_PROVIDER.get(tier_key, "openai")
    model_name = ROUTING_TIER_MODEL.get(tier_key, OPENAI_MODEL)
    return provider, model_name


def _default_classifier_model() -> BaseChatModel:
    # Classification itself always runs on the cheap "fast" tier model,
    # regardless of which tier it ultimately routes the query to.
    provider, model_name = _model_for_tier(ModelTier.FAST)
    return build_chat_model(provider=provider, model=model_name, temperature=0.0)


def _fallback_decision(with_rag_requested: bool, matched_rules: tuple[str, ...]) -> RoutingDecision:
    tool_names = ALL_TOOL_NAMES + ((RAG_TOOL_NAME,) if with_rag_requested else ())
    provider, model_name = _model_for_tier(ModelTier.CAPABLE)
    return RoutingDecision(
        model_tier=ModelTier.CAPABLE,
        provider=provider,
        model_name=model_name,
        tool_names=tool_names,
        rag_engaged=with_rag_requested,
        matched_rules=matched_rules,
    )


def classify_query(
    query: str,
    with_rag_requested: bool,
    model: Optional[BaseChatModel] = None,
) -> RoutingDecision:
    """Classify a query into a routing decision using an LLM classifier.

    Classification always runs on the cheap "fast" tier model (via
    structured output) so the extra call stays low-cost regardless of
    which tier the query is ultimately routed to. Falls back to a safe,
    maximal default (capable tier, full tool set) if classification is
    disabled or the call fails for any reason -- the same graceful-
    degradation pattern used elsewhere in this codebase (e.g. the
    sentiment tool's keyword fallback).
    """

    if not ROUTING_ENABLED:
        return _fallback_decision(with_rag_requested, ("routing_disabled",))

    classifier_model = model or _default_classifier_model()

    try:
        structured_model = classifier_model.with_structured_output(_RoutingClassification)
        result = structured_model.invoke(
            [
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        )
    except Exception:
        logger.exception("Routing classification failed, using safe fallback decision")
        return _fallback_decision(with_rag_requested, ("llm_classification_failed",))

    model_tier = ModelTier(result.model_tier)
    rag_engaged = with_rag_requested and result.needs_private_database
    tool_names = tuple(dict.fromkeys(result.relevant_tools)) or ALL_TOOL_NAMES
    if rag_engaged:
        tool_names = tool_names + (RAG_TOOL_NAME,)

    provider, model_name = _model_for_tier(model_tier)

    decision = RoutingDecision(
        model_tier=model_tier,
        provider=provider,
        model_name=model_name,
        tool_names=tool_names,
        rag_engaged=rag_engaged,
        matched_rules=(f"llm_classifier:{result.reasoning}",),
    )

    logger.info(
        "Routing decision | tier=%s | provider=%s | model=%s | tools=%s | "
        "rag_engaged=%s | reasoning=%s",
        decision.model_tier.value,
        decision.provider,
        decision.model_name,
        decision.tool_names,
        decision.rag_engaged,
        result.reasoning,
    )

    return decision
