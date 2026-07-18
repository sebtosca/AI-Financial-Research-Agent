import os

import pytest

from app.config import OPENAI_API_KEY
from app.routing.policy import ModelTier, classify_query


pytestmark = [pytest.mark.integration, pytest.mark.live]


def _live_tests_enabled() -> bool:
    return os.getenv("RUN_LIVE_AGENT_TESTS", "").lower() in {"1", "true", "yes"}


def _require_live_environment() -> None:
    if not _live_tests_enabled():
        pytest.skip("Set RUN_LIVE_AGENT_TESTS=true to run live agent tests")

    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is required for live routing classifier tests")


def test_live_classifier_routes_simple_price_query_to_fast_tier():
    _require_live_environment()

    decision = classify_query("What is the current price of NVDA?", with_rag_requested=True)

    assert decision.model_tier == ModelTier.FAST
    assert "query_private_database" not in decision.tool_names


def test_live_classifier_routes_comparison_query_to_capable_tier():
    _require_live_environment()

    decision = classify_query(
        "Compare MSFT, GOOGL, NVDA, and AMZN on AI investment attractiveness, "
        "including their AI research initiatives, and give a buy/hold/sell view.",
        with_rag_requested=True,
    )

    assert decision.model_tier == ModelTier.CAPABLE
    assert decision.rag_engaged is True
