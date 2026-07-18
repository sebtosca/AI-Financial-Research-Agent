import pytest

from app.routing.policy import ModelTier, _RoutingClassification, classify_query


class _FakeStructuredModel:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def invoke(self, messages):
        if self._error:
            raise self._error
        return self._result


class _FakeClassifierModel:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def with_structured_output(self, schema):
        return _FakeStructuredModel(result=self._result, error=self._error)


def _classification(**overrides) -> _RoutingClassification:
    defaults = dict(
        model_tier="fast",
        relevant_tools=["get_stock_price"],
        needs_private_database=False,
        reasoning="Simple price lookup.",
    )
    defaults.update(overrides)
    return _RoutingClassification(**defaults)


def test_simple_query_uses_classifier_output_for_fast_tier():
    fake_model = _FakeClassifierModel(result=_classification())

    decision = classify_query("What is the price of NVDA?", with_rag_requested=True, model=fake_model)

    assert decision.model_tier == ModelTier.FAST
    assert decision.tool_names == ("get_stock_price",)
    assert decision.rag_engaged is False
    assert decision.matched_rules == ("llm_classifier:Simple price lookup.",)


def test_capable_tier_and_rag_from_classifier_output():
    fake_model = _FakeClassifierModel(
        result=_classification(
            model_tier="capable",
            relevant_tools=["get_stock_price", "get_stock_history", "search_financial_news", "analyze_sentiment"],
            needs_private_database=True,
            reasoning="Full comparison and AI-initiative research request.",
        )
    )

    decision = classify_query(
        "Compare MSFT, GOOGL, and NVDA on AI investment attractiveness.",
        with_rag_requested=True,
        model=fake_model,
    )

    assert decision.model_tier == ModelTier.CAPABLE
    assert set(decision.tool_names) == {
        "get_stock_price",
        "get_stock_history",
        "search_financial_news",
        "analyze_sentiment",
        "query_private_database",
    }
    assert decision.rag_engaged is True


def test_with_rag_false_overrides_classifier_saying_rag_is_needed():
    fake_model = _FakeClassifierModel(
        result=_classification(needs_private_database=True)
    )

    decision = classify_query("Tell me about NVIDIA's AI roadmap.", with_rag_requested=False, model=fake_model)

    assert decision.rag_engaged is False
    assert "query_private_database" not in decision.tool_names


def test_empty_relevant_tools_falls_back_to_full_general_toolset():
    fake_model = _FakeClassifierModel(result=_classification(relevant_tools=[]))

    decision = classify_query("Hi", with_rag_requested=False, model=fake_model)

    assert set(decision.tool_names) == {
        "get_stock_price",
        "get_stock_history",
        "search_financial_news",
        "analyze_sentiment",
    }


def test_classification_failure_falls_back_to_safe_default():
    fake_model = _FakeClassifierModel(error=RuntimeError("classifier unavailable"))

    decision = classify_query("What is the price of NVDA?", with_rag_requested=True, model=fake_model)

    assert decision.model_tier == ModelTier.CAPABLE
    assert decision.matched_rules == ("llm_classification_failed",)
    assert set(decision.tool_names) == {
        "get_stock_price",
        "get_stock_history",
        "search_financial_news",
        "analyze_sentiment",
        "query_private_database",
    }


def test_routing_disabled_returns_safe_default_without_calling_model(monkeypatch):
    monkeypatch.setattr("app.routing.policy.ROUTING_ENABLED", False)

    class ExplodingModel:
        def with_structured_output(self, schema):
            raise AssertionError("classifier should not be invoked when routing is disabled")

    decision = classify_query("What is the price of NVDA?", with_rag_requested=True, model=ExplodingModel())

    assert decision.matched_rules == ("routing_disabled",)
