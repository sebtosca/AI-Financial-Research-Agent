from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.observability import tracing as tracing_module
from app.observability.tracing import build_trace_context, build_tracer, extract_run_id, submit_feedback
from app.routing.policy import ModelTier, RoutingDecision


def _decision(**overrides) -> RoutingDecision:
    defaults = dict(
        model_tier=ModelTier.CAPABLE,
        provider="openai",
        model_name="gpt-4o",
        tool_names=("get_stock_price", "query_private_database"),
        rag_engaged=True,
        matched_rules=("llm_classifier:Full comparison request.",),
    )
    defaults.update(overrides)
    return RoutingDecision(**defaults)


def test_build_trace_context_produces_expected_tags_and_metadata():
    run_id = uuid4()
    thread_id = uuid4()
    decision = _decision()

    context = build_trace_context(run_id, thread_id, decision)

    assert "model_tier:capable" in context.tags
    assert "provider:openai" in context.tags
    assert "rag_engaged:True" in context.tags
    assert context.metadata["run_id"] == str(run_id)
    assert context.metadata["thread_id"] == str(thread_id)
    assert context.metadata["matched_rules"] == list(decision.matched_rules)
    assert context.run_name == f"argent-run-{run_id}"


def test_build_tracer_returns_none_when_tracing_disabled(monkeypatch):
    monkeypatch.setattr(tracing_module, "LANGCHAIN_TRACING_V2", False)

    context = build_trace_context(uuid4(), uuid4(), _decision())

    assert build_tracer(context) is None


def test_build_tracer_returns_tracer_when_tracing_enabled(monkeypatch):
    monkeypatch.setattr(tracing_module, "LANGCHAIN_TRACING_V2", True)
    fake_tracer_class = Mock(return_value="fake-tracer-instance")
    monkeypatch.setattr("langchain_core.tracers.LangChainTracer", fake_tracer_class)

    context = build_trace_context(uuid4(), uuid4(), _decision())
    tracer = build_tracer(context)

    assert tracer == "fake-tracer-instance"
    _, kwargs = fake_tracer_class.call_args
    assert kwargs["tags"] == context.tags
    assert kwargs["metadata"] == context.metadata


def test_extract_run_id_returns_none_for_missing_tracer():
    assert extract_run_id(None) is None


def test_extract_run_id_returns_none_when_latest_run_unset():
    tracer = Mock(spec=[])  # no attributes at all, including latest_run

    assert extract_run_id(tracer) is None


def test_extract_run_id_returns_string_of_latest_run_id():
    run_uuid = uuid4()
    tracer = Mock()
    tracer.latest_run = Mock(id=run_uuid)

    assert extract_run_id(tracer) == str(run_uuid)


def test_submit_feedback_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing_module, "LANGSMITH_FEEDBACK_ENABLED", False)
    client_factory = Mock()
    monkeypatch.setattr(tracing_module, "_client", client_factory)

    submit_feedback("langsmith-run-id", rating=1, comment="great")

    client_factory.assert_not_called()


def test_submit_feedback_calls_client_when_enabled(monkeypatch):
    monkeypatch.setattr(tracing_module, "LANGSMITH_FEEDBACK_ENABLED", True)
    fake_client = Mock()
    monkeypatch.setattr(tracing_module, "_client", Mock(return_value=fake_client))

    submit_feedback("langsmith-run-id", rating=1, comment="great")

    fake_client.create_feedback.assert_called_once_with(
        run_id="langsmith-run-id", key="user_rating", score=1, comment="great"
    )


def test_submit_feedback_swallows_client_errors(monkeypatch):
    monkeypatch.setattr(tracing_module, "LANGSMITH_FEEDBACK_ENABLED", True)
    fake_client = Mock()
    fake_client.create_feedback.side_effect = RuntimeError("network unavailable")
    monkeypatch.setattr(tracing_module, "_client", Mock(return_value=fake_client))

    submit_feedback("langsmith-run-id", rating=-1, comment=None)  # must not raise
