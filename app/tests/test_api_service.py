from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.api import service as service_module
from app.api.schemas import AgentStage, EventType, RunRecord, ThreadRecord
from app.api.service import AgentRunService
from app.api.store import InMemoryRunStore
from app.routing.policy import ModelTier, RoutingDecision


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_process_update_translates_messages_in_order():
    store = InMemoryRunStore()
    service = AgentRunService(store)
    run_id = uuid4()
    update = {
        "agent": {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_stock_history",
                            "args": {
                                "ticker": "NVDA",
                                "period": "3y",
                                "secret": "excluded",
                            },
                            "id": "history-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        "tools": {
            "messages": [
                ToolMessage(
                    content='{"status":"success"}',
                    tool_call_id="history-call",
                    name="get_stock_history",
                )
            ]
        },
        "final": {"messages": [AIMessage(content="NVIDIA report")]},
    }

    report = await service._process_update(run_id, update)
    events = await store.events_after(run_id, 0)

    assert report == "NVIDIA report"
    assert [event.type for event in events] == [
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.REPORT_DELTA,
    ]
    assert events[0].stage == AgentStage.STOCK_HISTORY
    assert events[0].payload["input"] == {"ticker": "NVDA", "period": "3y"}


@pytest.mark.anyio
async def test_process_update_ignores_unknown_payloads():
    service = AgentRunService(InMemoryRunStore())

    report = await service._process_update(
        uuid4(),
        {"metadata": {"duration": 1.2}, "agent": {"messages": []}},
    )

    assert report is None


@pytest.mark.anyio
async def test_process_update_marks_tool_error_as_failed():
    store = InMemoryRunStore()
    service = AgentRunService(store)
    run_id = uuid4()

    await service._process_update(
        run_id,
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="Private analyst database unavailable",
                        tool_call_id="rag-call",
                        name="query_private_database",
                        status="error",
                    )
                ]
            }
        },
    )
    events = await store.events_after(run_id, 0)

    assert events[0].type == EventType.TOOL_FAILED
    assert events[0].stage == AgentStage.PRIVATE_RAG


@pytest.mark.anyio
async def test_execute_applies_routing_decision(monkeypatch):
    store = InMemoryRunStore()
    service = AgentRunService(store)
    thread = await store.create_thread(ThreadRecord(title="NVIDIA research"))
    query = "What is the current price of NVDA?"
    run = await store.create_run(
        RunRecord(thread_id=thread.id, query=query, with_rag=True)
    )

    expected_decision = RoutingDecision(
        model_tier=ModelTier.FAST,
        provider="openai",
        model_name="gpt-4o-mini",
        tool_names=("get_stock_price",),
        rag_engaged=False,
        matched_rules=("llm_classifier:Simple price lookup.",),
    )
    classify = Mock(return_value=expected_decision)
    monkeypatch.setattr(service_module, "classify_query", classify)

    class FakeAgent:
        def stream(self, *args, **kwargs):
            yield {"final": {"messages": [AIMessage(content="NVDA is at $100.")]}}

    create_agent = Mock(return_value=FakeAgent())
    monkeypatch.setattr(service_module, "create_financial_agent", create_agent)

    await service.execute(run.id)

    classify.assert_called_once_with(query, with_rag_requested=True)

    updated_run = await store.get_run(run.id)
    assert updated_run.model_tier == expected_decision.model_tier.value
    assert updated_run.provider == expected_decision.provider
    assert updated_run.model_name == expected_decision.model_name
    assert updated_run.tool_subset == list(expected_decision.tool_names)
    assert updated_run.rag_engaged == expected_decision.rag_engaged

    events = await store.events_after(run.id, 0)
    routing_events = [event for event in events if event.type == EventType.ROUTING_DECIDED]
    assert len(routing_events) == 1
    assert routing_events[0].stage == AgentStage.ROUTING
    assert routing_events[0].payload["model_tier"] == expected_decision.model_tier.value
    assert routing_events[0].payload["tools"] == list(expected_decision.tool_names)

    create_agent.assert_called_once()
    _, kwargs = create_agent.call_args
    assert kwargs["model_tier"] == expected_decision.model_tier.value
    assert [tool.name for tool in kwargs["tools"]] == list(expected_decision.tool_names)


@pytest.mark.anyio
async def test_execute_persists_token_and_cost_aggregates(monkeypatch):
    store = InMemoryRunStore()
    service = AgentRunService(store)
    thread = await store.create_thread(ThreadRecord(title="NVIDIA research"))
    query = "What is the current price of NVDA?"
    run = await store.create_run(RunRecord(thread_id=thread.id, query=query, with_rag=True))

    decision = RoutingDecision(
        model_tier=ModelTier.FAST,
        provider="openai",
        model_name="gpt-4o-mini",
        tool_names=("get_stock_price",),
        rag_engaged=False,
        matched_rules=("llm_classifier:test",),
    )
    monkeypatch.setattr(service_module, "classify_query", Mock(return_value=decision))

    final_message = AIMessage(content="NVDA is at $100.")
    final_message.usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}

    class FakeAgent:
        def stream(self, *args, **kwargs):
            yield {"final": {"messages": [final_message]}}

    monkeypatch.setattr(service_module, "create_financial_agent", Mock(return_value=FakeAgent()))

    await service.execute(run.id)

    updated_run = await store.get_run(run.id)
    assert updated_run.prompt_tokens == 100
    assert updated_run.completion_tokens == 20
    assert updated_run.estimated_cost_usd == pytest.approx(
        (100 / 1000) * 0.00015 + (20 / 1000) * 0.0006
    )


@pytest.mark.anyio
async def test_tool_call_duration_is_observed_between_start_and_completion():
    store = InMemoryRunStore()
    service = AgentRunService(store)
    run_id = uuid4()

    await service._process_update(
        run_id,
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_stock_price",
                                "args": {"ticker": "NVDA"},
                                "id": "price-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        },
    )

    assert "price-call" in service._tool_call_started

    await service._process_update(
        run_id,
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"status":"success"}',
                        tool_call_id="price-call",
                        name="get_stock_price",
                    )
                ]
            }
        },
    )

    assert "price-call" not in service._tool_call_started
