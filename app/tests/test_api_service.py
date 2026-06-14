from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.api.schemas import AgentStage, EventType
from app.api.service import AgentRunService
from app.api.store import InMemoryRunStore


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
