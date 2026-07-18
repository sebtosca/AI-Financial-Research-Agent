from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.api import routes as routes_module
from app.api.schemas import EventType, RunEvent, RunRecord, RunStatus, ThreadRecord
from app.api.store import InMemoryRunStore
from app.main import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_and_thread_lifecycle():
    app = create_app(InMemoryRunStore())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/v1/health/live")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        created = await client.post(
            "/api/v1/threads",
            json={"title": "NVIDIA research"},
        )
        assert created.status_code == 201
        thread_id = created.json()["id"]

        detail = await client.get(f"/api/v1/threads/{thread_id}")
        assert detail.status_code == 200
        assert detail.json()["title"] == "NVIDIA research"


@pytest.mark.anyio
async def test_create_run_queues_background_execution(monkeypatch):
    app = create_app(InMemoryRunStore())
    service = app.state.run_service
    monkeypatch.setattr(service, "execute", AsyncMock())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        thread = (await client.post("/api/v1/threads", json={})).json()
        response = await client.post(
            f"/api/v1/threads/{thread['id']}/runs",
            json={"query": "Analyze NVIDIA", "with_rag": True},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["with_rag"] is True
    service.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_create_run_enqueues_via_arq_when_configured():
    app = create_app(InMemoryRunStore())
    arq_pool = AsyncMock()
    app.state.arq_pool = arq_pool
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        thread = (await client.post("/api/v1/threads", json={})).json()
        response = await client.post(
            f"/api/v1/threads/{thread['id']}/runs",
            json={"query": "Analyze NVIDIA", "with_rag": True},
        )

    assert response.status_code == 202
    run_id = response.json()["id"]
    arq_pool.enqueue_job.assert_awaited_once_with("execute_run_task", run_id)


@pytest.mark.anyio
async def test_missing_resources_return_404():
    transport = httpx.ASGITransport(app=create_app(InMemoryRunStore()))
    missing_id = "00000000-0000-0000-0000-000000000000"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get(f"/api/v1/threads/{missing_id}")).status_code == 404
        assert (await client.get(f"/api/v1/runs/{missing_id}")).status_code == 404
        assert (
            await client.post(f"/api/v1/runs/{missing_id}/feedback", json={"rating": 1})
        ).status_code == 404


@pytest.mark.anyio
async def test_create_feedback_persists_locally(monkeypatch):
    submit_feedback = Mock()
    monkeypatch.setattr(routes_module, "submit_feedback", submit_feedback)

    store = InMemoryRunStore()
    thread = await store.create_thread(ThreadRecord(title="NVIDIA research"))
    run = await store.create_run(
        RunRecord(thread_id=thread.id, query="Analyze NVIDIA", with_rag=True)
    )
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/runs/{run.id}/feedback",
            json={"rating": 1, "comment": "Helpful report"},
        )

    assert response.status_code == 204
    stored_feedback = await store.list_feedback(run.id)
    assert len(stored_feedback) == 1
    assert stored_feedback[0].rating == 1
    assert stored_feedback[0].comment == "Helpful report"
    submit_feedback.assert_not_called()  # run has no langsmith_run_id


@pytest.mark.anyio
async def test_create_feedback_forwards_to_langsmith_when_run_has_trace_id(monkeypatch):
    submit_feedback = Mock()
    monkeypatch.setattr(routes_module, "submit_feedback", submit_feedback)

    store = InMemoryRunStore()
    thread = await store.create_thread(ThreadRecord(title="NVIDIA research"))
    run = await store.create_run(
        RunRecord(thread_id=thread.id, query="Analyze NVIDIA", with_rag=True)
    )
    await store.update_run(run.id, langsmith_run_id="fake-langsmith-run-id")
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/runs/{run.id}/feedback",
            json={"rating": -1, "comment": "Missed a source"},
        )

    assert response.status_code == 204
    submit_feedback.assert_called_once_with(
        "fake-langsmith-run-id", rating=-1, comment="Missed a source"
    )


@pytest.mark.anyio
async def test_terminal_event_replay_closes_stream():
    store = InMemoryRunStore()
    thread = await store.create_thread(ThreadRecord(title="Completed research"))
    run = await store.create_run(
        RunRecord(
            thread_id=thread.id,
            query="Analyze NVIDIA",
            with_rag=True,
            status=RunStatus.COMPLETED,
        )
    )
    await store.append_event(
        RunEvent(
            run_id=run.id,
            sequence=1,
            type=EventType.RUN_COMPLETED,
            payload={"report": "Finished"},
        )
    )
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/v1/runs/{run.id}/events")

    assert response.status_code == 200
    assert response.text.count("event: run.completed") == 1
