from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import httpx
import pytest

from app.api import routes as routes_module
from app.api.auth import hash_api_key
from app.api.schemas import ApiKeyRecord, ApiKeyRole, EventType, RunEvent, RunRecord, RunStatus, ThreadRecord
from app.api.store import InMemoryRunStore, RunStore
from app.main import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _create_key(store: RunStore, *, role: ApiKeyRole = ApiKeyRole.USER) -> tuple[UUID, dict]:
    raw_key = f"test-key-{uuid4().hex}"
    record = ApiKeyRecord(hashed_key=hash_api_key(raw_key), label="test", role=role)
    await store.create_api_key(record)
    return record.id, {"Authorization": f"Bearer {raw_key}"}


@pytest.mark.anyio
async def test_health_and_thread_lifecycle():
    store = InMemoryRunStore()
    app = create_app(store)
    _, headers = await _create_key(store)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/v1/health/live")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        created = await client.post(
            "/api/v1/threads",
            json={"title": "NVIDIA research"},
            headers=headers,
        )
        assert created.status_code == 201
        thread_id = created.json()["id"]

        detail = await client.get(f"/api/v1/threads/{thread_id}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["title"] == "NVIDIA research"


@pytest.mark.anyio
async def test_create_run_queues_background_execution(monkeypatch):
    store = InMemoryRunStore()
    app = create_app(store)
    service = app.state.run_service
    monkeypatch.setattr(service, "execute", AsyncMock())
    _, headers = await _create_key(store)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        thread = (await client.post("/api/v1/threads", json={}, headers=headers)).json()
        response = await client.post(
            f"/api/v1/threads/{thread['id']}/runs",
            json={"query": "Analyze NVIDIA", "with_rag": True},
            headers=headers,
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["with_rag"] is True
    service.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_create_run_enqueues_via_arq_when_configured():
    store = InMemoryRunStore()
    app = create_app(store)
    arq_pool = AsyncMock()
    app.state.arq_pool = arq_pool
    _, headers = await _create_key(store)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        thread = (await client.post("/api/v1/threads", json={}, headers=headers)).json()
        response = await client.post(
            f"/api/v1/threads/{thread['id']}/runs",
            json={"query": "Analyze NVIDIA", "with_rag": True},
            headers=headers,
        )

    assert response.status_code == 202
    run_id = response.json()["id"]
    arq_pool.enqueue_job.assert_awaited_once_with("execute_run_task", run_id)


@pytest.mark.anyio
async def test_missing_resources_return_404():
    store = InMemoryRunStore()
    _, headers = await _create_key(store)
    transport = httpx.ASGITransport(app=create_app(store))
    missing_id = "00000000-0000-0000-0000-000000000000"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.get(f"/api/v1/threads/{missing_id}", headers=headers)
        ).status_code == 404
        assert (
            await client.get(f"/api/v1/runs/{missing_id}", headers=headers)
        ).status_code == 404
        assert (
            await client.post(
                f"/api/v1/runs/{missing_id}/feedback", json={"rating": 1}, headers=headers
            )
        ).status_code == 404


@pytest.mark.anyio
async def test_requests_without_api_key_are_rejected():
    transport = httpx.ASGITransport(app=create_app(InMemoryRunStore()))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/api/v1/threads", json={})).status_code == 401
        assert (await client.get("/api/v1/health/live")).status_code == 200


@pytest.mark.anyio
async def test_user_cannot_see_another_users_thread():
    store = InMemoryRunStore()
    owner_id, owner_headers = await _create_key(store)
    _, other_headers = await _create_key(store)
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/threads", json={"title": "Private research"}, headers=owner_headers
        )
        thread_id = created.json()["id"]

        own_view = await client.get(f"/api/v1/threads/{thread_id}", headers=owner_headers)
        other_view = await client.get(f"/api/v1/threads/{thread_id}", headers=other_headers)
        thread_list = await client.get("/api/v1/threads", headers=other_headers)

    assert own_view.status_code == 200
    assert other_view.status_code == 404
    assert all(t["id"] != thread_id for t in thread_list.json())


@pytest.mark.anyio
async def test_admin_can_see_any_thread():
    store = InMemoryRunStore()
    _, owner_headers = await _create_key(store)
    _, admin_headers = await _create_key(store, role=ApiKeyRole.ADMIN)
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/threads", json={"title": "Private research"}, headers=owner_headers
        )
        thread_id = created.json()["id"]

        admin_view = await client.get(f"/api/v1/threads/{thread_id}", headers=admin_headers)

    assert admin_view.status_code == 200


@pytest.mark.anyio
async def test_create_feedback_persists_locally(monkeypatch):
    submit_feedback = Mock()
    monkeypatch.setattr(routes_module, "submit_feedback", submit_feedback)

    store = InMemoryRunStore()
    owner_id, headers = await _create_key(store)
    thread = await store.create_thread(ThreadRecord(title="NVIDIA research", owner_key_id=owner_id))
    run = await store.create_run(
        RunRecord(thread_id=thread.id, query="Analyze NVIDIA", with_rag=True, owner_key_id=owner_id)
    )
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/runs/{run.id}/feedback",
            json={"rating": 1, "comment": "Helpful report"},
            headers=headers,
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
    owner_id, headers = await _create_key(store)
    thread = await store.create_thread(ThreadRecord(title="NVIDIA research", owner_key_id=owner_id))
    run = await store.create_run(
        RunRecord(thread_id=thread.id, query="Analyze NVIDIA", with_rag=True, owner_key_id=owner_id)
    )
    await store.update_run(run.id, langsmith_run_id="fake-langsmith-run-id")
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/runs/{run.id}/feedback",
            json={"rating": -1, "comment": "Missed a source"},
            headers=headers,
        )

    assert response.status_code == 204
    submit_feedback.assert_called_once_with(
        "fake-langsmith-run-id", rating=-1, comment="Missed a source"
    )


@pytest.mark.anyio
async def test_terminal_event_replay_closes_stream():
    store = InMemoryRunStore()
    owner_id, headers = await _create_key(store)
    thread = await store.create_thread(ThreadRecord(title="Completed research", owner_key_id=owner_id))
    run = await store.create_run(
        RunRecord(
            thread_id=thread.id,
            query="Analyze NVIDIA",
            with_rag=True,
            status=RunStatus.COMPLETED,
            owner_key_id=owner_id,
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
        response = await client.get(f"/api/v1/runs/{run.id}/events", headers=headers)

    assert response.status_code == 200
    assert response.text.count("event: run.completed") == 1
