from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app import worker as worker_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_execute_run_task_calls_service_execute_with_parsed_uuid(monkeypatch):
    store = Mock()
    execute = AsyncMock()
    monkeypatch.setattr(
        worker_module,
        "AgentRunService",
        Mock(return_value=Mock(execute=execute)),
    )

    run_id = uuid4()
    await worker_module.execute_run_task({"run_store": store}, str(run_id))

    execute.assert_awaited_once_with(run_id)


@pytest.mark.anyio
async def test_startup_builds_store_and_stores_in_ctx(monkeypatch):
    fake_store = Mock()
    fake_store.initialize = AsyncMock()
    monkeypatch.setattr(worker_module, "create_run_store", Mock(return_value=fake_store))
    monkeypatch.setattr(worker_module, "REDIS_URL", None)

    ctx: dict = {}
    await worker_module.startup(ctx)

    assert ctx["run_store"] is fake_store
    assert ctx["redis_client"] is None
    fake_store.initialize.assert_awaited_once()


@pytest.mark.anyio
async def test_shutdown_closes_store_and_redis_client():
    store = Mock()
    store.close = AsyncMock()
    redis_client = Mock()
    redis_client.close = AsyncMock()

    await worker_module.shutdown({"run_store": store, "redis_client": redis_client})

    store.close.assert_awaited_once()
    redis_client.close.assert_awaited_once()


@pytest.mark.anyio
async def test_shutdown_handles_missing_context_gracefully():
    await worker_module.shutdown({})  # must not raise
