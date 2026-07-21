import os
from uuid import uuid4

import pytest

from app.api.schemas import ApiKeyRecord, EventType, RunEvent, RunRecord, RunStatus, ThreadRecord
from app.api.store import PostgresRunStore


pytestmark = pytest.mark.integration


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_postgres_store_lifecycle():
    database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    store = PostgresRunStore(database_url, min_size=1, max_size=2)
    await store.initialize()
    try:
        thread = await store.create_thread(
            ThreadRecord(title=f"PostgreSQL test {uuid4()}")
        )
        run = await store.create_run(
            RunRecord(
                thread_id=thread.id,
                query="Analyze NVIDIA",
                with_rag=True,
            )
        )
        updated = await store.update_run(
            run.id,
            status=RunStatus.RUNNING,
        )
        await store.append_event(
            RunEvent(
                run_id=run.id,
                sequence=1,
                type=EventType.RUN_STARTED,
            )
        )

        detail = await store.get_thread(thread.id)
        events = await store.events_after(run.id, 0)

        assert await store.healthcheck() is True
        assert updated.status == RunStatus.RUNNING
        assert detail is not None
        assert detail.runs[0].id == run.id
        assert events[0].type == EventType.RUN_STARTED
        assert await store.latest_event_sequence(run.id) == 1
        assert await store.cancel(run.id) is True
        assert await store.is_cancelled(run.id) is True

        api_key = await store.create_api_key(
            ApiKeyRecord(hashed_key=f"hash-{uuid4()}", label="postgres test")
        )
        assert await store.get_api_key_by_hash(api_key.hashed_key) is not None
        assert any(key.id == api_key.id for key in await store.list_api_keys())
        assert await store.revoke_api_key(api_key.id) is True
    finally:
        await store.close()
