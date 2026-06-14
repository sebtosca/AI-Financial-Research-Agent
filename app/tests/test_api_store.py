from uuid import uuid4

import pytest

from app.api.schemas import EventType, RunEvent, RunRecord, RunStatus, ThreadRecord
from app.api.store import PostgresRunStore, SqliteRunStore, create_run_store


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_sqlite_store_preserves_research_across_instances(tmp_path):
    database_path = tmp_path / "research.sqlite3"
    first_store = SqliteRunStore(database_path)
    thread = await first_store.create_thread(ThreadRecord(title="NVIDIA research"))
    run = await first_store.create_run(
        RunRecord(
            thread_id=thread.id,
            query="Analyze NVIDIA",
            with_rag=True,
        )
    )
    await first_store.update_run(
        run.id,
        status=RunStatus.COMPLETED,
        report="Saved NVIDIA report",
    )
    await first_store.append_event(
        RunEvent(
            id=uuid4(),
            run_id=run.id,
            sequence=1,
            type=EventType.RUN_COMPLETED,
            payload={"report": "Saved NVIDIA report"},
        )
    )

    restarted_store = SqliteRunStore(database_path)
    threads = await restarted_store.list_threads()
    detail = await restarted_store.get_thread(thread.id)
    events = await restarted_store.events_after(run.id, 0)

    assert [item.id for item in threads] == [thread.id]
    assert detail is not None
    assert detail.runs[0].report == "Saved NVIDIA report"
    assert events[0].type == EventType.RUN_COMPLETED
    assert await restarted_store.latest_event_sequence(run.id) == 1


def test_store_factory_uses_sqlite_without_database_url(tmp_path):
    store = create_run_store(
        database_url=None,
        sqlite_path=tmp_path / "research.sqlite3",
    )

    assert isinstance(store, SqliteRunStore)


def test_store_factory_uses_postgres_when_database_url_is_configured(tmp_path):
    store = create_run_store(
        database_url="postgresql://user:password@localhost/research",
        sqlite_path=tmp_path / "unused.sqlite3",
        postgres_min_size=0,
    )

    assert isinstance(store, PostgresRunStore)


def test_postgres_store_rejects_invalid_pool_sizes():
    with pytest.raises(ValueError, match="pool size"):
        PostgresRunStore(
            "postgresql://user:password@localhost/research",
            min_size=5,
            max_size=2,
        )
