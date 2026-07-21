import asyncio
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .event_fanout import EventFanout
from .schemas import (
    ApiKeyRecord,
    FeedbackRecord,
    RunEvent,
    RunRecord,
    RunStatus,
    ThreadDetail,
    ThreadRecord,
)


class RunStore(Protocol):
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def healthcheck(self) -> bool: ...
    async def create_thread(self, thread: ThreadRecord) -> ThreadRecord: ...
    async def list_threads(self) -> list[ThreadRecord]: ...
    async def get_thread(self, thread_id: UUID) -> ThreadDetail | None: ...
    async def create_run(self, run: RunRecord) -> RunRecord: ...
    async def get_run(self, run_id: UUID) -> RunRecord | None: ...
    async def update_run(self, run_id: UUID, **changes) -> RunRecord: ...
    async def append_event(self, event: RunEvent) -> None: ...
    async def events_after(self, run_id: UUID, sequence: int) -> list[RunEvent]: ...
    async def subscribe(self, run_id: UUID) -> asyncio.Queue[RunEvent]: ...
    async def unsubscribe(
        self, run_id: UUID, queue: asyncio.Queue[RunEvent]
    ) -> None: ...
    async def cancel(self, run_id: UUID) -> bool: ...
    async def is_cancelled(self, run_id: UUID) -> bool: ...
    async def latest_event_sequence(self, run_id: UUID) -> int: ...
    async def add_feedback(self, feedback: FeedbackRecord) -> None: ...
    async def list_feedback(self, run_id: UUID) -> list[FeedbackRecord]: ...
    async def create_api_key(self, key: ApiKeyRecord) -> ApiKeyRecord: ...
    async def get_api_key_by_hash(self, hashed_key: str) -> ApiKeyRecord | None: ...
    async def list_api_keys(self) -> list[ApiKeyRecord]: ...
    async def revoke_api_key(self, key_id: UUID) -> bool: ...


async def _no_op() -> None:
    return None


async def _healthy() -> bool:
    return True


class InMemoryRunStore:
    """Development store behind the same boundary used by the API service."""

    def __init__(self) -> None:
        self._threads: dict[UUID, ThreadRecord] = {}
        self._runs: dict[UUID, RunRecord] = {}
        self._events: dict[UUID, list[RunEvent]] = defaultdict(list)
        self._feedback: dict[UUID, list[FeedbackRecord]] = defaultdict(list)
        self._api_keys: dict[UUID, ApiKeyRecord] = {}
        self._subscribers: dict[UUID, set[asyncio.Queue[RunEvent]]] = defaultdict(set)
        self._cancelled: set[UUID] = set()
        self._lock = asyncio.Lock()

    initialize = close = staticmethod(_no_op)
    healthcheck = staticmethod(_healthy)

    async def create_thread(self, thread: ThreadRecord) -> ThreadRecord:
        async with self._lock:
            self._threads[thread.id] = thread
        return thread

    async def list_threads(self) -> list[ThreadRecord]:
        async with self._lock:
            return sorted(
                self._threads.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )

    async def get_thread(self, thread_id: UUID) -> ThreadDetail | None:
        async with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                return None
            runs = [run for run in self._runs.values() if run.thread_id == thread_id]
            return ThreadDetail(**thread.model_dump(), runs=runs)

    async def create_run(self, run: RunRecord) -> RunRecord:
        async with self._lock:
            self._runs[run.id] = run
            thread = self._threads[run.thread_id]
            thread.updated_at = datetime.now(timezone.utc)
        return run

    async def get_run(self, run_id: UUID) -> RunRecord | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def update_run(self, run_id: UUID, **changes) -> RunRecord:
        async with self._lock:
            run = self._runs[run_id]
            updated = run.model_copy(update=changes)
            self._runs[run_id] = updated
            return updated

    async def append_event(self, event: RunEvent) -> None:
        async with self._lock:
            self._events[event.run_id].append(event)
            subscribers = list(self._subscribers[event.run_id])
        for queue in subscribers:
            await queue.put(event)

    async def events_after(self, run_id: UUID, sequence: int) -> list[RunEvent]:
        async with self._lock:
            return [event for event in self._events[run_id] if event.sequence > sequence]

    async def subscribe(self, run_id: UUID) -> asyncio.Queue[RunEvent]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        async with self._lock:
            self._subscribers[run_id].add(queue)
        return queue

    async def unsubscribe(self, run_id: UUID, queue: asyncio.Queue[RunEvent]) -> None:
        async with self._lock:
            self._subscribers[run_id].discard(queue)

    async def cancel(self, run_id: UUID) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return False
            self._cancelled.add(run_id)
            return True

    async def is_cancelled(self, run_id: UUID) -> bool:
        async with self._lock:
            return run_id in self._cancelled

    async def add_feedback(self, feedback: FeedbackRecord) -> None:
        async with self._lock:
            self._feedback[feedback.run_id].append(feedback)

    async def list_feedback(self, run_id: UUID) -> list[FeedbackRecord]:
        async with self._lock:
            return list(self._feedback[run_id])

    async def create_api_key(self, key: ApiKeyRecord) -> ApiKeyRecord:
        async with self._lock:
            self._api_keys[key.id] = key
        return key

    async def get_api_key_by_hash(self, hashed_key: str) -> ApiKeyRecord | None:
        async with self._lock:
            for key in self._api_keys.values():
                if key.hashed_key == hashed_key:
                    return key
        return None

    async def list_api_keys(self) -> list[ApiKeyRecord]:
        async with self._lock:
            return sorted(self._api_keys.values(), key=lambda item: item.created_at)

    async def revoke_api_key(self, key_id: UUID) -> bool:
        async with self._lock:
            key = self._api_keys.get(key_id)
            if key is None:
                return False
            self._api_keys[key_id] = key.model_copy(
                update={"revoked_at": datetime.now(timezone.utc)}
            )
            return True

    async def latest_event_sequence(self, run_id: UUID) -> int:
        async with self._lock:
            events = self._events[run_id]
            return events[-1].sequence if events else 0


class SqliteRunStore:
    """Persistent run store for a single API process."""

    def __init__(self, database_path: Path, redis_client: Any | None = None) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._fanout = EventFanout(redis_client)
        self._cancelled: set[UUID] = set()
        self._lock = asyncio.Lock()
        self._create_schema()

    healthcheck = staticmethod(_healthy)

    async def initialize(self) -> None:
        await self._fanout.start()

    async def close(self) -> None:
        await self._fanout.stop()
        async with self._lock:
            self._connection.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS runs_thread_id ON runs(thread_id);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_run_sequence
                ON events(run_id, sequence);
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS feedback_run_id ON feedback(run_id);
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                hashed_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    async def create_thread(self, thread: ThreadRecord) -> ThreadRecord:
        async with self._lock:
            self._save_thread(thread)
            self._connection.commit()
        return thread

    async def list_threads(self) -> list[ThreadRecord]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT data FROM threads ORDER BY updated_at DESC"
            ).fetchall()
        return [ThreadRecord.model_validate_json(row["data"]) for row in rows]

    async def get_thread(self, thread_id: UUID) -> ThreadDetail | None:
        async with self._lock:
            thread_row = self._connection.execute(
                "SELECT data FROM threads WHERE id = ?", (str(thread_id),)
            ).fetchone()
            if thread_row is None:
                return None
            run_rows = self._connection.execute(
                "SELECT data FROM runs WHERE thread_id = ? ORDER BY created_at",
                (str(thread_id),),
            ).fetchall()

        thread = ThreadRecord.model_validate_json(thread_row["data"])
        runs = [RunRecord.model_validate_json(row["data"]) for row in run_rows]
        return ThreadDetail(**thread.model_dump(), runs=runs)

    async def create_run(self, run: RunRecord) -> RunRecord:
        async with self._lock:
            self._save_run(run)
            row = self._connection.execute(
                "SELECT data FROM threads WHERE id = ?", (str(run.thread_id),)
            ).fetchone()
            if row is None:
                raise KeyError(run.thread_id)
            thread = ThreadRecord.model_validate_json(row["data"])
            thread.updated_at = datetime.now(timezone.utc)
            self._save_thread(thread)
            self._connection.commit()
        return run

    async def get_run(self, run_id: UUID) -> RunRecord | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT data FROM runs WHERE id = ?", (str(run_id),)
            ).fetchone()
        return RunRecord.model_validate_json(row["data"]) if row else None

    async def update_run(self, run_id: UUID, **changes) -> RunRecord:
        async with self._lock:
            row = self._connection.execute(
                "SELECT data FROM runs WHERE id = ?", (str(run_id),)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = RunRecord.model_validate_json(row["data"]).model_copy(update=changes)
            self._save_run(run)
            self._connection.commit()
        return run

    async def append_event(self, event: RunEvent) -> None:
        async with self._lock:
            self._connection.execute(
                "INSERT INTO events (id, run_id, sequence, data) VALUES (?, ?, ?, ?)",
                (
                    str(event.id),
                    str(event.run_id),
                    event.sequence,
                    event.model_dump_json(),
                ),
            )
            self._connection.commit()
        await self._fanout.publish(event)

    async def events_after(self, run_id: UUID, sequence: int) -> list[RunEvent]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT data FROM events WHERE run_id = ? AND sequence > ? "
                "ORDER BY sequence",
                (str(run_id), sequence),
            ).fetchall()
        return [RunEvent.model_validate_json(row["data"]) for row in rows]

    async def subscribe(self, run_id: UUID) -> asyncio.Queue[RunEvent]:
        return self._fanout.subscribe(run_id)

    async def unsubscribe(self, run_id: UUID, queue: asyncio.Queue[RunEvent]) -> None:
        self._fanout.unsubscribe(run_id, queue)

    async def cancel(self, run_id: UUID) -> bool:
        async with self._lock:
            row = self._connection.execute(
                "SELECT data FROM runs WHERE id = ?", (str(run_id),)
            ).fetchone()
            if row is None:
                return False
            run = RunRecord.model_validate_json(row["data"])
            if run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return False
            self._cancelled.add(run_id)
            return True

    async def is_cancelled(self, run_id: UUID) -> bool:
        async with self._lock:
            return run_id in self._cancelled

    async def add_feedback(self, feedback: FeedbackRecord) -> None:
        async with self._lock:
            self._connection.execute(
                "INSERT INTO feedback (id, run_id, created_at, data) VALUES (?, ?, ?, ?)",
                (
                    str(feedback.id),
                    str(feedback.run_id),
                    feedback.created_at.isoformat(),
                    feedback.model_dump_json(),
                ),
            )
            self._connection.commit()

    async def list_feedback(self, run_id: UUID) -> list[FeedbackRecord]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT data FROM feedback WHERE run_id = ? ORDER BY created_at",
                (str(run_id),),
            ).fetchall()
        return [FeedbackRecord.model_validate_json(row["data"]) for row in rows]

    async def create_api_key(self, key: ApiKeyRecord) -> ApiKeyRecord:
        async with self._lock:
            self._connection.execute(
                "INSERT INTO api_keys (id, hashed_key, created_at, data) VALUES (?, ?, ?, ?)",
                (
                    str(key.id),
                    key.hashed_key,
                    key.created_at.isoformat(),
                    key.model_dump_json(),
                ),
            )
            self._connection.commit()
        return key

    async def get_api_key_by_hash(self, hashed_key: str) -> ApiKeyRecord | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT data FROM api_keys WHERE hashed_key = ?", (hashed_key,)
            ).fetchone()
        return ApiKeyRecord.model_validate_json(row["data"]) if row else None

    async def list_api_keys(self) -> list[ApiKeyRecord]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT data FROM api_keys ORDER BY created_at"
            ).fetchall()
        return [ApiKeyRecord.model_validate_json(row["data"]) for row in rows]

    async def revoke_api_key(self, key_id: UUID) -> bool:
        async with self._lock:
            row = self._connection.execute(
                "SELECT data FROM api_keys WHERE id = ?", (str(key_id),)
            ).fetchone()
            if row is None:
                return False
            key = ApiKeyRecord.model_validate_json(row["data"]).model_copy(
                update={"revoked_at": datetime.now(timezone.utc)}
            )
            self._connection.execute(
                "UPDATE api_keys SET data = ? WHERE id = ?",
                (key.model_dump_json(), str(key_id)),
            )
            self._connection.commit()
            return True

    async def latest_event_sequence(self, run_id: UUID) -> int:
        async with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS sequence "
                "FROM events WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return int(row["sequence"])

    def _save_thread(self, thread: ThreadRecord) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO threads (id, updated_at, data) VALUES (?, ?, ?)",
            (str(thread.id), thread.updated_at.isoformat(), thread.model_dump_json()),
        )

    def _save_run(self, run: RunRecord) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO runs (id, thread_id, created_at, data) "
            "VALUES (?, ?, ?, ?)",
            (
                str(run.id),
                str(run.thread_id),
                run.created_at.isoformat(),
                run.model_dump_json(),
            ),
        )


class PostgresRunStore:
    """PostgreSQL-backed store for durable production research history."""

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS research_threads (
            id UUID PRIMARY KEY,
            updated_at TIMESTAMPTZ NOT NULL,
            data JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS research_runs (
            id UUID PRIMARY KEY,
            thread_id UUID NOT NULL REFERENCES research_threads(id)
                ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL,
            cancellation_requested BOOLEAN NOT NULL DEFAULT FALSE,
            data JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS research_runs_thread_created_idx
            ON research_runs(thread_id, created_at)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_run_events (
            id UUID PRIMARY KEY,
            run_id UUID NOT NULL REFERENCES research_runs(id)
                ON DELETE CASCADE,
            sequence INTEGER NOT NULL,
            data JSONB NOT NULL,
            UNIQUE (run_id, sequence)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS research_events_run_sequence_idx
            ON research_run_events(run_id, sequence)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_run_feedback (
            id UUID PRIMARY KEY,
            run_id UUID NOT NULL REFERENCES research_runs(id)
                ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL,
            data JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS research_feedback_run_id_idx
            ON research_run_feedback(run_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_api_keys (
            id UUID PRIMARY KEY,
            hashed_key TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL,
            data JSONB NOT NULL
        )
        """,
    )

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 30.0,
        redis_client: Any | None = None,
    ) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if min_size < 0 or max_size < 1 or min_size > max_size:
            raise ValueError("Invalid PostgreSQL pool size configuration")

        self._pool = AsyncConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            open=False,
        )
        self._fanout = EventFanout(redis_client)

    async def initialize(self) -> None:
        await self._pool.open(wait=True)
        async with self._pool.connection() as connection:
            for statement in self._SCHEMA:
                await connection.execute(statement)
        await self._fanout.start()

    async def close(self) -> None:
        await self._fanout.stop()
        await self._pool.close()

    async def healthcheck(self) -> bool:
        try:
            async with self._pool.connection() as connection:
                await connection.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def create_thread(self, thread: ThreadRecord) -> ThreadRecord:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO research_threads (id, updated_at, data)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    data = EXCLUDED.data
                """,
                (thread.id, thread.updated_at, _json(thread)),
            )
        return thread

    async def list_threads(self) -> list[ThreadRecord]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT data FROM research_threads ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
        return [ThreadRecord.model_validate(row[0]) for row in rows]

    async def get_thread(self, thread_id: UUID) -> ThreadDetail | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT data FROM research_threads WHERE id = %s",
                (thread_id,),
            )
            thread_row = await cursor.fetchone()
            if thread_row is None:
                return None

            cursor = await connection.execute(
                """
                SELECT data FROM research_runs
                WHERE thread_id = %s
                ORDER BY created_at
                """,
                (thread_id,),
            )
            run_rows = await cursor.fetchall()

        thread = ThreadRecord.model_validate(thread_row[0])
        runs = [RunRecord.model_validate(row[0]) for row in run_rows]
        return ThreadDetail(**thread.model_dump(), runs=runs)

    async def create_run(self, run: RunRecord) -> RunRecord:
        updated_at = datetime.now(timezone.utc)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT data FROM research_threads WHERE id = %s FOR UPDATE",
                    (run.thread_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise KeyError(run.thread_id)

                thread = ThreadRecord.model_validate(row[0])
                thread.updated_at = updated_at
                await connection.execute(
                    """
                    UPDATE research_threads
                    SET updated_at = %s, data = %s
                    WHERE id = %s
                    """,
                    (updated_at, _json(thread), run.thread_id),
                )
                await connection.execute(
                    """
                    INSERT INTO research_runs (id, thread_id, created_at, data)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                    """,
                    (run.id, run.thread_id, run.created_at, _json(run)),
                )
        return run

    async def get_run(self, run_id: UUID) -> RunRecord | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT data FROM research_runs WHERE id = %s",
                (run_id,),
            )
            row = await cursor.fetchone()
        return RunRecord.model_validate(row[0]) if row else None

    async def update_run(self, run_id: UUID, **changes) -> RunRecord:
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT data FROM research_runs WHERE id = %s FOR UPDATE",
                    (run_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise KeyError(run_id)

                run = RunRecord.model_validate(row[0]).model_copy(update=changes)
                await connection.execute(
                    "UPDATE research_runs SET data = %s WHERE id = %s",
                    (_json(run), run_id),
                )
        return run

    async def append_event(self, event: RunEvent) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO research_run_events (id, run_id, sequence, data)
                VALUES (%s, %s, %s, %s)
                """,
                (event.id, event.run_id, event.sequence, _json(event)),
            )

        await self._fanout.publish(event)

    async def events_after(self, run_id: UUID, sequence: int) -> list[RunEvent]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT data FROM research_run_events
                WHERE run_id = %s AND sequence > %s
                ORDER BY sequence
                """,
                (run_id, sequence),
            )
            rows = await cursor.fetchall()
        return [RunEvent.model_validate(row[0]) for row in rows]

    async def subscribe(self, run_id: UUID) -> asyncio.Queue[RunEvent]:
        return self._fanout.subscribe(run_id)

    async def unsubscribe(self, run_id: UUID, queue: asyncio.Queue[RunEvent]) -> None:
        self._fanout.unsubscribe(run_id, queue)

    async def cancel(self, run_id: UUID) -> bool:
        terminal_statuses = [
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        ]
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                UPDATE research_runs
                SET cancellation_requested = TRUE
                WHERE id = %s AND NOT (data->>'status' = ANY(%s))
                RETURNING id
                """,
                (run_id, terminal_statuses),
            )
            return await cursor.fetchone() is not None

    async def is_cancelled(self, run_id: UUID) -> bool:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT cancellation_requested
                FROM research_runs WHERE id = %s
                """,
                (run_id,),
            )
            row = await cursor.fetchone()
        return bool(row and row[0])

    async def latest_event_sequence(self, run_id: UUID) -> int:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM research_run_events WHERE run_id = %s
                """,
                (run_id,),
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def add_feedback(self, feedback: FeedbackRecord) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO research_run_feedback (id, run_id, created_at, data)
                VALUES (%s, %s, %s, %s)
                """,
                (feedback.id, feedback.run_id, feedback.created_at, _json(feedback)),
            )

    async def list_feedback(self, run_id: UUID) -> list[FeedbackRecord]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT data FROM research_run_feedback
                WHERE run_id = %s ORDER BY created_at
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [FeedbackRecord.model_validate(row[0]) for row in rows]

    async def create_api_key(self, key: ApiKeyRecord) -> ApiKeyRecord:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO research_api_keys (id, hashed_key, created_at, data)
                VALUES (%s, %s, %s, %s)
                """,
                (key.id, key.hashed_key, key.created_at, _json(key)),
            )
        return key

    async def get_api_key_by_hash(self, hashed_key: str) -> ApiKeyRecord | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT data FROM research_api_keys WHERE hashed_key = %s",
                (hashed_key,),
            )
            row = await cursor.fetchone()
        return ApiKeyRecord.model_validate(row[0]) if row else None

    async def list_api_keys(self) -> list[ApiKeyRecord]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT data FROM research_api_keys ORDER BY created_at"
            )
            rows = await cursor.fetchall()
        return [ApiKeyRecord.model_validate(row[0]) for row in rows]

    async def revoke_api_key(self, key_id: UUID) -> bool:
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT data FROM research_api_keys WHERE id = %s FOR UPDATE",
                    (key_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    return False
                key = ApiKeyRecord.model_validate(row[0]).model_copy(
                    update={"revoked_at": datetime.now(timezone.utc)}
                )
                await connection.execute(
                    "UPDATE research_api_keys SET data = %s WHERE id = %s",
                    (_json(key), key_id),
                )
            return True


def _json(record: ThreadRecord | RunRecord | RunEvent | FeedbackRecord | ApiKeyRecord) -> Jsonb:
    return Jsonb(record.model_dump(mode="json"))


def create_run_store(
    *,
    database_url: str | None,
    sqlite_path: Path,
    postgres_min_size: int = 1,
    postgres_max_size: int = 10,
    postgres_timeout: float = 30.0,
    redis_client: Any | None = None,
) -> RunStore:
    if database_url:
        return PostgresRunStore(
            database_url,
            min_size=postgres_min_size,
            max_size=postgres_max_size,
            timeout=postgres_timeout,
            redis_client=redis_client,
        )
    return SqliteRunStore(sqlite_path, redis_client=redis_client)
