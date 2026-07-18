"""arq worker: executes agent runs enqueued by the API process.

Runs the exact same AgentRunService.execute() coroutine the API used to
run in-process via BackgroundTasks -- moving it here means a queued run
survives an API process restart, and can run on separate worker
containers from the API itself.
"""

import logging
from uuid import UUID

import redis.asyncio as redis_asyncio
from arq.connections import RedisSettings

from app.api.service import AgentRunService
from app.api.store import RunStore, create_run_store
from app.config import (
    DATABASE_POOL_MAX_SIZE,
    DATABASE_POOL_MIN_SIZE,
    DATABASE_POOL_TIMEOUT,
    DATABASE_URL,
    REDIS_URL,
    RUN_STORE_PATH,
    WORKER_CONCURRENCY,
)

logger = logging.getLogger(__name__)


async def execute_run_task(ctx: dict, run_id: str) -> None:
    store: RunStore = ctx["run_store"]
    service = AgentRunService(store)
    await service.execute(UUID(run_id))


async def startup(ctx: dict) -> None:
    redis_client = redis_asyncio.from_url(REDIS_URL) if REDIS_URL else None
    store = create_run_store(
        database_url=DATABASE_URL,
        sqlite_path=RUN_STORE_PATH,
        postgres_min_size=DATABASE_POOL_MIN_SIZE,
        postgres_max_size=DATABASE_POOL_MAX_SIZE,
        postgres_timeout=DATABASE_POOL_TIMEOUT,
        redis_client=redis_client,
    )
    await store.initialize()
    ctx["run_store"] = store
    ctx["redis_client"] = redis_client
    logger.info("Worker started | store=%s", type(store).__name__)


async def shutdown(ctx: dict) -> None:
    store = ctx.get("run_store")
    if store is not None:
        await store.close()
    redis_client = ctx.get("redis_client")
    if redis_client is not None:
        await redis_client.close()


class WorkerSettings:
    functions = [execute_run_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = (
        RedisSettings.from_dsn(REDIS_URL) if REDIS_URL else RedisSettings()
    )
    max_jobs = WORKER_CONCURRENCY
