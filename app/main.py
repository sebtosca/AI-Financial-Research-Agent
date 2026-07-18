from contextlib import asynccontextmanager

import redis.asyncio as redis_asyncio
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.api.routes import router
from app.api.service import AgentRunService
from app.api.store import RunStore, create_run_store
from app.config import (
    APP_CORS_ORIGINS,
    APP_NAME,
    DATABASE_POOL_MAX_SIZE,
    DATABASE_POOL_MIN_SIZE,
    DATABASE_POOL_TIMEOUT,
    DATABASE_URL,
    DEBUG,
    REDIS_URL,
    RUN_STORE_PATH,
)


def create_app(store: RunStore | None = None) -> FastAPI:
    redis_client = redis_asyncio.from_url(REDIS_URL) if REDIS_URL else None
    selected_store = store or create_run_store(
        database_url=DATABASE_URL,
        sqlite_path=RUN_STORE_PATH,
        postgres_min_size=DATABASE_POOL_MIN_SIZE,
        postgres_max_size=DATABASE_POOL_MAX_SIZE,
        postgres_timeout=DATABASE_POOL_TIMEOUT,
        redis_client=redis_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await selected_store.initialize()
        arq_pool = await create_pool(RedisSettings.from_dsn(REDIS_URL)) if REDIS_URL else None
        app.state.arq_pool = arq_pool
        try:
            yield
        finally:
            if arq_pool is not None:
                await arq_pool.close()
            await selected_store.close()
            if redis_client is not None:
                await redis_client.close()

    app = FastAPI(
        title=APP_NAME,
        version="1.0.0",
        debug=DEBUG,
        lifespan=lifespan,
    )
    app.state.run_service = AgentRunService(selected_store)
    app.state.arq_pool = None  # set for real in lifespan(); default covers tests that skip lifespan
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(APP_CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.mount("/metrics", make_asgi_app())

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": APP_NAME, "docs": "/docs"}

    return app


app = create_app()
