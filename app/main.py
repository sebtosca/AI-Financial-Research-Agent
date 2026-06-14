from contextlib import asynccontextmanager

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
    RUN_STORE_PATH,
)


def create_app(store: RunStore | None = None) -> FastAPI:
    selected_store = store or create_run_store(
        database_url=DATABASE_URL,
        sqlite_path=RUN_STORE_PATH,
        postgres_min_size=DATABASE_POOL_MIN_SIZE,
        postgres_max_size=DATABASE_POOL_MAX_SIZE,
        postgres_timeout=DATABASE_POOL_TIMEOUT,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await selected_store.initialize()
        try:
            yield
        finally:
            await selected_store.close()

    app = FastAPI(
        title=APP_NAME,
        version="1.0.0",
        debug=DEBUG,
        lifespan=lifespan,
    )
    app.state.run_service = AgentRunService(selected_store)
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
