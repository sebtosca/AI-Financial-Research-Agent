import asyncio
from functools import partial
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.observability.tracing import submit_feedback

from .auth import generate_api_key, hash_api_key, is_authorized_owner, require_admin, require_api_key
from .metrics import SSE_CONNECTIONS
from .schemas import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyPublic,
    ApiKeyRecord,
    EventType,
    FeedbackCreate,
    FeedbackRecord,
    HealthResponse,
    RunCreate,
    RunEvent,
    RunRecord,
    RunStatus,
    ThreadCreate,
    ThreadDetail,
    ThreadRecord,
)
from .service import AgentRunService, encode_sse


health_router = APIRouter(prefix="/api/v1")
router = APIRouter(prefix="/api/v1")
admin_router = APIRouter(prefix="/api/v1/admin")

TERMINAL_EVENT_TYPES = {
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
    EventType.RUN_CANCELLED,
}


def _service(request: Request) -> AgentRunService:
    return request.app.state.run_service


def _authorize_thread(thread: ThreadRecord, key: ApiKeyRecord) -> None:
    if not is_authorized_owner(key, thread.owner_key_id):
        raise HTTPException(status_code=404, detail="Thread not found")


def _authorize_run(run: RunRecord, key: ApiKeyRecord) -> None:
    if not is_authorized_owner(key, run.owner_key_id):
        raise HTTPException(status_code=404, detail="Run not found")


@router.post("/threads", response_model=ThreadRecord, status_code=201)
async def create_thread(
    payload: ThreadCreate,
    request: Request,
    key: ApiKeyRecord = Depends(require_api_key),
) -> ThreadRecord:
    thread = ThreadRecord(
        title=payload.title or "New financial research",
        owner_key_id=key.id,
    )
    return await _service(request).store.create_thread(thread)


@router.get("/threads", response_model=list[ThreadRecord])
async def list_threads(
    request: Request,
    key: ApiKeyRecord = Depends(require_api_key),
) -> list[ThreadRecord]:
    threads = await _service(request).store.list_threads()
    return [thread for thread in threads if is_authorized_owner(key, thread.owner_key_id)]


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
async def get_thread(
    thread_id: UUID,
    request: Request,
    key: ApiKeyRecord = Depends(require_api_key),
) -> ThreadDetail:
    thread = await _service(request).store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    _authorize_thread(thread, key)
    return thread


@router.post("/threads/{thread_id}/runs", response_model=RunRecord, status_code=202)
async def create_run(
    thread_id: UUID,
    payload: RunCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    key: ApiKeyRecord = Depends(require_api_key),
) -> RunRecord:
    service = _service(request)
    thread = await service.store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    _authorize_thread(thread, key)

    run = RunRecord(
        thread_id=thread_id,
        query=payload.query.strip(),
        with_rag=payload.with_rag,
        owner_key_id=key.id,
    )
    await service.store.create_run(run)
    await service.emit(run.id, EventType.RUN_CREATED)

    arq_pool = request.app.state.arq_pool
    if arq_pool is not None:
        await arq_pool.enqueue_job("execute_run_task", str(run.id))
    else:
        background_tasks.add_task(service.execute, run.id)

    return run


@router.get("/runs/{run_id}", response_model=RunRecord)
async def get_run(
    run_id: UUID,
    request: Request,
    key: ApiKeyRecord = Depends(require_api_key),
) -> RunRecord:
    run = await _service(request).store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _authorize_run(run, key)
    return run


@router.post("/runs/{run_id}/cancel", response_model=RunRecord)
async def cancel_run(
    run_id: UUID,
    request: Request,
    key: ApiKeyRecord = Depends(require_api_key),
) -> RunRecord:
    service = _service(request)
    existing = await service.store.get_run(run_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _authorize_run(existing, key)

    if not await service.store.cancel(run_id):
        raise HTTPException(status_code=409, detail="Run cannot be cancelled")
    run = await service.store.get_run(run_id)
    assert run is not None
    return run


@router.post("/runs/{run_id}/feedback", status_code=204)
async def create_feedback(
    run_id: UUID,
    payload: FeedbackCreate,
    request: Request,
    key: ApiKeyRecord = Depends(require_api_key),
) -> None:
    service = _service(request)
    run = await service.store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _authorize_run(run, key)

    feedback = FeedbackRecord(run_id=run_id, rating=payload.rating, comment=payload.comment)
    await service.store.add_feedback(feedback)

    if run.langsmith_run_id:
        await asyncio.to_thread(
            partial(
                submit_feedback,
                run.langsmith_run_id,
                rating=payload.rating,
                comment=payload.comment,
            )
        )


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
    key: ApiKeyRecord = Depends(require_api_key),
) -> StreamingResponse:
    service = _service(request)
    run = await service.store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _authorize_run(run, key)

    async def event_stream():
        queue = await service.store.subscribe(run_id)
        SSE_CONNECTIONS.inc()
        last_sequence = after
        try:
            for event in await service.store.events_after(run_id, after):
                yield encode_sse(event)
                last_sequence = event.sequence
                if event.type in TERMINAL_EVENT_TYPES:
                    return

            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    if event.sequence <= last_sequence:
                        continue
                    yield encode_sse(event)
                    last_sequence = event.sequence
                    if event.type in TERMINAL_EVENT_TYPES:
                        break
                except TimeoutError:
                    heartbeat = RunEvent(
                        run_id=run_id,
                        sequence=0,
                        type=EventType.HEARTBEAT,
                    )
                    yield encode_sse(heartbeat)
        finally:
            SSE_CONNECTIONS.dec()
            await service.store.unsubscribe(run_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@health_router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok", service="financial-research-api")


@health_router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    if not await _service(request).store.healthcheck():
        raise HTTPException(status_code=503, detail="Run store unavailable")
    return HealthResponse(status="ready", service="financial-research-api")


@admin_router.post("/api-keys", response_model=ApiKeyCreated, status_code=201)
async def create_api_key(
    payload: ApiKeyCreate,
    request: Request,
    _admin: ApiKeyRecord = Depends(require_admin),
) -> ApiKeyCreated:
    raw_key = generate_api_key()
    record = ApiKeyRecord(
        hashed_key=hash_api_key(raw_key),
        label=payload.label,
        role=payload.role,
    )
    await _service(request).store.create_api_key(record)
    return ApiKeyCreated(
        id=record.id,
        label=record.label,
        role=record.role,
        created_at=record.created_at,
        api_key=raw_key,
    )


@admin_router.get("/api-keys", response_model=list[ApiKeyPublic])
async def list_api_keys(
    request: Request,
    _admin: ApiKeyRecord = Depends(require_admin),
) -> list[ApiKeyPublic]:
    keys = await _service(request).store.list_api_keys()
    return [
        ApiKeyPublic(
            id=key.id,
            label=key.label,
            role=key.role,
            created_at=key.created_at,
            revoked_at=key.revoked_at,
        )
        for key in keys
    ]


@admin_router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: UUID,
    request: Request,
    _admin: ApiKeyRecord = Depends(require_admin),
) -> None:
    if not await _service(request).store.revoke_api_key(key_id):
        raise HTTPException(status_code=404, detail="API key not found")
