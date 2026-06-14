import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .metrics import SSE_CONNECTIONS
from .schemas import (
    EventType,
    FeedbackCreate,
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


router = APIRouter(prefix="/api/v1")

TERMINAL_EVENT_TYPES = {
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
    EventType.RUN_CANCELLED,
}


def _service(request: Request) -> AgentRunService:
    return request.app.state.run_service


@router.post("/threads", response_model=ThreadRecord, status_code=201)
async def create_thread(payload: ThreadCreate, request: Request) -> ThreadRecord:
    thread = ThreadRecord(title=payload.title or "New financial research")
    return await _service(request).store.create_thread(thread)


@router.get("/threads", response_model=list[ThreadRecord])
async def list_threads(request: Request) -> list[ThreadRecord]:
    return await _service(request).store.list_threads()


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
async def get_thread(thread_id: UUID, request: Request) -> ThreadDetail:
    thread = await _service(request).store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.post("/threads/{thread_id}/runs", response_model=RunRecord, status_code=202)
async def create_run(
    thread_id: UUID,
    payload: RunCreate,
    request: Request,
    background_tasks: BackgroundTasks,
) -> RunRecord:
    service = _service(request)
    if await service.store.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    run = RunRecord(
        thread_id=thread_id,
        query=payload.query.strip(),
        with_rag=payload.with_rag,
    )
    await service.store.create_run(run)
    await service.emit(run.id, EventType.RUN_CREATED)
    background_tasks.add_task(service.execute, run.id)
    return run


@router.get("/runs/{run_id}", response_model=RunRecord)
async def get_run(run_id: UUID, request: Request) -> RunRecord:
    run = await _service(request).store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/runs/{run_id}/cancel", response_model=RunRecord)
async def cancel_run(run_id: UUID, request: Request) -> RunRecord:
    service = _service(request)
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
) -> None:
    if await _service(request).store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _ = payload


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
) -> StreamingResponse:
    service = _service(request)
    if await service.store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")

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


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok", service="financial-research-api")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    if not await _service(request).store.healthcheck():
        raise HTTPException(status_code=503, detail="Run store unavailable")
    return HealthResponse(status="ready", service="financial-research-api")
