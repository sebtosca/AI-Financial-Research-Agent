from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStage(str, Enum):
    PLANNING = "planning"
    STOCK_PRICE = "stock_price"
    STOCK_HISTORY = "stock_history"
    NEWS_SEARCH = "news_search"
    SENTIMENT = "sentiment"
    PRIVATE_RAG = "private_rag"
    SYNTHESIS = "synthesis"


class EventType(str, Enum):
    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    STAGE_STARTED = "stage.started"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    REPORT_DELTA = "report.delta"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    HEARTBEAT = "heartbeat"


class ThreadCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ThreadRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RunCreate(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    with_rag: bool = True


class RunRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    thread_id: UUID
    query: str
    with_rag: bool
    status: RunStatus = RunStatus.QUEUED
    report: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    sequence: int
    type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    stage: AgentStage | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ThreadDetail(ThreadRecord):
    runs: list[RunRecord] = Field(default_factory=list)


class FeedbackCreate(BaseModel):
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=2000)


class HealthResponse(BaseModel):
    status: str
    service: str
