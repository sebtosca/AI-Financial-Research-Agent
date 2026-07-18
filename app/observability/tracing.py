"""Explicit LangSmith trace tagging and feedback wiring.

Makes LangSmith an explicit, first-class part of run lifecycle
observability instead of relying solely on global env-var auto-
instrumentation: every run is tagged/named by its routing decision, its
root trace id is captured synchronously, and user feedback can be
forwarded to that trace.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import UUID

from app.config import LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2, LANGSMITH_FEEDBACK_ENABLED
from app.routing.policy import RoutingDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceContext:
    tags: list[str]
    metadata: dict[str, Any]
    run_name: str


def build_trace_context(run_id: UUID, thread_id: UUID, decision: RoutingDecision) -> TraceContext:
    return TraceContext(
        tags=[
            f"model_tier:{decision.model_tier.value}",
            f"provider:{decision.provider}",
            f"rag_engaged:{decision.rag_engaged}",
        ],
        metadata={
            "run_id": str(run_id),
            "thread_id": str(thread_id),
            "matched_rules": list(decision.matched_rules),
        },
        run_name=f"argent-run-{run_id}",
    )


def build_tracer(trace_context: TraceContext):
    """Build a LangChainTracer for this run, or None when tracing is disabled."""

    if not LANGCHAIN_TRACING_V2:
        return None

    from langchain_core.tracers import LangChainTracer

    return LangChainTracer(
        project_name=LANGCHAIN_PROJECT,
        tags=trace_context.tags,
        metadata=trace_context.metadata,
    )


def extract_run_id(tracer: Any | None) -> str | None:
    """Read the LangSmith root run id off a tracer after its run has completed."""

    if tracer is None:
        return None

    latest_run = getattr(tracer, "latest_run", None)
    if latest_run is None:
        return None

    return str(latest_run.id)


@lru_cache(maxsize=1)
def _client():
    from langsmith import Client

    return Client()


def submit_feedback(langsmith_run_id: str, *, rating: int, comment: str | None) -> None:
    """Forward user feedback to a LangSmith trace. Logs and swallows failures --
    local persistence (app/api/store.py) is the source of truth, LangSmith is
    best-effort."""

    if not LANGSMITH_FEEDBACK_ENABLED:
        return

    try:
        _client().create_feedback(
            run_id=langsmith_run_id,
            key="user_rating",
            score=rating,
            comment=comment,
        )
    except Exception:
        logger.exception(
            "Failed to submit LangSmith feedback | langsmith_run_id=%s", langsmith_run_id
        )
