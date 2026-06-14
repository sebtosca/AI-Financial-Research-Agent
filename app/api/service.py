import asyncio
import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.graph import create_enhanced_financial_agent

from .metrics import ACTIVE_RUNS, RUN_DURATION, RUNS_TOTAL, TOOL_CALLS
from .schemas import (
    AgentStage,
    EventType,
    RunEvent,
    RunRecord,
    RunStatus,
)
from .store import RunStore


logger = logging.getLogger(__name__)

TOOL_STAGES = {
    "get_stock_price": AgentStage.STOCK_PRICE,
    "get_stock_history": AgentStage.STOCK_HISTORY,
    "search_financial_news": AgentStage.NEWS_SEARCH,
    "analyze_sentiment": AgentStage.SENTIMENT,
    "query_private_database": AgentStage.PRIVATE_RAG,
}


def _safe_tool_args(args: dict) -> dict:
    allowed = {"ticker", "period", "query"}
    return {key: value for key, value in args.items() if key in allowed}


class AgentRunService:
    def __init__(self, store: RunStore) -> None:
        self.store = store
        self._sequences: dict[UUID, int] = {}

    async def emit(
        self,
        run_id: UUID,
        event_type: EventType,
        stage: AgentStage | None = None,
        payload: dict | None = None,
    ) -> RunEvent:
        previous = self._sequences.get(run_id)
        if previous is None:
            previous = await self.store.latest_event_sequence(run_id)
        sequence = previous + 1
        self._sequences[run_id] = sequence
        event = RunEvent(
            run_id=run_id,
            sequence=sequence,
            type=event_type,
            stage=stage,
            payload=payload or {},
        )
        await self.store.append_event(event)
        return event

    async def execute(self, run_id: UUID) -> None:
        run = await self.store.get_run(run_id)
        if run is None:
            return

        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        ACTIVE_RUNS.inc()
        await self.store.update_run(
            run_id,
            status=RunStatus.RUNNING,
            started_at=started_at,
        )
        await self.emit(run_id, EventType.RUN_STARTED, AgentStage.PLANNING)

        try:
            agent = create_enhanced_financial_agent(
                with_rag=run.with_rag,
                with_memory=True,
            )
            stream = agent.stream(
                {"messages": [HumanMessage(content=run.query)]},
                config={"configurable": {"thread_id": str(run.thread_id)}},
                stream_mode="updates",
            )
            report = ""

            while True:
                if await self.store.is_cancelled(run_id):
                    await self._mark_cancelled(run_id)
                    return

                update = await asyncio.to_thread(next, stream, None)
                if update is None:
                    break

                update_report = await self._process_update(run_id, update)
                if update_report is not None:
                    report = update_report

            completed_at = datetime.now(timezone.utc)
            await self.store.update_run(
                run_id,
                status=RunStatus.COMPLETED,
                report=report,
                completed_at=completed_at,
            )
            await self.emit(
                run_id,
                EventType.RUN_COMPLETED,
                AgentStage.SYNTHESIS,
                {"report": report},
            )
            RUNS_TOTAL.labels(status="completed").inc()
        except Exception:
            logger.exception("Agent run failed | run_id=%s", run_id)
            await self.store.update_run(
                run_id,
                status=RunStatus.FAILED,
                error="The research run failed. Please retry.",
                completed_at=datetime.now(timezone.utc),
            )
            await self.emit(
                run_id,
                EventType.RUN_FAILED,
                payload={"message": "The research run failed. Please retry."},
            )
            RUNS_TOTAL.labels(status="failed").inc()
        finally:
            ACTIVE_RUNS.dec()
            RUN_DURATION.observe(time.perf_counter() - started)

    async def _process_update(self, run_id: UUID, update: dict) -> str | None:
        report = None

        for message in self._iter_messages(update):
            message_report = await self._process_message(run_id, message)
            if message_report is not None:
                report = message_report

        return report

    @staticmethod
    def _iter_messages(update: dict) -> Iterable[BaseMessage]:
        for node_update in update.values():
            if not isinstance(node_update, dict):
                continue

            yield from node_update.get("messages", [])

    async def _process_message(
        self,
        run_id: UUID,
        message: BaseMessage,
    ) -> str | None:
        if isinstance(message, ToolMessage):
            await self._emit_tool_completed(run_id, message)
            return None

        if not isinstance(message, AIMessage):
            return None

        if message.tool_calls:
            await self._emit_tool_starts(run_id, message)
            return None

        if not message.content:
            return None

        report = str(message.content)
        await self.emit(
            run_id,
            EventType.REPORT_DELTA,
            AgentStage.SYNTHESIS,
            {"content": report},
        )
        return report

    async def _emit_tool_starts(self, run_id: UUID, message: AIMessage) -> None:
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name", "unknown")
            await self.emit(
                run_id,
                EventType.TOOL_STARTED,
                TOOL_STAGES.get(tool_name, AgentStage.PLANNING),
                {
                    "tool": tool_name,
                    "input": _safe_tool_args(tool_call.get("args", {})),
                },
            )

    async def _emit_tool_completed(
        self,
        run_id: UUID,
        message: ToolMessage,
    ) -> None:
        tool_name = message.name or "unknown"
        failed = message.status == "error"
        status = "failed" if failed else "success"
        TOOL_CALLS.labels(tool=tool_name, status=status).inc()
        await self.emit(
            run_id,
            EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED,
            TOOL_STAGES.get(tool_name, AgentStage.PLANNING),
            {
                "tool": tool_name,
                **({"message": str(message.content)} if failed else {}),
            },
        )

    async def _mark_cancelled(self, run_id: UUID) -> None:
        await self.store.update_run(
            run_id,
            status=RunStatus.CANCELLED,
            completed_at=datetime.now(timezone.utc),
        )
        await self.emit(run_id, EventType.RUN_CANCELLED)
        RUNS_TOTAL.labels(status="cancelled").inc()


def encode_sse(event: RunEvent) -> str:
    data = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.type.value}\ndata: {data}\n\n"
