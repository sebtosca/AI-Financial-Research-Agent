import asyncio
import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.graph import create_financial_agent, tools_from_names
from app.observability.tracing import build_trace_context, build_tracer, extract_run_id
from app.providers.pricing import estimate_cost_usd
from app.routing.policy import classify_query

from .metrics import (
    ACTIVE_RUNS,
    LLM_COST_USD_TOTAL,
    LLM_TOKENS_TOTAL,
    ROUTING_DECISIONS_TOTAL,
    RUN_DURATION,
    RUNS_TOTAL,
    TOOL_CALL_DURATION,
    TOOL_CALLS,
)
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
        self._tool_call_started: dict[str, float] = {}
        self._token_usage: dict[UUID, dict[str, int]] = {}

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
            decision = classify_query(run.query, with_rag_requested=run.with_rag)
            ROUTING_DECISIONS_TOTAL.labels(model_tier=decision.model_tier.value).inc()
            self._token_usage[run_id] = {"prompt_tokens": 0, "completion_tokens": 0}
            await self.store.update_run(
                run_id,
                model_tier=decision.model_tier.value,
                provider=decision.provider,
                model_name=decision.model_name,
                tool_subset=list(decision.tool_names),
                rag_engaged=decision.rag_engaged,
            )
            await self.emit(
                run_id,
                EventType.ROUTING_DECIDED,
                AgentStage.ROUTING,
                {
                    "model_tier": decision.model_tier.value,
                    "provider": decision.provider,
                    "model_name": decision.model_name,
                    "tools": list(decision.tool_names),
                    "rag_engaged": decision.rag_engaged,
                    "matched_rules": list(decision.matched_rules),
                },
            )

            agent = create_financial_agent(
                agent_type="full",
                with_memory=True,
                tools=tools_from_names(decision.tool_names),
                model_tier=decision.model_tier.value,
            )
            trace_context = build_trace_context(run_id, run.thread_id, decision)
            tracer = build_tracer(trace_context)
            stream_config = {
                "configurable": {"thread_id": str(run.thread_id)},
                "tags": trace_context.tags,
                "metadata": trace_context.metadata,
                "run_name": trace_context.run_name,
            }
            if tracer is not None:
                stream_config["callbacks"] = [tracer]

            stream = agent.stream(
                {"messages": [HumanMessage(content=run.query)]},
                config=stream_config,
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

                update_report = await self._process_update(run_id, update, decision.model_name)
                if update_report is not None:
                    report = update_report

            usage = self._token_usage.get(run_id, {"prompt_tokens": 0, "completion_tokens": 0})
            estimated_cost_usd = estimate_cost_usd(
                decision.model_name, usage["prompt_tokens"], usage["completion_tokens"]
            )
            completed_at = datetime.now(timezone.utc)
            await self.store.update_run(
                run_id,
                status=RunStatus.COMPLETED,
                report=report,
                completed_at=completed_at,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                estimated_cost_usd=estimated_cost_usd,
                langsmith_run_id=extract_run_id(tracer),
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
            self._token_usage.pop(run_id, None)
            ACTIVE_RUNS.dec()
            RUN_DURATION.observe(time.perf_counter() - started)

    async def _process_update(
        self, run_id: UUID, update: dict, model_name: str = "unknown"
    ) -> str | None:
        report = None

        for message in self._iter_messages(update):
            message_report = await self._process_message(run_id, message, model_name)
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
        model_name: str = "unknown",
    ) -> str | None:
        if isinstance(message, ToolMessage):
            await self._emit_tool_completed(run_id, message)
            return None

        if not isinstance(message, AIMessage):
            return None

        self._record_token_usage(run_id, message, model_name)

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

    def _record_token_usage(self, run_id: UUID, message: AIMessage, model_name: str) -> None:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            return

        prompt_tokens = usage.get("input_tokens", 0) or 0
        completion_tokens = usage.get("output_tokens", 0) or 0

        totals = self._token_usage.setdefault(
            run_id, {"prompt_tokens": 0, "completion_tokens": 0}
        )
        totals["prompt_tokens"] += prompt_tokens
        totals["completion_tokens"] += completion_tokens

        LLM_TOKENS_TOTAL.labels(model=model_name, direction="prompt").inc(prompt_tokens)
        LLM_TOKENS_TOTAL.labels(model=model_name, direction="completion").inc(completion_tokens)
        LLM_COST_USD_TOTAL.labels(model=model_name).inc(
            estimate_cost_usd(model_name, prompt_tokens, completion_tokens)
        )

    async def _emit_tool_starts(self, run_id: UUID, message: AIMessage) -> None:
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                self._tool_call_started[tool_call_id] = time.perf_counter()
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
        started = self._tool_call_started.pop(message.tool_call_id, None)
        if started is not None:
            TOOL_CALL_DURATION.labels(tool=tool_name).observe(time.perf_counter() - started)
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
