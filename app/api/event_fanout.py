"""Delivers RunEvents to local SSE subscribers, optionally through Redis
pub/sub so events reach subscribers connected to a different process than
the one executing the run (e.g. an API replica vs. the worker).

When no Redis client is configured, events are delivered directly to local
subscribers (today's single-process behavior). When one is configured,
publish() always goes through Redis and the same process's own listener
delivers it back to local subscribers -- this avoids double-delivery and
keeps the code path identical regardless of whether the event originated
locally or in another process.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from .schemas import RunEvent

logger = logging.getLogger(__name__)

_CHANNEL_PATTERN = "run-events:*"
_CHANNEL_PREFIX = "run-events:"


class EventFanout:
    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client
        self._subscribers: dict[UUID, set[asyncio.Queue[RunEvent]]] = defaultdict(set)
        self._pubsub: Any | None = None
        self._listener_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._redis is None:
            return

        self._pubsub = self._redis.pubsub()
        await self._pubsub.psubscribe(_CHANNEL_PATTERN)
        self._listener_task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        if self._pubsub is not None:
            await self._pubsub.aclose()
            self._pubsub = None

    async def _listen(self) -> None:
        assert self._pubsub is not None
        try:
            async for message in self._pubsub.listen():
                if message.get("type") != "pmessage":
                    continue
                try:
                    event = RunEvent.model_validate_json(message["data"])
                except Exception:
                    logger.exception("Failed to parse run event from Redis pub/sub")
                    continue
                await self._deliver_local(event)
        except asyncio.CancelledError:
            pass

    async def publish(self, event: RunEvent) -> None:
        if self._redis is not None:
            await self._redis.publish(
                f"{_CHANNEL_PREFIX}{event.run_id}", event.model_dump_json()
            )
        else:
            await self._deliver_local(event)

    async def _deliver_local(self, event: RunEvent) -> None:
        for queue in list(self._subscribers[event.run_id]):
            await queue.put(event)

    def subscribe(self, run_id: UUID) -> asyncio.Queue[RunEvent]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        self._subscribers[run_id].add(queue)
        return queue

    def unsubscribe(self, run_id: UUID, queue: asyncio.Queue[RunEvent]) -> None:
        self._subscribers[run_id].discard(queue)
