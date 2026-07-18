import asyncio
from uuid import uuid4

import fakeredis.aioredis as fakeredis_asyncio
import pytest

from app.api.event_fanout import EventFanout
from app.api.schemas import EventType, RunEvent


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_local_delivery_without_redis():
    fanout = EventFanout(redis_client=None)
    run_id = uuid4()
    queue = fanout.subscribe(run_id)
    event = RunEvent(run_id=run_id, sequence=1, type=EventType.RUN_STARTED)

    await fanout.publish(event)

    delivered = await asyncio.wait_for(queue.get(), timeout=1)
    assert delivered == event


@pytest.mark.anyio
async def test_unsubscribe_stops_delivery():
    fanout = EventFanout(redis_client=None)
    run_id = uuid4()
    queue = fanout.subscribe(run_id)
    fanout.unsubscribe(run_id, queue)

    await fanout.publish(RunEvent(run_id=run_id, sequence=1, type=EventType.RUN_STARTED))

    assert queue.empty()


@pytest.mark.anyio
async def test_cross_process_fanout_via_redis_pubsub():
    """Simulates two API replicas + one worker: a subscriber on one
    EventFanout instance receives an event published from a completely
    separate EventFanout instance, bridged through shared Redis pub/sub."""

    server = fakeredis_asyncio.FakeServer()
    publisher_fanout = EventFanout(redis_client=fakeredis_asyncio.FakeRedis(server=server))
    subscriber_fanout = EventFanout(redis_client=fakeredis_asyncio.FakeRedis(server=server))

    await subscriber_fanout.start()
    try:
        run_id = uuid4()
        queue = subscriber_fanout.subscribe(run_id)
        await asyncio.sleep(0.05)  # let the psubscribe register before publishing

        event = RunEvent(run_id=run_id, sequence=1, type=EventType.TOOL_STARTED)
        await publisher_fanout.publish(event)

        delivered = await asyncio.wait_for(queue.get(), timeout=2)
        assert delivered.run_id == event.run_id
        assert delivered.type == event.type
    finally:
        await subscriber_fanout.stop()


@pytest.mark.anyio
async def test_publisher_also_receives_its_own_published_event_when_subscribed():
    """A single process that both publishes and subscribes (the common case:
    the worker executing a run while the API is also subscribed) must still
    receive the event exactly once via its own listener -- not directly."""

    server = fakeredis_asyncio.FakeServer()
    fanout = EventFanout(redis_client=fakeredis_asyncio.FakeRedis(server=server))
    await fanout.start()
    try:
        run_id = uuid4()
        queue = fanout.subscribe(run_id)
        await asyncio.sleep(0.05)

        event = RunEvent(run_id=run_id, sequence=1, type=EventType.RUN_COMPLETED)
        await fanout.publish(event)

        first = await asyncio.wait_for(queue.get(), timeout=2)
        assert first.type == EventType.RUN_COMPLETED
        assert queue.empty()  # delivered exactly once, not duplicated
    finally:
        await fanout.stop()
