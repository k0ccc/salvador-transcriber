import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from salvador_transcriber.domain.events import Event
from salvador_transcriber.domain.sink import EventSink


@dataclass(eq=False, slots=True)
class _Subscription:
    call_id: str
    hotel_id: str
    queue: asyncio.Queue[Event]


class DemoEventHub:
    """Fan out to the configured sink and bounded per-viewer drop-new queues."""

    def __init__(self, downstream: EventSink, *, queue_limit: int = 256) -> None:
        if queue_limit < 1:
            raise ValueError("demo event queue limit must be positive")
        self._downstream = downstream
        self._queue_limit = queue_limit
        self._subscriptions: set[_Subscription] = set()
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    async def publish(self, event: Event) -> None:
        await self._downstream.publish(event)
        for subscription in tuple(self._subscriptions):
            if subscription.call_id != event.call_id or subscription.hotel_id != event.hotel_id:
                continue
            try:
                subscription.queue.put_nowait(event)
            except asyncio.QueueFull:
                self._dropped_events += 1

    @asynccontextmanager
    async def subscribe(
        self,
        call_id: str,
        hotel_id: str,
    ) -> AsyncIterator[asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_limit)
        subscription = _Subscription(call_id=call_id, hotel_id=hotel_id, queue=queue)
        self._subscriptions.add(subscription)
        try:
            yield queue
        finally:
            self._subscriptions.discard(subscription)
