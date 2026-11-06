from typing import Protocol

from salvador_transcriber.domain.events import Event


class EventSink(Protocol):
    async def publish(self, event: Event) -> None:
        """Publish one normalized domain event."""


class InMemoryEventSink:
    """Small local sink; each instance owns its event list."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    async def publish(self, event: Event) -> None:
        self._events.append(event)
