import logging
from typing import Protocol

from salvador_transcriber.domain.events import (
    AudioGapEvent,
    CallCompletedEvent,
    ChannelStatusEvent,
    Event,
    TranscriptEvent,
)
from salvador_transcriber.domain.sink import EventSink
from salvador_transcriber.storage.repository import (
    ExportStatus,
    StoredUtterance,
    TranscriptRepository,
)

logger = logging.getLogger(__name__)


class FinalEventExporter(Protocol):
    async def publish(self, event: TranscriptEvent) -> None:
        """Accept one non-empty final utterance for asynchronous export."""


class DurableEventSink:
    """Persist before fan-out and suppress duplicate finals for every downstream."""

    def __init__(
        self,
        *,
        repository: TranscriptRepository,
        downstream: EventSink,
        exporter: FinalEventExporter | None = None,
    ) -> None:
        self._repository = repository
        self._downstream = downstream
        self._exporter = exporter

    async def publish(self, event: Event) -> None:
        if isinstance(event, TranscriptEvent):
            if not event.is_final:
                await self._downstream.publish(event)
                return
            result = await self._repository.save_final(event)
            if not result.created:
                if self._is_conflict(event, result.record):
                    logger.error(
                        "Conflicting final ignored for call_id=%s utterance_id=%s",
                        event.call_id,
                        event.utterance_id,
                    )
                return
            await self._downstream.publish(event)
            if not event.text.strip():
                await self._repository.record_export_result(
                    call_id=event.call_id,
                    utterance_id=event.utterance_id,
                    status=ExportStatus.SKIPPED_EMPTY,
                    attempts=0,
                )
            elif self._exporter is None:
                await self._repository.record_export_result(
                    call_id=event.call_id,
                    utterance_id=event.utterance_id,
                    status=ExportStatus.DISABLED,
                    attempts=0,
                )
            else:
                await self._exporter.publish(event)
            return

        if isinstance(event, AudioGapEvent | CallCompletedEvent | ChannelStatusEvent):
            created = await self._repository.save_event(event)
            if created:
                await self._downstream.publish(event)

    @staticmethod
    def _is_conflict(event: TranscriptEvent, stored: StoredUtterance) -> bool:
        return (
            event.hotel_id != stored.hotel_id
            or event.role is not stored.role
            or event.sequence != stored.sequence
            or event.revision != stored.revision
            or event.text != stored.text
            or event.start_ms != stored.start_ms
            or event.end_ms != stored.end_ms
            or event.connection_generation != stored.connection_generation
        )
