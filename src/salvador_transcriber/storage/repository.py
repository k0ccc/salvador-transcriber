from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from salvador_transcriber.domain.events import (
    AudioGapEvent,
    CallCompletedEvent,
    ChannelStatusEvent,
    EventType,
    GapReason,
    TranscriptEvent,
)
from salvador_transcriber.domain.state import ChannelState, Role


class ExportStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DISABLED = "disabled"
    SKIPPED_EMPTY = "skipped_empty"


@dataclass(frozen=True, slots=True)
class StoredUtterance:
    event_id: str
    call_id: str
    hotel_id: str
    utterance_id: str
    role: Role
    sequence: int
    revision: int
    text: str
    start_ms: int
    end_ms: int
    connection_generation: int
    occurred_at: datetime
    export_status: ExportStatus
    export_attempts: int
    export_error: str | None


@dataclass(frozen=True, slots=True)
class StoredGap:
    event_id: str
    call_id: str
    hotel_id: str
    role: Role
    start_ms: int | None
    end_ms: int | None
    reason: GapReason
    connection_generation: int
    dropped_frames: int
    dropped_bytes: int
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class StoredServiceEvent:
    event_id: str
    call_id: str
    hotel_id: str
    event_type: EventType
    role: Role | None
    status: ChannelState | None
    connection_generation: int | None
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class Transcript:
    call_id: str
    hotel_id: str
    utterances: tuple[StoredUtterance, ...]
    gaps: tuple[StoredGap, ...]
    service_events: tuple[StoredServiceEvent, ...]


@dataclass(frozen=True, slots=True)
class FinalSaveResult:
    created: bool
    record: StoredUtterance


class TranscriptNotFoundError(Exception):
    """The requested call does not exist in the requested hotel context."""


class CallContextMismatchError(Exception):
    """A durable call id cannot be reused with another hotel id."""


class ExportStatusStore(Protocol):
    async def record_export_result(
        self,
        *,
        call_id: str,
        utterance_id: str,
        status: ExportStatus,
        attempts: int,
        error: str | None = None,
    ) -> None:
        """Record the terminal result of one logical export."""


class TranscriptRepository(ExportStatusStore, Protocol):
    async def start(self) -> None:
        """Open the durable store and start its asynchronous boundary."""

    async def close(self) -> None:
        """Drain all accepted operations and close the store."""

    async def save_final(self, event: TranscriptEvent) -> FinalSaveResult:
        """Persist the first final for a call/utterance key."""

    async def save_event(
        self,
        event: AudioGapEvent | CallCompletedEvent | ChannelStatusEvent,
    ) -> bool:
        """Persist a gap or service event, returning whether it was new."""

    async def get_transcript(self, call_id: str, hotel_id: str) -> Transcript:
        """Read one hotel-scoped call in deterministic timeline order."""
