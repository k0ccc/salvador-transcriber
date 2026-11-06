from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvador_transcriber.domain.state import ChannelState, Role


class EventType(StrEnum):
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    AUDIO_GAP = "audio_gap"
    CALL_COMPLETED = "call_completed"
    CHANNEL_STATUS = "channel_status"


class GapReason(StrEnum):
    QUEUE_OVERFLOW = "queue_overflow"
    OPENAI_DISCONNECT = "openai_disconnect"
    RECONNECT_TIMEOUT = "reconnect_timeout"
    INVALID_INFLIGHT_UTTERANCE = "invalid_inflight_utterance"
    SHUTDOWN_TIMEOUT = "shutdown_timeout"


class DomainEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    call_id: str = Field(min_length=1)
    hotel_id: str = Field(min_length=1)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TranscriptEvent(DomainEvent):
    event_type: Literal[EventType.TRANSCRIPT_PARTIAL, EventType.TRANSCRIPT_FINAL]
    utterance_id: str = Field(min_length=1)
    role: Role
    sequence: int = Field(ge=1)
    revision: int = Field(ge=1)
    text: str
    is_final: bool
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    connection_generation: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.is_final != (self.event_type == EventType.TRANSCRIPT_FINAL):
            raise ValueError("event_type and is_final must describe the same transcript state")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not be earlier than start_ms")
        return self


class AudioGapEvent(DomainEvent):
    event_type: Literal[EventType.AUDIO_GAP] = EventType.AUDIO_GAP
    role: Role
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    reason: GapReason
    connection_generation: int = Field(ge=1)
    dropped_frames: int = Field(default=0, ge=0)
    dropped_bytes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        if self.start_ms is not None and self.end_ms is not None:
            if self.end_ms < self.start_ms:
                raise ValueError("end_ms must not be earlier than start_ms")
        return self


class CallCompletedEvent(DomainEvent):
    event_type: Literal[EventType.CALL_COMPLETED] = EventType.CALL_COMPLETED


class ChannelStatusEvent(DomainEvent):
    event_type: Literal[EventType.CHANNEL_STATUS] = EventType.CHANNEL_STATUS
    role: Role
    status: ChannelState
    connection_generation: int = Field(ge=1)


Event = Annotated[
    TranscriptEvent | AudioGapEvent | CallCompletedEvent | ChannelStatusEvent,
    Field(discriminator="event_type"),
]
