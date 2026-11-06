from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from salvador_transcriber.audio.pcm import PCMFormat
from salvador_transcriber.domain.events import EventType, GapReason
from salvador_transcriber.domain.state import CallState, ChannelState, Role
from salvador_transcriber.storage.repository import ExportStatus


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CallCreateRequest(APIModel):
    call_id: str = Field(min_length=1)
    hotel_id: str = Field(min_length=1)


class CallResponse(APIModel):
    call_id: str
    hotel_id: str
    state: CallState


class StartCommand(APIModel):
    type: Literal["start"] = "start"
    hotel_id: str = Field(min_length=1)
    audio: PCMFormat


class StopCommand(APIModel):
    type: Literal["stop"] = "stop"


class PingCommand(APIModel):
    type: Literal["ping"] = "ping"


IngressCommand = Annotated[StartCommand | StopCommand | PingCommand, Field(discriminator="type")]


class IngressEnvelope(RootModel[IngressCommand]):
    pass


class IngressResponse(APIModel):
    type: Literal["accepted", "ready", "pong", "warning", "error"]
    code: str
    message: str


class HealthResponse(APIModel):
    status: Literal["ok"]
    backend: Literal["fake", "openai"]


class UtteranceResponse(APIModel):
    event_id: str
    utterance_id: str
    role: Role
    sequence: int
    revision: int
    text: str
    start_ms: int
    end_ms: int
    connection_generation: int
    export_status: ExportStatus
    export_attempts: int


class GapResponse(APIModel):
    event_id: str
    role: Role
    start_ms: int | None
    end_ms: int | None
    reason: GapReason
    connection_generation: int
    dropped_frames: int
    dropped_bytes: int


class ServiceEventResponse(APIModel):
    event_id: str
    event_type: EventType
    role: Role | None
    status: ChannelState | None
    connection_generation: int | None


class TranscriptResponse(APIModel):
    call_id: str
    hotel_id: str
    utterances: tuple[UtteranceResponse, ...]
    gaps: tuple[GapResponse, ...]
    service_events: tuple[ServiceEventResponse, ...]
