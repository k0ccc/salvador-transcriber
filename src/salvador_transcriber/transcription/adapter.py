from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvador_transcriber.domain.state import Role


class AdapterEventType(StrEnum):
    PARTIAL = "partial"
    FINAL = "final"


class AdapterEvent(BaseModel):
    """Provider-neutral transcript result produced by a transcriber."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_type: AdapterEventType
    revision: int = Field(default=1, ge=1)
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    provider_event_id: str | None = Field(default=None, min_length=1)
    provider_item_id: str | None = Field(default=None, min_length=1)
    provider_content_index: int | None = Field(default=None, ge=0)
    # False means end_ms is only a conservative transport bound, not confirmed speech end.
    speech_end_confirmed: bool = True

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not be earlier than start_ms")
        return self


class Transcriber(Protocol):
    async def connect(self) -> None: ...

    async def send_audio(self, pcm: bytes) -> None: ...

    def events(self) -> AsyncIterator[AdapterEvent]: ...

    async def finish_input(self) -> None: ...

    async def close(self) -> None: ...


class TranscriberFactory(Protocol):
    def __call__(self, role: Role) -> Transcriber: ...


@runtime_checkable
class ProviderFaultInjector(Protocol):
    """Opt-in demo fault hook; production adapters are never interrupted by default."""

    async def interrupt_for_demo(self) -> None: ...


class TranscriberStateError(RuntimeError):
    pass
