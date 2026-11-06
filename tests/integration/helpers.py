import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import TypeVar

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from salvador_transcriber.audio.pcm import encode_pcm16le
from salvador_transcriber.config import Settings
from salvador_transcriber.domain.events import Event, TranscriptEvent
from salvador_transcriber.domain.sink import EventSink
from salvador_transcriber.domain.state import Role
from salvador_transcriber.export.http import HTTPSalvadorExporter
from salvador_transcriber.main import create_app
from salvador_transcriber.sessions.call_session import CallSession
from salvador_transcriber.sessions.channel_session import ChannelSession
from salvador_transcriber.storage.repository import Transcript, TranscriptRepository
from salvador_transcriber.storage.sqlite import SQLiteTranscriptRepository
from salvador_transcriber.transcription.adapter import (
    AdapterEvent,
    AdapterEventType,
    Transcriber,
)
from salvador_transcriber.transcription.fake import FakeTranscriber

PCM_FRAME = encode_pcm16le([100] * 320)
DEFAULT_TIMEOUT_SECONDS = 2.0
_T = TypeVar("_T")


def adapter_event(
    event_type: AdapterEventType,
    text: str,
    revision: int = 1,
    *,
    item_id: str | None = None,
    event_id: str | None = None,
    start_ms: int = 0,
    end_ms: int = 20,
) -> AdapterEvent:
    return AdapterEvent(
        event_type=event_type,
        text=text,
        revision=revision,
        start_ms=start_ms,
        end_ms=end_ms,
        provider_item_id=item_id,
        provider_event_id=event_id,
    )


def start_payload(hotel_id: str) -> dict[str, object]:
    return {
        "type": "start",
        "hotel_id": hotel_id,
        "audio": {
            "encoding": "pcm16le",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration_ms": 20,
        },
    }


def transcript_events(
    events: Iterable[Event],
    *,
    final: bool | None = None,
) -> list[TranscriptEvent]:
    return [
        event
        for event in events
        if isinstance(event, TranscriptEvent) and (final is None or event.is_final is final)
    ]


class RecordingEventSink(EventSink):
    """Event log with an asyncio condition instead of timing-based synchronization."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._condition = asyncio.Condition()

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    async def publish(self, event: Event) -> None:
        async with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    async def wait_for(
        self,
        predicate: Callable[[tuple[Event, ...]], bool],
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> tuple[Event, ...]:
        async with asyncio.timeout(timeout_seconds):
            async with self._condition:
                await self._condition.wait_for(lambda: predicate(self.events))
        return self.events


class ControlledFake(FakeTranscriber):
    """Fake with explicit audio/connect gates and optional final on input completion."""

    def __init__(
        self,
        events: Iterable[AdapterEvent] = (),
        *,
        block_send: bool = False,
        block_connect: bool = False,
        final_on_finish: AdapterEvent | None = None,
    ) -> None:
        super().__init__(events)
        self._block_send = block_send
        self._block_connect = block_connect
        self._final_on_finish = final_on_finish
        self._audio_condition = asyncio.Condition()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.connect_started = asyncio.Event()
        self.release_connect = asyncio.Event()

    async def connect(self) -> None:
        if self._block_connect:
            self.connect_started.set()
            await self.release_connect.wait()
        await super().connect()

    async def send_audio(self, pcm: bytes) -> None:
        if self._block_send:
            self.send_started.set()
            await self.release_send.wait()
        await super().send_audio(pcm)
        async with self._audio_condition:
            self._audio_condition.notify_all()

    async def wait_audio(
        self,
        count: int,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        async with asyncio.timeout(timeout_seconds):
            async with self._audio_condition:
                await self._audio_condition.wait_for(lambda: len(self.audio_chunks) >= count)

    async def finish_input(self) -> None:
        if self._final_on_finish is not None:
            self.emit(self._final_on_finish)
        await super().finish_input()


class RoleFactory:
    """Return preallocated per-role fakes, including reconnect generations."""

    def __init__(
        self,
        *,
        guest: Iterable[Transcriber] = (),
        operator: Iterable[Transcriber] = (),
    ) -> None:
        self._adapters = {
            Role.GUEST: deque(guest),
            Role.OPERATOR: deque(operator),
        }

    def __call__(self, role: Role) -> Transcriber:
        try:
            return self._adapters[role].popleft()
        except IndexError as error:
            raise AssertionError(f"no fake adapter left for {role.value}") from error


async def _poll_until(
    predicate: Callable[[], bool],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    async with asyncio.timeout(timeout_seconds):
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0)


async def _emit(fake: FakeTranscriber, event: AdapterEvent) -> None:
    fake.emit(event)


async def _fail(fake: FakeTranscriber, error: Exception) -> None:
    fake.fail(error)


async def _set_event(event: asyncio.Event) -> None:
    event.set()


async def _wait_event(event: asyncio.Event) -> None:
    async with asyncio.timeout(DEFAULT_TIMEOUT_SECONDS):
        await event.wait()


class IntegrationHarness:
    """Synchronous WebSocket tests with all async coordination on the app loop."""

    def __init__(
        self,
        database_path: Path,
        transcriber_factory: RoleFactory,
        *,
        settings: Mapping[str, object] | None = None,
        repository: SQLiteTranscriptRepository | None = None,
        exporter: HTTPSalvadorExporter | None = None,
    ) -> None:
        values: dict[str, object] = {
            "_env_file": None,
            "sqlite_path": database_path,
            "transcriber_backend": "fake",
            "audio_target_sample_rate": 16000,
            "reconnect_backoff_seconds": 0.0,
            "reconnect_jitter_seconds": 0.0,
            "shutdown_drain_timeout_seconds": 0.5,
            "shutdown_final_grace_seconds": 0.1,
            "shutdown_export_drain_timeout_seconds": 0.5,
        }
        if settings is not None:
            values.update(settings)
        self.events = RecordingEventSink()
        self.repository = repository
        self.app: FastAPI = create_app(
            Settings(**values),  # type: ignore[arg-type]
            transcriber_factory=transcriber_factory,
            event_sink=self.events,
            repository=repository,
            exporter=exporter,
        )
        self.client = TestClient(self.app)

    def __enter__(self) -> "IntegrationHarness":
        self.client.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.client.__exit__(*exc_info)

    def run(
        self,
        function: Callable[..., Awaitable[_T]],
        *args: object,
    ) -> _T:
        portal = self.client.portal
        assert portal is not None
        return portal.call(function, *args)

    def create_call(self, call_id: str, hotel_id: str) -> None:
        response = self.client.post(
            "/calls",
            json={"call_id": call_id, "hotel_id": hotel_id},
        )
        assert response.status_code == 201, response.text

    def start_channel(self, websocket: WebSocketTestSession, hotel_id: str) -> None:
        websocket.send_json(start_payload(hotel_id))
        response = websocket.receive_json()
        assert response["type"] == "ready", response

    @staticmethod
    def stop_channel(websocket: WebSocketTestSession) -> None:
        websocket.send_json({"type": "stop"})
        response = websocket.receive_json()
        assert response["code"] == "call_completed", response

    def emit(self, fake: FakeTranscriber, event: AdapterEvent) -> None:
        self.run(_emit, fake, event)

    def fail(self, fake: FakeTranscriber, error: Exception) -> None:
        self.run(_fail, fake, error)

    def release(self, event: asyncio.Event) -> None:
        self.run(_set_event, event)

    def wait_event(self, event: asyncio.Event) -> None:
        self.run(_wait_event, event)

    def wait_audio(self, fake: ControlledFake, count: int) -> None:
        self.run(fake.wait_audio, count)

    def wait_events(
        self,
        predicate: Callable[[tuple[Event, ...]], bool],
    ) -> tuple[Event, ...]:
        return self.run(self.events.wait_for, predicate)

    def wait_until(self, predicate: Callable[[], bool]) -> None:
        self.run(_poll_until, predicate)

    def call(self, call_id: str) -> CallSession:
        return self.run(self.app.state.registry.get_call, call_id)

    def channel(self, call_id: str, role: Role) -> ChannelSession:
        return self.call(call_id).channels[role]

    def transcript(self, call_id: str, hotel_id: str) -> dict[str, object]:
        response = self.client.get(
            f"/calls/{call_id}/transcript",
            params={"hotel_id": hotel_id},
        )
        assert response.status_code == 200, response.text
        body: dict[str, object] = response.json()
        return body

    def wait_stored_transcript(
        self,
        call_id: str,
        hotel_id: str,
        predicate: Callable[[Transcript], bool],
    ) -> Transcript:
        repository: TranscriptRepository = self.app.state.repository

        async def wait() -> Transcript:
            async with asyncio.timeout(DEFAULT_TIMEOUT_SECONDS):
                while True:
                    transcript = await repository.get_transcript(call_id, hotel_id)
                    if predicate(transcript):
                        return transcript
                    await asyncio.sleep(0)

        return self.run(wait)
