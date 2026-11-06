import asyncio
import base64
import json
from collections.abc import Awaitable

import pytest
from pydantic import SecretStr, ValidationError

from salvador_transcriber.audio.pcm import PCMFormat
from salvador_transcriber.config import Settings
from salvador_transcriber.domain.events import TranscriptEvent
from salvador_transcriber.domain.sink import InMemoryEventSink
from salvador_transcriber.domain.state import Role
from salvador_transcriber.sessions.call_session import CallSession
from salvador_transcriber.transcription.adapter import AdapterEvent, TranscriberStateError
from salvador_transcriber.transcription.factory import (
    TranscriberConfigurationError,
    build_transcriber_factory,
)
from salvador_transcriber.transcription.fake import FakeTranscriber
from salvador_transcriber.transcription.openai_realtime import (
    OPENAI_REALTIME_TRANSCRIPTION_URL,
    OpenAIRealtimeConfig,
    OpenAIRealtimeProtocolError,
    OpenAIRealtimeProviderError,
    OpenAIRealtimeTranscriber,
    WebSocketLike,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.incoming: asyncio.Queue[str | Exception] = asyncio.Queue()
        self.closed = False

    def enqueue(self, payload: dict[str, object]) -> None:
        self.incoming.put_nowait(json.dumps(payload))

    async def send(self, message: str) -> None:
        if self.closed:
            raise RuntimeError("WebSocket is closed")
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        item = await self.incoming.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.incoming.put_nowait(RuntimeError("closed"))

    @property
    def sent_json(self) -> list[dict[str, object]]:
        return [json.loads(message) for message in self.sent]


class FakeConnector:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket
        self.url: str | None = None
        self.headers: dict[str, str] | None = None
        self.open_timeout: float | None = None

    def __call__(
        self,
        url: str,
        *,
        additional_headers: dict[str, str],
        open_timeout: float,
        max_size: None,
        compression: None,
    ) -> Awaitable[WebSocketLike]:
        self.url = url
        self.headers = additional_headers
        self.open_timeout = open_timeout

        async def connect() -> WebSocketLike:
            return self.websocket

        return connect()


def config(**overrides: object) -> OpenAIRealtimeConfig:
    values: dict[str, object] = {
        "api_key": SecretStr("unit-test-key"),
        "language": "ru",
    }
    values.update(overrides)
    return OpenAIRealtimeConfig(**values)  # type: ignore[arg-type]


def adapter_with_socket(
    **config_overrides: object,
) -> tuple[OpenAIRealtimeTranscriber, FakeWebSocket, FakeConnector]:
    websocket = FakeWebSocket()
    connector = FakeConnector(websocket)
    adapter = OpenAIRealtimeTranscriber(config(**config_overrides), connector=connector)
    return adapter, websocket, connector


async def read_events(adapter: OpenAIRealtimeTranscriber, count: int) -> list[AdapterEvent]:
    result: list[AdapterEvent] = []
    async for event in adapter.events():
        result.append(event)
        if len(result) == count:
            break
    return result


def completed_event(
    event_id: str,
    item_id: str,
    transcript: str,
) -> dict[str, object]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": event_id,
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }


async def test_session_configuration_audio_encoding_and_finish() -> None:
    adapter, websocket, connector = adapter_with_socket(
        vad_threshold=0.65,
        vad_prefix_padding_ms=350,
        vad_silence_duration_ms=800,
    )

    await adapter.connect()
    await adapter.send_audio(b"\x01\x00\x02\x00")
    await adapter.finish_input()
    await adapter.finish_input()
    await adapter.close()
    await adapter.close()

    assert connector.url == OPENAI_REALTIME_TRANSCRIPTION_URL
    assert connector.headers == {"Authorization": "Bearer unit-test-key"}
    assert connector.open_timeout == 10.0
    assert websocket.sent_json == [
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "gpt-live-transcribe", "language": "ru"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.65,
                            "prefix_padding_ms": 350,
                            "silence_duration_ms": 800,
                        },
                    }
                },
            },
        },
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"\x01\x00\x02\x00").decode("ascii"),
        },
        {"type": "input_audio_buffer.commit"},
    ]
    assert websocket.closed is True


async def test_partial_revisions_vad_boundaries_and_final_dedup_handoff() -> None:
    adapter, websocket, _ = adapter_with_socket()
    await adapter.connect()
    for payload in [
        {
            "type": "input_audio_buffer.speech_started",
            "event_id": "vad-start-1",
            "item_id": "item-1",
            "audio_start_ms": 120,
        },
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": "delta-1",
            "item_id": "item-1",
            "content_index": 0,
            "delta": "Здра",
        },
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": "delta-2",
            "item_id": "item-1",
            "content_index": 0,
            "delta": "вствуйте",
        },
        {
            "type": "input_audio_buffer.speech_stopped",
            "event_id": "vad-stop-1",
            "item_id": "item-1",
            "audio_end_ms": 940,
        },
        completed_event("final-1", "item-1", "Здравствуйте"),
        completed_event("final-1", "item-1", "Здравствуйте"),
        completed_event("final-duplicate", "item-1", "Здравствуйте"),
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": "late-delta",
            "item_id": "item-1",
            "delta": " поздно",
        },
        completed_event("final-2", "item-2", "Вторая реплика"),
    ]:
        websocket.enqueue(payload)

    events = await read_events(adapter, 5)
    await adapter.close()

    assert [event.text for event in events] == [
        "Здра",
        "Здравствуйте",
        "Здравствуйте",
        "Здравствуйте",
        "Вторая реплика",
    ]
    assert [event.revision for event in events[:3]] == [1, 2, 3]
    assert [event.event_type.value for event in events] == [
        "partial",
        "partial",
        "final",
        "final",
        "final",
    ]
    assert all(event.provider_item_id == "item-1" for event in events[:4])
    assert events[2].speech_end_confirmed is True
    assert events[4].speech_end_confirmed is False
    assert all(event.start_ms == 120 for event in events[:3])
    assert [event.end_ms for event in events[:3]] == [120, 120, 940]
    assert events[2].provider_event_id == "final-1"
    assert events[3].provider_event_id == "final-duplicate"
    assert events[4].provider_item_id == "item-2"
    assert events[4].revision == 1


async def test_distinct_duplicate_finals_reach_domain_dedup_and_conflict_metrics() -> None:
    adapter, websocket, _ = adapter_with_socket()
    sink = InMemoryEventSink()

    def factory(role: Role) -> OpenAIRealtimeTranscriber | FakeTranscriber:
        return adapter if role is Role.GUEST else FakeTranscriber()

    call = CallSession(
        call_id="openai-duplicates",
        hotel_id="hotel-1",
        transcriber_factory=factory,
        event_sink=sink,
        target_sample_rate=24000,
        max_audio_message_bytes=65536,
        shutdown_final_grace_seconds=0.0,
    )
    for payload in (
        completed_event("final-1", "item-1", "Первый авторитетный текст"),
        completed_event("final-2", "item-1", "Первый авторитетный текст"),
        completed_event("final-3", "item-1", "Конфликтующий повтор"),
    ):
        websocket.enqueue(payload)
    await call.start_channel(Role.GUEST, "hotel-1", PCMFormat(sample_rate=16000))
    channel = call.channels[Role.GUEST]
    try:
        async with asyncio.timeout(0.5):
            while channel.duplicate_final_count < 2:  # noqa: ASYNC110
                await asyncio.sleep(0)
    finally:
        await call.close()

    finals = [
        event for event in sink.events if isinstance(event, TranscriptEvent) and event.is_final
    ]
    assert [event.text for event in finals] == ["Первый авторитетный текст"]
    assert channel.duplicate_final_count == 2
    assert channel.final_conflict_count == 1


async def test_unknown_event_is_ignored_and_malformed_known_event_fails() -> None:
    adapter, websocket, _ = adapter_with_socket()
    await adapter.connect()
    websocket.enqueue({"type": "provider.future_event", "payload": {"text": "not emitted"}})
    websocket.enqueue(completed_event("final-1", "item-1", "Готово"))

    events = await read_events(adapter, 1)
    assert [event.text for event in events] == ["Готово"]
    await adapter.close()

    malformed, malformed_socket, _ = adapter_with_socket()
    await malformed.connect()
    malformed_socket.enqueue(
        {"type": "conversation.item.input_audio_transcription.delta", "event_id": "bad"}
    )
    with pytest.raises(OpenAIRealtimeProtocolError, match="item_id"):
        _ = [event async for event in malformed.events()]
    await malformed.close()


async def test_provider_error_is_safe_and_terminal_for_reader() -> None:
    adapter, websocket, _ = adapter_with_socket()
    await adapter.connect()
    websocket.enqueue(
        {
            "type": "error",
            "event_id": "error-1",
            "error": {
                "type": "invalid_request_error",
                "code": "bad_session",
                "message": "unit-test-key and sensitive full transcript must not be copied",
            },
        }
    )

    with pytest.raises(OpenAIRealtimeProviderError) as caught:
        _ = [event async for event in adapter.events()]
    await adapter.close()

    message = str(caught.value)
    assert "type=invalid_request_error" in message
    assert "code=bad_session" in message
    assert "unit-test-key" not in message
    assert "sensitive full transcript" not in message


async def test_demo_interrupt_closes_provider_socket_without_marking_adapter_closed() -> None:
    adapter, websocket, _ = adapter_with_socket()
    await adapter.connect()

    await adapter.interrupt_for_demo()

    assert websocket.closed is True
    # ChannelSession may now replace this generation through its normal reconnect path.
    await adapter.close()


async def test_close_unblocks_reader_and_rejects_later_audio() -> None:
    adapter, websocket, _ = adapter_with_socket()
    await adapter.connect()

    async def collect() -> list[AdapterEvent]:
        return [event async for event in adapter.events()]

    reader = asyncio.create_task(collect())
    await asyncio.sleep(0)
    await adapter.close()

    assert await reader == []
    assert websocket.closed is True
    with pytest.raises(TranscriberStateError):
        await adapter.send_audio(b"\x00\x00")


async def test_two_adapters_do_not_share_websocket_audio_or_item_state() -> None:
    first, first_socket, _ = adapter_with_socket()
    second, second_socket, _ = adapter_with_socket()
    await first.connect()
    await second.connect()
    await first.send_audio(b"\x01\x00")
    await second.send_audio(b"\x02\x00\x03\x00")
    first_socket.enqueue(completed_event("shared-event-id", "shared-item", "Первый"))
    second_socket.enqueue(completed_event("shared-event-id", "shared-item", "Второй"))

    first_events = await read_events(first, 1)
    second_events = await read_events(second, 1)
    await first.close()

    assert first_events[0].text == "Первый"
    assert second_events[0].text == "Второй"
    assert first_socket.sent_json[1]["audio"] == base64.b64encode(b"\x01\x00").decode("ascii")
    assert second_socket.sent_json[1]["audio"] == base64.b64encode(b"\x02\x00\x03\x00").decode(
        "ascii"
    )
    assert second_socket.closed is False
    await second.close()


def test_factory_selects_backend_and_real_mode_never_falls_back() -> None:
    fake_factory = build_transcriber_factory(Settings(_env_file=None))
    assert isinstance(fake_factory(Role.GUEST), FakeTranscriber)

    with pytest.raises(TranscriberConfigurationError, match="OPENAI_API_KEY"):
        build_transcriber_factory(Settings(_env_file=None, transcriber_backend="openai"))

    openai_factory = build_transcriber_factory(
        Settings(
            _env_file=None,
            transcriber_backend="openai",
            openai_api_key=SecretStr("unit-test-key"),
        )
    )
    first = openai_factory(Role.GUEST)
    second = openai_factory(Role.OPERATOR)
    assert isinstance(first, OpenAIRealtimeTranscriber)
    assert isinstance(second, OpenAIRealtimeTranscriber)
    assert first is not second


def test_openai_mode_rejects_non_24khz_pcm() -> None:
    with pytest.raises(ValidationError, match="24000"):
        Settings(
            _env_file=None,
            transcriber_backend="openai",
            audio_target_sample_rate=16000,
            openai_api_key=SecretStr("unit-test-key"),
        )
