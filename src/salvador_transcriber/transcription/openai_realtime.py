"""Asynchronous OpenAI GA Realtime transcription WebSocket adapter.

Protocol fields were verified on 2026-11-05 against the official ``openai-python``
3.46.0 sources at commit ``ea5d1606f441d80dca1e9e7f99b74581f41408f8``:
https://github.com/openai/openai-python/tree/ea5d1606f441d80dca1e9e7f99b74581f41408f8/src/openai/types/realtime
The HTML documentation was not accessible from the development environment (HTTP 403).
"""

import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import SecretStr
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from salvador_transcriber.transcription.adapter import (
    AdapterEvent,
    AdapterEventType,
    TranscriberStateError,
)

logger = logging.getLogger(__name__)

OPENAI_REALTIME_TRANSCRIPTION_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
OPENAI_PCM_SAMPLE_RATE = 24000

_DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
_COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"
_FAILED_EVENT = "conversation.item.input_audio_transcription.failed"
_SPEECH_STARTED_EVENT = "input_audio_buffer.speech_started"
_SPEECH_STOPPED_EVENT = "input_audio_buffer.speech_stopped"
_COMMITTED_EVENT = "input_audio_buffer.committed"
_ERROR_EVENT = "error"
_IDENTIFIED_EVENTS = {
    _DELTA_EVENT,
    _COMPLETED_EVENT,
    _FAILED_EVENT,
    _SPEECH_STARTED_EVENT,
    _SPEECH_STOPPED_EVENT,
    _ERROR_EVENT,
}
_IGNORED_EVENTS = {
    "session.created",
    "session.updated",
    "input_audio_buffer.cleared",
    "rate_limits.updated",
}


class OpenAIRealtimeError(RuntimeError):
    """Base error that never includes audio, credentials, or transcript text."""


class OpenAIRealtimeConnectionError(OpenAIRealtimeError):
    """The provider connection or a transport operation failed."""


class OpenAIRealtimeProtocolError(OpenAIRealtimeError):
    """A relevant provider event did not match the verified schema."""


class OpenAIRealtimeProviderError(OpenAIRealtimeError):
    """The provider returned an error event."""


class WebSocketLike(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


class WebSocketConnector(Protocol):
    def __call__(
        self,
        url: str,
        *,
        additional_headers: dict[str, str],
        open_timeout: float,
        max_size: None,
        compression: None,
    ) -> Awaitable[WebSocketLike]: ...


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeConfig:
    api_key: SecretStr
    model: str = "gpt-live-transcribe"
    language: str | None = None
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 700
    connect_timeout_seconds: float = 10.0
    websocket_url: str = OPENAI_REALTIME_TRANSCRIPTION_URL


@dataclass(slots=True)
class _ItemState:
    text: str = ""
    revision: int = 0
    start_ms: int | None = None
    end_ms: int | None = None
    final_emitted: bool = False


async def _default_connector(
    url: str,
    *,
    additional_headers: dict[str, str],
    open_timeout: float,
    max_size: None,
    compression: None,
) -> WebSocketLike:
    return await connect(
        url,
        additional_headers=additional_headers,
        open_timeout=open_timeout,
        max_size=max_size,
        compression=compression,
    )


def _required_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OpenAIRealtimeProtocolError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _required_string(payload: dict[str, object], field: str, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise OpenAIRealtimeProtocolError(f"{context}.{field} must be a non-empty string")
    return value


def _required_integer(payload: dict[str, object], field: str, context: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIRealtimeProtocolError(f"{context}.{field} must be a non-negative integer")
    return value


def _content_index(payload: dict[str, object], context: str) -> int:
    value = payload.get("content_index", 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIRealtimeProtocolError(f"{context}.content_index must be non-negative")
    return value


class OpenAIRealtimeTranscriber:
    """Own one WebSocket and all mutable provider state for exactly one channel."""

    def __init__(
        self,
        config: OpenAIRealtimeConfig,
        *,
        connector: WebSocketConnector = _default_connector,
    ) -> None:
        self._config = config
        self._connector = connector
        self._websocket: WebSocketLike | None = None
        self._events_started = False
        self._input_finished = False
        self._closed = False
        self._items: dict[str, _ItemState] = {}
        self._seen_event_ids: set[str] = set()
        self._audio_samples_sent = 0
        self._uncommitted_audio_samples = 0

    @property
    def _audio_duration_ms(self) -> int:
        return self._audio_samples_sent * 1000 // OPENAI_PCM_SAMPLE_RATE

    def _session_update(self) -> dict[str, object]:
        transcription: dict[str, object] = {"model": self._config.model}
        if self._config.language is not None:
            transcription["language"] = self._config.language
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": OPENAI_PCM_SAMPLE_RATE},
                        "transcription": transcription,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": self._config.vad_threshold,
                            "prefix_padding_ms": self._config.vad_prefix_padding_ms,
                            "silence_duration_ms": self._config.vad_silence_duration_ms,
                        },
                    }
                },
            },
        }

    def _require_websocket(self) -> WebSocketLike:
        if self._websocket is None or self._closed:
            raise TranscriberStateError("OpenAI Realtime transcriber is not connected")
        return self._websocket

    async def _send_json(self, payload: dict[str, object]) -> None:
        websocket = self._require_websocket()
        try:
            await websocket.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        except Exception as error:
            raise OpenAIRealtimeConnectionError(
                "failed to send an OpenAI Realtime event"
            ) from error

    async def connect(self) -> None:
        if self._websocket is not None or self._closed:
            raise TranscriberStateError("OpenAI Realtime transcriber cannot be connected twice")
        headers = {"Authorization": f"Bearer {self._config.api_key.get_secret_value()}"}
        try:
            self._websocket = await self._connector(
                self._config.websocket_url,
                additional_headers=headers,
                open_timeout=self._config.connect_timeout_seconds,
                max_size=None,
                compression=None,
            )
            await self._send_json(self._session_update())
        except Exception as error:
            try:
                await self.close()
            except OpenAIRealtimeError:
                pass
            if isinstance(error, OpenAIRealtimeError):
                raise
            raise OpenAIRealtimeConnectionError("failed to connect to OpenAI Realtime") from error

    async def send_audio(self, pcm: bytes) -> None:
        self._require_websocket()
        if self._input_finished:
            raise TranscriberStateError("OpenAI Realtime input is already finished")
        if len(pcm) % 2:
            raise ValueError("OpenAI Realtime PCM16 audio must contain an even number of bytes")
        if not pcm:
            return
        encoded = base64.b64encode(pcm).decode("ascii")
        await self._send_json({"type": "input_audio_buffer.append", "audio": encoded})
        sample_count = len(pcm) // 2
        self._audio_samples_sent += sample_count
        self._uncommitted_audio_samples += sample_count

    async def finish_input(self) -> None:
        self._require_websocket()
        if self._input_finished:
            return
        if self._uncommitted_audio_samples:
            await self._send_json({"type": "input_audio_buffer.commit"})
            self._uncommitted_audio_samples = 0
        self._input_finished = True

    def _item(self, item_id: str) -> _ItemState:
        return self._items.setdefault(item_id, _ItemState())

    def _boundaries(self, item: _ItemState) -> tuple[int, int]:
        # VAD supplies exact stream-relative boundaries. The fallback covers manual
        # commits and can only claim the conservative bounds of audio sent so far.
        start_ms = item.start_ms if item.start_ms is not None else 0
        end_ms = item.end_ms
        if end_ms is None:
            end_ms = max(self._audio_duration_ms, start_ms)
        if end_ms < start_ms:
            raise OpenAIRealtimeProtocolError("provider audio_end_ms precedes audio_start_ms")
        return start_ms, end_ms

    def _map_delta(self, payload: dict[str, object], event_id: str) -> AdapterEvent | None:
        item_id = _required_string(payload, "item_id", _DELTA_EVENT)
        content_index = _content_index(payload, _DELTA_EVENT)
        delta = payload.get("delta", "")
        if delta is None:
            delta = ""
        if not isinstance(delta, str):
            raise OpenAIRealtimeProtocolError(f"{_DELTA_EVENT}.delta must be a string")
        item = self._item(item_id)
        if item.final_emitted:
            return None
        item.text += delta
        item.revision += 1
        start_ms, end_ms = self._boundaries(item)
        return AdapterEvent(
            event_type=AdapterEventType.PARTIAL,
            revision=item.revision,
            text=item.text,
            start_ms=start_ms,
            end_ms=end_ms,
            provider_event_id=event_id,
            provider_item_id=item_id,
            provider_content_index=content_index,
            speech_end_confirmed=item.end_ms is not None,
        )

    def _map_completed(self, payload: dict[str, object], event_id: str) -> AdapterEvent | None:
        item_id = _required_string(payload, "item_id", _COMPLETED_EVENT)
        content_index = _content_index(payload, _COMPLETED_EVENT)
        transcript = payload.get("transcript")
        if not isinstance(transcript, str):
            raise OpenAIRealtimeProtocolError(f"{_COMPLETED_EVENT}.transcript must be a string")
        item = self._item(item_id)
        # Distinct completed events are forwarded even for an already finalized item.
        # ChannelSession owns authoritative dedup/conflict metrics; exact transport
        # retransmissions with the same event_id are still suppressed above.
        item.final_emitted = True
        item.text = transcript
        item.revision += 1
        start_ms, end_ms = self._boundaries(item)
        return AdapterEvent(
            event_type=AdapterEventType.FINAL,
            revision=item.revision,
            text=transcript,
            start_ms=start_ms,
            end_ms=end_ms,
            provider_event_id=event_id,
            provider_item_id=item_id,
            provider_content_index=content_index,
            speech_end_confirmed=item.end_ms is not None,
        )

    def _map_event(self, payload: dict[str, object]) -> AdapterEvent | None:
        event_type = _required_string(payload, "type", "provider event")
        if event_type == _COMMITTED_EVENT:
            self._uncommitted_audio_samples = 0
            return None
        if event_type in _IGNORED_EVENTS:
            return None
        if event_type not in _IDENTIFIED_EVENTS:
            logger.warning("Ignoring unsupported OpenAI Realtime event type=%s", event_type)
            return None

        event_id = _required_string(payload, "event_id", event_type)
        if event_id in self._seen_event_ids:
            return None
        self._seen_event_ids.add(event_id)

        if event_type == _DELTA_EVENT:
            return self._map_delta(payload, event_id)
        if event_type == _COMPLETED_EVENT:
            return self._map_completed(payload, event_id)
        if event_type == _SPEECH_STARTED_EVENT:
            item_id = _required_string(payload, "item_id", event_type)
            self._item(item_id).start_ms = _required_integer(payload, "audio_start_ms", event_type)
            return None
        if event_type == _SPEECH_STOPPED_EVENT:
            item_id = _required_string(payload, "item_id", event_type)
            self._item(item_id).end_ms = _required_integer(payload, "audio_end_ms", event_type)
            return None
        if event_type == _FAILED_EVENT:
            item_id = _required_string(payload, "item_id", event_type)
            error = _required_object(payload.get("error"), f"{event_type}.error")
            code = error.get("code")
            safe_code = code if isinstance(code, str) else "unknown"
            raise OpenAIRealtimeProviderError(
                f"OpenAI transcription failed for item {item_id}: code={safe_code}"
            )

        error = _required_object(payload.get("error"), "error.error")
        provider_type = error.get("type")
        provider_code = error.get("code")
        safe_type = provider_type if isinstance(provider_type, str) else "unknown"
        safe_code = provider_code if isinstance(provider_code, str) else "unknown"
        raise OpenAIRealtimeProviderError(
            f"OpenAI Realtime returned an error: type={safe_type}, code={safe_code}"
        )

    async def events(self) -> AsyncIterator[AdapterEvent]:
        websocket = self._require_websocket()
        if self._events_started:
            raise TranscriberStateError("OpenAI Realtime events already have a reader")
        self._events_started = True
        while not self._closed:
            try:
                raw = await websocket.recv()
            except ConnectionClosedOK:
                return
            except ConnectionClosedError as error:
                if self._closed:
                    return
                raise OpenAIRealtimeConnectionError("OpenAI Realtime connection closed") from error
            except Exception as error:
                if self._closed:
                    return
                raise OpenAIRealtimeConnectionError(
                    "failed to read from OpenAI Realtime"
                ) from error
            if isinstance(raw, bytes):
                raise OpenAIRealtimeProtocolError("OpenAI Realtime returned a binary event")
            try:
                decoded: object = json.loads(raw)
            except json.JSONDecodeError as error:
                raise OpenAIRealtimeProtocolError(
                    "OpenAI Realtime returned invalid JSON"
                ) from error
            event = self._map_event(_required_object(decoded, "provider event"))
            if event is not None:
                yield event

    async def interrupt_for_demo(self) -> None:
        """Close only the provider socket so ChannelSession performs a real reconnect."""

        websocket = self._require_websocket()
        try:
            await websocket.close()
        except Exception as error:
            raise OpenAIRealtimeConnectionError(
                "failed to interrupt OpenAI Realtime for the demo"
            ) from error

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._websocket is None:
            return
        try:
            await self._websocket.close()
        except Exception as error:
            raise OpenAIRealtimeConnectionError("failed to close OpenAI Realtime") from error
