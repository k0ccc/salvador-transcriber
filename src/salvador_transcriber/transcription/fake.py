import asyncio
from collections.abc import AsyncIterator, Iterable

from salvador_transcriber.transcription.adapter import (
    AdapterEvent,
    TranscriberStateError,
)

_CLOSE = object()
_QueueItem = AdapterEvent | Exception | object


class FakeTranscriber:
    """Scriptable, network-free transcriber with per-instance mutable state."""

    def __init__(
        self,
        events: Iterable[AdapterEvent] = (),
        *,
        event_delay: float = 0.0,
    ) -> None:
        if event_delay < 0:
            raise ValueError("event_delay must not be negative")
        self._script = tuple(events)
        self._event_delay = event_delay
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._audio_chunks: list[bytes] = []
        self._connected = False
        self._input_finished = False
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def input_finished(self) -> bool:
        return self._input_finished

    @property
    def audio_chunks(self) -> tuple[bytes, ...]:
        return tuple(self._audio_chunks)

    @property
    def audio_bytes(self) -> bytes:
        return b"".join(self._audio_chunks)

    async def connect(self) -> None:
        if self._connected or self._closed:
            raise TranscriberStateError("fake transcriber cannot be connected twice")
        self._connected = True

    async def send_audio(self, pcm: bytes) -> None:
        if not self._connected or self._closed:
            raise TranscriberStateError("fake transcriber is not connected")
        if self._input_finished:
            raise TranscriberStateError("fake transcriber input is already finished")
        self._audio_chunks.append(pcm)

    def emit(self, event: AdapterEvent) -> None:
        if self._closed:
            raise TranscriberStateError("cannot emit an event after close")
        self._queue.put_nowait(event)

    def fail(self, error: Exception) -> None:
        if self._closed:
            raise TranscriberStateError("cannot fail a closed fake transcriber")
        self._queue.put_nowait(error)

    async def events(self) -> AsyncIterator[AdapterEvent]:
        if not self._connected:
            raise TranscriberStateError("fake transcriber must be connected before reading")
        for event in self._script:
            if self._event_delay:
                await asyncio.sleep(self._event_delay)
            yield event
        while True:
            item = await self._queue.get()
            if item is _CLOSE:
                return
            if isinstance(item, Exception):
                raise item
            if isinstance(item, AdapterEvent):
                yield item

    async def finish_input(self) -> None:
        if not self._connected or self._closed:
            raise TranscriberStateError("fake transcriber is not connected")
        self._input_finished = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(_CLOSE)
