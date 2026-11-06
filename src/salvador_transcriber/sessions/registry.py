import asyncio

from salvador_transcriber.domain.sink import EventSink
from salvador_transcriber.observability.metrics import MetricsRegistry
from salvador_transcriber.sessions.call_session import CallSession
from salvador_transcriber.sessions.errors import (
    CallAlreadyExistsError,
    CallNotFoundError,
    SessionStateError,
)
from salvador_transcriber.transcription.adapter import TranscriberFactory


class CallRegistry:
    """Atomic in-process registry; no audio or utterance state is global."""

    def __init__(
        self,
        *,
        transcriber_factory: TranscriberFactory,
        event_sink: EventSink,
        target_sample_rate: int,
        max_audio_message_bytes: int,
        audio_queue_limit: int = 100,
        reconnect_max_attempts: int = 5,
        reconnect_backoff_seconds: float = 0.1,
        reconnect_max_backoff_seconds: float = 2.0,
        reconnect_jitter_seconds: float = 0.05,
        reconnect_connect_timeout_seconds: float = 10.0,
        shutdown_drain_timeout_seconds: float = 2.0,
        shutdown_final_grace_seconds: float = 0.2,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._transcriber_factory = transcriber_factory
        self._event_sink = event_sink
        self._target_sample_rate = target_sample_rate
        self._max_audio_message_bytes = max_audio_message_bytes
        self._audio_queue_limit = audio_queue_limit
        self._reconnect_max_attempts = reconnect_max_attempts
        self._reconnect_backoff_seconds = reconnect_backoff_seconds
        self._reconnect_max_backoff_seconds = reconnect_max_backoff_seconds
        self._reconnect_jitter_seconds = reconnect_jitter_seconds
        self._reconnect_connect_timeout_seconds = reconnect_connect_timeout_seconds
        self._shutdown_drain_timeout_seconds = shutdown_drain_timeout_seconds
        self._shutdown_final_grace_seconds = shutdown_final_grace_seconds
        self._metrics = metrics or MetricsRegistry()
        self._calls: dict[str, CallSession] = {}
        self._lock = asyncio.Lock()
        self._accepting = True

    @property
    def call_ids(self) -> tuple[str, ...]:
        return tuple(self._calls)

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    async def create_call(self, call_id: str, hotel_id: str) -> CallSession:
        async with self._lock:
            if not self._accepting:
                raise SessionStateError("call registry is shutting down")
            if call_id in self._calls:
                raise CallAlreadyExistsError(f"call {call_id} already exists")
            call = CallSession(
                call_id=call_id,
                hotel_id=hotel_id,
                transcriber_factory=self._transcriber_factory,
                event_sink=self._event_sink,
                target_sample_rate=self._target_sample_rate,
                max_audio_message_bytes=self._max_audio_message_bytes,
                audio_queue_limit=self._audio_queue_limit,
                reconnect_max_attempts=self._reconnect_max_attempts,
                reconnect_backoff_seconds=self._reconnect_backoff_seconds,
                reconnect_max_backoff_seconds=self._reconnect_max_backoff_seconds,
                reconnect_jitter_seconds=self._reconnect_jitter_seconds,
                reconnect_connect_timeout_seconds=self._reconnect_connect_timeout_seconds,
                shutdown_drain_timeout_seconds=self._shutdown_drain_timeout_seconds,
                shutdown_final_grace_seconds=self._shutdown_final_grace_seconds,
                metrics=self._metrics,
            )
            self._calls[call_id] = call
            return call

    async def get_call(self, call_id: str) -> CallSession:
        async with self._lock:
            try:
                return self._calls[call_id]
            except KeyError as error:
                raise CallNotFoundError(f"call {call_id} does not exist") from error

    async def close_call(self, call_id: str) -> CallSession:
        call = await self.get_call(call_id)
        await call.close()
        async with self._lock:
            if self._calls.get(call_id) is call:
                del self._calls[call_id]
        return call

    async def close_all(self) -> None:
        async with self._lock:
            self._accepting = False
            calls = tuple(self._calls.values())
        await asyncio.gather(*(call.close() for call in calls), return_exceptions=True)
        async with self._lock:
            for call in calls:
                if self._calls.get(call.call_id) is call:
                    del self._calls[call.call_id]
