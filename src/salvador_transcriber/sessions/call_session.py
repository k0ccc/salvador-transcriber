import asyncio
from collections.abc import Mapping
from functools import partial
from types import MappingProxyType

from salvador_transcriber.audio.pcm import PCMFormat
from salvador_transcriber.domain.events import CallCompletedEvent
from salvador_transcriber.domain.sink import EventSink
from salvador_transcriber.domain.state import CallState, ChannelState, Role
from salvador_transcriber.observability.metrics import MetricsRegistry
from salvador_transcriber.sessions.channel_session import ChannelSession
from salvador_transcriber.sessions.errors import (
    HotelMismatchError,
    RoleOwnershipError,
    SessionStateError,
)
from salvador_transcriber.transcription.adapter import TranscriberFactory


class CallSession:
    """Own two independent channels and the per-call sequence counter."""

    def __init__(
        self,
        *,
        call_id: str,
        hotel_id: str,
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
        self.call_id = call_id
        self.hotel_id = hotel_id
        self._state = CallState.CREATED
        self._event_sink = event_sink
        self._sequence = 0
        self._sequence_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._completion_published = False
        self._metrics = metrics or MetricsRegistry()
        self._metrics.call_opened(call_id)
        channels = {
            role: ChannelSession(
                call_id=call_id,
                hotel_id=hotel_id,
                role=role,
                transcriber=transcriber_factory(role),
                event_sink=event_sink,
                sequence_allocator=self._next_sequence,
                target_sample_rate=target_sample_rate,
                max_audio_message_bytes=max_audio_message_bytes,
                transcriber_factory=partial(transcriber_factory, role),
                audio_queue_limit=audio_queue_limit,
                reconnect_max_attempts=reconnect_max_attempts,
                reconnect_backoff_seconds=reconnect_backoff_seconds,
                reconnect_max_backoff_seconds=reconnect_max_backoff_seconds,
                reconnect_jitter_seconds=reconnect_jitter_seconds,
                reconnect_connect_timeout_seconds=reconnect_connect_timeout_seconds,
                shutdown_drain_timeout_seconds=shutdown_drain_timeout_seconds,
                shutdown_final_grace_seconds=shutdown_final_grace_seconds,
                metrics=self._metrics,
            )
            for role in Role
        }
        self._channels = channels
        self.channels: Mapping[Role, ChannelSession] = MappingProxyType(channels)

    @property
    def state(self) -> CallState:
        return self._state

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    async def _next_sequence(self) -> int:
        async with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _check_hotel(self, hotel_id: str) -> None:
        if hotel_id != self.hotel_id:
            raise HotelMismatchError(
                f"call {self.call_id} belongs to a different hotel",
            )

    async def start_channel(self, role: Role, hotel_id: str, pcm_format: PCMFormat) -> None:
        self._check_hotel(hotel_id)
        async with self._lifecycle_lock:
            if self._state not in {CallState.CREATED, CallState.ACTIVE}:
                raise SessionStateError(f"call {self.call_id} does not accept channels")
            channel = self.channels[role]
            if channel.state != ChannelState.CONNECTING:
                raise RoleOwnershipError(f"role {role} is already owned or terminal")
            await channel.start(pcm_format)
            self._state = CallState.ACTIVE

    async def accept_audio(self, role: Role, hotel_id: str, data: bytes) -> None:
        self._check_hotel(hotel_id)
        if self._state != CallState.ACTIVE:
            raise SessionStateError(f"call {self.call_id} does not accept audio")
        await self.channels[role].accept_audio(data)

    async def close_channel(self, role: Role) -> None:
        await self.channels[role].close()

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._state in {CallState.CLOSED, CallState.FAILED}:
                return
            self._state = CallState.CLOSING
            results = await asyncio.gather(
                *(channel.close() for channel in self.channels.values()),
                return_exceptions=True,
            )
            self._state = (
                CallState.FAILED
                if any(isinstance(result, Exception) for result in results)
                or any(channel.state is ChannelState.FAILED for channel in self.channels.values())
                else CallState.CLOSED
            )
            try:
                if not self._completion_published:
                    self._completion_published = True
                    await self._event_sink.publish(
                        CallCompletedEvent(call_id=self.call_id, hotel_id=self.hotel_id),
                    )
            finally:
                self._metrics.call_closed(self.call_id)
