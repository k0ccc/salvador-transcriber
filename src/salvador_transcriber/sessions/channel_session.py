import asyncio
import logging
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from salvador_transcriber.audio.pcm import AudioTimeline, PCMFormat, PCMFrameSplitter
from salvador_transcriber.audio.resampler import StatefulLinearResampler
from salvador_transcriber.domain.events import (
    AudioGapEvent,
    EventType,
    GapReason,
    TranscriptEvent,
)
from salvador_transcriber.domain.sink import EventSink
from salvador_transcriber.domain.state import ChannelState, Role
from salvador_transcriber.observability.metrics import MetricsRegistry
from salvador_transcriber.sessions.errors import SessionStateError
from salvador_transcriber.transcription.adapter import (
    AdapterEvent,
    AdapterEventType,
    ProviderFaultInjector,
    Transcriber,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _AudioFrame:
    pcm: bytes
    start_ms: int
    end_ms: int
    received_at: float


@dataclass(frozen=True, slots=True)
class _AudioTiming:
    start_ms: int
    end_ms: int
    received_at: float


@dataclass(slots=True)
class _TimelineSegment:
    provider_start_ms: int
    provider_end_ms: int
    call_start_ms: int
    call_end_ms: int


@dataclass(slots=True)
class _GapAccumulator:
    reason: GapReason
    generation: int
    start_ms: int | None = None
    end_ms: int | None = None
    dropped_frames: int = 0
    dropped_bytes: int = 0

    def add(self, frame: _AudioFrame) -> None:
        if self.start_ms is None:
            self.start_ms = frame.start_ms
        else:
            self.start_ms = min(self.start_ms, frame.start_ms)
        if self.end_ms is None:
            self.end_ms = frame.end_ms
        else:
            self.end_ms = max(self.end_ms, frame.end_ms)
        self.dropped_frames += 1
        self.dropped_bytes += len(frame.pcm)


class ChannelSession:
    """Own one bounded audio pipeline and one connection generation at a time."""

    def __init__(
        self,
        *,
        call_id: str,
        hotel_id: str,
        role: Role,
        transcriber: Transcriber,
        event_sink: EventSink,
        sequence_allocator: Callable[[], Awaitable[int]],
        target_sample_rate: int,
        max_audio_message_bytes: int,
        transcriber_factory: Callable[[], Transcriber] | None = None,
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
        if audio_queue_limit < 1:
            raise ValueError("audio_queue_limit must be positive")
        self.call_id = call_id
        self.hotel_id = hotel_id
        self.role = role
        self._transcriber = transcriber
        self._transcriber_factory = transcriber_factory
        self._event_sink = event_sink
        self._sequence_allocator = sequence_allocator
        self._target_sample_rate = target_sample_rate
        self._max_audio_message_bytes = max_audio_message_bytes
        self._reconnect_max_attempts = reconnect_max_attempts
        self._reconnect_backoff_seconds = reconnect_backoff_seconds
        self._reconnect_max_backoff_seconds = reconnect_max_backoff_seconds
        self._reconnect_jitter_seconds = reconnect_jitter_seconds
        self._reconnect_connect_timeout_seconds = reconnect_connect_timeout_seconds
        self._shutdown_drain_timeout_seconds = shutdown_drain_timeout_seconds
        self._shutdown_final_grace_seconds = shutdown_final_grace_seconds
        self._metrics = metrics or MetricsRegistry()
        self._channel_key = f"{call_id}:{role.value}"
        self._state = ChannelState.CONNECTING
        self._generation = 1
        self._lifecycle_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._splitter: PCMFrameSplitter | None = None
        self._resampler: StatefulLinearResampler | None = None
        self._timeline: AudioTimeline | None = None
        self._audio_queue: asyncio.Queue[_AudioFrame | None] = asyncio.Queue(
            maxsize=audio_queue_limit
        )
        self._current_frame: _AudioFrame | None = None
        self._overflow_gap: _GapAccumulator | None = None
        self._recent_timings: deque[_AudioTiming] = deque(maxlen=max(256, audio_queue_limit * 4))
        self._utterance_counter = 0
        self._utterance_id: str | None = None
        self._sequence: int | None = None
        self._revision = 0
        self._last_adapter_revision = 0
        self._final: TranscriptEvent | None = None
        self._provider_item_key: str | None = None
        self._provider_finals: dict[str, TranscriptEvent] = {}
        self._duplicate_final_count = 0
        self._final_conflict_count = 0
        self._finalized_event = asyncio.Event()
        self._last_partial_boundaries: tuple[int, int] | None = None
        self._generation_audio_offset_ms: int | None = 0
        self._generation_provider_samples = 0
        self._audio_submitted = False
        self._timeline_segments: deque[_TimelineSegment] = deque(maxlen=1024)
        self._closed = False
        self._close_error: Exception | None = None
        self._metrics.channel_opened(self._channel_key, role, self._state)

    @property
    def state(self) -> ChannelState:
        return self._state

    @property
    def connection_generation(self) -> int:
        return self._generation

    @property
    def duplicate_final_count(self) -> int:
        return self._duplicate_final_count

    @property
    def final_conflict_count(self) -> int:
        return self._final_conflict_count

    @property
    def queue_depth(self) -> int:
        return self._audio_queue.qsize()

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    async def interrupt_provider_for_demo(self) -> None:
        """Trigger the opt-in provider fault without closing the ingress or the call."""

        if not isinstance(self._transcriber, ProviderFaultInjector):
            raise SessionStateError("provider fault injection is unavailable for this backend")
        await self._transcriber.interrupt_for_demo()

    @property
    def timeline(self) -> AudioTimeline:
        if self._timeline is None:
            raise SessionStateError(f"channel {self.role} has not been started")
        return self._timeline

    def _set_state(self, state: ChannelState) -> None:
        self._state = state
        self._metrics.channel_state(self._channel_key, state)

    def _update_queue_depth(self) -> None:
        self._metrics.audio_queue_depth(self._channel_key, self._audio_queue.qsize())

    async def start(self, pcm_format: PCMFormat) -> None:
        async with self._lifecycle_lock:
            if self._state != ChannelState.CONNECTING or self._audio_task is not None:
                raise SessionStateError(f"channel {self.role} is already owned or terminal")
            try:
                await asyncio.wait_for(
                    self._transcriber.connect(),
                    timeout=self._reconnect_connect_timeout_seconds,
                )
            except Exception:
                self._set_state(ChannelState.FAILED)
                try:
                    await self._transcriber.close()
                except Exception:
                    logger.exception("Failed to close channel %s after connect failure", self.role)
                raise
            self._splitter = PCMFrameSplitter(
                pcm_format,
                max_message_bytes=self._max_audio_message_bytes,
            )
            self._resampler = StatefulLinearResampler(
                pcm_format.sample_rate,
                self._target_sample_rate,
            )
            self._timeline = AudioTimeline(pcm_format.sample_rate)
            self._frame_samples = pcm_format.frame_samples
            self._set_state(ChannelState.READY)
            self._audio_task = asyncio.create_task(
                self._process_audio(),
                name=f"audio-{self.call_id}-{self.role}",
            )
            self._start_reader(self._transcriber, self._generation)

    def _start_reader(self, transcriber: Transcriber, generation: int) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            raise SessionStateError(f"channel {self.role} already has an active reader")
        self._reader_task = asyncio.create_task(
            self._read_events(transcriber, generation),
            name=f"transcript-{self.call_id}-{self.role}-g{generation}",
        )

    async def accept_audio(self, data: bytes) -> None:
        if self._state not in {
            ChannelState.READY,
            ChannelState.DEGRADED,
            ChannelState.RECONNECTING,
        }:
            raise SessionStateError(f"channel {self.role} does not accept audio in {self._state}")
        if self._splitter is None or self._resampler is None or self._timeline is None:
            raise SessionStateError(f"channel {self.role} audio pipeline is not initialized")

        pending_samples = self._splitter.pending_bytes // 2
        buffered_start_sample = self._timeline.samples - pending_samples
        frames = self._splitter.push(data)
        self._timeline.consume(data)
        received_at = time.monotonic()
        frame_samples = self._frame_samples
        for index, pcm in enumerate(frames):
            start_sample = buffered_start_sample + index * frame_samples
            frame = _AudioFrame(
                pcm=pcm,
                start_ms=self._timeline.samples_to_ms(start_sample),
                end_ms=self._timeline.samples_to_ms(start_sample + frame_samples),
                received_at=received_at,
            )
            self._recent_timings.append(
                _AudioTiming(
                    start_ms=frame.start_ms,
                    end_ms=frame.end_ms,
                    received_at=received_at,
                )
            )
            try:
                self._audio_queue.put_nowait(frame)
            except asyncio.QueueFull:
                self._record_drop(frame, GapReason.QUEUE_OVERFLOW)
                self._metrics.increment(
                    "audio_queue_overflows_total",
                    labels={"role": self.role.value},
                )
            else:
                self._metrics.increment(
                    "audio_frames_accepted_total",
                    labels={"role": self.role.value},
                )
            self._update_queue_depth()

    def _record_drop(self, frame: _AudioFrame, reason: GapReason) -> None:
        if self._overflow_gap is None:
            self._overflow_gap = _GapAccumulator(reason=reason, generation=self._generation)
        if (
            self._overflow_gap.reason is not reason
            or self._overflow_gap.generation != self._generation
        ):
            raise SessionStateError("pending audio gap must be flushed before changing its reason")
        self._overflow_gap.add(frame)
        self._metrics.increment(
            "audio_frames_dropped_total",
            labels={"role": self.role.value, "reason": reason.value},
        )

    async def _flush_overflow_gap(self) -> None:
        pending = self._overflow_gap
        if pending is None:
            return
        self._overflow_gap = None
        try:
            await self._publish_gap(
                reason=pending.reason,
                generation=pending.generation,
                start_ms=pending.start_ms,
                end_ms=pending.end_ms,
                dropped_frames=pending.dropped_frames,
                dropped_bytes=pending.dropped_bytes,
            )
        except BaseException:
            if self._overflow_gap is None:
                self._overflow_gap = pending
            raise

    async def _publish_gap(
        self,
        *,
        reason: GapReason,
        generation: int,
        start_ms: int | None,
        end_ms: int | None,
        dropped_frames: int = 0,
        dropped_bytes: int = 0,
    ) -> None:
        gap = AudioGapEvent(
            call_id=self.call_id,
            hotel_id=self.hotel_id,
            role=self.role,
            start_ms=start_ms,
            end_ms=end_ms,
            reason=reason,
            connection_generation=generation,
            dropped_frames=dropped_frames,
            dropped_bytes=dropped_bytes,
        )
        await self._event_sink.publish(gap)
        self._metrics.increment(
            "audio_gaps_total",
            labels={"role": self.role.value, "reason": reason.value},
        )

    async def _process_audio(self) -> None:
        while True:
            item = await self._audio_queue.get()
            try:
                if item is None:
                    await self._flush_overflow_gap()
                    await self._finish_audio_input()
                    return
                self._current_frame = item
                await self._flush_overflow_gap()
                await self._process_frame(item)
            finally:
                self._audio_queue.task_done()
                self._update_queue_depth()

    async def _process_frame(self, frame: _AudioFrame) -> None:
        reconnect_task = self._reconnect_task
        if reconnect_task is not None:
            await asyncio.shield(reconnect_task)
        if self._state is ChannelState.FAILED:
            self._record_drop(frame, GapReason.RECONNECT_TIMEOUT)
            await self._flush_overflow_gap()
            self._current_frame = None
            return
        if self._state not in {ChannelState.READY, ChannelState.CLOSING}:
            self._record_drop(frame, GapReason.OPENAI_DISCONNECT)
            await self._flush_overflow_gap()
            self._current_frame = None
            return
        if self._resampler is None:
            raise SessionStateError("resampler is not initialized")

        started_at = time.monotonic()
        if self._generation_audio_offset_ms is None:
            self._generation_audio_offset_ms = frame.start_ms
        try:
            converted = self._resampler.push(frame.pcm)
            if converted:
                await self._transcriber.send_audio(converted)
                self._record_provider_audio(frame, converted)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._current_frame = None
            self._metrics.increment(
                "transcriber_errors_total",
                labels={"role": self.role.value, "kind": "send"},
            )
            if self._state is ChannelState.CLOSING:
                self._close_error = error
                self._record_drop(frame, GapReason.SHUTDOWN_TIMEOUT)
                await self._flush_overflow_gap()
            elif self._state is ChannelState.RECONNECTING:
                self._metrics.increment(
                    "audio_frames_dropped_total",
                    labels={
                        "role": self.role.value,
                        "reason": GapReason.OPENAI_DISCONNECT.value,
                    },
                )
                await self._publish_gap(
                    reason=GapReason.OPENAI_DISCONNECT,
                    generation=self._generation,
                    start_ms=frame.start_ms,
                    end_ms=frame.end_ms,
                    dropped_frames=1,
                    dropped_bytes=len(frame.pcm),
                )
            else:
                self._schedule_reconnect(error, frame=frame)
        else:
            self._current_frame = None
            self._metrics.observe(
                "audio_frame_processing_seconds",
                time.monotonic() - started_at,
                labels={"role": self.role.value},
            )

    def _record_provider_audio(self, frame: _AudioFrame, converted: bytes) -> None:
        self._audio_submitted = True
        provider_start_ms = (
            self._generation_provider_samples * 1000 + self._target_sample_rate // 2
        ) // self._target_sample_rate
        self._generation_provider_samples += len(converted) // 2
        provider_end_ms = (
            self._generation_provider_samples * 1000 + self._target_sample_rate // 2
        ) // self._target_sample_rate
        if self._timeline_segments:
            previous = self._timeline_segments[-1]
            if (
                previous.provider_end_ms == provider_start_ms
                and previous.call_end_ms == frame.start_ms
            ):
                previous.provider_end_ms = provider_end_ms
                previous.call_end_ms = frame.end_ms
                return
        self._timeline_segments.append(
            _TimelineSegment(
                provider_start_ms=provider_start_ms,
                provider_end_ms=provider_end_ms,
                call_start_ms=frame.start_ms,
                call_end_ms=frame.end_ms,
            )
        )

    def _map_provider_time(self, value_ms: int) -> int | None:
        for segment in reversed(self._timeline_segments):
            if segment.provider_start_ms <= value_ms <= segment.provider_end_ms:
                delta_ms = value_ms - segment.provider_start_ms
                return min(segment.call_end_ms, segment.call_start_ms + delta_ms)
        return None

    async def _finish_audio_input(self) -> None:
        if self._resampler is None:
            return
        tail = self._resampler.flush()
        if tail:
            await self._transcriber.send_audio(tail)
            self._audio_submitted = True
        await self._transcriber.finish_input()

    def _schedule_reconnect(self, error: Exception, *, frame: _AudioFrame | None = None) -> None:
        if self._state in {
            ChannelState.CLOSING,
            ChannelState.CLOSED,
            ChannelState.FAILED,
        }:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        generation = self._generation
        self._set_state(ChannelState.RECONNECTING)
        self._metrics.increment("disconnects_total", labels={"role": self.role.value})
        self._metrics.increment(
            "transcriber_errors_total",
            labels={"role": self.role.value, "kind": type(error).__name__},
        )
        self._reconnect_task = asyncio.create_task(
            self._reconnect(generation, frame),
            name=f"reconnect-{self.call_id}-{self.role}-g{generation + 1}",
        )

    async def _reconnect(self, disconnected_generation: int, frame: _AudioFrame | None) -> None:
        started_at = time.monotonic()
        partial_start: int | None = None
        partial_end: int | None = None
        if self._utterance_id is not None and self._final is None:
            if self._last_partial_boundaries is not None:
                partial_start, partial_end = self._last_partial_boundaries
        if frame is not None:
            partial_start = frame.start_ms
            partial_end = frame.end_ms
        if frame is not None:
            self._metrics.increment(
                "audio_frames_dropped_total",
                labels={"role": self.role.value, "reason": GapReason.OPENAI_DISCONNECT.value},
            )
        try:
            await self._publish_gap(
                reason=GapReason.OPENAI_DISCONNECT,
                generation=disconnected_generation,
                start_ms=partial_start,
                end_ms=partial_end,
                dropped_frames=1 if frame is not None else 0,
                dropped_bytes=len(frame.pcm) if frame is not None else 0,
            )
        except Exception:
            self._metrics.increment(
                "session_errors_total",
                labels={"role": self.role.value, "kind": "gap_publish"},
            )
            logger.exception("Failed to publish disconnect gap for channel %s", self.role)
        self._reset_current_utterance()

        old_reader = self._reader_task
        await self._close_adapter_quietly(self._transcriber)
        if old_reader is not None and not old_reader.done():
            old_reader.cancel()
            await asyncio.gather(old_reader, return_exceptions=True)
        if self._reader_task is old_reader:
            self._reader_task = None

        for attempt in range(1, self._reconnect_max_attempts + 1):
            if self._state is not ChannelState.RECONNECTING:
                return
            delay = min(
                self._reconnect_max_backoff_seconds,
                self._reconnect_backoff_seconds * (2 ** (attempt - 1)),
            )
            delay += random.uniform(0.0, self._reconnect_jitter_seconds)
            if delay:
                await asyncio.sleep(delay)
            self._metrics.increment(
                "reconnect_attempts_total",
                labels={"role": self.role.value},
            )
            candidate: Transcriber | None = None
            try:
                if self._transcriber_factory is None:
                    raise SessionStateError("no transcriber factory is configured for reconnect")
                candidate = self._transcriber_factory()
                await asyncio.wait_for(
                    candidate.connect(),
                    timeout=self._reconnect_connect_timeout_seconds,
                )
            except asyncio.CancelledError:
                if candidate is not None:
                    await self._close_adapter_quietly(candidate)
                raise
            except Exception as error:
                self._metrics.increment(
                    "transcriber_errors_total",
                    labels={"role": self.role.value, "kind": "reconnect"},
                )
                logger.warning(
                    "Reconnect attempt %s failed for call_id=%s role=%s: %s",
                    attempt,
                    self.call_id,
                    self.role.value,
                    type(error).__name__,
                )
                if candidate is not None:
                    await self._close_adapter_quietly(candidate)
                continue

            try:
                await self._flush_overflow_gap()
            except Exception:
                await self._close_adapter_quietly(candidate)
                self._set_state(ChannelState.FAILED)
                logger.exception("Failed to flush audio gap before reconnect")
                return
            self._transcriber = candidate
            self._generation += 1
            self._provider_finals.clear()
            self._reset_current_utterance()
            self._generation_audio_offset_ms = None
            self._generation_provider_samples = 0
            self._audio_submitted = False
            self._timeline_segments.clear()
            self._set_state(ChannelState.READY)
            self._start_reader(candidate, self._generation)
            self._metrics.increment(
                "reconnect_successes_total",
                labels={"role": self.role.value},
            )
            self._metrics.observe(
                "reconnect_duration_seconds",
                time.monotonic() - started_at,
                labels={"role": self.role.value},
            )
            return

        if self._state is ChannelState.RECONNECTING:
            self._set_state(ChannelState.FAILED)
            self._metrics.increment(
                "reconnect_exhausted_total",
                labels={"role": self.role.value},
            )

    async def _read_events(self, transcriber: Transcriber, generation: int) -> None:
        try:
            async for event in transcriber.events():
                if (
                    generation != self._generation
                    or transcriber is not self._transcriber
                    or self._state not in {ChannelState.READY, ChannelState.CLOSING}
                ):
                    continue
                normalized = await self._normalize(event, generation)
                if normalized is None:
                    continue
                if generation != self._generation:
                    continue
                await self._event_sink.publish(normalized)
                if normalized.is_final:
                    self._finalized_event.set()
                    self._metrics.increment(
                        "transcript_finals_total",
                        labels={"role": self.role.value},
                    )
                    self._observe_final_latency(
                        normalized,
                        speech_end_confirmed=event.speech_end_confirmed,
                    )
                else:
                    self._metrics.increment(
                        "transcript_partials_total",
                        labels={"role": self.role.value},
                    )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._state not in {ChannelState.CLOSING, ChannelState.CLOSED}:
                self._schedule_reconnect(error)
        else:
            if self._state is ChannelState.READY:
                self._schedule_reconnect(SessionStateError("transcriber event stream ended"))

    def _observe_final_latency(
        self,
        event: TranscriptEvent,
        *,
        speech_end_confirmed: bool,
    ) -> None:
        if not speech_end_confirmed:
            self._mark_latency_unavailable(event, "speech_end_unconfirmed")
            return
        final_internal = time.monotonic()
        for timing in reversed(self._recent_timings):
            if timing.start_ms < event.end_ms <= timing.end_ms:
                self._metrics.observe(
                    "final_latency_seconds",
                    final_internal - timing.received_at,
                    labels={"role": self.role.value},
                )
                self._metrics.latency.record(
                    call_id=event.call_id,
                    role=event.role,
                    utterance_id=event.utterance_id,
                    connection_generation=event.connection_generation,
                    speech_end_ms=event.end_ms,
                    speech_end_ingress_monotonic=timing.received_at,
                    final_internal_monotonic=final_internal,
                )
                return
        self._mark_latency_unavailable(event, "ingress_timing_not_found")

    def _mark_latency_unavailable(self, event: TranscriptEvent, reason: str) -> None:
        self._metrics.increment(
            "final_latency_unavailable_total",
            labels={"role": self.role.value},
        )
        self._metrics.latency.mark_unavailable(
            call_id=event.call_id,
            role=event.role,
            utterance_id=event.utterance_id,
            connection_generation=event.connection_generation,
            reason=reason,
        )

    async def _normalize(self, event: AdapterEvent, generation: int) -> TranscriptEvent | None:
        mapped_start = self._map_provider_time(event.start_ms)
        mapped_end = self._map_provider_time(event.end_ms)
        if mapped_start is not None and mapped_end is not None:
            event = event.model_copy(update={"start_ms": mapped_start, "end_ms": mapped_end})
        else:
            offset_ms = self._generation_audio_offset_ms or 0
            if offset_ms:
                event = event.model_copy(
                    update={
                        "start_ms": event.start_ms + offset_ms,
                        "end_ms": event.end_ms + offset_ms,
                    }
                )
        if event.event_type == AdapterEventType.PARTIAL:
            return await self._normalize_partial(event, generation)
        return await self._normalize_final(event, generation)

    async def _normalize_partial(
        self,
        event: AdapterEvent,
        generation: int,
    ) -> TranscriptEvent | None:
        provider_key = self._provider_key(event)
        if provider_key is not None and provider_key in self._provider_finals:
            return None
        if (
            self._utterance_id is None
            or self._final is not None
            or (provider_key is not None and provider_key != self._provider_item_key)
        ):
            await self._begin_utterance(provider_key)
        if event.revision <= self._last_adapter_revision:
            return None
        self._last_adapter_revision = event.revision
        self._revision += 1
        self._last_partial_boundaries = (event.start_ms, event.end_ms)
        return TranscriptEvent(
            event_type=EventType.TRANSCRIPT_PARTIAL,
            call_id=self.call_id,
            hotel_id=self.hotel_id,
            utterance_id=self._require_utterance_id(),
            role=self.role,
            sequence=self._require_sequence(),
            revision=self._revision,
            text=event.text,
            is_final=False,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            connection_generation=generation,
        )

    async def _normalize_final(
        self,
        event: AdapterEvent,
        generation: int,
    ) -> TranscriptEvent | None:
        provider_key = self._provider_key(event)
        previous_final = (
            self._provider_finals.get(provider_key) if provider_key is not None else self._final
        )
        if previous_final is not None:
            self._duplicate_final_count += 1
            self._metrics.increment(
                "transcript_duplicate_finals_total",
                labels={"role": self.role.value},
            )
            if (
                event.text != previous_final.text
                or event.start_ms != previous_final.start_ms
                or event.end_ms != previous_final.end_ms
            ):
                self._final_conflict_count += 1
                self._metrics.increment(
                    "transcript_final_conflicts_total",
                    labels={"role": self.role.value},
                )
            return None
        if (
            self._utterance_id is None
            or self._final is not None
            or (provider_key is not None and provider_key != self._provider_item_key)
        ):
            await self._begin_utterance(provider_key)
        self._last_adapter_revision = max(self._last_adapter_revision, event.revision)
        self._revision += 1
        final = TranscriptEvent(
            event_type=EventType.TRANSCRIPT_FINAL,
            call_id=self.call_id,
            hotel_id=self.hotel_id,
            utterance_id=self._require_utterance_id(),
            role=self.role,
            sequence=self._require_sequence(),
            revision=self._revision,
            text=event.text,
            is_final=True,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            connection_generation=generation,
        )
        self._final = final
        if provider_key is not None:
            self._provider_finals[provider_key] = final
        return final

    @staticmethod
    def _provider_key(event: AdapterEvent) -> str | None:
        if event.provider_item_id is None:
            return None
        return f"{event.provider_item_id}:{event.provider_content_index or 0}"

    async def _begin_utterance(self, provider_item_key: str | None = None) -> None:
        self._utterance_counter += 1
        self._utterance_id = f"{self.role.value}_{self._utterance_counter:04d}"
        self._sequence = await self._sequence_allocator()
        self._revision = 0
        self._last_adapter_revision = 0
        self._final = None
        self._provider_item_key = provider_item_key
        self._last_partial_boundaries = None

    def _reset_current_utterance(self) -> None:
        self._utterance_id = None
        self._sequence = None
        self._revision = 0
        self._last_adapter_revision = 0
        self._final = None
        self._provider_item_key = None
        self._last_partial_boundaries = None

    def _require_utterance_id(self) -> str:
        if self._utterance_id is None:
            raise SessionStateError("utterance has not been allocated")
        return self._utterance_id

    def _require_sequence(self) -> int:
        if self._sequence is None:
            raise SessionStateError("sequence has not been allocated")
        return self._sequence

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            previous_state = self._state
            self._set_state(ChannelState.CLOSING)
            if previous_state is ChannelState.CONNECTING:
                await self._close_transcriber()
                self._closed = True
                self._set_state(ChannelState.CLOSED)
                self._metrics.channel_closed(self._channel_key)
                return

            can_drain = previous_state in {ChannelState.READY, ChannelState.DEGRADED}
            try:
                if can_drain:
                    await asyncio.wait_for(
                        self._drain_and_finish(),
                        timeout=self._shutdown_drain_timeout_seconds,
                    )
                else:
                    await self._cancel_background_tasks()
                    await self._drop_all_audio(GapReason.SHUTDOWN_TIMEOUT)
            except TimeoutError as error:
                self._close_error = error
                self._metrics.increment(
                    "shutdown_timeouts_total",
                    labels={"role": self.role.value},
                )
            except Exception as error:
                self._close_error = error
                self._metrics.increment(
                    "session_errors_total",
                    labels={"role": self.role.value, "kind": "close"},
                )
                logger.exception("Failed to drain channel %s", self.role)

            await self._cancel_background_tasks()
            if self._close_error is not None:
                try:
                    await self._drop_all_audio(GapReason.SHUTDOWN_TIMEOUT)
                    await self._invalidate_inflight(GapReason.SHUTDOWN_TIMEOUT)
                except Exception:
                    logger.exception("Failed to publish shutdown gap for channel %s", self.role)
            await self._close_transcriber()
            self._closed = True
            self._set_state(
                ChannelState.FAILED if self._close_error is not None else ChannelState.CLOSED
            )
            self._metrics.channel_closed(self._channel_key)

    async def _drain_and_finish(self) -> None:
        if self._splitter is not None and self._timeline is not None:
            remainder = self._splitter.flush()
            if remainder:
                start_sample = self._timeline.samples - len(remainder) // 2
                frame = _AudioFrame(
                    pcm=remainder,
                    start_ms=self._timeline.samples_to_ms(start_sample),
                    end_ms=self._timeline.position_ms,
                    received_at=time.monotonic(),
                )
                await self._audio_queue.put(frame)
                self._metrics.increment(
                    "audio_frames_accepted_total",
                    labels={"role": self.role.value},
                )
                self._update_queue_depth()
        await self._audio_queue.put(None)
        self._update_queue_depth()
        if self._audio_task is not None:
            await self._audio_task

        # Let already queued provider events reach the reader before closing it.
        await asyncio.sleep(0)
        # A provider may return a final without any preceding partial. Once audio was
        # submitted, always allow the configured grace period for such a final. An
        # existing partial also retains the previous invalidation-on-timeout behavior.
        if self._audio_submitted or (self._utterance_id is not None and self._final is None):
            self._finalized_event.clear()
            try:
                await asyncio.wait_for(
                    self._finalized_event.wait(),
                    timeout=self._shutdown_final_grace_seconds,
                )
            except TimeoutError:
                await self._invalidate_inflight(GapReason.SHUTDOWN_TIMEOUT)
        await self._close_transcriber()
        reader = self._reader_task
        if reader is not None:
            await reader

    async def _invalidate_inflight(self, reason: GapReason) -> None:
        if self._utterance_id is None or self._final is not None:
            return
        boundaries = self._last_partial_boundaries
        await self._publish_gap(
            reason=reason,
            generation=self._generation,
            start_ms=boundaries[0] if boundaries is not None else None,
            end_ms=boundaries[1] if boundaries is not None else None,
        )
        self._reset_current_utterance()

    async def _drop_all_audio(self, reason: GapReason) -> None:
        await self._flush_overflow_gap()
        current = self._current_frame
        if current is not None:
            self._record_drop(current, reason)
            self._current_frame = None
        while True:
            try:
                item = self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if item is not None:
                    self._record_drop(item, reason)
            finally:
                self._audio_queue.task_done()
        self._update_queue_depth()
        await self._flush_overflow_gap()

    async def _cancel_background_tasks(self) -> None:
        tasks = {
            task
            for task in (self._reconnect_task, self._reader_task, self._audio_task)
            if task is not None and task is not asyncio.current_task() and not task.done()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self._shutdown_drain_timeout_seconds,
                )
            except TimeoutError as error:
                self._close_error = self._close_error or error
                self._metrics.increment(
                    "shutdown_timeouts_total",
                    labels={"role": self.role.value},
                )
                logger.error("Background task cancellation timed out for channel %s", self.role)
        self._reconnect_task = None
        self._reader_task = None
        self._audio_task = None

    async def _close_adapter_quietly(self, transcriber: Transcriber) -> None:
        try:
            await asyncio.wait_for(
                transcriber.close(),
                timeout=min(1.0, self._shutdown_drain_timeout_seconds),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._metrics.increment(
                "session_errors_total",
                labels={"role": self.role.value, "kind": "close"},
            )
            logger.exception("Failed to close transcriber for channel %s", self.role)

    async def _close_transcriber(self) -> None:
        try:
            await asyncio.wait_for(
                self._transcriber.close(),
                timeout=min(1.0, self._shutdown_drain_timeout_seconds),
            )
        except Exception as error:
            self._close_error = self._close_error or error
            self._metrics.increment(
                "session_errors_total",
                labels={"role": self.role.value, "kind": "close"},
            )
