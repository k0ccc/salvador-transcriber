import asyncio
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from salvador_transcriber.audio.pcm import PCMFormat, encode_pcm16le
from salvador_transcriber.config import Settings
from salvador_transcriber.domain.events import (
    AudioGapEvent,
    EventType,
    GapReason,
    TranscriptEvent,
)
from salvador_transcriber.domain.sink import InMemoryEventSink
from salvador_transcriber.domain.state import ChannelState, Role
from salvador_transcriber.export.http import HTTPSalvadorExporter
from salvador_transcriber.export.sink import DurableEventSink
from salvador_transcriber.main import create_app
from salvador_transcriber.sessions.call_session import CallSession
from salvador_transcriber.sessions.errors import SessionStateError
from salvador_transcriber.storage.repository import ExportStatus
from salvador_transcriber.storage.sqlite import SQLiteTranscriptRepository
from salvador_transcriber.transcription.adapter import AdapterEvent, AdapterEventType, Transcriber
from salvador_transcriber.transcription.fake import FakeTranscriber

PCM_FRAME = encode_pcm16le([1] * 320)
PCM_FORMAT = PCMFormat(sample_rate=16000)


def adapter_event(
    event_type: AdapterEventType,
    text: str,
    revision: int = 1,
    *,
    item_id: str | None = None,
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
    )


async def wait_until(predicate: Callable[[], bool], timeout_seconds: float = 1.0) -> None:
    async with asyncio.timeout(timeout_seconds):
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.005)


def make_call(
    call_id: str,
    factory: Callable[[Role], Transcriber],
    sink: InMemoryEventSink | DurableEventSink,
    **overrides: object,
) -> CallSession:
    values: dict[str, object] = {
        "call_id": call_id,
        "hotel_id": "hotel-1",
        "transcriber_factory": factory,
        "event_sink": sink,
        "target_sample_rate": 16000,
        "max_audio_message_bytes": 65536,
        "reconnect_backoff_seconds": 0.0,
        "reconnect_jitter_seconds": 0.0,
        "shutdown_drain_timeout_seconds": 0.5,
        "shutdown_final_grace_seconds": 0.05,
    }
    values.update(overrides)
    return CallSession(**values)  # type: ignore[arg-type]


def finals(sink: InMemoryEventSink) -> list[TranscriptEvent]:
    return [event for event in sink.events if isinstance(event, TranscriptEvent) and event.is_final]


def gaps(sink: InMemoryEventSink) -> list[AudioGapEvent]:
    return [event for event in sink.events if isinstance(event, AudioGapEvent)]


class BlockingSendFake(FakeTranscriber):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def send_audio(self, pcm: bytes) -> None:
        self.send_started.set()
        await self.release_send.wait()
        await super().send_audio(pcm)


class BlockingConnectFake(FakeTranscriber):
    def __init__(self) -> None:
        super().__init__()
        self.connect_started = asyncio.Event()
        self.release_connect = asyncio.Event()

    async def connect(self) -> None:
        self.connect_started.set()
        await self.release_connect.wait()
        await super().connect()


class FailingConnectFake(FakeTranscriber):
    async def connect(self) -> None:
        raise RuntimeError("artificial connect failure")


class FinalOnFinishFake(FakeTranscriber):
    async def finish_input(self) -> None:
        self.emit(adapter_event(AdapterEventType.FINAL, "available at shutdown", 1))
        await super().finish_input()


class DelayedFinalOnFinishFake(FakeTranscriber):
    async def finish_input(self) -> None:
        await super().finish_input()
        asyncio.create_task(self._emit_final_after_provider_delay())

    async def _emit_final_after_provider_delay(self) -> None:
        await asyncio.sleep(0.01)
        if not self.closed:
            self.emit(adapter_event(AdapterEventType.FINAL, "delayed final without partial", 1))


async def test_overflow_drop_new_has_precise_gap_and_metrics() -> None:
    blocked = BlockingSendFake()
    sink = InMemoryEventSink()

    def factory(role: Role) -> FakeTranscriber:
        return blocked if role is Role.GUEST else FakeTranscriber()

    call = make_call("overflow", factory, sink, audio_queue_limit=1)
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    channel = call.channels[Role.GUEST]

    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    await asyncio.wait_for(blocked.send_started.wait(), timeout=0.2)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)

    assert channel.queue_depth == 1
    assert (
        channel.metrics.counter_value(
            "audio_frames_dropped_total",
            labels={"role": "guest", "reason": "queue_overflow"},
        )
        == 1
    )

    blocked.release_send.set()
    await wait_until(lambda: len(blocked.audio_chunks) >= 2 and channel.queue_depth == 0)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    await wait_until(lambda: bool(gaps(sink)))
    await wait_until(lambda: len(blocked.audio_chunks) == 3)
    blocked.emit(adapter_event(AdapterEventType.FINAL, "after gap", start_ms=40, end_ms=60))
    await wait_until(lambda: bool(finals(sink)))
    await call.close()

    gap = gaps(sink)[0]
    assert gap.reason is GapReason.QUEUE_OVERFLOW
    assert (gap.start_ms, gap.end_ms) == (40, 60)
    assert (gap.dropped_frames, gap.dropped_bytes) == (1, len(PCM_FRAME))
    assert blocked.audio_bytes == PCM_FRAME * 3
    assert (finals(sink)[0].start_ms, finals(sink)[0].end_ms) == (60, 80)
    assert (
        channel.metrics.counter_value("audio_frames_accepted_total", labels={"role": "guest"}) == 3
    )
    assert (
        channel.metrics.histogram_count("audio_frame_processing_seconds", labels={"role": "guest"})
        == 3
    )


async def test_blocked_channel_does_not_block_other_channel_or_call() -> None:
    blocked = BlockingSendFake()
    other = FakeTranscriber()
    operator = FakeTranscriber()
    sink = InMemoryEventSink()

    def first_factory(role: Role) -> FakeTranscriber:
        return blocked if role is Role.GUEST else operator

    first = make_call("first", first_factory, sink, audio_queue_limit=1)
    second = make_call("second", lambda role: other, sink)
    await first.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await first.start_channel(Role.OPERATOR, "hotel-1", PCM_FORMAT)
    await second.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)

    await first.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    await asyncio.wait_for(blocked.send_started.wait(), timeout=0.2)
    await asyncio.wait_for(
        second.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME),
        timeout=0.1,
    )
    await asyncio.wait_for(
        first.accept_audio(Role.OPERATOR, "hotel-1", PCM_FRAME),
        timeout=0.1,
    )
    await wait_until(lambda: bool(other.audio_chunks) and bool(operator.audio_chunks))

    assert not gaps(sink)
    blocked.release_send.set()
    await first.close()
    await second.close()


async def test_reconnect_preserves_sqlite_finals_and_separates_generations(
    tmp_path: Path,
) -> None:
    initial = FakeTranscriber(
        [
            adapter_event(AdapterEventType.FINAL, "before", item_id="shared-item"),
            adapter_event(AdapterEventType.PARTIAL, "incomplete", item_id="old-item"),
        ]
    )
    reconnected = BlockingConnectFake()
    operator = FakeTranscriber()
    guest_adapters = iter([initial, reconnected])

    def factory(role: Role) -> FakeTranscriber:
        if role is Role.GUEST:
            return next(guest_adapters)
        return operator

    async with SQLiteTranscriptRepository(tmp_path / "reconnect.db") as repository:
        downstream = InMemoryEventSink()
        sink = DurableEventSink(repository=repository, downstream=downstream)
        call = make_call("reconnect", factory, sink, reconnect_max_attempts=2)
        await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
        await call.start_channel(Role.OPERATOR, "hotel-1", PCM_FORMAT)
        await wait_until(
            lambda: (
                len(finals(downstream)) == 1
                and any(
                    isinstance(event, TranscriptEvent) and not event.is_final
                    for event in downstream.events
                )
            )
        )

        await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
        await wait_until(lambda: bool(initial.audio_chunks))
        initial.fail(RuntimeError("artificial disconnect"))
        await asyncio.wait_for(reconnected.connect_started.wait(), timeout=0.5)
        channel = call.channels[Role.GUEST]
        assert channel.state is ChannelState.RECONNECTING
        await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)

        operator.emit(adapter_event(AdapterEventType.FINAL, "operator alive"))
        await wait_until(lambda: len(finals(downstream)) == 2)
        reconnected.release_connect.set()
        await wait_until(
            lambda: channel.state is ChannelState.READY and channel.connection_generation == 2
        )

        await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
        reconnected.emit(adapter_event(AdapterEventType.FINAL, "after", item_id="shared-item"))
        await wait_until(lambda: len(finals(downstream)) == 3)
        await call.close()

        transcript = await repository.get_transcript("reconnect", "hotel-1")
        guest_finals = [item for item in transcript.utterances if item.role is Role.GUEST]
        assert [item.text for item in guest_finals] == ["before", "after"]
        assert [item.connection_generation for item in guest_finals] == [1, 2]
        assert guest_finals[1].start_ms == 20
        assert guest_finals[1].end_ms == 40
        assert len({item.utterance_id for item in guest_finals}) == 2
        assert any(item.text == "operator alive" for item in transcript.utterances)
        assert any(gap.reason is GapReason.OPENAI_DISCONNECT for gap in transcript.gaps)
        assert all(item.text != "incomplete" for item in transcript.utterances)
        assert initial.audio_bytes + reconnected.audio_bytes == PCM_FRAME * 3
        assert (
            channel.metrics.counter_value("reconnect_successes_total", labels={"role": "guest"})
            == 1
        )


async def test_reconnect_exhaustion_is_terminal_only_for_affected_channel() -> None:
    initial = FakeTranscriber()
    operator = FakeTranscriber()
    attempts = [FailingConnectFake(), FailingConnectFake()]
    replacements = iter(attempts)

    # The initial adapter is created before start; later factory calls are failures.
    created = 0

    def counting_factory(role: Role) -> FakeTranscriber:
        nonlocal created
        if role is Role.OPERATOR:
            return operator
        created += 1
        return initial if created == 1 else next(replacements)

    sink = InMemoryEventSink()
    call = make_call(
        "exhausted",
        counting_factory,
        sink,
        reconnect_max_attempts=2,
        reconnect_backoff_seconds=0.02,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await call.start_channel(Role.OPERATOR, "hotel-1", PCM_FORMAT)
    initial.fail(RuntimeError("disconnect"))
    channel = call.channels[Role.GUEST]
    await wait_until(lambda: channel.state is ChannelState.RECONNECTING)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    await wait_until(lambda: channel.state is ChannelState.FAILED)
    await wait_until(lambda: any(gap.reason is GapReason.RECONNECT_TIMEOUT for gap in gaps(sink)))

    await call.accept_audio(Role.OPERATOR, "hotel-1", PCM_FRAME)
    await wait_until(lambda: bool(operator.audio_chunks))
    with pytest.raises(SessionStateError):
        await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    assert channel.metrics.counter_value("reconnect_attempts_total", labels={"role": "guest"}) == 2
    assert channel.metrics.counter_value("reconnect_exhausted_total", labels={"role": "guest"}) == 1
    await call.close()


async def test_stop_during_backoff_cancels_reconnect_and_is_idempotent() -> None:
    initial = FakeTranscriber()
    created = 0

    def factory(role: Role) -> FakeTranscriber:
        nonlocal created
        if role is Role.OPERATOR:
            return FakeTranscriber()
        created += 1
        return initial if created == 1 else FakeTranscriber()

    sink = InMemoryEventSink()
    call = make_call(
        "stop-backoff",
        factory,
        sink,
        reconnect_backoff_seconds=30.0,
        reconnect_max_attempts=5,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    initial.fail(RuntimeError("disconnect"))
    channel = call.channels[Role.GUEST]
    await wait_until(lambda: channel.state is ChannelState.RECONNECTING)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)

    await asyncio.wait_for(call.close(), timeout=0.5)
    event_count = len(sink.events)
    await asyncio.wait_for(call.close(), timeout=0.5)

    assert len(sink.events) == event_count
    assert initial.closed
    assert channel.metrics.counter_value("reconnect_attempts_total", labels={"role": "guest"}) == 0
    assert any(
        gap.reason is GapReason.SHUTDOWN_TIMEOUT and gap.dropped_frames == 1 for gap in gaps(sink)
    )


async def test_shutdown_drains_final_emitted_by_finish_input() -> None:
    finisher = FinalOnFinishFake()
    sink = InMemoryEventSink()
    call = make_call(
        "drain",
        lambda role: finisher if role is Role.GUEST else FakeTranscriber(),
        sink,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)

    await call.close()

    assert [event.text for event in finals(sink)] == ["available at shutdown"]
    assert finisher.input_finished
    assert finisher.closed


async def test_shutdown_waits_for_delayed_final_without_partial() -> None:
    finisher = DelayedFinalOnFinishFake()
    sink = InMemoryEventSink()
    call = make_call(
        "delayed-drain",
        lambda role: finisher if role is Role.GUEST else FakeTranscriber(),
        sink,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)

    await call.close()
    await asyncio.sleep(0.02)

    assert [event.text for event in finals(sink)] == ["delayed final without partial"]
    assert finisher.input_finished
    assert finisher.closed


async def test_shutdown_deadline_cancels_send_and_records_uncertain_audio() -> None:
    blocked = BlockingSendFake()
    sink = InMemoryEventSink()
    call = make_call(
        "deadline",
        lambda role: blocked if role is Role.GUEST else FakeTranscriber(),
        sink,
        shutdown_drain_timeout_seconds=0.03,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    await asyncio.wait_for(blocked.send_started.wait(), timeout=0.2)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)

    await asyncio.wait_for(call.close(), timeout=0.5)
    event_count = len(sink.events)
    await call.close()

    shutdown_gaps = [gap for gap in gaps(sink) if gap.reason is GapReason.SHUTDOWN_TIMEOUT]
    assert len(shutdown_gaps) == 1
    assert shutdown_gaps[0].dropped_frames == 2
    assert shutdown_gaps[0].dropped_bytes == len(PCM_FRAME) * 2
    assert len(sink.events) == event_count
    assert blocked.closed


async def test_shutdown_grace_expiry_invalidates_partial_with_gap() -> None:
    fake = FakeTranscriber(
        [adapter_event(AdapterEventType.PARTIAL, "not final", item_id="partial-item")]
    )
    sink = InMemoryEventSink()
    call = make_call(
        "partial-deadline",
        lambda role: fake if role is Role.GUEST else FakeTranscriber(),
        sink,
        shutdown_final_grace_seconds=0.01,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await wait_until(
        lambda: any(
            isinstance(event, TranscriptEvent) and not event.is_final for event in sink.events
        )
    )

    await call.close()

    assert not finals(sink)
    assert any(
        gap.reason is GapReason.SHUTDOWN_TIMEOUT
        and gap.dropped_frames == 0
        and (gap.start_ms, gap.end_ms) == (0, 20)
        for gap in gaps(sink)
    )


async def test_final_latency_is_observed_when_audio_timing_is_available() -> None:
    fake = FakeTranscriber()
    sink = InMemoryEventSink()
    call = make_call(
        "latency",
        lambda role: fake if role is Role.GUEST else FakeTranscriber(),
        sink,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    fake.emit(adapter_event(AdapterEventType.FINAL, "timed", end_ms=20))
    await wait_until(lambda: bool(finals(sink)))
    channel = call.channels[Role.GUEST]

    assert channel.metrics.histogram_count("final_latency_seconds", labels={"role": "guest"}) == 1
    observation = channel.metrics.latency.observations("latency")[0]
    assert observation.speech_end_ms == 20
    assert observation.final_internal_monotonic >= observation.speech_end_ingress_monotonic
    await call.close()


async def test_final_latency_is_unavailable_without_confirmed_speech_end() -> None:
    fake = FakeTranscriber()
    sink = InMemoryEventSink()
    call = make_call(
        "latency-unavailable",
        lambda role: fake if role is Role.GUEST else FakeTranscriber(),
        sink,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCM_FORMAT)
    await call.accept_audio(Role.GUEST, "hotel-1", PCM_FRAME)
    fake.emit(
        adapter_event(AdapterEventType.FINAL, "not exactly timed", end_ms=20).model_copy(
            update={"speech_end_confirmed": False}
        )
    )
    await wait_until(lambda: bool(finals(sink)))
    channel = call.channels[Role.GUEST]

    assert channel.metrics.histogram_count("final_latency_seconds", labels={"role": "guest"}) == 0
    assert channel.metrics.latency.observations("latency-unavailable") == ()
    unavailable = channel.metrics.latency.unavailable("latency-unavailable")
    assert [item.reason for item in unavailable] == ["speech_end_unconfirmed"]
    await call.close()


async def test_export_overflow_marks_only_delivery_failed(tmp_path: Path) -> None:
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await release_request.wait()
        return httpx.Response(200)

    async with (
        SQLiteTranscriptRepository(tmp_path / "export.db") as repository,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
    ):
        exporter = HTTPSalvadorExporter(
            endpoint="http://salvador.test/events",
            status_store=repository,
            queue_limit=1,
            max_attempts=1,
            client=client,
        )
        await exporter.start()
        events = [
            TranscriptEvent(
                event_type=EventType.TRANSCRIPT_FINAL,
                call_id="call",
                hotel_id="hotel",
                utterance_id=f"guest_{index:04d}",
                role=Role.GUEST,
                sequence=index,
                revision=1,
                text=f"final {index}",
                is_final=True,
                start_ms=0,
                end_ms=20,
                connection_generation=1,
            )
            for index in range(1, 4)
        ]
        for event in events:
            await repository.save_final(event)
        await exporter.publish(events[0])
        await asyncio.wait_for(request_started.wait(), timeout=0.2)
        await exporter.publish(events[1])
        await exporter.publish(events[2])

        assert exporter.dropped_events == 1
        transcript = await repository.get_transcript("call", "hotel")
        assert len(transcript.utterances) == 3
        assert transcript.utterances[2].export_status is ExportStatus.FAILED
        assert transcript.utterances[2].export_error == "queue_overflow"
        release_request.set()
        await exporter.close()
        transcript = await repository.get_transcript("call", "hotel")
        assert [item.export_status for item in transcript.utterances] == [
            ExportStatus.SENT,
            ExportStatus.SENT,
            ExportStatus.FAILED,
        ]


async def test_export_shutdown_race_marks_newly_dequeued_delivery_failed(
    tmp_path: Path,
) -> None:
    request_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await asyncio.Event().wait()
        return httpx.Response(200)

    async with (
        SQLiteTranscriptRepository(tmp_path / "export-race.db") as repository,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
    ):
        exporter = HTTPSalvadorExporter(
            endpoint="http://salvador.test/events",
            status_store=repository,
            max_attempts=1,
            client=client,
        )
        await exporter.start()
        event = TranscriptEvent(
            event_type=EventType.TRANSCRIPT_FINAL,
            call_id="call",
            hotel_id="hotel",
            utterance_id="guest_0001",
            role=Role.GUEST,
            sequence=1,
            revision=1,
            text="durable",
            is_final=True,
            start_ms=0,
            end_ms=20,
            connection_generation=1,
        )
        await repository.save_final(event)
        await exporter.publish(event)

        await asyncio.wait_for(exporter.close(timeout_seconds=0.01), timeout=0.5)

        assert request_started.is_set()
        transcript = await repository.get_transcript("call", "hotel")
        assert transcript.utterances[0].export_status is ExportStatus.FAILED
        assert transcript.utterances[0].export_error == "shutdown_timeout"


async def test_export_shutdown_deadline_marks_inflight_delivery_failed(
    tmp_path: Path,
) -> None:
    request_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await asyncio.Event().wait()
        return httpx.Response(200)

    async with (
        SQLiteTranscriptRepository(tmp_path / "export-timeout.db") as repository,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
    ):
        exporter = HTTPSalvadorExporter(
            endpoint="http://salvador.test/events",
            status_store=repository,
            max_attempts=1,
            client=client,
        )
        await exporter.start()
        event = TranscriptEvent(
            event_type=EventType.TRANSCRIPT_FINAL,
            call_id="call",
            hotel_id="hotel",
            utterance_id="guest_0001",
            role=Role.GUEST,
            sequence=1,
            revision=1,
            text="durable",
            is_final=True,
            start_ms=0,
            end_ms=20,
            connection_generation=1,
        )
        await repository.save_final(event)
        await exporter.publish(event)
        await asyncio.wait_for(request_started.wait(), timeout=0.2)

        await asyncio.wait_for(exporter.close(timeout_seconds=0.01), timeout=0.5)

        transcript = await repository.get_transcript("call", "hotel")
        assert len(transcript.utterances) == 1
        assert transcript.utterances[0].text == "durable"
        assert transcript.utterances[0].export_status is ExportStatus.FAILED
        assert transcript.utterances[0].export_error == "shutdown_timeout"


def test_metrics_endpoints_expose_technical_state_without_call_content(
    tmp_path: Path,
) -> None:
    app = create_app(Settings(_env_file=None, sqlite_path=tmp_path / "metrics.db"))
    with TestClient(app) as client:
        client.post("/calls", json={"call_id": "private-call", "hotel_id": "private-hotel"})
        active = client.get("/metrics/state")
        assert active.status_code == 200
        assert active.json()["active_calls"] == 1
        assert active.json()["active_channels"] == 2

        with client.websocket_connect("/ws/calls/private-call/guest") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "hotel_id": "private-hotel",
                    "audio": {"sample_rate": 16000, "channels": 1},
                }
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(PCM_FRAME)
            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["code"] == "call_completed"

        state_response = client.get("/metrics/state")
        prometheus_response = client.get("/metrics")
        assert state_response.status_code == 200
        assert state_response.json()["active_calls"] == 0
        assert state_response.json()["active_channels"] == 0
        assert "private-call" not in state_response.text
        assert "private-hotel" not in state_response.text
        assert "salvador_audio_frames_accepted_total" in prometheus_response.text
        assert "salvador_audio_frame_processing_seconds_count" in prometheus_response.text
