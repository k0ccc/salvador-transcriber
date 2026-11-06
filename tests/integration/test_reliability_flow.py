import json
from pathlib import Path
from typing import cast

import httpx

from salvador_transcriber.domain.events import EventType, GapReason
from salvador_transcriber.domain.state import ChannelState, Role
from salvador_transcriber.export.http import HTTPSalvadorExporter, idempotency_key
from salvador_transcriber.observability.metrics import MetricsRegistry
from salvador_transcriber.storage.repository import ExportStatus
from salvador_transcriber.storage.sqlite import SQLiteTranscriptRepository
from salvador_transcriber.transcription.adapter import AdapterEventType
from tests.integration.helpers import (
    PCM_FRAME,
    ControlledFake,
    IntegrationHarness,
    RoleFactory,
    adapter_event,
    transcript_events,
)


def utterances(body: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], body["utterances"])


def test_duplicate_provider_events_and_stale_revisions_keep_one_authoritative_final(
    tmp_path: Path,
) -> None:
    guest = ControlledFake()
    factory = RoleFactory(guest=[guest], operator=[ControlledFake()])

    with IntegrationHarness(tmp_path / "duplicates.db", factory) as harness:
        harness.create_call("call-duplicates", "hotel-duplicates")
        with harness.client.websocket_connect("/ws/calls/call-duplicates/guest") as guest_ws:
            harness.start_channel(guest_ws, "hotel-duplicates")
            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(guest, 1)
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Новый partial",
                    2,
                    item_id="item-1",
                    event_id="delta-2",
                ),
            )
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Устаревший partial",
                    1,
                    item_id="item-1",
                    event_id="delta-1",
                ),
            )
            final = adapter_event(
                AdapterEventType.FINAL,
                "Авторитетный финал",
                3,
                item_id="item-1",
                event_id="provider-final",
            )
            harness.emit(guest, final)
            harness.emit(guest, final)
            harness.emit(
                guest,
                final.model_copy(update={"provider_event_id": "provider-final-new-id"}),
            )
            harness.emit(
                guest,
                final.model_copy(
                    update={
                        "provider_event_id": "provider-conflict",
                        "text": "Конфликтующий финал",
                        "revision": 4,
                    }
                ),
            )
            channel = harness.channel("call-duplicates", Role.GUEST)
            harness.wait_until(lambda: channel.duplicate_final_count == 3)
            harness.stop_channel(guest_ws)

        events = transcript_events(harness.events.events)
        assert [event.text for event in events] == ["Новый partial", "Авторитетный финал"]
        assert len(transcript_events(harness.events.events, final=True)) == 1
        stored = utterances(harness.transcript("call-duplicates", "hotel-duplicates"))
        assert len(stored) == 1
        assert stored[0]["text"] == "Авторитетный финал"
        assert stored[0]["revision"] == 2
        assert channel.duplicate_final_count == 3
        assert channel.final_conflict_count == 1
        metrics = harness.app.state.metrics
        assert (
            metrics.counter_value("transcript_duplicate_finals_total", labels={"role": "guest"})
            == 3
        )
        assert (
            metrics.counter_value("transcript_final_conflicts_total", labels={"role": "guest"}) == 1
        )
        assert metrics.counter_value("transcript_finals_total", labels={"role": "guest"}) == 1


def test_websocket_queue_overflow_drops_new_frames_and_persists_precise_gap(
    tmp_path: Path,
) -> None:
    guest = ControlledFake(block_send=True)
    factory = RoleFactory(guest=[guest], operator=[ControlledFake()])

    with IntegrationHarness(
        tmp_path / "overflow.db",
        factory,
        settings={"audio_queue_limit": 1},
    ) as harness:
        harness.create_call("call-overflow", "hotel-overflow")
        with harness.client.websocket_connect("/ws/calls/call-overflow/guest") as guest_ws:
            harness.start_channel(guest_ws, "hotel-overflow")
            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_event(guest.send_started)
            guest_ws.send_bytes(PCM_FRAME)
            guest_ws.send_bytes(PCM_FRAME)
            channel = harness.channel("call-overflow", Role.GUEST)
            assert channel.queue_depth == 1

            harness.release(guest.release_send)
            harness.wait_audio(guest, 2)
            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(guest, 3)
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Финал после потери кадра",
                    1,
                    item_id="after-overflow",
                    start_ms=40,
                    end_ms=60,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 1)
            harness.stop_channel(guest_ws)

        body = harness.transcript("call-overflow", "hotel-overflow")
        stored = utterances(body)
        assert len(stored) == 1
        assert (stored[0]["start_ms"], stored[0]["end_ms"]) == (60, 80)
        gaps = cast(list[dict[str, object]], body["gaps"])
        overflow_gaps = [gap for gap in gaps if gap["reason"] == GapReason.QUEUE_OVERFLOW]
        assert len(overflow_gaps) == 1
        gap = overflow_gaps[0]
        assert (gap["start_ms"], gap["end_ms"]) == (40, 60)
        assert (gap["dropped_frames"], gap["dropped_bytes"]) == (1, len(PCM_FRAME))
        metrics = harness.app.state.metrics
        assert metrics.counter_value("audio_frames_accepted_total", labels={"role": "guest"}) == 3
        assert (
            metrics.counter_value(
                "audio_frames_dropped_total",
                labels={"role": "guest", "reason": "queue_overflow"},
            )
            == 1
        )
        assert metrics.counter_value("audio_queue_overflows_total", labels={"role": "guest"}) == 1
        assert (
            metrics.counter_value(
                "audio_gaps_total",
                labels={"role": "guest", "reason": "queue_overflow"},
            )
            == 1
        )


def test_stop_during_reconnect_backoff_cancels_recovery_and_drains_bounded_audio(
    tmp_path: Path,
) -> None:
    initial = ControlledFake()
    replacement = ControlledFake()
    factory = RoleFactory(guest=[initial, replacement], operator=[ControlledFake()])

    with IntegrationHarness(
        tmp_path / "stop-reconnect.db",
        factory,
        settings={
            "reconnect_backoff_seconds": 30.0,
            "reconnect_max_backoff_seconds": 30.0,
            "shutdown_drain_timeout_seconds": 0.2,
        },
    ) as harness:
        harness.create_call("call-stop-reconnect", "hotel-stop")
        with harness.client.websocket_connect("/ws/calls/call-stop-reconnect/guest") as guest_ws:
            harness.start_channel(guest_ws, "hotel-stop")
            channel = harness.channel("call-stop-reconnect", Role.GUEST)
            harness.fail(initial, RuntimeError("artificial disconnect"))
            harness.wait_until(lambda: channel.state is ChannelState.RECONNECTING)
            guest_ws.send_bytes(PCM_FRAME)
            harness.stop_channel(guest_ws)

        assert initial.closed
        assert not replacement.connected
        assert channel.state is ChannelState.CLOSED
        metrics = harness.app.state.metrics
        assert metrics.counter_value("reconnect_attempts_total", labels={"role": "guest"}) == 0
        body = harness.transcript("call-stop-reconnect", "hotel-stop")
        gaps = cast(list[dict[str, object]], body["gaps"])
        assert any(
            gap["reason"] == GapReason.SHUTDOWN_TIMEOUT
            and gap["dropped_frames"] == 1
            and gap["dropped_bytes"] == len(PCM_FRAME)
            for gap in gaps
        )
        service_events = cast(list[dict[str, object]], body["service_events"])
        assert [
            event["event_type"]
            for event in service_events
            if event["event_type"] == EventType.CALL_COMPLETED
        ] == [EventType.CALL_COMPLETED]


def test_websocket_stop_drains_final_emitted_by_finish_input_exactly_once(
    tmp_path: Path,
) -> None:
    final_on_finish = adapter_event(
        AdapterEventType.FINAL,
        "Финал, доступный при drain",
        1,
        item_id="shutdown-final",
    )
    guest = ControlledFake(final_on_finish=final_on_finish)
    factory = RoleFactory(guest=[guest], operator=[ControlledFake()])

    with IntegrationHarness(tmp_path / "shutdown-drain.db", factory) as harness:
        harness.create_call("call-drain", "hotel-drain")
        with harness.client.websocket_connect("/ws/calls/call-drain/guest") as guest_ws:
            harness.start_channel(guest_ws, "hotel-drain")
            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(guest, 1)
            harness.stop_channel(guest_ws)

        assert guest.input_finished
        assert guest.closed
        first_read = harness.transcript("call-drain", "hotel-drain")
        second_read = harness.transcript("call-drain", "hotel-drain")
        assert first_read == second_read
        stored = utterances(first_read)
        assert [item["text"] for item in stored] == ["Финал, доступный при drain"]
        assert len(transcript_events(harness.events.events, final=True)) == 1
        service_events = cast(list[dict[str, object]], first_read["service_events"])
        assert sum(event["event_type"] == EventType.CALL_COMPLETED for event in service_events) == 1


def test_full_path_export_retry_reuses_idempotency_key_after_provider_duplicate(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 200)

    repository = SQLiteTranscriptRepository(tmp_path / "export-retry.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    exporter = HTTPSalvadorExporter(
        endpoint="http://salvador.test/events",
        status_store=repository,
        max_attempts=2,
        backoff_seconds=0,
        jitter_seconds=0,
        client=http_client,
    )
    guest = ControlledFake()
    factory = RoleFactory(guest=[guest], operator=[ControlledFake()])

    with IntegrationHarness(
        tmp_path / "export-retry.db",
        factory,
        repository=repository,
        exporter=exporter,
    ) as harness:
        try:
            harness.create_call("call-export-retry", "hotel-export")
            with harness.client.websocket_connect("/ws/calls/call-export-retry/guest") as guest_ws:
                harness.start_channel(guest_ws, "hotel-export")
                guest_ws.send_bytes(PCM_FRAME)
                harness.wait_audio(guest, 1)
                final = adapter_event(
                    AdapterEventType.FINAL,
                    "Экспортируемый финал",
                    1,
                    item_id="export-item",
                    event_id="export-provider-event",
                )
                harness.emit(guest, final)
                harness.emit(guest, final)
                channel = harness.channel("call-export-retry", Role.GUEST)
                harness.wait_until(lambda: channel.duplicate_final_count == 1)
                harness.stop_channel(guest_ws)

            stored = harness.wait_stored_transcript(
                "call-export-retry",
                "hotel-export",
                lambda transcript: (
                    len(transcript.utterances) == 1
                    and transcript.utterances[0].export_status is ExportStatus.SENT
                ),
            )
            assert len(requests) == 2
            assert len(stored.utterances) == 1
            assert stored.utterances[0].export_attempts == 2
            domain_final = transcript_events(harness.events.events, final=True)
            assert len(domain_final) == 1
            expected_key = idempotency_key(domain_final[0])
            assert {request.headers["Idempotency-Key"] for request in requests} == {expected_key}
            payloads = [json.loads(request.content) for request in requests]
            assert payloads[0] == payloads[1]
            assert payloads[0]["event_type"] == "guest_utterance_completed"
            body = harness.transcript("call-export-retry", "hotel-export")
            assert len(utterances(body)) == 1
            assert utterances(body)[0]["export_status"] == "sent"
        finally:
            harness.run(http_client.aclose)


def test_full_path_export_failure_keeps_local_final_and_marks_delivery_failed(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503)

    repository = SQLiteTranscriptRepository(tmp_path / "export-failure.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    export_metrics = MetricsRegistry()
    exporter = HTTPSalvadorExporter(
        endpoint="http://salvador.test/events",
        status_store=repository,
        max_attempts=1,
        backoff_seconds=0,
        jitter_seconds=0,
        metrics=export_metrics,
        client=http_client,
    )
    guest = ControlledFake()
    factory = RoleFactory(guest=[guest], operator=[ControlledFake()])

    with IntegrationHarness(
        tmp_path / "export-failure.db",
        factory,
        repository=repository,
        exporter=exporter,
    ) as harness:
        try:
            harness.create_call("call-export-failure", "hotel-export")
            with harness.client.websocket_connect(
                "/ws/calls/call-export-failure/guest"
            ) as guest_ws:
                harness.start_channel(guest_ws, "hotel-export")
                guest_ws.send_bytes(PCM_FRAME)
                harness.wait_audio(guest, 1)
                harness.emit(
                    guest,
                    adapter_event(
                        AdapterEventType.FINAL,
                        "Локальный финал не должен потеряться",
                        1,
                        item_id="durable-item",
                    ),
                )
                harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 1)
                harness.stop_channel(guest_ws)

            stored = harness.wait_stored_transcript(
                "call-export-failure",
                "hotel-export",
                lambda transcript: (
                    len(transcript.utterances) == 1
                    and transcript.utterances[0].export_status is ExportStatus.FAILED
                ),
            )
            assert len(requests) == 1
            assert len(stored.utterances) == 1
            assert stored.utterances[0].text == "Локальный финал не должен потеряться"
            assert stored.utterances[0].export_attempts == 1
            assert stored.utterances[0].export_error == "HTTP 503"
            body = harness.transcript("call-export-failure", "hotel-export")
            assert utterances(body)[0]["text"] == "Локальный финал не должен потеряться"
            assert utterances(body)[0]["export_status"] == "failed"
            assert export_metrics.counter_value("export_failures_total") == 1
            assert export_metrics.histogram_count("export_delivery_seconds") == 1
        finally:
            harness.run(http_client.aclose)
