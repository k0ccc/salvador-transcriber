import json
from pathlib import Path

import httpx

from salvador_transcriber.domain.events import EventType, TranscriptEvent
from salvador_transcriber.domain.sink import InMemoryEventSink
from salvador_transcriber.domain.state import Role
from salvador_transcriber.export.http import HTTPSalvadorExporter, idempotency_key
from salvador_transcriber.export.sink import DurableEventSink
from salvador_transcriber.storage.repository import ExportStatus
from salvador_transcriber.storage.sqlite import SQLiteTranscriptRepository


def final_event(
    *,
    utterance_id: str = "guest_0001",
    role: Role = Role.GUEST,
    sequence: int = 1,
    text: str = "hello",
    event_id: str = "event-1",
) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=event_id,
        event_type=EventType.TRANSCRIPT_FINAL,
        call_id="call-1",
        hotel_id="hotel-1",
        utterance_id=utterance_id,
        role=role,
        sequence=sequence,
        revision=1,
        text=text,
        is_final=True,
        start_ms=100,
        end_ms=900,
        connection_generation=1,
    )


async def test_export_payload_roles_idempotency_duplicate_and_shutdown_drain(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202)

    transport = httpx.MockTransport(handler)
    async with (
        SQLiteTranscriptRepository(tmp_path / "transcript.db") as repository,
        httpx.AsyncClient(transport=transport) as client,
    ):
        exporter = HTTPSalvadorExporter(
            endpoint="http://salvador.test/events",
            status_store=repository,
            max_attempts=1,
            backoff_seconds=0,
            jitter_seconds=0,
            client=client,
        )
        await exporter.start()
        downstream = InMemoryEventSink()
        sink = DurableEventSink(
            repository=repository,
            downstream=downstream,
            exporter=exporter,
        )
        guest = final_event()
        operator = final_event(
            utterance_id="operator_0001",
            role=Role.OPERATOR,
            sequence=2,
            text="operator answer",
            event_id="event-2",
        )
        empty = final_event(
            utterance_id="guest_0002",
            sequence=3,
            text=" ",
            event_id="event-3",
        )
        await sink.publish(guest)
        await sink.publish(operator)
        await sink.publish(empty)
        await sink.publish(guest.model_copy(update={"event_id": "duplicate"}))
        await exporter.close()

        assert len(requests) == 2
        payloads = [json.loads(request.content) for request in requests]
        assert [payload["event_type"] for payload in payloads] == [
            "guest_utterance_completed",
            "operator_utterance_completed",
        ]
        assert payloads[0] == {
            "event_type": "guest_utterance_completed",
            "call_id": "call-1",
            "hotel_id": "hotel-1",
            "utterance_id": "guest_0001",
            "role": "guest",
            "sequence": 1,
            "text": "hello",
            "start_ms": 100,
            "end_ms": 900,
        }
        assert requests[0].headers["Idempotency-Key"] == idempotency_key(guest)
        assert requests[1].headers["Idempotency-Key"] == idempotency_key(operator)
        transcript = await repository.get_transcript("call-1", "hotel-1")
        assert len(transcript.utterances) == 3
        assert [item.export_status for item in transcript.utterances] == [
            ExportStatus.SENT,
            ExportStatus.SENT,
            ExportStatus.SKIPPED_EMPTY,
        ]
        assert (
            len([event for event in downstream.events if isinstance(event, TranscriptEvent)]) == 3
        )


async def test_http_retry_uses_same_key_and_records_success(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200)

    async with (
        SQLiteTranscriptRepository(tmp_path / "transcript.db") as repository,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
    ):
        exporter = HTTPSalvadorExporter(
            endpoint="http://salvador.test/events",
            status_store=repository,
            max_attempts=2,
            backoff_seconds=0,
            jitter_seconds=0,
            client=client,
        )
        await exporter.start()
        event = final_event()
        await repository.save_final(event)
        await exporter.publish(event)
        await exporter.close()

        assert len(requests) == 2
        assert {request.headers["Idempotency-Key"] for request in requests} == {
            idempotency_key(event),
        }
        transcript = await repository.get_transcript("call-1", "hotel-1")
        assert transcript.utterances[0].export_status is ExportStatus.SENT
        assert transcript.utterances[0].export_attempts == 2


async def test_timeout_keeps_local_final_and_marks_delivery_failed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout", request=request)

    async with (
        SQLiteTranscriptRepository(tmp_path / "transcript.db") as repository,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
    ):
        exporter = HTTPSalvadorExporter(
            endpoint="http://salvador.test/events",
            status_store=repository,
            timeout_seconds=0.01,
            max_attempts=1,
            backoff_seconds=0,
            jitter_seconds=0,
            client=client,
        )
        await exporter.start()
        downstream = InMemoryEventSink()
        sink = DurableEventSink(
            repository=repository,
            downstream=downstream,
            exporter=exporter,
        )
        await sink.publish(final_event())
        await exporter.close()

        transcript = await repository.get_transcript("call-1", "hotel-1")
        assert len(transcript.utterances) == 1
        assert transcript.utterances[0].text == "hello"
        assert transcript.utterances[0].export_status is ExportStatus.FAILED
        assert transcript.utterances[0].export_attempts == 1
        assert transcript.utterances[0].export_error == "ConnectTimeout"
        assert len(downstream.events) == 1
