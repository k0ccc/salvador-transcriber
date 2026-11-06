import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from salvador_transcriber.domain.events import (
    AudioGapEvent,
    CallCompletedEvent,
    ChannelStatusEvent,
    EventType,
    GapReason,
    TranscriptEvent,
)
from salvador_transcriber.domain.sink import InMemoryEventSink
from salvador_transcriber.domain.state import ChannelState, Role
from salvador_transcriber.export.sink import DurableEventSink
from salvador_transcriber.storage.repository import (
    CallContextMismatchError,
    ExportStatus,
    FinalSaveResult,
    TranscriptNotFoundError,
)
from salvador_transcriber.storage.sqlite import SQLiteTranscriptRepository


def transcript_event(
    *,
    call_id: str = "call-a",
    hotel_id: str = "hotel-a",
    utterance_id: str = "guest_0001",
    role: Role = Role.GUEST,
    sequence: int = 1,
    text: str = "final text",
    start_ms: int = 100,
    end_ms: int = 200,
    event_id: str = "event-1",
    is_final: bool = True,
) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=event_id,
        event_type=EventType.TRANSCRIPT_FINAL if is_final else EventType.TRANSCRIPT_PARTIAL,
        call_id=call_id,
        hotel_id=hotel_id,
        utterance_id=utterance_id,
        role=role,
        sequence=sequence,
        revision=1,
        text=text,
        is_final=is_final,
        start_ms=start_ms,
        end_ms=end_ms,
        connection_generation=1,
    )


async def test_partial_and_duplicate_final_do_not_create_extra_transcript_rows(
    tmp_path: Path,
) -> None:
    async with SQLiteTranscriptRepository(tmp_path / "transcript.db") as repository:
        downstream = InMemoryEventSink()
        sink = DurableEventSink(repository=repository, downstream=downstream)
        partial = transcript_event(is_final=False, text="partial")
        final = transcript_event()

        await sink.publish(partial)
        await sink.publish(final)
        await sink.publish(final.model_copy(update={"event_id": "provider-duplicate"}))
        await sink.publish(
            final.model_copy(update={"event_id": "conflict", "text": "changed"}),
        )

        transcript = await repository.get_transcript("call-a", "hotel-a")
        assert len(transcript.utterances) == 1
        assert transcript.utterances[0].text == "final text"
        assert transcript.utterances[0].export_status is ExportStatus.DISABLED
        assert [event.event_id for event in downstream.events] == [
            partial.event_id,
            final.event_id,
        ]


async def test_calls_hotels_roles_and_deterministic_timeline_are_isolated(
    tmp_path: Path,
) -> None:
    async with SQLiteTranscriptRepository(tmp_path / "transcript.db") as repository:
        events = [
            transcript_event(
                utterance_id="operator_0001",
                role=Role.OPERATOR,
                sequence=2,
                text="operator at 100",
                start_ms=100,
                end_ms=250,
                event_id="event-operator",
            ),
            transcript_event(
                utterance_id="guest_0002",
                sequence=3,
                text="early overlapping guest",
                start_ms=50,
                end_ms=300,
                event_id="event-early",
            ),
            transcript_event(
                utterance_id="guest_0001",
                sequence=1,
                text="guest at 100",
                start_ms=100,
                end_ms=150,
                event_id="event-guest",
            ),
            transcript_event(
                utterance_id="guest_0000",
                sequence=1,
                text="utterance tie breaker",
                start_ms=100,
                end_ms=150,
                event_id="event-tie-breaker",
            ),
            transcript_event(
                call_id="call-b",
                hotel_id="hotel-b",
                text="other call",
                event_id="event-other-call",
            ),
        ]
        for event in events:
            assert (await repository.save_final(event)).created is True

        call_a = await repository.get_transcript("call-a", "hotel-a")
        call_b = await repository.get_transcript("call-b", "hotel-b")
        assert [item.text for item in call_a.utterances] == [
            "early overlapping guest",
            "utterance tie breaker",
            "guest at 100",
            "operator at 100",
        ]
        assert [item.role for item in call_a.utterances] == [
            Role.GUEST,
            Role.GUEST,
            Role.GUEST,
            Role.OPERATOR,
        ]
        assert [item.text for item in call_b.utterances] == ["other call"]
        with pytest.raises(TranscriptNotFoundError):
            await repository.get_transcript("call-a", "hotel-b")
        with pytest.raises(CallContextMismatchError):
            await repository.save_final(
                transcript_event(hotel_id="hotel-b", event_id="wrong-hotel"),
            )


async def test_sqlite_restart_recovers_finals_gaps_and_service_status(tmp_path: Path) -> None:
    path = tmp_path / "transcript.db"
    final = transcript_event()
    gap = AudioGapEvent(
        event_id="gap-1",
        call_id="call-a",
        hotel_id="hotel-a",
        role=Role.GUEST,
        start_ms=300,
        end_ms=400,
        reason=GapReason.QUEUE_OVERFLOW,
        connection_generation=1,
        dropped_frames=2,
        dropped_bytes=640,
    )
    status = ChannelStatusEvent(
        event_id="status-1",
        call_id="call-a",
        hotel_id="hotel-a",
        role=Role.GUEST,
        status=ChannelState.CLOSED,
        connection_generation=1,
    )
    completed = CallCompletedEvent(
        event_id="completed-1",
        call_id="call-a",
        hotel_id="hotel-a",
    )
    repository = SQLiteTranscriptRepository(path)
    await repository.start()
    await repository.save_final(final)
    await repository.save_event(gap)
    await repository.save_event(status)
    await repository.save_event(completed)
    await repository.close()

    async with SQLiteTranscriptRepository(path) as recovered:
        transcript = await recovered.get_transcript("call-a", "hotel-a")
        assert [item.text for item in transcript.utterances] == ["final text"]
        assert transcript.gaps[0].reason is GapReason.QUEUE_OVERFLOW
        assert transcript.gaps[0].dropped_bytes == 640
        assert {item.event_type for item in transcript.service_events} == {
            EventType.CHANNEL_STATUS,
            EventType.CALL_COMPLETED,
        }


async def test_sqlite_work_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with SQLiteTranscriptRepository(tmp_path / "transcript.db") as repository:
        original = repository._save_final

        def slow_save(connection: sqlite3.Connection, event: TranscriptEvent) -> FinalSaveResult:
            time.sleep(0.1)
            return original(connection, event)

        monkeypatch.setattr(repository, "_save_final", slow_save)
        write = asyncio.create_task(repository.save_final(transcript_event()))
        await asyncio.sleep(0)
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.03)
        assert (await write).created is True


async def test_close_drains_accepted_final_writes(tmp_path: Path) -> None:
    path = tmp_path / "transcript.db"
    repository = SQLiteTranscriptRepository(path)
    await repository.start()
    tasks = [
        asyncio.create_task(
            repository.save_final(
                transcript_event(
                    utterance_id=f"guest_{index:04d}",
                    sequence=index,
                    event_id=f"event-{index}",
                ),
            ),
        )
        for index in range(1, 21)
    ]
    await asyncio.sleep(0)

    await repository.close()
    results = await asyncio.gather(*tasks)
    assert all(result.created for result in results)

    async with SQLiteTranscriptRepository(path) as recovered:
        transcript = await recovered.get_transcript("call-a", "hotel-a")
        assert len(transcript.utterances) == 20
