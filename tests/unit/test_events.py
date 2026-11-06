import json

import pytest
from pydantic import TypeAdapter, ValidationError

from salvador_transcriber.domain.events import (
    AudioGapEvent,
    Event,
    EventType,
    GapReason,
    TranscriptEvent,
)
from salvador_transcriber.domain.state import Role


def transcript_payload() -> dict[str, object]:
    return {
        "event_id": "event-1",
        "event_type": EventType.TRANSCRIPT_FINAL,
        "call_id": "call-1",
        "hotel_id": "hotel-1",
        "utterance_id": "guest_0001",
        "role": Role.GUEST,
        "sequence": 1,
        "revision": 2,
        "text": "Здравствуйте",
        "is_final": True,
        "start_ms": 100,
        "end_ms": 900,
        "connection_generation": 1,
        "occurred_at": "2026-01-01T00:00:00Z",
    }


def test_transcript_event_json_round_trip() -> None:
    event = TranscriptEvent.model_validate_json(json.dumps(transcript_payload(), default=str))
    restored = TypeAdapter(Event).validate_json(event.model_dump_json())

    assert isinstance(restored, TranscriptEvent)
    assert restored == event
    assert restored.event_id == "event-1"
    assert restored.role is Role.GUEST
    assert restored.is_final is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "caller"),
        ("start_ms", -1),
        ("end_ms", 99),
        ("sequence", 0),
        ("revision", 0),
    ],
)
def test_transcript_event_rejects_invalid_values(field: str, value: object) -> None:
    payload = transcript_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        TranscriptEvent.model_validate_json(json.dumps(payload, default=str))


def test_transcript_event_rejects_inconsistent_final_flags() -> None:
    payload = transcript_payload()
    payload["event_type"] = EventType.TRANSCRIPT_PARTIAL

    with pytest.raises(ValidationError, match="event_type and is_final"):
        TranscriptEvent.model_validate_json(json.dumps(payload, default=str))


def test_transcript_event_rejects_unknown_fields() -> None:
    payload = transcript_payload()
    payload["provider_item_id"] = "not-part-of-the-domain-contract"

    with pytest.raises(ValidationError):
        TranscriptEvent.model_validate_json(json.dumps(payload, default=str))


def test_audio_gap_allows_unknown_boundaries() -> None:
    event = AudioGapEvent(
        call_id="call-1",
        hotel_id="hotel-1",
        role=Role.OPERATOR,
        reason=GapReason.QUEUE_OVERFLOW,
        connection_generation=1,
        dropped_frames=2,
        dropped_bytes=640,
    )

    assert event.start_ms is None
    assert event.end_ms is None
    assert AudioGapEvent.model_validate_json(event.model_dump_json()) == event
