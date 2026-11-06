from pathlib import Path

from fastapi.testclient import TestClient

from salvador_transcriber.config import Settings
from salvador_transcriber.demo.events import DemoEventHub
from salvador_transcriber.domain.events import EventType, TranscriptEvent
from salvador_transcriber.domain.sink import InMemoryEventSink
from salvador_transcriber.domain.state import Role
from salvador_transcriber.main import create_app
from salvador_transcriber.transcription.adapter import AdapterEvent, AdapterEventType
from salvador_transcriber.transcription.fake import FakeTranscriber


def transcript_event(utterance_id: str, *, final: bool = False) -> TranscriptEvent:
    return TranscriptEvent(
        event_type=(EventType.TRANSCRIPT_FINAL if final else EventType.TRANSCRIPT_PARTIAL),
        call_id="call-1",
        hotel_id="hotel-1",
        utterance_id=utterance_id,
        role=Role.GUEST,
        sequence=1,
        revision=2 if final else 1,
        text="final" if final else "partial",
        is_final=final,
        start_ms=0,
        end_ms=20,
        connection_generation=1,
    )


def test_demo_page_is_packaged_and_has_two_independent_sources(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, sqlite_path=tmp_path / "demo.db"))
    with TestClient(app) as client:
        response = client.get("/demo")
        assert response.status_code == 200
        assert 'id="guestFile"' in response.text
        assert 'id="operatorFile"' in response.text
        assert "streamRole" in response.text
        assert "https://" not in response.text


async def test_demo_event_hub_isolated_and_bounded_drop_new() -> None:
    downstream = InMemoryEventSink()
    hub = DemoEventHub(downstream, queue_limit=1)
    async with hub.subscribe("call-1", "hotel-1") as queue:
        await hub.publish(transcript_event("guest_0001"))
        await hub.publish(transcript_event("guest_0001", final=True))
        assert queue.qsize() == 1
        assert (await queue.get()).utterance_id == "guest_0001"
        assert hub.dropped_events == 1
        assert len(downstream.events) == 2
    assert hub.subscriber_count == 0


def test_event_websocket_streams_partial_final_and_completion(tmp_path: Path) -> None:
    def factory(role: Role) -> FakeTranscriber:
        if role is Role.OPERATOR:
            return FakeTranscriber()
        return FakeTranscriber(
            [
                AdapterEvent(
                    event_type=AdapterEventType.PARTIAL,
                    text="промежуточный",
                    start_ms=0,
                    end_ms=20,
                ),
                AdapterEvent(
                    event_type=AdapterEventType.FINAL,
                    text="финальный",
                    revision=2,
                    start_ms=0,
                    end_ms=20,
                ),
            ]
        )

    app = create_app(
        Settings(_env_file=None, sqlite_path=tmp_path / "events.db"),
        transcriber_factory=factory,
    )
    with TestClient(app) as client:
        client.post("/calls", json={"call_id": "call-1", "hotel_id": "hotel-1"})
        with (
            client.websocket_connect("/ws/calls/call-1/events?hotel_id=hotel-1") as events,
            client.websocket_connect("/ws/calls/call-1/guest") as channel,
        ):
            channel.send_json(
                {
                    "type": "start",
                    "hotel_id": "hotel-1",
                    "audio": {"sample_rate": 16000, "channels": 1},
                }
            )
            assert channel.receive_json()["type"] == "ready"
            assert events.receive_json()["event_type"] == "transcript_partial"
            assert events.receive_json()["event_type"] == "transcript_final"
            channel.send_json({"type": "stop"})
            assert channel.receive_json()["code"] == "call_completed"
            assert events.receive_json()["event_type"] == "call_completed"


def test_provider_fault_endpoint_is_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, sqlite_path=tmp_path / "fault.db"))
    with TestClient(app) as client:
        response = client.post(
            "/demo/calls/missing/guest/provider-disconnect",
            params={"hotel_id": "hotel-1"},
        )
        assert response.status_code == 404
