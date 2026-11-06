from pathlib import Path

from fastapi.testclient import TestClient

from salvador_transcriber.config import Settings
from salvador_transcriber.domain.state import Role
from salvador_transcriber.main import create_app
from salvador_transcriber.transcription.adapter import AdapterEvent, AdapterEventType
from salvador_transcriber.transcription.fake import FakeTranscriber


def test_http_and_websocket_lifecycle(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, sqlite_path=tmp_path / "api.db"))

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json() == {"status": "ok", "backend": "fake"}
        created = client.post("/calls", json={"call_id": "call-1", "hotel_id": "hotel-1"})
        assert created.status_code == 201
        assert created.json()["state"] == "created"

        with client.websocket_connect("/ws/calls/call-1/guest") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "hotel_id": "hotel-1",
                    "audio": {
                        "encoding": "pcm16le",
                        "sample_rate": 16000,
                        "channels": 1,
                        "frame_duration_ms": 20,
                    },
                },
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(b"\x00\x00" * 320)
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"
            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["code"] == "call_completed"

        assert app.state.registry.call_ids == ()


def test_websocket_requires_start_and_immutable_hotel(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, sqlite_path=tmp_path / "api.db"))

    with TestClient(app) as client:
        client.post("/calls", json={"call_id": "call-1", "hotel_id": "hotel-1"})
        with client.websocket_connect("/ws/calls/call-1/operator") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "hotel_id": "hotel-2",
                    "audio": {"sample_rate": 16000, "channels": 1},
                },
            )
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "start_rejected"


def test_final_transcript_read_endpoint_is_hotel_scoped(tmp_path: Path) -> None:
    def factory(role: Role) -> FakeTranscriber:
        return FakeTranscriber(
            [
                AdapterEvent(
                    event_type=AdapterEventType.PARTIAL,
                    text="partial",
                    revision=1,
                    start_ms=10,
                    end_ms=20,
                ),
                AdapterEvent(
                    event_type=AdapterEventType.FINAL,
                    text="stored final",
                    revision=2,
                    start_ms=10,
                    end_ms=20,
                ),
            ],
        )

    app = create_app(
        Settings(_env_file=None, sqlite_path=tmp_path / "api.db"),
        transcriber_factory=factory,
    )
    with TestClient(app) as client:
        client.post("/calls", json={"call_id": "call-1", "hotel_id": "hotel-1"})
        with client.websocket_connect("/ws/calls/call-1/guest") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "hotel_id": "hotel-1",
                    "audio": {"sample_rate": 16000, "channels": 1},
                },
            )
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["code"] == "call_completed"

        response = client.get("/calls/call-1/transcript", params={"hotel_id": "hotel-1"})
        assert response.status_code == 200
        body = response.json()
        assert [item["text"] for item in body["utterances"]] == ["stored final"]
        assert body["utterances"][0]["export_status"] == "disabled"
        assert body["utterances"][0]["start_ms"] == 10
        assert body["utterances"][0]["end_ms"] == 20
        assert {item["event_type"] for item in body["service_events"]} == {"call_completed"}
        assert (
            client.get(
                "/calls/call-1/transcript",
                params={"hotel_id": "hotel-2"},
            ).status_code
            == 404
        )
