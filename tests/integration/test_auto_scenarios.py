import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import cast

from salvador_transcriber.audio.pcm import encode_pcm16le
from salvador_transcriber.domain.events import GapReason
from salvador_transcriber.domain.state import ChannelState, Role
from salvador_transcriber.transcription.adapter import AdapterEventType
from tests.integration.helpers import (
    PCM_FRAME,
    ControlledFake,
    IntegrationHarness,
    RoleFactory,
    adapter_event,
    start_payload,
    transcript_events,
)


def utterances(body: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], body["utterances"])


def test_auto_s1_regular_conversation_reaches_sqlite_and_read_api(tmp_path: Path) -> None:
    guest = ControlledFake()
    operator = ControlledFake()
    factory = RoleFactory(guest=[guest], operator=[operator])

    with IntegrationHarness(tmp_path / "regular.db", factory) as harness:
        harness.create_call("call-regular", "hotel-regular")
        with (
            harness.client.websocket_connect("/ws/calls/call-regular/guest") as guest_ws,
            harness.client.websocket_connect("/ws/calls/call-regular/operator") as operator_ws,
        ):
            harness.start_channel(guest_ws, "hotel-regular")
            harness.start_channel(operator_ws, "hotel-regular")

            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(guest, 1)
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Гость",
                    1,
                    item_id="guest-item",
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=False)) == 1)
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Гость спрашивает о бронировании",
                    2,
                    item_id="guest-item",
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=False)) == 2)
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Гость спрашивает о бронировании",
                    3,
                    item_id="guest-item",
                    event_id="guest-final",
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 1)

            operator_ws.send_bytes(PCM_FRAME)
            operator_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(operator, 2)
            harness.emit(
                operator,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Оператор подтверждает бронирование",
                    1,
                    item_id="operator-item",
                    start_ms=20,
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 2)
            harness.stop_channel(guest_ws)

        guest_chain = [
            event for event in transcript_events(harness.events.events) if event.role is Role.GUEST
        ]
        assert [event.revision for event in guest_chain] == [1, 2, 3]
        assert len({event.utterance_id for event in guest_chain}) == 1
        assert len({event.sequence for event in guest_chain}) == 1
        assert guest_chain[-1].is_final is True

        body = harness.transcript("call-regular", "hotel-regular")
        stored = utterances(body)
        assert [item["role"] for item in stored] == ["guest", "operator"]
        assert [item["text"] for item in stored] == [
            "Гость спрашивает о бронировании",
            "Оператор подтверждает бронирование",
        ]
        assert [(item["start_ms"], item["end_ms"]) for item in stored] == [(0, 20), (20, 40)]
        assert len({item["utterance_id"] for item in stored}) == 2
        assert {item["export_status"] for item in stored} == {"disabled"}

        metrics = harness.app.state.metrics
        assert metrics.counter_value("transcript_partials_total", labels={"role": "guest"}) == 2
        assert metrics.counter_value("transcript_finals_total", labels={"role": "guest"}) == 1
        assert metrics.counter_value("transcript_finals_total", labels={"role": "operator"}) == 1
        # This checks internal timestamp plumbing only, not real OpenAI latency or the 2 s target.
        assert metrics.histogram_count("final_latency_seconds", labels={"role": "guest"}) == 1
        assert metrics.histogram_count("final_latency_seconds", labels={"role": "operator"}) == 1
        prometheus = harness.client.get("/metrics").text
        assert "salvador_final_latency_seconds_count" in prometheus
        for sensitive in (
            "call-regular",
            "hotel-regular",
            "Гость спрашивает",
            "Оператор подтверждает",
            "test-secret-key",
        ):
            assert sensitive not in prometheus


def test_auto_s2_overlapping_channels_keep_text_roles_and_timeline_independent(
    tmp_path: Path,
) -> None:
    guest = ControlledFake()
    operator = ControlledFake()
    factory = RoleFactory(guest=[guest], operator=[operator])
    guest_frame = encode_pcm16le([101] * 320)
    operator_frame = encode_pcm16le([202] * 320)

    with IntegrationHarness(tmp_path / "overlap.db", factory) as harness:
        harness.create_call("call-overlap", "hotel-overlap")
        with (
            harness.client.websocket_connect("/ws/calls/call-overlap/guest") as guest_ws,
            harness.client.websocket_connect("/ws/calls/call-overlap/operator") as operator_ws,
        ):
            harness.start_channel(guest_ws, "hotel-overlap")
            harness.start_channel(operator_ws, "hotel-overlap")
            barrier = threading.Barrier(2)

            def send_guest() -> None:
                barrier.wait()
                guest_ws.send_bytes(guest_frame)
                guest_ws.send_bytes(guest_frame)

            def send_operator() -> None:
                barrier.wait()
                operator_ws.send_bytes(operator_frame)
                operator_ws.send_bytes(operator_frame)

            with ThreadPoolExecutor(max_workers=2) as pool:
                guest_send = pool.submit(send_guest)
                operator_send = pool.submit(send_operator)
                guest_send.result(timeout=2)
                operator_send.result(timeout=2)
            harness.wait_audio(guest, 2)
            harness.wait_audio(operator, 2)

            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Речь гостя",
                    1,
                    item_id="guest-overlap",
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=False)) == 1)
            harness.emit(
                operator,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Речь оператора",
                    1,
                    item_id="operator-overlap",
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=False)) == 2)

            # Reverse network completion order relative to sequence allocation.
            harness.emit(
                operator,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Речь оператора",
                    2,
                    item_id="operator-overlap",
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 1)
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Речь гостя",
                    2,
                    item_id="guest-overlap",
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 2)
            harness.stop_channel(operator_ws)

        assert guest is not operator
        assert guest.audio_bytes == guest_frame * 2
        assert operator.audio_bytes == operator_frame * 2
        network_finals = transcript_events(harness.events.events, final=True)
        assert [event.role for event in network_finals] == [Role.OPERATOR, Role.GUEST]

        body = harness.transcript("call-overlap", "hotel-overlap")
        stored = utterances(body)
        assert [item["text"] for item in stored] == ["Речь гостя", "Речь оператора"]
        assert [item["role"] for item in stored] == ["guest", "operator"]
        assert [(item["start_ms"], item["end_ms"]) for item in stored] == [(0, 40), (0, 40)]
        assert [item["sequence"] for item in stored] == [1, 2]
        assert utterances(harness.transcript("call-overlap", "hotel-overlap")) == stored


def test_auto_s3_fake_server_vad_short_pause_and_completed_speech(tmp_path: Path) -> None:
    guest = ControlledFake()
    factory = RoleFactory(guest=[guest], operator=[ControlledFake()])

    with IntegrationHarness(tmp_path / "pauses.db", factory) as harness:
        harness.create_call("call-pauses", "hotel-pauses")
        with harness.client.websocket_connect("/ws/calls/call-pauses/guest") as guest_ws:
            harness.start_channel(guest_ws, "hotel-pauses")
            for _ in range(3):
                guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(guest, 3)

            # Scripted server-VAD result: a short internal pause does not split this utterance.
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Фраза с короткой паузой",
                    1,
                    item_id="short-pause",
                    start_ms=0,
                    end_ms=60,
                ),
            )
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Фраза с короткой паузой",
                    2,
                    item_id="short-pause",
                    start_ms=0,
                    end_ms=60,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 1)
            assert len(transcript_events(harness.events.events, final=True)) == 1

            for _ in range(3):
                guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(guest, 6)
            harness.emit(
                guest,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Реплика после длинной паузы",
                    1,
                    item_id="long-pause",
                    start_ms=100,
                    end_ms=120,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 2)
            harness.stop_channel(guest_ws)

        stored = utterances(harness.transcript("call-pauses", "hotel-pauses"))
        assert [item["text"] for item in stored] == [
            "Фраза с короткой паузой",
            "Реплика после длинной паузы",
        ]
        assert [(item["start_ms"], item["end_ms"]) for item in stored] == [(0, 60), (100, 120)]
        # These assertions validate orchestration of fake VAD events, not OpenAI VAD calibration.


def test_auto_s4_reconnect_preserves_finals_gap_and_operator_progress(tmp_path: Path) -> None:
    initial = ControlledFake()
    reconnected = ControlledFake(block_connect=True)
    operator = ControlledFake()
    factory = RoleFactory(guest=[initial, reconnected], operator=[operator])

    with IntegrationHarness(tmp_path / "reconnect.db", factory) as harness:
        harness.create_call("call-reconnect", "hotel-reconnect")
        with (
            harness.client.websocket_connect("/ws/calls/call-reconnect/guest") as guest_ws,
            harness.client.websocket_connect("/ws/calls/call-reconnect/operator") as operator_ws,
        ):
            harness.start_channel(guest_ws, "hotel-reconnect")
            harness.start_channel(operator_ws, "hotel-reconnect")
            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(initial, 1)
            harness.emit(
                initial,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Финал до разрыва",
                    1,
                    item_id="shared-provider-item",
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 1)

            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(initial, 2)
            harness.emit(
                initial,
                adapter_event(
                    AdapterEventType.PARTIAL,
                    "Незавершённый старый partial",
                    1,
                    item_id="old-partial",
                    start_ms=20,
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=False)) == 1)
            harness.fail(initial, RuntimeError("artificial disconnect"))
            harness.wait_event(reconnected.connect_started)
            channel = harness.channel("call-reconnect", Role.GUEST)
            assert channel.state is ChannelState.RECONNECTING

            guest_ws.send_bytes(PCM_FRAME)
            operator_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(operator, 1)
            harness.emit(
                operator,
                adapter_event(AdapterEventType.FINAL, "Оператор продолжает работу", 1),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 2)

            harness.release(reconnected.release_connect)
            harness.wait_until(
                lambda: channel.state is ChannelState.READY and channel.connection_generation == 2
            )
            harness.wait_audio(reconnected, 1)
            guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(reconnected, 2)
            harness.emit(
                reconnected,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Новый финал после reconnect",
                    1,
                    item_id="shared-provider-item",
                    start_ms=0,
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 3)
            harness.stop_channel(operator_ws)

        body = harness.transcript("call-reconnect", "hotel-reconnect")
        stored = utterances(body)
        guest_stored = [item for item in stored if item["role"] == "guest"]
        assert [item["text"] for item in guest_stored] == [
            "Финал до разрыва",
            "Новый финал после reconnect",
        ]
        assert [item["connection_generation"] for item in guest_stored] == [1, 2]
        assert len({item["utterance_id"] for item in guest_stored}) == 2
        assert all(item["text"] != "Незавершённый старый partial" for item in stored)
        assert any(item["text"] == "Оператор продолжает работу" for item in stored)
        gaps = cast(list[dict[str, object]], body["gaps"])
        assert any(
            gap["reason"] == GapReason.OPENAI_DISCONNECT
            and gap["role"] == "guest"
            and (gap["start_ms"], gap["end_ms"]) == (20, 40)
            for gap in gaps
        )
        # The streaming resampler retains one sample until more input or final flush arrives.
        assert initial.audio_bytes == (PCM_FRAME * 2)[:-2]
        assert initial.audio_bytes + reconnected.audio_bytes == PCM_FRAME * 4
        metrics = harness.app.state.metrics
        assert metrics.counter_value("disconnects_total", labels={"role": "guest"}) == 1
        assert metrics.counter_value("reconnect_successes_total", labels={"role": "guest"}) == 1


def test_auto_s5_two_simultaneous_calls_are_isolated_and_stop_independently(
    tmp_path: Path,
) -> None:
    a_guest = ControlledFake()
    a_operator = ControlledFake()
    b_guest = ControlledFake()
    b_operator = ControlledFake()
    factory = RoleFactory(
        guest=[a_guest, b_guest],
        operator=[a_operator, b_operator],
    )

    with IntegrationHarness(tmp_path / "two-calls.db", factory) as harness:
        harness.create_call("call-a", "hotel-a")
        harness.create_call("call-b", "hotel-b")
        with ExitStack() as stack:
            a_guest_ws = stack.enter_context(
                harness.client.websocket_connect("/ws/calls/call-a/guest")
            )
            a_operator_ws = stack.enter_context(
                harness.client.websocket_connect("/ws/calls/call-a/operator")
            )
            b_guest_ws = stack.enter_context(
                harness.client.websocket_connect("/ws/calls/call-b/guest")
            )
            b_operator_ws = stack.enter_context(
                harness.client.websocket_connect("/ws/calls/call-b/operator")
            )
            for websocket, hotel in (
                (a_guest_ws, "hotel-a"),
                (a_operator_ws, "hotel-a"),
                (b_guest_ws, "hotel-b"),
                (b_operator_ws, "hotel-b"),
            ):
                harness.start_channel(websocket, hotel)
            assert set(harness.app.state.registry.call_ids) == {"call-a", "call-b"}

            for websocket, fake in (
                (a_guest_ws, a_guest),
                (a_operator_ws, a_operator),
                (b_guest_ws, b_guest),
                (b_operator_ws, b_operator),
            ):
                websocket.send_bytes(PCM_FRAME)
                harness.wait_audio(fake, 1)

            for fake, text, item_id in (
                (a_guest, "Гость звонка A", "a-guest-1"),
                (a_operator, "Оператор звонка A", "a-operator-1"),
                (b_guest, "Гость звонка B", "b-guest-1"),
                (b_operator, "Оператор звонка B", "b-operator-1"),
            ):
                harness.emit(
                    fake,
                    adapter_event(AdapterEventType.FINAL, text, 1, item_id=item_id),
                )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 4)

            with harness.client.websocket_connect("/ws/calls/call-a/guest") as mismatched_ws:
                mismatched_ws.send_json(start_payload("hotel-b"))
                rejected = mismatched_ws.receive_json()
                assert rejected["type"] == "error"
                assert rejected["code"] == "start_rejected"

            harness.stop_channel(a_guest_ws)
            assert harness.app.state.registry.call_ids == ("call-b",)

            b_guest_ws.send_bytes(PCM_FRAME)
            harness.wait_audio(b_guest, 2)
            harness.emit(
                b_guest,
                adapter_event(
                    AdapterEventType.FINAL,
                    "Гость B продолжает после завершения A",
                    1,
                    item_id="b-guest-2",
                    start_ms=20,
                    end_ms=40,
                ),
            )
            harness.wait_events(lambda events: len(transcript_events(events, final=True)) == 5)
            harness.stop_channel(b_operator_ws)

        stored_a = utterances(harness.transcript("call-a", "hotel-a"))
        stored_b = utterances(harness.transcript("call-b", "hotel-b"))
        assert {item["text"] for item in stored_a} == {
            "Гость звонка A",
            "Оператор звонка A",
        }
        assert {item["text"] for item in stored_b} == {
            "Гость звонка B",
            "Оператор звонка B",
            "Гость B продолжает после завершения A",
        }
        assert [item["sequence"] for item in stored_b] == [1, 2, 3]
        assert all(item["role"] in {"guest", "operator"} for item in stored_a + stored_b)
        for event in transcript_events(harness.events.events, final=True):
            if event.call_id == "call-a":
                assert event.hotel_id == "hotel-a"
                assert "звонка B" not in event.text
                assert "Гость B" not in event.text
            else:
                assert event.call_id == "call-b"
                assert event.hotel_id == "hotel-b"
                assert "звонка A" not in event.text
        assert harness.client.get("/metrics/state").json()["active_calls"] == 0
