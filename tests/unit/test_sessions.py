from collections.abc import Iterable

import pytest

from salvador_transcriber.audio.pcm import PCMFormat, encode_pcm16le
from salvador_transcriber.domain.events import CallCompletedEvent, TranscriptEvent
from salvador_transcriber.domain.sink import InMemoryEventSink
from salvador_transcriber.domain.state import CallState, ChannelState, Role
from salvador_transcriber.sessions.call_session import CallSession
from salvador_transcriber.sessions.errors import HotelMismatchError, RoleOwnershipError
from salvador_transcriber.sessions.registry import CallRegistry
from salvador_transcriber.transcription.adapter import AdapterEvent, AdapterEventType
from salvador_transcriber.transcription.fake import FakeTranscriber


def adapter_event(
    event_type: AdapterEventType,
    text: str,
    revision: int,
    *,
    start_ms: int = 0,
    end_ms: int = 100,
    provider_item_id: str | None = None,
) -> AdapterEvent:
    return AdapterEvent(
        event_type=event_type,
        text=text,
        revision=revision,
        start_ms=start_ms,
        end_ms=end_ms,
        provider_item_id=provider_item_id,
    )


def make_call(
    call_id: str,
    hotel_id: str,
    scripts: dict[Role, Iterable[AdapterEvent]],
    sink: InMemoryEventSink,
    fakes: dict[Role, FakeTranscriber],
) -> CallSession:
    def factory(role: Role) -> FakeTranscriber:
        fake = FakeTranscriber(scripts.get(role, ()))
        fakes[role] = fake
        return fake

    return CallSession(
        call_id=call_id,
        hotel_id=hotel_id,
        transcriber_factory=factory,
        event_sink=sink,
        target_sample_rate=16000,
        max_audio_message_bytes=65536,
    )


def transcripts(sink: InMemoryEventSink) -> list[TranscriptEvent]:
    return [event for event in sink.events if isinstance(event, TranscriptEvent)]


async def test_partial_final_chain_and_roles_are_independent() -> None:
    sink = InMemoryEventSink()
    fakes: dict[Role, FakeTranscriber] = {}
    call = make_call(
        "call-1",
        "hotel-1",
        {
            Role.GUEST: [
                adapter_event(AdapterEventType.PARTIAL, "Здра", 1),
                adapter_event(AdapterEventType.PARTIAL, "Здравствуйте", 2),
                adapter_event(AdapterEventType.FINAL, "Здравствуйте", 3),
            ],
            Role.OPERATOR: [
                adapter_event(AdapterEventType.FINAL, "Добрый день", 1),
            ],
        },
        sink,
        fakes,
    )
    pcm_format = PCMFormat(sample_rate=16000)
    await call.start_channel(Role.GUEST, "hotel-1", pcm_format)
    await call.start_channel(Role.OPERATOR, "hotel-1", pcm_format)
    await call.close()

    events = transcripts(sink)
    guest = [event for event in events if event.role is Role.GUEST]
    operator = [event for event in events if event.role is Role.OPERATOR]
    assert [event.revision for event in guest] == [1, 2, 3]
    assert len({event.utterance_id for event in guest}) == 1
    assert len({event.sequence for event in guest}) == 1
    assert guest[-1].is_final is True
    assert len(operator) == 1
    assert operator[0].utterance_id == "operator_0001"
    assert operator[0].sequence != guest[0].sequence
    assert all(fake.closed for fake in fakes.values())


async def test_stale_partial_and_duplicate_finals_are_not_published() -> None:
    sink = InMemoryEventSink()
    fakes: dict[Role, FakeTranscriber] = {}
    call = make_call(
        "call-1",
        "hotel-1",
        {
            Role.GUEST: [
                adapter_event(AdapterEventType.PARTIAL, "new", 2),
                adapter_event(AdapterEventType.PARTIAL, "old", 1),
                adapter_event(AdapterEventType.FINAL, "authoritative", 3),
                adapter_event(AdapterEventType.FINAL, "authoritative", 3),
                adapter_event(AdapterEventType.FINAL, "conflicting", 4),
            ],
        },
        sink,
        fakes,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCMFormat(sample_rate=16000))
    await call.close()

    events = transcripts(sink)
    assert [event.text for event in events] == ["new", "authoritative"]
    assert events[-1].text == "authoritative"
    channel = call.channels[Role.GUEST]
    assert channel.duplicate_final_count == 2
    assert channel.final_conflict_count == 1


async def test_provider_items_create_separate_utterances_and_keep_final_dedup() -> None:
    sink = InMemoryEventSink()
    fakes: dict[Role, FakeTranscriber] = {}
    call = make_call(
        "call-1",
        "hotel-1",
        {
            Role.GUEST: [
                adapter_event(AdapterEventType.FINAL, "first", 1, provider_item_id="item-1"),
                adapter_event(AdapterEventType.FINAL, "first", 2, provider_item_id="item-1"),
                adapter_event(AdapterEventType.FINAL, "second", 1, provider_item_id="item-2"),
                adapter_event(AdapterEventType.PARTIAL, "late", 2, provider_item_id="item-2"),
                adapter_event(AdapterEventType.FINAL, "third", 1, provider_item_id="item-3"),
            ],
        },
        sink,
        fakes,
    )
    await call.start_channel(Role.GUEST, "hotel-1", PCMFormat(sample_rate=16000))
    await call.close()

    events = transcripts(sink)
    assert [event.text for event in events] == ["first", "second", "third"]
    assert [event.utterance_id for event in events] == [
        "guest_0001",
        "guest_0002",
        "guest_0003",
    ]
    assert call.channels[Role.GUEST].duplicate_final_count == 1


async def test_two_calls_do_not_share_identifiers_audio_or_sequence() -> None:
    sink = InMemoryEventSink()
    fakes_a: dict[Role, FakeTranscriber] = {}
    fakes_b: dict[Role, FakeTranscriber] = {}
    call_a = make_call(
        "call-a",
        "hotel-a",
        {Role.GUEST: [adapter_event(AdapterEventType.FINAL, "A", 1)]},
        sink,
        fakes_a,
    )
    call_b = make_call(
        "call-b",
        "hotel-b",
        {Role.GUEST: [adapter_event(AdapterEventType.FINAL, "B", 1)]},
        sink,
        fakes_b,
    )
    pcm_format = PCMFormat(sample_rate=16000)
    await call_a.start_channel(Role.GUEST, "hotel-a", pcm_format)
    await call_b.start_channel(Role.GUEST, "hotel-b", pcm_format)
    audio_a = encode_pcm16le([1] * 320)
    audio_b = encode_pcm16le([2] * 320)
    await call_a.accept_audio(Role.GUEST, "hotel-a", audio_a)
    await call_b.accept_audio(Role.GUEST, "hotel-b", audio_b)
    await call_a.close()

    assert fakes_a[Role.GUEST].audio_bytes == audio_a
    assert fakes_b[Role.GUEST].audio_bytes == audio_b[:-2]
    assert call_a.state is CallState.CLOSED
    assert call_b.state is CallState.ACTIVE
    await call_b.accept_audio(Role.GUEST, "hotel-b", audio_b)
    await call_b.close()
    assert fakes_b[Role.GUEST].audio_bytes == audio_b + audio_b

    events = transcripts(sink)
    assert {(event.call_id, event.hotel_id) for event in events} == {
        ("call-a", "hotel-a"),
        ("call-b", "hotel-b"),
    }
    assert all(event.utterance_id == "guest_0001" for event in events)
    assert all(event.sequence == 1 for event in events)


async def test_close_is_idempotent_and_publishes_call_completed_once() -> None:
    sink = InMemoryEventSink()
    fakes: dict[Role, FakeTranscriber] = {}
    call = make_call("call-1", "hotel-1", {}, sink, fakes)
    pcm_format = PCMFormat(sample_rate=16000)
    await call.start_channel(Role.GUEST, "hotel-1", pcm_format)
    await call.start_channel(Role.OPERATOR, "hotel-1", pcm_format)

    await call.close()
    await call.close()

    completed = [event for event in sink.events if isinstance(event, CallCompletedEvent)]
    assert len(completed) == 1
    assert call.state is CallState.CLOSED
    assert all(channel.state is ChannelState.CLOSED for channel in call.channels.values())
    assert all(fake.input_finished for fake in fakes.values())


async def test_registry_rejects_hotel_mismatch_and_role_conflict() -> None:
    sink = InMemoryEventSink()
    fakes: dict[Role, FakeTranscriber] = {}

    def factory(role: Role) -> FakeTranscriber:
        fake = FakeTranscriber()
        fakes[role] = fake
        return fake

    registry = CallRegistry(
        transcriber_factory=factory,
        event_sink=sink,
        target_sample_rate=16000,
        max_audio_message_bytes=65536,
    )
    call = await registry.create_call("call-1", "hotel-1")
    pcm_format = PCMFormat(sample_rate=16000)
    await call.start_channel(Role.GUEST, "hotel-1", pcm_format)

    with pytest.raises(RoleOwnershipError):
        await call.start_channel(Role.GUEST, "hotel-1", pcm_format)
    with pytest.raises(HotelMismatchError):
        await call.start_channel(Role.OPERATOR, "hotel-2", pcm_format)

    await registry.close_call("call-1")
    assert registry.call_ids == ()
