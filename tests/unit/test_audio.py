import pytest
from pydantic import ValidationError

from salvador_transcriber.audio.pcm import (
    AudioTimeline,
    AudioValidationError,
    PCMFormat,
    PCMFrameSplitter,
    encode_pcm16le,
    validate_pcm16le,
)
from salvador_transcriber.audio.resampler import StatefulLinearResampler


def test_pcm_validation_rejects_unaligned_and_oversized_payloads() -> None:
    with pytest.raises(AudioValidationError, match="even number") as unaligned:
        validate_pcm16le(b"\x00", sample_rate=16000)
    assert unaligned.value.code == "unaligned_pcm16"

    with pytest.raises(AudioValidationError, match="exceeds") as oversized:
        validate_pcm16le(b"\x00\x00" * 3, sample_rate=16000, max_bytes=4)
    assert oversized.value.code == "audio_message_too_large"


def test_pcm_format_rejects_stereo_and_unsupported_rate() -> None:
    with pytest.raises(ValidationError):
        PCMFormat(sample_rate=16000, channels=2)
    with pytest.raises(ValidationError, match="unsupported sample rate"):
        PCMFormat(sample_rate=44100)


def test_frame_splitter_preserves_remainder_and_sample_order() -> None:
    pcm_format = PCMFormat(sample_rate=8000, frame_duration_ms=10)
    splitter = PCMFrameSplitter(pcm_format)
    first = encode_pcm16le(list(range(50)))
    second = encode_pcm16le(list(range(50, 100)))

    assert splitter.push(first) == []
    frames = splitter.push(second)

    assert len(frames) == 1
    assert frames[0] == encode_pcm16le(list(range(80)))
    assert splitter.pending_bytes == 40
    assert splitter.flush() == encode_pcm16le(list(range(80, 100)))
    assert splitter.flush() == b""


@pytest.mark.parametrize(
    ("input_rate", "input_samples"),
    [(8000, 80), (16000, 160), (48000, 480)],
)
def test_resampler_duration_and_chunk_invariance(input_rate: int, input_samples: int) -> None:
    source = encode_pcm16le([((index * 7919) % 65536) - 32768 for index in range(input_samples)])
    whole_resampler = StatefulLinearResampler(input_rate, 16000)
    chunked_resampler = StatefulLinearResampler(input_rate, 16000)

    whole = whole_resampler.push(source) + whole_resampler.flush()
    chunks = []
    for offset in range(0, len(source), 14):
        chunks.append(chunked_resampler.push(source[offset : offset + 14]))
    chunks.append(chunked_resampler.flush())
    chunked = b"".join(chunks)

    assert len(whole) == 160 * 2
    assert chunked == whole
    assert chunked_resampler.expected_output_samples == 160


def test_resampler_flush_emits_tail_and_is_idempotent() -> None:
    resampler = StatefulLinearResampler(8000, 16000)

    assert resampler.push(encode_pcm16le([123])) == b""
    tail = resampler.flush()

    assert tail == encode_pcm16le([123, 123])
    assert resampler.flush() == b""
    with pytest.raises(AudioValidationError, match="after flush"):
        resampler.push(b"\x00\x00")


def test_audio_timeline_uses_integer_sample_count_without_drift() -> None:
    timeline = AudioTimeline(8000)

    for _ in range(8001):
        timeline.consume(b"\x00\x00")

    assert timeline.samples == 8001
    assert timeline.position_ms == 1000
    assert timeline.consume(b"\x00\x00" * 7999) == (1000, 2000)
