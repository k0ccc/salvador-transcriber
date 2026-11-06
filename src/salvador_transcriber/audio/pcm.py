import sys
from array import array
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_INPUT_SAMPLE_RATES = frozenset({8000, 16000, 48000})
PCM16_BYTES_PER_SAMPLE = 2


class AudioValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PCMFormat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    encoding: Literal["pcm16le"] = "pcm16le"
    sample_rate: int
    channels: Literal[1] = 1
    frame_duration_ms: int = Field(default=20, ge=10, le=1000)

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        if value not in SUPPORTED_INPUT_SAMPLE_RATES:
            supported = ", ".join(str(rate) for rate in sorted(SUPPORTED_INPUT_SAMPLE_RATES))
            raise ValueError(f"unsupported sample rate {value}; expected one of: {supported}")
        return value

    @model_validator(mode="after")
    def validate_frame_alignment(self) -> Self:
        if (self.sample_rate * self.frame_duration_ms) % 1000 != 0:
            raise ValueError("frame duration must contain a whole number of PCM samples")
        return self

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_duration_ms // 1000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * PCM16_BYTES_PER_SAMPLE

    def validate_payload(self, data: bytes, *, max_bytes: int | None = None) -> None:
        validate_pcm16le(data, sample_rate=self.sample_rate, max_bytes=max_bytes)


def validate_pcm16le(
    data: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    max_bytes: int | None = None,
) -> None:
    if not isinstance(data, bytes):
        raise AudioValidationError("invalid_pcm_type", "PCM payload must be bytes")
    if channels != 1:
        raise AudioValidationError("unsupported_channels", "only mono PCM is supported")
    if sample_rate not in SUPPORTED_INPUT_SAMPLE_RATES:
        raise AudioValidationError(
            "unsupported_sample_rate",
            f"unsupported input sample rate: {sample_rate}",
        )
    if len(data) % PCM16_BYTES_PER_SAMPLE != 0:
        raise AudioValidationError(
            "unaligned_pcm16",
            "PCM16 little-endian payload must contain an even number of bytes",
        )
    if max_bytes is not None and len(data) > max_bytes:
        raise AudioValidationError(
            "audio_message_too_large",
            f"PCM payload exceeds the {max_bytes}-byte limit",
        )


def decode_pcm16le(data: bytes) -> list[int]:
    if len(data) % PCM16_BYTES_PER_SAMPLE != 0:
        raise AudioValidationError(
            "unaligned_pcm16",
            "PCM16 little-endian payload must contain an even number of bytes",
        )
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return list(samples)


def encode_pcm16le(samples: list[int]) -> bytes:
    encoded = array("h", samples)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


class PCMFrameSplitter:
    """Split arbitrary messages into fixed frames without losing the remainder."""

    def __init__(self, pcm_format: PCMFormat, *, max_message_bytes: int | None = None) -> None:
        self._format = pcm_format
        self._max_message_bytes = max_message_bytes
        self._remainder = b""

    @property
    def pending_bytes(self) -> int:
        return len(self._remainder)

    def push(self, data: bytes) -> list[bytes]:
        self._format.validate_payload(data, max_bytes=self._max_message_bytes)
        buffered = self._remainder + data
        frame_bytes = self._format.frame_bytes
        complete_length = len(buffered) - len(buffered) % frame_bytes
        self._remainder = buffered[complete_length:]
        return [
            buffered[offset : offset + frame_bytes]
            for offset in range(0, complete_length, frame_bytes)
        ]

    def flush(self) -> bytes:
        remainder = self._remainder
        self._remainder = b""
        return remainder


class AudioTimeline:
    """Integer sample counter; millisecond conversion happens only at boundaries."""

    def __init__(self, sample_rate: int) -> None:
        if sample_rate not in SUPPORTED_INPUT_SAMPLE_RATES:
            raise AudioValidationError(
                "unsupported_sample_rate",
                f"unsupported input sample rate: {sample_rate}",
            )
        self.sample_rate = sample_rate
        self._samples = 0

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def position_ms(self) -> int:
        return self.samples_to_ms(self._samples)

    def samples_to_ms(self, samples: int) -> int:
        return samples * 1000 // self.sample_rate

    def consume(self, data: bytes) -> tuple[int, int]:
        validate_pcm16le(data, sample_rate=self.sample_rate)
        start_ms = self.position_ms
        self._samples += len(data) // PCM16_BYTES_PER_SAMPLE
        return start_ms, self.position_ms
