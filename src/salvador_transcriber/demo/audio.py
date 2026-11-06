import asyncio
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from salvador_transcriber.audio.pcm import PCMFormat


class AudioSourceError(ValueError):
    pass


class AudioSender(Protocol):
    async def send(self, message: str | bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class AudioSource:
    path: Path
    sample_rate: int
    pcm: bytes
    frame_duration_ms: int = 20

    @property
    def frame_bytes(self) -> int:
        return self.sample_rate * self.frame_duration_ms // 1000 * 2

    @property
    def duration_ms(self) -> int:
        return len(self.pcm) // 2 * 1000 // self.sample_rate

    def frames(self) -> Iterator[bytes]:
        for offset in range(0, len(self.pcm), self.frame_bytes):
            yield self.pcm[offset : offset + self.frame_bytes]


def load_audio_source(path: str | Path, *, pcm_sample_rate: int = 16_000) -> AudioSource:
    """Read mono PCM16LE WAV or headerless PCM without mixing any channels."""

    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".wav":
        try:
            with wave.open(str(source_path), "rb") as container:
                channels = container.getnchannels()
                sample_width = container.getsampwidth()
                sample_rate = container.getframerate()
                pcm = container.readframes(container.getnframes())
        except (OSError, wave.Error) as error:
            raise AudioSourceError(f"cannot read WAV source {source_path}") from error
        if channels != 1:
            raise AudioSourceError("WAV source must be mono; stereo downmix is not performed")
        if sample_width != 2:
            raise AudioSourceError("WAV source must contain PCM16 samples")
    elif suffix == ".pcm":
        sample_rate = pcm_sample_rate
        try:
            pcm = source_path.read_bytes()
        except OSError as error:
            raise AudioSourceError(f"cannot read PCM source {source_path}") from error
    else:
        raise AudioSourceError("audio source must have a .wav or .pcm extension")

    try:
        pcm_format = PCMFormat(sample_rate=sample_rate)
        pcm_format.validate_payload(pcm)
    except ValueError as error:
        raise AudioSourceError(str(error)) from error
    if not pcm:
        raise AudioSourceError("audio source must not be empty")
    return AudioSource(path=source_path, sample_rate=sample_rate, pcm=pcm)


async def stream_audio_realtime(sender: AudioSender, source: AudioSource) -> None:
    """Send independent source frames with monotonic drift-corrected pacing."""

    started_at = time.monotonic()
    frame_seconds = source.frame_duration_ms / 1000
    for index, frame in enumerate(source.frames(), start=1):
        await sender.send(frame)
        delay = started_at + index * frame_seconds - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
