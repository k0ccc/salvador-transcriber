from salvador_transcriber.audio.pcm import (
    AudioValidationError,
    decode_pcm16le,
    encode_pcm16le,
)


class StatefulLinearResampler:
    """Constant-memory streaming linear resampler for PCM16 mono.

    The output sample count after flush is the nearest integer to
    input_samples * output_rate / input_rate. One trailing input sample is
    retained between pushes so chunk boundaries do not change the result.
    """

    def __init__(self, input_rate: int, output_rate: int) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise AudioValidationError("invalid_sample_rate", "sample rates must be positive")
        self.input_rate = input_rate
        self.output_rate = output_rate
        self._buffer: list[int] = []
        self._buffer_start = 0
        self._total_input_samples = 0
        self._output_index = 0
        self._flushed = False

    @property
    def input_samples(self) -> int:
        return self._total_input_samples

    @property
    def expected_output_samples(self) -> int:
        numerator = self._total_input_samples * self.output_rate
        return (numerator + self.input_rate // 2) // self.input_rate

    def push(self, data: bytes) -> bytes:
        if self._flushed:
            raise AudioValidationError("resampler_flushed", "cannot push audio after flush")
        samples = decode_pcm16le(data)
        self._buffer.extend(samples)
        self._total_input_samples += len(samples)
        return self._produce(final=False)

    def flush(self) -> bytes:
        if self._flushed:
            return b""
        self._flushed = True
        return self._produce(final=True)

    def _produce(self, *, final: bool) -> bytes:
        output: list[int] = []
        target = self.expected_output_samples if final else None

        while target is None or self._output_index < target:
            numerator = self._output_index * self.input_rate
            left_index, remainder = divmod(numerator, self.output_rate)
            if left_index >= self._total_input_samples:
                break
            if not final and left_index + 1 >= self._total_input_samples:
                break

            buffer_index = left_index - self._buffer_start
            left = self._buffer[buffer_index]
            if left_index + 1 < self._total_input_samples:
                right = self._buffer[buffer_index + 1]
            else:
                right = left
            value = round(left + (right - left) * remainder / self.output_rate)
            output.append(max(-32768, min(32767, value)))
            self._output_index += 1

        if self._total_input_samples:
            next_left = (self._output_index * self.input_rate) // self.output_rate
            keep_index = min(next_left, self._total_input_samples - 1)
            discard_count = max(0, keep_index - self._buffer_start)
            if discard_count:
                del self._buffer[:discard_count]
                self._buffer_start = keep_index

        return encode_pcm16le(output)
