from salvador_transcriber.config import Settings
from salvador_transcriber.domain.state import Role
from salvador_transcriber.transcription.adapter import Transcriber, TranscriberFactory
from salvador_transcriber.transcription.fake import FakeTranscriber
from salvador_transcriber.transcription.openai_realtime import (
    OpenAIRealtimeConfig,
    OpenAIRealtimeTranscriber,
)


class TranscriberConfigurationError(RuntimeError):
    pass


class _FakeTranscriberFactory:
    def __call__(self, role: Role) -> Transcriber:
        return FakeTranscriber()


class _OpenAITranscriberFactory:
    def __init__(self, config: OpenAIRealtimeConfig) -> None:
        self._config = config

    def __call__(self, role: Role) -> Transcriber:
        return OpenAIRealtimeTranscriber(self._config)


def build_transcriber_factory(settings: Settings) -> TranscriberFactory:
    if settings.transcriber_backend == "fake":
        return _FakeTranscriberFactory()

    api_key = settings.openai_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise TranscriberConfigurationError(
            "SALVADOR_OPENAI_API_KEY is required when SALVADOR_TRANSCRIBER_BACKEND=openai"
        )
    config = OpenAIRealtimeConfig(
        api_key=api_key,
        model=settings.openai_model,
        language=settings.openai_transcription_language,
        vad_threshold=settings.openai_vad_threshold,
        vad_prefix_padding_ms=settings.openai_vad_prefix_padding_ms,
        vad_silence_duration_ms=settings.openai_vad_silence_duration_ms,
        connect_timeout_seconds=settings.openai_connect_timeout_seconds,
    )
    return _OpenAITranscriberFactory(config)
