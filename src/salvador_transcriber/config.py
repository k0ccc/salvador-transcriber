from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings; the API key is redacted by ``SecretStr``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SALVADOR_",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    transcriber_backend: Literal["fake", "openai"] = "fake"
    audio_target_sample_rate: int = Field(default=24000, ge=1000, le=192000)
    audio_frame_duration_ms: int = Field(default=20, ge=10, le=1000)
    max_audio_message_bytes: int = Field(default=65536, ge=2, le=10_000_000)
    audio_queue_limit: int = Field(default=100, ge=1, le=100_000)
    reconnect_max_attempts: int = Field(default=5, ge=1, le=100)
    reconnect_backoff_seconds: float = Field(default=0.1, ge=0.0, le=60.0)
    reconnect_max_backoff_seconds: float = Field(default=2.0, gt=0.0, le=300.0)
    reconnect_jitter_seconds: float = Field(default=0.05, ge=0.0, le=60.0)
    reconnect_connect_timeout_seconds: float = Field(default=10.0, gt=0.0, le=120.0)
    shutdown_drain_timeout_seconds: float = Field(default=2.0, gt=0.0, le=120.0)
    shutdown_final_grace_seconds: float = Field(default=0.2, ge=0.0, le=120.0)
    shutdown_export_drain_timeout_seconds: float = Field(default=2.0, gt=0.0, le=120.0)
    metrics_enabled: bool = True
    demo_provider_faults_enabled: bool = False
    openai_api_key: SecretStr | None = None
    openai_model: str = Field(default="gpt-4o-transcribe", min_length=1)
    openai_transcription_language: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2}$",
    )
    openai_vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    openai_vad_prefix_padding_ms: int = Field(default=300, ge=0)
    openai_vad_silence_duration_ms: int = Field(default=700, ge=0)
    openai_connect_timeout_seconds: float = Field(default=10.0, gt=0.0, le=120.0)
    sqlite_path: Path = Path("var/salvador-transcriber.db")
    sqlite_busy_timeout_ms: int = Field(default=5000, ge=1, le=120000)
    sqlite_queue_limit: int = Field(default=1000, ge=1, le=100_000)
    exporter_enabled: bool = False
    exporter_queue_limit: int = Field(default=1000, ge=1, le=100_000)
    exporter_endpoint: AnyHttpUrl | None = None
    exporter_timeout_seconds: float = Field(default=5.0, gt=0.0, le=120.0)
    exporter_max_attempts: int = Field(default=3, ge=1, le=10)
    exporter_backoff_seconds: float = Field(default=0.1, ge=0.0, le=60.0)
    exporter_jitter_seconds: float = Field(default=0.05, ge=0.0, le=60.0)

    @model_validator(mode="after")
    def validate_openai_audio(self) -> Self:
        if self.transcriber_backend == "openai" and self.audio_target_sample_rate != 24000:
            raise ValueError("OpenAI Realtime PCM input requires audio_target_sample_rate=24000")
        if self.exporter_enabled and self.exporter_endpoint is None:
            raise ValueError("exporter_endpoint is required when exporter_enabled=true")
        if self.shutdown_final_grace_seconds > self.shutdown_drain_timeout_seconds:
            raise ValueError("shutdown_final_grace_seconds must not exceed the drain timeout")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
