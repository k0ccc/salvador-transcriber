import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from salvador_transcriber.api.routes import router
from salvador_transcriber.config import Settings, get_settings
from salvador_transcriber.demo.events import DemoEventHub
from salvador_transcriber.domain.sink import EventSink, InMemoryEventSink
from salvador_transcriber.export.http import HTTPSalvadorExporter
from salvador_transcriber.export.sink import DurableEventSink
from salvador_transcriber.observability.metrics import MetricsRegistry
from salvador_transcriber.sessions.registry import CallRegistry
from salvador_transcriber.storage.repository import TranscriptRepository
from salvador_transcriber.storage.sqlite import SQLiteTranscriptRepository
from salvador_transcriber.transcription.adapter import TranscriberFactory
from salvador_transcriber.transcription.factory import build_transcriber_factory


def create_app(
    settings: Settings | None = None,
    *,
    transcriber_factory: TranscriberFactory | None = None,
    event_sink: EventSink | None = None,
    repository: TranscriptRepository | None = None,
    exporter: HTTPSalvadorExporter | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    metrics = MetricsRegistry()
    downstream = event_sink or InMemoryEventSink()
    demo_event_hub = DemoEventHub(downstream)
    store = repository or SQLiteTranscriptRepository(
        resolved_settings.sqlite_path,
        busy_timeout_ms=resolved_settings.sqlite_busy_timeout_ms,
        queue_limit=resolved_settings.sqlite_queue_limit,
    )
    resolved_exporter = exporter
    if resolved_exporter is None and resolved_settings.exporter_enabled:
        if resolved_settings.exporter_endpoint is None:
            raise ValueError("exporter_endpoint is required when exporter_enabled=true")
        resolved_exporter = HTTPSalvadorExporter(
            endpoint=str(resolved_settings.exporter_endpoint),
            status_store=store,
            timeout_seconds=resolved_settings.exporter_timeout_seconds,
            max_attempts=resolved_settings.exporter_max_attempts,
            backoff_seconds=resolved_settings.exporter_backoff_seconds,
            jitter_seconds=resolved_settings.exporter_jitter_seconds,
            queue_limit=resolved_settings.exporter_queue_limit,
            metrics=metrics,
        )
    sink = DurableEventSink(
        repository=store,
        downstream=demo_event_hub,
        exporter=resolved_exporter,
    )
    factory = transcriber_factory or build_transcriber_factory(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await store.start()
        if resolved_exporter is not None:
            await resolved_exporter.start()
        try:
            yield
        finally:
            registry: CallRegistry = app.state.registry
            try:
                await registry.close_all()
            finally:
                try:
                    if resolved_exporter is not None:
                        await resolved_exporter.close(
                            timeout_seconds=resolved_settings.shutdown_export_drain_timeout_seconds
                        )
                finally:
                    await store.close()

    app = FastAPI(title="Salvador Transcriber", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.event_sink = sink
    app.state.local_event_sink = downstream
    app.state.demo_event_hub = demo_event_hub
    app.state.repository = store
    app.state.exporter = resolved_exporter
    app.state.metrics = metrics
    app.state.registry = CallRegistry(
        transcriber_factory=factory,
        event_sink=sink,
        target_sample_rate=resolved_settings.audio_target_sample_rate,
        max_audio_message_bytes=resolved_settings.max_audio_message_bytes,
        audio_queue_limit=resolved_settings.audio_queue_limit,
        reconnect_max_attempts=resolved_settings.reconnect_max_attempts,
        reconnect_backoff_seconds=resolved_settings.reconnect_backoff_seconds,
        reconnect_max_backoff_seconds=resolved_settings.reconnect_max_backoff_seconds,
        reconnect_jitter_seconds=resolved_settings.reconnect_jitter_seconds,
        reconnect_connect_timeout_seconds=resolved_settings.reconnect_connect_timeout_seconds,
        shutdown_drain_timeout_seconds=resolved_settings.shutdown_drain_timeout_seconds,
        shutdown_final_grace_seconds=resolved_settings.shutdown_final_grace_seconds,
        metrics=metrics,
    )
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    uvicorn.run(
        "salvador_transcriber.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
