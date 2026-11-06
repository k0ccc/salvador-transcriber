import asyncio
import hashlib
import logging
import random
import time

import httpx

from salvador_transcriber.domain.events import TranscriptEvent
from salvador_transcriber.domain.state import Role
from salvador_transcriber.observability.metrics import MetricsRegistry
from salvador_transcriber.storage.repository import ExportStatus, ExportStatusStore

logger = logging.getLogger(__name__)


def salvador_event_type(event: TranscriptEvent) -> str:
    if event.role is Role.GUEST:
        return "guest_utterance_completed"
    return "operator_utterance_completed"


def export_payload(event: TranscriptEvent) -> dict[str, object]:
    return {
        "event_type": salvador_event_type(event),
        "call_id": event.call_id,
        "hotel_id": event.hotel_id,
        "utterance_id": event.utterance_id,
        "role": event.role.value,
        "sequence": event.sequence,
        "text": event.text,
        "start_ms": event.start_ms,
        "end_ms": event.end_ms,
    }


def idempotency_key(event: TranscriptEvent) -> str:
    digest = hashlib.sha256()
    for value in (event.call_id, event.utterance_id, salvador_event_type(event)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class HTTPSalvadorExporter:
    """Bounded asynchronous delivery; local finals survive every export failure."""

    def __init__(
        self,
        *,
        endpoint: str,
        status_store: ExportStatusStore,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.1,
        jitter_seconds: float = 0.05,
        queue_limit: int = 1000,
        metrics: MetricsRegistry | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if queue_limit < 1:
            raise ValueError("queue_limit must be positive")
        self._endpoint = endpoint
        self._status_store = status_store
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._jitter_seconds = jitter_seconds
        self._metrics = metrics or MetricsRegistry()
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue(maxsize=queue_limit)
        self._lifecycle_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._inflight: TranscriptEvent | None = None
        self._accepting = False
        self._closed = False
        self._dropped_events = 0

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Salvador exporter is closed")
            if self._worker_task is not None:
                return
            self._accepting = True
            self._worker_task = asyncio.create_task(
                self._worker(),
                name="salvador-http-exporter",
            )

    async def publish(self, event: TranscriptEvent) -> None:
        if not event.is_final or not event.text.strip():
            return
        if not self._accepting:
            raise RuntimeError("Salvador exporter is not accepting events")
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped_events += 1
            self._metrics.increment("export_queue_overflows_total")
            await self._status_store.record_export_result(
                call_id=event.call_id,
                utterance_id=event.utterance_id,
                status=ExportStatus.FAILED,
                attempts=0,
                error="queue_overflow",
            )
            logger.error(
                "Export queue overflow; durable final retained for call_id=%s utterance_id=%s",
                event.call_id,
                event.utterance_id,
            )

    async def close(self, *, timeout_seconds: float | None = None) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._accepting = False
            task = self._worker_task
            abandoned_inflight: TranscriptEvent | None = None
            timed_out = False
            try:
                if task is not None:
                    if timeout_seconds is None:
                        await self._queue.join()
                    else:
                        await asyncio.wait_for(self._queue.join(), timeout=timeout_seconds)
            except TimeoutError:
                timed_out = True
                self._metrics.increment("export_shutdown_timeouts_total")
            finally:
                # The worker may dequeue an event while close() waits on queue.join().
                # Capture it before cancellation clears the worker's inflight marker.
                if timed_out:
                    abandoned_inflight = self._inflight
                if task is not None and not task.done():
                    task.cancel()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                if timed_out:
                    await self._mark_abandoned_failed(abandoned_inflight)
                self._worker_task = None
                if self._owns_client:
                    await self._client.aclose()

    async def _mark_abandoned_failed(self, inflight: TranscriptEvent | None) -> None:
        abandoned: list[TranscriptEvent] = []
        if inflight is not None:
            abandoned.append(inflight)
        self._inflight = None
        while True:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            abandoned.append(event)
            self._queue.task_done()
        for event in abandoned:
            await self._status_store.record_export_result(
                call_id=event.call_id,
                utterance_id=event.utterance_id,
                status=ExportStatus.FAILED,
                attempts=0,
                error="shutdown_timeout",
            )
            self._metrics.increment("export_shutdown_dropped_total")

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            self._inflight = event
            started_at = time.monotonic()
            try:
                try:
                    await self._deliver(event)
                except Exception:
                    self._metrics.increment("export_errors_total")
                    logger.exception(
                        "Export processing failed for call_id=%s utterance_id=%s",
                        event.call_id,
                        event.utterance_id,
                    )
            finally:
                self._metrics.observe(
                    "export_delivery_seconds",
                    time.monotonic() - started_at,
                )
                self._inflight = None
                self._queue.task_done()

    async def _deliver(self, event: TranscriptEvent) -> None:
        key = idempotency_key(event)
        payload = export_payload(event)
        last_error = "ExportFailed"
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(
                    self._endpoint,
                    json=payload,
                    headers={"Idempotency-Key": key},
                    timeout=self._timeout_seconds,
                )
                if 200 <= response.status_code < 300:
                    await self._status_store.record_export_result(
                        call_id=event.call_id,
                        utterance_id=event.utterance_id,
                        status=ExportStatus.SENT,
                        attempts=attempt,
                    )
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.TimeoutException as error:
                last_error = type(error).__name__
            except httpx.HTTPError as error:
                last_error = type(error).__name__
            if attempt < self._max_attempts:
                delay = self._backoff_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0.0, self._jitter_seconds)
                await asyncio.sleep(delay)
        await self._status_store.record_export_result(
            call_id=event.call_id,
            utterance_id=event.utterance_id,
            status=ExportStatus.FAILED,
            attempts=self._max_attempts,
            error=last_error,
        )
        self._metrics.increment("export_failures_total")
        logger.warning(
            "Export failed for call_id=%s hotel_id=%s role=%s utterance_id=%s: %s",
            event.call_id,
            event.hotel_id,
            event.role.value,
            event.utterance_id,
            last_error,
        )
