import asyncio
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from salvador_transcriber.domain.events import (
    AudioGapEvent,
    CallCompletedEvent,
    ChannelStatusEvent,
    EventType,
    GapReason,
    TranscriptEvent,
)
from salvador_transcriber.domain.state import ChannelState, Role
from salvador_transcriber.storage.repository import (
    CallContextMismatchError,
    ExportStatus,
    FinalSaveResult,
    StoredGap,
    StoredServiceEvent,
    StoredUtterance,
    Transcript,
    TranscriptNotFoundError,
)

_T = TypeVar("_T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY,
    hotel_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS final_utterances (
    call_id TEXT NOT NULL,
    hotel_id TEXT NOT NULL,
    utterance_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('guest', 'operator')),
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    text TEXT NOT NULL,
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms >= start_ms),
    connection_generation INTEGER NOT NULL CHECK (connection_generation >= 1),
    occurred_at TEXT NOT NULL,
    export_status TEXT NOT NULL CHECK (
        export_status IN ('pending', 'sent', 'failed', 'disabled', 'skipped_empty')
    ),
    export_attempts INTEGER NOT NULL DEFAULT 0 CHECK (export_attempts >= 0),
    export_error TEXT,
    exported_at TEXT,
    PRIMARY KEY (call_id, utterance_id),
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);

CREATE INDEX IF NOT EXISTS final_utterances_timeline
    ON final_utterances(call_id, hotel_id, start_ms, sequence, utterance_id);

CREATE TABLE IF NOT EXISTS audio_gaps (
    event_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL,
    hotel_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('guest', 'operator')),
    start_ms INTEGER,
    end_ms INTEGER,
    reason TEXT NOT NULL,
    connection_generation INTEGER NOT NULL,
    dropped_frames INTEGER NOT NULL,
    dropped_bytes INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);

CREATE INDEX IF NOT EXISTS audio_gaps_timeline
    ON audio_gaps(call_id, hotel_id, start_ms, end_ms, event_id);

CREATE TABLE IF NOT EXISTS service_events (
    event_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL,
    hotel_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('call_completed', 'channel_status')),
    role TEXT CHECK (role IN ('guest', 'operator')),
    status TEXT,
    connection_generation INTEGER,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);

CREATE INDEX IF NOT EXISTS service_events_timeline
    ON service_events(call_id, hotel_id, occurred_at, event_id);
"""


@dataclass(slots=True)
class _Operation:
    action: Callable[[sqlite3.Connection], object]
    future: asyncio.Future[object]


class SQLiteTranscriptRepository:
    """Serialize SQLite work on one worker task without blocking the event loop."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        queue_limit: int = 1000,
    ) -> None:
        if queue_limit < 1:
            raise ValueError("queue_limit must be positive")
        self._database = str(path)
        self._busy_timeout_ms = busy_timeout_ms
        self._queue: asyncio.Queue[_Operation | None] = asyncio.Queue(maxsize=queue_limit)
        self._lifecycle_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._accepting = False
        self._closed = False

    async def __aenter__(self) -> "SQLiteTranscriptRepository":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("SQLite repository is closed")
            if self._worker_task is not None:
                return
            self._ready = asyncio.get_running_loop().create_future()
            self._accepting = True
            self._worker_task = asyncio.create_task(
                self._worker(),
                name="sqlite-transcript-repository",
            )
            await self._ready

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._accepting = False
            task = self._worker_task
            if task is not None:
                await self._queue.put(None)
                await task
                self._worker_task = None

    async def save_final(self, event: TranscriptEvent) -> FinalSaveResult:
        if not event.is_final:
            raise ValueError("only final transcript events may be persisted")
        return await self._submit(lambda connection: self._save_final(connection, event))

    async def save_event(
        self,
        event: AudioGapEvent | CallCompletedEvent | ChannelStatusEvent,
    ) -> bool:
        return await self._submit(lambda connection: self._save_event(connection, event))

    async def get_transcript(self, call_id: str, hotel_id: str) -> Transcript:
        return await self._submit(
            lambda connection: self._read_transcript(connection, call_id, hotel_id),
        )

    async def record_export_result(
        self,
        *,
        call_id: str,
        utterance_id: str,
        status: ExportStatus,
        attempts: int,
        error: str | None = None,
    ) -> None:
        await self._submit(
            lambda connection: self._record_export_result(
                connection,
                call_id=call_id,
                utterance_id=utterance_id,
                status=status,
                attempts=attempts,
                error=error,
            ),
        )

    async def _submit(self, action: Callable[[sqlite3.Connection], _T]) -> _T:
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        async with self._lifecycle_lock:
            if not self._accepting or self._worker_task is None:
                raise RuntimeError("SQLite repository is not accepting operations")
            await self._queue.put(_Operation(action=action, future=future))
        return cast(_T, await future)

    def _connect(self) -> sqlite3.Connection:
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._database,
            timeout=self._busy_timeout_ms / 1000,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.executescript(_SCHEMA)
        return connection

    async def _worker(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            try:
                connection = await asyncio.to_thread(self._connect)
            except Exception as error:
                if self._ready is not None and not self._ready.done():
                    self._ready.set_exception(error)
                return
            if self._ready is not None and not self._ready.done():
                self._ready.set_result(None)
            while True:
                operation = await self._queue.get()
                try:
                    if operation is None:
                        return
                    try:
                        result = await asyncio.to_thread(operation.action, connection)
                    except Exception as error:
                        if not operation.future.cancelled():
                            operation.future.set_exception(error)
                    else:
                        if not operation.future.cancelled():
                            operation.future.set_result(result)
                finally:
                    self._queue.task_done()
        finally:
            if connection is not None:
                await asyncio.to_thread(connection.close)

    @staticmethod
    def _ensure_call(
        connection: sqlite3.Connection,
        *,
        call_id: str,
        hotel_id: str,
        occurred_at: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO calls(call_id, hotel_id, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(call_id) DO NOTHING
            """,
            (call_id, hotel_id, occurred_at.isoformat()),
        )
        row = connection.execute(
            "SELECT hotel_id FROM calls WHERE call_id = ?",
            (call_id,),
        ).fetchone()
        if row is None or row["hotel_id"] != hotel_id:
            raise CallContextMismatchError("call_id is already bound to another hotel_id")

    def _save_final(
        self,
        connection: sqlite3.Connection,
        event: TranscriptEvent,
    ) -> FinalSaveResult:
        with connection:
            self._ensure_call(
                connection,
                call_id=event.call_id,
                hotel_id=event.hotel_id,
                occurred_at=event.occurred_at,
            )
            cursor = connection.execute(
                """
                INSERT INTO final_utterances(
                    call_id, hotel_id, utterance_id, event_id, role, sequence, revision,
                    text, start_ms, end_ms, connection_generation, occurred_at,
                    export_status, export_attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0)
                ON CONFLICT(call_id, utterance_id) DO NOTHING
                """,
                (
                    event.call_id,
                    event.hotel_id,
                    event.utterance_id,
                    event.event_id,
                    event.role.value,
                    event.sequence,
                    event.revision,
                    event.text,
                    event.start_ms,
                    event.end_ms,
                    event.connection_generation,
                    event.occurred_at.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM final_utterances WHERE call_id = ? AND utterance_id = ?",
                (event.call_id, event.utterance_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("saved final utterance could not be read back")
        return FinalSaveResult(created=cursor.rowcount == 1, record=self._utterance(row))

    def _save_event(
        self,
        connection: sqlite3.Connection,
        event: AudioGapEvent | CallCompletedEvent | ChannelStatusEvent,
    ) -> bool:
        with connection:
            self._ensure_call(
                connection,
                call_id=event.call_id,
                hotel_id=event.hotel_id,
                occurred_at=event.occurred_at,
            )
            if isinstance(event, AudioGapEvent):
                cursor = connection.execute(
                    """
                    INSERT INTO audio_gaps(
                        event_id, call_id, hotel_id, role, start_ms, end_ms, reason,
                        connection_generation, dropped_frames, dropped_bytes, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (
                        event.event_id,
                        event.call_id,
                        event.hotel_id,
                        event.role.value,
                        event.start_ms,
                        event.end_ms,
                        event.reason.value,
                        event.connection_generation,
                        event.dropped_frames,
                        event.dropped_bytes,
                        event.occurred_at.isoformat(),
                    ),
                )
            else:
                role = event.role.value if isinstance(event, ChannelStatusEvent) else None
                status = event.status.value if isinstance(event, ChannelStatusEvent) else None
                generation = (
                    event.connection_generation if isinstance(event, ChannelStatusEvent) else None
                )
                cursor = connection.execute(
                    """
                    INSERT INTO service_events(
                        event_id, call_id, hotel_id, event_type, role, status,
                        connection_generation, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (
                        event.event_id,
                        event.call_id,
                        event.hotel_id,
                        event.event_type.value,
                        role,
                        status,
                        generation,
                        event.occurred_at.isoformat(),
                    ),
                )
                if isinstance(event, CallCompletedEvent):
                    connection.execute(
                        "UPDATE calls SET completed_at = ? WHERE call_id = ?",
                        (event.occurred_at.isoformat(), event.call_id),
                    )
        return cursor.rowcount == 1

    def _read_transcript(
        self,
        connection: sqlite3.Connection,
        call_id: str,
        hotel_id: str,
    ) -> Transcript:
        call = connection.execute(
            "SELECT call_id FROM calls WHERE call_id = ? AND hotel_id = ?",
            (call_id, hotel_id),
        ).fetchone()
        if call is None:
            raise TranscriptNotFoundError("call transcript not found")
        utterance_rows = connection.execute(
            """
            SELECT * FROM final_utterances
            WHERE call_id = ? AND hotel_id = ?
            ORDER BY start_ms, sequence, utterance_id
            """,
            (call_id, hotel_id),
        ).fetchall()
        gap_rows = connection.execute(
            """
            SELECT * FROM audio_gaps
            WHERE call_id = ? AND hotel_id = ?
            ORDER BY
                COALESCE(start_ms, 9223372036854775807),
                COALESCE(end_ms, 9223372036854775807),
                event_id
            """,
            (call_id, hotel_id),
        ).fetchall()
        service_rows = connection.execute(
            """
            SELECT * FROM service_events
            WHERE call_id = ? AND hotel_id = ?
            ORDER BY occurred_at, event_id
            """,
            (call_id, hotel_id),
        ).fetchall()
        return Transcript(
            call_id=call_id,
            hotel_id=hotel_id,
            utterances=tuple(self._utterance(row) for row in utterance_rows),
            gaps=tuple(self._gap(row) for row in gap_rows),
            service_events=tuple(self._service_event(row) for row in service_rows),
        )

    @staticmethod
    def _record_export_result(
        connection: sqlite3.Connection,
        *,
        call_id: str,
        utterance_id: str,
        status: ExportStatus,
        attempts: int,
        error: str | None,
    ) -> None:
        exported_at = datetime.now(UTC).isoformat() if status is ExportStatus.SENT else None
        stored_error = error[:200] if status is ExportStatus.FAILED and error else None
        with connection:
            cursor = connection.execute(
                """
                UPDATE final_utterances
                SET export_status = ?, export_attempts = ?, export_error = ?, exported_at = ?
                WHERE call_id = ? AND utterance_id = ?
                """,
                (
                    status.value,
                    attempts,
                    stored_error,
                    exported_at,
                    call_id,
                    utterance_id,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("cannot record export result for an unknown final utterance")

    @staticmethod
    def _utterance(row: sqlite3.Row) -> StoredUtterance:
        return StoredUtterance(
            event_id=row["event_id"],
            call_id=row["call_id"],
            hotel_id=row["hotel_id"],
            utterance_id=row["utterance_id"],
            role=Role(row["role"]),
            sequence=row["sequence"],
            revision=row["revision"],
            text=row["text"],
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            connection_generation=row["connection_generation"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            export_status=ExportStatus(row["export_status"]),
            export_attempts=row["export_attempts"],
            export_error=row["export_error"],
        )

    @staticmethod
    def _gap(row: sqlite3.Row) -> StoredGap:
        return StoredGap(
            event_id=row["event_id"],
            call_id=row["call_id"],
            hotel_id=row["hotel_id"],
            role=Role(row["role"]),
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            reason=GapReason(row["reason"]),
            connection_generation=row["connection_generation"],
            dropped_frames=row["dropped_frames"],
            dropped_bytes=row["dropped_bytes"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
        )

    @staticmethod
    def _service_event(row: sqlite3.Row) -> StoredServiceEvent:
        return StoredServiceEvent(
            event_id=row["event_id"],
            call_id=row["call_id"],
            hotel_id=row["hotel_id"],
            event_type=EventType(row["event_type"]),
            role=Role(row["role"]) if row["role"] is not None else None,
            status=ChannelState(row["status"]) if row["status"] is not None else None,
            connection_generation=row["connection_generation"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
        )
