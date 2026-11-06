import argparse
import asyncio
import json
import os
import socket
import sys
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
import uvicorn
from pydantic import SecretStr
from websockets.asyncio.client import ClientConnection, connect

from salvador_transcriber.config import Settings
from salvador_transcriber.demo.audio import AudioSource, load_audio_source, stream_audio_realtime
from salvador_transcriber.domain.events import EventType, TranscriptEvent
from salvador_transcriber.domain.sink import InMemoryEventSink
from salvador_transcriber.domain.state import Role
from salvador_transcriber.main import create_app
from salvador_transcriber.observability.latency import BoundedLatencyCollector, summarize_latency

SCENARIO_NAMES = ("normal", "overlap", "pauses", "reconnect", "two_calls")
JsonObject = dict[str, object]


class LiveConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LiveOptions:
    api_key: SecretStr
    output: Path
    scenario: str
    guest: AudioSource
    operator: AudioSource
    second_guest: AudioSource | None
    second_operator: AudioSource | None
    after_disconnect: AudioSource | None
    final_timeout_seconds: float


@dataclass(slots=True)
class _CallConnection:
    call_id: str
    hotel_id: str
    guest: ClientConnection
    operator: ClientConnection


def latency_report(
    collector: BoundedLatencyCollector,
    call_ids: Iterable[str],
    *,
    include_reconnect: bool,
) -> JsonObject:
    selected_calls = set(call_ids)
    observations = tuple(
        item
        for call_id in selected_calls
        for item in collector.observations(call_id)
        if include_reconnect or not item.reconnect
    )
    unavailable = tuple(
        item
        for call_id in selected_calls
        for item in collector.unavailable(call_id)
        if include_reconnect or item.connection_generation == 1
    )
    return {
        "status": "available" if observations else "unavailable",
        "reason": None if observations else "no_confirmed_speech_end_measurements",
        "statistics": summarize_latency(observations),
        "observations": [item.as_dict() for item in observations],
        "unavailable": [item.as_dict() for item in unavailable],
    }


def _numeric(mapping: object, key: str) -> float:
    if not isinstance(mapping, dict):
        return 0.0
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def reconnect_report(before: JsonObject, after: JsonObject) -> JsonObject:
    """Return counter deltas and the actual reconnect histogram, never a target estimate."""

    before_counters = before.get("counters")
    after_counters = after.get("counters")
    keys: set[str] = set()
    if isinstance(before_counters, dict):
        keys.update(str(key) for key in before_counters)
    if isinstance(after_counters, dict):
        keys.update(str(key) for key in after_counters)
    interesting = {
        key
        for key in keys
        if key.startswith(("disconnects_total", "reconnect_", "transcriber_errors_total"))
    }
    deltas = {
        key: _numeric(after_counters, key) - _numeric(before_counters, key)
        for key in sorted(interesting)
    }
    histograms = after.get("histograms")
    duration: object = None
    if isinstance(histograms, dict):
        duration = histograms.get('reconnect_duration_seconds{role="guest"}')
    return {"counter_deltas": deltas, "reconnect_duration_seconds": duration}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real OpenAI two-channel scenarios and write a raw JSON report."
    )
    parser.add_argument("--scenario", choices=(*SCENARIO_NAMES, "all"), required=True)
    parser.add_argument("--guest-audio", required=True)
    parser.add_argument("--operator-audio", required=True)
    parser.add_argument("--second-guest-audio")
    parser.add_argument("--second-operator-audio")
    parser.add_argument("--after-disconnect-audio")
    parser.add_argument("--pcm-sample-rate", type=int, default=16_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--confirm-live-openai", action="store_true", required=True)
    return parser


def load_options(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> LiveOptions:
    key = environ.get("OPENAI_API_KEY") or environ.get("SALVADOR_OPENAI_API_KEY")
    if key is None or not key.strip():
        raise LiveConfigurationError(
            "OPENAI_API_KEY or SALVADOR_OPENAI_API_KEY must be explicitly set"
        )
    scenario = cast(str, args.scenario)
    if not args.confirm_live_openai:
        raise LiveConfigurationError("explicit --confirm-live-openai is required")
    if scenario in {"all", "two_calls"} and not (
        args.second_guest_audio and args.second_operator_audio
    ):
        raise LiveConfigurationError("two_calls requires both second audio files")
    if scenario in {"all", "reconnect"} and not args.after_disconnect_audio:
        raise LiveConfigurationError("reconnect requires --after-disconnect-audio")
    if args.final_timeout_seconds <= 0:
        raise LiveConfigurationError("final timeout must be positive")

    pcm_rate = cast(int, args.pcm_sample_rate)
    return LiveOptions(
        api_key=SecretStr(key),
        output=Path(cast(str, args.output)),
        scenario=scenario,
        guest=load_audio_source(cast(str, args.guest_audio), pcm_sample_rate=pcm_rate),
        operator=load_audio_source(cast(str, args.operator_audio), pcm_sample_rate=pcm_rate),
        second_guest=(
            load_audio_source(cast(str, args.second_guest_audio), pcm_sample_rate=pcm_rate)
            if args.second_guest_audio
            else None
        ),
        second_operator=(
            load_audio_source(cast(str, args.second_operator_audio), pcm_sample_rate=pcm_rate)
            if args.second_operator_audio
            else None
        ),
        after_disconnect=(
            load_audio_source(cast(str, args.after_disconnect_audio), pcm_sample_rate=pcm_rate)
            if args.after_disconnect_audio
            else None
        ),
        final_timeout_seconds=cast(float, args.final_timeout_seconds),
    )


class LiveScenarioRunner:
    """Run a real uvicorn backend; this class is never used by the regular tests."""

    def __init__(self, options: LiveOptions) -> None:
        self._options = options
        self._settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            transcriber_backend="openai",
            openai_api_key=options.api_key,
            sqlite_path=options.output.with_suffix(".db"),
            demo_provider_faults_enabled=True,
            shutdown_drain_timeout_seconds=5.0,
            shutdown_final_grace_seconds=2.0,
        )
        self._client: httpx.AsyncClient | None = None
        self._local_sink: InMemoryEventSink | None = None
        self._latency_collector: BoundedLatencyCollector | None = None
        self._scenario_call_ids: list[str] = []
        self._scenario_errors: list[JsonObject] = []

    async def run(self) -> JsonObject:
        options = self._options
        report: JsonObject = {
            "schema_version": 1,
            "started_at": datetime.now(UTC).isoformat(),
            "backend": "openai",
            "scenario_selection": options.scenario,
            "configuration": {
                "model": self._settings.openai_model,
                "language": self._settings.openai_transcription_language,
                "vad_threshold": self._settings.openai_vad_threshold,
                "vad_prefix_padding_ms": self._settings.openai_vad_prefix_padding_ms,
                "vad_silence_duration_ms": self._settings.openai_vad_silence_duration_ms,
                "audio_target_sample_rate": self._settings.audio_target_sample_rate,
                "reconnect_max_attempts": self._settings.reconnect_max_attempts,
            },
            "audio": [
                self._audio_metadata(source)
                for source in (
                    options.guest,
                    options.operator,
                    options.second_guest,
                    options.second_operator,
                    options.after_disconnect,
                )
                if source is not None
            ],
            "scenarios": [],
            "fatal_errors": [],
        }
        app = create_app(self._settings)
        local_sink = app.state.local_event_sink
        if not isinstance(local_sink, InMemoryEventSink):
            raise TypeError("live runner requires the default in-memory event sink")
        self._local_sink = local_sink
        self._latency_collector = app.state.metrics.latency
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        server_task = asyncio.create_task(server.serve(), name="live-scenario-uvicorn")
        try:
            await self._wait_server(server, server_task)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=options.final_timeout_seconds,
            ) as client:
                self._client = client
                selected = SCENARIO_NAMES if options.scenario == "all" else (options.scenario,)
                for name in selected:
                    result = await self._run_scenario(name)
                    cast(list[JsonObject], report["scenarios"]).append(result)
                report["latency_normal"] = self._combined_latency(
                    cast(list[JsonObject], report["scenarios"]),
                    include_reconnect=False,
                )
        except Exception as error:
            cast(list[JsonObject], report["fatal_errors"]).append(self._error(error))
        finally:
            server.should_exit = True
            await asyncio.gather(server_task, return_exceptions=True)
            self._client = None

        scenarios = cast(list[JsonObject], report["scenarios"])
        failed = any(item.get("status") != "completed" for item in scenarios)
        report["status"] = "failed" if failed or report["fatal_errors"] else "completed"
        report["finished_at"] = datetime.now(UTC).isoformat()
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _audio_metadata(source: AudioSource) -> JsonObject:
        return {
            "path": str(source.path),
            "sample_rate": source.sample_rate,
            "duration_ms": source.duration_ms,
            "frame_duration_ms": source.frame_duration_ms,
        }

    async def _wait_server(
        self,
        server: uvicorn.Server,
        server_task: asyncio.Task[None],
    ) -> None:
        async with asyncio.timeout(self._options.final_timeout_seconds):
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError("uvicorn stopped before becoming ready")
                await asyncio.sleep(0.02)

    async def _run_scenario(self, name: str) -> JsonObject:
        if self._client is None:
            raise RuntimeError("live HTTP client is not running")
        before = await self._metrics()
        self._scenario_call_ids = []
        self._scenario_errors = []
        try:
            if name == "normal":
                await self._normal("live-normal", self._options.guest, self._options.operator)
            elif name == "overlap":
                await self._overlap()
            elif name == "pauses":
                await self._pauses()
            elif name == "reconnect":
                await self._reconnect()
            elif name == "two_calls":
                await self._two_calls()
            else:
                raise LiveConfigurationError(f"unknown scenario {name}")
        except Exception as error:
            self._scenario_errors.append(self._error(error))
        after = await self._metrics()
        call_ids = tuple(self._scenario_call_ids)
        transcripts: JsonObject = {}
        for call_id in call_ids:
            try:
                transcripts[call_id] = await self._transcript(call_id, f"hotel-{call_id}")
            except Exception as error:
                self._scenario_errors.append(self._error(error))
        result: JsonObject = {
            "name": name,
            "status": "failed" if self._scenario_errors else "completed",
            "call_ids": list(call_ids),
            "errors": list(self._scenario_errors),
            "events": self._events(call_ids),
            "transcripts": transcripts,
            "latency": latency_report(
                self._settings_latency(),
                call_ids,
                include_reconnect=name == "reconnect",
            ),
            "metrics": after,
        }
        if name == "reconnect":
            result["reconnect"] = reconnect_report(before, after)
        return result

    def _settings_latency(self) -> BoundedLatencyCollector:
        if self._latency_collector is None:
            raise RuntimeError("latency collector is not initialized")
        return self._latency_collector

    def _combined_latency(
        self,
        scenarios: Sequence[JsonObject],
        *,
        include_reconnect: bool,
    ) -> JsonObject:
        call_ids = [
            call_id
            for scenario in scenarios
            if scenario.get("name") != "reconnect"
            for call_id in cast(list[str], scenario.get("call_ids", []))
        ]
        return latency_report(
            self._settings_latency(),
            call_ids,
            include_reconnect=include_reconnect,
        )

    async def _normal(self, call_id: str, guest: AudioSource, operator: AudioSource) -> None:
        async with self._open_call(call_id, guest, operator) as connection:
            await stream_audio_realtime(connection.guest, guest)
            await self._wait_final(call_id, Role.GUEST)
            await stream_audio_realtime(connection.operator, operator)
            await self._wait_final(call_id, Role.OPERATOR)

    async def _overlap(self) -> None:
        call_id = "live-overlap"
        async with self._open_call(call_id, self._options.guest, self._options.operator) as conn:
            await asyncio.gather(
                stream_audio_realtime(conn.guest, self._options.guest),
                stream_audio_realtime(conn.operator, self._options.operator),
            )
            await self._wait_final(call_id, Role.GUEST)
            await self._wait_final(call_id, Role.OPERATOR)

    async def _pauses(self) -> None:
        call_id = "live-pauses"
        async with self._open_call(call_id, self._options.guest, self._options.operator) as conn:
            await stream_audio_realtime(conn.guest, self._options.guest)
            await self._wait_final(call_id, Role.GUEST)

    async def _reconnect(self) -> None:
        after_source = self._options.after_disconnect
        if after_source is None:
            raise LiveConfigurationError("after-disconnect source is required")
        call_id = "live-reconnect"
        async with self._open_call(call_id, self._options.guest, self._options.operator) as conn:
            await stream_audio_realtime(conn.guest, self._options.guest)
            await self._wait_final(call_id, Role.GUEST)
            await stream_audio_realtime(conn.operator, self._options.operator)
            await self._wait_final(call_id, Role.OPERATOR)
            response = await self._require_client().post(
                f"/demo/calls/{call_id}/guest/provider-disconnect",
                params={"hotel_id": f"hotel-{call_id}"},
            )
            response.raise_for_status()
            await self._wait_reconnect()
            await stream_audio_realtime(conn.guest, after_source)
            await self._wait_final(call_id, Role.GUEST, minimum_generation=2)
            await stream_audio_realtime(conn.operator, self._options.operator)
            await self._wait_final(call_id, Role.OPERATOR, minimum_count=2)

    async def _two_calls(self) -> None:
        second_guest = self._options.second_guest
        second_operator = self._options.second_operator
        if second_guest is None or second_operator is None:
            raise LiveConfigurationError("second call sources are required")
        await asyncio.gather(
            self._normal("live-two-a", self._options.guest, self._options.operator),
            self._normal("live-two-b", second_guest, second_operator),
        )

    @asynccontextmanager
    async def _open_call(
        self,
        call_id: str,
        guest: AudioSource,
        operator: AudioSource,
    ) -> AsyncIterator[_CallConnection]:
        client = self._require_client()
        hotel_id = f"hotel-{call_id}"
        response = await client.post(
            "/calls",
            json={"call_id": call_id, "hotel_id": hotel_id},
        )
        response.raise_for_status()
        self._scenario_call_ids.append(call_id)
        base = str(client.base_url).replace("http://", "ws://").replace("https://", "wss://")
        guest_ws = await connect(f"{base}/ws/calls/{call_id}/guest")
        operator_ws: ClientConnection | None = None
        try:
            operator_ws = await connect(f"{base}/ws/calls/{call_id}/operator")
            await self._start_channel(guest_ws, hotel_id, guest)
            await self._start_channel(operator_ws, hotel_id, operator)
            yield _CallConnection(call_id, hotel_id, guest_ws, operator_ws)
        finally:
            try:
                if operator_ws is not None:
                    await guest_ws.send(json.dumps({"type": "stop"}))
                    raw = await asyncio.wait_for(
                        guest_ws.recv(),
                        timeout=self._options.final_timeout_seconds,
                    )
                    if isinstance(raw, bytes):
                        raise RuntimeError("ingress returned binary stop acknowledgement")
                    acknowledgement = cast(JsonObject, json.loads(raw))
                    if acknowledgement.get("code") != "call_completed":
                        raise RuntimeError("call did not complete cleanly")
            except Exception as error:
                self._scenario_errors.append(self._error(error))
                try:
                    await client.post(f"/calls/{call_id}/stop")
                except Exception as stop_error:
                    self._scenario_errors.append(self._error(stop_error))
            finally:
                await asyncio.gather(
                    guest_ws.close(),
                    operator_ws.close() if operator_ws is not None else asyncio.sleep(0),
                    return_exceptions=True,
                )

    async def _start_channel(
        self,
        websocket: ClientConnection,
        hotel_id: str,
        source: AudioSource,
    ) -> None:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "hotel_id": hotel_id,
                    "audio": {
                        "encoding": "pcm16le",
                        "sample_rate": source.sample_rate,
                        "channels": 1,
                        "frame_duration_ms": source.frame_duration_ms,
                    },
                }
            )
        )
        raw = await websocket.recv()
        if isinstance(raw, bytes):
            raise RuntimeError("ingress returned an unexpected binary response")
        response = cast(JsonObject, json.loads(raw))
        if response.get("type") != "ready":
            raise RuntimeError(f"ingress rejected channel start: {response.get('code')}")

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("live HTTP client is not initialized")
        return self._client

    def _events(self, call_ids: Iterable[str]) -> list[JsonObject]:
        if self._local_sink is None:
            return []
        selected = set(call_ids)
        return [
            cast(JsonObject, event.model_dump(mode="json"))
            for event in self._local_sink.events
            if event.call_id in selected
        ]

    def _finals(
        self,
        call_id: str,
        role: Role,
        *,
        minimum_generation: int = 1,
    ) -> list[TranscriptEvent]:
        if self._local_sink is None:
            return []
        return [
            event
            for event in self._local_sink.events
            if isinstance(event, TranscriptEvent)
            and event.event_type is EventType.TRANSCRIPT_FINAL
            and event.call_id == call_id
            and event.role is role
            and event.connection_generation >= minimum_generation
        ]

    async def _wait_final(
        self,
        call_id: str,
        role: Role,
        *,
        minimum_count: int = 1,
        minimum_generation: int = 1,
    ) -> None:
        deadline = time.monotonic() + self._options.final_timeout_seconds
        while time.monotonic() < deadline:
            finals = self._finals(
                call_id,
                role,
                minimum_generation=minimum_generation,
            )
            if len(finals) >= minimum_count:
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"timed out waiting for {role.value} final")

    async def _wait_reconnect(self) -> None:
        deadline = time.monotonic() + self._options.final_timeout_seconds
        key = 'reconnect_successes_total{role="guest"}'
        while time.monotonic() < deadline:
            metrics = await self._metrics()
            if _numeric(metrics.get("counters"), key) >= 1:
                return
            await asyncio.sleep(0.05)
        raise TimeoutError("timed out waiting for guest reconnect")

    async def _metrics(self) -> JsonObject:
        response = await self._require_client().get("/metrics/state")
        response.raise_for_status()
        return cast(JsonObject, response.json())

    async def _transcript(self, call_id: str, hotel_id: str) -> JsonObject:
        response = await self._require_client().get(
            f"/calls/{call_id}/transcript",
            params={"hotel_id": hotel_id},
        )
        response.raise_for_status()
        return cast(JsonObject, response.json())

    def _error(self, error: Exception) -> JsonObject:
        secret = self._options.api_key.get_secret_value()
        message = str(error).replace(secret, "[redacted]")[:500]
        return {"type": type(error).__name__, "message": message}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        options = load_options(args, os.environ if environ is None else environ)
    except (LiveConfigurationError, ValueError) as error:
        print(f"live runner configuration error: {error}", file=sys.stderr)
        return 2
    report = asyncio.run(LiveScenarioRunner(options).run())
    return 0 if report.get("status") == "completed" else 1
