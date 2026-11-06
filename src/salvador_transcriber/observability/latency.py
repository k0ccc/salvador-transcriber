import math
import statistics
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from salvador_transcriber.domain.state import Role


@dataclass(frozen=True, slots=True)
class LatencyObservation:
    """One valid final-latency measurement on a single monotonic clock."""

    call_id: str
    role: Role
    utterance_id: str
    connection_generation: int
    speech_end_ms: int
    speech_end_ingress_monotonic: float
    final_internal_monotonic: float
    final_latency_ms: float
    reconnect: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "role": self.role.value,
            "utterance_id": self.utterance_id,
            "connection_generation": self.connection_generation,
            "speech_end_ms": self.speech_end_ms,
            "speech_end_ingress_monotonic": self.speech_end_ingress_monotonic,
            "final_internal_monotonic": self.final_internal_monotonic,
            "final_latency_ms": self.final_latency_ms,
            "reconnect": self.reconnect,
        }


@dataclass(frozen=True, slots=True)
class UnavailableLatency:
    """A final for which an honest end-to-end latency cannot be reconstructed."""

    call_id: str
    role: Role
    utterance_id: str
    connection_generation: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "role": self.role.value,
            "utterance_id": self.utterance_id,
            "connection_generation": self.connection_generation,
            "reason": self.reason,
        }


class BoundedLatencyCollector:
    """Keep bounded raw measurements for local reports, not Prometheus labels."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity < 1:
            raise ValueError("latency collector capacity must be positive")
        self._observations: deque[LatencyObservation] = deque(maxlen=capacity)
        self._unavailable: deque[UnavailableLatency] = deque(maxlen=capacity)

    def record(
        self,
        *,
        call_id: str,
        role: Role,
        utterance_id: str,
        connection_generation: int,
        speech_end_ms: int,
        speech_end_ingress_monotonic: float,
        final_internal_monotonic: float,
    ) -> None:
        latency_ms = (final_internal_monotonic - speech_end_ingress_monotonic) * 1000
        if latency_ms < 0:
            self.mark_unavailable(
                call_id=call_id,
                role=role,
                utterance_id=utterance_id,
                connection_generation=connection_generation,
                reason="negative_monotonic_interval",
            )
            return
        self._observations.append(
            LatencyObservation(
                call_id=call_id,
                role=role,
                utterance_id=utterance_id,
                connection_generation=connection_generation,
                speech_end_ms=speech_end_ms,
                speech_end_ingress_monotonic=speech_end_ingress_monotonic,
                final_internal_monotonic=final_internal_monotonic,
                final_latency_ms=latency_ms,
                reconnect=connection_generation > 1,
            )
        )

    def mark_unavailable(
        self,
        *,
        call_id: str,
        role: Role,
        utterance_id: str,
        connection_generation: int,
        reason: str,
    ) -> None:
        self._unavailable.append(
            UnavailableLatency(
                call_id=call_id,
                role=role,
                utterance_id=utterance_id,
                connection_generation=connection_generation,
                reason=reason,
            )
        )

    def observations(self, call_id: str | None = None) -> tuple[LatencyObservation, ...]:
        return tuple(
            item for item in self._observations if call_id is None or item.call_id == call_id
        )

    def unavailable(self, call_id: str | None = None) -> tuple[UnavailableLatency, ...]:
        return tuple(
            item for item in self._unavailable if call_id is None or item.call_id == call_id
        )


def summarize_latency(observations: Iterable[LatencyObservation]) -> dict[str, object]:
    """Return nearest-rank p95 and the fraction at or below the two-second target."""

    values = sorted(item.final_latency_ms for item in observations)
    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "min_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "within_2000ms_count": 0,
            "within_2000ms_ratio": None,
        }
    p95_index = max(0, math.ceil(0.95 * count) - 1)
    within_target = sum(value <= 2000 for value in values)
    return {
        "count": count,
        "min_ms": values[0],
        "median_ms": statistics.median(values),
        "p95_ms": values[p95_index],
        "max_ms": values[-1],
        "within_2000ms_count": within_target,
        "within_2000ms_ratio": within_target / count,
    }
