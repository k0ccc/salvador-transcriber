from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from salvador_transcriber.domain.state import ChannelState, Role
from salvador_transcriber.observability.latency import BoundedLatencyCollector

Labels: TypeAlias = tuple[tuple[str, str], ...]
MetricKey: TypeAlias = tuple[str, Labels]


@dataclass(slots=True)
class _Histogram:
    count: int = 0
    total: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0

    def observe(self, value: float) -> None:
        if self.count == 0:
            self.minimum = value
            self.maximum = value
        else:
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)
        self.count += 1
        self.total += value


class MetricsRegistry:
    """Small in-process registry with bounded, low-cardinality exported labels.

    Call identifiers are used only as internal dictionary keys to maintain gauges;
    they are never emitted in snapshots or Prometheus text.
    """

    def __init__(self) -> None:
        self.latency = BoundedLatencyCollector()
        self._counters: dict[MetricKey, float] = defaultdict(float)
        self._histograms: dict[MetricKey, _Histogram] = {}
        self._calls: set[str] = set()
        self._channels: dict[str, tuple[Role, ChannelState]] = {}
        self._queue_depths: dict[str, int] = {}

    @staticmethod
    def _key(name: str, labels: Mapping[str, str] | None = None) -> MetricKey:
        return name, tuple(sorted((labels or {}).items()))

    def increment(
        self,
        name: str,
        *,
        value: float = 1.0,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self._counters[self._key(name, labels)] += value

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        key = self._key(name, labels)
        histogram = self._histograms.setdefault(key, _Histogram())
        histogram.observe(max(0.0, value))

    def call_opened(self, call_id: str) -> None:
        self._calls.add(call_id)

    def call_closed(self, call_id: str) -> None:
        self._calls.discard(call_id)

    def channel_opened(self, channel_key: str, role: Role, state: ChannelState) -> None:
        self._channels[channel_key] = (role, state)

    def channel_state(self, channel_key: str, state: ChannelState) -> None:
        current = self._channels.get(channel_key)
        if current is not None:
            self._channels[channel_key] = (current[0], state)

    def channel_closed(self, channel_key: str) -> None:
        self._channels.pop(channel_key, None)
        self._queue_depths.pop(channel_key, None)

    def audio_queue_depth(self, channel_key: str, depth: int) -> None:
        self._queue_depths[channel_key] = max(0, depth)

    def counter_value(self, name: str, *, labels: Mapping[str, str] | None = None) -> float:
        return self._counters.get(self._key(name, labels), 0.0)

    def histogram_count(self, name: str, *, labels: Mapping[str, str] | None = None) -> int:
        histogram = self._histograms.get(self._key(name, labels))
        return histogram.count if histogram is not None else 0

    @staticmethod
    def _number(value: float) -> int | float:
        return int(value) if value.is_integer() else value

    @staticmethod
    def _format_labels(labels: Labels) -> str:
        if not labels:
            return ""
        rendered = ",".join(
            f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
            for key, value in labels
        )
        return "{" + rendered + "}"

    def _state_counts(self) -> dict[tuple[Role, ChannelState], int]:
        counts: dict[tuple[Role, ChannelState], int] = defaultdict(int)
        for role, state in self._channels.values():
            counts[(role, state)] += 1
        return counts

    def snapshot(self) -> dict[str, object]:
        counters = {
            f"{name}{self._format_labels(labels)}": self._number(value)
            for (name, labels), value in sorted(self._counters.items())
        }
        histograms: dict[str, object] = {}
        for (name, labels), histogram in sorted(self._histograms.items()):
            suffix = self._format_labels(labels)
            histograms[f"{name}{suffix}"] = {
                "count": histogram.count,
                "sum": histogram.total,
                "min": histogram.minimum,
                "max": histogram.maximum,
            }
        states = {
            f"{role.value}/{state.value}": count
            for (role, state), count in sorted(self._state_counts().items())
        }
        return {
            "active_calls": len(self._calls),
            "active_channels": len(self._channels),
            "channels_by_state": states,
            "audio_queue_depth": sum(self._queue_depths.values()),
            "audio_queue_depth_max": max(self._queue_depths.values(), default=0),
            "counters": counters,
            "histograms": histograms,
        }

    def prometheus(self) -> str:
        lines: list[str] = []
        lines.extend(
            (
                "# TYPE salvador_active_calls gauge",
                f"salvador_active_calls {len(self._calls)}",
                "# TYPE salvador_active_channels gauge",
                f"salvador_active_channels {len(self._channels)}",
            )
        )
        for (role, state), count in sorted(self._state_counts().items()):
            state_labels = (("role", role.value), ("state", state.value))
            lines.append(f"salvador_channel_state{self._format_labels(state_labels)} {count}")
        lines.append(f"salvador_audio_queue_depth {sum(self._queue_depths.values())}")
        max_depth = max(self._queue_depths.values(), default=0)
        lines.append(f"salvador_audio_queue_depth_max {max_depth}")
        for (name, labels), value in sorted(self._counters.items()):
            metric_name = name if name.startswith("salvador_") else f"salvador_{name}"
            lines.append(f"{metric_name}{self._format_labels(labels)} {self._number(value)}")
        for (name, labels), histogram in sorted(self._histograms.items()):
            metric_name = name if name.startswith("salvador_") else f"salvador_{name}"
            suffix = self._format_labels(labels)
            lines.append(f"{metric_name}_count{suffix} {histogram.count}")
            lines.append(f"{metric_name}_sum{suffix} {histogram.total}")
            lines.append(f"{metric_name}_min{suffix} {histogram.minimum}")
            lines.append(f"{metric_name}_max{suffix} {histogram.maximum}")
        return "\n".join(lines) + "\n"
