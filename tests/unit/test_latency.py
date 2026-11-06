import pytest

from salvador_transcriber.demo.live import latency_report
from salvador_transcriber.domain.state import Role
from salvador_transcriber.observability.latency import (
    BoundedLatencyCollector,
    LatencyObservation,
    summarize_latency,
)


def observation(latency_ms: float, *, generation: int = 1) -> LatencyObservation:
    return LatencyObservation(
        call_id="call-1",
        role=Role.GUEST,
        utterance_id=f"guest_{int(latency_ms):04d}",
        connection_generation=generation,
        speech_end_ms=1000,
        speech_end_ingress_monotonic=10.0,
        final_internal_monotonic=10.0 + latency_ms / 1000,
        final_latency_ms=latency_ms,
        reconnect=generation > 1,
    )


def test_latency_statistics_include_target_ratio_and_nearest_rank_p95() -> None:
    result = summarize_latency(observation(value) for value in (100.0, 200.0, 300.0, 400.0, 2500.0))

    assert result == {
        "count": 5,
        "min_ms": 100.0,
        "median_ms": 300.0,
        "p95_ms": 2500.0,
        "max_ms": 2500.0,
        "within_2000ms_count": 4,
        "within_2000ms_ratio": 0.8,
    }


def test_empty_latency_statistics_are_unavailable_not_zero() -> None:
    result = summarize_latency([])

    assert result["count"] == 0
    assert result["p95_ms"] is None
    assert result["within_2000ms_ratio"] is None


def test_bounded_collector_computes_monotonic_latency_and_drops_oldest() -> None:
    collector = BoundedLatencyCollector(capacity=2)
    for index in range(3):
        collector.record(
            call_id="call-1",
            role=Role.GUEST,
            utterance_id=f"guest_{index:04d}",
            connection_generation=1,
            speech_end_ms=1000 + index,
            speech_end_ingress_monotonic=10.0 + index,
            final_internal_monotonic=10.1 + index,
        )

    observations = collector.observations("call-1")
    assert len(observations) == 2
    assert [item.utterance_id for item in observations] == ["guest_0001", "guest_0002"]
    assert observations[0].final_latency_ms == pytest.approx(100.0)


def test_negative_interval_and_reconnect_filter_are_explicit() -> None:
    collector = BoundedLatencyCollector()
    collector.record(
        call_id="call-1",
        role=Role.GUEST,
        utterance_id="guest_0001",
        connection_generation=1,
        speech_end_ms=1000,
        speech_end_ingress_monotonic=11.0,
        final_internal_monotonic=10.0,
    )
    assert collector.observations() == ()
    assert collector.unavailable()[0].reason == "negative_monotonic_interval"

    collector.record(
        call_id="call-1",
        role=Role.GUEST,
        utterance_id="guest_0002",
        connection_generation=2,
        speech_end_ms=1000,
        speech_end_ingress_monotonic=20.0,
        final_internal_monotonic=20.2,
    )
    normal = latency_report(collector, ["call-1"], include_reconnect=False)
    recovered = latency_report(collector, ["call-1"], include_reconnect=True)
    assert normal["status"] == "unavailable"
    assert normal["statistics"]["count"] == 0  # type: ignore[index]
    assert recovered["status"] == "available"
    assert recovered["statistics"]["count"] == 1  # type: ignore[index]
