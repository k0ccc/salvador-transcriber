import wave
from pathlib import Path

import pytest

from salvador_transcriber.demo.audio import AudioSourceError, load_audio_source
from salvador_transcriber.demo.live import (
    LiveConfigurationError,
    build_argument_parser,
    load_options,
    main,
    reconnect_report,
)


def write_wav(path: Path, *, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x01\x00" * 320 * channels)


def required_args(guest: Path, operator: Path, output: Path) -> list[str]:
    return [
        "--scenario",
        "normal",
        "--guest-audio",
        str(guest),
        "--operator-audio",
        str(operator),
        "--output",
        str(output),
        "--confirm-live-openai",
    ]


def test_audio_loader_accepts_mono_wav_and_raw_pcm(tmp_path: Path) -> None:
    wav_path = tmp_path / "guest.wav"
    pcm_path = tmp_path / "operator.pcm"
    write_wav(wav_path)
    pcm_path.write_bytes(b"\x02\x00" * 320)

    wav = load_audio_source(wav_path)
    pcm = load_audio_source(pcm_path)
    assert wav.sample_rate == 16000
    assert wav.duration_ms == 20
    assert len(list(wav.frames())) == 1
    assert pcm.pcm == b"\x02\x00" * 320


def test_audio_loader_rejects_stereo_instead_of_downmixing(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    write_wav(path, channels=2)
    with pytest.raises(AudioSourceError, match="mono"):
        load_audio_source(path)


def test_live_options_require_key_and_explicit_scenario_files(tmp_path: Path) -> None:
    parser = build_argument_parser()
    guest = tmp_path / "guest.pcm"
    operator = tmp_path / "operator.pcm"
    guest.write_bytes(b"\x00\x00" * 320)
    operator.write_bytes(b"\x01\x00" * 320)
    args = parser.parse_args(required_args(guest, operator, tmp_path / "report.json"))

    with pytest.raises(LiveConfigurationError, match="OPENAI_API_KEY"):
        load_options(args, {})
    options = load_options(args, {"OPENAI_API_KEY": "test-only-key"})
    assert options.guest.path == guest
    assert options.api_key.get_secret_value() == "test-only-key"

    two_call_args = required_args(guest, operator, tmp_path / "report.json")
    two_call_args[1] = "two_calls"
    two_calls = parser.parse_args(two_call_args)
    with pytest.raises(LiveConfigurationError, match="second audio"):
        load_options(two_calls, {"OPENAI_API_KEY": "test-only-key"})


def test_live_main_without_key_does_not_start_network(tmp_path: Path) -> None:
    args = required_args(
        tmp_path / "missing-guest.pcm",
        tmp_path / "missing-operator.pcm",
        tmp_path / "report.json",
    )
    assert main(args, environ={}) == 2
    assert not (tmp_path / "report.json").exists()


def test_reconnect_report_uses_actual_counter_delta_and_histogram() -> None:
    before = {"counters": {'reconnect_successes_total{role="guest"}': 1}}
    after = {
        "counters": {
            'reconnect_successes_total{role="guest"}': 2,
            'disconnects_total{role="guest"}': 1,
        },
        "histograms": {
            'reconnect_duration_seconds{role="guest"}': {
                "count": 1,
                "sum": 0.25,
                "min": 0.25,
                "max": 0.25,
            }
        },
    }
    result = reconnect_report(before, after)
    assert result["counter_deltas"]['reconnect_successes_total{role="guest"}'] == 1
    assert result["counter_deltas"]['disconnects_total{role="guest"}'] == 1
    assert result["reconnect_duration_seconds"]["count"] == 1  # type: ignore[index]
