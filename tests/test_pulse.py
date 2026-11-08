"""pulse.assess_pulse のユニットテスト。合成音とビート列だけで動く。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chordscope.analyzers.beats import estimate_beats_and_meter
from chordscope.analyzers.pulse import assess_pulse
from chordscope.audio import DEFAULT_SR, AudioBuffer
from chordscope.models import BeatResult
from tests._synth import click_track

_BPM = 100.0
_DURATION_SEC = 30.0


def _beats(times: np.ndarray) -> BeatResult:
    return BeatResult(beat_times=times.tolist(), downbeat_times=[], beats_per_bar=4, method="x")


def _regular_beats(bpm: float, duration: float) -> np.ndarray:
    return np.arange(0.5, duration - 0.1, 60.0 / bpm)


@pytest.fixture(scope="module")
def click_buffer() -> AudioBuffer:
    return click_track(_BPM, _DURATION_SEC, DEFAULT_SR)


@pytest.fixture(scope="module")
def noise_buffer() -> AudioBuffer:
    rng = np.random.default_rng(0)
    samples = (0.1 * rng.standard_normal(int(_DURATION_SEC * DEFAULT_SR))).astype(np.float32)
    return AudioBuffer(
        samples=samples, sample_rate=DEFAULT_SR, duration=_DURATION_SEC, path=Path("noise.wav")
    )


def test_steady_click_track_is_not_uncertain(click_buffer: AudioBuffer) -> None:
    res = assess_pulse(click_buffer, _beats(_regular_beats(_BPM, _DURATION_SEC)))

    assert res.uncertain is False
    assert res.reasons == []
    assert res.grid_ratio == pytest.approx(1.0)


def test_mid_band_pulse_matches_click_tempo(click_buffer: AudioBuffer) -> None:
    res = assess_pulse(click_buffer, _beats(_regular_beats(_BPM, _DURATION_SEC)))

    mid = next(b for b in res.bands if b.band == "mid")
    assert mid.matches_beat is True
    assert mid.clarity >= 0.1


def test_noise_without_pulse_is_uncertain_for_weak_mid_band(noise_buffer: AudioBuffer) -> None:
    res = assess_pulse(noise_buffer, _beats(_regular_beats(_BPM, _DURATION_SEC)))

    assert res.uncertain is True
    assert "weak_mid_band_pulse" in res.reasons


def test_scattered_beat_intervals_are_uncertain_for_low_grid_ratio(
    click_buffer: AudioBuffer,
) -> None:
    """拍のない曲で、追跡器が間隔のばらばらな拍を返したときの形 (Decay Music など)。"""
    rng = np.random.default_rng(1)
    times = np.cumsum(rng.uniform(0.3, 1.2, size=40))

    res = assess_pulse(click_buffer, _beats(times))

    assert res.grid_ratio < 0.9
    assert "low_grid_ratio" in res.reasons


def test_beat_off_the_audio_pulse_octave_is_uncertain(click_buffer: AudioBuffer) -> None:
    """追跡器の拍が、音の周期と 2 の冪倍の関係にない (1.37 倍)。"""
    res = assess_pulse(click_buffer, _beats(_regular_beats(_BPM * 1.37, _DURATION_SEC)))

    assert "no_band_matches_beat" in res.reasons


def test_too_few_beats_is_uncertain(click_buffer: AudioBuffer) -> None:
    res = assess_pulse(click_buffer, _beats(np.array([1.0, 1.6])))

    assert res.uncertain is True
    assert res.reasons == ["too_few_beats"]


@pytest.mark.integration
@pytest.mark.slow
def test_free_rhythm_vocal_is_uncertain(pd_encina_buffer: AudioBuffer) -> None:
    """ルネサンス声楽。追跡器の拍はそろって見える (grid 0.9 前後) が、音に拍の周期が弱い。"""
    beats, _meter = estimate_beats_and_meter(pd_encina_buffer)

    res = assess_pulse(pd_encina_buffer, beats)

    assert res.uncertain is True
    assert "weak_mid_band_pulse" in res.reasons


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("fixture", ["pd_sousa_buffer", "pd_joplin_full_buffer"])
def test_march_and_ragtime_are_not_uncertain(fixture: str, request: pytest.FixtureRequest) -> None:
    buffer: AudioBuffer = request.getfixturevalue(fixture)
    beats, _meter = estimate_beats_and_meter(buffer)

    res = assess_pulse(buffer, beats)

    assert res.uncertain is False, res.reasons
