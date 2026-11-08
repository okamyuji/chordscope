"""madmom へ渡す音声のサンプルレートが時刻と BPM を歪めないことを確かめる。

madmom の Processor は ndarray を 44100 Hz とみなす。22050 Hz の配列をそのまま渡すと
時刻が半分に、BPM が 2 倍になる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chordscope.analyzers.beats import estimate_beats_and_meter
from chordscope.analyzers.tempo import estimate_bpm_madmom
from chordscope.audio import DEFAULT_SR, MADMOM_SR, AudioBuffer, madmom_samples

_BPM = 100.0
_DURATION_SEC = 20.0


def _click_track(bpm: float, duration: float, sr: int) -> AudioBuffer:
    samples = np.zeros(int(duration * sr), dtype=np.float32)
    click_len = int(0.03 * sr)
    t = np.arange(click_len) / sr
    for i, onset in enumerate(np.arange(0.5, duration - 0.1, 60.0 / bpm)):
        freq = 1500.0 if i % 4 == 0 else 1000.0
        click = np.sin(2 * np.pi * freq * t) * np.exp(-t * 120.0)
        start = int(onset * sr)
        samples[start : start + click_len] += click.astype(np.float32)[: len(samples) - start]
    return AudioBuffer(samples=samples, sample_rate=sr, duration=duration, path=Path("click.wav"))


@pytest.fixture(scope="module")
def click_buffer() -> AudioBuffer:
    return _click_track(_BPM, _DURATION_SEC, DEFAULT_SR)


def test_beat_times_span_the_whole_buffer_at_default_sample_rate(
    click_buffer: AudioBuffer,
) -> None:
    beats, _meter = estimate_beats_and_meter(click_buffer)

    assert beats.beat_times[-1] >= 0.8 * _DURATION_SEC


def test_beat_interval_matches_click_tempo_at_default_sample_rate(
    click_buffer: AudioBuffer,
) -> None:
    beats, _meter = estimate_beats_and_meter(click_buffer)

    median_ibi = float(np.median(np.diff(beats.beat_times)))
    assert median_ibi == pytest.approx(60.0 / _BPM, rel=0.05)


def test_madmom_bpm_matches_click_tempo_at_default_sample_rate(
    click_buffer: AudioBuffer,
) -> None:
    assert estimate_bpm_madmom(click_buffer) == pytest.approx(_BPM, rel=0.05)


def test_madmom_samples_doubles_length_of_22050_buffer(click_buffer: AudioBuffer) -> None:
    resampled = madmom_samples(click_buffer)

    assert len(resampled) == 2 * len(click_buffer.samples)


def test_madmom_samples_returns_44100_buffer_as_is() -> None:
    buffer = _click_track(_BPM, 1.0, MADMOM_SR)

    assert madmom_samples(buffer) is buffer.samples


def test_beat_interval_matches_click_tempo_at_44100() -> None:
    buffer = _click_track(_BPM, _DURATION_SEC, MADMOM_SR)

    beats, _meter = estimate_beats_and_meter(buffer)

    median_ibi = float(np.median(np.diff(beats.beat_times)))
    assert median_ibi == pytest.approx(60.0 / _BPM, rel=0.05)
