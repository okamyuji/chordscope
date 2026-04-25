"""データモデル: 不変性・バリデーションテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chordscope.models import (
    BeatResult,
    ChordResult,
    ChordSegment,
    GenreResult,
    GenreScore,
    HarmonicAnalysis,
    KeyResult,
    MeterResult,
    TempoResult,
    TrackAnalysis,
)


def _make_track() -> TrackAnalysis:
    return TrackAnalysis(
        file_path=Path("/tmp/x.mp3"),
        duration_sec=10.0,
        sample_rate=22050,
        tempo=TempoResult(bpm=120.0, bpm_candidates=[120.0], confidence=1.0, method="m"),
        beats=BeatResult(beat_times=[0.0], downbeat_times=[0.0], beats_per_bar=4, method="m"),
        meter=MeterResult(numerator=4, denominator=4, confidence=1.0),
        key=KeyResult(tonic="C", mode="major", confidence=1.0, correlation=0.9),
        chords=ChordResult(
            segments=[ChordSegment(start=0.0, end=1.0, label="C:maj")],
            method="m",
            unique_chords=["C:maj"],
        ),
        harmony=HarmonicAnalysis(),
        genre=GenreResult(
            top=GenreScore(label="Pop music", score=0.9),
            distribution=[GenreScore(label="Pop music", score=0.9)],
            model_id="x",
            aggregation="mean",
        ),
    )


def test_track_is_frozen() -> None:
    t = _make_track()
    with pytest.raises(ValidationError):
        t.duration_sec = 99.0  # type: ignore[misc]


def test_meter_invalid_numerator() -> None:
    with pytest.raises(ValidationError):
        MeterResult(numerator=1, denominator=4, confidence=0.5)


def test_genre_score_in_range() -> None:
    with pytest.raises(ValidationError):
        GenreScore(label="x", score=1.5)


def test_track_serializes_json() -> None:
    t = _make_track()
    js = t.model_dump_json()
    assert "C:maj" in js
    assert "Pop music" in js
