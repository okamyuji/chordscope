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
    KeyChange,
    KeyResult,
    KeySegment,
    MeterResult,
    ModulationResult,
    TempoCurveResult,
    TempoResult,
    TempoSegment,
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


def test_key_segment_validates_mode() -> None:
    with pytest.raises(ValidationError):
        KeySegment(
            start_sec=0.0,
            end_sec=4.0,
            tonic="C",
            mode="dorian",  # type: ignore[arg-type]
            confidence=0.5,
            correlation=0.4,
        )


def test_key_change_validates_relation() -> None:
    with pytest.raises(ValidationError):
        KeyChange(
            at_sec=10.0,
            from_tonic="C",
            from_mode="major",
            to_tonic="G",
            to_mode="major",
            interval_semitones=7,
            relation="weird",  # type: ignore[arg-type]
        )


def test_modulation_result_is_frozen() -> None:
    mod = ModulationResult(window_sec=16.0, hop_sec=4.0, segments=[], changes=[], method="x")
    with pytest.raises(ValidationError):
        mod.window_sec = 8.0  # type: ignore[misc]


def test_tempo_curve_validates_trend() -> None:
    with pytest.raises(ValidationError):
        TempoCurveResult(
            window_beats=8,
            global_bpm=120.0,
            bpm_min=110.0,
            bpm_max=130.0,
            bpm_std=2.0,
            segments=[],
            trend="oscillating",  # type: ignore[arg-type]
            method="x",
        )


def test_tempo_segment_validates_label() -> None:
    with pytest.raises(ValidationError):
        TempoSegment(
            start_sec=0.0,
            end_sec=4.0,
            local_bpm=120.0,
            delta_pct=0.0,
            label="medium",  # type: ignore[arg-type]
        )


def test_track_with_modulation_and_tempo_curve_serializes() -> None:
    t = _make_track().model_copy(
        update={
            "modulation": ModulationResult(
                window_sec=16.0,
                hop_sec=4.0,
                segments=[
                    KeySegment(
                        start_sec=0.0,
                        end_sec=20.0,
                        tonic="C",
                        mode="major",
                        confidence=0.8,
                        correlation=0.7,
                    )
                ],
                changes=[],
                method="ks-test",
            ),
            "tempo_curve": TempoCurveResult(
                window_beats=8,
                global_bpm=120.0,
                bpm_min=118.0,
                bpm_max=124.0,
                bpm_std=1.5,
                segments=[
                    TempoSegment(
                        start_sec=0.0,
                        end_sec=10.0,
                        local_bpm=120.0,
                        delta_pct=0.0,
                        label="stable",
                    )
                ],
                trend="stable",
                method="curve-test",
            ),
        }
    )
    payload = t.model_dump_json()
    assert "ks-test" in payload
    assert "curve-test" in payload
