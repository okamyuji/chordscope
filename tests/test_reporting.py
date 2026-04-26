"""出力レポートのスモークテスト。"""

from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

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
    StyleNotes,
    TempoCurveResult,
    TempoResult,
    TempoSegment,
    TrackAnalysis,
)
from chordscope.reporting.console import render_track
from chordscope.reporting.json_report import write_json
from chordscope.reporting.markdown import render_markdown, write_markdown


def _track() -> TrackAnalysis:
    return TrackAnalysis(
        file_path=Path("/tmp/sample.mp3"),
        duration_sec=180.0,
        sample_rate=22050,
        tempo=TempoResult(bpm=120.5, bpm_candidates=[120.0, 121.0], confidence=0.9, method="m1+m2"),
        beats=BeatResult(
            beat_times=[0.0, 0.5, 1.0], downbeat_times=[0.0], beats_per_bar=4, method="m"
        ),
        meter=MeterResult(numerator=4, denominator=4, confidence=0.8),
        key=KeyResult(
            tonic="A", mode="minor", confidence=0.7, correlation=0.65, second_best=("C", "major")
        ),
        chords=ChordResult(
            segments=[
                ChordSegment(start=0.0, end=2.0, label="A:min"),
                ChordSegment(start=2.0, end=4.0, label="F:maj"),
            ],
            method="madmom",
            unique_chords=["A:min", "F:maj"],
        ),
        harmony=HarmonicAnalysis(
            roman_numerals=["i", "VI"],
            cadences=["Plagal Cadence (iv→i)"],
            modulations=[],
            chord_categories={"T": 1, "SD": 1},
        ),
        modulation=ModulationResult(
            window_sec=16.0,
            hop_sec=4.0,
            segments=[
                KeySegment(
                    start_sec=0.0,
                    end_sec=60.0,
                    tonic="A",
                    mode="minor",
                    confidence=0.7,
                    correlation=0.5,
                ),
                KeySegment(
                    start_sec=60.0,
                    end_sec=180.0,
                    tonic="C",
                    mode="major",
                    confidence=0.65,
                    correlation=0.45,
                ),
            ],
            changes=[
                KeyChange(
                    at_sec=60.0,
                    from_tonic="A",
                    from_mode="minor",
                    to_tonic="C",
                    to_mode="major",
                    interval_semitones=3,
                    relation="relative",
                ),
            ],
            method="ks-test",
        ),
        tempo_curve=TempoCurveResult(
            window_beats=8,
            global_bpm=120.5,
            bpm_min=118.0,
            bpm_max=128.0,
            bpm_std=3.5,
            segments=[
                TempoSegment(
                    start_sec=0.0,
                    end_sec=90.0,
                    local_bpm=119.0,
                    delta_pct=-1.2,
                    label="stable",
                ),
                TempoSegment(
                    start_sec=90.0,
                    end_sec=180.0,
                    local_bpm=128.0,
                    delta_pct=6.2,
                    label="fast",
                ),
            ],
            trend="accelerando",
            method="curve-test",
        ),
        genre=GenreResult(
            top=GenreScore(label="Pop music", score=0.61),
            distribution=[
                GenreScore(label="Pop music", score=0.61),
                GenreScore(label="Rock music", score=0.22),
            ],
            model_id="MIT/ast-finetuned-audioset-10-10-0.4593",
            aggregation="mean",
        ),
        style_notes=[
            StyleNotes(
                style="jpop",
                findings=["王道進行検出"],
                metrics={"ohyou_progression_count": 1.0},
            )
        ],
    )


def test_render_console_does_not_crash() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    render_track(_track(), console)
    out = buf.getvalue()
    assert "120.50 BPM" in out
    assert "A minor" in out
    assert "Pop music" in out
    # 新セクション
    assert "trend=accelerando" in out
    assert "modulations=1件" in out
    assert "Tempo curve segments" in out
    assert "Key changes" in out


def test_render_markdown_includes_tempo_curve_and_modulation() -> None:
    md = render_markdown(_track())
    assert "テンポ変動" in md
    assert "転調検出" in md
    assert "accelerando" in md
    assert "relative" in md
    assert "C major" in md or "C major" in md  # 念のため
    assert "A minor" in md


def test_write_json(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    write_json(_track(), out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tempo"]["bpm"] == 120.5
    assert payload["chords"]["segments"][0]["label"] == "A:min"
    assert payload["genre"]["top"]["label"] == "Pop music"


def test_render_markdown_includes_sections() -> None:
    md = render_markdown(_track())
    assert "# 分析レポート: sample.mp3" in md
    assert "120.50 BPM" in md
    assert "A:min" in md
    assert "ジャンル分布" in md
    assert "ローマ数字" in md
    assert "王道進行検出" in md


def test_write_markdown(tmp_path: Path) -> None:
    out = tmp_path / "x.md"
    write_markdown(_track(), out)
    assert out.exists()
    assert "分析レポート" in out.read_text(encoding="utf-8")
