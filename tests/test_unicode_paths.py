"""日本語・全角空白・半角空白を含むファイル名でも全パイプラインが動作することを検証。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chordscope.analyzers.chords import recognize_chords
from chordscope.analyzers.tempo import estimate_tempo
from chordscope.audio import load_audio
from chordscope.discovery import discover_audio_files
from chordscope.reporting.json_report import write_json
from chordscope.reporting.markdown import render_markdown, write_markdown

# Unicode/空白を意図的に混ぜた検証用ファイル名
TRICKY_FILENAMES = [
    "曲タイトル　日本語テスト.ogg",  # 日本語 + 全角空白
    "with half width spaces.ogg",  # 半角空白
    "café español ñ.ogg",  # 拉丁系アクセント
    "曲 - artist (live).ogg",  # 日本語 + 半角空白 + ()
]


@pytest.fixture(scope="module")
def tricky_audio_dir(pd_encina_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """PD音源 (Encina) を上記の各種ファイル名にコピーした一時ディレクトリ。"""
    base = tmp_path_factory.mktemp("tricky_names")
    for name in TRICKY_FILENAMES:
        shutil.copy(pd_encina_path, base / name)
    return base


@pytest.mark.integration
def test_discovery_finds_unicode_filenames(tricky_audio_dir: Path) -> None:
    files = discover_audio_files([tricky_audio_dir])
    found = sorted(f.name for f in files)
    assert found == sorted(TRICKY_FILENAMES), (
        f"見つからないファイル: {set(TRICKY_FILENAMES) - set(found)}"
    )


@pytest.mark.integration
def test_load_audio_handles_unicode_filenames(tricky_audio_dir: Path) -> None:
    for name in TRICKY_FILENAMES:
        path = tricky_audio_dir / name
        buf = load_audio(path)
        assert buf.duration > 60.0, f"{name}: 読み込めていない (duration={buf.duration})"


@pytest.mark.integration
@pytest.mark.slow
def test_full_analysis_chain_with_unicode_filename(tricky_audio_dir: Path, tmp_path: Path) -> None:
    """日本語ファイル名で BPM 推定・コード認識・JSON/Markdown 書き出しが完走。"""
    path = tricky_audio_dir / TRICKY_FILENAMES[0]
    buf = load_audio(path)
    tempo = estimate_tempo(buf)
    chords = recognize_chords(buf, engine="madmom")
    assert tempo.bpm > 0
    assert len(chords.segments) > 0
    # 出力 (ファイル名に Unicode を含む)
    from chordscope.models import (
        BeatResult,
        HarmonicAnalysis,
        KeyResult,
        MeterResult,
        TrackAnalysis,
    )

    track = TrackAnalysis(
        file_path=path,
        duration_sec=buf.duration,
        sample_rate=buf.sample_rate,
        tempo=tempo,
        beats=BeatResult(beat_times=[], downbeat_times=[], beats_per_bar=4, method="dummy"),
        meter=MeterResult(numerator=4, denominator=4, confidence=0.5),
        key=KeyResult(tonic="C", mode="major", confidence=0.5, correlation=0.5),
        chords=chords,
        harmony=HarmonicAnalysis(),
        plot_paths={"waveform_beats": tmp_path / f"{path.stem}__plot.png"},
    )
    json_out = tmp_path / f"{path.stem}.json"
    md_out = tmp_path / f"{path.stem}.md"
    write_json(track, json_out)
    write_markdown(track, md_out)
    assert json_out.exists() and json_out.stat().st_size > 0
    assert md_out.exists() and md_out.stat().st_size > 0
    md_text = md_out.read_text(encoding="utf-8")
    assert path.name in md_text  # 日本語ファイル名がそのままタイトルに


def test_markdown_url_encodes_unicode_image_names() -> None:
    """グラフ画像リンクは URL エンコードされる (日本語/空白を含むため)。"""
    from chordscope.models import (
        BeatResult,
        ChordResult,
        ChordSegment,
        HarmonicAnalysis,
        KeyResult,
        MeterResult,
        TempoResult,
        TrackAnalysis,
    )

    track = TrackAnalysis(
        file_path=Path("/tmp/曲 名前.mp3"),
        duration_sec=30.0,
        sample_rate=22050,
        tempo=TempoResult(bpm=120.0, bpm_candidates=[120.0], confidence=1.0, method="m"),
        beats=BeatResult(beat_times=[], downbeat_times=[], beats_per_bar=4, method="m"),
        meter=MeterResult(numerator=4, denominator=4, confidence=1.0),
        key=KeyResult(tonic="C", mode="major", confidence=1.0, correlation=1.0),
        chords=ChordResult(
            segments=[ChordSegment(start=0, end=1, label="C:maj")],
            method="m",
            unique_chords=["C:maj"],
        ),
        harmony=HarmonicAnalysis(),
        plot_paths={"waveform_beats": Path("/tmp/曲 名前__waveform.png")},
    )
    md = render_markdown(track)
    # スペースが %20 に、日本語が UTF-8 percent-encoded になる
    assert "%20" in md or "%E6" in md  # 空白または日本語が percent encoded
    # マークダウン画像構文は破綻していない
    assert "![waveform_beats](" in md
