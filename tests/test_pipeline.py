"""パイプライン全体の統合テスト。

- PD 音源とユーザー任意音源の両方で `analyze_file` が完走する
- TrackAnalysis の必須フィールドが埋まる
- グラフ画像が指定先に書き出される
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chordscope.analyzers.genre import GenreClassifier
from chordscope.pipeline import AnalysisOptions, analyze_file


@pytest.fixture(scope="module")
def shared_classifier() -> GenreClassifier:
    return GenreClassifier()


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_on_pd_joplin(
    pd_joplin_sample_path: Path,
    shared_classifier: GenreClassifier,
    tmp_path: Path,
) -> None:
    opts = AnalysisOptions(
        chord_engine="madmom",
        enable_genre=True,
        enabled_styles=("jazz", "classic", "jpop", "rock"),
        plots=True,
        plot_dir=tmp_path / "plots",
        genre_classifier=shared_classifier,
    )
    result = analyze_file(pd_joplin_sample_path, opts)
    assert result.duration_sec > 0
    assert result.tempo.bpm > 0
    assert result.beats.beats_per_bar in (2, 3, 4, 5, 6, 7)
    assert result.key.tonic in {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
    assert len(result.chords.segments) > 0
    assert result.genre is not None
    assert len(result.style_notes) == 4
    assert len(result.plot_paths) == 4
    for p in result.plot_paths.values():
        assert p.exists() and p.stat().st_size > 0


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_on_user_audio(
    user_audio_path: Path,
    shared_classifier: GenreClassifier,
    tmp_path: Path,
) -> None:
    """ユーザー任意の音源 (ファイル名固定なし) で全パイプラインが完走する。"""
    opts = AnalysisOptions(
        chord_engine="madmom",
        enable_genre=True,
        enabled_styles=("jazz", "classic", "jpop", "rock"),
        plots=True,
        plot_dir=tmp_path / "plots",
        genre_classifier=shared_classifier,
    )
    result = analyze_file(user_audio_path, opts)
    # ファイルの内容に依存しない最低限の合理性のみ検証。
    assert result.tempo.bpm > 30.0
    assert result.tempo.bpm < 300.0
    assert len(result.chords.segments) >= 1
    assert result.genre is not None
    for p in result.plot_paths.values():
        assert p.exists() and p.stat().st_size > 5_000


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_with_librosa_template_engine(
    pd_joplin_sample_path: Path,
    tmp_path: Path,
) -> None:
    """フォールバックエンジンでも全パイプライン完走。"""
    opts = AnalysisOptions(
        chord_engine="librosa-template",
        enable_genre=False,
        enabled_styles=("jazz",),
        plots=False,
        plot_dir=None,
    )
    result = analyze_file(pd_joplin_sample_path, opts)
    assert "librosa" in result.chords.method
    assert len(result.chords.segments) > 0
    assert result.genre is None
