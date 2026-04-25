"""PD音源を ground truth とした AB テスト群 (analyzers)。

各テストは「物理的に検証可能な範囲・系統」に分析結果が入っているかを確認する。
厳密な数値一致ではなく、音楽学的な合理性の検証。
"""

from __future__ import annotations

import pytest

from chordscope.analyzers.beats import estimate_beats_and_meter
from chordscope.analyzers.chords import recognize_chords
from chordscope.analyzers.key import estimate_key
from chordscope.analyzers.style import analyze_styles
from chordscope.analyzers.tempo import estimate_tempo
from chordscope.analyzers.theory import analyze_harmony


@pytest.mark.integration
@pytest.mark.slow
def test_tempo_in_known_range_for_joplin(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    """Maple Leaf Rag は ragtime tempo (一般に 100-180 BPM)。"""
    result = estimate_tempo(pd_joplin_sample_buffer)
    assert 70.0 <= result.bpm <= 220.0, f"BPM {result.bpm} は ragtime レンジ外"
    assert len(result.bpm_candidates) >= 2  # 複数手法で集約しているはず


@pytest.mark.integration
@pytest.mark.slow
def test_meter_for_joplin_should_be_two_or_four(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    """ラグタイムは 2/4 または 4/4 拍子。"""
    _beats, meter = estimate_beats_and_meter(pd_joplin_sample_buffer)
    assert meter.numerator in (2, 4), f"想定外の拍子分子: {meter.numerator}"


@pytest.mark.integration
@pytest.mark.slow
def test_key_for_encina_returns_valid_label(pd_encina_buffer) -> None:  # type: ignore[no-untyped-def]
    """Encina の声楽はモード旋法だが、単一キー推定は major/minor のいずれかを返す。"""
    result = estimate_key(pd_encina_buffer)
    assert result.tonic in {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
    assert result.mode in {"major", "minor"}
    assert -1.0 <= result.correlation <= 1.0


@pytest.mark.integration
@pytest.mark.slow
def test_chords_madmom_produces_segments(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    """madmom DeepChroma は最低限のコード区間を返す。"""
    result = recognize_chords(pd_joplin_sample_buffer, engine="madmom")
    assert len(result.segments) > 0
    # 全セグメントは start < end
    for seg in result.segments:
        assert seg.end > seg.start
    # 1 種類以上のコードを検出
    assert len(result.unique_chords) >= 1


@pytest.mark.integration
@pytest.mark.slow
def test_chords_librosa_template_engine(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    """フォールバックエンジンも単独で動作する。"""
    result = recognize_chords(pd_joplin_sample_buffer, engine="librosa-template")
    assert len(result.segments) > 0
    assert "viterbi" in result.method


@pytest.mark.integration
@pytest.mark.slow
def test_theory_analysis_produces_romans(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    """コード列を music21 でローマ数字に展開できる。"""
    chords = recognize_chords(pd_joplin_sample_buffer, engine="madmom")
    key = estimate_key(pd_joplin_sample_buffer)
    harmony = analyze_harmony(chords, key)
    assert len(harmony.roman_numerals) > 0
    # 機能カテゴリも何らかに分類される
    if harmony.chord_categories:
        assert sum(harmony.chord_categories.values()) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_style_analyses_jazz_for_ragtime(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    """Maple Leaf Rag は Jazz/Ragtime 系のため、Jazz 観点で何らかの finding が出ることを期待。"""
    tempo = estimate_tempo(pd_joplin_sample_buffer)
    beats, meter = estimate_beats_and_meter(pd_joplin_sample_buffer)
    key = estimate_key(pd_joplin_sample_buffer)
    chords = recognize_chords(pd_joplin_sample_buffer, engine="madmom")
    harmony = analyze_harmony(chords, key)
    notes = analyze_styles(
        buffer=pd_joplin_sample_buffer,
        tempo=tempo,
        beats=beats,
        meter=meter,
        key=key,
        chords=chords,
        harmony=harmony,
        enabled=["jazz", "classic", "jpop", "rock"],
    )
    assert {n.style for n in notes} == {"jazz", "classic", "jpop", "rock"}
    jazz = next(n for n in notes if n.style == "jazz")
    # 7th 比率と syncopation index は metrics に含まれる
    assert "seventh_ratio" in jazz.metrics
    assert "syncopation_index" in jazz.metrics
