"""narrative レポートの構造テスト。"""

from __future__ import annotations

from pathlib import Path

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
from chordscope.reporting.narrative import render_narrative, write_narrative


def _track() -> TrackAnalysis:
    return TrackAnalysis(
        file_path=Path("/tmp/曲タイトル.mp3"),
        duration_sec=240.0,
        sample_rate=22050,
        tempo=TempoResult(
            bpm=128.5, bpm_candidates=[128.0, 128.5, 130.0], confidence=0.85, method="m1+m2"
        ),
        beats=BeatResult(
            beat_times=[0.0, 0.5, 1.0],
            downbeat_times=[0.0],
            beats_per_bar=4,
            method="madmom",
        ),
        meter=MeterResult(numerator=4, denominator=4, confidence=0.95),
        key=KeyResult(
            tonic="A",
            mode="minor",
            confidence=0.7,
            correlation=0.62,
            second_best=("C", "major"),
        ),
        chords=ChordResult(
            segments=[
                ChordSegment(start=0.0, end=2.0, label="A:min"),
                ChordSegment(start=2.0, end=4.0, label="F:maj"),
                ChordSegment(start=4.0, end=6.0, label="C:maj"),
                ChordSegment(start=6.0, end=8.0, label="G:maj"),
                ChordSegment(start=8.0, end=10.0, label="A:min"),
            ],
            method="madmom-DeepChroma",
            unique_chords=["A:min", "F:maj", "C:maj", "G:maj"],
        ),
        harmony=HarmonicAnalysis(
            roman_numerals=["i", "VI", "III", "VII", "i"],
            cadences=["Plagal Cadence (iv→i)", "Authentic Cadence (V→i)"],
            modulations=["Possible key change near chord index 8 (A → C)"],
            chord_categories={"T": 3, "SD": 1, "D": 1},
        ),
        modulation=ModulationResult(
            window_sec=16.0,
            hop_sec=4.0,
            segments=[
                KeySegment(
                    start_sec=0.0,
                    end_sec=84.0,
                    tonic="A",
                    mode="minor",
                    confidence=0.7,
                    correlation=0.55,
                ),
                KeySegment(
                    start_sec=84.0,
                    end_sec=240.0,
                    tonic="C",
                    mode="major",
                    confidence=0.65,
                    correlation=0.5,
                ),
            ],
            changes=[
                KeyChange(
                    at_sec=84.0,
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
            global_bpm=128.5,
            bpm_min=122.0,
            bpm_max=138.0,
            bpm_std=4.5,
            segments=[
                TempoSegment(
                    start_sec=0.0,
                    end_sec=72.0,
                    local_bpm=122.0,
                    delta_pct=-5.1,
                    label="slow",
                ),
                TempoSegment(
                    start_sec=72.0,
                    end_sec=240.0,
                    local_bpm=138.0,
                    delta_pct=7.4,
                    label="fast",
                ),
            ],
            trend="accelerando",
            method="curve-test",
        ),
        genre=GenreResult(
            top=GenreScore(label="Pop music", score=0.42),
            distribution=[
                GenreScore(label="Pop music", score=0.42),
                GenreScore(label="Rock music", score=0.31),
                GenreScore(label="Jazz", score=0.12),
            ],
            model_id="MIT/ast-finetuned-audioset-10-10-0.4593",
            aggregation="mean",
        ),
        style_notes=[
            StyleNotes(
                style="Pop music",
                findings=[
                    "AST 推論スコア 42.00%",
                    "三和音中心 (100%) — Rock/Pop 系のシンプルな和声",
                ],
                metrics={"ast_score": 0.42, "triad_ratio": 1.0, "power_chord_ratio": 0.05},
            ),
            StyleNotes(
                style="Jazz",
                findings=["AST 推論スコア 12.00%", "シンコペーション指数 0.55"],
                metrics={"ast_score": 0.12, "syncopation_index": 0.55, "ii_v_i_count": 1.0},
            ),
        ],
    )


def test_render_narrative_contains_all_sections() -> None:
    md = render_narrative(_track())
    expected_headers = [
        "# 音楽分析: 曲タイトル.mp3",
        "## 1. 楽曲の骨格",
        "## 2. 和声の流れと特徴",
        "## 3. リズム・グルーヴの性格",
        "## 4. ジャンル傾向",
        "## 5. この曲を特徴づける指標",
        "## 6. コード進行の時系列概観",
        "## 7. 算出根拠",
    ]
    for h in expected_headers:
        assert h in md, f"見出し欠落: {h}"


def test_render_narrative_includes_key_facts() -> None:
    md = render_narrative(_track())
    # 骨格
    assert "128.50" in md  # BPM
    assert "4/4" in md
    assert "A minor" in md
    # 和声
    assert "`A:min`" in md  # 出現上位コード
    assert "Plagal Cadence" in md
    # 旧 harmony.modulations は track.modulation がある場合は出ない (置き換え)
    # 新方式の「時系列キー解析で N 箇所の転調を検出」が出る
    assert "転調を検出" in md
    # ジャンル
    assert "Pop music" in md
    assert "Jazz" in md
    # 特徴指標
    assert "三和音比率" in md
    assert "シンコペーション指数" in md
    # 時系列 (8 ビン目安)
    assert "0:00-" in md
    # 技術メモ
    assert "madmom-DeepChroma" in md
    assert "MIT/ast" in md


def test_narrative_handles_no_genre() -> None:
    track = _track()
    track_no_genre = track.model_copy(update={"genre": None, "style_notes": []})
    md = render_narrative(track_no_genre)
    assert "ジャンル分類は実行されていません" in md


def test_narrative_includes_modulation_changes() -> None:
    md = render_narrative(_track())
    # 第 2 章に転調イベントが時刻付きで列挙される
    assert "1 箇所" in md  # changes count
    assert "A minor" in md
    assert "C major" in md
    assert "relative" in md
    # フォーマット: "X:XX で <from> → <to>"
    assert "1:24" in md  # 84.0 秒 = 1:24


def test_narrative_includes_tempo_curve_trend_and_segments() -> None:
    md = render_narrative(_track())
    # 第 3 章にテンポ推移
    assert "加速 (accelerando)" in md
    assert "122.0" in md
    assert "138.0" in md
    # 速い区間 / 遅い区間 がそれぞれ列挙される
    assert "速い区間" in md
    assert "遅い区間" in md


def test_narrative_handles_no_modulation_or_tempo_curve() -> None:
    """新フィールドが None でも narrative が落ちず、フォールバックが働く。"""
    track = _track()
    track_minimal = track.model_copy(update={"modulation": None, "tempo_curve": None})
    md = render_narrative(track_minimal)
    # 既存の `harmony.modulations` フォールバックは出る
    assert "転調候補" in md


def test_write_narrative_writes_unicode_filename(tmp_path: Path) -> None:
    track = _track()
    out = tmp_path / "曲タイトル_analysis.md"
    write_narrative(track, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "音楽分析:" in content


def test_narrative_logical_order() -> None:
    """章 1 (骨格) は章 2 (和声) より前に出現する等、論理順序の保証。"""
    md = render_narrative(_track())
    positions = {
        title: md.index(title)
        for title in ("## 1. ", "## 2. ", "## 3. ", "## 4. ", "## 5. ", "## 6. ", "## 7. ")
    }
    sorted_keys = sorted(positions, key=lambda k: positions[k])
    assert sorted_keys == ["## 1. ", "## 2. ", "## 3. ", "## 4. ", "## 5. ", "## 6. ", "## 7. "]
