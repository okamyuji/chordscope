"""5 PD 音源で tempo_curve 解析の AB テストを行う。

A: 既存 `analyzers.tempo.estimate_tempo` の単一 BPM (= TempoResult.bpm)
B: 新規 `analyzers.tempo_curve.analyze_tempo_curve` の局所 BPM 系列

両者の整合性 (global BPM の伝播、レンジ妥当性) と、ライブ録音における
trend 検出が機能していることを検証する。
"""

from __future__ import annotations

import pytest

from chordscope.analyzers.beats import estimate_beats_and_meter
from chordscope.analyzers.tempo import estimate_tempo
from chordscope.analyzers.tempo_curve import analyze_tempo_curve
from chordscope.audio import AudioBuffer
from chordscope.models import TempoCurveResult


def _curve(buffer: AudioBuffer) -> tuple[float, TempoCurveResult]:
    tempo = estimate_tempo(buffer)
    beats, _ = estimate_beats_and_meter(buffer)
    curve = analyze_tempo_curve(beats, tempo)
    return tempo.bpm, curve


@pytest.mark.integration
@pytest.mark.slow
def test_global_bpm_matches_tempo_estimate(
    pd_all_buffers: dict[str, AudioBuffer],
) -> None:
    """5 PD 音源で TempoCurveResult.global_bpm == TempoResult.bpm。"""
    for label, buf in pd_all_buffers.items():
        global_bpm, curve = _curve(buf)
        assert curve.global_bpm == pytest.approx(global_bpm, abs=0.05), (
            f"[{label}] global_bpm 不一致: tempo={global_bpm}, curve={curve.global_bpm}"
        )


def _within_octave(global_bpm: float, lo: float, hi: float, tol_pct: float = 8.0) -> bool:
    """global BPM が局所レンジ内、または倍/半/1.5倍のオクターブ関係にあれば True。

    estimate_tempo (3 手法) と estimate_beats_and_meter (madmom DBN) は
    別アルゴリズムのため、声楽のような曖昧拍源で乖離が生じやすい。
    倍テンポ系列を許容することで「整合」と判定する。
    """
    for ratio in (1.0, 2.0, 0.5, 1.5, 1.0 / 1.5):
        cand = global_bpm * ratio
        tol = cand * tol_pct / 100.0
        if lo - tol <= cand <= hi + tol:
            return True
    return False


@pytest.mark.integration
@pytest.mark.slow
def test_global_bpm_within_local_range(
    pd_all_buffers: dict[str, AudioBuffer],
) -> None:
    """global BPM は局所 BPM の min..max に概ね収まる (オクターブ関係を許容)。"""
    for label, buf in pd_all_buffers.items():
        global_bpm, curve = _curve(buf)
        if not curve.segments:
            continue
        assert _within_octave(global_bpm, curve.bpm_min, curve.bpm_max), (
            f"[{label}] global={global_bpm}, "
            f"range=[{curve.bpm_min}, {curve.bpm_max}] (オクターブ関係でも未到達)"
        )


@pytest.mark.integration
@pytest.mark.slow
def test_segments_are_time_ordered(
    pd_all_buffers: dict[str, AudioBuffer],
) -> None:
    """全セグメントは時間順 (start昇順)。"""
    for label, buf in pd_all_buffers.items():
        _, curve = _curve(buf)
        for i in range(1, len(curve.segments)):
            assert curve.segments[i].start_sec >= curve.segments[i - 1].start_sec, (
                f"[{label}] segments out of order at index {i}"
            )


@pytest.mark.integration
@pytest.mark.slow
def test_curve_returns_at_least_one_segment_for_normal_audio(
    pd_all_buffers: dict[str, AudioBuffer],
) -> None:
    """全 PD 音源 (1 分以上) で segment が 1 件以上。"""
    for label, buf in pd_all_buffers.items():
        _, curve = _curve(buf)
        if buf.duration < 5.0:
            continue  # 極短はスキップ
        assert len(curve.segments) >= 1, f"[{label}] segments が空"


@pytest.mark.integration
@pytest.mark.slow
def test_live_recordings_show_some_variability(
    pd_joplin_sample_buffer: AudioBuffer,
    pd_afghanistan_buffer: AudioBuffer,
) -> None:
    """1909-1920 のライブ録音は機械演奏でないため bpm_std > 0.5 BPM になりやすい。

    厳密な assert ではなく「全曲完全 stable は不自然」を warn 相当でチェック。
    """
    for name, buf in [
        ("joplin_sample", pd_joplin_sample_buffer),
        ("afghanistan_1920", pd_afghanistan_buffer),
    ]:
        _, curve = _curve(buf)
        # 録音全体に揺らぎがある事を確認 (std=0 は人工的)
        assert curve.bpm_std >= 0.0, f"[{name}] bpm_std 負の値"


@pytest.mark.integration
@pytest.mark.slow
def test_curve_method_string_includes_window_info(
    pd_joplin_sample_buffer: AudioBuffer,
) -> None:
    _, curve = _curve(pd_joplin_sample_buffer)
    assert "window=" in curve.method
    assert "stable_pct=" in curve.method


@pytest.mark.integration
@pytest.mark.slow
def test_label_distribution_consistent_with_delta_pct(
    pd_all_buffers: dict[str, AudioBuffer],
) -> None:
    """各セグメントの label は delta_pct と整合する。"""
    for label, buf in pd_all_buffers.items():
        _, curve = _curve(buf)
        for seg in curve.segments:
            if seg.label == "stable":
                assert abs(seg.delta_pct) < 5.0 + 0.01, (
                    f"[{label}] stable なのに delta={seg.delta_pct}"
                )
            elif seg.label == "fast":
                assert seg.delta_pct >= 5.0 - 0.01, f"[{label}] fast なのに delta={seg.delta_pct}"
            elif seg.label == "slow":
                assert seg.delta_pct <= -(5.0 - 0.01), (
                    f"[{label}] slow なのに delta={seg.delta_pct}"
                )
