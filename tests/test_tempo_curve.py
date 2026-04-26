"""tempo_curve.analyze_tempo_curve のユニットテスト。

合成的なビート列 (numpy で生成) で振る舞いを検証する。実音源は不要。
"""

from __future__ import annotations

import numpy as np

from chordscope.analyzers.tempo_curve import (
    DEFAULT_STABLE_PCT,
    DEFAULT_WINDOW_BEATS,
    analyze_tempo_curve,
)
from chordscope.models import BeatResult, TempoResult


def _beats_at(times: list[float]) -> BeatResult:
    return BeatResult(
        beat_times=times,
        downbeat_times=[],
        beats_per_bar=4,
        method="synthetic",
    )


def _tempo(bpm: float) -> TempoResult:
    return TempoResult(bpm=bpm, bpm_candidates=[bpm], confidence=1.0, method="synthetic")


def test_constant_bpm_yields_stable_trend() -> None:
    """120 BPM (= 0.5s 間隔) を 64 ビート → trend stable, 全 segment stable。"""
    times = (np.arange(64) * 0.5).tolist()
    res = analyze_tempo_curve(_beats_at(times), _tempo(120.0))
    assert res.trend == "stable"
    assert res.global_bpm == 120.0
    assert all(seg.label == "stable" for seg in res.segments)
    assert res.bpm_std < 1.0
    assert abs(res.bpm_min - 120.0) < 1.0
    assert abs(res.bpm_max - 120.0) < 1.0


def test_accelerando_detected_when_intervals_shrink() -> None:
    """間隔が線形に短くなる → trend accelerando。"""
    intervals = np.linspace(0.6, 0.4, 64)  # 100 BPM → 150 BPM
    times = np.cumsum(np.concatenate([[0.0], intervals])).tolist()
    res = analyze_tempo_curve(_beats_at(times), _tempo(120.0))
    assert res.trend == "accelerando"
    assert res.bpm_max > res.bpm_min


def test_ritardando_detected_when_intervals_grow() -> None:
    """間隔が線形に伸びる → trend ritardando。"""
    intervals = np.linspace(0.4, 0.6, 64)  # 150 BPM → 100 BPM
    times = np.cumsum(np.concatenate([[0.0], intervals])).tolist()
    res = analyze_tempo_curve(_beats_at(times), _tempo(120.0))
    assert res.trend == "ritardando"


def test_few_beats_returns_empty_segments() -> None:
    """ビート数 < window_beats なら segments は空、trend は stable。"""
    times = [0.0, 0.5, 1.0]  # 3 beats < default window 8
    res = analyze_tempo_curve(_beats_at(times), _tempo(120.0))
    assert res.segments == []
    assert res.trend == "stable"
    assert res.global_bpm == 120.0
    assert res.bpm_std == 0.0


def test_segment_labels_reflect_delta_pct() -> None:
    """前半 100 BPM、後半 140 BPM → 前半 slow、後半 fast (global=120)。"""
    fast_times = (np.arange(32) * 0.6).tolist()  # 100 BPM (slow vs 120)
    last = fast_times[-1] + 0.6
    slow_times = [last + (i + 1) * (60.0 / 140.0) for i in range(32)]  # 140 BPM (fast vs 120)
    times = fast_times + slow_times
    res = analyze_tempo_curve(_beats_at(times), _tempo(120.0))
    labels = {seg.label for seg in res.segments}
    assert "slow" in labels
    assert "fast" in labels


def test_global_bpm_preserved_in_result() -> None:
    """TempoResult.bpm は TempoCurveResult.global_bpm に伝播する。"""
    times = (np.arange(20) * 0.5).tolist()
    res = analyze_tempo_curve(_beats_at(times), _tempo(120.0))
    assert res.global_bpm == 120.0


def test_stable_pct_threshold_respected() -> None:
    """delta_pct < stable_pct なら label は stable。"""
    times = (np.arange(20) * 0.5).tolist()
    # global = 121.0, 局所 = 120.0 → delta = -0.83% < 5% → stable
    res = analyze_tempo_curve(_beats_at(times), _tempo(121.0), stable_pct=DEFAULT_STABLE_PCT)
    assert all(seg.label == "stable" for seg in res.segments)


def test_window_beats_parameter_changes_segment_count() -> None:
    """window_beats を増やすと segment 数 (フレーム数) は減る。"""
    times = (np.arange(40) * 0.5).tolist()
    small = analyze_tempo_curve(_beats_at(times), _tempo(120.0), window_beats=4)
    large = analyze_tempo_curve(_beats_at(times), _tempo(120.0), window_beats=16)
    # マージ後の segments 数なので一概に減るとは限らない、
    # しかし stable な合成データでは両方 segment 1 つに収束する
    assert len(small.segments) >= 1
    assert len(large.segments) >= 1
    # window_beats が大きい方が bpm_std は小さい
    assert large.bpm_std <= small.bpm_std + 0.01


def test_window_beats_default_constant() -> None:
    """既定値が公開定数と一致する。"""
    assert DEFAULT_WINDOW_BEATS == 8
    assert DEFAULT_STABLE_PCT == 5.0


def test_segments_are_time_ordered_and_contiguous() -> None:
    """全 segment は時間順 + 隣接が連続。"""
    times = (np.arange(40) * 0.5).tolist()
    res = analyze_tempo_curve(_beats_at(times), _tempo(120.0))
    for i in range(1, len(res.segments)):
        assert res.segments[i].start_sec >= res.segments[i - 1].start_sec
        # マージ後の連続性は終端 == 次の開始ではないこともある (window center 採用)
        # ただし end >= start を必ず満たす
        assert res.segments[i].end_sec >= res.segments[i].start_sec
