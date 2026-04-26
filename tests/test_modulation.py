"""modulation モジュールの純関数テスト。

実音源を使うテストは tests/test_modulation_ab.py 側に分離する。
ここでは `classify_relation`, `_smooth_keys`, `_window_to_segments`,
`_normalize_interval`, `_build_changes` といった純関数を検証する。
"""

from __future__ import annotations

from chordscope.analyzers.modulation import (
    _build_changes,
    _normalize_interval,
    _smooth_keys,
    _window_to_segments,
    classify_relation,
)
from chordscope.models import KeySegment


def test_classify_relation_dominant() -> None:
    """C major → G major は完全 5 度上 = dominant。"""
    assert classify_relation("C", "major", "G", "major") == "dominant"


def test_classify_relation_subdominant() -> None:
    """C major → F major は完全 5 度下 (= 4 度上) = subdominant。"""
    assert classify_relation("C", "major", "F", "major") == "subdominant"


def test_classify_relation_relative_major_to_minor() -> None:
    """C major → A minor は relative。"""
    assert classify_relation("C", "major", "A", "minor") == "relative"


def test_classify_relation_relative_minor_to_major() -> None:
    """A minor → C major は relative。"""
    assert classify_relation("A", "minor", "C", "major") == "relative"


def test_classify_relation_parallel() -> None:
    """C major → C minor は parallel。"""
    assert classify_relation("C", "major", "C", "minor") == "parallel"


def test_classify_relation_chromatic() -> None:
    """C major → C# major は半音 = chromatic。"""
    assert classify_relation("C", "major", "C#", "major") == "chromatic"


def test_classify_relation_other_for_distant_keys() -> None:
    """C major → D major (長 2 度上) は other。"""
    assert classify_relation("C", "major", "D", "major") == "other"


def test_normalize_interval_collapses_to_minus6_plus6() -> None:
    """[-6, +6] に収まる。"""
    assert _normalize_interval(7) == -5  # 5 度上 → 4 度下
    assert _normalize_interval(-7) == 5
    assert _normalize_interval(0) == 0
    assert _normalize_interval(6) == 6
    assert _normalize_interval(-6) == 6  # 6 と -6 は等価
    assert _normalize_interval(12) == 0


def test_smooth_keys_removes_single_flap() -> None:
    """前後同じキーで 1 窓だけ違うフラップは削除。"""
    raw = [
        ("C", "major", 0.5, 0.5),
        ("C", "major", 0.5, 0.5),
        ("F", "major", 0.4, 0.4),  # フラップ
        ("C", "major", 0.5, 0.5),
        ("C", "major", 0.5, 0.5),
    ]
    smoothed = _smooth_keys(raw)
    assert all(k[0] == "C" and k[1] == "major" for k in smoothed)


def test_smooth_keys_keeps_real_change() -> None:
    """連続する変化は残る (フラップではない)。"""
    raw = [
        ("C", "major", 0.5, 0.5),
        ("C", "major", 0.5, 0.5),
        ("G", "major", 0.5, 0.5),
        ("G", "major", 0.5, 0.5),
        ("G", "major", 0.5, 0.5),
    ]
    smoothed = _smooth_keys(raw)
    keys = [(t, m) for t, m, _, _ in smoothed]
    # 変化点の前後でキーが入れ替わっていることを確認
    assert ("C", "major") in keys
    assert ("G", "major") in keys


def test_smooth_keys_short_input_passthrough() -> None:
    """3 件未満ならそのまま返す。"""
    raw = [("C", "major", 0.5, 0.5), ("G", "major", 0.5, 0.5)]
    assert _smooth_keys(raw) == raw


def test_window_to_segments_merges_consecutive() -> None:
    """連続する同じキー窓は 1 つの KeySegment にマージされる。"""
    keys = [
        ("C", "major", 0.6, 0.5),
        ("C", "major", 0.7, 0.55),
        ("G", "major", 0.5, 0.45),
    ]
    starts = [0.0, 4.0, 8.0]
    ends = [16.0, 20.0, 24.0]
    segments = _window_to_segments(keys, starts, ends)
    assert len(segments) == 2
    assert segments[0].tonic == "C"
    assert segments[0].mode == "major"
    assert segments[0].end_sec == 20.0  # マージ後の終端
    assert segments[1].tonic == "G"


def test_window_to_segments_empty_input() -> None:
    assert _window_to_segments([], [], []) == []


def test_build_changes_emits_events_at_boundaries() -> None:
    """隣接 KeySegment の (tonic, mode) が変わった点は KeyChange になる。"""
    segments = [
        KeySegment(
            start_sec=0.0, end_sec=20.0, tonic="C", mode="major", confidence=0.7, correlation=0.5
        ),
        KeySegment(
            start_sec=20.0, end_sec=40.0, tonic="G", mode="major", confidence=0.6, correlation=0.4
        ),
        KeySegment(
            start_sec=40.0, end_sec=60.0, tonic="A", mode="minor", confidence=0.65, correlation=0.45
        ),
    ]
    changes = _build_changes(segments)
    assert len(changes) == 2
    assert changes[0].at_sec == 20.0
    assert changes[0].from_tonic == "C"
    assert changes[0].to_tonic == "G"
    assert changes[0].relation == "dominant"
    assert changes[1].at_sec == 40.0
    assert changes[1].from_tonic == "G"
    assert changes[1].to_tonic == "A"


def test_build_changes_no_event_when_keys_identical() -> None:
    """同じキーが 2 セグメント続いていても change は発火しない。"""
    segments = [
        KeySegment(
            start_sec=0.0, end_sec=10.0, tonic="C", mode="major", confidence=0.7, correlation=0.5
        ),
        KeySegment(
            start_sec=10.0, end_sec=20.0, tonic="C", mode="major", confidence=0.6, correlation=0.4
        ),
    ]
    assert _build_changes(segments) == []
