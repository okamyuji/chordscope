"""ビート時刻列から局所 BPM 時系列を構築。

`BeatResult.beat_times[]` (madmom RNN+DBN downbeat tracking で得たビート位置) を
window_beats 単位の中央値で平滑化し、局所 BPM 系列を組み立てる。

global BPM (= TempoResult.bpm) との相対差で各区間に "slow" / "stable" / "fast" を
割り当て、隣接同ラベルをマージして TempoSegment 列にする。

trend (accelerando / ritardando / variable / stable) は局所 BPM 列に対する
1 次回帰係数 + 標準偏差から判定する。
"""

from __future__ import annotations

import numpy as np

from chordscope.analyzers.tempo import _canonicalize
from chordscope.models import BeatResult, TempoCurveResult, TempoResult, TempoSegment

DEFAULT_WINDOW_BEATS = 8
DEFAULT_STABLE_PCT = 5.0
# trend 判定: 標準偏差がこの BPM 未満なら全体的に "stable"
_STABLE_STD_BPM = 2.0
# trend 判定: 1 次回帰係数 (BPM/sec) の閾値。曲全長で +/-0.5 BPM/sec * dur=240s = +/-120 BPM 変化、
# 実際は曲全長で ±2 BPM 以上の傾きがあれば accelerando/ritardando と扱う。
_TREND_SLOPE_THRESHOLD_BPM_PER_SEC = 0.02


def _local_bpms(beat_times: np.ndarray, window_beats: int) -> tuple[np.ndarray, np.ndarray]:
    """各 window の中央値間隔から局所 BPM とその開始時刻 (window 中央) を返す。

    返り値: (window_centers_sec, local_bpms)
    """
    intervals = np.diff(beat_times)
    if len(intervals) < window_beats:
        return np.empty(0), np.empty(0)
    n_windows = len(intervals) - window_beats + 1
    centers = np.empty(n_windows)
    bpms = np.empty(n_windows)
    for i in range(n_windows):
        med = float(np.median(intervals[i : i + window_beats]))
        if med <= 0:
            bpms[i] = 0.0
        else:
            bpms[i] = _canonicalize(60.0 / med)
        # window 中央の時刻
        centers[i] = float((beat_times[i] + beat_times[i + window_beats]) / 2.0)
    return centers, bpms


def _label_for_delta(delta_pct: float, stable_pct: float) -> str:
    if abs(delta_pct) < stable_pct:
        return "stable"
    return "fast" if delta_pct > 0 else "slow"


def _classify_trend(centers: np.ndarray, bpms: np.ndarray) -> str:
    """局所 BPM 列から全体の傾向を分類する。"""
    if len(bpms) < 2:
        return "stable"
    std = float(bpms.std())
    if std < _STABLE_STD_BPM:
        return "stable"
    # 1 次回帰
    slope, _ = np.polyfit(centers, bpms, 1)
    if slope > _TREND_SLOPE_THRESHOLD_BPM_PER_SEC:
        return "accelerando"
    if slope < -_TREND_SLOPE_THRESHOLD_BPM_PER_SEC:
        return "ritardando"
    return "variable"


def _merge_segments(
    centers: np.ndarray,
    bpms: np.ndarray,
    beat_times: np.ndarray,
    window_beats: int,
    global_bpm: float,
    stable_pct: float,
) -> list[TempoSegment]:
    """同一ラベルの連続する局所 BPM をマージして TempoSegment 列にする。"""
    if len(bpms) == 0:
        return []
    raw: list[tuple[float, float, float, str]] = []
    for i, (_c, b) in enumerate(zip(centers, bpms, strict=False)):
        s = float(beat_times[i])
        e = float(beat_times[i + window_beats])
        delta = (b - global_bpm) / global_bpm * 100.0 if global_bpm > 0 else 0.0
        label = _label_for_delta(delta, stable_pct)
        raw.append((s, e, float(b), label))

    merged: list[TempoSegment] = []
    cur_s, cur_e, cur_bpms, cur_label = raw[0][0], raw[0][1], [raw[0][2]], raw[0][3]
    for s, e, b, label in raw[1:]:
        if label == cur_label:
            cur_e = e
            cur_bpms.append(b)
        else:
            mean_bpm = float(np.mean(cur_bpms))
            delta = (mean_bpm - global_bpm) / global_bpm * 100.0 if global_bpm > 0 else 0.0
            merged.append(
                TempoSegment(
                    start_sec=round(cur_s, 3),
                    end_sec=round(cur_e, 3),
                    local_bpm=round(mean_bpm, 2),
                    delta_pct=round(delta, 2),
                    label=cur_label,  # type: ignore[arg-type]
                )
            )
            cur_s, cur_e, cur_bpms, cur_label = s, e, [b], label
    # 末尾
    mean_bpm = float(np.mean(cur_bpms))
    delta = (mean_bpm - global_bpm) / global_bpm * 100.0 if global_bpm > 0 else 0.0
    merged.append(
        TempoSegment(
            start_sec=round(cur_s, 3),
            end_sec=round(cur_e, 3),
            local_bpm=round(mean_bpm, 2),
            delta_pct=round(delta, 2),
            label=cur_label,  # type: ignore[arg-type]
        )
    )
    return merged


def analyze_tempo_curve(
    beats: BeatResult,
    tempo: TempoResult,
    *,
    window_beats: int = DEFAULT_WINDOW_BEATS,
    stable_pct: float = DEFAULT_STABLE_PCT,
) -> TempoCurveResult:
    """beat_times[] を平滑化して局所 BPM 系列を組み立てる。

    ビート数が window_beats 以下しかない場合は segments=[], trend="stable" を返す。
    """
    beat_times = np.asarray(beats.beat_times, dtype=float)
    global_bpm = float(tempo.bpm)
    method = f"beat-interval-median (window={window_beats}, stable_pct={stable_pct})"

    centers, bpms = _local_bpms(beat_times, window_beats)
    if len(bpms) == 0:
        return TempoCurveResult(
            window_beats=window_beats,
            global_bpm=round(global_bpm, 2),
            bpm_min=round(global_bpm, 2),
            bpm_max=round(global_bpm, 2),
            bpm_std=0.0,
            segments=[],
            trend="stable",
            method=method,
        )

    trend = _classify_trend(centers, bpms)
    segments = _merge_segments(
        centers,
        bpms,
        beat_times,
        window_beats,
        global_bpm,
        stable_pct,
    )

    return TempoCurveResult(
        window_beats=window_beats,
        global_bpm=round(global_bpm, 2),
        bpm_min=round(float(bpms.min()), 2),
        bpm_max=round(float(bpms.max()), 2),
        bpm_std=round(float(bpms.std()), 3),
        segments=segments,
        trend=trend,  # type: ignore[arg-type]
        method=method,
    )
