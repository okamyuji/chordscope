"""局所 KS 解析による時系列転調検出。

`key.py` が曲全体の平均クロマに対し 24 個のプロファイルを相関させて
1 つの最尤キーを返すのに対し、本モジュールは window_sec 単位の
スライディングウィンドウで KS を再適用し、KeySegment 列を作る。
隣接する KeySegment の (tonic, mode) が変化した境界を `KeyChange` として
出力する。

単発のフラップ (1 窓だけ違うキー) は majority 抑制で除去する。
"""

from __future__ import annotations

import numpy as np

from chordscope.analyzers.key import (
    KEY_HOP_LENGTH,
    PITCH_NAMES,
    avg_chroma_from_matrix,
    compute_chroma,
    correlate_chroma,
)
from chordscope.audio import AudioBuffer
from chordscope.models import KeyChange, KeyResult, KeySegment, ModulationResult

DEFAULT_WINDOW_SEC = 16.0
DEFAULT_HOP_SEC = 4.0
DEFAULT_MIN_CONFIDENCE = 0.20
# 短すぎるセグメント (window_sec * 比率 未満) は前後にマージする
DEFAULT_MIN_SEGMENT_RATIO = 0.75
# smoothing パス数
DEFAULT_SMOOTHING_PASSES = 2


def _ks_for_window(chroma_window: np.ndarray) -> tuple[str, str, float, float]:
    """1 窓のクロマから最尤 (tonic, mode, confidence, correlation) を返す。"""
    chroma_mean = avg_chroma_from_matrix(chroma_window)
    maj, mn = correlate_chroma(chroma_mean)
    flat: list[tuple[float, str, str]] = []
    for i in range(12):
        flat.append((maj[i], PITCH_NAMES[i], "major"))
        flat.append((mn[i], PITCH_NAMES[i], "minor"))
    flat.sort(reverse=True, key=lambda t: t[0])
    best_corr, best_tonic, best_mode = flat[0]
    second_corr, _, _ = flat[1]
    gap = max(0.0, best_corr - second_corr)
    confidence = float(min(1.0, gap * 5.0))
    return best_tonic, best_mode, confidence, float(best_corr)


def _smooth_keys(
    raw: list[tuple[str, str, float, float]],
    *,
    passes: int = DEFAULT_SMOOTHING_PASSES,
) -> list[tuple[str, str, float, float]]:
    """1 窓のフラップを抑制する (multi-pass)。

    各 pass で raw[i] が前後 raw[i-1]/raw[i+1] と同じ (tonic, mode) で raw[i] だけ違うなら、
    raw[i] を前のキーで上書きする。passes 回繰り返すことで連続 2 窓のフラップにも対応。
    両端は touch しない。
    """
    if len(raw) < 3:
        return list(raw)
    out = list(raw)
    for _ in range(passes):
        new = list(out)
        for i in range(1, len(out) - 1):
            prev_t, prev_m, _, _ = out[i - 1]
            cur_t, cur_m, _, _ = out[i]
            nxt_t, nxt_m, _, _ = out[i + 1]
            if (prev_t, prev_m) == (nxt_t, nxt_m) and (prev_t, prev_m) != (cur_t, cur_m):
                _, _, conf_a, corr_a = out[i - 1]
                _, _, conf_b, corr_b = out[i + 1]
                new[i] = (
                    prev_t,
                    prev_m,
                    (conf_a + conf_b) / 2.0,
                    (corr_a + corr_b) / 2.0,
                )
        out = new
    return out


def _merge_short_segments(segments: list[KeySegment], min_duration_sec: float) -> list[KeySegment]:
    """duration < min_duration_sec のセグメントを前のセグメントに吸収する。

    前がない (先頭) なら次のセグメントに吸収する。
    """
    if not segments:
        return segments
    out: list[KeySegment] = []
    for seg in segments:
        dur = seg.end_sec - seg.start_sec
        if dur >= min_duration_sec:
            out.append(seg)
            continue
        if out:
            # 前のセグメントの end を伸ばすだけで、キーはそのまま
            prev = out[-1]
            out[-1] = KeySegment(
                start_sec=prev.start_sec,
                end_sec=seg.end_sec,
                tonic=prev.tonic,
                mode=prev.mode,  # type: ignore[arg-type]
                confidence=prev.confidence,
                correlation=prev.correlation,
            )
        else:
            # 先頭の短いセグメントは保持 (次の loop で次のセグメントが取り込む)
            out.append(seg)
    # もう一度: 先頭が短く、次があればマージ
    if len(out) >= 2 and (out[0].end_sec - out[0].start_sec) < min_duration_sec:
        nxt = out[1]
        out[0:2] = [
            KeySegment(
                start_sec=out[0].start_sec,
                end_sec=nxt.end_sec,
                tonic=nxt.tonic,
                mode=nxt.mode,  # type: ignore[arg-type]
                confidence=nxt.confidence,
                correlation=nxt.correlation,
            )
        ]
    return out


def _window_to_segments(
    window_keys: list[tuple[str, str, float, float]],
    window_starts_sec: list[float],
    window_ends_sec: list[float],
) -> list[KeySegment]:
    """連続する同じ (tonic, mode) を 1 つの KeySegment にマージする。"""
    if not window_keys:
        return []
    segments: list[KeySegment] = []
    cur_tonic, cur_mode, confs, corrs = (
        window_keys[0][0],
        window_keys[0][1],
        [window_keys[0][2]],
        [window_keys[0][3]],
    )
    cur_start = window_starts_sec[0]
    cur_end = window_ends_sec[0]
    for i in range(1, len(window_keys)):
        t, m, c, r = window_keys[i]
        if (t, m) == (cur_tonic, cur_mode):
            cur_end = window_ends_sec[i]
            confs.append(c)
            corrs.append(r)
        else:
            segments.append(
                KeySegment(
                    start_sec=round(cur_start, 3),
                    end_sec=round(cur_end, 3),
                    tonic=cur_tonic,
                    mode=cur_mode,  # type: ignore[arg-type]
                    confidence=round(float(np.mean(confs)), 3),
                    correlation=round(float(np.mean(corrs)), 4),
                )
            )
            cur_tonic, cur_mode = t, m
            cur_start = window_starts_sec[i]
            cur_end = window_ends_sec[i]
            confs, corrs = [c], [r]
    segments.append(
        KeySegment(
            start_sec=round(cur_start, 3),
            end_sec=round(cur_end, 3),
            tonic=cur_tonic,
            mode=cur_mode,  # type: ignore[arg-type]
            confidence=round(float(np.mean(confs)), 3),
            correlation=round(float(np.mean(corrs)), 4),
        )
    )
    return segments


def _semitone_index(tonic: str) -> int:
    return PITCH_NAMES.index(tonic)


def _normalize_interval(diff: int) -> int:
    """[-6, +6] に収める。"""
    diff = diff % 12
    if diff > 6:
        diff -= 12
    return diff


def classify_relation(from_tonic: str, from_mode: str, to_tonic: str, to_mode: str) -> str:
    """前後のキーから音楽理論的な関係を分類する。"""
    from_idx = _semitone_index(from_tonic)
    to_idx = _semitone_index(to_tonic)
    diff = (to_idx - from_idx) % 12  # 0..11
    if from_tonic == to_tonic and from_mode != to_mode:
        return "parallel"
    # +9 = -3 (Cmaj→Amin), +3 (Amin→Cmaj) は relative
    if from_mode != to_mode and (
        (from_mode == "major" and to_mode == "minor" and diff == 9)
        or (from_mode == "minor" and to_mode == "major" and diff == 3)
    ):
        return "relative"
    if from_mode == to_mode:
        if diff == 7:
            return "dominant"
        if diff == 5:
            return "subdominant"
        if diff in (1, 11):
            return "chromatic"
    return "other"


def _build_changes(segments: list[KeySegment]) -> list[KeyChange]:
    changes: list[KeyChange] = []
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        if (a.tonic, a.mode) == (b.tonic, b.mode):
            continue
        diff = _normalize_interval(_semitone_index(b.tonic) - _semitone_index(a.tonic))
        rel = classify_relation(a.tonic, a.mode, b.tonic, b.mode)
        changes.append(
            KeyChange(
                at_sec=round(b.start_sec, 3),
                from_tonic=a.tonic,
                from_mode=a.mode,  # type: ignore[arg-type]
                to_tonic=b.tonic,
                to_mode=b.mode,  # type: ignore[arg-type]
                interval_semitones=diff,
                relation=rel,  # type: ignore[arg-type]
            )
        )
    return changes


def detect_modulation(
    buffer: AudioBuffer,
    key: KeyResult,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    hop_length: int = KEY_HOP_LENGTH,
) -> ModulationResult:
    """音声バッファに対して時系列キー推定を行い、転調イベントを抽出する。

    入力: AudioBuffer と全曲 KeyResult (参考情報。フォールバック時に使う)
    """
    method = f"sliding-KS (window={window_sec}s, hop={hop_sec}s, min_conf={min_confidence})"
    sr = buffer.sample_rate
    duration = buffer.duration

    # 短すぎる音源 (window_sec の 1.5 倍未満) は単一セグメントを返す
    if duration < window_sec * 1.5:
        if duration <= 0:
            return ModulationResult(
                window_sec=window_sec, hop_sec=hop_sec, segments=[], changes=[], method=method
            )
        seg = KeySegment(
            start_sec=0.0,
            end_sec=round(duration, 3),
            tonic=key.tonic,
            mode=key.mode,
            confidence=key.confidence,
            correlation=key.correlation,
        )
        return ModulationResult(
            window_sec=window_sec, hop_sec=hop_sec, segments=[seg], changes=[], method=method
        )

    chroma = compute_chroma(buffer, hop_length=hop_length)
    frames_per_sec = sr / hop_length
    win_frames = int(window_sec * frames_per_sec)
    hop_frames = int(hop_sec * frames_per_sec)
    if win_frames <= 0 or hop_frames <= 0:
        return ModulationResult(
            window_sec=window_sec, hop_sec=hop_sec, segments=[], changes=[], method=method
        )

    n_frames = chroma.shape[1]
    raw: list[tuple[str, str, float, float]] = []
    starts: list[float] = []
    ends: list[float] = []
    for start in range(0, n_frames - win_frames + 1, hop_frames):
        end = start + win_frames
        sub = chroma[:, start:end]
        t, m, conf, corr = _ks_for_window(sub)
        raw.append((t, m, conf, corr))
        starts.append(start / frames_per_sec)
        ends.append(end / frames_per_sec)
    if not raw:
        return ModulationResult(
            window_sec=window_sec, hop_sec=hop_sec, segments=[], changes=[], method=method
        )

    # confidence しきい値: 弱い窓は前の窓のキーで埋める (フラップ抑制の前段)
    filled: list[tuple[str, str, float, float]] = []
    last_strong: tuple[str, str, float, float] | None = None
    for entry in raw:
        if entry[2] >= min_confidence:
            filled.append(entry)
            last_strong = entry
        elif last_strong is not None:
            filled.append((last_strong[0], last_strong[1], entry[2], entry[3]))
        else:
            filled.append(entry)

    smoothed = _smooth_keys(filled)
    segments = _window_to_segments(smoothed, starts, ends)
    segments = _merge_short_segments(segments, window_sec * DEFAULT_MIN_SEGMENT_RATIO)
    changes = _build_changes(segments)

    return ModulationResult(
        window_sec=window_sec,
        hop_sec=hop_sec,
        segments=segments,
        changes=changes,
        method=method,
    )
