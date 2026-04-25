"""BPM 推定。

複数手法 (madmom DBNBeat, librosa.beat.beat_track, librosa onset autocorr) を実行し、
中央値を採用。各候補との一致度を信頼度に換算する (精度最優先方針)。
"""

from __future__ import annotations

from statistics import median

import librosa
import numpy as np
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor

from chordscope.audio import AudioBuffer
from chordscope.models import TempoResult

# 同一とみなす BPM 差 (整数刻み・倍テンポ補正を考慮)
_BPM_TOLERANCE = 2.0


def _agreement(values: list[float], target: float, tol: float = _BPM_TOLERANCE) -> float:
    if not values:
        return 0.0
    matches = sum(1 for v in values if _bpm_close(v, target, tol))
    return matches / len(values)


def _bpm_close(a: float, b: float, tol: float) -> bool:
    """半倍/倍テンポも近いとみなす。"""
    return (
        abs(a - b) <= tol
        or abs(a * 2 - b) <= tol
        or abs(a / 2 - b) <= tol
        or abs(a * 1.5 - b) <= tol  # 3/2 ずれ (3拍子と4拍子の混同)
    )


def estimate_bpm_madmom(buffer: AudioBuffer) -> float:
    """madmom RNN ビート活性 + テンポ推定 (DBN/comb)。"""
    proc = RNNBeatProcessor()  # 学習済みネット
    activations = proc(buffer.samples)
    tempo_proc = TempoEstimationProcessor(fps=100)
    candidates = tempo_proc(activations)
    if candidates is None or len(candidates) == 0:
        msg = "madmom failed to estimate tempo"
        raise RuntimeError(msg)
    # candidates は (bpm, strength) 配列。最強候補を採用
    return float(candidates[0][0])


def estimate_bpm_librosa(buffer: AudioBuffer) -> float:
    """librosa beat_track。返却値は scalar (>=0.10)。"""
    tempo, _ = librosa.beat.beat_track(y=buffer.samples, sr=buffer.sample_rate, units="time")
    return float(np.atleast_1d(tempo)[0])


def estimate_bpm_onset_autocorr(buffer: AudioBuffer) -> float:
    """オンセット強度の自己相関ベースの BPM (librosa.feature.tempo)。"""
    onset_env = librosa.onset.onset_strength(y=buffer.samples, sr=buffer.sample_rate)
    bpm = librosa.feature.tempo(onset_envelope=onset_env, sr=buffer.sample_rate)
    return float(np.atleast_1d(bpm)[0])


_CANONICAL_LOW = 60.0
_CANONICAL_HIGH = 180.0


def _canonicalize(bpm: float) -> float:
    """BPM を倍/半分系で [60, 180) BPM レンジへ畳み込む (オクターブ正規化)。

    BPM 推定器は 16 分音符・8 分音符レートを拾って 2x/4x になることが多い。
    """
    if bpm <= 0:
        return bpm
    while bpm < _CANONICAL_LOW:
        bpm *= 2.0
    while bpm >= _CANONICAL_HIGH * 2:
        bpm /= 2.0
    # _CANONICAL_HIGH (180) より大きくても 2x すると 360 を超えるので、半分にした方が妥当な場合のみ畳む
    if bpm >= _CANONICAL_HIGH and bpm / 2.0 >= _CANONICAL_LOW:
        bpm /= 2.0
    return bpm


def estimate_tempo(buffer: AudioBuffer) -> TempoResult:
    """複数手法の BPM を取得しオクターブ正規化後に中央値を採用する。"""
    raw_candidates: list[float] = []
    methods_used: list[str] = []
    for name, fn in [
        ("madmom-dbn", estimate_bpm_madmom),
        ("librosa-beat-track", estimate_bpm_librosa),
        ("librosa-onset-autocorr", estimate_bpm_onset_autocorr),
    ]:
        try:
            raw_candidates.append(fn(buffer))
            methods_used.append(name)
        except Exception:  # 1 手法失敗しても他で集約
            continue
    if not raw_candidates:
        msg = "All BPM estimators failed"
        raise RuntimeError(msg)
    canonical = [_canonicalize(c) for c in raw_candidates]
    chosen = float(median(canonical))
    confidence = _agreement(canonical, chosen)
    return TempoResult(
        bpm=round(chosen, 2),
        bpm_candidates=[round(c, 2) for c in raw_candidates],
        confidence=round(confidence, 3),
        method=" + ".join(methods_used),
    )
