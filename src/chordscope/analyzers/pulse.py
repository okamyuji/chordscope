"""拍の確かさの判定。

追跡器 (madmom) は拍のない曲にも拍を返す。その拍から出したテンポ変動やシンコペーションは
意味を持たないので、次の 2 系統の根拠で「拍が不確か」を判定し、理由とともに残す。

- 追跡器の自己一致: 拍間隔が中央値の近くにそろっているか
- 音そのものの周期: 低音・中音・高音帯のスペクトルフラックスに、拍の周期が現れるか、
  その周期が追跡器の拍と 2 の冪倍で一致するか

しきい値は 26 曲 (ファンク、ミニマル、映画音楽、ルネサンス声楽、ラグタイム、行進曲、
ロック) で拍の有無を分けた値。中音帯の 0.1 は、拍のない側の最大 0.073 と拍のある側の
最小 0.119 の間にある。
"""

from __future__ import annotations

import librosa
import numpy as np

from chordscope.audio import AudioBuffer
from chordscope.models import BandPulse, BeatResult, PulseReason, PulseResult

GRID_TOLERANCE = 0.15
MIN_GRID_RATIO = 0.9
MIN_MID_CLARITY = 0.1
MIN_INTERVALS = 4
_OCTAVE_TOLERANCE = np.log2(1.08)
_BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", 20.0, 200.0),
    ("mid", 200.0, 2000.0),
    ("high", 2000.0, 8000.0),
)
_N_FFT = 2048
_HOP = 512
_MIN_LAG_SEC, _MAX_LAG_SEC = 0.25, 2.0  # 30〜240 BPM


def _grid_stats(beat_times: np.ndarray) -> tuple[float, float]:
    """拍間隔が中央値 ±15% に入る割合と、その間隔だけの変動係数。

    外れた間隔 (取りこぼしや重複) を CV に入れると、テンポの揺れと追跡の失敗が混ざる。
    """
    intervals = np.diff(beat_times)
    on_grid = np.abs(intervals / np.median(intervals) - 1.0) < GRID_TOLERANCE
    grid = intervals[on_grid]
    cv = float(grid.std() / grid.mean()) if len(grid) > 1 else 0.0
    return float(on_grid.mean()), cv


def _band_flux(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    band = np.log1p(mag[(freqs >= lo) & (freqs < hi)].sum(axis=0))
    return np.maximum(np.diff(band, prepend=band[0]), 0.0)


def _periodicity(flux: np.ndarray, frame_sec: float) -> tuple[float, float | None]:
    """拍周期の範囲での自己相関の最大値と、そのラグの秒数。"""
    x = flux - flux.mean()
    spec = np.fft.rfft(x, n=2 * len(x))
    ac = np.fft.irfft(spec * np.conj(spec))[: len(x)]
    lo, hi = int(_MIN_LAG_SEC / frame_sec), int(_MAX_LAG_SEC / frame_sec)
    if ac[0] <= 0 or hi >= len(ac):
        return 0.0, None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(ac[lag] / ac[0]), lag * frame_sec


def _matches(period: float | None, beat_period: float) -> bool:
    """周期が拍と 2 の冪倍 (±8%) で一致するか。追跡器は拍の倍や半分を拾うので、オクターブ違いは一致とみなす。"""
    if period is None:
        return False
    r = float(np.log2(period / beat_period))
    return abs(r - round(r)) < _OCTAVE_TOLERANCE


def _band_pulses(buffer: AudioBuffer, beat_period: float) -> list[BandPulse]:
    """低音・中音・高音帯ごとの拍周期の明瞭さ、その BPM、追跡した拍との一致。

    STFT は曲全体を一度に作る。1 時間の曲で最大メモリが約 0.9 GB 増える。
    """
    mag = np.abs(librosa.stft(buffer.samples, n_fft=_N_FFT, hop_length=_HOP))
    freqs = librosa.fft_frequencies(sr=buffer.sample_rate, n_fft=_N_FFT)
    frame_sec = _HOP / buffer.sample_rate
    out = []
    for name, lo, hi in _BANDS:
        clarity, period = _periodicity(_band_flux(mag, freqs, lo, hi), frame_sec)
        out.append(
            BandPulse(
                band=name,  # type: ignore[arg-type]
                clarity=round(clarity, 3),
                bpm=round(60.0 / period, 1) if period else None,
                matches_beat=_matches(period, beat_period),
            )
        )
    return out


def assess_pulse(buffer: AudioBuffer, beats: BeatResult) -> PulseResult:
    """拍の確かさを判定する。計算は止めず、uncertain と理由を返すだけにする。"""
    method = (
        f"grid±{GRID_TOLERANCE:.0%}>={MIN_GRID_RATIO} / "
        f"band-flux autocorr (mid>={MIN_MID_CLARITY}, octave match)"
    )
    beat_times = np.asarray(beats.beat_times, dtype=float)
    if len(beat_times) - 1 < MIN_INTERVALS:
        return PulseResult(
            grid_ratio=0.0, ibi_cv=0.0, uncertain=True, reasons=["too_few_beats"], method=method
        )
    grid_ratio, ibi_cv = _grid_stats(beat_times)
    bands = _band_pulses(buffer, float(np.median(np.diff(beat_times))))
    mid = next(b for b in bands if b.band == "mid")
    checks: list[tuple[PulseReason, bool]] = [
        ("low_grid_ratio", grid_ratio < MIN_GRID_RATIO),
        ("weak_mid_band_pulse", mid.clarity < MIN_MID_CLARITY),
        ("no_band_matches_beat", not any(b.matches_beat for b in bands)),
    ]
    reasons = [reason for reason, hit in checks if hit]
    return PulseResult(
        grid_ratio=round(grid_ratio, 3),
        ibi_cv=round(ibi_cv, 4),
        bands=bands,
        uncertain=bool(reasons),
        reasons=reasons,
        method=method,
    )
