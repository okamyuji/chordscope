"""調性 (キー) 推定。

Krumhansl-Schmuckler アルゴリズム（実装は Temperley 1999 の改良プロファイル）。
クロマグラム平均と 24 個 (long+short × 12 root) のプロファイルとの相関から最尤キーを返す。
精度最優先方針: librosa の CQT クロマで平均し、ノイズ除去のため上位 80% 区間を集計。
"""

from __future__ import annotations

import librosa
import numpy as np

from chordscope.audio import AudioBuffer
from chordscope.models import KeyResult

# Temperley (1999) Kostka-Payne 改良プロファイル
KS_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
KS_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 旧 private 名のエイリアス (modulation.py が使うので削除しない)
_KS_MAJOR = KS_MAJOR
_KS_MINOR = KS_MINOR
_PITCH_NAMES = PITCH_NAMES

KEY_HOP_LENGTH = 2048


def compute_chroma(buffer: AudioBuffer, *, hop_length: int = KEY_HOP_LENGTH) -> np.ndarray:
    """CQT クロマを計算する。`avg_chroma` と `modulation.detect_modulation` で共用。"""
    return librosa.feature.chroma_cqt(
        y=buffer.samples, sr=buffer.sample_rate, hop_length=hop_length
    )


def avg_chroma_from_matrix(chroma: np.ndarray) -> np.ndarray:
    """事前計算済みクロマ行列から「強度の高い 80% フレーム」の平均ベクトルを返す。"""
    energy = chroma.sum(axis=0)
    if len(energy) > 4:
        threshold = np.quantile(energy, 0.2)
        mask = energy >= threshold
        chroma = chroma[:, mask]
    return chroma.mean(axis=1)


def _avg_chroma(buffer: AudioBuffer) -> np.ndarray:
    """CQT クロマ → 強度の高い 80% フレームを平均。"""
    return avg_chroma_from_matrix(compute_chroma(buffer))


def correlate_chroma(chroma_mean: np.ndarray) -> tuple[list[float], list[float]]:
    """各 root に major/minor プロファイルとの相関を計算 (public API)。"""
    return _correlations(chroma_mean)


def _correlations(chroma_mean: np.ndarray) -> tuple[list[float], list[float]]:
    """各 root に major/minor プロファイルとの相関を計算。"""
    maj_corrs: list[float] = []
    min_corrs: list[float] = []
    chroma_centered = chroma_mean - chroma_mean.mean()
    for shift in range(12):
        maj = np.roll(KS_MAJOR, shift)
        mn = np.roll(KS_MINOR, shift)
        maj_centered = maj - maj.mean()
        mn_centered = mn - mn.mean()
        maj_corr = float(
            np.dot(chroma_centered, maj_centered)
            / (np.linalg.norm(chroma_centered) * np.linalg.norm(maj_centered) + 1e-12)
        )
        min_corr = float(
            np.dot(chroma_centered, mn_centered)
            / (np.linalg.norm(chroma_centered) * np.linalg.norm(mn_centered) + 1e-12)
        )
        maj_corrs.append(maj_corr)
        min_corrs.append(min_corr)
    return maj_corrs, min_corrs


def estimate_key(buffer: AudioBuffer) -> KeyResult:
    chroma_mean = _avg_chroma(buffer)
    maj, mn = _correlations(chroma_mean)
    flat: list[tuple[float, str, str]] = []
    for i in range(12):
        flat.append((maj[i], PITCH_NAMES[i], "major"))
        flat.append((mn[i], PITCH_NAMES[i], "minor"))
    flat.sort(reverse=True, key=lambda t: t[0])
    best_corr, best_tonic, best_mode = flat[0]
    second_corr, second_tonic, second_mode = flat[1]
    # 信頼度 = 1位と2位の差を [0,1] にマップ。両正で接近していると曖昧。
    gap = max(0.0, best_corr - second_corr)
    confidence = float(min(1.0, gap * 5.0))
    return KeyResult(
        tonic=best_tonic,
        mode=best_mode,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        correlation=round(best_corr, 4),
        second_best=(second_tonic, second_mode),
    )
