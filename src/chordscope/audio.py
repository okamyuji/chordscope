"""音声ファイル読み込みのラッパー。

librosa.load を中心に、サンプルレート 22050 Hz・モノラル化を共通化する。
mp3/m4a/aac/mp4 などはバックエンドに ffmpeg が必要。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

DEFAULT_SR = 22050  # librosa default; 多くの推定器の前提


@dataclass(frozen=True)
class AudioBuffer:
    """生波形と付随情報。"""

    samples: np.ndarray  # 1D float32, mono
    sample_rate: int
    duration: float
    path: Path


def load_audio(path: Path, *, sr: int | None = DEFAULT_SR, mono: bool = True) -> AudioBuffer:
    """音声を読み込み AudioBuffer で返す。

    - sr=None なら原音そのままのサンプルレート。
    - 入力が長すぎる場合でも全長読み込む（精度最優先方針）。
    """
    samples, used_sr = librosa.load(str(path), sr=sr, mono=mono)
    samples = np.asarray(samples, dtype=np.float32)
    duration = float(len(samples) / float(used_sr))
    return AudioBuffer(samples=samples, sample_rate=int(used_sr), duration=duration, path=path)
