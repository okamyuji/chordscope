"""テスト用の合成音声。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chordscope.audio import AudioBuffer


def click_track(bpm: float, duration: float, sr: int) -> AudioBuffer:
    """0.5 秒から bpm 間隔で 30 ms のクリックを並べる。4 拍ごとに高い音にする。"""
    samples = np.zeros(int(duration * sr), dtype=np.float32)
    click_len = int(0.03 * sr)
    t = np.arange(click_len) / sr
    for i, onset in enumerate(np.arange(0.5, duration - 0.1, 60.0 / bpm)):
        freq = 1500.0 if i % 4 == 0 else 1000.0
        click = np.sin(2 * np.pi * freq * t) * np.exp(-t * 120.0)
        start = int(onset * sr)
        samples[start : start + click_len] += click.astype(np.float32)[: len(samples) - start]
    return AudioBuffer(samples=samples, sample_rate=sr, duration=duration, path=Path("click.wav"))
