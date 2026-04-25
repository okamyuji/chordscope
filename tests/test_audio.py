"""音声 I/O テスト。"""

from __future__ import annotations

import numpy as np
import pytest

from chordscope.audio import DEFAULT_SR, load_audio


@pytest.mark.integration
def test_load_pd_audio(pd_encina_path) -> None:  # type: ignore[no-untyped-def]
    buf = load_audio(pd_encina_path)
    assert buf.sample_rate == DEFAULT_SR
    assert buf.duration > 60.0  # 95 秒の音源
    assert buf.duration < 200.0
    assert buf.samples.dtype == np.float32
    assert buf.samples.ndim == 1


@pytest.mark.integration
def test_load_audio_native_sr(pd_joplin_sample_path) -> None:  # type: ignore[no-untyped-def]
    buf = load_audio(pd_joplin_sample_path, sr=None)
    assert buf.sample_rate > 0
    assert buf.duration > 100.0
