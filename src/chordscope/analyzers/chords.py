"""コード認識。

主バックエンド: madmom DeepChromaProcessor + DeepChromaChordRecognitionProcessor。
これは Korzeniowski/Widmer の DNN/CRF ベースで、MIREX で長年高い精度を示している。

副バックエンド: librosa CQT クロマ + 拡張テンプレート + Viterbi 平滑化。
madmom が壊れた場合 (例: モデルファイル欠損) のフェイルセーフ。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import librosa
import numpy as np
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

from chordscope.audio import AudioBuffer
from chordscope.models import ChordResult, ChordSegment

ChordEngine = Literal["madmom", "librosa-template"]
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _unique_chords(segments: Iterable[ChordSegment]) -> list[str]:
    seen: list[str] = []
    for s in segments:
        if s.label not in seen:
            seen.append(s.label)
    return seen


def recognize_chords_madmom(buffer: AudioBuffer) -> ChordResult:
    """madmom Deep Chroma + DCC で和音区間を認識する。

    madmom は内部的に専用サンプルレート (44100Hz) で扱うため、
    入力ファイルパスを直接使う。バッファ経由では精度が落ちる。
    """
    chroma_proc = DeepChromaProcessor()
    rec_proc = DeepChromaChordRecognitionProcessor()
    chroma = chroma_proc(str(buffer.path))
    raw = rec_proc(chroma)
    # raw は recarray: fields ('start', 'end', 'label')
    segments: list[ChordSegment] = []
    for row in raw:
        start = float(row[0])
        end = float(row[1])
        label = str(row[2])
        if end <= start:
            continue
        segments.append(ChordSegment(start=start, end=end, label=label))
    return ChordResult(
        segments=segments,
        method="madmom-DeepChroma+DCC",
        unique_chords=_unique_chords(segments),
    )


# ----- librosa template fallback -----

# テンプレート定義: (suffix, intervals)
_CHORD_TEMPLATES: list[tuple[str, list[int]]] = [
    ("maj", [0, 4, 7]),
    ("min", [0, 3, 7]),
    ("7", [0, 4, 7, 10]),
    ("maj7", [0, 4, 7, 11]),
    ("min7", [0, 3, 7, 10]),
    ("dim", [0, 3, 6]),
    ("aug", [0, 4, 8]),
    ("sus2", [0, 2, 7]),
    ("sus4", [0, 5, 7]),
    ("m7b5", [0, 3, 6, 10]),
]


def _build_templates() -> tuple[np.ndarray, list[str]]:
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for root in range(12):
        for suffix, intervals in _CHORD_TEMPLATES:
            v = np.zeros(12, dtype=np.float32)
            for interval in intervals:
                v[(root + interval) % 12] = 1.0
            v /= np.linalg.norm(v) + 1e-12
            rows.append(v)
            labels.append(f"{_PITCH_NAMES[root]}:{suffix}")
    # No-chord (silence) row
    rows.append(np.full(12, 1.0 / np.sqrt(12), dtype=np.float32))
    labels.append("N")
    return np.stack(rows), labels


def _viterbi_smooth(scores: np.ndarray, self_loop: float = 0.85) -> np.ndarray:
    """シンプルなビタビ平滑。

    scores: shape (T, K), 各時間ステップでのテンプレートとの類似度 (大=良)。
    遷移行列: 自己ループ self_loop, それ以外は (1-self_loop)/(K-1)。
    """
    t, k = scores.shape
    if t == 0:
        return np.empty(0, dtype=int)
    log_scores = np.log(np.clip(scores, 1e-9, None))
    log_self = np.log(self_loop)
    log_other = np.log((1.0 - self_loop) / (k - 1))
    dp = np.full((t, k), -np.inf)
    back = np.zeros((t, k), dtype=np.int32)
    dp[0] = log_scores[0]
    for i in range(1, t):
        # ベクトル化: 各 next k に対して、prev からの最良
        prev = dp[i - 1]
        # 自己ループ vs 他遷移の最良を比較
        best_other = prev.max() + log_other
        best_other_idx = int(prev.argmax())
        for j in range(k):
            self_score = prev[j] + log_self
            if self_score >= best_other:
                dp[i, j] = self_score + log_scores[i, j]
                back[i, j] = j
            else:
                dp[i, j] = best_other + log_scores[i, j]
                back[i, j] = best_other_idx
    path = np.zeros(t, dtype=int)
    path[-1] = int(dp[-1].argmax())
    for i in range(t - 2, -1, -1):
        path[i] = back[i + 1, path[i + 1]]
    return path


def recognize_chords_librosa(buffer: AudioBuffer, *, hop_length: int = 4096) -> ChordResult:
    """librosa CQT クロマ + 拡張テンプレートマッチ + Viterbi 平滑。"""
    chroma = librosa.feature.chroma_cqt(
        y=buffer.samples, sr=buffer.sample_rate, hop_length=hop_length
    )
    # 各フレームを正規化
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-12
    chroma_n = chroma / norms
    templates, labels = _build_templates()
    scores = (templates @ chroma_n).T  # shape (T, K)
    scores = (scores + 1.0) / 2.0  # cosine [-1,1] -> [0,1]
    path = _viterbi_smooth(scores)
    times = librosa.frames_to_time(
        np.arange(scores.shape[0]), sr=buffer.sample_rate, hop_length=hop_length
    )
    segments: list[ChordSegment] = []
    if len(path) == 0:
        return ChordResult(segments=[], method="librosa-template+viterbi", unique_chords=[])
    cur_label = labels[path[0]]
    cur_start = float(times[0])
    for i in range(1, len(path)):
        new_label = labels[path[i]]
        if new_label != cur_label:
            segments.append(ChordSegment(start=cur_start, end=float(times[i]), label=cur_label))
            cur_label = new_label
            cur_start = float(times[i])
    end_time = float(buffer.duration)
    segments.append(ChordSegment(start=cur_start, end=end_time, label=cur_label))
    return ChordResult(
        segments=segments,
        method="librosa-template+viterbi",
        unique_chords=_unique_chords(segments),
    )


def recognize_chords(buffer: AudioBuffer, engine: ChordEngine = "madmom") -> ChordResult:
    """フロントドア。エンジン指定で振り分け。"""
    if engine == "madmom":
        return recognize_chords_madmom(buffer)
    if engine == "librosa-template":
        return recognize_chords_librosa(buffer)
    msg = f"Unknown chord engine: {engine}"
    raise ValueError(msg)
