"""ビート・ダウンビート・拍子推定。

madmom の RNNDownBeatProcessor + DBNDownBeatTrackingProcessor を主軸に、
2/3/4/5/6/7 拍子候補をそれぞれ独立に走らせて拍頭分布から最尤を選ぶ。

madmom 内部実装は単一 DBN で複数 beats_per_bar 候補を扱うパスが numpy 2.x の
inhomogeneous shape エラーで動かないため、候補ごとに分けて呼ぶ。
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

from chordscope.audio import AudioBuffer
from chordscope.models import BeatResult, MeterResult

_BEATS_PER_BAR_CANDIDATES = [2, 3, 4, 5, 6, 7]


def _run_dbn_single(activations: np.ndarray, beats_per_bar: int) -> np.ndarray | None:
    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar], fps=100)
    try:
        return dbn(activations)
    except Exception:
        return None


def _score_beats(beats: np.ndarray) -> float:
    """ダウンビート位置の周期一貫性で簡易スコア。

    各ダウンビート間隔のばらつき (CV: 標準偏差/平均) が小さいほど高得点。
    """
    if beats is None or len(beats) < 2:
        return 0.0
    downbeats = beats[beats[:, 1] == 1, 0]
    if len(downbeats) < 3:
        return 0.0
    diffs = np.diff(downbeats)
    if diffs.mean() <= 0:
        return 0.0
    cv = float(diffs.std() / diffs.mean())
    return float(1.0 / (1.0 + cv))


_PARSIMONY_TOLERANCE = 0.92  # 上位スコアの 92% 以上なら、より小さい bpb を優先


def estimate_beats_and_meter(buffer: AudioBuffer) -> tuple[BeatResult, MeterResult]:
    """ビート・ダウンビート・拍子分子を独立 DBN 群で推定する。

    倍数的に互換な拍子 (例: 6/4 と 2/4 と 3/4) では、Occam's razor で
    最も小さい bpb を優先する (有意なスコア差がない場合)。
    """
    rnn = RNNDownBeatProcessor()
    activations = rnn(buffer.samples)
    scored: list[tuple[float, int, np.ndarray]] = []
    for bpb in _BEATS_PER_BAR_CANDIDATES:
        result = _run_dbn_single(activations, bpb)
        if result is None or len(result) == 0:
            continue
        score = _score_beats(result)
        scored.append((score, bpb, result))
    if not scored:
        msg = "Downbeat tracking failed for all meter candidates"
        raise RuntimeError(msg)
    max_score = max(s for s, _, _ in scored)
    threshold = max_score * _PARSIMONY_TOLERANCE
    eligible = [(s, b, r) for s, b, r in scored if s >= threshold]
    eligible.sort(key=lambda t: t[1])  # 小さい bpb 優先
    _score, _winning_bpb, beats = eligible[0]
    # beats は shape (N, 2): [time, beat_position(1=downbeat,2..,N=最終拍)]
    beat_times: list[float] = [float(t) for t in beats[:, 0]]
    positions = beats[:, 1].astype(int)
    downbeat_times: list[float] = [
        float(beats[i, 0]) for i in range(len(beats)) if int(beats[i, 1]) == 1
    ]
    # 拍子分子 = 同一小節内の最大 beat 位置
    if len(downbeat_times) >= 2:
        # 最後のダウンビート以降を除外し、各小節の拍数を集計
        bar_lengths: list[int] = []
        last_db = None
        count = 0
        for pos in positions:
            if pos == 1 and last_db is not None:
                bar_lengths.append(count)
                count = 1
            else:
                count += 1
            last_db = pos
        if bar_lengths:
            counter = Counter(bar_lengths)
            numerator, freq = counter.most_common(1)[0]
            confidence = freq / sum(counter.values())
        else:
            numerator = int(np.max(positions))
            confidence = 0.5
    else:
        numerator = int(np.max(positions))
        confidence = 0.5
    if numerator not in _BEATS_PER_BAR_CANDIDATES:
        # 想定外なら最も近い候補に丸める
        numerator = min(_BEATS_PER_BAR_CANDIDATES, key=lambda c: abs(c - numerator))
    beat_result = BeatResult(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        beats_per_bar=numerator,
        method="madmom-RNN+DBN-DownBeat",
    )
    meter_result = MeterResult(
        numerator=numerator,
        denominator=4,
        confidence=round(float(confidence), 3),
    )
    return beat_result, meter_result
