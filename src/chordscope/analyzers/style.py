"""ジャンル別の観点分析。

各ジャンル特有の特徴を別軸で計測する。ジャンル分類モデルとは独立 (モデル非依存)。

- Jazz: 7th 系コード比率、ii-V-I 進行頻度、テンポ範囲、シンコペーション度
- Classic: 機能和声整合度、調性変化数、テンポ変動度、coda 検出は割愛
- J-Pop: コード進行の循環度 (王道進行 IV-V-iii-vi 等)、サビっぽい盛り上がり区間
- Rock: パワーコード/トライアド比率、4拍ロック度 (4/4 + 強拍均等)、テンポ
"""

from __future__ import annotations

from collections import Counter

import librosa
import numpy as np

from chordscope.audio import AudioBuffer
from chordscope.models import (
    BeatResult,
    ChordResult,
    HarmonicAnalysis,
    KeyResult,
    MeterResult,
    StyleNotes,
    TempoResult,
)


def _chord_quality_distribution(chords: ChordResult) -> Counter[str]:
    counter: Counter[str] = Counter()
    for seg in chords.segments:
        if seg.label in ("N", "X", ""):
            continue
        parts = seg.label.split("/")[0].split(":")
        qual = parts[1] if len(parts) > 1 else "maj"
        counter[qual] += 1
    return counter


def _ratio(counter: Counter[str], targets: set[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(counter.get(t, 0) for t in targets) / total


def _ii_v_i_count(romans: list[str]) -> int:
    count = 0
    for i in range(len(romans) - 2):
        a = romans[i].rstrip("0123456789")
        b = romans[i + 1].rstrip("0123456789")
        c = romans[i + 2].rstrip("0123456789")
        if a in ("ii", "II") and b == "V" and c in ("I", "i"):
            count += 1
    return count


def _ohyou_progression_count(romans: list[str]) -> int:
    """J-Pop 王道 IV-V-iii-vi の出現数 (回転は問わず 4 連の包含)。"""
    target = ["IV", "V", "iii", "vi"]
    count = 0
    if len(romans) < 4:
        return 0
    bare = [r.rstrip("0123456789") for r in romans]
    for i in range(len(bare) - 3):
        if bare[i : i + 4] == target:
            count += 1
    return count


def _syncopation_index(beats: BeatResult, buffer: AudioBuffer) -> float:
    """オンセット強度のうち、拍位置からズレた分の比率。0.0=ピッタリ拍/1.0=完全裏。"""
    if not beats.beat_times:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=buffer.samples, sr=buffer.sample_rate)
    times = librosa.times_like(onset_env, sr=buffer.sample_rate)
    if onset_env.sum() <= 0:
        return 0.0
    beat_array = np.asarray(beats.beat_times)
    # 各時間フレームに対して最寄りビートとの距離 [秒] を求める
    diffs = np.min(np.abs(times[:, None] - beat_array[None, :]), axis=1)
    # 平均ビート間隔の半分以上ズレた成分の重み比
    median_iter = float(np.median(np.diff(beat_array))) if len(beat_array) > 1 else 0.5
    threshold = median_iter / 4.0
    off = onset_env[diffs > threshold].sum()
    return float(off / onset_env.sum())


def _power_chord_ratio(buffer: AudioBuffer) -> float:
    """近似: 偶数次 (M3) のクロマエネルギーが薄く、root + 5th が支配的なフレーム比率。"""
    chroma = librosa.feature.chroma_cqt(y=buffer.samples, sr=buffer.sample_rate, hop_length=4096)
    n = chroma.shape[1]
    if n == 0:
        return 0.0
    matched = 0
    for i in range(n):
        v = chroma[:, i]
        if v.sum() == 0:
            continue
        v = v / v.sum()
        # root, 5th が突出 + 3rd (major/minor) が小さい
        top2 = np.argsort(v)[::-1][:2]
        # 5度関係 (差が 7 半音) または逆向き 5
        diff = abs(int(top2[0]) - int(top2[1]))
        if diff in (5, 7):
            third_pos = (int(top2[0]) + 4) % 12
            third_pos_min = (int(top2[0]) + 3) % 12
            if v[third_pos] < 0.08 and v[third_pos_min] < 0.08:
                matched += 1
    return matched / n


def analyze_styles(
    *,
    buffer: AudioBuffer,
    tempo: TempoResult,
    beats: BeatResult,
    meter: MeterResult,
    key: KeyResult,
    chords: ChordResult,
    harmony: HarmonicAnalysis,
    enabled: list[str],
) -> list[StyleNotes]:
    """有効化されたジャンル観点で分析を行い StyleNotes のリストを返す。"""
    notes: list[StyleNotes] = []
    qual = _chord_quality_distribution(chords)
    seventh_set = {"7", "maj7", "min7", "9", "min9", "maj9", "13", "minmaj7", "m7b5", "hdim7"}
    triad_set = {"maj", "min", "sus2", "sus4", "aug", "dim"}

    if "jazz" in enabled:
        seventh_ratio = _ratio(qual, seventh_set)
        ii_v_i = _ii_v_i_count(harmony.roman_numerals)
        sync = _syncopation_index(beats, buffer)
        findings: list[str] = []
        if seventh_ratio >= 0.4:
            findings.append(f"7th 系コード比率が高い ({seventh_ratio:.0%}) — Jazz/Fusion 傾向")
        if ii_v_i >= 2:
            findings.append(f"ii-V-I 進行を {ii_v_i} 回検出 — Jazz 標準カデンツ")
        if sync >= 0.4:
            findings.append(f"シンコペーション指数 {sync:.2f} — スウィング/裏拍の強さ")
        if 80 <= tempo.bpm <= 220:
            findings.append(f"テンポ {tempo.bpm} BPM は Jazz の典型レンジ")
        notes.append(
            StyleNotes(
                style="jazz",
                findings=findings,
                metrics={
                    "seventh_ratio": round(seventh_ratio, 3),
                    "ii_v_i_count": float(ii_v_i),
                    "syncopation_index": round(sync, 3),
                },
            )
        )

    if "classic" in enabled:
        functional_categories = harmony.chord_categories or {}
        total_classified = sum(functional_categories.values())
        functional_ratio = (
            sum(functional_categories.get(k, 0) for k in ("T", "SD", "D")) / total_classified
            if total_classified
            else 0.0
        )
        modulation_count = len(harmony.modulations)
        cadence_count = len(harmony.cadences)
        findings = []
        if functional_ratio >= 0.7:
            findings.append(
                f"機能和声に整合する和音比率 {functional_ratio:.0%} — Classic / 教会旋法的"
            )
        if cadence_count >= 2:
            findings.append(f"カデンツ検出数 {cadence_count}（{', '.join(harmony.cadences[:3])}…）")
        if modulation_count >= 1:
            findings.append(f"転調候補 {modulation_count} 箇所 — 古典/ロマン派的展開")
        notes.append(
            StyleNotes(
                style="classic",
                findings=findings,
                metrics={
                    "functional_ratio": round(functional_ratio, 3),
                    "modulation_count": float(modulation_count),
                    "cadence_count": float(cadence_count),
                },
            )
        )

    if "jpop" in enabled:
        ohyou = _ohyou_progression_count(harmony.roman_numerals)
        # サビ候補: 振幅 RMS の上位 20% 連続区間長 / 全長
        rms = librosa.feature.rms(y=buffer.samples)[0]
        threshold = float(np.quantile(rms, 0.8))
        loud_ratio = float((rms >= threshold).sum() / max(len(rms), 1))
        findings = []
        if ohyou >= 1:
            findings.append(f"王道進行 IV-V-iii-vi を {ohyou} 回検出 — J-Pop/Anison 典型")
        if 0.18 <= loud_ratio <= 0.32:
            findings.append(f"高音量帯 (サビ候補) 比率 {loud_ratio:.0%} — A→B→サビの構成と整合")
        if 100 <= tempo.bpm <= 160:
            findings.append(f"テンポ {tempo.bpm} BPM は J-Pop ボーカル主体の典型")
        notes.append(
            StyleNotes(
                style="jpop",
                findings=findings,
                metrics={
                    "ohyou_progression_count": float(ohyou),
                    "loud_segment_ratio": round(loud_ratio, 3),
                },
            )
        )

    if "rock" in enabled:
        triad_ratio = _ratio(qual, triad_set)
        power_ratio = _power_chord_ratio(buffer)
        findings = []
        if triad_ratio >= 0.7:
            findings.append(f"三和音中心 ({triad_ratio:.0%}) — Rock/Pop 系のシンプルな和声")
        if power_ratio >= 0.15:
            findings.append(f"パワーコード支配フレーム {power_ratio:.0%} — Hard Rock/Punk 傾向")
        if meter.numerator == 4 and 90 <= tempo.bpm <= 200:
            findings.append(f"4/4 拍子 + {tempo.bpm} BPM — 王道 Rock リズム")
        notes.append(
            StyleNotes(
                style="rock",
                findings=findings,
                metrics={
                    "triad_ratio": round(triad_ratio, 3),
                    "power_chord_ratio": round(power_ratio, 3),
                },
            )
        )

    return notes
