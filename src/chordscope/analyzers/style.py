"""ジャンル別観点の動的ディスパッチ。

AST AudioSet の top-K ジャンル推論結果を駆動源として、各ラベルに対応する
ヒューリスティック関数 (登録ある場合) を呼び出す。未登録ラベルは AST スコア
だけを report する。固定ラベル (`style = ["jazz", "rock", ...]`) も並行指定可能。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import librosa
import numpy as np

from chordscope.audio import AudioBuffer
from chordscope.models import (
    BeatResult,
    ChordResult,
    GenreResult,
    HarmonicAnalysis,
    KeyResult,
    MeterResult,
    StyleNotes,
    TempoResult,
)

# ---------- 共有特徴量計算 ----------


@dataclass(frozen=True)
class StyleContext:
    """各ヒューリスティックに渡す事前計算済み特徴量バンドル。"""

    buffer: AudioBuffer
    tempo: TempoResult
    beats: BeatResult
    meter: MeterResult
    key: KeyResult
    chords: ChordResult
    harmony: HarmonicAnalysis
    chord_quality_dist: Counter[str]
    seventh_ratio: float
    triad_ratio: float
    syncopation_index: float
    power_chord_ratio: float
    loud_segment_ratio: float
    ii_v_i_count: int
    ohyou_progression_count: int
    twelve_bar_blues_match: int


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
    """J-Pop 王道 IV-V-iii-vi の出現数。"""
    target = ["IV", "V", "iii", "vi"]
    if len(romans) < 4:
        return 0
    bare = [r.rstrip("0123456789") for r in romans]
    return sum(1 for i in range(len(bare) - 3) if bare[i : i + 4] == target)


def _twelve_bar_blues_match(romans: list[str]) -> int:
    """12 小節ブルースのコード進行 (I-I-I-I-IV-IV-I-I-V-IV-I-V) との部分一致数。"""
    pattern = ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"]
    if len(romans) < len(pattern):
        return 0
    bare = [r.rstrip("0123456789").upper() for r in romans]
    return sum(
        1 for i in range(len(bare) - len(pattern) + 1) if bare[i : i + len(pattern)] == pattern
    )


def _syncopation_index(beats: BeatResult, buffer: AudioBuffer) -> float:
    if not beats.beat_times:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=buffer.samples, sr=buffer.sample_rate)
    times = librosa.times_like(onset_env, sr=buffer.sample_rate)
    if onset_env.sum() <= 0:
        return 0.0
    beat_array = np.asarray(beats.beat_times)
    diffs = np.min(np.abs(times[:, None] - beat_array[None, :]), axis=1)
    median_iter = float(np.median(np.diff(beat_array))) if len(beat_array) > 1 else 0.5
    threshold = median_iter / 4.0
    off = onset_env[diffs > threshold].sum()
    return float(off / onset_env.sum())


def _power_chord_ratio(buffer: AudioBuffer) -> float:
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
        top2 = np.argsort(v)[::-1][:2]
        diff = abs(int(top2[0]) - int(top2[1]))
        if diff in (5, 7):
            third_pos = (int(top2[0]) + 4) % 12
            third_pos_min = (int(top2[0]) + 3) % 12
            if v[third_pos] < 0.08 and v[third_pos_min] < 0.08:
                matched += 1
    return matched / n


def _loud_segment_ratio(buffer: AudioBuffer) -> float:
    rms = librosa.feature.rms(y=buffer.samples)[0]
    if len(rms) == 0:
        return 0.0
    threshold = float(np.quantile(rms, 0.8))
    return float((rms >= threshold).sum() / len(rms))


def _build_context(
    *,
    buffer: AudioBuffer,
    tempo: TempoResult,
    beats: BeatResult,
    meter: MeterResult,
    key: KeyResult,
    chords: ChordResult,
    harmony: HarmonicAnalysis,
) -> StyleContext:
    qual = _chord_quality_distribution(chords)
    seventh_set = {"7", "maj7", "min7", "9", "min9", "maj9", "13", "minmaj7", "m7b5", "hdim7"}
    triad_set = {"maj", "min", "sus2", "sus4", "aug", "dim"}
    return StyleContext(
        buffer=buffer,
        tempo=tempo,
        beats=beats,
        meter=meter,
        key=key,
        chords=chords,
        harmony=harmony,
        chord_quality_dist=qual,
        seventh_ratio=_ratio(qual, seventh_set),
        triad_ratio=_ratio(qual, triad_set),
        syncopation_index=_syncopation_index(beats, buffer),
        power_chord_ratio=_power_chord_ratio(buffer),
        loud_segment_ratio=_loud_segment_ratio(buffer),
        ii_v_i_count=_ii_v_i_count(harmony.roman_numerals),
        ohyou_progression_count=_ohyou_progression_count(harmony.roman_numerals),
        twelve_bar_blues_match=_twelve_bar_blues_match(harmony.roman_numerals),
    )


# ---------- ヒューリスティック関数群 ----------

HeuristicFn = Callable[[StyleContext], tuple[list[str], dict[str, float]]]


def _h_jazz(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.seventh_ratio >= 0.4:
        findings.append(f"7th 系コード比率 {ctx.seventh_ratio:.0%} — Jazz/Fusion 傾向")
    if ctx.ii_v_i_count >= 2:
        findings.append(f"ii-V-I 進行を {ctx.ii_v_i_count} 回検出 — Jazz 標準カデンツ")
    if ctx.syncopation_index >= 0.4:
        findings.append(f"シンコペーション指数 {ctx.syncopation_index:.2f} — スウィング/裏拍の強さ")
    if 80 <= ctx.tempo.bpm <= 220:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Jazz の典型レンジ")
    return findings, {
        "seventh_ratio": round(ctx.seventh_ratio, 3),
        "ii_v_i_count": float(ctx.ii_v_i_count),
        "syncopation_index": round(ctx.syncopation_index, 3),
    }


def _h_classic(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    cats = ctx.harmony.chord_categories or {}
    total = sum(cats.values())
    func_ratio = sum(cats.get(k, 0) for k in ("T", "SD", "D")) / total if total else 0.0
    findings: list[str] = []
    if func_ratio >= 0.7:
        findings.append(f"機能和声整合度 {func_ratio:.0%} — Classic / 古典的")
    if len(ctx.harmony.cadences) >= 2:
        findings.append(
            f"カデンツ検出 {len(ctx.harmony.cadences)} 種 ({', '.join(ctx.harmony.cadences[:3])})"
        )
    if len(ctx.harmony.modulations) >= 1:
        findings.append(f"転調候補 {len(ctx.harmony.modulations)} 箇所 — 古典/ロマン派的展開")
    return findings, {
        "functional_ratio": round(func_ratio, 3),
        "modulation_count": float(len(ctx.harmony.modulations)),
        "cadence_count": float(len(ctx.harmony.cadences)),
    }


def _h_jpop(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.ohyou_progression_count >= 1:
        findings.append(f"王道進行 IV-V-iii-vi を {ctx.ohyou_progression_count} 回検出")
    if 0.18 <= ctx.loud_segment_ratio <= 0.32:
        findings.append(
            f"高音量帯 (サビ候補) 比率 {ctx.loud_segment_ratio:.0%} — A→B→サビ構成と整合"
        )
    if 100 <= ctx.tempo.bpm <= 160:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は J-Pop ボーカル主体の典型")
    return findings, {
        "ohyou_progression_count": float(ctx.ohyou_progression_count),
        "loud_segment_ratio": round(ctx.loud_segment_ratio, 3),
    }


def _h_rock(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.triad_ratio >= 0.7:
        findings.append(f"三和音中心 ({ctx.triad_ratio:.0%}) — Rock/Pop 系のシンプルな和声")
    if ctx.power_chord_ratio >= 0.15:
        findings.append(f"パワーコード支配 {ctx.power_chord_ratio:.0%} — Hard Rock/Punk 傾向")
    if ctx.meter.numerator == 4 and 90 <= ctx.tempo.bpm <= 200:
        findings.append(f"4/4 + {ctx.tempo.bpm} BPM — 王道 Rock リズム")
    return findings, {
        "triad_ratio": round(ctx.triad_ratio, 3),
        "power_chord_ratio": round(ctx.power_chord_ratio, 3),
    }


def _h_blues(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    dom7_ratio = _ratio(ctx.chord_quality_dist, {"7"})
    findings: list[str] = []
    if ctx.twelve_bar_blues_match >= 1:
        findings.append(f"12 小節ブルース進行を {ctx.twelve_bar_blues_match} 回検出 — Blues 標準形")
    if dom7_ratio >= 0.3:
        findings.append(f"ドミナント 7th 比率 {dom7_ratio:.0%} — Blues 特有の和声")
    if 60 <= ctx.tempo.bpm <= 130:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Blues の典型レンジ")
    return findings, {
        "twelve_bar_match": float(ctx.twelve_bar_blues_match),
        "dom7_ratio": round(dom7_ratio, 3),
    }


def _h_hiphop(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    chord_change_density = (
        len(ctx.chords.segments) / ctx.buffer.duration if ctx.buffer.duration > 0 else 0.0
    )
    if 70 <= ctx.tempo.bpm <= 110:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は HipHop/Trap の典型 (60-110)")
    if ctx.syncopation_index >= 0.45:
        findings.append(f"シンコペーション {ctx.syncopation_index:.2f} — リズム重心が裏に寄る")
    if chord_change_density < 0.4:
        findings.append(f"コード変化頻度 {chord_change_density:.2f}/秒 — 反復的・ループ志向")
    return findings, {
        "syncopation_index": round(ctx.syncopation_index, 3),
        "chord_change_density": round(chord_change_density, 3),
    }


def _h_electronic(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    edm_subgenre = None
    if 120 <= ctx.tempo.bpm <= 130:
        edm_subgenre = "House"
    elif 130 <= ctx.tempo.bpm <= 150:
        edm_subgenre = "Techno/Trance"
    elif 160 <= ctx.tempo.bpm <= 180:
        edm_subgenre = "Drum and Bass"
    elif 65 <= ctx.tempo.bpm <= 75 or 138 <= ctx.tempo.bpm <= 142:
        edm_subgenre = "Dubstep"
    if edm_subgenre:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM — {edm_subgenre} レンジ")
    if ctx.meter.numerator == 4 and ctx.meter.confidence >= 0.85:
        findings.append("4/4 高信頼度 — 4-on-the-floor キック前提の構成")
    return findings, {"bpm": ctx.tempo.bpm, "meter_confidence": ctx.meter.confidence}


def _h_country(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    triple = _ratio(ctx.chord_quality_dist, {"maj", "min"})
    findings: list[str] = []
    if triple >= 0.85:
        findings.append(f"単純三和音中心 ({triple:.0%}) — Country/Folk 和声")
    if 90 <= ctx.tempo.bpm <= 140:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Country の典型レンジ")
    if ctx.meter.numerator in (3, 4):
        findings.append(f"{ctx.meter.numerator}/4 拍子 — Country/Bluegrass の典型")
    return findings, {"simple_triad_ratio": round(triple, 3)}


def _h_soul_funk_rnb(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.seventh_ratio >= 0.3:
        findings.append(f"7th コード {ctx.seventh_ratio:.0%} — Soul/Funk/R&B 和声")
    if ctx.syncopation_index >= 0.5:
        findings.append(f"シンコペ {ctx.syncopation_index:.2f} — Funk グルーヴ")
    if 80 <= ctx.tempo.bpm <= 130:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Soul/R&B の典型")
    return findings, {
        "seventh_ratio": round(ctx.seventh_ratio, 3),
        "syncopation_index": round(ctx.syncopation_index, 3),
    }


def _h_metal(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.power_chord_ratio >= 0.25:
        findings.append(f"パワーコード比率 {ctx.power_chord_ratio:.0%} — Metal/Hard Rock")
    if ctx.tempo.bpm >= 140:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM — Speed/Thrash Metal レンジ")
    if ctx.key.mode == "minor":
        findings.append(f"短調 ({ctx.key.tonic} minor) — Metal で支配的")
    return findings, {"power_chord_ratio": round(ctx.power_chord_ratio, 3)}


def _h_reggae(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if 60 <= ctx.tempo.bpm <= 100:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Reggae の典型 (60-100)")
    if ctx.syncopation_index >= 0.4:
        findings.append(f"裏拍重心 (シンコペ {ctx.syncopation_index:.2f}) — Reggae スカ刻み")
    if ctx.meter.numerator == 4:
        findings.append("4/4 拍子 — Reggae 標準")
    return findings, {"syncopation_index": round(ctx.syncopation_index, 3)}


def _h_folk(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    simple = _ratio(ctx.chord_quality_dist, {"maj", "min"})
    findings: list[str] = []
    if simple >= 0.85:
        findings.append(f"単純三和音 {simple:.0%} — Folk/アコースティック")
    if 70 <= ctx.tempo.bpm <= 130:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Folk の典型")
    if ctx.meter.numerator in (3, 4, 6):
        findings.append(f"{ctx.meter.numerator}/4 — Folk/Traditional の典型拍子")
    return findings, {"simple_chord_ratio": round(simple, 3)}


def _h_latin(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.syncopation_index >= 0.5:
        findings.append(
            f"高シンコペーション {ctx.syncopation_index:.2f} — Latin (Salsa/Samba/Bossa) の特徴"
        )
    if 90 <= ctx.tempo.bpm <= 200:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Latin ダンスの典型")
    return findings, {"syncopation_index": round(ctx.syncopation_index, 3)}


def _h_gospel(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    plagal_count = sum(1 for c in ctx.harmony.cadences if "Plagal" in c)
    if plagal_count >= 1:
        findings.append(f"Plagal Cadence (IV→I) を {plagal_count} 箇所検出 — Gospel 定型")
    if ctx.seventh_ratio >= 0.2:
        findings.append(f"7th コード {ctx.seventh_ratio:.0%} — Gospel/Soul の彩り")
    return findings, {"plagal_count": float(plagal_count)}


def _h_punk(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.power_chord_ratio >= 0.2 and ctx.tempo.bpm >= 150:
        findings.append(
            f"パワーコード {ctx.power_chord_ratio:.0%} + 高速 BPM {ctx.tempo.bpm} — Punk/HC"
        )
    if ctx.triad_ratio >= 0.85:
        findings.append(f"三和音純度 {ctx.triad_ratio:.0%} — Punk のシンプル和声")
    return findings, {"power_chord_ratio": round(ctx.power_chord_ratio, 3)}


def _h_ambient(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    chord_density = (
        len(ctx.chords.segments) / ctx.buffer.duration if ctx.buffer.duration > 0 else 0.0
    )
    if chord_density < 0.2:
        findings.append(f"コード変化 {chord_density:.2f}/秒 — 持続音中心 (Ambient/Drone/New Age)")
    if ctx.tempo.confidence < 0.5:
        findings.append(f"テンポ信頼度 {ctx.tempo.confidence} — 拍が曖昧 (Ambient 特性)")
    return findings, {"chord_change_density": round(chord_density, 3)}


def _h_flamenco(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.key.mode == "minor" and ctx.syncopation_index >= 0.5:
        findings.append(f"短調 + 高シンコペ {ctx.syncopation_index:.2f} — Flamenco/Spanish 風")
    if 90 <= ctx.tempo.bpm <= 220:
        findings.append(f"テンポ {ctx.tempo.bpm} BPM は Flamenco compas レンジ")
    return findings, {"syncopation_index": round(ctx.syncopation_index, 3)}


def _h_ska(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if 100 <= ctx.tempo.bpm <= 180 and ctx.syncopation_index >= 0.4:
        findings.append(f"テンポ {ctx.tempo.bpm} + 裏拍 — Ska/Reggae 派生")
    return findings, {}


def _h_anime(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.ohyou_progression_count >= 1 and 110 <= ctx.tempo.bpm <= 180:
        findings.append(f"王道進行 + テンポ {ctx.tempo.bpm} BPM — Anime/J-Pop 派生の典型")
    return findings, {"ohyou_progression_count": float(ctx.ohyou_progression_count)}


def _h_enka(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.key.mode == "minor" and 60 <= ctx.tempo.bpm <= 100:
        findings.append(f"短調 + 緩テンポ {ctx.tempo.bpm} BPM — Enka/演歌 の典型")
    if len(ctx.harmony.modulations) >= 2:
        findings.append("転調多 — Enka の感情起伏表現")
    return findings, {}


def _h_world(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.meter.numerator in (5, 7):
        findings.append(f"{ctx.meter.numerator}/4 拍子 — 非西洋圏 (バルカン・中東等) の特徴")
    if ctx.syncopation_index >= 0.55:
        findings.append(f"高シンコペ {ctx.syncopation_index:.2f} — World/Ethnic 由来")
    return findings, {"meter_numerator": float(ctx.meter.numerator)}


def _h_vocal(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.power_chord_ratio < 0.05 and ctx.triad_ratio >= 0.8:
        findings.append("ハーモニー中心・歪み少 — Vocal/Choral/A cappella 系")
    return findings, {}


def _h_soundtrack(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if len(ctx.harmony.modulations) >= 3:
        findings.append(f"転調候補 {len(ctx.harmony.modulations)} — Soundtrack/Film score 展開")
    return findings, {"modulation_count": float(len(ctx.harmony.modulations))}


def _h_musical(ctx: StyleContext) -> tuple[list[str], dict[str, float]]:
    findings: list[str] = []
    if ctx.seventh_ratio >= 0.3 and ctx.ohyou_progression_count == 0:
        findings.append("7th 多様 + ポップ進行少 — Musical Theater/Show tunes")
    return findings, {}


# ラベル → ヒューリスティック関数。AST AudioSet ラベル, 一般ジャンル名, 短縮名 すべて受ける。
# キーは小文字。検索は完全一致 → 長キー部分一致の順で行う。
_HEURISTIC_REGISTRY: dict[str, HeuristicFn] = {
    # コア 4
    "jazz": _h_jazz,
    "classical music": _h_classic,
    "classical": _h_classic,
    "classic": _h_classic,
    "j-pop": _h_jpop,
    "jpop": _h_jpop,
    "rock music": _h_rock,
    "rock": _h_rock,
    "pop music": _h_jpop,
    # ブルース系
    "blues": _h_blues,
    "rhythm and blues": _h_soul_funk_rnb,
    # ヒップホップ
    "hip hop music": _h_hiphop,
    "hip hop": _h_hiphop,
    "hiphop": _h_hiphop,
    "rap": _h_hiphop,
    "trap": _h_hiphop,
    # エレクトロニック
    "electronic music": _h_electronic,
    "electronic": _h_electronic,
    "house music": _h_electronic,
    "house": _h_electronic,
    "techno": _h_electronic,
    "trance music": _h_electronic,
    "trance": _h_electronic,
    "drum and bass": _h_electronic,
    "dubstep": _h_electronic,
    "edm": _h_electronic,
    # カントリー / フォーク
    "country": _h_country,
    "bluegrass": _h_country,
    "folk music": _h_folk,
    "folk": _h_folk,
    # ソウル/ファンク
    "soul music": _h_soul_funk_rnb,
    "soul": _h_soul_funk_rnb,
    "funk": _h_soul_funk_rnb,
    # メタル / パンク
    "heavy metal": _h_metal,
    "metal": _h_metal,
    "punk rock": _h_punk,
    "punk": _h_punk,
    "grunge": _h_punk,
    # レゲエ / スカ
    "reggae": _h_reggae,
    "ska": _h_ska,
    # ラテン
    "latin music": _h_latin,
    "latin": _h_latin,
    "salsa music": _h_latin,
    "salsa": _h_latin,
    "bossa nova": _h_latin,
    "samba": _h_latin,
    "tango": _h_latin,
    "mambo": _h_latin,
    "cumbia": _h_latin,
    "music of latin america": _h_latin,
    # ゴスペル / クリスチャン
    "gospel music": _h_gospel,
    "gospel": _h_gospel,
    "christian music": _h_gospel,
    "christian": _h_gospel,
    # アンビエント
    "ambient music": _h_ambient,
    "ambient": _h_ambient,
    "new-age music": _h_ambient,
    "new age": _h_ambient,
    # フラメンコ / 民族
    "flamenco": _h_flamenco,
    "music of asia": _h_world,
    "music of africa": _h_world,
    "music of the middle east": _h_world,
    "music of bollywood": _h_world,
    "carnatic music": _h_world,
    "world": _h_world,
    "afrobeat": _h_world,
    # 日本特化
    "anime": _h_anime,
    "vocaloid": _h_anime,
    "enka": _h_enka,
    # ボーカル / 合唱
    "a capella": _h_vocal,
    "vocal music": _h_vocal,
    "choir": _h_vocal,
    "chant": _h_vocal,
    "opera": _h_vocal,
    # サウンドトラック
    "soundtrack music": _h_soundtrack,
    "theme music": _h_soundtrack,
    "video game music": _h_soundtrack,
    # ミュージカル
    "musical": _h_musical,
    "show tunes": _h_musical,
}


def _lookup_heuristic(label: str) -> HeuristicFn | None:
    """ラベル文字列にマッチするヒューリスティックを返す。

    完全一致を優先。次に長いキーから順に部分一致を探す (短い "rock" が
    "Punk rock" にマッチするのを避けるため)。
    """
    lower = label.lower()
    if lower in _HEURISTIC_REGISTRY:
        return _HEURISTIC_REGISTRY[lower]
    for key in sorted(_HEURISTIC_REGISTRY, key=len, reverse=True):
        if key in lower:
            return _HEURISTIC_REGISTRY[key]
    return None


# ---------- メイン API ----------


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
    genre: GenreResult | None = None,
    top_k: int = 8,
) -> list[StyleNotes]:
    """Style notes を生成する。

    `enabled` 中の "auto" は GenreResult から top-K ジャンルを動的展開する。
    "auto" 以外の項目は固定ジャンル名として扱う (例: ["jazz", "rock"])。
    両方を混在させると "auto" 展開分 + 固定指定分の和集合になる。
    """
    ctx = _build_context(
        buffer=buffer,
        tempo=tempo,
        beats=beats,
        meter=meter,
        key=key,
        chords=chords,
        harmony=harmony,
    )
    target_labels: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for entry in enabled:
        if entry.lower() == "auto":
            if genre is None:
                continue
            for g in genre.distribution[:top_k]:
                if g.label not in seen:
                    target_labels.append((g.label, g.score))
                    seen.add(g.label)
        elif entry not in seen:
            target_labels.append((entry, None))
            seen.add(entry)
    notes: list[StyleNotes] = []
    for label, score in target_labels:
        h = _lookup_heuristic(label)
        if h is not None:
            findings, metrics = h(ctx)
        else:
            findings, metrics = [], {}
        if score is not None:
            findings.insert(0, f"AST 推論スコア {score:.2%}")
            metrics["ast_score"] = round(score, 4)
        if h is None:
            findings.append("ヒューリスティック未登録 (詳細解析なし)")
        notes.append(StyleNotes(style=label, findings=findings, metrics=metrics))
    return notes
