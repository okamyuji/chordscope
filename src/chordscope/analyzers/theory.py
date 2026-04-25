"""コード列に対する音楽理論分析 (music21 利用)。

- ローマ数字解析: 推定キーで各和音をローマ数字に変換
- 機能和声: トニック (T)、サブドミナント (SD)、ドミナント (D) に分類
- カデンツ検出: V→I (Authentic), IV→I (Plagal), V→vi (Deceptive), I-IV-V-I (Standard)
- 転調候補: 一定区間でローマ数字解析の整合性が崩れる箇所を検出
"""

from __future__ import annotations

from collections import Counter

from music21 import chord as m21chord
from music21 import key as m21key
from music21 import pitch as m21pitch
from music21 import roman as m21roman

from chordscope.models import ChordResult, HarmonicAnalysis, KeyResult

# 音楽理論ラベル → グループ
_FUNCTION_GROUPS = {
    "T": {"I", "i", "vi", "VI", "iii", "III"},
    "SD": {"IV", "iv", "ii", "II", "ii6", "iio", "iiø"},
    "D": {"V", "v", "vii", "viio", "viiø", "V7"},
}


def _label_to_pitches(label: str) -> list[str] | None:
    """'C:maj7' のようなラベルをピッチ名のリストに変換。

    madmom Chordino 形式: '<root>:<quality>[/<bass>]'。'N' は無和音。
    """
    if label in ("N", "X", ""):
        return None
    parts = label.split("/", maxsplit=1)[0].split(":")
    root_name = parts[0]
    qual = parts[1] if len(parts) > 1 else "maj"
    try:
        root = m21pitch.Pitch(root_name)
    except Exception:
        return None
    intervals_map = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "7": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dim": [0, 3, 6],
        "dim7": [0, 3, 6, 9],
        "hdim7": [0, 3, 6, 10],
        "m7b5": [0, 3, 6, 10],
        "aug": [0, 4, 8],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "minmaj7": [0, 3, 7, 11],
        "9": [0, 4, 7, 10, 14],
        "min9": [0, 3, 7, 10, 14],
        "maj9": [0, 4, 7, 11, 14],
        "13": [0, 4, 7, 10, 14, 21],
        "6": [0, 4, 7, 9],
        "min6": [0, 3, 7, 9],
    }
    intervals = intervals_map.get(qual, [0, 4, 7])
    pitches: list[str] = []
    for iv in intervals:
        p = root.transpose(iv)
        pitches.append(p.nameWithOctave)
    return pitches


def _to_roman(label: str, k: m21key.Key) -> str | None:
    pitches = _label_to_pitches(label)
    if pitches is None:
        return None
    try:
        c = m21chord.Chord(pitches)
        return m21roman.romanNumeralFromChord(c, k).figure
    except Exception:
        return None


def _classify(rn: str) -> str:
    base = rn.rstrip("0123456789").rstrip("ø°+")
    for cat, labels in _FUNCTION_GROUPS.items():
        if base in labels:
            return cat
    return "OTHER"


def _detect_cadences(romans: list[str]) -> list[str]:
    """連続する 2-3 コードでカデンツ候補を抽出。"""
    findings: list[str] = []
    for i in range(len(romans) - 1):
        a, b = romans[i].rstrip("0123456789"), romans[i + 1].rstrip("0123456789")
        if a == "V" and b in ("I", "i"):
            findings.append(f"Authentic Cadence ({romans[i]}→{romans[i + 1]})")
        if a in ("IV", "iv") and b in ("I", "i"):
            findings.append(f"Plagal Cadence ({romans[i]}→{romans[i + 1]})")
        if a == "V" and b in ("vi", "VI"):
            findings.append(f"Deceptive Cadence ({romans[i]}→{romans[i + 1]})")
        if a in ("ii", "II") and b == "V":
            findings.append(f"ii→V motion ({romans[i]}→{romans[i + 1]})")
    # 重複は順序保持で除去
    seen: list[str] = []
    for f in findings:
        if f not in seen:
            seen.append(f)
    return seen


def _detect_modulations(labels: list[str], window: int = 8) -> list[str]:
    """各 window でローマ数字解析した結果が変化する箇所を転調候補として返す。

    あくまで簡易的: window 単位でクロマからキーを再推定し、変化があればフラグ。
    実装簡略のため、コードの root 出現分布の変化を検出する。
    """
    if len(labels) < window * 2:
        return []
    findings: list[str] = []
    prev_top: str | None = None
    for i in range(0, len(labels) - window, window):
        win = labels[i : i + window]
        roots = []
        for lbl in win:
            if lbl == "N":
                continue
            roots.append(lbl.split(":")[0])
        if not roots:
            continue
        c = Counter(roots)
        top = c.most_common(1)[0][0]
        if prev_top is not None and top != prev_top:
            findings.append(f"Possible key change near chord index {i} ({prev_top} → {top})")
        prev_top = top
    return findings


def analyze_harmony(chords: ChordResult, key: KeyResult) -> HarmonicAnalysis:
    """ChordResult + KeyResult からローマ数字・カデンツ・転調を抽出。"""
    if not chords.segments:
        return HarmonicAnalysis()
    k = m21key.Key(key.tonic, key.mode)
    romans: list[str] = []
    categories: Counter[str] = Counter()
    for seg in chords.segments:
        rn = _to_roman(seg.label, k)
        if rn is None:
            continue
        romans.append(rn)
        categories[_classify(rn)] += 1
    cadences = _detect_cadences(romans)
    labels_only = [s.label for s in chords.segments]
    modulations = _detect_modulations(labels_only)
    return HarmonicAnalysis(
        roman_numerals=romans,
        cadences=cadences,
        modulations=modulations,
        chord_categories=dict(categories),
    )
