"""分析結果を表す不変なデータモデル群。

すべて pydantic.BaseModel ベース。`model_config = ConfigDict(frozen=True)` で
インスタンス化後の変更を禁止し、不変性 (グローバルコーディング規約) を保つ。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BeatResult",
    "ChordResult",
    "ChordSegment",
    "GenreLabel",
    "GenreResult",
    "GenreScore",
    "HarmonicAnalysis",
    "KeyChange",
    "KeyResult",
    "KeySegment",
    "MeterResult",
    "ModulationResult",
    "StyleName",
    "StyleNotes",
    "TempoCurveResult",
    "TempoResult",
    "TempoSegment",
    "TrackAnalysis",
]

GenreLabel = str
# StyleName は固定 Literal を撤廃し、AST AudioSet ラベルや任意ジャンル名を許容する。
# heuristic レジストリに対応キーがあれば固有 finding が出る。未登録は AST スコアのみ報告。
StyleName = str


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class TempoResult(_Frozen):
    """BPM と推定信頼度。複数アルゴリズムで推定し最も合意が取れた値を採用する。"""

    bpm: float = Field(..., description="採用 BPM (整数寄り)")
    bpm_candidates: list[float] = Field(default_factory=list, description="各推定器の BPM")
    confidence: float = Field(..., ge=0.0, le=1.0, description="推定一致度から導いた信頼度")
    method: str = Field(..., description="採用した推定手法名")


class BeatResult(_Frozen):
    """ビート位置とダウンビート位置 (秒)。"""

    beat_times: list[float] = Field(default_factory=list)
    downbeat_times: list[float] = Field(default_factory=list)
    beats_per_bar: int = Field(..., ge=2, le=12, description="1 小節あたりの拍数 (拍子分子)")
    method: str


class MeterResult(_Frozen):
    """拍子。分母は便宜上 4 を既定とする。"""

    numerator: int = Field(..., ge=2, le=12)
    denominator: int = Field(default=4)
    confidence: float = Field(..., ge=0.0, le=1.0)


class KeyResult(_Frozen):
    """調性 (Krumhansl-Schmuckler)。"""

    tonic: str = Field(..., description="主音 (例: 'C', 'F#')")
    mode: Literal["major", "minor"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    correlation: float = Field(..., description="プロファイル相関係数")
    second_best: tuple[str, str] | None = Field(
        default=None, description="次点 (tonic, mode)。転調候補の参考"
    )


class ChordSegment(_Frozen):
    """1 つの和音区間。"""

    start: float
    end: float
    label: str = Field(..., description="例: 'C:maj', 'A:min7', 'N' (=無音)")


class ChordResult(_Frozen):
    """全曲のコード認識結果。"""

    segments: list[ChordSegment]
    method: str
    unique_chords: list[str] = Field(default_factory=list)


class HarmonicAnalysis(_Frozen):
    """music21 ベースの機能分析結果。"""

    roman_numerals: list[str] = Field(default_factory=list, description="ローマ数字表記")
    cadences: list[str] = Field(default_factory=list, description="検出されたカデンツ種別")
    modulations: list[str] = Field(default_factory=list, description="転調候補")
    chord_categories: dict[str, int] = Field(
        default_factory=dict,
        description="トニック/ドミナント/サブドミナント等の出現数",
    )


class GenreScore(_Frozen):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class GenreResult(_Frozen):
    """学習済みモデルによるジャンル分類結果。"""

    top: GenreScore
    distribution: list[GenreScore] = Field(default_factory=list)
    model_id: str
    aggregation: Literal["mean", "max"]


class StyleNotes(_Frozen):
    """ジャンル別の観点分析。"""

    style: StyleName
    findings: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class KeySegment(_Frozen):
    """ある時間区間における局所キー推定。"""

    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    tonic: str
    mode: Literal["major", "minor"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    correlation: float


class KeyChange(_Frozen):
    """転調イベント (前後 2 つの KeySegment の境界)。"""

    at_sec: float = Field(..., ge=0.0)
    from_tonic: str
    from_mode: Literal["major", "minor"]
    to_tonic: str
    to_mode: Literal["major", "minor"]
    interval_semitones: int = Field(..., description="前後の主音差。-6..+6 に正規化済み")
    relation: Literal["dominant", "subdominant", "relative", "parallel", "chromatic", "other"]


class ModulationResult(_Frozen):
    """全曲の局所キー時系列と転調イベント。"""

    window_sec: float = Field(..., gt=0.0)
    hop_sec: float = Field(..., gt=0.0)
    segments: list[KeySegment] = Field(default_factory=list)
    changes: list[KeyChange] = Field(default_factory=list)
    method: str


class TempoSegment(_Frozen):
    """ある区間の局所 BPM とラベル。"""

    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    local_bpm: float = Field(..., gt=0.0)
    delta_pct: float = Field(..., description="global BPM 比 (-15.0 = 15% 遅い)")
    label: Literal["slow", "stable", "fast"]


class TempoCurveResult(_Frozen):
    """局所 BPM 時系列 + 統計。"""

    window_beats: int = Field(..., ge=2)
    global_bpm: float = Field(..., gt=0.0)
    bpm_min: float = Field(..., ge=0.0)
    bpm_max: float = Field(..., ge=0.0)
    bpm_std: float = Field(..., ge=0.0)
    segments: list[TempoSegment] = Field(default_factory=list)
    trend: Literal["stable", "accelerando", "ritardando", "variable"]
    method: str


class TrackAnalysis(_Frozen):
    """1 曲分の最終分析結果。"""

    file_path: Path
    duration_sec: float
    sample_rate: int

    tempo: TempoResult
    beats: BeatResult
    meter: MeterResult
    key: KeyResult
    chords: ChordResult
    harmony: HarmonicAnalysis
    modulation: ModulationResult | None = None
    tempo_curve: TempoCurveResult | None = None
    genre: GenreResult | None = None
    style_notes: list[StyleNotes] = Field(default_factory=list)

    plot_paths: dict[str, Path] = Field(default_factory=dict)
