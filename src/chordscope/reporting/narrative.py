"""音楽分析を論理的フローで出力する narrative レポート。

既存の TrackAnalysis から事実データを再構成し、章立て + 接続詞で
「単なる羅列」を「論理的な分析文」に変換する。LLM 不要、テンプレートのみ。

章立て:
  1. 楽曲の骨格 (BPM/拍子/調/長さ/コード総数)
  2. 和声の流れと特徴 (主軸調 + コード頻度 + 機能分類 + カデンツ + 転調)
  3. リズム・グルーヴの性格 (テンポレンジ + シンコペーション)
  4. ジャンル傾向 (AST top-K + style notes クロス参照)
  5. この曲を特徴づける指標 (style ctx の各指標を逸脱観点で)
  6. コード進行の時系列概観 (時間ビン × 主要コード)
  7. 算出根拠 (技術メモ)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from chordscope.models import TrackAnalysis


def _section_header(num: int, title: str) -> str:
    return f"## {num}. {title}"


def _tempo_band(bpm: float) -> str:
    if bpm < 70:
        return "落ち着いたバラード/スロー"
    if bpm < 100:
        return "ミディアムスロー"
    if bpm < 130:
        return "ミディアム"
    if bpm < 160:
        return "アップテンポ"
    return "高速"


def _format_skeleton(track: TrackAnalysis) -> list[str]:
    lines = [_section_header(1, "楽曲の骨格 (一目で分かる基本)"), ""]
    lines.append(
        f"- BPM: **{track.tempo.bpm:.2f}** "
        f"(候補 {track.tempo.bpm_candidates}, 信頼度 {track.tempo.confidence:.0%})"
    )
    lines.append(
        f"- 拍子: **{track.meter.numerator}/{track.meter.denominator}** "
        f"(信頼度 {track.meter.confidence:.0%})"
    )
    second = (
        f", 次点 {track.key.second_best[0]} {track.key.second_best[1]}"
        if track.key.second_best
        else ""
    )
    lines.append(
        f"- 調性: **{track.key.tonic} {track.key.mode}** "
        f"(KS 相関 {track.key.correlation}, 信頼度 {track.key.confidence:.0%}{second})"
    )
    minutes = int(track.duration_sec // 60)
    seconds = int(track.duration_sec % 60)
    lines.append(f"- 楽曲長: **{minutes} 分 {seconds:02d} 秒**")
    lines.append(
        f"- コード総数: **{len(track.chords.segments)} セグメント** / "
        f"**{len(track.chords.unique_chords)} 種**"
    )
    lines.append("")
    return lines


def _format_harmony(track: TrackAnalysis) -> list[str]:
    lines = [_section_header(2, "和声の流れと特徴"), ""]
    lines.append(f"- 主軸は **{track.key.tonic} {track.key.mode}** の自然音階")
    chord_counter = Counter(s.label for s in track.chords.segments if s.label not in ("N", "X", ""))
    top5 = chord_counter.most_common(5)
    if top5:
        chord_str = " / ".join(f"`{lbl}` ({cnt}回)" for lbl, cnt in top5)
        lines.append(f"- 出現上位 5 コード: {chord_str}")
    cats = track.harmony.chord_categories or {}
    if cats:
        cat_str = " / ".join(f"{k}={v}" for k, v in cats.items())
        total = sum(cats.values())
        functional = sum(cats.get(k, 0) for k in ("T", "SD", "D"))
        ratio = functional / total if total else 0.0
        lines.append(f"- 機能分類: {cat_str}")
        if ratio >= 0.7:
            interpretation = "**古典的な機能和声に厳密**"
        elif ratio >= 0.4:
            interpretation = "機能和声寄りだがやや自由"
        else:
            interpretation = "機能和声に厳密でない**ジャズ/ポップ寄り**の和声配置"
        lines.append(f"  - 機能和声整合度 {ratio:.0%} → {interpretation}")
    if track.harmony.cadences:
        cad_text = ", ".join(track.harmony.cadences[:3])
        suffix = " 他" if len(track.harmony.cadences) > 3 else ""
        lines.append(f"- カデンツ検出 ({len(track.harmony.cadences)} 種): {cad_text}{suffix}")
    if track.harmony.modulations:
        lines.append(
            f"- 転調候補 **{len(track.harmony.modulations)} 箇所** → "
            "セクションごとに調が揺らぐ展開的構成"
        )
    lines.append("")
    return lines


def _format_rhythm(track: TrackAnalysis) -> list[str]:
    lines = [_section_header(3, "リズム・グルーヴの性格"), ""]
    bpm = track.tempo.bpm
    band = _tempo_band(bpm)
    lines.append(
        f"- {track.meter.numerator}/{track.meter.denominator} + {bpm:.2f} BPM → **{band}**"
    )
    sync_idx: float | None = None
    for sn in track.style_notes:
        if "syncopation_index" in sn.metrics:
            sync_idx = sn.metrics["syncopation_index"]
            break
    if sync_idx is not None:
        if sync_idx >= 0.5:
            sync_desc = "**裏拍重心の強いシンコペーション**"
        elif sync_idx >= 0.3:
            sync_desc = "適度なシンコペーション"
        else:
            sync_desc = "拍ジャストに近い**ストレート**なリズム"
        lines.append(f"- シンコペーション指数 {sync_idx:.2f} → {sync_desc}")
    if track.beats.beat_times:
        lines.append(
            f"- 検出ビート数 {len(track.beats.beat_times)} "
            f"(うちダウンビート {len(track.beats.downbeat_times)})"
        )
    lines.append("")
    return lines


def _format_genre(track: TrackAnalysis) -> list[str]:
    lines = [_section_header(4, "ジャンル傾向 (AST AudioSet 推論)"), ""]
    if track.genre is None:
        lines.append("- ジャンル分類は実行されていません (`--no-genre`)")
        lines.append("")
        return lines
    style_by_label = {sn.style: sn for sn in track.style_notes}
    for g in track.genre.distribution:
        cross_finding = ""
        sn = style_by_label.get(g.label)
        if sn:
            extras = [f for f in sn.findings if not f.startswith("AST 推論スコア")]
            if extras:
                cross_finding = f" ← {extras[0]}"
        lines.append(f"- **{g.label}** ({g.score:.2%}){cross_finding}")
    lines.append("")
    return lines


_METRIC_ANNOTATIONS: dict[str, tuple[str, str]] = {
    "triad_ratio": ("三和音比率", "Rock/Pop 寄りのシンプルな和声"),
    "seventh_ratio": ("7th 比率", "ジャズ的色彩"),
    "power_chord_ratio": ("パワーコード比率", "Hard Rock/Punk 度"),
    "syncopation_index": ("シンコペーション指数", "裏拍の強さ"),
    "ohyou_progression_count": ("王道進行 IV-V-iii-vi 出現数", "J-Pop/Anison 度"),
    "ii_v_i_count": ("ii-V-I 進行数", "Jazz 標準カデンツ度"),
    "twelve_bar_match": ("12 小節ブルース一致数", "Blues 進行度"),
    "chord_change_density": ("コード変化頻度 (/秒)", "和声密度"),
    "loud_segment_ratio": ("高音量帯比率", "サビ的盛り上がり比率"),
    "dom7_ratio": ("ドミナント 7th 比率", "Blues 特有度"),
    "modulation_count": ("転調候補数", "展開度"),
    "cadence_count": ("カデンツ数", "終止形の明示度"),
    "functional_ratio": ("機能和声整合度", "古典度"),
    "simple_triad_ratio": ("単純三和音比率", "Country/Folk 度"),
    "simple_chord_ratio": ("単純コード比率", "アコースティック度"),
    "plagal_count": ("Plagal Cadence 数", "Gospel 定型度"),
    "meter_numerator": ("拍子分子", "拍子の特殊性"),
}


def _format_distinctive(track: TrackAnalysis) -> list[str]:
    lines = [_section_header(5, "この曲を特徴づける指標"), ""]
    aggregated: dict[str, float] = {}
    for sn in track.style_notes:
        for k, v in sn.metrics.items():
            if k != "ast_score":
                aggregated.setdefault(k, v)
    if not aggregated:
        lines.append("- (style notes の metrics が空)")
        lines.append("")
        return lines
    for k, v in sorted(aggregated.items(), key=lambda kv: -kv[1]):
        if k not in _METRIC_ANNOTATIONS:
            continue
        jp, hint = _METRIC_ANNOTATIONS[k]
        if isinstance(v, float) and 0 < v < 1 and k.endswith("_ratio"):
            lines.append(f"- {jp}: **{v:.2%}** ({hint})")
        else:
            lines.append(f"- {jp}: **{v:g}** ({hint})")
    lines.append("")
    return lines


def _format_timeline(track: TrackAnalysis, n_bins: int = 8) -> list[str]:
    lines = [_section_header(6, "コード進行の時系列概観"), ""]
    if not track.chords.segments or track.duration_sec <= 0:
        lines.append("- (コード認識結果なし)")
        lines.append("")
        return lines
    bin_size = track.duration_sec / n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        in_bin = [s for s in track.chords.segments if s.start < end and s.end > start]
        labels = [s.label for s in in_bin if s.label not in ("N", "X", "")]
        if not labels:
            most_common = "(無音/無和音)"
        else:
            counter = Counter(labels)
            most_common = " / ".join(f"`{lbl}`({c})" for lbl, c in counter.most_common(3))
        m1, s1 = int(start // 60), int(start % 60)
        m2, s2 = int(end // 60), int(end % 60)
        lines.append(f"- {m1}:{s1:02d}-{m2}:{s2:02d}: 主要コード {most_common}")
    lines.append("")
    return lines


def _format_technical(track: TrackAnalysis) -> list[str]:
    lines = [_section_header(7, "算出根拠 (技術メモ)"), ""]
    lines.append(f"- BPM 推定エンジン: {track.tempo.method}")
    lines.append(f"- ビート/ダウンビート: {track.beats.method}")
    lines.append(f"- コード認識エンジン: {track.chords.method}")
    if track.genre:
        lines.append(f"- ジャンル分類モデル: {track.genre.model_id}")
        lines.append(f"  - 集約方法: {track.genre.aggregation}")
    lines.append(f"- サンプルレート: {track.sample_rate} Hz")
    lines.append("")
    return lines


def render_narrative(track: TrackAnalysis) -> str:
    """TrackAnalysis から論理フロー型の音楽分析 Markdown を生成する。"""
    out: list[str] = []
    out.append(f"# 音楽分析: {track.file_path.name}")
    out.append("")
    out.append(f"_ファイル: `{track.file_path}`_")
    out.append("")
    out.extend(_format_skeleton(track))
    out.extend(_format_harmony(track))
    out.extend(_format_rhythm(track))
    out.extend(_format_genre(track))
    out.extend(_format_distinctive(track))
    out.extend(_format_timeline(track))
    out.extend(_format_technical(track))
    return "\n".join(out)


def write_narrative(track: TrackAnalysis, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_narrative(track), encoding="utf-8")
