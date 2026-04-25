"""Markdown レポート出力。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from chordscope.models import TrackAnalysis


def render_markdown(track: TrackAnalysis) -> str:
    lines: list[str] = []
    lines.append(f"# 分析レポート: {track.file_path.name}")
    lines.append("")
    lines.append(f"- ファイル: `{track.file_path}`")
    lines.append(f"- 長さ: {track.duration_sec:.2f} 秒")
    lines.append(f"- サンプルレート: {track.sample_rate} Hz")
    lines.append("")
    lines.append("## 概要")
    lines.append("")
    lines.append("| 項目 | 値 | 詳細 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Tempo | {track.tempo.bpm:.2f} BPM | candidates={track.tempo.bpm_candidates}, conf={track.tempo.confidence} |"
    )
    lines.append(
        f"| Meter | {track.meter.numerator}/{track.meter.denominator} | conf={track.meter.confidence} |"
    )
    lines.append(
        f"| Key | {track.key.tonic} {track.key.mode} | corr={track.key.correlation}, conf={track.key.confidence}, 2nd={track.key.second_best} |"
    )
    if track.genre is not None:
        lines.append(
            f"| Genre top | {track.genre.top.label} ({track.genre.top.score:.2%}) | model={track.genre.model_id} |"
        )
    lines.append("")
    if track.genre is not None and track.genre.distribution:
        lines.append("## ジャンル分布 (top 10)")
        lines.append("")
        lines.append("| ラベル | スコア |")
        lines.append("|---|---:|")
        for g in track.genre.distribution[:10]:
            lines.append(f"| {g.label} | {g.score:.2%} |")
        lines.append("")
    lines.append(
        f"## コード進行 (engine={track.chords.method}, total={len(track.chords.segments)})"
    )
    lines.append("")
    lines.append("| start (s) | end (s) | chord |")
    lines.append("|---:|---:|---|")
    for seg in track.chords.segments:
        lines.append(f"| {seg.start:.2f} | {seg.end:.2f} | `{seg.label}` |")
    lines.append("")
    if track.chords.unique_chords:
        lines.append("### ユニークコード")
        lines.append("")
        lines.append(", ".join(f"`{c}`" for c in track.chords.unique_chords))
        lines.append("")
    if track.harmony.roman_numerals:
        lines.append("## 音楽理論分析")
        lines.append("")
        lines.append("### ローマ数字 (先頭 32)")
        lines.append("")
        lines.append(" ".join(track.harmony.roman_numerals[:32]))
        lines.append("")
        if track.harmony.chord_categories:
            lines.append("### 機能分類")
            lines.append("")
            lines.append("| 機能 | 出現数 |")
            lines.append("|---|---:|")
            for k, v in track.harmony.chord_categories.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        if track.harmony.cadences:
            lines.append("### カデンツ")
            lines.append("")
            for c in track.harmony.cadences:
                lines.append(f"- {c}")
            lines.append("")
        if track.harmony.modulations:
            lines.append("### 転調候補")
            lines.append("")
            for m in track.harmony.modulations:
                lines.append(f"- {m}")
            lines.append("")
    if track.style_notes:
        lines.append("## ジャンル別観点分析")
        lines.append("")
        for sn in track.style_notes:
            lines.append(f"### {sn.style}")
            lines.append("")
            for f in sn.findings:
                lines.append(f"- {f}")
            if sn.metrics:
                lines.append("")
                lines.append("metrics: " + ", ".join(f"{k}={v}" for k, v in sn.metrics.items()))
            lines.append("")
    if track.plot_paths:
        lines.append("## グラフ画像")
        lines.append("")
        for kind, path in track.plot_paths.items():
            # 日本語/空白を含むファイル名は URL エンコードしないと一部 Markdown レンダラで表示されない
            href = quote(path.name)
            lines.append(f"### {kind}")
            lines.append("")
            lines.append(f"![{kind}]({href})")
            lines.append("")
    return "\n".join(lines)


def write_markdown(track: TrackAnalysis, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(track), encoding="utf-8")
