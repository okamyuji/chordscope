"""rich を用いたコンソールレポート出力。"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chordscope.models import TrackAnalysis


def render_track(track: TrackAnalysis, console: Console | None = None) -> None:
    """1 曲分の分析結果をコンソール表示。"""
    c = console or Console()
    c.rule(f"[bold cyan]{track.file_path.name}")
    meta = Table.grid(padding=(0, 2))
    meta.add_row("Path", str(track.file_path))
    meta.add_row("Duration", f"{track.duration_sec:.1f} sec")
    meta.add_row("Sample rate", f"{track.sample_rate} Hz")
    c.print(meta)

    overview = Table(title="Overview", show_header=True, header_style="bold magenta")
    overview.add_column("Item")
    overview.add_column("Value")
    overview.add_column("Detail")
    overview.add_row(
        "Tempo",
        f"{track.tempo.bpm:.2f} BPM",
        f"candidates={track.tempo.bpm_candidates} confidence={track.tempo.confidence}",
    )
    overview.add_row(
        "Meter",
        f"{track.meter.numerator}/{track.meter.denominator}",
        f"confidence={track.meter.confidence}",
    )
    overview.add_row(
        "Key",
        f"{track.key.tonic} {track.key.mode}",
        f"corr={track.key.correlation} confidence={track.key.confidence}, 2nd={track.key.second_best}",
    )
    if track.genre is not None:
        top = track.genre.top
        overview.add_row(
            "Genre (top)", f"{top.label} ({top.score:.2%})", f"model={track.genre.model_id}"
        )
    c.print(overview)

    if track.genre is not None and len(track.genre.distribution) > 1:
        gtable = Table(title="Genre distribution (top 10)", header_style="bold green")
        gtable.add_column("Label")
        gtable.add_column("Score", justify="right")
        for g in track.genre.distribution[:10]:
            gtable.add_row(g.label, f"{g.score:.2%}")
        c.print(gtable)

    chord_table = Table(
        title=f"Chords (first 20 / total {len(track.chords.segments)}, engine={track.chords.method})",
        header_style="bold yellow",
    )
    chord_table.add_column("Start (s)", justify="right")
    chord_table.add_column("End (s)", justify="right")
    chord_table.add_column("Chord")
    for seg in track.chords.segments[:20]:
        chord_table.add_row(f"{seg.start:.2f}", f"{seg.end:.2f}", seg.label)
    c.print(chord_table)

    if track.harmony.roman_numerals:
        rn_summary = " ".join(track.harmony.roman_numerals[:24])
        c.print(Panel(rn_summary, title="Roman numerals (first 24)"))
    if track.harmony.cadences:
        c.print(Panel("\n".join(track.harmony.cadences), title="Cadences"))
    if track.harmony.modulations:
        c.print(Panel("\n".join(track.harmony.modulations), title="Modulation candidates"))
    if track.harmony.chord_categories:
        cat_text = "  ".join(f"{k}={v}" for k, v in track.harmony.chord_categories.items())
        c.print(Panel(cat_text, title="Chord function counts"))

    for sn in track.style_notes:
        if not sn.findings and not sn.metrics:
            continue
        body_lines = list(sn.findings)
        if sn.metrics:
            body_lines.append("")
            body_lines.append("metrics: " + ", ".join(f"{k}={v}" for k, v in sn.metrics.items()))
        c.print(Panel("\n".join(body_lines), title=f"Style notes [{sn.style}]"))

    if track.plot_paths:
        plots_table = Table(title="Plots", header_style="bold blue")
        plots_table.add_column("Kind")
        plots_table.add_column("Path")
        for kind, path in track.plot_paths.items():
            plots_table.add_row(kind, str(path))
        c.print(plots_table)
