"""1 曲分の全分析を束ねるオーケストレータ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chordscope.analyzers.beats import estimate_beats_and_meter
from chordscope.analyzers.chords import recognize_chords
from chordscope.analyzers.genre import GenreClassifier
from chordscope.analyzers.key import estimate_key
from chordscope.analyzers.modulation import detect_modulation
from chordscope.analyzers.style import analyze_styles
from chordscope.analyzers.tempo import estimate_tempo
from chordscope.analyzers.tempo_curve import analyze_tempo_curve
from chordscope.analyzers.theory import analyze_harmony
from chordscope.audio import AudioBuffer, load_audio
from chordscope.config import AnalysisConfig
from chordscope.models import TrackAnalysis
from chordscope.reporting.plots import render_all_plots


@dataclass(frozen=True)
class AnalysisOptions:
    """ランタイム指定。"""

    chord_engine: str = "madmom"
    enable_genre: bool = True
    # "auto" は AST top-K ジャンルを動的に展開する。固定リストとの混在も可。
    enabled_styles: tuple[str, ...] = ("auto",)
    style_top_k: int = 5
    plots: bool = True
    plot_dir: Path | None = None
    genre_classifier: GenreClassifier | None = None  # 再利用したい場合に渡す


def options_from_config(
    config: AnalysisConfig, *, plot_dir: Path | None, plots: bool
) -> AnalysisOptions:
    return AnalysisOptions(
        chord_engine=config.chord_engine,
        enable_genre=config.genre,
        enabled_styles=tuple(config.style),
        style_top_k=config.style_top_k,
        plots=plots,
        plot_dir=plot_dir,
    )


def analyze_file(path: Path, opts: AnalysisOptions) -> TrackAnalysis:
    """1 ファイルを分析し TrackAnalysis を返す。"""
    buffer: AudioBuffer = load_audio(path)
    tempo = estimate_tempo(buffer)
    beats, meter = estimate_beats_and_meter(buffer)
    key = estimate_key(buffer)
    modulation = detect_modulation(buffer, key)
    tempo_curve = analyze_tempo_curve(beats, tempo)
    chords = recognize_chords(buffer, engine=opts.chord_engine)  # type: ignore[arg-type]
    harmony = analyze_harmony(chords, key)
    genre = None
    if opts.enable_genre:
        clf = opts.genre_classifier or GenreClassifier()
        genre = clf.predict(buffer)
    styles = analyze_styles(
        buffer=buffer,
        tempo=tempo,
        beats=beats,
        meter=meter,
        key=key,
        chords=chords,
        harmony=harmony,
        enabled=list(opts.enabled_styles),
        genre=genre,
        top_k=opts.style_top_k,
    )
    plot_paths: dict[str, Path] = {}
    if opts.plots and opts.plot_dir is not None:
        plot_paths = render_all_plots(
            buffer,
            beats=beats,
            chords=chords,
            tempo=tempo,
            tempo_curve=tempo_curve,
            modulation=modulation,
            out_dir=opts.plot_dir,
        )
    return TrackAnalysis(
        file_path=path,
        duration_sec=buffer.duration,
        sample_rate=buffer.sample_rate,
        tempo=tempo,
        beats=beats,
        meter=meter,
        key=key,
        chords=chords,
        harmony=harmony,
        modulation=modulation,
        tempo_curve=tempo_curve,
        genre=genre,
        style_notes=styles,
        plot_paths=plot_paths,
    )
