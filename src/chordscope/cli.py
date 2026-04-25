"""Typer ベースの CLI。"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from chordscope.analyzers.genre import GenreClassifier
from chordscope.config import AppConfig, load_config
from chordscope.discovery import discover_audio_files
from chordscope.pipeline import AnalysisOptions, analyze_file, options_from_config
from chordscope.reporting.console import render_track
from chordscope.reporting.json_report import write_json
from chordscope.reporting.markdown import write_markdown

app = typer.Typer(
    name="chordscope",
    help="汎用音楽分析 CLI: BPM・拍子・コード・調性・音楽理論・ジャンル",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()


def _split_csv(values: list[str] | None) -> list[str] | None:
    """typer の --opt a,b と --opt a --opt b の両方を許容するコールバック。"""
    if not values:
        return values
    out: list[str] = []
    for v in values:
        out.extend(s.strip() for s in v.split(",") if s.strip())
    return out


@app.command("analyze")
def analyze(
    paths: list[Path] = typer.Argument(
        None,
        help="分析対象のファイルまたはディレクトリ (複数指定可)。指定がなければ設定ファイルの roots を使用。",
        exists=False,
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="TOML 設定ファイルへのパス",
        exists=False,
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="出力ディレクトリ (デフォルト: 設定ファイルの output.directory)",
    ),
    chord_engine: str | None = typer.Option(
        None, help="madmom か librosa-template (設定ファイル値を上書き)"
    ),
    no_genre: bool = typer.Option(False, help="ジャンル分類をスキップ"),
    no_plots: bool = typer.Option(False, help="グラフ画像生成をスキップ"),
    formats: list[str] | None = typer.Option(
        None,
        callback=_split_csv,
        help="出力フォーマット: console, json, markdown。"
        " カンマ区切り (例: --formats json,markdown) または複数指定 (--formats json --formats markdown) のどちらでも可。",
    ),
) -> None:
    """指定されたパスを再帰的に分析する。"""
    cfg: AppConfig = load_config(config)
    roots = list(paths) if paths else list(cfg.discovery.roots)
    if not roots:
        _console.print(
            "[red]エラー:[/red] 分析対象が指定されていません。"
            " 引数または --config で discovery.roots を指定してください。"
        )
        raise typer.Exit(code=2)
    files = discover_audio_files(roots, extensions=cfg.discovery.extensions)
    if not files:
        _console.print("[yellow]対応する音声ファイルが見つかりません。[/yellow]")
        raise typer.Exit(code=1)
    _console.print(f"[green]{len(files)} 件のファイルを発見:[/green]")
    for f in files:
        _console.print(f"  - {f}")

    # 出力先決定
    out_dir = output_dir or cfg.output.directory
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    formats_set = set(formats) if formats else set(cfg.output.formats)
    plots = (not no_plots) and cfg.output.plots
    enable_genre = (not no_genre) and cfg.analysis.genre

    # オプション組み立て
    opts = options_from_config(cfg.analysis, plot_dir=out_dir / "plots", plots=plots)
    opts = AnalysisOptions(
        chord_engine=chord_engine or opts.chord_engine,
        enable_genre=enable_genre,
        enabled_styles=opts.enabled_styles,
        plots=plots,
        plot_dir=opts.plot_dir,
        genre_classifier=GenreClassifier() if enable_genre else None,
    )

    for path in files:
        _console.rule(f"Analyzing: {path.name}")
        try:
            track = analyze_file(path, opts)
        except Exception as e:
            _console.print(f"[red]Failed to analyze {path}: {e}[/red]")
            continue
        if "console" in formats_set:
            render_track(track, _console)
        stem = path.stem
        if "json" in formats_set:
            write_json(track, out_dir / f"{stem}.json")
        if "markdown" in formats_set:
            write_markdown(track, out_dir / f"{stem}.md")
        _console.print(f"[green]Saved outputs to {out_dir}[/green]")


@app.command("init-config")
def init_config(
    out: Path = typer.Argument(Path("./chordscope.toml"), help="出力先 TOML パス"),
) -> None:
    """雛形となる TOML 設定ファイルを書き出す。"""
    if out.exists():
        _console.print(f"[red]既に存在します: {out}[/red]")
        raise typer.Exit(code=1)
    out.write_text(_DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    _console.print(f"[green]Wrote template to {out}[/green]")


_DEFAULT_CONFIG_TEMPLATE = """# chordscope 設定ファイル
[discovery]
roots = ["~/Music"]
extensions = ["mp3", "wav", "flac", "ogg", "aac", "aiff", "mp4", "m4a"]

[output]
directory = "~/music-analysis-out"
formats = ["console", "json", "markdown"]
plots = true

[analysis]
genre = true
style = ["jazz", "classic", "jpop", "rock"]
chord_engine = "madmom"  # madmom | librosa-template
"""


def main() -> None:
    """エントリポイント (`uv run chordscope`)。"""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
