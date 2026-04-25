"""設定ファイルロードテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chordscope.config import AppConfig, load_config


def test_default_when_path_none() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.discovery.roots == []
    assert "mp3" in cfg.discovery.extensions
    assert cfg.analysis.chord_engine == "madmom"


def test_load_minimal_toml(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text(
        """
[discovery]
roots = ["~/Music"]
extensions = ["mp3", "WAV", ".flac"]

[output]
directory = "~/out"
formats = ["json", "markdown"]
plots = false

[analysis]
genre = false
style = ["jazz"]
chord_engine = "librosa-template"
""",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.discovery.roots[0].name == "Music"
    assert cfg.discovery.extensions == ["mp3", "wav", "flac"]
    assert cfg.output.formats == ["json", "markdown"]
    assert cfg.output.plots is False
    assert cfg.analysis.genre is False
    assert cfg.analysis.style == ["jazz"]
    assert cfg.analysis.chord_engine == "librosa-template"


def test_load_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
