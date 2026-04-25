"""TOML 設定ファイルロード。

CLI 引数を補完する。設定ファイル例:

```toml
[discovery]
roots = ["~/Music", "/Volumes/External/audio"]
extensions = ["mp3", "wav", "flac", "ogg", "aac", "aiff", "mp4"]

[output]
directory = "~/music-analysis-out"
formats = ["console", "json", "markdown"]
plots = true

[analysis]
genre = true
style = ["jazz", "classic", "jpop", "rock"]
chord_engine = "madmom"  # "madmom" or "librosa-template"
```
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_EXTENSIONS = ("mp3", "wav", "flac", "ogg", "aac", "aiff", "mp4", "m4a")
ChordEngine = Literal["madmom", "librosa-template"]
ReportFormat = Literal["console", "json", "markdown", "narrative"]
# style は固定 Literal を撤廃。"auto" (AST top-K 自動採用) または任意ジャンル名のリスト。
StyleChoice = str
DEFAULT_STYLE_TOP_K = 5


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    roots: list[Path] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_EXTENSIONS))

    @field_validator("roots", mode="before")
    @classmethod
    def _expand(cls, v: list[str | Path]) -> list[Path]:
        return [Path(str(p)).expanduser() for p in v]

    @field_validator("extensions")
    @classmethod
    def _normalize_ext(cls, v: list[str]) -> list[str]:
        return [e.lower().lstrip(".") for e in v]


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    directory: Path = Field(default_factory=lambda: Path("./music-analysis-out"))
    formats: list[ReportFormat] = Field(
        default_factory=lambda: ["console", "json", "markdown"]  # type: ignore[arg-type]
    )
    plots: bool = True

    @field_validator("directory", mode="before")
    @classmethod
    def _expand_dir(cls, v: str | Path) -> Path:
        return Path(str(v)).expanduser()


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    genre: bool = True
    # "auto" は AST top-K を style notes に動的展開する。任意ジャンル名のリストでも可。
    style: list[str] = Field(default_factory=lambda: ["auto"])
    style_top_k: int = Field(default=DEFAULT_STYLE_TOP_K, ge=1, le=50)
    chord_engine: ChordEngine = "madmom"


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def load_config(path: Path | None) -> AppConfig:
    """TOML 設定ファイルをロードして AppConfig を返す。

    path=None または存在しないパスを与えた場合はすべてデフォルト値の AppConfig。
    """
    if path is None:
        return AppConfig()
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return AppConfig.model_validate(data)
