"""音声ファイル探索テスト。"""

from __future__ import annotations

from pathlib import Path

from chordscope.discovery import discover_audio_files


def _touch(p: Path, content: bytes = b"") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_recursive_discovery_filters_by_extension(tmp_path: Path) -> None:
    _touch(tmp_path / "a.mp3")
    _touch(tmp_path / "sub" / "b.WAV")
    _touch(tmp_path / "sub" / "deep" / "c.flac")
    _touch(tmp_path / "ignore.txt")
    _touch(tmp_path / "img.png")
    found = discover_audio_files([tmp_path])
    names = [f.name for f in found]
    assert sorted(names) == ["a.mp3", "b.WAV", "c.flac"]


def test_dedupes_when_multiple_roots_overlap(tmp_path: Path) -> None:
    _touch(tmp_path / "x.mp3")
    found = discover_audio_files([tmp_path, tmp_path / "x.mp3", tmp_path])
    assert len(found) == 1


def test_unknown_extension_excluded(tmp_path: Path) -> None:
    _touch(tmp_path / "bad.exe")
    found = discover_audio_files([tmp_path])
    assert found == []


def test_missing_root_silently_ignored(tmp_path: Path) -> None:
    found = discover_audio_files([tmp_path / "nope"])
    assert found == []


def test_extension_filter_case_insensitive(tmp_path: Path) -> None:
    _touch(tmp_path / "X.MP3")
    found = discover_audio_files([tmp_path], extensions=["mp3"])
    assert len(found) == 1
