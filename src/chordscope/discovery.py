"""音声ファイル探索ユーティリティ。

CLI 引数で指定されたルート、設定ファイル discovery.roots を再帰的にスキャンし、
対応拡張子のファイルを列挙する。シンボリックリンクは追跡しない (循環ループ防止)。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from chordscope.config import DEFAULT_EXTENSIONS


def _normalize_extensions(extensions: Iterable[str]) -> set[str]:
    return {e.lower().lstrip(".") for e in extensions}


def discover_audio_files(
    roots: Iterable[Path],
    *,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    follow_symlinks: bool = False,
) -> list[Path]:
    """指定ルートを再帰的に走査し、対応拡張子の音声ファイルをソート済みで返す。

    - 同じファイルを複数のルートが指していても重複排除する。
    - 存在しないルートは黙って無視する (CLI 側で警告を出すこと)。
    """
    extset = _normalize_extensions(extensions)
    seen: set[Path] = set()
    results: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower().lstrip(".") in extset:
                resolved = root.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    results.append(resolved)
            continue
        for path in _walk(root, follow_symlinks=follow_symlinks):
            if not path.is_file():
                continue
            if path.suffix.lower().lstrip(".") not in extset:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append(resolved)
    return sorted(results)


def _walk(root: Path, *, follow_symlinks: bool) -> Iterator[Path]:
    """rglob だと循環シンボリックリンクで詰まるため自前 DFS。"""
    stack: list[Path] = [root]
    visited: set[Path] = set()
    while stack:
        current = stack.pop()
        try:
            real = current.resolve()
        except OSError:
            continue
        if real in visited:
            continue
        visited.add(real)
        if current.is_file():
            yield current
            continue
        try:
            children = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for child in children:
            if child.is_symlink() and not follow_symlinks:
                if child.is_file():
                    yield child
                continue
            stack.append(child)
