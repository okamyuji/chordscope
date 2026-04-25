"""_compat 互換シムの単体テスト。"""

from __future__ import annotations

import collections
import collections.abc

import numpy as np

from chordscope import _compat


def test_collections_aliases_present() -> None:
    _compat.patch_collections()
    assert collections.MutableSequence is collections.abc.MutableSequence  # type: ignore[attr-defined]
    assert collections.MutableMapping is collections.abc.MutableMapping  # type: ignore[attr-defined]
    assert collections.Iterable is collections.abc.Iterable  # type: ignore[attr-defined]
    assert collections.Callable is collections.abc.Callable  # type: ignore[attr-defined]


def test_numpy_aliases_present() -> None:
    """np.float, np.int, np.complex 属性が解決可能 (madmom が要求)。"""
    _compat.patch_numpy()
    for name in ("float", "int", "complex"):
        assert hasattr(np, name), f"np.{name} が解決できません"


def test_apply_all_idempotent() -> None:
    _compat.apply_all()
    _compat.apply_all()
    assert hasattr(collections, "MutableSequence")
    assert hasattr(np, "float")


def test_madmom_imports_after_compat() -> None:
    _compat.apply_all()
    import madmom

    assert madmom.__version__ != ""
