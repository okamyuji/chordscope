"""madmom と numpy/Python の deprecated API ギャップを埋める互換シム。

madmom 0.16.1 (最終リリース 2018-11) は古い API に依存しており、
Python 3.10+ や NumPy 1.24+ では import 時にクラッシュする。
本モジュールは `chordscope` パッケージ初期化の最初で読み込まれ、
不在エイリアスのみを補完する (既存属性は変更しない)。
"""

from __future__ import annotations

import collections
import collections.abc
import warnings


def patch_collections() -> None:
    """collections モジュールに MutableSequence 等のエイリアスを補完する。

    Python 3.10 以降、これらは `collections.abc` のみに存在する。
    """
    aliases = (
        "MutableSequence",
        "MutableMapping",
        "MutableSet",
        "Mapping",
        "Sequence",
        "Set",
        "Iterable",
        "Iterator",
        "Container",
        "Hashable",
        "Sized",
        "Callable",
    )
    for name in aliases:
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def patch_numpy() -> None:
    """numpy に np.float, np.int, np.complex のエイリアスを補完する。

    NumPy 1.24 で削除されたが madmom が SEGMENT_DTYPE 等で参照する。
    `np.bool` / `np.object` は NumPy が独自の再導入予定を持ち、`hasattr` 自体が
    FutureWarning を発するため対象外とする。
    """
    import numpy as np

    aliases: dict[str, type] = {
        "float": float,
        "int": int,
        "complex": complex,
    }
    for name, builtin in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)


def patch_madmom_downbeat_inhomogeneous() -> None:
    """madmom DBNDownBeatTrackingProcessor.process を numpy >=1.24 互換に置換する。

    madmom 0.16.1 の元実装は `np.argmax(np.asarray(results)[:, 1])` を含み、
    `results = [(path_array, score), ...]` の最終次元が異種長のため、
    NumPy 1.24 以降の inhomogeneous-shape 厳格化で ValueError となる。
    本パッチは元の process を 1 行だけ書き換えた完全互換版に差し替える。
    """
    import itertools as it

    import numpy as _np
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        _process_dbn,
    )

    def patched_process(self, activations, **_kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202, ANN003
        first = 0
        if self.threshold:
            idx = _np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, int(_np.min(idx)))
                last = min(len(activations), int(_np.max(idx)) + 1)
            else:
                last = first
            activations = activations[first:last]
        if not activations.any():
            return _np.empty((0, 2))
        results = list(self.map(_process_dbn, zip(self.hmms, it.repeat(activations))))
        # === BUG FIX: 元コードの `np.argmax(np.asarray(results)[:, 1])` を置換 ===
        scores = [float(r[1]) for r in results]
        best = int(_np.argmax(scores))
        path, _ = results[best]
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = _np.empty(0, dtype=int)
            beat_range = om.pointers[path] >= 1
            idx = _np.nonzero(_np.diff(beat_range.astype(int)))[0] + 1
            if beat_range[0]:
                idx = _np.r_[0, idx]
            if beat_range[-1]:
                idx = _np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = int(_np.argmax(activations[left:right])) // 2 + left
                    beats = _np.hstack((beats, peak))
        else:
            beats = _np.nonzero(_np.diff(beat_numbers))[0] + 1
        return _np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T

    DBNDownBeatTrackingProcessor.process = patched_process


def silence_known_warnings() -> None:
    """madmom 由来で不可避な DeprecationWarning を抑制する。"""
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Implementing implicit namespace packages.*",
        category=DeprecationWarning,
    )


def apply_all() -> None:
    """全ての互換パッチを冪等に適用する。"""
    import contextlib

    patch_collections()
    patch_numpy()
    silence_known_warnings()
    with contextlib.suppress(Exception):
        # madmom 未インストール時は何もしない
        patch_madmom_downbeat_inhomogeneous()


# モジュール読み込み時に自動適用する
apply_all()
