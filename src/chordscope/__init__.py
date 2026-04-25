"""chordscope: 汎用音楽分析ライブラリ。

BPM・拍子・コード・調性・音楽理論・ジャンルを一括分析する CLI/ライブラリ。
パッケージ初期化時に `_compat` が deprecated API のエイリアスを補う。
"""

from chordscope import _compat as _compat  # 先頭で deprecated-API パッチを適用

__version__ = "0.1.0"

__all__ = ["__version__"]
