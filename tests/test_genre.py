"""ジャンル分類器のAB テスト。

AST AudioSet モデルは初回ロード時に大型ファイル (~330MB) をダウンロードするため、
slow + integration マーク。CI で省きたい場合は `pytest -m "not slow"` で除外。
"""

from __future__ import annotations

import pytest

from chordscope.analyzers.genre import GenreClassifier


@pytest.mark.integration
@pytest.mark.slow
def test_genre_classifier_returns_distribution(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    clf = GenreClassifier()
    result = clf.predict(pd_joplin_sample_buffer)
    assert result.top.score > 0.0
    assert len(result.distribution) >= 5
    # モデル ID は AST AudioSet
    assert "ast" in result.model_id.lower()


@pytest.mark.integration
@pytest.mark.slow
def test_genre_for_renaissance_in_classical_keywords(pd_encina_buffer) -> None:  # type: ignore[no-untyped-def]
    """Encina の声楽 → Top10 に Classical/Vocal/Choir/Music いずれかが含まれる。"""
    clf = GenreClassifier()
    result = clf.predict(pd_encina_buffer)
    top_labels = [g.label.lower() for g in result.distribution[:10]]
    expected_keywords = ["classical", "vocal", "choir", "music", "song", "chant", "opera"]
    assert any(any(kw in lbl for kw in expected_keywords) for lbl in top_labels), (
        f"想定キーワードを top10 に含まず: {top_labels}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_genre_for_ragtime_in_jazz_or_piano_keywords(pd_joplin_sample_buffer) -> None:  # type: ignore[no-untyped-def]
    """Maple Leaf Rag → Top10 に Jazz/Piano/Ragtime/Music 系統が含まれる。"""
    clf = GenreClassifier()
    result = clf.predict(pd_joplin_sample_buffer)
    top_labels = [g.label.lower() for g in result.distribution[:10]]
    expected_keywords = ["jazz", "piano", "ragtime", "music", "blues", "rock"]
    assert any(any(kw in lbl for kw in expected_keywords) for lbl in top_labels), (
        f"想定キーワードを top10 に含まず: {top_labels}"
    )
