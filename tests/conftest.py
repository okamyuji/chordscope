"""共有 pytest フィクスチャ。

音源は 2 種類で構成される (合成音声は使わない)。

1. PD 音源: Wikimedia Commons から取得し `~/.cache/chordscope-tests/` にキャッシュする。
   100 年以上前の録音で PD が確実なものを採用する。
2. ユーザー任意音源: 環境変数 `CHORDSCOPE_TEST_USER_AUDIO_DIR` (既定:
   `<repo>/tests/fixtures/user-audio/`) を再帰スキャンし、対応拡張子の 1 件目を選ぶ。

オフライン時のダウンロード失敗および音源不在は当該テストを `pytest.skip` する。
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from chordscope.audio import AudioBuffer, load_audio
from chordscope.config import DEFAULT_EXTENSIONS
from chordscope.discovery import discover_audio_files

DEFAULT_CACHE = Path.home() / ".cache" / "chordscope-tests"
USER_AGENT = "chordscope-test/0.1 (https://github.com/okamyuji/chordscope)"

# プロジェクトルート (= conftest.py の 1 階層上)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ユーザーが任意の音源を投入する公式ディレクトリ。空でも skip するので存在するだけで OK。
DEFAULT_USER_AUDIO_DIR = PROJECT_ROOT / "tests" / "fixtures" / "user-audio"


@dataclass(frozen=True)
class PdAudio:
    """パブリックドメイン音源のメタデータ。"""

    name: str  # ファイルキャッシュ名 (拡張子付き)
    url: str  # 直接ダウンロード URL (Wikimedia Commons)
    sha256: str | None  # 期待 SHA-256 (None なら検証スキップ。初回 DL 後に固定する)
    description: str
    expected_genre_keywords: tuple[str, ...]  # 想定ジャンル系統 (緩いマッチ用)
    expected_key: tuple[str, str] | None  # (tonic, mode) 既知ならテスト
    expected_bpm_range: tuple[float, float] | None


# 全て 100 年以上前の録音または PD 確定音源。Wikimedia Commons から取得。
PD_AUDIO_FIXTURES: tuple[PdAudio, ...] = (
    PdAudio(
        name="encina_fata_la_parte.ogg",
        url="https://upload.wikimedia.org/wikipedia/commons/e/e4/Juan_del_Encina_--_Fata_la_parte.ogg",
        sha256=None,
        description="Juan del Encina (1469-1529): Fata la parte. ルネサンス声楽。",
        expected_genre_keywords=("Classical", "Vocal", "Folk", "Choir"),
        expected_key=None,
        expected_bpm_range=None,
    ),
    PdAudio(
        name="joplin_maple_leaf_rag_1916.ogg",
        url="https://upload.wikimedia.org/wikipedia/commons/e/e9/Maple_Leaf_Rag_-_played_by_Scott_Joplin_1916_sample.ogg",
        sha256=None,
        description="Scott Joplin 自身による Maple Leaf Rag (1916 録音、131 秒抜粋)。ラグタイム。",
        expected_genre_keywords=("Jazz", "Ragtime", "Piano", "Rock", "Blues"),
        expected_key=(
            "A",
            "major",
        ),  # Maple Leaf Rag は A♭ メジャー (理論上)。録音はピッチ揺れあり。
        expected_bpm_range=(80.0, 200.0),
    ),
    PdAudio(
        name="joplin_maple_leaf_rag_full.ogg",
        url="https://upload.wikimedia.org/wikipedia/commons/0/09/Scott_Joplin_-_Maple_Leaf_Rag.ogg",
        sha256=None,
        description="Scott Joplin Maple Leaf Rag (フル音源)。ラグタイム。",
        expected_genre_keywords=("Jazz", "Ragtime", "Piano", "Blues"),
        expected_key=None,
        expected_bpm_range=(80.0, 200.0),
    ),
    PdAudio(
        name="afghanistan_1920_foxtrot.mp3",
        url="https://upload.wikimedia.org/wikipedia/commons/7/77/%22Afghanistan%22_fox-trot_by_Charles_A._Prince%27s_Dance_Orchestra_%281920%29.mp3",
        sha256=None,
        description="Charles A. Prince's Dance Orchestra: Afghanistan (1920)。ダンスオーケストラ初期ジャズ。",
        expected_genre_keywords=("Jazz", "Swing", "Big band", "Orchestra"),
        expected_key=None,
        expected_bpm_range=(70.0, 200.0),
    ),
    PdAudio(
        name="sousa_stars_and_stripes_forever.ogg",
        url="https://upload.wikimedia.org/wikipedia/commons/9/90/Sousa%27s_Band_-_Stars_and_Stripes_Forever.ogg",
        sha256=None,
        description=(
            "Sousa's Band: The Stars and Stripes Forever (1909 録音、約 4 分)。"
            "Sousa 行進曲。trio で完全 4 度上に転調する典型例。"
        ),
        expected_genre_keywords=("March", "Brass", "Wind", "Orchestra", "Classical"),
        expected_key=None,
        expected_bpm_range=(80.0, 220.0),
    ),
)


def _cache_dir() -> Path:
    p = Path(os.environ.get("CHORDSCOPE_TEST_CACHE", str(DEFAULT_CACHE))).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _ensure(audio: PdAudio, cache: Path) -> Path:
    """キャッシュにファイルがなければダウンロードする。失敗時は pytest.skip。"""
    target = cache / audio.name
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        _download(audio.url, target)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        if target.exists():
            target.unlink(missing_ok=True)
        pytest.skip(f"PD音源ダウンロード失敗 ({audio.name}): {e}")
    if audio.sha256 is not None:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != audio.sha256:
            target.unlink(missing_ok=True)
            pytest.skip(
                f"PD音源 SHA-256 不一致 ({audio.name}): expected {audio.sha256}, got {digest}"
            )
    return target


# ---------- Session-scoped fixtures ----------


@pytest.fixture(scope="session")
def pd_cache_dir() -> Path:
    return _cache_dir()


@pytest.fixture(scope="session")
def pd_audio_dir(pd_cache_dir: Path) -> Path:
    """全 PD 音源を取得してディレクトリを返す。テスト探索の起点として使う。"""
    for audio in PD_AUDIO_FIXTURES:
        _ensure(audio, pd_cache_dir)
    return pd_cache_dir


@pytest.fixture(scope="session")
def pd_encina_path(pd_cache_dir: Path) -> Path:
    return _ensure(PD_AUDIO_FIXTURES[0], pd_cache_dir)


@pytest.fixture(scope="session")
def pd_joplin_sample_path(pd_cache_dir: Path) -> Path:
    return _ensure(PD_AUDIO_FIXTURES[1], pd_cache_dir)


@pytest.fixture(scope="session")
def pd_joplin_full_path(pd_cache_dir: Path) -> Path:
    return _ensure(PD_AUDIO_FIXTURES[2], pd_cache_dir)


@pytest.fixture(scope="session")
def pd_afghanistan_path(pd_cache_dir: Path) -> Path:
    return _ensure(PD_AUDIO_FIXTURES[3], pd_cache_dir)


@pytest.fixture(scope="session")
def pd_sousa_path(pd_cache_dir: Path) -> Path:
    return _ensure(PD_AUDIO_FIXTURES[4], pd_cache_dir)


@pytest.fixture(scope="session")
def pd_encina_buffer(pd_encina_path: Path) -> AudioBuffer:
    return load_audio(pd_encina_path)


@pytest.fixture(scope="session")
def pd_joplin_sample_buffer(pd_joplin_sample_path: Path) -> AudioBuffer:
    return load_audio(pd_joplin_sample_path)


@pytest.fixture(scope="session")
def pd_joplin_full_buffer(pd_joplin_full_path: Path) -> AudioBuffer:
    return load_audio(pd_joplin_full_path)


@pytest.fixture(scope="session")
def pd_afghanistan_buffer(pd_afghanistan_path: Path) -> AudioBuffer:
    return load_audio(pd_afghanistan_path)


@pytest.fixture(scope="session")
def pd_sousa_buffer(pd_sousa_path: Path) -> AudioBuffer:
    return load_audio(pd_sousa_path)


@pytest.fixture(scope="session")
def pd_all_buffers(
    pd_encina_buffer: AudioBuffer,
    pd_joplin_sample_buffer: AudioBuffer,
    pd_joplin_full_buffer: AudioBuffer,
    pd_afghanistan_buffer: AudioBuffer,
    pd_sousa_buffer: AudioBuffer,
) -> dict[str, AudioBuffer]:
    """5 PD 音源を識別子→AudioBuffer のマップで取得 (AB テスト用)。"""
    return {
        "encina_renaissance": pd_encina_buffer,
        "joplin_sample_ragtime": pd_joplin_sample_buffer,
        "joplin_full_ragtime": pd_joplin_full_buffer,
        "afghanistan_1920_jazz": pd_afghanistan_buffer,
        "sousa_march": pd_sousa_buffer,
    }


def _resolve_user_audio_dir() -> Path:
    """環境変数で指定されたディレクトリを返す。未指定時はリポジトリ既定。"""
    raw = os.environ.get("CHORDSCOPE_TEST_USER_AUDIO_DIR")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_USER_AUDIO_DIR


@pytest.fixture(scope="session")
def user_audio_path() -> Path:
    """対応拡張子のファイル 1 件を任意ディレクトリから返す (ファイル名は不定)。

    `CHORDSCOPE_TEST_USER_AUDIO_DIR` を再帰スキャンする。未指定時は
    `tests/fixtures/user-audio/` が対象。何も見つからない場合は skip。
    """
    base = _resolve_user_audio_dir()
    if not base.exists():
        pytest.skip(f"ユーザー音源ディレクトリが存在しません: {base}")
    candidates = discover_audio_files([base], extensions=DEFAULT_EXTENSIONS)
    if not candidates:
        pytest.skip(
            f"ユーザー音源が見つかりません ({base})。"
            f" 環境変数 CHORDSCOPE_TEST_USER_AUDIO_DIR で別ディレクトリを指定可能。"
        )
    return candidates[0]


@pytest.fixture(scope="session")
def user_audio_buffer(user_audio_path: Path) -> AudioBuffer:
    return load_audio(user_audio_path)


@pytest.fixture(autouse=True)
def _stable_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
