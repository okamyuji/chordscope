# chordscope

汎用音楽分析 CLI。1 曲または複数曲を再帰的に分析し、BPM・拍子・コード進行・調性・音楽理論的な機能解析・ジャンル分布・ジャンル別観点を 1 コマンドで生成する。

- **言語/ランタイム**: Python 3.11.x (uv 管理)
- **インターフェース**: CLI (`chordscope analyze ...`)
- **出力**: コンソール (rich) / JSON / Markdown レポート / matplotlib PNG グラフ
- **設計方針**: 速度よりも分析精度を優先する

---

## 主な機能

| 機能 | 実装 |
|---|---|
| **BPM 推定** | madmom (RNN+DBN) + librosa beat_track + librosa onset autocorr の 3 手法を集約。各候補を [60, 180) BPM レンジへオクターブ正規化 → 中央値を採用 |
| **ビート/ダウンビート** | madmom RNN+DBN downbeat tracking。numpy 1.24+ で発生する inhomogeneous-shape バグを `_compat.py` で完全互換 monkey-patch |
| **拍子 (numerator)** | 2/3/4/5/6/7 の各候補で独立 DBN を実行し、信頼度 92% 以内の中で最小の値を優先 (Occam's razor) |
| **調性 (キー) 推定** | Krumhansl-Schmuckler (Temperley 改良プロファイル) を CQT クロマで適用。次点候補も保持 |
| **コード認識** | madmom Deep Chroma + DCC を主バックエンド。`librosa-template` 副バックエンドは 10 種 × 12 ルート計 120 テンプレート + Viterbi 平滑 |
| **音楽理論分析** | music21 でローマ数字解析、機能分類 (T/SD/D)、カデンツ検出 (Authentic / Plagal / Deceptive / ii→V)、転調候補 |
| **転調検出 (Modulation)** | 16 秒スライディングウィンドウ + 4 秒ホップで Krumhansl-Schmuckler を再適用し、`KeySegment` 列と `KeyChange` (時刻 / from→to / 関係) を出力。フラップ抑制 (multi-pass smoothing + 短セグメントマージ) で旋法音楽でも安定 |
| **テンポ変動 (Tempo Curve)** | `beat_times[]` の中央値間隔から局所 BPM を算出し、global BPM 比 ±5% でセグメント分類 (slow / stable / fast)。線形回帰で trend (stable / accelerando / ritardando / variable) を判定 |
| **ジャンル分類** | HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593` (AST + AudioSet 527 クラス)。AudioSet 出力を 163 ジャンルカタログで正規化し、umbrella/mood/機能ラベル除外後に **TOP-5 + family 多様性** (1 family 最大 2 件) で出力 |
| **ジャンル別観点** | **動的ディスパッチ**: AST top-K ラベルに応じて 80+ キーのヒューリスティックレジストリ (jazz/classic/jpop/rock/blues/hiphop/electronic/country/soul/funk/r&b/metal/punk/reggae/ska/folk/bluegrass/latin/salsa/bossa nova/gospel/ambient/flamenco/world/anime/enka/vocal/soundtrack/musical 等) を起動。未登録ラベルは AST スコアのみ報告 |
| **ジャンルカタログ** | 163 件 / 14 family (rock / pop / jazz / blues / classical / folk / country / electronic / hiphop / metal / world / latin / religious / soundtrack)。Discogs Styles + AudioSet ontology + 主要音楽辞典から抽出 |
| **可視化** | 波形 + ビート/ダウンビート overlay、メルスペクトログラム、CQT クロマ + コード overlay、テンポグラム + BPM 線 |
| **対応フォーマット** | mp3 / wav / flac / ogg / aac / aiff / mp4 / m4a (再帰スキャン) |
| **ファイル名** | 半角空白・全角空白・日本語・各種 Unicode を含むファイル名を全段で取り扱う |
| **音楽分析文 (narrative)** | 既存の分析データを章立て + 接続詞で「論理フロー型」に組み立てた Markdown を別ファイル出力 (`<stem>_analysis.md`)。LLM 不要、ハルシネーションゼロ |

---

## 動作要件

- macOS / Linux
- Python **3.11.x** (madmom 0.16.1 互換のため。3.12/3.13 は不可)
- mise または uv で Python 管理
- ffmpeg (mp3/mp4/aac/m4a の読み込みに必須)

```bash
# リポジトリを取得 (任意の作業ディレクトリで)
git clone https://github.com/okamyuji/chordscope.git
cd chordscope

# Python 3.11 を mise で取得
mise install python@3.11

# プロジェクト依存解決と仮想環境作成
uv sync
```

uv が `madmom` のビルドに必要な `Cython`, `numpy` を build dep として自動補完する設定 (`[tool.uv].extra-build-dependencies`) を含む。

---

## 使い方

### 単発の分析

```bash
# パスを直接指定 (ファイルでもディレクトリでも可、ディレクトリは再帰)
uv run chordscope analyze ~/Music/album/

# 出力フォーマット指定 (カンマ区切りまたは複数指定どちらも可)
uv run chordscope analyze ~/Music/track.flac --formats console,json,markdown,narrative
uv run chordscope analyze ~/Music/track.flac --formats narrative --formats json

# narrative のみ出力 (論理フロー型分析文)
uv run chordscope analyze ~/Music/track.flac --formats narrative

# 出力先指定
uv run chordscope analyze ~/Music/ -o ./out

# ジャンル分類とグラフ画像をスキップ (高速化)
uv run chordscope analyze ~/Music/ --no-genre --no-plots

# コードエンジン切り替え (madmom が遅い場合の代替)
uv run chordscope analyze ~/Music/ --chord-engine librosa-template
```

### 設定ファイルで運用

```bash
# 雛形を生成
uv run chordscope init-config ./chordscope.toml

# 編集後、--config で読み込み
uv run chordscope analyze --config ./chordscope.toml
```

設定ファイル (`configs/example.toml` を参照):

```toml
[discovery]
roots = ["~/Music", "/Volumes/External/audio"]
extensions = ["mp3", "wav", "flac", "ogg", "aac", "aiff", "mp4", "m4a"]

[output]
directory = "~/music-analysis-out"
formats = ["console", "json", "markdown"]
plots = true

[analysis]
genre = true
# style = ["auto"] は AST top-K のジャンル推論結果に応じて動的に解析対象を選ぶ。
# 任意の固定リスト ["jazz", "blues", "latin"] や両者の混在も可能。
style = ["auto"]
style_top_k = 5  # AST 推論の TOP-K を style notes 対象にする
chord_engine = "madmom"  # madmom | librosa-template
```

引数で指定したパスと設定ファイルの `discovery.roots` の両方を使える。CLI フラグは設定ファイルの値を上書きする。

---

## 出力ファイル

```
<output_dir>/
├── <stem>.json              # 全分析結果 (機械可読)
├── <stem>.md                # Markdown レポート (網羅型・章ごとに全データを羅列)
├── <stem>_analysis.md       # 音楽分析文 (論理フロー型・章立て + 接続詞で読み物に)
└── plots/
    ├── <stem>__waveform_beats.png
    ├── <stem>__mel_spectrogram.png
    ├── <stem>__chromagram_chords.png
    ├── <stem>__tempogram.png
    ├── <stem>__tempo_curve.png      # 局所 BPM 折れ線 + slow/stable/fast 帯
    └── <stem>__key_timeline.png     # 時系列キー帯 (色 = 主音)、転調点に縦線
```

`<stem>_analysis.md` の章立て:

1. **楽曲の骨格** (BPM/拍子/調/長さ/コード総数)
2. **和声の流れと特徴** (主軸調 / コード頻度 / 機能分類 / カデンツ / **転調イベント時刻付き**)
3. **リズム・グルーヴの性格** (テンポレンジ / シンコペーション / **テンポ推移 trend + 速い/遅い区間**)
4. **ジャンル傾向** (AST top-K + style notes クロス参照)
5. **この曲を特徴づける指標** (各 metric を逸脱の大きい順に)
6. **コード進行の時系列概観** (8 区間 × 主要コード)
7. **算出根拠** (採用エンジン名)

JSON のスキーマは `src/chordscope/models.py` の `TrackAnalysis` (Pydantic, frozen) を参照。`<stem>` は入力ファイルの拡張子なし basename で、Unicode を含むそのままの文字列を使う。Markdown 内の画像リンクは URL エンコード済みなので、日本語/空白を含むファイル名でも GitHub 等で正しく描画される。

---

## プロジェクト構成

```
chordscope/
├── src/chordscope/
│   ├── __init__.py            # _compat を最初に読み込んで madmom を使えるようにする
│   ├── _compat.py             # collections/numpy エイリアス補完 + madmom DBN バグ fix
│   ├── audio.py               # librosa.load ラッパー (AudioBuffer)
│   ├── config.py              # TOML 設定ファイルローダー (Pydantic)
│   ├── discovery.py           # 再帰スキャン (シンボリックリンク循環防止)
│   ├── models.py              # 分析結果のイミュータブルなデータモデル群
│   ├── pipeline.py            # 1 ファイルの全分析を束ねるオーケストレータ
│   ├── cli.py                 # Typer CLI (analyze / init-config)
│   ├── genre_catalog.py       # 163 ジャンル × 14 family カタログ (Discogs+AudioSet 由来)
│   ├── analyzers/
│   │   ├── tempo.py           # BPM 推定 (3 手法 + オクターブ正規化)
│   │   ├── tempo_curve.py     # 局所 BPM 時系列 + trend (accelerando/ritardando)
│   │   ├── beats.py           # ビート/ダウンビート/拍子 (parsimony)
│   │   ├── key.py             # 調性 (KS-Temperley)
│   │   ├── modulation.py      # 時系列 KS スライディング窓 + フラップ抑制で転調検出
│   │   ├── chords.py          # コード認識 (madmom / librosa-template)
│   │   ├── theory.py          # 機能和声・カデンツ (music21、転調候補は粗推定として残置)
│   │   ├── genre.py           # AST + AudioSet ジャンル分類 (TOP-5 + family 多様性)
│   │   └── style.py           # ジャンル別観点 (動的ディスパッチ + 80+ ヒューリスティック)
│   └── reporting/
│       ├── console.py         # rich コンソール出力
│       ├── json_report.py     # JSON 出力
│       ├── markdown.py        # Markdown レポート (画像 URL エンコード)
│       ├── narrative.py       # 論理フロー型音楽分析文 (LLM 不要)
│       └── plots.py           # matplotlib グラフ画像 6 種
├── tests/                     # pytest (91 件: ユニット 65 + 統合 AB 26)
│   ├── conftest.py            # PD音源 DL fixture + ユーザー任意音源 fixture
│   ├── fixtures/user-audio/   # ユーザーの実音源を投入する場所
│   └── test_*.py              # ユニット + 統合 + Unicode パス + パイプライン
├── configs/example.toml
├── pyproject.toml             # uv build / ruff / mypy / pytest 一括設定
├── .pre-commit-config.yaml
└── README.md
```

---

## 開発

```bash
# 全品質ゲート
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest -m "not slow"                   # 高速ユニット (65 件、約 7 秒)
uv run pytest -m "slow"                       # 統合 AB (26 件、約 7-10 分: PD 音源 + AST モデル DL)
uv run pytest                                 # 全 91 件

# pre-commit hook 登録
uv run pre-commit install
uv run pre-commit run --all-files
```

### テスト音源の構成

| 種別 | 場所 | 備考 |
|---|---|---|
| **PD 音源** | `~/.cache/chordscope-tests/` | Wikimedia Commons から初回 DL してキャッシュ。Encina (Renaissance), Joplin Maple Leaf Rag (1916 / full), 1920s Fox-trot, Sousa "Stars and Stripes Forever" (1909 行進曲、trio で完全 4 度上に転調) |
| **ユーザー任意音源** | `tests/fixtures/user-audio/` | 既定パス (環境変数 `CHORDSCOPE_TEST_USER_AUDIO_DIR` で上書き可能)。ファイルが無ければ該当テストのみ skip |

PD 音源テストは厳密な数値一致ではなく「音楽学的に妥当な範囲・系統」を AB テストする。例えばラグタイム → ジャンルラベル top-K に Jazz/Blues/Classical 系統が含まれること、など。

### ジャンル分析の AB テスト結果 (PD 音源)

| 音源 | TOP-5 (family 多様性適用) | 確認できた事実 |
|---|---|---|
| Joplin "Maple Leaf Rag" (1916, ラグタイム) | New-age music / Classical music / Jazz / Rhythm and blues / Soul music | 期待 family {jazz, blues, classical} 全て top-5 内。Jazz には「シンコペ 0.76 / BPM 117 は Jazz 典型」の固有 finding |
| Encina "Fata la parte" (15c. ルネサンス声楽) | A capella (70%) / Chant / Opera / Classical music / Music of Asia | 期待 family {religious, classical, world} 全て top-5 内。A capella は楽曲の本質と一致 |

### 品質ゲート結果 (2026-04-26 時点)

| ゲート | 結果 |
|---|---|
| ruff check | All checks passed |
| ruff format --check | 42 files already formatted |
| pytest ユニット (`-m "not slow"`) | 65 passed |
| pytest 統合 AB (`-m "slow"`) | 26 passed (modulation AB 5 + tempo curve AB 7 + 既存 14) |

---

## 設計上の注意

- **madmom 0.16.1 互換シム**: 最終リリース 2018-11 以後更新がなく、`collections.MutableSequence`, `np.float`, `np.int` 等の deprecated API を使う。さらに `DBNDownBeatTrackingProcessor.process` は numpy 1.24+ の inhomogeneous-array 厳格化で `np.argmax(np.asarray(results)[:, 1])` が ValueError になる。`_compat.py` がパッケージ初期化時に互換シム + 該当メソッドの完全互換 monkey-patch を適用する。
- **AST + AudioSet モデル**: 約 330 MB を初回利用時に HuggingFace Hub から `~/.cache/huggingface/` にダウンロードする。
- **精度最優先**: 入力音源は全長を解析する (サンプリング省略なし)。10 秒窓 / 5 秒ホップでジャンル推論を行い、softmax/sigmoid を平均集約する。
- **マルチアルゴリズム集約**: BPM は 3 手法 → オクターブ正規化 → 中央値 + 一致度信頼度。拍子は各候補で独立 DBN → 信頼度 92% 以内なら最小値優先。
- **ファイル名 Unicode**: 全パイプライン (discovery / load / madmom / 出力) で日本語・全角空白・拉丁系アクセント・括弧を含むファイル名を扱う。Markdown 画像リンクは `urllib.parse.quote` でエンコード。

---

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `ImportError: cannot import name 'MutableSequence' from 'collections'` | `chordscope` パッケージを直接 import せず、サブモジュール経由で import している場合に発生する。`from chordscope import ...` を一度実行して `_compat` を起動する |
| madmom 関連で `ValueError: setting an array element with a sequence` | `_compat.patch_madmom_downbeat_inhomogeneous` が適用されていない。`from chordscope import _compat` でロードされる |
| ジャンル分類が極端に遅い (初回) | AST モデル ~330 MB をダウンロード中。`~/.cache/huggingface/` にキャッシュされて 2 回目以降は高速 |
| `librosa.load` で mp3/m4a が読めない | `ffmpeg` をインストール (`brew install ffmpeg`) |
| Python 3.12/3.13 で uv sync が失敗 | madmom が build できない。`mise install python@3.11 && mise use python@3.11` で 3.11 系に切替 |

---

## ライセンス

MIT
