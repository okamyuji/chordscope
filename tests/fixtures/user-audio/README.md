# user-audio フィクスチャ

このディレクトリに対応拡張子の音声ファイル (mp3/wav/flac/ogg/aac/aiff/mp4/m4a) を 1 つ以上置くと、
`tests/test_pipeline.py::test_full_pipeline_on_user_audio` が動作します。

ファイルが無い場合、該当テストは自動的に skip されます。

別ディレクトリを使う場合: 環境変数 `CHORDSCOPE_TEST_USER_AUDIO_DIR` を指定してください。

```sh
CHORDSCOPE_TEST_USER_AUDIO_DIR=/path/to/your/audio uv run pytest -m integration
```

## 注意

ここに置く音声ファイルは Git にコミットしないでください (.gitignore で除外済)。
