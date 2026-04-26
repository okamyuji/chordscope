"""matplotlib による分析結果の可視化。

生成するグラフ:
- waveform: 波形 + ビート/ダウンビート overlay
- spectrogram: メルスペクトログラム (dB)
- chromagram: CQT クロマ + コード区間オーバーレイ
- tempogram: テンポグラム + 推定 BPM ライン
"""

from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # ヘッドレス環境での描画
import matplotlib.pyplot as plt
import numpy as np

from chordscope.audio import AudioBuffer
from chordscope.models import (
    BeatResult,
    ChordResult,
    ModulationResult,
    TempoCurveResult,
    TempoResult,
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _safe_title(name: str, max_len: int = 50) -> str:
    """matplotlib のデフォルトフォントが扱えない CJK 等のグリフを '?' に置換する。"""
    try:
        name.encode("latin-1")
        return name[:max_len]
    except UnicodeEncodeError:
        return "(non-ascii filename)"


def waveform_with_beats(buffer: AudioBuffer, beats: BeatResult, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(buffer.samples, sr=buffer.sample_rate, ax=ax, alpha=0.6)
    for t in beats.beat_times:
        ax.axvline(t, color="gray", alpha=0.4, linewidth=0.6)
    for t in beats.downbeat_times:
        ax.axvline(t, color="red", alpha=0.8, linewidth=1.0)
    ax.set_title(f"Waveform with beats (red=downbeat) — {_safe_title(buffer.path.name)}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return _save(fig, out_path)


def mel_spectrogram(buffer: AudioBuffer, out_path: Path) -> Path:
    mel = librosa.feature.melspectrogram(
        y=buffer.samples, sr=buffer.sample_rate, n_mels=128, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        mel_db, sr=buffer.sample_rate, x_axis="time", y_axis="mel", ax=ax, fmax=8000
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel spectrogram — {_safe_title(buffer.path.name)}")
    return _save(fig, out_path)


def chromagram_with_chords(buffer: AudioBuffer, chords: ChordResult, out_path: Path) -> Path:
    chroma = librosa.feature.chroma_cqt(y=buffer.samples, sr=buffer.sample_rate)
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.specshow(chroma, sr=buffer.sample_rate, x_axis="time", y_axis="chroma", ax=ax)
    # コード区間ラベルを上部に重ねる
    last_text_x = -np.inf
    for seg in chords.segments:
        if seg.label in ("N", "X", ""):
            continue
        if seg.start - last_text_x < 1.5:
            continue
        ax.text(
            seg.start,
            12.2,
            seg.label,
            fontsize=7,
            color="white",
            backgroundcolor="black",
            alpha=0.7,
        )
        last_text_x = seg.start
    ax.set_title(f"Chromagram with chord overlay — {_safe_title(buffer.path.name)}")
    return _save(fig, out_path)


def tempogram_with_bpm(buffer: AudioBuffer, tempo: TempoResult, out_path: Path) -> Path:
    onset_env = librosa.onset.onset_strength(y=buffer.samples, sr=buffer.sample_rate)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=buffer.sample_rate)
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.specshow(tempogram, sr=buffer.sample_rate, x_axis="time", y_axis="tempo", ax=ax)
    ax.axhline(tempo.bpm, color="red", linewidth=2, label=f"{tempo.bpm:.1f} BPM")
    ax.legend(loc="upper right")
    ax.set_title(f"Tempogram — {_safe_title(buffer.path.name)}")
    return _save(fig, out_path)


def tempo_curve_plot(buffer: AudioBuffer, tempo_curve: TempoCurveResult, out_path: Path) -> Path:
    """局所 BPM 時系列をプロットする。"""
    fig, ax = plt.subplots(figsize=(12, 3))
    if tempo_curve.segments:
        # 各セグメントの中央時刻と local_bpm を結ぶ
        xs: list[float] = []
        ys: list[float] = []
        for seg in tempo_curve.segments:
            mid = (seg.start_sec + seg.end_sec) / 2.0
            xs.append(mid)
            ys.append(seg.local_bpm)
        ax.plot(xs, ys, marker="o", linewidth=1.5, color="tab:blue", label="local BPM")
        # 領域塗り (slow=青/stable=緑/fast=赤)
        color_map = {"slow": "#cce5ff", "stable": "#d4edda", "fast": "#f8d7da"}
        for seg in tempo_curve.segments:
            ax.axvspan(seg.start_sec, seg.end_sec, color=color_map[seg.label], alpha=0.4)
    ax.axhline(
        tempo_curve.global_bpm,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"global {tempo_curve.global_bpm:.1f} BPM",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Local BPM")
    ax.set_title(
        f"Tempo curve (trend={tempo_curve.trend}, "
        f"range={tempo_curve.bpm_min:.1f}..{tempo_curve.bpm_max:.1f}) — "
        f"{_safe_title(buffer.path.name)}"
    )
    ax.legend(loc="upper right")
    return _save(fig, out_path)


def key_timeline_plot(buffer: AudioBuffer, modulation: ModulationResult, out_path: Path) -> Path:
    """時系列のキー帯を描く。"""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    cmap = plt.get_cmap("hsv")
    pitch_to_index = {
        n: i
        for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    }
    if modulation.segments:
        for seg in modulation.segments:
            color_val = pitch_to_index[seg.tonic] / 12.0
            color = cmap(color_val)
            ax.axvspan(
                seg.start_sec, seg.end_sec, color=color, alpha=0.4 if seg.mode == "major" else 0.7
            )
            ax.text(
                (seg.start_sec + seg.end_sec) / 2.0,
                0.5,
                f"{seg.tonic} {seg.mode[:3]}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    for ch in modulation.changes:
        ax.axvline(ch.at_sec, color="black", linewidth=1.2, linestyle=":")
    ax.set_xlim(0, max(buffer.duration, 1.0))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"Key timeline (changes={len(modulation.changes)}) — {_safe_title(buffer.path.name)}"
    )
    return _save(fig, out_path)


def render_all_plots(
    buffer: AudioBuffer,
    *,
    beats: BeatResult,
    chords: ChordResult,
    tempo: TempoResult,
    tempo_curve: TempoCurveResult | None = None,
    modulation: ModulationResult | None = None,
    out_dir: Path,
) -> dict[str, Path]:
    stem = buffer.path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {
        "waveform_beats": waveform_with_beats(
            buffer, beats, out_dir / f"{stem}__waveform_beats.png"
        ),
        "mel_spectrogram": mel_spectrogram(buffer, out_dir / f"{stem}__mel_spectrogram.png"),
        "chromagram_chords": chromagram_with_chords(
            buffer, chords, out_dir / f"{stem}__chromagram_chords.png"
        ),
        "tempogram": tempogram_with_bpm(buffer, tempo, out_dir / f"{stem}__tempogram.png"),
    }
    if tempo_curve is not None:
        out["tempo_curve"] = tempo_curve_plot(
            buffer, tempo_curve, out_dir / f"{stem}__tempo_curve.png"
        )
    if modulation is not None:
        out["key_timeline"] = key_timeline_plot(
            buffer, modulation, out_dir / f"{stem}__key_timeline.png"
        )
    return out
