"""学習済みモデルによるジャンル分類 (広範ラベル対応)。

主モデル: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Audio Spectrogram Transformer + AudioSet 527 クラス。
- AudioSet ontology の "Music" / "Music genre" / "Musical instrument" 配下を含む。
- multi-label sigmoid 出力 (各ラベル独立確率)。

`Music genre` サブツリーの代表ラベル (Pop music, Rock music, Jazz, Classical music,
Country, Funk, Disco, Hip hop music, Heavy metal, Reggae, Folk music, Blues,
Gospel music, Soul music, Latin music, Electronic music, Ambient music, Dance music,
House music, Techno, Drum and bass, Salsa music, Flamenco, Music of Bollywood,
Music of Africa, Music of Asia, Music of Latin America, Christian music,
Independent music, ...) など 50+ を扱える。

精度最優先方針: 30 秒窓・15 秒ホップで全長を覆い、各窓のロジットを平均集約する。
"""

from __future__ import annotations

from typing import Literal

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from chordscope.audio import AudioBuffer
from chordscope.models import GenreResult, GenreScore

DEFAULT_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
TARGET_SR = 16000
WINDOW_SEC = 10.0  # AST は 10s 学習
HOP_SEC = 5.0  # 50% オーバーラップ

# AudioSet ontology から "音楽ジャンル" 系統のラベルを列挙 (大文字小文字含め完全一致)。
# 完全に網羅していない場合は実行時に "music" を含むラベルでフォールバック判定する。
_GENRE_KEYWORDS = (
    "music",  # 'Pop music', 'Rock music', 'Folk music', etc.
)
_DIRECT_GENRE_LABELS = frozenset(
    {
        "Jazz",
        "Blues",
        "Reggae",
        "Hip hop music",
        "Heavy metal",
        "Techno",
        "Punk rock",
        "Disco",
        "Funk",
        "Soul music",
        "Country",
        "Classical music",
        "Opera",
        "Rhythm and blues",
        "Salsa music",
        "Flamenco",
        "Gospel music",
        "Christian music",
        "Music of Bollywood",
        "Music of Africa",
        "Music of Asia",
        "Music of Latin America",
        "Music of the Middle East",
        "Music for children",
        "Independent music",
        "Electronic music",
        "Ambient music",
        "Drum and bass",
        "House music",
        "Trance music",
        "Dubstep",
        "Ska",
        "Afrobeat",
        "Bluegrass",
        "Chant",
        "Mantra",
        "Lullaby",
        "Theme music",
        "Soundtrack music",
        "Video game music",
        "Children's music",
        "New-age music",
        "Vocal music",
        "Folk music",
        "Pop music",
        "Rock music",
        "Rock and roll",
        "Psychedelic rock",
        "Grunge",
        "Progressive rock",
        "Swing music",
        "Bossa nova",
        "Mambo",
        "Carnatic music",
        "A capella",
        "Beatboxing",
        "Cumbia",
        "Tango",
        "Bolero",
    }
)


# 上位概念ラベル: AudioSet の親階層で常時高スコアになるため除外する。
# (これらが top を独占すると個別ジャンルが見えなくなる)
_UMBRELLA_LABELS = frozenset(
    {
        # Music ontology の親階層 (どの音楽でも高スコア)
        "Music",
        "Musical instrument",
        "Singing",
        "Speech",
        "Sound effect",
        "Background music",
        # 楽器カテゴリ (ジャンルではない)
        "Plucked string instrument",
        "Bowed string instrument",
        "Brass instrument",
        "Woodwind instrument",
        "Keyboard (musical)",
        "Percussion",
        "Drum kit",
        "Drum",
        "Bass (instrument role)",
        "Mallet percussion",
        "Vocal music",
        # ムード / 感情ラベル (ジャンルではない)
        "Happy music",
        "Funny music",
        "Sad music",
        "Tender music",
        "Exciting music",
        "Angry music",
        "Scary music",
        # 機能ラベル (ジャンルではない)
        "Theme music",
        "Jingle (music)",
        "Soundtrack music",
        "Lullaby",
        "Video game music",
        "Christmas music",
        "Wedding music",
        "Birthday music",
        # 一般語
        "Independent music",
    }
)


def _is_genre_label(label: str) -> bool:
    if label in _UMBRELLA_LABELS:
        return False
    if label in _DIRECT_GENRE_LABELS:
        return True
    lower = label.lower()
    return "music" in lower and "background" not in lower


class GenreClassifier:
    """モデルを 1 度だけロードして使い回すクラス。"""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        # PyTorch 推論モード (dropout/batchnorm 凍結)
        self.model.train(False)
        self.id2label: dict[int, str] = self.model.config.id2label
        self._genre_indices: list[int] = [
            i for i, lbl in self.id2label.items() if _is_genre_label(lbl)
        ]
        if not self._genre_indices:
            # フォールバック: 全ラベルを対象にする (非 AudioSet モデル等)
            self._genre_indices = list(self.id2label.keys())
        self._is_multi_label = self._detect_multi_label()

    def _detect_multi_label(self) -> bool:
        problem = getattr(self.model.config, "problem_type", None)
        if problem == "multi_label_classification":
            return True
        # AST AudioSet の既定は multi-label
        return self.model_id.startswith("MIT/ast")

    def _windows(self, samples: np.ndarray, sr: int) -> list[np.ndarray]:
        win = int(WINDOW_SEC * sr)
        hop = int(HOP_SEC * sr)
        if len(samples) <= win:
            return [samples]
        out: list[np.ndarray] = []
        for start in range(0, len(samples) - win + 1, hop):
            out.append(samples[start : start + win])
        if (len(samples) - win) % hop != 0:
            out.append(samples[-win:])
        return out

    def predict(
        self,
        buffer: AudioBuffer,
        *,
        aggregation: Literal["mean", "max"] = "mean",
        top_k: int = 5,
        family_diversity: bool = True,
    ) -> GenreResult:
        """AST 推論結果から TOP-K のジャンル分布を返す。

        - `top_k` (既定 5): 出力件数。
        - `family_diversity`: True なら親 family 重複を抑え多様性を確保する。
          (例: top にロック系ばかり並ばないように、 1 family あたり 2 件まで)
        """
        from chordscope.genre_catalog import family_of

        if buffer.sample_rate != TARGET_SR:
            samples = librosa.resample(
                buffer.samples, orig_sr=buffer.sample_rate, target_sr=TARGET_SR
            )
        else:
            samples = buffer.samples
        wins = self._windows(samples, TARGET_SR)
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for w in wins:
                inputs = self.extractor(w, sampling_rate=TARGET_SR, return_tensors="pt")
                logits = self.model(**inputs).logits
                if self._is_multi_label:
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                all_probs.append(probs)
        stacked = np.stack(all_probs)
        agg = stacked.mean(axis=0) if aggregation == "mean" else stacked.max(axis=0)
        # ジャンル関連ラベルだけ抽出して並び替え
        genre_scores: list[GenreScore] = []
        for idx in self._genre_indices:
            label = str(self.id2label[int(idx)])
            score = float(agg[idx])
            genre_scores.append(GenreScore(label=label, score=round(score, 4)))
        genre_scores.sort(key=lambda g: g.score, reverse=True)
        if not genre_scores:
            msg = "No genre labels available in model"
            raise RuntimeError(msg)
        if family_diversity:
            family_count: dict[str, int] = {}
            picked: list[GenreScore] = []
            family_cap = max(2, top_k // 3)  # top_k=5 なら 1 family 最大 2 件
            for g in genre_scores:
                fam = family_of(g.label) or "_other"
                if family_count.get(fam, 0) >= family_cap:
                    continue
                picked.append(g)
                family_count[fam] = family_count.get(fam, 0) + 1
                if len(picked) >= top_k:
                    break
            distribution = picked
        else:
            distribution = genre_scores[:top_k]
        return GenreResult(
            top=distribution[0],
            distribution=distribution,
            model_id=self.model_id,
            aggregation=aggregation,
        )
