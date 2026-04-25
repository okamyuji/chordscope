"""音楽ジャンル広範カタログ (~150 件)。

Discogs Styles + AudioSet ontology + 主要音楽辞典から抽出した広範ジャンル一覧。
各エントリは canonical 名 + 親 family + 短い説明 + AudioSet/AST が出力する別名群。

このカタログは:
- AST AudioSet 出力ラベルから canonical 名への正規化
- TOP-K 出力時の親 family 多様性チェック
- markdown レポートでの説明文表示
に使われる。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenreEntry:
    canonical: str
    family: str
    description: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


# 親 family は 14 個に集約
FAMILIES = (
    "rock",  # rock 系
    "pop",  # pop 系
    "jazz",  # jazz 系
    "blues",  # blues 系
    "classical",  # 西洋古典
    "folk",  # フォーク・トラディショナル
    "country",  # カントリー
    "electronic",  # エレクトロニック / EDM
    "hiphop",  # ヒップホップ / R&B / ソウル / ファンク
    "metal",  # ヘヴィメタル / ハードロック
    "world",  # 世界各地 (アジア / アフリカ / 中東等)
    "latin",  # ラテンアメリカ
    "religious",  # 宗教音楽 (ゴスペル / 詠唱 / 賛美歌)
    "soundtrack",  # 映像伴奏 / 劇伴 / ゲーム音楽 / 子供向け
)

CATALOG: tuple[GenreEntry, ...] = (
    # ---- rock 系 (15) ----
    GenreEntry("Rock", "rock", "ロックの総称", ("Rock music", "rock")),
    GenreEntry("Rock and roll", "rock", "1950s 初期ロック", ("Rock 'n' roll",)),
    GenreEntry("Hard rock", "rock", "1970s 大音量ギター・ロック"),
    GenreEntry("Classic rock", "rock", "1960-1970s クラシックロック"),
    GenreEntry("Progressive rock", "rock", "プログレッシブ・ロック", ("Prog rock",)),
    GenreEntry("Psychedelic rock", "rock", "サイケデリック・ロック"),
    GenreEntry("Garage rock", "rock", "1960s ガレージ"),
    GenreEntry("Indie rock", "rock", "インディーズ・ロック"),
    GenreEntry("Alternative rock", "rock", "オルタナティブ", ("Alternative",)),
    GenreEntry("Grunge", "rock", "1990s グランジ"),
    GenreEntry("Punk rock", "rock", "パンクロック", ("Punk",)),
    GenreEntry("Post-punk", "rock", "ポストパンク"),
    GenreEntry("Hardcore punk", "rock", "ハードコア", ("Hardcore",)),
    GenreEntry("Surf rock", "rock", "サーフロック"),
    GenreEntry("Emo", "rock", "エモ"),
    # ---- pop 系 (15) ----
    GenreEntry("Pop", "pop", "ポップス全般", ("Pop music",)),
    GenreEntry("J-Pop", "pop", "日本語ポップス", ("Japanese pop",)),
    GenreEntry("K-Pop", "pop", "韓国ポップス", ("Korean pop",)),
    GenreEntry("C-Pop", "pop", "中華圏ポップス", ("Chinese pop",)),
    GenreEntry("Mandopop", "pop", "マンドポップ"),
    GenreEntry("Cantopop", "pop", "広東語ポップス"),
    GenreEntry("Anime", "pop", "アニメ主題歌"),
    GenreEntry("Vocaloid", "pop", "ボーカロイド"),
    GenreEntry("Enka", "pop", "演歌"),
    GenreEntry("Kayoukyoku", "pop", "歌謡曲"),
    GenreEntry("Synthpop", "pop", "シンセポップ"),
    GenreEntry("Dance-pop", "pop", "ダンスポップ"),
    GenreEntry("Teen pop", "pop", "ティーンポップ"),
    GenreEntry("Bubblegum pop", "pop", "バブルガムポップ"),
    GenreEntry("Adult contemporary", "pop", "AC, 大人向けポップス"),
    # ---- jazz 系 (12) ----
    GenreEntry("Jazz", "jazz", "ジャズ全般"),
    GenreEntry("Bebop", "jazz", "ビバップ"),
    GenreEntry("Swing", "jazz", "スウィング", ("Swing music",)),
    GenreEntry("Cool jazz", "jazz", "クールジャズ"),
    GenreEntry("Hard bop", "jazz", "ハードバップ"),
    GenreEntry("Modal jazz", "jazz", "モードジャズ"),
    GenreEntry("Free jazz", "jazz", "フリージャズ"),
    GenreEntry("Fusion", "jazz", "フュージョン", ("Jazz fusion",)),
    GenreEntry("Smooth jazz", "jazz", "スムーズジャズ"),
    GenreEntry("Big band", "jazz", "ビッグバンド"),
    GenreEntry("Dixieland", "jazz", "ディキシーランド"),
    GenreEntry("Ragtime", "jazz", "ラグタイム"),
    # ---- blues 系 (8) ----
    GenreEntry("Blues", "blues", "ブルース全般"),
    GenreEntry("Delta blues", "blues", "デルタ・ブルース"),
    GenreEntry("Chicago blues", "blues", "シカゴ・ブルース"),
    GenreEntry("Electric blues", "blues", "エレクトリック・ブルース"),
    GenreEntry("Boogie-woogie", "blues", "ブギウギ"),
    GenreEntry("Jump blues", "blues", "ジャンプ・ブルース"),
    GenreEntry("Rhythm and blues", "blues", "リズム&ブルース", ("R&B",)),
    GenreEntry("British blues", "blues", "ブリティッシュ・ブルース"),
    # ---- classical 系 (12) ----
    GenreEntry("Classical music", "classical", "西洋クラシック総称", ("Classical",)),
    GenreEntry("Baroque", "classical", "バロック (1600-1750)"),
    GenreEntry("Renaissance music", "classical", "ルネサンス音楽 (1400-1600)"),
    GenreEntry("Medieval", "classical", "中世音楽"),
    GenreEntry("Romantic", "classical", "ロマン派 (1800-1900)"),
    GenreEntry("Modern classical", "classical", "現代クラシック"),
    GenreEntry("Opera", "classical", "オペラ"),
    GenreEntry("Chamber music", "classical", "室内楽"),
    GenreEntry("Symphony", "classical", "交響曲"),
    GenreEntry("Concerto", "classical", "協奏曲"),
    GenreEntry("Choral", "classical", "合唱曲", ("Choir",)),
    GenreEntry("Minimalism", "classical", "ミニマル音楽"),
    # ---- folk 系 (10) ----
    GenreEntry("Folk music", "folk", "フォーク全般", ("Folk",)),
    GenreEntry("American folk", "folk", "アメリカン・フォーク"),
    GenreEntry("British folk", "folk", "ブリティッシュ・フォーク"),
    GenreEntry("Celtic music", "folk", "ケルト音楽"),
    GenreEntry("Sea shanty", "folk", "海の労働歌"),
    GenreEntry("Folk rock", "folk", "フォーク・ロック"),
    GenreEntry("Singer-songwriter", "folk", "シンガーソングライター"),
    GenreEntry("Indie folk", "folk", "インディー・フォーク"),
    GenreEntry("Acoustic", "folk", "アコースティック"),
    GenreEntry("Traditional", "folk", "トラディショナル"),
    # ---- country 系 (8) ----
    GenreEntry("Country", "country", "カントリー全般", ("Country music",)),
    GenreEntry("Bluegrass", "country", "ブルーグラス"),
    GenreEntry("Honky-tonk", "country", "ホンキートンク"),
    GenreEntry("Outlaw country", "country", "アウトロー・カントリー"),
    GenreEntry("Country rock", "country", "カントリー・ロック"),
    GenreEntry("Country pop", "country", "カントリー・ポップ"),
    GenreEntry("Western swing", "country", "ウェスタン・スウィング"),
    GenreEntry("Americana", "country", "アメリカーナ"),
    # ---- electronic 系 (18) ----
    GenreEntry("Electronic music", "electronic", "エレクトロニック総称", ("Electronic", "EDM")),
    GenreEntry("House", "electronic", "ハウス", ("House music",)),
    GenreEntry("Techno", "electronic", "テクノ"),
    GenreEntry("Trance", "electronic", "トランス", ("Trance music",)),
    GenreEntry("Drum and bass", "electronic", "ドラムンベース", ("DnB",)),
    GenreEntry("Dubstep", "electronic", "ダブステップ"),
    GenreEntry("Garage", "electronic", "UK Garage"),
    GenreEntry("Breakbeat", "electronic", "ブレイクビーツ"),
    GenreEntry("Ambient", "electronic", "アンビエント", ("Ambient music",)),
    GenreEntry("Chillout", "electronic", "チルアウト"),
    GenreEntry("Lo-fi", "electronic", "Lo-fi Beats"),
    GenreEntry("IDM", "electronic", "Intelligent Dance Music"),
    GenreEntry("Industrial", "electronic", "インダストリアル"),
    GenreEntry("Synthwave", "electronic", "シンセウェーブ"),
    GenreEntry("Vaporwave", "electronic", "ヴェイパーウェイヴ"),
    GenreEntry("Future bass", "electronic", "フューチャーベース"),
    GenreEntry("Hardcore (electronic)", "electronic", "ハードコアテクノ"),
    GenreEntry("Disco", "electronic", "ディスコ"),
    # ---- hiphop / soul / funk / R&B (12) ----
    GenreEntry("Hip hop", "hiphop", "ヒップホップ", ("Hip hop music",)),
    GenreEntry("Rap", "hiphop", "ラップ"),
    GenreEntry("Trap", "hiphop", "トラップ"),
    GenreEntry("Boom bap", "hiphop", "ブームバップ"),
    GenreEntry("Conscious hip hop", "hiphop", "コンシャス・ヒップホップ"),
    GenreEntry("Gangsta rap", "hiphop", "ギャングスタ・ラップ"),
    GenreEntry("R&B", "hiphop", "Contemporary R&B"),
    GenreEntry("Soul", "hiphop", "ソウル", ("Soul music",)),
    GenreEntry("Funk", "hiphop", "ファンク"),
    GenreEntry("Neo-soul", "hiphop", "ネオソウル"),
    GenreEntry("Motown", "hiphop", "モータウン"),
    GenreEntry("New jack swing", "hiphop", "ニュージャック・スウィング"),
    # ---- metal 系 (10) ----
    GenreEntry("Heavy metal", "metal", "ヘヴィメタル", ("Metal",)),
    GenreEntry("Thrash metal", "metal", "スラッシュメタル"),
    GenreEntry("Death metal", "metal", "デスメタル"),
    GenreEntry("Black metal", "metal", "ブラックメタル"),
    GenreEntry("Power metal", "metal", "パワーメタル"),
    GenreEntry("Doom metal", "metal", "ドゥームメタル"),
    GenreEntry("Speed metal", "metal", "スピードメタル"),
    GenreEntry("Symphonic metal", "metal", "シンフォニックメタル"),
    GenreEntry("Nu metal", "metal", "ニューメタル"),
    GenreEntry("Metalcore", "metal", "メタルコア"),
    # ---- world 系 (15) ----
    GenreEntry("Music of Africa", "world", "アフリカ音楽 (umbrella)"),
    GenreEntry("Afrobeat", "world", "アフロビート"),
    GenreEntry("Highlife", "world", "ハイライフ"),
    GenreEntry("Music of Asia", "world", "アジア音楽 (umbrella)"),
    GenreEntry("Music of Bollywood", "world", "ボリウッド"),
    GenreEntry("Carnatic music", "world", "カルナータカ音楽"),
    GenreEntry("Hindustani classical", "world", "ヒンドゥスターニー古典"),
    GenreEntry(
        "Music of the Middle East", "world", "中東音楽 (umbrella)", ("Middle Eastern music",)
    ),
    GenreEntry("Klezmer", "world", "クレズマー"),
    GenreEntry("Balkan", "world", "バルカン音楽"),
    GenreEntry("Throat singing", "world", "ホーミー"),
    GenreEntry("Gamelan", "world", "ガムラン"),
    GenreEntry("Traditional Japanese", "world", "邦楽 (雅楽・民謡)"),
    GenreEntry("Traditional Chinese", "world", "中国伝統音楽"),
    GenreEntry("Polynesian", "world", "ポリネシア音楽"),
    # ---- latin 系 (12) ----
    GenreEntry("Latin music", "latin", "ラテン音楽 (umbrella)", ("Music of Latin America",)),
    GenreEntry("Salsa", "latin", "サルサ", ("Salsa music",)),
    GenreEntry("Bossa nova", "latin", "ボサノバ"),
    GenreEntry("Samba", "latin", "サンバ"),
    GenreEntry("Tango", "latin", "タンゴ"),
    GenreEntry("Mambo", "latin", "マンボ"),
    GenreEntry("Cumbia", "latin", "クンビア"),
    GenreEntry("Reggaeton", "latin", "レゲトン"),
    GenreEntry("Bachata", "latin", "バチャータ"),
    GenreEntry("Merengue", "latin", "メレンゲ"),
    GenreEntry("Latin jazz", "latin", "ラテン・ジャズ"),
    GenreEntry("Flamenco", "latin", "フラメンコ"),
    # ---- religious 系 (8) ----
    GenreEntry("Gospel", "religious", "ゴスペル", ("Gospel music",)),
    GenreEntry("Christian music", "religious", "クリスチャン音楽"),
    GenreEntry("Chant", "religious", "詠唱"),
    GenreEntry("Mantra", "religious", "マントラ"),
    GenreEntry("Hymn", "religious", "賛美歌"),
    GenreEntry("Spiritual", "religious", "スピリチュアル"),
    GenreEntry("A capella", "religious", "アカペラ"),
    GenreEntry("Devotional", "religious", "信仰歌"),
    # ---- soundtrack / その他 (8) ----
    GenreEntry("Soundtrack", "soundtrack", "サウンドトラック", ("Soundtrack music",)),
    GenreEntry("Film score", "soundtrack", "映画音楽"),
    GenreEntry("Video game music", "soundtrack", "ゲーム音楽"),
    GenreEntry("Anime soundtrack", "soundtrack", "アニメサントラ"),
    GenreEntry("Musical theater", "soundtrack", "ミュージカル", ("Show tunes", "Musical")),
    GenreEntry("Children's music", "soundtrack", "子供向け音楽"),
    GenreEntry("Lullaby", "soundtrack", "子守唄"),
    GenreEntry("Reggae", "world", "レゲエ"),
)


# 検索高速化のため小文字 alias → entry のインデックスを構築
_ALIAS_INDEX: dict[str, GenreEntry] = {}
for _entry in CATALOG:
    _ALIAS_INDEX[_entry.canonical.lower()] = _entry
    for _alias in _entry.aliases:
        _ALIAS_INDEX[_alias.lower()] = _entry


def find_entry(label: str) -> GenreEntry | None:
    """ラベル文字列に最もマッチするカタログエントリを返す。"""
    lower = label.lower().strip()
    if lower in _ALIAS_INDEX:
        return _ALIAS_INDEX[lower]
    # 部分一致 (longest-first)
    for key in sorted(_ALIAS_INDEX, key=len, reverse=True):
        if key in lower:
            return _ALIAS_INDEX[key]
    return None


def family_of(label: str) -> str | None:
    e = find_entry(label)
    return e.family if e else None


def all_families() -> tuple[str, ...]:
    return FAMILIES


def catalog_size() -> int:
    return len(CATALOG)
