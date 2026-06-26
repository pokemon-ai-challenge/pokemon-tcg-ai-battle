"""Meta deck knowledge base based on actual competitive tier ranking data.

Source: pokeka-win-decks.jp + torecamap.co.jp (scraped 2026-06-26, merged cache)
  Tier 1 (S): dragapult_ex (102), hydrapple_ex/カミツオロチex (95), terasta_bullet (3)
  Tier 2 (A): ogerpon_bullet (99), raging_bolt_ex/タケルライコex (98),
              mega_lucario_ex (98), olivia_ex/オリーヴァex (94),
              crustle/イワパレス (4), maries_obstagoon_ex/マリィのオーロンゲex (5)
  Tier 3 (B): alakazam/フーディン (97), Nのゾロアークex (102), ゲッコウガex (102),
              シロナのガブリアスex (102), ロケット団のドンカラス (98), + others

Tier mapping: cache tier 1 → "S", tier 2 → "A", tier 3 → "B".

To regenerate from a fresh cache:
  cd cardlist_referenced/tier_deck_data
  python load_tier_cache.py       # prints current tier summary
  # Then resolve JP card names with JP_Card_Data.csv and update recipes below.

Recipes resolved from data/JP_Card_Data.csv via normalized JP name matching + manual disambiguation.
See docs/strategy-knowledge.md [META-004] for resolution notes.
"""
from dataclasses import dataclass


@dataclass
class MetaDeck:
    name: str
    archetype: str
    tier: str
    key_pokemon_ids: list[int]      # primary recognizers for estimate_deck_candidates
    recipe: list[tuple[int, int]]   # (card_id, count), must total 60


# ---------------------------------------------------------------------------
# Tier 1 (S) Deck Recipes
# ---------------------------------------------------------------------------

# --- Dragapult ex (ドラパルトex) ---
# Blaziken ex energy-cycle variant. Dreepy×4 → Dragapult ex via Rare Candy.
# Blaziken ex (326 JTG) recycles energies from discard to bench.
# Energy: {R}×3 + {P}×3 + {D}×2
_DRAGAPULT_EX: list[tuple[int, int]] = [
    # Pokemon
    (119, 4),    # Dreepy — 70 HP, Buddy-Buddy Poffin searchable
    (120, 4),    # Drakloak — Stage 1, Recon Directive draw ability
    (121, 2),    # Dragapult ex — 320 HP, Phantom Dive {R}{P} 200 + 6 bench spread
    (324, 2),    # Torchic (JTG) — Basic, Blaziken line
    (325, 1),    # Combusken (JTG) — Stage 1
    (326, 2),    # Blaziken ex (JTG) — Stage 2, energy acceleration
    (112, 2),    # Munkidori — ability support
    (1071, 2),   # Meowth ex (POR) — draw ability
    (140, 1),    # Fezandipiti ex — 1-prize draw
    (791, 1),    # Moltres (PFL) — tech attacker
    (1063, 1),   # Chien-Pao (POR) — energy tech
    (235, 1),    # Budew — chip damage tech
    # Items
    (1086, 4),   # Buddy-Buddy Poffin — search 2 Basic ≤70 HP (Dreepy!)
    (1121, 4),   # Ultra Ball
    (1152, 4),   # Poké Pad
    (1227, 4),   # Lillie's Determination
    (1182, 3),   # Boss's Orders
    (1079, 2),   # Rare Candy — Dreepy → Dragapult ex (skip Drakloak)
    (1097, 2),   # Night Stretcher
    (1231, 2),   # Dawn
    (1198, 1),   # Iono
    (1250, 2),   # trainer tech
    (1080, 1),   # Unfair Stamp (ACE SPEC)
    # Energy
    (2, 3),      # Basic {R} Energy
    (5, 3),      # Basic {P} Energy
    (7, 2),      # Basic {D} Energy
]
# Total: 4+4+2+2+1+2+2+2+1+1+1+1 + 4+4+4+4+3+2+2+2+1+2+1 + 3+3+2 = 60

# --- Hydrapple ex (カミツオロチex) ---
# Self-healing Stage 2 attacker. Applin→Dipplin→Hydrapple ex + Chikorita→Meganium energy line.
# Teal Mask Ogerpon ex (96) as sub-attacker. Energy: {G}×15.
_HYDRAPPLE_EX: list[tuple[int, int]] = [
    # Pokemon
    (149, 2),    # Applin (SCR) — Basic, Hydrapple line
    (921, 2),    # Dipplin (SCR) — Stage 1
    (150, 2),    # Hydrapple ex — Stage 2, 330 HP, self-heal + energy attack
    (708, 2),    # Chikorita (MEG) — Basic, Meganium line
    (709, 2),    # Bayleef (MEG) — Stage 1
    (710, 2),    # Meganium (MEG) — Stage 2, energy acceleration
    (96, 4),     # Teal Mask Ogerpon ex (TWM) — sub-attacker
    (655, 2),    # Celebi — bench utility
    (1071, 2),   # Meowth ex (POR) — draw
    (140, 1),    # Fezandipiti ex — draw
    (920, 1),    # tech Pokémon
    # Items
    (1094, 4),   # Nest Ball
    (1121, 4),   # Ultra Ball
    (1227, 4),   # Lillie's Determination
    (1261, 4),   # trainer card
    (1231, 2),   # Dawn
    (1182, 1),   # Boss's Orders
    (1184, 1),   # trainer
    (1188, 1),   # trainer
    (1097, 1),   # Night Stretcher
    (1080, 1),   # Unfair Stamp (ACE SPEC)
    # Energy
    (1, 15),     # Basic {G} Energy
]
# Total: 2+2+2+2+2+2+4+2+2+1+1 + 4+4+4+4+2+1+1+1+1+1 + 15 = 60


# ---------------------------------------------------------------------------
# Tier 2 (A) Deck Recipes
# ---------------------------------------------------------------------------

# --- Ogerpon Bullet (オーガポンバレット) ---
# Multi-type "bullet" deck with diverse attackers. No single evolution line;
# uses Energy Switch (1116) to freely move multi-type energy across attackers.
# Key attackers: オーガポン+みどりのめんex(96), メガガルーラex(756), ラティアスex(184).
_OGERPON_BULLET: list[tuple[int, int]] = [
    # Pokemon
    (96, 2),     # オーガポン+みどりのめんex (TWM) — main grass attacker
    (108, 1),    # オーガポン+いどのめんex — water sub-attacker
    (756, 2),    # メガガルーラex — normal/mega sub-attacker
    (112, 2),    # マシマシラ — bench utility
    (184, 2),    # ラティアスex — psychic sub-attacker
    (1071, 2),   # ニャースex — draw engine
    (272, 2),    # リーリエのピッピex — draw/setup
    (858, 1),    # コダック (Psyduck) — tech
    (978, 1),    # ナゲツケサル — tech
    (209, 1),    # パオジアン (M2a) — water tech
    (75, 1),     # テツノイサハex — lightning sub-attacker
    (140, 1),    # キチキギスex (Fezandipiti ex) — draw
    # Items
    (1121, 4),   # ハイパーボール
    (1116, 4),   # エネルギーつけかえ — moves energy across multi-type attackers
    (1182, 3),   # ボスの指令
    (1198, 3),   # アカマツ (Iono)
    (1250, 3),   # ゼロの大空洞 (stadium)
    (1097, 2),   # 夜のタンカ
    (1205, 2),   # シアノ
    (1210, 2),   # タケシのスカウト
    (1227, 2),   # リーリエの決心
    (1188, 1),   # 暗号マニアの解読
    (1080, 1),   # アンフェアスタンプ (ACE SPEC)
    # Energy (multi-type)
    (1, 6),      # 基本草エネルギー
    (7, 3),      # 基本悪エネルギー
    (5, 2),      # 基本超エネルギー
    (16, 2),     # プリズムエネルギー
    (3, 1),      # 基本水エネルギー
    (6, 1),      # 基本闘エネルギー
]
# Total: 2+1+2+2+2+2+2+1+1+1+1+1 + 4+4+3+3+3+2+2+2+2+1+1 + 6+3+2+2+1+1 = 60

# --- Raging Bolt ex (タケルライコex) ---
# Ancient lightning attacker + Teal Mask Ogerpon ex as bench support.
# Iron Hands ex (75) sub-attacker. Energy: {G}×11 + {L}×3 + {F}×3.
_RAGING_BOLT_EX: list[tuple[int, int]] = [
    # Pokemon
    (63, 3),     # Raging Bolt ex (TEF) — main ancient Pokémon attacker
    (62, 1),     # Koraidon (TEF) — energy search ability
    (75, 2),     # Iron Hands ex — sub-attacker, Floodgate Hammer ability
    (96, 4),     # Teal Mask Ogerpon ex (TWM) — bench utility/sub
    (1071, 1),   # Meowth ex (POR)
    (140, 1),    # Fezandipiti ex
    (184, 1),    # Latias ex
    # Items
    (1094, 4),   # Nest Ball
    (1121, 4),   # Ultra Ball
    (1198, 4),   # Pokémon Catcher
    (1122, 3),   # Pokégear 3.0
    (1124, 4),   # trainer
    (1227, 3),   # Lillie's Determination
    (1182, 1),   # Boss's Orders
    (1163, 1),   # trainer
    (1127, 1),   # trainer
    (1118, 1),   # trainer
    (1097, 1),   # Night Stretcher
    (1213, 1),   # Judge
    (1250, 1),   # trainer tech
    (1080, 1),   # Unfair Stamp (ACE SPEC)
    # Energy
    (1, 11),     # Basic {G} Energy
    (4, 3),      # Basic {L} Energy
    (6, 3),      # Basic {F} Energy
]
# Total: 3+1+2+4+1+1+1 + 4+4+4+3+4+3+1+1+1+1+1+1+1+1 + 11+3+3 = 60

# --- Mega Lucario ex (メガルカリオex) ---
# Riolu → Mega Lucario ex. Lunatone/Solrock draw engine.
# Makuhita → Hariyama as sub-attacker/bench sitter.
# Aura Jab {F}=130 + recycle 3 {F} from discard. Energy: {F}×10 + ロック闘×2.
_MEGA_LUCARIO_EX: list[tuple[int, int]] = [
    # Pokemon
    (677, 4),    # Riolu (MEG) — Basic, Mega Lucario ex line
    (678, 3),    # Mega Lucario ex (MEG) — 340 HP Mega Pokémon ex
    (673, 2),    # Makuhita (MEG) — Basic, Hariyama line
    (674, 2),    # Hariyama (MEG) — Stage 1, sub-attacker
    (675, 2),    # Lunatone (MEG) — draw engine (Lunar Cycle: discard {F} → draw 3)
    (676, 2),    # Solrock (MEG) — required partner for Lunatone ability
    (1071, 1),   # Meowth ex (POR)
    # Items
    (1121, 4),   # Ultra Ball
    (1141, 4),   # Power Protein (trainer)
    (1142, 4),   # Fight Gong (trainer)
    (1152, 3),   # Poké Pad
    (1227, 4),   # Lillie's Determination
    (1182, 2),   # Boss's Orders
    (1192, 2),   # Carmine
    (1213, 3),   # Judge
    (1206, 1),   # trainer
    (1174, 1),   # trainer
    (1158, 1),   # trainer
    (1123, 1),   # Switch
    (1252, 1),   # trainer tech
    (1256, 1),   # Gravity Mountain (stadium)
    # Energy
    (6, 10),     # Basic {F} Energy
    (20, 2),     # ロック【闘】エネルギー (special fighting energy)
]
# Total: 4+3+2+2+2+2+1 + 4+4+4+3+4+2+2+3+1+1+1+1+1+1 + 10+2 = 60

# --- Olivia ex (オリーヴァex) ---
# Grass self-heal Stage 2 deck. ミニーブ→オリーニョ→オリーヴァex (MC) as main attacker.
# メガニウム (MEG M1S) for energy acceleration. オーガポン+みどりのめんex (96) as sub-attacker.
# Energy: {G}×12.
_OLIVIA_EX: list[tuple[int, int]] = [
    # Pokemon
    (402, 2),    # ミニーブ (MC) — Basic, Olivia line
    (403, 2),    # オリーニョ (MC) — Stage 1
    (404, 2),    # オリーヴァex (MC) — Stage 2, self-heal attacker
    (708, 2),    # チコリータ (MEG M1S) — Basic, Meganium energy line
    (709, 2),    # ベイリーフ (MEG M1S) — Stage 1
    (710, 2),    # メガニウム (MEG M1S) — Stage 2, energy acceleration
    (96, 4),     # オーガポン+みどりのめんex (TWM) — sub-attacker
    (1071, 2),   # ニャースex — draw
    (140, 1),    # キチキギスex (Fezandipiti ex) — draw
    (235, 1),    # スボミー — bench utility
    # Items
    (1094, 4),   # むしとりセット (Nest Ball)
    (1121, 4),   # ハイパーボール
    (1227, 4),   # リーリエの決心
    (1261, 4),   # 活力の森 (stadium)
    (1152, 2),   # ポケパッド
    (1231, 2),   # ヒカリ
    (1182, 2),   # ボスの指令
    (1116, 1),   # エネルギーつけかえ
    (1097, 1),   # 夜のタンカ
    (1080, 1),   # アンフェアスタンプ (ACE SPEC)
    (1201, 1),   # ブライア
    (1213, 1),   # ジャッジマン
    (1184, 1),   # スイレンのお世話
    # Energy
    (1, 12),     # 基本草エネルギー (consensus 13→12 to reach exactly 60)
]
# Total: 2+2+2+2+2+2+4+2+1+1 + 4+4+4+4+2+2+2+1+1+1+1+1+1 + 12 = 60

# --- Terasta Bullet (テラスタルバレット) ---
# Multi-type Terastal "bullet" deck. No single evolution line; freely moves
# multi-type energy via Energy Switch (1116). Same strategy as Ogerpon Bullet
# but emphasizes Terastal attackers. Consensus from 2 torecamap recipes.
# スペシャルレッドカード substitute: 基本草エネルギー+1.
_TERASTA_BULLET: list[tuple[int, int]] = [
    # Pokemon
    (96, 2),     # オーガポンみどりのめんex — grass attacker
    (108, 2),    # オーガポンいどのめんex — water attacker
    (272, 2),    # リーリエのピッピex — draw/setup
    (756, 2),    # メガガルーラex — normal/mega attacker
    (1071, 2),   # ニャースex — draw engine
    (184, 2),    # ラティアスex — psychic attacker
    (140, 2),    # キチキギスex — draw
    (112, 1),    # マシマシラ — bench utility
    (75, 1),     # テツノイサハex — lightning attacker
    # Items/Trainers
    (1121, 4),   # ハイパーボール
    (1116, 4),   # エネルギーつけかえ
    (1205, 4),   # シアノ
    (1198, 4),   # アカマツ
    (1250, 3),   # ゼロの大空洞
    (1188, 2),   # 暗号マニアの解読
    (1182, 2),   # ボスの指令
    (1097, 2),   # 夜のタンカ
    (1122, 2),   # ポケギア3.0
    (1210, 1),   # タケシのスカウト
    (1080, 1),   # アンフェアスタンプ (ACE SPEC)
    # Energy (multi-type)
    (1, 7),      # 基本草エネルギー (6 consensus + 1 substitute)
    (5, 4),      # 基本超エネルギー
    (11, 2),     # ミストエネルギー
    (3, 2),      # 基本水エネルギー
]
# Total: 2+2+2+2+2+2+2+1+1 + 4+4+4+4+3+2+2+2+2+1+1 + 7+4+2+2 = 60

# --- Crustle (イワパレス) ---
# Dwebble(344)→Crustle(345) as main tank/attacker with MegaKangaskhan ex(756).
# Special energy disruption: ミストエネルギー(11), スパイクエネルギー(14),
# グロウ【草】エネルギー(18), ロック【闘】エネルギー(20).
# オーガポンいしずえのめんex(117) as rock-type sub-attacker.
# Consensus from 3 torecamap recipes. ポケモンセンターのお姉さん substitute: アカマツ×2.
_CRUSTLE: list[tuple[int, int]] = [
    # Pokemon
    (344, 4),    # イシズマイ (MC) — Basic, Crustle line
    (345, 3),    # イワパレス (MC) — Stage 1, main tank
    (756, 4),    # メガガルーラex — secondary attacker
    (117, 1),    # オーガポンいしずえのめんex — rock-type sub-attacker
    # Items/Trainers
    (1122, 4),   # ポケギア3.0
    (1227, 4),   # リーリエの決心
    (1182, 4),   # ボスの指令
    (1198, 2),   # アカマツ (substitute for ポケモンセンターのお姉さん + extra)
    (1147, 3),   # ジャンボアイス
    (1219, 3),   # ロケット団のラムダ
    (1225, 3),   # トウコ
    (1086, 2),   # なかよしポフィン
    (1121, 2),   # ハイパーボール
    (1123, 1),   # ポケモンいれかえ
    (1177, 1),   # せいなるおまもり
    (1159, 1),   # ヒーローマント
    (1194, 1),   # アクロマの執念
    (1197, 1),   # クセロシキのたくらみ
    (1257, 1),   # ロケット団のファクトリー
    # Energy (multi-type disruption)
    (11, 4),     # ミストエネルギー
    (14, 4),     # スパイクエネルギー
    (18, 3),     # グロウ【草】エネルギー
    (20, 2),     # ロック【闘】エネルギー
    (1, 2),      # 基本草エネルギー
]
# Total: 4+3+4+1 + 4+4+4+2+3+3+3+2+2+1+1+1+1+1+1 + 4+4+3+2+2 = 60

# --- Marie's Obstagoon ex (マリィのオーロンゲex) ---
# Marnie's Zigzagoon(646)→Linoone(647)→Obstagoon ex(648) SVOM evolution chain.
# Dark-type disruption with Maschiff(112), Snover(103)→Abomasnow(104).
# スペシャルレッドカード substitute: 基本悪エネルギー+1.
# Consensus from 4 torecamap recipes.
_MARIES_OBSTAGOON_EX: list[tuple[int, int]] = [
    # Pokemon
    (646, 4),    # マリィのベロバー (SVOM) — Basic, Obstagoon line
    (647, 2),    # マリィのギモー (SVOM) — Stage 1
    (648, 3),    # マリィのオーロンゲex (SVOM) — Stage 2, main attacker
    (112, 4),    # マシマシラ — bench utility/draw
    (103, 2),    # ユキワラシ (MC) — Basic, Abomasnow line
    (104, 2),    # ユキメノコ (SV8a) — Stage 1
    (689, 1),    # イベルタル (MC) — tech
    (235, 1),    # スボミー — chip damage tech
    # Items/Trainers
    (1152, 4),   # ポケパッド
    (1227, 4),   # リーリエの決心
    (1219, 4),   # ロケット団のラムダ
    (1259, 4),   # スパイクタウンジム
    (1086, 3),   # なかよしポフィン
    (1182, 3),   # ボスの指令
    (1079, 2),   # ふしぎなアメ (skip stage 1 evolution)
    (1097, 2),   # 夜のタンカ
    (1206, 3),   # アオキの手際 (2 consensus + 1 substitute)
    (1080, 1),   # アンフェアスタンプ (ACE SPEC)
    (1174, 1),   # ふうせん
    # Energy
    (7, 10),     # 基本悪エネルギー (9 consensus + 1 substitute)
]
# Total: 4+2+3+4+2+2+1+1 + 4+4+4+4+3+3+2+2+3+1+1 + 10 = 60

# --- Alakazam (フーディン) ---
# Abra(741)→Kadabra(742)→Alakazam(743) MEG chain. Hand-size damage deck.
# Alakazam: Powerful Hand — 2 counters per hand card.
# Dunsparce(65)+Dudunsparce(66) as search/draw engine.
# Telepathos {P} energy (ID 19) special energy.
# Tier 3 (B) in actual cache: 98 Tier-3 entries vs 3 Tier-2 entries.
_ALAKAZAM: list[tuple[int, int]] = [
    # Pokemon
    (741, 4),    # Abra (MEG) — Basic, 50 HP
    (742, 4),    # Kadabra (MEG) — Stage 1, Psychic Draw ability (draw 2 on evolve)
    (743, 4),    # Alakazam (MEG) — Stage 2, Psychic Draw ability (draw 3), Powerful Hand attack
    (65, 3),     # Dunsparce (TEF) — Basic, search/draw tech
    (66, 3),     # Dudunsparce (TEF) — Stage 1, evolved form
    (140, 1),    # Fezandipiti ex — draw
    (1013, 1),   # Shaymin (POR) — search tech
    (858, 1),    # Psyduck — tech
    # Items
    (1086, 4),   # Buddy-Buddy Poffin — search 2 Basic ≤70 HP (Abra 50 HP!)
    (1152, 4),   # Poké Pad
    (1225, 4),   # Touko (supporter)
    (1079, 3),   # Rare Candy — Abra → Alakazam (skip Kadabra)
    (1182, 3),   # Boss's Orders
    (1231, 3),   # Dawn
    (1264, 3),   # Battle Colosseum (stadium)
    (1081, 2),   # Regimental Hammer (trainer)
    (1184, 1),   # trainer
    (1186, 1),   # trainer
    (1129, 1),   # Sacred Ash (trainer)
    (1146, 1),   # Biwa (trainer)
    (1097, 1),   # Night Stretcher
    # Energy
    (19, 4),     # Telepathos {P} Special Energy
    (5, 3),      # Basic {P} Energy
    (13, 1),     # Rich Energy (special)
]
# Total: 4+4+4+3+3+1+1+1 + 4+4+4+3+3+3+3+2+1+1+1+1+1 + 4+3+1 = 60


# ---------------------------------------------------------------------------
# DECK_CATALOG: master registry
# ---------------------------------------------------------------------------

DECK_CATALOG: dict[str, MetaDeck] = {
    # --- Tier 1 (S) ---
    "dragapult_ex": MetaDeck(
        name="dragapult_ex",
        archetype="spread-damage",
        tier="S",
        key_pokemon_ids=[121, 119, 120, 326],  # Dragapult ex, Dreepy, Drakloak, Blaziken ex
        recipe=_DRAGAPULT_EX,
    ),
    "hydrapple_ex": MetaDeck(
        name="hydrapple_ex",
        archetype="grass-self-heal",
        tier="S",
        key_pokemon_ids=[150, 921, 149, 96, 710],  # Hydrapple ex, Dipplin, Applin, Ogerpon ex, Meganium
        recipe=_HYDRAPPLE_EX,
    ),
    "terasta_bullet": MetaDeck(
        name="terasta_bullet",
        archetype="multi-type-bullet",
        tier="S",
        key_pokemon_ids=[96, 756, 184, 108, 140],  # Ogerpon, MegaKangaskhan, Latias, Ogerpon well, Fezandipiti
        recipe=_TERASTA_BULLET,
    ),
    # --- Tier 2 (A) ---
    "ogerpon_bullet": MetaDeck(
        name="ogerpon_bullet",
        archetype="multi-type-bullet",
        tier="A",
        key_pokemon_ids=[96, 756, 112, 184, 108],  # Ogerpon みどりのめん, メガガルーラex, マシマシラ, ラティアスex, Ogerpon いどのめん
        recipe=_OGERPON_BULLET,
    ),
    "raging_bolt_ex": MetaDeck(
        name="raging_bolt_ex",
        archetype="ancient-lightning",
        tier="A",
        key_pokemon_ids=[63, 75, 96, 62],  # Raging Bolt ex, Iron Hands ex, Ogerpon ex, Koraidon
        recipe=_RAGING_BOLT_EX,
    ),
    "mega_lucario_ex": MetaDeck(
        name="mega_lucario_ex",
        archetype="fighting-mega",
        tier="A",
        key_pokemon_ids=[678, 677, 675, 676, 674],  # Mega Lucario ex, Riolu, Lunatone, Solrock, Hariyama
        recipe=_MEGA_LUCARIO_EX,
    ),
    "olivia_ex": MetaDeck(
        name="olivia_ex",
        archetype="grass-self-heal",
        tier="A",
        key_pokemon_ids=[404, 402, 403, 710, 96],  # Olivia ex, ミニーブ, オリーニョ, Meganium, Ogerpon ex
        recipe=_OLIVIA_EX,
    ),
    "crustle": MetaDeck(
        name="crustle",
        archetype="tank-disruption",
        tier="A",
        key_pokemon_ids=[345, 344, 756, 117],  # Crustle, Dwebble, MegaKangaskhan ex, Ogerpon rock
        recipe=_CRUSTLE,
    ),
    "maries_obstagoon_ex": MetaDeck(
        name="maries_obstagoon_ex",
        archetype="dark-disruption",
        tier="A",
        key_pokemon_ids=[648, 646, 647, 112, 103],  # Marnie's Obstagoon ex, Zigzagoon, Linoone, Munkidori, Snover
        recipe=_MARIES_OBSTAGOON_EX,
    ),
    # --- Tier 3 (B) ---
    "alakazam": MetaDeck(
        name="alakazam",
        archetype="hand-damage",
        tier="B",
        key_pokemon_ids=[743, 742, 741, 66, 65],  # Alakazam, Kadabra, Abra, Dudunsparce, Dunsparce
        recipe=_ALAKAZAM,
    ),
}


# ---------------------------------------------------------------------------
# Derived lookup tables (built from DECK_CATALOG)
# ---------------------------------------------------------------------------

DECK_RECIPES: dict[str, list[tuple[int, int]]] = {
    name: deck.recipe for name, deck in DECK_CATALOG.items()
}

DECK_TIERS: dict[str, str] = {
    name: deck.tier for name, deck in DECK_CATALOG.items()
}


def _build_card_to_decks() -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for deck_name, deck in DECK_CATALOG.items():
        for card_id, _ in deck.recipe:
            result.setdefault(card_id, []).append(deck_name)
    return result


CARD_TO_DECKS: dict[int, list[str]] = _build_card_to_decks()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tier_s_decks() -> list[str]:
    """Return names of all Tier 1 (S) decks."""
    return [name for name, deck in DECK_CATALOG.items() if deck.tier == "S"]


def tier_a_decks() -> list[str]:
    """Return names of all Tier 2 (A) decks."""
    return [name for name, deck in DECK_CATALOG.items() if deck.tier == "A"]


def tier_b_decks() -> list[str]:
    """Return names of all Tier 3 (B) decks."""
    return [name for name, deck in DECK_CATALOG.items() if deck.tier == "B"]


def estimate_deck_candidates(revealed_card_ids: list[int]) -> list[tuple[str, float]]:
    """Estimate opponent deck archetype from revealed card IDs.

    Uses weighted scoring: each revealed card contributes 1/N weight to each deck
    that includes it (N = number of decks sharing that card).
    Returns list of (deck_name, normalized_score) sorted by descending score.
    """
    scores: dict[str, float] = {name: 0.0 for name in DECK_CATALOG}
    for card_id in revealed_card_ids:
        decks_with_card = CARD_TO_DECKS.get(card_id, [])
        if not decks_with_card:
            continue
        weight = 1.0 / len(decks_with_card)
        for deck_name in decks_with_card:
            scores[deck_name] += weight
    max_score = max(scores.values()) if scores else 0.0
    if max_score == 0.0:
        return [(name, 0.0) for name in DECK_CATALOG]
    return sorted(
        [(name, score / max_score) for name, score in scores.items()],
        key=lambda x: -x[1],
    )
