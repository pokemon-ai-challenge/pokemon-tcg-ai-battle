"""カードIDの定数と優先順位（ユーザー仕様をそのまま定数化したもの）。

デッキ: フーライン（ケーシィ→ユンゲラー→フーディン）・ノコライン（ノコッチ→ノココッチ）・
シェイミ を中核とするフーディンのハンドパワーデッキ（sample_submission/deck.csv）。
"""

from __future__ import annotations

# --- フーライン（ケーシィ→ユンゲラー→フーディン） ---
CASEY = 741
KADABRA = 742
ALAKAZAM = 743
FUDIN_LINE: tuple[int, ...] = (CASEY, KADABRA, ALAKAZAM)

# --- ノコライン（ノコッチ→ノココッチ） ---
DUNSPARCE = 65
DUDUNSPARCE = 66
NOKO_LINE: tuple[int, ...] = (DUNSPARCE, DUDUNSPARCE)

# --- シェイミ ---
SHAYMIN = 343

# 原則ベンチに固定し、自分から場（バトル場）に出さないポケモン。
# ノココッチ: にげあしドロー要員。シェイミ: はなのカーテンでベンチを守る要員。
BENCH_ONLY: tuple[int, ...] = (DUDUNSPARCE, SHAYMIN)

# --- エネルギー ---
BASIC_PSYCHIC_ENERGY = 5
TELEPATH_ENERGY = 19
RICH_ENERGY = 13  # ACE SPEC・1枚のみ。必ずノコラインにつける。
ENERGY_CARD_IDS: tuple[int, ...] = (BASIC_PSYCHIC_ENERGY, TELEPATH_ENERGY, RICH_ENERGY)

# --- グッズ ---
RARE_CANDY = 1079
ENHANCED_HAMMER = 1081
BUDDY_BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
SACRED_ASH = 1129
WONDROUS_PATCH = 1146
POKE_PAD = 1152

# --- サポート ---
BOSS_ORDERS = 1182
LANAS_AID = 1184
XEROSICS_MACHINATIONS = 1197
TOUKO = 1225  # 進化ポケモン+エネルギーを1枚ずつサーチ（Hilda）
HIKARI = 1231  # たね+1進化+2進化を1枚ずつサーチ（Dawn）

# --- スタジアム ---
BATTLE_COLOSSEUM = 1264

# --- 各ポケモンの攻撃に必要なエネルギー総数（これ以上は付けない） ---
ENERGY_REQUIRED_COUNT: dict[int, int] = {
    CASEY: 1,
    KADABRA: 1,
    ALAKAZAM: 1,
    DUNSPARCE: 2,
    DUDUNSPARCE: 3,
    SHAYMIN: 2,
}

# --- ポケモン全体の優先順位（展開・エネルギー付け・気絶後の後継など共通） ---
# 上から優先度高: フーライン（進化段階問わず同格）→ノコッチ→ノココッチ→シェイミ。
POKEMON_PRIORITY: list[int] = [ALAKAZAM, KADABRA, CASEY, DUNSPARCE, DUDUNSPARCE, SHAYMIN]

# 場にいる/手札にある基本（たね）ポケモンのみの優先順位（夜のタンカ・なかよしポフィンで使用）。
BASIC_POKEMON_IDS: tuple[int, ...] = (CASEY, DUNSPARCE, SHAYMIN)

# 各基本ポケモンの進化先（無ければ None）。「進化先が場・手札にいない」判定に使う。
EVOLUTION_SUCCESSOR: dict[int, int] = {
    CASEY: KADABRA,
    DUNSPARCE: DUDUNSPARCE,
}

# --- 進化の優先順位（複数の進化が同時に選べる場合のタイブレーク用。進化できるなら基本的に必ず
# 進化する。743/742はどちらもフーライン系列で同格の最優先、66がその次） ---
EVOLUTION_PRIORITY: list[int] = [ALAKAZAM, KADABRA, DUDUNSPARCE]
RARE_CANDY_PRIMARY_TARGET = ALAKAZAM

# --- 気絶後・にげる時の後継優先順位 ---
KO_REPLACEMENT_PRIORITY: list[int] = [ALAKAZAM, KADABRA, CASEY, DUNSPARCE, DUDUNSPARCE, SHAYMIN]

# --- 山札を消費するドロー/サーチ系カード。手札がこの枚数以上ならこれ以上使わない ---
DECK_DRAW_STOP_HAND_SIZE = 20

# --- グッズ使用優先順位（MAINターンの5.グッズで使う） ---
ITEM_USE_PRIORITY: list[int] = [
    BUDDY_BUDDY_POFFIN,
    ENHANCED_HAMMER,
    WONDROUS_PATCH,
    NIGHT_STRETCHER,
    SACRED_ASH,
    POKE_PAD,
]

# --- サポート使用優先順位（1ターン1枚。ボスの指令は別枠で攻撃直前に判定） ---
SUPPORTER_USE_PRIORITY: list[int] = [
    XEROSICS_MACHINATIONS,
    TOUKO,
    HIKARI,
    LANAS_AID,
]

# --- 捨てたくないカード（回収手段が無いもの） ---
PROTECTED_CARD_IDS: tuple[int, ...] = (RARE_CANDY, HIKARI, BOSS_ORDERS)

# --- TO_HAND/TO_DECK/LOOK 等、どの効果由来か特定できない場合の汎用フォールバック優先順位 ---
SEARCH_FALLBACK_PRIORITY: list[int] = [
    HIKARI,
    TOUKO,
    POKE_PAD,
    BUDDY_BUDDY_POFFIN,
    RARE_CANDY,
    RICH_ENERGY,
    *POKEMON_PRIORITY,
]
