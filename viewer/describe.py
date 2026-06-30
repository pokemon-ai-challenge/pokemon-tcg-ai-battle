"""Japanese presentation layer for the viewer.

Turns the engine's two data shapes into ONE uniform "view" dict that the
frontend renders:

  * AI vs AI : god-view snapshots from cg.game.visualize_data() -- enums are
    strings, cards carry a `name`, both hands are face up.
  * Human vs AI : the live agent observation from cg.game (fog of war) -- enums
    are ints, cards have no `name`, opponent hand is None.

To hide those differences, every enum is normalised to its UPPER_SNAKE key
(`_enum_key`) and card names are resolved from the embedded name first, then the
JP card database, then the engine's English name.
"""

from __future__ import annotations

import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SAMPLE = os.path.join(_REPO, "sample_submission")
if _SAMPLE not in sys.path:
    sys.path.insert(0, _SAMPLE)

from cg.api import (  # noqa: E402
    AreaType,
    EnergyType,
    OptionType,
    SelectContext,
    SelectType,
    SpecialConditionType,
    all_attack,
    all_card_data,
)

_JP_CARD_CSV = os.path.join(_REPO, "data", "JP_Card_Data.csv")


# --------------------------------------------------------------------------- #
# Static databases (loaded once)
# --------------------------------------------------------------------------- #
_CARD_DB = None
_ATTACK_DB = None
_JP_NAMES = None


def _card_db():
    global _CARD_DB
    if _CARD_DB is None:
        _CARD_DB = {c.cardId: c for c in all_card_data()}
    return _CARD_DB


def _attack_db():
    global _ATTACK_DB
    if _ATTACK_DB is None:
        _ATTACK_DB = {a.attackId: a for a in all_attack()}
    return _ATTACK_DB


def _jp_names() -> dict[int, str]:
    """card id -> Japanese name, parsed from data/JP_Card_Data.csv."""
    global _JP_NAMES
    if _JP_NAMES is not None:
        return _JP_NAMES
    names: dict[int, str] = {}
    try:
        with open(_JP_CARD_CSV, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    cid = int(row[0])
                except ValueError:
                    continue
                name = (row[1] or "").strip()
                if name and cid not in names:
                    names[cid] = name
    except OSError:
        pass
    _JP_NAMES = names
    return names


def card_name(card_id, embedded: str | None = None) -> str:
    """Best display name for a card id: JP name > embedded EN > engine EN > id."""
    jp = _jp_names().get(card_id)
    if jp:
        return jp
    if embedded:
        return embedded
    c = _card_db().get(card_id)
    if c is not None and c.name:
        return c.name
    return f"#{card_id}"


# --------------------------------------------------------------------------- #
# Enum normalisation (int OR engine PascalCase string -> UPPER_SNAKE key)
# --------------------------------------------------------------------------- #
def _enum_key(val, enum_cls) -> str | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        try:
            return enum_cls(val).name
        except Exception:
            return None
    s = str(val)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).upper()


ENERGY_JP = {
    "COLORLESS": "無", "GRASS": "草", "FIRE": "炎", "WATER": "水",
    "LIGHTNING": "雷", "PSYCHIC": "超", "FIGHTING": "闘", "DARKNESS": "悪",
    "METAL": "鋼", "DRAGON": "竜", "RAINBOW": "虹", "TEAM_ROCKET": "RR",
}

CONDITION_JP = {
    "POISON": "どく", "BURN": "やけど", "SLEEP": "ねむり",
    "PARALYZE": "まひ", "CONFUSE": "こんらん",
}

CONTEXT_JP = {
    "MAIN": "メイン", "SETUP_ACTIVE_POKEMON": "バトル場のポケモンを選ぶ",
    "SETUP_BENCH_POKEMON": "ベンチに出すポケモンを選ぶ", "SWITCH": "入れ替えるポケモンを選ぶ",
    "TO_ACTIVE": "バトル場に出す", "TO_BENCH": "ベンチに出す", "TO_FIELD": "場に出す",
    "TO_HAND": "手札に加える", "DISCARD": "トラッシュする", "TO_DECK": "山札に戻す",
    "TO_DECK_BOTTOM": "山札の下に戻す", "TO_PRIZE": "サイドに置く", "NOT_MOVE": "残すカードを選ぶ",
    "DAMAGE_COUNTER": "ダメカンを置く", "DAMAGE_COUNTER_ANY": "ダメカンを好きに置く",
    "DAMAGE": "ダメージを与える", "REMOVE_DAMAGE_COUNTER": "ダメカンを取り除く",
    "HEAL": "回復する", "EVOLVES_FROM": "進化元を選ぶ", "EVOLVES_TO": "進化先を選ぶ",
    "DEVOLVE": "退化させる", "ATTACH_FROM": "つける先のポケモンを選ぶ",
    "ATTACH_TO": "つけるカードを選ぶ", "DETACH_FROM": "外すポケモンを選ぶ",
    "LOOK": "見るカードを選ぶ", "EFFECT_TARGET": "効果の対象を選ぶ",
    "DISCARD_ENERGY_CARD": "エネルギーをトラッシュ", "DISCARD_TOOL_CARD": "どうぐをトラッシュ",
    "SWITCH_ENERGY_CARD": "エネルギーを付け替える", "DISCARD_CARD_OR_ATTACHED_CARD": "トラッシュする",
    "DISCARD_ENERGY": "エネルギーをトラッシュ", "TO_HAND_ENERGY": "エネルギーを手札に戻す",
    "TO_DECK_ENERGY": "エネルギーを山札に戻す", "SWITCH_ENERGY": "エネルギーを入れ替える",
    "SKILL_ORDER": "効果の発動順を選ぶ", "ATTACK": "ワザを選ぶ", "DISABLE_ATTACK": "ワザを封じる",
    "EVOLVE": "進化させる", "DRAW_COUNT": "引く枚数を選ぶ", "DAMAGE_COUNTER_COUNT": "ダメカンの数を選ぶ",
    "REMOVE_DAMAGE_COUNTER_COUNT": "取り除くダメカンの数を選ぶ", "IS_FIRST": "先攻にしますか？",
    "MULLIGAN": "引き直しますか？", "ACTIVATE": "効果を使いますか？", "FIRST_EFFECT": "最初の効果を選びますか？",
    "MORE_DEVOLVE": "さらに退化させますか？", "COIN_HEAD": "オモテを選びますか？",
    "AFFECT_SPECIAL_CONDITION": "状態異常を選ぶ", "RECOVER_SPECIAL_CONDITION": "回復する状態異常を選ぶ",
}


# --------------------------------------------------------------------------- #
# Card / pokemon resolution from a players array
# --------------------------------------------------------------------------- #
_AREA = {int(a): a.name for a in AreaType}


def _area_list(player: dict, area_key: str):
    if player is None or area_key is None:
        return []
    return {
        "HAND": player.get("hand") or [],
        "ACTIVE": player.get("active") or [],
        "BENCH": player.get("bench") or [],
        "DISCARD": player.get("discard") or [],
        "DECK": player.get("deck") or [],
        "PRIZE": player.get("prize") or [],
    }.get(area_key, [])


def _card_at(players, pi, area, idx):
    """Return (card_id, name) at players[pi].<area>[idx], or (None, None)."""
    area_key = _enum_key(area, AreaType)
    if pi is None or not (0 <= pi < len(players)):
        return None, None
    lst = _area_list(players[pi], area_key)
    if idx is None or not (0 <= idx < len(lst)):
        return None, None
    card = lst[idx]
    if card is None:
        return None, None
    cid = card.get("id")
    return cid, card_name(cid, card.get("name"))


def _pokemon_name(players, pi, area, idx) -> str:
    cid, name = _card_at(players, pi, area, idx)
    return name or "ポケモン"


# --------------------------------------------------------------------------- #
# Option -> Japanese label
# --------------------------------------------------------------------------- #
def describe_option(opt: dict, players, actor_index: int, context_key: str | None) -> dict:
    """Return {label, kind} describing a single option in Japanese."""
    kind = _enum_key(opt.get("type"), OptionType) or "?"
    pi = opt.get("playerIndex")
    pi = actor_index if pi is None else pi

    def res(area, idx, default="カード"):
        cid, name = _card_at(players, pi, area, idx)
        return name or default

    if kind == "PLAY":
        cid, name = _card_at(players, actor_index, AreaType.HAND, opt.get("index"))
        verb = "出す"
        c = _card_db().get(cid) if cid is not None else None
        if c is not None and int(c.cardType) not in (0,):  # not a Pokemon
            verb = "使う"
        return {"label": f"『{name or 'カード'}』を{verb}", "kind": kind}

    if kind == "ATTACH":
        what = res(opt.get("area"), opt.get("index"), "カード")
        target = _pokemon_name(players, pi, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        return {"label": f"{what}を{target}につける", "kind": kind}

    if kind == "EVOLVE":
        evo = res(opt.get("area"), opt.get("index"), "進化カード")
        target = _pokemon_name(players, pi, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        return {"label": f"{target}を『{evo}』に進化", "kind": kind}

    if kind == "ABILITY":
        who = res(opt.get("area"), opt.get("index"), "ポケモン")
        return {"label": f"特性：{who}", "kind": kind}

    if kind == "ATTACK":
        atk = _attack_db().get(opt.get("attackId"))
        if atk is not None:
            dmg = f"（{atk.damage}）" if atk.damage else ""
            return {"label": f"ワザ：{atk.name}{dmg}", "kind": kind}
        return {"label": "ワザを使う", "kind": kind}

    if kind == "RETREAT":
        return {"label": "にげる", "kind": kind}
    if kind == "END":
        return {"label": "ターン終了", "kind": kind}
    if kind == "YES":
        return {"label": "はい", "kind": kind}
    if kind == "NO":
        return {"label": "いいえ", "kind": kind}

    if kind in ("CARD",):
        name = res(opt.get("area"), opt.get("index"))
        prefix = CONTEXT_JP.get(context_key or "", "")
        if prefix and prefix not in ("メイン",):
            return {"label": f"{prefix}：{name}", "kind": kind}
        return {"label": name, "kind": kind}

    if kind in ("TOOL_CARD", "ENERGY_CARD"):
        target = _pokemon_name(players, pi, opt.get("area"), opt.get("index"))
        return {"label": f"{target} のカードを選択", "kind": kind}

    if kind == "ENERGY":
        target = _pokemon_name(players, pi, opt.get("area"), opt.get("index"))
        return {"label": f"{target} のエネルギー", "kind": kind}

    if kind == "NUMBER":
        return {"label": f"{opt.get('number')}", "kind": kind}

    if kind == "SKILL":
        cid = opt.get("cardId")
        if cid:
            return {"label": f"効果：{card_name(cid)}", "kind": kind}
        return {"label": "状態異常の処理", "kind": kind}

    if kind == "SPECIAL_CONDITION":
        sc = _enum_key(opt.get("specialConditionType"), SpecialConditionType)
        return {"label": CONDITION_JP.get(sc, "状態異常"), "kind": kind}

    return {"label": kind, "kind": kind}


# --------------------------------------------------------------------------- #
# Log -> Japanese line
# --------------------------------------------------------------------------- #
def describe_log(log: dict) -> str | None:
    from cg.api import LogType
    t = _enum_key(log.get("type"), LogType)
    p = log.get("playerIndex")
    who = f"P{p}" if p is not None else ""
    cid = log.get("cardId")
    nm = card_name(cid) if cid else ""

    if t == "DRAW":
        return f"{who}: {nm or 'カード'} を引いた"
    if t == "DRAW_REVERSE":
        return f"{who}: 山札からカードを引いた"
    if t == "PLAY":
        return f"{who}: {nm} をプレイ"
    if t == "ATTACH":
        tgt = card_name(log.get("cardIdTarget")) if log.get("cardIdTarget") else ""
        return f"{who}: {nm} を {tgt} につけた"
    if t == "EVOLVE":
        tgt = card_name(log.get("cardIdTarget")) if log.get("cardIdTarget") else ""
        return f"{who}: {tgt} を {nm} に進化"
    if t == "ATTACK":
        atk = _attack_db().get(log.get("attackId"))
        an = atk.name if atk else "ワザ"
        return f"{who}: {nm} の {an}"
    if t == "HP_CHANGE":
        v = log.get("value")
        if v is not None and v < 0:
            return f"{who}: {nm} に {abs(v)} ダメージ"
        if v:
            return f"{who}: {nm} のHPが {v} 回復"
        return None
    if t in ("POISONED", "BURNED", "ASLEEP", "PARALYZED", "CONFUSED"):
        jp = {"POISONED": "どく", "BURNED": "やけど", "ASLEEP": "ねむり",
              "PARALYZED": "まひ", "CONFUSED": "こんらん"}[t]
        if log.get("isRecover"):
            return f"{who}: {jp} が回復"
        return f"{who}: {nm or '相手'} が {jp}"
    if t == "SWITCH":
        return f"{who}: ポケモンを入れ替えた"
    if t == "COIN":
        return f"{who}: コイン {'オモテ' if log.get('head') else 'ウラ'}"
    if t == "TURN_START":
        return f"--- P{p} のターン開始 ---"
    if t == "RESULT":
        r = log.get("result")
        if r == 2:
            return "=== 引き分け ==="
        return f"=== P{r} の勝ち ==="
    return None


# --------------------------------------------------------------------------- #
# Pokemon / player normalisation
# --------------------------------------------------------------------------- #
def _norm_pokemon(poke: dict) -> dict | None:
    if poke is None:
        return None
    cid = poke.get("id")
    c = _card_db().get(cid)
    stage = "basic"
    ex = mega = False
    if c is not None:
        if c.stage2:
            stage = "stage2"
        elif c.stage1:
            stage = "stage1"
        ex = bool(c.ex)
        mega = bool(c.megaEx)
    tools = poke.get("tools") or []
    return {
        "id": cid,
        "name": card_name(cid, poke.get("name")),
        "hp": poke.get("hp", 0),
        "maxHp": poke.get("maxHp", 0) or poke.get("hp", 0),
        "energies": [int(e) for e in (poke.get("energies") or [])],
        "toolCount": len(tools),
        "tools": [card_name(t.get("id"), t.get("name")) for t in tools],
        "stage": stage,
        "ex": ex,
        "megaEx": mega,
        "appearThisTurn": bool(poke.get("appearThisTurn")),
    }


def _norm_hand(player: dict, show: bool) -> list | None:
    if not show:
        return None
    hand = player.get("hand")
    if hand is None:
        return None
    out = []
    for card in hand:
        cid = card.get("id")
        c = _card_db().get(cid)
        ctype = int(c.cardType) if c is not None else -1
        out.append({"id": cid, "name": card_name(cid, card.get("name")), "cardType": ctype})
    return out


def _norm_player(player: dict, index: int, label: str, show_hand: bool) -> dict:
    prize = player.get("prize") or []
    return {
        "index": index,
        "label": label,
        "active": _norm_pokemon(player["active"][0]) if player.get("active") else None,
        "bench": [_norm_pokemon(p) for p in (player.get("bench") or [])],
        "benchMax": player.get("benchMax", 5),
        "deckCount": player.get("deckCount", 0),
        "discardCount": len(player.get("discard") or []),
        "handCount": player.get("handCount", len(player.get("hand") or [])),
        "hand": _norm_hand(player, show_hand),
        "prizeTotal": len(prize),
        "status": {
            "poisoned": bool(player.get("poisoned")),
            "burned": bool(player.get("burned")),
            "asleep": bool(player.get("asleep")),
            "paralyzed": bool(player.get("paralyzed")),
            "confused": bool(player.get("confused")),
        },
    }


def _stadium(current: dict):
    st = current.get("stadium") or []
    if st:
        cid = st[0].get("id")
        return {"id": cid, "name": card_name(cid, st[0].get("name"))}
    return None


def _options_block(select: dict, players, actor_index: int, selected) -> list:
    if not select:
        return []
    ctx = _enum_key(select.get("context"), SelectContext)
    sel_set = set(selected or [])
    out = []
    for i, opt in enumerate(select.get("option") or []):
        d = describe_option(opt, players, actor_index, ctx)
        out.append({"idx": i, "label": d["label"], "kind": d["kind"],
                    "selected": i in sel_set})
    return out


def _select_block(select: dict):
    if not select:
        return None
    ctx = _enum_key(select.get("context"), SelectContext)
    return {
        "type": _enum_key(select.get("type"), SelectType),
        "context": ctx,
        "contextLabel": CONTEXT_JP.get(ctx or "", ctx or ""),
        "minCount": select.get("minCount", 0),
        "maxCount": select.get("maxCount", 0),
    }


def _logs_block(logs) -> list:
    out = []
    for lg in logs or []:
        line = describe_log(lg)
        if line:
            out.append(line)
    return out


# --------------------------------------------------------------------------- #
# Public: build a uniform view
# --------------------------------------------------------------------------- #
def build_view(snapshot: dict, labels: list[str], *, god_view: bool,
               bottom_index: int = 0, human_index: int | None = None) -> dict:
    """Normalise one engine snapshot/observation into the frontend view schema.

    snapshot: {select, logs, current, selected?} (visualize_data element OR live obs)
    labels:   [player0_label, player1_label]
    god_view: True => both hands shown; False => only human_index's hand shown.
    """
    current = snapshot.get("current") or {}
    players_raw = current.get("players") or [{}, {}]
    select = snapshot.get("select")
    selected = snapshot.get("selected")
    acting = current.get("yourIndex")

    players = []
    for i, praw in enumerate(players_raw):
        if god_view:
            show_hand = True
        else:
            show_hand = (human_index is not None and i == human_index)
        players.append(_norm_player(praw, i, labels[i] if i < len(labels) else f"P{i}",
                                    show_hand))

    actor_index = acting if acting is not None else 0
    return {
        "turn": current.get("turn", 0),
        "actingIndex": acting if select else None,
        "result": current.get("result", -1),
        "firstPlayer": current.get("firstPlayer", -1),
        "godView": god_view,
        "bottomIndex": bottom_index,
        "players": players,
        "stadium": _stadium(current),
        "select": _select_block(select),
        "options": _options_block(select, players_raw, actor_index, selected),
        "logs": _logs_block(snapshot.get("logs")),
    }
