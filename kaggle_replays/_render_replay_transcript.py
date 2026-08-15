#!/usr/bin/env python3
"""Kaggle実ラダーのリプレイを、LLMが読んで戦術分析できる日本語対戦記録(Markdown)に変換する。

読み取り専用の分析ツール。cg/ data/ sample_submission/ は一切変更しない。新規ファイルは
本ファイル1本のみ(既存資産は import して再利用し、重複実装を避ける)。

--- 依拠するスキーマ理解 ---
[確実] cg/api.py の dataclass 定義(LogType/OptionType/SelectType/SelectContext/AreaType/
CardType の数値、Observation/State/PlayerState/Pokemon/Card/Option/SelectData のフィールド名)。

[検証済み・_audit_hammer_replay.py 由来] steps[k][side]['action'] は steps[k-1][side]['select']
への回答である(「1つ前行モデル」、同ファイルの docstring 参照)。own_index の解決・相手アーキ
タイプ推定はそのファイルの実装をそのまま import して再利用する(重複実装しない)。

[本ファイル固有・追加で検証した事実]
1. steps[k][own_index]['observation'] の 'logs' フィールドは「直前の own_index の"本物の"
   決定以降に起きたイベント」であり、own_index が動かない間(相手の手番中など)は行が
   進んでも logs も select も current も一切更新されず、同一内容が action=[] のまま
   繰り返される("埋め草行")。したがって「直前行の own_index 側 logs と完全一致する行は
   埋め草」として捨てて良い(remainingOverageTime が変化しないことでも同じ結論が裏取りできる、
   episode-92386848 で実測確認済み)。
2. MAIN(SelectType.MAIN)の select は必ず END 選択肢を含むため minCount=1 が常に成立し、
   「本物の決定」には必ず非空の action が対になる。したがって
   `steps[k][own_index]['select']` が MAIN で、`steps[k+1][own_index]['action']` が非空なら、
   それは埋め草ではなく実際に own_index が選んだ本物の決定である
   (deck選択・クラッシュハンマーのエネルギー選択などでも同じ隣接ペアリングが成立することを
   _audit_hammer_replay.py 側で確認済み)。
3. ある試合が「相手の連続ターンの途中で(こちらへの介入なしに)終局する」場合、own_index の
   observation ストリームには最終盤の出来事(相手の最後の攻撃など)が一切届かない
   (episode-92386848 で実測: own_index 側は turn=7 で凍結し turn=8 の相手の攻撃は見えないが、
   opp_index 側の observation には turn=8 の攻撃ログが存在する)。これは「相手の非公開情報」
   ではなく双方に見える公開ログ(攻撃・HP変化・きぜつ等)なので、own_index 側にその turn の
   ログが1件も無い場合に限り opp_index 側の同turnログで補完する(手札の中身など本当に隠れて
   いる情報には一切触れない)。補完した行動には "(相手視点ログで補完)" と明記する。

使い方:
  python _render_replay_transcript.py --ref 55464390 --ref-deck kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/01.csv --only-losses
  python _render_replay_transcript.py --ref 55464390 --episode 92386848 --ref-deck ...  # 単発検証用
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
if str(_SUB) not in sys.path:
    sys.path.insert(0, str(_SUB))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cg.api import (  # noqa: E402
    AreaType,
    CardType,
    LogType,
    OptionType,
    SelectType,
    all_attack,
    all_card_data,
)

import _audit_hammer_replay as _audit  # noqa: E402  既存の own_index解決/相手アーキ推定を再利用

CRUSHING_HAMMER_ID = 1120
JP_CSV_PATH = _ROOT / "data" / "JP_Card_Data.csv"

_STATUS_LABEL = {
    LogType.POISONED: "どく",
    LogType.BURNED: "やけど",
    LogType.ASLEEP: "ねむり",
    LogType.PARALYZED: "まひ",
    LogType.CONFUSED: "こんらん",
}

_RESULT_REASON_LABEL = {
    1: "サイド0枚",
    2: "山札切れ",
    3: "バトルポケモン不在",
    4: "カード効果",
}


# ---------------------------------------------------------------------------
# カード名/技名の日本語辞書(data/JP_Card_Data.csv 由来)
# ---------------------------------------------------------------------------

_BASIC_ENERGY_RE = re.compile(r"^基本【(.+?)】エネルギー$")


def build_jp_names(csv_path: Path = JP_CSV_PATH) -> dict[int, str]:
    """cardId -> カード名(日本語)。1カードにつき複数行(技/特性ごと)あるので先勝ちで採用。

    圧縮のため基本エネルギーは「基本【草】エネルギー」→「草エネルギー」に短縮する
    (意味は変わらない機械的な短縮。1試合あたり何十回も出てくる最頻出語のため)。
    """
    names: dict[int, str] = {}
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["カード ID"])
            except (KeyError, ValueError):
                continue
            if cid in names:
                continue
            raw = row.get("カード名") or f"id={cid}"
            m = _BASIC_ENERGY_RE.match(raw)
            names[cid] = f"{m.group(1)}エネルギー" if m else raw
    return names


def build_jp_attack_names(csv_path: Path = JP_CSV_PATH) -> dict[int, str]:
    """attackId -> ワザ名(日本語)。

    JP_Card_Data.csv 自体には attackId が無いため、cg.api.all_card_data() の
    CardData.attacks(カードごとの attackId の並び順)と、CSV 側の「実ワザ行」
    (コスト列が n/a でない行。特性行やテラスタル注記行を除外するため)をカードID単位で
    zip 対応させて構築する。行数が一致しないカード(パース対象外の特殊カード等)は
    諦めて all_attack() の英語名にフォールバックする(捏造を避けるための保険)。
    """
    cards = {c.cardId: c for c in all_card_data()}
    attacks_en = {a.attackId: a for a in all_attack()}

    rows_by_card: dict[int, list[dict]] = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["カード ID"])
            except (KeyError, ValueError):
                continue
            if (row.get("コスト") or "n/a") == "n/a":
                continue
            rows_by_card[cid].append(row)

    result: dict[int, str] = {}
    for cid, card in cards.items():
        jp_rows = rows_by_card.get(cid, [])
        if card.attacks and len(jp_rows) == len(card.attacks):
            for aid, jr in zip(card.attacks, jp_rows):
                jp_name = (jr.get("ワザ名") or "").strip()
                if jp_name and jp_name != "n/a":
                    result[aid] = jp_name

    for aid, a in attacks_en.items():
        result.setdefault(aid, a.name)
    return result


# ---------------------------------------------------------------------------
# リプレイ内のイベント列(own_index / opp_index それぞれの視点)を復元する
# ---------------------------------------------------------------------------

def extract_events(steps: list, viewer_index: int) -> tuple[dict[int, list[dict]], int]:
    """viewer_index 視点で「本物の新規ログのみ」を turn 番号ごとに束ねる。

    埋め草行(前回と logs が完全一致する行)は捨てる。TURN_START ログの出現回数で
    turn 番号を前方カウントし、行末の current.turn で毎回補正する(取りこぼし対策)。
    """
    events: dict[int, list[dict]] = defaultdict(list)
    prev_logs: list | None = None
    running_turn = 0
    for row in steps:
        obs = row[viewer_index].get("observation") or {}
        cur = obs.get("current")
        logs = obs.get("logs") or []
        if cur is None:
            continue
        if logs == prev_logs:
            continue  # 埋め草行(この viewer は動いていない)
        prev_logs = logs
        for log in logs:
            if log.get("type") == LogType.TURN_START:
                running_turn += 1
            events[running_turn].append(log)
        if cur.get("turn") is not None:
            running_turn = cur["turn"]
    return dict(events), running_turn


def build_turn_snapshots(steps: list, viewer_index: int) -> dict[int, dict]:
    """viewer_index 視点で、各 turn 番号について最初に観測できた current スナップショットを集める。"""
    result: dict[int, dict] = {}
    for row in steps:
        cur = row[viewer_index].get("observation", {}).get("current")
        if cur is not None:
            t = cur.get("turn")
            if t not in result:
                result[t] = cur
    return result


# ---------------------------------------------------------------------------
# 盤面の対象ポケモン解決 / 選択肢の日本語説明(「他候補」用)
# ---------------------------------------------------------------------------

def _pokemon_at(cur: dict, player_index: int | None, area: int | None, index: int | None) -> dict | None:
    if player_index is None:
        return None
    try:
        player = cur["players"][player_index]
    except (KeyError, IndexError, TypeError):
        return None
    if area == AreaType.ACTIVE:
        lst = player.get("active") or []
        return lst[0] if lst else None
    if area == AreaType.BENCH:
        lst = player.get("bench") or []
        return lst[index] if index is not None and 0 <= index < len(lst) else None
    return None


def describe_option(
    opt: dict,
    cur: dict,
    own_index: int,
    jp_names: dict[int, str],
    jp_attacks: dict[int, str],
    card_types: dict[int, int] | None = None,
) -> str:
    """MAIN select の1つの option を短い日本語で説明する(「他候補」欄の表示用、ベストエフォート)。"""

    def name(cid: int | None) -> str:
        return jp_names.get(cid, f"id={cid}") if cid is not None else "?"

    t = opt.get("type")
    hand = cur["players"][own_index].get("hand") or []

    if t == OptionType.PLAY:
        idx = opt.get("index")
        if idx is not None and 0 <= idx < len(hand):
            cid = hand[idx]["id"]
            if card_types is not None and card_types.get(cid) == CardType.POKEMON:
                return f"{name(cid)}を場に出す"
            return f"{name(cid)}を使用"
        return "手札を使用"
    if t == OptionType.ATTACH:
        pidx = opt.get("playerIndex") if opt.get("playerIndex") is not None else own_index
        tgt = _pokemon_at(cur, pidx, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        card_name = "カード"
        if opt.get("area") == AreaType.HAND:
            idx = opt.get("index")
            if idx is not None and 0 <= idx < len(hand):
                card_name = name(hand[idx]["id"])
        return f"{card_name}を{name(tgt['id']) if tgt else '対象不明'}に付与"
    if t == OptionType.EVOLVE:
        pidx = opt.get("playerIndex") if opt.get("playerIndex") is not None else own_index
        tgt = _pokemon_at(cur, pidx, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        card_name = "進化カード"
        if opt.get("area") == AreaType.HAND:
            idx = opt.get("index")
            if idx is not None and 0 <= idx < len(hand):
                card_name = name(hand[idx]["id"])
        return f"{name(tgt['id']) if tgt else '対象不明'}を{card_name}に進化"
    if t == OptionType.ABILITY:
        p = _pokemon_at(cur, own_index, opt.get("area"), opt.get("index"))
        return f"{name(p['id']) if p else '?'}の特性を使用"
    if t == OptionType.DISCARD:
        p = _pokemon_at(cur, own_index, opt.get("area"), opt.get("index"))
        return f"{name(p['id']) if p else '?'}をトラッシュ"
    if t == OptionType.RETREAT:
        return "にげる"
    if t == OptionType.ATTACK:
        aid = opt.get("attackId")
        return f"「{jp_attacks.get(aid, f'attackId={aid}')}」で攻撃"
    if t == OptionType.END:
        return "ターン終了"
    if t == OptionType.SKILL:
        return "処理順選択"
    return f"(選択肢type={t})"


_MAX_ALT_CANDIDATES = 3  # 1メモあたりの「他候補」表示件数上限(圧縮方針)
_MAX_ALT_NOTES_PER_TURN = 2  # 1turnあたりの選択肢メモ表示件数上限(圧縮方針)


def collect_alt_notes(
    steps: list,
    own_index: int,
    jp_names: dict[int, str],
    jp_attacks: dict[int, str],
    card_types: dict[int, int],
) -> dict[int, list[str]]:
    """MAIN選択肢が5個以上あった「本物の」own_index の決定について、turn番号ごとに
    「選んだもの + 他候補」の1行メモを集める。

    圧縮のため: (1) 他候補は重複除去のうえ上位3件まで、(2) 1turnあたりは選択肢数が多い
    (=分岐が大きい)決定を上位3件までに絞る。"""
    raw: dict[int, list[tuple[int, str]]] = defaultdict(list)  # turn -> [(nopt, text)]
    for k in range(len(steps) - 1):
        obs = steps[k][own_index].get("observation") or {}
        sel = obs.get("select")
        cur = obs.get("current")
        if sel is None or cur is None:
            continue
        if sel.get("type") != SelectType.MAIN:
            continue
        options = sel.get("option") or []
        if len(options) < 5:
            continue
        ans = steps[k + 1][own_index].get("action") or []
        if not ans:
            continue  # 埋め草行(本物の決定ではない)
        chosen_idx = ans[0]
        if not (0 <= chosen_idx < len(options)):
            continue
        alt_descs: list[str] = []
        for i, opt in enumerate(options):
            if i == chosen_idx:
                continue
            try:
                desc = describe_option(opt, cur, own_index, jp_names, jp_attacks, card_types)
            except Exception:  # noqa: BLE001 - ベストエフォートの注釈なので失敗は無視
                continue
            if desc not in alt_descs:  # 同一内容(草エネ付与の別選択肢など)は重複除去
                alt_descs.append(desc)
            if len(alt_descs) >= _MAX_ALT_CANDIDATES:
                break
        if not alt_descs:
            continue
        try:
            chosen_desc = describe_option(options[chosen_idx], cur, own_index, jp_names, jp_attacks, card_types)
        except Exception:  # noqa: BLE001
            chosen_desc = "?"
        turn = cur.get("turn")
        text = f"- 選択肢メモ(MAIN{len(options)}択, 選択={chosen_desc}): 他候補: " + ", ".join(alt_descs)
        raw[turn].append((len(options), text))

    notes: dict[int, list[str]] = {}
    for turn, items in raw.items():
        # 分岐が大きい(選択肢数が多い)決定を優先して上位3件のみ採用
        top = sorted(items, key=lambda x: -x[0])[:_MAX_ALT_NOTES_PER_TURN]
        notes[turn] = [text for _, text in top]
    return notes


# ---------------------------------------------------------------------------
# ログ列 -> 圧縮された日本語の「行動」フレーズ列
# ---------------------------------------------------------------------------

def describe_board_side(player_state: dict, jp_names: dict[int, str]) -> str:
    def name(cid: int) -> str:
        return jp_names.get(cid, f"id={cid}")

    parts: list[str] = []
    for p in player_state.get("active") or []:
        if p is None:
            parts.append("(伏せ)")
        else:
            parts.append(
                f"{name(p['id'])}(アクティブ,HP{p['hp']}/{p['maxHp']},エネ{len(p.get('energies') or [])})"
            )
    for p in player_state.get("bench") or []:
        parts.append(
            f"{name(p['id'])}(ベンチ,HP{p['hp']}/{p['maxHp']},エネ{len(p.get('energies') or [])})"
        )
    return ", ".join(parts) if parts else "なし"


def render_turn_actions(
    logs: list[dict],
    own_index: int,
    jp_names: dict[int, str],
    jp_attacks: dict[int, str],
    card_types: dict[int, int],
) -> list[str]:
    """1turn分のログ列を、意味のある行動だけの短い日本語フレーズ列に圧縮する。"""

    def name(cid: int | None) -> str:
        return jp_names.get(cid, f"id={cid}") if cid is not None else "?"

    phrases: list[str] = []
    consumed: set[int] = set()
    n = len(logs)

    for i, log in enumerate(logs):
        if i in consumed:
            continue
        t = log.get("type")
        pidx = log.get("playerIndex")
        who = "自分" if pidx == own_index else ("相手" if pidx is not None else "")

        if t == LogType.ATTACH:
            phrases.append(f"{who}: {name(log.get('cardId'))}を{name(log.get('cardIdTarget'))}に付与")

        elif t == LogType.PLAY:
            cid = log.get("cardId")
            ct = card_types.get(cid)
            # Pokemon/エネルギー/道具は ATTACH や MOVE_CARD 側の表現で足りるため PLAY 単体は出さない
            if ct in (CardType.POKEMON, CardType.TOOL, CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                continue
            text = f"{who}: {name(cid)}使用"
            if cid == CRUSHING_HAMMER_ID:
                for j in range(i + 1, min(i + 4, n)):
                    if j in consumed:
                        continue
                    lg2 = logs[j]
                    if lg2.get("type") == LogType.COIN and lg2.get("playerIndex") == pidx:
                        head = lg2.get("head")
                        text += f"(コイン:{'表' if head else '裏'})"
                        consumed.add(j)
                        if head:
                            for k2 in range(j + 1, min(j + 3, n)):
                                if k2 in consumed:
                                    continue
                                lg3 = logs[k2]
                                if (
                                    lg3.get("type") == LogType.MOVE_CARD
                                    and lg3.get("fromArea") == AreaType.ENERGY
                                    and lg3.get("toArea") == AreaType.DISCARD
                                ):
                                    text += f"→相手の{name(lg3.get('cardId'))}を1枚トラッシュ"
                                    consumed.add(k2)
                                    break
                        break
            phrases.append(text)

        elif t == LogType.EVOLVE:
            phrases.append(f"{who}: {name(log.get('cardIdTarget'))}→{name(log.get('cardId'))}に進化")

        elif t == LogType.DEVOLVE:
            phrases.append(f"{who}: {name(log.get('cardIdTarget'))}が{name(log.get('cardId'))}に退化")

        elif t == LogType.SWITCH:
            phrases.append(
                f"{who}: {name(log.get('cardIdBench'))}と{name(log.get('cardIdActive'))}を交代"
            )

        elif t == LogType.CHANGE:
            phrases.append(f"{who}: {name(log.get('cardIdBefore'))}→{name(log.get('cardIdAfter'))}")

        elif t == LogType.ATTACK:
            aid = log.get("attackId")
            aname = jp_attacks.get(aid, f"attackId={aid}")
            text = f"{who}: 攻撃「{aname}」"
            for j in range(i + 1, min(i + 3, n)):
                if j in consumed:
                    continue
                lg2 = logs[j]
                if lg2.get("type") == LogType.HP_CHANGE and (lg2.get("value") or 0) < 0:
                    text += f"({abs(lg2['value'])}ダメ)"
                    consumed.add(j)
                    break
            phrases.append(text)

        elif t == LogType.MOVE_CARD:
            frm, to = log.get("fromArea"), log.get("toArea")
            cid = log.get("cardId")
            if frm == AreaType.PRIZE and to == AreaType.HAND:
                phrases.append(f"{who}: サイドを1枚獲得")
            elif (
                to == AreaType.DISCARD
                and frm in (AreaType.ACTIVE, AreaType.BENCH)
                and card_types.get(cid) == CardType.POKEMON
            ):
                phrases.append(f"{who}: {name(cid)}きぜつ")
            elif frm == AreaType.ENERGY and to == AreaType.DISCARD:
                phrases.append(f"{who}: {name(cid)}がトラッシュ")
            elif frm == AreaType.TOOL and to == AreaType.DISCARD:
                phrases.append(f"{who}: 道具{name(cid)}がトラッシュ")
            elif frm == AreaType.BENCH and to == AreaType.ACTIVE:
                phrases.append(f"{who}: {name(cid)}をバトル場に")
            elif frm == AreaType.HAND and to in (AreaType.ACTIVE, AreaType.BENCH):
                phrases.append(f"{who}: {name(cid)}を場に出す")
            # それ以外(デッキサーチ/シャッフル等)は圧縮方針により省略

        elif t in _STATUS_LABEL:
            label = _STATUS_LABEL[t]
            if log.get("isRecover"):
                phrases.append(f"{who}: {name(log.get('cardId'))}の{label}が回復")
            else:
                phrases.append(f"{who}: {name(log.get('cardId'))}が{label}に")

        elif t == LogType.COIN:
            phrases.append(f"{who}: コイン{'表' if log.get('head') else '裏'}")

        # 以下は圧縮方針により省略: DRAW/DRAW_REVERSE/SHUFFLE/HAS_BASIC_POKEMON/
        # MOVE_CARD_REVERSE/MOVE_ATTACHED/TURN_START/TURN_END/RESULT/単独のHP_CHANGE

    return _collapse_runs(phrases)


def _collapse_runs(phrases: list[str]) -> list[str]:
    """連続して同一内容が並ぶフレーズ("草エネルギーがトラッシュ"を5連続 等)を
    "フレーズ ×N" にまとめて圧縮する(情報は保持したまま行数だけ削る)。"""
    collapsed: list[str] = []
    i = 0
    while i < len(phrases):
        j = i
        while j < len(phrases) and phrases[j] == phrases[i]:
            j += 1
        run = j - i
        collapsed.append(phrases[i] if run == 1 else f"{phrases[i]} ×{run}")
        i = j
    return collapsed


# ---------------------------------------------------------------------------
# 終局理由の推定
# ---------------------------------------------------------------------------

def infer_win_condition(
    steps: list,
    own_index: int,
    opp_index: int,
    own_events: dict[int, list[dict]],
    opp_events: dict[int, list[dict]],
    total_turns: int,
    card_types: dict[int, int],
    jp_names: dict[int, str],
) -> str:
    """終局理由をベストエフォートで推定する。断定できない場合は「詳細不明」を返す(捏造しない)。"""
    # 1. RESULT ログが両視点のどちらかに記録されていれば最優先で使う
    for viewer in (own_index, opp_index):
        for row in steps:
            for lg in row[viewer]["observation"].get("logs") or []:
                if lg.get("type") == LogType.RESULT:
                    reason = lg.get("reason")
                    return _RESULT_REASON_LABEL.get(reason, f"reason={reason}")

    # 2. 最終turnの行動列(own優先、無ければ相手視点で補完したもの)から「きぜつ」を逆順に探す
    last_turn_events = own_events.get(total_turns) or opp_events.get(total_turns) or []
    for lg in reversed(last_turn_events):
        cid = lg.get("cardId")
        if (
            lg.get("type") == LogType.MOVE_CARD
            and lg.get("toArea") == AreaType.DISCARD
            and lg.get("fromArea") in (AreaType.ACTIVE, AreaType.BENCH)
            and card_types.get(cid) == CardType.POKEMON
        ):
            return f"{jp_names.get(cid, f'id={cid}')}のきぜつによる決着"

    return "詳細不明(打ち切り)"


# ---------------------------------------------------------------------------
# 自分デッキの短い要約行
# ---------------------------------------------------------------------------

def build_deck_summary(
    ref_deck: list[int], card_types: dict[int, int], jp_names: dict[int, str], label: str
) -> str:
    counts = Counter(ref_deck)
    items = [(cid, n) for cid, n in counts.items() if card_types.get(cid) != CardType.BASIC_ENERGY]
    items.sort(key=lambda x: (-x[1], x[0]))
    top = items[:5]
    parts = [f"{jp_names.get(cid, f'id={cid}')}{n}" for cid, n in top]
    return f"{label}({'/'.join(parts)})"


# ---------------------------------------------------------------------------
# 1試合分のMarkdown組み立て
# ---------------------------------------------------------------------------

def build_transcript(
    eid: str,
    data: dict[str, Any],
    own_index: int,
    deck_summary: str,
    jp_names: dict[int, str],
    jp_attacks: dict[int, str],
    card_types: dict[int, int],
) -> str:
    opp_index = 1 - own_index
    steps = data["steps"]

    own_events, own_max_turn = extract_events(steps, own_index)
    opp_events, opp_max_turn = extract_events(steps, opp_index)
    total_turns = max(own_max_turn, opp_max_turn, 1)

    own_snaps = build_turn_snapshots(steps, own_index)
    opp_snaps = build_turn_snapshots(steps, opp_index)

    alt_notes = collect_alt_notes(steps, own_index, jp_names, jp_attacks, card_types)

    first_player = None
    for snaps in (own_snaps, opp_snaps):
        for t in sorted(snaps):
            fp = snaps[t].get("firstPlayer", -1)
            if fp is not None and fp != -1:
                first_player = fp
                break
        if first_player is not None:
            break
    if first_player is None:
        first_player = own_index  # 不明時のフォールバック(通常は発生しない)

    team_names = data.get("info", {}).get("TeamNames") or ["?", "?"]
    opp_arch = _audit._classify_opponent(data, own_index)
    rewards = data.get("rewards") or [None, None]
    own_reward = rewards[own_index] if own_index < len(rewards) else None
    won = own_reward is not None and own_reward > 0
    win_cond = infer_win_condition(
        steps, own_index, opp_index, own_events, opp_events, total_turns, card_types, jp_names
    )

    lines: list[str] = []
    result_label = "勝ち" if won else "負け"
    lines.append(
        f"# Episode {eid} | 対面: {opp_arch}({team_names[opp_index]}) | "
        f"結果: {result_label}({win_cond}) | {total_turns}ターン"
    )
    lines.append(f"自分デッキ: {deck_summary}")
    lines.append("")

    for turn in range(1, total_turns + 1):
        owner = first_player if turn % 2 == 1 else (1 - first_player)
        tag = "自分" if owner == own_index else "相手"

        snap = own_snaps.get(turn) or opp_snaps.get(turn)
        header = f"## T{turn}({tag})"
        if snap is not None:
            owner_player = snap["players"][owner]
            my_prize = len(snap["players"][own_index].get("prize") or [])
            opp_prize = len(snap["players"][opp_index].get("prize") or [])
            header += (
                f" 手札{owner_player.get('handCount')}枚 山札{owner_player.get('deckCount')} "
                f"サイド{my_prize}-{opp_prize}"
            )
        lines.append(header)

        if snap is not None:
            self_txt = describe_board_side(snap["players"][own_index], jp_names)
            opp_txt = describe_board_side(snap["players"][opp_index], jp_names)
            lines.append(f"- 場: 自分[{self_txt}] 相手[{opp_txt}]")

        events = own_events.get(turn) or []
        supplemented = False
        if not events and opp_events.get(turn):
            events = opp_events[turn]
            supplemented = True
        phrases = render_turn_actions(events, own_index, jp_names, jp_attacks, card_types)
        if phrases:
            suffix = "(相手視点ログで補完)" if supplemented else ""
            lines.append(f"- 行動{suffix}: " + " / ".join(phrases))

        for note in alt_notes.get(turn, []):
            lines.append(note)

        lines.append("")

    lines.append(f"## 終局: T{total_turns} {win_cond}")
    lines.append("")
    return "\n".join(lines)


def build_index_line(eid: str, md: str) -> str:
    first_line = md.splitlines()[0]
    tail = first_line.split("|", 1)[1].strip() if "|" in first_line else first_line
    return f"- [{eid}]({eid}.md) {tail}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kaggleリプレイを日本語の対戦記録Markdownに変換する(読み取り専用の分析ツール)。"
    )
    ap.add_argument("--ref", default="55464390", help="submission ref (_submission_episode_map.json のキー)")
    ap.add_argument("--map", default=str(_HERE / "_submission_episode_map.json"))
    ap.add_argument(
        "--ref-deck",
        default=str(_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2" / "ogerpon_teal_ex" / "01.csv"),
        help="自分側判定・デッキ要約に使う参照デッキCSV(60枚)",
    )
    ap.add_argument("--only-losses", action="store_true", help="負け試合のみレンダリングする")
    ap.add_argument("--limit", type=int, default=None, help="レンダリングする試合数の上限")
    ap.add_argument("--episode", default=None, help="特定の episode id 1件だけレンダリングする(検証用)")
    ap.add_argument("--out-dir", default=None, help="出力先(既定: kaggle_replays/transcripts/<ref>)")
    args = ap.parse_args()

    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    if args.episode is not None:
        eids = [args.episode]
    else:
        if args.ref not in mapping:
            print(f"ref {args.ref} が map に無い(未fetch?)。map内: {list(mapping.keys())}")
            sys.exit(1)
        eids = mapping[args.ref]

    ref_deck = _audit.load_ref_deck(Path(args.ref_deck))
    jp_names = build_jp_names()
    jp_attacks = build_jp_attack_names()
    card_types = {c.cardId: c.cardType for c in all_card_data()}
    deck_label = f"{Path(args.ref_deck).parent.name}/{Path(args.ref_deck).stem}"
    deck_summary = build_deck_summary(ref_deck, card_types, jp_names, deck_label)

    out_dir = Path(args.out_dir) if args.out_dir else _HERE / "transcripts" / args.ref
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped_resolve = 0
    skipped_filter = 0
    failed = 0
    index_rows: list[str] = []

    for eid in eids:
        if args.limit is not None and rendered >= args.limit:
            break
        data = _audit._load_episode(eid)
        if data is None:
            print(f"episode {eid}: リプレイファイル無し、skip", file=sys.stderr)
            continue
        own_index, reason = _audit._resolve_own_index(data, ref_deck)
        if own_index is None:
            skipped_resolve += 1
            continue
        rewards = data.get("rewards") or [None, None]
        own_reward = rewards[own_index] if own_index < len(rewards) else None
        won = own_reward is not None and own_reward > 0
        if args.only_losses and won:
            skipped_filter += 1
            continue
        try:
            md = build_transcript(eid, data, own_index, deck_summary, jp_names, jp_attacks, card_types)
        except Exception as e:  # noqa: BLE001 - 1件の失敗で全体を止めない
            print(f"episode {eid}: レンダリング失敗 ({e!r})、skip", file=sys.stderr)
            failed += 1
            continue
        (out_dir / f"{eid}.md").write_text(md, encoding="utf-8")
        index_rows.append(build_index_line(eid, md))
        rendered += 1

    index_lines = [
        f"# Transcripts index: submission {args.ref}",
        "",
        f"このrunでレンダリングした {rendered} 試合"
        f"(own_index解決失敗 {skipped_resolve} 試合 / フィルタ除外 {skipped_filter} 試合 / 失敗 {failed} 試合)",
        f"参照デッキ: {args.ref_deck}",
        "",
        *index_rows,
    ]
    (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"rendered={rendered} skipped_resolve={skipped_resolve} skipped_filter={skipped_filter} failed={failed}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
