#!/usr/bin/env python3
"""Kaggle実ラダーのリプレイから「クラッシュハンマー(cardId=1120)の無駄撃ち」を監査する。

無駄撃ちの定義(ユーザー観測に基づく): 自分のターンにクラッシュハンマーを使い、コイン成功で
相手のポケモンからエネルギーを1個剥がしたが、その**同じターン内に自分の攻撃でそのポケモンを
きぜつさせた**ケース(=エネルギーを剥がす前にそのまま倒せていたなら、剥がす意味が薄かった)。

対象: kaggle_replays/_submission_episode_map.json のキー "55464390"(mixogerpon提出)の
57エピソード。リプレイ本体は kaggle_replays/replays/episode-<id>-replay.json。

--- スキーマの確度について ---
このスクリプトが依存する Kaggle リプレイ JSON のスキーマ解釈は、以下の2段階の確度に分かれる。

[確実] cg/api.py の dataclass 定義そのもの(LogType/OptionType/SelectType/SelectContext/
AreaType の数値、Card/Pokemon/PlayerState/State のフィールド名)。これは提出コードが実際に
依存する契約なので誤りようがない。

[検証済み・本ファイル固有] steps[k][side]['action'] が「どの select への回答か」のペアリング。
本ファイルを書く過程で実測した結果、**action は同じ行(steps[k])の select ではなく、1つ前の
行(steps[k-1])の select への回答である**(= steps[k][side]['observation'] は
steps[k][side]['action'] を適用した"結果"として既に反映済みの状態を表す)。根拠:
  1. steps[1][side]['action'] が60枚デッキ(steps[0]の select=None への回答として既知/メモリ記録済み)。
  2. 実リプレイ1本(episode-92385834)の全行で「action が同じ行の select オプション数に収まるか」
     と「action が1つ前の行の select オプション数に収まるか」を突き合わせたところ、
     同一行仮説は側0/側1とも 19/23 件が範囲外(オプション数超過)になったのに対し、
     1つ前行仮説は側0/側1とも 1件のみ不一致(それが steps[1]、上記1.の特殊ケースそのもの)で
     ほぼ完全に一致した。
  なお `_analyze_attackplan_submissions.py` は同一行仮説で書かれている(コメントには
  "手動検証で確認済み"とあるが、これは before/after の current 取得ロジックについての検証で
  あり、action-select のペアリング自体は本監査で見つかった通り疑わしい可能性がある。本ファイルは
  独自に検証した「1つ前行」モデルを採用する。既存スクリプトの修正はスコープ外なので行わない)。

[推定] クラッシュハンマーの PLAY ログと COIN ログが常に隣接する(コイン結果が入る前に
別のログが挟まらない)という前提、同ターン内での「対象の消失」を常に自分の攻撃によるKOと
みなしてよいという前提(進化による serial 変化・特殊な自壊効果等は理論上ありうるが、
自分のターン中に相手が能動的にそれらを起こすことは通常ルール上できないため妥当と判断)。

読み取り専用。cg/ data/ sample_submission/ は一切変更しない。
"""

from __future__ import annotations

import argparse
import json
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

from cg.api import all_card_data, to_observation_class  # noqa: E402
from ptcg_ai.opponent_modeling import rough_predictor as rp  # noqa: E402

OUR_TEAM = "MORIOKA Tsuoi"
CRUSHING_HAMMER_ID = 1120

# --- cg/api.py の Enum 値(確実。api.py の定義をそのまま数値化) ---
LOG_TURN_START = 2
LOG_PLAY = 10
LOG_ATTACK = 15
LOG_COIN = 22

SELECT_TYPE_ENERGY = 4
SELECT_CONTEXT_DISCARD_ENERGY = 30

AREA_ACTIVE = 4
AREA_BENCH = 5

DEFAULT_REF_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2" / "ogerpon_teal_ex" / "01.csv"


def load_ref_deck(path: Path) -> list[int]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted(int(x) for x in lines)


def _card_name_map() -> dict[int, str]:
    return {c.cardId: c.name for c in all_card_data()}


def _load_episode(eid: str) -> dict[str, Any] | None:
    path = _HERE / "replays" / f"episode-{eid}-replay.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_own_index(data: dict[str, Any], ref_deck: list[int]) -> tuple[int | None, str]:
    """自分側(own_index)を決定する。

    まずチーム名(OUR_TEAM)で候補を絞り、続いて steps[1] の60枚デッキ(action)が
    参照デッキ(ogerpon_teal_ex/01.csv)と一致するかで確認する。両者(自分視点候補の相手側)も
    参照デッキと一致する場合は「デッキ内容だけでは自分側を一意に決められない」としてスキップする
    (指示に基づく仕様。チーム名だけなら一意に決まる場面でも、デッキ一致基準を優先してスキップ扱いにする)。
    """
    team_names = data.get("info", {}).get("TeamNames") or []
    if OUR_TEAM not in team_names:
        return None, "team_not_found"
    own_index = team_names.index(OUR_TEAM)
    opp_index = 1 - own_index

    steps = data["steps"]
    if len(steps) < 2:
        return None, "too_short"
    own_deck_action = steps[1][own_index].get("action") or []
    opp_deck_action = steps[1][opp_index].get("action") or []
    if len(own_deck_action) != 60 or len(opp_deck_action) != 60:
        return None, "deck_action_malformed"

    own_matches = sorted(own_deck_action) == ref_deck
    opp_matches = sorted(opp_deck_action) == ref_deck

    if own_matches and opp_matches:
        return None, "mirror_both_match_ref_deck"
    if not own_matches:
        return None, "own_deck_not_mixogerpon"
    return own_index, "ok"


def _classify_opponent(data: dict[str, Any], own_index: int) -> str:
    """`_archetype_winloss.py` の classify_opponent と同じロジック(相手アーキタイプ推定)。
    独立の throwaway 診断間でのコード共有のため、ロジックをここに複製している
    (import すると同ファイルの __main__ 実行ガード外の依存が増えるため複製を選択)。
    """
    steps = data["steps"]
    best_dt = "unknown"
    best_evidence = -1
    for k in range(len(steps)):
        obs = steps[k][own_index]["observation"]
        cur = obs.get("current")
        if cur is None:
            continue
        try:
            state = to_observation_class({"current": cur, "select": obs.get("select"), "logs": []}).current
            pred = rp.predict(state)
        except Exception:  # noqa: BLE001 - 予測失敗はunknown扱いで継続
            continue
        dt = pred.get("deck_type") or "unknown"
        ev = pred.get("evidence_count", 0) or 0
        if dt != "unknown" and ev > best_evidence:
            best_evidence = ev
            best_dt = dt
    return best_dt


def _own_turn_numbers(steps: list, own_index: int) -> set[int]:
    """TURN_START(playerIndex=own_index)ログが現れた turn 値の集合 = 自分のターン数。"""
    turns: set[int] = set()
    for row in steps:
        obs = row[own_index].get("observation") or {}
        cur = obs.get("current")
        logs = obs.get("logs") or []
        if cur is None:
            continue
        for log in logs:
            if log.get("type") == LOG_TURN_START and log.get("playerIndex") == own_index:
                turns.add(cur.get("turn"))
    return turns


def _find_hammer_plays(steps: list, own_index: int) -> list[dict[str, Any]]:
    """自分が実際にプレイしたクラッシュハンマーのイベントを、行インデックス付きで抽出する。

    obs.logs は「前回の自分の選択以降のイベント」で、自分が行動しない間の行では同じ内容が
    繰り返し現れる(idle side の観測は最新の自分の意思決定時点のまま更新されない)ため、
    (serial, playerIndex) で重複排除して「最初に登場した行」だけを採用する。
    """
    seen_serials: set[int] = set()
    plays: list[dict[str, Any]] = []
    for k, row in enumerate(steps):
        obs = row[own_index].get("observation") or {}
        logs = obs.get("logs") or []
        for i, log in enumerate(logs):
            if log.get("type") != LOG_PLAY:
                continue
            if log.get("cardId") != CRUSHING_HAMMER_ID:
                continue
            if log.get("playerIndex") != own_index:
                continue
            serial = log.get("serial")
            if serial in seen_serials:
                continue
            seen_serials.add(serial)
            # コイン結果ログは同じ logs リスト内で PLAY の直後に来る想定(推定: 自動解決のため
            # 他カードのログが割り込まない)。念のため PLAY より後ろの範囲全体から type=22 を探す。
            head = None
            for log2 in logs[i + 1:]:
                if log2.get("type") == LOG_COIN and log2.get("playerIndex") == own_index:
                    head = log2.get("head")
                    break
            plays.append({"row": k, "serial": serial, "head": head})
    return plays


def _resolve_target(steps: list, own_index: int, opp_index: int, play_row: int) -> dict[str, Any]:
    """コイン成功時、剥がし対象のポケモンを特定する。

    その follow-up select(ENERGY/DISCARD_ENERGY)は play_row と同じ行の observation.select に
    現れる。その回答は「1つ前行モデル」により steps[play_row+1] の own_index 側 action に入る。
    対象ポケモンの実体(id/serial)は play_row 時点(まだ剥がされる前)の current から解決する。
    """
    obs = steps[play_row][own_index]["observation"]
    sel = obs.get("select")
    if sel is None or sel.get("type") != SELECT_TYPE_ENERGY or sel.get("context") != SELECT_CONTEXT_DISCARD_ENERGY:
        return {"zone": "unexpected_select_schema", "pokemon": None}

    options = sel.get("option") or []
    if play_row + 1 >= len(steps):
        return {"zone": "unresolved_no_next_row", "pokemon": None}
    answer = steps[play_row + 1][own_index].get("action") or []
    if len(answer) != 1 or not (0 <= answer[0] < len(options)):
        return {"zone": "unresolved_action_mismatch", "pokemon": None}

    chosen = options[answer[0]]
    area = chosen.get("area")
    idx = chosen.get("index")
    cur = obs.get("current")
    opp_player = cur["players"][opp_index]

    if area == AREA_ACTIVE:
        pokemon = opp_player["active"][0] if opp_player.get("active") else None
        zone = "active"
    elif area == AREA_BENCH:
        bench = opp_player.get("bench") or []
        pokemon = bench[idx] if idx is not None and 0 <= idx < len(bench) else None
        zone = "bench"
    else:
        pokemon = None
        zone = f"unknown_area_{area}"

    return {"zone": zone, "pokemon": pokemon}


def _scan_same_turn_outcome(
    steps: list, own_index: int, opp_index: int, play_row: int, target_serial: int, hammer_turn: int, zone: str
) -> dict[str, Any]:
    """同ターン内でのKO判定と、次ラウンドでの対象自主退場(参考カウント)を調べる。

    vacated_next_round(参考カウント)は「対象がアクティブから自主退場したか」を見る指標であり、
    対象がそもそもベンチだった場合は意味を成さない(ベンチのポケモンは元々アクティブではないため、
    「次ラウンドの相手アクティブが対象と別」はベンチ対象では常にほぼ真になり無意味)。
    そのため zone!="active" のときは None(該当なし)を返す。
    """
    attacked_this_turn = False
    attacker_card_id: int | None = None
    same_turn_ko = False
    last_row_same_turn = play_row
    target_present_at_turn_end = True

    for k in range(play_row + 1, len(steps)):
        obs = steps[k][own_index].get("observation") or {}
        cur = obs.get("current")
        if cur is None:
            continue
        if cur.get("turn") != hammer_turn:
            break
        last_row_same_turn = k

        logs = obs.get("logs") or []
        for log in logs:
            if log.get("type") == LOG_ATTACK and log.get("playerIndex") == own_index:
                attacked_this_turn = True
                attacker_card_id = log.get("cardId")

        opp_player = cur["players"][opp_index]
        present = {
            p["serial"]
            for p in [*(opp_player.get("active") or []), *(opp_player.get("bench") or [])]
            if p is not None
        }
        if target_serial not in present:
            target_present_at_turn_end = False
            if attacked_this_turn:
                same_turn_ko = True

    vacated_next_round: bool | None = None
    if zone == "active" and not same_turn_ko and target_present_at_turn_end:
        for k in range(last_row_same_turn + 1, len(steps)):
            obs = steps[k][own_index].get("observation") or {}
            cur = obs.get("current")
            if cur is None:
                continue
            if cur.get("turn", -1) <= hammer_turn:
                continue
            opp_player = cur["players"][opp_index]
            active_list = opp_player.get("active") or []
            cur_active = active_list[0] if active_list else None
            vacated_next_round = cur_active is None or cur_active.get("serial") != target_serial
            break

    return {
        "attacked_this_turn": attacked_this_turn,
        "attacker_card_id": attacker_card_id,
        "same_turn_ko": same_turn_ko,
        "target_present_at_turn_end": target_present_at_turn_end,
        "vacated_next_round": vacated_next_round,
    }


def analyze_episode(eid: str, ref_deck: list[int]) -> dict[str, Any] | None:
    data = _load_episode(eid)
    if data is None:
        return {"episode": eid, "skip_reason": "replay_missing"}

    own_index, reason = _resolve_own_index(data, ref_deck)
    if own_index is None:
        return {"episode": eid, "skip_reason": reason}
    opp_index = 1 - own_index
    steps = data["steps"]

    own_turns = _own_turn_numbers(steps, own_index)
    hammer_plays_raw = _find_hammer_plays(steps, own_index)

    events: list[dict[str, Any]] = []
    for play in hammer_plays_raw:
        row = play["row"]
        obs = steps[row][own_index]["observation"]
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        head = play["head"]

        if head is False:
            events.append({
                "turn": turn, "zone": "coin_fail", "target_id": None, "target_serial": None,
                "same_turn_ko": False, "vacated_next_round": None, "attacker_card_id": None,
            })
            continue
        if head is None:
            events.append({
                "turn": turn, "zone": "coin_result_unresolved", "target_id": None, "target_serial": None,
                "same_turn_ko": False, "vacated_next_round": None, "attacker_card_id": None,
            })
            continue

        target = _resolve_target(steps, own_index, opp_index, row)
        pokemon = target["pokemon"]
        if pokemon is None:
            events.append({
                "turn": turn, "zone": target["zone"], "target_id": None, "target_serial": None,
                "same_turn_ko": False, "vacated_next_round": None, "attacker_card_id": None,
            })
            continue

        outcome = _scan_same_turn_outcome(
            steps, own_index, opp_index, row, pokemon["serial"], turn, target["zone"]
        )
        events.append({
            "turn": turn,
            "zone": target["zone"],
            "target_id": pokemon.get("id"),
            "target_serial": pokemon.get("serial"),
            "same_turn_ko": outcome["same_turn_ko"],
            "vacated_next_round": outcome["vacated_next_round"],
            "attacker_card_id": outcome["attacker_card_id"],
        })

    opp_arch = _classify_opponent(data, own_index)
    rewards = data.get("rewards") or [None, None]
    own_reward = rewards[own_index] if own_index < len(rewards) else None
    won = own_reward is not None and own_reward > 0

    return {
        "episode": eid,
        "skip_reason": None,
        "own_index": own_index,
        "opponent_archetype": opp_arch,
        "won": won,
        "own_turns": len(own_turns),
        "hammer_events": events,
    }


def summarize(results: list[dict[str, Any]], name_map: dict[int, str]) -> dict[str, Any]:
    analyzed = [r for r in results if r.get("skip_reason") is None]
    skipped = [r for r in results if r.get("skip_reason") is not None]
    skip_reasons = Counter(r["skip_reason"] for r in skipped)

    total_turns = sum(r["own_turns"] for r in analyzed)
    all_events = [(r, ev) for r in analyzed for ev in r["hammer_events"]]

    zone_counts = Counter(ev["zone"] for _, ev in all_events)
    same_turn_ko_events = [(r, ev) for r, ev in all_events if ev["same_turn_ko"]]
    vacated_next_round_events = [
        (r, ev) for r, ev in all_events
        if not ev["same_turn_ko"] and ev.get("vacated_next_round") is True
    ]

    by_archetype: dict[str, dict[str, int]] = defaultdict(lambda: {
        "games": 0, "hammer_used_games": 0, "hammer_plays": 0, "same_turn_ko": 0,
    })
    for r in analyzed:
        arch = r["opponent_archetype"]
        by_archetype[arch]["games"] += 1
        if r["hammer_events"]:
            by_archetype[arch]["hammer_used_games"] += 1
        by_archetype[arch]["hammer_plays"] += len(r["hammer_events"])
        by_archetype[arch]["same_turn_ko"] += sum(1 for ev in r["hammer_events"] if ev["same_turn_ko"])

    examples = []
    for r, ev in same_turn_ko_events[:5]:
        target_name = name_map.get(ev["target_id"], f"id={ev['target_id']}")
        attacker_name = name_map.get(ev["attacker_card_id"], f"id={ev['attacker_card_id']}")
        zone_label = "アクティブ" if ev["zone"] == "active" else "ベンチ"
        examples.append({
            "episode": r["episode"],
            "turn": ev["turn"],
            "description": (
                f"T{ev['turn']}: {zone_label}の{target_name}にハンマー"
                f"→同ターン{attacker_name}の攻撃でKO"
            ),
        })

    return {
        "games_total": len(results),
        "games_analyzed": len(analyzed),
        "games_skipped": len(skipped),
        "skip_reasons": dict(skip_reasons),
        "own_turns_total": total_turns,
        "hammer_plays_total": len(all_events),
        "hammer_target_zone_breakdown": dict(zone_counts),
        "same_turn_ko_waste_total": len(same_turn_ko_events),
        "reference_vacated_next_round_total": len(vacated_next_round_events),
        "games_with_hammer_play": sum(1 for r in analyzed if r["hammer_events"]),
        "by_opponent_archetype": dict(by_archetype),
        "waste_examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="実ラダーのリプレイからクラッシュハンマーの無駄撃ち(同ターンKOと重複)を監査する。"
    )
    ap.add_argument("--ref", default="55464390", help="submission ref (kaggle_replays/_submission_episode_map.json のキー)")
    ap.add_argument("--map", default=str(_HERE / "_submission_episode_map.json"))
    ap.add_argument("--ref-deck", default=str(DEFAULT_REF_DECK), help="自分側判定に使う参照デッキCSV(60枚)")
    ap.add_argument("--out", default=None, help="集計結果の出力先(既定: kaggle_replays/_audit_hammer_replay_<ref>.json)")
    args = ap.parse_args()

    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    if args.ref not in mapping:
        print(f"ref {args.ref} が map に無い(未fetch?)。map内: {list(mapping.keys())}")
        sys.exit(1)
    eids = mapping[args.ref]

    ref_deck = load_ref_deck(Path(args.ref_deck))
    name_map = _card_name_map()

    results = [analyze_episode(eid, ref_deck) for eid in eids]
    summary = summarize(results, name_map)
    summary["ref"] = args.ref

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = Path(args.out) if args.out else _HERE / f"_audit_hammer_replay_{args.ref}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "per_episode": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
