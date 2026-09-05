"""Kaggle実戦リプレイからattack_plan v0が絡む攻撃(ATTACK選択)を抽出し、
- 直接ダメージが0だった回数(残存する「無意味な攻撃」の有無)
- 勝敗との相関
を集計する(throwaway diagnostic)。

対象: 54883836(publicScore 596.0) / 54883922(749.8) の2提出(同一tarball、attack_plan v0 ON)。
コードは同一なので、この2つはA/Bではなく単純にプール対象として扱う。
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).parent
OUR_TEAM = "MORIOKA Tsuoi"
SELECT_TYPE_MAIN = 0
OPTION_TYPE_ATTACK = 13
STATUS_FIELDS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def _all_pokemon(player: dict) -> dict[int, dict]:
    out = {}
    for poke in [*player["active"], *player["bench"]]:
        if poke is not None:
            out[poke["serial"]] = poke
    return out


def _pokemon_list(entries: list) -> dict[int, dict]:
    return {poke["serial"]: poke for poke in entries if poke is not None}


def _hp_loss_list(before_entries: list, after_entries: list) -> int:
    before = _pokemon_list(before_entries)
    after = _pokemon_list(after_entries)
    total = 0
    for serial, poke in before.items():
        if serial in after:
            total += max(0, poke["hp"] - after[serial]["hp"])
    return total


def _energy_removed(before_player: dict, after_player: dict) -> bool:
    old = {c["serial"] for poke in _all_pokemon(before_player).values() for c in poke["energyCards"]}
    new = {c["serial"] for poke in _all_pokemon(after_player).values() for c in poke["energyCards"]}
    return bool(old - new)


def analyze_episode(path: Path, our_index: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data["steps"]
    rewards = data["rewards"]
    won = rewards[our_index] is not None and rewards[our_index] > 0

    opp_index = 1 - our_index
    attacks = []
    for i, step in enumerate(steps[:-1]):
        entry = step[our_index]
        action = entry.get("action") or []
        if len(action) != 1 or not isinstance(action[0], int):
            continue
        select = entry["observation"].get("select")
        if not select or select.get("type") != SELECT_TYPE_MAIN:
            continue
        options = select.get("option") or []
        idx = action[0]
        if idx < 0 or idx >= len(options) or options[idx].get("type") != OPTION_TYPE_ATTACK:
            continue

        before_state = entry["observation"]["current"]
        # リプレイ形式は「自分の手番でない側」のobservationがそのプレイヤーの直近の
        # 実際の意思決定時点のまま更新されない(コピーが引き継がれる)ため、次ステップを
        # 単純に見ると古い値を読んでしまう(手動検証で確認済み)。正しい「攻撃解決後」の
        # 状態は、相手側(opp_index)が次に実際に行動する(action非空になる)ステップまで
        # スキャンして取得する必要がある。相手が行動する前に決着した場合は result!=-1 の
        # ステップを使う。
        after_state = None
        for j in range(i + 1, len(steps)):
            opp_entry = steps[j][opp_index]
            cur = opp_entry["observation"].get("current")
            if cur is None:
                continue
            if opp_entry.get("action") or cur.get("result", -1) != -1:
                after_state = cur
                break
        if after_state is None:
            continue

        before_me, before_opp = before_state["players"][our_index], before_state["players"][opp_index]
        after_me, after_opp = after_state["players"][our_index], after_state["players"][opp_index]

        direct = _hp_loss_list(before_opp["active"], after_opp["active"])
        bench = _hp_loss_list(before_opp["bench"], after_opp["bench"])
        ko = any(s not in _all_pokemon(after_opp) for s in _all_pokemon(before_opp))
        prize_taken = sum(1 for c in before_me["prize"] if c is not None) > sum(
            1 for c in after_me["prize"] if c is not None
        )
        status = any(not before_opp.get(f) and after_opp.get(f) for f in STATUS_FIELDS)
        energy_removed = _energy_removed(before_opp, after_opp)
        draw = (after_me.get("handCount") or 0) > (before_me.get("handCount") or 0)
        useful = bool(bench or status or energy_removed or draw)
        meaningless = direct <= 0 and not (ko or prize_taken or useful)

        attacks.append({
            "step": i, "direct_damage": direct, "ko": ko, "prize_taken": prize_taken,
            "useful_side_effect": useful, "meaningless": meaningless,
        })

    return {"episode": path.stem, "won": won, "our_index": our_index, "attacks": attacks}


def main() -> None:
    mapping = json.loads((_HERE / "_submission_episode_map.json").read_text(encoding="utf-8"))
    results = []
    for ref, episode_ids in mapping.items():
        for eid in episode_ids:
            path = _HERE / "replays" / f"episode-{eid}-replay.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            team_names = data["info"]["TeamNames"]
            if OUR_TEAM not in team_names:
                continue
            our_index = team_names.index(OUR_TEAM)
            result = analyze_episode(path, our_index)
            result["submission_ref"] = ref
            results.append(result)

    out_path = _HERE / "_analyze_attackplan_submissions_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    total_games = len(results)
    wins = sum(1 for r in results if r["won"])
    total_attacks = sum(len(r["attacks"]) for r in results)
    zero_dmg_attacks = sum(1 for r in results for a in r["attacks"] if a["direct_damage"] <= 0)
    meaningless_attacks = sum(1 for r in results for a in r["attacks"] if a["meaningless"])
    games_with_meaningless = [r for r in results if any(a["meaningless"] for a in r["attacks"])]
    games_without_meaningless = [r for r in results if not any(a["meaningless"] for a in r["attacks"])]

    def win_rate(rs):
        return (sum(1 for r in rs if r["won"]) / len(rs)) if rs else None

    print(f"games analyzed: {total_games} (wins: {wins}, win rate: {wins/total_games:.3f})")
    print(f"total our ATTACK actions: {total_attacks}")
    print(f"  zero direct damage: {zero_dmg_attacks} ({zero_dmg_attacks/total_attacks:.1%})" if total_attacks else "")
    print(f"  meaningless (0dmg AND no useful effect, i.e. truly wasted): {meaningless_attacks}")
    print(f"games containing >=1 meaningless attack: {len(games_with_meaningless)} "
          f"(win rate {win_rate(games_with_meaningless)})")
    print(f"games with 0 meaningless attacks: {len(games_without_meaningless)} "
          f"(win rate {win_rate(games_without_meaningless)})")

    print("\nper-submission breakdown:")
    for ref in mapping:
        rs = [r for r in results if r["submission_ref"] == ref]
        w = sum(1 for r in rs if r["won"])
        ma = sum(1 for r in rs for a in r["attacks"] if a["meaningless"])
        print(f"  {ref}: games={len(rs)} wins={w} win_rate={w/len(rs):.3f} meaningless_attacks={ma}")


if __name__ == "__main__":
    main()
