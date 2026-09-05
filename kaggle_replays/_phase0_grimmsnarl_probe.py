"""Phase 0: マリィのオーロンゲex の特性/効果がエンジンでどう select 提示されるかを実測する。

目的（方針書 §6）:
  - パンクアップ(648) の悪エネ5枚配分の出方
  - アドレナブレイン(112) ののせ替え(元ダメカン/個数/先) の出方
  - いてつくとばり(104) が自動か選択か
  - スパイクタウンジム(1259) 毎ターンサーチの出方
  - 先攻/後攻それぞれ最初の番(turn 1 / turn 2)で EVOLVE/ATTACK/PLAY(サポ) が出るか

やり方:
  grimmsnarl 側を模倣ポリシー(policy_weights_marnie_grimmsnarl_ex)で操縦し、rule_based 相手に
  数ゲーム回す。grimmsnarl の全 decision の select(context/type/option/effect/contextCard/deck)
  を JSONL に落とし、キー特性まわりだけ後段で要約する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE = _ROOT / "sample_submission"
_LEAGUE = _ROOT / "league"
for p in (str(_ROOT), str(_SAMPLE), str(_LEAGUE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.api import all_card_data, SelectType, SelectContext, OptionType, AreaType  # noqa: E402
import run_league  # noqa: E402
from run_match import play_match  # noqa: E402


def nm(cls, v):
    """生 int / IntEnum のどちらでも enum 名に変換する（不明値は int のまま）。"""
    if v is None:
        return None
    try:
        return cls(int(v)).name
    except Exception:
        return v

CARD = {c.cardId: c.name for c in all_card_data()}

DECK_PATH = _ROOT / "kaggle_replays/meta_analysis/archetype_decks/marnie_grimmsnarl_ex/01.csv"
WEIGHTS = "sample_submission/ptcg_ai/learning/policy_weights_marnie_grimmsnarl_ex.json"
OUT = Path(__file__).resolve().parent / "_phase0_grimmsnarl_probe_log.jsonl"

KEY_IDS = {648: "パンクアップ/オーロンゲ", 112: "アドレナブレイン/マシマシラ",
           104: "いてつくとばり/ユキメノコ", 1259: "スパイクタウンジム"}


def _cn(cid):
    if cid is None:
        return None
    return f"{cid}:{CARD.get(cid, '?')}"


def _opt(o):
    d = {"type": nm(OptionType, o.type)}
    for k in ("number", "area", "index", "playerIndex", "inPlayArea", "inPlayIndex",
              "attackId", "cardId", "count", "energyIndex", "toolIndex"):
        v = getattr(o, k, None)
        if v is not None:
            d[k] = nm(AreaType, v) if k in ("area", "inPlayArea") else v
    if o.cardId is not None:
        d["card"] = _cn(o.cardId)
    return d


def make_logger(inner, records, tag):
    step = {"n": 0}

    def logged(obs):
        step["n"] += 1
        sel = obs.select
        cur = obs.current
        if sel is not None and cur is not None:
            rec = {
                "tag": tag,
                "step": step["n"],
                "turn": cur.turn,
                "firstPlayer": getattr(cur, "firstPlayer", None),
                "yourIndex": cur.yourIndex,
                "type": nm(SelectType, sel.type),
                "context": nm(SelectContext, sel.context),
                "min": sel.minCount,
                "max": sel.maxCount,
                "remainDmg": sel.remainDamageCounter,
                "remainEnergy": sel.remainEnergyCost,
                "deck": None if sel.deck is None else [_cn(c.id) for c in sel.deck][:20],
                "contextCard": _cn(sel.contextCard.id) if sel.contextCard else None,
                "effect": _cn(sel.effect.id) if sel.effect else None,
                "options": [_opt(o) for o in sel.option][:25],
            }
            records.append(rec)
        return inner(obs)

    return logged


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    deck_g = run_league.read_deck_csv_file(str(DECK_PATH))
    deck_opp = run_league.read_deck_csv_file(str(_SAMPLE / "deck.csv"))

    pilot = run_league.build_agent("ml_policy", WEIGHTS, "ml_lethal_attackplan_v0only")
    opp = run_league.build_agent("rule_based", None, "ml_lethal_attackplan_v0only")

    records = []
    results = []
    for g in range(n_games):
        # 先攻/後攻の両方を拾うため、席を入れ替える
        if g % 2 == 0:
            a0 = make_logger(pilot, records, "G")   # grimmsnarl = player 0
            r = play_match(a0, opp, deck_g, deck_opp, seed=1000 + g)
        else:
            a1 = make_logger(pilot, records, "G")   # grimmsnarl = player 1
            r = play_match(opp, a1, deck_opp, deck_g, seed=1000 + g)
        results.append({"game": g, "winner": r.winner, "turns": r.turns,
                        "steps": r.steps, "error": r.error, "sec": round(r.seconds, 1)})
        print(f"game {g}: winner={r.winner} turns={r.turns} steps={r.steps} "
              f"err={r.error} sec={r.seconds:.1f}", flush=True)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    print(f"\nwrote {len(records)} decision records -> {OUT}")
    print("results:", json.dumps(results, ensure_ascii=False))

    # ---- 要約: キー特性まわりの context を抽出 ----
    print("\n=== distinct (context, type) over grimmsnarl decisions ===")
    from collections import Counter
    ctx = Counter((str(r["context"]), str(r["type"])) for r in records)
    for (c, t), n in ctx.most_common():
        print(f"  {n:4d}  {c:28s} {t}")

    print("\n=== decisions where effect/contextCard is a KEY ability card ===")
    for r in records:
        eff = r["effect"] or ""
        cc = r["contextCard"] or ""
        hit = [k for k in KEY_IDS if f"{k}:" in eff or f"{k}:" in cc]
        if hit:
            opts = ", ".join(o["type"] + (f"({o.get('card')})" if o.get("card") else "") for o in r["options"][:8])
            print(f"  turn{r['turn']:>3} ctx={r['context']:<22} type={r['type']:<10} "
                  f"eff={r['effect']} cc={r['contextCard']} deck={'Y' if r['deck'] else '-'} "
                  f"min/max={r['min']}/{r['max']} rDmg={r['remainDmg']} | opts=[{opts}]")

    print("\n=== turn 1 & turn 2 MAIN option types (先後の初手可能行動) ===")
    for r in records:
        if r["turn"] in (1, 2) and r["context"] == "MAIN":
            types = sorted({o["type"] for o in r["options"]})
            print(f"  turn{r['turn']} first={r['firstPlayer']} you={r['yourIndex']} MAIN opts={types}")


if __name__ == "__main__":
    main()
