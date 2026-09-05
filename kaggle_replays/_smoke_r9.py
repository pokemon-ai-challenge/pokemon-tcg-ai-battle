"""og_r9(Fix-D 山札僅少ブレーキ / Fix-E ボスのKOゲート)の実対戦スモーク。

目的は**勝率ではない**(4試合ではノイズ床にすら届かない。`project_local_ab_noise_floor`)。
確認するのは次の3点だけ:

  1. エラー0(invalid action / 例外 / タイムアウトが出ない = 提出しても即ERROR にならない)
  2. 2つのゲートが実対戦の decision 経路で**実際に発火する頻度**(スナップショットは通っても
     実戦の decision に届いていない、という取りこぼしを検出する)
  3. 片側ON config(deckbrake / bossgate)でも同じく動くこと(独立ablationの実行確認)

デッキ/重みは og 系(`_diag_og_hammer.py` と同じ g2top2_v032 + mixogerpon)、相手は
`train_league.FIELD_PRESETS["mix"]` から share 比例で抽出する。

使い方(cwd はどこでもよい。内部で production cwd に移る):
    python kaggle_replays/_smoke_r9.py --games 4
    python kaggle_replays/_smoke_r9.py --games 4 --config abl_5_full_og_r9_deckbrake
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_META_DIR = _ROOT / "kaggle_replays" / "meta_analysis"
_OG_DECK = _ROOT / "kaggle_replays" / "deck_search" / "candidates_ogerpon_stage3" / "g2top2_v032.csv"
_OG_WEIGHTS = _WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"
_GAME_OFFSET = 4_910_000

# 報告する r9 のカウンタ(`ml_policy_agent._GUARD_STATS` のキー)。
_R9_KEYS = (
    "low_deck_draw_brake_fired",
    "low_deck_draw_brake_skipped_ko",
    "low_deck_draw_brake_inconclusive",
    "boss_lethal_gate_fired",
    "boss_lethal_gate_redirected",
    "boss_lethal_gate_denial_kept",
    "boss_lethal_gate_inconclusive",
)


def _load_mix_field() -> list[tuple[str, float, str, str]]:
    """torch 依存で重いので import は呼び出し時まで遅延する(`_diag_og_hammer.py` と同じ)。"""
    for _p in (str(_ROOT / "kaggle_replays" / "rl"), str(_ROOT / "league")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import train_league  # noqa: E402 - 遅延import

    return list(train_league.FIELD_PRESETS["mix"])


def _pick_opponent(field, rng: random.Random):
    shares = [share for _arch, share, _gen, _deckdir in field]
    draw = rng.random() * sum(shares)
    acc = 0.0
    idx = len(field) - 1
    for i, share in enumerate(shares):
        acc += share
        if draw <= acc:
            idx = i
            break
    arch, _share, gen, deckdir = field[idx]
    return arch, str(_WDIR / f"policy_weights_{arch}{gen}.json"), _META_DIR / deckdir / arch / "01.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description="og_r9 の実対戦スモーク(エラー0と発火頻度の確認)")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--config", default="abl_5_full_og_r9")
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()

    own_cfg = agents.load_config_copy(args.config)
    own_cfg["policy_weights_path"] = str(_OG_WEIGHTS)
    own_agent = agents.make_ml_policy_agent(own_cfg)
    own_deck = runner.load_deck(_OG_DECK)
    # ローカル評価専用の是正(`_diag_og_hammer.py` と同じ): `_get_deck()` は cwd の deck.csv を
    # 読むが、ここで実際に使うのは og_v032。食い違うと隠れ状態が組めず探索が黙って死ぬ。
    MA._deck_cache = list(own_deck)

    field = _load_mix_field()
    rng = random.Random(args.seed)
    totals = {k: 0 for k in _R9_KEYS}
    rows: list[dict] = []
    errors = 0
    started = time.perf_counter()

    for game_index in range(args.games):
        arch, opp_weights, opp_deck_path = _pick_opponent(field, rng)
        opp_cfg = agents.load_config_copy("abl_5_full")  # r9 キー無し=相手側では不発
        opp_cfg["policy_weights_path"] = opp_weights
        opponent = agents.make_ml_policy_agent(opp_cfg)
        opp_deck = runner.load_deck(opp_deck_path)

        own_first = game_index % 2 == 0
        own_index = 0 if own_first else 1
        MA.reset_guard_stats()
        if own_first:
            result = runner.play_game(own_agent, opponent, own_deck, opp_deck)
        else:
            result = runner.play_game(opponent, own_agent, opp_deck, own_deck)
        stats = MA.get_guard_stats()
        for key in _R9_KEYS:
            totals[key] += stats.get(key, 0)
        if result.error:
            errors += 1
        row = {
            "game_id": _GAME_OFFSET + game_index,
            "opp_arch": arch,
            "own_first": own_first,
            "own_win": (result.winner == own_index) if result.winner is not None else None,
            "win_condition": result.primary_win_condition,
            "turns": result.turns,
            "error": result.error,
            **{k: stats.get(k, 0) for k in _R9_KEYS},
        }
        rows.append(row)
        print("  game#{} opp={} own_first={} win={} cond={} turns={} err={} | {} ({:.1f}分)".format(
            row["game_id"], arch, own_first, row["own_win"], row["win_condition"],
            row["turns"], row["error"],
            {k: row[k] for k in _R9_KEYS if row[k]}, (time.perf_counter() - started) / 60.0),
            file=sys.stderr, flush=True)

    summary = {
        "config": args.config,
        "games": len(rows),
        "errors": errors,
        "wins": sum(1 for r in rows if r["own_win"] is True),
        "totals": totals,
        "per_game": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
