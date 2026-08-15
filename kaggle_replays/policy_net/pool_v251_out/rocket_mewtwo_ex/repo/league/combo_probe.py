#!/usr/bin/env python3
"""self-play で「リッチエネルギー付与(ATTACH, cardId=13)」と特性使用(ABILITY)の発火回数を
config 別に数える behavioral 検証。card_advantage 項 + extra_candidate_types(ABILITY/ATTACH)で、
模倣学習が見落とすドローコンボ(リッチ付与→+4ドロー→にげあしドローで+3ドロー&1枚制限札を山札回収)
を実際に打つようになるかを、デッキ特化せずに確かめる。

両サイド同一 agent/config の self-play。カウントは両プレイヤー分の合計。config を替えて比較する。

使い方(デスクトップ):
  python combo_probe.py --games 25 --configs abl_5_full_combo,abl_5_full
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_match import play_match  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

os.chdir(_ROOT / "sample_submission")
from cg.api import OptionType  # noqa: E402
from ptcg_ai.core import config as cfgmod  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402

RICH_ENERGY_ID = 13  # リッチエネルギー(ACE SPEC 特殊エネ)


def probe(config_name: str, deck: list[int], games: int, seed_start: int = 0) -> dict:
    cfg = cfgmod.load_config(config_name)
    c = {"rich_attach": 0, "ability": 0, "attach": 0, "decisions": 0}

    def agent(obs):
        action = ml_policy_agent.agent(obs, config=cfg)
        try:
            if obs.select and obs.select.option and action:
                opt = obs.select.option[action[0]]
                c["decisions"] += 1
                if opt.type == OptionType.ATTACH:
                    c["attach"] += 1
                    if opt.cardId == RICH_ENERGY_ID:
                        c["rich_attach"] += 1
                if opt.type == OptionType.ABILITY:
                    c["ability"] += 1
        except Exception:
            pass
        return action

    wins = errors = 0
    for i in range(games):
        r = play_match(agent, agent, deck, deck, seed=seed_start + i)
        if r.error is not None:
            errors += 1
    c["games"] = games
    c["errors"] = errors
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--deck", default=None, help="self-play に使うデッキ(既定 sample_submission/deck.csv)")
    ap.add_argument("--configs", default="abl_5_full_combo,abl_5_full")
    args = ap.parse_args()

    deck = read_deck_csv_file(args.deck)
    print(f"deck cards: {len(deck)}  (リッチ枚数: {deck.count(RICH_ENERGY_ID)})", flush=True)
    for name in args.configs.split(","):
        name = name.strip()
        t = time.time()
        c = probe(name, deck, args.games)
        per = c["rich_attach"] / c["games"] if c["games"] else 0
        print(
            f"{name:22s} games={c['games']} err={c['errors']}  "
            f"rich_attach={c['rich_attach']} ({per:.2f}/game)  ability={c['ability']}  "
            f"attach_total={c['attach']}  decisions={c['decisions']}  ({time.time()-t:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
