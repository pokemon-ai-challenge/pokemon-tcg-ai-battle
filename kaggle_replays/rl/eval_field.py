"""方針(c)検証: alakazam(learner)の対フィールド加重勝率を full agent で測る。

単一相手(crustle)RL が他マッチを壊していないか(overfit)を検出するため、7アーキ全部に対する
alakazam の勝率を meta-share 加重で出し、production(policy_weights.json)と RL 候補を比較する。

相手は各アーキの模倣policy(policy_weights_<arch>.json)= round_robin の "field" 定義に合わせる。
alakazam 側だけ weights を差し替える。full agent(ml_lethal_attackplan_v0only overlay込み)。

使い方: python kaggle_replays/rl/eval_field.py --rl-weights policy_weights_alakazam_rl_vscrustle.json --games 150
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# (arch, meta_share) — round_robin と同じ7アーキ。相手は各 imitation policy。share は2026-07実測。
FIELD = [
    ("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
    ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
    ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181),
]

# フィールド定義は train_league.py を単一の出典にする(mix の内容が2箇所に散ると必ずずれる)。
# eval 固有の july7(旧7アーキ)だけここで足す。
from train_league import FIELD_PRESETS as _TL_PRESETS, _expand as _tl_expand  # noqa: E402

FIELD_PRESETS = dict(_TL_PRESETS)
FIELD_PRESETS["july7"] = _tl_expand(FIELD, "")

# --field-preset で差し替わる実効値(main で設定)。既定は従来どおり july7。
_ACTIVE = {"field": FIELD_PRESETS["july7"]}


def alakazam_vs(arch, ala_weights, games, workers, config_base):
    """学習側デッキは _ACTIVE["learner_deck"](既定=field定義の alakazam デッキ)。"""
    """alakazam(ala_weights) vs arch(imitation) の alakazam 勝率。"""
    summary = run_league.run_league(
        agent_a_name="ml_policy", agent_b_name="ml_policy", games=games,
        deck_a_path=_ACTIVE["learner_deck"],
        deck_b_path=str(DECKDIR.parent / _ACTIVE["deckdir_of"][arch] / arch / "01.csv"),
        seed_start=0, progress_every=0,
        weights_a_path=ala_weights,
        weights_b_path=str(WDIR / f"policy_weights_{arch}{_ACTIVE['gen_of'][arch]}.json"),
        config_base=config_base, workers=workers, log=lambda m: None,
    )
    ov = summary["overall"]
    return ov["win_rate"], ov["wins"], ov["games"]


def field_weighted(ala_weights, games, workers, label, config_base):
    total = sum(r[1] for r in _ACTIVE["field"])
    acc = 0.0
    print(f"--- {label} ---", flush=True)
    rows = {}
    for arch, share, _gen, _dd in _ACTIVE["field"]:
        wr, w, n = alakazam_vs(arch, ala_weights, games, workers, config_base)
        acc += share * wr
        rows[arch] = wr
        print(f"  vs {arch:<22} {wr*100:5.1f}%  ({w}/{n})", flush=True)
    fw = acc / total
    print(f"  => 対フィールド加重勝率: {fw*100:.2f}%", flush=True)
    return fw, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-weights", required=True, help="alakazam RL 候補(WDIR内のファイル名 or 絶対)")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--field-preset", default="july7", choices=sorted(FIELD_PRESETS),
                    help="相手フィールドの世代。july7=従来(既定で挙動不変)、g2=2026-08 BC 11アーキ")
    ap.add_argument("--learner-deck", default=None,
                    help="学習側(alakazam)のデッキCSV。既定=field定義の archetype_decks*/alakazam/01.csv。"
                         "提出デッキで測るときは sample_submission/deck.csv 等を渡す")
    ap.add_argument("--baseline-weights", default=None,
                    help="比較元。既定 None = production の policy_weights.json")
    ap.add_argument("--config-base", default="ml_lethal_attackplan_v0only",
                    help="両者に注入する base config。素ポリシー比較は abl_2_policy_only "
                         "(lethal/attack_plan/pipeline すべて OFF)。既定は本番相当の overlay。")
    args = ap.parse_args()

    _ACTIVE["field"] = FIELD_PRESETS[args.field_preset]
    _ACTIVE["gen_of"] = {a: g for a, _s, g, _d in _ACTIVE["field"]}
    _ACTIVE["deckdir_of"] = {a: d for a, _s, _g, d in _ACTIVE["field"]}
    # 学習側(alakazam)のデッキは field 側の alakazam 定義に合わせる。無ければ gen2。
    _ACTIVE["ala_deckdir"] = _ACTIVE["deckdir_of"].get("alakazam", "archetype_decks_g2")
    _ACTIVE["learner_deck"] = (args.learner_deck if args.learner_deck
                               else str(DECKDIR.parent / _ACTIVE["ala_deckdir"] / "alakazam" / "01.csv"))
    print(f"学習側デッキ: {_ACTIVE['learner_deck']}", flush=True)

    rlw = args.rl_weights
    if not Path(rlw).is_absolute():
        rlw = str(WDIR / rlw)
    if not Path(rlw).exists():
        print(f"RL重みが無い: {rlw}"); return

    # full agent の read_deck_csv() は cwd 相対の deck.csv → 無ければ Kaggle パスへフォールバックする
    # 契約(rule_based_agent.read_deck_csv)。run_league の並列パスは _worker_init で chdir するが、
    # 逐次(workers<=1)パスは chdir しないため、main プロセスの cwd を sample_submission にそろえる。
    # deck/weights は絶対パスで渡しているので chdir の影響を受けない。
    os.chdir(_ROOT / "sample_submission")

    print(f"=== alakazam 対フィールド加重勝率(config={args.config_base}, {args.games}試合/マッチ) ===", flush=True)
    basew = args.baseline_weights
    if basew and not Path(basew).is_absolute():
        basew = str(WDIR / basew)
    base_label = f"baseline ({Path(basew).name})" if basew else "production (policy_weights.json)"
    base_fw, base_rows = field_weighted(basew, args.games, args.workers, base_label, args.config_base)
    rl_fw, rl_rows = field_weighted(rlw, args.games, args.workers, f"RL ({Path(rlw).name})", args.config_base)
    print(f"\n=== 比較 ===", flush=True)
    print(f"  production 対フィールド {base_fw*100:.2f}%  ->  RL {rl_fw*100:.2f}%  Δ{(rl_fw-base_fw)*100:+.2f}pt")
    print(f"  マッチ別Δ:")
    for arch, _s, _g, _d in _ACTIVE["field"]:
        d = (rl_rows[arch] - base_rows[arch]) * 100
        print(f"    {arch:<22} {base_rows[arch]*100:5.1f}% -> {rl_rows[arch]*100:5.1f}%  ({d:+.1f}pt)")


if __name__ == "__main__":
    main()
