"""Phase19.12 §3/§40: Frozen Value が production runtime で本当に呼ばれるかの実測。

コードを読むだけでは見落とし得るので、`ValueModel` の構築と全 predict 経路に
カウンタを刺し、production entrypoint(main.agent 相当 = core.agent -> ml_policy_agent、
config は abl_5_full)で実試合を回して呼び出し回数を数える。

対照として leaf_eval.kind="value" の config でも回し、
「カウンタが刺さっていること自体」を確認する(0 が計装ミスでない証拠)。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from ptcg_ai.learning import value_model as VM  # noqa: E402
from ptcg_ai.learning import value_shadow_log as VSL  # noqa: E402
from ptcg_ai.search import leaf_eval as LE  # noqa: E402
from ptcg_ai.search import pimc as PIMC  # noqa: E402

CALLS: Counter = Counter()


def instrument() -> None:
    orig_init = VM.ValueModel.__init__

    def init(self, *a, **k):
        CALLS["ValueModel.__init__"] += 1
        return orig_init(self, *a, **k)
    VM.ValueModel.__init__ = init

    for name in ("predict_win_prob", "predict_win_prob_from_state",
                 "predict_win_prob_from_dict", "predict_win_prob_from_features"):
        fn = getattr(VM.ValueModel, name)

        def mk(fn=fn, name=name):
            def wrapped(self, *a, **k):
                CALLS["ValueModel." + name] += 1
                return fn(self, *a, **k)
            return wrapped
        setattr(VM.ValueModel, name, mk())

    for mod, name, key in ((LE.ValueModelEvaluator, "evaluate", "ValueModelEvaluator.evaluate"),
                           (LE.BlendedEvaluator, "evaluate", "BlendedEvaluator.evaluate"),
                           (LE.HandcraftedEvaluator, "evaluate", "HandcraftedEvaluator.evaluate")):
        fn = getattr(mod, name)

        def mk(fn=fn, key=key):
            def wrapped(self, *a, **k):
                CALLS[key] += 1
                return fn(self, *a, **k)
            return wrapped
        setattr(mod, name, mk())

    for mod, name, key in ((VSL, "record", "value_shadow_log.record"),
                           (PIMC, "_leaf_score", "pimc._leaf_score")):
        fn = getattr(mod, name, None)
        if fn is None:
            continue

        def mk(fn=fn, key=key):
            def wrapped(*a, **k):
                CALLS[key] += 1
                return fn(*a, **k)
            return wrapped
        setattr(mod, name, mk())


def run(config_name: str, games: int, mutate=None) -> dict:
    cfg = agents.load_config_copy(config_name)
    if mutate:
        cfg = mutate(copy.deepcopy(cfg))
    a = agents.make_ml_policy_agent(cfg)
    b = agents.make_ml_policy_agent(copy.deepcopy(cfg))
    deck = runner.load_deck()
    CALLS.clear()
    res = []
    for g in range(games):
        r = runner.play_game(a, b, list(deck), list(deck))
        res.append(getattr(r, "winner", None))
    return {"config": config_name, "games": games, "results": [str(x) for x in res],
            "calls": dict(CALLS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--out", default=str(_ROOT / "kaggle_replays" / "value_net" / "phase1912_callsite.json"))
    args = ap.parse_args()
    instrument()
    out = {}

    # (1) 本番そのまま
    out["production_abl_5_full"] = run("abl_5_full", args.games)

    # (2) 対照: leaf_eval を value に差し替え(計装が生きている証拠)
    def to_value(c):
        c.setdefault("pipeline", {})["leaf_eval"] = {"kind": "value"}
        return c
    out["control_leaf_eval_value"] = run("abl_5_full", 1, to_value)

    # (3) 対照: value_shadow_logging を有効化
    def to_shadow(c):
        c["value_shadow_logging"] = True
        return c
    out["control_value_shadow_logging"] = run("abl_5_full", 1, to_shadow)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
