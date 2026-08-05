"""build_submission.py が作った tar.gz を、Kaggle相当の条件で動かして確認する。

**cwd をバンドル展開先にし、`__file__` の無い状態で main.py を exec() するのが必須。**
これを端折ると、テストスクリプト側で先に `cg` を import してしまい `sys.modules["cg"]`
が汚れたまま main.py の import が(気づかれずに)失敗し、対局中ずっと保険用の
ダミー行動を返し続ける、という事故になる(勝率が不自然に0%近くまで落ちるのに
例外は一切出ないので、フォールバック回数を数えるまで気づけなかった)。

使い方:
```bash
cd opponents/rule_agents
python3 verify_submission.py submissions/grimmsnarl_rule_submission.tar.gz --opponent dragapult_rule --games 100
```
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

OPPONENTS = {
    "dragapult_rule": (_ROOT / "opponents" / "dragapult_ex_deck.csv", "opponents.dragapult_rule_agent"),
    "grimmsnarl_rule": (_HERE / "decks" / "marnie_grimmsnarl_ex.csv", "opponents.rule_agents.grimmsnarl"),
    "lucario_rule": (_HERE / "decks" / "mega_lucario_ex.csv", "opponents.rule_agents.lucario"),
    "archaludon_rule": (_HERE / "decks" / "archaludon_ex.csv", "opponents.rule_agents.archaludon"),
}


def read_deck(path: Path) -> list[int]:
    return [int(v) for v in path.read_text().split() if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball", type=Path)
    ap.add_argument("--opponent", choices=sorted(OPPONENTS), default="dragapult_rule")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="verify_submission_"))
    with tarfile.open(args.tarball) as tar:
        tar.extractall(workdir)

    # --- Kaggle 相当の条件を作る -------------------------------------------------
    os.chdir(workdir)  # main.py の os.getcwd() 候補と一致させる
    sys.path.insert(0, str(workdir))

    main_ns: dict = {}
    code = (workdir / "main.py").read_text(encoding="utf-8")
    exec(compile(code, "main.py", "exec"), main_ns)  # __file__ 無し = Kaggle 相当
    assert "torch" not in sys.modules, "torch は読み込まれないはず"

    rule_agent = main_ns["_get_agent"]()
    if rule_agent is None:
        print("FAIL: rule_agents の import に失敗しています(フォールバックのみで動く状態)")
        sys.exit(1)

    fallback_calls = [0]
    orig_fallback = main_ns["_fallback_action"]

    def counting_fallback(obs):
        fallback_calls[0] += 1
        return orig_fallback(obs)

    main_ns["_fallback_action"] = counting_fallback
    submission_agent = main_ns["agent"]

    import cg.game as cg_game  # noqa: E402  (main.py が import 済みの cg を再利用)
    to_observation_class = sys.modules["cg.api"].to_observation_class
    battle_start, battle_select, battle_finish = (
        cg_game.battle_start, cg_game.battle_select, cg_game.battle_finish,
    )

    # --- 対戦相手 -----------------------------------------------------------------
    sys.path.insert(0, str(_ROOT))
    import importlib
    opp_deck_path, opp_module = OPPONENTS[args.opponent]
    opponent_agent = importlib.import_module(opp_module).agent

    deck0 = read_deck(workdir / "deck.csv")
    deck1 = read_deck(opp_deck_path)

    wins = 0
    errors = 0
    max_ms = 0.0
    for g in range(args.games):
        random.seed(args.seed_start + g)
        p0_first = (g % 2 == 0)
        d0, d1 = (deck0, deck1) if p0_first else (deck1, deck0)
        obs_dict, start = battle_start(d0, d1)
        if start.errorType != 0:
            errors += 1
            continue
        steps = 0
        try:
            while steps < 3000:
                obs = to_observation_class(obs_dict)
                if obs.current is None or obs.current.result != -1:
                    res = None if obs.current is None else obs.current.result
                    if res is not None:
                        my_idx = 0 if p0_first else 1
                        wins += 1 if res == my_idx else 0
                    break
                p = obs.current.yourIndex
                is_mine = (p == 0) == p0_first
                t0 = time.perf_counter()
                act = submission_agent(obs_dict) if is_mine else opponent_agent(obs)
                if is_mine:
                    max_ms = max(max_ms, (time.perf_counter() - t0) * 1000)
                sel = obs.select
                assert isinstance(act, list)
                assert sel.minCount <= len(act) <= sel.maxCount, (act, sel.minCount, sel.maxCount)
                assert len(set(act)) == len(act), "重複した選択"
                assert all(0 <= i < len(sel.option) for i in act), "範囲外の選択"
                obs_dict = battle_select(act)
                steps += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  game {g}: ERROR {e!r}")
        finally:
            battle_finish()

    n_ok = args.games - errors
    print(f"games={args.games} errors={errors} win_rate={wins/n_ok if n_ok else float('nan'):.3f} "
          f"max_agent_call_ms={max_ms:.1f} fallback_calls={fallback_calls[0]}")
    if fallback_calls[0] > 0:
        print("WARNING: フォールバックが呼ばれている(rule_agents の import か判断ロジックで"
              "例外が起きている可能性)。原因を特定してから提出すること。")
    if errors > 0:
        print("FAIL: エラーが発生した対局がある")
        sys.exit(1)
    if fallback_calls[0] > 0:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
