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
import subprocess
import sys
import tarfile
import tempfile
import textwrap
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


# 「main.py を exec() して、判断ロジックが本当に読み込めるか」だけを確かめる小さな台本。
# 別プロセスで走らせるのが要点。同じプロセスで検証すると、検証スクリプト側が先に
# import した cg などが sys.modules に残り、本番では失敗する読み込みが通ってしまう。
_IMPORT_PROBE = textwrap.dedent(
    """
    import os, sys, traceback
    ns = {}
    code = open(os.path.join(sys.argv[1], "main.py"), encoding="utf-8").read()
    try:
        exec(compile(code, "main.py", "exec"), ns)   # __file__ 無し = Kaggle 相当
    except Exception:
        print("EXEC_FAILED")
        traceback.print_exc()
        raise SystemExit(2)
    agent = ns["_get_agent"]()
    print("LOADED" if agent is not None else "FALLBACK_ONLY")
    raise SystemExit(0 if agent is not None else 3)
    """
)


def check_load_environments(bundle: Path) -> bool:
    """実行環境を変えて「判断ロジックが読み込めるか」を確かめる。

    Kaggle 上で実際に踏んだ事故は、対局自体はエラー無く完走するのに判断ロジックが
    一度も動かない、というものだった。勝敗や例外だけを見ていても気づけないので、
    読み込みの成否そのものを、環境を変えて明示的に確認する。
    """
    neutral = Path(tempfile.mkdtemp(prefix="verify_cwd_"))
    cases = [
        ("cwd=バンドル直下", bundle, None),
        ("cwd=無関係な場所 + PYTHONPATH でバンドルを指定", neutral, str(bundle)),
    ]
    all_ok = True
    for label, cwd, pythonpath in cases:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        proc = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE, str(bundle)],
            cwd=str(cwd), env=env, capture_output=True, encoding="utf-8", errors="replace",
        )
        ok = proc.returncode == 0
        all_ok = all_ok and ok
        print(f"  [{'OK ' if ok else 'NG '}] {label}: {(proc.stdout or '').strip().splitlines()[-1:] or ['(出力なし)']}")
        if not ok and proc.stderr:
            for line in proc.stderr.strip().splitlines()[-12:]:
                print(f"        {line}")
    return all_ok


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

    # --- 1. 実行環境を変えて「判断ロジックが読み込めるか」を先に確かめる -------------
    print("[1] 読み込み確認(別プロセス・環境を変えて)")
    if not check_load_environments(workdir):
        print("FAIL: 判断ロジックを読み込めない環境がある。このまま提出すると"
              "「エラーは出ないのに一度も判断が効かない」試合になる。")
        sys.exit(1)

    print("[2] 実対戦(合法性・フォールバック発生の確認)")

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
