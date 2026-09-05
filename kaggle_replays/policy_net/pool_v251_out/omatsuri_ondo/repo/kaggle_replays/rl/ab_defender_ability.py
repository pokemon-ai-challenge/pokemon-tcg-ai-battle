#!/usr/bin/env python3
"""防御側特性の考慮(damage_prevented / survives_via_ability)のA/B比較。

sample_submission/ptcg_ai/board_evaluation/attack_features.py の打点計算は、長らく
弱点・抵抗力しか見ておらず、防御側の特性を一切参照していなかった。そのため、
イワパレス(crustle, cardId=345)の特性
    "Prevent all damage done to this Pokémon by attacks from your opponent's Pokémon {ex}"
に対して、marnie デッキの主砲「マリィのオーロンゲ ex」(ex アタッカー)の打点を満額で
計算していた(実際は0)。これが marnie の最悪マッチアップ(crustle 戦)の最有力の説明。

この修正(damage_prevented() / survives_via_ability() の追加、can_ko() /
resolve_damage() からの呼び出し)は既に実装済み。ここでは効果を測定するだけで、
再実装はしない。

測定は4アーム(各600試合、既定):

    A. 修正あり(既定)   vs crustle   ← 効果を見たい本命
    B. 修正なし(env無効化) vs crustle   ← 比較対象
    C. 修正あり(既定)   vs alakazam  ← 対照。変わらないはず
    D. 修正なし(env無効化) vs alakazam  ← 同上

「修正なし」は環境変数 PTCG_DISABLE_DEFENDER_ABILITY=1 で実現する
(attack_features.py 側にA/B切り替え用に実装済み)。

集計ロジック(勝率・先攻後攻内訳・サイド差・試合長)は自前で書かず、
kaggle_replays/rl/eval_diagnostics.py をサブプロセスとして呼び出し、その標準出力を
パースするだけにする(同じ集計を二重に書かない)。環境変数はサブプロセスの env に
直接渡す方式にする。eval_diagnostics.py 自身が multiprocessing.Pool でワーカーを
起動するが、Windows(spawn)でもサブプロセスの環境変数はそのままそのプロセスの
os.environ になるので、Pool の子プロセスにも確実に継承される。

使い方:
    python ab_defender_ability.py
    python ab_defender_ability.py --games 600 --workers 4
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EVAL_DIAGNOSTICS = _HERE / "eval_diagnostics.py"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LEARNER = "marnie_grimmsnarl_ex"
LEARNER_WEIGHTS = "policy_weights_marnie_grimmsnarl_ex_bc2.json"

# (アームラベル, 相手, 修正を無効化するか)
ARMS: list[tuple[str, str, bool]] = [
    ("A", "crustle", False),
    ("B", "crustle", True),
    ("C", "alakazam", False),
    ("D", "alakazam", True),
]

# eval_diagnostics.py の出力(日本語)から必要な数値だけ拾う。
# フォーマットは eval_diagnostics.py 側の print 文と1対1で対応させている
# (向こうのフォーマットを変えたらここも直す必要がある)。
_VALID_RE = re.compile(r"有効試合\s+(\d+)/(\d+)")
_OVERALL_RE = re.compile(r"^\s*全体\s+([\d.]+)\s+\(n=(\d+)\)", re.MULTILINE)
_FIRST_RE = re.compile(r"^\s*先攻\s+([\d.]+)\s+\(n=(\d+)\)", re.MULTILINE)
_SECOND_RE = re.compile(r"^\s*後攻\s+([\d.]+)\s+\(n=(\d+)\)", re.MULTILINE)
_TURNS_RE = re.compile(r"平均\s+([\d.]+)\s+ターン")
_SIDE_DIFF_RE = re.compile(r"平均\s+([+-][\d.]+)\s+SD\s+([\d.]+)")


def _run_arm(opponent: str, disable_fix: bool, games: int, workers: int) -> dict:
    """eval_diagnostics.py をサブプロセスで実行し、標準出力をパースして返す。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if disable_fix:
        env["PTCG_DISABLE_DEFENDER_ABILITY"] = "1"
    else:
        # 親プロセスの環境に紛れ込んでいる可能性を考慮し、明示的に外す。
        env.pop("PTCG_DISABLE_DEFENDER_ABILITY", None)

    cmd = [
        sys.executable, str(_EVAL_DIAGNOSTICS),
        "--learner", LEARNER,
        "--learner-weights", LEARNER_WEIGHTS,
        "--opponents", opponent,
        "--games", str(games),
        "--workers", str(workers),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval_diagnostics.py が失敗(returncode={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    out = proc.stdout
    result: dict = {"stdout": out}

    m = _VALID_RE.search(out)
    if m:
        result["valid_n"], result["total_n"] = int(m.group(1)), int(m.group(2))

    m = _OVERALL_RE.search(out)
    if m:
        result["win_rate"], result["n"] = float(m.group(1)), int(m.group(2))

    m = _FIRST_RE.search(out)
    if m:
        result["first_wr"], result["first_n"] = float(m.group(1)), int(m.group(2))

    m = _SECOND_RE.search(out)
    if m:
        result["second_wr"], result["second_n"] = float(m.group(1)), int(m.group(2))

    m = _TURNS_RE.search(out)
    if m:
        result["avg_turns"] = float(m.group(1))

    m = _SIDE_DIFF_RE.search(out)
    if m:
        result["side_diff_mean"], result["side_diff_sd"] = float(m.group(1)), float(m.group(2))

    return result


def _fmt(x, spec: str = "{:.4f}") -> str:
    return spec.format(x) if x is not None else "-"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"learner={LEARNER} weights={LEARNER_WEIGHTS} games/arm={args.games} workers={args.workers}")
    print("アーム: A=crustle修正あり B=crustle修正なし C=alakazam修正あり D=alakazam修正なし\n")

    results: dict[str, dict] = {}
    t_start = time.time()
    for label, opponent, disable_fix in ARMS:
        fix_desc = "修正なし(PTCG_DISABLE_DEFENDER_ABILITY=1)" if disable_fix else "修正あり(既定)"
        print(f"[アーム{label}] 開始: opponent={opponent} {fix_desc} games={args.games}", flush=True)
        t0 = time.time()
        r = _run_arm(opponent, disable_fix, args.games, args.workers)
        elapsed = time.time() - t0
        results[label] = r
        print(
            f"[アーム{label}] 完了 ({elapsed / 60:.1f}分): "
            f"win_rate={_fmt(r.get('win_rate'))} (n={r.get('n', '-')}) "
            f"有効試合={r.get('valid_n', '-')}/{r.get('total_n', '-')}",
            flush=True,
        )

    total_elapsed = time.time() - t_start
    print(f"\n全4アーム完了。合計 {total_elapsed / 60:.1f}分\n")

    print("=" * 78)
    print("防御側特性 A/B比較 結果")
    print("=" * 78)
    header = (
        f"{'アーム':<8}{'相手':<10}{'修正':<8}{'勝率':>8}{'n':>6}"
        f"{'先攻':>8}{'後攻':>8}{'平均ターン':>10}"
    )
    print(header)
    for label, opponent, disable_fix in ARMS:
        r = results[label]
        fix = "なし" if disable_fix else "あり"
        n = r.get("n")
        print(
            f"{label:<8}{opponent:<10}{fix:<8}"
            f"{_fmt(r.get('win_rate')):>8}{(n if n is not None else '-'):>6}"
            f"{_fmt(r.get('first_wr')):>8}{_fmt(r.get('second_wr')):>8}"
            f"{_fmt(r.get('avg_turns'), '{:.1f}'):>10}"
        )

    a, b = results["A"].get("win_rate"), results["B"].get("win_rate")
    c, d = results["C"].get("win_rate"), results["D"].get("win_rate")
    print("\n=== 差分(修正あり − 修正なし) ===")
    if a is not None and b is not None:
        print(f"  crustle  (A-B): {a - b:+.4f}   (A={a:.4f} B={b:.4f})  ← ここで差が出るのが期待される結果")
    if c is not None and d is not None:
        print(f"  alakazam (C-D): {c - d:+.4f}   (C={c:.4f} D={d:.4f})  ← ここは差が出ないはず(対照)")


if __name__ == "__main__":
    main()
