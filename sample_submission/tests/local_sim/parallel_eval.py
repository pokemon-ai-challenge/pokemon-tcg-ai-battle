#!/usr/bin/env python3
"""複数プロセスで対局を並列実行し、勝率を測る。

## なぜプロセス並列なのか（スレッドでは動かない）

``cg`` は ctypes 経由でネイティブライブラリを呼ぶが、``battle_start`` /
``battle_select`` / ``battle_finish`` は **プロセス全体で1つの対局状態** を共有する。
スレッドを分けても同じ対局を奪い合うだけで壊れる。プロセスを分ければ各自が
独立したライブラリのインスタンスを持つため、そのまま並列化できる。

## 何を測れるか

- 相手デッキを差し替えたガントレット（``--opponents`` で複数指定）
- 機能フラグの A/B（``--arm`` を複数指定すると条件ごとに測る）

いずれも先手・後手を交互に入れ替えるので、着手順の有利は打ち消される。

## 使い方

    # 自デッキ同士（従来と同じミラーマッチ）で ML policy の A/B
    python tests/local_sim/parallel_eval.py --games 1000 --arm off --arm on

    # 上位アーキタイプ5種を相手にしたガントレット
    python tests/local_sim/parallel_eval.py --games 200 --opponents all --arm off --arm on

    # 直接対戦: 汎用方策 vs ルールベース（ミラーデッキ）
    python tests/local_sim/parallel_eval.py --games 1000 --head-to-head

    # 直接対戦: 汎用方策 vs アーキタイプ専用方策（相手は専用デッキ＋専用モデル）
    python tests/local_sim/parallel_eval.py --games 200 --head-to-head --opponent-pool         --opponents archetype1,archetype2,archetype3,archetype4

相手の方策はランダム固定。したがって本スクリプトが測るのは「相手のデッキ構成が
変わっても勝てるか」であって「強い相手に勝てるか」ではない。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LOCAL_SIM_DIR = Path(__file__).resolve().parent
SAMPLE_SUBMISSION_ROOT = LOCAL_SIM_DIR.parents[1]
OPPONENT_DECKS = LOCAL_SIM_DIR / "opponent_decks.json"
# アーキタイプ別の学習済み重み（再生成可能なので .gitignore 済み。
# 無ければ相手はルールベースになる旨を警告して続行する）
ARCHETYPE_WEIGHTS_DIR = (
    SAMPLE_SUBMISSION_ROOT.parent / "kaggle_replays" / "policy_prior" / "output"
)

# 機能フラグの組み合わせ。名前 -> 環境変数
ARMS: dict[str, dict[str, str]] = {
    "off": {"PTCG_ML_POLICY": "0"},
    "on": {"PTCG_ML_POLICY": "1"},
    "belief_off": {"PTCG_REAL_HIDDEN_STATE": "0"},
    "belief_on": {"PTCG_REAL_HIDDEN_STATE": "1"},
}

STEP_CAP = 2000            # これを超えたら未決着として無効試合に数える


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def two_proportion(k0: int, n0: int, k1: int, n1: int) -> tuple[float, float, float]:
    """(差, 差の95%CI半幅, p値)。プール分散で検定し、CIは非プール分散で出す。"""
    if not n0 or not n1:
        return (float("nan"),) * 3
    p0, p1 = k0 / n0, k1 / n1
    pool = (k0 + k1) / (n0 + n1)
    se = math.sqrt(pool * (1 - pool) * (1 / n0 + 1 / n1))
    z = (p1 - p0) / se if se else 0.0
    se_d = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    return (p1 - p0, 1.96 * se_d, math.erfc(abs(z) / math.sqrt(2)))


def _run_chunk(job: dict) -> dict:
    """1ワーカー分の対局。子プロセスで実行されるので import はこの中で行う。

    環境変数は ``cg`` / ``ptcg_ai`` を import する前に設定する。フラグは呼び出しの
    たびに ``os.environ`` を読む実装だが、import 時に読む実装へ変わっても壊れない
    ようにこの順序を守る。
    """
    import random

    os.environ.update(job["env"])
    if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
        sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))
    os.chdir(SAMPLE_SUBMISSION_ROOT)

    from cg.api import Observation, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from main import agent
    from ptcg_ai.hidden_information import match_context

    rng = random.Random(job["seed"])
    my_deck, opp_deck = job["my_deck"], job["opp_deck"]

    def random_agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return opp_deck
        return rng.sample(range(len(obs.select.option)), obs.select.maxCount)

    # --- 直接対戦モード -----------------------------------------------------
    # 両プレイヤーが同じ agent() を通るため、フラグをプレイヤー単位で切り替える。
    # 本番コードは変更せず、ハーネス側で「手番のプレイヤーが対照群側なら方策を
    # 使わない(None を返す)」ようにラップする。None は既存の素通し経路であり、
    # 呼び出し側はそのまま router.route へ落ちる。
    ml_seat = {"value": 0}
    if job["head_to_head"]:
        from ptcg_ai.action_selection import selector
        from ptcg_ai.learning import policy_model

        _orig_ml = selector._ml_policy_action
        # 自分側: --own-weights が指定されていればそれを使う（RL世代の重み等の評価用）。
        # 無指定なら既定の重み（提出しているモデル）のまま、従来どおり。
        own_model = (
            policy_model.PolicyModel(job["own_weights"]) if job.get("own_weights")
            else policy_model.PolicyModel()
        )
        # 相手側: 重みが指定されていればそれを使う。無指定なら相手はルールベース
        opp_model = (
            policy_model.PolicyModel(job["opp_weights"]) if job["opp_weights"] else None
        )

        def _per_player(obs, select):
            """手番のプレイヤーごとに使うモデルを差し替える。

            selector はモデルをモジュール変数に1つだけ持つ設計なので、呼ぶ直前に
            差し替える。本番コードは変更しない（差し替えはこのハーネス内だけ）。
            """
            acting = obs.current.yourIndex if obs.current is not None else 0
            if acting == ml_seat["value"]:
                selector._POLICY_MODEL_CACHE = own_model
                return _orig_ml(obs, select)
            if opp_model is None:
                return None                      # 相手はルールベース
            selector._POLICY_MODEL_CACHE = opp_model
            return _orig_ml(obs, select)

        selector._ml_policy_action = _per_player

    counts = Counter()
    steps_total = 0
    for i in range(job["games"]):
        match_context.reset()
        seat = (job["seat_start"] + i) % 2      # 自分（実験群）が先手か後手か
        if job["head_to_head"]:
            # 両者とも agent()。学習方策を使うのは seat 側だけ
            ml_seat["value"] = seat
            p0 = p1 = agent
            d0, d1 = my_deck, opp_deck
        else:
            p0, p1 = (agent, random_agent) if seat == 0 else (random_agent, agent)
            d0, d1 = (my_deck, opp_deck) if seat == 0 else (opp_deck, my_deck)

        obs_dict, start = battle_start(d0, d1)
        if start.errorType != 0:
            battle_finish()
            raise RuntimeError(f"battle_start errorType={start.errorType}")
        steps = 0
        try:
            while True:
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    result = obs.current.result
                    break
                acting = obs.current.yourIndex if obs.current is not None else 0
                obs_dict = battle_select((p0 if acting == 0 else p1)(obs_dict))
                steps += 1
                if steps > STEP_CAP:
                    result = -1
                    break
        finally:
            battle_finish()

        steps_total += steps
        if result == seat:
            counts["win"] += 1
        elif result in (0, 1):
            counts["loss"] += 1
        else:
            counts["invalid"] += 1

    return {
        "arm": job["arm"], "opponent": job["opponent"],
        "win": counts["win"], "loss": counts["loss"], "invalid": counts["invalid"],
        "steps": steps_total,
    }


def build_jobs(arms, opponents, my_deck, games, workers, seed, head_to_head=False,
               opp_weights=None, own_weights=None):
    """条件 × 相手デッキ を、ワーカー数ぶんのチャンクに割る。

    ``own_weights``: head-to-head モードで自分側に使う重みJSONのパス（任意）。
    省略時は従来どおり既定の重み（``ptcg_ai.learning.policy_model.PolicyModel()``）。
    RL世代の重みを凍結プールに対して評価する際に使う（``rl/generation.py`` 参照）。
    """
    opp_weights = opp_weights or {}
    jobs = []
    for arm in arms:
        for opp_name, opp_deck in opponents:
            base, extra = divmod(games, workers)
            offset = 0
            for w in range(workers):
                n = base + (1 if w < extra else 0)
                if n == 0:
                    continue
                jobs.append({
                    "arm": arm, "env": ARMS[arm],
                    "opponent": opp_name, "my_deck": my_deck, "opp_deck": opp_deck,
                    "games": n,
                    # 着席をチャンク間で連続させ、全体で先手後手が半々になるようにする
                    "seat_start": offset % 2,
                    # 種は arm に依存させない。条件間で相手のランダム挙動を揃えると
                    # (common random numbers) 条件間の分散が減り検出力が上がる。
                    # hash() は文字列に対しプロセスごとにランダム化されるため使えない
                    # (実行するたびに種が変わり再現しなくなる)。crc32 を使う。
                    "seed": seed + w * 7919 + zlib.crc32(opp_name.encode()) % 100000,
                    "head_to_head": head_to_head,
                    "opp_weights": opp_weights.get(opp_name),
                    "own_weights": own_weights,
                })
                offset += n
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=200, help="条件×相手デッキ あたりの試合数")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), default=None,
                    help="測る条件。複数指定可。既定は off と on")
    ap.add_argument("--opponents", default="self",
                    help="'self'（自デッキ＝ミラー）/ 'all' / opponent_decks.json のキーをカンマ区切り")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="並列プロセス数。既定は 論理コア数-2（操作不能を避けるため）")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--opponent-pool", action="store_true",
                    help="相手にアーキタイプ専用の学習方策を使わせる（--head-to-head と併用）。"
                         "相手は「そのアーキタイプのデッキ + そのアーキタイプ専用モデル」になる")
    ap.add_argument("--head-to-head", action="store_true",
                    help="学習方策 vs ルールベースの直接対戦。両者とも agent() を通し、"
                         "手番のプレイヤーに応じて方策を切り替える。--arm は on 固定")
    ap.add_argument("--own-weights", default=None,
                    help="head-to-head モードで自分側に使う重みJSONのパス（省略時は既定の"
                         "提出モデル）。RL世代の重みを評価する用途("
                         "kaggle_replays/policy_prior/rl/generation.py)")
    ap.add_argument("--json-out", default=None,
                    help="集計結果(arm×opponentごとのwin/loss/invalid、および全体合算)を"
                         "JSONで書き出す（任意。プログラムからの呼び出し用）")
    args = ap.parse_args()

    arms = ["on"] if args.head_to_head else (args.arm or ["off", "on"])

    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))
    os.chdir(SAMPLE_SUBMISSION_ROOT)
    from main import read_deck_csv
    my_deck = read_deck_csv()

    if args.opponents == "self":
        opponents = [("self", my_deck)]
    else:
        pool = json.loads(OPPONENT_DECKS.read_text(encoding="utf-8"))
        names = sorted(pool) if args.opponents == "all" else args.opponents.split(",")
        opponents = [(n, pool[n]["deck"]) for n in names]

    opp_weights: dict[str, str] = {}
    if args.opponent_pool:
        if not args.head_to_head:
            raise SystemExit("--opponent-pool は --head-to-head と併用してください")
        missing = []
        for name, _ in opponents:
            path = ARCHETYPE_WEIGHTS_DIR / f"policy_weights_{name}.json"
            if path.exists():
                opp_weights[name] = str(path)
            else:
                missing.append(name)
        if missing:
            print(f"  ! 専用モデルが見つからない相手（ルールベースになります）: {missing}")
            print(f"    {ARCHETYPE_WEIGHTS_DIR} に policy_weights_<name>.json が必要です。")
            print("    再生成: python policy_prior/train.py --archetype <name> "
                  "--out policy_prior/output/policy_weights_<name>.json")

    jobs = build_jobs(arms, opponents, my_deck, args.games, args.workers, args.seed,
                      head_to_head=args.head_to_head, opp_weights=opp_weights,
                      own_weights=args.own_weights)
    total = sum(j["games"] for j in jobs)
    if args.head_to_head:
        who = "アーキタイプ専用の学習方策" if args.opponent_pool else "ルールベース"
        print(f"直接対戦モード: 汎用の学習方策 vs {who}（勝率50%が「差なし」の基準）")
        print("  注意: match_context は両プレイヤーで共有され Belief が汚染される。"
              "両者に等しく影響するため比較は成立するが、完全にクリーンではない。")
    print(f"{len(arms)}条件 × 相手{len(opponents)}種 × {args.games}試合 = {total:,}試合")
    print(f"{args.workers} プロセスで実行（論理コア {os.cpu_count()}）", flush=True)

    import time
    t0 = time.time()
    agg: dict[tuple[str, str], Counter] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
        for r in pool_exec.map(_run_chunk, jobs):
            key = (r["arm"], r["opponent"])
            c = agg.setdefault(key, Counter())
            for k in ("win", "loss", "invalid", "steps"):
                c[k] += r[k]
            done += r["win"] + r["loss"] + r["invalid"]
            print(f"  ... {done:,}/{total:,} ({time.time() - t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n完了: {elapsed:.0f}s ({elapsed / total:.2f}s/試合 実効)")

    print(f"\n{'条件':<12}{'相手':<14}{'勝率':>8}{'95%CI':>18}{'試合':>8}{'無効':>6}")
    print("-" * 68)
    for arm in arms:
        for opp_name, _ in opponents:
            c = agg.get((arm, opp_name))
            if not c:
                continue
            n = c["win"] + c["loss"]
            lo, hi = wilson(c["win"], n)
            print(f"{arm:<12}{opp_name:<14}{c['win']/n:>7.1%}"
                  f"  [{lo:>5.1%},{hi:>6.1%}]{n:>8,}{c['invalid']:>6}")

    # 条件が2つなら相手デッキごとに差を出す
    if len(arms) == 2:
        a, b = arms
        print(f"\n=== 差（{b} - {a}）===")
        for opp_name, _ in opponents + ([("合計", None)] if len(opponents) > 1 else []):
            if opp_name == "合計":
                ca = sum((agg[(a, o)] for o, _ in opponents), Counter())
                cb = sum((agg[(b, o)] for o, _ in opponents), Counter())
            else:
                ca, cb = agg.get((a, opp_name)), agg.get((b, opp_name))
            if not ca or not cb:
                continue
            n0, n1 = ca["win"] + ca["loss"], cb["win"] + cb["loss"]
            d, hw, p = two_proportion(ca["win"], n0, cb["win"], n1)
            mark = "有意" if p < 0.05 else "有意差なし"
            print(f"  {opp_name:<14}{d:+7.1%}  [{d-hw:+6.1%},{d+hw:+6.1%}]  p={p:.3f}  {mark}")

    if args.json_out:
        # プログラムから読みやすいよう、arm×opponent の内訳に加えて全相手合算
        # (「合計勝率」。設計書 §5.2: 多重比較を避けるため主指標は合算にする)も書き出す。
        by_arm_opponent = {}
        combined = {}
        for arm in arms:
            combined_c = Counter()
            for opp_name, _ in opponents:
                c = agg.get((arm, opp_name))
                if not c:
                    continue
                by_arm_opponent[f"{arm}:{opp_name}"] = dict(c)
                combined_c += c
            combined[arm] = dict(combined_c)
        payload = {
            "arms": arms,
            "opponents": [n for n, _ in opponents],
            "games_per_cell": args.games,
            "by_arm_opponent": by_arm_opponent,
            "combined": combined,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON集計を書き出しました: {out_path}")


if __name__ == "__main__":
    main()
