"""相手プールに対する並列トラジェクトリ収集(単一 Pool、タスクごとに相手を切替)。

``collect_parallel.py`` は「学習側1モデル vs 相手1モデル固定」でしか収集できない。
本モジュールは「学習側1モデル vs 相手プール」に拡張する(相互鍛錬フェーズ1)。

設計方針:
  - ``collect_parallel._play_one`` のロジックはコピーせず、必ず import して呼ぶ
    (二重管理を避ける)。
  - Pool の initializer で学習側 PolicyModel を1つ、相手 PolicyModel を相手の数だけロードし、
    学習側は ``collect_parallel._W`` に、相手プールは本モジュール側の ``_POOL`` に保持する。
  - task 関数は呼び出しごとに ``collect_parallel._W["opp"]`` / ``["deck_o"]`` を差し替えてから
    ``collect_parallel._play_one`` を呼ぶ(1ワーカーが同じプロセス内で使い回すグローバルの
    差し替えなので、Pool 自体は1つのままでよい)。

``collect_parallel.py`` はチーム既存資産のため変更しない。

seed の意味について(重要、誤解しやすい):
  - ``seed0``(および ``build_tasks`` が各タスクに割り当てる seed)は
    ``collect_parallel._play_one`` 内の ``rng = random.Random(seed ^ ...)`` を経由して
    **方策のsoftmaxサンプリングのみ**を決める。
  - cg 側(``sample_submission/cg/game.py`` の ``lib.BattleStart``)にはシードを渡す引数が
    存在せず、``cg/`` 内に seed/srand を設定する呼び出しも見当たらない。つまり **山札シャッフルや
    初手といった試合内乱数は Python 側の seed では制御できない**。
  - 実際、同一 seed0・同一コマンドで48試合を4回実行したところ wins は 3, 6, 14, 7 とばらついた
    (workers=1 に落としても再現しない)。
  - したがって「同一シードで対を作る」共通乱数分散削減は成立しない。A/B比較は対応なしの統計量
    (Wilson区間・二標本比率検定など)で扱う。本モジュールが提供できる分散対策は
    ``build_tasks`` による**相手ごとの先攻/後攻の均等化**のみ。
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_parallel  # noqa: E402

# 相手プールごとの seed 空間の間隔。per(相手あたりの試合数)はこれ未満でなければならない。
# 超えると相手間で seed が衝突する。seed は方策サンプリングのみを決める(上記 docstring)ので
# 試合が同一になるわけではないが、サンプリングの乱数列が重複する理由が無いので避ける。
SEED_STRIDE = 100_000

# ワーカーごとのグローバル状態(Pool initializer でセット)。相手プール専用。
# collect_parallel._W は学習側(pm/deck_l/temp)専用のまま使う。
_POOL: dict = {"opps": [], "decks": [], "names": []}


def _init_worker_pool(weights_path, opponents, deck_l, temperature):
    """Pool initializer。学習側1つ + 相手プールをまとめてロードする。

    opponents: [(name, opp_weights_path_or_None, deck_o), ...]
    """
    from ptcg_ai.learning.policy_model import PolicyModel

    pm = PolicyModel(weights_path)
    if not pm.is_ready:
        raise RuntimeError(f"learner policy not ready (path?): {weights_path}")
    collect_parallel._W["pm"] = pm
    collect_parallel._W["deck_l"] = deck_l
    collect_parallel._W["temp"] = temperature

    names, opps, decks = [], [], []
    for name, opp_weights, deck_o in opponents:
        opp = PolicyModel(opp_weights)
        # opp_weights=None は production alakazam(絶対デフォルトパス)として正当。
        # それ以外で未ロードは fail-loud(黙って壊れた方策で回すと偽陽性になる)。
        if not opp.is_ready:
            raise RuntimeError(f"opponent policy not ready (path?): name={name} weights={opp_weights}")
        names.append(name)
        opps.append(opp)
        decks.append(deck_o)
    _POOL["names"] = names
    _POOL["opps"] = opps
    _POOL["decks"] = decks


def _play_one_pool(task):
    """task = (opp_idx, learner_index, seed)。相手を差し替えてから _play_one を呼ぶ。"""
    opp_idx, learner_index, seed = task
    collect_parallel._W["opp"] = _POOL["opps"][opp_idx]
    collect_parallel._W["deck_o"] = _POOL["decks"][opp_idx]
    result = collect_parallel._play_one((learner_index, seed))
    result["opponent"] = _POOL["names"][opp_idx]
    result["opp_idx"] = opp_idx
    result["learner_index"] = learner_index
    return result


def build_tasks(k, n_games, seed0):
    """相手プール用のタスク列を生成する純関数(cg 非依存、副作用なし、単体テスト対象)。

    各相手に n_games // k 試合を割り当て、相手ごとの先攻/後攻(learner_index)を均等にするため
    偶数に切り下げる。相手 i のシードは ``seed0 + i * SEED_STRIDE + g`` で、相手間のシード空間が
    重ならないようにしてある(per が SEED_STRIDE 以上だと衝突するので ValueError)。

    戻り: (tasks, per, dropped_games)
      tasks: [(opp_idx, learner_index, seed), ...]
      per: 相手1人あたりの試合数(偶数に丸め済み。0 以上)
      dropped_games: 丸め・端数で切り捨てられた試合数(= n_games - per * k)
    """
    if k <= 0:
        raise ValueError(f"k は正の整数でなければならない: k={k}")
    if n_games < 0:
        raise ValueError(f"n_games は非負でなければならない: n_games={n_games}")

    per = n_games // k
    if per % 2 != 0:
        per -= 1  # 偶数に丸める(相手ごとの先攻/後攻を均等にするため)。
    if per < 0:
        per = 0
    if per >= SEED_STRIDE:
        raise ValueError(f"per({per}) が SEED_STRIDE({SEED_STRIDE}) 以上。シードが相手間で衝突する。")

    dropped_games = n_games - per * k

    tasks = []
    for i in range(k):
        for g in range(per):
            tasks.append((i, g % 2, seed0 + i * SEED_STRIDE + g))

    return tasks, per, dropped_games


def parallel_collect_pool(weights_path, opponents, deck_l, n_games, seed0,
                           temperature, workers):
    """相手プールに対して n_games 収集する。

    opponents: [(name, opp_weights_path_or_None, deck_o), ...]
    戻り: (trajectories, stats)
    """
    k = len(opponents)
    if k == 0:
        raise ValueError("opponents は空にできない")

    names = [name for name, _, _ in opponents]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"opponents の name に重複がある(統計が黙って合算される): {dupes}")

    tasks, per, dropped_games = build_tasks(k, n_games, seed0)

    per_opponent = {
        name: {"games": 0, "valid": 0, "wins": 0, "errors": 0,
               "seat0": 0, "seat1": 0, "steps": 0}
        for name in names
    }
    total = {"games": 0, "valid": 0, "wins": 0, "errors": 0,
             "seat0": 0, "seat1": 0, "steps": 0}

    if per == 0:
        # per==0 (n_games < k、または丸めで0になった場合)はタスクが空なので、Pool を
        # 起動しても起動コストが無駄になるだけ。作らずに空の trajectories と全ゼロの
        # stats を即返す。
        stats = {"per_opponent": per_opponent, "total": total,
                 "dropped_games": dropped_games, "per": per}
        return [], stats

    with Pool(processes=workers, initializer=_init_worker_pool,
              initargs=(weights_path, opponents, deck_l, temperature)) as pool:
        results = pool.map(_play_one_pool, tasks,
                            chunksize=max(1, len(tasks) // (workers * 4)) if tasks else 1)

    trajs = []
    for r in results:
        name = r["opponent"]
        bucket = per_opponent[name]
        bucket["games"] += 1
        total["games"] += 1
        if r["learner_index"] == 0:
            bucket["seat0"] += 1
            total["seat0"] += 1
        else:
            bucket["seat1"] += 1
            total["seat1"] += 1

        if r["error"] is not None:
            bucket["errors"] += 1
            total["errors"] += 1
            continue
        bucket["valid"] += 1
        total["valid"] += 1
        if r["reward"] >= 1.0:
            bucket["wins"] += 1
            total["wins"] += 1
        if r["steps"]:
            bucket["steps"] += len(r["steps"])
            total["steps"] += len(r["steps"])
            trajs.append(r)

    stats = {
        "per_opponent": per_opponent,
        "total": total,
        "dropped_games": dropped_games,
        "per": per,
    }
    return trajs, stats


if __name__ == "__main__":
    # 手動スモークテスト用(本体の検証は test_collect_pool.py)。
    print("collect_pool.py はライブラリモジュールです。検証は test_collect_pool.py を実行してください。")
