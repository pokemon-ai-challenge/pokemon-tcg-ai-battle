"""並列トラジェクトリ収集(multiprocessing、torch非依存ワーカー)。

cg 収集は CPU律速で単一プロセスだと1コアしか使わない。ワーカーを **pure-Python PolicyModel**
(M0 で torch と数値一致を保証済)にして softmax サンプリングで行動を選び、全コアで並列収集する。
torch は main の PPO 更新だけで使う(ワーカーは torch を import しないので起動が軽い)。

各イテレーション:
  main: 現在の torch policy を temp JSON にエクスポート
      -> parallel_collect(temp JSON) でワーカーが収集(現在の重みを読む)
      -> main が PPO 更新
temp JSON 経由の重み配布なので、ワーカーとの torch テンソル共有が要らず堅牢。
"""

from __future__ import annotations

import math
import random
import sys
from multiprocessing import Pool
from multiprocessing import TimeoutError as MpTimeoutError
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_STEPS = 3000

# ワーカーごとのグローバル状態(Pool initializer でセット)。
_W: dict = {}


def _init_worker(weights_path, deck_l, deck_o, temperature):
    from cg.api import to_observation_class  # noqa: F401 (import 確認)
    from ptcg_ai.learning.policy_model import PolicyModel
    _W["pm"] = PolicyModel(weights_path)
    _W["deck_l"] = deck_l
    _W["deck_o"] = deck_o
    _W["temp"] = temperature


def _softmax_sample(scores, temperature, rng):
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    z = sum(exps)
    probs = [e / z for e in exps]
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i, math.log(max(probs[i], 1e-12))
    return len(probs) - 1, math.log(max(probs[-1], 1e-12))


def _valid_wall_guard_action(action, select) -> bool:
    """`ml_policy_agent._is_valid_action` と同内容(モジュール依存を増やさないため複製、
    同ファイルのdocstringにある既存の複製方針と同じ)。"""
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (select.minCount <= len(action) <= select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(select.option) for i in action)


def _play_one(task):
    """1試合を pure-Python 方策(sampling)で。learner の単一選択を記録して dict で返す。

    2026-08-14: learner側の意思決定に wall_guard(Guard A/B + pending target)を pre-step
    として追加(requirements-kamitsuorochi-2026-08-12.md §step2「根本原因、コード確認で確定」)。
    crustle対面のRL(gen0-25、kamitsuorochi_vs_crustle_v1)は
    これが無いまま行われており、本番(ml_policy_agent.py)の対応する呼び出し順序
    (pending -> guard_a -> guard_b -> モデル)と揃っていなかった(collect_parallel.py には
    wall_guard の呼び出しが一切無かった)。相手側(opponent)には適用しない
    (wall_guardは自分の0打点攻撃を直すためのもので、相手の忠実さを変える理由が無い)。
    """
    from cg.api import LogType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import encoder
    from ptcg_ai.search import wall_guard

    learner_index, seed = task
    pm = _W["pm"]
    temp = _W["temp"]
    rng = random.Random(seed ^ 0x5DEECE66D)
    random.seed(seed)
    wall_guard.reset_pending_target()
    wg_config = {"enabled": True}

    deck0, deck1 = (_W["deck_l"], _W["deck_o"]) if learner_index == 0 else (_W["deck_o"], _W["deck_l"])
    steps = []
    reward = 0.0
    winner = None
    error = None
    final_prize_self = None
    final_prize_opp = None
    went_first = None
    final_turn = None
    end_reason = None

    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"steps": [], "reward": 0.0, "winner": None, "error": f"start {start_data.errorType}"}
    n = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"; break
            if cur.result != -1:
                winner = cur.result
                reward = 1.0 if cur.result == learner_index else 0.0
                # 終局時の診断: 残りサイドと先攻/後攻。勝敗(1bit)より情報量が多く、
                # 先攻後攻はポケカでは実質2つの別ゲーム(先攻は1ターン目に攻撃できない)。
                # 追加コストはゼロ(この時点の cur から読むだけ)。
                try:
                    prize_self = len(cur.players[learner_index].prize or [])
                    prize_opp = len(cur.players[1 - learner_index].prize or [])
                    final_prize_self = prize_self
                    final_prize_opp = prize_opp
                    went_first = (cur.firstPlayer == learner_index) if cur.firstPlayer is not None else None
                    final_turn = cur.turn
                except Exception:  # noqa: BLE001 - 診断が本流を壊さないこと
                    pass
                # 終局理由(LogType.RESULT の reason: 1=サイド0 2=山札切れ 3=バトル場不在 4=カード効果)。
                # obs.logs は「前回選択以降のイベント」なので、終局直前の this obs に含まれるはず。
                # スモークテストで実地確認済み(reason フィールド名は api.py コメント通り)。
                try:
                    result_logs = [lg for lg in obs.logs if lg.type == LogType.RESULT]
                    if result_logs:
                        end_reason = result_logs[-1].reason
                except Exception:  # noqa: BLE001 - 診断が本流を壊さないこと
                    pass
                break
            if n >= MAX_STEPS:
                error = "max_steps"; break
            select = obs.select
            if cur.yourIndex == learner_index:
                wg_action = None
                if select is not None and select.option:
                    # 本番(ml_policy_agent._select_action)と同じ順序: pending -> Guard A -> Guard B。
                    # いずれも盤面(obs)だけから判定するpre-stepで、モデルのスコアは使わない。
                    # ml_policy_agent の _try_wall_guard_* と同じく例外はここで握りつぶし、
                    # 「このガードは発火しない」として通常経路にフォールバックする
                    # (1試合が丸ごと error 扱いで捨てられるのを防ぐ)。
                    try:
                        wg_action = wall_guard.try_consume_pending_target(obs, wg_config)
                        if wg_action is None:
                            wg_action = wall_guard.guard_a_retreat(obs, wg_config)
                        if wg_action is None:
                            wg_action = wall_guard.guard_b_boss_orders(obs, wg_config)
                    except Exception:  # noqa: BLE001 - ガードの失敗で1試合を丸ごと捨てない
                        wg_action = None
                    if wg_action is not None and not _valid_wall_guard_action(wg_action, select):
                        wg_action = None

                if wg_action is not None:
                    # wall_guard の決定はモデルのサンプリングではないので PPO の学習対象には
                    # しない(本番でもこの経路ではモデルを一切呼ばない)。
                    action = wg_action
                elif select is not None and select.option and select.maxCount == 1:
                    sf = encoder.encode_state_from_state(cur)
                    of = encoder.encode_options_from_state(cur, select)
                    ci = encoder.encode_option_card_ids(cur, select)
                    if of:
                        scores = [pm._forward(sf, of[i], ci[i]) for i in range(len(of))]
                        idx, logp = _softmax_sample(scores, temp, rng)
                        steps.append({"state_feat": sf, "option_feats": of,
                                      "card_ids": ci, "chosen_idx": idx, "logprob": logp})
                        action = [idx]
                    else:
                        action = [0]
                elif select is not None and select.option:
                    # 複数選択: greedy(記録しない)
                    scores = pm.score_options(obs)
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                              if scores else list(range(count)))
                else:
                    action = []
            else:
                # 相手も同じ pure-Python 方策(argmax)。相手を production alakazam にするには
                # 別 weights を渡す設計にできるが、PoC は learner と同一方策プールで対戦させず
                # opponent 用 PolicyModel を別に持つ(下記 _W["opp"])。
                opp = _W.get("opp")
                if select is None or not select.option:
                    action = []
                elif select.maxCount == 1:
                    oi = opp.select_option(obs) if opp is not None else 0
                    action = [oi if oi is not None else 0]
                else:
                    osc = opp.score_options(obs) if opp is not None else []
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:count]
                              if osc else list(range(count)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()
    return {"steps": steps, "reward": reward, "winner": winner, "error": error,
            "prize_self": final_prize_self, "prize_opp": final_prize_opp,
            "went_first": went_first, "final_turn": final_turn, "end_reason": end_reason}


def _init_worker2(weights_path, opp_weights, deck_l, deck_o, temperature):
    from ptcg_ai.learning.policy_model import PolicyModel
    pm = PolicyModel(weights_path)
    # 未ロード(パス誤り等)を黙って index0 固定の壊れた方策として使うと結果が偽陽性になる。
    # 明示的に fail-loud にする(learner は必ずロード済であるべき)。
    if not pm.is_ready:
        raise RuntimeError(f"learner policy not ready (path?): {weights_path}")
    _W["pm"] = pm
    opp = PolicyModel(opp_weights)  # opp_weights=None -> production alakazam(絶対デフォルトパス)
    if not opp.is_ready:
        raise RuntimeError(f"opponent policy not ready (path?): {opp_weights}")
    _W["opp"] = opp
    _W["deck_l"] = deck_l
    _W["deck_o"] = deck_o
    _W["temp"] = temperature


# 2026-08-13 実測: initializer(_init_worker2)が例外を投げると、multiprocessing.Pool は
# 死んだ子プロセスを黙って無限に再spawnし続ける(Pool 自体に「初期化失敗が続いたら諦める」
# 仕組みが無い)。親プロセスには何も伝播しないので pool.map() は一生完了せず、実際には
# 一瞬で分かる設定ミス(重みのパス間違い・次元不一致)が「タイムアウトまでハング」に化ける。
# ローカル再現: 166次元(旧)の重みを故意に渡すと90秒で321回 respawn し、例外は一度も
# 親へは上がらない。詳細は kaggle_replays/rl/distributed/README.md の Kaggle 運用ノート参照。
#
# 対策は2段構え:
#  1. _preflight_check_policies: 本番 Pool を作る**前**に親プロセス側で PolicyModel を
#     読み込み、is_ready を確認する。既知の失敗モード(パス間違い・次元不一致)はこれで
#     ミリ秒〜数秒で検出でき、Pool を1つも作らずに済む。
#  2. _probe_pool_initializer: 1に引っかからない未知の失敗モード向けの一般的な安全網。
#     本番と全く同じ initializer/initargs で processes=1 の使い捨て Pool を作り、
#     ダミータスクの結果を短いタイムアウトで待つ。initializer が失敗し続けていれば
#     タイムアウトで検出でき、1800秒待たずに数十秒で「原因不明だが初期化が終わらない」
#     ことが分かる(健全なときは実際の初期化時間だけで完了し、追加コストはほぼ無い)。
_PROBE_TIMEOUT_SECONDS = 30


def _preflight_check_policies(weights_path, opp_weights) -> None:
    """Pool を作る前に、learner/opponent 両方の PolicyModel が読み込めることを確認する。
    失敗なら即座に例外(Pool の無限 respawn ループに入る前に落とす)。"""
    from ptcg_ai.learning.policy_model import PolicyModel
    pm = PolicyModel(weights_path)
    if not pm.is_ready:
        raise RuntimeError(
            f"[preflight] learner policy not ready (path/dim mismatch?): {weights_path}")
    opp = PolicyModel(opp_weights)
    if not opp.is_ready:
        raise RuntimeError(
            f"[preflight] opponent policy not ready (path/dim mismatch?): {opp_weights}")


def _noop(x):
    return x


def _probe_pool_initializer(weights_path, opp_weights, deck_l, deck_o, temperature) -> None:
    """本番の Pool を作る前に、initializer が実際に完走するかを短いタイムアウトで確認する
    一般的な安全網(未知の失敗モード向け。既知の失敗モードは _preflight_check_policies が
    先に、もっと速く・具体的なメッセージで捕まえる)。"""
    probe = Pool(processes=1, initializer=_init_worker2,
                 initargs=(weights_path, opp_weights, deck_l, deck_o, temperature))
    try:
        probe.apply_async(_noop, (1,)).get(timeout=_PROBE_TIMEOUT_SECONDS)
    except MpTimeoutError:
        raise RuntimeError(
            f"[preflight] Pool initializer が {_PROBE_TIMEOUT_SECONDS}秒以内に完走しなかった。"
            "initializer が例外を出し続けて multiprocessing.Pool が子プロセスを無限に "
            "再spawn している可能性が高い(標準エラー出力に子プロセスのトレースバックが "
            "出ているはずなので確認する)。"
        ) from None
    finally:
        probe.terminate()
        probe.join()


def parallel_collect(weights_path, opp_weights, deck_l, deck_o, n_games, seed0,
                     temperature, workers):
    """ワーカー並列で n_games 収集。learner は weights_path(temp JSON)、相手は opp_weights。
    戻り: (trajectories(list[dict]), wins, valid, errors)。

    本番 Pool を作る前に、initializer が確実に完走することを確認する(上記コメント参照)。
    ここで検出された失敗は例外としてすぐ伝播する(呼び出し元は worker.py の main で、
    そこで SystemExit / traceback として表面化する)。"""
    _preflight_check_policies(weights_path, opp_weights)
    _probe_pool_initializer(weights_path, opp_weights, deck_l, deck_o, temperature)

    tasks = [(g % 2, seed0 + g) for g in range(n_games)]
    with Pool(processes=workers, initializer=_init_worker2,
              initargs=(weights_path, opp_weights, deck_l, deck_o, temperature)) as pool:
        results = pool.map(_play_one, tasks, chunksize=max(1, n_games // (workers * 4)))
    trajs, wins, valid, errors = [], 0, 0, 0
    for r in results:
        if r["error"] is not None:
            errors += 1
            continue
        valid += 1
        wins += 1 if r["reward"] >= 1.0 else 0
        if r["steps"]:
            trajs.append(r)
    return trajs, wins, valid, errors
