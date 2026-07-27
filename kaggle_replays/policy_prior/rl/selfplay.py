"""自己対戦、または相手プールとの対戦のログ収集。プロセス並列(``sample_submission/tests/
local_sim/parallel_eval.py`` と同じ作法)。

## なぜプロセス並列なのか

``cg`` はプロセス全体で1つの対局状態を持つ(``battle_start`` / ``battle_select`` /
``battle_finish``)。スレッドを分けても同じ対局を奪い合うだけで壊れるため、
プロセスを分けて各自が独立したライブラリのインスタンスを持つようにする
(``parallel_eval.py`` の docstring 参照。本モジュールも同じ制約に従う)。

## 相手の多様化(オプション)

``opponent_pool`` を渡すと、1試合ごとに相手プールから(デッキ, 専用モデルの重み)を
巡回で選び、純粋なミラーマッチ(同デッキ・同方策)ではなく相手依存の対局を作る
(``sample_submission/tests/local_sim/parallel_eval.py`` の ``--opponent-pool`` と同じ
考え方)。省略時(``None``)は従来どおりの純粋な自己対戦(同デッキ・同方策)。

## 【最重要・正しさに関わる】どちらの側の decision を記録するか

方策勾配 ``∇_w log pi(a)`` は「自分の方策 pi が選んだ行動 a」に対する勾配である。

- **純粋な自己対戦**(``opponent_pool=None``): 両席とも同じ方策 ``policy`` を使って
  行動を選んでいるので、両席の decision を記録してよい(1試合 -> 最大2エピソード)。
  これが従来の挙動。
- **相手プールとの対戦**(``opponent_pool`` 指定): 相手側の行動は「相手の方策」が
  選んだものであり、「自分の方策」が選んだものではない。もしここで相手側の
  decision も記録し、自分の方策 pi の確率を使って ``log pi(a_opponent)`` の勾配を
  計算すると、方策勾配ではない別の量(他人が選んだ行動を自分の方策のパラメータで
  説明しようとする、オフポリシーの補正を欠いた量)を最適化してしまう。これは
  符号を間違えるのと同様に「それらしく動くが学習が壊れている」バグになる
  (``rl.reinforce`` の docstring 参照: 勾配バグは発見が難しい)。
  したがって相手プールモードでは **自分側の decision だけ** を記録する
  (1試合 -> 最大1エピソード)。相手側の行動は「相手の方策の argmax」
  (本番の推論と同じ、温度サンプリングなし)で選ぶが記録しない。

## 何を記録するか

``ptcg_ai.action_selection.selector._ml_policy_action`` と同じ経路(MAIN文脈のみ)を
モンキーパッチして差し替える。差し替え後の関数は、記録対象の席については argmax の
代わりに温度付き softmax サンプリングを行い、各 decision について

- 選択肢ごとの特徴 dict(``policy.filter_known`` で語彙に絞ったもの)
- サンプリングで選ばれた index
- その時点の自分・相手の残りサイド枚数(``observable_state`` の ``n_prize``)

を記録する。特徴抽出(``ptcg_ai.learning.policy_features``)・状態抽出
(``observable_state``)・Option解決(``resolve_option``)はいずれも唯一の実装を
そのまま import する(再実装しない)。

試合終了後、決着がついた試合だけを対象に、記録対象の席(純粋自己対戦なら0/1
両方、相手プールモードなら自分の着席のみ)ごとに「その席の decision 列 + その席
から見た試合結果(勝ち+1/負け-1) + 終端状態(その席から見た最終サイド枚数)」を
1エピソードとして返す。lethal search が発火した decision(本番と同じ挙動)や MAIN
以外の文脈はこの方策を経由しないため記録されない(設計書 §8-5: MAIN 以外は RL の
対象外)。
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICY_PRIOR_DIR = _HERE.parent
_REPO_ROOT = _POLICY_PRIOR_DIR.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
_OPPONENT_DECKS_PATH = _SAMPLE_SUBMISSION_DIR / "tests" / "local_sim" / "opponent_decks.json"
# アーキタイプ別の学習済み重み(再生成可能なので .gitignore 済み。無ければ
# resolve_opponent_pool が警告して純粋な自己対戦にフォールバックする)。
_ARCHETYPE_WEIGHTS_DIR = _POLICY_PRIOR_DIR / "output"

STEP_CAP = 2000  # parallel_eval.py と同じ: これを超えたら未決着として無効試合に数える


def resolve_opponent_pool(names: str | list[str] | None) -> list[dict] | None:
    """アーキタイプ名から ``collect_selfplay(opponent_pool=...)`` 用のリストを作る。

    Args:
        names: ``None``(既定)なら ``None`` を返す(=純粋な自己対戦)。
            カンマ区切り文字列(例 ``"archetype1,archetype2"``)、``"all"``
            (``opponent_decks.json`` の全キー)、または名前のリストを受け付ける。

    相手デッキは ``opponent_decks.json``、相手専用モデルは
    ``policy_prior/output/policy_weights_<name>.json`` から読む
    (``sample_submission/tests/local_sim/parallel_eval.py`` の ``--opponent-pool``
    と同じレイアウト)。専用モデルの重みは再生成可能なので .gitignore 済みで、
    別環境には無いことがある。1つでも見つからなければ例外にせず、警告を出して
    ``None``(=従来どおりの純粋な自己対戦)にフォールバックする(一部だけ欠けた
    プールで学習すると相手の分布が意図せず偏るため、部分利用はしない)。
    """
    if names is None:
        return None
    decks = json.loads(_OPPONENT_DECKS_PATH.read_text(encoding="utf-8"))
    if isinstance(names, str):
        name_list = sorted(decks) if names == "all" else [n.strip() for n in names.split(",") if n.strip()]
    else:
        name_list = list(names)
    if not name_list:
        return None

    missing = []
    opponents = []
    for name in name_list:
        if name not in decks:
            raise ValueError(f"opponent_decks.json に '{name}' がありません(既知: {sorted(decks)})")
        weights_path = _ARCHETYPE_WEIGHTS_DIR / f"policy_weights_{name}.json"
        if not weights_path.exists():
            missing.append(name)
            continue
        opponents.append(
            {"name": name, "deck": decks[name]["deck"], "weights_path": str(weights_path)}
        )

    if missing:
        print(
            f"警告: 相手専用モデルが見つかりません({missing})。"
            f"{_ARCHETYPE_WEIGHTS_DIR} に policy_weights_<name>.json が必要です"
            "(再生成: python policy_prior/train.py --archetype <name> "
            "--out policy_prior/output/policy_weights_<name>.json)。"
            "純粋な自己対戦にフォールバックします。",
            file=sys.stderr,
        )
        return None
    return opponents


def _game_opponent_and_seat(
    pool_size: int, opponent_start_index: int, seat_start: int, game_i: int
) -> tuple[int, int]:
    """相手プールモードで、ジョブ内の ``game_i`` 番目の対局に使う

    (相手プール内インデックス, 自分の着席(0/1))

    を返す純粋関数(``cg`` に依存しない)。相手は巡回(round-robin)で選び、着席は
    交互に入れ替える。``opponent_start_index`` / ``seat_start`` はジョブが受け持つ
    最初の対局より前に何試合分が既に処理されたか(累積試合数)を表し、これにより
    ワーカーをまたいでも全体で巡回・交互化が連続する(1プロセスで全試合を順に
    処理した場合と同じ割り当てになる)。純粋関数なので実対局を回さずに単体テスト
    できる(``tests/test_selfplay_opponents.py`` 参照)。
    """
    global_i = opponent_start_index + game_i
    opp_index = global_i % pool_size
    seat = (seat_start + game_i) % 2
    return opp_index, seat


def _build_jobs(
    weights_path: str,
    games: int,
    temperature: float,
    workers: int,
    seed: int,
    deck: list[int],
    opponent_pool: list[dict] | None,
) -> list[dict]:
    """ワーカーごとのジョブ辞書を組み立てる純粋関数(``cg`` に依存しない)。

    ``offset``(これまでの累積試合数)をワーカー間で引き継ぐことで、相手の巡回
    (``opponent_start_index``)と先手/後手の交互化(``seat_start``)が単一プロセスで
    全試合を順に処理した場合と同じ結果になるようにする(``parallel_eval.build_jobs``
    の ``seat_start`` と同じ考え方)。単体テストで割り当ての均等性・交互性を検証
    できるよう、実際のジョブ実行(``_run_selfplay_chunk``)とは切り離してある。
    """
    jobs = []
    base, extra = divmod(games, workers)
    offset = 0
    for w in range(workers):
        n = base + (1 if w < extra else 0)
        if n == 0:
            continue
        jobs.append(
            {
                "weights_path": str(weights_path),
                "temperature": temperature,
                "deck": deck,
                "games": n,
                "seed": seed + w * 7919,
                "opponent_pool": opponent_pool,
                "opponent_start_index": offset % len(opponent_pool) if opponent_pool else 0,
                "seat_start": offset % 2,
            }
        )
        offset += n
    return jobs


def _run_selfplay_chunk(job: dict) -> dict:
    """1ワーカー分の自己対戦(または相手プールとの対戦)。子プロセスで実行されるので
    import はこの中で行う(``parallel_eval._run_chunk`` と同じ理由: 環境変数・
    sys.path・cwd を子プロセス側で独立に整えるため)。
    """
    if str(_SAMPLE_SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))
    if str(_POLICY_PRIOR_DIR) not in sys.path:
        sys.path.insert(0, str(_POLICY_PRIOR_DIR))
    os.chdir(_SAMPLE_SUBMISSION_DIR)

    from cg.api import Observation, SelectContext, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from main import agent
    from ptcg_ai.action_selection import selector
    from ptcg_ai.hidden_information import match_context
    from ptcg_ai.learning import policy_features
    from ptcg_ai.learning.observable_state import observable_state
    from ptcg_ai.learning.semantic_action import resolve_option

    from rl.policy import LinearPolicy, argmax_index, sample_index, softmax  # noqa: E402

    rng = random.Random(job["seed"])
    own_policy = LinearPolicy.load(job["weights_path"])
    temperature = float(job["temperature"])
    own_deck = job["deck"]

    opponent_pool: list[dict] | None = job.get("opponent_pool")
    pure_selfplay = opponent_pool is None
    # 相手専用モデルは対局中変わらないので、ジョブ開始時に1回だけロードする
    # (試合ごとに読み直さない)。
    opp_policies = (
        [LinearPolicy.load(o["weights_path"]) for o in opponent_pool]
        if opponent_pool
        else []
    )
    opponent_start_index = int(job.get("opponent_start_index", 0))
    seat_start = int(job.get("seat_start", 0))

    # 1試合ぶんの可変状態(次の battle_start の直前に設定し、モンキーパッチ内で参照する)。
    # クロージャで参照するので dict に包む(int/str は再代入すると別オブジェクトになり
    # 内側の関数から見えなくなるため)。
    game_ctx: dict = {"my_seat": 0, "opp_policy": None}

    # 1試合分のバッファ(seat 0/1 それぞれの decision 列)。クロージャで参照・毎試合クリアする。
    buffers: dict[int, list[dict]] = {0: [], 1: []}

    def _sampling_ml_action(obs: Observation, select):
        """``selector._ml_policy_action`` の差し替え版。MAIN以外は None(通常のrouterへ)。

        記録するのは「自分の方策が選んだ行動」だけ(モジュール docstring 参照)。
        純粋な自己対戦では両席とも「自分の方策」なので両方記録する。相手プール
        モードでは ``game_ctx["my_seat"]`` の席だけが「自分の方策」であり、
        もう一方は相手の専用方策(argmax、記録しない)。
        """
        if select.context != SelectContext.MAIN or obs.current is None:
            return None
        try:
            current_dict = dataclasses.asdict(obs.current)
            me = current_dict.get("yourIndex", 0)
            state = observable_state(current_dict, me)
            actions = [
                resolve_option(dataclasses.asdict(option), current_dict, me)
                for option in select.option
            ]
            is_own = pure_selfplay or (me == game_ctx["my_seat"])
            active_policy = own_policy if is_own else game_ctx["opp_policy"]
            feats = [
                policy_features.extract_features(
                    state, a, active_policy.card_attributes, active_policy.frequent_card_ids
                )
                for a in actions
            ]
            scores = active_policy.score_many(feats)
            if is_own:
                # 自分の方策: 学習用に温度付きサンプリング(記録対象)
                probs = softmax(scores, temperature)
                idx = sample_index(probs, rng)
            else:
                # 相手の方策: 本番の推論と同じ argmax(サンプリングしない・記録しない)
                idx = argmax_index(scores)
        except Exception:  # noqa: BLE001 -- サンプリングが失敗してもターンを止めない
            return None
        if is_own:
            buffers[me].append(
                {
                    "features": [own_policy.filter_known(f) for f in feats],
                    "chosen_index": idx,
                    "own_n_prize": state["own"]["n_prize"],
                    "opp_n_prize": state["opponent"]["n_prize"],
                }
            )
        return [idx]

    orig_ml_action = selector._ml_policy_action
    selector._ml_policy_action = _sampling_ml_action

    episodes: list[dict] = []
    try:
        for game_i in range(job["games"]):
            match_context.reset()
            buffers[0].clear()
            buffers[1].clear()

            if pure_selfplay:
                d0, d1 = own_deck, own_deck
                my_seat = 0  # 未使用(is_own は常に True)だが診断用に固定しておく
                opp_name = None
            else:
                opp_i, my_seat = _game_opponent_and_seat(
                    len(opponent_pool), opponent_start_index, seat_start, game_i
                )
                opp = opponent_pool[opp_i]
                opp_name = opp.get("name")
                game_ctx["opp_policy"] = opp_policies[opp_i]
                # 先手・後手を交互に入れ替える(着手順の有利が偏らないように)
                game_ctx["my_seat"] = my_seat
                d0, d1 = (own_deck, opp["deck"]) if my_seat == 0 else (opp["deck"], own_deck)

            obs_dict, start = battle_start(d0, d1)
            if start.errorType != 0:
                battle_finish()
                raise RuntimeError(f"battle_start errorType={start.errorType}")

            steps = 0
            result = -1
            final_current: dict | None = None
            try:
                while True:
                    obs = to_observation_class(obs_dict)
                    if obs.current is not None and obs.current.result != -1:
                        result = obs.current.result
                        final_current = dataclasses.asdict(obs.current)
                        break
                    obs_dict = battle_select(agent(obs_dict))
                    steps += 1
                    if steps > STEP_CAP:
                        result = -1
                        break
            finally:
                battle_finish()

            if result not in (0, 1) or final_current is None:
                continue  # 未決着(無効試合)はログに残さない

            # 記録対象の席: 純粋自己対戦なら0/1両方、相手プールモードなら自分の席だけ
            # (もう一方の buffers は _sampling_ml_action が積まないので自然に空になる)。
            for seat in (0, 1):
                decisions = buffers[seat]
                if not decisions:
                    continue
                final_state = observable_state(final_current, seat)
                episode = {
                    "result": 1 if result == seat else -1,
                    "decisions": decisions,
                    "terminal": {
                        "own_n_prize": final_state["own"]["n_prize"],
                        "opp_n_prize": final_state["opponent"]["n_prize"],
                    },
                    # 診断・テスト用(reinforce.py は使わない): この decision 列を
                    # 記録した席(0/1)。相手プールモードでの先手/後手の交互化を
                    # 外側から検証できるようにする。
                    "seat": seat,
                }
                if opp_name is not None:
                    episode["opponent"] = opp_name
                episodes.append(episode)
                # 次の試合のために新しいリストを積む(参照を使い回さない)
                buffers[seat] = []
    finally:
        selector._ml_policy_action = orig_ml_action

    return {"episodes": episodes, "games_played": job["games"]}


def collect_selfplay(
    weights_path: str | Path,
    games: int,
    temperature: float,
    workers: int | None = None,
    seed: int = 20260727,
    deck: list[int] | None = None,
    opponent_pool: list[dict] | None = None,
) -> list[dict]:
    """自己対戦(または相手プールとの対戦)を並列実行し、決着がついた試合ぶんの
    エピソードを集めて返す。

    Args:
        weights_path: 現在の方策の重みJSON(``LinearPolicy`` 契約)。
        games: 総試合数(ワーカー間で均等に割る)。
        temperature: サンプリング温度。
        workers: 並列プロセス数。省略時は論理コア数-2。
        seed: 乱数種の基準値(ワーカーごとにずらす)。
        deck: 自分側に使う60枚デッキ。省略時は ``sample_submission/deck.csv``。
        opponent_pool: 指定すると純粋な自己対戦の代わりに相手プールと対戦する。
            各要素は ``{"name": str, "deck": list[int], "weights_path": str}``。
            1試合ごとに巡回で選ぶ(全体で各相手がほぼ均等に当たる)ので、
            ``games`` が ``len(opponent_pool)`` の倍数でなくても差は最大1試合に
            収まる。省略時(``None``、既定)は従来どおりの純粋な自己対戦
            (同デッキ・同方策で、両席の decision を記録する)。
            指定時は自分側の decision だけを記録する(モジュール docstring 参照:
            相手側の行動を自分の方策の勾配で学習するとオフポリシーになるため)。
    """
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    # ワーカーは sample_submission へ chdir するので、相対パスのまま渡すと解決できない。
    # 呼び出し元のカレントディレクトリを基準に絶対パスへ解決してから渡す。
    weights_path = str(Path(weights_path).resolve())
    if opponent_pool is not None:
        opponent_pool = [
            {**o, "weights_path": str(Path(o["weights_path"]).resolve())}
            for o in opponent_pool
        ]
        if not opponent_pool:
            raise ValueError("opponent_pool を指定する場合は空にできません")
    if deck is None:
        if str(_SAMPLE_SUBMISSION_DIR) not in sys.path:
            sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))
        cwd = os.getcwd()
        os.chdir(_SAMPLE_SUBMISSION_DIR)
        try:
            from main import read_deck_csv

            deck = read_deck_csv()
        finally:
            os.chdir(cwd)

    jobs = _build_jobs(weights_path, games, temperature, workers, seed, deck, opponent_pool)

    episodes: list[dict] = []
    games_played = 0
    with ProcessPoolExecutor(max_workers=workers) as pool_exec:
        for r in pool_exec.map(_run_selfplay_chunk, jobs):
            episodes.extend(r["episodes"])
            games_played += r["games_played"]
    return episodes
