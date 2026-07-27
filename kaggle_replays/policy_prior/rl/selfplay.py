"""自己対戦(ミラーマッチ)のログ収集。プロセス並列(``sample_submission/tests/local_sim/
parallel_eval.py`` と同じ作法)。

## なぜプロセス並列なのか

``cg`` はプロセス全体で1つの対局状態を持つ(``battle_start`` / ``battle_select`` /
``battle_finish``)。スレッドを分けても同じ対局を奪い合うだけで壊れるため、
プロセスを分けて各自が独立したライブラリのインスタンスを持つようにする
(``parallel_eval.py`` の docstring 参照。本モジュールも同じ制約に従う)。

## 何を記録するか

``ptcg_ai.action_selection.selector._ml_policy_action`` と同じ経路(MAIN文脈のみ)を
モンキーパッチして差し替える。差し替え後の関数は、argmax の代わりに温度付き
softmax サンプリングを行い、各 decision について

- 選択肢ごとの特徴 dict(``policy.filter_known`` で語彙に絞ったもの)
- サンプリングで選ばれた index
- その時点の自分・相手の残りサイド枚数(``observable_state`` の ``n_prize``)

を記録する。特徴抽出(``ptcg_ai.learning.policy_features``)・状態抽出
(``observable_state``)・Option解決(``resolve_option``)はいずれも唯一の実装を
そのまま import する(再実装しない)。

試合終了後、決着がついた試合だけを対象に、着席(0/1)ごとに
「その席の decision 列 + その席から見た試合結果(勝ち+1/負け-1) + 終端状態
(その席から見た最終サイド枚数)」を1エピソードとして返す。lethal search が
発火した decision(本番と同じ挙動)や MAIN 以外の文脈はこの方策を経由しない
ため記録されない(設計書 §8-5: MAIN 以外は RL の対象外)。
"""

from __future__ import annotations

import dataclasses
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICY_PRIOR_DIR = _HERE.parent
_REPO_ROOT = _POLICY_PRIOR_DIR.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"

STEP_CAP = 2000  # parallel_eval.py と同じ: これを超えたら未決着として無効試合に数える


def _run_selfplay_chunk(job: dict) -> dict:
    """1ワーカー分の自己対戦。子プロセスで実行されるので import はこの中で行う
    (``parallel_eval._run_chunk`` と同じ理由: 環境変数・sys.path・cwd を子プロセス側で
    独立に整えるため)。
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

    from rl.policy import LinearPolicy, sample_index, softmax  # noqa: E402

    rng = random.Random(job["seed"])
    policy = LinearPolicy.load(job["weights_path"])
    temperature = float(job["temperature"])
    deck = job["deck"]

    # 1試合分のバッファ(seat 0/1 それぞれの decision 列)。クロージャで参照・毎試合クリアする。
    buffers: dict[int, list[dict]] = {0: [], 1: []}

    def _sampling_ml_action(obs: Observation, select):
        """``selector._ml_policy_action`` の差し替え版。MAIN以外は None(通常のrouterへ)。"""
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
            feats = [
                policy_features.extract_features(
                    state, a, policy.card_attributes, policy.frequent_card_ids
                )
                for a in actions
            ]
            scores = policy.score_many(feats)
            probs = softmax(scores, temperature)
            idx = sample_index(probs, rng)
        except Exception:  # noqa: BLE001 -- サンプリングが失敗してもターンを止めない
            return None
        buffers[me].append(
            {
                "features": [policy.filter_known(f) for f in feats],
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
        for _ in range(job["games"]):
            match_context.reset()
            buffers[0].clear()
            buffers[1].clear()

            obs_dict, start = battle_start(deck, deck)
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

            for seat in (0, 1):
                decisions = buffers[seat]
                if not decisions:
                    continue
                final_state = observable_state(final_current, seat)
                episodes.append(
                    {
                        "result": 1 if result == seat else -1,
                        "decisions": decisions,
                        "terminal": {
                            "own_n_prize": final_state["own"]["n_prize"],
                            "opp_n_prize": final_state["opponent"]["n_prize"],
                        },
                    }
                )
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
) -> list[dict]:
    """自己対戦を並列実行し、決着がついた試合ぶんのエピソードを集めて返す。

    Args:
        weights_path: 現在の方策の重みJSON(``LinearPolicy`` 契約)。
        games: 総試合数(ワーカー間で均等に割る)。
        temperature: サンプリング温度。
        workers: 並列プロセス数。省略時は論理コア数-2。
        seed: 乱数種の基準値(ワーカーごとにずらす)。
        deck: 自己対戦に使う60枚デッキ。省略時は ``sample_submission/deck.csv``。
    """
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    # ワーカーは sample_submission へ chdir するので、相対パスのまま渡すと解決できない。
    # 呼び出し元のカレントディレクトリを基準に絶対パスへ解決してから渡す。
    weights_path = str(Path(weights_path).resolve())
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

    jobs = []
    base, extra = divmod(games, workers)
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
            }
        )

    episodes: list[dict] = []
    games_played = 0
    with ProcessPoolExecutor(max_workers=workers) as pool_exec:
        for r in pool_exec.map(_run_selfplay_chunk, jobs):
            episodes.extend(r["episodes"])
            games_played += r["games_played"]
    return episodes
