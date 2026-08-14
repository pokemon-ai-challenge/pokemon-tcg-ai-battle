"""能力評価コーパスを構築する（Step 1-18）。

自己対戦から局面を採取し、**Phase 1/2 を使わない独立オラクル**（`tools/lethal_oracle.py`）
でラベル付けして `tests/fixtures/lethal_capability.jsonl` へ書き出す。

採取方針:
  * サイドが減った終盤ほどリーサルが存在しやすいので、終盤を厚めに採る。
  * ただし positive だけでは意味が無い。`expected_lethal = False` の局面も残し、
    そのうち「相手のバトルポケモンは倒せるが勝ちきれない」ものを near miss として区別する。
  * オラクルの手順は必ず**再生検証**してから保存する（偽の positive を作らない）。

使い方:
    python tools/build_capability_corpus.py [games] [out]
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import read_deck_csv
from ptcg_ai.action_selection import router, selector
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search.lethal.engine import HiddenState
from tools import lethal_oracle

OUT_DEFAULT = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"

# 各カテゴリの採取上限。1 カテゴリに偏らせない。
QUOTA = {
    "deterministic_lethal": 40,
    "multi_step_lethal": 40,
    "chance_lethal": 40,
    "opponent_choice": 25,
    "near_miss": 30,
    "negative": 30,
}


def categorize(oracle: lethal_oracle.OracleResult) -> str | None:
    if oracle.expected_lethal:
        if not oracle.chance_free:
            return "chance_lethal"
        if (oracle.minimum_depth or 0) >= 3:
            return "multi_step_lethal"
        return "deterministic_lethal"
    if oracle.truncated:
        return None  # 打ち切られた局面は negative の根拠にできない
    # 「相手を倒せるが勝ちきれない」が最も情報量が多いので先に振り分ける。
    # なお B4（ターン中の相手選択）は排他カテゴリではなく
    # `has_opponent_choice` フラグとして全行に付くので、そちらで選べる。
    if oracle.can_knock_out_active:
        return "near_miss"
    if oracle.has_opponent_choice:
        return "opponent_choice"
    return "negative"


def opponent_action(obs, deck, mode, rng):
    if mode == "random":
        count = rng.randint(obs.select.minCount, min(obs.select.maxCount,
                                                     len(obs.select.option)))
        return rng.sample(range(len(obs.select.option)), count) if count else []
    return selector.select_action(obs, deck, {"lethal_search": {"enabled": False}})


def collect(games: int, out_path: Path) -> None:
    deck = read_deck_csv()
    rows: list[dict] = []
    taken: Counter = Counter()
    skipped: Counter = Counter()
    started = time.perf_counter()

    for game in range(games):
        mode = "random" if game % 2 == 0 else "rule"
        rng = random.Random(70_000 + game)
        obs_dict, start = battle_start(list(deck), list(deck))
        if start.errorType != 0:
            continue
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    break
                if obs.select is None:
                    obs_dict = battle_select(list(deck))
                    continue
                if obs.current.yourIndex == 0:
                    if _should_probe(obs, rng, taken):
                        _probe(obs, obs_dict, deck, rows, taken, skipped)
                    action = router.route(obs) or [0]
                else:
                    action = opponent_action(obs, deck, mode, rng)
                obs_dict = battle_select(action)
        finally:
            battle_finish()
        if all(taken[key] >= quota for key, quota in QUOTA.items()):
            break
        if game % 10 == 9:
            print(f"  game {game + 1}/{games}  collected={len(rows)} {dict(taken)}",
                  flush=True)

    out_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\n書き出し: {out_path}  {len(rows)} 件  "
          f"({time.perf_counter() - started:.0f}s)")
    print("  category:", dict(taken))
    print("  skipped :", dict(skipped))


def _should_probe(obs, rng, taken) -> bool:
    """全決定を検証すると重すぎるので、終盤を厚くしつつ間引く。"""
    state = obs.current
    if state is None:
        return False
    # ゲーム開始時のセットアップ選択（ターン 0・サイド未配置）は採らない。
    # 実測（Step 1-18 第 1 回）: これを入れないと turn==0 の初期配置が 20 件混入し、
    # opponent_choice 25 件中 20 件を占め、12 件の完全重複を生んでいた。
    # リーサル評価には使えない局面なので除外する。
    if state.turn < 1 or not state.players[0].prize:
        return False
    prizes = len(state.players[0].prize)
    probability = {6: 0.05, 5: 0.10, 4: 0.20, 3: 0.35, 2: 0.60, 1: 0.90}.get(prizes, 0.5)
    return rng.random() < probability


def _probe(obs, obs_dict, deck, rows, taken, skipped) -> None:
    stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
    if stub is None:
        skipped["no_hidden_state"] += 1
        return
    hidden = HiddenState.from_stub(stub)
    try:
        oracle = lethal_oracle.solve(
            obs, hidden, 0, max_depth=6, time_limit_ms=6_000.0, node_limit=20_000
        )
    except Exception:  # noqa: BLE001
        skipped["oracle_error"] += 1
        return

    category = categorize(oracle)
    if category is None:
        skipped["truncated"] += 1
        return
    if taken[category] >= QUOTA[category]:
        skipped[f"quota_{category}"] += 1
        return

    if oracle.expected_lethal:
        # 保存する前に、その手順が本当に勝ちへ到達するか実エンジンで再生する。
        verified = lethal_oracle.replay_verify(
            obs, HiddenState.from_stub(stub), 0, oracle.oracle_sequence
        )
        if not verified:
            skipped["replay_failed"] += 1
            return

    taken[category] += 1
    rows.append({
        "id": f"cap{len(rows):04d}",
        "category": category,
        "prizes_left": len(obs.current.players[0].prize),
        "turn": obs.current.turn,
        "oracle": oracle.as_dict(),
        "hidden_stub": {k: list(v) if isinstance(v, (list, tuple)) else v
                        for k, v in stub.items()},
        "obs": obs_dict,
    })


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DEFAULT
    collect(count, destination)
