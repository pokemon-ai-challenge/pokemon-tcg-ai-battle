"""自己対戦の相手多様化(``rl.selfplay.collect_selfplay`` の ``opponent_pool``)の検証。

## なぜこのテストが正しさの核心なのか

方策勾配 ``∇_w log pi(a)`` は「自分の方策 pi が選んだ行動 a」に対する勾配である。
相手プールモードで相手側の decision まで記録し、それに対して自分の方策 pi の
確率で勾配を計算すると、方策勾配ではない別の量(他人が選んだ行動を自分の方策の
パラメータで説明しようとする、オフポリシーの補正を欠いた量)を最適化してしまう。
これは符号を間違えるのと同種の「それらしく動くが学習が壊れている」バグであり、
発見が非常に難しい(``rl/reinforce.py`` の docstring 参照)。

``test_opponent_mode_records_only_own_seat_decisions`` と
``test_pure_selfplay_records_both_seat_decisions`` はこれを実対局で検証する。

## 実対局は厳密な回数を保証しない(engine内部の乱数)

``cg`` の対局は Python 側の ``random.Random(seed)`` だけでなく、エンジン内部の
シャッフル・コイントスなど Python から制御できない乱数も使っている。そのため
同じ ``seed`` を渡しても「1試合が必ず両席ぶんの decision を生む」「STEP_CAP 内に
必ず決着する」until保証はない。したがって実対局を使うテストは、厳密な件数の
一致ではなく「相手プールモードは1試合につき最大1エピソード」「純粋自己対戦は
それより明らかに多い」という構造的な不等式で検証する(flaky にしない)。

一方、相手の巡回選択・先手/後手の交互化そのものは実対局に依存しない純粋な
算術(``rl.selfplay._game_opponent_and_seat`` / ``_build_jobs``)なので、そちらは
実対局を回さずに決定的に検証する。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
_POLICY_PRIOR_DIR = _REPO_ROOT / "kaggle_replays" / "policy_prior"
for p in (_SAMPLE_SUBMISSION_DIR, _POLICY_PRIOR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl.selfplay import (  # noqa: E402
    _build_jobs,
    _game_opponent_and_seat,
    collect_selfplay,
    resolve_opponent_pool,
)

_OWN_WEIGHTS = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "policy_weights.json"


def _read_own_deck() -> list[int]:
    cwd = Path.cwd()
    import os

    os.chdir(_SAMPLE_SUBMISSION_DIR)
    try:
        from main import read_deck_csv

        return read_deck_csv()
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def own_deck() -> list[int]:
    return _read_own_deck()


def _opponent_pool(deck: list[int], n: int) -> list[dict]:
    """テスト用の相手プール。デッキ・重みは実在するミラーデッキ/汎用モデルを
    使い回す(相手のデッキ・モデルの中身自体はこのテストの主眼ではないため)。
    """
    return [
        {"name": f"dummy{i}", "deck": deck, "weights_path": str(_OWN_WEIGHTS)}
        for i in range(n)
    ]


# --- 1. 【最重要・実対局】相手プールモードでは自分側の decision だけが記録される ---


def test_opponent_mode_records_only_own_seat_decisions(own_deck):
    games = 6
    episodes = collect_selfplay(
        weights_path=_OWN_WEIGHTS,
        games=games,
        temperature=1.0,
        workers=1,
        seed=20260728,
        deck=own_deck,
        opponent_pool=_opponent_pool(own_deck, n=2),
    )
    # 構造的な保証: 相手側は記録されないので、1試合につき最大1エピソードしか
    # 生まれない(engineの内部乱数で試合内容が変わっても、これは崩れない)。
    assert len(episodes) <= games, (
        f"相手プールモードは1試合→最大1エピソードのはずが {len(episodes)} 件"
        f"(games={games})。相手側の decision が混入している疑い"
    )
    assert len(episodes) >= 1, "決着がついた試合が1つも無く、検証できていない"
    for ep in episodes:
        assert ep["decisions"], "エピソードに decision が1件も無い"
        assert "opponent" in ep, "相手プールモードなのに opponent 名が付いていない"


def test_pure_selfplay_records_both_seat_decisions(own_deck):
    """対比: opponent_pool を指定しない従来の自己対戦では、両席とも自分の方策
    (同じ重み)なので両方記録してよい(1試合→最大2エピソード)。

    上のテストの構造的上限(相手プールモードは episodes <= games)を踏まえ、
    ここでは episodes > games を示すことで「相手プールモードでは起こり得ない
    ほど多くのエピソードが両席から出ている」= 両席記録の証拠とする
    (厳密に 2*games である必要はない。engine内部の乱数で片方の席が0
    decision の試合が混ざっても不等式は崩れない想定)。
    """
    games = 6
    episodes = collect_selfplay(
        weights_path=_OWN_WEIGHTS,
        games=games,
        temperature=1.0,
        workers=1,
        seed=20260728,
        deck=own_deck,
        # opponent_pool は省略(None) = 純粋な自己対戦
    )
    for ep in episodes:
        assert "opponent" not in ep, "純粋な自己対戦なのに opponent 名が付いている"
    assert len(episodes) <= 2 * games
    assert len(episodes) > games, (
        f"純粋な自己対戦なら両席とも記録され、相手プールモードの上限(<=games)を"
        f"超えるはずが {len(episodes)} 件(games={games})しか無かった。"
        "両席とも記録されているか確認すること"
    )


# --- 2. 相手プールの巡回選択・先手後手の交互化(純粋関数を直接検証、実対局なし) ---


def test_game_opponent_and_seat_round_robin():
    pool_size = 3
    assignments = [_game_opponent_and_seat(pool_size, 0, 0, i) for i in range(9)]
    opp_indices = [a[0] for a in assignments]
    seats = [a[1] for a in assignments]
    assert opp_indices == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert seats == [0, 1, 0, 1, 0, 1, 0, 1, 0]


def test_game_opponent_and_seat_continues_from_offset():
    """offset(累積試合数)を渡すと、そこから巡回・交互化が続く
    (ワーカーをまたいで全体で連続させるための性質)。
    """
    pool_size = 2
    # offset=0 から5試合分進めた場合と、offset=5 から4試合分進めた場合を
    # 連結すると、offset=0 から9試合分進めた場合と一致するはず。
    combined = [_game_opponent_and_seat(pool_size, 0, 0, i) for i in range(9)]
    part1 = [_game_opponent_and_seat(pool_size, 0, 0, i) for i in range(5)]
    part2 = [_game_opponent_and_seat(pool_size, 5 % pool_size, 5 % 2, i) for i in range(4)]
    assert part1 + part2 == combined


def test_build_jobs_balances_opponent_usage_across_workers():
    """複数ワーカーに分割しても、全ジョブを合わせると相手の巡回・先手後手の
    交互化が「1プロセスで順番に処理した場合」と同じになる(不均等にならない)。
    """
    pool = _opponent_pool([1, 2, 3], n=3)
    games, workers = 17, 4  # 17は3の倍数でも4の倍数でもない(端数ケース)
    jobs = _build_jobs("w.json", games, 1.0, workers, seed=0, deck=[1, 2, 3], opponent_pool=pool)
    assert sum(j["games"] for j in jobs) == games

    # 各ジョブの中身(_run_selfplay_chunk が行うのと同じ計算)を再現し、
    # 全ジョブを通しての相手・着席の割り当てが「1プロセスで順に処理した場合」と
    # 一致することを確認する。
    all_assignments = []
    for job in jobs:
        for i in range(job["games"]):
            all_assignments.append(
                _game_opponent_and_seat(
                    len(pool), job["opponent_start_index"], job["seat_start"], i
                )
            )
    expected = [_game_opponent_and_seat(len(pool), 0, 0, i) for i in range(games)]
    assert all_assignments == expected

    opp_counts = Counter(idx for idx, _ in all_assignments)
    assert max(opp_counts.values()) - min(opp_counts.values()) <= 1, dict(opp_counts)

    seat_counts = Counter(seat for _, seat in all_assignments)
    assert max(seat_counts.values()) - min(seat_counts.values()) <= 1, dict(seat_counts)


def test_build_jobs_pure_selfplay_has_no_opponent_pool():
    jobs = _build_jobs("w.json", 10, 1.0, 3, seed=0, deck=[1, 2, 3], opponent_pool=None)
    for job in jobs:
        assert job["opponent_pool"] is None
        assert job["opponent_start_index"] == 0


# --- 3. resolve_opponent_pool: 専用モデルが無い場合のフォールバック -----------


def test_resolve_opponent_pool_none_means_pure_selfplay():
    assert resolve_opponent_pool(None) is None


def test_resolve_opponent_pool_falls_back_with_warning_when_weights_missing(
    monkeypatch, tmp_path, capsys
):
    """専用モデルが見つからないときは例外にせず、警告して None(=純粋な自己対戦)
    にフォールバックする(重みは .gitignore 済みで別環境には無いことがあるため)。
    """
    import rl.selfplay as selfplay_module

    decks_path = tmp_path / "opponent_decks.json"
    decks_path.write_text(
        '{"archetypeX": {"deck": [1, 2, 3]}}', encoding="utf-8"
    )
    empty_weights_dir = tmp_path / "no_weights_here"
    empty_weights_dir.mkdir()

    monkeypatch.setattr(selfplay_module, "_OPPONENT_DECKS_PATH", decks_path)
    monkeypatch.setattr(selfplay_module, "_ARCHETYPE_WEIGHTS_DIR", empty_weights_dir)

    result = selfplay_module.resolve_opponent_pool("archetypeX")
    assert result is None
    captured = capsys.readouterr()
    assert "警告" in captured.err
    assert "archetypeX" in captured.err


def test_resolve_opponent_pool_builds_pool_when_weights_present(monkeypatch, tmp_path):
    import json

    import rl.selfplay as selfplay_module

    decks_path = tmp_path / "opponent_decks.json"
    decks_path.write_text(
        json.dumps({"archetypeX": {"deck": [1, 2, 3]}}), encoding="utf-8"
    )
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weights_file = weights_dir / "policy_weights_archetypeX.json"
    weights_file.write_text(_OWN_WEIGHTS.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(selfplay_module, "_OPPONENT_DECKS_PATH", decks_path)
    monkeypatch.setattr(selfplay_module, "_ARCHETYPE_WEIGHTS_DIR", weights_dir)

    result = selfplay_module.resolve_opponent_pool("archetypeX")
    assert result == [
        {"name": "archetypeX", "deck": [1, 2, 3], "weights_path": str(weights_file)}
    ]


def test_resolve_opponent_pool_unknown_name_raises(monkeypatch, tmp_path):
    import rl.selfplay as selfplay_module

    decks_path = tmp_path / "opponent_decks.json"
    decks_path.write_text('{"archetypeX": {"deck": [1, 2, 3]}}', encoding="utf-8")
    monkeypatch.setattr(selfplay_module, "_OPPONENT_DECKS_PATH", decks_path)
    monkeypatch.setattr(selfplay_module, "_ARCHETYPE_WEIGHTS_DIR", tmp_path)

    with pytest.raises(ValueError):
        selfplay_module.resolve_opponent_pool("does_not_exist")
