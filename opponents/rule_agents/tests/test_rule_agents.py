"""ルールベース3種が「エンジンのルールを破らない」ことを確かめる回帰テスト。

中身の強さではなく、契約を守っているかだけを見る:

- 返す選択肢は `minCount`〜`maxCount` の個数で、重複が無く、範囲内であること
- 3種のどの組み合わせでも、例外を出さずに1試合を最後まで進められること

対戦の中身は乱数に依存するので勝敗は検証しない（勝率は
`opponents/rule_agents/diagnose.py` で測る）。
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
_ROOT = _PKG.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

AGENTS = {
    "grimmsnarl": ("opponents.rule_agents.grimmsnarl", "marnie_grimmsnarl_ex.csv"),
    "lucario": ("opponents.rule_agents.lucario", "mega_lucario_ex.csv"),
    "archaludon": ("opponents.rule_agents.archaludon", "archaludon_ex.csv"),
}

MAX_STEPS = 3000


def _deck(name: str) -> list[int]:
    text = (_PKG / "decks" / AGENTS[name][1]).read_text(encoding="utf-8")
    return [int(v) for v in text.replace(",", "\n").split() if v.strip()]


def _agent(name: str):
    return importlib.import_module(AGENTS[name][0]).agent


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_deck_is_60_cards(name: str) -> None:
    assert len(_deck(name)) == 60


@pytest.mark.parametrize(
    "a,b",
    [("grimmsnarl", "lucario"), ("grimmsnarl", "archaludon"), ("lucario", "archaludon")],
)
def test_match_finishes_without_illegal_selection(a: str, b: str) -> None:
    """1試合を通して回し、毎手の選択がエンジンの制約を満たすことを確かめる。"""
    random.seed(12345)
    agents = [_agent(a), _agent(b)]
    obs_dict, start = battle_start(_deck(a), _deck(b))
    assert start.errorType == 0

    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            assert obs.current is not None
            if obs.current.result != -1:
                break
            assert steps < MAX_STEPS, "試合が終わらない（判断がループしている可能性）"

            select = obs.select
            action = agents[obs.current.yourIndex](obs)

            assert isinstance(action, list)
            assert select.minCount <= len(action) <= select.maxCount, (
                f"選択個数が範囲外: {len(action)} not in "
                f"[{select.minCount}, {select.maxCount}] (context={select.context})"
            )
            assert len(set(action)) == len(action), f"選択肢が重複している: {action}"
            assert all(0 <= i < len(select.option) for i in action), (
                f"選択肢の範囲外: {action} (option数={len(select.option)})"
            )

            obs_dict = battle_select(action)
            steps += 1
    finally:
        battle_finish()
