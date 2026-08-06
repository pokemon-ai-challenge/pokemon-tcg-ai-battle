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

from opponents.rule_agents import framework as fw  # noqa: E402

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


def _pokemon(card_id: int) -> "Pokemon":
    from cg.api import Pokemon

    return Pokemon(
        id=card_id, serial=1, hp=100, maxHp=100, appearThisTurn=False,
        energies=[], energyCards=[], tools=[], preEvolution=[],
    )


# 実戦の敗北(2026-08-05 ep90196278, vs クラスタゲ)で見つかった不具合の回帰テスト。
# クラスタゲ／ニンフィアの特性は「Pokémon ex からのワザのダメージ」を防ぐ。
# こちらの主軸(オーロンゲex/ブリジュラスex/メガルカリオex)は全員 Pokémon ex なので、
# これらが相手の場にいる間はワザで一切ダメージを与えられない。
CRUSTLE = 345
SYLVEON = 330
DUSKULL = 355  # ダミー: 壁を持たない、ただの相手ポケモン


@pytest.mark.parametrize("wall_id", [CRUSTLE, SYLVEON])
@pytest.mark.parametrize("attacker_id", [648, 190, 678])  # オーロンゲex/ブリジュラスex/メガルカリオex
def test_ex_pokemon_wall_blocks_our_main_attackers(wall_id: int, attacker_id: int) -> None:
    attacker = _pokemon(attacker_id)
    wall = _pokemon(wall_id)
    assert fw.is_damage_immune(attacker, wall, False) is True


def test_ex_pokemon_wall_does_not_block_non_ex_attacker() -> None:
    # オーロンゲデッキのマンキーは ex ではない。壁は ex 相手だけを防ぐ。
    attacker = _pokemon(112)  # Munkidori
    wall = _pokemon(CRUSTLE)
    assert fw.is_damage_immune(attacker, wall, False) is False


def test_non_wall_pokemon_never_blocks() -> None:
    attacker = _pokemon(648)
    non_wall = _pokemon(DUSKULL)
    assert fw.is_damage_immune(attacker, non_wall, False) is False


# 実戦の敗北(2026-08-06 ep90534503, vs Alakazam)で見つかった不具合の回帰テスト。
# ニュートラルゾーン(スタジアム)は「ルールを持たないポケモンは、相手の Pokémon ex
# からのワザのダメージを受けない」を場に出ている間ずっと適用する。攻撃した側だけでなく
# 双方に効く。これを見落とし、シャドーバレットを9回連続でダメージ0のまま撃ち続けていた。
NEUTRALIZATION_ZONE = 1247


@pytest.mark.parametrize("attacker_id", [648, 190, 678])  # オーロンゲex/ブリジュラスex/メガルカリオex
def test_neutralization_zone_blocks_our_main_attackers_vs_rule_box_free_defender(
    attacker_id: int,
) -> None:
    attacker = _pokemon(attacker_id)
    # ルールを持たない(ex でも megaEx でもない)ただのポケモン。
    defender = _pokemon(DUSKULL)
    assert fw.is_damage_immune(attacker, defender, False, NEUTRALIZATION_ZONE) is True
    # スタジアムが無ければ通る。
    assert fw.is_damage_immune(attacker, defender, False, 0) is False


def test_neutralization_zone_does_not_block_when_defender_has_rule_box() -> None:
    # 相手も ex なら「ルールを持たない」の対象外で、壁は効かない。
    attacker = _pokemon(648)
    defender = _pokemon(190)  # Archaludon ex
    assert fw.is_damage_immune(attacker, defender, False, NEUTRALIZATION_ZONE) is False
