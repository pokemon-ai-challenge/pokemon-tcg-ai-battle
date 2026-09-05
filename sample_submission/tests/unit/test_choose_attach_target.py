"""energy_tool_turn._choose_attach_target の回帰テスト（PRレビュー指摘対応）。

_choose_attach_target（ATTACH_FROM: Wonder Patch 等で提示された選択肢の中からエネルギー/
どうぐの付与先ポケモンを選ぶ）は、energy_eval.best_energy_target と同じ
「ENERGY_REQUIRED_COUNT に既に達しているポケモンは候補から外す」フィルタ
（energy_eval.is_energy_already_sufficient）を通していなかった。

そのため、選択肢に「既に必要エネルギーを満たしたポケモン」と「まだ不足しているポケモン」の
両方が含まれる場面で、active_value（エネルギー本数が多いほど加点）が高いという理由だけで
前者が選ばれ、周回コンボ対象でもないのに無駄にエネルギーが積まれる可能性があった。

このテストは、is_energy_already_sufficient フィルタが _choose_attach_target にも
適用されていることを確認する。
"""

from types import SimpleNamespace

from cg.api import AreaType

from ptcg_ai.action_selection.handlers import energy_tool_turn

_ALAKAZAM_ID = 743  # ENERGY_REQUIRED_COUNT=1
_KADABRA_ID = 742  # ENERGY_REQUIRED_COUNT=1


def _pokemon(card_id: int, energies=None, hp=100, max_hp=100):
    return SimpleNamespace(id=card_id, energies=list(energies or []), hp=hp, maxHp=max_hp)


def _option(area, index, player_index=0):
    return SimpleNamespace(
        area=area, index=index, playerIndex=player_index, toolIndex=None, energyIndex=None
    )


def _make_obs(bench):
    options = [_option(AreaType.BENCH, i) for i in range(len(bench))]
    select = SimpleNamespace(option=options)
    own_player = SimpleNamespace(active=[None], bench=bench, hand=[])
    opponent_player = SimpleNamespace(active=[None], bench=[], hand=[])
    state = SimpleNamespace(yourIndex=0, players=[own_player, opponent_player])
    return SimpleNamespace(select=select, current=state)


def test_prefers_not_yet_sufficient_candidate_over_already_charged_one():
    # index0: 既にENERGY_REQUIRED_COUNT(1)を満たしたアラカザム。
    # index1: まだ0本のカダブラ（不足）。
    already_charged = _pokemon(_ALAKAZAM_ID, energies=["e"], hp=140, max_hp=140)
    still_needs_energy = _pokemon(_KADABRA_ID, energies=[], hp=80, max_hp=80)

    obs = _make_obs(bench=[already_charged, still_needs_energy])
    result = energy_tool_turn._choose_attach_target(obs)

    # 既に足りているアラカザム(index0)ではなく、不足しているカダブラ(index1)が選ばれるべき。
    assert result == [1]


def test_falls_back_to_scoring_when_all_candidates_already_sufficient():
    # 提示された選択肢が全員「既に充足済み」でも、必ず合法手を返す（フォールバックの確認）。
    a = _pokemon(_ALAKAZAM_ID, energies=["e"], hp=140, max_hp=140)
    b = _pokemon(_KADABRA_ID, energies=["e"], hp=80, max_hp=80)

    obs = _make_obs(bench=[a, b])
    result = energy_tool_turn._choose_attach_target(obs)

    assert result in ([0], [1])
