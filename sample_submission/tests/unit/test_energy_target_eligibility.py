"""Issue 1 回帰テスト: エネルギー付与先はデッキ方針上のアタッカーに限定する。

エネルギー付与先は「デッキ方針上のアタッカー（ENERGY_TARGET_PRIORITY）」に限定し、
ドローエンジン前段のノコッチ系列（card_id 65/66）には基本的に付けない。

ただし例外として、ノコッチ系列は「リッチエネルギーの周回コンボ」の受け皿としてなら
エネルギー付与の対象になってよい（進化前のノコッチ(65)に付けても、進化でノココッチ(66)へ
引き継がれるためどちらの段階でもよい）。この例外は
(a) 手札にリッチエネルギー(energy_recycle_card_id)がある、かつ
(b) 自分のバトルポケモンが攻撃に必要なエネルギーを既に満たしている、または
    代替供給手段（ワンダーパッチ等）が使える
の両方が揃った場合のみ成立する（_energy_recycle_bonus > 0 と同じゲート）。

is_attach_eligible は、対象がフーディン系列/キチキギスex（ENERGY_TARGET_PRIORITY）なら
state を参照せず True/False を返す。ノコッチ系列（65/66）は周回条件の判定に state が
必要なため、state=None のときは安全側で False を返す。
"""

from types import SimpleNamespace

from ptcg_ai.rule_based.main_turn_parts import energy_eval

_ALAKAZAM_ID = 743  # フーディン（ENERGY_REQUIRED_COUNT=1）
_RICH_ENERGY_ID = 13  # リッチエネルギー


def _pokemon(card_id: int):
    return SimpleNamespace(id=card_id, energies=[])


def _energy(card_id: int = 99):
    return SimpleNamespace(id=card_id)


def _card(card_id: int):
    return SimpleNamespace(id=card_id)


def _fake_state(*, active_energy_count: int, hand_ids: tuple[int, ...]):
    """own側=フーディンがバトル場、指定エネルギー数・手札を持つ最小限の State モック。"""
    own_active = SimpleNamespace(id=_ALAKAZAM_ID, energies=[_energy() for _ in range(active_energy_count)])
    own_player = SimpleNamespace(
        active=[own_active],
        bench=[],
        hand=[_card(cid) for cid in hand_ids],
        discard=[],
    )
    opponent_player = SimpleNamespace(active=[None], bench=[], hand=[], discard=[])
    return SimpleNamespace(yourIndex=0, players=[own_player, opponent_player], stadium=[])


def test_dunsparce_is_not_energy_eligible_without_state():
    # state が無い（判定不能）ときは安全側で対象外。
    assert energy_eval.is_attach_eligible(_pokemon(65), state=None) is False


def test_shaymin_is_not_energy_eligible():
    # シェイミ(343)は周回対象でもないので、周回条件が揃っていても対象外。
    state = _fake_state(active_energy_count=1, hand_ids=(_RICH_ENERGY_ID,))
    assert energy_eval.is_attach_eligible(_pokemon(343), state) is False


def test_attackers_are_energy_eligible():
    # フーディン系列(741/742/743)とキチキギスex(140)は付与対象。
    for card_id in (741, 742, 743, 140):
        assert energy_eval.is_attach_eligible(_pokemon(card_id), state=None) is True


def test_dunsparce_line_is_energy_eligible_when_recycle_conditions_met():
    # 主力フーディンが必要エネルギーを既に満たし、手札にリッチエネがある＝周回OK。
    # ノコッチ(65)・ノココッチ(66)のどちらでも受け皿になれる。
    state = _fake_state(active_energy_count=1, hand_ids=(_RICH_ENERGY_ID,))
    assert energy_eval.is_attach_eligible(_pokemon(65), state) is True
    assert energy_eval.is_attach_eligible(_pokemon(66), state) is True


def test_dunsparce_line_is_not_energy_eligible_without_rich_energy_in_hand():
    # 主力の準備は整っていても、手札にリッチエネが無ければ周回コンボは成立しない。
    state = _fake_state(active_energy_count=1, hand_ids=())
    assert energy_eval.is_attach_eligible(_pokemon(65), state) is False


def test_dunsparce_line_is_not_energy_eligible_when_attacker_not_ready_and_no_backup():
    # 手札にリッチエネはあっても、主力がまだ準備不足かつ代替供給手段も無ければ、
    # 主力のエネルギー確保を妨げないよう周回コンボは見送る。
    state = _fake_state(active_energy_count=0, hand_ids=(_RICH_ENERGY_ID,))
    assert energy_eval.is_attach_eligible(_pokemon(65), state) is False
