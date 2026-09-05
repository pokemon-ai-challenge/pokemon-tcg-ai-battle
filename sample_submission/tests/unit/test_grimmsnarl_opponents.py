"""マリィのオーロンゲex ルールベース対戦相手(opponents/grimmsnarl_*)のテスト。

対象: ``opponents/grimmsnarl_core.py`` / ``opponents/grimmsnarl_rule_01.py``〜``_05.py`` /
``opponents/grimmsnarl_ex_deck_01.csv``〜``_05.csv``。

実装契約: ``sample_submission/docs/plans/opponent-training/
grimmsnarl-rule-implementation-design.md``(特に §E build_profile 期待値表、
§F 合法手フィルタの不変条件)。

本ファイルは ``opponents/`` の対戦相手(sparring partner)に対するテストであり、
``sample_submission/ptcg_ai``(production)には一切触れない。「強くなったか」(勝率)は
評価しない――ここで確認するのは「壊れていないか／契約を満たすか」のみ:

    1. build_profile が5デッキで契約§Eの期待値表と一致するか(決定的・最重要)。
    2. 各デッキCSVがデッキ構築ルール(60枚・同名4枚以内・ACE SPECは1枚・Basicあり)を
       満たすか。
    3. 各デッキのラッパ agent が rule_based 相手の実対戦で契約§Fの合法手フィルタ
       (min/maxCount・範囲・重複なし)を一度も破らずに完走するか(回帰の要)。
    4. build_profile が未知/空デッキでも例外を投げず動くか(エッジケース)。

乱数(cg は完全CRNではない)には依存しない不変条件のみを assert する。勝敗そのもの
(勝率)は一切 assert しない――勝敗の偏りを見るのは evaluation-scientist の仕事。

想定実行コマンド(本体側):
    cd sample_submission && python -m pytest tests/unit/test_grimmsnarl_opponents.py -v
    # または repo root から:
    python -m pytest sample_submission/tests/unit/test_grimmsnarl_opponents.py -v
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pytest

# --- sys.path bootstrap -----------------------------------------------------
# opponents/ はリポジトリルート直下(sample_submission/ の外)にあり、その内部で
# `from cg.api import ...` している(cg は sample_submission/cg/ 配下)。そのため
# このテストは「repo root」と「sample_submission」の両方を sys.path に通す必要がある
# (league/run_match.py 冒頭のブートストラップと同じ考え方)。
# cwd に依らず __file__ 基準で解決するので、
#   `cd sample_submission && python -m pytest tests/unit/test_grimmsnarl_opponents.py`
#   `python -m pytest sample_submission/tests/unit/test_grimmsnarl_opponents.py`(repo root)
# のどちらでも動く。
SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
LEAGUE_ROOT = REPO_ROOT / "league"

for _candidate in (REPO_ROOT, SAMPLE_SUBMISSION_ROOT, LEAGUE_ROOT):
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:
        sys.path.insert(0, _candidate_str)

# cg エンジン(cg.dll/libcg.so)は import 時にロードされる(sample_submission/cg/sim.py)。
# ロードできない「環境」の問題だけをここでモジュール単位スキップにする
# (league/tests/test_run_match.py の fixture skip と同じ理由)。
# opponents.grimmsnarl_* 側の import は意図的にこの try の外に置く: こちらが本当に壊れて
# いる(実装バグ)場合はスキップで握り潰さず、collection error として素直に落としたい。
try:
    from cg.api import CardType, Observation  # noqa: F401 - cg engine 可用性チェック
except Exception as exc:  # noqa: BLE001 - cg engine 未ロード環境ではモジュールごとスキップ
    pytest.skip(f"cg engine(cg.dll/libcg.so) が利用できません: {exc}", allow_module_level=True)

from opponents import grimmsnarl_core
from opponents import (
    grimmsnarl_rule_01,
    grimmsnarl_rule_02,
    grimmsnarl_rule_03,
    grimmsnarl_rule_04,
    grimmsnarl_rule_05,
)


_RULE_MODULES = {
    "01": grimmsnarl_rule_01,
    "02": grimmsnarl_rule_02,
    "03": grimmsnarl_rule_03,
    "04": grimmsnarl_rule_04,
    "05": grimmsnarl_rule_05,
}

# 契約 §E build_profile 期待値表(grimmsnarl-rule-implementation-design.md)。
# build_profile が返す13キーすべてを網羅する(dict完全一致で比較する)。
EXPECTED_PROFILES: dict[str, dict] = {
    "01": {
        "has_snow": True, "has_nokocchi": False, "has_subomi": False, "has_morpeko": False,
        "rare_candy": 3, "boss": 2, "xerosic": 0, "lambda": 4, "hikari": 1,
        "has_balloon": False, "has_hero_mantle": False, "has_handy_circ": False,
        "dark_energy": 10,
    },
    "02": {
        "has_snow": False, "has_nokocchi": True, "has_subomi": False, "has_morpeko": True,
        "rare_candy": 3, "boss": 0, "xerosic": 2, "lambda": 0, "hikari": 4,
        "has_balloon": False, "has_hero_mantle": True, "has_handy_circ": False,
        "dark_energy": 10,
    },
    "03": {
        "has_snow": True, "has_nokocchi": False, "has_subomi": False, "has_morpeko": False,
        "rare_candy": 3, "boss": 2, "xerosic": 0, "lambda": 4, "hikari": 1,
        "has_balloon": False, "has_hero_mantle": False, "has_handy_circ": True,
        "dark_energy": 10,
    },
    "04": {
        "has_snow": True, "has_nokocchi": False, "has_subomi": False, "has_morpeko": False,
        "rare_candy": 3, "boss": 2, "xerosic": 1, "lambda": 4, "hikari": 1,
        "has_balloon": False, "has_hero_mantle": False, "has_handy_circ": False,
        "dark_energy": 10,
    },
    "05": {
        "has_snow": True, "has_nokocchi": False, "has_subomi": True, "has_morpeko": True,
        "rare_candy": 4, "boss": 2, "xerosic": 0, "lambda": 4, "hikari": 0,
        "has_balloon": True, "has_hero_mantle": False, "has_handy_circ": False,
        "dark_energy": 9,
    },
}


# ---------------------------------------------------------------------------
# 1. build_profile 単体(契約§E, 決定的・最重要)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deck_id", sorted(EXPECTED_PROFILES))
def test_build_profile_matches_contract_table(deck_id):
    """build_profile(deck) の13キー全てが契約§Eの期待値表と一致する。"""
    module = _RULE_MODULES[deck_id]
    profile = grimmsnarl_core.build_profile(module.DECK)
    assert profile == EXPECTED_PROFILES[deck_id]


@pytest.mark.parametrize("deck_id", sorted(_RULE_MODULES))
def test_wrapper_profile_is_actually_derived_from_build_profile(deck_id):
    """ラッパ(grimmsnarl_rule_XX)の module 定数 PROFILE が、
    build_profile(DECK) の呼び出し結果そのものであることを保証する
    (将来リファクタで PROFILE がハードコード化される regression を防ぐ)。
    """
    module = _RULE_MODULES[deck_id]
    assert module.PROFILE == grimmsnarl_core.build_profile(module.DECK)


# ---------------------------------------------------------------------------
# 2. デッキ妥当性(デッキ構築ルール準拠)
# ---------------------------------------------------------------------------
def test_ace_spec_cards_are_flagged_correctly():
    """アンフェアスタンプ(1080)/ヒーローマント(1159)が ACE SPEC として認識されること
    (=デッキ構築ルール上「1枚まで」の前提が card_table から読めていること)。
    """
    assert grimmsnarl_core.card_table[1080].aceSpec is True
    assert grimmsnarl_core.card_table[1159].aceSpec is True


@pytest.mark.parametrize("deck_id", sorted(_RULE_MODULES))
def test_deck_csv_is_a_legal_60_card_deck(deck_id):
    """各デッキCSVが「60枚・カードID整数・同名4枚(ACE SPECは1枚)以内・Basic1枚以上」を満たす。"""
    deck = _RULE_MODULES[deck_id].DECK

    assert len(deck) == 60, f"deck_{deck_id}: expected 60 cards, got {len(deck)}"
    assert all(isinstance(cid, int) for cid in deck), f"deck_{deck_id}: contains a non-int card id"

    counts = Counter(deck)
    for cid, cnt in counts.items():
        data = grimmsnarl_core.card_table.get(cid)
        assert data is not None, f"deck_{deck_id}: unknown card id {cid} (not in all_card_data())"
        if data.cardType == CardType.BASIC_ENERGY:
            continue  # 基本エネルギーは枚数制限なし
        limit = 1 if data.aceSpec else 4
        assert cnt <= limit, (
            f"deck_{deck_id}: card {cid} ({data.name}) appears {cnt} times "
            f"(limit={limit}, aceSpec={data.aceSpec})"
        )

    assert any(grimmsnarl_core.card_table[cid].basic for cid in deck), (
        f"deck_{deck_id}: no Basic Pokemon in deck"
    )


# ---------------------------------------------------------------------------
# 3. 合法手不変条件(契約§F、回帰の要)
# ---------------------------------------------------------------------------
def _make_legality_checked_agent(inner_agent, label: str):
    """inner_agent(obs) -> list[int] を包み、契約§Fの合法手フィルタを毎 decision で assert する。

    違反時は AssertionError を送出する。``play_match`` はエージェントの例外を捕捉して
    ``MatchResult.error`` に詰める設計(``run_match.py`` の try/except)なので、この
    AssertionError は試合をクラッシュさせず、最終的に ``assert result.error is None``
    の失敗メッセージとして表面化する(何の入力で何が違反したかは decisions 経由でも追跡できる)。

    契約(sample_submission/docs/plans/opponent-training/
    grimmsnarl-rule-implementation-design.md §F):
        (a) minCount <= len(action) <= maxCount
        (b) 各 index は 0 <= i < len(option)
        (c) 重複禁止
        (d) 初回(select=None)は60枚
    (d)は play_match の対戦ループでは発火しない(battle_start に直接デッキを渡す設計のため。
    obs.select is None は Kaggle 基盤が直接 agent を呼ぶときのみ発生する)。そのため (d) は
    別途 test_agent_returns_full_deck_on_initial_selection で単独に検証する。
    """
    decisions: list[dict] = []

    def checked(obs):
        action = inner_agent(obs)

        if obs.select is None:
            assert isinstance(action, list), (
                f"{label}: deck selection must return a list, got {type(action)!r}"
            )
            assert len(action) == 60, f"{label}: deck selection length != 60 ({len(action)})"
            assert all(isinstance(i, int) for i in action), f"{label}: deck selection has non-int element"
            decisions.append({"kind": "deck_select", "len": len(action)})
            return action

        select = obs.select
        n_options = len(select.option)

        assert isinstance(action, list), (
            f"{label}: action must be a list, got {type(action)!r} (context={select.context})"
        )
        assert all(isinstance(i, int) for i in action), (
            f"{label}: action has a non-int element (context={select.context}): {action}"
        )
        assert select.minCount <= len(action) <= select.maxCount, (
            f"{label}: len(action)={len(action)} not in "
            f"[{select.minCount}, {select.maxCount}] (context={select.context}, n_options={n_options})"
        )
        assert all(0 <= i < n_options for i in action), (
            f"{label}: action index out of range "
            f"(context={select.context}, n_options={n_options}): {action}"
        )
        assert len(set(action)) == len(action), (
            f"{label}: duplicate indices in action (context={select.context}): {action}"
        )

        decisions.append({"kind": "select", "context": int(select.context), "len": len(action)})
        return action

    return checked, decisions


@pytest.fixture(scope="module")
def rule_based_harness():
    """rule_based_agent と run_match.play_match を遅延 import する(cg engine / ptcg_ai
    依存が満たせない環境ではこのテストだけをスキップする。league/tests/test_run_match.py の
    ``rule_based_agent`` fixture と同じ方針)。
    """
    try:
        from ptcg_ai.rule_based.rule_based_agent import agent as rule_based_agent_fn, read_deck_csv
        from run_match import play_match
    except Exception as exc:  # noqa: BLE001 - 依存未整備環境ではスキップ
        pytest.skip(f"rule_based_agent / run_match が利用できません: {exc}")
    return rule_based_agent_fn, read_deck_csv, play_match


_GAMES_PER_DECK = 2  # 軽さ優先(各デッキ2試合、先後を入れ替える)。


@pytest.mark.parametrize("deck_id", sorted(_RULE_MODULES))
def test_grimmsnarl_agent_only_produces_legal_moves_vs_rule_based(deck_id, rule_based_harness, monkeypatch):
    """rule_based 相手の実対戦全体を通じて、全 decision が契約§Fの合法手フィルタを
    満たし(不正手0件)、試合が正常完走(error なし・winner が 0/1)することを確認する。

    ``rule_based_agent.read_deck_csv()`` は cwd 相対で "deck.csv" を読む契約なので、
    対戦中だけ cwd を sample_submission/ にする(run_league.py の
    ``os.chdir(_SAMPLE_SUBMISSION_DIR)`` と同じ理由。monkeypatch.chdir はテスト終了時に
    自動で元の cwd へ戻すので他テストへの副作用はない)。
    """
    rule_based_agent_fn, read_deck_csv, play_match = rule_based_harness

    monkeypatch.chdir(SAMPLE_SUBMISSION_ROOT)

    module = _RULE_MODULES[deck_id]
    gs_deck = module.DECK
    rb_deck = read_deck_csv()

    checked_agent, decisions = _make_legality_checked_agent(module.agent, label=f"grimmsnarl_rule_{deck_id}")

    for i in range(_GAMES_PER_DECK):
        seed = 5000 + int(deck_id) * 10 + i
        if i % 2 == 0:
            result = play_match(checked_agent, rule_based_agent_fn, gs_deck, rb_deck, seed=seed)
        else:
            result = play_match(rule_based_agent_fn, checked_agent, rb_deck, gs_deck, seed=seed)

        assert result.error is None, (
            f"deck_{deck_id} game={i}: 対戦がエラー終了(不正手検知の可能性を含む): {result.error}"
        )
        assert result.winner in (0, 1), f"deck_{deck_id} game={i}: winner={result.winner!r}(異常終了)"
        assert result.turns is not None and result.turns > 0

    assert len(decisions) > 0, f"deck_{deck_id}: grimmsnarl agent が一度も呼ばれなかった(テスト自体が無意味になる)"


@pytest.mark.parametrize("deck_id", sorted(_RULE_MODULES))
def test_agent_returns_full_deck_on_initial_selection(deck_id):
    """契約(d): obs.select is None のとき、agent は自分の60枚デッキをそのまま返す。

    play_match の対戦ループでは発火しない経路(デッキは battle_start に直接渡される)なので、
    ここで単独に(cg 対戦を回さず)直接 agent を呼んで検証する。
    """
    module = _RULE_MODULES[deck_id]
    obs = Observation(select=None, logs=[], current=None)
    result = module.agent(obs)
    assert result == module.DECK


# ---------------------------------------------------------------------------
# 4. build_profile エッジケース
# ---------------------------------------------------------------------------
def test_build_profile_handles_empty_deck_without_raising():
    """60枚に満たない/空のデッキでも例外を投げず、フラグ類は素直に False/0 になる。"""
    profile = grimmsnarl_core.build_profile([])
    assert profile["has_snow"] is False
    assert profile["has_morpeko"] is False
    assert profile["dark_energy"] == 0
    assert profile["boss"] == 0
    assert profile["rare_candy"] == 0


def test_build_profile_handles_unknown_card_ids_without_raising():
    """all_card_data() に存在しないカードIDが混ざっても例外を投げず無視される(defaultdict契約)。"""
    profile = grimmsnarl_core.build_profile([999999, 999999, 7, 7])
    assert profile["dark_energy"] == 2
    assert profile["has_snow"] is False
