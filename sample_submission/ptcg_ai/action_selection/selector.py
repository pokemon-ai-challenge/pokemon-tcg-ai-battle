"""Action selection: try lethal search first, then the rule-based router.

``select_action()`` is the normal-turn entry point called from
``ptcg_ai.rule_based.rule_based_agent``. Search modules never talk to
the agent entry points directly; this module wires the config, builds
the hidden state for the search and validates whatever the search
returns, falling back to the rule-based ``router.route()`` and
guaranteeing a legal action even when everything else fails.

Hidden state for the search (EXP-A44 #P2): the real determinization
samplers (``hidden_information.own_hidden_state`` /
``opponent_hidden_state``, kept per-match by
``hidden_information.match_context``) are used through
``hidden_information.search_adapter.to_search_begin_kwargs``. The old
dummy stub (``hidden_information.search_state_stub``) is kept as the
fallback used when ``OpponentHiddenState.is_ready`` is False (no
archetype pool loaded yet) or when anything above raises.

Stage order (EXP-A44 #P3): ``lethal_simple`` -> ``risk_determinization``
-> ``router.route`` -> ``fallback.safe_choice``. Each stage returns
``None`` (or raises, caught here) to fall through to the next one.
``risk_determinization`` is gated by its own config section
(``configs/rule_risk.json``, default OFF; absent in
``configs/rule_lethal.json``) and only re-ranks the same candidate set
``rule_based.main_turn_parts.proposals`` would have produced (see
``_build_candidate_provider``).
"""

from __future__ import annotations

import random

from cg.api import Observation, SelectData

from ptcg_ai.action_selection import fallback, router
from ptcg_ai.core.config import load_config
from ptcg_ai.hidden_information import match_context, search_adapter
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.rule_based.main_turn_parts import proposals
from ptcg_ai.search import lethal_simple, risk_determinization

_SEARCH_MODULES = {
    "lethal_simple": lethal_simple,
    "risk_determinization": risk_determinization,
}

_CONFIG_CACHE: dict | None = None


def _config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


def _hidden_state_factory(obs: Observation, full_deck: list[int], rng: random.Random | None = None):
    """search_begin() 用の隠し情報を作る callable を返す（EXP-A44 #P2）。

    ``match_context``（``rule_based_agent.agent()`` が毎ターン ``update()`` 済み）が保持する
    ``OwnHiddenState`` / ``OpponentHiddenState`` を ``search_adapter.to_search_begin_kwargs`` で
    ``search_begin()`` の引数へ変換する。呼ぶたびに ``sample()`` が呼ばれるので、呼ぶたびに
    別の決定化サンプルになる（``lethal_simple`` の ``verify_shuffles`` が期待する挙動）。

    ``OpponentHiddenState.is_ready`` が False（代表リスト未ロード）のとき、および途中で
    何らかの例外が起きたときは、旧来のダミースタブ（``search_state_stub``）へフォールバックする
    （要件書 §5.6: stub は削除せずフォールバックとして残す）。
    """

    def factory() -> dict | None:
        try:
            opponent_state = match_context.get_opponent_state()
            if opponent_state.is_ready:
                own_state = match_context.get_own_state()
                return search_adapter.to_search_begin_kwargs(own_state, opponent_state, obs, rng)
        except Exception:
            pass
        try:
            return build_dummy_search_state(obs, full_deck, rng)
        except Exception:
            return None

    return factory


def _build_candidate_provider(obs: Observation):
    """``risk_determinization`` 用の ``candidate_provider`` を作る（EXP-A44 #P3、要件書 §5.5）。

    ``proposals.collect_proposals(obs)`` の結果に、``proposals.decide()`` と同じ setup-first
    フィルタ（自己枯渇する下準備が残っている間は attack 等より先に消化する不変条件）を適用した
    集合から、``proposals._total_score`` の降順で ``list[tuple[action, rule_score]]`` を返す。

    ``decide()`` 自体が持つ ``_setup_budget``（1ターンあたりの下準備採用回数の上限。無限ループの
    安全弁）は、ここでは意図的に消費しない: この関数は「候補を並べ替えて見せるだけ」で、
    実際にどれかを選んで手を進めるのは呼び出し側（``risk_determinization.search()`` が返した
    action、または router 経由の ``decide()`` 自身）であり、二重にカウントすると
    ``decide()`` 側の予算管理と整合しなくなる。setup-first の核となる不変条件
    （下準備が残る間は attack を候補に混ぜない）はこの関数でも変わらず守られる。
    """

    def provider() -> list[tuple[list[int], float]]:
        all_proposals = proposals.collect_proposals(obs)
        candidates = all_proposals
        if proposals.SETUP_BEFORE_ATTACK:
            setup = [p for p in all_proposals if p.category in proposals._SELF_DEPLETING_SETUP]
            if setup:
                candidates = setup
        ordered = sorted(candidates, key=proposals._total_score, reverse=True)
        return [(p.select, proposals._total_score(p)) for p in ordered]

    return provider


def is_valid_action(action, select: SelectData) -> bool:
    """Check the contract required by the competition runner."""
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (select.minCount <= len(action) <= select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(select.option) for i in action)


def select_action(obs: Observation, full_deck: list[int], config: dict | None = None) -> list[int]:
    """Choose the action for the current selection.

    Args:
        obs: Observation passed to the agent (``obs.select`` must be set).
        full_deck: Our own 60-card deck list (used to build the dummy
            hidden state handed to the search).
        config: Agent config dict (``lethal_search`` section is used).
            Defaults to ``core.config.load_config()``.

    Returns:
        list[int]: A legal selection.
    """
    select = obs.select
    if config is None:
        config = _config()
    lethal_config = (config or {}).get("lethal_search") or {}
    risk_config = (config or {}).get("risk_determinization") or {}

    # 相手デッキ予測の更新は lethal_search / router のどちらに進む前にも必ず通したいので、
    # この関数の一番手前で行う。予測結果は priorities/*.py が
    # opponent_tracker.current_matchup_plan() 経由で参照する。失敗しても通常運用は継続する。
    if obs.current is not None:
        try:
            opponent_tracker.update(obs)
        except Exception:
            pass

    if lethal_config.get("enabled", False) and obs.current is not None:
        module = _SEARCH_MODULES.get(lethal_config.get("module", "lethal_simple"))
        if module is not None:
            context = {
                "observation": obs,
                "config": lethal_config,
                # 実サンプラ(OwnHiddenState/OpponentHiddenState)経由。is_ready==False や
                # 例外時はダミースタブへ内部でフォールバックする(_hidden_state_factory参照)。
                "hidden_state_factory": _hidden_state_factory(obs, full_deck),
            }
            try:
                action = module.search(obs.current, select.option, context)
            except Exception:
                action = None
            if action is not None and is_valid_action(action, select):
                return action

    if risk_config.get("enabled", False) and obs.current is not None:
        module = _SEARCH_MODULES.get(risk_config.get("module", "risk_determinization"))
        if module is not None:
            context = {
                "observation": obs,
                "config": risk_config,
                "hidden_state_factory": _hidden_state_factory(obs, full_deck),
                "candidate_provider": _build_candidate_provider(obs),
            }
            try:
                action = module.search(obs.current, select.option, context)
            except Exception:
                action = None
            if action is not None and is_valid_action(action, select):
                return action

    # No certain lethal (and no risk-layer override): play the normal rule-based policy.
    try:
        action = router.route(obs)
    except Exception:
        action = None
    if action is not None and is_valid_action(action, select):
        return action
    return fallback.safe_choice(obs)
