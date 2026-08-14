"""模倣ポリシー Agent の入口(Step2、イシュー未起票、ml-value-network の後続)。

`core.agent` から `AGENT_TYPE == "ml_policy"` のときに呼ばれる Agent 実装。
`ptcg_ai.learning.policy_model.PolicyModel`(選択肢スコアリング)で通常ターンの選択肢を
選ぶ。`rule_based/` `action_selection/` は変更しない(担当領域を尊重する)。
`action_selection/selector.py`(担当Bのルーター)は呼ばない方針を維持しているが、
`rule_based/main_turn_parts`(`proposals.py`/`weights.py`)は ATTACK専用ハイブリッド
(下記)のために読み取り専用で import・呼び出しする例外を1つだけ設けている
(`sample_submission/docs/plans/ml-value-network/step2-design.md` §1 の接続点方針からの
唯一の逸脱。attack-rulebased-hybrid-strategy.md §4.2 で境界を確認済み)。

デッキ選択(初回)は現行 `deck.csv`(フーディン)をそのまま使う。`rule_based_agent.py` の
`read_deck_csv()` は deck.csv を読むだけの共通ユーティリティ(rule_based 自体のゲーム判断
ロジックではない)であり、`main.py` 自身もここから re-export して使っている安定した部品
なので、そのまま再利用する(コード重複を避ける)。

選択肢が複数選択(`maxCount > 1`)の意思決定点は Step2 の学習スコープ外
(step2-design.md §2.4)。スコアの高い選択肢から `minCount`〜`maxCount` 件を独立に選ぶ
貪欲フォールバックで対応する。

さらに、模倣ポリシーが唯一 `rule_based` に劣る領域(ATTACK 選択、一致率 60.5% vs 94.9%)を
補うため、選択前に確定リーサル探索(`search.lethal_simple`)を試すハイブリッド構成を取る
(step2-lethal-hybrid.md)。`action_selection/selector.py` は最終フォールバックが
rule_based の router に固定されているため import せず、同じ制御フロー(config 読み込み・
contract 検証)だけを複製し、フォールバック先を PolicyModel に差し替えている。

確定リーサルが見つからない非リーサルATTACKについては、`config["attack_hybrid"]["enabled"]`
で opt-in するとさらに `_try_attack_hybrid` を試す(attack-rulebased-hybrid-implementation-plan.md
Step1)。`rule_based.main_turn_parts.proposals.collect_proposals(obs)` を呼び、rule_base自身が
「他にやるべき展開が残っていない」と判断できる局面に限定して"attack"カテゴリを選んだ場合
だけそれを採用する。既定(config未指定/`attack_hybrid`キー無し)では常に無効で挙動は不変。
"""

import os
import time

from cg.api import Observation, OptionType, SelectData
from ptcg_ai.core.config import load_config
from ptcg_ai.hidden_information import match_context, search_adapter
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.learning import value_shadow_log
from ptcg_ai.learning.policy_model import PolicyModel
from ptcg_ai.rule_based.main_turn_parts import proposals as rb_proposals
from ptcg_ai.rule_based.main_turn_parts import weights as rb_weights
from ptcg_ai.rule_based.rule_based_agent import read_deck_csv
from ptcg_ai.search import attack_plan, lethal_simple, pimc, pipeline
from ptcg_ai.search.lethal import entry as lethal_phase1

# `_try_attack_hybrid` 用。draw/board/ability/energyのいずれかが提案されている
# (=まだ他にやるべき展開が残っている)行ではATTACKゲートを発火させない。
#
# Step0のオフライン実測(results/2026-07-21_attack_hybrid_gate_offline.md)で、この絞り込み
# 無しに「rule_baseの最高スコアカテゴリが"attack"か」だけで判定すると一致率23.2%まで劣化する
# ことが分かった。原因はATTACKのKOボーナス(1000点)がboard等の得点(20点程度)を常に上回り、
# 進化を複数回してから最後に攻撃するような同一ターン内の連続したMAIN選択の途中でも即攻撃を
# 選んでしまうこと(rule_baseがPLAY/ABILITY/ATTACHで一致率が低いのと同根)。この4カテゴリの
# 提案が1つも無い行に絞ると一致率89.7%まで回復する(発火頻度は狭まるが、意図した安全側の
# 低recall・高精度の介入)。
_ATTACK_HYBRID_BLOCKING_CATEGORIES = {"draw", "board", "ability", "energy"}

# 探索モジュールの登録表。**登録しただけでは挙動は変わらない**: どれを使うかは
# config の `lethal_search.module` が決める(既定 `abl_5_full` は `lethal_simple`)。
# `lethal_phase1` は Phase 1/2 の確定リーサル探索(search/lethal/)。design.md 付録 Z.1 の
# `RNG_NON_INTERFERENCE_VERIFIED` が False のため既定 config では選ばない。opt-in は
# `configs/abl_5_full_lethalphase2.json`(PTCG_AI_ML_CONFIG で切り替え)。
_SEARCH_MODULES = {
    "lethal_simple": lethal_simple,
    "lethal_phase1": lethal_phase1,
    "pimc": pimc,
}

# `rule_based` と共有の `rule_lethal.json` ではなく専用 config を読む。
# 計測(40試合)では max_remaining_prizes を 2 -> 3 に上げると確定リーサルの検出が
# 28 -> 55 件に増え、かつ「リーサル発火した試合は 40/40 で勝利」(verify_rejects=0)と
# 誤検出は観測されなかった(詳細・実戦ログでの追加検証は step2-lethal-hybrid.md §6.5/§6.6)。
# この値を共有 config 側で変えると `rule_based` の挙動まで変わってしまうため、
# ml_policy 専用の config に分けている。
# `PTCG_AI_ML_CONFIG` で上書き可能(Stage1 A/B計測で ml_lethal_estimated 等に
# 切り替えるため)。Kaggle提出時はこの環境変数を設定できないため、提出したい config を
# デフォルト値としてここに直接指定する。
#
# 2026-07-25: 意思決定パイプライン full(abl_5_full: belief決定化N=8 + 相手デッキ推定
# + Policy top-k先読み + lethal_simple)を提出用の既定に採用。ローカルのミラー自己対戦
# ablation(docs/plans/decision-pipeline/ablation-results-2026-07-25.md)では、fullは素の
# policy_only/searchN1 を有意に上回った一方、現行本番 ml_lethal_attackplan_v0only との
# 直接A/B(200試合)では有意差なしだった。ミラー自己対戦では測れない「実フィールドでの
# 転移」を確かめるための提出(ref 54956037)。
#
# ※ Kaggleのこのコンペはレーティング型で publicScore は提出直後の初期値から時間をかけて
#   収束する。投入直後の 600.0 は未収束の初期値であり、本番相当(717.2, ref 54883922)との
#   比較は score が落ち着いてから行う。劣後が確定したら ml_lethal_attackplan_v0only に戻す。
#
# (履歴: 2026-07-22 は ml_lethal_attackplan_v0only を提出。ロック闘エネルギー等で攻撃が
# 0ダメージになる局面の事後veto。ローカル400試合では有意差未確認・エラー0件だった。)
#
# 2026-08-14: 提出用の既定を abl_5_full_lethalphase2 に変更(champion climb の
# パイプラインのうち1段目の確定リーサルだけを Phase 1/2 探索器へ差し替えたもの)。
# pipeline / Policy 重み / デッキは climb(LB 823.5)と同一。
# **注意**: design.md 付録 Z.1 の production 接続条件のうち
# `RNG_NON_INTERFERENCE_VERIFIED` は False のまま(探索の有無だけを変えて本番 RNG を
# 比較する手段が SDK に無い= not verified。証明ロジック側の問題ではない)。
# 誤 PROVEN_WIN 0 / 不正行動 0 / replay 検証 100% / fallback 100% / リーク 0 は満たす。
# 元に戻すときはこの既定値を "abl_5_full" へ戻すだけでよい。
_CONFIG_NAME = os.environ.get("PTCG_AI_ML_CONFIG", "abl_5_full_lethalphase2")

_model: PolicyModel | None = None
_config_cache: dict | None = None
_deck_cache: list[int] | None = None
# パイプライン(pipeline.py)の動的時間予算用。1試合ごとにデッキ選択ターン(obs.select is None)
# でリセットする。league の worker はプロセスを跨いで再利用されるため、試合境界での
# リセットが必要(config["pipeline"]["time_budget"] を指定したときのみ参照される)。
_match_start_perf: float | None = None
_selects_seen: int = 0
# config で明示された `policy_weights_path` ごとにキャッシュする(既定パス=グローバル `_model`
# は変えず、注入されたパスだけ別枠に積む)。同一プロセス内で複数の候補重み(config違い)を
# 混線なく head-to-head させるための注入点(policymodel-skill-concentration-implementation-plan.md
# Step4)。`_config_cache`/`_model` と同じく、明示 config を渡さない限り挙動は不変。
_model_cache_by_weights_path: dict[str, PolicyModel] = {}


def agent(obs: Observation, config: dict | None = None) -> list[int]:
    """obs を見てデッキ返却 or 選択肢スコアリングによる選択を行う。

    Args:
        obs: core.agent から渡される Observation。
        config: 使う agent config。`None`(既定)なら `_get_config()`(モジュール
            グローバルにキャッシュされた `PTCG_AI_ML_CONFIG` の config)を使う。本番
            (`core.agent`)や `main.py` からは1引数 `agent(obs)` でのみ呼ばれるため
            挙動は不変。明示 config を渡す口は、同一プロセス内で `ml_lethal` と
            `ml_pimc` のような別configの2エージェントを混線なく head-to-head させる
            ために用意している(selector.select_action の `config` 引数の ml_policy 版。
            pimc-production-validation-implementation-plan.md §1)。`config["policy_weights_path"]`
            を指定すると、既定の `ptcg_ai/learning/policy_weights.json` の代わりにそのパスの
            重みJSONを読んだ `PolicyModel` を使う(policymodel-skill-concentration-
            implementation-plan.md Step4。候補重みと本番重みをプロセス内で混線なく
            head-to-head させるための注入点。パスごとにモデルをキャッシュするだけで、
            `_config_cache`/既定の `_model` には触れないため、この引数を渡さない限り
            挙動は不変)。

    Returns:
        list[int]: 初回はデッキの60枚のカードIDリスト。通常ターンは選択肢インデックスのリスト。
    """
    # 非公開情報推定レイヤー(hidden_information)の更新。純粋な副作用追加であり、
    # 失敗しても意思決定を止めない(match_context.update内部でtry/exceptしている)。
    # rule_based_agent.py と同じ呼び出し方(agent()の先頭で毎ターン呼ぶだけ)。
    match_context.update(obs)

    effective_config = config if config is not None else _get_config()

    # value network の shadow mode ログ(stage1-wiring-implementation-plan.md Step4)。
    # config の value_shadow_logging が true のときだけ、勝率予測をログに残すだけの
    # 純粋な副作用追加(value_shadow_log.record 内部でtry/exceptしている)。意思決定には使わない。
    if effective_config.get("value_shadow_logging", False):
        value_shadow_log.record(obs)

    if obs.select is None:
        # 新しい試合の開始。pipeline の動的時間予算のカウンタをリセットする
        # (pipeline 無効時も無害なただの代入)。
        global _match_start_perf, _selects_seen
        _match_start_perf = time.perf_counter()
        _selects_seen = 0
        return read_deck_csv()
    return _select_action(obs, config)


def _get_model(config: dict | None = None) -> PolicyModel:
    """`config["policy_weights_path"]` があればそのパスの重みを読んだ `PolicyModel` を返す
    (プロセス内でパスごとにキャッシュ)。無ければ従来どおりモジュールグローバルの
    既定モデル(`_model`)を返す(挙動不変)。"""
    weights_path = (config or {}).get("policy_weights_path")
    if weights_path is None:
        global _model
        if _model is None:
            _model = PolicyModel()
        return _model

    model = _model_cache_by_weights_path.get(weights_path)
    if model is None:
        model = PolicyModel(weights_path)
        _model_cache_by_weights_path[weights_path] = model
    return model


def _get_config() -> dict:
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config(_CONFIG_NAME)
    return _config_cache


def _get_deck() -> list[int]:
    global _deck_cache
    if _deck_cache is None:
        _deck_cache = read_deck_csv()
    return _deck_cache


def _is_valid_action(action, select: SelectData) -> bool:
    """コンペランナーが要求する contract を満たすかを検証する。

    `action_selection/selector.py` の同名関数と同じ内容だが、`action_selection/` への
    依存をゼロに保つため意図的に複製している(step2-lethal-hybrid.md §4)。
    """
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (select.minCount <= len(action) <= select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(select.option) for i in action)


def _try_lethal(obs: Observation, config: dict | None = None) -> list[int] | None:
    """確定リーサルが見つかればその最初の選択を返す。無ければ None。

    模倣ポリシーは打点計算・詰み筋の読みを内在的に持たないため、`rule_based` が既に
    運用している確定リーサル探索をここで先に試す。探索は例外を投げない設計だが、
    呼び出し側でも二重に保護する(リーサル探索の失敗が対戦を落としてはならない)。

    Returns:
        list[int] | None: リーサル手順の最初の選択。見つからない/無効/例外時は None。
    """
    if obs.current is None:
        return None

    if config is None:
        config = _get_config()
    lethal_config = (config or {}).get("lethal_search") or {}
    if not lethal_config.get("enabled", False):
        return None

    module = _SEARCH_MODULES.get(lethal_config.get("module", "lethal_simple"))
    if module is None:
        return None

    try:
        hidden_state_source = lethal_config.get("hidden_state_source", "dummy")
        if hidden_state_source == "estimated":
            # 実推定(hidden_information.match_context、agent()冒頭で毎ターン更新済み)。
            factory = lambda: search_adapter.to_search_begin_kwargs(
                match_context.get_own_state(obs.current.yourIndex),
                match_context.get_opponent_state(obs.current.yourIndex),
                obs,
            )
        else:
            # ダミースタブ(既定、既存configとの後方互換)。
            factory = lambda: build_dummy_search_state(obs, _get_deck())
        context = {
            "observation": obs,
            "config": lethal_config,
            "hidden_state_factory": factory,
            # 自分のデッキ構成(60枚)。`lethal_phase1` は境界で multiset へ落とし、
            # Phase 2 の outcome 列挙に使う(無いと Phase 2 は必ず UNKNOWN)。
            # 実際の山札順ではないので隠れ情報の漏洩にはならない。`lethal_simple`/`pimc`
            # はこのキーを見ないので挙動は不変。
            "full_deck": _get_deck(),
        }
        action = module.search(obs.current, obs.select.option, context)
    except Exception:
        return None

    if action is not None and _is_valid_action(action, obs.select):
        return action
    return None


def _try_attack_hybrid(obs: Observation, config: dict | None = None) -> list[int] | None:
    """rule_base が『このターンは attack カテゴリを選ぶべき』と判定した場合のみ、
    そのATTACK選択のインデックスを返す。それ以外(config無効/ATTACK型選択肢が無い/
    draw・board・ability・energyのいずれかが提案されている/rule_baseが他カテゴリを
    選ぶ)は None を返し、呼び出し側はPolicyModelに委ねる。

    `rule_based/main_turn_parts`(`proposals.py`/`weights.py`)を読み取り専用で
    import・呼び出しするだけで、ファイル自体は変更しない
    (attack-rulebased-hybrid-strategy.md §4.2 で境界を確認済み)。

    Returns:
        list[int] | None: rule_base が選んだATTACKの選択(1要素)。条件を満たさない/
            例外時/contract違反時は None。
    """
    if obs.current is None:
        return None

    if config is None:
        config = _get_config()
    hybrid_config = (config or {}).get("attack_hybrid") or {}
    if not hybrid_config.get("enabled", False):
        return None

    select = obs.select
    if not any(option.type == OptionType.ATTACK for option in select.option):
        return None

    try:
        rb_props = rb_proposals.collect_proposals(obs)
    except Exception:
        return None
    if not rb_props:
        return None

    categories_present = {p.category for p in rb_props}
    if categories_present & _ATTACK_HYBRID_BLOCKING_CATEGORIES:
        return None

    def total_score(p):
        return p.score + rb_weights.CATEGORY_BASE_WEIGHT.get(p.category, 0.0)

    best = max(rb_props, key=total_score)
    if best.category != "attack":
        return None

    if not _is_valid_action(best.select, select):
        return None
    return best.select


def _try_attack_plan(
    obs: Observation,
    chosen_action: list[int],
    config: dict | None = None,
    policy_scores: list[float] | None = None,
) -> list[int] | None:
    """Post-validate the final normal-policy choice; disabled by default."""
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    plan_config = (effective_config or {}).get("attack_plan") or {}
    if not plan_config.get("enabled", False) or len(chosen_action) != 1:
        return None
    try:
        if plan_config.get("hidden_state_source", "dummy") == "estimated":
            factory = lambda: search_adapter.to_search_begin_kwargs(
                match_context.get_own_state(obs.current.yourIndex),
                match_context.get_opponent_state(obs.current.yourIndex), obs,
            )
        else:
            full_deck = _get_deck()
            factory = lambda: build_dummy_search_state(obs, full_deck)
        action = attack_plan.search(obs.current, obs.select.option, {
            "observation": obs, "chosen_action": chosen_action,
            "config": plan_config, "hidden_state_factory": factory,
            "policy_scores": policy_scores or [],
        })
    except Exception:
        return None
    return action if action is not None and _is_valid_action(action, obs.select) else None


# PolicyModelがconsequence特徴(Tier3 Stage3c、meta.consequence_fieldsを持つ重み)を使う
# 場合の仮実行に割り当てる時間予算。既定重み(consequence特徴なし)ではこの値は一切参照
# されない(PolicyModel._consequence_fields が空なら factory/deadline は無視される)。
_MODEL_TIME_BUDGET_MS = 100


def _model_hidden_state_factory(obs: Observation, config: dict | None):
    """PolicyModelのconsequence特徴計算に使う hidden_state_factory。

    `_try_lethal`/`_try_attack_plan` と同じ dummy/estimated 切替
    (config["policy_model"]["hidden_state_source"]、既定 "dummy")。consequence特徴を
    使わない重みではこの factory は呼ばれずコストゼロ(遅延評価のlambdaのみ構築)。
    """
    effective_config = config if config is not None else _get_config()
    model_config = (effective_config or {}).get("policy_model") or {}
    if model_config.get("hidden_state_source", "dummy") == "estimated":
        return lambda: search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(obs.current.yourIndex),
            match_context.get_opponent_state(obs.current.yourIndex), obs,
        )
    full_deck = _get_deck()
    return lambda: build_dummy_search_state(obs, full_deck)


def _dynamic_pipeline_time_limit_ms(pipeline_config: dict) -> float:
    """`config["pipeline"]["time_budget"]` があれば「残り時間 ÷ 推定残り選択数」で1手予算を
    算出する。無ければ固定 `time_limit_ms`(既定は pipeline.DEFAULTS の値)を返す。

    予算切れ・未初期化でも必ず正の値を返す(pipeline 側の deadline は壁時計で二重に保護
    されており、この関数の失敗が反則負けにつながることはない)。
    """
    base = float(pipeline_config.get("time_limit_ms", pipeline.DEFAULTS["time_limit_ms"]))
    budget = pipeline_config.get("time_budget")
    if not budget or _match_start_perf is None:
        return base
    try:
        total_ms = float(budget["total_ms"])
        min_ms = float(budget.get("min_ms", 50))
        max_ms = float(budget.get("max_ms", 2000))
        assumed_total = int(budget.get("assumed_total_selects", 400))
        elapsed_ms = (time.perf_counter() - _match_start_perf) * 1000.0
        remaining_ms = total_ms - elapsed_ms
        if remaining_ms <= 0:
            return min_ms
        remaining_selects = max(1, assumed_total - _selects_seen)
        per_move = remaining_ms / remaining_selects
        return max(min_ms, min(max_ms, per_move))
    except Exception:
        return base


def _try_pipeline(obs: Observation, config: dict | None = None) -> list[int] | None:
    """`config["pipeline"]["enabled"]` が真のときだけ統合パイプライン(`search.pipeline`)を
    試す。適用外(単一選択の MAIN 以外)・失敗・予算切れ前に評価不能なら None を返し、
    呼び出し側は既存の Policy top1 経路にフォールバックする。

    本番 config(`ml_lethal_attackplan_v0only`)は `pipeline` キーを持たないため常に None を
    返し、本番挙動は完全に不変。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    pipeline_config = (effective_config or {}).get("pipeline") or {}
    if not pipeline_config.get("enabled", False):
        return None

    # 動的時間予算を反映した config を1回だけ組み立てる(元 config は不変)。
    run_config = dict(pipeline_config)
    run_config["time_limit_ms"] = _dynamic_pipeline_time_limit_ms(pipeline_config)

    try:
        if pipeline_config.get("hidden_state_source", "estimated") == "dummy":
            full_deck = _get_deck()
            factory = lambda: build_dummy_search_state(obs, full_deck)
        else:
            factory = lambda: search_adapter.to_search_begin_kwargs(
                match_context.get_own_state(obs.current.yourIndex),
                match_context.get_opponent_state(obs.current.yourIndex),
                obs,
            )
        model = _get_model(config)
        action = pipeline.search(obs.current, obs.select.option, {
            "observation": obs,
            "config": run_config,
            "hidden_state_factory": factory,
            "model_hidden_state_factory": _model_hidden_state_factory(obs, config),
            "policy_model": model,
        })
    except Exception:
        return None

    if action is not None and _is_valid_action(action, obs.select):
        return action
    return None


def _select_action(obs: Observation, config: dict | None = None) -> list[int]:
    select = obs.select

    # pipeline の動的時間予算用に、意思決定を要する選択のたびにカウンタを進める
    # (pipeline 無効時も無害)。
    global _selects_seen
    _selects_seen += 1

    lethal_action = _try_lethal(obs, config=config)
    if lethal_action is not None:
        return lethal_action

    pipeline_action = _try_pipeline(obs, config=config)
    if pipeline_action is not None:
        return pipeline_action

    model = _get_model(config)
    model_factory = _model_hidden_state_factory(obs, config)
    model_deadline = time.perf_counter() + _MODEL_TIME_BUDGET_MS / 1000
    attack_hybrid_action = _try_attack_hybrid(obs, config=config)
    if attack_hybrid_action is not None:
        baseline_action = attack_hybrid_action
        policy_scores = None
    elif select.maxCount == 1:
        idx = model.select_option(obs, model_factory, model_deadline)
        baseline_action = [idx if idx is not None else 0]
        effective_config = config if config is not None else _get_config()
        plan_config = (effective_config or {}).get("attack_plan") or {}
        policy_scores = (
            model.score_options(obs, model_factory, model_deadline)
            if plan_config.get("enabled", False) else None
        )
    else:
        baseline_action = _greedy_multi_select(obs, model, select, model_factory, model_deadline)
        policy_scores = None

    planned_action = _try_attack_plan(obs, baseline_action, config, policy_scores)
    return planned_action if planned_action is not None else baseline_action


def _greedy_multi_select(
    obs: Observation,
    model: PolicyModel,
    select: SelectData,
    hidden_state_factory=None,
    deadline: float | None = None,
) -> list[int]:
    """maxCount > 1(Step2 学習スコープ外)向けの貪欲フォールバック。

    各選択肢を独立にスコアリングし、上位から minCount〜maxCount 件を選ぶ。組み合わせの
    最適性は保証しない(step2-design.md §2.4 の既知の制約)。
    """
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))

    scores = model.score_options(obs, hidden_state_factory, deadline)
    if not scores:
        return list(range(count))

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return ranked[:count]

