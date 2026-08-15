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

from cg.api import AreaType, Observation, OptionType, SelectContext, SelectData, SelectType
from ptcg_ai.core.config import load_config
from ptcg_ai.hidden_information import match_context, search_adapter
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.learning import value_shadow_log
from ptcg_ai.learning.policy_model import PolicyModel
from ptcg_ai.learning.strategy_residual import StrategyResidual
from ptcg_ai.rule_based.main_turn_parts import proposals as rb_proposals
from ptcg_ai.rule_based.main_turn_parts import weights as rb_weights
from ptcg_ai.rule_based.rule_based_agent import read_deck_csv
from ptcg_ai.search import attack_plan, lethal_simple, pimc, pipeline

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

_SEARCH_MODULES = {
    "lethal_simple": lethal_simple,
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
_CONFIG_NAME = os.environ.get("PTCG_AI_ML_CONFIG", "abl_5_full")

_model: PolicyModel | None = None
_config_cache: dict | None = None
_deck_cache: list[int] | None = None
# パイプライン(pipeline.py)の動的時間予算用。1試合ごとにデッキ選択ターン(obs.select is None)
# でリセットする。league の worker はプロセスを跨いで再利用されるため、試合境界での
# リセットが必要(config["pipeline"]["time_budget"] を指定したときのみ参照される)。
_match_start_perf: float | None = None
_selects_seen: int = 0
# 時間予算v2(`time_budget["mode"] == "expensive_roots"` のときだけ参照)。実測では1試合の総select
# 数は約72.5に対し、重い探索(pipeline.search)が走る決定は約29回しかない。「全selectで割る」旧式は
# 予算を大幅に余らせるため、**探索が実際に走った回数**で残り時間を割る。`_pipeline_breaker_tripped`
# は1手が割当を大きく超過したときに立てるサーキットブレーカ(その試合はそれ以降探索しない)。
# どちらも `agent()` の試合開始パスでリセットする(mode キーが無い config では未参照=挙動不変)。
_expensive_roots_seen: int = 0
_pipeline_breaker_tripped: bool = False
# config で明示された `policy_weights_path` ごとにキャッシュする(既定パス=グローバル `_model`
# は変えず、注入されたパスだけ別枠に積む)。同一プロセス内で複数の候補重み(config違い)を
# 混線なく head-to-head させるための注入点(policymodel-skill-concentration-implementation-plan.md
# Step4)。`_config_cache`/`_model` と同じく、明示 config を渡さない限り挙動は不変。
_model_cache_by_weights_path: dict[str, PolicyModel] = {}
# 動的ルーティング(config["routing"])の現在の差替先。decision毎に _select_action 冒頭で
# _compute_route が更新する。None=既定(base/climb)を使う。routing キー無しの config では常に
# None のままなので本番挙動は不変(config-gated)。相手デッキ予測の argmax が閾値以上かつ専用が
# あるときだけ、その decision の _get_model が専用重みを返す(=途中で相手予測が変われば切替わる)。
_current_route_path: str | None = None
# ハンマーのターゲット・リダイレクト(config-gated `hammer_veto.redirect_target`)用の
# モジュール状態。ハンマーは「PLAY する decision」と「剥がすエネを選ぶ decision」が分かれて
# いるため、PLAY 決定時に計算済みの `ko_search.can_ko_this_turn` の結果を次の decision まで
# 持ち越す。試合を跨いで残ると誤判定になるので `agent()` の試合開始パスでクリアする
# (league の worker はプロセスを跨いで再利用されるため必須)。redirect_target キーが無い
# config では誰も読まないので本番挙動は不変。
_hammer_ko_cache: dict | None = None


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
        global _expensive_roots_seen, _pipeline_breaker_tripped
        global _hammer_ko_cache
        _match_start_perf = time.perf_counter()
        _selects_seen = 0
        # 時間予算v2 のカウンタ/ブレーカも試合境界でリセット(mode 未指定なら未参照のまま)。
        _expensive_roots_seen = 0
        _pipeline_breaker_tripped = False
        # ハンマーのターゲット・リダイレクト用のKO判定キャッシュも試合境界で捨てる
        # (前の試合の判定が次の試合の最初のハンマーに漏れないようにする)。
        _hammer_ko_cache = None
        return read_deck_csv()
    return _select_action(obs, config)


def _get_model(config: dict | None = None) -> PolicyModel:
    """`config["policy_weights_path"]` があればそのパスの重みを読んだ `PolicyModel` を返す
    (プロセス内でパスごとにキャッシュ)。無ければ従来どおりモジュールグローバルの
    既定モデル(`_model`)を返す(挙動不変)。"""
    # 動的ルーティング: この decision で専用が選ばれていれば(=_current_route_path 有効)、
    # base(既定/policy_weights_path)より優先してその専用重みを返す。ロード失敗/未readyなら
    # 安全側で base にフォールバック(routing キー無し=常に None なので本番不変)。
    if _current_route_path is not None:
        routed = _model_cache_by_weights_path.get(_current_route_path)
        if routed is None:
            try:
                routed = PolicyModel(_current_route_path, deck_card_ids=_get_deck())
            except Exception:  # noqa: BLE001 - ルーティング失敗が意思決定を止めてはならない
                routed = None
            _model_cache_by_weights_path[_current_route_path] = routed  # None もキャッシュ(再試行抑止)
        if routed is not None and getattr(routed, "is_ready", False):
            return routed
        # フォールスルー: base を使う(下の既定ロジック)。

    # deck を渡す: 入力拡張(meta.extra_features)重みは runtime で own_resource_features を
    # デッキ基準に計算する必要がある(parity)。既定(flag無し)重みは deck を無視するので不変。
    weights_path = (config or {}).get("policy_weights_path")
    if weights_path is None:
        global _model
        if _model is None:
            _model = PolicyModel(deck_card_ids=_get_deck())
        return _model

    model = _model_cache_by_weights_path.get(weights_path)
    if model is None:
        model = PolicyModel(weights_path, deck_card_ids=_get_deck())
        _model_cache_by_weights_path[weights_path] = model
    return model


def _compute_route(obs: Observation, config: dict | None) -> str | None:
    """この decision で使う専用重みパスを返す(無ければ None=base/climb)。

    `config["routing"]`(既定 config には無い=本番不変)が有効なとき、現在の相手デッキ予測の
    argmax アーキが `confidence_threshold` 以上 かつ `specialists` にそのアーキの専用重みが
    あれば、そのパスを返す。**decision 毎に呼ぶ**ので、相手予測が試合途中で変われば差替先も
    切替わる(序盤=予測未確定は None=climb、Crustle 確定後は crustle 専用、等)。予測器未
    ロード/State無し/例外/低確信は全て None(安全側=base)。予測は belief 特徴と同じ純経路
    (現在 State の相手可視カード → HybridDeckPredictor.predict)を使う。
    """
    r = (config or {}).get("routing")
    if not r or not r.get("enabled"):
        return None
    if obs is None or obs.current is None:
        return None
    # 状態条件ルート(優先): 山札が低い=延命スペシャリストへ。deck管理は終盤限定スキルなので、
    # 全ゲームでなく"山切れ間近"だけ専用に差し替える(climbは終盤以外を担い総合力を保つ)。予測器
    # 不要・アーキ条件より先(山切れは最優先の状況)。既定 config に routing キー無し=本番不変。
    low = r.get("low_deck")
    if low and low.get("weights"):
        try:
            me = obs.current.yourIndex
            if int(obs.current.players[me].deckCount or 0) <= int(low.get("deck_threshold", 8)):
                return low["weights"]
        except Exception:  # noqa: BLE001
            pass
    # カード検出ルート(予測器にクラスが無いアーキ用。相手の場/トラッシュにシグネチャカードが
    # あればその専用へ)。例: ロケット団ミュウツーex(431)/フリーザー(414)。予測器不要・早期発火。
    card_routes = r.get("card_routes") or []
    if card_routes:
        try:
            opp = obs.current.players[1 - obs.current.yourIndex]
            seen: set = set()
            for pk in (list(opp.active or []) + list(opp.bench or [])):
                pid = getattr(pk, "id", None) if pk is not None else None
                if pid is not None:
                    seen.add(pid)
            for c in (opp.discard or []):
                cid = getattr(c, "id", None) if c is not None else None
                if cid is not None:
                    seen.add(cid)
            for cr in card_routes:
                if cr.get("weights") and (set(cr.get("any_of") or []) & seen):
                    return cr["weights"]
        except Exception:  # noqa: BLE001
            pass

    specialists = r.get("specialists") or {}
    if not specialists:
        return None
    try:
        predictor = match_context._get_predictor()
        if predictor is None or not getattr(predictor, "is_ready", False):
            return None
        from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge
        knowledge = OpponentKnowledge()
        knowledge.update_from_state(obs.current)
        observed = knowledge.get_prediction_features().get("observed_cards", {})
        probs = predictor.predict(observed, int(getattr(obs.current, "turn", 0) or 0))
    except Exception:  # noqa: BLE001 - 予測失敗がルーティング/意思決定を止めてはならない
        return None
    if not probs:
        return None
    top_arch = max(probs, key=probs.get)
    thr = float(r.get("confidence_threshold", 0.6))
    if probs[top_arch] >= thr and top_arch in specialists:
        return specialists[top_arch]
    return None


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
            full_deck = _get_deck()
            factory = lambda: build_dummy_search_state(obs, full_deck)
        context = {
            "observation": obs,
            "config": lethal_config,
            "hidden_state_factory": factory,
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


_ENHANCED_HAMMER_ID = 1081  # 改造ハンマー: 相手アクティブの特殊エネのみ剥がせる
_CRUSHING_HAMMER_ID = 1120  # クラッシュハンマー: コイン成功で相手の任意ポケモンからエネを1枚剥がせる
# config["hammer_veto"]["card_ids"] 省略時の既定値=現行挙動(改造ハンマーのみ対象、本番不変)。
_DEFAULT_HAMMER_VETO_CARD_IDS = [_ENHANCED_HAMMER_ID]


def _opp_has_energy_off_active(state, *, special_only: bool) -> bool:
    """相手の「アクティブ以外(=ベンチ)」のポケモンにエネが付いているか。

    ハンマー浪費vetoの guard。剥がしても無駄なのは「このターンKOする相手=アクティブ」の
    エネだけ。ベンチ等の(KOしない)ポケモンにエネがあれば、そこへのハンマーは正当なので
    veto してはいけない(ユーザー指摘: ベンチに対象がいて使いたい場面を潰さない)。

    ``special_only=True`` なら特殊エネのみを対象に判定する(改造ハンマー=相手アクティブの
    特殊エネしか剥がせないため)。``special_only=False`` ならエネの種類を問わず判定する
    (クラッシュハンマー=コイン成功で相手の任意のポケモンから任意のエネを1枚剥がせるため。
    浪費の条件=「KOできるアクティブから剥がしても無駄」自体は改造ハンマーと同じ)。
    判定不能は False(=guard効かず、従来どおり veto を続行=浪費防止優先)。
    """
    try:
        from cg.api import CardType
        from ptcg_ai.shared import card_cache
        me = state.yourIndex
        opp = state.players[1 - me]
        for pk in (opp.bench or []):
            if pk is None:
                continue
            for en in (getattr(pk, "energyCards", None) or []):
                if not special_only:
                    return True  # 種類問わずベンチにエネがあれば正当な対象=guard発火
                try:
                    if card_cache.get_card(en.id).cardType == CardType.SPECIAL_ENERGY:
                        return True
                except Exception:  # noqa: BLE001 - 未知IDは無視
                    continue
    except Exception:  # noqa: BLE001
        return False
    return False


def _hammer_veto_guard_blocks(state, card_id: int) -> bool:
    """打とうとしているハンマー(card_id)に応じた guard を選び、veto を止めるべきかを返す。

    True ならベンチに正当な対象がいる=veto してはいけない(呼び出し側は None を返す)。
    False なら guard は掛からず veto を続行してよい。改造ハンマー(1081)は特殊エネのみ、
    クラッシュハンマー(1120)は任意のエネで判定する(_opp_has_energy_off_active 参照)。
    それ以外の card_id(config で将来未知のハンマーIDを指定した場合)は「判定不能」として
    False を返す(=guardを掛けない、従来どおり安全側でveto継続)。
    """
    if card_id == _ENHANCED_HAMMER_ID:
        return _opp_has_energy_off_active(state, special_only=True)
    if card_id == _CRUSHING_HAMMER_ID:
        return _opp_has_energy_off_active(state, special_only=False)
    return False


def _try_hammer_veto(obs: Observation, chosen_action: list[int], config: dict | None = None) -> list[int] | None:
    """ハンマー系カードの浪費を事後veto。config-gated(hammer_veto)、既定OFF=本番不変。

    観測バグ: 「このターンに相手アクティブをKOできる(=剥がしても結末が変わらない)のに
    ハンマーを打つ」浪費。lethal探索の"詰み認識"の親戚として、**KO可能なら**ハンマーPLAYを
    却下し、ハンマー以外の最善手(方策スコア最大)に差し替える。KO不能ならハンマーは有用な
    可能性があるので干渉しない(安全側)。手書きの相性ルールではなく「KOできる=剥がし不要」
    という盤面事実に基づく veto。

    対象カードは `config["hammer_veto"]["card_ids"]`(既定 `_DEFAULT_HAMMER_VETO_CARD_IDS`=
    [1081]=改造ハンマーのみ、現行挙動不変)。[1120] を指定するとクラッシュハンマーにも同じ
    判定を適用する(guard はカードごとに切り替わる。`_hammer_veto_guard_blocks` 参照)。

    `config["hammer_veto"]["ignore_bench_guard"]`(任意キー、既定なし=現行 guard 維持)を真に
    すると、上記 guard を無視して「KOできるなら常に veto(=温存)」する。実ラダーのリプレイ
    監査で、guard が掛かる盤面(ベンチにエネあり)でも実際の狙いは**アクティブ**だった浪費が
    16/16 件あり、guard が浪費を素通ししていたことが分かったため。「温存」ではなく「撃つなら
    ベンチへ」で直す道は `redirect_target`(`_try_hammer_redirect`)。どちらが良いかを測る
    ための ablation 用に独立キーにしてある(両方ONなら播き時の veto が先に効く)。

    判定は best_effective_attack_damage(エネ充足・弱点・ロック込みの実効最大打点を仮実行で
    算出) >= 相手アクティブの残HP。推論時のみで学習コストは無い。ハンマーを選んだ時だけ
    仮実行するので通常手番のコストはゼロ。

    `config["hammer_veto"]["veto_mode"]`(既定 `"swap"`=現行挙動、`"shadow"` で shadow
    モード)。実測(1500試合A/B)で veto+redirect の合成は勝率-0.87pt(ns)、veto発火
    1.01回/試合は実ラダーの無駄撃ち頻度0.29回/試合の3.5倍=ko_searchの楽観誤りで正当な
    ハンマーまで抑止している疑いがある一方、redirect(0.135回/試合)は的が絞れている。
    redirect 単独の効果を切り分けたいが、`_try_hammer_redirect` はこの関数が播く
    `_hammer_ko_cache` に依存する(ターゲット時のko_search再実行は効果解決中の1枚ズレで
    死ぬ。`_hammer_redirect_can_ko` の実測メモ参照)ため、redirect だけを有効にしたくても
    このKO判定+キャッシュ構築部分は走らせる必要がある。`veto_mode="shadow"` はまさにそれ:
    KO判定とキャッシュ構築は行うが、**手は差し替えない**(常に None を返す=呼び出し側は
    選んだ手をそのまま使う)。キャッシュ破棄(veto発火時に捨てる下の処理)も shadow では
    通らない(redirect が使うキャッシュを保持したいため)。

    `config["hammer_veto"]["apply_veto"]`(既定 True=現行挙動)は上記 shadow の別名で、
    **False にすると `veto_mode="shadow"` と同じ**(ko_search とキャッシュ構築は走るが
    PLAY の差し替えはしない=redirect 用のキャッシュだけを供給する)。どちらか一方でも
    shadow を指示していれば shadow になる。キー自体が無い既存 config は True 扱いなので
    挙動は不変。
    """
    global _hammer_ko_cache
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    hv_config = (effective_config or {}).get("hammer_veto") or {}
    if not hv_config.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    hammer_ids = hv_config.get("card_ids") or _DEFAULT_HAMMER_VETO_CARD_IDS
    from ptcg_ai.learning import encoder as _enc
    chosen_card_id = _enc._resolve_card_id(select.option[idx], obs.current)
    if chosen_card_id not in hammer_ids:
        return None  # 選んだ手が対象ハンマーでない=無関係

    # このターン中に(準備込みで)KOできるか=ターン先読み探索(瞬間打点でなく、エネ装着→攻撃等
    # の手順を探索)。KOできるなら剥がしは無駄。KO不能ならハンマーは有用な可能性→干渉しない。
    try:
        from ptcg_ai.search import ko_search
        factory = _model_hidden_state_factory(obs, config)
        ko_cfg = hv_config.get("ko_search") or {}
        deadline = time.perf_counter() + float(ko_cfg.get("time_limit_ms", hv_config.get("time_limit_ms", 100))) / 1000.0
        can_ko = bool(ko_search.can_ko_this_turn(obs, factory, ko_cfg, deadline))
    except Exception:  # noqa: BLE001 - veto の失敗が意思決定を止めてはならない
        return None
    # このKO判定を「直後に来るターゲット選択」へ持ち越す(`_try_hammer_redirect` が読む)。
    # redirect_target キーが無ければ誰も読まない純粋な副作用。
    _hammer_ko_cache = _make_hammer_ko_cache(obs, chosen_card_id, can_ko)
    # shadow モード: KO判定とキャッシュ構築はここまでで完了、手は差し替えずに戻る
    # (呼び出し側は選ばれた元の手をそのまま使う)。以降のguard/差替/キャッシュ破棄は
    # 一切通らない=redirect用のキャッシュはそのまま保持される。既定("swap")では
    # このブロックは素通りする(veto_mode キー自体が無い既存configも同様)。
    # `apply_veto=false` は `veto_mode="shadow"` の別名(どちらでも shadow になる)。
    if hv_config.get("veto_mode", "swap") == "shadow" or not hv_config.get("apply_veto", True):
        return None
    if not can_ko:
        return None

    # guard: KOしない相手(ベンチ)に正当な対象があれば、そこへのハンマーは正当→vetoしない。
    # 判定基準はカードごとに切り替わる(_hammer_veto_guard_blocks)。
    # `ignore_bench_guard`(任意キー)が真ならこの guard を無視して常に veto する。
    if not hv_config.get("ignore_bench_guard", False) and _hammer_veto_guard_blocks(obs.current, chosen_card_id):
        return None

    # ここに来たら: ハンマーを打とうとしているが今アクティブをKOできる=浪費。
    # 対象ハンマー(hammer_ids)以外で方策スコア最大の手に差し替える。
    try:
        model = _get_model(config)
        factory2 = _model_hidden_state_factory(obs, config)
        deadline2 = time.perf_counter() + _MODEL_TIME_BUDGET_MS / 1000.0
        scores = model.score_options(obs, factory2, deadline2)
    except Exception:  # noqa: BLE001
        return None
    if not scores or len(scores) != len(select.option):
        return None
    best_alt = best_score = None
    for i, sc in enumerate(scores):
        if _enc._resolve_card_id(select.option[i], obs.current) in hammer_ids:
            continue  # 全ての対象ハンマーPLAYを除外
        if best_score is None or sc > best_score:
            best_score, best_alt = sc, i
    if best_alt is None:
        return None
    action = [best_alt]
    if not _is_valid_action(action, select):
        return None
    # ハンマーは打たれない=直後にターゲット選択は来ない。持ち越しキャッシュを捨てる。
    _hammer_ko_cache = None
    return action


def _make_hammer_ko_cache(obs: Observation, card_id: int, can_ko: bool) -> dict | None:
    """ハンマーPLAY決定時のKO判定を、直後のターゲット選択へ渡すためのキャッシュを作る。"""
    try:
        return {
            "card_id": card_id,
            "can_ko": bool(can_ko),
            "turn": getattr(obs.current, "turn", None),
            "your_index": obs.current.yourIndex,
            # `_select_action` が decision ごとに進めるカウンタ。ターゲット選択は
            # ハンマーPLAYの**次の** decision に必ず来るので、差が1でなければ古いとみなす。
            "select_seq": _selects_seen,
        }
    except Exception:  # noqa: BLE001 - キャッシュ作成の失敗が意思決定を止めてはならない
        return None


def _hammer_target_select_card_id(select: SelectData, hammer_ids) -> int | None:
    """この select が対象ハンマーの「剥がすエネを選ぶ select」ならそのカードIDを返す。

    実データ(Kaggle実ラダーのリプレイ、`kaggle_replays/replays/episode-92385834-replay.json`
    ほか。1120 の PLAY ログが載る行の observation.select を実測)で確認したスキーマ:

        select.type    = SelectType.ENERGY(4)
        select.context = SelectContext.DISCARD_ENERGY(30)
        select.effect  = Card(id=1120, playerIndex=<自分>, serial=…)  ← 効果元のハンマー
        select.option[i] = Option(type=OptionType.ENERGY(6),
                                  area=AreaType.ACTIVE(4) or AreaType.BENCH(5),
                                  index=<そのエリア内のindex>,
                                  playerIndex=<相手>, energyIndex=…, count=1)

    つまり「どのハンマーの選択か」は `select.effect.id` で直接分かり、「どの option が
    アクティブ狙いか」は `option.area` で分かる(cardId は None)。判定不能(effect が無い/
    type・context が違う)は None を返し、呼び出し側は介入しない(安全側)。
    """
    try:
        if select.type != SelectType.ENERGY or select.context != SelectContext.DISCARD_ENERGY:
            return None
        effect = select.effect
        effect_id = getattr(effect, "id", None) if effect is not None else None
        if effect_id is None or effect_id not in hammer_ids:
            return None
        return int(effect_id)
    except Exception:  # noqa: BLE001
        return None


def _hammer_redirect_can_ko(obs: Observation, card_id: int, hv_config: dict, config: dict | None) -> bool:
    """ターゲット選択の時点で「このターン相手アクティブをKOできるか」を返す。

    第一候補は直前のハンマーPLAY決定時のキャッシュ(`_hammer_ko_cache`)。同じ試合・同じ
    ターン・同じハンマー・**1つ前の decision** で作られたものだけを信用する。条件を満たさない
    (`hammer_veto.enabled=false` で redirect だけ有効な構成、lethal 経路で veto を通らずに
    ハンマーが打たれた場合など)ときは `ko_search` を張り直す(`time_limit_ms`、既定100ms)。
    例外・判定不能はすべて False(=介入しない、安全側)。

    **実測の注意(2026-08-14、og_v032 で計測)**: この張り直しは実際にはほぼ常に False になる。
    ターゲット選択の時点では、打ったハンマー自身が手札からもトラッシュからも見えない
    (効果解決中でどのゾーンにも現れない)ため、`build_dummy_search_state` の整合チェック
    「見えない札の枚数 == deckCount + 伏せサイド」が**ちょうど1枚ずれて** None を返し、
    `ko_search` は hidden_state を得られない(実測 11/11 で None、内訳も pool が常に +1)。
    つまり redirect を機能させるには `hammer_veto.enabled=true` かつ `card_ids` に対象ハンマー
    が入っていること(=PLAY 決定時にキャッシュが作られること)が実質の前提。張り直しは
    「無いよりまし」の安全弁として残してあるだけで、当てにしてはいけない。
    """
    cache = _hammer_ko_cache
    try:
        if (
            cache is not None
            and cache.get("card_id") == card_id
            and cache.get("your_index") == obs.current.yourIndex
            and cache.get("turn") == getattr(obs.current, "turn", None)
            and _selects_seen - int(cache.get("select_seq", -99)) == 1
        ):
            return bool(cache.get("can_ko", False))
    except Exception:  # noqa: BLE001
        pass
    try:
        from ptcg_ai.search import ko_search
        ko_cfg = hv_config.get("ko_search") or {}
        limit_ms = float(ko_cfg.get("time_limit_ms", hv_config.get("time_limit_ms", 100)))
        deadline = time.perf_counter() + limit_ms / 1000.0
        factory = _model_hidden_state_factory(obs, config)
        return bool(ko_search.can_ko_this_turn(obs, factory, ko_cfg, deadline))
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return False


def _try_hammer_redirect(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """ハンマーの**ターゲット**をアクティブ→ベンチへ振り替える。config-gated
    (`hammer_veto.redirect_target`)、キーが無ければ常に None=本番不変。

    実ラダー56試合のリプレイ監査(`kaggle_replays/_audit_hammer_replay_55464390.json`)で、
    クラッシュハンマー(1120)のアクティブ命中35件のうち16件が「同ターンに自分の攻撃でその
    相手をKO」する無駄撃ちで、**16件すべてアクティブ狙い**だった。倒す相手のエネを剥がしても
    どうせトラッシュに行くので効果がゼロ。しかも `_try_hammer_veto` の guard は「アクティブ
    以外にエネがあれば正当だろう」と見て veto を止めるため、この16件を素通ししていた
    (ベンチにエネはあるのに、実際に撃っていたのはアクティブ)。

    そこで「打つのをやめる」のではなく「**打つ先を変える**」で直す。ハンマーの PLAY と
    ターゲット選択は別 decision なので、PLAY 決定時のKO判定を持ち越して(`_hammer_ko_cache`)、
    直後のターゲット選択で:

      - このターンKOできる(=アクティブから剥がしても無駄)
      - かつ選択肢にアクティブ以外(ベンチ)の対象がある
      - かつ現在の選択が相手アクティブ狙い

    の3条件が揃ったときだけ、ベンチ対象(複数なら方策スコア最上位)に差し替える。
    どれか判定できなければ介入しない(既存の選択のまま)。

    前提: KO判定は PLAY 決定時のキャッシュ頼み(理由は `_hammer_redirect_can_ko` の実測メモ)。
    したがって `redirect_target` は `hammer_veto.enabled=true` かつ `card_ids` に対象ハンマー
    が入っている構成(例: `configs/abl_5_full_og_hammer.json`)と併せて使う。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    hv_config = (effective_config or {}).get("hammer_veto") or {}
    if not hv_config.get("redirect_target", False):
        return None
    select = obs.select
    if len(chosen_action) != 1 or select.maxCount != 1:
        return None
    hammer_ids = hv_config.get("card_ids") or _DEFAULT_HAMMER_VETO_CARD_IDS
    card_id = _hammer_target_select_card_id(select, hammer_ids)
    if card_id is None:
        return None  # 対象ハンマーのターゲット選択ではない=無関係
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None

    opp_index = 1 - obs.current.yourIndex
    chosen_option = select.option[idx]
    # 今の選択が「相手アクティブのエネ」でなければ介入不要(既にベンチ等を狙っている)。
    if getattr(chosen_option, "area", None) != AreaType.ACTIVE:
        return None
    if getattr(chosen_option, "playerIndex", None) != opp_index:
        return None

    # 振替先候補 = 相手ベンチ(=このターンKOしない相手)のエネ。area が読めない選択肢は
    # 「判定不能」として候補に入れない(安全側)。
    candidates = [
        i for i, opt in enumerate(select.option)
        if getattr(opt, "area", None) == AreaType.BENCH and getattr(opt, "playerIndex", None) == opp_index
    ]
    if not candidates:
        return None  # アクティブしか対象が無い=振替先が無い

    if not _hammer_redirect_can_ko(obs, card_id, hv_config, config):
        return None  # KOできない=アクティブから剥がすのは無駄ではない→干渉しない

    # 候補が複数あれば方策スコア上位。スコアが取れない場合でも「アクティブに撃つ」よりは
    # ましなので、先頭候補で振り替える(介入自体は行う)。
    best = candidates[0]
    try:
        model = _get_model(config)
        factory = _model_hidden_state_factory(obs, config)
        deadline = time.perf_counter() + _MODEL_TIME_BUDGET_MS / 1000.0
        scores = model.score_options(obs, factory, deadline)
        if scores and len(scores) == len(select.option):
            best = max(candidates, key=lambda i: scores[i])
    except Exception:  # noqa: BLE001 - スコアリング失敗は先頭候補にフォールバック
        pass
    action = [best]
    return action if _is_valid_action(action, select) else None


_SACRED_ASH_ID = 1129


def _count_trash_pokemon(state, me: int) -> int:
    """自分のトラッシュ(discard)にあるポケモンの枚数(=せいなるはいで山に戻せる枚数の上限根拠)。"""
    try:
        from cg.api import CardType
        from ptcg_ai.shared import card_cache
        cnt = 0
        for c in (state.players[me].discard or []):
            if c is None:
                continue
            try:
                if card_cache.get_card(c.id).cardType == CardType.POKEMON:
                    cnt += 1
            except Exception:  # noqa: BLE001 - 未知IDは無視
                continue
        return cnt
    except Exception:  # noqa: BLE001
        return 0


_DUDUNSPARCE_ID = 66  # ノココッチ。特性「にげあしドロー」=3枚引いた後このポケモンを山に戻す。


def _try_loop(obs: Observation, config: dict | None = None) -> list[int] | None:
    """ノコッチループの強制(config-gated `loop`, 既定OFF=本番不変)。極低deck限定の決定的ルール。

    山札が危険域(`deck_threshold` 以下)で、ノココッチ(id66)の**にげあしドロー(特性)**が使えるなら
    それを選ぶ。にげあしドローは「3枚引く→ノココッチ自身を山に戻す」ので、山が1〜2枚でも必ず
    山>=1 を保てる(自滅deckoutを防ぎ、ループで延命=相手が先に山切れするのを待てる)。MLスペシャリスト
    が"遅い差し込みのoff-distribution"で失敗したのに対し、極低deckのループ play はほぼ決定的なので
    ルールが確実(ユーザー案)。lethal(勝ち)の次に置く=勝ち>延命。
    """
    if obs.current is None or obs.select is None:
        return None
    eff = config if config is not None else _get_config()
    lp = (eff or {}).get("loop") or {}
    if not lp.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or not select.option:
        return None
    me = obs.current.yourIndex
    if int(obs.current.players[me].deckCount or 0) > int(lp.get("deck_threshold", 3)):
        return None  # 危険域でない
    # 対面限定ゲート(`require_route`): 専用ルートが立っている decision のみループ許可。グローバル
    # loop は偽fieldで-52pt悪化したが、実ラダーの deckout 敗は crustle に集中(全敗がデッキ切れ)。
    # crustle config は routing.specialists={crustle:...} のみ=ルート非None⟺crustle検出。
    # _current_route_path は _select_action 冒頭でこの decision 用にセット済み。
    if lp.get("require_route") and _current_route_path is None:
        return None
    from ptcg_ai.learning import encoder as _enc
    for i, opt in enumerate(select.option):
        if opt.type == OptionType.ABILITY and _enc._resolve_card_id(opt, obs.current) == _DUDUNSPARCE_ID:
            action = [i]
            return action if _is_valid_action(action, select) else None
    return None


def _try_keep3(obs: Observation, config: dict | None = None) -> list[int] | None:
    """クセロシキ等「手札を N 枚捨てて 3 枚に減らす」discard select で、keep-set 評価に基づき捨て札を
    選ぶ。config-gated(`keep_eval`)、既定OFF=本番不変。keep_eval.best_keep_discard_indices が組合せ
    (進化シナジー/重複サポート等)込みで最良 keep を選び、捨てる index を返す。
    """
    if obs.current is None or obs.select is None:
        return None
    eff = config if config is not None else _get_config()
    ke = (eff or {}).get("keep_eval") or {}
    if not ke.get("enabled", False):
        return None
    select = obs.select
    from cg.api import SelectContext
    if getattr(select, "context", None) != SelectContext.DISCARD or not select.option:
        return None
    if select.minCount != select.maxCount or select.maxCount < 1:
        return None  # 「N枚ちょうど捨てる」形(クセロシキ等)のみ対象
    from ptcg_ai.learning import encoder as _enc
    option_ids = [_enc._resolve_card_id(o, obs.current) for o in select.option]
    n = len(option_ids)
    discard_count = select.maxCount
    keep_count = n - discard_count
    if keep_count <= 0:
        return None

    me = obs.current.yourIndex
    st = obs.current
    deck = int(st.players[me].deckCount or 0)
    can_ko = False
    try:
        from ptcg_ai.search import ko_search
        can_ko = ko_search.can_ko_this_turn(
            obs, _model_hidden_state_factory(obs, config), {}, time.perf_counter() + 0.08)
    except Exception:  # noqa: BLE001
        can_ko = False
    energy_short = False
    try:
        a = st.players[me].active
        a0 = a[0] if a else None
        energy_short = a0 is not None and not (getattr(a0, "energyCards", None) or [])
    except Exception:  # noqa: BLE001
        pass
    alakazam_in_play = False
    try:
        pool = (list(st.players[me].active or []) + list(st.players[me].bench or []))
        alakazam_in_play = any(pk is not None and getattr(pk, "id", None) == 743 for pk in pool)
    except Exception:  # noqa: BLE001
        pass
    ctx = {
        "deck_low": deck <= int(ke.get("deck_low_threshold", 8)),
        "energy_short": energy_short,
        "opp_hand_big": int(st.players[1 - me].handCount or 0) >= 5,
        "can_ko": can_ko,
        "alakazam_in_play": alakazam_in_play,
    }
    try:
        from ptcg_ai.board_evaluation import keep_eval
        discard_idx = keep_eval.best_keep_discard_indices(option_ids, keep_count, st, ctx)
    except Exception:  # noqa: BLE001
        return None
    if discard_idx is None or len(discard_idx) != discard_count:
        return None
    action = sorted(discard_idx)
    return action if _is_valid_action(action, select) else None


def _try_survival(obs: Observation, chosen_action: list[int], config: dict | None = None) -> list[int] | None:
    """せいなるはい(1129, 唯一の山回復)の"早撃ち"防止veto。config-gated(`survival`)、既定OFF=本番不変。

    診断(vs crustle, deckout敗14件): 多く(11件)は せいなるはい を「トラッシュのポケモンがまだ
    少ない/山にまだ余裕がある」うちに早撃ちして、いざ山切れという時には既に消費済み、が原因。
    せいなるはいは「トラッシュのポケモンを最大5枚 山にもどす」ので、トラッシュのポケモンが少ない
    うちに撃つと戻せる枚数が少なく無駄。→ せいなるはいを選んだ手が **トラッシュのポケモン
    < min_trash_pokemon かつ 山 > deck_critical**(今すぐ回復が要るわけでない)なら、温存して他の
    最善手(方策スコア最大)に差し替える。ユーザー指摘「トラッシュのポケモン数が特徴に無いので
    早撃ちする」への対応(手書き相性でなく"回復量が最大化できるまで待つ"という盤面事実ベース)。
    """
    if obs.current is None or obs.select is None:
        return None
    eff = config if config is not None else _get_config()
    sv = (eff or {}).get("survival") or {}
    if not sv.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    from ptcg_ai.learning import encoder as _enc
    if _enc._resolve_card_id(select.option[idx], obs.current) != _SACRED_ASH_ID:
        return None  # せいなるはい以外は対象外

    me = obs.current.yourIndex
    deck = obs.current.players[me].deckCount
    if deck <= int(sv.get("sacred_ash_deck_critical", 6)):
        return None  # 山が危険域=今すぐ使ってよい(強制回復)→干渉しない
    if _count_trash_pokemon(obs.current, me) >= int(sv.get("sacred_ash_min_trash_pokemon", 5)):
        return None  # トラッシュに十分ポケモン=最大回復できる=使ってよい

    # 早撃ち(トラッシュ薄い&山にまだ余裕)→温存。せいなるはい以外で方策スコア最大の手へ差替。
    try:
        model = _get_model(config)
        scores = model.score_options(
            obs, _model_hidden_state_factory(obs, config),
            time.perf_counter() + _MODEL_TIME_BUDGET_MS / 1000,
        )
    except Exception:  # noqa: BLE001 - veto の失敗が意思決定を止めてはならない
        return None
    if not scores or len(scores) != len(select.option):
        return None
    best_alt = best_score = None
    for i in range(len(scores)):
        if _enc._resolve_card_id(select.option[i], obs.current) == _SACRED_ASH_ID:
            continue
        if best_score is None or scores[i] > best_score:
            best_score, best_alt = scores[i], i
    if best_alt is None:
        return None
    action = [best_alt]
    return action if _is_valid_action(action, select) else None


def _apply_action_vetoes(obs: Observation, action: list[int], config: dict | None = None) -> list[int]:
    """事後veto(改造ハンマー浪費→ターゲット振替→自滅deckout)を順に適用。すべて config-gated
    で既定OFF=本番不変。

    差替が起きたら次のvetoも差替後の手に適用する。abl_5_full ではキーが無く各vetoは即Noneを
    返すので action は不変(本番挙動は変わらない)。`_try_hammer_veto`(MAIN の PLAY 選択)と
    `_try_hammer_redirect`(ENERGY/DISCARD_ENERGY のターゲット選択)は対象 select が排他なので、
    同じ decision で両方が発火することはない。
    """
    for _veto in (_try_hammer_veto, _try_hammer_redirect, _try_survival):
        alt = _veto(obs, action, config=config)
        if alt is not None:
            action = alt
    return action


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

    `time_budget["mode"] == "expensive_roots"`(時間予算v2, 任意キー)のときだけ分母を変える:
    「(soft_stop_ms - 経過) ÷ 残りの重い探索回数(assumed_expensive_roots - 実行済み)」。
    実測で総select数(約72.5)と重い探索の回数(約29)が大きく違い、旧式(全selectで割る)は
    540秒プールの2割弱しか使えていなかったことへの是正。`mode` キーが無ければ従来式のまま。
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
            return min_ms   # ハードライン(total_ms)は v2 でも従来どおり
        if budget.get("mode") == "expensive_roots":
            soft_stop_ms = float(budget.get("soft_stop_ms", total_ms))
            assumed_roots = int(budget.get("assumed_expensive_roots", 60))
            remaining_soft_ms = soft_stop_ms - elapsed_ms
            if remaining_soft_ms <= 0:
                return min_ms
            remaining_roots = max(1, assumed_roots - _expensive_roots_seen)
            return max(min_ms, min(max_ms, remaining_soft_ms / remaining_roots))
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

    `time_budget["mode"] == "expensive_roots"`(時間予算v2)のときだけ、以下を追加で行う:
      (a) 経過が `soft_stop_ms` を超えた/ブレーカ発動済みなら探索を始めない(None=既存経路へ)
      (b) `pipeline.search` が**実際に決定化評価(Step3)へ入った**回数を `_expensive_roots_seen`
          に数える(v2 の分母)。`context["on_expensive_root"]` コールバック経由で通知を受ける。
          呼ぶ前に数えると、適用外 select(非MAIN/複数選択)や top1 shortcut で即 return する
          安い呼び出しまで分母に入り(実測94回/試合 vs 実際の探索29回)、1手予算が過小になる。
      (c) 実測時間が「割当 + `overshoot_breaker_ms`」を超えたらブレーカを立て、その試合は
          以降 pipeline を使わない(1試合600秒プールを1手の暴走で溶かさないための保険)
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    pipeline_config = (effective_config or {}).get("pipeline") or {}
    if not pipeline_config.get("enabled", False):
        return None

    global _expensive_roots_seen, _pipeline_breaker_tripped
    budget = pipeline_config.get("time_budget") or {}
    expensive_mode = budget.get("mode") == "expensive_roots"
    if expensive_mode:
        if _pipeline_breaker_tripped:
            return None
        try:
            if _match_start_perf is not None and "soft_stop_ms" in budget:
                elapsed_ms = (time.perf_counter() - _match_start_perf) * 1000.0
                if elapsed_ms >= float(budget["soft_stop_ms"]):
                    return None     # 残り時間は探索以外(select 処理)に残す
        except Exception:   # noqa: BLE001 - 予算計算の失敗が意思決定を止めてはならない
            pass

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
        search_context = {
            "observation": obs,
            "config": run_config,
            "hidden_state_factory": factory,
            "model_hidden_state_factory": _model_hidden_state_factory(obs, config),
            "policy_model": model,
        }
        if expensive_mode:
            # 「重い探索」を実際に始めた回数(v2 の分母)。pipeline 側が決定化評価へ入る
            # 直前に1回だけ呼ぶ(安い即 return は数えない)。
            def _count_expensive_root() -> None:
                global _expensive_roots_seen
                _expensive_roots_seen += 1

            search_context["on_expensive_root"] = _count_expensive_root
        search_start = time.perf_counter()
        action = pipeline.search(obs.current, obs.select.option, search_context)
        if expensive_mode:
            spent_ms = (time.perf_counter() - search_start) * 1000.0
            overshoot_ms = float(budget.get("overshoot_breaker_ms", 250))
            if spent_ms > float(run_config["time_limit_ms"]) + overshoot_ms:
                _pipeline_breaker_tripped = True
    except Exception:
        return None

    if action is not None and _is_valid_action(action, obs.select):
        return action
    return None


# vs ミル(crustle)自滅対策の回復札(step: vs-crustle 縦串P0)。
# 山札切れを直接防ぐ = deckCount を増やすカードは Sacred Ash(1129, Item)だけ:
#   「トラッシュのポケモン最大5枚を山札に戻す」= deckCount 直接 +5、かつ Item なので
#   サポート枠(ドロー)と競合しない。
# 診断(_diag_deckout_decision_trace)は Night Stretcher(1097)/Lana's Aid(1184)も "sustain" に
# 束ねていたが、両者はトラッシュ→"手札"回収で deckCount を増やさず山切れを防がない。特に
# Lana's Aid は Supporter でドロー枠を消費する(=テンポ損)。実際、3枚すべてにバイアスした
# 初回スモーク(n=8)は治療 12.5% vs baseline 50%・avg_turns 21->24 と逆効果で、原因は
# 手札回収札/サポートを温存名目で撃って自分のクロックを遅らせたこと。よって Sacred Ash 単独に
# 絞る。config["deck_sustain"]["recovery_card_ids"] で上書き可能(ablation 用)。
_DECK_SUSTAIN_RECOVERY_IDS = (1129,)


def _try_deck_sustain(obs: Observation, config: dict | None = None) -> list[int] | None:
    """残り山札が逼迫した MAIN(単一選択)で、山札切れ自滅を避けるよう軽く介入する。2挙動:

    (1) 回収バイアス: 山札復帰札(Sacred Ash=1129, ``recovery_card_ids``)が『ポリシー的に妥当な
        候補』(生スコア上位 ``max_rank`` 位以内)なら、それを選ぶ。deckCount を直接増やして
        deckout を遅らせるので先行/後攻不問。
    (2) ループ抑制(戦略依存, ``suppress_deckburn_when_ahead``): サイドで先行/五分のときだけ、
        ポリシー top1 が Dudunsparce「Run Away Draw」(ABILITY, ``deckburn_ability_ids``=66。
        3枚引いて更に山を焼くループ手)なら、最善の非 deck-burn 手に振る。後攻/劣勢では抑制
        しない — このデッキ(Alakazam「Powerful Hand」)は手札枚数=打点で、劣勢時にドローを
        止めると打点も札も失うため(『引かないと目当てが来ない』, ユーザー指摘)。

    どちらも該当しなければ None を返し、既存フロー(pipeline / baseline)に委ねる。ソフトさは
    (1)は ``max_rank`` ゲート、(2)は「top1 を次善手に振るだけ(非 deck-burn 手が無ければ不介入)」
    で担保する(安全側の低recall・高精度介入。attack_hybrid と同じ思想)。

    本番 config(``deck_sustain`` キー無し)では常に None を返し本番挙動は不変。``_try_lethal``
    の直後・``_try_pipeline`` の前に呼ばれる: 確定リーサル(今勝てる) > 山札温存 > 一般探索。
    pipeline のロールアウトも同じ模倣ポリシーを使い同じ盲点を持つため、探索側では直らない。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    ds_config = (effective_config or {}).get("deck_sustain") or {}
    if not ds_config.get("enabled", False):
        return None

    select = obs.select
    # 適用範囲: 回復札は自ターンの MAIN で手札から撃つ単一選択(pipeline と同じスコープ)。
    if select.type != SelectType.MAIN or select.maxCount != 1 or not select.option:
        return None

    state = obs.current
    me = state.yourIndex
    try:
        my_deck_count = state.players[me].deckCount
    except (IndexError, AttributeError, TypeError):
        return None
    if my_deck_count > int(ds_config.get("deck_count_threshold", 8)):
        return None

    recovery_ids = set(ds_config.get("recovery_card_ids") or _DECK_SUSTAIN_RECOVERY_IDS)
    recovery_idxs = [
        i for i, opt in enumerate(select.option)
        if opt.type == OptionType.PLAY and opt.cardId in recovery_ids
    ]
    # (2) ループ抑制(戦略依存)の準備。山札逼迫 かつ サイドで先行/五分のときだけ、更に山を
    #     焼く Dudunsparce「Run Away Draw」(ABILITY, cardId 66)を止め、次善手に振る。後攻/劣勢
    #     では抑制しない(札と打点が要る=『引かないと目当てが来ない』を尊重)。
    suppress_on = bool(ds_config.get("suppress_deckburn_when_ahead", False))
    burn_ids = set(ds_config.get("deckburn_ability_ids") or (66,))

    def _is_burn(i: int) -> bool:
        opt = select.option[i]
        return opt.type == OptionType.ABILITY and opt.cardId in burn_ids

    burn_present = suppress_on and any(_is_burn(i) for i in range(len(select.option)))
    if burn_present:
        try:
            # サイド残が少ないほど勝ちに近い。my <= opp = 先行/五分 のときだけ抑制する。
            ahead = len(state.players[me].prize) <= len(state.players[1 - me].prize)
        except (IndexError, AttributeError, TypeError):
            ahead = False
        burn_present = ahead

    if not recovery_idxs and not burn_present:
        return None  # やることが無い → 既存フローに委ねる(スコアリングもしない)。

    model = _get_model(config)
    model_factory = _model_hidden_state_factory(obs, config)
    model_deadline = time.perf_counter() + _MODEL_TIME_BUDGET_MS / 1000
    try:
        scores = model.score_options(obs, model_factory, model_deadline)
    except Exception:
        return None
    if not scores or len(scores) != len(select.option):
        return None
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # (1) 回収バイアス優先: 山札復帰札(Sacred Ash)が『妥当な候補』(上位 max_rank 位以内)なら
    #     それを拾う。deckout を直接遅らせるので先行/後攻不問。
    if recovery_idxs:
        best_recovery = max(recovery_idxs, key=lambda i: scores[i])
        if ranked.index(best_recovery) < int(ds_config.get("max_rank", 4)):
            action = [best_recovery]
            if _is_valid_action(action, select):
                return action

    # (2) ループ抑制: ポリシー top1 が deck-burn(Run Away Draw)なら、最善の非 deck-burn 手へ
    #     振る(ソフト=次善手に委ねるだけ。非 deck-burn 手が無ければ触らない)。
    if burn_present and _is_burn(ranked[0]):
        non_burn = [i for i in ranked if not _is_burn(i)]
        if non_burn:
            action = [non_burn[0]]
            if _is_valid_action(action, select):
                return action

    return None


# --- (ii) 戦略残差NN (docs/plans/strategy-residual-nn-experiment.md)。config-gated, 既定OFF=本番不変。---
# 凍結した模倣 score に option 別 bias を足して argmax。self-play RL が学習した重みを config で注入する。
_strategy_residual_default: StrategyResidual | None = None
_strategy_residual_by_path: dict[str, StrategyResidual] = {}


def _get_strategy_residual(config: dict | None) -> StrategyResidual:
    """``config["strategy_residual"]["weights_path"]`` の残差を返す(パスごとにキャッシュ)。
    パス無しなら空の残差(``is_ready=False`` = bias 全 0 = 無効)。"""
    sr = (config or {}).get("strategy_residual") or {}
    path = sr.get("weights_path")
    if not path:
        global _strategy_residual_default
        if _strategy_residual_default is None:
            _strategy_residual_default = StrategyResidual(None)
        return _strategy_residual_default
    res = _strategy_residual_by_path.get(path)
    if res is None:
        res = StrategyResidual(path)
        _strategy_residual_by_path[path] = res
    return res


def _try_strategy_residual(obs: Observation, config: dict | None = None) -> list[int] | None:
    """凍結模倣の score に学習残差 bias を足し、結論が変わるならその手を返す(config-gated)。

    最終行動 = ``argmax(score_imit + alpha * bias)``。bias が模倣の top1 を覆さないなら None を返し
    既存フローに委ねる(= 残差が無効/未学習なら本番挙動不変)。適用範囲は単一選択の MAIN。
    ``strategy_residual`` キーが無い本番 config では常に None。``_try_deck_sustain`` の後・
    ``_try_pipeline`` の前に呼ばれる(学習した②③戦略 > 一般探索)。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    sr_config = (effective_config or {}).get("strategy_residual") or {}
    if not sr_config.get("enabled", False):
        return None

    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or not select.option:
        return None

    model = _get_model(config)
    model_factory = _model_hidden_state_factory(obs, config)
    model_deadline = time.perf_counter() + _MODEL_TIME_BUDGET_MS / 1000
    try:
        scores = model.score_options(obs, model_factory, model_deadline)
    except Exception:
        return None
    if not scores or len(scores) != len(select.option):
        return None

    residual = _get_strategy_residual(effective_config)
    bias = residual.bias_options(obs, select)
    if len(bias) != len(scores):
        return None

    alpha = float(sr_config.get("alpha", 1.0))
    biased = [scores[i] + alpha * bias[i] for i in range(len(scores))]
    best = max(range(len(biased)), key=lambda i: biased[i])
    base = max(range(len(scores)), key=lambda i: scores[i])
    if best == base:
        return None  # 残差が模倣の結論を変えない → 委ねる(挙動不変)。

    action = [best]
    return action if _is_valid_action(action, select) else None


def _select_action(obs: Observation, config: dict | None = None) -> list[int]:
    select = obs.select

    # pipeline の動的時間予算用に、意思決定を要する選択のたびにカウンタを進める
    # (pipeline 無効時も無害)。
    global _selects_seen
    _selects_seen += 1

    # 動的ルーティング: この decision で使う専用重みを決めて global に置く(以降の全 _get_model が
    # 参照)。routing キー無しの config では常に None=base のままなので本番不変(config-gated)。
    global _current_route_path
    _current_route_path = _compute_route(obs, config if config is not None else _get_config())

    lethal_action = _try_lethal(obs, config=config)
    if lethal_action is not None:
        return lethal_action

    # 確定リーサル(今勝てる)の次に、極低deckでのノコッチループ強制(config-gated `loop`, 既定OFF=
    # 本番不変)。にげあしドローで山>=1を保ち延命=相手(ループ無し)が先に山切れするのを待つ。
    loop_action = _try_loop(obs, config=config)
    if loop_action is not None:
        return loop_action

    # クセロシキ等の手札 discard select を keep-set 評価で処理(config-gated `keep_eval`, 既定OFF=
    # 本番不変)。非MAIN select なので lethal/loop/pipeline は素通り。base方策より前に置く。
    keep_action = _try_keep3(obs, config=config)
    if keep_action is not None:
        return keep_action

    # 確定リーサル(今勝てる)の次に、山札切れ自滅の回避を優先する。既定 config(deck_sustain
    # キー無し)では常に None なので本番挙動は不変。pipeline より前に置くのは、pipeline の
    # ロールアウトも同じ模倣ポリシーで同じ盲点を持ち探索では直らないため(vs-crustle 縦串P0)。
    sustain_action = _try_deck_sustain(obs, config=config)
    if sustain_action is not None:
        return sustain_action

    # 学習した②③戦略残差(config-gated, 既定OFF=本番不変)。凍結模倣を bias するだけで、
    # 結論が変わらない/未学習なら None を返す。一般探索より前に置く(学習戦略 > 探索)。
    residual_action = _try_strategy_residual(obs, config=config)
    if residual_action is not None:
        return residual_action

    pipeline_action = _try_pipeline(obs, config=config)
    if pipeline_action is not None:
        # abl_5_full では MAIN 決定は pipeline が返すので、事後veto群(ハンマー浪費/自滅deckout)は
        # pipeline の出力にも掛ける(config-gated, 既定OFF=本番不変)。
        return _apply_action_vetoes(obs, pipeline_action, config=config)

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
    final_action = planned_action if planned_action is not None else baseline_action
    # 事後veto群(ハンマー浪費/自滅deckout, config-gated 既定OFF=本番不変)。base/attack_plan経路にも掛ける。
    return _apply_action_vetoes(obs, final_action, config=config)


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

