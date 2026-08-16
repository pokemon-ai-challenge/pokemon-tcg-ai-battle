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
import random
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
# ボスの指令の対象ゲート(config-gated `boss_lethal_gate`)用。ハンマーと同じく「PLAY する
# decision」と「引きずり出す相手を選ぶ decision」が分かれるため、PLAY 決定時に計算した
# 対象別のKO可否を次の decision まで持ち越す。試合を跨いで残ると誤判定になるので
# `agent()` の試合開始パスでクリアする。キーが無い config では誰も読まない。
_boss_target_cache: dict | None = None


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
        global _hammer_ko_cache, _boss_target_cache
        _match_start_perf = time.perf_counter()
        _selects_seen = 0
        # 時間予算v2 のカウンタ/ブレーカも試合境界でリセット(mode 未指定なら未参照のまま)。
        _expensive_roots_seen = 0
        _pipeline_breaker_tripped = False
        # ハンマーのターゲット・リダイレクト用のKO判定キャッシュも試合境界で捨てる
        # (前の試合の判定が次の試合の最初のハンマーに漏れないようにする)。
        _hammer_ko_cache = None
        # ボスの対象ゲート用の持ち越しキャッシュも同じ理由で試合境界で捨てる。
        _boss_target_cache = None
        # リーサル探索の専用RNG(config-gated `lethal_search.isolated_rng`)を試合開始時に
        # 決定論的にシードし直す。目的は「探索の実行量(見つかった/見つからない、
        # verify_shuffles の回数)が変わってもグローバル random の消費量が変わらない」こと
        # (paired seed A/B の交絡除去、乱数消費テストは test_ml_policy_agent.py 参照)。
        # キー無し(既定)では何もしない = グローバル random の消費量は完全に不変。
        # `isolated_rng_seed` を明示すればグローバル random にも一切触れない(試合を跨いで
        # 固定シード=再現性テスト用)。未指定ならグローバル random から1回だけ引く
        # (=試合単位では再現的、グローバル列はこの1回だけ進む)。
        lethal_cfg = (effective_config or {}).get("lethal_search") or {}
        if lethal_cfg.get("isolated_rng"):
            seed = lethal_cfg.get("isolated_rng_seed")
            if seed is None:
                seed = random.getrandbits(64)
            lethal_simple.reset_rng(seed)
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


# ローカル評価harness専用の「自分の60枚」上書き(player_index -> card_ids)。
# `match_context._own_deck_override`(estimated 経路用)の dummy 経路版。
#
# 本番(Kaggle)は 1プロセス1エージェントで、自分のデッキは常に `deck.csv` なので
# この dict は空のまま = `_get_deck_for()` は `_get_deck()` に等しく挙動は完全に不変。
#
# 一方ローカルの `run_match.play_match` / `measurement.runner.play_game` は
# **1プロセスで両陣営の ml_policy を動かす**が、デッキは `battle_start(deck0, deck1)` で
# エンジンへ直接渡されるため、`_get_deck()`(=CWD の deck.csv)と実盤面が食い違う。
# その結果 `build_dummy_search_state` は「観測に見えるカードがデッキに無い」で **必ず None**
# を返し、dummy 経路の隠れ状態が作れない = `_model_hidden_state_factory` に依存する
# `ko_search` 系ガード(briar_gate / low_deck_draw_brake / boss_lethal_gate /
# ability_draw_brake / terastal_rotation / hammer_veto)と `lethal_search`(既定 dummy)が
# **その陣営で一度も発火しない**。`match_context.set_own_deck_override` は estimated 経路
# しか救わないため、dummy 経路にも同じ宣言口を用意する。
_own_deck_override: dict[int, list[int]] = {}


def set_own_deck_override(player_index: int, card_ids: list[int] | None) -> None:
    """``player_index`` の「自分の60枚」を明示する(ローカル評価harness用)。

    ``None`` を渡すと解除。production からは呼ばれない(呼ばなければ従来どおり
    ``_get_deck()`` = ``deck.csv`` を読む)。``match_context.set_own_deck_override`` と
    対になる宣言口で、harness は両方を試合開始時に設定する。
    """
    if card_ids is None:
        _own_deck_override.pop(player_index, None)
    else:
        _own_deck_override[player_index] = list(card_ids)


def clear_own_deck_override() -> None:
    """``set_own_deck_override`` の宣言を全て解除する(harness/テストの後始末用)。"""
    _own_deck_override.clear()


def _get_deck_for(obs: Observation | None) -> list[int]:
    """この obs を受け取っている側の「自分の60枚」を返す。

    harness が ``set_own_deck_override`` で宣言していればその陣営の60枚を、無ければ
    従来どおり ``_get_deck()``(=``deck.csv``)を返す(本番はこちら)。
    ``obs.current`` が無い/`yourIndex` が取れない場合も安全側で ``_get_deck()``。
    戻り値は ``_get_deck()`` と同じく**共有リストのまま**返す(呼び出し側は読むだけ。
    毎 decision で60要素のコピーを作らないため)。
    """
    if _own_deck_override:
        try:
            player_index = obs.current.yourIndex  # type: ignore[union-attr]
        except AttributeError:
            return _get_deck()
        declared = _own_deck_override.get(player_index)
        if declared is not None:
            return declared
    return _get_deck()


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
        # config-gated: `isolated_rng` が true なら、リーサル探索専用の random.Random
        # (lethal_simple がインスタンスを保持。試合開始時に agent() がシード)を渡す。
        # 既定(キー無し/false)は rng=None のままで、build_dummy_search_state /
        # to_search_begin_kwargs はグローバル random にフォールバックするため既存挙動と
        # バイト単位で同一(呼び出しモジュールが lethal_simple 以外でもこの隔離は有効)。
        rng = lethal_simple.get_rng() if lethal_config.get("isolated_rng") else None
        if hidden_state_source == "estimated":
            # 実推定(hidden_information.match_context、agent()冒頭で毎ターン更新済み)。
            factory = lambda: search_adapter.to_search_begin_kwargs(
                match_context.get_own_state(obs.current.yourIndex),
                match_context.get_opponent_state(obs.current.yourIndex),
                obs,
                rng=rng,
            )
        else:
            # ダミースタブ(既定、既存configとの後方互換)。
            full_deck = _get_deck_for(obs)
            factory = lambda: build_dummy_search_state(obs, full_deck, rng=rng)
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
            full_deck = _get_deck_for(obs)
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


# --------------------------------------------------------------------------------------
# 手札連動ダメージへの生存ガード / ブライアの空撃ちゲート
# (og_r7 の実ラダー負け分析 Fix-B / Fix-C。どちらも config-gated で既定OFF=本番不変)
# --------------------------------------------------------------------------------------

# 事後veto群の発火カウンタ(`search/lethal_simple.py` 等と同じ流儀のモジュールグローバル)。
# 意思決定には使わない純粋な計測用で、ローカル計測スクリプトから読む。
_GUARD_STATS_ZERO = {
    "hand_damage_guard_fired": 0,      # ガードが「即死する」と判定した回数
    "hand_damage_guard_to_judge": 0,   # うちジャッジマンへ差し替えた回数
    "hand_damage_guard_to_other": 0,   # うち「手札を増やさない手」へ差し替えた回数
    "hand_damage_guard_to_boss": 0,    # うち(to_other の内訳)ボスの指令へ差し替えた回数(I3計測用)
    "hand_damage_guard_no_substitute": 0,  # I3: 代替(ジャッジ/ボス)不在で介入を見送った回数
    "briar_gate_fired": 0,             # KO不能ターンのブライアを差し替えた回数
    # r9 Fix-D: 山札僅少ブレーキ(`low_deck_draw_brake`)
    "low_deck_draw_brake_fired": 0,        # 山札を減らすドローサポートを差し替えた回数
    "low_deck_draw_brake_skipped_ko": 0,   # 今ターンKOできるので通した回数(=ドローして勝ちに行く)
    "low_deck_draw_brake_inconclusive": 0,  # KO探索が完走せず介入を見送った回数
    # r9 Fix-E: ボスのKOゲート(`boss_lethal_gate`)
    "boss_lethal_gate_fired": 0,        # KO可能な対象が皆無でボスPLAYを差し替えた回数
    "boss_lethal_gate_redirected": 0,   # 対象をKO可能な相手へ振り替えた回数
    "boss_lethal_gate_denial_kept": 0,  # エネ除去例外(最多エネ)で許可した回数
    "boss_lethal_gate_inconclusive": 0,  # 対象評価が完走せず介入を見送った回数
}
_GUARD_STATS = dict(_GUARD_STATS_ZERO)


def get_guard_stats() -> dict:
    return dict(_GUARD_STATS)


def reset_guard_stats() -> None:
    _GUARD_STATS.update(_GUARD_STATS_ZERO)


_JUDGE_ID = 1213      # ジャッジマン(サポート): 両者が手札を山に戻して4枚引く
_LILLIE_ID = 1227     # リーリエの決心(サポート): 手札を戻して6枚引く(自サイド6枚なら8枚)
_HARLEQUIN_ID = 1223  # クラウン(サポート): 手札を戻してコイン次第で自分は5枚 or 3枚引く
_BRIAR_ID = 1201      # ブライア(サポート): このターンKOしたとき、サイドをもう1枚多く取る
_BOSS_ORDER_ID = 1182  # ボスの指令(サポート): 相手のベンチをアクティブへ引きずり出す。手札は
                        # 増減しないので `hand_damage_guard` の「代替サポート」候補になる(I3)

# ジャッジマン使用後の自分の手札枚数(カードテキストで確定)。
_JUDGE_HAND_SIZE = 4

# 「手札を増やすサポート」使用後の自分の手札枚数の見積り。
# リーリエは自サイド6枚のとき8枚引くので `_refill_hand_size` で個別に補正する。
# クラウンはコイン依存(5 or 3)なので**多い方**=安全側(=被弾が大きい側)を採る。
_HAND_REFILL_SIZES = {_LILLIE_ID: 6, _HARLEQUIN_ID: 5}


def _opponent_field_card_ids(state, me: int) -> set[int]:
    """相手の場(アクティブ+ベンチ)に見えているポケモンの CardData id 集合。判定不能は空集合。"""
    seen: set[int] = set()
    try:
        opp = state.players[1 - me]
        for pk in (list(opp.active or []) + list(opp.bench or [])):
            card_id = getattr(pk, "id", None) if pk is not None else None
            if card_id is not None:
                seen.add(int(card_id))
    except Exception:  # noqa: BLE001 - 収集の失敗が意思決定を止めてはならない
        return set()
    return seen


def _opponent_field_imminent(state, me: int, card_id: int) -> bool:
    """`card_id` の相手ポケモンのうち、**アクティブに居る** か **エネルギーが1個以上付いている**
    個体が1体でも居るか(``hand_damage_guard`` の ``imminent: "active_or_energized"`` 用、I2)。

    Iteration 1 は「相手の場に見えている」だけで発火していたが、mega_froslass_ex 対面で
    -6.2pt(n=400)の退行を出した(og_r7 の実ラダー分析)。ベンチ・エネ0の個体は
    「前に出す→エネ付与→攻撃」を経てようやく打点が立つ=このターンではなく相手の**次の**手番の
    さらに先まで猶予がある可能性がある一方、アクティブ or エネ付きの個体は最短1手番で
    打点が立ちうる。判定不能(例外/フィールド欠損)は False(=発火させない側)。
    """
    try:
        opp = state.players[1 - me]
        for pk in (opp.active or []):
            if pk is not None and getattr(pk, "id", None) == card_id:
                return True
        for pk in (opp.bench or []):
            if pk is not None and getattr(pk, "id", None) == card_id \
                    and len(getattr(pk, "energies", None) or []) > 0:
                return True
    except Exception:  # noqa: BLE001 - 収集の失敗が意思決定を止めてはならない
        return False
    return False


def _effective_active_hp(state, me: int, guard_config: dict) -> int | None:
    """自分のバトルポケモンの実効HP。読めなければ None。

    実効HP = 残りHP + ``effective_hp_bonus``(config、既定0)。スタジアム/どうぐによる
    ダメージ軽減はカード効果の一般解が無く、ここでは解かない。既定0は「素の残りHP」=
    軽減を数えない側で、軽減がある盤面ではガードが**多めに**発火する(生存側に倒す)。
    既知の対面で補正したい場合だけ config で注入する(ablation 用の逃がし口)。
    """
    try:
        active = state.players[me].active or []
        mon = active[0] if active else None
        hp = getattr(mon, "hp", None) if mon is not None else None
        if not isinstance(hp, int):
            return None
        return hp + int(guard_config.get("effective_hp_bonus", 0))
    except Exception:  # noqa: BLE001
        return None


def _refill_hand_size(card_id: int | None, state, me: int, guard_config: dict) -> int | None:
    """そのカードを使った直後の「自分の手札枚数」の見積り。対象外カードなら None。

    対象は `_HAND_REFILL_SIZES`(config ``refill_hand_sizes`` で上書き可)。ドロー効果の
    精密なシミュレーションはしない: 手札連動打点の判定に必要なのは「手札を増やすサポートを
    使うと何枚になるか」だけで、リーリエ/クラウンはどちらもカードテキストで枚数が決まる
    (= 現在手札に依存しない)ため、この表引きで十分。
    """
    if card_id is None:
        return None
    sizes = guard_config.get("refill_hand_sizes") or _HAND_REFILL_SIZES
    size = sizes.get(card_id) if isinstance(sizes, dict) else None
    if size is None:
        # JSON の dict はキーが文字列になりうるので文字列キーでも引く。
        size = sizes.get(str(card_id)) if isinstance(sizes, dict) else None
    if size is None:
        return None
    if card_id == _LILLIE_ID:
        try:
            # 「自分のサイドの残り枚数が6枚なら、引く枚数は8枚になる」。
            if len(state.players[me].prize or []) == 6:
                return 8
        except Exception:  # noqa: BLE001
            pass
    return int(size)


def _is_supporter(card_id: int | None) -> bool:
    """card_id がサポートか。未知ID/例外は False(=介入しない、安全側)。"""
    if card_id is None:
        return False
    try:
        from cg.api import CardType
        from ptcg_ai.shared import card_cache
        return card_cache.get_card(int(card_id)).cardType == CardType.SUPPORTER
    except Exception:  # noqa: BLE001
        return False


def _policy_scores(obs: Observation, select: SelectData, config: dict | None) -> list[float] | None:
    """事後veto群で共通の「方策スコア」取得。長さが合わない/例外は None。"""
    try:
        model = _get_model(config)
        scores = model.score_options(
            obs, _model_hidden_state_factory(obs, config),
            time.perf_counter() + _MODEL_TIME_BUDGET_MS / 1000,
        )
    except Exception:  # noqa: BLE001 - スコアリング失敗が意思決定を止めてはならない
        return None
    if not scores or len(scores) != len(select.option):
        return None
    return list(scores)


def _try_hand_damage_guard(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """手札枚数に比例する相手ワザで即死する手を避ける。config-gated(``hand_damage_guard``)、
    キーが無ければ常に None=本番不変。

    実ラダーの負け分析(og_r7): 「リーリエの決心で手札を6枚に増やす → 相手メガユキメノコex
    の うらみぶし(相手の手札の枚数×50)が300ダメージ → 240HPのオーガポンが即死」という
    自滅が複数試合で起きていた。相手フーディンの ハンドパワー(自分の手札の枚数×2個のダメカン
    = 手札×20)も同型で、こちらは**相手の**手札枚数が打点になる。

    config 例(``imminent`` は I2、``substitute_only`` は I3。どちらも任意で、無ければ I1 と
    同じ presence-only + 「代替が無くても封じる」挙動=既定挙動不変)::

        "hand_damage_guard": {"enabled": true, "substitute_only": true, "sources": [
          {"card_id": 861, "per_card": 50, "count": "own_hand", "imminent": "active_or_energized"},
          {"card_id": 743, "per_card": 20, "count": "opp_hand", "imminent": "active_or_energized"}
        ]}

    発火条件:
      - 相手の場(アクティブ+ベンチ)に ``card_id`` が見えている
      - ``imminent: "active_or_energized"`` が指定された source は、さらに **その card_id の
        個体がアクティブに居る、またはエネルギーが1個以上付いている** ことも要る
        (`_opponent_field_imminent`)。I1(presence-only)は mega_froslass_ex 対面で -6.2pt
        (n=400、境界有意)の退行を出した。ベンチ・エネ0の個体は「前に出す→エネ付与→攻撃」を
        経てようやく打点が立つため、アクティブ/エネ付きの個体だけに絞って過剰発火を抑える(I2)
      - 今ターンの確定勝ち(lethal)が無い。`_select_action` は `_try_lethal` が None を
        返した decision でしか事後veto群に到達しないので、この条件は呼び出し位置で担保される
        (ここで探索を張り直すと二重コストになるため張らない)
      - 選んだ手が**サポートの PLAY**(ジャッジマン自身を除く)。エネ付与や攻撃を取り上げる
        ことは無い(サポート枠は1ターン1枚なので、差し替えても失うのは「別のサポート」だけ)

    判定:
      - ``own_hand`` 型: 選んだサポートが手札を増やす札(リーリエ/クラウン)で、使用後の
        手札見積り × ``per_card`` >= 自分アクティブの実効HP なら即死。
      - ``opp_hand`` 型: 相手の handCount × ``per_card`` >= 実効HP なら即死。

    差し替え:
      1. ジャッジマン(1213)が選択肢にあり、かつジャッジ後(手札4枚)なら全ての発火 source で
         即死しなくなるなら、ジャッジマンにする(リーリエより優先)。
      2. ジャッジ不在/ジャッジでも助からない場合、``own_hand`` 型が発火していれば「手札を
         増やす手」以外で方策スコア最大の手にする(サポート未使用=ENDも候補に含む)。
         ``substitute_only: true`` (I3) のときは、この候補を **ジャッジマン(1213)/
         ボスの指令(1182)の PLAY のみ** に絞る(優先順位はジャッジ > ボス)。どちらも
         選択肢に無ければ **介入しない**(リーリエ/クラウンをそのまま許す)。
      3. ``opp_hand`` 型だけが発火していてジャッジが無いなら、代替の当てが無いので介入しない
         (``substitute_only`` の有無に関わらず、ボスの指令は相手の手札を減らさないので
         opp_hand 型の脅威は解決しない=対象外のまま)。

    Iteration 3 の背景: I1(presence-only)はフィールド計測で mega_froslass 対面 -6.2pt の
    退行を出した。実敗着局面(93335550 row168 / 93324616 row132 / 93309236 row130)を
    再監査すると、いずれも代替(ジャッジマンまたはボスの指令)が手札にあり差し替えで解決して
    いた=退行の有害枝は「代替が無いのにリーリエ/クラウンを封じるだけ封じてドローエンジンを
    止める」純損のケースだった。``substitute_only`` はこの純損ケースだけを介入対象から外す
    (代替が実在する場合の挙動は従来どおり不変)。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    guard = (effective_config or {}).get("hand_damage_guard") or {}
    if not guard.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None

    state = obs.current
    me = state.yourIndex
    from ptcg_ai.learning import encoder as _enc
    chosen_option = select.option[idx]
    if chosen_option.type != OptionType.PLAY:
        return None
    chosen_card_id = _enc._resolve_card_id(chosen_option, state)
    if chosen_card_id == _JUDGE_ID or not _is_supporter(chosen_card_id):
        return None  # ジャッジ自身/非サポートには干渉しない

    eff_hp = _effective_active_hp(state, me, guard)
    if eff_hp is None or eff_hp <= 0:
        return None
    on_field = _opponent_field_card_ids(state, me)
    if not on_field:
        return None

    predicted_own_hand = _refill_hand_size(chosen_card_id, state, me, guard)
    try:
        opp_hand_count = int(state.players[1 - me].handCount or 0)
    except Exception:  # noqa: BLE001
        return None

    own_hand_threat = False
    fired = False
    judge_saves = True  # 発火した source すべてが「手札4枚なら死なない」なら True
    for source in (guard.get("sources") or []):
        try:
            card_id = int(source["card_id"])
            per_card = int(source["per_card"])
            kind = source.get("count", "own_hand")
        except Exception:  # noqa: BLE001 - 壊れた source は無視(安全側)
            continue
        if card_id not in on_field or per_card <= 0:
            continue
        if source.get("imminent") == "active_or_energized" \
                and not _opponent_field_imminent(state, me, card_id):
            continue  # I2: アクティブでもエネ付きでもない=切迫していない個体は見逃す
        if kind == "own_hand":
            if predicted_own_hand is None:
                continue  # 選んだ手は手札を増やさない=この source では死なない
            count = predicted_own_hand
        elif kind == "opp_hand":
            count = opp_hand_count
        else:
            continue
        if count * per_card < eff_hp:
            continue
        fired = True
        if kind == "own_hand":
            own_hand_threat = True
        if _JUDGE_HAND_SIZE * per_card >= eff_hp:
            # ジャッジ後(手札4枚)でも死ぬ=ジャッジへの差し替えは救いにならない。
            judge_saves = False
    if not fired:
        return None

    _GUARD_STATS["hand_damage_guard_fired"] += 1
    scores = _policy_scores(obs, select, config)

    # 1) ジャッジマンへの差し替え(手札を4枚に落として被弾を下げる)。
    if judge_saves:
        judge_idxs = [
            i for i, opt in enumerate(select.option)
            if opt.type == OptionType.PLAY and _enc._resolve_card_id(opt, state) == _JUDGE_ID
        ]
        if judge_idxs:
            best_judge = (
                max(judge_idxs, key=lambda i: scores[i]) if scores else judge_idxs[0]
            )
            action = [best_judge]
            if _is_valid_action(action, select):
                _GUARD_STATS["hand_damage_guard_to_judge"] += 1
                return action

    # 2) ジャッジが無い/救えない: 「手札を増やす手」だけを避ける(own_hand 型のときだけ)。
    if not own_hand_threat or scores is None:
        return None
    substitute_only = bool(guard.get("substitute_only", False))
    best_alt = best_score = None
    for i, opt in enumerate(select.option):
        if i == idx:
            continue
        if substitute_only:
            # I3: 代替サポート(ジャッジマン優先、無ければボスの指令)以外への逃げは
            # 「封じるだけ封じてドローエンジンを止める」純損になるため候補から外す。
            if opt.type != OptionType.PLAY:
                continue
            if _enc._resolve_card_id(opt, state) not in (_JUDGE_ID, _BOSS_ORDER_ID):
                continue
        elif opt.type == OptionType.PLAY and _refill_hand_size(
            _enc._resolve_card_id(opt, state), state, me, guard
        ) is not None:
            continue  # 手札を増やす手は全部除外
        if best_score is None or scores[i] > best_score:
            best_score, best_alt = scores[i], i
    if best_alt is None:
        if substitute_only:
            # 代替(ジャッジ/ボス)が手札に無い=差し替えても純損なので介入しない
            # (リーリエ/クラウンをそのまま許す。I3 が避けたい退行パターン)。
            _GUARD_STATS["hand_damage_guard_no_substitute"] += 1
        return None
    action = [best_alt]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["hand_damage_guard_to_other"] += 1
    if select.option[best_alt].type == OptionType.PLAY \
            and _enc._resolve_card_id(select.option[best_alt], state) == _BOSS_ORDER_ID:
        _GUARD_STATS["hand_damage_guard_to_boss"] += 1
    return action


def _try_briar_gate(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """ブライア(1201)の空撃ちを止める。config-gated(``briar_gate``)、キーが無ければ常に
    None=本番不変。

    ブライアは「このターン相手のポケモンをきぜつさせたとき、サイドをもう1枚多く取る」札で、
    **KOできないターンに撃つと完全に無駄**(サポート枠も1枚失う)。

    実測メモ: og_r7 の負け分析で挙がった 93309236 T9 / 93326434 T8 のブライアは、どちらも
    `ko_search` 基準では KO 可能=このゲートでは**発火しない**(93309236 の敗因は「ブライアが
    無駄」ではなく「サポート枠をジャッジマンに使えば生き残れた」で、そちらは
    `_try_hand_damage_guard` が拾う)。このゲートが直すのは純粋な空撃ちだけ。

    判定は `_try_hammer_veto` と同じ流儀の `ko_search.can_ko_this_turn`(瞬間打点ではなく
    エネ装着→攻撃まで含めたターン先読み)。KO可能なら正当な使用なので干渉しない。KO不能なら
    ブライア以外で方策スコア最大の手に差し替える。

    **重要**: `can_ko_this_turn` の False は「KOできない」と「判定不能(隠れ状態を組めない/
    予算切れ)」を区別しない。ハンマーvetoでは False 側が安全側だったが、ブライアでは逆で、
    予算切れをKO不能と誤読すると**正当なブライアまで潰す**(実測: 93326434 T8 は 300ms なら
    can_ko=True だが 100ms では時間切れ)。そこで `report` を受け取り、探索が実際に完走した
    (`searched and not aborted`)ときだけ介入する。既定予算もハンマー(100ms)より厚い 300ms。
    判定不能/例外はすべて None(=介入しない、安全側)。

    ``briar_gate`` は bool(``true`` で有効)。dict を渡すと ``{"enabled": true,
    "time_limit_ms": 300, "ko_search": {...}}`` として ko_search の予算も指定できる
    (ablation 用。bool のときは既定 300ms)。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    gate = (effective_config or {}).get("briar_gate")
    if isinstance(gate, dict):
        if not gate.get("enabled", True):
            return None
    elif gate:
        gate = {}   # bool の true = 既定パラメータで有効
    else:
        return None  # キー無し / false / None = 無効(本番不変)

    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    from ptcg_ai.learning import encoder as _enc
    briar_ids = set(gate.get("card_ids") or (_BRIAR_ID,))
    if _enc._resolve_card_id(select.option[idx], obs.current) not in briar_ids:
        return None  # ブライア以外は無関係

    try:
        from ptcg_ai.search import ko_search
        ko_cfg = gate.get("ko_search") or {}
        limit_ms = float(ko_cfg.get("time_limit_ms", gate.get("time_limit_ms", 300)))
        deadline = time.perf_counter() + limit_ms / 1000.0
        report: dict = {}
        can_ko = bool(ko_search.can_ko_this_turn(
            obs, _model_hidden_state_factory(obs, config), ko_cfg, deadline, report=report))
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return None
    if can_ko:
        return None  # KOできる=サイド追加が実現する=正当な使用
    if not report.get("searched") or report.get("aborted"):
        return None  # 判定不能(隠れ状態を組めない/予算切れ)=介入しない

    scores = _policy_scores(obs, select, config)
    if scores is None:
        return None
    best_alt = best_score = None
    for i, opt in enumerate(select.option):
        if _enc._resolve_card_id(opt, obs.current) in briar_ids:
            continue
        if best_score is None or scores[i] > best_score:
            best_score, best_alt = scores[i], i
    if best_alt is None:
        return None
    action = [best_alt]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["briar_gate_fired"] += 1
    return action


# --------------------------------------------------------------------------------------
# r9 Fix-D: 山札僅少ブレーキ / Fix-E: ボスのKOゲート
# (実ラダー68試合・32敗の全数調査に基づく。どちらも独立の config キーで既定OFF=本番不変)
# --------------------------------------------------------------------------------------

# ブレーキ対象=「自分の手札を山に戻して引き直す」ドローサポートと、その**引く枚数**。
# 山札の増減はこの引く枚数と手札枚数の差で決まる(実測で検証、`_draw_supporter_deck_delta`)。
# ジャッジマン(1213)は意図的にこの表に入れない(下記 `_try_low_deck_draw_brake` 参照)。
_DEFAULT_DRAW_SUPPORTER_DRAWS = {_LILLIE_ID: 6, _HARLEQUIN_ID: 5}


def _draw_supporter_deck_delta(card_id: int | None, state, me: int, brake: dict) -> int | None:
    """そのドローサポートを使ったときの**自分の山札の増減**。対象外カード/判定不能は None。

    実測(93517227 row104→105 / 93526443 row147→149 / 93517227 row124→125)で確認した式::

        山札の増減 = (手札枚数 - 1) - 引く枚数

    「-1」は使ったサポート自身がトラッシュへ行く分(山に戻るのは残りの手札)。リーリエは
    自サイド6枚なら8枚引く補正があるので、枚数の解決は `_refill_hand_size` に委ねる
    (`hand_damage_guard` と同じ表・同じ補正を使う=引く枚数の定義をリポジトリ内で一本化する)。
    """
    draws = brake.get("draw_counts") or _DEFAULT_DRAW_SUPPORTER_DRAWS
    draw = _refill_hand_size(card_id, state, me, {"refill_hand_sizes": draws})
    if draw is None:
        return None
    try:
        hand = len(state.players[me].hand or [])
    except Exception:  # noqa: BLE001
        return None
    return (hand - 1) - int(draw)


def _try_low_deck_draw_brake(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """山札が僅少なときの「山札を減らすドロー」を止める。config-gated
    (``low_deck_draw_brake``)、キーが無ければ常に None=本番不変。

    実ラダー68試合・32敗の全数調査: **32敗中5敗(15.6%)が自分の山札0による敗北**で、相手の
    山札は14〜29枚残っていた。うち3敗はサイドを3〜4枚先取した優勢形での自滅
    (93526443: T21 山5・手札3 でリーリエの決心 → 山2 → T23 山0)。一方 T10 時点の山札中央値は
    勝ち13.5 / 負け19.5 で、**速く回すこと自体は勝ちと正相関**。したがって抑制は
    「本当に山を減らす1手」だけに絞る必要がある。

    config 例::

        "low_deck_draw_brake": {"enabled": true, "deck_threshold": 6}

    発火条件(すべてAND):
      - 自分の ``deckCount`` <= ``deck_threshold``(既定6)
      - 選んだ手が **山札を減らすドローサポートの PLAY**。既定の対象はリーリエの決心(1227)と
        クラウン(1223)(``card_ids`` で上書き可)。さらに ``require_net_deck_loss``(既定 true)
        のときは `_draw_supporter_deck_delta` < 0、すなわち**実際に山札が減る**ことも要る
      - このターン確定KOが取れない(`ko_search.can_ko_this_turn`)。KOが取れるなら引いて
        勝ちに行ってよい(ブレーキは負け筋の回避であって、勝ち筋を止める道具ではない)

    **ジャッジマン(1213)は対象外**(明示的に弾く)。ジャッジは互いの手札を山に戻して4枚引く
    ので、手札5枚以上なら**山札はむしろ増える**=山札切れ対策としては有効な札。同じ理由で
    リーリエ/クラウンも「手札が引く枚数より多い」局面では山札が増える(93517227 row104 は
    山10・手札9 でリーリエ=山13 に**増えている**)。この「増える/減らない使い方」まで潰すと
    ドローエンジンを止めるだけの純損になるため、既定では実増減を見る(``require_net_deck_loss``)。
    literal な「リーリエ/クラウンなら常に」挙動を測りたい場合だけ false にする(ablation 用)。

    KO探索の False は「KOできない」と「判定不能」を区別しないため、`_try_briar_gate` と同じく
    **探索が完走したときだけ**介入する(``require_conclusive_ko_search``、既定 true)。
    判定不能でブレーキを掛けると、ユーザー指摘の「過度な抑制」側に倒れるため。

    r13(``use_closed_form_ko``、既定なし=OFF): 上記のKO判定を**閉形式ファストパス →
    従来の探索 → それも判定不能なら不介入**の3段にする(`_ko_verdict_for_brake`)。
    実対戦ではオーガポンが複数体並ぶ盤面で探索が予算内に完走せず、このガードは
    120試合で発火0回だった。閉形式(`search/closed_form_ko.py`)は探索なしで
    「今すぐ攻撃すればKO」/「攻撃でKOする手は存在しない」を O(1) で判定する。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    brake = (effective_config or {}).get("low_deck_draw_brake") or {}
    if not brake.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    option = select.option[idx]
    if option.type != OptionType.PLAY:
        return None

    state = obs.current
    me = state.yourIndex
    from ptcg_ai.learning import encoder as _enc
    card_id = _enc._resolve_card_id(option, state)
    if card_id == _JUDGE_ID:
        return None  # ジャッジは山札を増やす側=山札切れ対策としてはむしろ有効
    brake_ids = set(brake.get("card_ids") or _DEFAULT_DRAW_SUPPORTER_DRAWS.keys())
    brake_ids.discard(_JUDGE_ID)  # config で誤って入れられてもジャッジは止めない
    if card_id not in brake_ids:
        return None

    try:
        deck_count = int(state.players[me].deckCount)
    except Exception:  # noqa: BLE001
        return None
    if deck_count > int(brake.get("deck_threshold", 6)):
        return None
    if brake.get("require_net_deck_loss", True):
        delta = _draw_supporter_deck_delta(card_id, state, me, brake)
        if delta is None or delta >= 0:
            return None  # 山札が減らない使い方=止める理由がない

    # このターンKOできるなら引いて勝ちに行ってよい(ブレーキは掛けない)。
    # r13: 「閉形式ファストパス → 従来の探索 → それも判定不能なら不介入」の3段判定
    # (`_ko_verdict_for_brake`)。``use_closed_form_ko`` が無ければ1段目は素通りするので、
    # キーの無い config では r9〜r12 と完全に同一の探索だけが走る。
    if _ko_verdict_for_brake(obs, brake, config, "low_deck_draw_brake") is not False:
        return None  # True=KOできるので通す / None=判定不能=安全側で介入しない

    scores = _policy_scores(obs, select, config)
    if scores is None:
        return None
    best_alt = best_score = None
    for i, opt in enumerate(select.option):
        # 対象のドローサポートは全部除外(別の1枚に差し替えても山札は同じだけ減る)。
        if opt.type == OptionType.PLAY and _enc._resolve_card_id(opt, state) in brake_ids:
            continue
        if best_score is None or scores[i] > best_score:
            best_score, best_alt = scores[i], i
    if best_alt is None:
        return None
    action = [best_alt]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["low_deck_draw_brake_fired"] += 1
    return action


def _opponent_bench_energy_counts(state, me: int) -> dict[int, int]:
    """相手ベンチの index -> エネルギー個数。判定不能は空 dict。"""
    out: dict[int, int] = {}
    try:
        for i, pk in enumerate(state.players[1 - me].bench or []):
            if pk is None:
                continue
            out[i] = len(getattr(pk, "energies", None) or [])
    except Exception:  # noqa: BLE001
        return {}
    return out


def _boss_target_key(entry_or_option) -> tuple | None:
    """対象の照合キー (area, index, playerIndex)。選択肢の並び順に依存しない同一性判定用。"""
    try:
        if isinstance(entry_or_option, dict):
            return (entry_or_option["area"], entry_or_option["index"], entry_or_option["player_index"])
        return (
            getattr(entry_or_option, "area", None),
            getattr(entry_or_option, "index", None),
            getattr(entry_or_option, "playerIndex", None),
        )
    except Exception:  # noqa: BLE001
        return None


def _boss_allowed_targets(state, me: int, targets: list[dict], gate: dict) -> tuple[set, set]:
    """(許可する対象キーの集合, そのうちエネ除去例外で許可した対象キーの集合)。

    許可 = 「このターンKOできる」対象。``allow_energy_denial``(既定 true)のときは例外として
    「相手ベンチで**最多のエネルギー**を持つ(かつ1個以上)」対象も許可する。KO できなくても、
    育ったアタッカーを前に出して倒す/エネを盤面から抜くのは正当な使い方(実ラダー 93528233 T6
    の成功例=8エネのオーガポンを引きずり出し、草8個を盤面から永久に除去した)。
    """
    allowed: set = set()
    denial: set = set()
    for t in targets:
        key = _boss_target_key(t)
        if key is not None and t.get("can_ko"):
            allowed.add(key)
    if not gate.get("allow_energy_denial", True):
        return allowed, denial
    energies = _opponent_bench_energy_counts(state, me)
    if not energies:
        return allowed, denial
    max_energy = max(energies.values())
    if max_energy < max(1, int(gate.get("min_denial_energy", 1))):
        return allowed, denial
    for t in targets:
        key = _boss_target_key(t)
        if key is None or t.get("area") != AreaType.BENCH:
            continue
        if energies.get(t.get("index"), -1) == max_energy:
            allowed.add(key)
            denial.add(key)
    return allowed, denial


def _try_boss_lethal_gate(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """ボスの指令(1182)を「今ターン倒せる相手がいるとき」だけ許可する。config-gated
    (``boss_lethal_gate``)、キーが無ければ常に None=本番不変。

    実ラダー 93503044 T13: メガガルーラex(HP330)に対し自分は8エネ=270ダメージで**KO不可**
    なのに ボスの指令で引きずり出し、60HP残しで逃げられ、以降12ターン3サイド分をベンチに
    放置した。逆に成功例(93528233 T6 ほか)は「引きずり出した相手を倒せる」か
    「相手の最大エネ源を引き剥がす」ときだけだった。

    config 例::

        "boss_lethal_gate": {"enabled": true, "allow_energy_denial": true}

    この関数(PLAY 決定側)は **「許可できる対象が1体も居ない」ときだけ**ボスを次善手へ
    差し替える。対象が居るなら差し替えず、**どの相手を出すか**は直後の対象選択で
    `_try_boss_target_redirect` が直す(ボスの PLAY と対象選択は別 decision なので、
    ハンマーの redirect と同じ持ち越しキャッシュ方式を使う)。

    KO判定は打点式を手書きせず `search.boss_target_eval`(ボスPLAY→対象→`ko_search` のDFS)
    に委ねる。弱点・特性・「エネ加速してから攻撃」まで**エンジンの真実**で解くので、
    オーガポン専用式ではない。評価が完走しなかった/判定不能のときは介入しない(安全側)。
    """
    global _boss_target_cache
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    gate = (effective_config or {}).get("boss_lethal_gate") or {}
    if not gate.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    option = select.option[idx]
    if option.type != OptionType.PLAY:
        return None
    from ptcg_ai.learning import encoder as _enc
    boss_ids = set(gate.get("card_ids") or (_BOSS_ORDER_ID,))
    if _enc._resolve_card_id(option, obs.current) not in boss_ids:
        return None  # ボス以外は無関係(コストゼロ)

    state = obs.current
    me = state.yourIndex
    # このボスPLAYについての判定をこれから作り直すので、古い持ち越しは先に捨てる
    # (判定不能で早期 return したときに前の decision のキャッシュが残らないようにする)。
    _boss_target_cache = None
    try:
        from ptcg_ai.search import boss_target_eval
        eval_cfg = gate.get("target_eval") or {}
        limit_ms = float(eval_cfg.get("time_limit_ms", gate.get("time_limit_ms", 400)))
        result = boss_target_eval.evaluate_targets(
            obs, _model_hidden_state_factory(obs, config), idx, eval_cfg,
            time.perf_counter() + limit_ms / 1000.0)
    except Exception:  # noqa: BLE001 - 評価失敗が意思決定を止めてはならない
        return None
    if result is None or result.get("any_aborted"):
        _GUARD_STATS["boss_lethal_gate_inconclusive"] += 1
        return None  # 判定不能=介入しない(安全側)

    targets = result.get("targets") or []
    allowed, denial = _boss_allowed_targets(state, me, targets, gate)
    if allowed:
        # 許可できる相手が居る=ボス自体は正当。どれを出すかは対象選択で直す。
        _boss_target_cache = {
            "your_index": me,
            "turn": getattr(state, "turn", None),
            "select_seq": _selects_seen,
            "allowed": allowed,
            "denial": denial,
        }
        return None

    # 倒せる相手も剥がす価値のある相手も居ない=ボスは盤面を悪化させるだけ。次善手へ。
    _boss_target_cache = None
    scores = _policy_scores(obs, select, config)
    if scores is None:
        return None
    best_alt = best_score = None
    for i, opt in enumerate(select.option):
        if opt.type == OptionType.PLAY and _enc._resolve_card_id(opt, state) in boss_ids:
            continue
        if best_score is None or scores[i] > best_score:
            best_score, best_alt = scores[i], i
    if best_alt is None:
        return None
    action = [best_alt]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["boss_lethal_gate_fired"] += 1
    return action


def _is_boss_target_select(select: SelectData, boss_ids) -> bool:
    """この select がボスの「引きずり出す相手を選ぶ」select か(実測: CARD / SWITCH / effect=ボス)。"""
    try:
        if select.type != SelectType.CARD or select.context != SelectContext.SWITCH:
            return False
        effect = getattr(select, "effect", None)
        effect_id = getattr(effect, "id", None) if effect is not None else None
        return effect_id is not None and effect_id in boss_ids
    except Exception:  # noqa: BLE001
        return False


def _try_boss_target_redirect(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """ボスの**対象**を「今ターン倒せる相手」へ振り替える。config-gated(``boss_lethal_gate``)。

    直前の PLAY 決定で `_try_boss_lethal_gate` が作ったキャッシュ(同じ試合・同じターン・
    **1つ前の decision**)だけを信用する。ハンマーの redirect と同じ理由で、対象選択の時点では
    効果解決中のボスがどのゾーンにも見えず `build_dummy_search_state` の整合チェックが1枚ずれる
    ため、ここで評価を張り直すことはできない(`_hammer_redirect_can_ko` の実測メモ参照)。
    キャッシュが無い/古い/対象が判別できない場合は介入しない(安全側=従来挙動)。
    """
    global _boss_target_cache
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    gate = (effective_config or {}).get("boss_lethal_gate") or {}
    if not gate.get("enabled", False):
        return None
    select = obs.select
    boss_ids = set(gate.get("card_ids") or (_BOSS_ORDER_ID,))
    if not _is_boss_target_select(select, boss_ids):
        return None
    if len(chosen_action) != 1 or select.maxCount != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None

    cache = _boss_target_cache
    _boss_target_cache = None  # 1回きり(次の decision へは持ち越さない)
    try:
        if (
            cache is None
            or cache.get("your_index") != obs.current.yourIndex
            or cache.get("turn") != getattr(obs.current, "turn", None)
            or _selects_seen - int(cache.get("select_seq", -99)) != 1
        ):
            return None
    except Exception:  # noqa: BLE001
        return None
    allowed = cache.get("allowed") or set()
    if not allowed:
        return None
    if _boss_target_key(select.option[idx]) in allowed:
        if _boss_target_key(select.option[idx]) in (cache.get("denial") or set()):
            _GUARD_STATS["boss_lethal_gate_denial_kept"] += 1
        return None  # 既に許可対象を選んでいる=介入不要

    candidates = [i for i, opt in enumerate(select.option) if _boss_target_key(opt) in allowed]
    if not candidates:
        return None
    best = candidates[0]
    scores = _policy_scores(obs, select, config)
    if scores is not None:
        best = max(candidates, key=lambda i: scores[i])
    action = [best]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["boss_lethal_gate_redirected"] += 1
    return action


# --------------------------------------------------------------------------------------
# r10: 実ラダーのリプレイ分析で確定した「誤爆余地のほぼ無い無料改善」3種。
#   Fix-F `tool_stadium_guard`        どうぐ無効スタジアム下でのどうぐ装着を止める
#   Fix-G `energy_to_active_first`    攻撃不能なアクティブを差し置いてベンチにエネを付けない
#   Fix-H `search_pick_pokemon_first` 「山札の上からN枚見て手札に加える」でポケモンを拾う
# いずれも**独立の config キー**で、キーが無ければ即 None=本番挙動不変(config-gated)。
#
# 採用判定の方法論について: どれも実ラダー31敗中1〜3件の頻度で、勝率A/B(ローカルの
# ノイズ床 ±7pt)では原理的に判定できない。よって判定は (1) 実局面スナップショットで
# 正しい手を選ぶこと、(2) 発火率・誤爆率メトリクス(`kaggle_replays/_metrics_gate.py`)で
# 行う。勝率A/Bは「-3pt級の退行が無いこと」の確認にだけ使う。
# --------------------------------------------------------------------------------------

# ジャミングタワー(data/JP_Card_Data.csv で実ID確認済み: 1246 / スタジアム /
# 「おたがいのポケモン全員についている『ポケモンのどうぐ』の効果は、すべてなくなる。」)。
# config `tool_stadium_guard.nullifying_stadium_ids` で上書き可能(将来の同型スタジアム用)。
_JAMMING_TOWER_ID = 1246
_DEFAULT_NULLIFYING_STADIUM_IDS = (_JAMMING_TOWER_ID,)

# 「みどりのまい」型の**自分自身にエネを付ける特性**を持つポケモン。オーガポン みどりのめん ex
# (96)の みどりのまい =「自分の手札から基本【草】エネルギーを1枚選び、このポケモンにつける。
# その後、山札を1枚引く」。特性テキストの汎用パースは誤爆の温床なので、対象は明示リストで
# 持つ(config `energy_to_active_first.energy_ability_card_ids` で上書き可能)。
_TEAL_DANCE_CARD_IDS = (96,)

_R10_GUARD_STATS_ZERO = {
    # Fix-F: どうぐ無効スタジアム下のどうぐ装着veto
    "tool_stadium_guard_fired": 0,             # 無効スタジアム下のどうぐ装着を差し替えた回数
    "tool_stadium_guard_no_alternative": 0,    # 代替手が無く介入を見送った回数
    "tool_stadium_guard_misfire_no_stadium": 0,  # 番人: 無効スタジアムでないのに発火した回数(=0のはず)
    # Fix-G: 攻撃不能時のアクティブ優先エネ付け
    "energy_to_active_first_fired": 0,           # ベンチ→アクティブへ付け先を振り替えた回数
    "energy_to_active_first_no_active_option": 0,  # アクティブへ付けられず見送った回数
    "energy_to_active_first_misfire_can_attack": 0,  # 番人: 既に攻撃可能なのに発火した回数(=0のはず)
    # Fix-H: サーチ(looking)でポケモンを優先
    "search_pick_pokemon_first_fired": 0,       # ポケモンを拾うよう差し替えた回数
    "search_pick_pokemon_first_appended": 0,    # うち枠が余っていて「足した」回数
    "search_pick_pokemon_first_replaced": 0,    # うち別の候補と「入れ替えた」回数
    "search_pick_pokemon_first_misfire_bench_full": 0,  # 番人: ベンチ満杯なのに発火した回数(=0のはず)
}
# 既存の `_GUARD_STATS_ZERO` リテラルには触れず、追加登録だけする(同ファイルを同時に触る
# 他の作業とのコンフリクトを避けるため)。`get_guard_stats`/`reset_guard_stats` はそのまま動く。
_GUARD_STATS_ZERO.update(_R10_GUARD_STATS_ZERO)
_GUARD_STATS.update(_R10_GUARD_STATS_ZERO)


def _current_stadium_card_id(state) -> int | None:
    """今 場に出ているスタジアムの CardData id。無い/読めないなら None。"""
    try:
        stadium = state.stadium or []
        if not stadium:
            return None
        card = stadium[0]
        return int(card.id) if card is not None else None
    except Exception:  # noqa: BLE001 - 読み取り失敗が意思決定を止めてはならない
        return None


def _card_type_of(card_id: int | None):
    """card_id の CardType。未知ID/例外は None(=判定不能、呼び出し側は介入しない)。"""
    if card_id is None:
        return None
    try:
        from ptcg_ai.shared import card_cache
        return card_cache.get_card(int(card_id)).cardType
    except Exception:  # noqa: BLE001
        return None


def _is_pokemon_tool(card_id: int | None) -> bool:
    """「ポケモンのどうぐ」か。カードIDのハードコードではなく分類(CardType.TOOL)で判定する
    (data/JP_Card_Data.csv の『ポケモンのどうぐ』28枚に対応する engine 側の分類)。"""
    from cg.api import CardType
    return _card_type_of(card_id) == CardType.TOOL


def _is_energy_card(card_id: int | None) -> bool:
    """基本/特殊エネルギーか。未知ID/例外は False(=介入しない、安全側)。"""
    from cg.api import CardType
    return _card_type_of(card_id) in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)


def _is_basic_pokemon(card_id: int | None) -> bool:
    """たねポケモン(=引いてすぐベンチに出せる)か。未知ID/例外は False。"""
    if card_id is None:
        return False
    try:
        from cg.api import CardType
        from ptcg_ai.shared import card_cache
        card = card_cache.get_card(int(card_id))
        return card.cardType == CardType.POKEMON and bool(card.basic)
    except Exception:  # noqa: BLE001
        return False


def _try_tool_stadium_guard(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """どうぐの効果が無効化されるスタジアムの下で「ポケモンのどうぐ」を付ける手を却下する。
    config-gated(``tool_stadium_guard``)、キーが無ければ常に None=本番不変。

    実ラダーの実例(episode 93408551 T10 row129): 場が **ジャミングタワー**(1246、
    「おたがいのポケモン全員についている『ポケモンのどうぐ』の効果は、すべてなくなる」)なのに
    **ヒーローマント**(1159、最大HP+100)をHP10のオーガポンに装着した。効果は最初から
    無効なので完全な無駄撃ちで、そのターンHP10のまま気絶しマントもトラッシュへ流れた。

    config 例::

        "tool_stadium_guard": {"enabled": true, "nullifying_stadium_ids": [1246]}

    発火条件(すべて満たすときだけ):
      - 場のスタジアムが ``nullifying_stadium_ids``(既定 [1246])に含まれる
      - 選んだ手が **ポケモンのどうぐの ATTACH / PLAY**(判定は cardId のハードコードでは
        なく `CardType.TOOL` = カード分類。将来どうぐが増えても効く)
      - 単一選択の MAIN(どうぐを付ける decision の形)

    差し替え先はどうぐ以外で方策スコア最大の手。代替が無い/スコアが取れない場合は介入しない
    (安全側)。「先にスタジアムを張り替えてからどうぐを付ける」判断は方策/探索の領分なので
    ここではやらない(このガードは無駄撃ちを止めるだけ)。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    guard = (effective_config or {}).get("tool_stadium_guard") or {}
    if not guard.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None

    state = obs.current
    nullifying = set(guard.get("nullifying_stadium_ids") or _DEFAULT_NULLIFYING_STADIUM_IDS)
    stadium_id = _current_stadium_card_id(state)
    if stadium_id is None or stadium_id not in nullifying:
        return None  # どうぐが有効な盤面=干渉しない

    from ptcg_ai.learning import encoder as _enc
    chosen = select.option[idx]
    if chosen.type not in (OptionType.ATTACH, OptionType.PLAY):
        return None
    if not _is_pokemon_tool(_enc._resolve_card_id(chosen, state)):
        return None  # どうぐ以外(エネ付け等)には干渉しない

    scores = _policy_scores(obs, select, config)
    if scores is None:
        return None
    best_alt = best_score = None
    for i, opt in enumerate(select.option):
        if opt.type in (OptionType.ATTACH, OptionType.PLAY) \
                and _is_pokemon_tool(_enc._resolve_card_id(opt, state)):
            continue  # 他のどうぐ装着も同じ理由で無駄=候補から外す
        if best_score is None or scores[i] > best_score:
            best_score, best_alt = scores[i], i
    if best_alt is None:
        _GUARD_STATS["tool_stadium_guard_no_alternative"] += 1
        return None
    action = [best_alt]
    if not _is_valid_action(action, select):
        return None
    if _current_stadium_card_id(state) not in nullifying:  # 番人(到達しないはず)
        _GUARD_STATS["tool_stadium_guard_misfire_no_stadium"] += 1
    _GUARD_STATS["tool_stadium_guard_fired"] += 1
    return action


def _energy_attach_target_area(option, state, ability_ids: set[int]):
    """この選択肢が「エネルギーを自分の場のポケモンに付ける手」なら、その**付け先の AreaType**
    (ACTIVE / BENCH)を返す。そうでなければ None。

    2系統ある(実データ episode 93408551 T6 で確認):
      - 手貼り/道具的なエネ付け = ``OptionType.ATTACH``。``area``/``index`` が手札のエネを、
        ``inPlayArea``/``inPlayIndex`` が付け先の場のポケモンを指す(cg/api.py の
        OptionType.ATTACH コメント参照)。同じ手札 index の ATTACH が付け先の数だけ並ぶ。
      - 「みどりのまい」型の特性 = ``OptionType.ABILITY``。``area``/``index`` が特性を使う
        ポケモン自身を指し、エネはそのポケモン自身に付く(=付け先 == そのポケモン)。
    """
    try:
        from ptcg_ai.learning import encoder as _enc
        if option.type == OptionType.ATTACH:
            if not _is_energy_card(_enc._resolve_card_id(option, state)):
                return None
            return getattr(option, "inPlayArea", None)
        if option.type == OptionType.ABILITY:
            card_id = _enc._resolve_card_id(option, state)
            if card_id is None or int(card_id) not in ability_ids:
                return None
            return getattr(option, "area", None)
    except Exception:  # noqa: BLE001 - 判定不能は None(=介入しない、安全側)
        return None
    return None


def _try_energy_to_active_first(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """アクティブが攻撃コストを満たしていないのにベンチへエネを付ける手を、アクティブへの
    エネ付けに振り替える。config-gated(``energy_to_active_first``)、キーが無ければ常に
    None=本番不変。

    実ラダーの実例(episode 93408551 T6 rows84/86/88): アクティブのオーガポン(草2、
    まんようしぐれ=草3が必要なので **ATTACK 選択肢が出ていない**)を差し置いて、
    みどりのまい(特性)3回をすべて**ベンチ**個体に使い、手貼り権も未使用のままターン終了。
    そのターンの攻撃を丸ごと損した(相手はその間に自由に殴れる)。

    config 例(``energy_ability_card_ids`` は任意。既定は [96]=オーガポン みどりのめん ex)::

        "energy_to_active_first": {"enabled": true}

    発火条件(すべて満たすときだけ):
      - 単一選択の MAIN で、自分のアクティブが場に居る
      - **選択肢に ATTACK が1件も無い** = 今のアクティブは攻撃コストを満たしていない
        (engine が出す選択肢が唯一の正確な根拠。自前のコスト計算はしない)
      - 選んだ手が「エネルギーをベンチに付ける手」(`_energy_attach_target_area` == BENCH)
      - 同じ select に「エネルギーをアクティブに付ける手」がある

    振替先は、まず**同じ種類**(特性なら特性、手貼りなら手貼り)のアクティブ向け選択肢から
    方策スコア最大を選ぶ。同種が無ければ種類を問わずアクティブ向けから選ぶ。アクティブへ
    付けられる選択肢が1つも無ければ介入しない(=ベンチ育成が唯一の選択肢の局面は尊重する)。

    **既知の適用範囲(要判断)**: 「ワザを持たない補助ポケモン(キチキギスex 等)がアクティブに
    居座り、本命のアタッカーはベンチ」という盤面では、アクティブへ寄せるのは損になりうる。
    og_v032(ポケモンはオーガポン96のみ=常にワザを持つ)では起こり得ないため、現状は
    「アクティブにワザがあるか」の条件を**入れていない**。他デッキでこのキーを有効にする場合は
    設計側で要判断(入れるなら `card_cache.get_card(active.id).attacks` の空判定を足すだけ)。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    guard = (effective_config or {}).get("energy_to_active_first") or {}
    if not guard.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None

    state = obs.current
    me = state.yourIndex
    try:
        active = state.players[me].active or []
        if not active or active[0] is None:
            return None  # アクティブが居ない/伏せ中=対象外
    except Exception:  # noqa: BLE001
        return None

    if any(opt.type == OptionType.ATTACK for opt in select.option):
        return None  # 既に攻撃できる=このガードの出番ではない

    ability_ids = {int(x) for x in (guard.get("energy_ability_card_ids") or _TEAL_DANCE_CARD_IDS)}
    if _energy_attach_target_area(select.option[idx], state, ability_ids) != AreaType.BENCH:
        return None  # ベンチへのエネ付け以外(ドロー/進化/攻撃準備等)には干渉しない

    candidates = [
        i for i, opt in enumerate(select.option)
        if _energy_attach_target_area(opt, state, ability_ids) == AreaType.ACTIVE
    ]
    if not candidates:
        _GUARD_STATS["energy_to_active_first_no_active_option"] += 1
        return None

    same_kind = [i for i in candidates if select.option[i].type == select.option[idx].type]
    pool = same_kind or candidates
    best = pool[0]
    scores = _policy_scores(obs, select, config)
    if scores is not None:
        best = max(pool, key=lambda i: scores[i])
    action = [best]
    if not _is_valid_action(action, select):
        return None
    if any(opt.type == OptionType.ATTACK for opt in select.option):  # 番人(到達しないはず)
        _GUARD_STATS["energy_to_active_first_misfire_can_attack"] += 1
    _GUARD_STATS["energy_to_active_first_fired"] += 1
    return action


def _looking_card_id(option, state) -> int | None:
    """``area == LOOKING`` の選択肢が指す実カードの id。判定不能なら None。

    実データ(episode 93473767 T3 row37 / 93408551 T10 row131)で確認したスキーマ:

        state.looking          = [1, 1, 1251, 1, 1120, 96, 1]   ← 見えている7枚の実体
        select.option[i]       = Option(type=CARD, area=LOOKING, index=<looking内index>)
        select.option[i].cardId = None                          ← option 単体では盲目

    つまり **option には cardId が入らないが、`state.looking` を引けば実体が分かる**。
    `encoder._resolve_card_id` は LOOKING を解決しない(`_AREA_TO_PLAYER_ZONE` に無い)ため、
    方策から見ると候補は識別不能=このガードが供給する追加情報になる。
    """
    try:
        if getattr(option, "area", None) != AreaType.LOOKING:
            return None
        looking = getattr(state, "looking", None)
        index = getattr(option, "index", None)
        if not looking or index is None or not (0 <= index < len(looking)):
            return None
        card = looking[index]
        return int(card.id) if card is not None else None
    except Exception:  # noqa: BLE001
        return None


def _try_search_pick_pokemon_first(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """「山札の上からN枚を見て手札に加える」サーチで、たねポケモンを1枚は拾うようにする。
    config-gated(``search_pick_pokemon_first``)、キーが無ければ常に None=本番不変。

    実ラダーの実例(episode 93473767 T3 row37): むしとりセット(1094、「山札を上から7枚見て、
    その中から【草】ポケモンと基本【草】エネルギーを合計2枚まで手札に加える」)で、候補に
    オーガポン ex(96)が出ていたのに基本草エネルギー2枚を選んだ。ベンチには空きがあり、
    ベンチ要員を1体増やせる場面だった。

    観測可能性の結論(実データで確認済み): 選択肢の ``cardId`` は None だが
    ``state.looking[option.index]`` に実体が入っており、**エージェントは実行時に候補の中身を
    見られる**(`_looking_card_id` の docstring 参照)。方策側は LOOKING を解決しないので、
    この情報を使えるのはこのガードだけ。

    config 例::

        "search_pick_pokemon_first": {"enabled": true}

    発火条件:
      - ``SelectType.CARD`` かつ context が ``contexts``(既定 ["TO_HAND"])
      - **全ての選択肢が LOOKING 由来で実体を解決できる**(1つでも解決できなければ介入しない)
      - ベンチに空きがある(``require_bench_space`` 既定 true)
      - 現在の選択にたねポケモンが1枚も入っていない
      - 未選択の候補にたねポケモンがある

    差し替えは「枠が余っていれば足す」、埋まっていれば「方策スコア最小の選択を1つだけ
    入れ替える」。**入れる/入れ替えるのは1枚だけ**なので、たねポケモンを2枚以上取りに行って
    カードテキスト上不正な選択になることは無い。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    guard = (effective_config or {}).get("search_pick_pokemon_first") or {}
    if not guard.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.CARD or not select.option:
        return None

    from cg.api import SelectContext as _SC
    contexts = set()
    for name in (guard.get("contexts") or ("TO_HAND",)):
        try:
            contexts.add(int(_SC[name]) if isinstance(name, str) else int(name))
        except Exception:  # noqa: BLE001 - 未知の context 名は無視(安全側)
            continue
    if int(getattr(select, "context", -1)) not in contexts:
        return None

    state = obs.current
    card_ids = [_looking_card_id(opt, state) for opt in select.option]
    if any(cid is None for cid in card_ids):
        return None  # LOOKING 以外/実体不明が混ざる select = 介入しない(安全側)

    chosen = [i for i in chosen_action if 0 <= i < len(select.option)]
    if len(chosen) != len(chosen_action):
        return None
    if any(_is_basic_pokemon(card_ids[i]) for i in chosen):
        return None  # 既にポケモンを取っている=介入不要

    me = state.yourIndex
    bench_free = True
    try:
        player = state.players[me]
        bench_free = len(player.bench or []) < int(player.benchMax or 0)
    except Exception:  # noqa: BLE001
        bench_free = False
    if guard.get("require_bench_space", True) and not bench_free:
        return None  # ベンチ満杯=拾っても出せない

    candidates = [
        i for i in range(len(select.option))
        if i not in chosen and _is_basic_pokemon(card_ids[i])
    ]
    if not candidates:
        return None
    pick = candidates[0]  # 方策は LOOKING を識別できないのでスコアで選べない=先頭を決定的に取る

    scores = _policy_scores(obs, select, config)
    if len(chosen) < select.maxCount:
        action = sorted([*chosen, pick])
        appended = True
    else:
        if not chosen:
            return None
        drop = chosen[-1] if scores is None else min(chosen, key=lambda i: scores[i])
        action = sorted([i for i in chosen if i != drop] + [pick])
        appended = False
    if not _is_valid_action(action, select) or action == sorted(chosen):
        return None
    if guard.get("require_bench_space", True) and not bench_free:  # 番人(到達しないはず)
        _GUARD_STATS["search_pick_pokemon_first_misfire_bench_full"] += 1
    _GUARD_STATS["search_pick_pokemon_first_fired"] += 1
    _GUARD_STATS[
        "search_pick_pokemon_first_appended" if appended else "search_pick_pokemon_first_replaced"
    ] += 1
    return action


# --------------------------------------------------------------------------------------
# r11 Fix-I: 特性ドロー(みどりのまい型)の垂れ流し抑制(`ability_draw_brake`)
# (実ラダー32敗中3敗(9.4%)が自分の山札0による敗北。真犯人はリーリエ/クラウン(型は既に
# `low_deck_draw_brake` で対処済み)ではなく、オーガポン みどりのめん ex(96)の特性
# 「みどりのまい」(自分自身にエネ装着+1ドロー、各個体1回/ターン)。盤面に複数体並ぶと
# 毎ターン最大N ドロー=デッキの加速エンジンがそのまま山札消費エンジンになる。episode
# 93517227(壁無し・サイド2-6でリード・毎ターン1サイド獲得中)は T9〜T11 でこの特性を
# 連打し、山札 5→2→0 で敗北した。独立configキーで既定OFF=本番不変。)
# --------------------------------------------------------------------------------------

_R11_GUARD_STATS_ZERO = {
    "ability_draw_brake_fired": 0,               # みどりのまい型の特性ドローを差し替えた回数
    "ability_draw_brake_fired_already_lethal": 0,  # うち「足す前から既にKO可能」で発火した内訳
    "ability_draw_brake_fired_futile": 0,          # うち「最大まで足してもKO不可」で発火した内訳
    "ability_draw_brake_skipped_pivotal": 0,     # このエネ加速がKO成否に寄与しうるので通した回数
    "ability_draw_brake_inconclusive": 0,        # KO探索が完走せず介入を見送った回数
    "ability_draw_brake_no_alternative": 0,      # 代替手が無く介入を見送った回数
    "ability_draw_brake_misfire_no_attack": 0,   # 番人: 攻撃不能なのに発火した回数(=0のはず)
}
_GUARD_STATS_ZERO.update(_R11_GUARD_STATS_ZERO)
_GUARD_STATS.update(_R11_GUARD_STATS_ZERO)


def _try_ability_draw_brake(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """山札が僅少なときの「みどりのまい」型(自分自身にエネ装着+1ドロー)特性連打を、
    今ターンのKO成否に影響しないときだけ止める。config-gated(``ability_draw_brake``)、
    キーが無ければ常に None=本番不変。

    実ラダーの実例(episode 93517227、壁無し・サイド2-6でリードし毎ターン1サイド獲得中):
    T9 row122(山5、アクティブのオーガポンは8エネで相手アクティブ(HP70)へ攻撃すれば
    確実にKOできる=足す必要が無い)/ T11 row146(山5、アクティブ4エネで相手アクティブ
    (HP100)へ攻撃すれば足りる)のどちらも、攻撃せず**別個体**のみどりのまいを選び続け、
    山札は T9末〜T11 で 5→2→0 まで落ちて敗北した。「エネ加速をやめれば勝っていた試合」。

    config 例::

        "ability_draw_brake": {"enabled": true, "deck_threshold": 8, "time_limit_ms": 300}

    発火条件(すべてAND):
      - **絶対例外(最優先)**: 選択肢に ``OptionType.ATTACK`` が1件も無い(=アクティブが
        攻撃コストを満たしていない)なら**何があってもveto しない**。攻撃可能にするための
        エネ加速は最優先(実例: 同episode T13 row158、山0でも攻撃不能な局面では素通しする)。
      - 自分の ``deckCount`` <= ``deck_threshold``(既定8)
      - 選んだ手が **みどりのまい型の ABILITY**(既定対象はオーガポン みどりのめん ex(96)、
        ``card_ids`` で上書き可。``energy_to_active_first`` の ``_TEAL_DANCE_CARD_IDS`` と
        同じ既定値を共有する)
      - そのエネ追加で**今ターンのKO成否が変わらない**。以下のどちらかが成立すること:
          (a) 既に現在の打点でKO可能 = 今の select にある ATTACK 選択肢のどれかを今すぐ
              実行すれば今ターンKOできる(`ability_draw_eval.already_ko_without_more_energy`。
              `search_step` してからの `ko_search.can_ko_from_node`、`boss_target_eval` と
              同じ「search_step→KO判定」の流儀)。足す必要が無いので veto してよい。
          (b) 最大まで足してもKO不可 = このターンの理論上最善手順(エネ加速の連打を含む)
              でも `ko_search.can_ko_this_turn` が KO を発見できない(判定不能ではなく
              完走した上での False)。足しても無駄なので veto してよい。
        どちらも成立しない(=このエネ加速がKO達成に寄与しうる)なら veto しない。

    KO探索の False は「KOできない」と「判定不能」を区別しないため、``low_deck_draw_brake``/
    ``briar_gate`` と同じく**両方の探索が完走したときだけ**その結果を根拠にする。(a) が
    判定不能でも (b) が完走して False を返せば veto してよい((b) 単独で十分条件)。
    両方とも判定不能なら介入しない(安全側)。

    差し替え先は、同型の ABILITY 選択肢(みどりのまい型全個体)を除いた中で方策スコア最大の手。
    代替が無ければ介入しない。

    r13(``use_closed_form_ko``、既定なし=OFF): (a)(b) の判定を**閉形式ファストパス →
    従来の探索 → それも判定不能なら不介入**の3段にする(`_ability_draw_ko_verdict`)。
    実対戦ではオーガポンが複数体並ぶ盤面で (a)(b) の探索が 300ms 予算内に完走せず、
    このガードは120試合で発火0回だった(=実質的に死んでいた)。閉形式
    (`search/closed_form_ko.py`)は「まんようしぐれ = 30+30×両バトル場のエネ数」を
    既知パターンとして持ち、探索なしで (a)=`can_ko_now` / (b)=`is_ko_impossible_this_turn`
    を O(1) で判定する。弱点・抵抗・軽減が結論を変えうる局面では None を返して黙るので、
    段を足しても誤爆側には倒れない。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    brake = (effective_config or {}).get("ability_draw_brake") or {}
    if not brake.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None

    # 絶対例外: アクティブが攻撃コストを満たしていない(=ATTACK選択肢が無い)なら、
    # 攻撃可能にするためのエネ加速が最優先。何があってもvetoしない(最優先で判定)。
    if not any(opt.type == OptionType.ATTACK for opt in select.option):
        return None

    option = select.option[idx]
    if option.type != OptionType.ABILITY:
        return None

    state = obs.current
    me = state.yourIndex
    from ptcg_ai.learning import encoder as _enc
    card_id = _enc._resolve_card_id(option, state)
    ability_ids = {int(x) for x in (brake.get("card_ids") or _TEAL_DANCE_CARD_IDS)}
    if card_id is None or int(card_id) not in ability_ids:
        return None  # みどりのまい型対象外のABILITYには干渉しない

    try:
        deck_count = int(state.players[me].deckCount)
    except Exception:  # noqa: BLE001
        return None
    if deck_count > int(brake.get("deck_threshold", 8)):
        return None

    # r13: 「閉形式ファストパス → 従来の探索 → それも判定不能なら不介入」の3段判定
    # (`_ability_draw_ko_verdict`)。閉形式は ``use_closed_form_ko`` が無ければ素通りするので、
    # キーの無い config では r11 と完全に同一の探索だけが走る。
    veto_reason = _ability_draw_ko_verdict(obs, brake, config)
    if veto_reason is None:
        return None

    scores = _policy_scores(obs, select, config)
    if scores is None:
        return None
    best_alt = best_score = None
    for i, opt in enumerate(select.option):
        if opt.type == OptionType.ABILITY:
            other_id = _enc._resolve_card_id(opt, state)
            if other_id is not None and int(other_id) in ability_ids:
                continue  # 同型の特性はどれも同じ理由(KO成否に無関係)で除外
        if best_score is None or scores[i] > best_score:
            best_score, best_alt = scores[i], i
    if best_alt is None:
        _GUARD_STATS["ability_draw_brake_no_alternative"] += 1
        return None
    action = [best_alt]
    if not _is_valid_action(action, select):
        return None
    if not any(opt.type == OptionType.ATTACK for opt in select.option):  # 番人(到達しないはず)
        _GUARD_STATS["ability_draw_brake_misfire_no_attack"] += 1
    if veto_reason == "already_lethal":
        _GUARD_STATS["ability_draw_brake_fired_already_lethal"] += 1
    else:
        _GUARD_STATS["ability_draw_brake_fired_futile"] += 1
    _GUARD_STATS["ability_draw_brake_fired"] += 1
    return action


# --------------------------------------------------------------------------------------
# r13: KO判定の「閉形式ファストパス」(``use_closed_form_ko``、各ブレーキの中の独立キー)
#
# 問題(実測): `ability_draw_brake`(r11 Fix-I)と `low_deck_draw_brake`(r9 Fix-D)は
# スナップショットの単純局面では正しく発火するが、**実対戦120試合で発火0回**だった。
# 原因は KO 可否の判定に使っているエンジン探索(`ko_search.can_ko_this_turn` /
# `ability_draw_eval.already_ko_without_more_energy`)が、みどりのまいを持つオーガポンが
# 複数体並ぶ分岐の大きい盤面では 300ms 予算内に完走せず、すべて「判定不能」→安全側で
# 介入見送りになっていたこと。= ガードが実質的に死んでいた。
#
# 対策: 本デッキのアタッカーはオーガポン みどりのめん ex(96)のみで、そのワザ
# 「まんようしぐれ」(attackId 120)は 30 + 30×(両バトルポケモンのエネ数)という閉形式。
# 相手アクティブの残HPは観測できるので、KO可否は探索なしで O(1) 判定できる
# (`search/closed_form_ko.py`。実リプレイ250件突合は `kaggle_replays/_probe_closed_form_ko.py`)。
#
# 判定は3段: **閉形式ファストパス → 従来の探索 → それも判定不能なら介入しない**。
# 閉形式は「弱点・抵抗・軽減が結論を変えうる局面では None(判定不能)を返す」保守設計なので、
# 段を足しても誤爆側には倒れない。``use_closed_form_ko`` が無ければ1段目は素通り=r12以前と
# 完全に同一挙動(本番 config は無キー=不変)。
# --------------------------------------------------------------------------------------

# 閉形式が判定不能を返した理由の内訳(「発火が増えないときに原因を特定する」ための計測)。
# `closed_form_ko._Reason` の値と 1:1 で、キーは事前に固定登録する
# (`reset_guard_stats` が `_GUARD_STATS_ZERO` の update なので動的キーは残ってしまうため)。
#
# ``ko_possible_with_prep`` だけは「閉形式が両方の問いに**確定**で答えたが、ガードが要る粒度に
# 届かなかった」ケース(=今すぐ攻撃してもKOできない、しかし準備込みならKOできるかもしれない、
# の中間帯)。これは閉形式の欠陥ではなく、この帯だけは探索でしか詰められないことを意味する。
_CLOSED_FORM_KO_REASONS = (
    "no_state", "no_active", "no_opponent_active", "unknown_attack", "no_payable_attack",
    "opponent_resistance", "damage_modifier_in_play", "unknown_hp", "within_margin", "error",
    "no_attack_option", "ko_possible_with_prep", "unknown",
)

_R13_GUARD_STATS_ZERO = {
    # 閉形式ファストパスの解決内訳(ガード横断の合計)
    "closed_form_ko_true": 0,        # 「今すぐ攻撃すればKOできる」と即断した回数
    "closed_form_ko_false": 0,       # 「攻撃でKOする手は存在しない」と即断した回数
    "closed_form_ko_unresolved": 0,  # 判定不能で従来の探索へフォールバックした回数
    # ガード別(どちらのブレーキがファストパスで解決したか)
    "ability_draw_brake_closed_form_ko": 0,
    "ability_draw_brake_closed_form_no_ko": 0,
    "ability_draw_brake_closed_form_unresolved": 0,
    "low_deck_draw_brake_closed_form_ko": 0,
    "low_deck_draw_brake_closed_form_no_ko": 0,
    "low_deck_draw_brake_closed_form_unresolved": 0,
}
_R13_GUARD_STATS_ZERO.update(
    {f"closed_form_ko_reason_{r}": 0 for r in _CLOSED_FORM_KO_REASONS}
)
_GUARD_STATS_ZERO.update(_R13_GUARD_STATS_ZERO)
_GUARD_STATS.update(_R13_GUARD_STATS_ZERO)


def _closed_form_ko_fastpath(obs: Observation, brake: dict, guard: str) -> bool | None:
    """閉形式によるKO可否のファストパス。``use_closed_form_ko`` が無ければ常に None。

    Returns:
        ``True``  = 今すぐ攻撃すれば相手のバトルポケモンをKOできる(下界で確定)
        ``False`` = このターン攻撃でKOする手は**そもそも存在しない**(上界で確定)
        ``None``  = 判定不能。呼び出し側は従来のエンジン探索へフォールバックする。

    ``True`` を返すには**エンジンが実際に ATTACK 選択肢を出していること**も要求する。
    盤面のスカラーだけでは見えない一時効果(実例: カブルモ 506「スノットアップ」=
    「次の相手の番、ワザが使えない」。episode 93517227 row158 で、こちらの草4エネの
    オーガポンに ATTACK 選択肢が1つも出ていなかった)で攻撃自体が封じられていることが
    あるため。閉形式は静的HP・エネ数しか見ないので、この一点だけはエンジンに聞く。
    """
    if not brake.get("use_closed_form_ko", False):
        return None
    try:
        from ptcg_ai.search import closed_form_ko
        state = obs.current
        me = state.yourIndex
        cf_cfg = {"ko_margin": int(brake.get("ko_margin", 0))}
        report: dict = {}
        blocked = None
        now = closed_form_ko.can_ko_now(state, me, cf_cfg, report)
        if now is True:
            if any(o.type == OptionType.ATTACK for o in obs.select.option):
                _GUARD_STATS["closed_form_ko_true"] += 1
                _GUARD_STATS[f"{guard}_closed_form_ko"] += 1
                return True
            # 攻撃が封じられている(ATTACK選択肢が出ていない)ので True にはできない。
            # この理由は他の判定不能理由より優先して記録する(原因追跡のため)。
            blocked = "no_attack_option"
        report2: dict = {}
        impossible = closed_form_ko.is_ko_impossible_this_turn(state, me, cf_cfg, report2)
        if impossible is True:
            _GUARD_STATS["closed_form_ko_false"] += 1
            _GUARD_STATS[f"{guard}_closed_form_no_ko"] += 1
            return False
        if blocked is None and now is False and impossible is False:
            # 閉形式は両方に確定で答えたが、ガードが要る粒度に届かない中間帯
            # (今すぐではKOできない / しかし準備込みならKOできるかもしれない)。
            # ここだけは探索でしか詰められない=閉形式の欠陥ではないので別ラベルにする。
            blocked = "ko_possible_with_prep"
        reason = str(blocked or report2.get("reason") or report.get("reason") or "unknown")
        if reason not in _CLOSED_FORM_KO_REASONS:
            reason = "unknown"
        _GUARD_STATS[f"closed_form_ko_reason_{reason}"] += 1
        _GUARD_STATS["closed_form_ko_unresolved"] += 1
        _GUARD_STATS[f"{guard}_closed_form_unresolved"] += 1
        return None
    except Exception:  # noqa: BLE001 - ファストパスの失敗が意思決定を止めてはならない
        return None


def _search_ko_verdict(
    obs: Observation, brake: dict, config: dict | None, guard: str
) -> bool | None:
    """従来のエンジン探索(`ko_search.can_ko_this_turn`)によるKO可否。r9/r11 と同一挙動。

    ``True``=KOできる / ``False``=完走した上でKOできない / ``None``=判定不能。
    カウンタ(``{guard}_skipped_ko`` / ``{guard}_inconclusive``)の増やし方も従来と同じで、
    例外時だけは(従来どおり)どのカウンタも動かさずに ``None`` を返す。
    """
    try:
        from ptcg_ai.search import ko_search
        ko_cfg = brake.get("ko_search") or {}
        limit_ms = float(ko_cfg.get("time_limit_ms", brake.get("time_limit_ms", 300)))
        report: dict = {}
        can_ko = bool(ko_search.can_ko_this_turn(
            obs, _model_hidden_state_factory(obs, config), ko_cfg,
            time.perf_counter() + limit_ms / 1000.0, report=report))
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return None
    if can_ko:
        _GUARD_STATS[f"{guard}_skipped_ko"] += 1
        return True
    if brake.get("require_conclusive_ko_search", True) and (
            not report.get("searched") or report.get("aborted")):
        _GUARD_STATS[f"{guard}_inconclusive"] += 1
        return None
    return False


def _ko_verdict_for_brake(
    obs: Observation, brake: dict, config: dict | None, guard: str
) -> bool | None:
    """「閉形式ファストパス → 従来の探索 → それも不能なら None」の3段判定。"""
    verdict = _closed_form_ko_fastpath(obs, brake, guard)
    if verdict is True:
        _GUARD_STATS[f"{guard}_skipped_ko"] += 1  # 従来の探索が True を出したときと同じ扱い
        return True
    if verdict is False:
        return False
    return _search_ko_verdict(obs, brake, config, guard)


def _ability_draw_ko_verdict(
    obs: Observation, brake: dict, config: dict | None
) -> str | None:
    """`_try_ability_draw_brake` の (a)/(b) 判定。veto理由か、vetoしないなら None。

    (a) ``"already_lethal"``: 足す前から既にKO可能=足す必要が無い。
    (b) ``"futile"``: 最大まで足してもKO不可=足しても無駄。

    1段目は閉形式(`closed_form_ko`)。(a) は `can_ko_now`、(b) は
    `is_ko_impossible_this_turn`(攻撃でKOする手がそもそも存在しないことの健全な上界判定)。
    閉形式が判定不能なら2段目として r11 と同じ探索
    (`ability_draw_eval.already_ko_without_more_energy` → `ko_search.can_ko_this_turn`)に落ち、
    それも判定不能なら None(=介入しない、安全側)。
    """
    guard = "ability_draw_brake"
    fast = _closed_form_ko_fastpath(obs, brake, guard)
    if fast is True:
        return "already_lethal"
    if fast is False:
        return "futile"

    from ptcg_ai.search import ability_draw_eval, ko_search
    ko_cfg = brake.get("ko_search") or {}
    limit_ms = float(ko_cfg.get("time_limit_ms", brake.get("time_limit_ms", 300)))
    deadline = time.perf_counter() + limit_ms / 1000.0
    hidden_factory = _model_hidden_state_factory(obs, config)

    # (a) 既に現在の打点でKO可能なら足す必要が無い=veto可。
    try:
        already_ko, _aborted_a = ability_draw_eval.already_ko_without_more_energy(
            obs, hidden_factory, ko_cfg, deadline)
    except Exception:  # noqa: BLE001
        already_ko = False
    if already_ko:
        return "already_lethal"

    # (b) 最大まで足してもKO不可なら足しても無駄=veto可(判定不能なら介入しない)。
    report: dict = {}
    try:
        can_ko_overall = bool(ko_search.can_ko_this_turn(
            obs, hidden_factory, ko_cfg, deadline, report=report))
    except Exception:  # noqa: BLE001
        can_ko_overall, report = False, {"searched": False}
    if can_ko_overall:
        _GUARD_STATS["ability_draw_brake_skipped_pivotal"] += 1
        return None  # このエネ加速がKOに寄与しうる=温存せず使わせる
    if not report.get("searched") or report.get("aborted"):
        _GUARD_STATS["ability_draw_brake_inconclusive"] += 1
        return None  # (a)(b)とも判定不能=安全側で介入しない
    return "futile"


# --------------------------------------------------------------------------------------
# r12: 「テラスタル退避」ゲート(``terastal_rotation``)。
#
# オーガポンex系の特性「テラスタル」=「このポケモンは、ベンチにいるかぎり、ワザのダメージを
# 受けない」(``CardData.tera``)。傷ついたアクティブをベンチに逃がすのは、にげるコストが軽い
# 限り原則ノーリスク。実ラダーの実例:
#   - episode 93408551 T10(決定的): アクティブ HP10/エネ4、ベンチに HP150/エネ4 の健康な
#     個体。ブライア+KOで3サイド取ったのは正しいが、「にげる(コスト1)→HP150の個体で攻撃」
#     でも同じKO・同じ3サイドが取れた上に、HP10の個体をベンチ(ワザダメ無効)へ退避できた。
#     実際はHP10のまま残し、次ターンに70ダメージで落とされて敗北した。
#   - episode 93473767 T11: アクティブHP60/260、ベンチHP240/240・エネ3。KOは元々不可能
#     なので、傷んだ個体をベンチに逃がすのが明確に上。
#   - 反例(発火してはいけない): 相手がドラパルトex(121)の場合、ファントムダイブ
#     (「ダメカン6個を、相手のベンチポケモンに好きなようにのせる」)はダメージ**カウンタを
#     置く**効果であり「ワザのダメージ」ではないため、テラスタルを貫通する(実測: episode
#     93408551 T10 でベンチのオーガポン(HP150→150、maxHp210)が6個=60ダメージ分の
#     ダメカンを受けている)。よって相手の場にこのアーキタイプが居るときは退避の価値が下がる
#     ため発火しない。
# 独立configキー、既定OFF=本番不変。
# --------------------------------------------------------------------------------------

# ドラパルトex(data/JP_Card_Data.csv で実ID確認済み: 121。ワザ「ファントムダイブ」が
# ダメージカウンタをベンチへ直接置く=テラスタルを貫通する)。config
# `terastal_rotation.skip_vs_bench_damage_archetypes` で上書き可能。
_DEFAULT_SKIP_BENCH_DAMAGE_IDS = (121,)

_R12_GUARD_STATS_ZERO = {
    "terastal_rotation_fired": 0,                       # にげるへ差し替えた回数
    "terastal_rotation_skipped_bench_damage_archetype": 0,  # 相手にドラパルトex等が居て見送った回数
    "terastal_rotation_skipped_ko_mismatch": 0,          # にげるとKO成否が変わるため見送った回数
    "terastal_rotation_inconclusive": 0,                 # KO整合性の探索が完走せず見送った回数
    "terastal_rotation_misfire_not_tera": 0,             # 番人: テラスタルでないのに発火した回数(=0のはず)
}
_GUARD_STATS_ZERO.update(_R12_GUARD_STATS_ZERO)
_GUARD_STATS.update(_R12_GUARD_STATS_ZERO)


def _card_is_tera(card_id: int | None) -> bool:
    """テラスタル(ベンチにいる限りワザのダメージを受けない)を持つ種族か。``CardData.tera``
    を使う(ハードコードIDではなく分類。将来オーガポン以外のテラポケモンが増えても効く)。
    未知ID/例外は False(安全側=対象外扱い)。
    """
    if card_id is None:
        return False
    try:
        from ptcg_ai.shared import card_cache
        return bool(card_cache.get_card(int(card_id)).tera)
    except Exception:  # noqa: BLE001
        return False


def _opponent_max_attack_damage(state, me: int) -> int | None:
    """相手アクティブの「想定1発打点」の保守的近似(探索しない)。

    相手アクティブが持つ技のうち ``CardData``/``Attack`` の静的 ``damage`` フィールドの
    最大値を返す。**探索(search_step)は一切しない**。実データでの検証(episode 93408551
    T10: ドラパルトexのファントムダイブ=静的damage 200、実際にアクティブへ与えた直接ダメージも
    200で一致)では妥当な近似になったが、「エネルギー数に応じて追加ダメージ」等のテキスト
    効果(例: オーガポン自身の まんようしぐれ=静的30だが実際は30+30×エネ数)や「手札枚数に
    応じて追加ダメージ」等は静的フィールドに反映されないため、**過小評価になりうる**
    (episode 93473767 T11: Mega Froslass exのResentful Refrainは静的damage 0だが実際は
    200。この局面はもう一方の技Absolute Snowの静的150が閾値を満たしたため判定は変わらな
    かったが、一般には安全側/過小評価の近似であることに注意)。相手アクティブが伏せ中/不明
    なら None(判定不能)。
    """
    try:
        opp = 1 - me
        opp_active = state.players[opp].active or []
        if not opp_active or opp_active[0] is None:
            return None
        card_id = int(opp_active[0].id)
        from ptcg_ai.shared import card_cache
        card = card_cache.get_card(card_id)
        if not card.attacks:
            return 0
        return max(card_cache.get_attack(int(aid)).damage for aid in card.attacks)
    except Exception:  # noqa: BLE001
        return None


def _try_terastal_rotation(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """傷ついたテラスタル持ちのアクティブを、より健康な同族のベンチ個体と入れ替える
    (にげるコストが安く、ベンチにいる間はワザのダメージを受けないため)。
    config-gated(``terastal_rotation``)、キーが無ければ常に None=本番不変。

    実ラダーの実例・反例はモジュール冒頭のコメント参照(episode 93408551 T10 / 93473767 T11 /
    ドラパルトexの反例)。

    config 例::

        "terastal_rotation": {
            "enabled": true, "damage_margin": 0,
            "skip_vs_bench_damage_archetypes": [121]
        }

    発火条件(すべてAND):
      - 選んだ手が ``ATTACK`` または ``END``(このターンの主要な行動を差し置いてまで
        にげるを割り込ませるのはこの2択のときだけ。PLAY/ATTACH/ABILITY/EVOLVE等の他の
        展開が残っている局面には干渉しない)。
      - 自分のアクティブが居て、テラスタル持ち(``_card_is_tera``)。
      - ``RETREAT`` が合法選択肢にある(にげるコストを払えない/エネが足りない局面には
        干渉しない)。
      - 相手の場(アクティブ/ベンチ)に ``skip_vs_bench_damage_archetypes``(既定
        [121]=ドラパルトex)のカードが**いない**(反例ガード。先にチェックすることで
        不要な探索を避ける)。
      - ベンチに、自分のアクティブより**残HPが多い**テラスタル持ちの個体がいる(条件2)。
        複数いれば最もHPが高い個体を退避先候補にする。
      - 自分のアクティブの残HPが「相手アクティブの想定1発打点」(``_opponent_max_attack_damage``
        + ``damage_margin``、既定0)以下(条件1、次ターン落ちる見込み)。
      - にげても**今ターンのKO成否が変わらない**(条件3、``retreat_safety_eval.evaluate``。
        現アクティブでKOが元々無ければ自動的に満たす)。探索が完走しない(``None``)場合は
        介入しない(既存ガードと同じ安全側)。

    差し替え先は退避先の個体を選ぶ処理そのものではなく、この decision を ``RETREAT`` に
    置き換えるだけ(退避後、どのベンチ個体へ交代するか/その後の攻撃は既存の意思決定に委ねる。
    1手で「にげる+攻撃」まで組む必要はない、という仕様)。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    guard = (effective_config or {}).get("terastal_rotation") or {}
    if not guard.get("enabled", False):
        return None
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    if select.option[idx].type not in (OptionType.ATTACK, OptionType.END):
        return None

    state = obs.current
    me = state.yourIndex
    try:
        active_list = state.players[me].active or []
        if not active_list or active_list[0] is None:
            return None
        active = active_list[0]
    except Exception:  # noqa: BLE001
        return None
    if not _card_is_tera(active.id):
        return None  # テラスタルでない個体には退避のノーリスク性が成立しない

    retreat_idx = next((i for i, o in enumerate(select.option) if o.type == OptionType.RETREAT), None)
    if retreat_idx is None:
        return None  # にげるコストを払えない/合法手が無い

    skip_ids = {int(x) for x in (guard.get("skip_vs_bench_damage_archetypes")
                                  or _DEFAULT_SKIP_BENCH_DAMAGE_IDS)}
    if skip_ids:
        opp = 1 - me
        try:
            opp_player = state.players[opp]
            opp_ids = {int(p.id) for p in (opp_player.active or []) if p is not None}
            opp_ids |= {int(p.id) for p in (opp_player.bench or []) if p is not None}
        except Exception:  # noqa: BLE001
            opp_ids = set()
        if opp_ids & skip_ids:
            _GUARD_STATS["terastal_rotation_skipped_bench_damage_archetype"] += 1
            return None

    try:
        bench = state.players[me].bench or []
    except Exception:  # noqa: BLE001
        return None
    candidates = [
        (i, mon) for i, mon in enumerate(bench)
        if mon is not None and _card_is_tera(mon.id) and mon.hp > active.hp
    ]
    if not candidates:
        return None  # 条件2: より健康な退避先が無い
    target_index, _target_mon = max(candidates, key=lambda pair: pair[1].hp)

    margin = int(guard.get("damage_margin", 0))
    expected_hit = _opponent_max_attack_damage(state, me)
    if expected_hit is None:
        return None  # 条件1判定不能(相手アクティブが伏せ中等)=安全側で不介入
    if active.hp > expected_hit + margin:
        return None  # 条件1: まだ危険域ではない

    from ptcg_ai.search import retreat_safety_eval
    ko_cfg = guard.get("ko_search") or {}
    limit_ms = float(ko_cfg.get("time_limit_ms", guard.get("time_limit_ms", 400)))
    deadline = time.perf_counter() + limit_ms / 1000.0
    hidden_factory = _model_hidden_state_factory(obs, config)
    result = retreat_safety_eval.evaluate(
        obs, hidden_factory, retreat_idx, target_index, ko_cfg, deadline)
    if result is None:
        _GUARD_STATS["terastal_rotation_inconclusive"] += 1
        return None
    if not result.get("parity", False):
        _GUARD_STATS["terastal_rotation_skipped_ko_mismatch"] += 1
        return None

    action = [retreat_idx]
    if not _is_valid_action(action, select):
        return None
    if not _card_is_tera(active.id):  # 番人(到達しないはず)
        _GUARD_STATS["terastal_rotation_misfire_not_tera"] += 1
    _GUARD_STATS["terastal_rotation_fired"] += 1
    return action


# --------------------------------------------------------------------------------------
# r14: 壁デッキ検知 → 非exアタッカー(カプ・ブルル)を起用して育てる操縦ルート
#      (``wall_attacker_route``。独立configキー、キー無し=完全不変)
#
# 因果(実測):
#   crustle 系の壁(イワパレス345 / いしずえのめんex117)は、それぞれ「相手がex」
#   「相手が特性持ち」ならワザのダメージを**完全に無効化**する。og_v032 唯一のアタッカー
#   オーガポン みどりのめん ex(96)は **ex かつ 特性持ち**なので両方に打点0。
#   1枚を カプ・ブルル(920、非ex・特性なし・ウッドハンマー固定220)に替えると crustle
#   17.75%→22.75%(n=400、+5.0pt)だが、稼働診断60試合では
#     場出し率 88.3% / **4エネ到達率 6.7%** / エネ0のまま終了 72%
#   = **ブルルは置かれているだけでエネルギーが供給されていない**(+5.0ptはポケモンが5枚に
#   なった副次効果)。既存 `energy_to_active_first` は「アクティブが攻撃不能ならアクティブに
#   寄せる」ガードなので、ブルルがアクティブに出てこない限り発火しない。
#   (注: 同じ診断の「ウッドハンマー0.0回/試合」は `Log` に text フィールドが存在しないため
#    構造的に常に0だった計測バグで、行動の事実ではない。`_diag_bulu_usage.py` で
#    LogType.ATTACK の attackId を数えるよう修正済み。)
#
# 3つの差し替え(すべて veto 連鎖の中で「選ばれた手の差し替え」として実装):
#   (a) `_try_wall_attacker_energy`  エネの行き先をブルルへ / ブルルがバトル場ならNの筋書き優先
#   (b) `_try_wall_attacker_retreat` ブルルが撃てるようになったら にげる
#       `_try_wall_attacker_switch`  交代先の選択でブルルを選ぶ
#   (c) `_try_wall_attacker_attack`  ブルルがバトル場で撃てるのに撃たない手を ATTACK に戻す
#
# 判定部は `search/wall_attacker_route.py`(壁の無効化はカードIDではなくテキストで判定)。
# --------------------------------------------------------------------------------------

# (c)/(a2) で「差し替えてよい」とみなす手の種類。
#
# **仕様(「ATTACK 以外が選ばれていたら ATTACK に差し替える」)からの意図的な絞り込み**:
# このエンジンでは ATTACK を選ぶとその時点でターンが終わる。PLAY/ATTACH/ABILITY/EVOLVE を
# ATTACK で潰すと、そのターンの展開(エネ加速・ドロー・ベンチ展開)を丸ごと捨てることになり、
# 「攻撃を増やす」ために「攻撃以外の価値」を確実に失う。END と RETREAT は
#   END    = このターンもう何もしない(攻撃を撃たないなら純損)
#   RETREAT= せっかく前に出したアタッカーを下げる(このルートの目的そのものを打ち消す)
# のどちらも「差し替えて失うものが無い」ので、既定はこの2種類に限定する。
# 仕様どおり全種類にしたい場合は config の ``attack_override_types`` で明示指定する。
_WALL_ROUTE_DEFAULT_OVERRIDE_TYPES = ("END", "RETREAT")

# (b1) が差し替えを見送る PLAY のカードID。
#
# RETREAT は ATTACK と違い**ターンを終わらせない**(にげた後もう一度 MAIN の選択が来る)ので、
# 差し替えても元の手を失わない=次の decision でそのまま選び直せる。よって (b1) は手の種類を
# 絞らないのが既定。唯一の例外は ボスの指令(1182): 相手のベンチを引きずり出して
# 「壁でない相手」を作る手であり、そのターンのKO計画そのものになりうるため潰さない。
#
# ここを ATTACK/END だけに絞っていた初版は、実対戦10試合の稼働診断で
# 「ブルルが4エネ到達 かつ にげるが合法」な decision が7回あり、そこで選ばれていた手は
# ABILITY 3 / PLAY 2 / RETREAT 2 で **ATTACK と END は0回**、つまり (b1) がほぼ発火しなかった。
_WALL_ROUTE_DEFAULT_RETREAT_SKIP_PLAY_IDS = (_BOSS_ORDER_ID,)

_R14_GUARD_STATS_ZERO = {
    "wall_attacker_energy_fired": 0,            # (a1) エネの付け先をアタッカーへ振り替えた
    "wall_attacker_energy_no_option": 0,        # (a1) アタッカーに付けられる選択肢が無かった
    "wall_attacker_energy_skipped_ko": 0,       # (a1) 現アクティブで+1エネKOが確定=譲った
    "wall_attacker_energy_move_fired": 0,       # (a2) Nの筋書き(ベンチ→バトル場)を優先した
    "wall_attacker_retreat_fired": 0,           # (b1) にげるへ差し替えた
    "wall_attacker_retreat_skipped_ko": 0,      # (b1) 現アクティブでKOが取れる=譲った(番人)
    "wall_attacker_switch_fired": 0,            # (b2) 交代先をアタッカーへ差し替えた
    "wall_attacker_attack_fired": 0,            # (c)  ATTACK へ差し替えた
    "wall_attacker_attack_skipped_self_ko": 0,  # (c)  自傷で自滅するだけなので見送った
    "wall_attacker_misfire_no_wall": 0,         # 番人: 壁が居ないのに発火した(=0のはず)
}
_GUARD_STATS_ZERO.update(_R14_GUARD_STATS_ZERO)
_GUARD_STATS.update(_R14_GUARD_STATS_ZERO)


def _wall_route_gate(obs: Observation, config: dict | None):
    """``wall_attacker_route`` の共通ゲート。``(guard_config, ctx)`` か None を返す。

    キーが無い/``enabled`` でない/相手の場に壁が見えない/自分の場に指定アタッカーが
    居ない、のいずれでも None(=本番configでは常に None で完全不変)。
    """
    if obs.current is None or obs.select is None:
        return None
    effective_config = config if config is not None else _get_config()
    guard = (effective_config or {}).get("wall_attacker_route") or {}
    if not guard.get("enabled", False):
        return None
    from ptcg_ai.search import wall_attacker_route
    ctx = wall_attacker_route.evaluate(obs.current, obs.current.yourIndex, guard)
    if ctx is None:
        return None
    if not ctx.get("wall_ids"):  # 番人(到達しないはず)
        _GUARD_STATS["wall_attacker_misfire_no_wall"] += 1
        return None
    return guard, ctx


def _wall_route_override_types(guard: dict) -> set:
    """``attack_override_types``(既定 END/RETREAT)を OptionType の集合に解決する。"""
    names = guard.get("attack_override_types") or _WALL_ROUTE_DEFAULT_OVERRIDE_TYPES
    out = set()
    for name in names:
        try:
            out.add(OptionType[name] if isinstance(name, str) else OptionType(int(name)))
        except Exception:  # noqa: BLE001 - 未知の名前は無視(安全側=差し替え対象を増やさない)
            continue
    return out


def _wall_route_retreat_may_override(option, state, guard: dict) -> bool:
    """(b1) がこの手を ``RETREAT`` に差し替えてよいか(既定はボスの指令の PLAY 以外すべて)。

    ``retreat_override_types`` を config で明示した場合はその集合だけを対象にする
    (ablation 用。初版の ATTACK/END 限定に戻したいときはこれを使う)。
    """
    types = guard.get("retreat_override_types")
    if types:
        allowed = set()
        for name in types:
            try:
                allowed.add(OptionType[name] if isinstance(name, str) else OptionType(int(name)))
            except Exception:  # noqa: BLE001
                continue
        return option.type in allowed
    if option.type != OptionType.PLAY:
        return True
    skip = {int(x) for x in (guard.get("retreat_skip_play_card_ids")
                             or _WALL_ROUTE_DEFAULT_RETREAT_SKIP_PLAY_IDS)}
    try:
        from ptcg_ai.learning import encoder as _enc
        card_id = _enc._resolve_card_id(option, state)
    except Exception:  # noqa: BLE001
        return False  # 判定不能な PLAY は潰さない(安全側)
    return card_id is None or int(card_id) not in skip


def _try_wall_attacker_energy(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """(a) エネルギーの行き先を、壁に通る非exアタッカー(ブルル)へ寄せる。

    (a1) エネ装着(手貼り/みどりのまい型の特性)の選択で、付け先がアタッカー以外なら
         **アタッカーに付ける選択肢に差し替える**(アタッカーのエネが必要数未満のときだけ)。
         みどりのまいは自分自身にしか付かない仕様なので、実際に振り替わるのは主に手貼り。
    (a2) アタッカーが既にバトル場に居てエネが足りず、END/RETREAT を選ぼうとしていて、
         Nの筋書き(1221、ベンチのエネを最大2個バトル場へ移す)で必要数に届くなら、
         そのサポートの PLAY に差し替える。

    config-gated(``wall_attacker_route``)。キーが無ければ常に None=本番不変。
    """
    resolved = _wall_route_gate(obs, config)
    if resolved is None:
        return None
    guard, ctx = resolved
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    if not ctx["zero_damage_vs_any"]:
        return None  # 現アタッカーで殴れる=このルートの出番ではない
    if ctx["attacker_energy"] >= ctx["attack_cost"]:
        return None  # 既に必要数に到達=これ以上寄せる理由が無い

    from ptcg_ai.search import wall_attacker_route
    state = obs.current
    me = state.yourIndex
    ability_ids = {int(x) for x in (guard.get("energy_ability_card_ids") or _TEAL_DANCE_CARD_IDS)}
    chosen = select.option[idx]

    # --- (a1) エネ装着の付け先を振り替える ---------------------------------------------
    chosen_target = wall_attacker_route.energy_attach_target(chosen, state, ability_ids)
    if chosen_target is not None:
        if wall_attacker_route.option_targets_attacker(chosen, state, ctx, ability_ids):
            return None  # 既にアタッカーへ付けている
        if chosen_target[0] == AreaType.ACTIVE:
            # 付け先が現アクティブの場合だけ、「あと1エネでKOが取れる」なら譲る
            # (閉形式が True を返したときだけ。壁が場に居ると壁自身の無効化テキストが
            # `_damage_modifier_risk` に引っかかって常に判定不能になるため、これは
            # 実質「壁がベンチに居るだけで相手アクティブは殴れる」局面向けの保険)。
            try:
                from ptcg_ai.search import closed_form_ko
                if closed_form_ko.can_ko_with_more_energy(state, me, 1) is True:
                    _GUARD_STATS["wall_attacker_energy_skipped_ko"] += 1
                    return None
            except Exception:  # noqa: BLE001
                pass
        candidates = [
            i for i, opt in enumerate(select.option)
            if wall_attacker_route.option_targets_attacker(opt, state, ctx, ability_ids)
        ]
        if not candidates:
            _GUARD_STATS["wall_attacker_energy_no_option"] += 1
            return None
        best = candidates[0]
        scores = _policy_scores(obs, select, config)
        if scores is not None:
            best = max(candidates, key=lambda i: scores[i])
        action = [best]
        if not _is_valid_action(action, select):
            return None
        _GUARD_STATS["wall_attacker_energy_fired"] += 1
        return action

    # --- (a2) Nの筋書き(ベンチ→バトル場)を優先する -------------------------------------
    if not ctx["active_is_attacker"] or not ctx["attacker_ready_with_move"]:
        return None
    if chosen.type not in _wall_route_override_types(guard):
        return None
    move_ids = {int(x) for x in (guard.get("energy_move_card_ids")
                                 or wall_attacker_route.DEFAULTS["energy_move_card_ids"])}
    from ptcg_ai.learning import encoder as _enc
    candidates = [
        i for i, opt in enumerate(select.option)
        if opt.type == OptionType.PLAY and (_enc._resolve_card_id(opt, state) in move_ids)
    ]
    if not candidates:
        return None
    action = [candidates[0]]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["wall_attacker_energy_move_fired"] += 1
    return action


def _try_wall_attacker_retreat(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """(b1) 壁がバトル場に居て現アクティブの打点が0のとき、撃てるようになったアタッカーへ
    交代するため ``RETREAT`` へ差し替える。config-gated(``wall_attacker_route``)。

    **「今ターンKOが取れるならそちらを優先」の扱い(仕様からの明示的な逸脱)**:
    仕様は「ko_search / closed_form_ko で判定、判定不能なら安全側で介入しない」だが、
    それを字義どおりに実装するとこのルートは**永久に発火しない**。壁のカードテキスト
    ("Prevent all damage ...")が `closed_form_ko._damage_modifier_risk` に必ず引っかかり、
    壁が場に居る局面では `can_ko_now` が常に ``None`` を返すためである。
    そもそもこのルートの発火条件は「相手のバトルポケモンが**自分の現アクティブのワザの
    ダメージを完全に無効化する**ことがテキストで確定している」ことなので、
    **攻撃でKOを取る手は定義上存在しない**(ワザは相手のバトルポケモンにしか当たらない)。
    つまり KO 判定は不要で、無効化の証明の方が強い。それでも取りこぼしが無いことを
    番人として確かめるため `can_ko_now` が ``True`` を返した場合だけ譲る。

    **差し替え対象の手を絞らない理由**: RETREAT は ATTACK と違いターンを終わらせないので、
    どの手を差し替えても次の decision で同じ手を選び直せる(唯一の例外=ボスの指令は
    `_wall_route_retreat_may_override` で除外)。初版は `terastal_rotation` に倣って
    ATTACK/END に限定していたが、実対戦の稼働診断で「ブルルが4エネかつ にげるが合法」な
    decision の選択手は ABILITY/PLAY/RETREAT ばかりで ATTACK・END が0回=ほぼ発火しない
    ことが分かったため、実測に基づいて外した。
    """
    resolved = _wall_route_gate(obs, config)
    if resolved is None:
        return None
    guard, ctx = resolved
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    if not _wall_route_retreat_may_override(select.option[idx], obs.current, guard):
        return None  # ボスの指令(そのターンのKO計画)は潰さない
    if not ctx["zero_damage_vs_active"]:
        return None  # 相手のバトルポケモンが「打点0が確定する壁」でなければ出番なし
    if ctx["active_is_attacker"] or ctx["attacker_bench_index"] is None:
        return None  # 既にアタッカーがバトル場に居る
    if not (ctx["attacker_ready"] or ctx["attacker_ready_with_move"]):
        return None  # 出しても撃てない=ただの的になる

    retreat_idx = next(
        (i for i, o in enumerate(select.option) if o.type == OptionType.RETREAT), None)
    if retreat_idx is None:
        return None  # にげるコストを払えない

    try:
        from ptcg_ai.search import closed_form_ko
        if closed_form_ko.can_ko_now(obs.current, obs.current.yourIndex) is True:
            _GUARD_STATS["wall_attacker_retreat_skipped_ko"] += 1
            return None
    except Exception:  # noqa: BLE001
        pass

    action = [retreat_idx]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["wall_attacker_retreat_fired"] += 1
    return action


def _try_wall_attacker_switch(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """(b2) 「交代先を選ぶ」select(にげた直後 / バトル場がKOされた後)でアタッカーを選ぶ。

    (b1) が ``RETREAT`` に差し替えても、続く交代先の選択で別個体を選んでしまうと意味が無い
    (実測スキーマ: RETREAT → ENERGY(にげるコスト破棄) → CARD/SWITCH の順。
    `retreat_safety_eval.evaluate` が実エンジンで辿っているのと同じ流れ)。

    ボスの指令の対象選択も CARD/SWITCH だが、そちらの選択肢は**相手**のベンチを指す
    (``playerIndex`` が異なる)ので巻き込まない。二重の安全のため
    `_is_boss_target_select` でも除外する。
    """
    resolved = _wall_route_gate(obs, config)
    if resolved is None:
        return None
    guard, ctx = resolved
    select = obs.select
    if select.type != SelectType.CARD or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    if select.context not in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        return None
    if _is_boss_target_select(select, {_BOSS_ORDER_ID}):
        return None
    if ctx["opp_active_wall_id"] is None:
        return None  # 壁がバトル場に居ないなら交代先を強制する理由が無い
    if ctx["active_is_attacker"] or ctx["attacker_bench_index"] is None:
        return None
    if not (ctx["attacker_ready"] or ctx["attacker_ready_with_move"]):
        return None

    from ptcg_ai.search import wall_attacker_route
    target = wall_attacker_route.switch_option_index_for_attacker(
        select, obs.current.yourIndex, ctx)
    if target is None or target == chosen_action[0]:
        return None
    action = [target]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["wall_attacker_switch_fired"] += 1
    return action


def _try_wall_attacker_attack(
    obs: Observation, chosen_action: list[int], config: dict | None = None
) -> list[int] | None:
    """(c) アタッカーがバトル場で撃てるのに撃たない手(既定 END/RETREAT)を ATTACK に戻す。

    自傷で**自分だけが落ちる**撃ち方は避ける: ウッドハンマーの自傷30(閉形式の既知値)で
    アタッカーが気絶し、かつ相手のバトルポケモンをKOできない見込みなら発火しない。
    """
    resolved = _wall_route_gate(obs, config)
    if resolved is None:
        return None
    guard, ctx = resolved
    select = obs.select
    if select.type != SelectType.MAIN or select.maxCount != 1 or len(chosen_action) != 1:
        return None
    idx = chosen_action[0]
    if not (0 <= idx < len(select.option)):
        return None
    if not ctx["active_is_attacker"]:
        return None
    if select.option[idx].type not in _wall_route_override_types(guard):
        return None

    attack_indices = [i for i, o in enumerate(select.option) if o.type == OptionType.ATTACK]
    if not attack_indices:
        return None  # まだ撃てない(エネ不足等)
    best = attack_indices[0]
    scores = _policy_scores(obs, select, config)
    if scores is not None:
        best = max(attack_indices, key=lambda i: scores[i])

    if not _wall_route_attack_is_worth_it(obs.current, obs.current.yourIndex,
                                          select.option[best]):
        _GUARD_STATS["wall_attacker_attack_skipped_self_ko"] += 1
        return None

    action = [best]
    if not _is_valid_action(action, select):
        return None
    _GUARD_STATS["wall_attacker_attack_fired"] += 1
    return action


def _wall_route_attack_is_worth_it(state, me: int, attack_option) -> bool:
    """自傷で自分だけが落ちる撃ち方でないか。判定不能は True(=撃つ、従来どおり)。

    自傷ダメージは `closed_form_ko.self_damage_of`(ウッドハンマー=30)。自傷で自分の
    バトルポケモンが気絶する場合に限り、「相手のバトルポケモンをKOできる見込み」があるかを
    素の打点(弱点・抵抗・軽減を含まない)で確かめ、届かないなら見送る。
    """
    try:
        from ptcg_ai.search import closed_form_ko
        self_damage = closed_form_ko.self_damage_of(getattr(attack_option, "attackId", None))
        if self_damage <= 0:
            return True
        my_active = (state.players[me].active or [None])[0]
        if my_active is None or int(my_active.hp) > self_damage:
            return True  # 自傷では落ちない
        damage = closed_form_ko.estimate_attack_damage(
            state, me, getattr(attack_option, "attackId", None))
        opp_active = (state.players[1 - me].active or [None])[0]
        if damage is None or opp_active is None:
            return True  # 判定不能=従来どおり撃つ
        return int(damage) >= int(opp_active.hp)
    except Exception:  # noqa: BLE001
        return True


def _apply_action_vetoes(obs: Observation, action: list[int], config: dict | None = None) -> list[int]:
    """事後veto(改造ハンマー浪費→ターゲット振替→自滅deckout→ブライア空撃ち→手札連動打点の
    生存ガード)を順に適用。すべて config-gated で既定OFF=本番不変。

    差替が起きたら次のvetoも差替後の手に適用する。abl_5_full ではキーが無く各vetoは即Noneを
    返すので action は不変(本番挙動は変わらない)。`_try_hammer_veto`(MAIN の PLAY 選択)と
    `_try_hammer_redirect`(ENERGY/DISCARD_ENERGY のターゲット選択)は対象 select が排他なので、
    同じ decision で両方が発火することはない(`_try_boss_lethal_gate` と
    `_try_boss_target_redirect` も同様に MAIN / CARD-SWITCH で排他)。

    `_try_briar_gate` を `_try_hand_damage_guard` より前に置くのは、ブライアを弾いた結果
    選ばれた次善手が「手札を増やすサポート」だった場合に、生存ガードがそれも見られるように
    するため(実ラダー 93309236 T9 がまさにこの形)。r9 の2ゲートも同じ理由で
    `_try_hand_damage_guard` の前に置く(ブレーキ/ボスゲートが選び直した手を生存ガードが
    最後に検分する)。**既知の順序制約**: 逆に `_try_hand_damage_guard` がボスの指令へ
    差し替えた場合、その decision ではボスゲートは既に通過済みなので対象評価は走らない
    (=その回は r8 と同じ挙動になる。安全側の素通り)。
    """
    for _veto in (_try_hammer_veto, _try_hammer_redirect, _try_survival,
                  _try_briar_gate, _try_low_deck_draw_brake,
                  _try_boss_lethal_gate, _try_boss_target_redirect,
                  # r10(Fix-F/G/H)。いずれも対象 select が上記と排他か、対象カード種が
                  # 異なるため同じ decision で二重発火しない。生存ガードが最後に検分できる
                  # よう `_try_hand_damage_guard` の前に置く(r9の2ゲートと同じ理由)。
                  _try_tool_stadium_guard, _try_energy_to_active_first,
                  _try_search_pick_pokemon_first,
                  # r11 Fix-I。対象 select は ABILITY(みどりのまい型)の PLAY 決定で、
                  # `_try_energy_to_active_first`(ATTACK選択肢が無い decision だけに反応)とは
                  # 発火条件がATTACK有無で排他。低山札ドローの兄弟ガードとして
                  # `_try_low_deck_draw_brake` の並びに置きたいところだが、後段の
                  # `_try_hand_damage_guard` が差し替え後の手も検分できるよう最後手前に置く。
                  _try_ability_draw_brake,
                  # r12。対象 select は ATTACK/END を選んだ MAIN 決定で、`_try_ability_draw_brake`
                  # (ABILITY選択のみ対象)や `_try_energy_to_active_first`(ATTACK選択肢が無い
                  # decisionのみ対象)とは対象手の種類で排他。差し替え後の手(RETREAT)を
                  # `_try_hand_damage_guard` が最後に検分できるよう同じ理由でその直前に置く。
                  _try_terastal_rotation,
                  # r14 `wall_attacker_route`。ボス系ゲートより**後**に置くのが重要:
                  # ボスゲートが「ボスを出してKOを取る」手に差し替えた decision では
                  # 選ばれた手が PLAY になるため、(b1) の ATTACK/END 条件で自動的に
                  # 素通りする(= KOのチャンスをこのルートが潰さない)。
                  # 4関数は対象 select と対象手で相互排他:
                  #   (a) MAIN でエネ装着を選んだ decision / (b1) MAIN で ATTACK・END かつ
                  #   アクティブが壁に無効化されている / (b2) CARD の SWITCH・TO_ACTIVE /
                  #   (c) MAIN で END・RETREAT かつアクティブがアタッカー本人。
                  # (b1) と (c) は「アクティブがアタッカーか否か」で背反、
                  # (a2) と (c) は「エネが必要数に届いているか否か」で背反。
                  _try_wall_attacker_energy, _try_wall_attacker_retreat,
                  _try_wall_attacker_switch, _try_wall_attacker_attack,
                  _try_hand_damage_guard):
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
    full_deck = _get_deck_for(obs)
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
            full_deck = _get_deck_for(obs)
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

