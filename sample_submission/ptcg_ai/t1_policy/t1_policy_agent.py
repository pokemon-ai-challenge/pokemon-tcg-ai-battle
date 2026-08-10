"""T1(盤面Transformer)方策のAgent入口(NumPy推論、torch不要)。

`ml_policy_agent.py`(既存production)は変更しない。並存する新規モジュール。
lethal探索・attack hybrid・pipeline等の追加ロジックは持たない
(「Transformerそのものの実力を見る」ための提出であり、既存MLPと条件を揃えるため
なるべく素の構成にする)。

T1は単一選択(maxCount==1)の決定点しかモデル化していないため、複数選択決定点は
既存``PolicyModel``(production既定重み)のgreedyへフォールバックする
(蒸留・評価で一貫して使ってきた慣習と同じ)。

デッキはT1の学習に使ったalakazam_morioka(このモジュールと同じディレクトリに置く
``deck.csv``)を使う。
"""

from __future__ import annotations

import os
from pathlib import Path

from cg.api import Observation, OptionType
from ptcg_ai.learning import board_tokens as board_tokens_mod
from ptcg_ai.learning import card_vocab
from ptcg_ai.learning import encoder
from ptcg_ai.learning import legacy_feature_manifest as manifest
from ptcg_ai.learning import t1_numpy_policy as np_policy
from ptcg_ai.learning.policy_model import PolicyModel

_HERE = Path(__file__).resolve().parent
_FEATURE_PROFILE = "fuudin_v4"

_t1_policy = None
_t1_state_pm: PolicyModel | None = None
_fallback_pm: PolicyModel | None = None
_vocab = None
_deck_cache: list[int] | None = None


def _npz_path() -> Path:
    override = os.environ.get("T1_WEIGHTS_NPZ")
    if override:
        return Path(override)
    return _HERE / "weights.npz"


def _get_t1_policy():
    global _t1_policy
    if _t1_policy is None:
        _t1_policy = np_policy.load_t1_numpy_policy(str(_npz_path()))
    return _t1_policy


def _get_t1_state_pm() -> PolicyModel:
    """T1自身の状態エンコード(``encode_state_features``)専用。T1はfuudin_v4(389次元)で
    学習しているため、必ずfuudin_v4のprofileを持つ重み(v40)を使う。

    [重要] production既定の``PolicyModel()``(引数無し)は166次元profileで、
    T1の期待する389次元と食い違う(``run_fixed_pool_eval.py``で一度発見したのと
    同種のバグ)。複数選択のfallback用``_get_fallback_pm()``とは別インスタンスにする。"""
    global _t1_state_pm
    if _t1_state_pm is None:
        _t1_state_pm = PolicyModel(str(_HERE / "v40_teacher.json"))
    return _t1_state_pm


def _get_fallback_pm() -> PolicyModel:
    """複数選択決定点用(T1は単一選択しかモデル化していない)。production既定重み。"""
    global _fallback_pm
    if _fallback_pm is None:
        _fallback_pm = PolicyModel()
    return _fallback_pm


def _get_vocab():
    global _vocab
    if _vocab is None:
        _vocab = card_vocab.load_vocab()
    return _vocab


def _get_deck() -> list[int]:
    global _deck_cache
    if _deck_cache is None:
        text = (_HERE / "deck.csv").read_text(encoding="utf-8")
        _deck_cache = [int(line.strip()) for line in text.splitlines() if line.strip()]
    return _deck_cache


def _greedy_multi_select(pm: PolicyModel, cur, select) -> list[int]:
    scores = pm.score_options_from_state(cur, select)
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))
    if not scores:
        return list(range(count))
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:count]


def agent(obs: Observation) -> list[int]:
    if obs.select is None:
        return _get_deck()

    cur = obs.current
    select = obs.select
    if select.option and select.maxCount == 1:
        return _select_via_t1(cur, select)
    return _greedy_multi_select(_get_fallback_pm(), cur, select)


def _select_via_t1(cur, select) -> list[int]:
    vocab = _get_vocab()
    t1_state_pm = _get_t1_state_pm()  # legacy_state_feat/global特徴の抽出専用(fuudin_v4)

    legacy_state_feat = t1_state_pm.encode_state_features(cur)
    legacy_option_feats = encoder.encode_options_from_state(cur, select)
    option_card_ids = encoder.encode_option_card_ids(cur, select)
    legacy_global_feat = manifest.extract_global_features(legacy_state_feat, _FEATURE_PROFILE)

    tokens = board_tokens_mod.build_board_tokens(cur)

    inputs = np_policy.build_single_decision_inputs(
        tokens, tokens.card_ids, option_card_ids, legacy_option_feats, legacy_global_feat, vocab)

    scores = _get_t1_policy().forward(**inputs)[0]
    n_opt = len(select.option)
    idx = int(scores[:n_opt].argmax())
    if not (0 <= idx < n_opt):
        idx = 0
    return [idx]
