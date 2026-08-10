"""T1のライブ推論ラッパー(T1単独対戦評価、item1)。

**既存MLP経路(``ptcg_ai.learning.policy_model.PolicyModel``・
``kaggle_replays/rl/collect_parallel.py``・``kaggle_replays/rl/distributed/learner.py``)は
一切変更しない。** ここは並存する新規モジュールで、cgエンジンの1決定点
(``State``/``SelectData``)を受け取り、``collect_tokens.py``のoffline収集経路と
**同じ** ``board_tokens``/``encoder``呼び出しでtoken batchを組み立て、
T1 checkpointをforwardし、合法手maskを適用したうえで明示的argmaxで行動を返す。

温度サンプリングは一切行わない(既存``evaluate_pool``の``temperature=0.01``とは違い、
厳密なargmax)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import token_policy_t1 as t1  # noqa: E402
import token_batch as tb  # noqa: E402
import token_shard as ts  # noqa: E402
from ptcg_ai.learning import board_tokens as board_tokens_mod  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.learning import legacy_feature_manifest as manifest  # noqa: E402
from ptcg_ai.learning import card_vocab  # noqa: E402


class EmptyLegalMovesError(ValueError):
    """select.optionが空(合法手0件)で行動を選べない。"""


def build_single_decision_arrays(pm, cur, select, profile_name: str) -> dict:
    """1決定点ぶんの ``token_batch.build_batch`` 互換 arrays dict(n=1)を組み立てる。

    ``collect_tokens._build_decision`` と全く同じ ``board_tokens``/``encoder`` 呼び出しを
    使う(教師スコア計算だけは省く、T1推論に不要なため)。これにより offline収集経路と
    ライブ経路のtoken構築が構造的に同一になる(parity testで直接確認する)。
    """
    if not select.option:
        raise EmptyLegalMovesError("select.optionが空(合法手0件)")

    legacy_state_feat = pm.encode_state_features(cur)
    legacy_option_feats = encoder.encode_options_from_state(cur, select)
    option_card_ids = encoder.encode_option_card_ids(cur, select)
    legacy_global_feat = manifest.extract_global_features(legacy_state_feat, profile_name)

    tokens = board_tokens_mod.build_board_tokens(cur)
    option_target_indices = [
        board_tokens_mod.resolve_option_target_index(opt, cur, tokens) for opt in select.option
    ]

    n_opt = len(option_card_ids)
    n_board = len(tokens.card_ids)

    return {
        "board_counts": np.array([n_board], dtype=np.int32),
        "counts": np.array([n_opt], dtype=np.int32),
        "board_token_numeric_features": (
            np.asarray(tokens.numeric_features, dtype=np.float32)
            if n_board else np.zeros((0, 11), dtype=np.float32)),
        "board_token_card_ids": np.asarray(tokens.card_ids, dtype=np.int32),
        "board_token_zone_ids": np.asarray(tokens.zone_ids, dtype=np.int32),
        "legacy_option_features": np.asarray(legacy_option_feats, dtype=np.float32),
        "option_card_ids": np.asarray(option_card_ids, dtype=np.int32),
        # T1のforwardはteacher_logitsを使わない。build_option_batchの形状合わせのダミー。
        "teacher_logits": np.zeros(n_opt, dtype=np.float32),
        "option_target_token_indices": np.asarray(option_target_indices, dtype=np.int32),
        "chosen": np.zeros(1, dtype=np.int32),
        "legacy_global_features": np.asarray(legacy_global_feat, dtype=np.float32).reshape(1, -1),
    }


def _to_tensors(batch: dict, device) -> dict:
    b, o = batch["board"], batch["option"]
    return dict(
        board_numeric=torch.tensor(b["numeric"], dtype=torch.float32, device=device),
        board_card_idx=torch.tensor(b["card_idx"], dtype=torch.long, device=device),
        board_zone_ids=torch.tensor(b["zone_ids"], dtype=torch.long, device=device),
        board_owner_ids=torch.tensor(b["owner_ids"], dtype=torch.long, device=device),
        board_mask=torch.tensor(b["mask"], dtype=torch.bool, device=device),
        global_features=torch.tensor(batch["legacy_global_features"], dtype=torch.float32, device=device),
        option_features=torch.tensor(o["legacy_option_features"], dtype=torch.float32, device=device),
        option_card_idx=torch.tensor(o["card_idx"], dtype=torch.long, device=device),
        option_mask=torch.tensor(o["mask"], dtype=torch.bool, device=device),
    )


def t1_scores(model, vocab, pm, cur, select, profile_name: str, device="cpu") -> np.ndarray:
    """1決定点の全合法手ぶんのT1スコア(paddingなし、shape=(n_opt,))を返す。"""
    arrays = build_single_decision_arrays(pm, cur, select, profile_name)
    batch = tb.build_batch(arrays, vocab)
    inputs = _to_tensors(batch, device)
    model.eval()
    with torch.no_grad():
        scores = model(**inputs)[0]
    n_opt = int(arrays["counts"][0])
    return scores[:n_opt].cpu().numpy()


def t1_select_index(model, vocab, pm, cur, select, profile_name: str, device="cpu") -> int:
    """合法手mask適用後の明示的argmaxで行動indexを返す(温度サンプリングなし)。"""
    scores = t1_scores(model, vocab, pm, cur, select, profile_name, device=device)
    if scores.size == 0:
        raise EmptyLegalMovesError("select.optionが空(合法手0件)")
    return int(np.argmax(scores))


def load_t1_for_inference(checkpoint_path, device="cpu", meta_source_checkpoint=None):
    """checkpointを読み込み、現在の環境(vocab/profile/manifest)と照合する。
    ``T1OptionPolicy``/``load_checkpoint``は変更しない(既存T1.1契約をそのまま使う)。

    ``train_ppo_t1.save_ppo_checkpoint``で保存したPPO checkpointは
    ``feature_profile``等のメタ情報を持たない(model_state_dict/optimizer_state_dict/
    step/rng_state/extraのみ)。この場合は``meta_source_checkpoint``
    (既定: 蒸留seed=1のcheckpoint。PPOはTransformer構成・profileを変更しないため
    常に一致する)からメタ情報を借りてモデルの箱を作り、重みだけPPO checkpointの
    ``model_state_dict``で上書きする。"""
    vocab = card_vocab.load_vocab()
    payload_probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "feature_profile" in payload_probe:
        profile_name = payload_probe["feature_profile"]
        model, payload = t1.load_checkpoint(
            checkpoint_path,
            token_schema_version=ts.TOKEN_SHARD_FORMAT,
            vocabulary_version=vocab.version, vocabulary_hash=vocab.hash,
            card_vocab_size=vocab.size, feature_profile=profile_name,
            global_feature_manifest_hash=manifest.manifest_hash(profile_name),
            map_location=device)
        model.to(device)
        model.eval()
        return model, vocab, profile_name, payload

    # PPO checkpoint形式: メタ情報は meta_source_checkpoint から借りる。
    if meta_source_checkpoint is None:
        meta_source_checkpoint = (Path(__file__).resolve().parent / "runs" / "distill_v40"
                                  / "train" / "mixed_teacher_t1_seed1" / "best.pt")
    model, vocab, profile_name, payload = load_t1_for_inference(meta_source_checkpoint, device=device)
    # [value head追加(critic+GAE実験)との互換] value_fc1/value_fc2が追加される前に
    # 保存されたPPO checkpointは``model_state_dict``にこのキーが無い。
    # token_policy_t1.load_checkpointと同じ許容規則(value_fc*の欠落だけは許す、
    # それ以外の欠落や未知のキーはエラー)をここでも適用する。
    missing, unexpected = model.load_state_dict(payload_probe["model_state_dict"], strict=False)
    if unexpected:
        raise t1.CheckpointMismatchError(f"PPO checkpointに未知のパラメータがある: {unexpected}")
    if missing and not all(k.startswith("value_fc") for k in missing):
        raise t1.CheckpointMismatchError(f"PPO checkpointに想定外の欠落パラメータがある: {missing}")
    model.to(device)
    model.eval()
    return model, vocab, profile_name, payload_probe
