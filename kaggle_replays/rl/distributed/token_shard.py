"""新shard形式(トークン化 T0)の入出力。

設計書: ``transformer-tokenized-encoder-design.md`` T0。
調査根拠: ``transformer-tokenized-encoder-investigation.md`` §1。

## 既存shard形式(``common.py`` ``SHARD_FORMAT``)との関係

**既存形式は変更しない。** 既存の ``state``/``opts``/``cids``/``counts`` は
盤面を166〜389次元に要約し終えたあとの数値で、盤面トークン化に必要な生の情報
(スロットごとのcard id、選択肢→盤面トークンのポインタ)を復元できないことが分かっている
(investigation.md §1)。そのため本モジュールは既存 ``common.py`` を置き換えず、
**並存する新しいshard形式**として実装する。

新形式は既存の集約特徴(``legacy_state_features``/``legacy_option_features``/
``option_card_ids``)を**そのまま同梱する(dual-write)**。これにより:

- 新形式のshard1本から、旧形式相当のデータと新しいトークンデータの両方が取り出せる。
- 旧形式のshard reader(``common.py`` の ``read_shard``)は一切変更しないので、
  既存パイプライン(learner.py 等)は無傷のまま。

## トークンのポインタについて

``option_target_token_indices`` は「その選択肢が対象とする盤面トークンの、
**同じ決定点内でのローカルindex**」を保持する(全shardを通した通し番号ではない。
``board_counts`` から決定点ごとの範囲を復元し、その範囲内でのオフセットとして解釈する。
既存 ``learner.py`` の ``counts``/``offsets``/``seg`` と同じ考え方)。対象を持たない
選択肢は ``board_tokens.NO_TARGET``(-1)。

``Pokemon.serial``/``Card.serial`` 自体はここでは保存しない。ポインタ解決は
収集時(トークン列構築と同じタイミング)に完了させ、保存するのは解決済みの
ローカルindexだけで十分(serialは決定点ごとに使い捨てる中間キーであり、
保存後の学習で再利用する必要がない)。
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT / "sample_submission") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "sample_submission"))
from ptcg_ai.learning.board_tokens import BOARD_TOKEN_NUMERIC_DIM  # noqa: E402

# シャード形式のバージョン。中身の並び・意味を変えたら上げる。
# v2(T0.1): pointer監査用のdebug配列(任意)を追加。owner_idは保存しない
# (zone_idから決定的に導出する。board_tokens.owner_of_zone参照)。
TOKEN_SHARD_FORMAT = "ptcg-rl-token-shard/2"

# 複数shardを束ねるときに一致していなければならないmetaキー(D2.1で拡張)。
#   token_shard_format          : 配列の意味そのものが違う可能性がある(schema version)
#   extended_features_profile   : legacy_global_featuresの次元が違う
#   teacher_model_sha256        : 別モデルが出した教師logitsが混ざる(方策設定込み。
#                                  重みファイルが同一なら学習済みパラメータ・
#                                  consequence_fields等の設定も同一と判断できる)
#   vocabulary_version/hash     : card idのembedding index体系が違う
#   card_vocab_size             : 上と表裏(語彙サイズが違えばhashも通常違うが、
#                                  念のため独立に検査する)
#   global_feature_manifest_hash: legacy_global_featuresの列の意味が違う
#   legacy_state_dim/legacy_global_dim/legacy_option_dim/board_token_numeric_dim
#                                : 配列のshapeそのものの前提が違う
#   temperature                 : 教師logitsを収集した際のsoftmax温度(方策設定の一部)。
#                                  異なる温度で収集したshardを混ぜるとold_logpの意味が
#                                  変わる(教師の温度付き分布が違うため)
#   action_selection_mode       : 行動選択方式(現状は常に"softmax_temperature"。
#                                  将来argmax等の別方式を追加したときに、temperatureの
#                                  値だけでは区別できない収集方針の違いを検出するため)
_MERGE_CONSISTENCY_KEYS = (
    "token_shard_format", "extended_features_profile", "teacher_model_sha256",
    "vocabulary_version", "vocabulary_hash", "card_vocab_size",
    "global_feature_manifest_hash", "legacy_state_dim", "legacy_global_dim",
    "legacy_option_dim", "board_token_numeric_dim", "temperature",
    "action_selection_mode",
)

_MISSING = object()  # meta.get()の既定値と区別するための番兵(「キーが無い」を検出するため)


class ShardMetaMismatchError(ValueError):
    pass


def concatenate_shards(shard_paths: list) -> tuple[dict, dict]:
    """複数の ``.npz`` shardを読み込み、1つの ``arrays`` に結合する。

    決定点単位・選択肢単位・盤面トークン単位・試合単位の各配列を、それぞれ単純に
    axis0で連結するだけでよい(``counts``/``board_counts``/``lengths`` が
    ragged構造の復元に必要な情報を持っているため、shard境界をまたいでも
    ``decision_slice``/``iter_decisions``/``token_batch`` がそのまま使える)。

    ``_MERGE_CONSISTENCY_KEYS`` が全shardで一致しなければ ``ShardMetaMismatchError``。
    """
    if not shard_paths:
        raise ValueError("shard_pathsが空")
    all_arrays, all_meta = [], []
    for p in shard_paths:
        arrays, meta = read_token_shard(p)
        all_arrays.append(arrays)
        all_meta.append(meta)

    for key in _MERGE_CONSISTENCY_KEYS:
        raw_values = [m.get(key, _MISSING) for m in all_meta]
        if any(v is _MISSING for v in raw_values):
            missing_at = [str(p) for p, v in zip(shard_paths, raw_values) if v is _MISSING]
            raise ShardMetaMismatchError(
                f"shardに{key}が無い(古い形式のshardの可能性): {missing_at}")
        if len(set(raw_values)) > 1:
            raise ShardMetaMismatchError(
                f"shard間で{key}が一致しない: {set(raw_values)}\n  対象: {[str(p) for p in shard_paths]}")

    has_debug = all("debug_resolver" in a for a in all_arrays)
    has_global = all("legacy_global_features" in a for a in all_arrays)

    def _cat(key):
        return np.concatenate([a[key] for a in all_arrays])

    keys = ["legacy_state_features", "legacy_option_features", "option_card_ids", "counts",
           "board_token_numeric_features", "board_token_card_ids", "board_token_zone_ids",
           "board_counts", "option_target_token_indices", "teacher_logits", "chosen",
           "old_logp", "lengths", "rewards", "opp"]
    merged = {k: _cat(k) for k in keys}
    if has_global:
        merged["legacy_global_features"] = _cat("legacy_global_features")
    if has_debug:
        for k in ("board_token_serial", "debug_target_serial", "debug_target_card_id", "debug_resolver"):
            merged[k] = _cat(k)

    meta = dict(all_meta[0])
    meta["n_decisions"] = int(sum(m["n_decisions"] for m in all_meta))
    meta["n_trajectories"] = int(sum(m["n_trajectories"] for m in all_meta))
    meta["merged_from"] = [str(p) for p in shard_paths]
    meta["n_shards_merged"] = len(shard_paths)
    return merged, meta


DATASET_MANIFEST_SCHEMA_VERSION = 1


def build_dataset_manifest(shard_paths: list) -> dict:
    """複数shardの「束ね方」そのものを一意に固定するmanifestを作る(D2.1)。

    train/val分割(``split.json``)が正しいshard集合・正しい順序に対して行われた
    ものかを、``out_dir``のような不安定な識別子ではなく、この manifest の
    ``manifest_hash`` で確認できるようにする(§dataset manifestとsplitの固定)。
    """
    import hashlib
    entries = []
    for p in shard_paths:
        p = Path(p)
        _, meta = read_token_shard(p)
        entries.append({
            "path": str(p), "sha256": sha256_file(p),
            "n_trajectories": meta["n_trajectories"], "n_decisions": meta["n_decisions"],
        })
    payload = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "shards": entries,
        "total_trajectories": sum(e["n_trajectories"] for e in entries),
        "total_decisions": sum(e["n_decisions"] for e in entries),
    }
    hash_input = json.dumps(
        [(e["path"], e["sha256"]) for e in entries] + [payload["schema_version"]],
        ensure_ascii=False)
    payload["manifest_hash"] = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    return payload


def sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# resolve_option_target_index_debug() の resolver 文字列 <-> int の対応
# (npzは文字列配列より整数配列の方が扱いやすいため、保存時は整数化する)。
_RESOLVER_NAMES = [
    "skill_serial", "active_implicit", "area_index", "in_play_index",
    "no_target_no_fields", "no_target_unresolved_zone",
]
_RESOLVER_TO_INT = {name: i for i, name in enumerate(_RESOLVER_NAMES)}


def write_token_shard(path: Path, decisions: list[dict], trajectories: list[dict], meta: dict) -> Path:
    """トークン形式のshardを書き出す。

    Args:
        decisions: 1決定点1件の dict のリスト。各要素は次のキーを持つ
            (``collect_tokens.py`` が構築する):
                legacy_state_feat: list[float]           legacy_state_features の1行
                legacy_option_feats: list[list[float]]    legacy_option_features の該当範囲
                option_card_ids: list[int]
                board_numeric_feats: list[list[float]]    board_token_numeric_features の該当範囲
                board_card_ids: list[int]
                board_zone_ids: list[int]
                option_target_indices: list[int]          board_tokens.NO_TARGET を含みうる
                teacher_logits: list[float]                option_card_ids と同じ並び・同じ長さ
                chosen_idx: int
                logprob: float
                (任意・pointer監査用) debug_target_serial: list[int|None]
                (任意) debug_target_card_id: list[int|None]
                (任意) debug_resolver: list[str]              _RESOLVER_NAMES のいずれか
        trajectories: 1試合1件の dict のリスト。``{"steps": [...], "reward": float, "opp": int}``
            の形(``kaggle_replays/rl/collect_parallel.py`` の戻り値と同じ構造)。
            ``lengths``/``rewards``/``opp`` はここから作る。
        meta: 収集条件(run_id・generation・teacher_model_sha256等)。呼び出し側が用意する。

    Returns:
        書き出したファイルのパス。
    """
    if not decisions:
        raise SystemExit("収集できた決定点が0件。重みのパスや試合数を確認する。")

    option_counts = np.array([len(d["option_card_ids"]) for d in decisions], dtype=np.int32)
    board_counts = np.array([len(d["board_card_ids"]) for d in decisions], dtype=np.int32)

    for d in decisions:
        if len(d["teacher_logits"]) != len(d["option_card_ids"]):
            raise ValueError(
                "teacher_logits の長さが option_card_ids と一致しない: "
                f"{len(d['teacher_logits'])} != {len(d['option_card_ids'])}")
        if len(d["option_target_indices"]) != len(d["option_card_ids"]):
            raise ValueError(
                "option_target_indices の長さが option_card_ids と一致しない: "
                f"{len(d['option_target_indices'])} != {len(d['option_card_ids'])}")

    has_global = "legacy_global_feat" in decisions[0]

    arrays = {
        # --- dual-write: 既存(legacy)形式相当のデータ ---
        "legacy_state_features": np.array([d["legacy_state_feat"] for d in decisions], dtype=np.float32),
        "legacy_option_features": np.concatenate(
            [np.asarray(d["legacy_option_feats"], dtype=np.float32) for d in decisions]),
        "option_card_ids": np.concatenate(
            [np.asarray(d["option_card_ids"], dtype=np.int32) for d in decisions]),
        "counts": option_counts,
    }
    if has_global:
        arrays["legacy_global_features"] = np.array(
            [d["legacy_global_feat"] for d in decisions], dtype=np.float32)
    arrays.update({
        # --- 新規: 盤面トークン ---
        # BOARD_TOKEN_NUMERIC_DIM は board_tokens.py の定数(ポケモン1体あたり11次元)で固定。
        # 決定点によってトークン数(行数)は変わるが、1トークンあたりの次元数は常に一定なので、
        # 空の決定点(board_numeric_feats=[])が混ざっても形は (0, BOARD_TOKEN_NUMERIC_DIM) で揃う。
        "board_token_numeric_features": np.concatenate([
            np.asarray(d["board_numeric_feats"], dtype=np.float32).reshape(-1, BOARD_TOKEN_NUMERIC_DIM)
            for d in decisions
        ]) if decisions else np.zeros((0, BOARD_TOKEN_NUMERIC_DIM), dtype=np.float32),
        "board_token_card_ids": np.concatenate(
            [np.asarray(d["board_card_ids"], dtype=np.int32) for d in decisions]),
        "board_token_zone_ids": np.concatenate(
            [np.asarray(d["board_zone_ids"], dtype=np.int32) for d in decisions]),
        "board_counts": board_counts,
        # --- ポインタ・教師ロジット ---
        "option_target_token_indices": np.concatenate(
            [np.asarray(d["option_target_indices"], dtype=np.int32) for d in decisions]),
        "teacher_logits": np.concatenate(
            [np.asarray(d["teacher_logits"], dtype=np.float32) for d in decisions]),
        # --- 選択・確率 ---
        "chosen": np.array([d["chosen_idx"] for d in decisions], dtype=np.int32),
        "old_logp": np.array([d["logprob"] for d in decisions], dtype=np.float32),
        # --- 試合単位 ---
        "lengths": np.array([len(tr["steps"]) for tr in trajectories], dtype=np.int32),
        "rewards": np.array([tr["reward"] for tr in trajectories], dtype=np.float32),
        "opp": np.array([int(tr.get("opp", 0)) for tr in trajectories], dtype=np.int16),
    })

    # --- pointer監査情報(任意、--debug-pointers で収集した場合のみ) ---
    has_debug = "debug_resolver" in decisions[0]
    if has_debug:
        arrays["board_token_serial"] = np.concatenate([
            np.asarray(d["board_serials"], dtype=np.int32) for d in decisions])
        arrays["debug_target_serial"] = np.concatenate([
            np.asarray([-1 if s is None else s for s in d["debug_target_serial"]], dtype=np.int32)
            for d in decisions])
        arrays["debug_target_card_id"] = np.concatenate([
            np.asarray([-1 if c is None else c for c in d["debug_target_card_id"]], dtype=np.int32)
            for d in decisions])
        arrays["debug_resolver"] = np.concatenate([
            np.asarray([_RESOLVER_TO_INT[r] for r in d["debug_resolver"]], dtype=np.int32)
            for d in decisions])

    meta = dict(meta)
    if has_debug:
        meta["resolver_kind_names"] = _RESOLVER_NAMES
    meta.update({
        "token_shard_format": TOKEN_SHARD_FORMAT,
        "n_decisions": int(len(decisions)),
        "n_trajectories": int(len(trajectories)),
        "legacy_state_dim": int(arrays["legacy_state_features"].shape[1]),
        "legacy_option_dim": int(arrays["legacy_option_features"].shape[1]) if arrays["legacy_option_features"].size else 0,
        "legacy_global_dim": int(arrays["legacy_global_features"].shape[1]) if has_global else None,
        "board_token_numeric_dim": BOARD_TOKEN_NUMERIC_DIM,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
    })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.part")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    tmp.replace(path)
    return path


def read_token_shard(path: Path) -> tuple[dict, dict]:
    """(arrays, meta) を返す。"""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    if meta.get("token_shard_format") != TOKEN_SHARD_FORMAT:
        raise ValueError(
            f"shard形式が違う({meta.get('token_shard_format')} != {TOKEN_SHARD_FORMAT})")
    return arrays, meta


def decision_slice(arrays: dict, decision_index: int) -> dict:
    """1決定点ぶんのデータを取り出す(テスト・デバッグ用)。

    ``counts``/``board_counts`` から offset を復元し、その決定点に属する
    選択肢・盤面トークンの範囲だけを切り出す。
    """
    counts = arrays["counts"]
    board_counts = arrays["board_counts"]
    opt_off = int(counts[:decision_index].sum())
    opt_n = int(counts[decision_index])
    board_off = int(board_counts[:decision_index].sum())
    board_n = int(board_counts[decision_index])
    out = {
        "legacy_state_feat": arrays["legacy_state_features"][decision_index],
        "legacy_option_feats": arrays["legacy_option_features"][opt_off:opt_off + opt_n],
        "option_card_ids": arrays["option_card_ids"][opt_off:opt_off + opt_n],
        "board_numeric_feats": arrays["board_token_numeric_features"][board_off:board_off + board_n],
        "board_card_ids": arrays["board_token_card_ids"][board_off:board_off + board_n],
        "board_zone_ids": arrays["board_token_zone_ids"][board_off:board_off + board_n],
        "option_target_indices": arrays["option_target_token_indices"][opt_off:opt_off + opt_n],
        "teacher_logits": arrays["teacher_logits"][opt_off:opt_off + opt_n],
        "chosen_idx": int(arrays["chosen"][decision_index]),
        "logprob": float(arrays["old_logp"][decision_index]),
    }
    if "legacy_global_features" in arrays:
        out["legacy_global_feat"] = arrays["legacy_global_features"][decision_index]
    if "debug_resolver" in arrays:
        out["board_serials"] = arrays["board_token_serial"][board_off:board_off + board_n]
        out["debug_target_serial"] = arrays["debug_target_serial"][opt_off:opt_off + opt_n]
        out["debug_target_card_id"] = arrays["debug_target_card_id"][opt_off:opt_off + opt_n]
        out["debug_resolver"] = arrays["debug_resolver"][opt_off:opt_off + opt_n]
    return out


def iter_decisions(arrays: dict):
    """全決定点ぶんを ``decision_slice`` で1件ずつ返すジェネレータ(counts/board_countsに
    基づく決定点単位への復元。T0.1完了条件のE2Eテストで使う)。"""
    n = len(arrays["counts"])
    for i in range(n):
        yield decision_slice(arrays, i)
