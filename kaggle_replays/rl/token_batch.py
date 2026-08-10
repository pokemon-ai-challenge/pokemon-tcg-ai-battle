"""token shard(``distributed/token_shard.py``)からpaddingされたバッチを作る(T0.1)。

T1(Transformer本体)がtorch tensorに変換する前段。ここではnumpyのみで完結させる
(T0.1はTransformer本体を実装しないため、torch依存を持ち込まない)。

profileが異なるshard(166次元 vs 389次元)を無言で混ぜないよう、バッチ化の前に
``validate_profile_consistency`` で必ず揃っているか確認すること。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE / "distributed"), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg_ai.learning.board_tokens import owner_of_zone, ZONE_SELF_ACTIVE  # noqa: E402
from ptcg_ai.learning.card_vocab import CardVocab, PAD_INDEX  # noqa: E402

# paddingで埋める盤面トークンのzone_id。owner_of_zone()に通せる値である必要があるだけで、
# mask=Falseの位置なので実際の意味は持たない(計算には使われない)。
_PAD_ZONE_ID = ZONE_SELF_ACTIVE


class ProfileMismatchError(ValueError):
    pass


def validate_profile_consistency(meta_list: list[dict]) -> str | None:
    """複数shardのmetaを束ねる前に、extended_features_profileが揃っているか確認する。

    揃っていなければ ``ProfileMismatchError``(166次元shardと389次元shardを
    無言で混ぜない・無言でpaddingしない、というT0.1の決定に対応)。
    """
    profiles = {m.get("extended_features_profile") for m in meta_list}
    if len(profiles) > 1:
        raise ProfileMismatchError(
            f"extended_features_profileが揃っていない shard を混ぜようとした: {profiles}\n"
            "  166次元(profile=None)と389次元(profile='fuudin_v4'等)を混在させると、"
            "legacy_global_featuresの次元・意味が決定点ごとに変わってしまう。")
    return next(iter(profiles))


def _pad_2d(rows: list[np.ndarray], max_len: int, fill=0.0) -> np.ndarray:
    """可変長の行(各要素が (k_i, d) or (k_i,))を (n, max_len, d) or (n, max_len) に詰める。"""
    if not rows:
        return np.zeros((0, max_len), dtype=np.float32)
    if rows[0].ndim == 1:
        out = np.full((len(rows), max_len), fill, dtype=rows[0].dtype)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
        return out
    d = rows[0].shape[1]
    out = np.full((len(rows), max_len, d), fill, dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def build_board_batch(arrays: dict, vocab: CardVocab) -> dict:
    """盤面トークンをpaddingしたバッチにする。

    Returns:
        numeric:   (n, max_board, 11) float32。paddingは0。
        card_idx:  (n, max_board) int32。vocab.index_of() 済み(未登録カードはUNK)。paddingはPAD_INDEX。
        zone_ids:  (n, max_board) int32。paddingは _PAD_ZONE_ID(mask=Falseなので値は無意味)。
        owner_ids: (n, max_board) int32。zone_idsから owner_of_zone() で決定的に導出(保存はしない)。
        mask:      (n, max_board) bool。True=実トークン、False=padding。
    """
    board_counts = arrays["board_counts"]
    n = len(board_counts)
    max_board = int(board_counts.max()) if n and board_counts.max() > 0 else 0

    numeric_rows, card_id_rows, zone_rows = [], [], []
    off = 0
    for cnt in board_counts:
        cnt = int(cnt)
        numeric_rows.append(arrays["board_token_numeric_features"][off:off + cnt])
        card_id_rows.append(arrays["board_token_card_ids"][off:off + cnt])
        zone_rows.append(arrays["board_token_zone_ids"][off:off + cnt])
        off += cnt

    numeric = _pad_2d(numeric_rows, max_board, fill=0.0) if max_board else np.zeros((n, 0, 11), dtype=np.float32)
    zone_ids = _pad_2d(zone_rows, max_board, fill=_PAD_ZONE_ID).astype(np.int32) if max_board else np.zeros((n, 0), dtype=np.int32)

    mask = np.zeros((n, max_board), dtype=bool)
    for i, cnt in enumerate(board_counts):
        mask[i, :int(cnt)] = True

    card_idx = np.full((n, max_board), PAD_INDEX, dtype=np.int32)
    for i, row in enumerate(card_id_rows):
        for j, cid in enumerate(row):
            card_idx[i, j] = vocab.index_of(int(cid))

    owner_ids = np.vectorize(owner_of_zone)(zone_ids).astype(np.int32) if max_board else zone_ids

    return {"numeric": numeric, "card_idx": card_idx, "zone_ids": zone_ids,
            "owner_ids": owner_ids, "mask": mask}


def build_option_batch(arrays: dict, vocab: CardVocab) -> dict:
    """選択肢(既存65次元特徴+card id embedding入力+教師logits+pointer)をpaddingしたバッチにする。

    Returns:
        legacy_option_features: (n, max_opt, 65) float32。paddingは0。
        card_idx: (n, max_opt) int32。vocab index(選択肢の対象card idを既存card embeddingに使う分)。
        teacher_logits: (n, max_opt) float32。paddingは -inf(softmaxで無視されるように)。
        target_indices: (n, max_opt) int32。board_tokens.NO_TARGET(-1)を含む。
        mask: (n, max_opt) bool。
        chosen: (n,) int32。
    """
    counts = arrays["counts"]
    n = len(counts)
    max_opt = int(counts.max()) if n and counts.max() > 0 else 0

    opt_rows, cid_rows, logit_rows, target_rows = [], [], [], []
    off = 0
    for cnt in counts:
        cnt = int(cnt)
        opt_rows.append(arrays["legacy_option_features"][off:off + cnt])
        cid_rows.append(arrays["option_card_ids"][off:off + cnt])
        logit_rows.append(arrays["teacher_logits"][off:off + cnt])
        target_rows.append(arrays["option_target_token_indices"][off:off + cnt])
        off += cnt

    legacy_option_features = _pad_2d(opt_rows, max_opt, fill=0.0) if max_opt else np.zeros((n, 0, 65), dtype=np.float32)
    teacher_logits = _pad_2d(logit_rows, max_opt, fill=-np.inf) if max_opt else np.zeros((n, 0), dtype=np.float32)
    target_indices = _pad_2d(target_rows, max_opt, fill=-1).astype(np.int32) if max_opt else np.zeros((n, 0), dtype=np.int32)

    mask = np.zeros((n, max_opt), dtype=bool)
    for i, cnt in enumerate(counts):
        mask[i, :int(cnt)] = True

    card_idx = np.full((n, max_opt), PAD_INDEX, dtype=np.int32)
    for i, row in enumerate(cid_rows):
        for j, cid in enumerate(row):
            card_idx[i, j] = vocab.index_of(int(cid))

    return {"legacy_option_features": legacy_option_features, "card_idx": card_idx,
            "teacher_logits": teacher_logits, "target_indices": target_indices,
            "mask": mask, "chosen": arrays["chosen"].astype(np.int32)}


def build_batch(arrays: dict, vocab: CardVocab) -> dict:
    """1shardぶんの決定点をまとめてpaddingしたバッチにする(board + option + global)。"""
    board = build_board_batch(arrays, vocab)
    option = build_option_batch(arrays, vocab)
    return {
        "board": board,
        "option": option,
        "legacy_global_features": arrays["legacy_global_features"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# 試合単位のtrain/val分割・正規化統計(T1.1)
# ---------------------------------------------------------------------------

STD_FLOOR = 1e-3  # 定数特徴でNaNが出ないための下限値


def train_val_game_split(n_games: int, val_fraction: float = 0.2,
                         seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """試合(game)単位でtrain/valの**game ID**(``concatenate_shards``で結合した
    ``lengths``配列上の0始まりのグローバルなtrajectory index)を分ける。

    game IDだけを返す決定的な関数にして、``split.json``にはこのIDを保存する
    (D2.1: 「決定点index」ではなく「game ID」を保存する要求に対応。決定点indexは
    ``lengths``さえあれば`expand_game_ids_to_decision_indices`で毎回再構成できる、
    game IDの方が安定した識別子)。
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_games)
    n_val = int(round(n_games * val_fraction))
    if n_games > 1:
        n_val = max(1, min(n_games - 1, n_val))
    else:
        n_val = 0
    val_games = np.sort(order[:n_val])
    train_games = np.sort(order[n_val:])
    return train_games.astype(np.int64), val_games.astype(np.int64)


def train_val_test_game_split(n_games: int, val_fraction: float = 0.15, test_fraction: float = 0.15,
                              seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """試合(game)単位でtrain/val/testの3分割(グリムスナール等、testを別枠で
    最終報告に使いたい場合用の追加。既存``train_val_game_split``(2分割)は変更しない)。"""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_games)
    n_val = max(1, int(round(n_games * val_fraction))) if n_games > 2 else 0
    n_test = max(1, int(round(n_games * test_fraction))) if n_games > 2 else 0
    n_val = min(n_val, n_games - 1)
    n_test = min(n_test, n_games - n_val - 1) if n_games - n_val > 1 else 0
    val_games = np.sort(order[:n_val])
    test_games = np.sort(order[n_val:n_val + n_test])
    train_games = np.sort(order[n_val + n_test:])
    return train_games.astype(np.int64), val_games.astype(np.int64), test_games.astype(np.int64)


def expand_game_ids_to_decision_indices(lengths: np.ndarray, game_ids) -> np.ndarray:
    """game ID(trajectoryのグローバルindex)の配列を、対応する決定点indexの配列に展開する。"""
    offsets = np.cumsum(lengths) - lengths
    game_ids = np.asarray(game_ids, dtype=np.int64)
    if len(game_ids) == 0:
        return np.zeros(0, dtype=np.int64)
    ranges = [np.arange(offsets[g], offsets[g] + lengths[g]) for g in game_ids]
    return np.concatenate(ranges)


def train_val_decision_indices(arrays: dict, val_fraction: float = 0.2,
                               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """試合(``lengths``、trajectory)単位でtrain/valを分け、それぞれに属する決定点の
    グローバルindex配列を返す。**決定点単位ではなく試合単位で分けるので、同じ試合の
    決定点がtrain/valに跨がらない**(情報漏洩を防ぐ)。

    ``train_val_game_split`` + ``expand_game_ids_to_decision_indices`` の合成
    (後方互換の簡易版。game IDを保存・検査したい呼び出し元は分解して使う)。
    """
    lengths = arrays["lengths"]
    train_games, val_games = train_val_game_split(len(lengths), val_fraction, seed)
    train_idx = expand_game_ids_to_decision_indices(lengths, train_games)
    val_idx = expand_game_ids_to_decision_indices(lengths, val_games)
    return train_idx, val_idx


def slice_arrays_by_decisions(arrays: dict, decision_indices) -> dict:
    """指定した決定点indexだけを含む新しい ``arrays`` dictを作る(train/val分割後に
    ``build_batch``/``compute_normalization_stats`` へ渡す用)。"""
    decision_indices = np.asarray(decision_indices, dtype=np.int64)
    board_counts, counts = arrays["board_counts"], arrays["counts"]
    board_off = np.cumsum(board_counts) - board_counts
    opt_off = np.cumsum(counts) - counts

    def _gather(off, cnt, idx):
        if len(idx) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(off[i], off[i] + cnt[i]) for i in idx])

    board_sel = _gather(board_off, board_counts, decision_indices)
    opt_sel = _gather(opt_off, counts, decision_indices)

    out = {
        "board_counts": board_counts[decision_indices],
        "counts": counts[decision_indices],
        "legacy_state_features": arrays["legacy_state_features"][decision_indices],
        "board_token_numeric_features": arrays["board_token_numeric_features"][board_sel],
        "board_token_card_ids": arrays["board_token_card_ids"][board_sel],
        "board_token_zone_ids": arrays["board_token_zone_ids"][board_sel],
        "legacy_option_features": arrays["legacy_option_features"][opt_sel],
        "option_card_ids": arrays["option_card_ids"][opt_sel],
        "teacher_logits": arrays["teacher_logits"][opt_sel],
        "option_target_token_indices": arrays["option_target_token_indices"][opt_sel],
        "chosen": arrays["chosen"][decision_indices],
        "old_logp": arrays["old_logp"][decision_indices],
    }
    if "legacy_global_features" in arrays:
        out["legacy_global_features"] = arrays["legacy_global_features"][decision_indices]
    return out


def _stats_with_floor(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] == 0:
        return np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), STD_FLOOR).astype(np.float32)
    return mean, std


def compute_normalization_stats(arrays: dict, decision_indices) -> dict:
    """``board_token_numeric_features``(11次元)・``legacy_global_features``・
    ``legacy_option_features``のmean/stdを、指定した決定点(train splitのみ)から計算する。

    保存されている行はいずれも実データのみ(paddingは含まれない、paddingはバッチ構築時に
    ``token_batch``が追加するだけで保存はされない)なので、ここで計算した統計にpaddingは
    混ざらない。stdは``STD_FLOOR``で下限を設け、定数特徴でも0除算・NaNにならないようにする。
    """
    sub = slice_arrays_by_decisions(arrays, decision_indices)
    board_mean, board_std = _stats_with_floor(
        sub["board_token_numeric_features"], sub["board_token_numeric_features"].shape[1]
        if sub["board_token_numeric_features"].size else 11)
    option_mean, option_std = _stats_with_floor(
        sub["legacy_option_features"], sub["legacy_option_features"].shape[1]
        if sub["legacy_option_features"].size else 65)
    global_dim = sub["legacy_global_features"].shape[1] if "legacy_global_features" in sub else 0
    global_mean, global_std = _stats_with_floor(sub.get("legacy_global_features", np.zeros((0, global_dim))), global_dim)
    return {
        "board_numeric_mean": board_mean, "board_numeric_std": board_std,
        "global_mean": global_mean, "global_std": global_std,
        "option_mean": option_mean, "option_std": option_std,
    }


def global_stats_from_teacher(teacher_state_mean, teacher_state_std, profile: str | None) -> tuple[np.ndarray, np.ndarray]:
    """教師(既存MLP)の``state_mean``/``state_std``から、legacy_global_featuresに対応する
    列だけを抜き出す(``legacy_feature_manifest.global_columns``と同じ選び方)。

    **これは新規計算不要。** legacy_global_featuresは既存state_featuresの一部をそのまま
    抜き出したものなので、教師が既に持つ較正済みの統計をそのまま使える
    (``compute_normalization_stats``で再計算する必要があるのは``board_numeric``のみ。
    board tokenは全スロット共通の統計を使う新しい表現で、教師のスロット別統計を
    そのまま流用できないため)。"""
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    cols = manifest.global_columns(profile)
    return (np.asarray(teacher_state_mean, dtype=np.float32)[cols],
            np.asarray(teacher_state_std, dtype=np.float32)[cols])


# token_policy_t1.NORMALIZATION_MODESと同じ値(torch非依存に保つため定数を複製している。
# token_batchはnumpyのみで完結させる設計のため、torch依存のtoken_policy_t1をここから
# importしない)。
NORMALIZATION_MODES = ("teacher_stats", "student_train_stats")


def build_normalization_stats(mode: str, arrays: dict, train_idx, profile: str | None,
                              teacher_standardization: dict | None = None) -> dict:
    """``mode``に応じた正規化統計を組み立てる(唯一の窓口)。

    - ``student_train_stats``: board_numeric・global・optionのすべてをtrain splitのみ
      から新規計算する(``compute_normalization_stats``をそのまま返す)。
    - ``teacher_stats``: board_numericのみtrain splitから新規計算(教師に相当する統計が
      無いため)。global/optionは``teacher_standardization``
      (教師のpolicy_weights.jsonの``standardization``ブロック、
      ``{"state_mean":...,"state_std":...,"option_mean":...,"option_std":...}``)から
      再利用する。この場合 ``teacher_standardization`` は必須。
    """
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"normalization_modeは{NORMALIZATION_MODES}のいずれか: {mode!r}")

    if mode == "student_train_stats":
        return compute_normalization_stats(arrays, train_idx)

    if teacher_standardization is None:
        raise ValueError("mode='teacher_stats'にはteacher_standardizationが必須")
    board_only = compute_normalization_stats(arrays, train_idx)
    global_mean, global_std = global_stats_from_teacher(
        teacher_standardization["state_mean"], teacher_standardization["state_std"], profile)
    return {
        "board_numeric_mean": board_only["board_numeric_mean"],
        "board_numeric_std": board_only["board_numeric_std"],
        "global_mean": global_mean, "global_std": global_std,
        "option_mean": np.asarray(teacher_standardization["option_mean"], dtype=np.float32),
        "option_std": np.asarray(teacher_standardization["option_std"], dtype=np.float32),
    }
