#!/usr/bin/env python3
"""模倣ポリシー(Step2)のオフライン評価(step2-design.md §5.1)。

``features.npz`` の **test split(split==2)** のみを対象に、3つを突き合わせる:

1. ベースライン1: 一様ランダム選択。各意思決定点で ``1 / n_options_i`` の平均(期待Top-1一致率)。
2. ベースライン2: 現行 ``ptcg_ai.rule_based.rule_based_agent.agent`` の再実行。
   ``features.npz`` の ``row_index``(= ``policy_positions.jsonl.gz`` の行番号、build_features.py が
   順序を保った証拠として保存したもの)を使い、test split に属する行だけ元の観測(observation)を
   読み直して ``rule_based_agent(obs)`` を再実行し、記録済み ``chosen_index`` とどれだけ一致するかを
   測る。``rule_based_agent`` は読み取り専用で呼ぶだけ(``ptcg_ai/rule_based/`` は変更しない)。
3. 本命: 学習済み ``policy_weights.json``(train.py が書き出した重み。2026-07-20 に
   card_embedding(個別カード識別の学習可能な埋め込み)対応)の Top-1一致率。
   ``ptcg_ai/learning/policy_model.py`` の本番フォワードパス(標準化 -> state++option++
   card_embedding 連結 -> Linear -> ReLU -> Linear -> argmax)をこのファイル内に numpy で
   複製する(policy_model.py 自体は import しない独立実装。train.py の自己検証で本番経路との
   数値一致は別途確認済み)。

層別評価(必須): SelectType別、ターン帯別(1-2/3-5/6-10/11+、value_net の turn_band() と
同じ境界を複製)、および select_type=MAIN の中で実際に選ばれた選択肢の OptionType別
(PLAY/ATTACH/EVOLVE/... card_embedding 追加の狙いである PLAY/ATTACH の精度改善を
確認するため)。

出力: ``kaggle_replays/policy_net/evaluate_results.json`` に全指標を保存し、標準出力に要約を表示する。

このスクリプトはオフライン評価専用(提出物ではない)。numpy/torch は使わない(train.py と違い
評価のみなので numpy の行列積で十分)。

実行:
    PYTHONIOENCODING=utf-8 python kaggle_replays/policy_net/evaluate.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
if str(_SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

_FEATURES = _HERE / "features.npz"
_WEIGHTS = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "policy_weights.json"
_POLICY_POSITIONS = _REPO_ROOT / "kaggle_replays" / "training_data" / "policy_positions.jsonl.gz"
_OUT_PATH = _HERE / "evaluate_results.json"

_TRAIN, _VAL, _TEST = 0, 1, 2

# ターン帯定義(kaggle_replays/value_net/train.py の turn_band_of() と同じ境界を複製)。
_TURN_BANDS = ["1-2", "3-5", "6-10", "11+", "unknown"]


def turn_band_of(turn: int) -> str:
    if turn is None or turn < 0:
        return "unknown"
    if turn <= 2:
        return "1-2"
    if turn <= 5:
        return "3-5"
    if turn <= 10:
        return "6-10"
    return "11+"


# ---------------------------------------------------------------------------
# 本命モデル: policy_weights.json のフォワードパス(numpy、train.py の PolicyScorer と同じ計算)。
# ---------------------------------------------------------------------------
def load_weights() -> dict:
    with _WEIGHTS.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def model_scores_per_row(
    weights: dict,
    state_features: np.ndarray,
    option_features: np.ndarray,
    option_card_ids: np.ndarray,
) -> list[np.ndarray]:
    """test split の各行について、選択肢ごとのスコア(未較正logit)を返す。

    ``state_features``: (N, BASE_FEATURE_COUNT) 生特徴。``option_features``: object配列(N,)、
    各要素 (n_i, OPTION_FEATURE_COUNT) 生特徴。``option_card_ids``: object配列(N,)、各要素
    (n_i,) の生 card_id(int、embedding lookup用、標準化しない)。戻り値は行ごとの (n_i,)
    スコア配列のリスト(``option_features`` と同じ順序)。

    policy_model.py の ``PolicyModel._forward`` と同じ計算(state_std ++ option_std ++
    card_embedding(card_id) を連結 -> Linear -> ReLU -> Linear、最終層は活性化なし)を
    ベクトル化して再実装したもの。card_id が 0 未満または card_id_max 超過の場合は
    予約枠(index 0)にフォールバックする(policy_model.py と同じ規則)。
    """
    state_mean = np.asarray(weights["standardization"]["state_mean"], dtype=np.float64)
    state_std = np.asarray(weights["standardization"]["state_std"], dtype=np.float64)
    option_mean = np.asarray(weights["standardization"]["option_mean"], dtype=np.float64)
    option_std = np.asarray(weights["standardization"]["option_std"], dtype=np.float64)

    card_embedding = weights["card_embedding"]
    card_id_max = int(card_embedding["card_id_max"])
    table = np.asarray(card_embedding["table"], dtype=np.float64)  # (card_id_max+1, dim)

    state_std_arr = (state_features.astype(np.float64) - state_mean) / state_std

    sizes = [int(o.shape[0]) for o in option_features]
    all_options = np.concatenate([o.astype(np.float64) for o in option_features], axis=0)
    all_options_std = (all_options - option_mean) / option_std
    state_repeated = np.repeat(state_std_arr, sizes, axis=0)

    all_card_ids = np.concatenate([np.asarray(c, dtype=np.int64) for c in option_card_ids], axis=0)
    safe_idx = np.where((all_card_ids >= 0) & (all_card_ids <= card_id_max), all_card_ids, 0)
    embed = table[safe_idx]

    X = np.concatenate([state_repeated, all_options_std, embed], axis=1)

    W0 = np.asarray(weights["layers"][0]["weight"], dtype=np.float64)  # (32, in+dim)
    b0 = np.asarray(weights["layers"][0]["bias"], dtype=np.float64)
    W1 = np.asarray(weights["layers"][1]["weight"], dtype=np.float64)  # (1, 32)
    b1 = np.asarray(weights["layers"][1]["bias"], dtype=np.float64)

    h = np.maximum(0.0, X @ W0.T + b0)
    z = (h @ W1.T + b1)[:, 0]

    offsets = np.cumsum(sizes)[:-1]
    return np.split(z, offsets)


# ---------------------------------------------------------------------------
# 指標(単純平均のTop-1一致率。design.md §5.1 は「選択タイミングごとのTop-1一致率」のみを
# 要求しており、rank_at_fetch由来のsample weightでの重み付けはtrain.pyの損失専用)。
# ---------------------------------------------------------------------------
def summarize_group(mask: np.ndarray, model_match, b1_score, b2_match, b2_attempted) -> dict:
    n = int(mask.sum())
    out = {"n": n}
    if n == 0:
        out.update({"model_top1": None, "baseline1_uniform_random": None, "baseline2_rule_based": None})
        return out
    out["model_top1"] = float(model_match[mask].mean())
    out["baseline1_uniform_random"] = float(b1_score[mask].mean())
    attempted_mask = mask & b2_attempted
    n_attempted = int(attempted_mask.sum())
    out["baseline2_rule_based"] = float(b2_match[attempted_mask].mean()) if n_attempted else None
    out["baseline2_n_attempted"] = n_attempted
    out["baseline2_n_error"] = int((mask & ~b2_attempted).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--limit", type=int, default=None, help="test split 先頭N件のみ処理する(動作確認モード)"
    )
    args = parser.parse_args()

    print("=== 模倣ポリシー(Step2) オフライン評価(test split) ===\n")
    t_start = time.time()

    print(f"features.npz を読み込み: {_FEATURES}")
    data = np.load(_FEATURES, allow_pickle=True)
    split = data["split"].astype(np.int64)
    test_mask = split == _TEST
    n_test = int(test_mask.sum())
    print(f"test split 件数: {n_test} / 全{len(split)}件")

    state_features = data["state_features"][test_mask]
    option_features = data["option_features"][test_mask]
    option_card_ids = data["option_card_ids"][test_mask]
    chosen_index = data["chosen_index"][test_mask].astype(np.int64)
    turn = data["turn"][test_mask].astype(np.int64)
    select_type = data["select_type"][test_mask].astype(np.int64)
    select_context = data["select_context"][test_mask].astype(np.int64)
    row_index = data["row_index"][test_mask].astype(np.int64)

    if args.limit is not None:
        n_test = min(args.limit, n_test)
        state_features = state_features[:n_test]
        option_features = option_features[:n_test]
        option_card_ids = option_card_ids[:n_test]
        chosen_index = chosen_index[:n_test]
        turn = turn[:n_test]
        select_type = select_type[:n_test]
        select_context = select_context[:n_test]
        row_index = row_index[:n_test]
        print(f"動作確認モード: test split 先頭 {n_test} 件のみ使用")

    n_options = np.array([int(o.shape[0]) for o in option_features], dtype=np.int64)

    # -------------------- ベースライン1: 一様ランダム --------------------
    print("\n[ベースライン1] 一様ランダム選択の期待Top-1一致率を計算...")
    baseline1_score = 1.0 / n_options.astype(np.float64)

    # -------------------- 本命: policy_weights.json のフォワードパス --------------------
    print("[本命] policy_weights.json をロードしてフォワードパスを計算...")
    weights = load_weights()
    scores_per_row = model_scores_per_row(weights, state_features, option_features, option_card_ids)
    model_pred = np.array([int(np.argmax(s)) for s in scores_per_row], dtype=np.int64)
    model_match = (model_pred == chosen_index).astype(np.float64)
    print(f"  全体Top-1一致率(モデル) = {model_match.mean():.4f}")

    meta_test = weights.get("meta", {}).get("test_metrics", {})
    if meta_test.get("top1_accuracy") is not None:
        print(
            f"  参考: policy_weights.json meta.test_metrics.top1_accuracy = "
            f"{meta_test['top1_accuracy']:.4f}(train.py 側の集計、重み付き平均とは定義が異なる場合あり)"
        )

    # -------------------- ベースライン2: rule_based_agent 再実行 --------------------
    print("\n[ベースライン2] rule_based_agent(obs) を test split 全件で再実行...")
    print(f"  対象行数: {n_test}(policy_positions.jsonl.gz の該当行番号を row_index から特定)")

    # rule_based_agent は deck.csv を相対パス "deck.csv" で読むため、sample_submission を cwd にする。
    #
    # 注意(既知のリスク、step2-design.md §7): rule_based_agent は内部で
    # ptcg_ai.hidden_information.match_context というモジュールレベルの状態(相手の非公開情報推定)
    # を持ち越す。design.md は「rule_based_agent は observation のみに依存する決定的関数である
    # 前提」とした上で、内部状態依存があればリスクとして扱うとしている。ここでは意図的に
    # reset しない(実運用と同じ、状態を持ち越したままの挙動を評価する)。episode をまたいで
    # 状態が残る場合があるが、update() 内部は例外安全(失敗しても既存状態を維持するだけ)なので
    # クラッシュはしない。reset() を毎回呼ぶ代替案も試したが、HybridDeckPredictor の再計算が
    # 都度走り約16倍(500 rows/sec -> 31 rows/sec)遅くなり§5.1の実行時間目安(10分以内)を
    # 超えるため採用しない。
    os.chdir(_SAMPLE_SUBMISSION_DIR)
    from cg.api import OptionType, to_observation_class  # noqa: E402
    from ptcg_ai.rule_based.rule_based_agent import agent as rule_based_agent  # noqa: E402

    target_lines = {int(li): pos for pos, li in enumerate(row_index)}
    b2_match = np.zeros(n_test, dtype=np.float64)
    b2_attempted = np.zeros(n_test, dtype=bool)
    # 実際に選ばれた選択肢の OptionType(int値)。obs 構築に失敗した行は -1 のまま(層別で除外される)。
    chosen_option_type = np.full(n_test, -1, dtype=np.int64)
    n_error = 0
    n_processed = 0

    t0 = time.time()
    with gzip.open(_POLICY_POSITIONS, "rt", encoding="utf-8") as f:
        line_no = -1
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            line_no += 1
            if line_no not in target_lines:
                continue
            pos = target_lines[line_no]

            row = json.loads(raw_line)
            try:
                obs = to_observation_class({**row["observation"], "logs": []})
                chosen_idx = int(row["chosen_index"])
                # by_chosen_option_type 用: 実際に選ばれた選択肢の OptionType を記録する
                # (rule_based_agent の成否とは独立。obs 構築さえ成功すれば取れる)。
                chosen_option_type[pos] = int(obs.select.option[chosen_idx].type)
                result = rule_based_agent(obs)
                match = isinstance(result, list) and len(result) == 1 and result[0] == chosen_idx
                b2_match[pos] = 1.0 if match else 0.0
                b2_attempted[pos] = True
            except Exception as exc:  # noqa: BLE001 - 1行の失敗で全体を止めない
                n_error += 1
                if n_error <= 10:
                    print(f"    警告: rule_based_agent 失敗(row_index={line_no}): {exc!r}", file=sys.stderr)

            n_processed += 1
            if n_processed % 2000 == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed if elapsed > 0 else 0.0
                print(f"    {n_processed}/{n_test} 件処理済み(経過 {elapsed:.1f}s、{rate:.1f} rows/sec)")

    elapsed = time.time() - t0
    print(
        f"  完了: {n_processed}/{n_test} 件処理(経過 {elapsed:.1f}s)、"
        f"rule_based_error={n_error}件"
    )
    n_attempted_total = int(b2_attempted.sum())
    b2_overall_top1 = float(b2_match[b2_attempted].mean()) if n_attempted_total else None
    print(
        f"  全体Top-1一致率(rule_based, 成功{n_attempted_total}件のみ) = "
        f"{b2_overall_top1:.4f}" if b2_overall_top1 is not None else "  rule_based: 有効サンプルなし"
    )

    # -------------------- 全体サマリ --------------------
    print("\n=== 全体Top-1一致率(3者比較) ===")
    print(f"  モデル(policy_weights.json)      : {model_match.mean():.4f}  (n={n_test})")
    print(f"  ベースライン1(一様ランダム)      : {baseline1_score.mean():.4f}  (n={n_test})")
    if b2_overall_top1 is not None:
        print(f"  ベースライン2(rule_based_agent)  : {b2_overall_top1:.4f}  (n={n_attempted_total}, error={n_error})")
    else:
        print("  ベースライン2(rule_based_agent)  : 有効サンプルなし")

    # -------------------- 層別: SelectType --------------------
    print("\n=== SelectType別 Top-1一致率 ===")
    by_select_type = {}
    for st in sorted(set(select_type.tolist())):
        mask = select_type == st
        g = summarize_group(mask, model_match, baseline1_score, b2_match, b2_attempted)
        by_select_type[str(st)] = g
        b2_str = f"{g['baseline2_rule_based']:.4f}" if g["baseline2_rule_based"] is not None else "n/a"
        print(
            f"  select_type={st:2d}  n={g['n']:6d}  "
            f"model={g['model_top1']:.4f}  uniform={g['baseline1_uniform_random']:.4f}  "
            f"rule_based={b2_str}"
        )

    # -------------------- 層別: SelectContext --------------------
    by_select_context = {}
    for sc in sorted(set(select_context.tolist())):
        mask = select_context == sc
        g = summarize_group(mask, model_match, baseline1_score, b2_match, b2_attempted)
        by_select_context[str(sc)] = g

    # -------------------- 層別: ターン帯 --------------------
    print("\n=== ターン帯別 Top-1一致率 ===")
    bands = np.array([turn_band_of(int(t)) for t in turn])
    by_turn_band = {}
    for band in _TURN_BANDS:
        mask = bands == band
        if mask.sum() == 0:
            continue
        g = summarize_group(mask, model_match, baseline1_score, b2_match, b2_attempted)
        by_turn_band[band] = g
        b2_str = f"{g['baseline2_rule_based']:.4f}" if g["baseline2_rule_based"] is not None else "n/a"
        print(
            f"  turn {band:<7} n={g['n']:6d}  "
            f"model={g['model_top1']:.4f}  uniform={g['baseline1_uniform_random']:.4f}  "
            f"rule_based={b2_str}"
        )

    # -------------------- 層別: select_type=MAIN の中で選ばれた OptionType --------------------
    # card_embedding 追加の狙い(個別カード識別で PLAY/ATTACH の精度を上げる)が実際に効いたかを
    # 確認するための層別。select_type==0(MAIN)の意思決定点のみを対象に、実際に選ばれた
    # 選択肢の OptionType(PLAY/ATTACH/EVOLVE/...)で群分けする。
    print("\n=== select_type=MAIN 内、選ばれた選択肢の OptionType別 Top-1一致率 ===")
    main_select_mask = select_type == 0
    by_chosen_option_type = {}
    ot_values = sorted(set(chosen_option_type[main_select_mask].tolist()))
    for ot_val in ot_values:
        mask = main_select_mask & (chosen_option_type == ot_val)
        g = summarize_group(mask, model_match, baseline1_score, b2_match, b2_attempted)
        if ot_val == -1:
            label = "unresolved"
        else:
            try:
                label = OptionType(ot_val).name
            except ValueError:
                label = f"unknown_{ot_val}"
        by_chosen_option_type[label] = g
        b2_str = f"{g['baseline2_rule_based']:.4f}" if g["baseline2_rule_based"] is not None else "n/a"
        print(
            f"  {label:16s} n={g['n']:6d}  "
            f"model={g['model_top1']:.4f}  uniform={g['baseline1_uniform_random']:.4f}  "
            f"rule_based={b2_str}"
        )

    # -------------------- 結果書き出し --------------------
    out = {
        "test_n": n_test,
        "overall": {
            "model_top1_accuracy": float(model_match.mean()),
            "baseline1_uniform_random_top1_accuracy": float(baseline1_score.mean()),
            "baseline2_rule_based_top1_accuracy": b2_overall_top1,
            "baseline2_n_attempted": n_attempted_total,
            "baseline2_n_error": n_error,
        },
        "by_select_type": by_select_type,
        "by_select_context": by_select_context,
        "by_turn_band": by_turn_band,
        "by_chosen_option_type": by_chosen_option_type,
        "meta": {
            "policy_weights_path": str(_WEIGHTS),
            "policy_positions_path": str(_POLICY_POSITIONS),
            "elapsed_sec": time.time() - t_start,
        },
    }
    with _OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n結果を書き出しました: {_OUT_PATH}")
    print(f"総経過時間: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
