"""label-fidelity study(外部指摘対応): training labelの決定化数(N)を増やすと、
paired outcomeの符号がどれだけ安定するかを検証する。

`collect_ogerpon_counterfactuals.py` を高determinizations(既定64)で実行した結果を
読み込み、同じ64本のrollout列の先頭からN=4/8/16/32/64のprefixを切り出して
経験的delta(single win - ex win の平均)を再計算する(別々に再収集しない、nested
prefixそのもの)。

Phase1: 現行Model Aのpredicted_deltaでnegative/near-zero/positiveに層化(診断用、
選択バイアスの記述目的。以降の指標は全て収集できた全状態に対して計算する)。
Phase2: N別のtie率・N=64との符号一致率・相関・MAE・符号反転率。
Phase3: N=64での90% bootstrap CIによるconfident positive/negative/ambiguous分類。
Phase4: 仮説A(label noiseが主因)/仮説B(効果がほぼ識別不能)の判定材料を出力する。

判定そのもの(仮説A/Bのどちらを採用するか)はこのスクリプトの出力を見て人が(または
呼び出し側が)判断する。ここでは判定に使う数値をすべて機械的に計算するだけで、
bin境界やしきい値を結果を見てから変更しない(外部指摘: 都合のよい境界へ変更しない)。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_ogerpon_strategy as T  # noqa: E402
from ptcg_ai.learning.ogerpon_strategy_model import OgerponStrategyModel  # noqa: E402
from ptcg_ai.ml_policy import ogerpon_strategy as STRAT  # noqa: E402
from types import SimpleNamespace  # noqa: E402

N_LEVELS = (4, 8, 16, 32, 64)
POSITIVE_THRESHOLD = 0.01
NEGATIVE_THRESHOLD = -0.01


def extract_paired_outcomes(records: list[dict]) -> dict[str, dict]:
    """sample_idごとに、EX_TEMPO/SINGLE_PRIZE_ROTATION両方のwinがある決定化だけを対象に、
    決定化ID順にソートしたpaired_outcome(-1/0/+1)のリストを作る。
    """
    by_sample: dict[str, list[dict]] = {}
    for r in records:
        by_sample.setdefault(r["sample_id"], []).append(r)

    out = {}
    for sid, recs in by_sample.items():
        ex_by_det, single_by_det = {}, {}
        meta = None
        for r in recs:
            o = r["outcome"]
            if o.get("error") is not None or o.get("win") is None:
                continue
            if r["option_name"] == "EX_TEMPO":
                ex_by_det[r["determinization_id"]] = o["win"]
            elif r["option_name"] == "SINGLE_PRIZE_ROTATION":
                single_by_det[r["determinization_id"]] = o["win"]
            if meta is None:
                meta = r
        common = sorted(set(ex_by_det) & set(single_by_det))
        if len(common) < max(N_LEVELS):
            continue  # 全rollout(最大N)が揃っていない状態は除外(エラー等で欠けた分)
        paired = [single_by_det[d] - ex_by_det[d] for d in common]
        out[sid] = {
            "paired_outcome": paired,
            "opponent_archetype": meta["opponent_archetype"],
            "learner_index": meta["learner_index"],
            "n_dets_available": len(common),
        }
    return out


def score_with_model_a(records: list[dict], sample_ids: list[str], weights_path: str) -> dict[str, float]:
    """Model Aのpredicted_deltaを、各sample_idの(EX_TEMPO, SINGLE_PRIZE_ROTATION)特徴量
    から計算する(層化のためだけに使う。以降の指標には使わない)。
    """
    by_state = T.aggregate_by_state_option(records)
    model = OgerponStrategyModel(weights_path)
    out = {}
    for sid in sample_ids:
        opts = by_state.get(sid)
        if not opts or "EX_TEMPO" not in opts or "SINGLE_PRIZE_ROTATION" not in opts:
            continue
        ex, single = opts["EX_TEMPO"], opts["SINGLE_PRIZE_ROTATION"]
        raw_single = next(r for r in records if r["sample_id"] == sid and r["option_name"] == "SINGLE_PRIZE_ROTATION")
        encoded = {
            "continuous_features": ex["continuous_features"], "slot_card_ids": ex["slot_card_ids"],
            "option_features": {"EX_TEMPO": ex["option_features"], "SINGLE_PRIZE_ROTATION": single["option_features"]},
        }
        sc = SimpleNamespace(required_ko_gain=raw_single["first_action_identity"]["required_ko_gain"])
        comparison = STRAT.compare_options(model, encoded, sc)
        out[sid] = comparison["delta"]
    return out


def bootstrap_mean_ci(vals: list[float], n_boot: int = 3000, alpha: float = 0.10, seed: int = 0):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int((alpha / 2) * n_boot)], means[int((1 - alpha / 2) * n_boot)]


def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / ((vx ** 0.5) * (vy ** 0.5)) if vx > 0 and vy > 0 else float("nan")


def spearman(xs, ys):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1):
                ranks[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return ranks
    return pearson(rank(xs), rank(ys))


def analyze(data_path: str, model_a_weights: str) -> dict:
    records = T.load_records(Path(data_path))
    paired = extract_paired_outcomes(records)
    sample_ids = list(paired.keys())
    print(f"n states with full {max(N_LEVELS)} paired determinizations: {len(sample_ids)}")

    pred_delta_a = score_with_model_a(records, sample_ids, model_a_weights)

    def bucket(sid):
        d = pred_delta_a.get(sid)
        if d is None:
            return "unscored"
        if d <= NEGATIVE_THRESHOLD:
            return "negative"
        if d >= POSITIVE_THRESHOLD:
            return "positive"
        return "near_zero"

    bucket_counts = {"negative": 0, "near_zero": 0, "positive": 0, "unscored": 0}
    for sid in sample_ids:
        bucket_counts[bucket(sid)] += 1
    print("Model A predicted_delta stratification (descriptive only):", bucket_counts)

    # --- Phase 2: nested-prefix N別分析 ---
    delta_by_n: dict[int, dict[str, float]] = {n: {} for n in N_LEVELS}
    for sid in sample_ids:
        po = paired[sid]["paired_outcome"]
        for n in N_LEVELS:
            prefix = po[:n]
            delta_by_n[n][sid] = sum(prefix) / len(prefix)

    ref = delta_by_n[max(N_LEVELS)]  # N=64を基準("ground truth"近似)とする

    phase2 = {}
    for n in N_LEVELS:
        d_n = delta_by_n[n]
        tie_rate = sum(1 for sid in sample_ids if d_n[sid] == 0.0) / len(sample_ids)
        n_sign_agree = sum(1 for sid in sample_ids if (d_n[sid] > 0) == (ref[sid] > 0)
                           or (d_n[sid] == 0 and ref[sid] == 0))
        sign_agree_rate = n_sign_agree / len(sample_ids)
        xs = [d_n[sid] for sid in sample_ids]
        ys = [ref[sid] for sid in sample_ids]
        mae = sum(abs(x - y) for x, y in zip(xs, ys)) / len(sample_ids)
        pear = pearson(xs, ys)
        spear = spearman(xs, ys)
        # false discovery: N-levelでpositiveと判定したが、N=64(ref)ではpositiveでない割合
        n_pred_pos = [sid for sid in sample_ids if d_n[sid] > 0]
        pos_fdr = (sum(1 for sid in n_pred_pos if ref[sid] <= 0) / len(n_pred_pos)) if n_pred_pos else None
        n_pred_neg = [sid for sid in sample_ids if d_n[sid] < 0]
        neg_fdr = (sum(1 for sid in n_pred_neg if ref[sid] >= 0) / len(n_pred_neg)) if n_pred_neg else None
        phase2[n] = {
            "tie_rate": tie_rate, "sign_agreement_with_N64": sign_agree_rate,
            "pearson_with_N64": pear, "spearman_with_N64": spear, "mae_vs_N64": mae,
            "n_predicted_positive": len(n_pred_pos), "positive_false_discovery_rate": pos_fdr,
            "n_predicted_negative": len(n_pred_neg), "negative_false_discovery_rate": neg_fdr,
        }
        print(f"N={n}: tie_rate={tie_rate:.3f} sign_agree_vs_N64={sign_agree_rate:.3f} "
             f"pearson={pear:.3f} spearman={spear:.3f} mae={mae:.4f} "
             f"pos_fdr={pos_fdr} neg_fdr={neg_fdr}")

    # 符号反転率(段階を上げるごとに前段と符号が変わった割合)
    flip_rates = {}
    ordered = list(N_LEVELS)
    for i in range(1, len(ordered)):
        n_prev, n_cur = ordered[i - 1], ordered[i]
        d_prev, d_cur = delta_by_n[n_prev], delta_by_n[n_cur]
        n_flip = sum(1 for sid in sample_ids if (d_prev[sid] > 0) != (d_cur[sid] > 0)
                    or (d_prev[sid] == 0) != (d_cur[sid] == 0))
        flip_rates[f"{n_prev}->{n_cur}"] = n_flip / len(sample_ids)
    print("sign flip rates between successive N:", flip_rates)

    # bucket別のN=4 vs N=64 sign agreement
    phase2_by_bucket = {}
    for b in ("negative", "near_zero", "positive"):
        ids_in_bucket = [sid for sid in sample_ids if bucket(sid) == b]
        if not ids_in_bucket:
            continue
        n4, n64 = delta_by_n[4], delta_by_n[64]
        agree = sum(1 for sid in ids_in_bucket if (n4[sid] > 0) == (n64[sid] > 0)
                   or (n4[sid] == 0) == (n64[sid] == 0))
        phase2_by_bucket[b] = {"n": len(ids_in_bucket), "N4_vs_N64_sign_agreement": agree / len(ids_in_bucket)}
    print("N=4 vs N=64 sign agreement by Model-A-predicted bucket:", phase2_by_bucket)

    # matchup/side別
    by_arch, by_side = {}, {}
    n4, n64 = delta_by_n[4], delta_by_n[64]
    for sid in sample_ids:
        arch = paired[sid]["opponent_archetype"]
        side = paired[sid]["learner_index"]
        agree = (n4[sid] > 0) == (n64[sid] > 0) or (n4[sid] == 0) == (n64[sid] == 0)
        by_arch.setdefault(arch, []).append(agree)
        by_side.setdefault(side, []).append(agree)
    phase2_by_arch = {a: sum(v) / len(v) for a, v in by_arch.items()}
    phase2_by_side = {s: sum(v) / len(v) for s, v in by_side.items()}
    print("N=4 vs N=64 sign agreement by matchup:", phase2_by_arch)
    print("N=4 vs N=64 sign agreement by side:", phase2_by_side)

    # --- Phase 3: N=64でのconfident positive/negative/ambiguous分類 ---
    confident_positive, confident_negative, ambiguous = [], [], []
    for sid in sample_ids:
        po = paired[sid]["paired_outcome"]
        lo, hi = bootstrap_mean_ci(po, seed=hash(sid) % (2**31))
        if lo > 0:
            confident_positive.append(sid)
        elif hi < 0:
            confident_negative.append(sid)
        else:
            ambiguous.append(sid)
    print(f"\nPhase3: confident_positive={len(confident_positive)} "
         f"confident_negative={len(confident_negative)} ambiguous={len(ambiguous)} "
         f"(total={len(sample_ids)})")

    n_model_a_caught_cp = sum(1 for sid in confident_positive if pred_delta_a.get(sid, 0) > 0)
    cp_capture_rate = n_model_a_caught_cp / len(confident_positive) if confident_positive else None
    print(f"Model A predicted positive for confident_positive states: "
         f"{n_model_a_caught_cp}/{len(confident_positive)} ({cp_capture_rate})")

    strict_sids = [sid for sid in sample_ids
                  if pred_delta_a.get(sid) is not None]
    # strict subset among THIS sample (not the earlier 941-state dev set): use compare_options directly
    model = OgerponStrategyModel(model_a_weights)
    by_state = T.aggregate_by_state_option(records)
    n_strict = 0
    n_strict_confident_positive = 0
    for sid in sample_ids:
        opts = by_state.get(sid)
        if not opts or "EX_TEMPO" not in opts or "SINGLE_PRIZE_ROTATION" not in opts:
            continue
        ex, single = opts["EX_TEMPO"], opts["SINGLE_PRIZE_ROTATION"]
        raw_single = next(r for r in records if r["sample_id"] == sid and r["option_name"] == "SINGLE_PRIZE_ROTATION")
        encoded = {
            "continuous_features": ex["continuous_features"], "slot_card_ids": ex["slot_card_ids"],
            "option_features": {"EX_TEMPO": ex["option_features"], "SINGLE_PRIZE_ROTATION": single["option_features"]},
        }
        sc = SimpleNamespace(required_ko_gain=raw_single["first_action_identity"]["required_ko_gain"])
        comparison = STRAT.compare_options(model, encoded, sc)
        if comparison["lcb_delta"] >= 0.02:
            n_strict += 1
            if sid in confident_positive:
                n_strict_confident_positive += 1
    print(f"strict subset (lcb_delta>=0.02) in this sample: n={n_strict}, "
         f"of which confident_positive: {n_strict_confident_positive}")

    result = {
        "n_states": len(sample_ids),
        "bucket_counts": bucket_counts,
        "phase2_by_n": phase2,
        "sign_flip_rates": flip_rates,
        "phase2_by_bucket": phase2_by_bucket,
        "phase2_by_archetype": phase2_by_arch,
        "phase2_by_side": phase2_by_side,
        "phase3": {
            "n_confident_positive": len(confident_positive),
            "n_confident_negative": len(confident_negative),
            "n_ambiguous": len(ambiguous),
            "model_a_confident_positive_capture_rate": cp_capture_rate,
            "n_strict_in_sample": n_strict,
            "n_strict_that_are_confident_positive": n_strict_confident_positive,
        },
    }
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-a-weights", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = analyze(args.data, args.model_a_weights)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nsaved to {args.output}")
