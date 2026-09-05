"""バリューネットワーク(Step1, イシュー #72)のオフライン評価。

``features.npz`` の **test split(split==2)** のみで、勝率予測器(値ネット)を2つの自明な
ベースラインと突き合わせる。val split は学習側の較正フィッティングに使われているため、
最終判断は未使用の test split で行う(design.md §5.1)。

ベースライン1: サイド枚数差(prize_diff)のみの1変数ロジスティック回帰(train split で学習)。
ベースライン2: マッチアップ事前勝率表(train split の (自archetype, 相手archetype) 別実勝率)。
値ネット本体: ``ptcg_ai.learning.value_model.ValueModel`` の較正込み予測。

指標: AUC / logloss(sklearn)。全体・ターン帯別・マッチアップ別で層別評価する。

実行:
    PYTHONIOENCODING=utf-8 python kaggle_replays/value_net/evaluate.py
    python kaggle_replays/value_net/evaluate.py --features <features.npz> --weights <value_weights.json> \
        --out <evaluate_results.json>

このスクリプトはオフライン評価専用(提出物ではない)。numpy/sklearn を使ってよい。
値ネットの本番推論経路(ValueModel)が features.npz の X を素通しした結果と一致することを
スポットチェックで担保したうえで、ベクトル化した numpy フォワードで全 test 局面を高速評価する。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SAMPLE_SUBMISSION = _REPO_ROOT / "sample_submission"
if str(_SAMPLE_SUBMISSION) not in sys.path:
    sys.path.insert(0, str(_SAMPLE_SUBMISSION))

from ptcg_ai.learning.value_model import ValueModel, _turn_band_of  # noqa: E402

_DEFAULT_FEATURES = _HERE / "features.npz"
_DEFAULT_WEIGHTS = _SAMPLE_SUBMISSION / "ptcg_ai" / "learning" / "value_weights.json"
_DEFAULT_OUT = _HERE / "evaluate_results.json"
_DECK_LABELS = _REPO_ROOT / "kaggle_replays" / "deck_predictor" / "output" / "deck_labels.jsonl"

_TRAIN, _VAL, _TEST = 0, 1, 2
_TURN_BANDS = ["1-2", "3-5", "6-10", "11+", "unknown"]
# ベースライン2 のマッチアップ勝率表を全体平均へ縮小(shrinkage)する擬似カウント。
# 少数セルが 0/1 に張り付いて logloss が発散するのを防ぐ、prior 表の標準的な平滑化。
# rate = (勝ち数 + K*全体平均) / (件数 + K)。K が大きいほど全体平均に寄る。
_MATCHUP_SHRINKAGE_K = 20.0
# logloss を安定させるための確率クリップ(ベースライン2 の 0/1 張り付き対策)。
# sklearn log_loss 既定と同等の緩いクリップにし、値ネット raw が meta.test_metrics と一致するようにする。
_CLIP = 1e-12


# ---------------------------------------------------------------------------
# 値ネットのベクトル化フォワード(numpy)。ValueModel と数値一致することをスポットチェックで担保。
# ---------------------------------------------------------------------------
def load_weights(weights_path: Path) -> dict:
    with weights_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def vectorized_raw_proba(X: np.ndarray, weights: dict) -> np.ndarray:
    """標準化 → 各層フォワード → 最終 sigmoid の p_raw(較正前)をベクトル化計算する。"""
    mean = np.asarray(weights["standardization"]["mean"], dtype=np.float64)
    std = np.asarray(weights["standardization"]["std"], dtype=np.float64)
    h = (X.astype(np.float64) - mean) / std
    for layer in weights["layers"]:
        W = np.asarray(layer["W"], dtype=np.float64)  # [out][in]
        b = np.asarray(layer["b"], dtype=np.float64)
        z = h @ W.T + b
        if layer["activation"] == "relu":
            h = np.maximum(0.0, z)
        elif layer["activation"] == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-z))
        else:
            raise ValueError(f"unknown activation: {layer['activation']}")
    return h[:, 0]


def calibrate(p_raw: np.ndarray, turns: np.ndarray, weights: dict) -> np.ndarray:
    """ターン帯別温度で p_raw を較正する(ValueModel._calibrate のベクトル版)。"""
    buckets = {b["band"]: float(b["temperature"]) for b in weights["meta"]["calibration"]["buckets"]}
    p = np.clip(p_raw, 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p))
    temps = np.array([buckets.get(_turn_band_of(int(t)), 1.0) for t in turns], dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logit / temps))


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC / logloss / 件数。片側クラスしか無い場合 AUC は None。"""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=np.float64), _CLIP, 1 - _CLIP)
    n = len(y)
    out = {"n": int(n), "pos_rate": float(y.mean()) if n else None}
    if n == 0 or len(np.unique(y)) < 2:
        out["auc"] = None
        out["logloss"] = float(log_loss(y, p, labels=[0, 1])) if n else None
        return out
    out["auc"] = float(roc_auc_score(y, p))
    out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    return out


def fmt(m: dict) -> str:
    auc = "  n/a " if m["auc"] is None else f"{m['auc']:.4f}"
    ll = "  n/a " if m["logloss"] is None else f"{m['logloss']:.4f}"
    return f"AUC={auc}  logloss={ll}  n={m['n']}"


# ---------------------------------------------------------------------------
# デッキラベル(マッチアップ)
# ---------------------------------------------------------------------------
def load_deck_labels() -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    if not _DECK_LABELS.exists():
        return labels
    with _DECK_LABELS.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[(str(row["episode_id"]), int(row["player_index"]))] = str(row["archetype"])
    return labels


def build_matchup_arrays(episode_id, player_index, labels):
    """各局面の (自archetype, 相手archetype) を返す。ラベル不明は 'unknown'。"""
    self_arch = []
    opp_arch = []
    for ep, pi in zip(episode_id, player_index):
        ep = str(ep)
        pi = int(pi)
        self_arch.append(labels.get((ep, pi), "unknown"))
        opp_arch.append(labels.get((ep, 1 - pi), "unknown"))
    return np.array(self_arch), np.array(opp_arch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES), help="入力 features.npz")
    parser.add_argument("--weights", default=str(_DEFAULT_WEIGHTS), help="評価する value_weights.json")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="評価結果JSONの書き出し先")
    args = parser.parse_args()

    features_path = Path(args.features)
    weights_path = Path(args.weights)
    out_path = Path(args.out)

    print("=== バリューネットワーク オフライン評価(test split) ===\n")
    data = np.load(features_path, allow_pickle=False)
    X = data["X"]
    y = data["y"].astype(int)
    turn = data["turn"].astype(int)
    split = data["split"].astype(int)
    weight = data["weight"].astype(np.float64)
    episode_id = data["episode_id"]
    player_index = data["player_index"].astype(int)

    weights = load_weights(weights_path)
    feature_names = weights["feature_names"]
    prize_idx = feature_names.index("prize_diff")

    tr = split == _TRAIN
    te = split == _TEST
    print(f"train={int(tr.sum())}  val={int((split==_VAL).sum())}  test={int(te.sum())}")
    print(f"test pos_rate(勝率)={y[te].mean():.4f}\n")

    # -------------------- 値ネット本体 --------------------
    p_raw_all = vectorized_raw_proba(X, weights)
    p_cal_all = calibrate(p_raw_all, turn, weights)

    # スポットチェック: ValueModel の実推論経路が numpy ベクトル版と一致するか(300件)。
    model = ValueModel(weights_path)
    assert model.is_ready, f"value_weights.json をロードできません: {weights_path}"
    te_idx = np.where(te)[0]
    rng = random.Random(0)
    check = rng.sample(list(te_idx), min(300, len(te_idx)))
    max_err = 0.0
    for i in check:
        got = model.predict_win_prob_from_features(X[i].tolist(), int(turn[i]))
        max_err = max(max_err, abs(got - p_cal_all[i]))
    print(f"[スポットチェック] ValueModel vs numpy較正版の最大誤差(test 300件)={max_err:.3e}")
    assert max_err < 1e-6, "ValueModel と評価用ベクトル版が不一致(実装ズレ)"

    y_te = y[te]
    turn_te = turn[te]

    vn_cal = metrics(y_te, p_cal_all[te])
    vn_raw = metrics(y_te, p_raw_all[te])
    print(f"\n[値ネット 較正後]   {fmt(vn_cal)}")
    print(f"[値ネット 生(raw)] {fmt(vn_raw)}   <- meta.test_metrics と突き合わせる")
    meta_test = weights["meta"]["test_metrics"]
    print(f"[meta.test_metrics] AUC={meta_test['auc']:.4f}  logloss={meta_test['logloss']:.4f}")
    print(f"  整合性: dAUC={abs(vn_raw['auc']-meta_test['auc']):.2e}  "
          f"dLogloss={abs(vn_raw['logloss']-meta_test['logloss']):.2e}")

    # -------------------- ベースライン1: prize_diff 1変数ロジスティック回帰 --------------------
    xb_tr = X[tr, prize_idx].reshape(-1, 1).astype(np.float64)
    xb_te = X[te, prize_idx].reshape(-1, 1).astype(np.float64)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(xb_tr, y[tr], sample_weight=weight[tr])
    p_b1_te = lr.predict_proba(xb_te)[:, 1]
    b1 = metrics(y_te, p_b1_te)
    print(f"\n[ベースライン1 prize_diff] {fmt(b1)}  (coef={lr.coef_[0,0]:.3f} intercept={lr.intercept_[0]:.3f})")

    # -------------------- ベースライン2: マッチアップ事前勝率表 --------------------
    labels = load_deck_labels()
    b2_available = len(labels) > 0
    p_b2_te = None
    self_arch_te = opp_arch_te = None
    if b2_available:
        self_arch_all, opp_arch_all = build_matchup_arrays(episode_id, player_index, labels)
        # join できた局面が実際にあるか(全て unknown なら実質スキップ扱い)。
        joined_te = np.sum((self_arch_all[te] != "unknown") | (opp_arch_all[te] != "unknown"))
        if joined_te == 0:
            b2_available = False
    if b2_available:
        # train split で (自, 相手) 別の勝率表を作る。
        pair_sum = defaultdict(float)
        pair_cnt = defaultdict(int)
        sa_tr = self_arch_all[tr]
        oa_tr = opp_arch_all[tr]
        y_tr = y[tr]
        for sa, oa, yy in zip(sa_tr, oa_tr, y_tr):
            pair_sum[(sa, oa)] += yy
            pair_cnt[(sa, oa)] += 1
        global_mean = float(y_tr.mean())
        # 全体平均へ縮小した勝率表(擬似カウント K)。学習に現れないペアは全体平均。
        table = {
            k: (pair_sum[k] + _MATCHUP_SHRINKAGE_K * global_mean) / (pair_cnt[k] + _MATCHUP_SHRINKAGE_K)
            for k in pair_cnt
        }
        self_arch_te = self_arch_all[te]
        opp_arch_te = opp_arch_all[te]
        p_b2_te = np.array(
            [table.get((sa, oa), global_mean) for sa, oa in zip(self_arch_te, opp_arch_te)],
            dtype=np.float64,
        )
        b2 = metrics(y_te, p_b2_te)
        n_pairs = len(table)
        covered = np.mean([(sa, oa) in table for sa, oa in zip(self_arch_te, opp_arch_te)])
        print(f"[ベースライン2 マッチアップ表] {fmt(b2)}  "
              f"(学習ペア数={n_pairs} shrinkK={_MATCHUP_SHRINKAGE_K:g}, test被覆率={covered:.2%}, 全体平均={global_mean:.3f})")
    else:
        b2 = None
        print("[ベースライン2 マッチアップ表] スキップ(deck_labels.jsonl が無い/結合不能)")

    # -------------------- ターン帯別 --------------------
    print("\n=== ターン帯別(値ネット較正 vs ベースライン1) ===")
    band_te = np.array([_turn_band_of(int(t)) for t in turn_te])
    band_rows = []
    for band in _TURN_BANDS:
        mask = band_te == band
        if mask.sum() == 0:
            continue
        m_vn = metrics(y_te[mask], p_cal_all[te][mask])
        m_b1 = metrics(y_te[mask], p_b1_te[mask])
        band_rows.append((band, m_vn, m_b1))
        print(f"  turn {band:<7} 値ネット[{fmt(m_vn)}]  |  ベース1[{fmt(m_b1)}]")

    # -------------------- マッチアップ別(自archetype) --------------------
    matchup_rows = []
    if b2_available:
        print("\n=== 自archetype別(値ネット較正 vs ベースライン2 マッチアップ表) ===")
        # 件数上位の自archetype を個別に、それ以外は 'other'。
        counts = defaultdict(int)
        for a in self_arch_te:
            counts[a] += 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        # 個別表示は分類済みの主要 archetype のみ。deck_predictor の "other"(分類不能)と
        # ラベル欠損 "unknown" は集約バケットへ回す(件数上位でも個別化しない)。
        top = [a for a, _ in ranked if a not in ("unknown", "other")][:6]
        # 主要 archetype 以外(minor + 実ラベル "other")= "other"、ラベル欠損 = "unknown"。
        group = np.array(
            ["unknown" if a == "unknown" else (a if a in top else "other") for a in self_arch_te]
        )
        for arch in top + ["other", "unknown"]:
            mask = group == arch
            if mask.sum() == 0:
                continue
            m_vn = metrics(y_te[mask], p_cal_all[te][mask])
            m_b2 = metrics(y_te[mask], p_b2_te[mask])
            matchup_rows.append((arch, m_vn, m_b2))
            print(f"  {arch:<18} 値ネット[{fmt(m_vn)}]  |  ベース2[{fmt(m_b2)}]")

    # -------------------- 判断 --------------------
    print("\n=== 判断 ===")
    verdict, reasons = decide(vn_cal, b1, b2, band_rows)
    print(f"判定: {verdict}")
    for r in reasons:
        print(f"  - {r}")

    # 結果を JSON でも吐いておく(レポート作成の裏付け)。
    out = {
        "test_n": int(te.sum()),
        "test_pos_rate": float(y_te.mean()),
        "value_net_calibrated": vn_cal,
        "value_net_raw": vn_raw,
        "meta_test_metrics": meta_test,
        "baseline1_prize_diff": b1,
        "baseline2_matchup": b2,
        "baseline2_available": b2_available,
        "by_turn_band": [{"band": b, "value_net": v, "baseline1": bb} for b, v, bb in band_rows],
        "by_archetype": [{"archetype": a, "value_net": v, "baseline2": bb} for a, v, bb in matchup_rows],
        "verdict": verdict,
        "reasons": reasons,
        "spotcheck_max_err": max_err,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n結果を書き出しました: {out_path}")


def decide(vn, b1, b2, band_rows):
    reasons = []
    beats_b1 = vn["auc"] > b1["auc"]
    reasons.append(
        f"全体 AUC: 値ネット {vn['auc']:.4f} vs ベース1(prize_diff) {b1['auc']:.4f} -> "
        + ("上回る" if beats_b1 else "上回らない")
    )
    beats_b1_ll = vn["logloss"] < b1["logloss"]
    reasons.append(
        f"全体 logloss: 値ネット {vn['logloss']:.4f} vs ベース1 {b1['logloss']:.4f} -> "
        + ("良い" if beats_b1_ll else "悪い")
    )
    beats_b2 = True
    if b2 is not None and b2["auc"] is not None:
        beats_b2 = vn["auc"] > b2["auc"]
        reasons.append(
            f"全体 AUC: 値ネット {vn['auc']:.4f} vs ベース2(マッチアップ表) {b2['auc']:.4f} -> "
            + ("上回る" if beats_b2 else "上回らない")
        )
    # 序盤 1-2 帯で値ネットがベース1 に対しどうか。
    early = next((r for r in band_rows if r[0] == "1-2"), None)
    if early is not None:
        _, m_vn, m_b1 = early
        if m_vn["auc"] is not None and m_b1["auc"] is not None:
            reasons.append(
                f"序盤(1-2) AUC: 値ネット {m_vn['auc']:.4f} vs ベース1 {m_b1['auc']:.4f} -> "
                + ("上回る" if m_vn["auc"] > m_b1["auc"] else "上回らない(既知リスク: 序盤は弱い)")
            )
    if beats_b1 and beats_b1_ll and beats_b2:
        verdict = "PASS"
    elif not beats_b1 and not beats_b1_ll:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, reasons


if __name__ == "__main__":
    main()
