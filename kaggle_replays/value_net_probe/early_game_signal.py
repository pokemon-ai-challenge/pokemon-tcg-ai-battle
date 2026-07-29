"""序盤(ターン1-5)で価値関数のAUCが低い原因を切り分けるプローブ。

問い: ターン1-5でAUCが低いのは
  (a) 特徴量に序盤の情報が無い
  (b) 全ターン混ぜて学習したことによる希釈
  (c) 序盤は本質的に予測不能
のどれか。

既存の kaggle_replays/value_net/features.npz を読むだけ。試合シミュレーションは行わない。
既存の kaggle_replays/value_net/ 以下のファイルは一切変更しない（このファイルは新規）。

設計上の注意:
  - 同一 episode_id には両プレイヤー視点の行が入っており、ラベルは完全に反対。
    -> GroupKFold(n_splits=5, groups=episode_id) で分割し、リークを防ぐ。
  - 既存の split 列・weight 列は今回は使わない（前者はターン1-2の評価に足りない、
    後者は重み付きAUCの解釈が面倒なため）。
  - turn==0 の行（初期状態。両プレイヤーの初手が反映される前のスナップショット）は
    「ターン1-2」「3-5」「6-10」「11+」のいずれの帯にも属さないため、分析対象から除外する。
  - 信頼区間は試合単位（episode_id）のクラスタブートストラップ（復元抽出、1000回、
    パーセンタイル法95%区間）で出す。行を独立標本として扱わない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "kaggle_replays" / "value_net" / "features.npz"
OUT_DIR = Path(__file__).resolve().parent
# 出力ファイルパスは --out-prefix から main() 内で組み立てる(既定は下記と同じ名前になる:
# early_game_signal_results.json / early_game_signal.md)。

# sample_submission を sys.path に入れて encoder.FEATURE_NAMES を読む
sys.path.insert(0, str(REPO_ROOT / "sample_submission"))
from ptcg_ai.learning import encoder  # noqa: E402

FEATURE_NAMES = list(encoder.FEATURE_NAMES)

N_SPLITS = 5
N_BOOT = 1000
RNG_SEED = 0
BAND_ORDER = ["1-2", "3-5", "6-10", "11+"]
SINGLE_FEATURES = ["bench_count_diff", "is_first_player", "prize_diff", "self_energy_on_board"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=str, default=str(FEATURES_PATH),
                     help="解析対象のnpz。X/y/turn/<group-col>列を持つこと。既定は既存の value_net/features.npz。")
    ap.add_argument("--group-col", type=str, default="episode_id",
                     help="GroupKFold・クラスタブートストラップに使うグループ列名。既定 episode_id。")
    ap.add_argument("--out-prefix", type=str, default="early_game_signal",
                     help="出力する json/md のファイル名接頭辞。既定 early_game_signal（後方互換）。")
    ap.add_argument("--extra-baseline-col", type=str, default=None,
                     help="指定するとそのカテゴリ列のone-hotのみを特徴にしたLRベースラインを追加評価する"
                          "（例: opponent を渡すと「相手が誰かだけを見るモデル」のAUCが出る）。")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                     help="クラスタブートストラップの反復回数。既定1000。")
    return ap.parse_args()


def turn_band(turn: np.ndarray) -> np.ndarray:
    band = np.empty(turn.shape, dtype=object)
    band[(turn >= 1) & (turn <= 2)] = "1-2"
    band[(turn >= 3) & (turn <= 5)] = "3-5"
    band[(turn >= 6) & (turn <= 10)] = "6-10"
    band[turn >= 11] = "11+"
    return band


def cluster_bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    band_mask: np.ndarray,
    episode_id: np.ndarray,
    all_episodes: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, int]:
    """episode_id を復元抽出するクラスタブートストラップで95%区間を計算する。

    all_episodes: データセット全体のユニークな episode_id の配列（帯に依らない母集団）。
    band_mask: この帯に属する行のブールマスク（y, score, episode_id と同じ長さ）。
    戻り値: (lo, hi, n_valid_iters)
    """
    band_eid = episode_id[band_mask]
    band_y = y[band_mask]
    band_score = score[band_mask]

    # episode_id -> band内の行インデックス群、を事前に引けるようにする
    from collections import defaultdict

    idx_by_eid: dict = defaultdict(list)
    for i, e in enumerate(band_eid):
        idx_by_eid[e].append(i)

    n_episodes = len(all_episodes)
    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(all_episodes, size=n_episodes, replace=True)
        rows = []
        for e in sampled:
            rows.extend(idx_by_eid.get(e, ()))
        if not rows:
            continue
        yb = band_y[rows]
        sb = band_score[rows]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))

    if len(aucs) == 0:
        return (float("nan"), float("nan"), 0)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return (float(lo), float(hi), len(aucs))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    out_json = OUT_DIR / f"{args.out_prefix}_results.json"
    out_md = OUT_DIR / f"{args.out_prefix}.md"
    n_boot = args.n_boot

    t0 = time.time()
    print(f"loading {npz_path} ...")
    data = np.load(npz_path, allow_pickle=False)
    X_all = data["X"]
    y_all = data["y"].astype(np.int64)
    turn_all = data["turn"]
    episode_id_all = data[args.group_col]
    # weight, split は今回は使わない（指示により明示的に不使用）

    # turn==0 の行を除外（4つのターン帯のどれにも属さない初期状態スナップショット）
    keep = turn_all >= 1
    X = X_all[keep]
    y = y_all[keep]
    turn = turn_all[keep]
    episode_id = episode_id_all[keep]

    n_rows_total = len(y)
    n_episodes_total = len(np.unique(episode_id))
    print(f"rows (turn>=1): {n_rows_total}, episodes: {n_episodes_total}")
    print(f"(dropped turn==0 rows: {int((~keep).sum())})")

    band = turn_band(turn)

    # ---- 行数 / 試合数の集計 ----
    band_stats = {}
    for b in BAND_ORDER:
        m = band == b
        band_stats[b] = {
            "n_rows": int(m.sum()),
            "n_episodes": int(len(np.unique(episode_id[m]))),
        }
    print("band stats:", band_stats)

    all_episodes = np.unique(episode_id)
    rng_boot = np.random.default_rng(RNG_SEED)

    results: dict = {
        "meta": {
            "features_path": str(npz_path),
            "n_rows_total_turn_ge_1": n_rows_total,
            "n_episodes_total": n_episodes_total,
            "n_rows_turn0_dropped": int((~keep).sum()),
            "n_splits": N_SPLITS,
            "n_boot": n_boot,
            "note": f"weight列・split列は使用しない。GroupKFold(groups={args.group_col})を使用。",
        },
        "band_stats": band_stats,
        "single_feature": {},
        "models": {},
        "permutation_importance_early_gbm_turn1_2": [],
        "warnings": [],
    }

    # =========================================================
    # 1. 単一特徴ベースライン（学習不要。全行に対してそのままAUC）
    # =========================================================
    print("\n=== single-feature baselines ===")
    feat_idx = {name: FEATURE_NAMES.index(name) for name in SINGLE_FEATURES}
    for fname in SINGLE_FEATURES:
        fi = feat_idx[fname]
        col = X[:, fi].astype(np.float64)
        results["single_feature"][fname] = {}
        for b in BAND_ORDER:
            m = band == b
            auc_raw = safe_auc(y[m], col[m])
            flipped = False
            score_for_ci = col
            auc_report = auc_raw
            if not np.isnan(auc_raw) and auc_raw < 0.5:
                flipped = True
                auc_report = 1.0 - auc_raw
                score_for_ci = -col
            lo, hi, n_valid = cluster_bootstrap_ci(
                y, score_for_ci, m, episode_id, all_episodes, rng_boot, n_boot=n_boot
            )
            results["single_feature"][fname][b] = {
                "auc": auc_report,
                "flipped": flipped,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_boot_valid": n_valid,
            }
            print(f"  {fname:24s} band={b:5s} auc={auc_report:.4f} (flipped={flipped}) "
                  f"CI=[{lo:.4f},{hi:.4f}] n_boot={n_valid}")

    # =========================================================
    # GroupKFold の分割を1回作り、joint / early すべてで使い回す
    # =========================================================
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups=episode_id))

    early_mask = turn <= 5  # early-LR / early-GBM の学習データ制限（ターン1-5）

    def make_lr():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    def make_gbm():
        return HistGradientBoostingClassifier(random_state=0)

    def run_oof(model_factory, restrict_train_mask=None, need_models_for_perm=False, X_data=None):
        """GroupKFoldでOOF予測を作る。restrict_train_maskがあればその行だけで学習する。
        need_models_for_perm=Trueならfold毎の(model, val_idx)のリストも返す。
        X_data を渡すと、通常の特徴量行列 X の代わりにそちらを使う
        （--extra-baseline-col の one-hot ベースライン用）。
        """
        X_use = X if X_data is None else X_data
        oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_models = []
        for train_idx, val_idx in folds:
            if restrict_train_mask is not None:
                train_idx = train_idx[restrict_train_mask[train_idx]]
            model = model_factory()
            model.fit(X_use[train_idx], y[train_idx])
            proba = model.predict_proba(X_use[val_idx])[:, 1]
            oof[val_idx] = proba
            if need_models_for_perm:
                fold_models.append((model, val_idx))
        return oof, fold_models

    def summarize_model(name, oof):
        results["models"][name] = {}
        for b in BAND_ORDER:
            m = (band == b) & ~np.isnan(oof)
            if m.sum() == 0:
                results["models"][name][b] = {
                    "auc": None, "ci_lo": None, "ci_hi": None, "n_boot_valid": 0
                }
                continue
            auc = safe_auc(y[m], oof[m])
            lo, hi, n_valid = cluster_bootstrap_ci(
                y, oof, m, episode_id, all_episodes, rng_boot, n_boot=n_boot
            )
            results["models"][name][b] = {
                "auc": auc, "ci_lo": lo, "ci_hi": hi, "n_boot_valid": n_valid
            }
            print(f"  {name:12s} band={b:5s} auc={auc:.4f} CI=[{lo:.4f},{hi:.4f}] "
                  f"n_boot={n_valid} n_rows_in_band_with_pred={int(m.sum())}")

    # =========================================================
    # 2. joint-LR / 3. joint-GBM （全ターンの行で学習）
    # =========================================================
    print("\n=== joint-LR (train: all turns) ===")
    oof_joint_lr, _ = run_oof(make_lr)
    summarize_model("joint-LR", oof_joint_lr)

    print("\n=== joint-GBM (train: all turns) ===")
    oof_joint_gbm, _ = run_oof(make_gbm)
    summarize_model("joint-GBM", oof_joint_gbm)

    # =========================================================
    # 4. early-LR / 5. early-GBM （ターン1-5の行だけで学習、fold分割は共通）
    # =========================================================
    print("\n=== early-LR (train: turn<=5 rows only, same folds) ===")
    oof_early_lr, _ = run_oof(make_lr, restrict_train_mask=early_mask)
    summarize_model("early-LR", oof_early_lr)

    print("\n=== early-GBM (train: turn<=5 rows only, same folds) ===")
    oof_early_gbm, early_gbm_fold_models = run_oof(
        make_gbm, restrict_train_mask=early_mask, need_models_for_perm=True
    )
    summarize_model("early-GBM", oof_early_gbm)

    # early-LR/GBMは3-5帯までしか要求されていないが、6-10/11+も参考として一応出す
    # （学習データがturn<=5のみのため、band 6-10/11+はモデルの汎化を見る参考値）

    # =========================================================
    # extra baseline: --extra-baseline-col が指定されたら、そのカテゴリ列の one-hot だけを
    # 特徴にしたLRを評価する。「局面ではなく相手のデッキ(等)を当てているだけ」疑惑を
    # 直接測るためのベースライン（例: --extra-baseline-col opponent）。
    # =========================================================
    extra_baseline_name = None
    if args.extra_baseline_col:
        if args.extra_baseline_col not in data.files:
            raise ValueError(
                f"--extra-baseline-col={args.extra_baseline_col!r} が npz に存在しない "
                f"(利用可能な列: {list(data.files)})"
            )
        extra_baseline_name = f"onehot-{args.extra_baseline_col}"
        cat_all = data[args.extra_baseline_col]
        cat = cat_all[keep].reshape(-1, 1)

        def make_onehot_lr():
            return make_pipeline(
                OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=2000)
            )

        print(f"\n=== {extra_baseline_name} (train: all turns, feature: one-hot of "
              f"{args.extra_baseline_col!r} only) ===")
        oof_extra, _ = run_oof(make_onehot_lr, X_data=cat)
        summarize_model(extra_baseline_name, oof_extra)

    # =========================================================
    # 6. permutation importance: early-GBM, band=1-2
    #    各foldのearly-GBMモデルを、そのfoldの検証行のうちband=1-2のものに対して
    #    permutation importanceを計算し、サンプル数で重み付き平均する（OOFの考え方を維持）。
    # =========================================================
    print("\n=== permutation importance: early-GBM on turn1-2 (per-fold, weighted avg) ===")
    n_features = X.shape[1]
    importance_sum = np.zeros(n_features, dtype=np.float64)
    importance_weight = np.zeros(n_features, dtype=np.float64)
    total_val_rows_used = 0
    per_fold_info = []
    for fold_i, (model, val_idx) in enumerate(early_gbm_fold_models):
        val_band_mask = band[val_idx] == "1-2"
        val_band_idx = val_idx[val_band_mask]
        n_val_band = len(val_band_idx)
        per_fold_info.append(n_val_band)
        if n_val_band < 20:
            results["warnings"].append(
                f"fold {fold_i}: turn1-2 validation rows too few ({n_val_band}), skipped in permutation importance"
            )
            continue
        pi = permutation_importance(
            model, X[val_band_idx], y[val_band_idx],
            n_repeats=10, scoring="roc_auc", random_state=0
        )
        importance_sum += pi.importances_mean * n_val_band
        importance_weight += n_val_band
        total_val_rows_used += n_val_band
        print(f"  fold {fold_i}: n_val_turn1-2={n_val_band}")

    with np.errstate(invalid="ignore", divide="ignore"):
        importance_avg = np.where(importance_weight > 0, importance_sum / importance_weight, 0.0)

    top_idx = np.argsort(-importance_avg)[:15]
    perm_list = [
        {"feature": FEATURE_NAMES[i], "importance_mean_auc_drop": float(importance_avg[i])}
        for i in top_idx
    ]
    results["permutation_importance_early_gbm_turn1_2"] = perm_list
    results["meta"]["permutation_importance_total_val_rows_used"] = int(total_val_rows_used)
    results["meta"]["permutation_importance_per_fold_n_val_turn1_2"] = per_fold_info
    for row in perm_list:
        print(f"  {row['feature']:32s} {row['importance_mean_auc_drop']:.5f}")

    # =========================================================
    # 出力
    # =========================================================
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_json}")

    write_markdown(results, out_md, group_col=args.group_col, extra_baseline_name=extra_baseline_name)
    print(f"wrote {out_md}")

    print(f"\ntotal elapsed: {time.time() - t0:.1f}s")


def fmt_auc(entry) -> str:
    if entry is None or entry.get("auc") is None:
        return "N/A"
    auc = entry["auc"]
    lo = entry.get("ci_lo")
    hi = entry.get("ci_hi")
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)):
        return f"{auc:.4f} (CI計算不可)"
    return f"{auc:.4f} [{lo:.4f}, {hi:.4f}]"


def write_markdown(results: dict, out_md: Path, group_col: str = "episode_id",
                    extra_baseline_name: str | None = None) -> None:
    lines = []
    lines.append("# 序盤(ターン1-5)価値関数シグナル・プローブ 結果\n")
    lines.append(
        "問い: ターン1-5で価値関数のAUCが低いのは (a) 特徴量に序盤情報が無い / "
        "(b) 全ターン混合学習による希釈 / (c) 序盤は本質的に予測不能、のどれか。\n"
    )
    meta = results["meta"]
    lines.append(
        f"- データ: `{meta['features_path']}`\n"
        f"- 分析対象行数（turn>=1）: {meta['n_rows_total_turn_ge_1']} "
        f"（turn==0の {meta['n_rows_turn0_dropped']} 行は除外）\n"
        f"- 試合数（ユニーク{group_col}）: {meta['n_episodes_total']}\n"
        f"- 分割: GroupKFold(n_splits={meta['n_splits']}, groups={group_col})。"
        f"weight列・split列は不使用。\n"
        f"- ブートストラップ: {group_col}復元抽出、{meta['n_boot']}回、パーセンタイル法95%区間。\n"
    )

    lines.append("\n## ターン帯別の行数・試合数\n")
    lines.append("| ターン帯 | 行数 | 試合数(その帯に行がある episode 数) |")
    lines.append("|---|---|---|")
    for b in BAND_ORDER:
        bs = results["band_stats"][b]
        lines.append(f"| {b} | {bs['n_rows']} | {bs['n_episodes']} |")

    lines.append("\n## 1. 単一特徴ベースライン（学習なし、そのままAUC。AUC<0.5は反転して報告）\n")
    lines.append("| 特徴量 | " + " | ".join(BAND_ORDER) + " |")
    lines.append("|---|" + "---|" * len(BAND_ORDER))
    for fname, per_band in results["single_feature"].items():
        cells = []
        for b in BAND_ORDER:
            e = per_band[b]
            flip_note = "(反転)" if e["flipped"] else ""
            cells.append(f"{fmt_auc(e)}{flip_note}")
        lines.append(f"| {fname} | " + " | ".join(cells) + " |")

    lines.append("\n## 2-5. モデル比較（すべてOut-of-fold予測、95%区間つき）\n")
    lines.append("| モデル | " + " | ".join(BAND_ORDER) + " |")
    lines.append("|---|" + "---|" * len(BAND_ORDER))
    model_names = ["joint-LR", "joint-GBM", "early-LR", "early-GBM"]
    if extra_baseline_name is not None:
        model_names.append(extra_baseline_name)
    for name in model_names:
        cells = []
        for b in BAND_ORDER:
            e = results["models"][name].get(b)
            cells.append(fmt_auc(e))
        note = ""
        if name.startswith("early"):
            note = "（学習: turn<=5行のみ。6-10/11+は参考値）"
        elif name.startswith("onehot-"):
            note = "（学習: 全ターン。特徴量は指定列のone-hotのみ、局面情報は不使用）"
        lines.append(f"| {name}{note} | " + " | ".join(cells) + " |")

    lines.append(
        "\n注: early-LR/early-GBMの評価対象はターン1-2, 3-5が本来の比較対象。"
        "6-10, 11+はearlyモデルが学習していない領域への汎化なので参考程度。\n"
    )

    lines.append("\n## 6. permutation importance（early-GBM, ターン1-2の検証行, 上位15）\n")
    lines.append(
        f"fold毎のearly-GBMモデルを、そのfoldの検証データのうちターン1-2の行に対して適用し "
        f"(`n_repeats=10, scoring='roc_auc'`)、fold間をサンプル数で重み付き平均した値。"
        f"（使用した検証行の合計: {meta.get('permutation_importance_total_val_rows_used')}）\n"
    )
    lines.append("| 順位 | 特徴量 | importance (AUC低下量, 平均) |")
    lines.append("|---|---|---|")
    for i, row in enumerate(results["permutation_importance_early_gbm_turn1_2"], start=1):
        lines.append(f"| {i} | {row['feature']} | {row['importance_mean_auc_drop']:.5f} |")

    if results["warnings"]:
        lines.append("\n## 警告\n")
        for w in results["warnings"]:
            lines.append(f"- {w}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
