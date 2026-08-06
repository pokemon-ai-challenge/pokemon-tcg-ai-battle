#!/usr/bin/env python3
"""features.npz からバリューネットワーク(MLP)を学習し、value_weights.json を書き出す。

流れ:
  1. features.npz を読み、train/val/test split に分ける。
  2. 標準化(平均・標準偏差)を train split のみから計算する。
  3. ベースライン: ロジスティック回帰(sample_weight 使用)。
  4. 本命: 小型 MLP([64, 16], ReLU, 出力 sigmoid)。PyTorch で重み付き BCE loss + 早期終了(val 監視)。
  5. ターン帯別温度スケーリング(val split、T>=1.0 に制約)。
  6. sample_submission/ptcg_ai/learning/value_weights.json へエクスポート
     (W は [out_dim][in_dim] 行優先。PyTorch の nn.Linear.weight はそのままこの形)。
  7. 自己検証: 書き出した JSON を再読み込みし、純Python(numpy/torch非依存)のフォワードパスで
     val split から無作為抽出した50件を計算し、PyTorch の出力(較正前 p_raw)と比較(誤差 1e-4 以内)。
  8. sample_predictions.json: val split から無作為抽出した30件(観測データ付き)を書き出す。

使い方:
  PYTHONIOENCODING=utf-8 python train.py
  python train.py --features <features.npz> --out-weights <path> --metrics-out <path> \
      --sample-predictions-out <path>
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.learning.encoder import FEATURE_NAMES  # noqa: E402

_DEFAULT_FEATURES_PATH = _HERE / "features.npz"
_VALUE_POSITIONS_PATH = _HERE.parent / "training_data" / "value_positions.jsonl.gz"
_DEFAULT_WEIGHTS_OUT_PATH = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "value_weights.json"
_DEFAULT_SAMPLE_PREDICTIONS_OUT_PATH = _HERE / "sample_predictions.json"

_SEED = 42
_HIDDEN_SIZES = [64, 16]
_L2_WEIGHT_DECAY = 1e-4
_MAX_EPOCHS = 200
_PATIENCE = 10
_BATCH_SIZE = 4096
_LR = 1e-3

_SAMPLE_WEIGHT_SCHEME = {"1-50": 1.5, "51-200": 1.3, "201-1000": 1.1, "1001+": 1.0, "unknown": 1.0}

# ターン帯定義(kaggle_replays/extract_value_dataset.py の turn_band() と同じ境界を踏襲)。
_TURN_BANDS = [
    {"band": "1-2", "min_turn": 1, "max_turn": 2},
    {"band": "3-5", "min_turn": 3, "max_turn": 5},
    {"band": "6-10", "min_turn": 6, "max_turn": 10},
    {"band": "11+", "min_turn": 11, "max_turn": None},
    {"band": "unknown", "min_turn": None, "max_turn": None},
]

_T_BOUNDS = (1.0, 15.0)  # calibrate.py と同じ方針: T は 1.0 未満に張り付かせない(慎重側)。

_N_SELF_CHECK = 50
_SELF_CHECK_TOL = 1e-4
_N_SAMPLE_PREDICTIONS = 30


def turn_band_of(turn: int) -> str:
    """features.npz の turn 列(None は -1)から帯名を返す。turn_band() (extract_value_dataset.py) 踏襲。"""
    if turn is None or turn < 0:
        return "unknown"
    if turn <= 2:
        return "1-2"
    if turn <= 5:
        return "3-5"
    if turn <= 10:
        return "6-10"
    return "11+"


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


# ---------------------------------------------------------------------------
# 純Python(numpy/torch非依存)フォワードパス。自己検証専用。
# ---------------------------------------------------------------------------
def pure_python_forward(raw_features: list, weights_json: dict) -> float:
    """weights_json のスキーマ通りに、較正前の p_raw を素の Python で計算する。"""
    mean = weights_json["standardization"]["mean"]
    std = weights_json["standardization"]["std"]
    h = [(raw_features[i] - mean[i]) / std[i] for i in range(len(raw_features))]

    for layer in weights_json["layers"]:
        W = layer["W"]
        b = layer["b"]
        activation = layer["activation"]
        z = []
        for k in range(len(W)):
            row = W[k]
            s = b[k]
            for i in range(len(row)):
                s += row[i] * h[i]
            z.append(s)
        if activation == "relu":
            h = [max(0.0, v) for v in z]
        elif activation == "sigmoid":
            h = [sigmoid(v) for v in z]
        else:
            raise ValueError(f"未知の activation: {activation}")

    assert len(h) == 1
    return h[0]


# ---------------------------------------------------------------------------
# MLP (PyTorch)
# ---------------------------------------------------------------------------
class ValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: list[int]):
        super().__init__()
        dims = [in_dim] + hidden_sizes + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.linears = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = len(self.linears)
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < n - 1:
                x = torch.relu(x)
        return x.squeeze(-1)  # raw logit(sigmoid前)


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss_per_sample = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss_per_sample * weight).sum() / weight.sum()


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[ValueMLP, list]:
    torch.manual_seed(_SEED)
    rng = np.random.default_rng(_SEED)

    in_dim = X_train.shape[1]
    model = ValueMLP(in_dim, _HIDDEN_SIZES)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR, weight_decay=_L2_WEIGHT_DECAY)

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    w_train_t = torch.from_numpy(w_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    n = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0
    val_loss_history = []

    for epoch in range(1, _MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, _BATCH_SIZE):
            idx = perm[start : start + _BATCH_SIZE]
            idx_t = torch.from_numpy(idx.astype(np.int64))
            xb = X_train_t[idx_t]
            yb = y_train_t[idx_t]
            wb = w_train_t[idx_t]

            optimizer.zero_grad()
            logits = model(xb)
            loss = weighted_bce_with_logits(logits, yb, wb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(
                nn.functional.binary_cross_entropy_with_logits(val_logits, y_val_t, reduction="mean").item()
            )
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    epoch {epoch:3d}  train_loss={epoch_loss / n_batches:.4f}  val_loss={val_loss:.4f}"
                f"  (best={best_val_loss:.4f}, no_improve={epochs_since_improve})"
            )

        if epochs_since_improve >= _PATIENCE:
            print(f"    早期終了: epoch {epoch} (val_loss が {_PATIENCE} epoch 改善せず)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, val_loss_history


def mlp_predict_proba(model: ValueMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        probs = torch.sigmoid(logits)
    return probs.numpy()


def mlp_raw_logits(model: ValueMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
    return logits.numpy()


def extract_layers_json(model: ValueMLP) -> list[dict]:
    """PyTorch モデルから W[out][in] 形式の layers を作る。

    nn.Linear.weight は既に shape (out_features, in_features) なので転置不要。
    最終層のみ activation="sigmoid"、それ以外は "relu"。
    """
    layers = []
    n = len(model.linears)
    for i, lin in enumerate(model.linears):
        W = lin.weight.detach().numpy().astype(np.float64)
        b = lin.bias.detach().numpy().astype(np.float64)
        activation = "sigmoid" if i == n - 1 else "relu"
        layers.append({"W": W.tolist(), "b": b.tolist(), "activation": activation})
    return layers


# ---------------------------------------------------------------------------
# 温度スケーリング(ターン帯別)
# ---------------------------------------------------------------------------
def neg_log_likelihood_binary(temperature: float, logits: np.ndarray, y: np.ndarray) -> float:
    z = logits / temperature
    # 数値安定な binary cross entropy(logit 版)。
    # log(1+exp(-|z|)) + max(z,0) - z*y  == -log p(y|z) の安定形。
    loss = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - z * y
    return float(loss.mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(T*, nll_before(T=1), nll_after(T=T*)) を返す。calibrate.py と同じ方針(T>=1.0 に制約)。"""
    nll_before = neg_log_likelihood_binary(1.0, logits, y)
    if len(logits) == 0:
        return 1.0, nll_before, nll_before
    result = minimize_scalar(
        lambda t: neg_log_likelihood_binary(t, logits, y), bounds=_T_BOUNDS, method="bounded"
    )
    t_star = float(result.x)
    nll_after = neg_log_likelihood_binary(t_star, logits, y)
    if nll_after > nll_before:
        t_star = 1.0
        nll_after = nll_before
    return t_star, nll_before, nll_after


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES_PATH), help="入力 features.npz")
    parser.add_argument(
        "--out-weights", default=str(_DEFAULT_WEIGHTS_OUT_PATH),
        help="重みJSON書き出し先(既定: 本番パス。実験時は別パスを明示指定すること)",
    )
    parser.add_argument(
        "--metrics-out", default=None,
        help="学習指標(LR/MLP の val・test AUC・logloss、較正温度など)のJSON書き出し先"
        "(既定: 書き出さない)。",
    )
    parser.add_argument(
        "--sample-predictions-out", default=str(_DEFAULT_SAMPLE_PREDICTIONS_OUT_PATH),
        help="sample_predictions.json の書き出し先(既定: value_net/sample_predictions.json)",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    weights_out_path = Path(args.out_weights)
    sample_predictions_out_path = Path(args.sample_predictions_out)
    metrics_out_path = Path(args.metrics_out) if args.metrics_out else None

    print(f"features.npz を読み込み: {features_path}")
    data = np.load(features_path, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    turn = data["turn"].astype(np.int64)
    weight = data["weight"].astype(np.float64)
    split = data["split"].astype(np.int64)
    episode_id = data["episode_id"]
    player_index = data["player_index"]
    step_index = data["step_index"]

    assert X.shape[1] == len(FEATURE_NAMES), (
        f"特徴次元がencoder.FEATURE_NAMESと不一致: {X.shape[1]} != {len(FEATURE_NAMES)}"
    )

    train_mask = split == 0
    val_mask = split == 1
    test_mask = split == 2
    print(
        f"train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()} "
        f"(全{len(y)}件)"
    )

    X_train, y_train, w_train = X[train_mask], y[train_mask], weight[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], weight[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], weight[test_mask]
    turn_val = turn[val_mask]
    turn_test = turn[test_mask]

    # --- 標準化(train split のみから計算) ---
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std_clipped = np.where(std < 1e-6, 1e-6, std)

    X_train_std = (X_train - mean) / std_clipped
    X_val_std = (X_val - mean) / std_clipped
    X_test_std = (X_test - mean) / std_clipped

    # --- ベースライン: ロジスティック回帰 ---
    print("\n=== ベースライン: ロジスティック回帰 ===")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_std, y_train, sample_weight=w_train)
    print(f"  学習完了 ({time.time() - t0:.1f}s)")

    lr_val_proba = lr.predict_proba(X_val_std)[:, 1]
    lr_val_auc = roc_auc_score(y_val, lr_val_proba)
    lr_val_logloss = log_loss(y_val, lr_val_proba)
    print(f"  val AUC={lr_val_auc:.4f}  logloss={lr_val_logloss:.4f}")

    # --- 本命: MLP ---
    print("\n=== 本命: MLP([64, 16], ReLU, sigmoid出力) ===")
    t0 = time.time()
    model, val_loss_history = train_mlp(X_train_std, y_train, w_train, X_val_std, y_val)
    print(f"  学習完了 ({time.time() - t0:.1f}s, {len(val_loss_history)} epochs)")

    mlp_val_proba = mlp_predict_proba(model, X_val_std)
    mlp_val_auc = roc_auc_score(y_val, mlp_val_proba)
    mlp_val_logloss = log_loss(y_val, mlp_val_proba)
    print(f"  val AUC={mlp_val_auc:.4f}  logloss={mlp_val_logloss:.4f}")

    mlp_test_proba = mlp_predict_proba(model, X_test_std)
    mlp_test_auc = roc_auc_score(y_test, mlp_test_proba)
    mlp_test_logloss = log_loss(y_test, mlp_test_proba)
    print(f"  test AUC={mlp_test_auc:.4f}  logloss={mlp_test_logloss:.4f}")

    print("\n=== LR vs MLP (val split) 比較 ===")
    print(f"  LogisticRegression : AUC={lr_val_auc:.4f}  logloss={lr_val_logloss:.4f}")
    print(f"  MLP                : AUC={mlp_val_auc:.4f}  logloss={mlp_val_logloss:.4f}")

    # --- ターン帯別温度スケーリング(val split, MLP の生ロジット) ---
    print("\n=== ターン帯別 温度スケーリング(val split, T>=1.0 制約) ===")
    val_raw_logits = mlp_raw_logits(model, X_val_std)
    val_bands = np.array([turn_band_of(int(t)) for t in turn_val])

    calibration_buckets = []
    for band_def in _TURN_BANDS:
        band = band_def["band"]
        mask = val_bands == band
        n_band = int(mask.sum())
        if n_band == 0:
            t_star, nll_before, nll_after = 1.0, float("nan"), float("nan")
            print(f"  band={band:8s} n=0 (val split にサンプルなし) -> T=1.0 (フォールバック)")
        else:
            t_star, nll_before, nll_after = fit_temperature(val_raw_logits[mask], y_val[mask].astype(np.float64))
            print(
                f"  band={band:8s} n={n_band:6d}  T*={t_star:.3f}  "
                f"nll: {nll_before:.4f} -> {nll_after:.4f}"
            )
        calibration_buckets.append(
            {
                "band": band,
                "min_turn": band_def["min_turn"],
                "max_turn": band_def["max_turn"],
                "temperature": t_star,
            }
        )

    # --- JSON エクスポート ---
    layers_json = extract_layers_json(model)
    weights_json = {
        "schema_version": 1,
        "model_type": "mlp",
        "feature_names": list(FEATURE_NAMES),
        "standardization": {"mean": mean.tolist(), "std": std_clipped.tolist()},
        "layers": layers_json,
        "meta": {
            "calibration": {"axis": "turn_band", "buckets": calibration_buckets},
            "sample_weight_scheme": _SAMPLE_WEIGHT_SCHEME,
            "trained_positions": int(train_mask.sum()),
            "val_metrics": {"auc": float(mlp_val_auc), "logloss": float(mlp_val_logloss)},
            "test_metrics": {"auc": float(mlp_test_auc), "logloss": float(mlp_test_logloss)},
            "baseline_val_metrics_logreg": {"auc": float(lr_val_auc), "logloss": float(lr_val_logloss)},
        },
    }

    weights_out_path.parent.mkdir(parents=True, exist_ok=True)
    weights_out_path.write_text(json.dumps(weights_json, ensure_ascii=False), encoding="utf-8")
    print(f"\n重みを書き出しました: {weights_out_path}")

    # --- 自己検証 ---
    print("\n=== 自己検証: 純Pythonフォワードパス vs PyTorch出力(val split から50件) ===")
    reloaded = json.loads(weights_out_path.read_text(encoding="utf-8"))

    val_indices = np.where(val_mask)[0]
    rng = random.Random(_SEED)
    check_positions = rng.sample(range(len(val_indices)), min(_N_SELF_CHECK, len(val_indices)))

    max_abs_err = 0.0
    n_checked = 0
    for pos in check_positions:
        raw_x = X_val[pos]  # 標準化前の生特徴(元の X から val 部分を取り出したもの)
        expected = float(mlp_val_proba[pos])  # ライブラリ側の p_raw(較正前)
        got = pure_python_forward(raw_x.tolist(), reloaded)
        err = abs(got - expected)
        max_abs_err = max(max_abs_err, err)
        n_checked += 1

    self_check_pass = max_abs_err <= _SELF_CHECK_TOL
    print(f"  検証件数={n_checked}  最大誤差={max_abs_err:.8f}  許容={_SELF_CHECK_TOL}")
    print(f"  結果: {'PASS' if self_check_pass else 'FAIL'}")
    if not self_check_pass:
        print(
            "  エラー: 自己検証に失敗しました。W の向き・標準化の適用順序を確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 自己検証その2: 較正込みの数式(ln比 -> sigmoid(logit/T))が生ロジット直接割りと一致するか ---
    print("\n=== 自己検証: 較正数式(logit=ln(p/(1-p)) 経由) vs 生ロジット直接較正 ===")
    calib_max_err = 0.0
    for pos in check_positions[:10]:
        raw_x = X_val[pos]
        t = int(turn_val[pos])
        band = turn_band_of(t)
        temp = next(b["temperature"] for b in calibration_buckets if b["band"] == band)

        p_raw = pure_python_forward(raw_x.tolist(), reloaded)
        p_raw_clamped = min(max(p_raw, 1e-12), 1 - 1e-12)
        logit = math.log(p_raw_clamped / (1 - p_raw_clamped))
        p_calibrated = sigmoid(logit / temp)

        raw_x_std = (raw_x - mean) / std_clipped
        p_raw_direct = sigmoid(float(mlp_raw_logits(model, raw_x_std.reshape(1, -1))[0]) / temp)
        calib_max_err = max(calib_max_err, abs(p_calibrated - p_raw_direct))
    print(f"  最大誤差(較正込み経路の一致確認)={calib_max_err:.8f}")

    # --- sample_predictions.json ---
    print(f"\n=== sample_predictions.json 生成(val split から無作為{_N_SAMPLE_PREDICTIONS}件) ===")
    sample_positions = rng.sample(range(len(val_indices)), min(_N_SAMPLE_PREDICTIONS, len(val_indices)))
    sample_keys = set()
    key_to_pos = {}
    for pos in sample_positions:
        idx = val_indices[pos]
        key = (str(episode_id[idx]), int(player_index[idx]), int(step_index[idx]))
        sample_keys.add(key)
        key_to_pos[key] = pos

    observations_by_key: dict[tuple[str, int, int], dict] = {}
    with gzip.open(_VALUE_POSITIONS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["episode_id"]), int(row["player_index"]), int(row["step_index"]))
            if key in sample_keys:
                observations_by_key[key] = row
                if len(observations_by_key) == len(sample_keys):
                    break

    sample_predictions = []
    for key in sample_keys:
        pos = key_to_pos[key]
        idx = val_indices[pos]
        row = observations_by_key.get(key)
        if row is None:
            print(f"  警告: value_positions.jsonl.gz 内に見つかりませんでした: {key}", file=sys.stderr)
            continue

        raw_x = X_val[pos]
        t = int(turn_val[pos])
        band = turn_band_of(t)
        temp = next(b["temperature"] for b in calibration_buckets if b["band"] == band)

        p_raw = pure_python_forward(raw_x.tolist(), reloaded)
        p_raw_clamped = min(max(p_raw, 1e-12), 1 - 1e-12)
        logit = math.log(p_raw_clamped / (1 - p_raw_clamped))
        p_calibrated = sigmoid(logit / temp)

        sample_predictions.append(
            {
                "episode_id": key[0],
                "player_index": key[1],
                "step_index": key[2],
                "turn": t,
                "label": int(y_val[pos]),
                "observation": row["observation"],
                "feature_vector": raw_x.tolist(),
                "expected_win_prob": p_calibrated,
            }
        )

    sample_predictions_out_path.write_text(
        json.dumps(sample_predictions, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  書き出し完了: {sample_predictions_out_path} ({len(sample_predictions)}件)")

    # --- 最終サマリ ---
    print("\n" + "=" * 60)
    print("学習パイプライン完了サマリ")
    print("=" * 60)
    print(f"  LogisticRegression val: AUC={lr_val_auc:.4f} logloss={lr_val_logloss:.4f}")
    print(f"  MLP               val: AUC={mlp_val_auc:.4f} logloss={mlp_val_logloss:.4f}")
    print(f"  MLP               test: AUC={mlp_test_auc:.4f} logloss={mlp_test_logloss:.4f}")
    print("  ターン帯別温度:")
    for b in calibration_buckets:
        print(f"    {b['band']:8s} T={b['temperature']:.3f}")
    print(f"  自己検証: 最大誤差={max_abs_err:.8f} -> {'PASS' if self_check_pass else 'FAIL'}")
    print(f"  出力: {weights_out_path} ({weights_out_path.stat().st_size / 1e3:.1f} KB)")
    print(
        f"  出力: {sample_predictions_out_path} "
        f"({sample_predictions_out_path.stat().st_size / 1e3:.1f} KB)"
    )

    # --- --metrics-out(指定時のみ) ---
    if metrics_out_path is not None:
        metrics_json = {
            "features_path": str(features_path),
            "out_weights_path": str(weights_out_path),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "logreg_val_metrics": {"auc": float(lr_val_auc), "logloss": float(lr_val_logloss)},
            "mlp_val_metrics": {"auc": float(mlp_val_auc), "logloss": float(mlp_val_logloss)},
            "mlp_test_metrics": {"auc": float(mlp_test_auc), "logloss": float(mlp_test_logloss)},
            "calibration_buckets": calibration_buckets,
            "self_check_max_abs_err": max_abs_err,
            "self_check_pass": self_check_pass,
        }
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_out_path.write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  出力: {metrics_out_path}")


if __name__ == "__main__":
    main()
