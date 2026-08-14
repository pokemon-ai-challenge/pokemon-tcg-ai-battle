"""design.md(ogerpon_single_prize_mlp_design.md)§8, §10(Phase3 item2〜3): カプ・ブルル
中継戦略Q-critic(multi-head MLP ensemble)の学習・較正・純Python export。

``collect_ogerpon_counterfactuals.py`` が書き出したJSONL(1行 = 1つの
(state, option, determinization)のpaired rollout結果)を読み込み、``(state_id,
option_name)`` ごとに複数決定化の結果を集約してから学習する。encoderは
``sample_submission/ptcg_ai/learning/ogerpon_strategy_encoder.py`` を再利用する
(train/runtime parity)。

## 使い方

    python kaggle_replays/rl/train_ogerpon_strategy.py \
        --data kaggle_replays/rl/runs/ogerpon_counterfactuals.jsonl \
        --output sample_submission/ptcg_ai/learning/ogerpon_strategy_weights.json

## モデル構造(design.md §8.1)

    continuous_features(標準化) ++ option_features(標準化) ++ 12スロット分のcard embedding
                     ↓
    Linear(入力, 128) + ReLU
                     ↓
    Linear(128, 64) + ReLU
                     ↓
    共有表現64
      win_head / loop_complete_head / opponent_ko_count_head / signed_terminal_turns_head

推論側(``ogerpon_strategy_model.py``)と完全に同じ構造・順序で重みをexportする
(``kaggle_replays/rl/tests`` 側でparityは検証しない。純Python側のparityは
``--parity-check`` で学習直後に確認する)。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matchup_common import atomic_write_json, git_commit_sha, sha256_file  # noqa: E402

EX_TEMPO = "EX_TEMPO"
SINGLE_PRIZE_ROTATION = "SINGLE_PRIZE_ROTATION"
HEADS = ("win", "loop_complete", "opponent_ko_count", "signed_terminal_turns")
PARITY_MAX_ABS_DIFF = 1e-5

DEFAULTS = {
    "learning_rate": 3.0e-4, "weight_decay": 1.0e-4, "batch_size": 256,
    "max_epochs": 100, "early_stopping_patience": 10, "gradient_clip_norm": 1.0,
    "ensemble_seeds": [1103, 2207, 3301],
    "hidden1": 128, "hidden2": 64, "embedding_dim": 4,
    "loss_weights": {"win": 1.00, "loop_complete": 0.20, "opponent_ko_count": 0.10,
                     "signed_terminal_turns": 0.10, "pairwise_ranking": 0.25},
    "pairwise_ranking_min_gap": 0.02,  # 勝率差がこれ未満のペアはranking lossから除外する
    "min_calibration_samples": 10,     # これ未満ならvalidationでの較正をスキップ(温度1.0)
    "max_error_rate": 0.05,            # rollout error率がこれを超えたら学習を中止する
    "max_incomplete_pair_rate": 0.5,   # 片側Option欠損pairの比率がこれを超えたら学習を中止する
}


# ---------------------------------------------------------------------------
# 1. データ読み込み・(state_id, option_name)への集約
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_bad_float(v) -> bool:
    return isinstance(v, float) and (v != v or v in (float("inf"), float("-inf")))


def validate_dataset(records: list[dict], cfg: dict) -> dict:
    """学習前のデータ品質検証(design.md §9.6 + 外部レビュー指摘)。

    不正なoption_name・NaN/Infは1件でも致命的として中止する。rollout error率・
    不完全pair率は閾値超過で中止する(閾値以下は許容し、reportとして返す)。
    """
    n = len(records)
    n_errors = sum(1 for r in records if r.get("outcome", {}).get("error"))
    bad_option_names = sum(
        1 for r in records if r.get("option_name") not in (EX_TEMPO, SINGLE_PRIZE_ROTATION, None))
    nan_inf = 0
    for r in records:
        for key in ("continuous_features", "option_features"):
            for v in (r.get(key) or []):
                if _is_bad_float(v):
                    nan_inf += 1

    pair_options: dict[str, set] = defaultdict(set)
    for r in records:
        if r.get("option_name"):
            pair_options[r["pair_id"]].add(r["option_name"])
    n_pairs = len(pair_options)
    n_incomplete = sum(1 for opts in pair_options.values() if len(opts) != 2)

    error_rate = (n_errors / n) if n else 0.0
    incomplete_rate = (n_incomplete / n_pairs) if n_pairs else 0.0
    report = {
        "n_records": n, "n_errors": n_errors, "error_rate": error_rate,
        "n_bad_option_names": bad_option_names, "n_nan_inf_values": nan_inf,
        "n_pairs": n_pairs, "n_incomplete_pairs": n_incomplete,
        "incomplete_pair_rate": incomplete_rate,
    }
    if bad_option_names > 0:
        raise SystemExit(f"不正なoption_nameを検出。学習を中止する: {report}")
    if nan_inf > 0:
        raise SystemExit(f"NaN/Infを含む特徴量を検出。学習を中止する: {report}")
    if error_rate > cfg["max_error_rate"]:
        raise SystemExit(
            f"rollout error率が閾値超過({error_rate:.1%} > {cfg['max_error_rate']:.1%})。"
            f"学習を中止する: {report}")
    if incomplete_rate > cfg["max_incomplete_pair_rate"]:
        raise SystemExit(
            f"不完全なOptionペアの比率が閾値超過({incomplete_rate:.1%} > "
            f"{cfg['max_incomplete_pair_rate']:.1%})。学習を中止する: {report}")
    return report


def aggregate_by_state_option(records: list[dict]) -> dict[str, dict[str, dict]]:
    """``(sample_id, option_name)`` ごとに複数決定化のoutcomeを平均する。

    ``sample_id``(collector側で match_seed + learner_index + opponent_archetype + turn +
    state_id + 試合内発火連番 から作る一意ID)で集約する。``state_id``(公開盤面の内容
    ハッシュ)だけで集約すると、異なる試合が偶然同じ盤面内容に達した場合に、無関係な
    2つのtrigger事象のrolloutを1つの学習サンプルへ誤って混ぜてしまう(外部レビュー指摘)。
    ``state_id`` は情報として結果dictに残すが、集約キーとしては使わない。

    features(continuous_features/slot_card_ids/option_features)は同じ
    (sample_id, option_name)内で決定化に依らず同一(collector側の仕様)なので先頭行を使う。
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["sample_id"], r["option_name"])].append(r)

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (sample_id, option_name), rows in groups.items():
        wins, loops, kos, signed_turns = [], [], [], []
        for r in rows:
            o = r["outcome"]
            if o.get("error") is not None or o.get("win") is None:
                continue
            wins.append(float(o["win"]))
            if option_name == SINGLE_PRIZE_ROTATION:
                loops.append(1.0 if o.get("loop_complete") else 0.0)
            kos.append(float(o.get("opponent_ko_count") or 0))
            terminal_turns = float(o.get("terminal_turns") or 0)
            signed_turns.append(terminal_turns if o["win"] == 1 else -terminal_turns)
        if not wins:
            continue
        first = rows[0]
        out[sample_id][option_name] = {
            "sample_id": sample_id, "state_id": first.get("state_id"), "option_name": option_name,
            "opponent_archetype": first["opponent_archetype"],
            "match_seed": first["match_seed"], "trigger_kind": first["trigger_kind"],
            "learner_index": first["learner_index"],
            "continuous_features": first["continuous_features"],
            "slot_card_ids": first["slot_card_ids"],
            "option_features": first["option_features"],
            "win_rate": sum(wins) / len(wins),
            "loop_complete_rate": (sum(loops) / len(loops)) if loops else None,
            "mean_opp_kos": sum(kos) / len(kos),
            "mean_signed_turns": sum(signed_turns) / len(signed_turns),
            "n": len(wins),
        }
    return out


# ---------------------------------------------------------------------------
# 2. train/validation/test split(design.md §10.1: state_id単位、match単位で揃える)
# ---------------------------------------------------------------------------

def split_state_ids(by_state: dict, seed: int, train=0.70, val=0.15) -> tuple[list, list, list]:
    """sample_id(``aggregate_by_state_option`` の出力キー)をmatch_seed単位でグルーピングし
    (同一match由来の近接状態は同じsplitへ)、対面(opponent_archetype)x先後(learner_index)で
    層化した上で分割する(design.md §10.1)。層ごとの件数が少ない場合でも単純な按分に
    フォールバックし、壊れずに動く。
    """
    by_match: dict[int, list[str]] = defaultdict(list)
    match_stratum: dict[int, tuple] = {}
    for sid, opts in by_state.items():
        seeds = {o["match_seed"] for o in opts.values()}
        match_seed = min(seeds) if seeds else 0
        by_match[match_seed].append(sid)
        sample = next(iter(opts.values()))
        match_stratum[match_seed] = (sample["opponent_archetype"], sample["learner_index"])

    strata: dict[tuple, list[int]] = defaultdict(list)
    for match_seed in by_match:
        strata[match_stratum[match_seed]].append(match_seed)

    rng = random.Random(seed)
    train_keys, val_keys, test_keys = [], [], []
    for stratum_matches in strata.values():
        rng.shuffle(stratum_matches)
        n = len(stratum_matches)
        n_train = max(1, round(n * train)) if n > 2 else n
        n_val = max(0, round(n * val)) if n > 2 else 0
        train_keys += stratum_matches[:n_train]
        val_keys += stratum_matches[n_train:n_train + n_val]
        test_keys += stratum_matches[n_train + n_val:]

    def _flat(keys):
        out = []
        for k in keys:
            out += by_match[k]
        return out

    return _flat(train_keys), _flat(val_keys), _flat(test_keys)


# ---------------------------------------------------------------------------
# 3. 正規化(train splitだけで計算)
# ---------------------------------------------------------------------------

def compute_standardization(examples: list[dict]) -> dict:
    def _mean_std(vectors):
        n = len(vectors)
        dim = len(vectors[0])
        mean = [sum(v[i] for v in vectors) / n for i in range(dim)]
        var = [sum((v[i] - mean[i]) ** 2 for v in vectors) / n for i in range(dim)]
        std = [max(1e-6, v ** 0.5) for v in var]
        return mean, std

    cont_mean, cont_std = _mean_std([e["continuous_features"] for e in examples])
    opt_mean, opt_std = _mean_std([e["option_features"] for e in examples])
    return {
        "continuous_mean": cont_mean, "continuous_std": cont_std,
        "option_mean": opt_mean, "option_std": opt_std,
    }


def card_id_max_from_engine() -> int:
    from cg.api import all_card_data
    return max((c.cardId for c in all_card_data()), default=0)


# ---------------------------------------------------------------------------
# 4. PyTorchモデル
# ---------------------------------------------------------------------------

class OgerponQNet(nn.Module):
    def __init__(self, continuous_dim, option_dim, slot_count, card_id_max, embedding_dim,
                hidden1, hidden2):
        super().__init__()
        self.slot_count = slot_count
        self.embedding = nn.Embedding(card_id_max + 1, embedding_dim, padding_idx=0)
        input_dim = continuous_dim + option_dim + slot_count * embedding_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.heads = nn.ModuleDict({h: nn.Linear(hidden2, 1) for h in HEADS})

    def forward(self, continuous, option, slot_ids):
        emb = self.embedding(slot_ids).reshape(slot_ids.shape[0], -1)
        x = torch.cat([continuous, option, emb], dim=1)
        h = self.shared(x)
        return {name: layer(h).squeeze(-1) for name, layer in self.heads.items()}


def _to_tensors(examples: list[dict], std: dict, device):
    cont_mean = torch.tensor(std["continuous_mean"], dtype=torch.float32, device=device)
    cont_std = torch.tensor(std["continuous_std"], dtype=torch.float32, device=device)
    opt_mean = torch.tensor(std["option_mean"], dtype=torch.float32, device=device)
    opt_std = torch.tensor(std["option_std"], dtype=torch.float32, device=device)

    continuous = torch.tensor([e["continuous_features"] for e in examples],
                              dtype=torch.float32, device=device)
    option = torch.tensor([e["option_features"] for e in examples],
                          dtype=torch.float32, device=device)
    slots = torch.tensor([e["slot_card_ids"] for e in examples], dtype=torch.long, device=device)
    continuous = (continuous - cont_mean) / cont_std
    option = (option - opt_mean) / opt_std

    win = torch.tensor([e["win_rate"] for e in examples], dtype=torch.float32, device=device)
    loop_mask = torch.tensor([1.0 if e["loop_complete_rate"] is not None else 0.0 for e in examples],
                             dtype=torch.float32, device=device)
    loop = torch.tensor([e["loop_complete_rate"] or 0.0 for e in examples],
                        dtype=torch.float32, device=device)
    kos = torch.tensor([e["mean_opp_kos"] for e in examples], dtype=torch.float32, device=device)
    turns = torch.tensor([e["mean_signed_turns"] for e in examples], dtype=torch.float32, device=device)
    return continuous, option, slots, {"win": win, "loop_complete": loop, "opponent_ko_count": kos,
                                       "signed_terminal_turns": turns, "loop_mask": loop_mask}


def _pairwise_ranking_pairs(examples: list[dict], min_gap: float) -> list[tuple[int, int, float]]:
    """train_examples内で同じsample_id(同一trigger事象)の両Optionが揃っている行番号ペアを
    列挙する。``state_id``(内容ハッシュ)ではなく``sample_id``でマッチングする理由は
    ``aggregate_by_state_option`` と同じ(内容衝突で無関係な事象のEX_TEMPO/SINGLE_PRIZE_ROTATION
    を誤ってペアリングしないため)。

    戻り値は (single_prize側のindex, ex_tempo側のindex, sign) のリスト。sign は
    「実際にどちらのwin_rateが高いか」(+1ならSINGLE側が高い)。勝率差が
    ``min_gap`` 未満のペアは除外する(design.md §8.3: 乱数差を強い教師として扱わない)。
    """
    by_state: dict[str, dict[str, int]] = defaultdict(dict)
    for i, e in enumerate(examples):
        by_state[e["sample_id"]][e["option_name"]] = i
    pairs = []
    for sid, opts in by_state.items():
        if EX_TEMPO not in opts or SINGLE_PRIZE_ROTATION not in opts:
            continue
        i_single, i_ex = opts[SINGLE_PRIZE_ROTATION], opts[EX_TEMPO]
        gap = examples[i_single]["win_rate"] - examples[i_ex]["win_rate"]
        if abs(gap) < min_gap:
            continue
        pairs.append((i_single, i_ex, 1.0 if gap > 0 else -1.0))
    return pairs


def train_one_model(train_examples, val_examples, std, contract, cfg, seed, device):
    torch.manual_seed(seed)
    model = OgerponQNet(
        len(std["continuous_mean"]), len(std["option_mean"]), contract["slot_count"],
        contract["card_id_max"], cfg["embedding_dim"], cfg["hidden1"], cfg["hidden2"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.SmoothL1Loss()

    tr_cont, tr_opt, tr_slots, tr_y = _to_tensors(train_examples, std, device)
    pairs = _pairwise_ranking_pairs(train_examples, cfg["pairwise_ranking_min_gap"])

    best_val, best_state, patience = float("inf"), None, 0
    n = len(train_examples)
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n)
        batch_size = min(cfg["batch_size"], n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            out = model(tr_cont[idx], tr_opt[idx], tr_slots[idx])
            loss = cfg["loss_weights"]["win"] * bce(out["win"], tr_y["win"][idx]).mean()
            loop_bce = bce(out["loop_complete"], tr_y["loop_complete"][idx])
            masked = loop_bce * tr_y["loop_mask"][idx]
            denom = tr_y["loop_mask"][idx].sum().clamp(min=1.0)
            loss = loss + cfg["loss_weights"]["loop_complete"] * (masked.sum() / denom)
            loss = loss + cfg["loss_weights"]["opponent_ko_count"] * huber(
                out["opponent_ko_count"], tr_y["opponent_ko_count"][idx])
            loss = loss + cfg["loss_weights"]["signed_terminal_turns"] * huber(
                out["signed_terminal_turns"], tr_y["signed_terminal_turns"][idx])

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            opt.step()

        # pairwise ranking loss は state_id をまたいだ比較が要るため、ミニバッチではなく
        # train集合全体に対して epoch ごとに1回だけ追加の勾配ステップを踏む。
        if pairs and cfg["loss_weights"]["pairwise_ranking"] > 0:
            out_full = model(tr_cont, tr_opt, tr_slots)
            win_logit = out_full["win"]
            losses = []
            for i_single, i_ex, sign in pairs:
                diff = sign * (win_logit[i_single] - win_logit[i_ex])
                losses.append(-torch.log(torch.sigmoid(diff).clamp(min=1e-6)))
            rank_loss = cfg["loss_weights"]["pairwise_ranking"] * torch.stack(losses).mean()
            opt.zero_grad()
            rank_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            opt.step()

        model.eval()
        with torch.no_grad():
            if val_examples:
                v_cont, v_opt, v_slots, v_y = _to_tensors(val_examples, std, device)
                out = model(v_cont, v_opt, v_slots)
                val_loss = bce(out["win"], v_y["win"]).mean().item()
            else:
                out = model(tr_cont, tr_opt, tr_slots)
                val_loss = bce(out["win"], tr_y["win"]).mean().item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, patience = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ---------------------------------------------------------------------------
# 5. 較正(design.md §8.4): validation setで温度Tをグリッド探索する
#
# winとloop_completeは別headであり、別々に較正する(loop_completeはSINGLE_PRIZE_ROTATION
# の行にしか意味を持たない)。validationが``min_calibration_samples``未満の場合は
# train dataへ暗黙にフォールバックせず、温度1.0(未較正)のまま返す
# (外部レビュー指摘: heldoutではないデータでの較正はoverconfidenceバイアスを生む)。
# ---------------------------------------------------------------------------

def _ensemble_head_prob(models: list[nn.Module], cont, opt_feat, slots, head: str, temperature: float):
    with torch.no_grad():
        probs = [torch.sigmoid(m(cont, opt_feat, slots)[head] / temperature) for m in models]
    return torch.stack(probs, dim=0).mean(dim=0)


def _ensemble_win_prob(models, cont, opt_feat, slots, temperature: float):
    return _ensemble_head_prob(models, cont, opt_feat, slots, "win", temperature)


def calibrate_head_temperature(models, val_examples, std, device, head: str, min_samples: int,
                               grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)) -> tuple[float, str]:
    """``head`` の温度をvalidationで探索する。件数不足なら1.0(未較正)を返す。

    Returns:
        (temperature, status)。status は ``"calibrated"`` または
        ``"skipped_insufficient_validation"``。
    """
    if head == "loop_complete":
        usable = [e for e in val_examples
                 if e["option_name"] == SINGLE_PRIZE_ROTATION and e["loop_complete_rate"] is not None]
    else:
        usable = list(val_examples)
    if len(usable) < min_samples:
        return 1.0, "skipped_insufficient_validation"

    cont, opt_feat, slots, y = _to_tensors(usable, std, device)
    target = y[head]
    best_t, best_nll = 1.0, float("inf")
    eps = 1e-6
    for t in grid:
        p = _ensemble_head_prob(models, cont, opt_feat, slots, head, t).clamp(eps, 1 - eps)
        nll = -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean().item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t, "calibrated"


def evaluate_metrics(models, examples, std, device, temperature: float) -> dict:
    if not examples:
        return {}
    cont, opt_feat, slots, y = _to_tensors(examples, std, device)
    p = _ensemble_win_prob(models, cont, opt_feat, slots, temperature)
    target = y["win"]
    eps = 1e-6
    pc = p.clamp(eps, 1 - eps)
    log_loss = -(target * torch.log(pc) + (1 - target) * torch.log(1 - pc)).mean().item()
    brier = ((p - target) ** 2).mean().item()
    # ECE: 10-bin
    n_bins = 10
    ece = 0.0
    n = len(examples)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if mask.sum().item() == 0:
            continue
        conf = p[mask].mean().item()
        acc = target[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(conf - acc)
    return {"n": n, "log_loss": log_loss, "brier": brier, "ece": ece}


# ---------------------------------------------------------------------------
# 6. export(design.md §10.4)
# ---------------------------------------------------------------------------

def export_model_entry(model: nn.Module, calibration: dict, standardization: dict) -> dict:
    sd = model.state_dict()
    shared_layers = []
    for i in range(0, len([k for k in sd if k.startswith("shared") and "weight" in k])):
        w = sd[f"shared.{i * 2}.weight"].tolist()
        b = sd[f"shared.{i * 2}.bias"].tolist()
        shared_layers.append({"weight": w, "bias": b})
    heads = {}
    for head in HEADS:
        heads[head] = {
            "weight": sd[f"heads.{head}.weight"].tolist(),
            "bias": sd[f"heads.{head}.bias"].tolist(),
        }
    embedding_table = sd["embedding.weight"].tolist()
    return {
        "card_embedding": {
            "dim": model.embedding.embedding_dim,
            "card_id_max": model.embedding.num_embeddings - 1,
            "table": embedding_table,
        },
        "standardization": standardization,
        "shared_layers": shared_layers,
        "heads": heads,
        "calibration": dict(calibration),
    }


def build_export_payload(models, calibration: dict, standardization, contract, meta) -> dict:
    return {
        "schema_version": 1,
        "model_type": "ogerpon_option_q_mlp_ensemble",
        "feature_contract": contract,
        "standardization": standardization,
        "models": [export_model_entry(m, calibration, standardization) for m in models],
        "calibration": dict(calibration),
        "decision_thresholds": {
            "strict_override_threshold": 0.02, "near_tie_enabled": False,
            "near_tie_epsilon": 0.015, "loop_threshold": 0.45, "uncertainty_coef": 1.0,
        },
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# 7. parity検証(design.md §10.4: PyTorchと純Python推論の最大絶対誤差 1e-5以下、全head)
#
# production用exportではこの検証を必須にする(外部レビュー指摘)。呼び出し側(main())は
# 全headの最大絶対誤差が PARITY_MAX_ABS_DIFF を超えたら非0終了し、weightsを配置しない。
# ---------------------------------------------------------------------------

def check_parity(payload: dict, models, std, examples: list[dict], device,
                 n_samples=200, seed=0) -> dict:
    """exportしたJSON(pure Python推論)とPyTorch本体の出力差を、全headについて比較する。

    固定seedのランダムサンプル(``examples`` が ``n_samples`` 未満なら全件)を使う。

    PyTorch側はfloat64(double)へ変換してから評価する。純Python側は素の``float``
    (=倍精度)で計算するため、既定のfloat32のままだと勝敗headでは丸め誤差がsigmoidに
    吸収されて見えなくても、無制約の回帰head(opponent_ko_count/signed_terminal_turns)
    では0.01〜0.1程度の差として顕在化する(実測で確認済み。ロジックの誤りではなく
    単純な精度不一致だった)。float64同士で比較して初めて「学習コードと推論コードが
    同じ計算をしているか」を検証できる。

    Returns:
        dict: ``{head: max_abs_diff, ...}`` に ``"n_samples"`` を加えたもの。
        exportしたJSONが読み込めなかった場合は全head ``inf``。
    """
    import copy
    import tempfile

    from ptcg_ai.learning.ogerpon_strategy_model import OgerponStrategyModel

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh)
        tmp_path = Path(fh.name)
    try:
        py_model = OgerponStrategyModel(tmp_path)
        if not py_model.is_ready:
            return {**{h: float("inf") for h in HEADS}, "n_samples": 0,
                   "error": "exported JSON did not load (is_ready=False)"}

        double_models = [copy.deepcopy(m).double().to(device) for m in models]
        pool = list(examples)
        sample = random.Random(seed).sample(pool, min(n_samples, len(pool))) if pool else []
        max_diffs = {h: 0.0 for h in HEADS}
        cont_mean = torch.tensor(std["continuous_mean"], dtype=torch.float64, device=device)
        cont_std = torch.tensor(std["continuous_std"], dtype=torch.float64, device=device)
        opt_mean = torch.tensor(std["option_mean"], dtype=torch.float64, device=device)
        opt_std = torch.tensor(std["option_std"], dtype=torch.float64, device=device)
        win_temp = payload["calibration"]["win_temperature"]
        loop_temp = payload["calibration"]["loop_complete_temperature"]

        for ex in sample:
            cont = torch.tensor([ex["continuous_features"]], dtype=torch.float64, device=device)
            opt_t = torch.tensor([ex["option_features"]], dtype=torch.float64, device=device)
            slots = torch.tensor([ex["slot_card_ids"]], dtype=torch.long, device=device)
            cont_n = (cont - cont_mean) / cont_std
            opt_n = (opt_t - opt_mean) / opt_std

            with torch.no_grad():
                per_model = [m(cont_n, opt_n, slots) for m in double_models]
            win_t = torch.stack([torch.sigmoid(p["win"][0] / win_temp) for p in per_model]).mean().item()
            loop_t = torch.stack(
                [torch.sigmoid(p["loop_complete"][0] / loop_temp) for p in per_model]).mean().item()
            ko_t = torch.stack([p["opponent_ko_count"][0] for p in per_model]).mean().item()
            turns_t = torch.stack([p["signed_terminal_turns"][0] for p in per_model]).mean().item()
            if ex["option_name"] != SINGLE_PRIZE_ROTATION:
                loop_t = 0.5  # ogerpon_strategy_model側もEX_TEMPOはloop_completeを0.5固定

            py_pred = py_model.predict(ex["continuous_features"], ex["slot_card_ids"],
                                       ex["option_features"], ex["option_name"])
            max_diffs["win"] = max(max_diffs["win"], abs(win_t - py_pred["win"]))
            max_diffs["loop_complete"] = max(max_diffs["loop_complete"], abs(loop_t - py_pred["loop_complete"]))
            max_diffs["opponent_ko_count"] = max(
                max_diffs["opponent_ko_count"], abs(ko_t - py_pred["opponent_ko_count"]))
            max_diffs["signed_terminal_turns"] = max(
                max_diffs["signed_terminal_turns"], abs(turns_t - py_pred["signed_terminal_turns"]))
        max_diffs["n_samples"] = len(sample)
        return max_diffs
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="collect_ogerpon_counterfactuals.py の出力JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ensemble-seeds", default=",".join(str(s) for s in DEFAULTS["ensemble_seeds"]))
    ap.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-error-rate", type=float, default=DEFAULTS["max_error_rate"])
    ap.add_argument("--max-incomplete-pair-rate", type=float, default=DEFAULTS["max_incomplete_pair_rate"])
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = dict(DEFAULTS)
    cfg["max_epochs"] = args.max_epochs
    cfg["batch_size"] = args.batch_size
    cfg["learning_rate"] = args.learning_rate
    cfg["max_error_rate"] = args.max_error_rate
    cfg["max_incomplete_pair_rate"] = args.max_incomplete_pair_rate
    seeds = [int(s) for s in args.ensemble_seeds.split(",") if s.strip()]

    data_path = Path(args.data)
    records = load_records(data_path)
    quality_report = validate_dataset(records, cfg)
    print(f"data quality: {quality_report}", flush=True)

    by_state = aggregate_by_state_option(records)
    train_ids, val_ids, test_ids = split_state_ids(by_state, args.split_seed)

    def _flatten(ids):
        out = []
        for sid in ids:
            out.extend(by_state[sid].values())
        return out

    train_examples = _flatten(train_ids)
    val_examples = _flatten(val_ids)
    test_examples = _flatten(test_ids)

    def _dist(examples):
        d = defaultdict(int)
        for e in examples:
            d[(e["opponent_archetype"], e["learner_index"], e["trigger_kind"])] += 1
        return dict(d)

    print(f"states: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
         f"| examples: train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}",
         flush=True)
    print(f"train distribution (archetype, side, trigger): {_dist(train_examples)}", flush=True)
    if not train_examples:
        raise SystemExit("学習データが0件。--data の内容を確認してください。")

    std = compute_standardization(train_examples)
    from ptcg_ai.learning import ogerpon_strategy_encoder as enc
    contract = {
        "continuous_feature_count": len(std["continuous_mean"]),
        "option_feature_count": len(std["option_mean"]),
        "slot_count": enc.SLOT_COUNT, "embedding_dim": cfg["embedding_dim"], "bulu_card_ids": [920],
        "encoder_version": enc.ENCODER_VERSION,
        "continuous_feature_names": list(enc.CONTINUOUS_FEATURE_NAMES),
        "option_feature_names": list(enc.OPTION_FEATURE_NAMES),
        "slot_names": list(enc.SLOT_NAMES),
    }
    card_id_max = card_id_max_from_engine()
    contract["card_id_max"] = card_id_max

    models, val_losses = [], []
    for seed in seeds:
        model, val_loss = train_one_model(train_examples, val_examples, std,
                                          {"slot_count": enc.SLOT_COUNT, "card_id_max": card_id_max},
                                          cfg, seed, device)
        models.append(model)
        val_losses.append(val_loss)
        print(f"[seed={seed}] best_val_bce={val_loss:.4f}", flush=True)

    win_temp, win_status = calibrate_head_temperature(
        models, val_examples, std, device, "win", cfg["min_calibration_samples"])
    loop_temp, loop_status = calibrate_head_temperature(
        models, val_examples, std, device, "loop_complete", cfg["min_calibration_samples"])
    calibration = {"win_temperature": win_temp, "loop_complete_temperature": loop_temp}
    print(f"calibration: win={win_temp} ({win_status}), loop_complete={loop_temp} ({loop_status})",
         flush=True)

    meta = {
        "git_commit": git_commit_sha(),
        "dataset_hash": sha256_file(data_path),
        "n_states": len(by_state),
        "n_examples": {"train": len(train_examples), "val": len(val_examples), "test": len(test_examples)},
        "data_quality": quality_report,
        "ensemble_val_bce": val_losses,
        "calibration_status": {"win": win_status, "loop_complete": loop_status},
        "test_metrics": evaluate_metrics(models, test_examples or val_examples, std, device, win_temp),
    }
    payload = build_export_payload(models, calibration, std, contract, meta)

    # design.md §10.4 + 外部レビュー指摘: parity checkは必須。全headの最大絶対誤差が
    # 閾値を超えたら非0終了し、weightsを一切配置しない(既存のoutputは上書きしない)。
    parity_pool = test_examples or val_examples or train_examples
    parity = check_parity(payload, models, std, parity_pool, device)
    meta["parity_max_abs_diff"] = parity
    payload["meta"] = meta
    print(f"parity (max abs diff per head, n={parity.get('n_samples', 0)}): "
         f"{ {h: parity[h] for h in HEADS} }", flush=True)

    worst = max(parity[h] for h in HEADS)
    if worst > PARITY_MAX_ABS_DIFF:
        raise SystemExit(
            f"FAIL: parity exceeds {PARITY_MAX_ABS_DIFF:.0e} (worst={worst:.2e}). "
            f"weights NOT written to {args.output}.")

    atomic_write_json(Path(args.output), payload)
    print(f"parity OK (worst={worst:.2e} <= {PARITY_MAX_ABS_DIFF:.0e})", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
