"""RL 対戦相手強化 v3: 並列収集(collect_parallel)+ GAE + 最終イテレート採用。

v2 との差: 収集をワーカー並列化(torch非依存ワーカー)。CPU律速の収集が ~workers 倍速くなるので
games/iter を大きく(既定512)して勾配分散を下げつつ、全体を高速化。
方策更新(PPO)は main の torch。各iter: 現policyをtemp JSONへ -> 並列収集 -> PPO更新。

すべて `kaggle_replays/rl/` 内・torch は main のみ・production 無変更。

2026-08-13 修正(roadmap Phase E / requirements-kamitsuorochi-2026-08-12.md §6 step1):
  - **checkpoint選択**: best-of-eval(winner's curse)をやめ、**最終イテレート**を常に採用する
    (実測: marnie best 0.800/n=200 → 最終 0.757/n=1200、-4.3pt。詳細は requirements doc §4-2 末尾)。
  - Critic/GAE/policy-logp まわりのユーティリティ(このファイル)は train_pool.py からも import
    される共有プリミティブ。ここでの変更は train_pool.py 経由の実行にも及ぶ。
    ``distributed/learner.py`` も ``Critic`` / ``compute_gae`` / ``wilson_lo`` をこのファイルから
    import しているため、**既存呼び出しの位置引数だけの呼び方(引数追加なし)は必ず動き続ける
    ように新引数はすべてキーワード・既定値付きで追加すること**(後方互換)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_parallel import parallel_collect  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
VALUE_NET_DIR = _ROOT / "kaggle_replays" / "value_net"

# defect#5 (報酬シェーピング) で使う特徴インデックス。ハードコードせず encoder.FEATURE_NAMES
# から引く(train_pool.py の _SELF_DECK_COUNT_IDX と同じ流儀)。
SELF_PRIZE_REMAINING_IDX = encoder.FEATURE_NAMES.index("self_prize_remaining")
OPP_PRIZE_REMAINING_IDX = encoder.FEATURE_NAMES.index("opp_prize_remaining")


class Critic(nn.Module):
    """状態価値関数。既定(hidden=64, output_activation=None)は元の実装とビット単位で同じ形
    (state_dim -> 64 -> 64 -> 1、線形出力、``net.0/net.2/net.4`` の3層)なので、
    ``distributed/learner.py`` が保存済み ``state/trainer_v<N>.pt`` を ``load_state_dict`` する
    経路は無変更で動く。

    defect#3(Critic のランダム初期化)向けに、hidden をタプルで・output_activation を
    ``"sigmoid"`` で指定できるように拡張した。これは ``Critic.from_value_net`` が
    ``kaggle_replays/value_net/value_weights_v251.json`` の学習済み勝率推定器
    (hidden=[64,16]、出力 sigmoid)をそのまま移植するために使う(下記参照)。
    引数を省略したときの挙動は完全に元のまま。
    """

    def __init__(self, state_dim, mean, std, hidden=64, output_activation=None):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        hs = tuple(hidden) if isinstance(hidden, (list, tuple)) else (hidden, hidden)
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hs:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation is not None:
            raise ValueError(f"未対応の output_activation: {output_activation!r}")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        safe = torch.where(self.std != 0, self.std, torch.ones_like(self.std))
        z = torch.where(self.std != 0, (x - self.mean) / safe, torch.zeros_like(x))
        return self.net(z).squeeze(-1)

    # ------------------------------------------------------------------ defect#3
    @classmethod
    def from_value_net(cls, state_dim, state_mean, state_std, feature_names,
                       value_net_payload, device="cpu"):
        """``value_net/train.py`` が出力した勝率推定 MLP(sigmoid出力)から Critic を warm-start する。

        value_net の入力は「学習側方策の ablate 済み(常時0埋め)特徴を除いた生き残り特徴」の
        部分集合(``value_net_payload["feature_names"]``、715次元中251次元)。列を
        ``feature_names``(呼び出し側の715次元の並び)の対応する位置へ差し込み、ablate 済みの
        列は 0 で埋める(その列は Critic.forward で std=0 -> z=0 になるので、重みが何であれ
        出力に寄与しない。0にするのはそれを明示するだけ)。

        隠れ層サイズ・出力活性化は value_net_payload の layers からそのまま引き継ぐ
        (現行は hidden=[64,16]、出力 sigmoid)ので、必ず ``Critic(..., hidden=(...),
        output_activation="sigmoid")`` と同じ形になる。

        戻り値: (critic, coverage) — coverage は feature_names のうち value_net 側に
        存在した列の割合(1.0 なら完全被覆。呼び出し側で警告に使える)。
        """
        layers = value_net_payload["layers"]
        vnames = value_net_payload["feature_names"]
        vstd = value_net_payload["standardization"]
        # value_net/train.py の schema は各層 {"W": [out][in], "b": [out], "activation": ...}
        # (policy_weights.json の "weight"/"bias" とはキー名が違う。別モデル・別 train.py の出力なので
        # schema が独立している。ここで吸収する)。
        hidden_sizes = [len(layer["b"]) for layer in layers[:-1]]
        out_act = "sigmoid" if layers[-1].get("activation") == "sigmoid" else None

        name_to_vidx = {n: i for i, n in enumerate(vnames)}
        covered = [name_to_vidx.get(n) for n in feature_names]
        coverage = sum(1 for c in covered if c is not None) / max(1, len(feature_names))

        critic = cls(state_dim, state_mean, state_std, hidden=hidden_sizes, output_activation=out_act)

        with torch.no_grad():
            # --- layer0: 列を 715(呼び出し側次元)へ写像。value_net 側は独自標準化(mean/std)を
            # 使っているので、ここで value_net の標準化を Critic 側の標準化に合成する:
            #   value_net は z_v = (x_v - mean_v)/std_v を食わせて学習している。
            #   Critic は既に (x - mean)/std を計算してから net に渡す設計なので、
            #   layer0 の重みは「Criticの標準化後の値」に対して定義し直す必要がある:
            #     z_v = (x - mean_v)/std_v = ((x-mean)/std) * (std/std_v) + (mean-mean_v)/std_v
            #   線形層なので W0行 * (std/std_v) をスケールし、バイアスに W0行・(mean-mean_v)/std_v を足す。
            first = layers[0]
            w0 = torch.tensor([[float(w) for w in row] for row in first["W"]])  # (h0, 251)
            b0 = torch.tensor([float(v) for v in first["b"]])  # (h0,)
            h0 = w0.shape[0]
            new_w0 = torch.zeros(h0, state_dim)
            extra_bias = torch.zeros(h0)
            for j, name in enumerate(feature_names):
                vidx = name_to_vidx.get(name)
                if vidx is None:
                    continue
                std_c = float(state_std[j]) if float(state_std[j]) != 0.0 else 1.0
                std_v = float(vstd["std"][vidx]) if float(vstd["std"][vidx]) != 0.0 else 1.0
                mean_c = float(state_mean[j])
                mean_v = float(vstd["mean"][vidx])
                scale = std_c / std_v
                new_w0[:, j] = w0[:, vidx] * scale
                extra_bias += w0[:, vidx] * ((mean_c - mean_v) / std_v)
            critic.net[0].weight.copy_(new_w0)
            critic.net[0].bias.copy_(b0 + extra_bias)

            # --- 残りの層はそのままコピー(次元は完全一致するので remap 不要)。
            # net の Linear 層は 0, 2, 4, ... の位置(間に ReLU/Sigmoid が挟まる)。
            linear_positions = [i for i, m in enumerate(critic.net) if isinstance(m, nn.Linear)]
            for li, layer in enumerate(layers[1:], start=1):
                w = torch.tensor([[float(v) for v in row] for row in layer["W"]])
                b = torch.tensor([float(v) for v in layer["b"]])
                pos = linear_positions[li]
                critic.net[pos].weight.copy_(w)
                critic.net[pos].bias.copy_(b)

        return critic.to(device), coverage


def elo_diff(wins: float, n: float) -> float:
    """勝率 -> Elo 差(``173.7 * logit(p)``)。roadmap-2026-08-05.md §6 の指標定義と同じ式。

    p=0/1(w=0 or w=n)は継続性補正(``eps = 0.5/n``)で厳密な 0 か 1 を避ける。素の
    logit(0)/logit(1) は ±inf になり、以降のソート・平均・CI計算がすべて壊れる
    (小標本の勝率0/n・n/nはRLの初期評価で普通に起こる)ため、これを既定の安全策にする。
    n<=0 は NaN を返す(未定義)。
    """
    if n <= 0:
        return float("nan")
    p = wins / n
    eps = 0.5 / n
    p = min(max(p, eps), 1.0 - eps)
    return 173.7 * math.log(p / (1.0 - p))


def wilson_lo(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (c - m) / d)


def build_padded(trajs, device):
    steps = [s for tr in trajs for s in tr["steps"]]
    n = len(steps)
    max_n = max(len(s["option_feats"]) for s in steps)
    sd = len(steps[0]["state_feat"])
    od = len(steps[0]["option_feats"][0])
    state_rows = torch.zeros(n, sd)
    option_pad = torch.zeros(n, max_n, od)
    card_pad = torch.zeros(n, max_n, dtype=torch.long)
    mask = torch.zeros(n, max_n)
    chosen = torch.zeros(n, dtype=torch.long)
    old_logp = torch.zeros(n)
    for i, s in enumerate(steps):
        state_rows[i] = torch.tensor(s["state_feat"])
        k = len(s["option_feats"])
        option_pad[i, :k] = torch.tensor(s["option_feats"])
        card_pad[i, :k] = torch.tensor(s["card_ids"])
        mask[i, :k] = 1.0
        chosen[i] = s["chosen_idx"]
        old_logp[i] = s["logprob"]
    return {"state_rows": state_rows.to(device), "option_pad": option_pad.to(device),
            "card_pad": card_pad.to(device), "mask": mask.to(device),
            "chosen": chosen.to(device), "old_logp": old_logp.to(device),
            "n": n, "max_n": max_n,
            "lengths": [len(tr["steps"]) for tr in trajs],
            "rewards": [tr["reward"] for tr in trajs]}


def prize_margin_potential(state_feat_row, c: float = 0.1):
    """defect#5: Φ(s) = c * (opp_prizes_remaining - self_prizes_remaining)。

    ``state_feat_row`` は 1決定点ぶんの state 特徴(len == len(encoder.FEATURE_NAMES))。
    生の ``list[float]`` / 1次元 ``torch.Tensor`` のどちらでも動く(スカラー同士の演算のみ)。

    **この関数は非終端の決定点状態にのみ呼ぶこと。終端(ゲーム終了後の状態)には呼ばない。**
    終端の潜在能力は常に定数 0.0 として扱う(compute_gae 側でハードコードしており、この関数を
    終端に対して評価することはない)。これにより Φ(terminal)≡0 は「たまたま特徴量がそう出る」
    のではなく、コード上の構造で保証される(終端 state_feat を作る/参照する経路が存在しない)。
    """
    return c * (float(state_feat_row[OPP_PRIZE_REMAINING_IDX]) - float(state_feat_row[SELF_PRIZE_REMAINING_IDX]))


def compute_gae(lengths, rewards, values, gamma, lam, device, state_rows=None, shaping_c=0.0):
    """GAE(λ)。``shaping_c`` を省略(既定0.0)すると元の実装とビット単位で同じ計算になる
    (defect#5 の報酬シェーピングは opt-in。distributed/learner.py は shaping_c を渡さずに
    呼んでいるので影響を受けない)。

    shaping_c != 0.0 のとき、各非終端遷移の報酬を potential-based shaping で置き換える:
        r_t = 0 + gamma*Φ(s_{t+1}) - Φ(s_t)                      (t < L-1)
        r_t = R + gamma*Φ(terminal) - Φ(s_{L-1}) = R - Φ(s_{L-1})  (t == L-1, Φ(terminal)≡0)
    Φ は ``prize_margin_potential`` で ``state_rows``(build_padded の state_rows、gamma スケール
    前の生特徴)から計算する。terminal の Φ は上記のとおり**関数を呼ばずに定数 0.0**として扱う
    (prize_margin_potential を終端状態に対して評価する経路自体を作らない)。
    """
    adv = torch.zeros(len(values)); vt = torch.zeros(len(values))
    v = values.detach().cpu(); off = 0
    phi = None
    if shaping_c:
        if state_rows is None:
            raise ValueError("shaping_c != 0 には state_rows が必要")
        sr = state_rows.detach().cpu()
        phi = shaping_c * (sr[:, OPP_PRIZE_REMAINING_IDX] - sr[:, SELF_PRIZE_REMAINING_IDX])
    for L, R in zip(lengths, rewards):
        last = 0.0
        for t in reversed(range(L)):
            idx = off + t
            if phi is not None:
                phi_t = float(phi[idx])
                if t == L - 1:
                    phi_next = 0.0  # Φ(terminal) ≡ 0 (定数。phi[] を読まない)
                    r_t = R + gamma * phi_next - phi_t
                else:
                    phi_next = float(phi[idx + 1])
                    r_t = 0.0 + gamma * phi_next - phi_t
            else:
                r_t = R if t == L - 1 else 0.0
            v_next = 0.0 if t == L - 1 else float(v[idx + 1])
            delta = r_t + gamma * v_next - float(v[idx])
            last = delta + gamma * lam * last
            adv[idx] = last; vt[idx] = last + float(v[idx])
        off += L
    return adv.to(device), vt.to(device)


def option_full_logp(policy, batch):
    """(n, max_n) の log π(option|state)。マスクされた選択肢は -inf 近傍。

    policy_logp_entropy / kl_to_bc の共通部分(defect#6 の KL(π‖π_BC) 計算に使うため、
    「選んだ行動だけ」ではなく全選択肢ぶんの logp が必要になって分離した)。数式は
    元の policy_logp_entropy と完全に同じ(リファクタのみ、数値は変わらない)。
    """
    n, max_n = batch["n"], batch["max_n"]
    sd = batch["state_rows"].shape[1]; od = batch["option_pad"].shape[2]
    sf = batch["state_rows"].unsqueeze(1).expand(n, max_n, sd).reshape(n * max_n, sd)
    of = batch["option_pad"].reshape(n * max_n, od)
    cf = batch["card_pad"].reshape(n * max_n)
    scores = policy.option_scores_flat(sf, of, cf).reshape(n, max_n)
    scores = torch.where(batch["mask"] > 0, scores, torch.full_like(scores, torch.finfo(scores.dtype).min))
    return torch.log_softmax(scores, dim=1)


def policy_logp_entropy(policy, batch):
    logp = option_full_logp(policy, batch)
    chosen_logp = logp.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)
    ent = -(logp.exp() * logp.masked_fill(batch["mask"] == 0, 0.0)).sum(dim=1)
    return chosen_logp, ent


def kl_per_step_to_bc(policy, bc_policy, batch):
    """defect#6: 決定点ごとの KL(π_policy(·|s) ‖ π_bc(·|s))。(n,) を返す(呼び出し側で .mean())。

    ``bc_policy`` は凍結済み(学習開始時の BC 重みをロードしたまま更新しない)想定。
    現在方策 policy に対して呼ぶときと、bc_policy 自身に対して呼んだとき(KL(bc‖bc)=0)の
    両方が正しく振る舞うことを test_kl_regularization.py で検証している(=「実は
    bc_policy ではなく policy 自身と比較している」を防ぐ回帰テスト)。
    """
    logp = option_full_logp(policy, batch)
    with torch.no_grad():
        logq = option_full_logp(bc_policy, batch)
    p = logp.exp()
    mask = batch["mask"] > 0
    terms = torch.where(mask, p * (logp - logq), torch.zeros_like(logp))
    return terms.sum(dim=1)


def ppo_update_step(policy, critic, opt_p, opt_v, batch, adv, vtarget, *,
                    clip, entropy_coef, kl_beta=0.0, bc_policy=None, update_policy=True):
    """PPO(clip)の1内部エポック更新。train_pool.py と train_v3.py の PPO ループ本体を
    1箇所にまとめたもの(defect#4/#6 を両方乗せる先を1つにするため新設)。

    - ``update_policy=False``(defect#4: critic warmup)のときは方策側の forward/backward/
      opt_p.step() を一切呼ばない -> policy のパラメータ・勾配は literally 触らない。
      critic の更新(opt_v.step())は update_policy に関係なく毎回行う。
    - ``kl_beta`` と ``bc_policy`` を両方与えたときだけ KL(π‖π_BC) を方策損失に加える
      (defect#6)。bc_policy 側は no_grad で評価するので KL の勾配は policy 側にしか流れない。

    戻り値: (pol_loss, val_loss, entropy, kl) の float。update_policy=False のときは
    pol_loss/entropy/kl は 0.0(計算していない)。
    """
    pl = ent_val = kl_val = 0.0
    if update_policy:
        new_logp, ent = policy_logp_entropy(policy, batch)
        ratio = torch.exp(new_logp - batch["old_logp"])
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        pol_loss = -torch.min(s1, s2).mean() - entropy_coef * ent.mean()
        if kl_beta and bc_policy is not None:
            kl_mean = kl_per_step_to_bc(policy, bc_policy, batch).mean()
            pol_loss = pol_loss + kl_beta * kl_mean
            kl_val = kl_mean.item()
        opt_p.zero_grad(); pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt_p.step()
        pl, ent_val = pol_loss.item(), ent.mean().item()
    v_pred = critic(batch["state_rows"])
    val_loss = ((v_pred - vtarget) ** 2).mean()
    opt_v.zero_grad(); val_loss.backward(); opt_v.step()
    return pl, val_loss.item(), ent_val, kl_val


def export_temp(policy, base_payload, path):
    payload = policy.to_json_payload(base_payload)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-arch", default="dragapult_ex")
    ap.add_argument("--learner-weights", default=None,
                    help="learner の初期重み(既定 policy_weights_<arch>.json)。alakazam は production の "
                         "policy_weights.json を指定する(policy_weights_alakazam.json は無い)。")
    ap.add_argument("--opponent-arch", default="alakazam")
    ap.add_argument("--opponent-weights", default=None)
    ap.add_argument("--iters", type=int, default=25,
                    help="既定25(旧60/50から変更。requirements-kamitsuorochi-2026-08-12.md §6 step1: "
                         "20-25で飽和という実測に基づく)。")
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-value", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()

    device = args.device
    arch = args.learner_arch
    learner_w = Path(args.learner_weights) if args.learner_weights else WDIR / f"policy_weights_{arch}.json"
    if not learner_w.is_absolute():
        learner_w = WDIR / learner_w.name
    out = WDIR / f"policy_weights_{arch}_rl_{args.tag}.json"
    tmp = _HERE / f"_tmp_policy_{arch}_{args.tag}.json"
    logpath = _HERE / f"_train_{arch}_{args.tag}.log"
    print(f"device={device} learner={arch} opp={args.opponent_arch} workers={args.workers} -> {out.name}", flush=True)

    deck_l = read_deck_csv_file(str(DECKDIR / arch / "01.csv"))
    deck_o = read_deck_csv_file(str(DECKDIR / args.opponent_arch / "01.csv"))

    base_payload = json.loads(learner_w.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    def do_eval(seed, ngames):
        export_temp(policy, base_payload, tmp)
        _, w, v, _ = parallel_collect(str(tmp), args.opponent_weights, deck_l, deck_o,
                                      ngames, seed, temperature=0.01, workers=args.workers)
        return (w / v if v else float("nan")), w, v

    history = []
    wr0, w0, v0 = do_eval(900000, args.eval_games)
    print(f"[iter 0] baseline greedy {w0}/{v0} = {wr0:.3f} (CI_lo {wilson_lo(w0,v0):.3f}) "
          f"Elo {elo_diff(w0, v0):+.1f}", flush=True)
    history.append({"iter": 0, "eval_winrate": wr0, "eval_wins": w0, "eval_valid": v0})
    # defect#1: best-of-eval(winner's curse)は選ばない。ここで追跡する best_wr/best_iter は
    # ログ表示専用の参考値で、どのイテレートを export するかには一切使わない
    # (下の main loop 末尾の export は常に「ループを抜けた時点の policy」= 最終イテレート)。
    best_wr, best_iter = wr0, 0

    t0 = time.time()
    for it in range(1, args.iters + 1):
        export_temp(policy, base_payload, tmp)
        trajs, wins, valid, errors = parallel_collect(
            str(tmp), args.opponent_weights, deck_l, deck_o,
            args.games_per_iter, seed0=it * 100000, temperature=args.temperature, workers=args.workers)
        if not trajs:
            print(f"[iter {it}] no trajs", flush=True); continue
        batch = build_padded(trajs, device)
        with torch.no_grad():
            values = critic(batch["state_rows"])
        adv, vtarget = compute_gae(batch["lengths"], batch["rewards"], values, args.gamma, args.lam, device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl = vl = en = 0.0
        for _ in range(args.epochs):
            new_logp, ent = policy_logp_entropy(policy, batch)
            ratio = torch.exp(new_logp - batch["old_logp"])
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
            pol_loss = -torch.min(s1, s2).mean() - args.entropy * ent.mean()
            opt_p.zero_grad(); pol_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt_p.step()
            v_pred = critic(batch["state_rows"])
            val_loss = ((v_pred - vtarget) ** 2).mean()
            opt_v.zero_grad(); val_loss.backward(); opt_v.step()
            pl, vl, en = pol_loss.item(), val_loss.item(), ent.mean().item()

        train_wr = wins / valid if valid else float("nan")
        print(f"[iter {it}] train_wr {train_wr:.3f} ({wins}/{valid} err{errors}) steps {batch['n']} "
              f"pol {pl:.4f} val {vl:.4f} ent {en:.3f} {time.time()-t0:.0f}s", flush=True)
        rec = {"iter": it, "train_winrate": train_wr, "steps": batch["n"], "pol_loss": pl, "val_loss": vl, "entropy": en}
        if it % args.eval_every == 0 or it == args.iters:
            wr, w, v = do_eval(900000, args.eval_games)
            rec.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
            star = ""
            if wr > best_wr:
                best_wr, best_iter = wr, it
                star = " (best-so-far, 参考値。exportには使わない)"
            print(f"    [eval] {w}/{v} = {wr:.3f} (CI_lo {wilson_lo(w,v):.3f}) "
                  f"Elo {elo_diff(w, v):+.1f} best_seen={best_wr:.3f}@{best_iter}{star}", flush=True)
        history.append(rec)
        logpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    # defect#1: 最終イテレート(ループを抜けた時点の policy)をそのまま export する。
    # best-of-eval を採用しない(理由は requirements-kamitsuorochi-2026-08-12.md §4-2 末尾:
    # 200試合evalのbestは winner's curse で水増しされる。marnie実測で best 0.800/n=200 → 最終
    # 0.757/n=1200、-4.3pt)。
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({
        "rl_finetuned": True, "rl_checkpoint": "final_iterate",
        "rl_final_iter": args.iters, "rl_baseline": wr0,
        "rl_best_seen_eval_winrate_FYI_NOT_EXPORTED": best_wr,
        "rl_best_seen_iter_FYI_NOT_EXPORTED": best_iter,
    })
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported FINAL ITERATE (iter {args.iters}, baseline {wr0:.3f}) -> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
