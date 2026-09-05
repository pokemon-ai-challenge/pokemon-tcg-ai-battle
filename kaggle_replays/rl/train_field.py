"""field-sampling RL: alakazam を meta-share サンプルの相手 mix に対して学習(overfit回避)。

単一相手RL(crustle)は overfit(狙ったマッチ+6ptも他が下がり field net ゼロ)。本スクリプトは毎ゲーム
相手を meta-share でサンプルし、対フィールド加重勝率を直接最適化する。eval は sampled-field の greedy 勝率
(=share加重)。best-ckpt をエクスポート。最終判定は eval_field.py(overlay込み)で別途行う。

train_v3 のヘルパー(Critic/GAE/PPO/padding)を再利用。収集のみ collect_field。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_field import parallel_collect_field  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402
from train_v3 import Critic, wilson_lo, build_padded, compute_gae, policy_logp_entropy, export_temp  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# round_robin/eval_field と同じ7アーキ(相手は各 imitation policy)。share は2026-07取得の実測デッキ数。
FIELD = [
    ("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
    ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
    ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181),
]

# gen2(2026-08取得, 4804デッキ)の実測フィールド。7月との主な違い:
#   - ミラー(alakazam 18.6%)が入る。実ラダーで最大の対面なのに従来のFIELDには居なかった
#   - オーガポン/メガユキメノコ/おまつりおんどの3アーキが新規に入る
#   - ブリジュラスが 1078 -> 287、メガルカリオが 1257 -> 403 に縮小
# 相手重みは policy_weights_<arch>_g2.json、デッキは archetype_decks_g2/<arch>/01.csv を使う。
FIELD_G2 = [
    ("marnie_grimmsnarl_ex", 1082), ("alakazam", 891), ("mega_lucario_ex", 403),
    ("dragapult_ex", 381), ("mega_froslass_ex", 353), ("ogerpon_teal_ex", 310),
    ("archaludon_ex", 287), ("crustle", 287), ("shirona_garchomp_ex", 158),
    ("omatsuri_ondo", 138), ("rocket_mewtwo_ex", 112),
]

# --field-preset の定義: (FIELD定義, 相手重みの接尾辞, デッキディレクトリ名)
FIELD_PRESETS = {
    "july7": (FIELD, "", "archetype_decks"),
    "g2": (FIELD_G2, "_g2", "archetype_decks_g2"),
}


def dense_step_rewards(trajs, gamma, coef):
    """各記録ステップの PBRS shaped 報酬列(build_padded と同じ step 順で flatten)。

    r'_t = 終局項(勝1/負0, 最終ステップのみ) + coef·(γ·Φ_{t+1} − Φ_t)。
    Φ_{t+1} は次の記録ステップのポテンシャル、最終ステップは終局(Φ=0)。
    PBRS(Ng+1999)ゆえ coef は Φ のスケール=最適方策を変えない。
    """
    rews = []
    for tr in trajs:
        steps = tr["steps"]
        L = len(steps)
        R = tr["reward"]
        for t in range(L):
            phi_t = steps[t].get("phi", 0.0)
            phi_next = steps[t + 1].get("phi", 0.0) if t < L - 1 else 0.0
            r_term = R if t == L - 1 else 0.0
            rews.append(r_term + coef * (gamma * phi_next - phi_t))
    return rews


def compute_gae_dense(lengths, per_step_rewards, values, gamma, lam, device):
    """per-step 報酬版 GAE(compute_gae は終局のみ報酬。こちらは各ステップに報酬がある)。"""
    adv = torch.zeros(len(values)); vt = torch.zeros(len(values))
    v = values.detach().cpu(); off = 0
    for L in lengths:
        last = 0.0
        for t in reversed(range(L)):
            idx = off + t
            r_t = per_step_rewards[idx]
            v_next = 0.0 if t == L - 1 else float(v[idx + 1])
            delta = r_t + gamma * v_next - float(v[idx])
            last = delta + gamma * lam * last
            adv[idx] = last; vt[idx] = last + float(v[idx])
        off += L
    return adv.to(device), vt.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-arch", default="alakazam")
    ap.add_argument("--learner-weights", default="policy_weights.json")
    ap.add_argument("--learner-deck-path", default=None,
                    help="学習側デッキの任意パス(既定=DECKDIR/<arch>/01.csv)。提出デッキ(xerosic等)で学習する用。")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--eval-games", type=int, default=400)
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
    ap.add_argument("--dense-reward", action="store_true",
                    help="PBRS 密報酬(Φ=サイド進捗差)を有効化。既定OFF(=従来の終局のみ報酬)。")
    ap.add_argument("--shaping-coef", type=float, default=1.0,
                    help="PBRS ポテンシャルのスケール係数(既定1.0)。PBRSゆえ最適方策は不変。")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--field-preset", default="july7", choices=sorted(FIELD_PRESETS),
                    help="相手フィールドの世代。july7=従来(2026-07 BC 7アーキ、既定で挙動不変)、"
                         "g2=2026-08 BC 11アーキ(ミラー・オーガポン等を含む実メタ)")
    ap.add_argument("--field-weights", default=None,
                    help="FIELD の相手 share をカンマ区切りで上書きする(順序は FIELD 定義順)。"
                         "既定 None = 従来の share。実ラダー分布で学習するための口。")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Nイテレーションごとに policy を別ファイルへ保存する(0=保存しない)。"
                         "学習中の eval(400戦=CI±5pt)で best を選ぶとノイズ選択になるため、"
                         "後段の高精度 screen(n=9600, CI±1.4pt)で選び直せるようにする。")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="最初のNイテレーションは **criticだけ** 更新し policy を凍結する。"
                         "0=従来と完全同一。critic が未収束のうちの policy 更新は advantage が"
                         "壊れており、best-checkpoint 選択もその窓に引きずられるため。")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="収集シードのオフセット。0=従来と完全同一。独立seed armを作るためだけの追加。")
    ap.add_argument("--tag", default="field")
    args = ap.parse_args()

    device = args.device
    arch = args.learner_arch
    lw = Path(args.learner_weights)
    learner_w = lw if lw.is_absolute() else WDIR / lw.name
    out = WDIR / f"policy_weights_{arch}_rl_{args.tag}.json"
    tmp = _HERE / f"_tmp_policy_{arch}_{args.tag}.json"
    logpath = _HERE / f"_train_{arch}_{args.tag}.log"

    field, opp_suffix, deckdir_name = FIELD_PRESETS[args.field_preset]
    deckdir = DECKDIR.parent / deckdir_name

    learner_deck = read_deck_csv_file(args.learner_deck_path if args.learner_deck_path
                                      else str(deckdir / arch / "01.csv"))
    opp_specs = [(str(WDIR / f"policy_weights_{a}{opp_suffix}.json"),
                  read_deck_csv_file(str(deckdir / a / "01.csv"))) for a, _ in field]
    missing = [p for p, _ in opp_specs if not Path(p).exists()]
    if missing:
        raise SystemExit("相手重みが無い: " + ", ".join(Path(m).name for m in missing))
    shares = [s for _, s in field]
    if args.field_weights:
        w = [float(x) for x in args.field_weights.split(",") if x != ""]
        if len(w) != len(field):
            raise SystemExit("field-weights の数が FIELD(%d)と一致しない" % len(field))
        shares = w
        print("field shares overridden ->", dict(zip([a for a, _ in field], shares)), flush=True)
    print(f"device={device} learner={arch} vs FIELD[{args.field_preset}]({len(field)}arch, "
          f"share-sampled, decks={deckdir_name}) workers={args.workers} -> {out.name}", flush=True)

    base_payload = json.loads(learner_w.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    def do_eval(seed, ngames):
        export_temp(policy, base_payload, tmp)
        _, w, v, _, _ = parallel_collect_field(str(tmp), opp_specs, shares, learner_deck,
                                               ngames, seed, temperature=0.01, workers=args.workers)
        return (w / v if v else float("nan")), w, v

    history = []
    wr0, w0, v0 = do_eval(900000, args.eval_games)
    print(f"[iter 0] baseline field-weighted greedy {w0}/{v0} = {wr0:.3f} (CI_lo {wilson_lo(w0,v0):.3f})", flush=True)
    history.append({"iter": 0, "eval_winrate": wr0, "eval_wins": w0, "eval_valid": v0})
    best_wr, best_iter = wr0, 0
    best_state = copy.deepcopy(policy.state_dict())

    t0 = time.time()
    for it in range(1, args.iters + 1):
        export_temp(policy, base_payload, tmp)
        trajs, wins, valid, errors, per_opp = parallel_collect_field(
            str(tmp), opp_specs, shares, learner_deck, args.games_per_iter,
            seed0=it * 100000, temperature=args.temperature, workers=args.workers)
        if not trajs:
            print(f"[iter {it}] no trajs", flush=True); continue
        batch = build_padded(trajs, device)
        with torch.no_grad():
            values = critic(batch["state_rows"])
        if args.dense_reward:
            step_rews = dense_step_rewards(trajs, args.gamma, args.shaping_coef)
            adv, vtarget = compute_gae_dense(batch["lengths"], step_rews, values, args.gamma, args.lam, device)
        else:
            adv, vtarget = compute_gae(batch["lengths"], batch["rewards"], values, args.gamma, args.lam, device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        warm = it <= args.critic_warmup
        pl = vl = en = 0.0
        for _ in range(args.epochs):
            if warm:
                # policy 凍結。critic だけを回す(advantage を作れるようにする)。
                v_pred = critic(batch["state_rows"])
                val_loss = ((v_pred - vtarget) ** 2).mean()
                opt_v.zero_grad(); val_loss.backward(); opt_v.step()
                vl = val_loss.item()
                continue
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
        if warm:
            print(f"[iter {it}] CRITIC-WARMUP train_wr {train_wr:.3f} "
                  f"({wins}/{valid}) val {vl:.4f} {time.time()-t0:.0f}s", flush=True)
            history.append({"iter": it, "critic_warmup": True,
                            "train_winrate": train_wr, "val_loss": vl})
            logpath.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                               encoding="utf-8")
            continue
        print(f"[iter {it}] train_wr {train_wr:.3f} ({wins}/{valid} err{errors}) steps {batch['n']} "
              f"pol {pl:.4f} val {vl:.4f} ent {en:.3f} {time.time()-t0:.0f}s", flush=True)
        rec = {"iter": it, "train_winrate": train_wr, "steps": batch["n"], "pol_loss": pl, "val_loss": vl, "entropy": en}
        if it % args.eval_every == 0 or it == args.iters:
            wr, w, v = do_eval(900000, args.eval_games)
            rec.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
            star = ""
            if wr > best_wr:
                best_wr, best_iter = wr, it
                best_state = copy.deepcopy(policy.state_dict()); star = " *BEST*"
            print(f"    [eval] field-weighted {w}/{v} = {wr:.3f} (CI_lo {wilson_lo(w,v):.3f}) best={best_wr:.3f}@{best_iter}{star}", flush=True)
        if args.save_every and it % args.save_every == 0:
            ck = WDIR / f"policy_weights_{arch}_rl_{args.tag}_it{it}.json"
            ck.write_text(json.dumps(policy.to_json_payload(copy.deepcopy(base_payload)),
                                     ensure_ascii=False), encoding="utf-8")
            rec["checkpoint"] = ck.name
        history.append(rec)
        logpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    policy.load_state_dict(best_state)
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({"rl_finetuned": True, "rl_best_iter": best_iter,
                                           "rl_best_eval_winrate": best_wr, "rl_baseline": wr0})
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported BEST (iter {best_iter}, field-weighted {best_wr:.3f}, baseline {wr0:.3f}) -> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
