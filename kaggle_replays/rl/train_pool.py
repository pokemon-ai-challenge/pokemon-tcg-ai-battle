"""RL 対戦相手強化(相互鍛錬 段階2): 学習側1モデル vs 相手プールで PPO する。

train_v3.py は「学習側1モデル vs 相手1モデル固定」でしか PPO できない(collect_parallel.parallel_collect
が固定の1相手しか取れないため)。本スクリプトは収集を collect_pool.parallel_collect_pool に差し替え、
相手プール(複数アーキタイプ)に対して学習できるようにする。

**アルゴリズムは再実装しない。** PPO 更新・GAE・パディング・評価用ユーティリティは
train_v3.py からそのまま import する(Critic, build_padded, compute_gae, policy_logp_entropy,
export_temp, wilson_lo)。本スクリプトが新たに書くのは main() の収集まわり(プール対応)と
CLI・ログ・best-checkpoint 管理のみ。

重要: 固定相手アーム(--train-opponents に1つだけ指定)も、このスクリプトで
「プール要素数1」として走らせる設計にしている。train_v3.py(collect_parallel 経由)と
train_pool.py(collect_pool 経由)とで固定相手アームの実装を分けると、
「相互鍛錬させたら悪化したのは相手プールのせいか、それとも実装差のせいか」が交絡して
切り分けられなくなる。そのため固定相手の比較実験も本スクリプト1本(k=1呼び出し)で行う。

評価用の相手プール(--eval-opponents)は学習中一切更新しない(凍結)。--train-opponents と
--eval-opponents を別引数にしているのは、学習用プールを将来変更しても比較可能な基準を
崩さないため、および「学習に使った相手だけに勝てるようになっただけ」を検出できるようにするため
(通常は両者を同じ値にして使う想定だが、意図的に分離できるようにしてある)。
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

from train_v3 import (  # noqa: E402
    Critic,
    build_padded,
    compute_gae,
    export_temp,
    policy_logp_entropy,
    wilson_lo,
)
from torch_policy import TorchOptionPolicy  # noqa: E402
from collect_pool import parallel_collect_pool  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
import pools  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# eval 系の seed は学習(train)の it*100000 空間([100000, iters*100000])と重ならないよう、
# 十分大きい定数オフセットに分離してある(重なっても正当性は壊れないが、意図を明確にするため)。
EVAL_SEED0 = 800_000_000
EVAL_FIXED_SEED0 = 810_000_000
FINAL_EVAL_SEED0 = 820_000_000
FINAL_EVAL_FIXED_SEED0 = 821_000_000

DEFAULT_POOL = "alakazam,crustle,marnie_grimmsnarl_ex,archaludon_ex"


def _parse_names(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_learner_weights_path(learner: str, override: str | None) -> Path:
    """学習側の初期重みファイルパスを解決する。

    --learner-weights が指定されていればそれを使う(相対パスは WDIR 直下として扱う、
    train_v3.py と同じ挙動)。省略時は pools.LEARNER_REGISTRY の既定を使う。
    alakazam は weights_path=None(production 既定)を意味するので、実ファイルとしては
    WDIR/policy_weights.json を指す(policy_weights_alakazam.json は存在しない)。
    """
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = WDIR / p.name
        return p
    weights_path, _deck_csv = pools.resolve_learner(learner)
    if weights_path is None:
        return WDIR / "policy_weights.json"
    return Path(weights_path)


_SELF_DECK_COUNT_IDX = encoder.FEATURE_NAMES.index("self_deck_count")


def looks_like_deckout(traj, threshold: int) -> bool:
    """学習側が負け、かつ最後の判断時点の自分の山札が threshold 以下ならデッキ切れとみなす。

    これは近似である。``collect_parallel._play_one`` が記録する ``traj["steps"]`` は学習側の
    (maxCount==1 の)判断のみで、最後に記録された判断はその後の相手の手番やカード効果を経た
    「最終盤面」そのものではない。そのため実際の最終 deckCount とはズレうる(近似の妥当性は
    ``test_deckout_detect.py`` で ``blunder_metrics.py`` の厳密判定(最終盤面を直接参照)と
    突き合わせて検証している)。

    self_deck_count のインデックスは ``encoder.FEATURE_NAMES.index("self_deck_count")`` から
    求めており、ハードコードしていない。
    """
    if traj["reward"] >= 1.0:
        return False
    steps = traj.get("steps")
    if not steps:
        return False
    last_deck_count = steps[-1]["state_feat"][_SELF_DECK_COUNT_IDX]
    return last_deck_count <= threshold


def summarize_per_opponent(stats: dict) -> dict:
    """collect_pool の stats['per_opponent'] から相手別勝率つきの軽量な dict を作る。"""
    out = {}
    for name, b in stats["per_opponent"].items():
        wr = (b["wins"] / b["valid"]) if b["valid"] else float("nan")
        out[name] = {
            "games": b["games"], "valid": b["valid"], "wins": b["wins"],
            "errors": b["errors"], "winrate": wr,
        }
    return out


def run_eval_pool(policy, base_payload, tmp_path, opponents, deck_l, n_games, seed0, workers):
    """凍結された相手プールに対して greedy(temperature=0.01)評価する(train_v3.do_eval と同じ温度)。

    戻り値: (winrate, wins, valid, per_opponent_summary)
    """
    export_temp(policy, base_payload, tmp_path)
    trajs, stats = parallel_collect_pool(
        str(tmp_path), opponents, deck_l,
        n_games=n_games, seed0=seed0, temperature=0.01, workers=workers,
    )
    del trajs
    total = stats["total"]
    wr = (total["wins"] / total["valid"]) if total["valid"] else float("nan")
    return wr, total["wins"], total["valid"], summarize_per_opponent(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learner", default="alakazam", help="pools.LEARNER_REGISTRY のキー。既定 alakazam。")
    ap.add_argument("--learner-weights", default=None,
                     help="学習側の初期重み(省略時は --learner のレジストリ既定。alakazam の既定は "
                          "production の policy_weights.json)。")
    ap.add_argument("--train-opponents", default=DEFAULT_POOL,
                     help="カンマ区切りの相手名(pools.LEARNER_REGISTRY のキー)。既定4種: "
                          f"{DEFAULT_POOL}。要素数1を指定すると固定相手アーム(プール要素数1として"
                          "同じコード経路で実行される)。")
    ap.add_argument("--eval-opponents", default=DEFAULT_POOL,
                     help="評価用の相手プール(カンマ区切り)。学習中は一切更新しない凍結プール。"
                          f"既定は --train-opponents と同じ4種: {DEFAULT_POOL}")
    ap.add_argument("--eval-fixed", default=None,
                     help="追加で単独評価したい相手(カンマ区切り、省略可)。指定すると "
                          "--eval-opponents とは別系列でログに記録する。")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games-per-iter", type=int, default=512,
                     help="1iter で収集する試合数。相手の数 x 2(先攻/後攻)の倍数にしないと "
                          "collect_pool.build_tasks の丸めで端数が出る(切り捨てられる)。")
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--final-eval-games", type=int, default=1200)
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
    ap.add_argument("--tag", default="pool")
    ap.add_argument("--force", action="store_true", help="出力重みファイルが既に存在しても上書きする。")
    ap.add_argument("--deckout-threshold", type=int, default=0,
                     help="looks_like_deckout の判定に使う自分の山札枚数の閾値(既定0)。"
                          "デッキ切れ率のログ記録には常に使う。--deckout-penalty が非0のときは "
                          "報酬シェーピングの判定にも使う。"
                          "既定を0にしているのは、test_deckout_paired.py で同一試合内の対応ありで "
                          "検証した結果、閾値0のとき厳密判定と完全一致したため(400試合・220敗中、"
                          "真陽性60/偽陽性0/偽陰性0、適合率1.000・再現率1.000)。"
                          "閾値を上げると偽陽性が入る(2で16件、適合率0.789)。"
                          "デッキ切れ負けは『自分のターン開始時に引けない』ことなので、"
                          "最後の判断時点で山札0なら必ずデッキ切れ、そうでなければ必ず違う。")
    ap.add_argument("--deckout-penalty", type=float, default=0.0,
                     help="デッキ切れ負けの軌跡に追加で引く罰則(既定0.0=無効、従来と完全に同じ挙動)。"
                          "有効時は学習側が負け かつ looks_like_deckout が True の軌跡の reward を "
                          "0.0 から 0.0 - deckout_penalty に書き換えてから build_padded に渡す。"
                          "注意: これは potential-based shaping ではないため、最適方策を変えうる。"
                          "「デッキ切れを避けるためなら勝率を多少犠牲にする」方策に寄る理論的リスクが"
                          "あるため、有効にした場合は必ず eval_winrate で確認すること。")
    args = ap.parse_args()

    device = args.device
    train_names = _parse_names(args.train_opponents)
    eval_names = _parse_names(args.eval_opponents)
    eval_fixed_names = _parse_names(args.eval_fixed) if args.eval_fixed else []
    if not train_names:
        raise ValueError(f"--train-opponents が空: {args.train_opponents!r}")
    if not eval_names:
        raise ValueError(f"--eval-opponents が空: {args.eval_opponents!r}")

    out = WDIR / f"policy_weights_{args.learner}_pool_{args.tag}.json"
    if out.exists() and not args.force:
        raise FileExistsError(
            f"出力先が既に存在する(既存の重みを保護するため上書きしない): {out}\n"
            "--force を指定すれば上書きできる。"
        )

    tmp = _HERE / f"_tmp_policy_pool_{args.learner}_{args.tag}.json"
    logpath = _HERE / f"_train_pool_{args.learner}_{args.tag}.log"

    learner_w = _resolve_learner_weights_path(args.learner, args.learner_weights)
    _weights_path_reg, learner_deck_csv = pools.resolve_learner(args.learner)
    deck_l = read_deck_csv_file(learner_deck_csv)

    train_opponents = pools.build_opponents(train_names)
    eval_opponents = pools.build_opponents(eval_names)
    eval_fixed_opponents = pools.build_opponents(eval_fixed_names) if eval_fixed_names else None

    print(f"device={device} learner={args.learner} weights={learner_w} "
          f"train_opponents={train_names} eval_opponents={eval_names} "
          f"eval_fixed={eval_fixed_names or None} workers={args.workers} -> {out.name}", flush=True)

    base_payload = json.loads(learner_w.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    history = []

    def do_baseline_or_iter_eval():
        wr, w, v, per_opp = run_eval_pool(
            policy, base_payload, tmp, eval_opponents, deck_l,
            args.eval_games, EVAL_SEED0, args.workers,
        )
        rec = {"eval_winrate": wr, "eval_wins": w, "eval_valid": v, "eval_per_opponent": per_opp}
        if eval_fixed_opponents is not None:
            fwr, fw, fv, fper = run_eval_pool(
                policy, base_payload, tmp, eval_fixed_opponents, deck_l,
                args.eval_games, EVAL_FIXED_SEED0, args.workers,
            )
            rec.update({"eval_fixed_winrate": fwr, "eval_fixed_wins": fw, "eval_fixed_valid": fv,
                        "eval_fixed_per_opponent": fper})
        return rec

    t0 = time.time()
    rec0 = do_baseline_or_iter_eval()
    rec0["iter"] = 0
    print(f"[iter 0] baseline greedy {rec0['eval_wins']}/{rec0['eval_valid']} = "
          f"{rec0['eval_winrate']:.3f} (CI_lo {wilson_lo(rec0['eval_wins'], rec0['eval_valid']):.3f})",
          flush=True)
    if eval_fixed_opponents is not None:
        print(f"    [eval_fixed] {rec0['eval_fixed_wins']}/{rec0['eval_fixed_valid']} = "
              f"{rec0['eval_fixed_winrate']:.3f}", flush=True)
    history.append(rec0)
    best_wr, best_iter = rec0["eval_winrate"], 0
    best_state = copy.deepcopy(policy.state_dict())

    def write_log():
        payload = {
            "args": vars(args),
            "train_opponents": train_names,
            "eval_opponents": eval_names,
            "eval_fixed": eval_fixed_names or None,
            "history": history,
        }
        logpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_log()

    for it in range(1, args.iters + 1):
        export_temp(policy, base_payload, tmp)
        trajs, stats = parallel_collect_pool(
            str(tmp), train_opponents, deck_l,
            n_games=args.games_per_iter, seed0=it * 100_000,
            temperature=args.temperature, workers=args.workers,
        )
        if not trajs:
            print(f"[iter {it}] no trajs", flush=True)
            continue

        # デッキ切れ率の診断ログは --deckout-penalty の有無にかかわらず常に計算・記録する
        # (学習中に下がっているか観察するため)。既定挙動を変えるのは reward の書き換えのみで、
        # それは --deckout-penalty が非0のときだけ行う。
        deckout_flags = [looks_like_deckout(tr, args.deckout_threshold) for tr in trajs]
        n_losses = sum(1 for tr in trajs if tr["reward"] < 1.0)
        n_deckouts = sum(deckout_flags)
        deckout_rate = (n_deckouts / n_losses) if n_losses else float("nan")

        if args.deckout_penalty:
            for tr, is_deckout in zip(trajs, deckout_flags):
                if is_deckout:
                    tr["reward"] = 0.0 - args.deckout_penalty

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

        total = stats["total"]
        train_wr = (total["wins"] / total["valid"]) if total["valid"] else float("nan")
        per_opp_train = summarize_per_opponent(stats)
        print(f"[iter {it}] train_wr {train_wr:.3f} ({total['wins']}/{total['valid']} "
              f"err{total['errors']}) steps {batch['n']} pol {pl:.4f} val {vl:.4f} ent {en:.3f} "
              f"deckout {n_deckouts}/{n_losses}={deckout_rate:.3f} "
              f"{time.time() - t0:.0f}s", flush=True)
        for name, b in per_opp_train.items():
            print(f"    per_opponent[{name}] {b['wins']}/{b['valid']} = {b['winrate']:.3f} "
                  f"err{b['errors']}", flush=True)

        rec = {
            "iter": it, "train_winrate": train_wr, "steps": batch["n"],
            "pol_loss": pl, "val_loss": vl, "entropy": en,
            "per_opponent_train": per_opp_train,
            "deckout_losses": n_deckouts, "deckout_total_losses": n_losses,
            "deckout_rate": deckout_rate, "deckout_threshold": args.deckout_threshold,
        }
        if it % args.eval_every == 0 or it == args.iters:
            erec = do_baseline_or_iter_eval()
            rec.update(erec)
            star = ""
            if erec["eval_winrate"] > best_wr:
                best_wr, best_iter = erec["eval_winrate"], it
                best_state = copy.deepcopy(policy.state_dict()); star = " *BEST*"
            print(f"    [eval] {erec['eval_wins']}/{erec['eval_valid']} = {erec['eval_winrate']:.3f} "
                  f"(CI_lo {wilson_lo(erec['eval_wins'], erec['eval_valid']):.3f}) "
                  f"best={best_wr:.3f}@{best_iter}{star}", flush=True)
            if eval_fixed_opponents is not None:
                print(f"    [eval_fixed] {erec['eval_fixed_wins']}/{erec['eval_fixed_valid']} = "
                      f"{erec['eval_fixed_winrate']:.3f}", flush=True)
        history.append(rec)
        write_log()

    # best-checkpoint を復元して最終評価する。
    policy.load_state_dict(best_state)

    final_wr, final_w, final_v, final_per_opp = run_eval_pool(
        policy, base_payload, tmp, eval_opponents, deck_l,
        args.final_eval_games, FINAL_EVAL_SEED0, args.workers,
    )
    print(f"[final eval] vs eval_opponents {final_w}/{final_v} = {final_wr:.3f} "
          f"(CI_lo {wilson_lo(final_w, final_v):.3f})", flush=True)
    final_eval = {
        "eval_opponents_winrate": final_wr, "eval_opponents_wins": final_w,
        "eval_opponents_valid": final_v, "eval_opponents_per_opponent": final_per_opp,
    }
    if eval_fixed_opponents is not None:
        ffinal_wr, ffinal_w, ffinal_v, ffinal_per_opp = run_eval_pool(
            policy, base_payload, tmp, eval_fixed_opponents, deck_l,
            args.final_eval_games, FINAL_EVAL_FIXED_SEED0, args.workers,
        )
        print(f"[final eval] vs eval_fixed {ffinal_w}/{ffinal_v} = {ffinal_wr:.3f} "
              f"(CI_lo {wilson_lo(ffinal_w, ffinal_v):.3f})", flush=True)
        final_eval.update({
            "eval_fixed_winrate": ffinal_wr, "eval_fixed_wins": ffinal_w,
            "eval_fixed_valid": ffinal_v, "eval_fixed_per_opponent": ffinal_per_opp,
        })

    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({
        "rl_pool_finetuned": True,
        "rl_pool_best_iter": best_iter,
        "rl_pool_best_eval_winrate": best_wr,
        "rl_pool_baseline_eval_winrate": rec0["eval_winrate"],
        "rl_pool_train_opponents": train_names,
        "rl_pool_eval_opponents": eval_names,
        "rl_pool_eval_fixed": eval_fixed_names or None,
        "rl_pool_final_eval": final_eval,
    })
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    history.append({"iter": "final", "final_eval": final_eval})
    write_log()

    print(f"exported BEST (iter {best_iter}, eval {best_wr:.3f}, baseline {rec0['eval_winrate']:.3f}) "
          f"-> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
