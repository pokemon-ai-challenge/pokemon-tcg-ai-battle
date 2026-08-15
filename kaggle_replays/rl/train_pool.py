"""RL 対戦相手強化(相互鍛錬 段階2): 学習側1モデル vs 相手プールで PPO する。

train_v3.py は「学習側1モデル vs 相手1モデル固定」でしか PPO できない(collect_parallel.parallel_collect
が固定の1相手しか取れないため)。本スクリプトは収集を collect_pool.parallel_collect_pool に差し替え、
相手プール(複数アーキタイプ)に対して学習できるようにする。

**アルゴリズムは再実装しない。** PPO 更新・GAE・パディング・評価用ユーティリティは
train_v3.py からそのまま import する(Critic, build_padded, compute_gae, elo_diff, export_temp,
ppo_update_step, wilson_lo)。本スクリプトが新たに書くのは main() の収集まわり(プール対応)と
CLI・ログ・critic warmup・チェックポイント export(defect#1修正後は常に最終イテレート)のみ。

重要: 固定相手アーム(--train-opponents に1つだけ指定)も、このスクリプトで
「プール要素数1」として走らせる設計にしている。train_v3.py(collect_parallel 経由)と
train_pool.py(collect_pool 経由)とで固定相手アームの実装を分けると、
「相互鍛錬させたら悪化したのは相手プールのせいか、それとも実装差のせいか」が交絡して
切り分けられなくなる。そのため固定相手の比較実験も本スクリプト1本(k=1呼び出し)で行う。

評価用の相手プール(--eval-opponents)は学習中一切更新しない(凍結)。--train-opponents と
--eval-opponents を別引数にしているのは、学習用プールを将来変更しても比較可能な基準を
崩さないため、および「学習に使った相手だけに勝てるようになっただけ」を検出できるようにするため
(通常は両者を同じ値にして使う想定だが、意図的に分離できるようにしてある)。

2026-08-13 修正(requirements-kamitsuorochi-2026-08-12.md §6 step1、roadmap-2026-08-05.md Phase E):
  defect#1 checkpoint選択: best-of-eval(winner's curse)をやめ、最終イテレートを常に export する。
  defect#2 iters: 既定を 20 -> 25(20-25で飽和という実測に基づく)。
  defect#3 Critic初期化: 既定でランダム初期化ではなく value_net/value_weights_v251.json
    (勝率を教師にした事前学習済みMLP)から warm-start する(--critic-init random で旧挙動に戻せる)。
  defect#4 Critic warmup: 既定5イテレーション、方策を凍結して critic のみ更新してから
    joint PPO に入る(--critic-warmup-iters 0 で無効化できる)。
  defect#5 報酬シェーピング: サイド差のポテンシャルシェーピング Φ=c(opp_prize-self_prize)、
    既定 c=0.1(--shaping-c 0 で無効化=旧挙動)。Φ(terminal)≡0 は train_v3.compute_gae 側で
    定数として保証(test_reward_shaping.py 参照)。
  defect#6 正則化: KL(π‖π_BC) を既定 β=0.02 で追加(--kl-beta 0 で無効化=旧挙動)。
  defect#7 評価指標: 勝率に加えて Elo差(173.7*logit(p))を全ての eval ログに併記する。
"""

from __future__ import annotations

import argparse
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
    elo_diff,
    export_temp,
    ppo_update_step,
    wilson_lo,
)
from torch_policy import TorchOptionPolicy  # noqa: E402
from collect_pool import parallel_collect_pool  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
import pools  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
VALUE_NET_WEIGHTS = _ROOT / "kaggle_replays" / "value_net" / "value_weights_v251.json"

# eval 系の seed は学習(train)の it*100000 空間([100000, iters*100000])と重ならないよう、
# 十分大きい定数オフセットに分離してある(重なっても正当性は壊れないが、意図を明確にするため)。
EVAL_SEED0 = 800_000_000
EVAL_FIXED_SEED0 = 810_000_000
FINAL_EVAL_SEED0 = 820_000_000
FINAL_EVAL_FIXED_SEED0 = 821_000_000
# defect#4: critic warmup 収集用の seed 空間。学習ループの it*100_000([100_000, iters*100_000])
# より下・EVAL_SEED0 より十分下なので、他のどの seed 空間とも重ならない。
WARMUP_SEED0 = 700_000_000

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
    ap.add_argument("--iters", type=int, default=25,
                     help="joint PPO のイテレーション数。既定25(旧20。requirements-kamitsuorochi-"
                          "2026-08-12.md §6 step1: 20-25で飽和という実測に基づく defect#2)。"
                          "critic warmup(--critic-warmup-iters)はこれとは別枠でこの前に走る。")
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
    ap.add_argument("--critic-init", choices=["value_net", "random"], default="value_net",
                     help="defect#3(最大の欠陥、'Critic のランダム初期化'): 既定は "
                          "kaggle_replays/value_net/ の学習済み勝率MLPから critic を warm-start する "
                          "(--value-net-weights で差し替え可)。--critic-init random で旧来のランダム"
                          "初期化(Critic(state_dim, mean, std) の既定コンストラクタ)に戻せる。")
    ap.add_argument("--value-net-weights", default=str(VALUE_NET_WEIGHTS),
                     help=f"--critic-init value_net のときに読む価値関数の重み(既定 {VALUE_NET_WEIGHTS.name})。"
                          "全アーキタイプ共通の汎用勝率推定器(kamitsuorochi専用ではない)。"
                          "state_dim=715 のうち、学習側方策の ablate 済み特徴を除いた列だけを使う"
                          "(名前で対応付け。学習側の非ablate特徴が value_net の feature_names に"
                          "無ければその列は0初期化になり、coverage としてログに出す)。")
    ap.add_argument("--critic-warmup-iters", type=int, default=5,
                     help="defect#4: joint PPO の前に、方策を凍結して critic だけを更新する"
                          "イテレーション数。既定5(0で無効化=旧挙動)。--iters には含まれない"
                          "(合計イテレーション数は critic-warmup-iters + iters)。")
    ap.add_argument("--shaping-c", type=float, default=0.1,
                     help="defect#5: サイド差ポテンシャルシェーピング Φ=c*(opp_prize_remaining-"
                          "self_prize_remaining) の係数。既定0.1。0を指定すると無効化(旧来の"
                          "終局のみ報酬に戻る、Φ(terminal)≡0の保証込みでも c=0 なら数式上シェーピング"
                          "項が消えるので実質同じ)。")
    ap.add_argument("--kl-beta", type=float, default=0.02,
                     help="defect#6: KL(π‖π_BC) の係数 β。既定0.02(0.01〜0.05の範囲でsweep可能、"
                          "plan doc根拠: crustle/lucarioがRLで悪化したのはこれが無いため)。"
                          "0を指定すると無効化(旧来のentropyのみの正則化)。π_BC は --learner-weights"
                          "(学習開始時の重み)をこのプロセス内でもう一度ロードして凍結したもの。")
    ap.add_argument("--extra-registry", default=None,
                     help="pools.LEARNER_REGISTRY に追加登録する JSON ファイル"
                          "({name: [weights_file, archetype], ...})へのパス(省略可)。"
                          "省略時は今までと完全に同じ挙動。train_league.py が世代ごとに動的な"
                          "名前(過去チェックポイント等)を --train-opponents / --learner に"
                          "渡すために使う(pools.load_extra_registry 参照)。")
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

    if args.extra_registry:
        pools.load_extra_registry(args.extra_registry)

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

    # defect#3: critic のランダム初期化(最大の欠陥)。既定は学習済み勝率MLPから warm-start する。
    critic_coverage = None
    if args.critic_init == "value_net":
        vn_path = Path(args.value_net_weights)
        value_net_payload = json.loads(vn_path.read_text(encoding="utf-8"))
        critic, critic_coverage = Critic.from_value_net(
            len(std["state_mean"]), std["state_mean"], std["state_std"],
            encoder.FEATURE_NAMES, value_net_payload, device=device,
        )
        print(f"critic_init=value_net <- {vn_path.name} (feature coverage {critic_coverage:.3f} "
              f"= {round(critic_coverage * len(encoder.FEATURE_NAMES))}/{len(encoder.FEATURE_NAMES)} dims; "
              "覆われない列は0初期化)", flush=True)
    else:
        critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
        print("critic_init=random (旧来のランダム初期化。--critic-init value_net で "
              "warm-start に切り替えられる)", flush=True)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    # defect#6: KL(π‖π_BC) の基準となる凍結BC方策。--learner-weights と全く同じ重みを
    # 独立にもう一度ロードする(policy とパラメータを共有しない別インスタンス)。
    # 学習が進んでも一切更新しない(requires_grad_(False) + eval())。
    bc_policy = None
    if args.kl_beta:
        bc_policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
        bc_policy.requires_grad_(False)
        bc_policy.eval()
        print(f"kl_beta={args.kl_beta} (π_BC = {learner_w.name} を凍結)", flush=True)
    else:
        print("kl_beta=0 (KL(π‖π_BC) 正則化は無効)", flush=True)

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
          f"{rec0['eval_winrate']:.3f} (CI_lo {wilson_lo(rec0['eval_wins'], rec0['eval_valid']):.3f}) "
          f"Elo {elo_diff(rec0['eval_wins'], rec0['eval_valid']):+.1f}",
          flush=True)
    if eval_fixed_opponents is not None:
        print(f"    [eval_fixed] {rec0['eval_fixed_wins']}/{rec0['eval_fixed_valid']} = "
              f"{rec0['eval_fixed_winrate']:.3f}", flush=True)
    history.append(rec0)
    # defect#1: best_wr/best_iter はログ表示専用の参考値(「学習中どこが良さそうに見えたか」を
    # 見るため)。best-of-eval の重みを保存・復元することは一切しない(winner's curse対策)。
    # export される重みは常にループを最後まで回した後の最終イテレートの policy そのもの。
    best_wr, best_iter = rec0["eval_winrate"], 0

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

    # ---------------------------------------------------------------- defect#4: critic warmup
    # 方策を凍結して critic だけを args.critic_warmup_iters 回更新する(joint PPO の前)。
    # --iters のカウントには含めない(合計イテレーション数は critic_warmup_iters + iters)。
    if args.critic_warmup_iters > 0:
        policy_norm_before = float(sum(p.detach().float().pow(2).sum() for p in policy.parameters()) ** 0.5)
        for wit in range(1, args.critic_warmup_iters + 1):
            export_temp(policy, base_payload, tmp)
            wtrajs, wstats = parallel_collect_pool(
                str(tmp), train_opponents, deck_l,
                n_games=args.games_per_iter, seed0=WARMUP_SEED0 + wit * 100_000,
                temperature=args.temperature, workers=args.workers,
            )
            if not wtrajs:
                print(f"[warmup {wit}/{args.critic_warmup_iters}] no trajs", flush=True)
                continue
            wbatch = build_padded(wtrajs, device)
            with torch.no_grad():
                wvalues = critic(wbatch["state_rows"])
            wadv, wvtarget = compute_gae(
                wbatch["lengths"], wbatch["rewards"], wvalues, args.gamma, args.lam, device,
                state_rows=wbatch["state_rows"] if args.shaping_c else None, shaping_c=args.shaping_c,
            )
            wadv = (wadv - wadv.mean()) / (wadv.std() + 1e-8)
            vl = 0.0
            for _ in range(args.epochs):
                _, vl, _, _ = ppo_update_step(
                    policy, critic, opt_p, opt_v, wbatch, wadv, wvtarget,
                    clip=args.clip, entropy_coef=args.entropy, update_policy=False,
                )
            wtotal = wstats["total"]
            wwr = (wtotal["wins"] / wtotal["valid"]) if wtotal["valid"] else float("nan")
            print(f"[warmup {wit}/{args.critic_warmup_iters}] val_loss {vl:.4f} "
                  f"collect_wr(frozen policy, informational) {wwr:.3f} steps {wbatch['n']} "
                  f"{time.time() - t0:.0f}s", flush=True)
        policy_norm_after = float(sum(p.detach().float().pow(2).sum() for p in policy.parameters()) ** 0.5)
        norm_delta = abs(policy_norm_after - policy_norm_before)
        print(f"[warmup done] policy param norm before={policy_norm_before:.6f} "
              f"after={policy_norm_after:.6f} delta={norm_delta:.3e} (0であるべき)", flush=True)
        assert norm_delta == 0.0, (
            f"critic warmup 中に policy パラメータのnormが変化した(delta={norm_delta}): "
            "update_policy=False(方策凍結)が効いていない可能性がある。ppo_update_step のバグ疑い。"
        )

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
        # defect#5: 報酬シェーピング(shaping_c=0 なら元の「終局のみ報酬」と完全に同じ計算)。
        adv, vtarget = compute_gae(
            batch["lengths"], batch["rewards"], values, args.gamma, args.lam, device,
            state_rows=batch["state_rows"] if args.shaping_c else None, shaping_c=args.shaping_c,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl = vl = en = kl = 0.0
        for _ in range(args.epochs):
            pl, vl, en, kl = ppo_update_step(
                policy, critic, opt_p, opt_v, batch, adv, vtarget,
                clip=args.clip, entropy_coef=args.entropy,
                kl_beta=args.kl_beta, bc_policy=bc_policy, update_policy=True,
            )

        total = stats["total"]
        train_wr = (total["wins"] / total["valid"]) if total["valid"] else float("nan")
        per_opp_train = summarize_per_opponent(stats)
        print(f"[iter {it}] train_wr {train_wr:.3f} ({total['wins']}/{total['valid']} "
              f"err{total['errors']}) steps {batch['n']} pol {pl:.4f} val {vl:.4f} ent {en:.3f} "
              f"kl_bc {kl:.4f} deckout {n_deckouts}/{n_losses}={deckout_rate:.3f} "
              f"{time.time() - t0:.0f}s", flush=True)
        for name, b in per_opp_train.items():
            print(f"    per_opponent[{name}] {b['wins']}/{b['valid']} = {b['winrate']:.3f} "
                  f"err{b['errors']}", flush=True)

        rec = {
            "iter": it, "train_winrate": train_wr, "steps": batch["n"],
            "pol_loss": pl, "val_loss": vl, "entropy": en, "kl_to_bc": kl,
            "per_opponent_train": per_opp_train,
            "deckout_losses": n_deckouts, "deckout_total_losses": n_losses,
            "deckout_rate": deckout_rate, "deckout_threshold": args.deckout_threshold,
        }
        if it % args.eval_every == 0 or it == args.iters:
            erec = do_baseline_or_iter_eval()
            rec.update(erec)
            erec_elo = elo_diff(erec["eval_wins"], erec["eval_valid"])
            rec["eval_elo_diff"] = erec_elo
            # defect#1: best_wr/best_iter は表示専用の参考値のまま更新するが、policyの状態を
            # 保存・復元することはしない(最終export は常に最終イテレート、下記参照)。
            star = ""
            if erec["eval_winrate"] > best_wr:
                best_wr, best_iter = erec["eval_winrate"], it
                star = " (best-so-far, 参考値。exportには使わない)"
            print(f"    [eval] {erec['eval_wins']}/{erec['eval_valid']} = {erec['eval_winrate']:.3f} "
                  f"(CI_lo {wilson_lo(erec['eval_wins'], erec['eval_valid']):.3f}) Elo {erec_elo:+.1f} "
                  f"best_seen={best_wr:.3f}@{best_iter}{star}", flush=True)
            if eval_fixed_opponents is not None:
                print(f"    [eval_fixed] {erec['eval_fixed_wins']}/{erec['eval_fixed_valid']} = "
                      f"{erec['eval_fixed_winrate']:.3f}", flush=True)
        history.append(rec)
        write_log()

    # defect#1: 最終イテレート(このループを最後まで回した時点の policy)をそのまま export する。
    # best-of-eval のチェックポイントに戻す処理はしない(winner's curse対策。根拠は
    # requirements-kamitsuorochi-2026-08-12.md §4-2 末尾: marnie実測で best 0.800/n=200 →
    # 最終 0.757/n=1200、-4.3pt)。

    final_wr, final_w, final_v, final_per_opp = run_eval_pool(
        policy, base_payload, tmp, eval_opponents, deck_l,
        args.final_eval_games, FINAL_EVAL_SEED0, args.workers,
    )
    final_elo = elo_diff(final_w, final_v)
    print(f"[final eval] vs eval_opponents {final_w}/{final_v} = {final_wr:.3f} "
          f"(CI_lo {wilson_lo(final_w, final_v):.3f}) Elo {final_elo:+.1f} "
          f"(baseline Elo {elo_diff(rec0['eval_wins'], rec0['eval_valid']):+.1f}) "
          "-- defect#7: 主指標はこのElo差(勝率ptではない)", flush=True)
    final_eval = {
        "eval_opponents_winrate": final_wr, "eval_opponents_wins": final_w,
        "eval_opponents_valid": final_v, "eval_opponents_per_opponent": final_per_opp,
        "eval_opponents_elo_diff": final_elo,
    }
    if eval_fixed_opponents is not None:
        ffinal_wr, ffinal_w, ffinal_v, ffinal_per_opp = run_eval_pool(
            policy, base_payload, tmp, eval_fixed_opponents, deck_l,
            args.final_eval_games, FINAL_EVAL_FIXED_SEED0, args.workers,
        )
        ffinal_elo = elo_diff(ffinal_w, ffinal_v)
        print(f"[final eval] vs eval_fixed {ffinal_w}/{ffinal_v} = {ffinal_wr:.3f} "
              f"(CI_lo {wilson_lo(ffinal_w, ffinal_v):.3f}) Elo {ffinal_elo:+.1f}", flush=True)
        final_eval.update({
            "eval_fixed_winrate": ffinal_wr, "eval_fixed_wins": ffinal_w,
            "eval_fixed_valid": ffinal_v, "eval_fixed_per_opponent": ffinal_per_opp,
            "eval_fixed_elo_diff": ffinal_elo,
        })

    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({
        "rl_pool_finetuned": True,
        "rl_pool_checkpoint": "final_iterate",  # defect#1: best-of-eval は使わない
        "rl_pool_final_iter": args.iters,
        "rl_pool_critic_warmup_iters": args.critic_warmup_iters,
        "rl_pool_critic_init": args.critic_init,
        "rl_pool_shaping_c": args.shaping_c,
        "rl_pool_kl_beta": args.kl_beta,
        "rl_pool_best_seen_iter_FYI_NOT_EXPORTED": best_iter,
        "rl_pool_best_seen_eval_winrate_FYI_NOT_EXPORTED": best_wr,
        "rl_pool_baseline_eval_winrate": rec0["eval_winrate"],
        "rl_pool_baseline_eval_elo_diff": elo_diff(rec0["eval_wins"], rec0["eval_valid"]),
        "rl_pool_train_opponents": train_names,
        "rl_pool_eval_opponents": eval_names,
        "rl_pool_eval_fixed": eval_fixed_names or None,
        "rl_pool_final_eval": final_eval,
    })
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    history.append({"iter": "final", "final_eval": final_eval})
    write_log()

    print(f"exported FINAL ITERATE (iter {args.iters}, eval_winrate {final_wr:.3f}, "
          f"Elo {final_elo:+.1f}, baseline {rec0['eval_winrate']:.3f}) -> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
