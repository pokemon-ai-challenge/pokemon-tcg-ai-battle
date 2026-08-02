"""learner: 各 worker のシャードを集約して PPO を1回更新し、次の世代のモデルを作る。

    python learner.py --run-dir <run_dir>            # 1世代ぶん更新する
    python learner.py --run-dir <run_dir> --status   # 何が揃っているか見るだけ

やっていること(1回の実行 = 1世代):
  1. いま配布中の世代 vN と、その model_v<N>.json のハッシュを確定する
  2. shards/v<N>/ のシャードを1つずつ検査する。世代・モデルのハッシュ・収集量・温度・
     相手設定のどれか1つでも食い違ったら、そのシャードは使わない
  3. 登録済みの worker が全部そろっていなければ止まる(--allow-partial で続行可)
  4. 合格したシャードだけを結合して PPO 更新 → model_v<N+1>.json
  5. 使ったシャードを consumed/ へ退避し、run.json の世代を N+1 に進める

critic(価値関数)と Adam の内部状態は state/trainer_v<N>.pt に持ち越す。ここを毎世代
作り直すと、GAE の基準がリセットされて学習が進まなくなる。

train_v3.py との意図的な違い:
  重要度比の計算に収集時と同じ温度を使う。worker は softmax(scores / T) で行動を選ぶのに、
  train_v3 は log_softmax(scores) で比を取っており、T != 1 のとき収集時の方策と学習対象の
  方策がずれる。分散化では「収集時と更新時の方策を揃える」ことが前提なので、ここを揃えた。
  T = 1.0(既定)なら train_v3 と同じ計算になる。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import common as C
from collect_parallel import parallel_collect
from run_league import read_deck_csv_file
from torch_policy import TorchOptionPolicy
from train_v3 import Critic, compute_gae, wilson_lo


# ------------------------------------------------------------------ バッチ構築
def build_batch(shards: list[tuple[dict, dict]], device) -> dict:
    """シャード群を1つの学習バッチにまとめる。

    選択肢は決定点ごとに数が違う。シャードは縦に連結した形(opts)+ 各決定点の
    行数(counts)で持っており、**その形のまま扱う**。

    以前は最大数までゼロ詰めして (n, max_n, dim) の四角に直していたが、選択肢は
    平均7.5個に対し最大42個あるため、計算の8割が「存在しない選択肢のゼロ」に
    費やされていた(結果は -inf で潰すので捨てるためだけの計算)。詰めるのをやめ、
    どの選択肢がどの決定点のものかを `seg` で持つ。
    """
    cat = lambda key: np.concatenate([a[key] for a, _ in shards])  # noqa: E731
    state = cat("state"); opts = cat("opts"); cids = cat("cids"); counts = cat("counts")
    chosen = cat("chosen"); logp = cat("logp")
    lengths = cat("lengths"); rewards = cat("rewards")

    n = len(counts)
    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    counts_t = t(counts.astype(np.int64))
    # offsets[i] = 決定点 i の選択肢が opts の何行目から始まるか
    offsets = torch.cumsum(counts_t, 0) - counts_t
    chosen_t = t(chosen.astype(np.int64))
    return {
        "state_rows": t(state.astype(np.float32)),
        "opts": t(opts.astype(np.float32)),
        "cids": t(cids.astype(np.int64)),
        "counts": counts_t,
        "offsets": offsets,
        "seg": torch.repeat_interleave(torch.arange(n, device=device), counts_t),
        "chosen": chosen_t,
        "chosen_row": offsets + chosen_t,     # 選んだ手が opts の何行目か
        "old_logp": t(logp.astype(np.float32)),
        "n": n,
        "lengths": [int(x) for x in lengths],
        "rewards": [float(x) for x in rewards],
    }


def slice_batch(batch: dict, idx: torch.Tensor) -> dict:
    """ミニバッチを切り出す(決定点単位。GAE は事前に全体で計算済み)。

    決定点を選ぶと、その決定点に属する選択肢の行もまとめて拾う必要がある。
    `offsets` と `counts` から行番号を作って一度に取り出す。
    """
    counts = batch["counts"].index_select(0, idx)
    off = batch["offsets"].index_select(0, idx)
    m = int(idx.numel())
    local_off = torch.cumsum(counts, 0) - counts        # 切り出した後の並びでの開始位置
    seg = torch.repeat_interleave(torch.arange(m, device=idx.device), counts)
    inner = torch.arange(int(counts.sum()), device=idx.device) - local_off.index_select(0, seg)
    rows = off.index_select(0, seg) + inner
    return {
        "state_rows": batch["state_rows"].index_select(0, idx),
        "opts": batch["opts"].index_select(0, rows),
        "cids": batch["cids"].index_select(0, rows),
        "seg": seg,
        "chosen_row": local_off + batch["chosen"].index_select(0, idx),
        "old_logp": batch["old_logp"].index_select(0, idx),
        "n": m,
    }


def _check_optimizer_shapes(opt) -> None:
    """Adam の moment がパラメータと同じ形か確かめる。

    `load_state_dict` は形を検査しないので、合わないまま進むと最初の step で落ちる。
    先に気づけるよう明示的に見る。
    """
    for group in opt.param_groups:
        for p in group["params"]:
            slot = opt.state.get(p)
            if not slot:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                t = slot.get(key)
                if t is not None and tuple(t.shape) != tuple(p.shape):
                    raise ValueError(
                        f"{key} の形が合わない: {tuple(t.shape)} != {tuple(p.shape)}")


def policy_logp_entropy(policy, batch, temperature: float):
    """選んだ行動の対数確率とエントロピー。収集時と同じ温度で softmax を取る。

    選択肢はゼロ詰めせず縦に連結したままなので、softmax は `seg` で区切って取る
    (決定点ごとの区間ごとに正規化する)。
    """
    seg, n = batch["seg"], batch["n"]
    scores = policy.option_scores_segmented(
        batch["state_rows"], batch["opts"], batch["cids"], seg) / temperature
    # log-sum-exp を安定させるための最大値。理論上は分子と分母で打ち消し合うので
    # 勾配は流さない(流しても打ち消えるが、無駄な計算と丸め誤差になる)。
    neg = torch.full((n,), float("-inf"), dtype=scores.dtype, device=scores.device)
    mx = neg.scatter_reduce(0, seg, scores, reduce="amax", include_self=False).detach()
    z = scores - mx.index_select(0, seg)
    den = torch.zeros(n, dtype=scores.dtype, device=scores.device).index_add(0, seg, torch.exp(z))
    logp = z - torch.log(den).index_select(0, seg)
    chosen_logp = logp.index_select(0, batch["chosen_row"])
    ent = -torch.zeros(n, dtype=scores.dtype, device=scores.device).index_add(
        0, seg, torch.exp(logp) * logp)
    return chosen_logp, ent


# ------------------------------------------------------------------ シャード収集
def default_inbox() -> Path:
    """ブラウザのダウンロード先。Colab の files.download() はここに落ちる。"""
    return Path.home() / "Downloads"


def import_inbox(run_dir: Path, run: dict, gen: int, model_sha: str, inbox: Path) -> list[str]:
    """ダウンロードフォルダに落ちているシャードを shards/v<gen>/ へ取り込む。

    今の世代・今のモデルに合うものだけを、その worker 名で受け入れる。合わないファイル
    (前の世代の残り、無関係な .npz)は黙って無視する。元ファイルは消さずコピーする。
    """
    if not inbox.is_dir():
        return []
    dest_dir = C.shard_dir(run_dir, gen)
    imported = []
    for p in sorted(inbox.glob("*.npz")):
        try:
            _, meta = C.read_shard(p)
            C.check_shard(meta, run, gen, model_sha)
        except Exception:
            continue          # 別の run / 世代 / 壊れたファイル
        wid = meta.get("worker_id")
        if wid not in run["workers"]:
            continue
        dest = dest_dir / f"{wid}.npz"
        if dest.exists():
            continue          # すでに提出済み
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        imported.append(f"{p.name} -> {wid}")
    return imported


def gather_shards(run_dir: Path, run: dict, gen: int, model_sha: str):
    """(合格シャード, 却下理由, 未提出の worker) を返す。"""
    accepted, rejected = [], []
    found_workers = set()
    sdir = C.shard_dir(run_dir, gen)
    paths = sorted(sdir.glob("*.npz")) if sdir.exists() else []
    for p in paths:
        try:
            arrays, meta = C.read_shard(p)
            C.check_shard(meta, run, gen, model_sha)
        except C.ShardRejected as exc:
            rejected.append((p, str(exc)))
            continue
        except Exception as exc:  # 壊れたファイル等
            rejected.append((p, f"読み込めない: {exc!r}"))
            continue
        accepted.append((arrays, meta, p))
        found_workers.add(meta["worker_id"])
    missing = [w for w in run["workers"] if w not in found_workers]
    return accepted, rejected, missing


def print_status(run_dir: Path, run: dict, gen: int, model_sha: str):
    accepted, rejected, missing = gather_shards(run_dir, run, gen, model_sha)
    print(f"run={run['run_id']}  いま配布中の世代 v{gen}")
    print(f"  model_v{gen}.json sha256={model_sha[:16]}...")
    print(f"  worker: {run['workers']}  1台あたり {run['games_per_worker']}試合")
    print(f"\n  提出済み({len(accepted)}件):")
    for _, meta, p in accepted:
        wr = meta["wins"] / meta["valid"] if meta["valid"] else float("nan")
        print(f"    OK  {p.name:20s} 決定点{meta['n_steps']:7d} 勝率{wr:.3f} "
              f"({meta['seconds']:.0f}s @ {meta['host']})")
    for p, why in rejected:
        print(f"    NG  {p.name:20s} {why}")
    if missing:
        print(f"\n  未提出: {', '.join(missing)}")
    else:
        print("\n  全 worker そろっている。learner.py を実行すれば v%d を作れる。" % (gen + 1))


# ------------------------------------------------------------------ 評価
def evaluate(run: dict, run_dir: Path, weights: Path, games: int, workers: int):
    """世代をまたいで比べたいので、評価相手は収集用プールとは切り離して固定する。"""
    ev = run.get("eval_opponent") or run["opponents"][0]
    deck_l = read_deck_csv_file(str(C.resolve_deck(run["learner_deck"])))
    deck_o = read_deck_csv_file(str(C.resolve_deck(ev.get("deck") or run["opponent_deck"])))
    opp = C.resolve_opponent_weights(run_dir, ev.get("weights"))
    _, w, v, _ = parallel_collect(str(weights), opp, deck_l, deck_o,
                                  games, C.EVAL_SEED_BASE, temperature=0.01, workers=workers)
    return (w / v if v else float("nan")), w, v


def evaluate_pool(run: dict, run_dir: Path, weights: Path, games: int, workers: int):
    """収集用プールの全相手に対して greedy で評価する。

    学習はプール平均を上げにいくので、固定1相手の勝率だけ見ていると
    「目的は達成しているのに指標は下がる」という読み違いが起きる。
    """
    deck_l = read_deck_csv_file(str(C.resolve_deck(run["learner_deck"])))
    opponents = run["opponents"]
    counts = C.split_games(games, opponents)
    per, tot_w, tot_v, offset = [], 0, 0, 0
    for opp, n in zip(opponents, counts):
        if n == 0:
            continue
        deck_o = read_deck_csv_file(
            str(C.resolve_deck(opp.get("deck") or run["opponent_deck"])))
        opp_w = C.resolve_opponent_weights(run_dir, opp.get("weights"))
        _, w, v, _ = parallel_collect(str(weights), opp_w, deck_l, deck_o, n,
                                      C.EVAL_SEED_BASE + 500_000 + offset,
                                      temperature=0.01, workers=workers)
        offset += n
        tot_w += w; tot_v += v
        per.append({"id": opp["id"], "wins": w, "valid": v,
                    "winrate": (w / v if v else float("nan"))})
    return per, (tot_w / tot_v if tot_v else float("nan")), tot_w, tot_v


# ------------------------------------------------------------------ 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--status", action="store_true", help="集計せず、揃い具合を表示して終了")
    ap.add_argument("--allow-partial", action="store_true",
                    help="未提出の worker があっても更新する(収集量が世代間で変わる点に注意)")
    ap.add_argument("--eval-games", type=int, default=0,
                    help="新モデルの評価試合数。0 なら評価しない")
    ap.add_argument("--eval-pool-games", type=int, default=None,
                    help="プール評価の試合数。既定は run.json の eval_pool_games。"
                         "0 を渡すとその世代はプール評価を飛ばす")
    ap.add_argument("--eval-workers", type=int, default=None)
    ap.add_argument("--device", default="cpu",
                    help="モデルが小さいので cpu で十分(GPUにしても速くならない)")
    ap.add_argument("--start-method", default="spawn", choices=["spawn", "fork", "default"],
                    help="評価対戦のプロセス起動方式(worker.py と同じ理由で既定 spawn)")
    ap.add_argument("--inbox", default=None,
                    help="シャードを拾うフォルダ。既定はブラウザのダウンロード先。"
                         "Colab から落とした .npz を置き直さずに済ませるためのもの")
    ap.add_argument("--no-inbox", action="store_true", help="ダウンロードフォルダを見ない")
    args = ap.parse_args()

    C.set_collection_start_method(args.start_method)
    run_dir = Path(args.run_dir)
    run = C.load_run(run_dir)
    gen = run["generation"]
    model = C.model_path(run_dir, gen)
    if not model.exists():
        raise SystemExit(f"モデルが無い: {model}")
    model_sha = C.sha256_file(model)

    if not args.no_inbox:
        inbox = Path(args.inbox) if args.inbox else default_inbox()
        for line in import_inbox(run_dir, run, gen, model_sha, inbox):
            print(f"[取り込み] {inbox.name}/{line}", flush=True)

    if args.status:
        print_status(run_dir, run, gen, model_sha)
        return

    next_model = C.model_path(run_dir, gen + 1)
    if next_model.exists():
        # 更新は「モデル保存 -> 評価 -> シャード退避 -> 世代を進める」の順。途中で落ちると
        # 次世代のモデルだけが残り、run.json は前の世代のままになる。この状態は
        #   ・次世代モデルがある
        #   ・この世代のシャードがまだ consumed に移っていない
        #   ・history にこの世代の記録が無い
        # の3つが揃うことで一意に見分けられる。中断とみなしてやり直す。
        done = set()
        if C.history_path(run_dir).exists():
            for line in C.history_path(run_dir).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["generation"])
        shards_left = list(C.shard_dir(run_dir, gen).glob("*.npz"))
        if gen not in done and shards_left:
            print(f"[復旧] v{gen+1} のモデルだけが残っている(更新が中断された)。"
                  "作り直す。", flush=True)
            next_model.unlink()
            st_next = C.trainer_state_path(run_dir, gen + 1)
            if st_next.exists():
                st_next.unlink()
        else:
            raise SystemExit(
                f"v{gen+1} はすでに存在する: {next_model}\n"
                "  この世代の更新は済んでいる。run.json の generation がずれていないか確認する。")

    accepted, rejected, missing = gather_shards(run_dir, run, gen, model_sha)
    for p, why in rejected:
        print(f"[除外] {p.name}: {why}", flush=True)
    if not accepted:
        raise SystemExit(f"使えるシャードが1件も無い(v{gen})。")
    if missing and not args.allow_partial:
        raise SystemExit(
            f"未提出の worker がある: {', '.join(missing)}\n"
            "  PPO は全 worker が同じ世代で同じ量を集めた前提。待つか --allow-partial を付ける。")
    if rejected:
        print("  ↑ 除外したデータは古い世代のもの。該当 worker に最新モデルを配り直して再収集する。",
              flush=True)

    shards = [(a, m) for a, m, _ in accepted]
    used_paths = [p for _, _, p in accepted]
    total_games = sum(m["valid"] for _, m in shards)
    total_wins = sum(m["wins"] for _, m in shards)

    print(f"\nv{gen} 集約: {len(shards)}台 / {total_games}試合 / "
          f"勝率 {total_wins}/{total_games}={total_wins/max(total_games,1):.3f}", flush=True)

    device = args.device
    ppo = run["ppo"]
    temperature = float(run["temperature"])
    batch = build_batch(shards, device)
    print(f"  決定点 {batch['n']} 件(選択肢 {len(batch['opts'])} 行 / "
          f"最大 {int(batch['counts'].max())} 個)", flush=True)

    base_payload = json.loads(model.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(model).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=ppo["lr_policy"])
    opt_v = torch.optim.Adam(critic.parameters(), lr=ppo["lr_value"])

    # critic と optimizer は世代をまたいで持ち越す(毎回初期化すると学習が進まない)。
    state_path = C.trainer_state_path(run_dir, gen)
    if state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=True)
        # 形が合わないものは引き継がずに作り直す。optimizer の load_state_dict は形を
        # 検査しないので、そのまま進めると更新の途中で落ちる(特徴や中間層を増やした
        # 直後がこれ)。ここで気づけるようにしておく。
        loaded, reset = [], []
        try:
            critic.load_state_dict(st["critic"])
            loaded.append("critic")
        except Exception as exc:
            reset.append(f"critic({exc.__class__.__name__})")
        for name, opt, key in (("opt_policy", opt_p, "opt_policy"),
                               ("opt_value", opt_v, "opt_value")):
            try:
                opt.load_state_dict(st[key])
                _check_optimizer_shapes(opt)
                loaded.append(name)
            except Exception as exc:
                # 作り直す(学習率などは run.json から復元されるので実害は小さい)
                fresh = torch.optim.Adam(
                    policy.parameters() if key == "opt_policy" else critic.parameters(),
                    lr=ppo["lr_policy"] if key == "opt_policy" else ppo["lr_value"])
                if key == "opt_policy":
                    opt_p = fresh
                else:
                    opt_v = fresh
                reset.append(f"{name}({exc.__class__.__name__})")
        print(f"  学習状態: 引き継ぎ {', '.join(loaded) or 'なし'}"
              + (f" / 作り直し {', '.join(reset)}" if reset else ""), flush=True)
    elif gen > 0:
        print(f"  警告: {state_path.name} が無い。critic を初期化して続行する"
              "(この世代の更新は品質が落ちる)。", flush=True)

    with torch.no_grad():
        values = critic(batch["state_rows"])
    adv, vtarget = compute_gae(batch["lengths"], batch["rewards"], values,
                               ppo["gamma"], ppo["lam"], device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    mb = int(ppo.get("minibatch_size", 16384))
    target_kl = float(ppo.get("target_kl", 0.0))   # 0 なら無効
    n = batch["n"]
    order_gen = torch.Generator().manual_seed(gen)
    pl = vl = en = 0.0
    kl = 0.0
    steps = 0
    stopped_at = None
    t0 = time.time()
    for epoch in range(ppo["epochs"]):
        perm = torch.randperm(n, generator=order_gen) if mb > 0 else torch.arange(n)
        chunks = ([perm[i:i + mb] for i in range(0, n, mb)] if mb > 0 else [perm])
        for idx in chunks:
            idx = idx.to(device)
            sub = slice_batch(batch, idx)
            new_logp, ent = policy_logp_entropy(policy, sub, temperature)
            ratio = torch.exp(new_logp - sub["old_logp"])
            a = adv[idx]
            s1 = ratio * a
            s2 = torch.clamp(ratio, 1 - ppo["clip"], 1 + ppo["clip"]) * a
            pol_loss = -torch.min(s1, s2).mean() - ppo["entropy"] * ent.mean()
            opt_p.zero_grad(); pol_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt_p.step()

            v_pred = critic(sub["state_rows"])
            val_loss = ((v_pred - vtarget[idx]) ** 2).mean()
            opt_v.zero_grad(); val_loss.backward(); opt_v.step()
            pl, vl, en = pol_loss.item(), val_loss.item(), ent.mean().item()
            steps += 1

        # エポックを増やすほど、収集時の方策から離れていく。離れすぎると PPO の前提
        # (収集時と更新対象が近い)が崩れるので、KL で見て打ち切る。
        if target_kl > 0:
            with torch.no_grad():
                lp, _ = policy_logp_entropy(policy, batch, temperature)
                r = torch.exp(lp - batch["old_logp"])
                kl = float(((r - 1) - (lp - batch["old_logp"])).mean())
            if kl > target_kl:
                stopped_at = epoch + 1
                break

    note = f" KL {kl:.4f}" + (f" (epoch {stopped_at} で打ち切り)" if stopped_at else "")
    print(f"  PPO更新 {time.time()-t0:.1f}s  更新{steps}回  "
          f"pol {pl:.4f} val {vl:.4f} ent {en:.3f}{note}", flush=True)

    # --- 保存 ---
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({
        "distributed_selfplay": True,
        "run_id": run["run_id"],
        "generation": gen + 1,
        "parent_sha256": model_sha,
        "games_used": total_games,
        "workers_used": sorted(m["worker_id"] for _, m in shards),
    })
    # 電源断などで書きかけのまま残ると、次回それを正常なモデルとして読んでしまう。
    # 一時ファイルに書いてから置き換える(置き換えは不可分な操作)。
    tmp_model = next_model.with_suffix(".json.tmp")
    tmp_model.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    st_path = C.trainer_state_path(run_dir, gen + 1)
    st_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_state = st_path.with_suffix(".pt.tmp")
    torch.save({"critic": critic.state_dict(),
                "opt_policy": opt_p.state_dict(),
                "opt_value": opt_v.state_dict()}, tmp_state)
    os.replace(tmp_state, st_path)
    os.replace(tmp_model, next_model)

    record = {
        "generation": gen, "next_generation": gen + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workers": sorted(m["worker_id"] for _, m in shards),
        "games": total_games, "wins": total_wins,
        "collect_winrate": total_wins / max(total_games, 1),
        "steps": batch["n"],
        "pol_loss": pl, "val_loss": vl, "entropy": en,
        "grad_steps": steps, "approx_kl": kl, "kl_early_stop_epoch": stopped_at,
        "rejected": [f"{p.name}: {why}" for p, why in rejected],
        "partial": bool(missing),
    }

    ew = args.eval_workers or (os.cpu_count() or 2)
    if args.eval_games > 0:
        wr, w, v = evaluate(run, run_dir, next_model, args.eval_games, ew)
        record.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
        print(f"  評価 v{gen+1}(固定相手): {w}/{v} = {wr:.3f} "
              f"(CI下限 {wilson_lo(w, v):.3f})", flush=True)

    # プール全体に対する評価。学習が上げにいっているのはこちらなので、
    # 世代をまたいで見るべき主指標はこの平均。
    pool_games = (args.eval_pool_games if args.eval_pool_games is not None
                  else int(run.get("eval_pool_games", 0)))
    if pool_games > 0 and len(run["opponents"]) > 1:
        per, avg, pw, pv = evaluate_pool(run, run_dir, next_model, pool_games, ew)
        record.update({"eval_pool": per, "eval_pool_winrate": avg,
                       "eval_pool_wins": pw, "eval_pool_valid": pv})
        print(f"  評価 v{gen+1}(プール平均): {pw}/{pv} = {avg:.3f} "
              f"(CI下限 {wilson_lo(pw, pv):.3f})", flush=True)
        for e in sorted(per, key=lambda x: x["winrate"]):
            print(f"      vs {e['id']:24s} {e['wins']:3d}/{e['valid']:3d} = {e['winrate']:.3f}",
                  flush=True)

    # --- 使ったシャードを退避し、世代を進める ---
    C.move_consumed(run_dir, gen, used_paths)
    run["generation"] = gen + 1
    C.save_run(run_dir, run)
    C.append_history(run_dir, record)

    print(f"\nv{gen+1} を作成: {next_model}")
    print(f"  sha256={C.sha256_file(next_model)[:16]}...")
    print(f"  次: run.json と models/model_v{gen+1}.json を各 worker へ配って再収集する。")


if __name__ == "__main__":
    main()
