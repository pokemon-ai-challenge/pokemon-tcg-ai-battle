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

    選択肢は決定点ごとに数が違うので、シャードでは縦に連結した形(opts)+ 各決定点の
    行数(counts)で持っている。ここで最大数までゼロ詰めして (n, max_n, dim) に直す。
    """
    cat = lambda key: np.concatenate([a[key] for a, _ in shards])  # noqa: E731
    state = cat("state"); opts = cat("opts"); cids = cat("cids"); counts = cat("counts")
    chosen = cat("chosen"); logp = cat("logp")
    lengths = cat("lengths"); rewards = cat("rewards")

    n = len(counts)
    max_n = int(counts.max())
    od = opts.shape[1]
    # sel[i, j] = (j < counts[i])。行優先で見ると opts の並びとちょうど一致するので、
    # ループなしでゼロ詰め配列に流し込める。
    sel = np.arange(max_n)[None, :] < counts[:, None]
    option_pad = np.zeros((n, max_n, od), dtype=np.float32); option_pad[sel] = opts
    card_pad = np.zeros((n, max_n), dtype=np.int64); card_pad[sel] = cids

    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return {
        "state_rows": t(state.astype(np.float32)),
        "option_pad": t(option_pad),
        "card_pad": t(card_pad),
        "mask": t(sel.astype(np.float32)),
        "chosen": t(chosen.astype(np.int64)),
        "old_logp": t(logp.astype(np.float32)),
        "n": n, "max_n": max_n,
        "lengths": [int(x) for x in lengths],
        "rewards": [float(x) for x in rewards],
    }


def slice_batch(batch: dict, idx: torch.Tensor) -> dict:
    """ミニバッチを切り出す(決定点単位。GAE は事前に全体で計算済み)。"""
    sub = {k: batch[k][idx] for k in
           ("state_rows", "option_pad", "card_pad", "mask", "chosen", "old_logp")}
    sub["n"] = int(idx.numel())
    sub["max_n"] = batch["max_n"]
    return sub


def policy_logp_entropy(policy, batch, temperature: float):
    """選んだ行動の対数確率とエントロピー。収集時と同じ温度で softmax を取る。"""
    n, max_n = batch["n"], batch["max_n"]
    sd = batch["state_rows"].shape[1]
    od = batch["option_pad"].shape[2]
    sf = batch["state_rows"].unsqueeze(1).expand(n, max_n, sd).reshape(n * max_n, sd)
    of = batch["option_pad"].reshape(n * max_n, od)
    cf = batch["card_pad"].reshape(n * max_n)
    scores = policy.option_scores_flat(sf, of, cf).reshape(n, max_n) / temperature
    scores = torch.where(batch["mask"] > 0, scores,
                         torch.full_like(scores, torch.finfo(scores.dtype).min))
    logp = torch.log_softmax(scores, dim=1)
    chosen_logp = logp.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)
    ent = -(logp.exp() * logp.masked_fill(batch["mask"] == 0, 0.0)).sum(dim=1)
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
    deck_l = read_deck_csv_file(str(C.resolve_deck(run["learner_deck"])))
    deck_o = read_deck_csv_file(str(C.resolve_deck(run["opponent_deck"])))
    opp = C.resolve_opponent_weights(run_dir, run["opponents"][0].get("weights"))
    _, w, v, _ = parallel_collect(str(weights), opp, deck_l, deck_o,
                                  games, C.EVAL_SEED_BASE, temperature=0.01, workers=workers)
    return (w / v if v else float("nan")), w, v


# ------------------------------------------------------------------ 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--status", action="store_true", help="集計せず、揃い具合を表示して終了")
    ap.add_argument("--allow-partial", action="store_true",
                    help="未提出の worker があっても更新する(収集量が世代間で変わる点に注意)")
    ap.add_argument("--eval-games", type=int, default=0,
                    help="新モデルの評価試合数。0 なら評価しない")
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
    print(f"  決定点 {batch['n']} 件(最大選択肢 {batch['max_n']})", flush=True)

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
        critic.load_state_dict(st["critic"])
        opt_p.load_state_dict(st["opt_policy"])
        opt_v.load_state_dict(st["opt_value"])
        print(f"  学習状態を引き継ぎ: {state_path.name}", flush=True)
    elif gen > 0:
        print(f"  警告: {state_path.name} が無い。critic を初期化して続行する"
              "(この世代の更新は品質が落ちる)。", flush=True)

    with torch.no_grad():
        values = critic(batch["state_rows"])
    adv, vtarget = compute_gae(batch["lengths"], batch["rewards"], values,
                               ppo["gamma"], ppo["lam"], device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    mb = int(ppo.get("minibatch_size", 16384))
    n = batch["n"]
    order_gen = torch.Generator().manual_seed(gen)
    pl = vl = en = 0.0
    t0 = time.time()
    for _ in range(ppo["epochs"]):
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
    print(f"  PPO更新 {time.time()-t0:.1f}s  pol {pl:.4f} val {vl:.4f} ent {en:.3f}", flush=True)

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
    next_model.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    C.trainer_state_path(run_dir, gen + 1).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"critic": critic.state_dict(),
                "opt_policy": opt_p.state_dict(),
                "opt_value": opt_v.state_dict()},
               C.trainer_state_path(run_dir, gen + 1))

    record = {
        "generation": gen, "next_generation": gen + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workers": sorted(m["worker_id"] for _, m in shards),
        "games": total_games, "wins": total_wins,
        "collect_winrate": total_wins / max(total_games, 1),
        "steps": batch["n"],
        "pol_loss": pl, "val_loss": vl, "entropy": en,
        "rejected": [f"{p.name}: {why}" for p, why in rejected],
        "partial": bool(missing),
    }

    if args.eval_games > 0:
        ew = args.eval_workers or (os.cpu_count() or 2)
        wr, w, v = evaluate(run, run_dir, next_model, args.eval_games, ew)
        record.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
        print(f"  評価 v{gen+1}: {w}/{v} = {wr:.3f} (CI下限 {wilson_lo(w, v):.3f})", flush=True)

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
