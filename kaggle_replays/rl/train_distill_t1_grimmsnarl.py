"""オーロンゲ(marnie_grimmsnarl_ex)T1の蒸留(ルールベース教師、BC championは混ぜない)。

``train_distill_t1.py``(フーディン用)は変更しない。同じ損失関数
(``masked_kl_loss``)・同じモデル構成(``build_model_config``)・同じバッチ化
(``training_batch_stream``/``make_eval_batches``)を再利用しつつ、
**試合単位でtrain/validation/testの3分割**にする点だけが違う(フーディンは
train/valの2分割)。教師のstandardization統計は無い(ルールベースのため)ので
``normalization_mode=student_train_stats``固定。

保存先はフーディン系(``runs/distill_v40/``)と完全に分離した
``runs/distill_grimmsnarl/``。既存のcheckpoint・optimizer state・正規化統計・
manifestには一切触れない。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import token_shard as ts  # noqa: E402
import token_batch as tb  # noqa: E402
import token_policy_t1 as t1  # noqa: E402
import train_distill_t1 as td  # noqa: E402

_FEATURE_PROFILE = "fuudin_v4"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    td.set_all_seeds(args.seed)

    shard_paths = [Path(p) for p in args.shards]
    dataset_manifest = ts.build_dataset_manifest(shard_paths)
    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, ensure_ascii=False, indent=1)

    arrays, shard_meta = ts.concatenate_shards(shard_paths)
    profile = shard_meta["extended_features_profile"]
    assert profile == _FEATURE_PROFILE, f"想定外のprofile: {profile}"

    from ptcg_ai.learning import card_vocab
    vocab = card_vocab.load_vocab()

    lengths = arrays["lengths"]
    n_games = len(lengths)
    train_games, val_games, test_games = tb.train_val_test_game_split(
        n_games, args.val_fraction, args.test_fraction, args.split_seed)
    split = {
        "schema_version": td.SPLIT_SCHEMA_VERSION, "dataset_manifest_hash": dataset_manifest["manifest_hash"],
        "split_seed": args.split_seed, "val_fraction": args.val_fraction, "test_fraction": args.test_fraction,
        "train_game_ids": train_games.tolist(), "val_game_ids": val_games.tolist(),
        "test_game_ids": test_games.tolist(),
    }
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)

    train_idx = tb.expand_game_ids_to_decision_indices(lengths, train_games)
    val_idx = tb.expand_game_ids_to_decision_indices(lengths, val_games)
    test_idx = tb.expand_game_ids_to_decision_indices(lengths, test_games)
    print(f"dataset_manifest_hash={dataset_manifest['manifest_hash'][:16]}...  "
         f"train試合={len(train_games)}(決定点{len(train_idx)})  "
         f"val試合={len(val_games)}(決定点{len(val_idx)})  "
         f"test試合={len(test_games)}(決定点{len(test_idx)})", flush=True)

    stats = tb.build_normalization_stats("student_train_stats", arrays, train_idx, profile)
    model_config = td.build_model_config(profile, vocab.size)
    manifest_hash = _manifest_hash(profile)

    model = t1.T1OptionPolicy(**model_config).to(args.device)
    with torch.no_grad():
        model.board_numeric_mean.copy_(torch.tensor(stats["board_numeric_mean"]))
        model.board_numeric_std.copy_(torch.tensor(stats["board_numeric_std"]))
        model.global_mean.copy_(torch.tensor(stats["global_mean"]))
        model.global_std.copy_(torch.tensor(stats["global_std"]))
        model.option_mean.copy_(torch.tensor(stats["option_mean"]))
        model.option_std.copy_(torch.tensor(stats["option_std"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stream = td.training_batch_stream(arrays, train_idx, vocab, args.batch_size, args.seed)
    best_val_kl = float("inf")
    log_path = out_dir / "train_log.jsonl"
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = next(stream)
        inputs, teacher_logits = _to_tensors(batch, args.device)
        model.train()
        scores = model(**inputs)
        loss = td.masked_kl_loss(scores, teacher_logits, inputs["option_mask"], 1.0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_metrics = _evaluate(model, vocab, arrays, val_idx, args.batch_size, args.device)
            entry = {"step": step, "train_loss": float(loss.item()), "elapsed_sec": time.time() - t0,
                    **val_metrics}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"step={step} train_loss={loss.item():.4f} val_kl={val_metrics['deployment_kl_t1']:.4f} "
                 f"top1={val_metrics['top1_agreement']:.4f}", flush=True)
            if val_metrics["deployment_kl_t1"] < best_val_kl:
                best_val_kl = val_metrics["deployment_kl_t1"]
                t1.save_checkpoint(
                    model, out_dir / "best.pt", model_config,
                    token_schema_version=ts.TOKEN_SHARD_FORMAT, vocabulary_version=vocab.version,
                    vocabulary_hash=vocab.hash, feature_profile=profile,
                    global_feature_manifest_hash=manifest_hash, normalization_mode="student_train_stats",
                    normalization_stats=stats, card_vocab_size=vocab.size)

    print(f"完了: best_val_kl={best_val_kl:.4f} -> {out_dir / 'best.pt'}", flush=True)

    # held-out testでの最終評価(checkpoint選択には使っていない、ここが唯一の使用箇所)
    best_model, _ = t1.load_checkpoint(
        out_dir / "best.pt", token_schema_version=ts.TOKEN_SHARD_FORMAT, vocabulary_version=vocab.version,
        vocabulary_hash=vocab.hash, card_vocab_size=vocab.size, feature_profile=profile,
        global_feature_manifest_hash=manifest_hash, map_location=args.device)
    best_model.to(args.device)
    test_metrics = _evaluate(best_model, vocab, arrays, test_idx, args.batch_size, args.device)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=1)
    print(f"test: kl={test_metrics['deployment_kl_t1']:.4f} top1={test_metrics['top1_agreement']:.4f} "
         f"top1_high_conf={test_metrics['top1_agreement_high_teacher_conf']:.4f}", flush=True)


def _manifest_hash(profile: str) -> str:
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    return manifest.manifest_hash(profile)


def _to_tensors(batch: dict, device: str):
    b, o = batch["board"], batch["option"]
    inputs = dict(
        board_numeric=torch.tensor(b["numeric"], dtype=torch.float32, device=device),
        board_card_idx=torch.tensor(b["card_idx"], dtype=torch.long, device=device),
        board_zone_ids=torch.tensor(b["zone_ids"], dtype=torch.long, device=device),
        board_owner_ids=torch.tensor(b["owner_ids"], dtype=torch.long, device=device),
        board_mask=torch.tensor(b["mask"], dtype=torch.bool, device=device),
        global_features=torch.tensor(batch["legacy_global_features"], dtype=torch.float32, device=device),
        option_features=torch.tensor(o["legacy_option_features"], dtype=torch.float32, device=device),
        option_card_idx=torch.tensor(o["card_idx"], dtype=torch.long, device=device),
        option_mask=torch.tensor(o["mask"], dtype=torch.bool, device=device),
    )
    teacher_logits = torch.tensor(o["teacher_logits"], dtype=torch.float32, device=device)
    return inputs, teacher_logits


def _evaluate(model, vocab, arrays, idx, batch_size, device) -> dict:
    import token_batch as tb_local
    all_metrics = []
    n_total = 0
    with torch.no_grad():
        for batch in td.make_eval_batches(arrays, idx, vocab, batch_size):
            inputs, teacher_logits = _to_tensors(batch, device)
            scores = model(**inputs)
            kl = td.masked_kl_loss(scores, teacher_logits, inputs["option_mask"], 1.0).item()
            m = td.compute_distill_metrics(scores, teacher_logits, inputs["option_mask"])
            n = inputs["option_mask"].shape[0]
            all_metrics.append((n, kl, m))
            n_total += n
    if n_total == 0:
        return {"deployment_kl_t1": float("nan"), "top1_agreement": float("nan"),
               "top1_agreement_high_teacher_conf": float("nan")}
    kl_w = sum(n * kl for n, kl, _ in all_metrics) / n_total
    out = {"deployment_kl_t1": kl_w}
    for key in all_metrics[0][2]:
        out[key] = sum(n * m[key] for n, _, m in all_metrics) / n_total
    return out


if __name__ == "__main__":
    main()
