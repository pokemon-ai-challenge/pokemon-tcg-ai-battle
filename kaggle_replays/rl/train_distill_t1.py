"""D2/D2.1: T1専用の蒸留trainer。

複数shard読み込み -> dataset manifest固定 -> metadata整合性検査 ->
試合単位train/val分割(dataset manifest hashに紐づけて固定・共有) ->
可変長board/optionのbatch化 -> paddingを除外したKL loss(T^2適用) ->
gradient clipping -> deployment(T=1)KLでbest checkpoint保存 -> 各種一致率の記録 -> resume。

    python train_distill_t1.py --shards a.npz b.npz --teacher-weights teacher.json \
        --out-dir runs/distill_v1 --temperature 1.0 --steps 500 --eval-every 50

温度違い(T=1 vs T=2)を比較する場合は、``--split-dir``を揃えたまま``--out-dir``だけ
変えて2回実行する(splitはdataset manifest hashに紐づくため、out-dirが違っても
同じshard集合・同じseedなら自動的に同じsplitを再利用する)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import token_batch as tb
import token_policy_t1 as t1
import token_shard as ts
from ptcg_ai.learning import card_vocab
from ptcg_ai.learning import legacy_feature_manifest as manifest

SPLIT_SCHEMA_VERSION = 1


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


def masked_kl_loss(scores: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """標準的な知識蒸留と同じ定義: KL(softmax(teacher_logits/T) || softmax(scores/T)) を
    決定点ごとに合法手方向へ合計し、そのあとバッチ内の決定点で平均する。**T^2を掛ける**
    (Hinton et al. 2015の慣行。T>1だと勾配が小さくなるのをT^2で補正する)。

    reduction順序(合法手方向で合計 → 決定点方向で平均)により、決定点あたりの
    合法手数(paddingを除いた実数)が違っても、各決定点はバッチ平均に等しい重みで
    寄与する(``test_reduction_weighting_independent_of_option_count``で確認)。

    paddingは**softmax/log_softmaxを計算する前に-infで潰す**(maskしたcontributionを
    後から0にするだけでは不十分。log_softmaxは行全体を正規化するため、paddingに
    でたらめな値が入っていると、paddingの寄与自体を0にしても実選択肢側の
    log_probsの値(正規化定数)が変わってしまい、決定点の重みがpaddingの中身に
    依存してしまう。``test_reduction_weighting_independent_of_option_count``で
    このバグを検出・回帰させないようにしている)。
    """
    neg_inf = torch.finfo(scores.dtype).min
    m_scores = scores.masked_fill(~mask, neg_inf)
    m_teacher_logits = teacher_logits.masked_fill(~mask, neg_inf)
    teacher_probs = torch.softmax(m_teacher_logits / temperature, dim=1)
    log_probs = torch.log_softmax(m_scores / temperature, dim=1)
    log_teacher = torch.log(teacher_probs.clamp_min(1e-12))
    contrib = teacher_probs * (log_teacher - log_probs)
    contrib = torch.where(mask, contrib, torch.zeros_like(contrib))
    per_decision_kl = contrib.sum(dim=1)  # 合法手方向の合計(決定点1件ぶん)
    return per_decision_kl.mean() * (temperature ** 2)  # 決定点方向の平均 × T^2


def compute_distill_metrics(scores: torch.Tensor, teacher_logits: torch.Tensor,
                            mask: torch.Tensor) -> dict:
    """top-1一致率・教師高確信度局面での一致率・教師entropy別一致率・
    student選択行動への教師確率。**常に共通T=1で計算する**(学習温度に関わらず、
    「実際にargmaxで手を選んだときの一致度」を測る指標のため)。"""
    neg_inf = torch.finfo(scores.dtype).min
    m_scores = scores.masked_fill(~mask, neg_inf)
    m_teacher_logits = teacher_logits.masked_fill(~mask, neg_inf)
    teacher_probs = torch.softmax(m_teacher_logits, dim=1)  # T=1固定

    student_argmax = m_scores.argmax(dim=1)
    teacher_argmax = m_teacher_logits.argmax(dim=1)
    top1_agree = (student_argmax == teacher_argmax).float()

    teacher_maxprob = teacher_probs.max(dim=1).values
    entropy = -(teacher_probs.clamp_min(1e-12) * teacher_probs.clamp_min(1e-12).log()).sum(dim=1)
    student_prob_under_teacher = teacher_probs.gather(1, student_argmax.unsqueeze(1)).squeeze(1)

    high_conf = teacher_maxprob >= 0.8
    out = {
        "top1_agreement": top1_agree.mean().item(),
        "top1_agreement_high_teacher_conf": (
            top1_agree[high_conf].mean().item() if high_conf.any() else float("nan")),
        "high_conf_fraction": high_conf.float().mean().item(),
        "student_prob_under_teacher_mean": student_prob_under_teacher.mean().item(),
    }
    if entropy.numel() >= 3:
        q = torch.quantile(entropy, torch.tensor([1 / 3, 2 / 3], device=entropy.device))
        low_mask = entropy <= q[0]
        mid_mask = (entropy > q[0]) & (entropy <= q[1])
        high_mask = entropy > q[1]
        for name, bmask in (("low", low_mask), ("mid", mid_mask), ("high", high_mask)):
            out[f"top1_agreement_entropy_{name}"] = (
                top1_agree[bmask].mean().item() if bmask.any() else float("nan"))
    return out


def build_model_config(profile: str | None, vocab_size: int) -> dict:
    return dict(global_dim=manifest.global_dim(profile), option_dim=65,
               board_vocab_size=vocab_size, num_zones=9)


def make_eval_batches(arrays: dict, decision_idx: np.ndarray, vocab, batch_size: int):
    """評価用: decision_idx を先頭から順にbatch_sizeごとに切るだけ(シャッフルしない)。"""
    for i in range(0, len(decision_idx), batch_size):
        chunk = decision_idx[i:i + batch_size]
        sub = tb.slice_arrays_by_decisions(arrays, chunk)
        yield tb.build_batch(sub, vocab)


def training_batch_stream(arrays: dict, train_idx: np.ndarray, vocab, batch_size: int, seed: int):
    """**resumeで厳密に再現可能な**無限バッチ列(D2.1補修)。

    1エポック(train_idxを1周)ごとに``seed + epoch``でシャッフルし直す。この関数は
    ``seed``/``train_idx``/``batch_size``だけで完全に決まる無限列なので、``step``個
    ぶん``next()``で読み飛ばしてから使えば、``step``ステップ目からの継続を厳密に
    再現できる(``resume``はこの「読み飛ばし」だけで実現する。旧実装は
    ``step``自体をシャッフルseedにしていたため、resume直後にepochの境目で
    シャッフル順が変わってしまい、中断なし学習と一致しなかった
    ``test_resume_bitwise_matches_straight_through_on_cpu``で検出)。
    """
    epoch = 0
    while True:
        order = train_idx.copy()
        np.random.RandomState(seed + epoch).shuffle(order)
        for i in range(0, len(order), batch_size):
            chunk = order[i:i + batch_size]
            sub = tb.slice_arrays_by_decisions(arrays, chunk)
            yield tb.build_batch(sub, vocab)
        epoch += 1


# ---------------------------------------------------------------------------
# dataset manifest / split(D2.1): out-dirではなくmanifest hashに紐づけて共有する
# ---------------------------------------------------------------------------

def _split_path(split_dir: Path, dataset_manifest_hash: str, seed: int, val_fraction: float) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    name = f"split_{dataset_manifest_hash[:16]}_seed{seed}_val{val_fraction}.json"
    return split_dir / name


def load_or_create_split(split_dir: Path, dataset_manifest: dict, lengths: np.ndarray,
                         seed: int, val_fraction: float) -> dict:
    path = _split_path(split_dir, dataset_manifest["manifest_hash"], seed, val_fraction)
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        if split["dataset_manifest_hash"] != dataset_manifest["manifest_hash"]:
            raise ValueError(
                f"splitファイル({path})のdataset_manifest_hashが現在のデータセットと"
                f"一致しない: {split['dataset_manifest_hash']} != {dataset_manifest['manifest_hash']}\n"
                "  shard集合・順序・中身が変わった状態で古いsplitを読もうとしている。")
        return split

    train_games, val_games = tb.train_val_game_split(len(lengths), val_fraction, seed)
    split = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_manifest_hash": dataset_manifest["manifest_hash"],
        "seed": seed, "val_fraction": val_fraction,
        "train_game_ids": train_games.tolist(), "val_game_ids": val_games.tolist(),
    }
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--teacher-weights", required=True,
                    help="正規化統計(teacher_stats時)・feature_profile確認用の教師重み")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-dir", default=None,
                    help="split.jsonの保存先。既定は--out-dirの親の'splits'"
                        "(out-dirをまたいでsplitを共有するため、既定でも--out-dir配下にはしない)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--normalization-mode", default="teacher_stats",
                    choices=list(tb.NORMALIZATION_MODES))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="モデル初期化・学習時シャッフルのseed")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir) if args.split_dir else out_dir.parent / "splits"
    log_path = out_dir / "train_log.jsonl"
    best_ckpt_path = out_dir / "best.pt"
    trainer_state_path = out_dir / "trainer_state.pt"

    set_all_seeds(args.seed)

    dataset_manifest = ts.build_dataset_manifest(args.shards)
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    arrays, shard_meta = ts.concatenate_shards(args.shards)
    profile = shard_meta.get("extended_features_profile")
    vocab = card_vocab.load_vocab()
    teacher_payload = json.loads(Path(args.teacher_weights).read_text(encoding="utf-8"))
    teacher_std = teacher_payload["standardization"]

    split = load_or_create_split(split_dir, dataset_manifest, arrays["lengths"],
                                 args.split_seed, args.val_fraction)
    train_idx = tb.expand_game_ids_to_decision_indices(arrays["lengths"], split["train_game_ids"])
    val_idx = tb.expand_game_ids_to_decision_indices(arrays["lengths"], split["val_game_ids"])
    print(f"dataset_manifest_hash={dataset_manifest['manifest_hash'][:16]}...  "
         f"train試合={len(split['train_game_ids'])}(決定点{len(train_idx)})  "
         f"val試合={len(split['val_game_ids'])}(決定点{len(val_idx)})")

    stats = tb.build_normalization_stats(args.normalization_mode, arrays, train_idx, profile,
                                         teacher_standardization=teacher_std)
    model_config = build_model_config(profile, vocab.size)
    manifest_hash = manifest.manifest_hash(profile)

    model = t1.T1OptionPolicy(**model_config).to(args.device)
    with torch.no_grad():
        model.board_numeric_mean.copy_(torch.tensor(stats["board_numeric_mean"]))
        model.board_numeric_std.copy_(torch.tensor(stats["board_numeric_std"]))
        model.global_mean.copy_(torch.tensor(stats["global_mean"]))
        model.global_std.copy_(torch.tensor(stats["global_std"]))
        model.option_mean.copy_(torch.tensor(stats["option_mean"]))
        model.option_std.copy_(torch.tensor(stats["option_std"]))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # resumeで検査する「今回の実行条件」(D2.1: 検査必須。省略できない)。
    # optimizer_config は opt.defaults から取る(betas/eps/weight_decay等の
    # 実効値そのもの。ハードコードした定数を別途持たない)。
    run_contract = {
        "dataset_manifest_hash": dataset_manifest["manifest_hash"],
        "split_hash": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        "model_config": model_config,
        "token_schema_version": shard_meta["token_shard_format"],
        "vocabulary_hash": vocab.hash, "card_vocab_size": vocab.size,
        "feature_profile": profile, "global_feature_manifest_hash": manifest_hash,
        "normalization_mode": args.normalization_mode,
        "temperature": args.temperature, "lr": args.lr, "grad_clip": args.grad_clip,
        "training_seed": args.seed,
        "optimizer_type": type(opt).__name__,
        "optimizer_config": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in opt.defaults.items()},
    }

    start_step = 0
    best_deployment_kl = float("inf")
    if args.resume and trainer_state_path.exists():
        # map_locationは常に"cpu"で読む。RNG state(torch.get_rng_state()のCPU側
        # ByteTensor・torch.cuda.get_rng_state_all()の各GPU分)はmap_locationで
        # 別デバイスへ移されると壊れる(torch.set_rng_stateはCPU tensorを要求する)。
        # モデル/optimizerのtensorはload_state_dict側が既存の(args.deviceへ.to済みの)
        # パラメータへ合わせてコピーするため、cpuで読んでも問題ない。
        state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        mismatches = [k for k in run_contract if state["run_contract"].get(k) != run_contract[k]]
        if mismatches:
            raise ValueError(
                f"resume条件が前回の実行と一致しない: {mismatches}\n"
                f"  前回: { {k: state['run_contract'].get(k) for k in mismatches} }\n"
                f"  今回: { {k: run_contract[k] for k in mismatches} }")
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        set_rng_state(state["rng_state"])
        start_step = state["step"]
        best_deployment_kl = state["best_deployment_kl"]
        print(f"resume: step={start_step} best_deployment_kl={best_deployment_kl:.4f}")

    def _evaluate(decision_idx) -> dict:
        model.eval()
        total_obj_kl, total_dep_kl, total_n = 0.0, 0.0, 0
        agg = {}
        with torch.no_grad():
            for batch in make_eval_batches(arrays, decision_idx, vocab, args.batch_size):
                inputs, teacher_logits = _to_tensors(batch, args.device)
                scores = model(**inputs)
                obj_kl = masked_kl_loss(scores, teacher_logits, inputs["option_mask"], args.temperature)
                dep_kl = masked_kl_loss(scores, teacher_logits, inputs["option_mask"], 1.0)
                n = inputs["option_mask"].shape[0]
                total_obj_kl += obj_kl.item() * n
                total_dep_kl += dep_kl.item() * n
                total_n += n
                m = compute_distill_metrics(scores, teacher_logits, inputs["option_mask"])
                for k, v in m.items():
                    agg.setdefault(k, []).append(v)
        model.train()
        result = {"train_objective_kl": total_obj_kl / max(total_n, 1),
                  "deployment_kl_t1": total_dep_kl / max(total_n, 1)}
        for k, vs in agg.items():
            vs = [v for v in vs if v == v]
            result[k] = float(np.mean(vs)) if vs else float("nan")
        return result

    model.train()
    step = start_step
    t0 = time.time()
    # training_batch_streamは(train_idx, args.seed, batch_size)だけで決まる無限列。
    # resume時は同じ列を再構築し、既に消費済みのstart_step個ぶんを読み飛ばすだけで
    # 継続を厳密に再現する(バッチ構築コストの再計算は発生するが、モデルの
    # forward/backwardはやり直さないので安価)。
    stream = training_batch_stream(arrays, train_idx, vocab, args.batch_size, args.seed)
    for _ in range(start_step):
        next(stream)

    with log_path.open("a", encoding="utf-8") as log_fh:
        while step < args.steps:
            batch = next(stream)
            inputs, teacher_logits = _to_tensors(batch, args.device)
            opt.zero_grad()
            scores = model(**inputs)
            loss = masked_kl_loss(scores, teacher_logits, inputs["option_mask"], args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                val_metrics = _evaluate(val_idx)
                record = {"step": step, "train_loss": loss.item(),
                         "elapsed_sec": round(time.time() - t0, 1), **val_metrics}
                log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_fh.flush()
                print(f"step={step} train_loss={loss.item():.4f} "
                     f"deployment_kl_t1={val_metrics['deployment_kl_t1']:.4f} "
                     f"top1={val_metrics['top1_agreement']:.3f}")

                # モデル選択(best checkpoint)は常にdeployment_kl_t1で行う
                # (train_objective_klは学習温度依存で、T=1/T=2間の比較に使えないため)。
                if val_metrics["deployment_kl_t1"] < best_deployment_kl:
                    best_deployment_kl = val_metrics["deployment_kl_t1"]
                    t1.save_checkpoint(
                        model, best_ckpt_path, model_config,
                        token_schema_version=shard_meta["token_shard_format"],
                        vocabulary_version=vocab.version, vocabulary_hash=vocab.hash,
                        feature_profile=profile, global_feature_manifest_hash=manifest_hash,
                        normalization_mode=args.normalization_mode, normalization_stats=stats,
                        card_vocab_size=vocab.size)

                torch.save({"model_state_dict": model.state_dict(),
                           "optimizer_state_dict": opt.state_dict(),
                           "rng_state": get_rng_state(),
                           "run_contract": run_contract,
                           "step": step, "best_deployment_kl": best_deployment_kl}, trainer_state_path)

    print(f"完了: step={step} best_deployment_kl_t1={best_deployment_kl:.4f}  -> {best_ckpt_path}")


if __name__ == "__main__":
    main()
