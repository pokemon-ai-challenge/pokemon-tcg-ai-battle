"""T1のGPUメモリ・推論/学習時間の実測スクリプト(T1.1補修版)。

前回計測(batch=256で推論のみが学習stepより遅い、batch=8192→16384で学習時間が
約17倍)が不自然だったため、warm-up・反復回数・同期タイミングを見直した。

    python measure_t1_perf.py --games 8 --batch-sizes 256,512,1024,2048,4096,8192,16384
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import collect_tokens as ct
import token_batch as tb
import token_policy_t1 as t1
from ptcg_ai.learning import card_vocab
from ptcg_ai.learning import legacy_feature_manifest as manifest
from run_league import read_deck_csv_file

_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v47.json"
_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"

WARMUP_ITERS = 10
MEASURE_ITERS = 30


def _repeat_to_batch(tensor: torch.Tensor, n: int) -> torch.Tensor:
    reps = -(-n // tensor.shape[0])
    return tensor.repeat(reps, *([1] * (tensor.dim() - 1)))[:n]


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _time_block(fn, device: str, n: int) -> list[float]:
    times = []
    for _ in range(n):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _report(label: str, times: list[float]) -> None:
    print(f"    {label:22s} mean={statistics.mean(times):8.2f}ms  "
          f"median={statistics.median(times):8.2f}ms  p95={_percentile(times, 0.95):8.2f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--batch-sizes", default="256,512,1024,2048,4096,8192,16384")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  warmup={WARMUP_ITERS}  measure_iters={MEASURE_ITERS}")

    deck = read_deck_csv_file(str(_DECK))
    trajs, wins, valid, errors = ct.parallel_collect_tokens(
        str(_TEACHER), deck, deck, n_games=args.games, seed0=7, temperature=1.0, workers=2)
    decisions = [step for tr in trajs for step in tr["steps"]]
    print(f"収集: {valid}試合 / 決定点 {len(decisions)}")

    import token_shard as ts
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "perf.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "perf", "generation": 0})
        arrays, meta = ts.read_token_shard(path)

    vocab = card_vocab.load_vocab()
    batch = tb.build_batch(arrays, vocab)
    print(f"1shardの決定点あたり: board_max={batch['board']['numeric'].shape[1]} "
          f"option_max={batch['option']['mask'].shape[1]}")

    model = t1.T1OptionPolicy(global_dim=manifest.global_dim("fuudin_v4"), option_dim=65,
                              board_vocab_size=vocab.size, num_zones=9).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    board_numeric = torch.tensor(batch["board"]["numeric"], dtype=torch.float32)
    board_card_idx = torch.tensor(batch["board"]["card_idx"], dtype=torch.long)
    board_zone_ids = torch.tensor(batch["board"]["zone_ids"], dtype=torch.long)
    board_owner_ids = torch.tensor(batch["board"]["owner_ids"], dtype=torch.long)
    board_mask = torch.tensor(batch["board"]["mask"], dtype=torch.bool)
    global_features = torch.tensor(batch["legacy_global_features"], dtype=torch.float32)
    option_features = torch.tensor(batch["option"]["legacy_option_features"], dtype=torch.float32)
    option_card_idx = torch.tensor(batch["option"]["card_idx"], dtype=torch.long)
    option_mask = torch.tensor(batch["option"]["mask"], dtype=torch.bool)

    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        inputs = dict(
            board_numeric=_repeat_to_batch(board_numeric, bs).to(device),
            board_card_idx=_repeat_to_batch(board_card_idx, bs).to(device),
            board_zone_ids=_repeat_to_batch(board_zone_ids, bs).to(device),
            board_owner_ids=_repeat_to_batch(board_owner_ids, bs).to(device),
            board_mask=_repeat_to_batch(board_mask, bs).to(device),
            global_features=_repeat_to_batch(global_features, bs).to(device),
            option_features=_repeat_to_batch(option_features, bs).to(device),
            option_card_idx=_repeat_to_batch(option_card_idx, bs).to(device),
            option_mask=_repeat_to_batch(option_mask, bs).to(device),
        )

        def _forward_only():
            with torch.no_grad():
                model(**inputs)

        def _forward_backward():
            opt.zero_grad(set_to_none=True)
            scores = model(**inputs)
            scores.masked_select(inputs["option_mask"]).sum().backward()

        def _forward_backward_step():
            opt.zero_grad(set_to_none=True)
            scores = model(**inputs)
            scores.masked_select(inputs["option_mask"]).sum().backward()
            opt.step()

        model.eval()
        for _ in range(WARMUP_ITERS):
            _forward_only()
        model.train()
        for _ in range(WARMUP_ITERS):
            _forward_backward_step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        print(f"  batch={bs}")
        model.eval()
        _report("forward_only", _time_block(_forward_only, device, MEASURE_ITERS))
        model.train()
        _report("forward+backward", _time_block(_forward_backward, device, MEASURE_ITERS))
        _report("forward+backward+step", _time_block(_forward_backward_step, device, MEASURE_ITERS))

        if device == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9
            print(f"    peak_allocated={peak_alloc:.3f}GB  peak_reserved={peak_reserved:.3f}GB")


if __name__ == "__main__":
    main()
