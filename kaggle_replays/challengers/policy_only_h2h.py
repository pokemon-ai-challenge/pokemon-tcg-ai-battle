"""ISMCTS v2.5 Phase I — Policy-only H2H(Search 完全に外す、唯一差分=Policy weights)。

Primary transfer test: Search-Distilled Policy 単体 が Original Policy 単体 より強いか(Search による補正なし)。
両者 config=abl_2_policy_only(lethal OFF / pipeline OFF = 純 Policy argmax)、deck 一致、weights のみ差し替え。
driver.run_sprt_ab 再利用。__main__ ガード必須。

使用: python policy_only_h2h.py --games 300 --workers 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for _p in (str(_SUB), str(_ROOT / "kaggle_replays" / "measurement")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import driver  # noqa: E402

_DEFAULT = _ROOT / "kaggle_replays" / "training" / "search_distilled_policy.json"


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--weights", default=str(_DEFAULT), help="candidate policy weights JSON(control=Original 既定)")
    ap.add_argument("--label", default="distilled_policy")
    ap.add_argument("--run-dir", default=str(_HERE / "results" / "policy_only"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cand_w = Path(args.weights).resolve()          # ★ 絶対パス化(spawn worker は cwd=sample_submission ゆえ相対は不可)
    if not cand_w.exists():
        print(f"[error] candidate policy not found: {cand_w}"); return 1
    # Phase I2 preflight(v2.6 path bug 再発防止): 候補が実ロード可能(is_ready)かを検証、index-0 fallback を禁止。
    import hashlib
    from ptcg_ai.learning.policy_model import PolicyModel
    _cand_hash = hashlib.sha1(cand_w.read_bytes()).hexdigest()[:12]
    _ctrl_w = (_SUB / "ptcg_ai" / "learning" / "policy_weights.json").resolve()
    _ctrl_hash = hashlib.sha1(_ctrl_w.read_bytes()).hexdigest()[:12]
    if not PolicyModel(weights_path=cand_w).is_ready:
        print(f"[error] candidate PolicyModel not ready (would fall back to index-0): {cand_w}"); return 1
    print(f"    [preflight] candidate={cand_w.name} sha1={_cand_hash} | control=policy_weights.json sha1={_ctrl_hash}")
    print(f"    [preflight] candidate!=control: {_cand_hash != _ctrl_hash}  (worker cwd=sample_submission, abs paths)")
    deck = _load_deck()
    # 唯一差分 = policy_weights_path。config は共通(abl_2_policy_only = lethal/pipeline OFF)。
    cand = driver.AgentSpec("abl_2_policy_only", policy_weights_path=str(cand_w), label=args.label)
    ctrl = driver.AgentSpec("abl_2_policy_only", label="original_policy")
    print(f"=== Policy-only H2H: {args.label} vs Original (no search) games={args.games} workers={args.workers} ===")
    print(f"    candidate weights = {cand_w.name}")
    t0 = time.perf_counter()
    rep = driver.run_sprt_ab(cand, ctrl, deck, delta_min=0.05, alpha=0.05, beta=0.10,
                             n_max=10_000_000, max_games=args.games, alternate_sides=True,
                             run_dir=Path(args.run_dir) / f"{args.label}_vs_original", resume=False,
                             workers=args.workers, progress_every=25, note="distilled policy-only vs original policy-only")
    r = rep["result"]; el = time.perf_counter() - t0
    print(f"\n=== RESULT ===")
    print(f"  distilled vs original: {r['candidate_wins']}/{rep['sprt']['n']} "
          f"wr={r['candidate_winrate']:.4f} CI{r['wilson95_ci']} dec={rep['decision']} "
          f"err={r['cand_errors']}/{r['ctrl_errors']} [{el:.0f}s]")
    print("  >0.50 = Search knowledge が Policy weight へ転移(Primary success)。~0.50 = 転移せず。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
