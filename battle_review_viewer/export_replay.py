"""Play one local match and save a god-view replay JSON for the review viewer.

The replay is the engine's own god-view frame list (visualize_data()): every
decision is one frame with {current, select, logs, selected}. We wrap it with a
small meta block and write it under replays/.

`play()` takes two agent callables (obs_dict -> list[int]) so both the CLI here
and the viewer server (serve_viewer.py /api/run) can drive any agents/decks.

CLI:
    python battle_review_viewer/export_replay.py --opponent random
    python battle_review_viewer/export_replay.py --opponent self --seed 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SAMPLE = _REPO / "sample_submission"
if str(_SAMPLE) not in sys.path:
    sys.path.insert(0, str(_SAMPLE))

from cg.game import battle_start, battle_finish, battle_select, visualize_data  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
import main as submission  # noqa: E402  (sample_submission/main.py)

REPLAYS_DIR = _HERE / "replays"


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    sel = obs.select
    if sel is None:
        return []
    n = len(sel.option)
    lo, hi = sel.minCount, min(sel.maxCount, n)
    if hi < lo:
        return []
    k = random.randint(lo, hi)
    return random.sample(range(n), k) if k > 0 else []


def play(deck0: list[int], deck1: list[int], agent0, agent1, seed: int | None = None):
    """Play deck0(agent0) vs deck1(agent1); return (frames, result)."""
    if seed is not None:
        random.seed(seed)
    obs_dict, start = battle_start(deck0, deck1)
    if start.errorType != 0:
        raise RuntimeError(f"battle_start failed (errorType={start.errorType})")
    result = -1
    try:
        steps = 0
        while steps < 5000:
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                result = obs.current.result
                break
            actor = obs.current.yourIndex if obs.current is not None else 0
            action = (agent0 if actor == 0 else agent1)(obs_dict)
            obs_dict = battle_select(action)
            steps += 1
        frames = json.loads(visualize_data())
    finally:
        battle_finish()
    return frames, result


def save_replay(frames, result, meta_extra=None, out_path=None, tag="match") -> Path:
    meta = {
        "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "frameCount": len(frames),
    }
    if meta_extra:
        meta.update(meta_extra)
    payload = {"meta": meta, "frames": frames}
    if out_path:
        out = Path(out_path)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = REPLAYS_DIR / f"replay-{stamp}-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=["random", "self"], default="random")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    deck = submission.read_deck_csv()
    agent1 = submission.agent if args.opponent == "self" else random_agent
    frames, result = play(deck, deck, submission.agent, agent1, seed)

    out = save_replay(
        frames, result,
        meta_extra={"opponent": args.opponent, "seed": seed,
                    "p0": "mppo", "p1": args.opponent, "deck0": "default", "deck1": "default"},
        out_path=args.output, tag=args.opponent,
    )
    print(f"saved {out}  (frames={len(frames)}, result={result})")


if __name__ == "__main__":
    main()
