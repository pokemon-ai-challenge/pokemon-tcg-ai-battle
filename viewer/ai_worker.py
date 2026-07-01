"""Subprocess worker that runs ONE external agent in isolation.

Why a subprocess: every dropped agent has its own ``main.py`` and may bundle its
own ``cg`` / ``tcg_rl`` / ``policy.npz``. Importing several of them in one Python
process would collide on those module names (and reload the native cg DLL). A
separate process per agent avoids all of that and mirrors how Kaggle runs agents.

Usage:
    python ai_worker.py <path>
  <path> = a directory containing ``main.py``  (submission-style, self-contained)
         | a single ``.py`` file defining ``agent(obs_dict) -> list[int]``

Protocol (line-delimited JSON):
    <- stdin : {"obs": <obs_dict>}\n      one request per decision; "" / EOF stops
    -> stdout: {"action": [ints]|null}\n  exactly one reply per request
First stdout line is {"ready": bool}. Any print()/logging from the loaded agent
is redirected to stderr so it can never corrupt the stdout protocol channel.
"""

import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SAMPLE = os.path.join(_REPO, "sample_submission")


def _load_agent(path: str):
    """Import the target and return its ``agent`` callable."""
    # Fallback deps (cg / tcg_rl) come from the repo submission folder, but only
    # if the agent does not bundle its own (its own dir is searched first).
    if _SAMPLE not in sys.path:
        sys.path.append(_SAMPLE)

    if os.path.isdir(path):
        sys.path.insert(0, path)
        import main as mod  # the agent's own main.py (dir is first on sys.path)
    else:
        d = os.path.dirname(path)
        if d not in sys.path:
            sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("ext_agent_main", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    if not hasattr(mod, "agent"):
        raise RuntimeError(f"{path} does not define agent(obs_dict)")
    return mod.agent


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("[ai_worker] missing <path> argument\n")
        sys.exit(2)
    target = sys.argv[1]

    # Redirect anything the agent prints to stderr; keep real stdout for protocol.
    real_out = sys.stdout
    sys.stdout = sys.stderr

    try:
        agent = _load_agent(target)
    except Exception as e:  # keep the process alive so the parent can fall back
        sys.stderr.write(f"[ai_worker] load failed for {target}: {e}\n")
        sys.stderr.flush()
        agent = None

    real_out.write(json.dumps({"ready": agent is not None}) + "\n")
    real_out.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            obs_dict = req.get("obs")
            action = agent(obs_dict) if agent is not None else None
            if not isinstance(action, list):
                action = None
        except Exception as e:
            sys.stderr.write(f"[ai_worker] decide error: {e}\n")
            sys.stderr.flush()
            action = None
        real_out.write(json.dumps({"action": action}, ensure_ascii=True) + "\n")
        real_out.flush()


if __name__ == "__main__":
    main()
