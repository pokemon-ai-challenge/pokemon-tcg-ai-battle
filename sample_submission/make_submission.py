"""Build submission.tar.gz with exactly the files the Kaggle agent needs.

Included (numpy + cg only — no torch/gymnasium):
    main.py, deck.csv, policy.npz, cg/, tcg_rl/{__init__,features,mlp_numpy}.py

Training-only modules (env.py, opponents.py, mc_agent.py, train_ppo.py,
export_policy.py) are intentionally excluded so the submission stays small and
free of torch/gym imports.

Run from the sample_submission folder:
    python make_submission.py
"""

from __future__ import annotations

import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "main.py",
    "deck.csv",
    "policy.npz",
    "tcg_rl/__init__.py",
    "tcg_rl/features.py",
    "tcg_rl/mlp_numpy.py",
]
DIRS = ["cg"]  # whole engine folder (cg.dll / libcg.so / *.py)

# cg files we don't need in the archive
CG_SKIP_SUFFIXES = (".pyc",)


def _add_file(tar: tarfile.TarFile, rel: str) -> None:
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"required file missing: {rel}")
    tar.add(path, arcname=rel)


def build(out: str = "submission.tar.gz") -> None:
    out_path = os.path.join(HERE, out)
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in FILES:
            _add_file(tar, rel)
        for d in DIRS:
            base = os.path.join(HERE, d)
            for root, _, files in os.walk(base):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith(CG_SKIP_SUFFIXES):
                        continue
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, HERE)
                    tar.add(full, arcname=rel.replace(os.sep, "/"))

    # report
    with tarfile.open(out_path, "r:gz") as tar:
        names = tar.getnames()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"built {out} ({size_mb:.1f} MB), {len(names)} entries")
    assert "main.py" in names, "main.py must be at archive root"
    assert "policy.npz" in names, "policy.npz missing (train + export first)"
    for must in ("tcg_rl/features.py", "tcg_rl/mlp_numpy.py", "deck.csv"):
        assert must in names, f"{must} missing"
    print("OK: required files present at archive root")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(HERE, "policy.npz")):
        print("policy.npz not found — run train_ppo.py first.", file=sys.stderr)
        sys.exit(1)
    build()
