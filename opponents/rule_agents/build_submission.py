"""rule_agents/ の手書きルールベース1種を、Kaggle に出せる tar.gz に固める。

sample_submission/ の本提出物(ML方策)とは別に、「このデッキのルールベース単体で
提出したらどうなるか」を試したいときに使う。torch/numpy 等の重みは無く、選んだ
デッキ1種の agent + framework + cg エンジンだけを詰める、小さく自己完結したバンドル。

## バンドルの中身

```
main.py            Kaggle エントリポイント(__file__ 非依存、失敗時は合法手にフォールバック)
deck.csv           選んだデッキの60枚(main.py の read_deck_csv 用)
cg/                sample_submission/cg のコピー(変更しない)
rule_agents/        framework.py + 選んだデッキの1ファイル + そのデッキCSV
```

デッキ間の依存が無いことは各 `<deck>.py` の import を見れば分かる(`from . import framework`
だけで、他の `grimmsnarl.py`/`lucario.py`/`archaludon.py` 同士は互いに依存しない)ため、
選んだ1ファイルだけをバンドルすれば動く。

## 使い方

```bash
cd opponents/rule_agents
python3 build_submission.py grimmsnarl --out submissions/grimmsnarl_rule.tar.gz
python3 build_submission.py lucario --out submissions/lucario_rule.tar.gz
python3 build_submission.py archaludon --out submissions/archaludon_rule.tar.gz
```

## 検証について

このスクリプトはファイルを固めるだけで、動作検証はしない。提出前に必ず
`verify_submission.py <tar.gz>` で
「`__file__` 無し・cwd=展開先」という Kaggle 相当の条件で複数試合を回し、
違法選択が出ないこと・フォールバックへ落ちていないことを確認すること
(過去に、テストスクリプト側の sys.path 汚染で `rule_agents` の import が
静かに失敗し、対局中ずっと保険用のダミー行動を返し続けていた事故があった)。
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SAMPLE_SUBMISSION = _HERE.parent.parent / "sample_submission"

DECKS = {
    "grimmsnarl": ("grimmsnarl.py", "marnie_grimmsnarl_ex.csv"),
    "lucario": ("lucario.py", "mega_lucario_ex.csv"),
    "archaludon": ("archaludon.py", "archaludon_ex.csv"),
}

_CG_FILES = ["__init__.py", "api.py", "game.py", "sim.py", "utils.py", "cg.dll", "libcg.so"]

_MAIN_PY = '''"""Kaggle submission entry point -- hand-written rule-based agent ({name} deck).

Pure Python, no torch/numpy dependency. Decision logic lives in rule_agents/{module},
scored against every legal option each call (see that module's docstring for the
deck's game plan). This file only handles Kaggle's runtime quirks and provides a
fallback so the agent never crashes out of a match.
"""

import os
import sys

# Make the submission folder importable regardless of the working directory.
# Kaggle runs main.py via exec() (no __file__), so __file__ can't be relied on.
# Register every plausible candidate directory instead of a single hardcoded one:
# the known Kaggle agent path, the current working directory (this is how
# sample_submission/main.py itself does it), and __file__'s directory when it
# happens to be available (e.g. when testing this file outside of exec()).
_CANDIDATE_DIRS = ["/kaggle_simulations/agent", os.getcwd()]
try:
    _CANDIDATE_DIRS.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
for _path in _CANDIDATE_DIRS:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import Observation, to_observation_class  # noqa: E402

_AGENT = None
_AGENT_LOAD_TRIED = False


def _candidate_paths(filename: str):
    return [os.path.join(d, filename) for d in _CANDIDATE_DIRS] + [filename]


def read_deck_csv() -> list[int]:
    """Read deck.csv and return a list of 60 card IDs."""
    for path in _candidate_paths("deck.csv"):
        if os.path.exists(path):
            with open(path, "r") as f:
                rows = [r for r in f.read().split("\\n") if r.strip()]
            return [int(rows[i]) for i in range(60)]
    raise FileNotFoundError("deck.csv not found")


def _get_agent():
    """Lazily import the rule-based agent; return None if unavailable (triggers fallback)."""
    global _AGENT, _AGENT_LOAD_TRIED
    if _AGENT_LOAD_TRIED:
        return _AGENT
    _AGENT_LOAD_TRIED = True
    try:
        from rule_agents.{package} import agent as _rule_agent
        _AGENT = _rule_agent
    except Exception:
        _AGENT = None
    return _AGENT


def _fallback_action(obs: Observation) -> list[int]:
    """A guaranteed-legal action used when the rule agent is unavailable or errors."""
    sel = obs.select
    if sel is None:
        return read_deck_csv()
    n = len(sel.option)
    if sel.minCount <= 0:
        return []
    return list(range(min(sel.minCount, n)))


def agent(obs_dict: dict) -> list[int]:
    """Competition entry point. Returns option indices (or the deck on turn 0)."""
    obs: Observation = to_observation_class(obs_dict)

    if obs.select is None:
        return read_deck_csv()

    try:
        rule_agent = _get_agent()
        if rule_agent is None:
            return _fallback_action(obs)
        action = rule_agent(obs)
        # final legality guard
        n = len(obs.select.option)
        action = [i for i in action if 0 <= i < n]
        if len(set(action)) != len(action):
            return _fallback_action(obs)
        if not (obs.select.minCount <= len(action) <= obs.select.maxCount):
            return _fallback_action(obs)
        return action
    except Exception:
        return _fallback_action(obs)
'''


def build(deck_name: str, out_path: Path) -> None:
    if deck_name not in DECKS:
        raise ValueError(f"unknown deck: {deck_name!r} (choices: {sorted(DECKS)})")
    module_file, deck_csv = DECKS[deck_name]
    module_name = module_file.removesuffix(".py")

    work = out_path.parent / f"_build_{deck_name}"
    if work.exists():
        shutil.rmtree(work)
    (work / "cg").mkdir(parents=True)
    (work / "rule_agents" / "decks").mkdir(parents=True)

    for f in _CG_FILES:
        shutil.copy(_SAMPLE_SUBMISSION / "cg" / f, work / "cg" / f)

    shutil.copy(_HERE / "__init__.py", work / "rule_agents" / "__init__.py")
    shutil.copy(_HERE / "framework.py", work / "rule_agents" / "framework.py")
    shutil.copy(_HERE / module_file, work / "rule_agents" / module_file)
    shutil.copy(_HERE / "decks" / deck_csv, work / "rule_agents" / "decks" / deck_csv)
    shutil.copy(_HERE / "decks" / deck_csv, work / "deck.csv")

    main_py = _MAIN_PY.format(name=deck_name, module=module_file, package=module_name)
    (work / "main.py").write_text(main_py, encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for item in sorted(work.rglob("*")):
            if item.is_file():
                tar.add(item, arcname=str(item.relative_to(work)))

    shutil.rmtree(work)
    print(f"built {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", choices=sorted(DECKS))
    ap.add_argument("--out", type=Path, default=None,
                    help="出力先(default: submissions/<deck>_rule_submission.tar.gz)")
    args = ap.parse_args()
    out = args.out or (_HERE / "submissions" / f"{args.deck}_rule_submission.tar.gz")
    build(args.deck, out)


if __name__ == "__main__":
    main()
