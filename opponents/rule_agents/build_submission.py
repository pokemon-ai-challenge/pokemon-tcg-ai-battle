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
import traceback

# Make the submission folder importable regardless of the working directory.
# Kaggle runs main.py via exec() (no __file__), so __file__ can't be relied on.
_CANDIDATE_DIRS = ["/kaggle_simulations/agent", os.getcwd()]
try:
    _CANDIDATE_DIRS.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
for _path in list(_CANDIDATE_DIRS):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import Observation, OptionType, SelectContext, to_observation_class  # noqa: E402

# `cg` が import できた時点で、そのファイルの場所からバンドルの位置が確定する。
# cwd も __file__ も当てにならない環境で、これが一番確実な手がかりになる。
try:
    import cg as _cg_pkg
    _bundle_dir = os.path.dirname(os.path.dirname(os.path.abspath(_cg_pkg.__file__)))
    if _bundle_dir not in _CANDIDATE_DIRS:
        _CANDIDATE_DIRS.append(_bundle_dir)
    if os.path.isdir(_bundle_dir) and _bundle_dir not in sys.path:
        sys.path.insert(0, _bundle_dir)
except Exception:
    pass

_AGENT = None
_AGENT_LOAD_TRIED = False
_LOAD_ERROR_REPORTED = False


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
    """Lazily import the rule-based agent; return None if unavailable (triggers fallback).

    読み込みに失敗したら **必ず stderr に理由を出す**。ここを黙って握りつぶすと、
    「エラーは一切出ないのに、判断が丸ごと効いていない」状態で試合が進んでしまう
    (実際に Kaggle 上で 118 手すべてが保険の行動になり、1度も攻撃しないまま負けた)。
    """
    global _AGENT, _AGENT_LOAD_TRIED, _LOAD_ERROR_REPORTED
    if _AGENT_LOAD_TRIED:
        return _AGENT
    _AGENT_LOAD_TRIED = True

    # 1回目: 普通の import（sys.path が正しく通っていればこれで済む）。
    try:
        from rule_agents.{package} import agent as _rule_agent
        _AGENT = _rule_agent
        return _AGENT
    except Exception:
        first_error = traceback.format_exc()

    # 2回目: ファイルの場所から直接読み込む。sys.path の状態に一切依存しないので、
    # 実行環境がどこを cwd にしていても、バンドルさえ見つかれば必ず読み込める。
    try:
        import importlib.util
        for _dir in _CANDIDATE_DIRS:
            pkg_dir = os.path.join(_dir, "rule_agents")
            mod_path = os.path.join(pkg_dir, "{module}")
            if not os.path.exists(mod_path):
                continue
            if "rule_agents" not in sys.modules:
                pkg_spec = importlib.util.spec_from_file_location(
                    "rule_agents", os.path.join(pkg_dir, "__init__.py"),
                    submodule_search_locations=[pkg_dir],
                )
                pkg = importlib.util.module_from_spec(pkg_spec)
                sys.modules["rule_agents"] = pkg
                pkg_spec.loader.exec_module(pkg)
            spec = importlib.util.spec_from_file_location("rule_agents.{package}", mod_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["rule_agents.{package}"] = mod
            spec.loader.exec_module(mod)
            _AGENT = mod.agent
            return _AGENT
    except Exception:
        first_error += "\\n--- ファイル指定での読み込みも失敗 ---\\n" + traceback.format_exc()

    _AGENT = None
    if not _LOAD_ERROR_REPORTED:
        _LOAD_ERROR_REPORTED = True
        print("[rule_agent] 判断ロジックの読み込みに失敗しました。保険の行動で進行します。",
              file=sys.stderr)
        print(f"[rule_agent] cwd={{os.getcwd()!r}} candidates={{_CANDIDATE_DIRS!r}}", file=sys.stderr)
        print(first_error, file=sys.stderr)
    return _AGENT


# 保険の行動でも、最低限「殴れるときは殴る」程度のことはする。
# 単純に先頭の選択肢を選び続けると、にげるを連打したり、ボスの指令を無駄打ちしたり、
# パンクアップで0枚しか付けなかったりと、目に見えて悪い試合になる。
_MAIN_PRIORITY = {{
    OptionType.EVOLVE: 6,
    OptionType.ATTACH: 5,
    OptionType.PLAY: 4,
    OptionType.ABILITY: 3,
    OptionType.ATTACK: 2,
    OptionType.RETREAT: 1,
    OptionType.END: 0,
}}

# 「選べるだけ選んだ方が得」な文脈(サーチ・付け先・ベンチ展開など)。
_TAKE_MAX_CONTEXTS = {{
    SelectContext.TO_HAND,
    SelectContext.ATTACH_TO,
    SelectContext.TO_BENCH,
    SelectContext.TO_FIELD,
    SelectContext.SETUP_BENCH_POKEMON,
}}


def _fallback_action(obs: Observation) -> list[int]:
    """判断ロジックが使えないときの保険。必ず合法な手を返す。"""
    sel = obs.select
    if sel is None:
        return read_deck_csv()
    n = len(sel.option)
    if n == 0:
        return []

    if sel.context == SelectContext.MAIN:
        # 手数が伸びすぎたら打ち切る(未知の効果でループしないための保険)。
        actions_taken = getattr(obs.current, "turnActionCount", 0) or 0
        best = 0
        best_score = -1
        for i, o in enumerate(sel.option):
            score = -1 if actions_taken > 40 and o.type != OptionType.END else \\
                _MAIN_PRIORITY.get(o.type, 0)
            if score > best_score:
                best, best_score = i, score
        return [best]

    take = sel.minCount
    if sel.context in _TAKE_MAX_CONTEXTS:
        take = max(sel.minCount, min(sel.maxCount, n))
    take = max(0, min(take, n))
    return list(range(take))


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
