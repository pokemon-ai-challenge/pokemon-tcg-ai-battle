"""rule_agents/ の手書きルールベース1種を、Kaggle に出せる tar.gz に固める。

sample_submission/ の本提出物(ML方策)とは別に、「このデッキのルールベース単体で
提出したらどうなるか」を試したいときに使う。torch/numpy 等の重みは無く、選んだ
デッキ1種の agent + framework + cg エンジンだけを詰める、小さく自己完結したバンドル。

## バンドルの中身

```
main.py            Kaggle エントリポイント(__file__ 非依存。判断ロジックが読めない/
                    不正な手を返したときは、誤魔化さず例外を送出して負ける)
deck.csv           選んだデッキの60枚(main.py の read_deck_csv 用)
cg/                sample_submission/cg のコピー(変更しない)
rule_agents/        framework.py + 選んだデッキの1ファイル + そのデッキCSV
```

**方針: 提出枠を「動いているふり」で埋めない。** 判断ロジックが読み込めない場合、
以前は「合法だが何も考えていない手」を返し続ける保険を持たせていたが、これだと
対局はエラー無く完走するのに実質何もしていない、という外から気づけない状態になる
(実際にこれで118手すべてが保険の行動という試合を2つ提出していた)。今は読み込みに
失敗するか、判断ロジックが不正な手を返したら、その場で例外を送出する。

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
deck's game plan).

**This submission has no silent fallback.** If the decision logic can't be loaded,
or it ever returns something the engine won't accept, this raises instead of
quietly playing dummy moves. A submission slot that plays nothing is worse than
a submission slot that visibly errors out -- an error is noticed and fixed; a
match that completes while doing nothing looks fine and isn't (this happened for
real: 118/118 decisions in two ranked matches were the old silent fallback,
and it was only caught by reading the replay JSON by hand).
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

from cg.api import Observation, to_observation_class  # noqa: E402

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
_LOAD_ERROR = None


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
    """Lazily import the rule-based agent.

    2通り試す(通常の import と、ファイル位置を直接指定した読み込み)。どちらも
    失敗したら None を返す。呼び出し側(agent())がそれを見て例外を送出する
    ("読めなかったら保険で誤魔化す"のではなく、はっきり負けにする方針)。
    """
    global _AGENT, _AGENT_LOAD_TRIED, _LOAD_ERROR
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
    _LOAD_ERROR = (
        f"cwd={{os.getcwd()!r}} candidates={{_CANDIDATE_DIRS!r}}\\n{{first_error}}"
    )
    return _AGENT


def agent(obs_dict: dict) -> list[int]:
    """Competition entry point. Returns option indices (or the deck on turn 0).

    判断ロジックが読み込めない、または不正な手を返した場合は例外を送出する。
    握りつぶして適当な手を返すと、対局はエラー無く終わるのに実質何もしていない
    という、外からは気づけない負け方をする(この提出物で一度実際に起きた)。
    それよりは、提出枠がはっきりエラーになる方が良い。
    """
    obs: Observation = to_observation_class(obs_dict)

    if obs.select is None:
        return read_deck_csv()

    rule_agent = _get_agent()
    if rule_agent is None:
        raise RuntimeError(
            "rule_agents の読み込みに失敗しました(判断ロジックが一切動いていません)。"
            "誤魔化さずここで止めます。詳細:\\n" + (_LOAD_ERROR or "(不明)")
        )

    action = rule_agent(obs)
    n = len(obs.select.option)
    sel = obs.select
    if not all(0 <= i < n for i in action):
        raise RuntimeError(f"範囲外の選択: {{action}} (option数={{n}})")
    if len(set(action)) != len(action):
        raise RuntimeError(f"選択肢が重複している: {{action}}")
    if not (sel.minCount <= len(action) <= sel.maxCount):
        raise RuntimeError(
            f"選択個数が範囲外: {{len(action)}} not in [{{sel.minCount}}, {{sel.maxCount}}]"
        )
    return action
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
