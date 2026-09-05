"""Kaggle Submission Candidate 1 build — v2.4 H4 + v2.11 Batched scorer(意味不変)を提出 tree へ移植。

sample_submission/ は変更せず、temp build dir に:
  - sample_submission の {main.py, deck.csv, cg, configs, ptcg_ai, decks} をコピー(__pycache__ 除外)
  - ptcg_ai/ismcts/ を新設: ismcts_core.py(=search/ismcts.py)/ batched_policy.py / _agent_impl.py(=ismcts_v1_agent.py 移植)
    / ismcts_agent.py(agent(obs) wrapper)/ rollout_student_h4.json(H4 weights)/ submission_config.json(v2.11 batched, 1350ms fixed)
  - ptcg_ai/core/agent.py の AGENT_TYPE を "ismcts" にし ismcts 分岐を追加
tar(main.py deck.csv cg configs ptcg_ai decks)。research asset のみ。
`python build_ismcts_submission.py <build_dir> <out_tarball>`
"""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
_SEARCH = _ROOT / "kaggle_replays" / "search"
_H4 = _ROOT / "kaggle_replays" / "training" / "rollout_student_h4.json"
_V211_CFG = _HERE / "configs" / "ismcts_v2_11_h4batch_t1350.json"


def _ign(_d, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def main(build_dir: str, out_tar: str):
    build = Path(build_dir)
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    # 1) sample_submission の必須ツリーをコピー
    for name in ("cg", "configs", "ptcg_ai", "decks"):
        src = _SUB / name
        if src.exists():
            shutil.copytree(src, build / name, ignore=_ign)
    for f in ("main.py", "deck.csv"):
        shutil.copy2(_SUB / f, build / f)

    # 2) ptcg_ai/ismcts/ を移植
    ism = build / "ptcg_ai" / "ismcts"
    ism.mkdir(parents=True)
    (ism / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(_SEARCH / "ismcts.py", ism / "ismcts_core.py")             # cg + ptcg_ai.search のみ=そのまま
    shutil.copy2(_HERE / "batched_policy.py", ism / "batched_policy.py")     # numpy + ptcg_ai.learning=そのまま
    shutil.copy2(_H4, ism / "rollout_student_h4.json")

    # _agent_impl.py = ismcts_v1_agent.py を package 用に移植(sys.path 除去・import 修正)
    agent_src = (_HERE / "ismcts_v1_agent.py").read_text(encoding="utf-8")
    block = ('_HERE = Path(__file__).resolve().parent\n'
             '_ROOT = _HERE.parents[1]\n'
             '_SUB = _ROOT / "sample_submission"\n'
             '_SEARCH = _ROOT / "kaggle_replays" / "search"\n'
             'for _p in (str(_SUB), str(_SEARCH)):\n'
             '    if _p not in sys.path:\n'
             '        sys.path.insert(0, _p)\n')
    assert block in agent_src, "sys.path block not found (source changed?)"
    agent_src = agent_src.replace(block, '_HERE = Path(__file__).resolve().parent\n')
    agent_src = agent_src.replace(
        'import ismcts  # noqa: E402  (kaggle_replays/search/ismcts.py)',
        'from ptcg_ai.ismcts import ismcts_core as ismcts  # noqa: E402')
    agent_src = agent_src.replace(
        '            p = _SUB / p            # sample_submission 相対を許可',
        '            p = _HERE / p            # ismcts module 相対を許可')
    agent_src = agent_src.replace(
        '            from batched_policy import BatchedPolicyModel',
        '            from ptcg_ai.ismcts.batched_policy import BatchedPolicyModel')
    agent_src = agent_src.replace(
        '_CONFIG_PATH = _HERE / "configs" / "ismcts_v1.json"',
        '_CONFIG_PATH = _HERE / "submission_config.json"')
    (ism / "_agent_impl.py").write_text(agent_src, encoding="utf-8")

    # ismcts_agent.py = agent(obs) wrapper(deck 選択は mpa へ委譲)
    (ism / "ismcts_agent.py").write_text(
        'from __future__ import annotations\n'
        'import json\n'
        'from pathlib import Path\n'
        'from cg.api import Observation\n'
        'from ptcg_ai.ismcts import _agent_impl\n'
        'from ptcg_ai.ml_policy import ml_policy_agent as _mpa\n'
        'from ptcg_ai.hidden_information import match_context as _mc\n'
        '\n'
        '_HERE = Path(__file__).resolve().parent\n'
        '_CONFIG = json.loads((_HERE / "submission_config.json").read_text(encoding="utf-8"))\n'
        '\n'
        'def agent(obs: Observation) -> list[int]:\n'
        '    if obs.select is None:\n'
        '        _mc.reset()\n'
        '        return _mpa.agent(obs, _CONFIG)   # 初回=デッキ選択(deck.csv 60枚)\n'
        '    return _agent_impl.agent(obs, _CONFIG)\n',
        encoding="utf-8")

    # submission_config.json = v2.11 batched(H4 weights を module 相対に)
    cfg = json.loads(_V211_CFG.read_text(encoding="utf-8"))
    cfg["name"] = "submission_v2_4_h4_v2_11_batch"
    cfg["ismcts"]["rollout_policy_weights"] = "rollout_student_h4.json"   # _HERE 相対
    cfg["ismcts"]["rollout_policy_batched"] = True
    cfg["ismcts"].pop("early_stop", None)                                # v2.12 OFF
    cfg["ismcts"].pop("log_trajectory", None)
    cfg["ismcts"]["budget_ms"] = 1350
    cfg["ismcts"]["instrument"] = False
    (ism / "submission_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) core/agent.py の AGENT_TYPE を ismcts に
    core = build / "ptcg_ai" / "core" / "agent.py"
    txt = core.read_text(encoding="utf-8")
    txt = txt.replace('AGENT_TYPE = "ml_policy"', 'AGENT_TYPE = "ismcts"')
    branch = ('    if AGENT_TYPE == "ismcts":\n'
              '        from ptcg_ai.ismcts.ismcts_agent import agent as ismcts_agent\n'
              '\n'
              '        return ismcts_agent(obs)\n'
              '\n'
              '    if AGENT_TYPE == "ml_policy":')
    txt = txt.replace('    if AGENT_TYPE == "ml_policy":', branch, 1)
    core.write_text(txt, encoding="utf-8")

    # 4) tar(__pycache__ 除外)
    out = Path(out_tar)
    if out.exists():
        out.unlink()
    with tarfile.open(out, "w:gz") as tf:
        for name in ("main.py", "deck.csv", "cg", "configs", "ptcg_ai", "decks"):
            p = build / name
            if p.exists():
                tf.add(p, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    nfiles = sum(1 for _ in tarfile.open(out, "r:gz"))
    print(f"built {out}  size={out.stat().st_size/1024:.0f}KB  files={nfiles}")
    print(f"  AGENT_TYPE=ismcts / batched=True / early_stop=OFF / budget_ms=1350 / rollout=FULL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
