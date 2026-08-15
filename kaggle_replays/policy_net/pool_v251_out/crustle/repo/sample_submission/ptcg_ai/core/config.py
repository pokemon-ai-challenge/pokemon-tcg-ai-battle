"""Config loading for ptcg_ai modules.

Configs are JSON files under ``sample_submission/configs/``.
On Kaggle the submission is extracted to ``/kaggle_simulations/agent/``,
and ``main.py`` is executed via ``exec()`` (no ``__file__``), so this module
resolves the configs directory from its own ``__file__`` with fallbacks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CONFIG_NAME = "rule_lethal"

# Used when no config file can be found. Keeps the agent runnable
# (lethal search disabled) instead of crashing.
DEFAULT_CONFIG: dict = {
    "name": "default",
    "lethal_search": {
        "enabled": False,
    },
}


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        here = Path(__file__).resolve()
        # config.py -> core -> ptcg_ai -> sample_submission
        dirs.append(here.parents[2] / "configs")
    except NameError:  # pragma: no cover - defensive for exec() contexts
        pass
    dirs.append(Path("/kaggle_simulations/agent/configs"))
    dirs.append(Path.cwd() / "configs")
    return dirs


def load_config(name: str | None = None) -> dict:
    """Load ``configs/<name>.json``.

    Args:
        name: Config name without extension. Defaults to the
            ``PTCG_AI_CONFIG`` environment variable or ``rule_lethal``.

    Returns:
        dict: Parsed config, or ``DEFAULT_CONFIG`` if the file is missing.
    """
    name = name or os.environ.get("PTCG_AI_CONFIG", DEFAULT_CONFIG_NAME)
    for directory in _candidate_dirs():
        path = directory / f"{name}.json"
        try:
            if path.is_file():
                with open(path, "r", encoding="utf-8") as file:
                    return json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
    return json.loads(json.dumps(DEFAULT_CONFIG))
