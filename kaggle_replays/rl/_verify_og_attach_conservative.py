"""抽出済みtarballを、環境変数PTCG_AI_ML_CONFIGを一切設定せず(Kaggle実行を模した状態)で
検証する。deck/config/挙動(手貼りが実際に発火するか)まで確認する。
"""
import sys

EXTRACT_DIR = r"C:\Users\morit\Desktop\pokehack\pokemon-tcg-ai-battle\build_ready\_verify_og_attach_conservative"
sys.path.insert(0, EXTRACT_DIR)

import os
assert "PTCG_AI_ML_CONFIG" not in os.environ, "このプロセスでは env var を一切設定していないこと"

import main  # noqa: E402
from ptcg_ai.rule_based.rule_based_agent import read_deck_csv  # noqa: E402

deck = read_deck_csv()
assert len(deck) == 60, f"deck size = {len(deck)}"
OGERPON_EX = 96
assert OGERPON_EX in deck, "deck does not contain Ogerpon-ex"
print("deck OK: 60 cards, contains Ogerpon-ex(96)")

from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
print("ml_policy_agent._CONFIG_NAME (env未設定時に実際に使われる名前):", ml_policy_agent._CONFIG_NAME)  # noqa: SLF001
cfg = ml_policy_agent._get_config()  # main.agent() が実際に使うのと同じ解決経路  # noqa: SLF001
print("resolved config name field:", cfg.get("name"))
og = cfg.get("ogerpon_planner", {})
print("ogerpon_planner block:", og)
assert og.get("enabled") is True, "ogerpon_planner.enabled が True になっていない"
assert og.get("attach_enabled") is True, "attach_enabled が True になっていない"
assert cfg.get("ogerpon_finish_only", {}).get("enabled", False) is False, \
    "finish_only が誤って有効化されている"
print("config swap OK: og_attach_conservative の中身が既定解決で読めている")

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from ptcg_ai.ml_policy import ogerpon_planner as P  # noqa: E402

errors = 0
bulu_attach_actual = 0
TAPU_BULU = 920
for g in range(8):
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        print(f"game {g}: start error {sd.errorType}")
        errors += 1
        continue
    n = 0
    err = None
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                err = "current None"
                break
            if cur.result != -1:
                break
            if n >= 2000:
                err = "max_steps"
                break
            sel = obs.select
            action = main.agent(obs_dict)
            if sel is not None:
                if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
                    err = f"bad type {action}"; break
                if not (sel.minCount <= len(action) <= sel.maxCount):
                    err = f"bad count {action}"; break
                if len(action) != len(set(action)):
                    err = f"dup {action}"; break
                if not all(0 <= i < len(sel.option) for i in action):
                    err = f"oob {action}"; break
                if int(sel.type) == 0 and action:  # SelectType.MAIN
                    opt = sel.option[action[0]]
                    if int(getattr(opt, "type", -1)) == 8:  # OptionType.ATTACH
                        mine = cur.players[cur.yourIndex]
                        tgt = P._resolve_attach_target(cur, cur.yourIndex, opt)  # noqa: SLF001
                        if tgt is not None and tgt.id == TAPU_BULU:
                            bulu_attach_actual += 1
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()
    if err is not None:
        print(f"game {g}: ERROR {err}")
        errors += 1
    else:
        print(f"game {g}: OK ({n} steps)")

print()
print(f"errors={errors}/8, bulu_attach_actual(observed)={bulu_attach_actual}")
assert errors == 0, "smoke test中にエラーが発生"
print("SMOKE TEST PASSED")
