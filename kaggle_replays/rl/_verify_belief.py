"""desktop 上で belief 学習の前提を検証: 予測器ロード / belief_init ロード / 次元一致。"""
import sys
sys.path.insert(0, ".")
ok = True
try:
    from ptcg_ai.hidden_information import match_context
    p = match_context._get_predictor()
    ready = p is not None and getattr(p, "is_ready", False)
    print("predictor ready:", ready, "classes:", len(getattr(p, "_classes", [])) if p else 0)
    if not ready:
        ok = False
except Exception as e:
    print("predictor load EXC:", repr(e)); ok = False

try:
    from ptcg_ai.learning.policy_model import PolicyModel
    m = PolicyModel("ptcg_ai/learning/policy_weights_alakazam_rl_belief_init.json")
    print("belief_init ready:", m.is_ready, "extra:", m._extra_features,
          "state_mean:", len(m._state_mean or []),
          "layer0_in:", (len(m._layers[0][0][0]) if m._layers else None))
    if not m.is_ready or m._extra_features != "opp_belief":
        ok = False
except Exception as e:
    print("belief_init load EXC:", repr(e)); ok = False

try:
    from ptcg_ai.learning.extra_features import compute_extra_features, opponent_belief_features
    print("extra_features import: OK")
except Exception as e:
    print("extra_features import EXC:", repr(e)); ok = False

print("VERIFY", "PASS" if ok else "FAIL")
