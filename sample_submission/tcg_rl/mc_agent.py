"""Flat Monte Carlo + heuristic agent for the Dragapult ex deck.

Ported and consolidated from the `reinforce_learning` branch into a single
class so it can serve as a training opponent for the PPO agent and as a place to
keep improving the board evaluation function.

Two modes:
  * heuristic-only (time_budget <= 0): fast, rule-based; the default opponent.
  * flat Monte Carlo (time_budget > 0): uses cg.api search to roll out candidate
    actions and pick the highest mean value. Slower; opt-in.

evaluate() is the clearly-marked improvement target (see EVALUATION NOTES).
"""

from __future__ import annotations

import os
import random
import time

from cg.api import (
    AreaType,
    Observation,
    OptionType,
    SelectContext,
    all_attack,
    all_card_data,
    search_begin,
    search_step,
    search_end,
    search_release,
    to_observation_class,
)

# --- Dragapult deck card ids -------------------------------------------------
DREEPY_ID = 119
DRAKLOAK_ID = 120
DRAGAPULT_EX_ID = 121
BLAZIKEN_EX_ID = 326
TORCHIC_ID = 410
COMBUSKEN_ID = 411
MUNKIDORI_ID = 112
CHI_YU_ID = 31
FEZANDIPITI_EX_ID = 140

_CARD_DB: dict[int, object] = {}
_ATTACK_DB: dict[int, object] = {}
_DB_LOADED = False


def _load_db() -> None:
    global _DB_LOADED
    if _DB_LOADED:
        return
    for c in all_card_data():
        _CARD_DB[c.cardId] = c
    for a in all_attack():
        _ATTACK_DB[a.attackId] = a
    _DB_LOADED = True


def _prize_value(card_id: int) -> int:
    c = _CARD_DB.get(card_id)
    if c is None:
        return 1
    if getattr(c, "megaEx", False):
        return 3
    if c.ex:
        return 2
    return 1


def _damage_progress(poke) -> float:
    if poke is None:
        return 0.0
    hp_ratio = poke.hp / max(poke.maxHp, 1)
    return _prize_value(poke.id) * (1.0 - hp_ratio)


# ============================================================================ #
#  EVALUATION NOTES — this is the function to keep improving.
#  Current terms (my_idx perspective, clamped to [-1,1]):
#    (1) prize differential                 weight 0.60   most important
#    (2) opponent damage-counter progress   weight 0.18   rewards ability damage
#    (3) active HP ratio differential       weight 0.10
#    (4) evolution-stage bonus              weight 0.08   Dragapult/Blaziken in play
#    (5) attacker energy-readiness          weight 0.04   NEW vs reference
#    (6) bench development differential      weight 0.02
#  Candidate future terms: deck-out risk, next-turn attack continuity, disruption
#  resources (Boss/switch) in hand, side-plan (2-2-2) awareness.
# ============================================================================ #
def evaluate(obs: Observation, my_idx: int) -> float:
    state = obs.current
    if state is None:
        return 0.0
    if state.result != -1:
        if state.result == my_idx:
            return 1.0
        if state.result == 1 - my_idx:
            return -1.0
        return 0.0

    me = state.players[my_idx]
    opp = state.players[1 - my_idx]

    prize = (len(opp.prize) - len(me.prize)) / 6.0

    def total_damage_progress(ps) -> float:
        total = 0.0
        if ps.active and ps.active[0] is not None:
            total += _damage_progress(ps.active[0]) * 1.0
        for p in ps.bench:
            total += _damage_progress(p) * 0.6
        return total

    damage = (total_damage_progress(opp) - total_damage_progress(me)) * 0.18

    def hp_r(ps) -> float:
        if not ps.active or ps.active[0] is None:
            return 0.0
        p = ps.active[0]
        return p.hp / max(p.maxHp, 1)

    hp = (hp_r(me) - hp_r(opp)) * 0.10

    evo_bonus = 0.0
    all_my = list(me.bench)
    if me.active and me.active[0] is not None:
        all_my.append(me.active[0])
    for p in all_my:
        if p.id == DRAGAPULT_EX_ID:
            evo_bonus += 0.05
        elif p.id == DRAKLOAK_ID:
            evo_bonus += 0.02
        elif p.id == BLAZIKEN_EX_ID:
            evo_bonus += 0.03

    # (5) NEW: attacker energy-readiness — reward energy on a real attacker that
    # is close to meeting its highest-damage attack cost (encourages building up).
    readiness = 0.0
    if me.active and me.active[0] is not None:
        a = me.active[0]
        need = _max_attack_cost(a.id)
        if need > 0:
            readiness = min(len(a.energies), need) / need * 0.04

    bench = (len(me.bench) - len(opp.bench)) * 0.02

    return max(-1.0, min(1.0, prize * 0.60 + damage + hp + evo_bonus + readiness + bench))


def _max_attack_cost(card_id: int) -> int:
    c = _CARD_DB.get(card_id)
    if c is None or not c.attacks:
        return 0
    best = 0
    for aid in c.attacks:
        atk = _ATTACK_DB.get(aid)
        if atk is not None:
            best = max(best, len(atk.energies or []))
    return best


class MonteCarloAgent:
    def __init__(self, deck: list[int] | None = None, rollouts: int = 0,
                 depth: int = 20, time_budget: float = 0.0):
        _load_db()
        self.deck = list(deck) if deck else self._read_deck()
        self.rollouts = rollouts
        self.depth = depth
        self.time_budget = time_budget
        self.opp_seen: list[int] = []

    # ------------------------------------------------------------------ public
    def act(self, obs: Observation) -> list[int]:
        if obs.select is None:
            return list(self.deck)
        self._update_opp_seen(obs.logs)
        if self.time_budget > 0 and self.rollouts != 0 and obs.search_begin_input:
            try:
                return self._flat_mc(obs)
            except Exception:
                pass
        return self.heuristic(obs)

    # -------------------------------------------------------------- heuristic
    def heuristic(self, obs: Observation) -> list[int]:
        sel = obs.select
        ctx = sel.context
        if ctx == SelectContext.MAIN:
            return self._h_main(obs)
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._h_setup_active(obs)
        if ctx == SelectContext.SETUP_BENCH_POKEMON:
            return self._h_setup_bench(obs)
        if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
            return self._rank_targets(obs)
        if ctx == SelectContext.DAMAGE_COUNTER_COUNT:
            return [len(sel.option) - 1] if sel.option else [0]
        if ctx == SelectContext.EFFECT_TARGET:
            r = self._rank_targets(obs)
            return r if len(r) >= sel.minCount else list(range(sel.minCount))
        if ctx in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return self._h_to_bench(obs)
        if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK_BOTTOM):
            return self._h_discard(obs)
        if ctx in (SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL):
            return self._h_heal(obs)
        if ctx in (SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT,
                   SelectContext.MULLIGAN, SelectContext.MORE_DEVOLVE,
                   SelectContext.COIN_HEAD):
            return self._h_yes(obs)
        if ctx == SelectContext.TO_HAND:
            return self._h_to_hand(obs)
        return list(range(sel.minCount))

    def _h_main(self, obs: Observation) -> list[int]:
        sel, st = obs.select, obs.current
        evo, abi, att, atk, play, end = [], [], [], [], [], []
        for i, opt in enumerate(sel.option):
            t = opt.type
            if t == OptionType.EVOLVE:
                evo.append(i)
            elif t == OptionType.ABILITY:
                abi.append(i)
            elif t == OptionType.ATTACH:
                att.append(i)
            elif t == OptionType.ATTACK:
                atk.append(i)
            elif t == OptionType.PLAY:
                play.append(i)
            elif t == OptionType.END:
                end.append(i)
        if evo:
            return [self._best_by_card(obs, evo,
                    {DRAGAPULT_EX_ID: 100, DRAKLOAK_ID: 80, BLAZIKEN_EX_ID: 60, COMBUSKEN_ID: 40})]
        if abi:
            return [self._best_by_card(obs, abi,
                    {MUNKIDORI_ID: 100, CHI_YU_ID: 90, FEZANDIPITI_EX_ID: 80})]
        if not st.energyAttached and att:
            return [self._best_attach(obs, att)]
        if atk:
            return [self._best_attack(obs, atk)]
        if play:
            return [play[0]]
        if end:
            return [end[0]]
        return [0]

    def _best_by_card(self, obs, opts, priority) -> int:
        sel = obs.select
        best_i, best_p = opts[0], -1
        for i in opts:
            cid = getattr(sel.option[i], "cardId", None)
            p = priority.get(cid, 1) if cid else 1
            if p > best_p:
                best_p, best_i = p, i
        return best_i

    def _best_attach(self, obs, opts) -> int:
        sel, st = obs.select, obs.current
        my = st.yourIndex
        ps = st.players[my]

        def prio(opt) -> int:
            area = getattr(opt, "inPlayArea", None)
            idx = getattr(opt, "inPlayIndex", None)
            if area is None:
                return 0
            poke = None
            if area == AreaType.ACTIVE and ps.active:
                poke = ps.active[0]
            elif area == AreaType.BENCH and idx is not None and idx < len(ps.bench):
                poke = ps.bench[idx]
            if poke is None:
                return 1
            if poke.id == DRAGAPULT_EX_ID:
                return 100
            if poke.id == DRAKLOAK_ID:
                return 80
            if poke.id == BLAZIKEN_EX_ID:
                return 60
            if area == AreaType.ACTIVE:
                return 40
            return 10

        return max(opts, key=lambda i: prio(sel.option[i]))

    def _best_attack(self, obs, opts) -> int:
        sel, st = obs.select, obs.current
        opp = st.players[1 - st.yourIndex]
        opp_hp = opp.active[0].hp if (opp.active and opp.active[0] is not None) else 0
        best_i, best_s = opts[0], -1
        for i in opts:
            opt = sel.option[i]
            dmg = 0
            if opt.attackId is not None and opt.attackId in _ATTACK_DB:
                dmg = _ATTACK_DB[opt.attackId].damage
            s = dmg + (10000 if opp_hp > 0 and dmg >= opp_hp else 0)
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    def _rank_targets(self, obs: Observation) -> list[int]:
        sel, st = obs.select, obs.current
        my = st.yourIndex
        count = min(sel.maxCount, len(sel.option))
        if count == 0:
            return []

        def score(i: int) -> float:
            opt = sel.option[i]
            area = getattr(opt, "area", None)
            idx = getattr(opt, "index", None)
            pidx = getattr(opt, "playerIndex", None)
            if pidx is None or area is None or pidx == my:
                return -9999.0
            ps = st.players[pidx]
            poke = None
            if area == AreaType.ACTIVE:
                poke = ps.active[0] if ps.active else None
            elif area == AreaType.BENCH and idx is not None and idx < len(ps.bench):
                poke = ps.bench[idx]
            if poke is None:
                return -9999.0
            pv = _prize_value(poke.id)
            return pv * (1.0 - poke.hp / max(poke.maxHp, 1)) + pv * 0.05

        return sorted(range(len(sel.option)), key=score, reverse=True)[:count]

    def _h_setup_active(self, obs) -> list[int]:
        sel = obs.select
        if not sel.option:
            return []
        for i, opt in enumerate(sel.option):
            if getattr(opt, "cardId", None) == DREEPY_ID:
                return [i]
        best_i, best_hp = 0, float("inf")
        for i, opt in enumerate(sel.option):
            cid = getattr(opt, "cardId", None)
            if cid in _CARD_DB and _CARD_DB[cid].hp < best_hp:
                best_hp, best_i = _CARD_DB[cid].hp, i
        return [best_i]

    def _h_setup_bench(self, obs) -> list[int]:
        sel = obs.select
        return list(range(min(sel.maxCount, len(sel.option))))

    def _h_to_bench(self, obs) -> list[int]:
        sel = obs.select
        count = min(sel.maxCount, len(sel.option))
        if count == 0:
            return []
        prio = {DREEPY_ID: 100, TORCHIC_ID: 80, MUNKIDORI_ID: 60,
                CHI_YU_ID: 50, FEZANDIPITI_EX_ID: 40}
        return sorted(range(len(sel.option)),
                      key=lambda i: prio.get(getattr(sel.option[i], "cardId", 0), 1),
                      reverse=True)[:count]

    def _h_discard(self, obs) -> list[int]:
        sel = obs.select
        count = min(sel.maxCount, len(sel.option))
        if count == 0:
            return []

        def keep(i: int) -> int:
            cid = getattr(sel.option[i], "cardId", None)
            c = _CARD_DB.get(cid) if cid else None
            if c is None:
                return 10
            if cid in (DREEPY_ID, TORCHIC_ID, MUNKIDORI_ID):
                return 100
            if cid in (DRAGAPULT_EX_ID, DRAKLOAK_ID, BLAZIKEN_EX_ID):
                return 90
            if c.basic and not (c.stage1 or c.stage2):
                return 20
            return 30

        return sorted(range(len(sel.option)), key=keep)[:count]

    def _h_heal(self, obs) -> list[int]:
        sel, st = obs.select, obs.current
        my = st.yourIndex
        ps = st.players[my]
        count = min(sel.maxCount, len(sel.option))
        if count == 0:
            return []

        def score(i: int) -> float:
            opt = sel.option[i]
            area = getattr(opt, "area", None)
            idx = getattr(opt, "index", None)
            if getattr(opt, "playerIndex", None) != my:
                return -1.0
            poke = None
            if area == AreaType.ACTIVE and ps.active:
                poke = ps.active[0]
            elif area == AreaType.BENCH and idx is not None and idx < len(ps.bench):
                poke = ps.bench[idx]
            if poke is None:
                return 0.0
            return _prize_value(poke.id) * (1.0 - poke.hp / max(poke.maxHp, 1))

        return sorted(range(len(sel.option)), key=score, reverse=True)[:count]

    def _h_to_hand(self, obs) -> list[int]:
        sel = obs.select
        count = min(sel.maxCount, len(sel.option))
        if count == 0:
            return []
        prio = {DRAGAPULT_EX_ID: 100, DRAKLOAK_ID: 90, MUNKIDORI_ID: 80,
                DREEPY_ID: 70, BLAZIKEN_EX_ID: 60, TORCHIC_ID: 50}
        return sorted(range(len(sel.option)),
                      key=lambda i: prio.get(getattr(sel.option[i], "cardId", 0), 1),
                      reverse=True)[:count]

    def _h_yes(self, obs) -> list[int]:
        sel = obs.select
        for i, opt in enumerate(sel.option):
            if opt.type == OptionType.YES:
                return [i]
        return [0]

    # -------------------------------------------------------------- flat MC
    def _legal_actions(self, obs, max_n: int = 12) -> list[list[int]]:
        sel = obs.select
        n, lo, hi = len(sel.option), sel.minCount, sel.maxCount
        if n == 0:
            return [[]] if lo == 0 else []
        cands: list[list[int]] = []
        if hi <= 1:
            cands = [[i] for i in range(n)]
            if lo == 0:
                cands.append([])
        elif lo == hi:
            cands.append(list(range(min(n, hi))))
        else:
            for cnt in range(lo, min(hi + 1, lo + 3)):
                if n >= cnt:
                    cands.append(list(range(cnt)))
        seen, uniq = set(), []
        for a in cands:
            k = tuple(sorted(a))
            if k not in seen:
                seen.add(k)
                uniq.append(a)
        return uniq[:max_n] if uniq else [list(range(lo))]

    def _rand_action(self, obs) -> list[int]:
        sel = obs.select
        n, lo, hi = len(sel.option), sel.minCount, sel.maxCount
        if n == 0:
            return []
        cnt = random.randint(lo, min(hi, n))
        return random.sample(range(n), cnt) if cnt > 0 else []

    def _flat_mc(self, obs: Observation) -> list[int]:
        my = obs.current.yourIndex
        deadline = time.time() + self.time_budget
        root = search_begin(
            obs, self.deck, self._pred_my_prize(obs.current.players[my]),
            self._pred_opp_deck(obs.current.players[1 - my].deckCount),
            self._pred_opp_prize(len(obs.current.players[1 - my].prize)),
            self._pred_opp_hand(obs.current.players[1 - my].handCount),
            self._pred_opp_active(obs, my),
        )
        cands = self._legal_actions(root.observation)
        if len(cands) == 1:
            search_end()
            return cands[0]
        vals = {tuple(sorted(a)): 0.0 for a in cands}
        counts = {tuple(sorted(a)): 0 for a in cands}
        idx = 0
        while time.time() < deadline:
            action = cands[idx % len(cands)]
            idx += 1
            try:
                ns = search_step(root.searchId, action)
                v = self._rollout(ns.searchId, ns.observation, my, deadline)
                search_release(ns.searchId)
                vals[tuple(sorted(action))] += v
                counts[tuple(sorted(action))] += 1
            except Exception:
                pass
        search_end()
        best_a, best_v = cands[0], float("-inf")
        for a in cands:
            k = tuple(sorted(a))
            if counts[k] > 0 and vals[k] / counts[k] > best_v:
                best_v, best_a = vals[k] / counts[k], a
        return best_a

    def _rollout(self, sid, ob, my, deadline) -> float:
        cur_id, cur = sid, ob
        created = []
        try:
            for _ in range(self.depth):
                if time.time() > deadline or (cur.current and cur.current.result != -1):
                    break
                action = self._rand_action(cur)
                if not action and cur.select.minCount > 0:
                    break
                ns = search_step(cur_id, action)
                created.append(ns.searchId)
                cur_id, cur = ns.searchId, ns.observation
        except Exception:
            pass
        finally:
            for s in created:
                try:
                    search_release(s)
                except Exception:
                    pass
        return evaluate(cur, my)

    # ------------------------------------------------------- opp prediction
    def _update_opp_seen(self, logs) -> None:
        for e in logs or []:
            cid = getattr(e, "cardId", None)
            if cid is not None and cid not in self.opp_seen:
                self.opp_seen.append(cid)

    def _pred_opp_deck(self, count: int) -> list[int]:
        pool = list(self.opp_seen) + random.sample(self.deck, len(self.deck))
        while len(pool) < count:
            pool.extend(self.deck)
        return pool[:count]

    def _pred_opp_hand(self, count: int) -> list[int]:
        return self._pred_opp_deck(count + 60)[:count]

    def _pred_opp_prize(self, count: int) -> list[int]:
        return self._pred_opp_deck(count + 60)[:count]

    def _pred_my_prize(self, me_state) -> list[int]:
        known = [c.id for c in me_state.prize if c is not None]
        while len(known) < len(me_state.prize):
            known.append(self.deck[len(known) % len(self.deck)])
        return known

    def _pred_opp_active(self, obs, my) -> list[int]:
        opp = obs.current.players[1 - my]
        if opp.active and opp.active[0] is None:
            for cid in self.deck:
                c = _CARD_DB.get(cid)
                if c is not None and c.basic:
                    return [cid]
        return []

    @staticmethod
    def _read_deck() -> list[int]:
        for path in ("deck.csv", "/kaggle_simulations/agent/deck.csv"):
            if os.path.exists(path):
                rows = [r for r in open(path).read().split("\n") if r.strip()]
                return [int(rows[i]) for i in range(60)]
        return []
