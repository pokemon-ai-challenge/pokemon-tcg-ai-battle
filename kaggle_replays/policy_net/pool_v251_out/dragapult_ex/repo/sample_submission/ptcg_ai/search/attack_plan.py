"""Conservative, opt-in post-validation for zero-damage attacks."""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable

from cg import api as cg_api
from cg.api import AreaType, LogType, OptionType, SelectType

DEFAULTS = {"enabled": False, "time_limit_ms": 100, "max_root_actions": 8,
            "max_combinations_per_select": 32, "shadow_only": True,
            "zero_damage_threshold": 0}
_ZERO = {"searches": 0, "attack_selections": 0, "zero_damage_attacks": 0,
         "meaningless_attacks": 0, "v0_alternatives": 0, "v1_rescues": 0,
         "shadow_candidates": 0, "timeouts": 0, "total_time_ms": 0.0,
         "max_time_ms": 0.0}
_stats = dict(_ZERO)
_durations: list[float] = []

class _Timeout(Exception): pass

@dataclass(frozen=True)
class AttackOutcome:
    direct_damage: int
    ko: bool = False
    prize_taken: bool = False
    won: bool = False
    useful_side_effect: bool = False
    coin_seen: bool = False
    @property
    def meaningful(self): return self.ko or self.prize_taken or self.won or self.useful_side_effect

@dataclass
class TransactionResult:
    node: object
    search_ids: list[int]
    logs: list

def reset_stats():
    _stats.update(_ZERO); _durations.clear()
def get_stats():
    out = dict(_stats); out["avg_time_ms"] = out["total_time_ms"] / out["searches"] if out["searches"] else 0.0
    values = sorted(_durations)
    for p in (50, 95, 99): out[f"p{p}_time_ms"] = values[min(len(values)-1, int((len(values)-1)*p/100))] if values else 0.0
    return out

def search(state, legal_actions, context):
    """Return a v0/v1 alternative, or None; timeout/uncertainty are rejected."""
    try:
        cfg = {**DEFAULTS, **(context.get("config") or {})}; obs = context.get("observation"); chosen = context.get("chosen_action")
        factory = _factory(context)
        if not cfg["enabled"] or not obs or not obs.select or not obs.current or not factory or not _is_attack(obs.select, chosen): return None
        _stats["searches"] += 1; _stats["attack_selections"] += 1
        started = time.perf_counter(); deadline = started + float(cfg["time_limit_ms"])/1000
        try:
            initial = _evaluate_attack(obs, chosen, state.yourIndex, factory, deadline)
            if initial is None or initial.coin_seen: return None
            if initial.direct_damage <= cfg["zero_damage_threshold"]: _stats["zero_damage_attacks"] += 1
            if not _meaningless(initial): return None
            _stats["meaningless_attacks"] += 1; scores = list(context.get("policy_scores") or [])
            action = _v0(obs, state.yourIndex, factory, scores, cfg, deadline)
            if action is not None: _stats["v0_alternatives"] += 1
            else:
                action = _v1(obs, chosen, state.yourIndex, factory, scores, cfg, deadline)
                if action is not None: _stats["v1_rescues"] += 1
            if action is not None and cfg["shadow_only"]: _stats["shadow_candidates"] += 1; return None
            return action
        except _Timeout:
            _stats["timeouts"] += 1; return None
        finally:
            elapsed = (time.perf_counter()-started)*1000; _stats["total_time_ms"] += elapsed; _stats["max_time_ms"] = max(_stats["max_time_ms"], elapsed)
            _durations.append(elapsed)
            if len(_durations)>4096: _durations.pop(0)
            try: cg_api.search_end()
            except Exception: pass
    except Exception: return None

def _factory(context):
    if context.get("hidden_state_factory") is not None: return context["hidden_state_factory"]
    hidden = context.get("hidden_state"); return (lambda: hidden) if hidden is not None else None
def _is_attack(select, action):
    return isinstance(action,list) and len(action)==1 and isinstance(action[0],int) and 0<=action[0]<len(select.option) and select.option[action[0]].type==OptionType.ATTACK
def _meaningless(outcome): return outcome.direct_damage <= 0 and not outcome.meaningful
def _check(deadline):
    if time.perf_counter() > deadline: raise _Timeout()
def _begin(obs, h): return cg_api.search_begin(obs,h["your_deck"],h["your_prize"],h["opponent_deck"],h["opponent_prize"],h["opponent_hand"],h["opponent_active"])
def _release(ids):
    for i in reversed(ids):
        try: cg_api.search_release(i)
        except Exception: pass
def _has_coin(logs): return any(getattr(log,"type",None)==LogType.COIN for log in logs)
def _is_deck_touching(logs):
    for log in logs:
        if getattr(log,"type",None) in (LogType.SHUFFLE,LogType.DRAW,LogType.DRAW_REVERSE): return True
        if getattr(log,"type",None) in (LogType.MOVE_CARD,LogType.MOVE_CARD_REVERSE) and (getattr(log,"fromArea",None)==AreaType.DECK or getattr(log,"toArea",None)==AreaType.DECK): return True
    return False
def _all(player): return {p.serial:p for p in [*player.active,*player.bench] if p is not None}
def _loss(before, after):
    after = {p.serial:p for p in after if p is not None}; return sum(max(0,p.hp-after[p.serial].hp) for p in before if p is not None and p.serial in after)
def _energy_removed(before,after,opponent):
    old={c.serial for p in _all(before.players[opponent]).values() for c in p.energyCards}; new={c.serial for p in _all(after.players[opponent]).values() for c in p.energyCards}; return bool(old-new)

def _evaluate_attack(obs, selection, me, factory, deadline):
    _check(deadline); hidden=factory()
    if hidden is None:return None
    root=_begin(obs,hidden); ids=[root.searchId]
    try:return _resolve_attack_damage(root,selection,me,deadline,ids)
    finally:_release(ids)
def _resolve_attack_damage(node,selection,me,deadline,ids):
    _check(deadline); before=node.observation.current
    try:child=cg_api.search_step(node.searchId,selection)
    except (ValueError,RuntimeError):return None
    ids.append(child.searchId); after=child.observation.current
    if before is None or after is None:return None
    enemy=1-me; logs=child.observation.logs or []; statuses=("poisoned","burned","asleep","paralyzed","confused")
    direct=_loss(before.players[enemy].active,after.players[enemy].active); bench=_loss(before.players[enemy].bench,[*after.players[enemy].active,*after.players[enemy].bench])
    ko=any(s not in _all(after.players[enemy]) for s in _all(before.players[enemy])); prize=len(after.players[me].prize)<len(before.players[me].prize)
    status=any(not getattr(before.players[enemy],x) and getattr(after.players[enemy],x) for x in statuses)
    return AttackOutcome(direct,ko,prize,after.result==me,bool(bench or status or _energy_removed(before,after,enemy) or after.players[me].handCount>before.players[me].handCount),_has_coin(logs))
def _score(scores,i):
    try:return float(scores[i])
    except (IndexError,TypeError,ValueError):return 0.0
def _v0(obs,me,factory,scores,cfg,deadline):
    bad=set()
    for i,opt in enumerate(obs.select.option):
        if opt.type==OptionType.ATTACK:
            result=_evaluate_attack(obs,[i],me,factory,deadline)
            if result is not None and not result.coin_seen and _meaningless(result):bad.add(i)
    choices=[i for i,opt in enumerate(obs.select.option) if i not in bad and opt.type!=OptionType.END]
    return [max(choices,key=lambda i:(_score(scores,i),-i))] if bad and choices else None

def _v1(obs,attack,me,factory,scores,cfg,deadline):
    results=[]
    for prep in _prep(obs.select,cfg):
        trans=_start_transaction(obs,prep,me,factory,cfg,deadline)
        if trans is None:continue
        try:
            end=trans.node.observation
            if (end.select is None or end.select.deck is not None or _is_deck_touching(trans.logs)
                    or any(end.current.players[i].deckCount != obs.current.players[i].deckCount for i in (0, 1))):continue
            out=_resolve_attack_damage(trans.node,attack,me,deadline,trans.search_ids)
            if out is None or out.coin_seen or not(out.direct_damage>0 or out.ko or out.prize_taken or out.won):continue
            results.append(((int(out.won),int(out.prize_taken),int(out.ko),out.direct_damage,_score(scores,prep[0]),-prep[0]),prep))
        finally:_release(trans.search_ids)
    return max(results,key=lambda x:x[0])[1] if results else None
def _prep(select,cfg):
    if select.type!=SelectType.MAIN:return
    count=0
    for i,opt in enumerate(select.option):
        if opt.type in (OptionType.ATTACK,OptionType.END) or not(select.minCount<=1<=select.maxCount):continue
        yield [i];count+=1
        if count>=int(cfg["max_root_actions"]):return
def _start_transaction(obs,first,me,factory,cfg,deadline):
    _check(deadline);h=factory()
    if h is None:return None
    root=_begin(obs,h);ids=[root.searchId]
    try:
        result=_transaction(root,first,me,cfg,deadline,ids,[])
        if result is None:_release(ids)
        return result
    except _Timeout:_release(ids);raise
    except Exception:_release(ids);return None
def _transaction(node,selection,me,cfg,deadline,ids,logs):
    _check(deadline)
    try:child=cg_api.search_step(node.searchId,selection)
    except (ValueError,RuntimeError):return None
    ids.append(child.searchId);state=child.observation.current;all_logs=[*logs,*(child.observation.logs or [])]
    if state is None or _has_coin(all_logs) or _is_deck_touching(all_logs):return None
    if state.result!=-1:return TransactionResult(child,ids,all_logs)
    select=child.observation.select
    if state.yourIndex!=me or select is None or select.deck is not None:return None
    if select.type==SelectType.MAIN:return TransactionResult(child,ids,all_logs)
    made=0
    for n in range(max(0,select.minCount),min(select.maxCount,len(select.option))+1):
        for combo in itertools.combinations(range(len(select.option)),n):
            mark=len(ids);result=_transaction(child,list(combo),me,cfg,deadline,ids,all_logs)
            if result is not None:return result
            _release(ids[mark:]);del ids[mark:];made+=1
            if made>=int(cfg["max_combinations_per_select"]):return None
    return None


