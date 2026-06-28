from dataclasses import dataclass
import re

from cg.api import Observation, OptionType, Pokemon

from src.decision.evaluation.attack_features import build_attack_lookup
from src.knowledge.deck_profiles import (
    AttackEffectProfile,
    attack_profile_requirements_met,
    get_attack_effect_profile,
)
from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


@dataclass(frozen=True)
class AttackOptionScore:
    option_index: int
    knock_out: bool
    immediate_damage: int
    overflow_damage: int
    effect_score: int

    @property
    def pressure_score(self) -> int:
        return self.immediate_damage + self.effect_score

    def sort_key(self) -> tuple[int, int, int, int, int]:
        # KO できるなら、余剰ダメージの前に効果と副作用の差を見る。
        if self.knock_out:
            return (
                1,
                self.effect_score,
                -self.overflow_damage,
                self.immediate_damage,
                -self.option_index,
            )

        # KO できないなら、有効打点と追加効果込みの圧力で比較する。
        return (
            0,
            self.pressure_score,
            self.effect_score,
            self.immediate_damage,
            -self.option_index,
        )


def choose_best_attack_option(
    obs: Observation,
    option_indexes: list[int] | None = None,
) -> int | None:
    """Choose an attack with attackId registry first, text fallback second."""
    if obs.select is None:
        raise ValueError("obs.select must not be None when choosing an attack.")

    attack_by_id = build_attack_lookup()
    target = resolve_opponent_active(obs)
    candidate_indexes = (
        list(range(len(obs.select.option)))
        if option_indexes is None
        else option_indexes
    )

    best_score: AttackOptionScore | None = None
    for option_index in candidate_indexes:
        # MAIN 文脈から attack 候補の部分集合を渡すことがあるので、
        # 範囲外インデックスは静かに無視する。
        if option_index < 0 or option_index >= len(obs.select.option):
            continue

        option = obs.select.option[option_index]
        if option.type != OptionType.ATTACK or option.attackId is None:
            continue

        attack = attack_by_id.get(option.attackId)
        if attack is None:
            continue

        # 即時打点と追加効果を分けて見て、最後に 1 つのスコアへまとめる。
        score = score_attack_option(
            obs=obs,
            option_index=option_index,
            attack_id=attack.attackId,
            printed_damage=attack.damage,
            attack_text=attack.text,
            target=target,
        )
        if best_score is None or score.sort_key() > best_score.sort_key():
            best_score = score

    return None if best_score is None else best_score.option_index


def propose_attack_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """MAIN 中の attack 候補から最良の 1 手を出す。"""
    best_attack_option = choose_best_attack_option(obs, buckets.attack)
    if best_attack_option is None:
        return None

    return MainActionProposal(
        action=[best_attack_option],
        score=MAIN_ACTION_BASE_WEIGHTS["attack"],
        label="attack",
    )


def score_attack_option(
    obs: Observation,
    option_index: int,
    attack_id: int,
    printed_damage: int,
    attack_text: str | None,
    target: Pokemon | None,
) -> AttackOptionScore:
    profile = get_attack_effect_profile(attack_id)
    # 既知技は profile ベース、未知技は後段で軽いフォールバック評価に落とす。
    immediate_damage = resolve_immediate_damage(obs, printed_damage, profile)
    effect_score = score_attack_effect(profile, attack_text)
    target_hp = target.hp if target is not None else None
    knock_out = target_hp is not None and immediate_damage >= target_hp
    overflow_damage = 0 if target_hp is None else max(0, immediate_damage - target_hp)

    return AttackOptionScore(
        option_index=option_index,
        knock_out=knock_out,
        immediate_damage=immediate_damage,
        overflow_damage=overflow_damage,
        effect_score=effect_score,
    )


def resolve_immediate_damage(
    obs: Observation,
    printed_damage: int,
    profile: AttackEffectProfile | None,
) -> int:
    if profile is None:
        return printed_damage

    # 盤面条件を満たさない技は、まず不発として扱う。
    if profile.fails_without_requirement and not profile_requirements_met(obs, profile):
        return 0

    return printed_damage


def profile_requirements_met(
    obs: Observation,
    profile: AttackEffectProfile,
) -> bool:
    # 盤面条件の実処理は knowledge 層へ寄せて、他判定からも再利用する。
    return attack_profile_requirements_met(obs, profile)


def resolve_opponent_active(obs: Observation) -> Pokemon | None:
    if obs.current is None:
        return None

    # 攻撃評価では、いま見えている相手バトル場だけを基準にする。
    opponent_index = 1 - obs.current.yourIndex
    active = obs.current.players[opponent_index].active
    if not active:
        return None
    return active[0]


def score_attack_effect(
    profile: AttackEffectProfile | None,
    attack_text: str | None,
) -> int:
    # 既知技は構造化プロフィール、未知技だけテキストを読む。
    if profile is not None:
        return score_attack_profile(profile)
    return score_unknown_attack_text(attack_text)


def score_attack_profile(profile: AttackEffectProfile) -> int:
    score = 0

    # 追加で盤面が良くなる要素は加点する。
    if profile.bench_damage > 0:
        score += 6 + min(profile.bench_damage // 10, 6)
    if profile.applies_special_condition:
        score += 6
    if profile.draws_cards > 0:
        score += min(profile.draws_cards * 2, 6)
    if profile.switches_opponent:
        score += 6
    if profile.accelerates_energy_to_bench > 0:
        score += 8 + min(profile.accelerates_energy_to_bench * 2, 6)

    # 自分の盤面を削る要素や不安定要素は減点する。
    if profile.requires_coin_flip:
        score -= 6
    if profile.discards_attached_energy > 0:
        score -= profile.discards_attached_energy * 8
    if profile.self_damage > 0:
        score -= max(4, profile.self_damage // 10)
    if profile.prevents_attack_next_turn:
        score -= 14

    return score


def score_unknown_attack_text(text: str | None) -> int:
    # 未登録技は強く信じすぎず、弱いヒューリスティックだけ返す。
    normalized = normalize_attack_text(text)
    if not normalized:
        return 0
    return bonus_score_from_text(normalized) - penalty_score_from_text(normalized)


def normalize_attack_text(text: str | None) -> str:
    if not text:
        return ""

    # 効果文の表記ゆれや文字化けを少し吸収してから判定する。
    return (
        text.replace("Pokémon", "Pokemon")
        .replace("pokémon", "pokemon")
        .replace("Pok\u00e9mon", "Pokemon")
        .replace("pok\u00e9mon", "pokemon")
        .replace("’", "'")
        .replace("\xa0", " ")
        .lower()
    )


def bonus_score_from_text(text: str) -> int:
    score = 0

    # 未登録技だけの暫定フォールバックなので、読む項目は最小限にする。
    if "your opponent's benched pokemon" in text:
        score += 10
    if "is now paralyzed" in text:
        score += 14
    if "is now asleep" in text or "is now confused" in text:
        score += 10
    if "is now poisoned" in text or "is now burned" in text:
        score += 8
    if "discard an energy from your opponent's active pokemon" in text:
        score += 10
    if "draw " in text:
        score += 6
    if "search your deck" in text:
        score += 6
    if "attach" in text and "energy" in text:
        score += 8

    return score


def penalty_score_from_text(text: str) -> int:
    score = 0

    # テキストフォールバックでも、重い副作用だけは見落とさないようにする。
    if "during your next turn, this pokemon can't use attacks" in text:
        score += 14
    elif "during your next turn, this pokemon can't use " in text:
        score += 12

    if "discard all energy from this pokemon" in text:
        score += 18
    elif re.search(r"discard (?:a|\d+) .*energy from this pokemon", text):
        score += 10

    self_damage_match = re.search(r"this pokemon(?: also)? does (\d+) damage to itself", text)
    if self_damage_match:
        score += max(4, int(self_damage_match.group(1)) // 10)

    return score
