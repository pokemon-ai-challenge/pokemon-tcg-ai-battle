from dataclasses import dataclass, field

from cg.api import EnergyType, Observation


@dataclass(frozen=True)
class PokemonProfile:
    """自デッキ内での役割を簡潔に持つプロフィール。"""

    active_role_bonus: int = 0
    bench_setup_bonus: int = 0


@dataclass(frozen=True)
class AttackEffectProfile:
    """自デッキで使う攻撃の追加効果と副作用を構造化して持つ。"""

    bench_damage: int = 0
    self_damage: int = 0
    discards_attached_energy: int = 0
    prevents_attack_next_turn: bool = False
    applies_special_condition: bool = False
    draws_cards: int = 0
    switches_opponent: bool = False
    requires_coin_flip: bool = False
    damage_depends_on_board: bool = False
    accelerates_energy_to_bench: int = 0
    required_benched_card_ids: frozenset[int] = field(default_factory=frozenset)
    fails_without_requirement: bool = False


@dataclass(frozen=True)
class ItemProfile:
    """グッズの主な役割を再利用しやすい形で持つ。"""

    searches_pokemon: int = 0
    searches_basic_pokemon: int = 0
    searches_basic_energy: int = 0
    recovers_pokemon_from_discard: int = 0
    recovers_basic_energy_from_discard: int = 0
    discard_cost: int = 0
    switches_own_active: bool = False
    active_damage_bonus: int = 0
    only_for_fighting_pokemon: bool = False
    excludes_rule_box: bool = False


@dataclass(frozen=True)
class SupporterProfile:
    """サポートの効果を大まかな行動カテゴリに分解して持つ。"""

    draw_cards: int = 0
    gust_effect: bool = False
    hand_reset: bool = False
    both_players_redraw: bool = False
    can_use_going_first: bool = False
    discard_hand: bool = False
    shuffle_hand_into_deck: bool = False


@dataclass(frozen=True)
class ToolProfile:
    """どうぐによる常在補正を持つ。"""

    retreat_cost_reduction: int = 0
    active_damage_bonus: int = 0
    active_damage_bonus_vs_ex: int = 0


@dataclass(frozen=True)
class StadiumProfile:
    """スタジアムの継続補正を持つ。"""

    stage2_hp_modifier: int = 0
    affects_both_players: bool = True


@dataclass(frozen=True)
class EnergyProfile:
    """基本エネルギーの種類を共通知識として持つ。"""

    provided_energy_type: EnergyType | None = None
    is_basic_energy: bool = False


POKEMON_PROFILES: dict[int, PokemonProfile] = {
    # Mega Lucario ex: このデッキの主力アタッカー。
    678: PokemonProfile(active_role_bonus=24),
    # Hariyama: 中盤以降の高打点要員。
    674: PokemonProfile(active_role_bonus=12),
    # Solrock: 条件付きだが場に出ると打点要員になれる。
    676: PokemonProfile(active_role_bonus=6),
    # Lunatone: Solrock の成立を支える補助寄り。
    675: PokemonProfile(active_role_bonus=3, bench_setup_bonus=6),
    # Riolu / Makuhita: 進化元として最低限の価値を持たせる。
    677: PokemonProfile(active_role_bonus=2, bench_setup_bonus=4),
    673: PokemonProfile(active_role_bonus=2, bench_setup_bonus=3),
}


ATTACK_EFFECTS: dict[int, AttackEffectProfile] = {
    # Solrock: Lunatone がいないと不発。
    980: AttackEffectProfile(
        damage_depends_on_board=True,
        required_benched_card_ids=frozenset({675}),
        fails_without_requirement=True,
    ),
    # Riolu: 次ターン同じ技を使えない。
    981: AttackEffectProfile(prevents_attack_next_turn=True),
    # Mega Lucario ex: 打点に加えてベンチへエネ加速。
    982: AttackEffectProfile(accelerates_energy_to_bench=3),
    # Mega Lucario ex: 高打点だが次ターン制約あり。
    983: AttackEffectProfile(prevents_attack_next_turn=True),
    # Hariyama: 自傷が重い。
    978: AttackEffectProfile(self_damage=70),
}


ITEM_PROFILES: dict[int, ItemProfile] = {
    # トラッシュ 2 枚で任意のポケモンサーチ。
    1121: ItemProfile(searches_pokemon=1, discard_cost=2),
    # ルールボックスなし限定のポケモンサーチ。
    1152: ItemProfile(searches_pokemon=1, excludes_rule_box=True),
    # Fighting デッキ専用のポケモン/基本エネルギーサーチ。
    1142: ItemProfile(
        searches_basic_pokemon=1,
        searches_basic_energy=1,
        only_for_fighting_pokemon=True,
    ),
    # 捨て札からポケモンか基本エネルギーを 1 枚回収。
    1097: ItemProfile(
        recovers_pokemon_from_discard=1,
        recovers_basic_energy_from_discard=1,
    ),
    # 自分のアクティブを入れ替える。
    1123: ItemProfile(switches_own_active=True),
    # Fighting ポケモンのアクティブ打点を一時的に上げる。
    1141: ItemProfile(active_damage_bonus=30, only_for_fighting_pokemon=True),
}


SUPPORTER_PROFILES: dict[int, SupporterProfile] = {
    # 確定 gust。
    1182: SupporterProfile(gust_effect=True),
    # 手札リセットから 5 枚ドロー。先攻 1 ターン目でも使える。
    1192: SupporterProfile(
        draw_cards=5,
        hand_reset=True,
        can_use_going_first=True,
        discard_hand=True,
    ),
    # お互いの手札を戻して 4 枚引かせる。
    1213: SupporterProfile(
        draw_cards=4,
        hand_reset=True,
        both_players_redraw=True,
        shuffle_hand_into_deck=True,
    ),
    # 自分の手札を戻して引き直す主力ドロー。
    1227: SupporterProfile(
        draw_cards=6,
        hand_reset=True,
        shuffle_hand_into_deck=True,
    ),
}


TOOL_PROFILES: dict[int, ToolProfile] = {
    # 逃げエネ軽減。
    1174: ToolProfile(retreat_cost_reduction=2),
    # ex への打点補正。
    1158: ToolProfile(active_damage_bonus_vs_ex=50),
}


STADIUM_PROFILES: dict[int, StadiumProfile] = {
    # お互いの Stage 2 を脆くする。
    1252: StadiumProfile(stage2_hp_modifier=-30, affects_both_players=True),
}


ENERGY_PROFILES: dict[int, EnergyProfile] = {
    6: EnergyProfile(provided_energy_type=EnergyType.FIGHTING, is_basic_energy=True),
}


def get_pokemon_profile(card_id: int) -> PokemonProfile | None:
    # 他の評価ロジックから「このカードは主力か」を引く入口。
    return POKEMON_PROFILES.get(card_id)


def get_attack_effect_profile(attack_id: int) -> AttackEffectProfile | None:
    # 攻撃評価だけでなく、将来のシミュレーション補助からも使う想定。
    return ATTACK_EFFECTS.get(attack_id)


def get_item_profile(card_id: int) -> ItemProfile | None:
    # グッズの役割知識を 1 箇所から引けるようにする。
    return ITEM_PROFILES.get(card_id)


def get_supporter_profile(card_id: int) -> SupporterProfile | None:
    # ドロー・gust・手札リセットなどの分類をここから参照する。
    return SUPPORTER_PROFILES.get(card_id)


def get_tool_profile(card_id: int) -> ToolProfile | None:
    return TOOL_PROFILES.get(card_id)


def get_stadium_profile(card_id: int) -> StadiumProfile | None:
    return STADIUM_PROFILES.get(card_id)


def get_energy_profile(card_id: int) -> EnergyProfile | None:
    return ENERGY_PROFILES.get(card_id)


def pokemon_active_role_bonus(card_id: int) -> int:
    # バトル場に出したいカードほど高い補正を返す。
    profile = get_pokemon_profile(card_id)
    if profile is None:
        return 0
    return profile.active_role_bonus


def pokemon_bench_setup_bonus(card_id: int) -> int:
    # ベンチに置いて価値があるカードの補助指標。
    profile = get_pokemon_profile(card_id)
    if profile is None:
        return 0
    return profile.bench_setup_bonus


def attack_profile_requirements_met(
    obs: Observation,
    profile: AttackEffectProfile,
) -> bool:
    # 「Lunatone がいるときだけ有効」などの盤面条件をここで共通判定する。
    if obs.current is None:
        return False
    if not profile.required_benched_card_ids:
        return True

    player = obs.current.players[obs.current.yourIndex]
    benched_ids = {pokemon.id for pokemon in player.bench}
    return profile.required_benched_card_ids.issubset(benched_ids)
