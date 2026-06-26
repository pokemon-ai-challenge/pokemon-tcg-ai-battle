"""BoardEvaluator: 盤面状態をスコアリングして行動選択に使う。

評価値 =
  side_lead_score           サイドリード（相手prizes - 自分prizes）
+ attack_ready_score        アクティブが攻撃可能か
+ bench_attacker_score      ベンチに攻撃準備できたアタッカーがいるか
+ evolution_progress_score  進化完成度
+ hand_score                手札の質
+ active_hp_score           アクティブHP割合（低いほど危険）
+ bench_stability_score     ベンチの厚さ
- prize_risk_score          アクティブKO時に相手が取るサイド価値
- deck_depletion_score      山札残り枚数リスク

Phase 3 での主な利用箇所:
  - choose_main_action: リトリートすべき局面の判定
  - 将来 (Phase 4): search_begin の候補枝刈り
"""
from dataclasses import dataclass
from cg.api import State, Pokemon
from src.knowledge.card_database import get_card_db, get_attack_db
from src.knowledge.deck_plan import DeckPlan
from src.decision.risk_evaluator import will_be_ko_next_turn


@dataclass
class EvalWeights:
    side_lead: float = 40.0           # サイドリード1枚差あたりの点数
    attack_ready: float = 25.0        # アクティブが今ターン攻撃できる
    bench_attacker_ready: float = 20.0 # ベンチに攻撃可能アタッカーがいる
    evolution_progress: float = 15.0  # 進化完成度（0.0〜1.0 の割合）
    hand_card: float = 2.0            # 手札1枚あたりの加点
    active_hp_ratio: float = 20.0     # アクティブ HP 割合（1.0 = 満タン）
    bench_count: float = 3.0          # ベンチポケモン1体あたり（DOWN: ベンチ過剰展開を防ぐ）
    prize_risk: float = 35.0          # KO時のサイド価値×HP消費割合
    deck_low_penalty: float = 5.0     # 山札残り10枚以下の1枚あたりペナルティ
    opponent_requirement: float = 1.0 # 相手の要求値（高いほど自分の壁が厚い）ARC-4


DEFAULT_WEIGHTS = EvalWeights()


def _get_energy_needed_for_attack(card_id: int) -> int:
    """カードが最も少ないエネルギーで使える攻撃に必要なエネルギー枚数（概算）。"""
    card_db = get_card_db()
    attack_db = get_attack_db()
    card = card_db.get(card_id)
    if not card:
        return 2  # デフォルト
    if not card.attacks:
        return 99
    min_cost = min(
        len(attack_db[atk_id].energies) if atk_id in attack_db else 2
        for atk_id in card.attacks
    )
    return max(1, min_cost)


def _prize_count(player_state) -> int:
    return len([c for c in player_state.prize if c is not None])


class BoardEvaluator:
    """盤面評価クラス。同じインスタンスを使い回すことで DB ロードを省く。"""

    def __init__(self, weights: EvalWeights = DEFAULT_WEIGHTS) -> None:
        self.w = weights

    def score(self, state: State, plan: DeckPlan) -> float:
        """現在の盤面スコアを返す。高いほど自分に有利。"""
        my_idx = state.yourIndex
        opp_idx = 1 - my_idx
        my = state.players[my_idx]
        opp = state.players[opp_idx]
        card_db = get_card_db()

        total = 0.0

        # 1. サイドリード: 相手prizeが多い（取られていない）= 自分が有利
        my_prizes = _prize_count(my)
        opp_prizes = _prize_count(opp)
        total += (opp_prizes - my_prizes) * self.w.side_lead

        # 2. アクティブ攻撃準備スコア
        my_active: Pokemon | None = my.active[0] if my.active else None
        if my_active:
            needed = _get_energy_needed_for_attack(my_active.id)
            have = len(my_active.energies)
            ratio = min(1.0, have / max(1, needed))
            total += ratio * self.w.attack_ready

        # 3. ベンチアタッカー準備スコア
        for bench_poke in my.bench:
            if bench_poke.id in plan.main_attacker_ids or bench_poke.id in plan.sub_attacker_ids:
                needed = _get_energy_needed_for_attack(bench_poke.id)
                have = len(bench_poke.energies)
                if have >= needed:
                    total += self.w.bench_attacker_ready
                    break

        # 4. 進化完成度
        all_my_pkmn: list[Pokemon] = [p for p in my.active if p] + my.bench
        if all_my_pkmn:
            evolved = sum(
                1 for p in all_my_pkmn
                if p.id in plan.main_attacker_ids or p.id in plan.sub_attacker_ids
            )
            total += (evolved / len(all_my_pkmn)) * self.w.evolution_progress

        # 5. 手札の質（枚数で近似）
        hand_count = len(my.hand) if my.hand else my.handCount
        total += hand_count * self.w.hand_card

        # 6. アクティブ HP 割合
        if my_active and my_active.maxHp > 0:
            hp_ratio = my_active.hp / my_active.maxHp
            total += hp_ratio * self.w.active_hp_ratio

        # 7. ベンチの厚さ
        total += len(my.bench) * self.w.bench_count

        # 8. 賞金リスク: アクティブがKOされると相手が取るサイド価値
        if my_active:
            card = card_db.get(my_active.id)
            if card:
                prizes_on_ko = 3 if card.megaEx else (2 if card.ex else 1)
                next_turn_ko = will_be_ko_next_turn(state, my_active)
                if next_turn_ko:
                    # 次ターンKO確定: フルペナルティ
                    total -= prizes_on_ko * self.w.prize_risk * 1.5
                    # エネルギー喪失コスト: KOされると付いているエネルギーが失われる
                    energy_loss = len(my_active.energies)
                    total -= energy_loss * 8.0
                    # ベンチ消滅リスク: ベンチが空なら負け確定に近い
                    if len(my.bench) == 0:
                        total -= 100.0
                elif my_active.maxHp > 0:
                    damage_taken_ratio = 1.0 - (my_active.hp / my_active.maxHp)
                    total -= prizes_on_ko * damage_taken_ratio * self.w.prize_risk

        # 9. 山札枯渇リスク
        if my.deckCount < 10:
            total -= (10 - my.deckCount) * self.w.deck_low_penalty

        # 10. 相手の要求値（ARC-4）
        total += self.opponent_requirement_score(state, plan) * self.w.opponent_requirement

        return total

    def opponent_requirement_score(self, state: State, plan: DeckPlan) -> float:
        """相手が自陣を突破するために必要なリソース（要求値）の高さを返す（ARC-4）。

        高いほど相手が倒しにくい盤面 → 自分に有利。
        計算要素:
          1. アクティブが非EX → 相手は複数回攻撃が必要になる（ボス不要）
          2. アクティブのHP残量が多い
          3. ベンチが少ない → 散布ダメージの標的が少ない
        """
        my_idx = state.yourIndex
        opp_idx = 1 - my_idx
        my = state.players[my_idx]
        card_db = get_card_db()
        attack_db = get_attack_db()

        total = 0.0

        active: Pokemon | None = my.active[0] if my.active else None
        if active:
            active_card = card_db.get(active.id)
            prizes_on_ko = 1
            if active_card:
                prizes_on_ko = 3 if active_card.megaEx else (2 if active_card.ex else 1)

            if prizes_on_ko == 1:
                # 非EX：相手の最大ダメージ期待値から KO に必要なターン数を推定
                opp_active = state.players[opp_idx].active
                max_opp_dmg = 0
                if opp_active and opp_active[0]:
                    opp_card = card_db.get(opp_active[0].id)
                    if opp_card and opp_card.attacks:
                        for atk_id in opp_card.attacks:
                            atk = attack_db.get(atk_id)
                            if atk:
                                max_opp_dmg = max(max_opp_dmg, atk.damage)
                if max_opp_dmg > 0 and active.hp > 0:
                    # 切り上げ除算: KO に何回攻撃が必要か
                    hits_to_ko = (active.hp + max_opp_dmg - 1) // max_opp_dmg
                    # 2回以上必要なら相手が追加ターンを消費する分だけ有利
                    total += max(0, hits_to_ko - 1) * 15.0

            # HP が多いほど要求値が高い（絶対値で評価）
            total += active.hp * 0.03

        # ベンチが少ない → 散布ダメージ被害減
        bench_size = len(my.bench)
        if bench_size <= 2:
            total += (3 - bench_size) * 4.0

        return total

    def should_retreat(self, state: State, plan: DeckPlan) -> bool:
        """アクティブを引っ込めるべきかを判断する。

        条件:
          - アクティブ HP が maxHp の 25% 未満
          - かつベンチに攻撃準備できたアタッカーがいる
        """
        my_idx = state.yourIndex
        my = state.players[my_idx]

        active: Pokemon | None = my.active[0] if my.active else None
        if not active or active.maxHp <= 0:
            return False

        hp_ratio = active.hp / active.maxHp
        if hp_ratio >= 0.25:
            return False

        # ベンチに攻撃できるアタッカーがいるか
        for bench_poke in my.bench:
            if bench_poke.id in plan.main_attacker_ids or bench_poke.id in plan.sub_attacker_ids:
                needed = _get_energy_needed_for_attack(bench_poke.id)
                if len(bench_poke.energies) >= needed:
                    return True

        return False

    def best_attack_option(
        self,
        state: State,
        attack_options: list[int],
        options,
    ) -> int:
        """攻撃選択肢のうち最善を返す（lethal優先、次にKO後の盤面評価）。

        現在は damage ベース選択（target_policy.choose_attack と同等）。
        Phase 4 で search_begin を使った深い評価に置き換える。
        """
        from src.knowledge.card_database import get_attack_db
        attack_db = get_attack_db()

        my_idx = state.yourIndex
        opp_idx = 1 - my_idx
        opp_active = state.players[opp_idx].active
        opp_hp: int | None = None
        if opp_active and opp_active[0]:
            opp_hp = opp_active[0].hp

        if opp_hp is not None:
            for i in attack_options:
                opt = options[i]
                if opt.attackId and opt.attackId in attack_db:
                    if attack_db[opt.attackId].damage >= opp_hp:
                        return i

        best_i = attack_options[0]
        best_dmg = -1
        for i in attack_options:
            opt = options[i]
            if opt.attackId and opt.attackId in attack_db:
                dmg = attack_db[opt.attackId].damage
                if dmg > best_dmg:
                    best_dmg = dmg
                    best_i = i
        return best_i


# モジュールレベルのシングルトン（使い回す）
_evaluator = BoardEvaluator()


def get_evaluator() -> BoardEvaluator:
    return _evaluator
