"""能力評価コーパス用の**独立** ground-truth オラクル（Step 1-18）。

Phase 1 / Phase 2 の実装を一切 import しない。
`SearchSession` でエンジンを直接叩き、合法手を素朴に反復深化するだけ。
これにより「solver が正しいと言ったから正しい」という循環を避ける。

保証の範囲（重要・誇張しないこと）:

  * 勝ち筋が **chance を含まない**場合
        -> 与えた信念の下で **厳密な ground truth**。
           再生で `result == me` を実際に確認しているので偽陽性は無い。
  * 勝ち筋が chance（shuffle / 未知順ドロー / コイン / 相手選択）を **含む**場合
        -> 「この 1 つの determinization では勝てる」までしか言えない。
           全 outcome で勝てること（= `PROVEN_WIN`）は主張しない。
           そのため `chance_free` フラグを必ず添えて記録する。

`expected_lethal = False` も同様に「max_depth まで探して見つからなかった」であり、
depth 上限を超える勝ち筋の不存在は主張しない（`searched_to_depth` を記録する）。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import OptionType
from ptcg_ai.search.lethal.engine import EngineError, SearchSession

MAX_BRANCH = 24  # 1 ノードあたりに展開する合法手の上限（爆発を抑える）


@dataclass
class OracleResult:
    """1 局面に対する独立検証の結果。"""

    expected_lethal: bool = False
    minimum_depth: int | None = None
    expected_root_action: list[int] | None = None
    oracle_sequence: list[list[int]] = field(default_factory=list)
    chance_free: bool = False
    capabilities: list[str] = field(default_factory=list)
    searched_to_depth: int = 0
    nodes: int = 0
    elapsed_ms: float = 0.0
    truncated: bool = False
    # near miss 判定用: 勝てなかったが相手のバトルポケモンは倒せた
    can_knock_out_active: bool = False
    # 今ターン中に相手の選択が挟まる線があったか(B4 カテゴリ判定用)
    has_opponent_choice: bool = False
    # 勝利時に自分のサイドが残っていたか(= サイド以外の勝利条件)
    non_prize_win: bool = False

    # --- 指示 5/6/7/17: 保証の強さを混ぜないための区別 -------------------
    # minimum_depth より浅い深さを**打ち切りなしで**全部探索できたか。
    # False なら minimum_depth は真の最小ではなく上界。
    proof_complete: bool = False
    # 全 outcome で確定と言えるか。単一 determinization しか見ていないので
    # chance を含む線では必ず False。
    outcome_complete: bool = False
    # --- 指示 16: B1 / B4 を別々のタグとして持つ ------------------------
    has_root_deck_reveal: bool = False
    has_midturn_deck_reveal: bool = False
    opponent_choice_count: int = 0
    opponent_choice_after_ko: bool = False
    # --- 指示 8: near miss の「不足量」 ---------------------------------
    min_opponent_active_hp: int | None = None
    opponent_bench_count: int = 0
    attach_still_available: bool = False
    # 内部: 打ち切りなしで完走した深さ
    completed_depths: set = field(default_factory=set)

    @property
    def minimum_depth_kind(self) -> str:
        if self.minimum_depth is None:
            return "unknown"
        return "exact" if self.proof_complete else "upper_bound"

    @property
    def ground_truth(self) -> str:
        """指示 17: 保証の強さ。後段の分析で混ぜないためのラベル。"""
        if not self.expected_lethal:
            return "DEPTH_LIMITED"
        if not self.chance_free:
            return "DETERMINIZATION_ONLY"
        if not self.proof_complete:
            return "GROUND_TRUTH_CHANCE_FREE"
        return "GROUND_TRUTH_EXACT"

    def as_dict(self) -> dict:
        return {
            "expected_lethal": self.expected_lethal,
            "minimum_depth": self.minimum_depth,
            "expected_root_action": self.expected_root_action,
            "oracle_sequence": self.oracle_sequence,
            "chance_free": self.chance_free,
            "capabilities": sorted(set(self.capabilities)),
            "searched_to_depth": self.searched_to_depth,
            "oracle_nodes": self.nodes,
            "oracle_ms": round(self.elapsed_ms, 1),
            "truncated": self.truncated,
            "can_knock_out_active": self.can_knock_out_active,
            "has_opponent_choice": self.has_opponent_choice,
            "non_prize_win": self.non_prize_win,
            # 指示 5: oracle_win / searched_to_depth / proof_complete を分けて保存
            "oracle_win": self.expected_lethal,
            "proof_complete": self.proof_complete,
            "outcome_complete": self.outcome_complete,
            "minimum_depth_kind": self.minimum_depth_kind,
            "ground_truth": self.ground_truth,
            "has_root_deck_reveal": self.has_root_deck_reveal,
            "has_midturn_deck_reveal": self.has_midturn_deck_reveal,
            "opponent_choice_count": self.opponent_choice_count,
            "opponent_choice_after_ko": self.opponent_choice_after_ko,
            "min_opponent_active_hp": self.min_opponent_active_hp,
            "opponent_bench_count": self.opponent_bench_count,
            "attach_still_available": self.attach_still_available,
        }


def _legal_actions(observation) -> list[list[int]]:
    """素朴な合法手の列挙。minCount==maxCount==1 以外は先頭から詰める。"""
    select = observation.select
    if select is None:
        return []
    option_count = len(select.option)
    low = max(select.minCount, 0)
    high = min(select.maxCount, option_count)
    if option_count == 0 or high == 0:
        return [[]] if low == 0 else []
    if low <= 1 and high >= 1:
        return [[i] for i in range(min(option_count, MAX_BRANCH))]
    size = low if low > 0 else 1
    return [list(range(min(size, option_count)))]


def _capability_of(observation, action: list[int]) -> list[str]:
    select = observation.select
    if select is None:
        return []
    names = []
    for index in action:
        if 0 <= index < len(select.option):
            names.append(OptionType(int(select.option[index].type)).name)
    return names


def _opponent_active_hp(observation, me: int) -> int | None:
    state = observation.current
    if state is None:
        return None
    actives = state.players[1 - me].active
    if not actives or actives[0] is None:
        return None
    return actives[0].hp


def _looks_like_deck_listing(observation) -> bool:
    """root の選択肢が「山札の中身を並べている」形かを見る（B1 タグ用）。"""
    select = observation.select
    state = observation.current
    if select is None or state is None:
        return False
    hand_ids = {c.id for c in (state.players[state.yourIndex].hand or [])}
    card_options = [o for o in select.option
                    if OptionType(int(o.type)).name == "CARD"]
    if len(card_options) < 3:
        return False
    # 手札に無いカードが選択肢へ並んでいるなら、山札を見せている。
    return any(getattr(o, "cardId", getattr(o, "id", None)) not in hand_ids
               for o in card_options)


def solve(
    observation,
    hidden,
    me: int,
    *,
    max_depth: int = 6,
    time_limit_ms: float = 20_000.0,
    node_limit: int = 60_000,
) -> OracleResult:
    """反復深化で「今ターン確定で勝てる最短手順」を探す。

    solver 側の枝刈り・置換表・証明規則を一切使わない素朴な全探索。
    見つけた手順は探索と同じセッション上で実際に再生済み（= engine replay 検証済み）。
    """
    result = OracleResult()
    started = time.perf_counter()
    prize_before = None

    with SearchSession(observation, hidden) as session:
        root_state = session.root.observation.current
        if root_state is not None:
            prize_before = len(root_state.players[me].prize)

        root_observation = session.root.observation
        if root_observation.select is not None and root_observation.current is not None:
            result.opponent_bench_count = sum(
                1 for x in root_observation.current.players[1 - me].bench
                if x is not None
            )
            kinds = {
                OptionType(int(o.type)).name for o in root_observation.select.option
            }
            result.attach_still_available = "ATTACH" in kinds
            result.has_root_deck_reveal = _looks_like_deck_listing(root_observation)

        for depth in range(1, max_depth + 1):
            result.searched_to_depth = depth
            truncated_before = result.truncated
            found = _search_to_depth(session, session.root, depth, me, result,
                                     started, time_limit_ms, node_limit, prize_before)
            if not result.truncated and not truncated_before:
                result.completed_depths.add(depth)
            if found is not None:
                result.expected_lethal = True
                result.minimum_depth = depth
                result.oracle_sequence = [list(a) for a, _ in found]
                result.expected_root_action = list(found[0][0])
                result.chance_free = all(chance_free for _, chance_free in found)
                # depth 1..N-1 をすべて完走していれば、N は真の最小。
                result.proof_complete = all(
                    d in result.completed_depths for d in range(1, depth)
                )
                result.outcome_complete = result.chance_free
                result.elapsed_ms = (time.perf_counter() - started) * 1000.0
                return result
            if result.truncated:
                break

    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


def _search_to_depth(session, node, depth, me, result, started, time_limit_ms,
                     node_limit, prize_before, parent_turn_ended=False):
    """深さ ``depth`` ちょうどまでの DFS。勝ち手順（action, chance_free）を返す。"""
    if (time.perf_counter() - started) * 1000.0 > time_limit_ms:
        result.truncated = True
        return None
    if result.nodes > node_limit:
        result.truncated = True
        return None

    observation = node.observation
    state = observation.current
    if state is None:
        return None
    if state.result == me:
        if prize_before is not None and len(state.players[me].prize) > 0:
            result.non_prize_win = True
        return []
    if state.result != -1:
        return None
    if depth == 0 or observation.select is None:
        return None
    if state.yourIndex != me:
        # ターン終了後に相手番へ移るのは当たり前なので数えない。
        # **ターン中に**相手の選択が挟まる場合だけが B4 の対象。
        if not parent_turn_ended:
            result.has_opponent_choice = True
            result.opponent_choice_count += 1
            if prize_before is not None and len(state.players[me].prize) < prize_before:
                result.opponent_choice_after_ko = True
        return None

    for action in _legal_actions(observation):
        result.nodes += 1
        try:
            child, events = session.step(node, action)
        except EngineError:
            continue
        chance_free = not (
            events.shuffled or events.coin_heads or events.drawn
        )
        if events.deck_listed_before:
            result.has_midturn_deck_reveal = True
        child_state = child.observation.current
        if child_state is not None and prize_before is not None:
            if len(child_state.players[me].prize) < prize_before:
                result.can_knock_out_active = True
        remaining = _opponent_active_hp(child.observation, me)
        if remaining is not None and remaining > 0:
            result.min_opponent_active_hp = (
                remaining if result.min_opponent_active_hp is None
                else min(result.min_opponent_active_hp, remaining)
            )
        tail = _search_to_depth(session, child, depth - 1, me, result, started,
                                time_limit_ms, node_limit, prize_before,
                                events.turn_ended)
        if tail is not None:
            result.capabilities.extend(_capability_of(observation, action))
            return [(action, chance_free)] + tail
        if result.truncated:
            return None
    return None


def replay_verify(observation, hidden, me: int, sequence: list[list[int]]) -> bool:
    """保存した oracle 手順を実エンジンで再生し、本当に勝てるか確認する。"""
    if not sequence:
        return False
    try:
        with SearchSession(observation, hidden) as session:
            node = session.root
            for action in sequence:
                state = node.observation.current
                if state is None or state.result != -1:
                    break
                node, _events = session.step(node, list(action))
            final = node.observation.current
            return final is not None and final.result == me
    except EngineError:
        return False
