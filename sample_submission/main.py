import os
from cg.api import (
    Observation, SelectContext, OptionType, CardType, AreaType,
    all_card_data, all_attack, to_observation_class,
)

# ============================================================
# RL モデルのロード（学習済み重みがあれば使用）
# ============================================================
_RL_MODEL = None
_RL_DEVICE = None

def _try_load_rl_model():
    global _RL_MODEL, _RL_DEVICE
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl", "weights.pt")
    if not os.path.exists(weights_path):
        return
    try:
        import torch
        from rl.model import PokemonRLModel
        device = torch.device("cpu")
        ckpt = torch.load(weights_path, map_location=device)
        model = PokemonRLModel().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _RL_MODEL = model
        _RL_DEVICE = device
        print(f"[RL] Loaded model from epoch {ckpt.get('epoch', '?')}")
    except Exception as e:
        print(f"[RL] Failed to load model: {e}")

_try_load_rl_model()


def _rl_choose_action(obs: Observation) -> list[int] | None:
    """RL モデルで行動選択。モデルがなければ None を返す。"""
    if _RL_MODEL is None:
        return None
    try:
        import torch
        from rl.encoder import encode_state, encode_option
        state_feat = encode_state(obs)
        opt_feats = [encode_option(opt, obs) for opt in obs.select.option]
        if not opt_feats:
            return None
        min_c = obs.select.minCount
        max_c = obs.select.maxCount
        s_t = torch.tensor(state_feat, dtype=torch.float32, device=_RL_DEVICE)
        o_t = torch.tensor(opt_feats, dtype=torch.float32, device=_RL_DEVICE)
        chosen = []
        available = list(range(len(opt_feats)))
        for _ in range(max(min_c, 1)):
            if not available:
                break
            avail_feats = torch.stack([o_t[i] for i in available])
            idx_in_avail, _, _ = _RL_MODEL.act(s_t, avail_feats, greedy=True)
            global_idx = available[idx_in_avail]
            chosen.append(global_idx)
            available.remove(global_idx)
            if len(chosen) >= max_c:
                break
        return chosen if chosen else None
    except Exception as e:
        print(f"[RL] act error: {e}")
        return None


# ============================================================
# カードID定数
# ============================================================

# ポケモン
ABRA       = 741   # ケーシィ
KADABRA    = 742   # ユンゲラー
ALAKAZAM   = 743   # フーディン
DUNSPARCE  = 65    # ノコッチ
DUDUNSPARCE = 66   # ノコ²チ
TOGEPI     = 959   # トゲピー
TOGEKISS   = 214   # トゲキッス
SHAYMIN    = 343   # シェイミ
PSYDUCK    = 858   # コダック
FAN_ROTOM  = 174   # スピンロトム

ALAKAZAM_LINE = {ABRA, KADABRA, ALAKAZAM}

# ベンチ要員の優先順位（低インデックスほど優先）
BENCH_PRIORITY = [ABRA, DUNSPARCE, TOGEPI, SHAYMIN, PSYDUCK, FAN_ROTOM, KADABRA, ALAKAZAM]

# せいなるはいで救いたいポケモン
SACRED_ASH_TARGETS = {ABRA, KADABRA, ALAKAZAM, DUNSPARCE, DUDUNSPARCE, TOGEPI, TOGEKISS}

# グッズ
RARE_CANDY       = 1079  # ふしぎなアメ
ENHANCED_HAMMER  = 1081  # 改造ハンマー
BUDDY_POFFIN     = 1086  # なかよしポフィン
NIGHT_STRETCHER  = 1097  # 夜のタンカ
SACRED_ASH       = 1129  # せいなるはい
WONDROUS_PATCH   = 1146  # ワンダーパッチ
POKE_PAD         = 1152  # ポケパッド

# サポート
BOSS_ORDERS = 1182  # ボスの指令
LANAS_AID   = 1184  # スイレンのお世話
HILDA       = 1225  # トウコ
DAWN        = 1231  # ヒカリ

# スタジアム
BATTLE_CAGE = 1264  # バトルコロシアム

# エネルギー（優先度順）
ENRICHING        = 13   # リッチエネルギー
TELEPATH_PSYCHIC = 19   # テレパス超エネルギー
BASIC_PSYCHIC    = 5    # 基本超エネルギー
ENERGY_PRIORITY  = {ENRICHING: 0, TELEPATH_PSYCHIC: 1, BASIC_PSYCHIC: 2}
ALL_ENERGY_IDS   = set(ENERGY_PRIORITY.keys())

# ============================================================
# カードデータキャッシュ
# ============================================================

_CARD_DATA: dict = {}
_ATTACK_DATA: dict = {}

def _get_card_data() -> dict:
    global _CARD_DATA
    if not _CARD_DATA:
        _CARD_DATA = {cd.cardId: cd for cd in all_card_data()}
    return _CARD_DATA

def _get_attack_data() -> dict:
    global _ATTACK_DATA
    if not _ATTACK_DATA:
        _ATTACK_DATA = {a.attackId: a for a in all_attack()}
    return _ATTACK_DATA

def _max_energy_needed(card_id: int) -> int:
    """そのポケモンのワザを使うために必要な最大エネルギー数。"""
    cd = _get_card_data().get(card_id)
    if not cd or not cd.attacks:
        return 0
    attacks = _get_attack_data()
    costs = [len(attacks[aid].energies) for aid in cd.attacks if aid in attacks]
    return max(costs) if costs else 0

# deck.csvからカードのIDを取得してリストを返す
def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as f:
        lines = f.read().split("\n")
    return [int(lines[i]) for i in range(60)]


def validate_choice(obs: Observation, chosen: list[int]) -> None:
    select = obs.select
    if not (select.minCount <= len(chosen) <= select.maxCount):
        raise ValueError(
            f"Action length must be between {select.minCount} and {select.maxCount}, got {len(chosen)}."
        )
    if len(chosen) != len(set(chosen)):
        raise ValueError("Duplicate select elements are not allowed.")
    if not all(0 <= i < len(select.option) for i in chosen):
        raise ValueError("Each selected index must be within the option range.")


def choose_minimum_required(obs: Observation) -> list[int]:
    return list(range(obs.select.minCount))

# ============================================================
# 盤面情報ヘルパー
# ============================================================

def _hand_ids(state, player_index: int) -> list[int]:
    """自分の手札カードIDリスト。"""
    hand = state.players[player_index].hand
    if hand is None:
        return []
    return [c.id for c in hand]


def _discard_ids(state, player_index: int) -> list[int]:
    return [c.id for c in state.players[player_index].discard]


def _active_id(state, player_index: int) -> int | None:
    active = state.players[player_index].active
    if not active or active[0] is None:
        return None
    return active[0].id


def _bench_ids(state, player_index: int) -> list[int]:
    return [p.id for p in state.players[player_index].bench]


def _in_play_ids(state, player_index: int) -> set[int]:
    """バトル場＋ベンチのIDセット。"""
    ids = set(_bench_ids(state, player_index))
    a = _active_id(state, player_index)
    if a is not None:
        ids.add(a)
    return ids


def _active_energy_count(state, player_index: int) -> int:
    active = state.players[player_index].active
    if not active or active[0] is None:
        return 0
    return len(active[0].energies)


def _active_has_special_energy(state, player_index: int) -> bool:
    """指定プレイヤーのバトルポケモンに特殊エネルギーがついているか。"""
    active = state.players[player_index].active
    if not active or active[0] is None:
        return False
    for ec in active[0].energyCards:
        cd = _get_card_data().get(ec.id)
        if cd and cd.cardType == CardType.SPECIAL_ENERGY:
            return True
    return False


def _hand_has(hand_ids: list[int], card_id: int) -> bool:
    return card_id in hand_ids

# ============================================================
# グッズ使用判定
# ============================================================

def _should_use_item(card_id: int, state, me_idx: int) -> bool:
    """このグッズを今ターン使うべきか判定する。"""
    opp_idx = 1 - me_idx
    hand = _hand_ids(state, me_idx)
    discard = _discard_ids(state, me_idx)

    if card_id == POKE_PAD:
        return True

    if card_id == RARE_CANDY:
        # バトル場がケーシィ かつ 手札にユンゲラーなし かつ フーディンあり
        active = _active_id(state, me_idx)
        return (
            active == ABRA
            and KADABRA not in hand
            and ALAKAZAM in hand
        )

    if card_id == ENHANCED_HAMMER:
        # 相手のバトルポケモンに特殊エネルギーがついている
        return _active_has_special_energy(state, opp_idx)

    if card_id == BUDDY_POFFIN:
        # ベンチに空きがあれば使う（デッキからポケモンを出せる）
        me = state.players[me_idx]
        return len(me.bench) < me.benchMax

    if card_id == NIGHT_STRETCHER:
        # トラッシュに基本ポケモン（優先対象）があれば使う
        basic_ids = {ABRA, DUNSPARCE, TOGEPI, SHAYMIN, PSYDUCK, FAN_ROTOM}
        return any(d in basic_ids for d in discard)

    if card_id == SACRED_ASH:
        # トラッシュに対象ポケモンがいれば使う
        return any(d in SACRED_ASH_TARGETS for d in discard)

    if card_id == WONDROUS_PATCH:
        # トラッシュに基本超エネルギーがあれば使う
        return BASIC_PSYCHIC in discard

    return False

# ============================================================
# サポート使用判定
# ============================================================

def _should_use_supporter(card_id: int, state, me_idx: int) -> bool:
    opp_idx = 1 - me_idx
    discard = _discard_ids(state, me_idx)

    if card_id == BOSS_ORDERS:
        # 相手ベンチに相手バトルより HPの低いポケモンがいれば使う
        opp = state.players[opp_idx]
        opp_active = opp.active[0] if opp.active else None
        if opp_active is None or not opp.bench:
            return False
        bench_min_hp = min(p.hp for p in opp.bench)
        return bench_min_hp < opp_active.hp

    if card_id == LANAS_AID:
        # (トラッシュにエネルギーがある) かつ
        # (バトルポケモンのエネルギーが0) かつ
        # (手札にエネルギーがない)
        has_energy_discard = any(d in ALL_ENERGY_IDS for d in discard)
        active_needs_energy = _active_energy_count(state, me_idx) == 0
        hand = _hand_ids(state, me_idx)
        hand_has_energy = any(h in ALL_ENERGY_IDS for h in hand)
        return has_energy_discard and active_needs_energy and not hand_has_energy

    if card_id in (HILDA, DAWN):
        return True

    return False

# ============================================================
# エネルギー貼付
# ============================================================

def _best_attach_option(obs: Observation) -> int | None:
    """エネルギーが必要数に達していないポケモンへ、優先度の高いエネルギーを貼る。

    貼る対象の優先順位: バトル場 > ベンチ
    エネルギーの優先順位: リッチ > テレパス超 > 基本超
    必要数以上はつけない（カードのワザコストから自動判定）
    """
    state = obs.current
    me_idx = state.yourIndex
    me = state.players[me_idx]

    # エネルギーを追加できるポケモンの serial を収集
    # (serial, 優先度) のリスト。バトル場=0、ベンチ=1 で優先度付け
    target_serials: dict[int, int] = {}  # serial → 優先度（低いほど先）

    active = me.active[0] if me.active else None
    if active is not None:
        needed = _max_energy_needed(active.id)
        if needed > 0 and len(active.energies) < needed:
            target_serials[active.serial] = 0  # バトル場優先

    for p in me.bench:
        needed = _max_energy_needed(p.id)
        if needed > 0 and len(p.energies) < needed:
            target_serials[p.serial] = 1  # ベンチは後回し

    if not target_serials:
        return None

    best_idx = None
    best_score = (999, 999)  # (貼り先優先度, エネルギー優先度)

    for i, option in enumerate(obs.select.option):
        if option.type != OptionType.ATTACH:
            continue
        if option.serial not in target_serials:
            continue
        if option.area != AreaType.HAND or option.index is None or not me.hand:
            continue

        energy_id = me.hand[option.index].id
        energy_pri = ENERGY_PRIORITY.get(energy_id, 999)
        target_pri = target_serials[option.serial]
        score = (target_pri, energy_pri)

        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx

# ============================================================
# ベンチ展開優先順位
# ============================================================

def _priority_for_bench(card_id: int, state, me_idx: int) -> int:
    """ベンチに出すポケモンの優先度（低いほど優先）。Fan Rotomは特別扱い。"""
    if card_id == FAN_ROTOM:
        # 最初のターン（turn<=2）かつベンチが空 → 出す
        if state.turn <= 2:
            return len(BENCH_PRIORITY)  # 低め優先度（基本は他を優先）
        return 9999  # それ以外は出さない
    try:
        return BENCH_PRIORITY.index(card_id)
    except ValueError:
        return 999

# ============================================================
# メインフェーズ
# ============================================================

def _supporter_priority(card_id: int) -> int:
    order = [BOSS_ORDERS, DAWN, HILDA, LANAS_AID]
    try:
        return order.index(card_id)
    except ValueError:
        return 999


def choose_main_action(obs: Observation) -> list[int]:
    """使えるものをすべて使い切る方針でのメインフェーズ行動選択。

    優先順位:
      1. ベンチ展開（スピンロトムは初手のみ、ベンチが空なら可）
      2. ポケパッド（無条件使用）
      3. その他アイテム（条件付き）
      4. サポーター（1ターン1枚）
      5. スタジアム（バトルコロシアム以外が場に出ていれば）
      6. エネルギー貼付（アラカズームラインに1個まで）
      7. 進化
      8. ワザ
      9. ターン終了
    """
    select = obs.select
    state = obs.current
    me_idx = state.yourIndex
    me = state.players[me_idx]
    hand = _hand_ids(state, me_idx)
    bench_full = len(me.bench) >= me.benchMax

    # カテゴリ別に候補を収集
    play_pokemon: list[tuple[int, int]] = []  # (option_index, priority)
    play_poke_pad: list[int] = []
    play_item: list[tuple[int, int]] = []  # (option_index, item_card_id)
    play_supporter: list[tuple[int, int]] = []  # (option_index, priority)
    play_stadium: list[int] = []
    ability_opts: list[int] = []
    evolve_opts: list[int] = []
    attack_opts: list[int] = []
    end_opts: list[int] = []

    # 現在フィールドに出ているスタジアムID
    current_stadium_id = state.stadium[0].id if state.stadium else None

    for i, option in enumerate(select.option):
        t = option.type

        if t == OptionType.PLAY:
            if me.hand is None or option.index is None:
                continue
            card = me.hand[option.index]
            cd = _get_card_data().get(card.id)
            if cd is None:
                continue

            if cd.cardType == CardType.POKEMON:
                if not bench_full:
                    pri = _priority_for_bench(card.id, state, me_idx)
                    if pri < 9999:
                        play_pokemon.append((i, pri))

            elif cd.cardType == CardType.ITEM:
                if card.id == POKE_PAD:
                    play_poke_pad.append(i)
                elif _should_use_item(card.id, state, me_idx):
                    play_item.append((i, card.id))

            elif cd.cardType == CardType.SUPPORTER and not state.supporterPlayed:
                if _should_use_supporter(card.id, state, me_idx):
                    pri = _supporter_priority(card.id)
                    play_supporter.append((i, pri))

            elif cd.cardType == CardType.STADIUM:
                # バトルコロシアム以外が出ていれば（または何も出ていなければ）出す
                if card.id == BATTLE_CAGE and current_stadium_id != BATTLE_CAGE:
                    play_stadium.append(i)

        elif t == OptionType.ABILITY:
            # ノコノコの特性はベンチに他のポケモンがいる場合のみ使う
            if option.area == AreaType.BENCH and option.index is not None:
                poke = me.bench[option.index] if option.index < len(me.bench) else None
                if poke and poke.id == DUDUNSPARCE and len(me.bench) <= 1:
                    continue  # ベンチにノコノコしかいない → スキップ
            ability_opts.append(i)
        elif t == OptionType.EVOLVE:
            evolve_opts.append(i)
        elif t == OptionType.ATTACK:
            attack_opts.append(i)
        elif t == OptionType.END:
            end_opts.append(i)

    # --- 行動選択 ---

    # 1. ベンチ展開（優先度順）
    if play_pokemon:
        play_pokemon.sort(key=lambda x: x[1])
        return [play_pokemon[0][0]]

    # 2. ポケパッド
    if play_poke_pad:
        return [play_poke_pad[0]]

    # 3. その他アイテム
    if play_item:
        return [play_item[0][0]]

    # 4. サポーター
    if play_supporter:
        play_supporter.sort(key=lambda x: x[1])
        return [play_supporter[0][0]]

    # 5. スタジアム
    if play_stadium:
        return [play_stadium[0]]

    # 6. エネルギー貼付
    if not state.energyAttached:
        attach_idx = _best_attach_option(obs)
        if attach_idx is not None:
            return [attach_idx]

    # 7. 特性
    if ability_opts:
        return [ability_opts[0]]

    # 8. 進化
    if evolve_opts:
        return [evolve_opts[0]]

    # 9. ワザ
    if attack_opts:
        return [attack_opts[0]]

    # 9. ターン終了
    if end_opts:
        return [end_opts[0]]

    return choose_minimum_required(obs)

# ============================================================
# 非メインフェーズの選択
# ============================================================

def _card_priority_for_to_hand(card_id: int) -> int:
    """TO_HAND / TO_BENCH で選ぶカードの優先度。"""
    order = [ABRA, ALAKAZAM, KADABRA, DUNSPARCE, DUDUNSPARCE,
             TOGEPI, TOGEKISS, SHAYMIN, PSYDUCK, FAN_ROTOM]
    try:
        return order.index(card_id)
    except ValueError:
        return 999


def choose_action(obs: Observation) -> list[int]:
    """SelectContext に応じて行動を選択する。"""
    select = obs.select
    ctx = select.context

    if ctx == SelectContext.MAIN:
        chosen = choose_main_action(obs)

    elif ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        # セットアップ：先頭のポケモン（ケーシィ優先）
        chosen = [0]
        for i, opt in enumerate(select.option):
            if opt.cardId == ABRA:
                chosen = [i]
                break

    elif ctx == SelectContext.SETUP_BENCH_POKEMON:
        # セットアップ：ベンチに出せるポケモンを優先度順に最大数まで出す
        options_sorted = sorted(
            range(len(select.option)),
            key=lambda i: _card_priority_for_to_hand(select.option[i].cardId or 0)
        )
        count = min(select.maxCount, len(select.option))
        chosen = options_sorted[:count]

    elif ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        # きぜつ後などバトル場に出すポケモンを選ぶ：フーディン優先
        ACTIVE_PRIORITY = [ALAKAZAM, KADABRA, ABRA, DUDUNSPARCE, DUNSPARCE,
                           TOGEKISS, SHAYMIN, PSYDUCK, FAN_ROTOM, TOGEPI]
        best = 0
        best_pri = 9999
        for i, opt in enumerate(select.option):
            cid = opt.cardId or 0
            pri = ACTIVE_PRIORITY.index(cid) if cid in ACTIVE_PRIORITY else 999
            if pri < best_pri:
                best_pri = pri
                best = i
        chosen = [best]

    elif ctx == SelectContext.TO_BENCH:
        # ポフィン/夜のタンカ等でデッキ/トラッシュからベンチへ
        count = min(select.maxCount, len(select.option))
        options_sorted = sorted(
            range(len(select.option)),
            key=lambda i: _card_priority_for_to_hand(select.option[i].cardId or 0)
        )
        chosen = options_sorted[:count]

    elif ctx == SelectContext.TO_HAND:
        # 夜のタンカ/せいなるはい等で手札へ
        count = min(select.maxCount, len(select.option))
        options_sorted = sorted(
            range(len(select.option)),
            key=lambda i: _card_priority_for_to_hand(select.option[i].cardId or 0)
        )
        chosen = options_sorted[:count]

    elif ctx == SelectContext.EFFECT_TARGET:
        # ボスの指令等：相手ベンチの中で最もHPが低いポケモンを選ぶ
        state = obs.current
        opp_idx = 1 - state.yourIndex
        opp_bench = state.players[opp_idx].bench

        best = 0
        best_hp = 9999
        for i, opt in enumerate(select.option):
            # option.index でベンチのインデックスを参照
            if opt.index is not None and opt.index < len(opp_bench):
                hp = opp_bench[opt.index].hp
                if hp < best_hp:
                    best_hp = hp
                    best = i
        count = min(select.minCount or 1, len(select.option))
        chosen = [best] if count == 1 else [best] + [j for j in range(len(select.option)) if j != best][:count - 1]

    else:
        chosen = choose_minimum_required(obs)

    validate_choice(obs, chosen)
    return chosen


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    # RL モデルが使えればそちらを使う
    rl_action = _rl_choose_action(obs)
    if rl_action is not None:
        try:
            validate_choice(obs, rl_action)
            return rl_action
        except ValueError:
            pass  # フォールバック

    return choose_action(obs)
