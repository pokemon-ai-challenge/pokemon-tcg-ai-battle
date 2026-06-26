from dataclasses import dataclass, field


@dataclass
class DeckPlan:
    """デッキ固有の戦術設定。ロジックはここを参照して汎用的に書く。"""
    name: str
    prefer_go_first: bool = True
    main_attacker_ids: list[int] = field(default_factory=list)
    sub_attacker_ids: list[int] = field(default_factory=list)
    # SETUP_ACTIVE 用の優先順位（先頭ほど優先してバトル場に出す）
    active_priority: list[int] = field(default_factory=list)
    # SETUP_BENCH / TO_BENCH 用の優先順位
    bench_priority: list[int] = field(default_factory=list)
    # ATTACH_FROM 用: エネルギーを付けたいポケモンの優先順位（pokemon card_id）
    energy_priority: list[int] = field(default_factory=list)
    # EVOLVES_TO / EVOLVE 用: 優先して進化させたいポケモンの card_id（進化後の ID）
    evolution_priority: list[int] = field(default_factory=list)
    # 進化ライン: {進化前 card_id: 進化後 card_id}
    evolution_lines: dict[int, int] = field(default_factory=dict)
    # DISCARD で捨てたくないカードの card_id セット
    do_not_discard_ids: set[int] = field(default_factory=set)
    # Crustle 等 EX 無効化アビリティ持ちと対戦時の非 ex エネルギー付与優先順位
    non_ex_energy_priority: list[int] = field(default_factory=list)
    # Crustle 等 EX 無効化アビリティ持ちと対戦時の非 ex アクティブ優先順位
    non_ex_active_priority: list[int] = field(default_factory=list)
    # 勝ち筋: このデッキで勝つための優先条件リスト（Phase 3 評価関数への入力）
    win_conditions: list[str] = field(default_factory=list)
    # 立て直し方針: 不利状況で取るべき行動の方針リスト
    recovery_rules: list[str] = field(default_factory=list)
    # アーキタイプ別対策: {meta_deck_name: 対策方針テキスト}
    archetype_counters: dict[str, str] = field(default_factory=dict)
    # バッファポケモン: 逃げエネが 0 でなくても育成前提で「壁役」として扱う card_id
    # ベンチにアタッカーが準備完了したら優先的に引っ込める
    buffer_pokemon_ids: list[int] = field(default_factory=list)


# メガルカリオex デッキ (deck.csv 構成)
# メインアタッカー: メガルカリオex(678) ← リオル(677)
# サブアタッカー:   ハリテヤマ(674)     ← マクノシタ(673)
# ユーティリティ:   ソルロック(676), ルナトーン(675), ニャースex(1071)
LUCARIO_PLAN = DeckPlan(
    name="mega_lucario_ex",
    prefer_go_first=True,
    main_attacker_ids=[678],
    sub_attacker_ids=[674],
    active_priority=[677, 673],
    bench_priority=[677, 673, 676, 675, 1071],
    energy_priority=[678, 674, 677],
    evolution_priority=[678, 674],
    evolution_lines={677: 678, 673: 674},
    do_not_discard_ids={677, 678, 673, 674, 676, 675, 1071},
    # Crustle 対面: Hariyama(674) の Wild Press 210 ダメで 1 撃 KO が狙える
    # Solrock(676) は HP が低く Crustle の攻撃で 1 撃されるため除外
    non_ex_energy_priority=[674, 677],
    non_ex_active_priority=[674],
    win_conditions=[
        "メガルカリオexを最速でアクティブに出してエネルギーを2枚つける",
        "相手の主力アタッカーを先に倒してサイドリードを取る",
        "進化ラインを2セット並べてリソース切れリスクを下げる",
    ],
    recovery_rules=[
        "メインアタッカーがKOされたら控えのリオルを即座に進化させる",
        "手札が2枚以下になったらサポーターを最優先で使う",
        "ベンチにポケモンがいなくなりそうなら補充を最優先にする",
    ],
    archetype_counters={
        # --- Tier S ---
        # 2026-06-26 実デッキベンチマーク: 23/30 = 77%
        "dragapult_ex": (
            "ドラパルトexはファントムダイブでベンチ6ダメカン散布+バトル場200。"
            "対策: ベンチを最小限に絞りリオル1体だけ置く。"
            "メガルカリオexが攻撃できる状態になったら即座にKOを狙う。"
        ),
        # 2026-06-26 実デッキベンチマーク: 17/30 = 57% ← 最も不利
        "hydrapple_ex": (
            "カミツオロチexは自己回復しながらメガニウムのエネ加速で攻撃し続ける。"
            "対策: まずメガニウムをKOしてエネ加速を止める（ボスの指令で引き出す）。"
            "オーガポンみどりのめんexは弱点をつかれるので早期処理優先。"
            "自己回復されると長期戦になるため、アウラジャブの格闘エネ回収サイクルで攻撃を継続する。"
        ),
        # 2026-06-26 実デッキベンチマーク: 25/30 = 83%
        "terasta_bullet": (
            "テラスタルバレットはエネルギーつけかえで多タイプ攻撃者を使い分けるデッキ。"
            "対策: 相手の攻撃者が確定する前に積極的に攻撃してサイドリードを作る。"
            "格闘タイプは多くのテラスタルアタッカーに等倍以上で入る。"
        ),
        # --- Tier A ---
        # 2026-06-26 実デッキベンチマーク: 27/30 = 90% ← 最も有利
        "raging_bolt_ex": (
            "タケルライコexは古代の雷タイプアタッカー。格闘が弱点。"
            "対策: メガルカリオexのオーラジャブで格闘弱点をつき一撃KOを狙う。"
            "相手のオーガポンみどりのめんexはベンチに置いてあることが多い—ボスで引き出してKO。"
        ),
        # 2026-06-26 実デッキベンチマーク: 23/30 = 77%（ミラーマッチ）
        "mega_lucario_ex": (
            "同じ格闘デッキとのミラーマッチ。先にセットアップした方が有利。"
            "対策: ルナトーンのルナサイクルドローを優先し先行展開する。"
            "相手のリオル・マクノシタを早期KOして進化を妨害する。"
        ),
        # 2026-06-26 実デッキベンチマーク: 22/30 = 73%
        "olivia_ex": (
            "オリーヴァexは草タイプの自己回復Stage2。メガニウムでエネ加速。"
            "対策: カミツオロチex同様、まずメガニウムをボスで引き出してKO。"
            "オリーヴァexへの攻撃は自己回復で無効化されやすいため加速源の破壊を優先。"
        ),
        # 2026-06-26 実デッキベンチマーク: 22/30 = 73%
        "maries_obstagoon_ex": (
            "マリィのオーロンゲexは悪タイプの妨害・コントロールデッキ。"
            "対策: ベンチのマリィのベロバー/ギモーを早期KOして進化ラインを崩す。"
            "ユキワラシ/ユキメノコはサポート役—ボスで引き出してKOしリソースを削る。"
        ),
        # 2026-06-26 実デッキベンチマーク: 20/30 = 67%
        "ogerpon_bullet": (
            "オーガポンバレットはエネルギーつけかえで多タイプを使い分ける。"
            "対策: 相手の攻撃者が変わる前に集中してKOする—攻撃対象を分散させない。"
            "マシマシラは能力でエネルギーを動かすため優先的に処理する。"
        ),
        # 2026-06-26 実デッキベンチマーク: 19/30 = 63%
        "crustle": (
            "イワパレスはHPが高く特殊エネルギー（スパイク/ミスト/グロウ草/ロック闘）で妨害してくる。"
            "対策: アウラジャブの格闘エネ回収サイクルで攻撃を途切れさせない。"
            "メガガルーラexはHPが高い—ハリテヤマのサブアタックも活用して削る。"
        ),
        # --- Tier B ---
        # 2026-06-26 実デッキベンチマーク: 18/30 = 60%
        "alakazam": (
            "フーディンはパワフルハンドで手札枚数×ダメージ。手札が多いほど強い。"
            "対策: こちらは手札を積極的に使って手札枚数を減らすことを意識する。"
            "フーディンは超タイプ—格闘は等倍。早期KOより継続ダメージで削る戦略。"
        ),
    },
)


RAGING_BOLT_PLAN = DeckPlan(
    name="raging_bolt_ex",
    prefer_go_first=True,
    main_attacker_ids=[63],            # Raging Bolt ex (Basic Ancient)
    sub_attacker_ids=[75, 96],         # Iron Hands ex, Ogerpon みどりのめんex
    active_priority=[63, 96],          # Raging Bolt ex → Ogerpon as backup
    bench_priority=[96, 75, 62, 1071, 140, 184],
    energy_priority=[63, 96, 75],      # Raging Bolt gets Grass; Iron Hands gets Lightning+Fighting
    evolution_priority=[],             # 全 Basic、進化なし
    evolution_lines={},
    do_not_discard_ids={63, 75, 96, 62, 1071, 140, 184, 209},
    win_conditions=[
        "タケルライコexを最速アクティブに出して草エネルギーを3枚つける",
        "ランペイジングサンダーで200+ダメージを連発してサイドリードを取る",
        "オーガポンexをサブアタッカーとして温存し草エネでいつでも攻撃できる状態に保つ",
    ],
    recovery_rules=[
        "タケルライコexがKOされたら控えのオーガポンexをすぐアクティブに出す",
        "手札が2枚以下になったらサポーターを最優先で使う",
        "テツノイサハexは雷エネルギーをかき集められる場合に使う",
    ],
    archetype_counters={
        "dragapult_ex": "ドラパルトexはベンチ散布。タケルライコ1体をアクティブに保ちベンチを最小限にして散布被害を抑える。",
        "hydrapple_ex": "カミツオロチexは自己回復。メガニウムをボスで引き出してKO。ランペイジング200で削り続ける。",
        "terasta_bullet": "多タイプ。タケルライコで高火力を連発してサイドリードを維持。",
        "raging_bolt_ex": "ミラー。先に草エネを3枚つけてランペイジングを先撃ちした方が有利。",
        "mega_lucario_ex": "格闘は弱点なし。同HPなら先にセットアップした方が勝つ。",
        "olivia_ex": "草タイプ自己回復。メガニウムをボスで引き出しKO後、オリーヴァexを削り続ける。",
        "ogerpon_bullet": "バレット系。高火力で次々KOしてサイドを取り切る。",
        "crustle": "特殊エネ妨害が厄介。草エネなら妨害されにくい。高HPイワパレスをランペイジングで削る。",
        "maries_obstagoon_ex": "悪コントロール。ベンチのベロバーをボスで引き出しKO。悪エネ10枚は弱点をつかれない。",
        "alakazam": "超タイプ→タケルライコに弱点なし。手札枚数を絞り、ランペイジングで高ダメージ維持。",
    },
)


HYDRAPPLE_PLAN = DeckPlan(
    name="hydrapple_ex",
    prefer_go_first=False,             # 後攻有利（メガニウム加速してから攻撃）
    main_attacker_ids=[150],           # Hydrapple ex (Stage 2)
    sub_attacker_ids=[96],             # Ogerpon みどりのめんex
    active_priority=[149, 96, 708],    # Applin, Ogerpon, Chikorita でセットアップ
    bench_priority=[149, 708, 96, 1071, 140, 655],
    energy_priority=[150, 96, 921],    # Hydrapple ex → Ogerpon → Dipplin
    evolution_priority=[150, 710, 921, 709],  # Hydrapple ex > Meganium > Dipplin > Bayleef
    evolution_lines={149: 921, 921: 150, 708: 709, 709: 710},
    do_not_discard_ids={149, 921, 150, 708, 709, 710, 96, 1071, 140, 655, 920},
    # Crustle 対面: Tapu Bulu(920) の Wood Hammer 220 で一撃 KO、次点 Bayleef(709) 50 ダメ
    non_ex_energy_priority=[920, 709, 149],
    non_ex_active_priority=[920, 709],
    win_conditions=[
        "チコリータ→ベイリーフ→メガニウムを早期完成させエネ加速を確立する",
        "カミツオロチexは自己回復しながら攻撃し長期戦で有利を作る",
        "オーガポンexをサブアタッカーとして早期攻撃に使う",
    ],
    recovery_rules=[
        "カミツオロチexがKOされたら控えのオーガポンexを即座にアクティブに",
        "メガニウムがKOされたら次のチコリータを急いで育てる",
        "手札が2枚以下になったらサポーターを最優先で使う",
    ],
    archetype_counters={
        "dragapult_ex": "ドラパルトexのベンチ散布がメガニウムに当たると痛い。ベンチを最小限に絞る。",
        "hydrapple_ex": "ミラー。先にメガニウムを育てた方が有利。メガニウムをボスで引き出す。",
        "terasta_bullet": "多タイプ。自己回復でサイド差を縮める。",
        "raging_bolt_ex": "タケルライコは草が弱点をつかない。メガニウム加速で圧倒する。",
        "mega_lucario_ex": "格闘タイプ。草には等倍。先に攻撃できれば有利。",
        "olivia_ex": "草ミラー。早くメガニウムを育てた方が有利。",
        "ogerpon_bullet": "バレット。オーガポンexで草弱点をつける相手を狙う。",
        "crustle": "特殊エネ妨害。草エネなら比較的安全。イワパレスを削り続ける。",
        "maries_obstagoon_ex": "悪コントロール。メガニウム加速でオーガポンが攻撃継続。",
        "alakazam": "超タイプ→草は等倍。手札を絞りながらランペイジング連打。",
    },
)


MARIES_OBSTAGOON_PLAN = DeckPlan(
    name="maries_obstagoon_ex",
    prefer_go_first=True,
    main_attacker_ids=[648],           # Marnie's Obstagoon ex (Stage 2)
    sub_attacker_ids=[112],            # Munkidori
    active_priority=[646, 103],        # Marnie's Zigzagoon, Snover でセットアップ
    bench_priority=[646, 103, 112, 689, 235],
    energy_priority=[648, 647, 646],   # Obstagoon ex → Linoone → Zigzagoon
    evolution_priority=[648, 647, 104],  # Obstagoon ex > Linoone > Abomasnow
    evolution_lines={646: 647, 647: 648, 103: 104},
    do_not_discard_ids={646, 647, 648, 103, 104, 112, 689, 235},
    # Crustle 対面: Yveltal(689) の Dark Feather 110 ダメ（2 回で KO）、次点 Morgrem(647) 60 ダメ
    non_ex_energy_priority=[689, 647, 646],
    non_ex_active_priority=[689, 647, 646],
    win_conditions=[
        "マリィのベロバー→ギモー→オーロンゲexの進化ラインを2セット並べる",
        "悪タイプの高火力と妨害効果で相手のリソースを削る",
        "ふしぎなアメでベロバーから直接オーロンゲexへ進化を狙う",
    ],
    recovery_rules=[
        "オーロンゲexがKOされたら控えのギモーを即座に進化させる",
        "手札が2枚以下になったらサポーターを最優先で使う",
        "ベンチにベロバーを必ず1体確保しておく",
    ],
    buffer_pokemon_ids=[646, 103],  # マリィのベロバー(646), スノーバー(103): 進化前は壁役
    archetype_counters={
        "dragapult_ex": "ドラパルトexは超タイプで悪に弱点なし。ベンチ散布を防ぐためベンチを絞る。",
        "hydrapple_ex": "草タイプ→悪に等倍。メガニウムをボスで引き出しKO。",
        "terasta_bullet": "多タイプ。悪の高火力でサイドリードを取る。",
        "raging_bolt_ex": "雷タイプ→悪に等倍。ランペイジングに対して高HPで耐える。",
        "mega_lucario_ex": "格闘は悪に弱点なし。オーロンゲexの妨害で相手のリソースを枯らす。",
        "olivia_ex": "草タイプ→悪に等倍。メガニウムをボスで引き出しKO。",
        "ogerpon_bullet": "バレット。オーロンゲの妨害で相手の多タイプを崩す。",
        "crustle": "特殊エネ妨害が辛い。悪エネは特殊エネでないので妨害されにくい。",
        "maries_obstagoon_ex": "ミラー。先に進化ラインを完成させた方が有利。",
        "alakazam": "超タイプ→悪に弱点。オーロンゲexの一撃でKOを狙う。",
    },
)


DRAGAPULT_PLAN = DeckPlan(
    name="dragapult_ex",
    prefer_go_first=True,
    main_attacker_ids=[121],           # Dragapult ex
    sub_attacker_ids=[326, 791],       # Blaziken ex, Moltres
    active_priority=[119, 324],        # Dreepy, Torchic（セットアップ用 Basic）
    bench_priority=[119, 324, 112, 1071, 140, 791],
    energy_priority=[121, 326, 791],   # Dragapult ex → Blaziken ex → Moltres
    evolution_priority=[121, 326, 120, 325],  # 最終進化優先、中間は後回し
    evolution_lines={
        119: 121,   # Dreepy → Dragapult ex（Rare Candy 経由含む）
        120: 121,   # Drakloak → Dragapult ex
        324: 326,   # Torchic → Blaziken ex（Rare Candy 経由含む）
        325: 326,   # Combusken → Blaziken ex
    },
    do_not_discard_ids={119, 120, 121, 324, 325, 326, 112, 1071, 140, 791, 1063},
    win_conditions=[
        "レアキャンディでドラパルトexを最速召喚してファントムダイブを連発する",
        "バシャーモexでエネルギーをベンチポケモンに循環してエネルギー切れを防ぐ",
        "ベンチへの6ダメカン散布を積み重ねてサイドリードを維持する",
    ],
    recovery_rules=[
        "ドラパルトexがKOされたら控えのドレディア進化ラインを即座に育てる",
        "バシャーモexでエネルギーを回収して次のアタッカーに回す",
        "手札が2枚以下になったらサポーターを最優先で使う",
    ],
    archetype_counters={
        "dragapult_ex": "ミラーマッチ。先にドラパルトを育てた方が有利。先攻ターンにファントムダイブを撃てれば優位。",
        "hydrapple_ex": "カミツオロチexの自己回復が厄介。ベンチ散布でメガニウムを削りながら攻撃を継続。",
        "terasta_bullet": "多タイプ。ベンチ散布で全員を削ってKOを積み重ねる。",
        "raging_bolt_ex": "HP280のタケルライコ。ファントムダイブ200+散布で2ターンで仕留める。ベンチのオーガポンも散布で削る。",
        "mega_lucario_ex": "格闘タイプには弱点なし。先にセットアップして先手を取る。ベンチのリオル・マクノシタを散布で削る。",
        "olivia_ex": "草タイプ自己回復。ベンチのメガニウムをダメカンで削ってKO。",
        "ogerpon_bullet": "バレット系でベンチ多展開→ファントムダイブ散布が特に有効。",
        "crustle": "イワパレスはHP高く特殊エネ妨害。ベンチのイシズマイを散布で削る。",
        "maries_obstagoon_ex": "悪コントロール。ベンチのベロバー進化前を散布で全滅させる。",
        "alakazam": "フーディン超タイプ→ドラパルトに弱点なし。ベンチのケーシィを散布で全滅させる。",
    },
)


_active_plan: DeckPlan | None = None


def set_active_plan(plan: DeckPlan) -> None:
    """ベンチマーク用: プランを一時的に上書きする。None を渡すとリセット。"""
    global _active_plan
    _active_plan = plan


def get_deck_plan() -> DeckPlan:
    """現在のデッキプランを返す。デッキ変更時はここの返り値を差し替える。"""
    if _active_plan is not None:
        return _active_plan
    return MARIES_OBSTAGOON_PLAN  # 現行デッキ: deck_maries_obstagoon.csv
