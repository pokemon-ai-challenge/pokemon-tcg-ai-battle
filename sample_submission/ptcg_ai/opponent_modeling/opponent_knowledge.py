"""相手デッキ予測器や将来の推論レイヤーが使う、相手の公開カード観測情報を蓄積するモジュール。

実際に見えたカードだけを記録する（``observed_cards``）。山札・手札・サイドの中身を
推定することはしない。それらは本モジュールの ``get_prediction_features()`` を入力として
使う、別の推論・予測モジュールの役目。

呼び出し方（重要・順序固定）: 新しい ``Observation`` を受け取るたびに、
``update_from_logs(obs.logs)`` を呼んでから ``update_from_state(obs.current)`` を呼ぶこと。
``logs`` は「この ``state`` に至るまでの出来事」なので時系列としては先に処理し、
盤面の完全スキャンである ``update_from_state`` を最後に当てて現在ゾーンを確定させる。
逆順で呼ぶと、盤面スキャンが正しく特定した現在ゾーンを、logs 側の暫定タグ
（`revealed` や、SWITCH ログ由来の active/bench 等）が後から上書きしてしまう
（実際に battle_review_viewer での検証中に踏んだ不具合）。
"""

from dataclasses import dataclass, field

from cg.api import (
    AreaType,
    Card,
    CardType,
    Log,
    LogType,
    Pokemon,
    State,
    all_card_data,
)

# AreaType -> 内部ゾーンキーの対応表。ここに無い AreaType は非公開領域（デッキ/手札/サイド）か
# 観測記録に無関係（PLAYER 等）なので、記録対象にしない。
_ZONE_BY_AREA = {
    AreaType.ACTIVE: "active",
    AreaType.BENCH: "bench",
    AreaType.DISCARD: "discard",
    AreaType.ENERGY: "energy",
    AreaType.TOOL: "tool",
    AreaType.PRE_EVOLUTION: "pre_evolution",
    AreaType.STADIUM: "stadium",
    AreaType.LOOKING: "revealed",
}

_ENERGY_CARD_TYPES = (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)


@dataclass
class ObservedCard:
    """1枚の実カード（``serial`` で識別）についての観測レコード。"""

    card_id: int  # CardData.cardId。同名でも再録違いで別の値になりうる
    name: str  # all_card_data() から解決した表示名（照合用の逆引き）
    is_pokemon: bool  # ポケモンカードか（features の observed_pokemon 用）
    is_energy: bool  # エネルギーカードか（features の observed_energies 用）
    is_tool: bool  # 道具カードか（features の observed_tools 用）
    serial: int | None = None  # 試合内で一意な実カードID。取れない場合のみ None
    zones_seen: set[str] = field(default_factory=set)  # これまで見えた全ゾーンの集合（履歴）
    current_zone: str | None = None  # 直近のスキャンで見えたゾーン。見えなくなったら None
    first_seen_turn: int | None = None  # 最初に観測したターン
    last_seen_turn: int | None = None  # 最後に観測したターン


class OpponentKnowledge:
    """1試合を通じて、相手の公開カード観測情報を蓄積する。"""

    def __init__(self, opponent_index: int | None = None):
        # 相手のプレイヤーインデックス。未指定なら最初の update_from_state 呼び出し時に
        # state.yourIndex から自動で確定する（毎回引数で渡さなくて済むようにするため）。
        self._opponent_index = opponent_index

        # 観測記録の主索引。serial（試合内で一意な実カードID）をキーにすることで、
        # 同じ1枚のカードがゾーンを移動しても「同じカード」として追跡できる。
        self._by_serial: dict[int, ObservedCard] = {}

        # serial が取れない観測のフォールバック保存先。(card_id, zone) をキーにすることで、
        # 同じカード/ゾーンの組み合わせを何度観測しても枚数を水増ししない
        # （取りこぼしよりも過大カウント防止を優先する）。
        self._no_serial: dict[tuple[int, str], ObservedCard] = {}

        # cardId ↔ 名前・種別の対応表。盤面上のカードは cardId しか持たないため、
        # 起動時に1回だけ all_card_data() を引いて逆引き辞書を作っておく。
        card_data = all_card_data()
        self._id_to_name: dict[int, str] = {c.cardId: c.name for c in card_data}
        self._id_to_type: dict[int, CardType] = {c.cardId: c.cardType for c in card_data}

        # Pokemon.energies（付属エネルギーの色）から集めた色ヒント。カード実体とは別に、
        # デッキのエネルギー色を推測する補助情報として保持する。
        self._energy_types_seen: set[int] = set()
        self._pending_logs: list[Log] = []

    # ------------------------------------------------------------------
    # 記録

    def update_from_state(self, state: State) -> None:
        """盤面スナップショットから、相手の公開カードを全て記録する。"""
        if self._opponent_index is None:
            self._opponent_index = 1 - state.yourIndex
        if self._pending_logs:
            self._replay_logs(self._pending_logs)
            self._pending_logs.clear()

        # 今回のスキャンで見えなかったカードは、公開ゾーンから消えたとみなす
        # （手札/デッキ/サイドへ戻った、単にこのスナップショットに写っていない、等）。
        for record in self._by_serial.values():
            record.current_zone = None
        for record in self._no_serial.values():
            record.current_zone = None

        turn = state.turn
        player = state.players[self._opponent_index]

        # バトル場: 伏せ中（active[0] is None）は正体不明なので記録しない。
        if player.active and player.active[0] is not None:
            self._observe_pokemon(player.active[0], "active", turn)
        # ベンチ: 複数体いるので1体ずつ記録する。
        for bench_pokemon in player.bench:
            self._observe_pokemon(bench_pokemon, "bench", turn)
        # トラッシュ: 山札に入っていたことが確定するカード（採用確認に有効）。
        for card in player.discard:
            self.observe_card(card.id, "discard", turn=turn, serial=card.serial)

        # スタジアムと一時公開(looking)は State 上で共有/所有者不明なゾーンだが、
        # Card.playerIndex を見ればどちらのリストから来たかに関わらず真の所有者が分かる。
        # スタジアム: 場に1枚だけ存在する共有カード。相手が出したものだけ記録する。
        for card in state.stadium:
            if card.playerIndex == self._opponent_index:
                self.observe_card(card.id, "stadium", turn=turn, serial=card.serial)
        # 一時公開: サーチ効果などで今だけ見えているカード。伏せ（None）は当然スキップ。
        if state.looking:
            for card in state.looking:
                if card is not None and card.playerIndex == self._opponent_index:
                    self.observe_card(card.id, "revealed", turn=turn, serial=card.serial)

    def _observe_pokemon(self, pokemon: Pokemon, zone: str, turn: int | None) -> None:
        """ポケモン本体と、それに付属するカード（エネルギー・道具・進化元）をまとめて記録する。"""
        # ポケモン本体（バトル場 or ベンチのどちらか。呼び出し元が zone で指定する）。
        self.observe_card(pokemon.id, zone, turn=turn, serial=pokemon.serial)
        # 付属エネルギーの色（EnergyType）はカード実体とは別に、色ヒントとして集計しておく。
        for energy in pokemon.energies:
            self._energy_types_seen.add(int(energy))
        # 付属エネルギーカードの実体（何のエネルギーカードが何枚ついているか）。
        for card in pokemon.energyCards:
            self.observe_card(card.id, "energy", turn=turn, serial=card.serial)
        # 付属道具。
        for card in pokemon.tools:
            self.observe_card(card.id, "tool", turn=turn, serial=card.serial)
        # 進化元（進化しても下の段のカードが露出したままになる）。進化ラインの採用確認に使える。
        for card in pokemon.preEvolution:
            self.observe_card(card.id, "pre_evolution", turn=turn, serial=card.serial)

    def update_from_logs(self, logs: list[Log]) -> None:
        """logs（プレイ/進化/付与/移動など）から、公開された相手カードを記録する。"""
        if self._opponent_index is None:
            self._pending_logs.extend(logs)
            return
        self._replay_logs(logs)

    def _replay_logs(self, logs: list[Log]) -> None:
        for log in logs:
            # 自分側のイベントは無視する（記録するのはあくまで相手のカードだけ）。
            if log.playerIndex is None or log.playerIndex != self._opponent_index:
                continue
            # 対応表に無い LogType（伏せ移動や結果通知など）はカード情報を持たないので無視する。
            handler = self._LOG_HANDLERS.get(log.type)
            if handler is not None:
                handler(self, log)

    def _handle_play(self, log: Log) -> None:
        # 手札からプレイされ、一瞬公開されたカード（サポート/グッズ等）。
        if log.cardId is not None:
            self.observe_card(log.cardId, "revealed", serial=log.serial)

    def _handle_evolve_like(self, log: Log) -> None:
        # 進化/退化したカード自体。着地先ゾーンは次の update_from_state で確定するので、
        # ここでは「観測済み」であることだけ記録しておく。
        if log.cardId is not None:
            self.observe_card(log.cardId, "revealed", serial=log.serial)

    def _handle_move_card(self, log: Log) -> None:
        # 公開領域間の移動。移動先が既知の公開ゾーンならそのゾーンで、
        # 非公開領域（手札/デッキ/サイド）ならとりあえず一時公開扱いにする。
        if log.cardId is None:
            return
        zone = _ZONE_BY_AREA.get(log.toArea, "revealed")
        self.observe_card(log.cardId, zone, serial=log.serial)

    def _handle_attach_like(self, log: Log) -> None:
        # エネルギー/道具の付与・付け替え。カード種別からエネルギーか道具かを判定する。
        if log.cardId is None:
            return
        self.observe_card(log.cardId, self._zone_for_attachable(log.cardId), serial=log.serial)

    def _handle_switch(self, log: Log) -> None:
        # バトル場とベンチのポケモンが入れ替わった。フィールド名と実際の意味が逆になっている点に注意:
        # cardIdActive は「(元)アクティブから来た＝ベンチへ移動する」カード、
        # cardIdBench は「(元)ベンチから来た＝アクティブへ移動する」カードを指す
        # （cg エンジンの実データで検証済み。api.py のコメント "Moving to the Bench/Active" と対応）。
        if log.cardIdActive is not None:
            self.observe_card(log.cardIdActive, "bench", serial=log.serialActive)
        if log.cardIdBench is not None:
            self.observe_card(log.cardIdBench, "active", serial=log.serialBench)

    def _handle_change(self, log: Log) -> None:
        # ポケモンが変化した（フォルムチェンジ等）。前後どちらのカードも観測済みとして残す。
        if log.cardIdBefore is not None:
            self.observe_card(log.cardIdBefore, "revealed", serial=log.serialBefore)
        if log.cardIdAfter is not None:
            self.observe_card(log.cardIdAfter, "revealed", serial=log.serialAfter)

    # LogType ごとのハンドラ対応表。MOVE_CARD_REVERSE・DRAW_REVERSE は cardId を持たない
    # 伏せ移動なので、ここに載せず記録対象から外す。
    _LOG_HANDLERS = {
        LogType.PLAY: _handle_play,
        LogType.EVOLVE: _handle_evolve_like,
        LogType.DEVOLVE: _handle_evolve_like,
        LogType.MOVE_CARD: _handle_move_card,
        LogType.ATTACH: _handle_attach_like,
        LogType.MOVE_ATTACHED: _handle_attach_like,
        LogType.SWITCH: _handle_switch,
        LogType.CHANGE: _handle_change,
    }

    def _zone_for_attachable(self, card_id: int) -> str:
        """ATTACH/MOVE_ATTACHED ログの cardId が、エネルギーか道具かをカード種別から判定する。"""
        card_type = self._id_to_type.get(card_id)
        if card_type in _ENERGY_CARD_TYPES:
            return "energy"
        if card_type == CardType.TOOL:
            return "tool"
        # どちらでもない・種別不明ならとりあえず一時公開扱いにしておく（安全側）。
        return "revealed"

    def observe_card(
        self,
        card_id: int,
        zone: str,
        turn: int | None = None,
        serial: int | None = None,
    ) -> None:
        """観測1件を記録する最小単位の API。``update_from_*`` 系メソッドが内部で呼ぶ。"""
        if serial is not None:
            # serial が分かっている＝実カードを一意に特定できるケース（通常はこちら）。
            record = self._by_serial.get(serial)
            if record is None:
                # 初めて見る serial → 新規レコードとして登録するだけ（枚数が+1される）。
                self._by_serial[serial] = self._new_record(card_id, zone, turn, serial)
                return
            # 既知の serial → 同じ1枚の再観測。枚数は増やさず、位置情報だけ更新する。
            record.zones_seen.add(zone)
            record.current_zone = zone
            if turn is not None:
                record.last_seen_turn = turn
                if record.first_seen_turn is None:
                    record.first_seen_turn = turn
            return

        # serial が取れないフォールバック経路。(card_id, zone) で既知かどうかだけ判定し、
        # 既知なら新規カウントせず位置情報のみ更新する（過大カウント防止を優先）。
        key = (card_id, zone)
        record = self._no_serial.get(key)
        if record is None:
            record = self._find_reusable_no_serial(card_id)
        if record is None:
            self._no_serial[key] = self._new_record(card_id, zone, turn, None)
            return
        record.zones_seen.add(zone)
        record.current_zone = zone
        if turn is not None:
            record.last_seen_turn = turn
            if record.first_seen_turn is None:
                record.first_seen_turn = turn

    def _find_reusable_no_serial(self, card_id: int) -> ObservedCard | None:
        for record in self._no_serial.values():
            if record.card_id == card_id:
                return record
        return None

    def _new_record(self, card_id: int, zone: str, turn: int | None, serial: int | None) -> ObservedCard:
        """card_id から名前・種別を引いて、初回観測時の ObservedCard を組み立てる。"""
        card_type = self._id_to_type.get(card_id)
        return ObservedCard(
            card_id=card_id,
            name=self._id_to_name.get(card_id, str(card_id)),
            is_pokemon=card_type == CardType.POKEMON,
            is_energy=card_type in _ENERGY_CARD_TYPES,
            is_tool=card_type == CardType.TOOL,
            serial=serial,
            zones_seen={zone},
            current_zone=zone,
            first_seen_turn=turn,
            last_seen_turn=turn,
        )

    # ------------------------------------------------------------------
    # 取得

    def get_observed_cards(self) -> list[ObservedCard]:
        """これまでに観測した全カード（履歴。現在は見えなくなったものも含む）。"""
        # serial 管理分とフォールバック分を合わせて返す。
        return list(self._by_serial.values()) + list(self._no_serial.values())

    def get_zone_cards(self) -> dict[str, list[ObservedCard]]:
        """現在見えているカードを、現在の公開ゾーンごとにグルーピングして返す。"""
        zone_cards: dict[str, list[ObservedCard]] = {}
        for record in self.get_observed_cards():
            if record.current_zone is None:
                continue  # 過去に見えたが、今は非公開ゾーンにいる（＝現在は見えない）カードは除外
            zone_cards.setdefault(record.current_zone, []).append(record)
        return zone_cards

    def get_prediction_features(self) -> dict:
        """相手デッキ予測器・将来の推論レイヤー向けの観測特徴量を返す。

        カウントの単位は ``card_id``（``serial`` で重複排除）。同名でも別カードID
        （再録違い）がありうるため。``observed_cards``（name単位）はアーキタイプ照合用の
        派生集計であり、``observed_card_ids`` の方が山札・サイド推定に使う正の値。
        """
        records = self.get_observed_cards()

        # 正の枚数カウント: card_id ごとに「distinct な ObservedCard の件数」を数える。
        # 1レコード = 1枚（serial 単位で重複排除済み）なので、これがそのまま観測枚数になる。
        observed_card_ids: dict[int, int] = {}
        for record in records:
            observed_card_ids[record.card_id] = observed_card_ids.get(record.card_id, 0) + 1

        # 同名カードの内訳（name -> {card_id: 枚数}）。再録違いを name 単位で束ねつつ、
        # どの card_id が何枚かという情報は失わないようにする。
        name_to_card_ids: dict[str, dict[int, int]] = {}
        for record in records:
            name_to_card_ids.setdefault(record.name, {})
            name_to_card_ids[record.name][record.card_id] = observed_card_ids[record.card_id]

        # card_id 単位の枚数を name で合算しただけの派生ビュー（逆変換はしない）。
        observed_cards: dict[str, int] = {
            name: sum(id_counts.values()) for name, id_counts in name_to_card_ids.items()
        }

        # 現在の公開ゾーンごとに、見えているカード名の一覧を作る（枚数はここでは持たない）。
        zone_cards = {
            zone: [record.name for record in zone_records]
            for zone, zone_records in self.get_zone_cards().items()
        }

        def _unique_names(predicate) -> list[str]:
            # 条件（ポケモン/エネルギー/道具のいずれか）に合うカードの名前を、
            # 観測順を保ったまま重複なく集める（dict のキー挿入順を利用した重複除去）。
            seen: dict[str, None] = {}
            for record in records:
                if predicate(record):
                    seen.setdefault(record.name, None)
            return list(seen.keys())

        return {
            "observed_card_ids": observed_card_ids,    # 正: card_id 単位の枚数（山札推定用）
            "observed_cards": observed_cards,          # 派生: name 単位に合算した枚数（照合用）
            "name_to_card_ids": name_to_card_ids,      # name -> {card_id: 枚数} の内訳
            "zone_cards": zone_cards,                  # 現在のゾーンごとのカード名一覧
            "observed_pokemon": _unique_names(lambda r: r.is_pokemon),
            "observed_energies": _unique_names(lambda r: r.is_energy),
            "observed_tools": _unique_names(lambda r: r.is_tool),
            "energy_types": sorted(self._energy_types_seen),  # 付属エネルギーから見えた色一覧
        }
