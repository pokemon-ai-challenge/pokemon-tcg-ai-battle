"""OpponentKnowledge の観測結果を、リプレイの神視点(visualize_data)と突き合わせるための検証ヘルパー。

visualize_data() が返す盤面は両プレイヤーの手札まで見える完全情報(ground truth)。
一方 OpponentKnowledge はエージェントが実際に受け取れる公開情報だけから観測を組み立てている。
この2つを同じ瞬間(同じ frame)で突き合わせることで、「本当は見えているのに観測し損ねている」
「見えなくなったのに観測が残ったまま」といった実装バグを、目視ではなくプログラムで検出する。

比較の単位は OpponentKnowledge 側と同じ serial（試合内で一意な実カードID）。
"""

from typing import Any


def _add(ground_truth: dict[int, dict[str, Any]], card: dict[str, Any] | None, zone: str) -> None:
    """神視点のカード1枚分を ground_truth に登録する。serial が無ければ突き合わせようがないので無視する。"""
    if card is None:
        return
    serial = card.get("serial")
    if serial is None:
        return
    ground_truth[serial] = {"card_id": card.get("id"), "name": card.get("name"), "zone": zone}


def _add_pokemon(ground_truth: dict[int, dict[str, Any]], pokemon: dict[str, Any] | None, zone: str) -> None:
    """ポケモン本体と、その付属カード（エネルギー・道具・進化元）をまとめて登録する。"""
    if pokemon is None:
        return
    _add(ground_truth, pokemon, zone)
    for card in pokemon.get("energyCards") or []:
        _add(ground_truth, card, "energy")
    for card in pokemon.get("tools") or []:
        _add(ground_truth, card, "tool")
    for card in pokemon.get("preEvolution") or []:
        _add(ground_truth, card, "pre_evolution")


def collect_ground_truth(
    current: dict[str, Any],
    opponent_index: int,
    revealed_active: bool = True,
) -> dict[int, dict[str, Any]]:
    """神視点の盤面(current)から、opponent_index 側の「今公開されているはずのカード」を集める。

    OpponentKnowledge が対象にしているのと同じゾーン(バトル場/ベンチ/トラッシュ/付属エネルギー・
    道具/進化元/スタジアム/一時公開)だけを見る。手札・デッキ・サイドは非公開領域なので含めない
    （OpponentKnowledge も対象外にしているのと同じ理由）。

    ``visualize_data()`` は人間のレビュー用に伏せカードの中身までそのまま見せてしまう
    （セットアップ中、両者同時公開前のバトル場ポケモンなど）。実際のエージェント視点の
    ``Observation`` では、その間バトル場は ``None``（伏せ）として渡される。この食い違いを
    「観測漏れ」と誤検出しないよう、``revealed_active``（実際の Observation 側でバトル場が
    公開済みかどうか）が False の間はバトル場を ground truth から除外する。

    戻り値: {serial: {"card_id":..., "name":..., "zone":...}}
    """
    ground_truth: dict[int, dict[str, Any]] = {}
    players = current.get("players") or []
    if not (0 <= opponent_index < len(players)):
        return ground_truth
    player = players[opponent_index]

    active_list = player.get("active") or []
    if active_list and revealed_active:
        _add_pokemon(ground_truth, active_list[0], "active")
    for bench_pokemon in player.get("bench") or []:
        _add_pokemon(ground_truth, bench_pokemon, "bench")
    for card in player.get("discard") or []:
        _add(ground_truth, card, "discard")

    # スタジアムと一時公開(looking)は共有領域。playerIndex で相手のものだけ拾う。
    for card in current.get("stadium") or []:
        if card and card.get("playerIndex") == opponent_index:
            _add(ground_truth, card, "stadium")
    for card in current.get("looking") or []:
        if card and card.get("playerIndex") == opponent_index:
            _add(ground_truth, card, "revealed")

    return ground_truth


def diff_against_ground_truth(observed_cards: list[Any], ground_truth: dict[int, dict[str, Any]]) -> dict[str, list]:
    """OpponentKnowledge.get_observed_cards() の結果と ground_truth を突き合わせて食い違いを返す。

    - missing: 神視点では今見えているのに、OpponentKnowledge が観測できていないカード（取りこぼしバグ）
    - extra: OpponentKnowledge は今見えていると思っているのに、神視点では見えないカード
      （ゾーンのクリア漏れ・所有者の取り違えバグ）
    - mismatched: 両方に存在するが card_id またはゾーンの記録が食い違っているカード（分類バグ）

    3つとも空リストなら、この瞬間については完全に一致している。
    """
    # OpponentKnowledge 側の「今現在、公開ゾーンに見えている」カードだけを対象にする
    # （current_zone が None＝過去に見えたが今は非公開領域にいるものは比較対象外）。
    tracked = {
        record.serial: record
        for record in observed_cards
        if record.serial is not None and record.current_zone is not None
    }

    missing = []
    mismatched = []
    for serial, truth in ground_truth.items():
        record = tracked.get(serial)
        if record is None:
            missing.append({"serial": serial, **truth})
            continue
        if record.card_id != truth["card_id"]:
            mismatched.append(
                {"serial": serial, "kind": "card_id", "expected": truth["card_id"], "actual": record.card_id}
            )
        elif record.current_zone != truth["zone"]:
            mismatched.append(
                {"serial": serial, "kind": "zone", "expected": truth["zone"], "actual": record.current_zone}
            )

    extra = [
        {"serial": serial, "card_id": record.card_id, "name": record.name, "zone": record.current_zone}
        for serial, record in tracked.items()
        if serial not in ground_truth
    ]

    return {"missing": missing, "extra": extra, "mismatched": mismatched}
