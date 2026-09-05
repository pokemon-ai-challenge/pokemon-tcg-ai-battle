"""カードを「切り順」の観点で機能分類する。

ポケカの切り順のセオリーは **不確定情報に関わる行動を先に、確定情報だけの行動を最後に**
という一点に集約される。そこから

  1. ドロー系 → サーチ系 の順（引いた結果を見てからサーチを使えば温存できる）
  2. サーチ同士なら当たり確率の低いものから
  3. 手張り・ベンチ出し・逃げ（自分の手札情報だけで完結する確定行動）は最後
  4. 複数枚必要ならドロー優先、失敗したときだけサーチ

が導かれる。これらはすべて **「いま何が使えるか」の中での相対順序**なので、
方策に渡すには「この選択肢は何系か」「他に何系が使えるか」を知る必要がある。

本モジュールは ``CardData.skills[].text`` から役割を判定する。テキストは英語。
判定は保守的にし、迷ったら False を返す（誤分類より未分類のほうが安全）。

注意: 個別カードの識別は方策の card_embedding（1,268種×8次元）が既に担っている。
ここで作るのは**集合として数えるためのラベル**であって、カードの正体の代用ではない。
"""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_ai.shared import card_cache


@dataclass(frozen=True)
class CardRole:
    """1カードの機能ラベル。複数該当しうる（サーチしつつドローする等）。"""

    is_draw: bool = False          # 山札から手札を増やす（不確定・先に使う側）
    is_search: bool = False        # 山札から指定のカードを取る（不確定だが狙って取れる）
    is_deck_look: bool = False     # 山札の上を見る/並べ替える（山札圧縮・確認）
    is_hand_disrupt: bool = False  # 相手の手札に干渉する
    is_energy_accel: bool = False  # 山札/トラッシュからエネルギーを付ける
    is_recovery: bool = False      # トラッシュから手札/山札へ戻す

    @property
    def is_uncertain(self) -> bool:
        """不確定情報に関わる（＝先に使うべき側）か。"""
        return self.is_draw or self.is_search or self.is_deck_look


_NO_ROLE = CardRole()

# 判定に使う語。text は英語。小文字化して部分一致で見る。
_DRAW = ("draw a card", "draw 1 card", "draw 2 cards", "draw 3 cards", "draw 4 cards",
         "draw 5 cards", "draw 6 cards", "draw 7 cards", "draw cards until",
         "draw that many cards")
_SEARCH = ("search your deck",)
_DECK_LOOK = ("look at the top", "discard the top")
_HAND_DISRUPT = ("opponent shuffles their hand", "opponent discards cards from their hand",
                 "each player shuffles their hand", "opponent's hand", "opponents hand")
_ENERGY_ACCEL = ("attach a basic energy", "attach an energy", "attach that energy",
                 "attach 1 energy", "attach 2 energy")
_RECOVERY = ("from your discard pile into your hand", "from your discard pile to your hand",
             "discard pile and put it into your hand", "shuffle them into your deck")

_cache: dict[int, CardRole] = {}


def _text_of(card) -> str:
    return " ".join((s.text or "") for s in (getattr(card, "skills", None) or [])).lower()


def role_of(card_id: int) -> CardRole:
    """card_id の機能ラベル。未知IDやテキスト無しは全 False。"""
    hit = _cache.get(card_id)
    if hit is not None:
        return hit

    try:
        card = card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001 - 未知IDでも決定的に「役割なし」へ倒す
        _cache[card_id] = _NO_ROLE
        return _NO_ROLE

    text = _text_of(card)
    if not text:
        _cache[card_id] = _NO_ROLE
        return _NO_ROLE

    role = CardRole(
        is_draw=any(k in text for k in _DRAW),
        is_search=any(k in text for k in _SEARCH),
        is_deck_look=any(k in text for k in _DECK_LOOK),
        is_hand_disrupt=any(k in text for k in _HAND_DISRUPT),
        is_energy_accel=any(k in text for k in _ENERGY_ACCEL),
        is_recovery=any(k in text for k in _RECOVERY),
    )
    _cache[card_id] = role
    return role
