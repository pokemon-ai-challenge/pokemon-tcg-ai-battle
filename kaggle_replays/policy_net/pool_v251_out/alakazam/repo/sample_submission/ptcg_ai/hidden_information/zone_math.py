"""超幾何分布の閉形式計算をまとめた純粋関数群。

「未確認プール（山札∪サイド等）の中に、対象カードが特定ゾーンへどれだけ落ちているか」を
標準ライブラリの ``math.comb`` だけで厳密計算する。反復計算・近似・外部ライブラリは使わない。

``OwnHiddenState``（自分側、Phase 1）だけでなく、``OpponentHiddenState``（相手側、Phase 2）からも
再利用する共通基盤なので、特定モジュール固有のロジックは持ち込まない。

用語（design.md §4.2 と対応）:
    M = pool_size    未確認プールの総枚数（例: 自分の deckCount + len(prize)）
    n = prize_size   対象ゾーンの総枚数（例: len(prize)）
    k = target_count 対象カードがプール中に残っている枚数
    j                対象ゾーンにちょうど j 枚落ちている、という仮定
"""

import math


def prob_in_prize_exact(pool_size: int, prize_size: int, target_count: int, j: int) -> float:
    """対象カードがちょうど ``j`` 枚、対象ゾーンに落ちている確率。

    ``P(j) = C(k, j) * C(M-k, n-j) / C(M, n)``（超幾何分布の確率質量関数）。

    ``j`` が組み合わせ論的にありえない値（``j < 0``、``j > target_count``、
    ``j > prize_size``、または残り ``M-k`` 枚の非対象カードで ``n-j`` 枚を賄えない等）の場合は
    例外を出さず ``0.0`` を返す（呼び出し側が範囲チェックを毎回書かずに済むようにするため）。
    """
    pool_size_int, prize_size_int, target_count_int, j_int = pool_size, prize_size, target_count, j

    if j_int < 0 or j_int > target_count_int or j_int > prize_size_int:
        return 0.0
    non_target_needed = prize_size_int - j_int
    non_target_available = pool_size_int - target_count_int
    if non_target_needed < 0 or non_target_needed > non_target_available:
        return 0.0

    denominator = math.comb(pool_size_int, prize_size_int)
    if denominator == 0:
        return 0.0
    numerator = math.comb(target_count_int, j_int) * math.comb(non_target_available, non_target_needed)
    return numerator / denominator


def prob_in_prize(pool_size: int, prize_size: int, target_count: int, at_least: int = 1) -> float:
    """対象カードが対象ゾーンに ``at_least`` 枚以上落ちている確率（``prob_in_prize_exact`` の累積）。

    ``at_least=0`` を全域（``j = 0 .. min(target_count, prize_size)``）で呼べば、
    合計は必ず 1.0 になる（分布として整合していることの検算に使える。テスト方針の観点1）。
    """
    lo = max(0, at_least)
    hi = min(target_count, prize_size)
    return sum(prob_in_prize_exact(pool_size, prize_size, target_count, j) for j in range(lo, hi + 1))


def expected_in_prize(pool_size: int, prize_size: int, target_count: int) -> float:
    """対象カードが対象ゾーンに落ちている期待枚数（表示用のショートカット）。

    ``target_count * prize_size / pool_size``。超幾何分布の期待値の閉形式そのもの。
    """
    if pool_size <= 0:
        return 0.0
    return target_count * prize_size / pool_size


# ----------------------------------------------------------------------
# 3ゾーン（山札 / 手札 / サイド）対応の多変量超幾何（Phase 2: OpponentHiddenState 用）
#
# 相手側は「未観測プール M 枚」を『山札・手札・サイド』という複数ゾーンに配分する必要がある。
# ただし、あるカードが「特定の1ゾーンに何枚あるか」だけを問う限り、他ゾーンの内訳は無関係で、
# 「そのゾーン vs それ以外（残り全部）」という2値超幾何に帰着できる（多変量超幾何の周辺分布は
# 各成分ごとに2値超幾何になる、という標準的な性質）。そのため下記の関数は上の2値関数
# （``prob_in_prize`` / ``expected_in_prize``）をゾーンごとに呼び直すだけの薄いラッパにしてあり、
# 超幾何の確率計算そのものは1箇所（``prob_in_prize_exact``）にしか実装されていない
# （計算ロジックの二重実装を避ける、という設計方針。design.md §5.3 実装内容の注記）。


def prob_in_zone(
    pool_size: int, zone_sizes: dict[str, int], target_count: int, at_least: int = 1
) -> dict[str, float]:
    """各ゾーンに対象カードが ``at_least`` 枚以上ある確率を、ゾーン名 -> 確率 で返す。

    ``zone_sizes`` は ``{"deck": d, "hand": h, "prize": p}`` のような「ゾーン名 -> そのゾーンの
    総枚数」。各ゾーンについて「そのゾーン(``prize_size``) vs それ以外(``pool_size - そのゾーン``)」の
    2値超幾何に帰着し、``prob_in_prize`` をそのまま再利用する。

    注意: 戻り値の各ゾーン確率は「そのゾーンに少なくとも ``at_least`` 枚ある確率」であり、
    ゾーン間で排他ではない（同じカードが山札にも手札にもある確率をそれぞれ独立に評価している）。
    そのため ``at_least>=1`` の確率をゾーン間で足しても 1 にはならない（枚数の期待値の整合を
    見たい場合は ``expected_in_zone`` を使う）。``target_count==1``（1枚しか無いカード）に限っては
    排他になるので各ゾーン確率の合計が 1（= 枚数）に一致する。
    """
    return {
        zone: prob_in_prize(pool_size, size, target_count, at_least)
        for zone, size in zone_sizes.items()
    }


def expected_in_zone(
    pool_size: int, zone_sizes: dict[str, int], target_count: int
) -> dict[str, float]:
    """各ゾーンに落ちている期待枚数を、ゾーン名 -> 期待枚数 で返す。

    各ゾーン ``target_count * zone_size / pool_size``（``expected_in_prize`` をゾーンごとに再利用）。
    ゾーンサイズの合計が ``pool_size`` に等しい（``d + h + p == M``）とき、全ゾーンの期待枚数の
    合計は ``target_count`` に一致する（1枚のカードが必ずどこかのゾーンに居る、という保存則）。
    """
    return {
        zone: expected_in_prize(pool_size, size, target_count) for zone, size in zone_sizes.items()
    }
