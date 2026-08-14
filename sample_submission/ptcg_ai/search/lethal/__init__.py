"""リーサル探索 Phase 1/2/3 の新パッケージ(設計: docs/plans/lethal-search-phase123/)。

Step 1-1 の時点では**探索本体は入っていない**。ここにあるのは、Step 0 の
capability report で確認した安全条件を実装前に固定するための土台だけである:

- ``types``       : Proof / ChanceClass / StopReason / 価値の種別(V_policy 等)
- ``scenario``    : 山札順列を包む型。**この型は outcome 生成の外へ出さない**(規則 R2)
- ``infokey``     : 決定ノードの情報集合キー。山札順序を含めない唯一の生成箇所(規則 R1/R2)
- ``engine``      : cg.api の薄いラッパ(資源解放・再生・再現性検査)
- ``chance``      : C/M/S の**能力ベース**分類(規則 R3)と B1 ガード(規則 R4)
- ``diagnostics`` : 診断ログに順列・非公開実体を書けなくする記録器(規則 R2b)
- ``value``       : critic 呼び出し契約(視点・UNAVAILABLE)。価値の種別を分離(規則 R7)

既存の ``ptcg_ai.search.lethal_simple`` は変更しない。selector への配線も
まだ行わない(Step 1-3)。
"""
