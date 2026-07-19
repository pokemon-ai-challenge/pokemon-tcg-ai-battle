"""リプレイに登場するカードの画像を web/card_images/ へ書き出すビルドスクリプト。

`cardlist_referenced/pdf_card_editor` の抽出処理（`pdf_card_tool`）をそのまま再利用し、
`card_id -> 画像ファイル` を静的に用意します。生成物:

    battle_review_viewer/web/card_images/<card_id>.<ext>   … カード画像
    battle_review_viewer/web/card_images/manifest.json      … {card_id: filename} の対応表

ビューア（`serve_viewer.py`）は依存ゼロのまま、この静的ファイルを配信するだけで画像表示できます。
フロントは manifest を見て「画像があるカードだけ <img>、無ければテキスト表示」にフォールバックします。

依存（pdfplumber / Pillow）は pdf_card_editor 側の venv に入っています。実行例:

    cardlist_referenced/pdf_card_editor/.venv/Scripts/python.exe \
        battle_review_viewer/build_card_assets.py

PDF（data/Card_ID List_JP.pdf）が無い環境ではエラーで落とさず、警告して何もせず終了します
（画像はあくまで任意レイヤで、無くてもビューアは動く方針）。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
VIEWER_DIR = Path(__file__).resolve().parent
PDF_TOOL_DIR = ROOT_DIR / "cardlist_referenced" / "pdf_card_editor"
DEFAULT_PDF = ROOT_DIR / "data" / "Card_ID List_JP.pdf"
DEFAULT_OUT = VIEWER_DIR / "web" / "card_images"
DEFAULT_REPLAY_DIR = VIEWER_DIR / "replays"
DEFAULT_DECK_CSV = ROOT_DIR / "sample_submission" / "deck.csv"


def _import_pdf_tool():
    if str(PDF_TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(PDF_TOOL_DIR))
    try:
        import pdf_card_tool  # noqa: WPS433 (lazy import on purpose)
    except ModuleNotFoundError as error:
        print(
            "[build_card_assets] 必要なモジュールを import できませんでした:"
            f" {error}.\n"
            "  pdfplumber / Pillow が入った venv で実行してください。例:\n"
            "  cardlist_referenced/pdf_card_editor/.venv/Scripts/python.exe "
            "battle_review_viewer/build_card_assets.py",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return pdf_card_tool


def _collect_card_ids_from_obj(obj, found: set[int]) -> None:
    """JSON を再帰的に走査し、カードらしき dict（id と name を持つ）の id を集める。"""
    if isinstance(obj, dict):
        card_id = obj.get("id")
        if isinstance(card_id, int) and ("name" in obj or "serial" in obj):
            found.add(card_id)
        for value in obj.values():
            _collect_card_ids_from_obj(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _collect_card_ids_from_obj(value, found)


def collect_card_ids(replay_dir: Path, deck_csv: Path, use_deck: bool) -> set[int]:
    found: set[int] = set()

    if replay_dir.exists():
        for replay_path in sorted(replay_dir.glob("*.json")):
            try:
                payload = json.loads(replay_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                print(f"[build_card_assets] リプレイ読み込み失敗 {replay_path.name}: {error}", file=sys.stderr)
                continue
            _collect_card_ids_from_obj(payload.get("frames", []), found)

    if use_deck and deck_csv.exists():
        for line in deck_csv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.isdigit():
                found.add(int(line))

    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract card images for the battle review viewer.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="カード一覧 PDF（既定: data/Card_ID List_JP.pdf）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="出力先ディレクトリ")
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR, help="走査するリプレイ JSON のディレクトリ")
    parser.add_argument("--deck", action="store_true", help="deck.csv の card_id も対象に含める")
    parser.add_argument("--all", action="store_true", help="リプレイに関係なく、PDF 内の全カードを抽出する")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(
            f"[build_card_assets] PDF が見つかりません: {args.pdf}\n"
            "  画像は任意レイヤです。PDF を data/ に置くと画像表示が有効になります。何もせず終了します。",
            file=sys.stderr,
        )
        return

    pdf_card_tool = _import_pdf_tool()

    catalog = pdf_card_tool.load_pdf_catalog(args.pdf)
    page_by_card_id = {card.card_id: card.target_page for card in catalog.cards}

    if args.all:
        wanted_ids = set(page_by_card_id)
        print(f"[build_card_assets] 全カード対象: {len(wanted_ids)} 件")
    else:
        wanted_ids = collect_card_ids(args.replay_dir, DEFAULT_DECK_CSV, args.deck)
        print(f"[build_card_assets] リプレイ/デッキから収集した card_id: {len(wanted_ids)} 件")

    target_pages = sorted({page_by_card_id[cid] for cid in wanted_ids if cid in page_by_card_id})
    missing_in_catalog = sorted(cid for cid in wanted_ids if cid not in page_by_card_id)
    if missing_in_catalog:
        print(f"[build_card_assets] カタログに無い card_id（画像化スキップ）: {missing_in_catalog[:20]}{' ...' if len(missing_in_catalog) > 20 else ''}")

    if not target_pages:
        print("[build_card_assets] 対象カードがありません。終了します。")
        return

    asset_map = pdf_card_tool.get_card_asset_map(args.pdf, target_pages)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    written = 0
    for asset in asset_map.values():
        ext = asset.image_path.suffix.lower() or ".png"
        file_name = f"{asset.card_id}{ext}"
        dest = args.out / file_name
        shutil.copyfile(asset.image_path, dest)
        manifest[str(asset.card_id)] = file_name
        written += 1

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[build_card_assets] 画像 {written} 件を書き出しました -> {args.out}")
    print(f"[build_card_assets] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
