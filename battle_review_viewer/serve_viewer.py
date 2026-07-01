"""Stdlib HTTP server for the Battle Review Viewer.

No third-party deps for serving. Card images are extracted from the JP card PDF
on demand (card_images.py) and cached by the underlying pdf tool.

    python battle_review_viewer/serve_viewer.py            # http://localhost:8010
    python battle_review_viewer/serve_viewer.py --port 9000

Endpoints:
    GET /                      -> web/index.html
    GET /web/<file>            -> web asset
    GET /api/replays           -> [{name, meta}]  (latest first)
    GET /api/replay?name=<f>   -> replay JSON {meta, frames}
    GET /api/cards             -> {cards:{id:{...}}, attacks:{id:{name,damage}}}
    GET /cards/<id>            -> card face image (jpg/png) or 404
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SAMPLE = _REPO / "sample_submission"
_VIEWER = _REPO / "viewer"
for p in (str(_HERE), str(_SAMPLE), str(_VIEWER)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.api import all_attack, all_card_data  # noqa: E402
import card_images  # noqa: E402  (battle_review_viewer/card_images.py)
import export_replay  # noqa: E402  (battle_review_viewer/export_replay.py)
import agents as vagents  # noqa: E402  (viewer/agents.py: AI registry + drop-in AIs)
import decks as vdecks  # noqa: E402  (viewer/decks.py: deck discovery/loading)

WEB = _HERE / "web"
REPLAYS = _HERE / "replays"

# cg.game is a single global battle -> serialise match runs.
_MATCH_LOCK = threading.RLock()
_CARDS = None


def run_match(body: dict) -> dict:
    """Play P0 vs P1 with the chosen agents/decks, save a replay, return its info."""
    p0, p1 = body.get("p0"), body.get("p1")
    d0, d1 = body.get("deck0"), body.get("deck1")
    seed = body.get("seed")
    with _MATCH_LOCK:
        a0 = vagents.build_agent(p0)
        a1 = vagents.build_agent(p1)
        deck0 = vdecks.load_deck(d0)
        deck1 = vdecks.load_deck(d1)
        frames, result = export_replay.play(deck0, deck1, a0, a1, seed)
        out = export_replay.save_replay(
            frames, result,
            meta_extra={"p0": p0, "p1": p1, "deck0": d0, "deck1": d1,
                        "opponent": p1, "seed": seed},
            tag=f"{p0}-vs-{p1}",
        )
    return {"name": out.name, "meta": {"frameCount": len(frames), "result": result,
                                       "p0": p0, "p1": p1}}


def cards_payload() -> dict:
    global _CARDS
    if _CARDS is None:
        cards = {}
        for c in all_card_data():
            cards[c.cardId] = {
                "name": c.name,
                "cardType": int(c.cardType),
                "energyType": int(c.energyType) if c.energyType is not None else None,
                "hp": c.hp,
                "ex": bool(c.ex),
                "basic": bool(c.basic),
                "stage1": bool(c.stage1),
                "stage2": bool(c.stage2),
            }
        for cid, nm in card_images.japanese_names().items():
            if cid in cards:
                cards[cid]["nameJp"] = nm
        attacks = {a.attackId: {"name": a.name, "damage": a.damage} for a in all_attack()}
        _CARDS = {"cards": cards, "attacks": attacks}
    return _CARDS


def list_replays() -> list[dict]:
    if not REPLAYS.is_dir():
        return []
    items = []
    for f in REPLAYS.glob("*.json"):
        meta = {}
        try:
            with f.open(encoding="utf-8") as fh:
                meta = json.load(fh).get("meta", {})
        except Exception:
            meta = {}
        items.append({"name": f.name, "meta": meta, "mtime": f.stat().st_mtime})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


class Handler(BaseHTTPRequestHandler):
    server_version = "BattleReviewViewer/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[review] " + (fmt % args) + "\n")

    def _send(self, code, body: bytes, ctype: str, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: Path, ctype: str | None = None, cache: str = "no-store"):
        if not path.is_file():
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype, cache)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                return self._json(run_match(body))
            except Exception as exc:
                return self._json({"error": str(exc)}, 400)
        return self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._file(WEB / "index.html", "text/html; charset=utf-8")
        if path.startswith("/web/"):
            rel = unquote(path[len("/web/"):])
            full = (WEB / rel).resolve()
            if not str(full).startswith(str(WEB.resolve())):
                return self._send(404, b"not found", "text/plain")
            return self._file(full)
        if path == "/api/replays":
            return self._json(list_replays())
        if path == "/api/replay":
            q = parse_qs(parsed.query)
            name = (q.get("name") or [""])[0]
            f = (REPLAYS / name).resolve()
            if not name or not str(f).startswith(str(REPLAYS.resolve())) or not f.is_file():
                return self._json({"error": "replay not found"}, 404)
            return self._file(f, "application/json; charset=utf-8")
        if path == "/api/cards":
            return self._json(cards_payload())
        if path == "/api/agents":
            return self._json(vagents.list_agents())
        if path == "/api/decks":
            return self._json([{k: v for k, v in d.items() if k != "path"}
                               for d in vdecks.list_decks()])
        if path.startswith("/cards/"):
            raw = path[len("/cards/"):].split(".", 1)[0]
            try:
                cid = int(raw)
            except ValueError:
                return self._send(400, b"bad id", "text/plain")
            img = card_images.card_image_path(cid)
            if img is None or not Path(img).is_file():
                return self._send(404, b"no image", "text/plain")
            ctype = "image/jpeg" if str(img).lower().endswith((".jpg", ".jpeg")) else "image/png"
            return self._file(Path(img), ctype, cache="public, max-age=604800")
        return self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Battle Review Viewer server")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Battle Review Viewer → http://{args.host}:{args.port}")
    print("停止するには Ctrl+C")
    if not card_images.available():
        print("[warn] data/Card_ID List_JP.pdf が見つかりません。カード画像はテキスト表示にフォールバックします。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
