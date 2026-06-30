"""Stdlib HTTP server for the Pokemon TCG battle viewer.

No third-party dependencies (only numpy, already required by the mPPO agent).
Run it and open http://localhost:8000 :

    python3 viewer/server.py            # default port 8000
    python3 viewer/server.py --port 9000

Endpoints (all JSON unless noted):
    GET  /                  -> static/index.html
    GET  /static/<file>     -> static asset
    GET  /api/meta          -> {agents, decks}
    POST /api/match         -> {ai0, ai1, deck0, deck1} -> AI vs AI replay
    POST /api/human/new     -> {humanSide, ai, humanDeck, aiDeck} -> first state
    POST /api/human/act     -> {action: [idx,...]} -> next state
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import agents as agents_mod  # noqa: E402
import decks as decks_mod  # noqa: E402
import engine as engine_mod  # noqa: E402

STATIC_DIR = os.path.join(_HERE, "static")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "TCGViewer/1.0"

    # -- low-level helpers ---------------------------------------------- #
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, message: str, code: int = 400):
        self._json({"error": message}, code)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("[viewer] " + (fmt % args) + "\n")

    # -- static --------------------------------------------------------- #
    def _serve_static(self, rel: str):
        rel = rel.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            return self._error("not found", 404)
        ext = os.path.splitext(full)[1].lower()
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    # -- routing -------------------------------------------------------- #
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/meta":
            return self._json({
                "agents": agents_mod.list_agents(),
                "decks": [
                    {"id": d["id"], "label": d["label"], "count": d["count"],
                     "valid": d["valid"], "error": d["error"]}
                    for d in decks_mod.list_decks()
                ],
            })
        return self._error("not found", 404)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self._read_json()
        try:
            if path == "/api/match":
                out = engine_mod.run_match(
                    body.get("ai0"), body.get("deck0"),
                    body.get("ai1"), body.get("deck1"))
                return self._json(out)
            if path == "/api/human/new":
                out = engine_mod.new_human_session(
                    int(body.get("humanSide", 0)),
                    body.get("ai"),
                    body.get("humanDeck"),
                    body.get("aiDeck"))
                return self._json(out)
            if path == "/api/human/act":
                action = body.get("action") or []
                action = [int(i) for i in action]
                return self._json(engine_mod.human_act(action))
            return self._error("not found", 404)
        except (ValueError, RuntimeError) as exc:
            return self._error(str(exc), 400)
        except Exception as exc:  # pragma: no cover - last-resort guard
            import traceback
            traceback.print_exc()
            return self._error(f"内部エラー: {exc}", 500)


def main():
    parser = argparse.ArgumentParser(description="Pokemon TCG battle viewer server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"ポケカ対戦ビジュアライザを起動しました → {url}")
    print("停止するには Ctrl+C")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
