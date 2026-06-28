import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
REPLAY_DIR = ROOT_DIR / "replays"


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/replays":
            self._send_json(
                [
                    {
                        "name": replay_path.name,
                        "size": replay_path.stat().st_size,
                    }
                    for replay_path in sorted(REPLAY_DIR.glob("*.json"))
                ]
            )
            return

        if self.path.startswith("/api/replays/"):
            file_name = unquote(self.path.removeprefix("/api/replays/"))
            replay_path = (REPLAY_DIR / file_name).resolve()
            if replay_path.parent != REPLAY_DIR.resolve() or not replay_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Replay not found.")
                return
            self._send_json(json.loads(replay_path.read_text(encoding="utf-8")))
            return

        super().do_GET()

    def _send_json(self, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), ViewerHandler)
    print("Battle review viewer: http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
