import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from live_match import LiveMatchSession

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
WEB_DIR = ROOT_DIR / "web"
REPLAY_DIR = ROOT_DIR / "replays"
DEFAULT_DECK_PATH = REPO_ROOT / "sample_submission" / "deck.csv"
LOCAL_DECK_DIR = REPO_ROOT / "sample_submission" / "local_decks"
EXPORT_SCRIPT = ROOT_DIR / "export_replay.py"
CARD_IMAGES_DIR = WEB_DIR / "card_images"
BUILD_ASSETS_SCRIPT = ROOT_DIR / "build_card_assets.py"
# 画像抽出は pdfplumber / Pillow が要るので、まず pdf_card_editor の venv python を使う。
# この venv は .gitignore 対象（Python の venv は git 管理しないのが通例）なので、
# GitHub から clone した直後は誰の環境にも存在しない。_ensure_pdf_editor_venv() が
# 無ければ自動で作る（ユーザーがターミナルで手動セットアップしなくて済むように）。
PDF_EDITOR_DIR = REPO_ROOT / "cardlist_referenced" / "pdf_card_editor"
PDF_EDITOR_VENV = PDF_EDITOR_DIR / ".venv"
PDF_EDITOR_PY = PDF_EDITOR_VENV / "Scripts" / "python.exe"
PDF_EDITOR_REQUIREMENTS = PDF_EDITOR_DIR / "pdf_tool_requirements.txt"
LIVE_MATCH = LiveMatchSession()

# 同時に複数の生成が走ると cg エンジンやファイルが競合するので、生成は1件ずつに直列化する。
GENERATE_LOCK = threading.Lock()

# カード画像ビルドの進捗（バックグラウンド1本）。フロントは status をポーリングする。
# stage: None（未開始）/ "venv_setup"（初回のみ、venv作成+依存インストール中）/ "extracting"（PDFから抽出中）。
_IMG_BUILD = {"running": False, "done": False, "error": None, "count": 0, "stage": None}
_IMG_LOCK = threading.Lock()


def _ensure_pdf_editor_venv() -> str | None:
    """pdf_card_editor 用の venv が無ければ作成し、依存関係を入れる。

    成功時（既にある場合を含む）は None、失敗時はエラーメッセージを返す。
    ネットワークが無い環境等では失敗しうるので、失敗時は README の手動手順に
    フォールバックできるよう、具体的なエラーメッセージを返す。
    """
    if PDF_EDITOR_PY.exists():
        return None
    if not PDF_EDITOR_REQUIREMENTS.exists():
        return f"requirements file not found: {PDF_EDITOR_REQUIREMENTS}"

    with _IMG_LOCK:
        _IMG_BUILD["stage"] = "venv_setup"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(PDF_EDITOR_VENV)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0 or not PDF_EDITOR_PY.exists():
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return f"venv creation failed: {detail[-1] if detail else 'unknown error'}"

        proc = subprocess.run(
            [str(PDF_EDITOR_PY), "-m", "pip", "install", "--disable-pip-version-check",
             "-r", str(PDF_EDITOR_REQUIREMENTS)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return f"pip install failed: {detail[-1] if detail else 'unknown error'}"
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    finally:
        with _IMG_LOCK:
            _IMG_BUILD["stage"] = None
    return None


def _count_card_images() -> int:
    if not CARD_IMAGES_DIR.exists():
        return 0
    return sum(1 for _ in CARD_IMAGES_DIR.glob("*.jpg")) + sum(1 for _ in CARD_IMAGES_DIR.glob("*.png"))


def card_images_status() -> dict:
    with _IMG_LOCK:
        state = dict(_IMG_BUILD)
    state["count"] = _count_card_images()
    return state


def _run_card_image_build(mode: str) -> None:
    error = _ensure_pdf_editor_venv()
    if error is not None:
        # venv 自体が用意できなければ抽出は試みない（pdfplumber/Pillow が無い環境で
        # sys.executable にフォールバックしても同じ ImportError で失敗するだけなので）。
        with _IMG_LOCK:
            _IMG_BUILD["running"] = False
            _IMG_BUILD["done"] = True
            _IMG_BUILD["error"] = f"venv setup failed: {error}"
            _IMG_BUILD["count"] = _count_card_images()
        return

    with _IMG_LOCK:
        _IMG_BUILD["stage"] = "extracting"
    py = str(PDF_EDITOR_PY) if PDF_EDITOR_PY.exists() else sys.executable
    cmd = [py, str(BUILD_ASSETS_SCRIPT), "--all" if mode == "all" else "--deck"]
    error = None
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            error = detail[-1] if detail else "build_card_assets.py failed."
        elif _count_card_images() == 0:
            # venv はあるが PDF 自体が data/ に無いケース（画像は任意レイヤ）。
            error = "no images produced (PDF missing under data/)."
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        with _IMG_LOCK:
            _IMG_BUILD["running"] = False
            _IMG_BUILD["done"] = True
            _IMG_BUILD["error"] = error
            _IMG_BUILD["count"] = _count_card_images()
            _IMG_BUILD["stage"] = None


def start_card_image_build(mode: str) -> dict:
    """カード画像ビルドをバックグラウンドで開始する（多重起動しない）。即座に status を返す。"""
    with _IMG_LOCK:
        if _IMG_BUILD["running"]:
            return dict(_IMG_BUILD)
        _IMG_BUILD.update({"running": True, "done": False, "error": None, "stage": None})
    threading.Thread(target=_run_card_image_build, args=(mode,), daemon=True).start()
    return card_images_status()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def list_decks() -> list[dict]:
    decks = [
        {
            "value": _repo_relative(DEFAULT_DECK_PATH),
            "label": "deck.csv (submission default)",
            "kind": "default",
        }
    ]
    if LOCAL_DECK_DIR.exists():
        for deck_path in sorted(LOCAL_DECK_DIR.glob("*.csv"), key=lambda path: path.name.lower()):
            decks.append(
                {
                    "value": _repo_relative(deck_path),
                    "label": f"local_decks/{deck_path.name}",
                    "kind": "local",
                }
            )
    return decks


def validate_deck_choice(value: str | None, field_name: str) -> str:
    default_value = _repo_relative(DEFAULT_DECK_PATH)
    if value is None or str(value).strip() == "":
        return default_value
    allowed = {deck["value"] for deck in list_decks()}
    if value not in allowed:
        raise ValueError(f"{field_name} must be a known deck under sample_submission/local_decks.")
    return value


def validate_cpu_policy(value: str | None, field_name: str, default: str = "self") -> str:
    policy = default if value is None or str(value).strip() == "" else str(value)
    if policy not in {"self", "random"}:
        raise ValueError(f"{field_name} must be 'self' or 'random'.")
    return policy


def generate_replay(
    player_policy: str,
    opponent: str,
    seed: int,
    player_deck: str | None = None,
    opponent_deck: str | None = None,
) -> dict:
    """UI からの依頼で export_replay.py を別プロセスとして実行し、新しい replay を1件作る。

    サーバープロセス自身も cg をロードしている（live_match 経由）ため、盤面状態の競合を避けて
    別プロセスで動かす。生成した JSON のファイル名を返す。
    """
    player_policy = validate_cpu_policy(player_policy, "playerPolicy")
    opponent = validate_cpu_policy(opponent, "opponent", default="random")

    player_deck = validate_deck_choice(player_deck, "playerDeck")
    opponent_deck = validate_deck_choice(opponent_deck, "opponentDeck")

    with GENERATE_LOCK:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"ui-{player_policy}-vs-{opponent}-seed{seed}-{timestamp}.json"
        output_path = REPLAY_DIR / name
        cmd = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--player-policy", player_policy,
            "--opponent", opponent,
            "--seed", str(seed),
            "--output", str(output_path),
            "--player-deck", player_deck,
            "--opponent-deck", opponent_deck,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0 or not output_path.exists():
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise RuntimeError(detail[-1] if detail else "export_replay.py failed.")
        return {
            "name": name,
            "stdout": (proc.stdout or "").strip(),
            "playerPolicy": player_policy,
            "opponent": opponent,
            "playerDeck": player_deck,
            "opponentDeck": opponent_deck,
        }


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
                        "modifiedAt": replay_path.stat().st_mtime,
                    }
                    for replay_path in sorted(
                        REPLAY_DIR.glob("*.json"),
                        key=lambda path: path.stat().st_mtime,
                        reverse=True,
                    )
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

        if self.path == "/api/live/state":
            self._send_json(LIVE_MATCH.get_state())
            return

        if self.path == "/api/card_images/status":
            self._send_json(card_images_status())
            return

        if self.path == "/api/decks":
            self._send_json(list_decks())
            return

        super().do_GET()

    def do_POST(self) -> None:
        try:
            body = self._read_json_body()
            if self.path == "/api/live/start":
                cpu_policy = body.get("cpuPolicy", "self")
                player_deck = validate_deck_choice(body.get("playerDeck"), "playerDeck")
                opponent_deck = validate_deck_choice(body.get("opponentDeck"), "opponentDeck")
                self._send_json(
                    LIVE_MATCH.start(
                        cpu_policy=cpu_policy,
                        player_deck=player_deck,
                        opponent_deck=opponent_deck,
                    )
                )
                return

            if self.path == "/api/live/action":
                action = body.get("action")
                if not isinstance(action, list):
                    raise ValueError("action must be list[int].")
                self._send_json(LIVE_MATCH.submit_action(action))
                return

            if self.path == "/api/live/undo":
                self._send_json(LIVE_MATCH.undo_last_human_turn())
                return

            if self.path == "/api/live/stop":
                self._send_json(LIVE_MATCH.stop())
                return

            if self.path == "/api/replays/generate":
                player_policy = body.get("playerPolicy", "self")
                opponent = body.get("opponent", "random")
                seed = body.get("seed", 7)
                player_deck = body.get("playerDeck")
                opponent_deck = body.get("opponentDeck")
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    raise ValueError("seed must be an integer.")
                self._send_json(generate_replay(player_policy, opponent, seed, player_deck, opponent_deck))
                return

            if self.path == "/api/card_images/build":
                mode = body.get("mode", "deck")
                if mode not in {"deck", "all"}:
                    raise ValueError("mode must be 'deck' or 'all'.")
                self._send_json(start_card_image_build(mode))
                return

            self.send_error(HTTPStatus.NOT_FOUND, "API route not found.")
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # UI 開発中は app.js / styles.css の古いキャッシュで live 操作が壊れて見えやすい。
        if self.path.split("?", 1)[0].endswith((".html", ".js", ".css")) or self.path in {"/", ""}:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    port = int(os.environ.get("PORT", 8765))
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DECK_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), ViewerHandler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Battle review viewer: {url}")

    # アプリのように、起動したら自動でブラウザを開く。Launch *.cmd から起動した場合は
    # そちら側が（Human vs CPU 用のクエリ付き URL などを）開くので VIEWER_NO_OPEN=1 で抑止する。
    if not os.environ.get("VIEWER_NO_OPEN"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
        server.shutdown()


if __name__ == "__main__":
    main()
