from __future__ import annotations

import ast
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SOURCE = APP_PATH.read_text(encoding="utf-8")
APP_AST = ast.parse(APP_SOURCE, filename=str(APP_PATH))


def _load_tier_helpers() -> dict:
    target_names = [
        "_tier_http_get",
    ]
    function_nodes = {
        node.name: node
        for node in APP_AST.body
        if isinstance(node, ast.FunctionDef) and node.name in target_names
    }
    namespace = {
        "os": os,
    }
    for name in target_names:
        module = ast.Module(body=[function_nodes[name]], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


HELPERS = _load_tier_helpers()


class TierHttpGetTests(unittest.TestCase):
    def test_retries_without_env_proxy_for_broken_loopback_proxy(self) -> None:
        call_trust_env: list[bool] = []

        class FakeProxyError(Exception):
            pass

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text
                self.encoding = None

            def raise_for_status(self) -> None:
                return None

        class FakeSession:
            def __init__(self) -> None:
                self.trust_env = True

            def get(self, url: str, headers: dict, timeout: int) -> FakeResponse:
                call_trust_env.append(self.trust_env)
                if self.trust_env:
                    raise FakeProxyError("proxy failed")
                return FakeResponse("ok")

        fake_requests = types.ModuleType("requests")
        fake_requests.Session = FakeSession
        fake_requests.exceptions = types.SimpleNamespace(ProxyError=FakeProxyError)

        with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:9"}, clear=False):
            with patch.dict(sys.modules, {"requests": fake_requests}):
                text = HELPERS["_tier_http_get"]("https://example.com")

        self.assertEqual(text, "ok")
        self.assertEqual(call_trust_env, [True, False])

    def test_reraises_proxy_error_when_proxy_env_is_not_broken_loopback(self) -> None:
        call_trust_env: list[bool] = []

        class FakeProxyError(Exception):
            pass

        class FakeSession:
            def __init__(self) -> None:
                self.trust_env = True

            def get(self, url: str, headers: dict, timeout: int) -> None:
                call_trust_env.append(self.trust_env)
                raise FakeProxyError("proxy failed")

        fake_requests = types.ModuleType("requests")
        fake_requests.Session = FakeSession
        fake_requests.exceptions = types.SimpleNamespace(ProxyError=FakeProxyError)

        env_patch = {
            "HTTP_PROXY": "http://proxy.example.com:8080",
            "HTTPS_PROXY": "http://proxy.example.com:8080",
            "ALL_PROXY": "http://proxy.example.com:8080",
            "http_proxy": "http://proxy.example.com:8080",
            "https_proxy": "http://proxy.example.com:8080",
            "all_proxy": "http://proxy.example.com:8080",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            with patch.dict(sys.modules, {"requests": fake_requests}):
                with self.assertRaises(FakeProxyError):
                    HELPERS["_tier_http_get"]("https://example.com")

        self.assertEqual(call_trust_env, [True])


if __name__ == "__main__":
    unittest.main()
