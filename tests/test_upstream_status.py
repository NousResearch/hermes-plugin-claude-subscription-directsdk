"""The first admitted HTTP error keeps its status for Hermes auxiliary fallback."""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from agent import auxiliary_client as aux

import directsdk

NATIVE = """
import json, os, sys, urllib.request, urllib.error
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type': 'result', 'num_turns': 0}), flush=True)
        continue
    break
try:
    urllib.request.urlopen(urllib.request.Request(os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages', data=b'{}'), timeout=5).read()
except urllib.error.HTTPError as error:
    print(json.dumps({'type': 'assistant', 'error': 'api_error', 'message': {'content': [{'type': 'text', 'text': str(error)}]}}), flush=True)
    print(json.dumps({'type': 'result', 'is_error': True, 'subtype': 'error_during_execution'}), flush=True)
"""


@pytest.mark.parametrize("status,should_fallback", [(429, True), (400, False)])
def test_admitted_status_drives_auxiliary_fallback(
    tmp_path, monkeypatch, status, should_fallback
):
    calls = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error"
                        if status == 429
                        else "invalid_request_error",
                        "message": "Error",
                    },
                }
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    peer = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / "native.py"
    native.write_text(NATIVE)
    client = directsdk.Client(
        command=[sys.executable, str(native)],
        env={
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{peer.server_port}",
        },
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            client.create(
                model="sonnet", messages=[{"role": "user", "content": "fixture"}]
            )
        exc = raised.value
        assert calls == ["/v1/messages"]
        assert exc.status_code == status
        assert not aux._is_payment_error(exc)
        assert aux._is_rate_limit_error(exc) is should_fallback

        # Drive Hermes' real provider-fallback rung, with only provider discovery stubbed.
        picked = []

        def choose(*args, **kwargs):
            picked.append(kwargs["reason"])
            return (
                SimpleNamespace(base_url="https://spare.example/v1"),
                "spare-model",
                "spare",
            )

        monkeypatch.setattr(aux, "_try_configured_fallback_chain", choose)
        route = SimpleNamespace(
            task="approval",
            tag="",
            resolved_provider="auto",
            client=client,
            main_runtime=None,
            base_info="",
            timeout=5,
            final_model="sonnet",
            route_info=None,
        )
        rung = aux._ladder_provider_fallback(exc, route)
        if should_fallback:
            step = next(rung)
            assert step.kind == "fallback"
            assert step.args[2] == "spare"
            assert picked == ["rate limit"]
            rung.close()
        else:
            with pytest.raises(StopIteration):
                next(rung)
            assert picked == []
    finally:
        client.close()
        peer.shutdown()
        thread.join()
        peer.server_close()
