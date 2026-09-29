"""Proxy selection and authenticated CONNECT routing for native admission."""
import base64
import http.client
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admission import Admission


@pytest.mark.parametrize(('host', 'no_proxy', 'expected'), [
    ('api.example.com', '', True),
    ('api.example.com', '.example.com', False),
    ('example.com', 'example.com', False),
    ('api.example.com', 'example.com:443', False),
    ('api.example.com', 'other.com, .example.com', False),
    ('api.example.com', '*', False),
    ('notexample.com', 'example.com', True),
])
def test_proxy_selection_by_no_proxy(host, no_proxy, expected):
    gate = Admission(f'https://{host}', 1, env={'HTTPS_PROXY': 'http://proxy.test:8080', 'NO_PROXY': no_proxy})
    try:
        assert (gate.proxy is not None) is expected
    finally:
        gate.close()


def test_no_proxy_port_must_match_upstream_port():
    mismatch = Admission('https://api.example.com:8443', 1, env={
        'HTTPS_PROXY': 'http://proxy.test:8080', 'NO_PROXY': 'api.example.com:443',
    })
    try:
        assert mismatch.proxy is not None
    finally:
        mismatch.close()
    match = Admission('https://api.example.com:8443', 1, env={
        'HTTPS_PROXY': 'http://proxy.test:8080', 'NO_PROXY': 'api.example.com:8443',
    })
    try:
        assert match.proxy is None
    finally:
        match.close()


def test_proxy_precedence_lowercase_and_unsupported_scheme():
    gate = Admission('https://api.example.com', 1, env={
        'HTTPS_PROXY': 'http://upper.test:3128', 'https_proxy': 'http://lower.test:3128',
        'ALL_PROXY': 'http://all.test:3128',
    })
    try:
        assert gate.proxy.hostname == 'upper.test'
    finally:
        gate.close()
    gate = Admission('https://api.example.com', 1, env={'https_proxy': 'http://lower.test:3128'})
    try:
        assert gate.proxy.hostname == 'lower.test'
    finally:
        gate.close()
    with pytest.raises(ValueError, match='HTTP CONNECT'):
        Admission('https://api.example.com', 1, env={'HTTPS_PROXY': 'socks5://proxy.test:1080'})


def test_authenticated_connect_and_request_route(monkeypatch):
    captured = {}
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_CONNECT(self):
            captured['connect'] = (self.path, self.headers.get('Proxy-Authorization'))
            self.send_response(200, 'Connection Established')
            self.end_headers()
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    gate = Admission('https://api.example.com/base', 1,
                     env={'HTTPS_PROXY': f'http://user:p%40ss@127.0.0.1:{proxy.server_port}'})
    # A fake HTTPSConnection makes the post-CONNECT request inspectable without a TLS fixture.
    class FakeConnection:
        def __init__(self, *args, **kwargs): self.sock = object()
        def set_tunnel(self, host, port, headers=None):
            captured['tunnel'] = (host, port, headers)
        def connect(self): pass
        def request(self, method, route, body, headers):
            captured['request'] = (method, route)
        def close(self): pass
    monkeypatch.setattr('admission.http.client.HTTPSConnection', FakeConnection)
    class Request:
        path = gate.prefix + '/v1/messages?beta=1'
        headers = {'Content-Length': '2', 'Content-Type': 'application/json'}
        rfile = __import__('io').BytesIO(b'{}')
        wfile = __import__('io').BytesIO()
        connection = type('Sock', (), {'settimeout': lambda *a: None})()
        close_connection = False
        def send_response(self, *a): pass
        def send_header(self, *a): pass
        def end_headers(self): pass
    try:
        # Invoke the real handler with a lightweight server binding and bypass response streaming.
        req = Request()
        req.server = gate.server
        req.server.admission = gate
        # Handler construction dispatches do_POST, so provide successful empty response behavior.
        class FakeResponse:
            status = 200
            def getheader(self, key): return None
            def getheaders(self): return []
            def read1(self, size): return b''
        FakeConnection.getresponse = lambda self: FakeResponse()
        from admission import Handler
        Handler.do_POST(req)
        expected = 'Basic ' + base64.b64encode(b'user:p@ss').decode()
        assert captured['tunnel'] == ('api.example.com', 443, {'Proxy-Authorization': expected})
        assert captured['request'] == ('POST', '/base/v1/messages?beta=1')
    finally:
        gate.close()
        proxy.shutdown(); thread.join(); proxy.server_close()
