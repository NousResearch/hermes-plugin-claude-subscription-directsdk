"""One upstream admission and first-response authority over native recovery."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evals'))
import directsdk
from directsdk_admission import upstream_closed

NATIVE = r'''
import json, os, sys, urllib.request, urllib.error
for line in sys.stdin:
    frame=json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type':'result','num_turns':0}),flush=True)
        continue
    break
url=os.environ['ANTHROPIC_BASE_URL']+'/v1/messages'
for _ in range(2):
    try:
        urllib.request.urlopen(urllib.request.Request(url,data=b'{}',headers={'Content-Type':'application/json'}),timeout=5).read()
    except urllib.error.HTTPError:
        break
print(json.dumps({'type':'assistant','message':{'id':'first','role':'assistant','content':[{'type':'text','text':'FIRST'}]}}))
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}))
print(json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':0,'output_tokens':0}}))
'''


@pytest.mark.parametrize('stop', ['end_turn', 'max_tokens', 'model_context_window_exceeded'])
def test_first_response_owns_usage_and_stops_recovery(tmp_path, stop):
    calls = []
    usage = {'input_tokens':0, 'output_tokens':0, 'cache_read_input_tokens':0, 'cache_creation_input_tokens':0}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            events = [
                {'type':'message_start','message':{'id':'first','role':'assistant','model':'sonnet','content':[], 'usage':usage}},
                {'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}},
                {'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'FIRST'}},
                {'type':'content_block_stop','index':0},
                {'type':'message_delta','delta':{'stop_reason':stop},'usage':usage},
                {'type':'message_stop'},
            ]
            self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        result=client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
        assert len(calls)==1
        assert result.choices[0].message.content=='FIRST'
        assert result.choices[0].finish_reason==('stop' if stop=='end_turn' else 'length')
        assert result.usage.prompt_tokens==0
        assert result.choices[0].message.reasoning_details[0]['messages'][0]['stop_reason']==stop
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_empty_tool_input_completes_the_capture(tmp_path):
    """A no-argument tool call streams an empty input_json_delta; the capture must still complete."""
    calls = []
    usage = {'input_tokens':0, 'output_tokens':0, 'cache_read_input_tokens':0, 'cache_creation_input_tokens':0}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            events = [
                {'type':'message_start','message':{'id':'first','role':'assistant','model':'sonnet','content':[], 'usage':usage}},
                {'type':'content_block_start','index':0,'content_block':{'type':'tool_use','id':'toolu_1','name':'mcp__hermes__list_things','input':{}}},
                {'type':'content_block_delta','index':0,'delta':{'type':'input_json_delta','partial_json':''}},
                {'type':'content_block_stop','index':0},
                {'type':'message_delta','delta':{'stop_reason':'tool_use'},'usage':usage},
                {'type':'message_stop'},
            ]
            self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    tools=[{'type':'function','function':{'name':'list_things','description':'list','parameters':{'type':'object','properties':{}}}}]
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        result=client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}],tools=tools)
        assert len(calls)==1
        call=result.choices[0].message.tool_calls[0]
        assert call.function.name=='list_things'
        assert json.loads(call.function.arguments)=={}
        assert result.choices[0].finish_reason=='tool_calls'
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_cancel_closes_the_active_upstream_socket(tmp_path, monkeypatch):
    entered, disconnected = threading.Event(), threading.Event()
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            entered.set()
            self.close_connection = True
            if upstream_closed(self.rfile):
                disconnected.set()
    spawned = []
    spawn = directsdk.Request.spawn
    monkeypatch.setattr(directsdk.Request, 'spawn', lambda self, *a, **k: spawned.append(spawn(self, *a, **k)) or spawned[-1])
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    client = directsdk.Client(command=[sys.executable, str(native)], env={'PATH':os.defpath, 'HOME':str(tmp_path), 'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client.create, model='sonnet', messages=[{'role':'user', 'content':'fixture'}])
            try:
                assert entered.wait(5)
            finally:
                client.cancel()
            with pytest.raises(RuntimeError, match='cancelled'):
                result.result(timeout=3)
            assert disconnected.wait(2)
            assert spawned and all(process.wait(timeout=5) is not None for process in spawned)
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


class _Read:
    def __init__(self, outcome): self.outcome = outcome
    def read(self, size):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize('outcome', [
    b'',
    ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host'),
    ConnectionAbortedError(10053, 'An established connection was aborted by the software in your host machine'),
], ids=['eof', 'winerror-10054', 'winerror-10053'])
def test_upstream_closed_counts_eof_reset_and_abort_as_closed(outcome):
    # Windows often reports the relay's teardown as 10054/10053 rather than EOF; both are a closed socket.
    assert upstream_closed(_Read(outcome)) is True


def test_upstream_closed_does_not_swallow_unrelated_errors():
    with pytest.raises(TimeoutError):
        upstream_closed(_Read(TimeoutError('timed out')))
    with pytest.raises(OSError):
        upstream_closed(_Read(OSError(22, 'invalid argument')))


def test_upstream_closed_sees_a_real_reset():
    import socket, struct
    left, right = socket.socketpair()
    with left, right:
        left.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))  # close() sends RST
        left.close()
        with right.makefile('rb') as rfile:
            assert upstream_closed(rfile) is True


def test_upstream_closed_never_passes_a_socket_that_is_still_open():
    import socket
    left, right = socket.socketpair()
    with left, right, right.makefile('rb') as rfile:
        verdict, done = [], threading.Event()
        watcher = threading.Thread(target=lambda: (verdict.append(upstream_closed(rfile)), done.set()), daemon=True)
        watcher.start()
        assert not done.wait(0.5)  # Open and silent: never reported closed.
        left.sendall(b'x')
        assert done.wait(5) and verdict == [False]  # Open and talking: not closed either.
        watcher.join(5)


def test_incomplete_upstream_error_names_the_first_attempt(tmp_path):
    """Native's retries are denied with ADMISSION_CONSUMED; the raised error must carry the first attempt's status."""
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body=b'{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 213000 tokens > 200000 maximum"}}'
            self.send_response(529); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        with pytest.raises(RuntimeError, match=r'status 529, capture incomplete.*upstream said: prompt is too long: 213000 tokens'):
            client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


@pytest.mark.parametrize('status', [429, 401, 400, 500, 529, 200])
def test_incomplete_upstream_error_carries_the_first_status(tmp_path, status):
    """Hermes routes on status_code: the first attempt's real HTTP status, none for a 200 cut short."""
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            if status == 200:
                # Headers and message_start, then the connection drops: no complete capture.
                self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
                self.wfile.write(b'data: {"type":"message_start","message":{"id":"cut","role":"assistant","content":[]}}\n\n')
                return
            body=b'{"type":"error","error":{"type":"api_error","message":"fixture"}}'
            self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        with pytest.raises(RuntimeError, match=rf'^Incomplete upstream response \(first upstream attempt: status {status}, capture incomplete') as raised:
            client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
        assert raised.value.status_code == (None if status == 200 else status)
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_invalid_stream_json_error_names_the_offending_line(tmp_path):
    """A native that prints a non-JSON stdout line (a shim banner) fails with that line in the error, not a bare label."""
    native=tmp_path/'native.py'; native.write_text("import sys\nprint('mise WARN tool not activated')\nsys.exit(0)\n")
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':'http://127.0.0.1:9'})
    try:
        with pytest.raises(RuntimeError, match=r"Invalid native stream-json output: 'mise WARN tool not activated"):
            client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
    finally:
        client.close()


def test_silent_upstream_fails_fast_instead_of_hanging(tmp_path, monkeypatch):
    """Headers, one event, then silence (a dead path that keeps TCP ESTAB): the relay must error out
    within the idle bound, not wait out the request timeout while Hermes sees nothing."""
    import time
    import admission
    monkeypatch.setattr(admission, 'UPSTREAM_IDLE_SECONDS', 1)
    release = threading.Event()
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            start = {'type':'message_start','message':{'id':'m','role':'assistant','model':'sonnet','content':[],'usage':{'input_tokens':0,'output_tokens':0}}}
            self.wfile.write(('data: '+json.dumps(start)+'\n\n').encode()); self.wfile.flush()
            release.wait(60)
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE.replace('timeout=5', 'timeout=60'))
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    began = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match=r'status 200, capture incomplete, relay failure UpstreamIdle'):
            client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}],timeout=20)
        assert time.monotonic() - began < 10
    finally:
        release.set(); client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_pings_keep_a_slow_upstream_alive(tmp_path, monkeypatch):
    """Long thinking streams only pings for a while; bytes inside the idle bound must never fail it."""
    import time
    import admission
    monkeypatch.setattr(admission, 'UPSTREAM_IDLE_SECONDS', 1)
    usage = {'input_tokens':0, 'output_tokens':0}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            send = lambda e: (self.wfile.write(('data: '+json.dumps(e)+'\n\n').encode()), self.wfile.flush())
            send({'type':'message_start','message':{'id':'first','role':'assistant','model':'sonnet','content':[],'usage':usage}})
            for _ in range(6):  # 3 s of pings, three times the patched idle bound
                time.sleep(.5); send({'type':'ping'})
            send({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})
            send({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'FIRST'}})
            send({'type':'content_block_stop','index':0})
            send({'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':usage}); send({'type':'message_stop'})
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE.replace('timeout=5', 'timeout=60'))
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        assert client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}],timeout=20).choices[0].message.content == 'FIRST'
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()
