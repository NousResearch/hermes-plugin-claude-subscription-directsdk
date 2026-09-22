"""Explicit context routing: which native route a request selects, and when `auto` may move a request
from the included 200K route to the metered [1m] route.

The fixtures are a fake native that relays one upstream request through the admission relay (like real
Claude Code) and a loopback peer standing in for the API. Nothing here is paid-model evidence.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import directsdk
import model_catalog as catalog

# Behaves like `claude -p --include-partial-messages`: replays history, makes exactly one upstream
# request on ANTHROPIC_BASE_URL, streams the text deltas it receives, then reports the API's answer.
NATIVE = r'''
import json, os, sys, urllib.request, urllib.error
model = sys.argv[sys.argv.index('--model') + 1]
with open(os.environ['SPAWNS'], 'a') as log:
    log.write(model + '\n')
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type': 'result', 'num_turns': 0}), flush=True)
        continue
    break
url = os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages'
request = urllib.request.Request(url, data=json.dumps({'model': model}).encode(), headers={'Content-Type': 'application/json'})
try:
    raw = urllib.request.urlopen(request, timeout=5).read().decode()
except urllib.error.HTTPError as error:
    body = error.read().decode()
    print(json.dumps({'type': 'assistant', 'error': 'api_error', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'API Error: %d %s' % (error.code, body)}]}}), flush=True)
    print(json.dumps({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True, 'num_turns': 1}), flush=True)
    sys.exit(1)
text, stop = '', None
for line in raw.splitlines():
    if not line.startswith('data: '):
        continue
    event = json.loads(line[6:])
    if event['type'] == 'content_block_delta' and event['delta']['type'] == 'text_delta':
        text += event['delta']['text']
        print(json.dumps({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'delta': event['delta']}}), flush=True)
    elif event['type'] == 'message_delta':
        stop = event['delta']['stop_reason']
print(json.dumps({'type': 'assistant', 'message': {'id': 'msg', 'role': 'assistant', 'content': [{'type': 'text', 'text': text}], 'stop_reason': stop}}), flush=True)
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}), flush=True)
print(json.dumps({'type': 'result', 'subtype': 'success', 'num_turns': 1, 'usage': {'input_tokens': 1, 'output_tokens': 1}}), flush=True)
'''

USAGE = {'input_tokens': 5, 'output_tokens': 1, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}
TOO_LONG = b'{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}'


def sse(text, stop):
    events = [
        {'type': 'message_start', 'message': {'id': 'msg', 'role': 'assistant', 'model': 'sonnet', 'content': [], 'usage': USAGE}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': stop}, 'usage': USAGE},
        {'type': 'message_stop'},
    ]
    return ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode()


@pytest.fixture
def harness(tmp_path):
    """``(client_factory, peer_calls, spawns)``: the peer answers the included route per ``mode``
    (``reject``: 400 prompt-too-long; ``truncate``: 200 that stops at the window) and the [1m] route
    with a complete answer; ``spawns()`` lists the native ``--model`` selections in order."""
    calls = []
    mode = {'value': 'reject'}

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body['model'])
            if not body['model'].endswith('[1m]') and mode['value'] == 'reject':
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(TOO_LONG)))
                self.end_headers()
                self.wfile.write(TOO_LONG)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            if body['model'].endswith('[1m]'):
                self.wfile.write(sse('FIRST', 'end_turn'))
            else:
                self.wfile.write(sse('partial', 'model_context_window_exceeded'))

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    spawn_log = tmp_path / 'spawns'
    clients = []

    def make(peer_mode='reject', policy=None, **env):
        mode['value'] = peer_mode
        spawn_log.write_text('')
        client = directsdk.Client(command=[sys.executable, str(native)], context_routing=policy,
                                  env={'PATH': os.defpath, 'HOME': str(tmp_path), 'SPAWNS': str(spawn_log),
                                       'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{peer.server_port}', **env})
        clients.append(client)
        return client

    def spawns():
        return spawn_log.read_text().split()

    try:
        yield make, calls, spawns
    finally:
        for client in clients:
            client.close()
        peer.shutdown()
        thread.join()
        peer.server_close()


def request(model='sonnet', **kw):
    return dict(model=model, messages=[{'role': 'user', 'content': 'fixture'}], **kw)


def admission(result):
    return result.usage.model_dump()['native_admission']


@pytest.mark.parametrize('stream', [False, True])
def test_auto_retries_a_context_window_rejection_on_the_1m_route(harness, stream):
    make, calls, spawns = harness
    client = make('reject')
    result = client.create(**request(stream=stream))
    if stream:
        chunks = list(result)
        assert ''.join(c.choices[0].delta.content or '' for c in chunks) == 'FIRST'
        result = chunks[-1]
    else:
        assert result.choices[0].message.content == 'FIRST'
    assert result.choices[0].finish_reason == 'stop'
    # The included route was tried first and rejected before anything reached the caller.
    assert spawns() == ['claude-sonnet-5', 'claude-sonnet-5[1m]'] == calls
    assert admission(result) == {'upstream_requests': 2, 'blocked_requests': 0, 'request_id': None,
                                 'route': 'claude-sonnet-5[1m]', 'routes': ['claude-sonnet-5', 'claude-sonnet-5[1m]']}
    assert not client._requests


def test_auto_escalates_an_unseen_truncation_but_never_reruns_a_streamed_one(harness):
    make, calls, spawns = harness
    client = make('truncate')
    complete = client.create(**request())
    assert complete.choices[0].message.content == 'FIRST'
    assert complete.choices[0].finish_reason == 'stop'
    assert spawns() == ['claude-sonnet-5', 'claude-sonnet-5[1m]']
    # Streaming already delivered the partial text: no second run, the truncation is reported as `length`.
    client = make('truncate')
    chunks = list(client.create(**request(stream=True)))
    assert ''.join(c.choices[0].delta.content or '' for c in chunks) == 'partial'
    assert chunks[-1].choices[0].finish_reason == 'length'
    assert spawns() == ['claude-sonnet-5']
    assert admission(chunks[-1])['routes'] == ['claude-sonnet-5']


def test_always_200k_never_selects_1m_and_surfaces_the_rejection(harness):
    make, calls, spawns = harness
    client = make('reject', CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING='always-200k')
    with pytest.raises(RuntimeError, match='prompt is too long: 250000 tokens > 200000 maximum'):
        client.create(**request())
    assert spawns() == ['claude-sonnet-5']
    # Even an id that carries the suffix stays on the included route under this policy.
    client = make('truncate', CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING='200k')
    truncated = client.create(**request(model='claude-sonnet-5[1m]'))
    assert truncated.choices[0].finish_reason == 'length'
    assert spawns() == ['claude-sonnet-5']


def test_always_1m_and_explicit_1m_ids_start_on_the_1m_route(harness):
    make, calls, spawns = harness
    for policy, model, env in ((None, 'claude-sonnet-5[1m]', {}), (None, 'opus[1m]', {}),
                               ('always-1m', 'sonnet', {}),
                               # An explicit client policy beats the environment.
                               ('always-1m', 'sonnet', {'CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING': 'always-200k'}),
                               (None, 'sonnet', {'CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING': '1m'})):
        client = make('reject', policy, **env)
        result = client.create(**request(model=model))
        assert result.choices[0].message.content == 'FIRST'
        assert spawns() == [catalog.canonical_model(model) + '[1m]'], (policy, model, env)
        assert admission(result)['upstream_requests'] == 1
    # Haiku has no 1M route: always-1m keeps it on 200K, asking for it explicitly is an error.
    assert catalog.route_plan('haiku', 'always-1m') == ('claude-haiku-4-5-20251001',)
    with pytest.raises(ValueError, match='1M'):
        client.create(**request(model='haiku[1m]'))


def test_a_rejection_on_every_route_is_reported_as_the_native_error(harness, tmp_path):
    make, calls, spawns = harness
    # Make the peer reject the [1m] route too by asking for a model whose plan has one route only.
    client = make('reject', 'always-200k')
    with pytest.raises(RuntimeError, match='Native API error: prompt is too long'):
        client.create(**request())
    # Under auto, an unknown id has no [1m] alternative to try.
    client = make('reject')
    with pytest.raises(RuntimeError, match='prompt is too long'):
        client.create(**request(model='unqualified-future-model'))
    assert spawns() == ['unqualified-future-model']


def test_a_misspelt_policy_fails_the_request_without_spawning(harness):
    make, calls, spawns = harness
    client = make('reject', 'sometimes')
    with pytest.raises(ValueError, match='CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING must be one of auto, always-1m, always-200k'):
        client.create(**request())
    assert spawns() == [] and calls == []


def test_catalog_routes_and_windows_follow_the_policy():
    assert catalog.context_routing_policy(env={}) == 'auto'
    assert catalog.context_routing_policy(' Always-1M ', env={}) == 'always-1m'
    assert catalog.context_routing_policy(env={'CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING': '200K'}) == 'always-200k'
    with pytest.raises(ValueError):
        catalog.context_routing_policy('1M-only', env={})
    assert catalog.route_plan('sonnet') == ('claude-sonnet-5', 'claude-sonnet-5[1m]')
    assert catalog.route_plan('claude-opus-4-8[1m]') == ('claude-opus-4-8[1m]',)
    assert catalog.route_plan('fable', 'always-1m') == ('claude-fable-5-1[1m]',)
    assert catalog.route_plan('fable[1m]', 'always-200k') == ('claude-fable-5-1',)
    assert catalog.route_plan('claude-haiku-4-5') == ('claude-haiku-4-5-20251001',)
    assert catalog.route_plan('unqualified-future-model', 'always-1m') == ('unqualified-future-model',)
    assert catalog.picker_route('opus') == 'claude-opus-5'
    assert catalog.picker_route('opus[1m]') == 'claude-opus-5[1m]'
    with pytest.raises(ValueError, match='1M'):
        catalog.picker_route('haiku[1m]')
    assert catalog.native_route('opus') == 'claude-opus-5'
    assert catalog.context_window('opus') == 200_000
    assert catalog.context_window('opus[1m]') == 1_000_000
    assert catalog.context_window('opus', 'always-1m') == 1_000_000
    assert catalog.context_window('haiku', 'always-1m') == 200_000
    assert catalog.context_window('haiku[1m]') is None
    assert catalog.context_window('unqualified-future-model') is None
    assert catalog.MODEL_METADATA['claude-sonnet-5'] == {'canonical_model': 'claude-sonnet-5', 'context_window': 200_000}
    assert catalog.MODEL_METADATA['claude-sonnet-5[1m]'] == {'canonical_model': 'claude-sonnet-5', 'context_window': 1_000_000}
    assert 'claude-haiku-4-5-20251001[1m]' not in catalog.MODEL_METADATA
    assert directsdk.context_window_error('input length and `max_tokens` exceed context limit: 199000 + 8192 > 200000')
    assert not directsdk.context_window_error('rate limit exceeded')
