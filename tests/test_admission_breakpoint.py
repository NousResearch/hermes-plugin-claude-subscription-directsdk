"""The message cache breakpoint must sit on content the next request replays unchanged.

Native attaches per-request context to the turn it answers and puts the single message
cache_control on or after it; the next request replays that turn without it, so the cached
prefix never recurs (issue #14, second cause). The relay anchors on the frame Hermes queried,
never on native's wording.
"""
import json
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from admission import Admission, QueriedTurnMismatch, pin_message_breakpoint, restore_queried_turn

MARKER = {'type': 'ephemeral'}
ASSISTANT = {'role': 'assistant', 'content': [
    {'type': 'thinking', 'thinking': 'signed', 'signature': 'sig'},
    {'type': 'tool_use', 'id': 't1', 'name': 'mcp__hermes__probe', 'input': {}}]}
RESULT = {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'real output'}
QUESTION = {'type': 'text', 'text': 'the real question'}


def wire(messages):
    return json.dumps({'model': 'm', 'messages': messages}).encode()


def marks(payload):
    return [(i, j) for i, m in enumerate(json.loads(payload)['messages'])
            for j, b in enumerate(m['content']) if 'cache_control' in b]


def unmarked(payload):
    body = json.loads(payload)
    for m in body['messages']:
        for b in m['content']:
            b.pop('cache_control', None)
    return body


def with_marker(block):
    return {**block, 'cache_control': MARKER}


# Shapes native has used plus wording no relay has seen: the breakpoint lands on the last
# block that is exactly Hermes' own content.
@pytest.mark.parametrize('queried,messages,expected', [
    # 2.1.280 tool round: reminder appended inside the tool_result, then a separate date message
    ([RESULT], [ASSISTANT,
                {'role': 'user', 'content': [{**RESULT, 'content': 'real output\n<system-reminder>userEmail</system-reminder>'}]},
                {'role': 'system', 'content': [with_marker({'type': 'text', 'text': "Today's date is 2026-09-23."})]}],
     (0, 1)),
    # an annotation after the host content, in wording no heuristic knows
    ([QUESTION], [ASSISTANT,
                  {'role': 'user', 'content': [QUESTION, with_marker({'type': 'text', 'text': 'Session context: v9 build'})]}],
     (1, 0)),
    # an annotation prepended to the newest turn: nothing of that turn recurs
    ([QUESTION], [ASSISTANT,
                  {'role': 'user', 'content': [{'type': 'text', 'text': 'Any new preamble'}, with_marker(QUESTION)]}],
     (0, 1)),
    # tool results then the user's own text, native's reminder in that text: the results recur
    ([RESULT, QUESTION], [ASSISTANT,
                          {'role': 'user', 'content': [RESULT, with_marker({**QUESTION, 'text': 'the real question\n<system-reminder>x</system-reminder>'})]}],
     (1, 0)),
    # every queried block unchanged, native's date in its own message after them
    ([RESULT], [ASSISTANT,
                {'role': 'user', 'content': [RESULT]},
                {'role': 'system', 'content': [with_marker({'type': 'text', 'text': "Today's date is 2026-09-23."})]}],
     (1, 0)),
    # a block native appends after unchanged tool results: the results recur
    ([RESULT], [ASSISTANT,
                {'role': 'user', 'content': [RESULT, with_marker({'type': 'text', 'text': 'Session context: v9 build'})]}],
     (1, 0)),
])
def test_breakpoint_moves_to_the_last_block_hermes_itself_sent(queried, messages, expected):
    raw = wire(messages)
    out = pin_message_breakpoint(raw, queried)
    assert marks(out) == [expected]
    assert unmarked(out) == unmarked(raw)  # only the directive moves, never content


@pytest.mark.parametrize('count', [2, 4])
def test_parallel_tool_results_do_not_pin_inside_a_partly_changed_user_message(count):
    results = [{'type': 'tool_result', 'tool_use_id': f't{i}', 'content': f'output {i}'}
               for i in range(count)]
    assistant = {'role': 'assistant', 'content': [
        {'type': 'tool_use', 'id': f't{i}', 'name': f'probe_{i}', 'input': {}}
        for i in range(count)]}
    changed_last = {**results[-1],
                    'content': results[-1]['content'] + '\n<system-reminder>native note</system-reminder>'}
    messages = [assistant, {'role': 'user', 'content': [
        *results[:-1], with_marker(changed_last)]}]
    raw = wire(messages)

    out = pin_message_breakpoint(raw, results)

    assert marks(out) == [(0, count - 1)]
    assert unmarked(out) == unmarked(raw)


@pytest.mark.parametrize('payload,queried', [
    (wire([ASSISTANT, {'role': 'user', 'content': [RESULT, with_marker(QUESTION)]}]), [RESULT, QUESTION]),
    (wire([{'role': 'user', 'content': [with_marker({'type': 'text', 'text': 'Any preamble'})]}]), [QUESTION]),
    (b'not json', [QUESTION]),
    (wire([ASSISTANT, {'role': 'user', 'content': [with_marker(RESULT)]}]), None),
])
def test_stable_breakpoint_or_nothing_to_anchor_forwards_unchanged(payload, queried):
    assert pin_message_breakpoint(payload, queried) == payload


def test_native_email_annotation_is_not_forwarded_as_tool_output():
    native = {**RESULT, 'content': 'real output\n<system-reminder>userEmail: matt@example.com</system-reminder>'}
    messages = [ASSISTANT, {'role': 'user', 'content': [native]},
                {'role': 'system', 'content': [{'type': 'text', 'text': "Today's date is 2026-09-30."}]}]
    out = json.loads(restore_queried_turn(wire(messages), [RESULT]))
    assert out['messages'][1]['content'] == [RESULT]
    assert out['messages'][2] == messages[2]


def test_original_reminder_in_tool_result_is_preserved():
    original = {**RESULT, 'content': 'real output\n<system-reminder>quoted fixture</system-reminder>'}
    native = {**original, 'content': original['content'] + '\n<system-reminder>userEmail</system-reminder>'}
    out = json.loads(restore_queried_turn(wire([ASSISTANT, {'role': 'user', 'content': [native]}]), [original]))
    assert out['messages'][1]['content'] == [original]


def test_parallel_results_and_native_trailing_block_restore_hermes_frame():
    second = {'type': 'tool_result', 'tool_use_id': 't2', 'content': 'second output'}
    native = [{**RESULT, 'content': RESULT['content'] + '\n<system-reminder>userEmail</system-reminder>'},
              second, {'type': 'text', 'text': 'native session context'}]
    out = json.loads(restore_queried_turn(wire([ASSISTANT, {'role': 'user', 'content': native}]), [RESULT, second]))
    assert out['messages'][1]['content'] == [RESULT, second]


def test_unmatched_native_tool_result_fails_closed():
    changed = {**RESULT, 'tool_use_id': 'different', 'content': 'real output\n<system-reminder>userEmail</system-reminder>'}
    raw = wire([ASSISTANT, {'role': 'user', 'content': [changed]}])
    with pytest.raises(QueriedTurnMismatch):
        restore_queried_turn(raw, [RESULT])


def test_ambiguous_native_tool_result_fails_closed():
    raw = wire([ASSISTANT, {'role': 'user', 'content': [RESULT, RESULT]}])
    with pytest.raises(QueriedTurnMismatch):
        restore_queried_turn(raw, [RESULT])


def test_unmatched_plain_user_turn_keeps_native_payload():
    raw = wire([ASSISTANT, {'role': 'user', 'content': [{**QUESTION, 'text': 'rewritten'}]}])
    assert restore_queried_turn(raw, [QUESTION]) == raw


def test_admission_forwards_hermes_tool_result_without_native_email():
    received = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(400)
            self.send_header('Content-Length', '0')
            self.end_headers()

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    gate = Admission(f'http://127.0.0.1:{peer.server_port}', 5, queried=[RESULT])
    native = {**RESULT, 'content': RESULT['content'] + '\n<system-reminder>userEmail</system-reminder>'}
    try:
        route = gate.url.removeprefix(f'http://127.0.0.1:{gate.server.server_port}') + '/v1/messages'
        conn = http.client.HTTPConnection('127.0.0.1', gate.server.server_port, timeout=5)
        conn.request('POST', route, wire([ASSISTANT, {'role': 'user', 'content': [native]}]),
                     {'Content-Type': 'application/json'})
        assert conn.getresponse().status == 400
        conn.close()
        assert received[0]['messages'][1]['content'] == [RESULT]
    finally:
        gate.close()
        peer.shutdown()
        thread.join()
        peer.server_close()


def test_admission_does_not_send_unmatched_tool_result_upstream():
    received = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200)
            self.end_headers()

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    gate = Admission(f'http://127.0.0.1:{peer.server_port}', 5, queried=[RESULT])
    changed = {**RESULT, 'tool_use_id': 'different'}
    try:
        route = gate.url.removeprefix(f'http://127.0.0.1:{gate.server.server_port}') + '/v1/messages'
        conn = http.client.HTTPConnection('127.0.0.1', gate.server.server_port, timeout=5)
        conn.request('POST', route, wire([ASSISTANT, {'role': 'user', 'content': [changed]}]),
                     {'Content-Type': 'application/json'})
        with pytest.raises(http.client.RemoteDisconnected):
            conn.getresponse()
        conn.close()
        assert gate.failure == 'Native request does not uniquely match Hermes tool results'
        assert received == []
    finally:
        gate.close()
        peer.shutdown()
        thread.join()
        peer.server_close()
