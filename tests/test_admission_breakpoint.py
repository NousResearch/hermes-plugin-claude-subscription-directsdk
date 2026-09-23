"""The message cache breakpoint must sit on content the next request replays unchanged.

Native attaches per-request context to the turn it answers and puts the single message
cache_control on or after it; the next request replays that turn without it, so the cached
prefix never recurs (issue #14, second cause). The relay anchors on the frame Hermes queried,
never on native's wording.
"""
import json

import pytest

from admission import pin_message_breakpoint

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
])
def test_breakpoint_moves_to_the_last_block_hermes_itself_sent(queried, messages, expected):
    raw = wire(messages)
    out = pin_message_breakpoint(raw, queried)
    assert marks(out) == [expected]
    assert unmarked(out) == unmarked(raw)  # only the directive moves, never content


@pytest.mark.parametrize('payload,queried', [
    (wire([ASSISTANT, {'role': 'user', 'content': [RESULT, with_marker(QUESTION)]}]), [RESULT, QUESTION]),
    (wire([{'role': 'user', 'content': [with_marker({'type': 'text', 'text': 'Any preamble'})]}]), [QUESTION]),
    (b'not json', [QUESTION]),
    (wire([ASSISTANT, {'role': 'user', 'content': [with_marker(RESULT)]}]), None),
])
def test_stable_breakpoint_or_nothing_to_anchor_forwards_unchanged(payload, queried):
    assert pin_message_breakpoint(payload, queried) == payload
