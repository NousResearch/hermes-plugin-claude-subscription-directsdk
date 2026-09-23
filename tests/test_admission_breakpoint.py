"""The message cache breakpoint must never ride the CLI's moving per-request injection.

Native >= 2.1.276 appends a <system-reminder> (date / userEmail) to the newest turn and
puts the single message cache_control on or after it; the next request moves the injection
to the new newest turn, so the cached prefix never reappears (issue #14, second cause).
"""
import json

from admission import relocate_message_breakpoint

MARKER = {'type': 'ephemeral'}
REMINDER = {'type': 'text', 'text': "<system-reminder>\nToday's date is 2026-09-23.\n</system-reminder>"}


def wire(messages):
    return json.dumps({'model': 'm', 'messages': messages}).encode()


def blocks_at(payload, index):
    return json.loads(payload)['messages'][index]['content']


def test_own_text_block_placement_moves_the_breakpoint():
    reminder = {**REMINDER, 'cache_control': MARKER}
    payload = relocate_message_breakpoint(wire([
        {'role': 'user', 'content': [{'type': 'text', 'text': 'question'}]},
        {'role': 'system', 'content': [reminder]},
    ]))
    messages = json.loads(payload)['messages']
    assert 'cache_control' not in messages[1]['content'][0]
    assert messages[0]['content'][0]['cache_control'] == MARKER
    assert messages[0]['content'][0]['text'] == 'question'  # content unchanged


def test_prefix_block_of_newest_user_message_moves_the_breakpoint():
    reminder = {**REMINDER, 'cache_control': MARKER}
    payload = relocate_message_breakpoint(wire([
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'answer'}]},
        {'role': 'user', 'content': [reminder, {'type': 'text', 'text': 'the real question'}]},
    ]))
    messages = json.loads(payload)['messages']
    assert 'cache_control' not in messages[1]['content'][0]
    assert messages[0]['content'][0]['cache_control'] == MARKER


def test_injection_inside_newest_tool_result_skips_thinking():
    tool_result = {'type': 'tool_result', 'tool_use_id': 't1',
                   'content': "<system-reminder>\nuserEmail: a@b.c\n</system-reminder>\nreal output",
                   'cache_control': MARKER}
    payload = relocate_message_breakpoint(wire([
        {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'signed', 'signature': 'sig'},
            {'type': 'tool_use', 'id': 't1', 'name': 'probe', 'input': {}}]},
        {'role': 'tool', 'content': [tool_result]},
    ]))
    messages = json.loads(payload)['messages']
    assert 'cache_control' not in messages[1]['content'][0]
    assert messages[0]['content'][1]['cache_control'] == MARKER  # tool_use, not the thinking block
    assert 'cache_control' not in messages[0]['content'][0]


def test_quoted_reminder_is_not_an_injection():
    quoted = {'type': 'text', 'text': 'the assistant said <system-reminder>\nToday\'s date is 2026-09-23.\n</system-reminder> yesterday',
              'cache_control': MARKER}
    payload = relocate_message_breakpoint(wire([{'role': 'user', 'content': [quoted]}]))
    assert blocks_at(payload, 0)[0] == quoted


def test_breakpoint_before_the_injection_is_left_alone():
    marked = {'type': 'text', 'text': 'question', 'cache_control': MARKER}
    messages = [{'role': 'user', 'content': [marked, REMINDER]}]
    assert json.loads(relocate_message_breakpoint(wire(messages)))['messages'] == messages


def test_unparseable_payload_forwards_unchanged():
    assert relocate_message_breakpoint(b'not json') == b'not json'