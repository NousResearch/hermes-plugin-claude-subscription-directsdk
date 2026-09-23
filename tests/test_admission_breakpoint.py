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


def test_unwrapped_date_line_is_an_injection():
    # Some model-specific CLI builds emit the date line without the <system-reminder> wrapper.
    bare = {'type': 'text', 'text': "Today's date is 2026-09-23.", 'cache_control': MARKER}
    payload = relocate_message_breakpoint(wire([
        {'role': 'user', 'content': [{'type': 'text', 'text': 'question'}]},
        {'role': 'system', 'content': [bare]},
    ]))
    messages = json.loads(payload)['messages']
    assert 'cache_control' not in messages[1]['content'][0]
    assert messages[0]['content'][0]['cache_control'] == MARKER


def test_breakpoint_before_the_injection_is_left_alone():
    marked = {'type': 'text', 'text': 'question', 'cache_control': MARKER}
    messages = [{'role': 'user', 'content': [marked, REMINDER]}]
    assert json.loads(relocate_message_breakpoint(wire(messages)))['messages'] == messages


def test_unparseable_payload_forwards_unchanged():
    assert relocate_message_breakpoint(b'not json') == b'not json'


EMAIL_REMINDER = ("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
                  "# userEmail\nThe user's email address is a@b.c.\n</system-reminder>")


def _tool_round(tool_content, date_role='system'):
    # observed on native 2.1.280 tool rounds: reminder appended to the newest string
    # tool_result, then a separate short date block that carries the only breakpoint
    return wire([
        {'role': 'assistant', 'content': [
            {'type': 'text', 'text': 'running'},
            {'type': 'tool_use', 'id': 't1', 'name': 'probe', 'input': {}},
        ]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': tool_content}]},
        {'role': date_role, 'content': [{'type': 'text', 'text': "Today's date is 2026-09-23.", 'cache_control': MARKER}]},
    ])


def _marks(payload):
    return [(i, j) for i, m in enumerate(json.loads(payload)['messages'])
            for j, b in enumerate(m['content']) if 'cache_control' in b]


def _without_marks(payload):
    body = json.loads(payload)
    for m in body['messages']:
        for b in m['content']:
            b.pop('cache_control', None)
    return body


def test_reminder_appended_to_string_tool_result_moves_marker_before_it():
    for role in ('system', 'user'):
        raw = _tool_round('1\n2\n3\n\n' + EMAIL_REMINDER + '\n', date_role=role)
        out = relocate_message_breakpoint(raw)
        assert _marks(out) == [(0, 1)]                  # the tool_use, before the moving reminder
        assert _without_marks(out) == _without_marks(raw)  # content untouched


def test_reminder_inside_list_tool_result_moves_marker_before_it():
    raw = _tool_round([{'type': 'text', 'text': 'real output'}, {'type': 'text', 'text': EMAIL_REMINDER}])
    assert _marks(relocate_message_breakpoint(raw)) == [(0, 1)]


def test_trailing_reminder_without_signature_is_not_an_injection():
    raw = _tool_round("match: '<system-reminder>'\nfoo </system-reminder>")
    assert _marks(relocate_message_breakpoint(raw)) == [(1, 0)]  # date block is still an injection


def test_signature_outside_the_trailing_segment_is_not_an_injection():
    content = "userEmail in docs\n<system-reminder>plain note</system-reminder>"
    raw = _tool_round(content)
    assert _marks(relocate_message_breakpoint(raw)) == [(1, 0)]


def test_only_the_last_complete_reminder_segment_counts():
    # not anchored at the start, so only the trailing-segment rule applies
    content = "output\n" + EMAIL_REMINDER + "\nmore\n<system-reminder>no signature here</system-reminder>"
    raw = _tool_round(content)
    assert _marks(relocate_message_breakpoint(raw)) == [(1, 0)]


def test_redacted_thinking_is_never_a_breakpoint_target():
    raw = wire([
        {'role': 'assistant', 'content': [
            {'type': 'text', 'text': 'ok'},
            {'type': 'redacted_thinking', 'data': 'opaque'}]},
        {'role': 'system', 'content': [{'type': 'text', 'text': "Today's date is 2026-09-23.", 'cache_control': MARKER}]},
    ])
    assert _marks(relocate_message_breakpoint(raw)) == [(0, 0)]


def test_tool_output_quoting_a_signature_in_a_trailing_reminder_is_not_an_injection():
    content = "output\n<system-reminder>Documentation mentions userEmail and Today's date is</system-reminder>"
    assert _marks(relocate_message_breakpoint(_tool_round(content))) == [(1, 0)]