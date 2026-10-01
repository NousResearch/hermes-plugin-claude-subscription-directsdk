"""Guard host tracing metadata at the request-scoped native transport boundary.

Tests own their request dictionaries; no process, network, or credential is used.
Host trace headers must not become generation fields or native identity overrides.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk


class RequestHeaders(unittest.TestCase):
    def test_host_traceparent_is_accepted_without_forwarding(self) -> None:
        trace = '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01'
        request = {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'hello'}],
            'extra_headers': {'traceparent': trace},
        }
        encoded, manifest, names = directsdk.request_body(request)
        self.assertEqual(json.loads(encoded), {'tools': []})
        self.assertEqual(manifest, [])
        self.assertEqual(names, set())
        self.assertEqual(request['extra_headers'], {'traceparent': trace})
        self.assertNotIn(trace, encoded)


    def test_empty_headers_preserve_request_translation(self) -> None:
        for headers in (None, {}):
            with self.subTest(headers=headers):
                encoded, _, _ = directsdk.request_body({'extra_headers': headers})
                self.assertEqual(json.loads(encoded), {'tools': []})

    def test_non_trace_headers_and_invalid_containers_fail_closed(self) -> None:
        invalid = ([], '', {'Authorization': 'test-not-a-secret'},
                   {'x-api-key': 'test-not-a-secret'}, {'traceparent': None},
                   {1: 'trace'}, {'traceparent': 'trace', 'x-custom': 'value'})
        for headers in invalid:
            with self.subTest(headers=headers):
                with self.assertRaisesRegex(ValueError, 'host traceparent metadata only'):
                    directsdk.request_body({'extra_headers': headers})

    def test_trace_header_name_is_case_insensitive(self) -> None:
        encoded, _, _ = directsdk.request_body({'extra_headers': {'TraceParent': 'host-trace'}})
        self.assertEqual(json.loads(encoded), {'tools': []})

    def test_streamed_tool_request_accepts_host_trace_header(self) -> None:
        import tempfile
        from test_directsdk import Contract

        fixture = Contract()
        with tempfile.TemporaryDirectory() as tmp:
            client = fixture.client(tmp)
            try:
                request = fixture.request()
                request['extra_headers'] = {'traceparent': 'host-trace'}
                stream = client.chat.completions.create(**request, stream=True)
                try:
                    chunks = list(stream)
                finally:
                    stream.close()
                self.assertEqual(chunks[-1].choices[0].finish_reason, 'tool_calls')
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
