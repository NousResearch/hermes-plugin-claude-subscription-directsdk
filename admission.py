"""Request-scoped native HTTP admission; credentials are forwarded, never persisted."""
import codecs
import copy
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import re
import secrets
import socket
import ssl
import threading
from urllib.parse import urlsplit


UNCACHEABLE = ('thinking', 'redacted_thinking')


class QueriedTurnMismatch(ValueError):
    """The native request could not be reconciled with Hermes' tool results."""


def _plain(block):
    return {k: v for k, v in block.items() if k != 'cache_control'} if isinstance(block, dict) else block


def pin_message_breakpoint(payload, queried):
    """Keep the single message ``cache_control`` on content the next request replays unchanged.

    Native attaches per-request context (today's date, the account-email reminder, whatever a
    later CLI adds) to the turn it answers and puts the message breakpoint on or after it. The
    next request replays that turn without it, so the cached prefix never recurs and every
    tool round re-writes the whole history (issue #14, second cause).

    What does recur is known without reading native's text: everything through the last
    assistant message, plus the leading blocks of the newest turn that equal the frame Hermes
    queried. The first block native added or changed ends that span. When that block is a
    tool_result (parallel calls, native's reminder on the last result), a breakpoint on the
    unchanged results before it measured no cache hit even though they replay byte-identical
    (#33, cause unknown), so the span ends at the preceding assistant message.
    The breakpoint never moves later, content never changes, and any payload that does not
    parse forwards as is."""
    if not queried:
        return payload
    try:
        body = json.loads(payload)
        messages = body['messages']
        blocks = [(i, j, b) for i, m in enumerate(messages) if isinstance(m.get('content'), list)
                  for j, b in enumerate(m['content'])]
        marked = [(i, j, b) for i, j, b in blocks if isinstance(b, dict) and 'cache_control' in b]
        if len(marked) != 1:
            return payload
        last = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), default=-1)
        stable = [(i, j, b) for i, j, b in blocks if i <= last]
        newest = messages[last + 1] if last + 1 < len(messages) else {}
        if newest.get('role') == 'user' and isinstance(newest.get('content'), list):
            prefix = []
            for j, (sent, host) in enumerate(zip(newest['content'], queried)):
                if _plain(sent) != _plain(host):
                    break
                prefix.append((last + 1, j, sent))
            rest = newest['content'][len(prefix):]
            if not (rest and isinstance(rest[0], dict) and rest[0].get('type') == 'tool_result'):
                stable += prefix
        target = next(((i, j, b) for i, j, b in reversed(stable)
                       if isinstance(b, dict) and b.get('type') not in UNCACHEABLE), None)
        i, j, block = marked[0]
        if target is None or (target[0], target[1]) >= (i, j):
            return payload
        target[2]['cache_control'] = block.pop('cache_control')
        return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return payload


def restore_queried_turn(payload, queried):
    """Restore a uniquely anchored tool-result frame without dropping protected content.

    Only trailing bare text is removable. Split/joined text runs are equivalent
    for matching, but never across media, metadata or cache directives. Restore
    the host's original representation and retain unambiguously mapped markers.
    """
    if not isinstance(queried, list) or not queried:
        return payload
    strict = any(isinstance(block, dict) and block.get('type') == 'tool_result' for block in queried)
    # An ordinary user-only frame is outside restoration scope, byte for byte.
    if not strict:
        return payload

    def unmatched():
        raise QueriedTurnMismatch('Native request does not uniquely match Hermes tool results')

    def bare_text(block):
        return (isinstance(block, dict) and set(block) == {'type', 'text'}
                and block['type'] == 'text' and isinstance(block['text'], str))

    def restore_content(sent, host):
        # Scalar/list equivalence is allowed only for text with no metadata or markers.
        if isinstance(host, str) or isinstance(sent, str):
            def text(value):
                if isinstance(value, str):
                    return value
                if isinstance(value, list) and all(bare_text(b) for b in value):
                    return ''.join(b['text'] for b in value)
                return None
            left, right = text(sent), text(host)
            if left is None or right is None or not left.startswith(right):
                return unmatched()
            return copy.deepcopy(host)
        if isinstance(sent, list) and isinstance(host, list):
            return restore_sequence(sent, host)
        if sent != host:
            return unmatched()
        return copy.deepcopy(host)

    def restore_block(sent, host):
        if not isinstance(sent, dict) or not isinstance(host, dict):
            return unmatched()
        restored = copy.deepcopy(host)
        if host.get('type') == 'tool_result':
            left_error, right_error = sent.get('is_error', False), host.get('is_error', False)
            # Missing means false; numeric 0 is not a protocol boolean.
            if type(left_error) is not bool or type(right_error) is not bool or left_error != right_error:
                return unmatched()
            ignored = {'content', 'is_error', 'cache_control'}
            if {k: v for k, v in sent.items() if k not in ignored} != {k: v for k, v in host.items() if k not in ignored}:
                return unmatched()
            if ('content' in sent) != ('content' in host):
                return unmatched()
            if 'content' in host:
                restored['content'] = restore_content(sent['content'], host['content'])
        elif _plain(sent) != _plain(host):
            return unmatched()
        if 'cache_control' in sent:
            if 'cache_control' in host and sent['cache_control'] != host['cache_control']:
                return unmatched()
            restored['cache_control'] = copy.deepcopy(sent['cache_control'])
        return restored

    def restore_sequence(sent, host):
        restored = []
        i = j = 0
        while i < len(host):
            if j >= len(sent):
                return unmatched()
            if bare_text(host[i]) and bare_text(sent[j]):
                host_end, sent_end = i + 1, j + 1
                while host_end < len(host) and bare_text(host[host_end]):
                    host_end += 1
                while sent_end < len(sent) and bare_text(sent[sent_end]):
                    sent_end += 1
                original = ''.join(b['text'] for b in host[i:host_end])
                actual = ''.join(b['text'] for b in sent[j:sent_end])
                # Extensions before a protected host block are insertions, not suffixes.
                if not (actual.startswith(original) if host_end == len(host) else actual == original):
                    return unmatched()
                restored.extend(copy.deepcopy(host[i:host_end]))
                i, j = host_end, sent_end
            else:
                restored.append(restore_block(sent[j], host[i]))
                i += 1
                j += 1
        if any(not bare_text(b) for b in sent[j:]):
            return unmatched()
        return restored

    try:
        body = json.loads(payload)
        messages = body['messages']
        last_assistant = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), default=-1)
        newest = messages[last_assistant + 1] if last_assistant + 1 < len(messages) else {}
        native = newest.get('content') if newest.get('role') == 'user' else None
        if not isinstance(native, list):
            return unmatched()

        result_ids = [block.get('tool_use_id') for block in queried
                      if isinstance(block, dict) and block.get('type') == 'tool_result']
        if any(not isinstance(result_id, str) or not result_id for result_id in result_ids) or len(set(result_ids)) != len(result_ids):
            return unmatched()

        native_ids = [b.get('tool_use_id') for b in native if isinstance(b, dict) and b.get('type') == 'tool_result']
        # Unique ordered result IDs anchor the complete frame; no substring search or
        # reminder-pattern stripping can accidentally choose another host span.
        if result_ids != native_ids:
            return unmatched()
        restored = restore_sequence(native, queried)
        if native == restored:
            return payload
        newest['content'] = restored
        return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    except QueriedTurnMismatch:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return unmatched()


class Capture:
    def __init__(self):
        self.message = None
        self.complete = False
        self.pending = ''
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.arguments = {}

    def feed(self, chunk):
        self.pending += self.decoder.decode(chunk)
        while match := re.search(r'\r?\n\r?\n', self.pending):
            frame, self.pending = self.pending[:match.start()], self.pending[match.end():]
            data = '\n'.join(line[5:].lstrip(' ') for line in frame.splitlines() if line.startswith('data:'))
            if data:
                self.event(json.loads(data))

    def event(self, event):
        handler = {
            'message_start': self._start,
            'content_block_start': self._block_start,
            'content_block_delta': self._block_delta,
            'content_block_stop': self._block_stop,
            'message_delta': self._delta,
            'message_stop': self._stop,
        }.get(event['type'])
        if handler:
            handler(event)

    def _start(self, event):
        self.message = copy.deepcopy(event['message'])

    def _block_start(self, event):
        self.message['content'].append(copy.deepcopy(event['content_block']))

    def _block_delta(self, event):
        delta = event['delta']
        block = self.message['content'][event['index']]
        field = {'text_delta':'text', 'thinking_delta':'thinking', 'signature_delta':'signature'}.get(delta['type'])
        if field:
            block[field] = block.get(field, '') + delta[field]
        elif delta['type'] == 'input_json_delta':
            index = event['index']
            self.arguments[index] = self.arguments.get(index, '') + delta['partial_json']
        elif delta['type'] == 'citations_delta':
            block.setdefault('citations', []).append(copy.deepcopy(delta['citation']))

    def _block_stop(self, event):
        index = event['index']
        if index in self.arguments:
            # A no-argument tool call streams one input_json_delta with an empty partial_json.
            raw = self.arguments.pop(index)
            self.message['content'][index]['input'] = json.loads(raw) if raw.strip() else {}

    def _delta(self, event):
        self.message.update(event.get('delta', {}))
        self.message['usage'].update(event.get('usage', {}))

    def _stop(self, event):
        self.complete = bool(self.message and self.message.get('stop_reason') and not self.arguments)


class Admission:
    def __init__(self, upstream, timeout, queried=None):
        self.upstream = urlsplit(upstream)
        self.queried = queried
        host = self.upstream.hostname
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == 'localhost'
        if (self.upstream.scheme != 'https' and not (self.upstream.scheme == 'http' and local)) or not host or self.upstream.username or self.upstream.password or self.upstream.query or self.upstream.fragment:
            raise ValueError('Native upstream must be HTTPS or a loopback HTTP fixture')
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sockets = set()
        self.cancelled = False
        self.used = False
        self.denied = 0
        self.request_id = None
        self.status = None
        self.failure = None
        self.capture = Capture()
        self.error_body = b''
        self.prefix = '/admit/' + secrets.token_urlsafe(32)
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.admission = self
        self.url = f'http://127.0.0.1:{self.server.server_port}' + self.prefix
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval':.05}, daemon=True)
        self.thread.start()

    def error_text(self):
        """The upstream's own message for a non-200 answer, '' when none was captured."""
        text = self.error_body.decode('utf-8', errors='replace')
        try:
            message = json.loads(text)['error']['message']
        except (ValueError, KeyError, TypeError):
            return text
        return message if isinstance(message, str) else text

    def abort(self):
        with self.lock:
            self.cancelled = True
            for sock in self.sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # Peer may have closed between the read and cancellation.

    def close(self):
        self.abort()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Native authorization and the per-call route must never enter logs.

    def do_POST(self):
        gate = self.server.admission
        path = urlsplit(self.path)
        if path.path != gate.prefix + '/v1/messages' or self.headers.get('Origin'):
            self.send_error(404)
            return
        with gate.lock:
            if gate.cancelled or gate.used:
                gate.denied += 1
                body = b'{"type":"error","error":{"type":"invalid_request_error","message":"HERMES_MODEL_ADMISSION_CONSUMED"}}'
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            gate.used = True
            gate.sockets.add(self.connection)
        conn = None
        upstream_socket = None
        try:
            self.connection.settimeout(gate.timeout)
            payload = self.rfile.read(int(self.headers['Content-Length']))
            payload = restore_queried_turn(payload, gate.queried)
            payload = pin_message_breakpoint(payload, gate.queried)
            target = gate.upstream
            if target.scheme == 'https':
                conn = http.client.HTTPSConnection(target.hostname, target.port, timeout=gate.timeout, context=ssl.create_default_context())
            else:
                conn = http.client.HTTPConnection(target.hostname, target.port, timeout=gate.timeout)
            conn.connect()
            upstream_socket = conn.sock
            with gate.lock:
                if gate.cancelled:
                    return
                gate.sockets.add(upstream_socket)
            # Request identity and payload remain native; only HTTP transfer encoding changes.
            headers = {k:v for k,v in self.headers.items() if k.lower() not in ('host','connection','content-length','transfer-encoding','proxy-authorization','proxy-connection','accept-encoding')}
            headers['Accept-Encoding'] = 'identity'
            route = target.path.rstrip('/') + '/v1/messages' + ('?' + path.query if path.query else '')
            conn.request('POST', route, payload, headers)
            del headers, payload
            response = conn.getresponse()
            gate.request_id = response.getheader('request-id') or response.getheader('x-request-id')
            gate.status = response.status
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in ('connection','transfer-encoding','server','date'):
                    self.send_header(key, value)
            self.send_header('Connection', 'close')
            self.end_headers()
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                if response.status == 200:
                    gate.capture.feed(chunk)
                elif len(gate.error_body) < 65536:
                    gate.error_body += chunk  # The rejection reason ("prompt is too long", "adaptive thinking is not supported"), bounded.
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
            gate.failure = str(exc) if isinstance(exc, QueriedTurnMismatch) else type(exc).__name__
        finally:
            with gate.lock:
                gate.sockets.discard(self.connection)
                gate.sockets.discard(upstream_socket)
            if conn:
                conn.close()
            self.close_connection = True
