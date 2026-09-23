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


INJECTION_ANCHORS = ('<system-reminder>', "Today's date is", 'userEmail:')


def _is_moving_injection(block):
    """The CLI's per-request context block (date / userEmail reminder), heuristically
    distinguished from content that quotes one. Native >= 2.1.276 wraps it in
    ``<system-reminder>``; some model-specific builds (e.g. claude-opus-5-5 on
    2.1.280.d84) emit the date line unwrapped. It is either the whole block (anchored at
    the start) or, in tool rounds, appended to the end of the newest string
    ``tool_result``; list content is checked per inner block."""
    if block.get('type') == 'tool_result':
        inner = block.get('content')
        if isinstance(inner, list):
            return any(_is_moving_injection(b) for b in inner if isinstance(b, dict))
        return _starts_or_ends_with_injection(inner)
    return _starts_or_ends_with_injection(block.get('text'))


REMINDER_OPEN, REMINDER_CLOSE = '<system-reminder>', '</system-reminder>'
# Observed body openings of the native per-request reminder (2.1.280): the userEmail
# context preamble, or the date line. Anchored at the body start, not a substring.
INJECTION_BODY_PREFIXES = ("As you answer the user's questions, you can use the following context:",
                           "Today's date is")


def _starts_or_ends_with_injection(text):
    """True when text starts with an injection anchor, or ends with one complete
    ``<system-reminder>`` segment whose body opens like the native reminder (observed:
    native appends it to string tool_result content). Linear, no regex. A tool output that
    ends with an exact copy of that reminder is indistinguishable without provenance; the
    cost of a wrong guess is cache hits only, content is never changed."""
    if not isinstance(text, str):
        return False
    if text.startswith(INJECTION_ANCHORS):
        return True
    tail = text.rstrip()
    if not tail.endswith(REMINDER_CLOSE):
        return False
    start = tail.rfind(REMINDER_OPEN)
    if start < 0:
        return False
    body = tail[start + len(REMINDER_OPEN):-len(REMINDER_CLOSE)]
    return REMINDER_CLOSE not in body and body.lstrip().startswith(INJECTION_BODY_PREFIXES)


def relocate_message_breakpoint(payload):
    """Move the single message ``cache_control`` off the CLI's moving per-request injection.

    Native >= 2.1.276 appends a ``<system-reminder>`` (``Today's date is ...``, userEmail)
    to the newest turn and puts the message cache breakpoint on or after it. The next
    request moves that injection to the new newest turn, so the cached prefix never
    reappears and every tool round re-writes the whole history (issue #14, second cause;
    design by @robbyczgw-cla). When the only message breakpoint sits at or after the first
    CLI injection following the last assistant message, move it to the last cacheable
    (non-thinking) block before that injection. Content is never changed — only the
    ``cache_control`` directive moves — so a wrong heuristic guess costs cache hits, never
    correctness. Every other payload, including anything that fails to parse, forwards
    unchanged."""
    try:
        body = json.loads(payload)
        messages = body.get('messages')
        if not isinstance(messages, list):
            return payload
        blocks = [(i, j, block) for i, msg in enumerate(messages)
                  if isinstance(msg, dict) and isinstance(msg.get('content'), list)
                  for j, block in enumerate(msg['content']) if isinstance(block, dict)]
        marked = [(i, j) for i, j, block in blocks if 'cache_control' in block]
        if len(marked) != 1:
            return payload
        last_assistant = max((i for i, msg in enumerate(messages)
                              if isinstance(msg, dict) and msg.get('role') == 'assistant'), default=-1)
        injection = next(((i, j) for i, j, block in blocks
                          if i > last_assistant and _is_moving_injection(block)), None)
        if injection is None or marked[0] < injection:
            return payload
        target = next(((i, j) for i, j, block in reversed(blocks)
                       if (i, j) < injection and block.get('type') not in ('thinking', 'redacted_thinking')), None)
        if target is None:
            return payload
        by_position = {(i, j): block for i, j, block in blocks}
        by_position[target]['cache_control'] = by_position[marked[0]].pop('cache_control')
        return json.dumps(body, ensure_ascii=False).encode()
    except (ValueError, TypeError):
        return payload


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
    def __init__(self, upstream, timeout):
        self.upstream = urlsplit(upstream)
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
            payload = relocate_message_breakpoint(payload)
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
            gate.failure = type(exc).__name__
        finally:
            with gate.lock:
                gate.sockets.discard(self.connection)
                gate.sockets.discard(upstream_socket)
            if conn:
                conn.close()
            self.close_connection = True
