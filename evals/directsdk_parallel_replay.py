"""Real Claude CLI replay qualification against a loopback fake upstream.

Run inside a network namespace containing only loopback, with loopback enabled:
  HERMES_AGENT_REPO=<core> PYTHONPATH=<core> python evals/directsdk_parallel_replay.py /path/to/claude
No account credentials, real inference, raw requests or headers are recorded.
Usage and assistant signatures are deliberately synthetic. Receipts prove wire
continuity for this binary/version, not vendor-cache savings or OAuth coverage.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))
from directsdk_cache_wire import THINKING, TOOLS, USAGE, content, digest, nodes
import directsdk


def require_loopback_namespace():
    if {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise RuntimeError("Qualification requires a loopback-only network namespace")


def prefix(wire, position):
    i, j = position
    result = content({key: wire[key] for key in ("system", "tools", "messages")})
    result["messages"] = result["messages"][:i + 1]
    result["messages"][-1]["content"] = result["messages"][-1]["content"][:j + 1]
    return result


def static_directives(wire):
    """Track tools/system marker locations and exact settings independently of text."""
    found = []

    def walk(value, path):
        if isinstance(value, dict):
            if "cache_control" in value:
                found.append({"path": path, "settings": copy.deepcopy(value["cache_control"])})
            for key, child in value.items():
                if key != "cache_control":
                    walk(child, path + "." + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    for key in ("system", "tools"):
        walk(wire[key], "$." + key)
    return found


def queried_frame(wire, host):
    expected_ids = [b["tool_use_id"] for b in host if b.get("type") == "tool_result"]
    matches = [(i, message["content"]) for i, message in enumerate(wire["messages"])
               if message.get("role") == "user" and
               [b["tool_use_id"] for b in message["content"] if b.get("type") == "tool_result"] == expected_ids]
    assert expected_ids and len(matches) == 1, "queried tool frame is not unique"
    return matches[0]


def changed_paths(before, after, path="$"):
    """Return field paths/types only; never include prompt or auth values."""
    if type(before) is not type(after):
        return [{"path": path, "before_type": type(before).__name__, "after_type": type(after).__name__}]
    if isinstance(before, dict):
        changes = [{"path": path + "." + key, "change": "key_presence"}
                   for key in sorted(set(before) ^ set(after))]
        for key in sorted(set(before) & set(after)):
            changes.extend(changed_paths(before[key], after[key], path + "." + key))
        return changes
    if isinstance(before, list):
        changes = [] if len(before) == len(after) else [{"path": path, "change": "list_length"}]
        for index, (left, right) in enumerate(zip(before, after)):
            changes.extend(changed_paths(left, right, f"{path}[{index}]"))
        return changes
    return [] if before == after else [{"path": path, "change": "value"}]


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path.split("?")[0] == "/v1/messages/count_tokens":
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"input_tokens":100}')
            return
        if self.path.split("?")[0] != "/v1/messages":
            self.send_error(403)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.wires.append(body)
        number = len(self.server.wires)
        blocks = [THINKING, *[
            {"type": "tool_use", "id": f"toolu_public_{number}_{lane}",
             "name": body["tools"][0]["name"], "input": {}}
            for lane in ("a", "b")]]
        message = {"id": f"msg_public_{number}", "type": "message", "role": "assistant",
                   "model": body["model"], "content": [], "stop_reason": None,
                   "stop_sequence": None, "usage": USAGE}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(event):
            self.wfile.write(("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()

        emit({"type": "message_start", "message": message})
        for index, block in enumerate(blocks):
            start = ({"type": "thinking", "thinking": "", "signature": ""}
                     if block["type"] == "thinking" else {**block, "input": {}})
            emit({"type": "content_block_start", "index": index, "content_block": start})
            deltas = ([{"type": "thinking_delta", "thinking": block["thinking"]},
                       {"type": "signature_delta", "signature": block["signature"]}]
                      if block["type"] == "thinking" else [{"type": "input_json_delta", "partial_json": "{}"}])
            for delta in deltas:
                emit({"type": "content_block_delta", "index": index, "delta": delta})
            emit({"type": "content_block_stop", "index": index})
        emit({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": USAGE})
        emit({"type": "message_stop"})


def run(binary, model):
    require_loopback_namespace()
    with tempfile.TemporaryDirectory(prefix="directsdk-parallel-") as tmp, ThreadingHTTPServer(("127.0.0.1", 0), Peer) as peer:
        peer.wires = []
        worker = threading.Thread(target=peer.serve_forever, daemon=True)
        worker.start()
        env = {"PATH": os.defpath, "HOME": tmp, "HERMES_HOME": tmp, "TMPDIR": tmp,
               "CLAUDE_CONFIG_DIR": str(Path(tmp) / "config"), "XDG_CONFIG_HOME": tmp,
               "ANTHROPIC_API_KEY": "offline-fixture-not-a-credential",
               "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{peer.server_port}",
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
               "DISABLE_ERROR_REPORTING": "1", "NO_PROXY": "127.0.0.1,localhost",
               "no_proxy": "127.0.0.1,localhost"}
        env.update({key: "http://127.0.0.1:1" for key in
                    ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")})
        version = subprocess.check_output([str(binary), "--version"], env=env,
                                          stdin=subprocess.DEVNULL, text=True, timeout=15).strip()
        client = directsdk.Client(command=str(binary), env=env, timeout=45)
        history = [{"role": "system", "content": "PUBLIC SYNTHETIC SYSTEM\n" * 300}]
        for index in range(12):
            history.extend([{"role": "user", "content": f"Public earlier question {index}"},
                            {"role": "assistant", "content": f"Public earlier answer {index}"}])
        history.append({"role": "user", "content": "Run two public parallel probes."})
        receipts, previous, expected_signatures = [], None, 0
        original_static_directives = None
        try:
            for round_number in range(6):
                recreated = round_number == 4
                if recreated:
                    client.close()
                    history = json.loads(json.dumps(history))
                    client = directsdk.Client(command=str(binary), env=env, timeout=45)
                original, start = copy.deepcopy(history), len(peer.wires)
                response = client.create(model=model, messages=history, tools=TOOLS)
                assert history == original, "caller history mutated"
                assert len(peer.wires) == start + 1, "unexpected extra inference request"
                wire = peer.wires[-1]
                directives = static_directives(wire)
                if original_static_directives is None:
                    original_static_directives = directives
                assert directives == original_static_directives, "static marker locations/settings changed"
                message_markers = [(i, j, b["type"]) for i, msg in enumerate(wire["messages"])
                                   for j, b in enumerate(msg["content"]) if "cache_control" in b]
                assert len(message_markers) == 1, "expected one message breakpoint"
                markers = [node["cache_control"] for node in nodes(wire) if "cache_control" in node]
                assert sum(node == THINKING for node in nodes(wire["messages"])) == expected_signatures, "signed history changed"
                if round_number:
                    host = directsdk.prepare_history(history)[1][-1]["message"]["content"]
                    frame_index, frame_content = queried_frame(wire, host)
                    current = content(frame_content)
                    assert current == host, json.dumps({"error": "queried host representation not restored",
                                                       "round": round_number, "differences": changed_paths(host, current)})
                receipt = {"round": round_number, "client_recreated": recreated,
                           "breakpoint": message_markers[0], "cache_controls": markers,
                           "static_directives": directives}
                if round_number:
                    receipt["separate_trailing_native_frames"] = len(wire["messages"]) - frame_index - 1
                if previous is not None:
                    old_prefix, old_markers, position = previous
                    replayed = prefix(wire, position)
                    receipt.update(previous_prefix_sha256=digest(old_prefix), replayed_prefix_sha256=digest(replayed),
                                   prefix_identical=replayed == old_prefix, marker_settings_identical=markers == old_markers,
                                   differing_paths=changed_paths(old_prefix, replayed))
                receipts.append(receipt)
                i, j, _ = message_markers[0]
                previous = prefix(wire, (i, j)), markers, (i, j)
                assistant = response.choices[0].message.model_dump()
                assert len(assistant["tool_calls"]) == 2, "expected parallel results"
                history.append(assistant)
                expected_signatures += 1
                # Rotate string/list encodings and absent/explicit default flags independently.
                for lane, call in enumerate(assistant["tool_calls"]):
                    value = "Public tool result <system-reminder>literal host text</system-reminder> " + call["id"]
                    shape = (round_number + lane) % 3
                    value = (value if shape == 0 else [{"type": "text", "text": value}] if shape == 1 else
                             [{"type": "text", "text": value[:12]}, {"type": "text", "text": value[12:]}])
                    result = {"role": "tool", "tool_call_id": call["id"], "content": value}
                    if (round_number + lane) % 2:
                        result["is_error"] = False
                    history.append(result)
            passed = all(r.get("prefix_identical", True) and r.get("marker_settings_identical", True) for r in receipts)
            return {"passed": passed, "native_version": version, "native_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "requests": len(peer.wires), "parallel_tools_per_response": 2, "client_recreation_round": 4,
                    "initial_replayed_turn_pairs": 12, "requested_model": model,
                    "input_variants": ["string", "one text block", "split text blocks", "absent is_error", "is_error false"],
                    "network_isolation": "loopback-only namespace", "credentials": "dummy, isolated home",
                    "synthetic_usage_not_cache_measurement": True, "signed_history_preserved": True,
                    "prefix_receipts": receipts}
        finally:
            client.close()
            peer.shutdown()
            worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--model", default="claude-sonnet-5")
    args = parser.parse_args()
    result = run(args.binary.resolve(), args.model)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
