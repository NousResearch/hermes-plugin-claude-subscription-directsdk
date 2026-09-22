"""One client must reuse one workspace directory across requests.

The native CLI embeds its working directory in the request (environment-context
block, native >= ~2.1.276), so a fresh random tempdir per request breaks the
server-side prompt cache on every round of a tool loop (see issue #14): the
shared prefix collapses to static system+tools and the conversation tail is
re-written as cache each round.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("directsdk_workspace", ROOT / "directsdk.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)

FAKE = '''import json, os, pathlib, sys
pathlib.Path(os.path.join(os.environ["HOME"], "cwds.log")).open("a").write(str(pathlib.Path.cwd()) + "\\n")
for line in sys.stdin: pass
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}], "id": "m", "stop_reason": "end_turn"}}), flush=True)
print(json.dumps({"type": "stream_event", "event": {"type": "message_stop"}}), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "usage": {"input_tokens": 1, "output_tokens": 1}}), flush=True)
'''


def test_requests_of_one_client_share_a_workspace_and_close_removes_it(tmp_path):
    script = tmp_path / "native.py"
    script.write_text(FAKE)
    client = native.Client(
        command=[sys.executable, str(script)],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    request = dict(
        model="sonnet",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "probe", "description": "d"}}],
        stream=True,
        timeout=SimpleNamespace(read=5),
    )
    try:
        list(client.create(**request))
        list(client.create(**request))
        cwds = (tmp_path / "cwds.log").read_text().splitlines()
        assert len(cwds) == 2, cwds
        assert cwds[0] == cwds[1], f"workspace moved between requests: {cwds}"
        workspace = cwds[0]
    finally:
        client.close()
    assert not Path(workspace).exists(), "workspace must be removed with the client"
