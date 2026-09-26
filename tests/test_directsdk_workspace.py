"""Every client of one OS user must reuse one workspace directory across requests.

The native CLI embeds its working directory in the request (environment-context
block, native >= ~2.1.276), so a fresh random tempdir per request breaks the
server-side prompt cache on every round of a tool loop (see issue #14): the
shared prefix collapses to static system+tools and the conversation tail is
re-written as cache each round. On Opus the block is a system message right after
the first user turn, so a per-client tempdir broke the cache whenever the host
built a new client.
"""
import importlib.util
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

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


REQUEST = dict(
    model="sonnet",
    messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "go"}],
    tools=[{"type": "function", "function": {"name": "probe", "description": "d"}}],
    stream=True,
    timeout=SimpleNamespace(read=5),
)


posix_only = pytest.mark.skipif(not hasattr(os, "getuid"), reason="the shared workspace is keyed by the POSIX uid")


def shared_path(tmp_path):
    return tmp_path / f"claude-directsdk-cwd-{os.getuid()}"


def fake_client(tmp_path):
    script = tmp_path / "native.py"
    script.write_text(FAKE)
    return native.Client(command=[sys.executable, str(script)], env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)})


def run_clients(tmp_path, count):
    clients = [fake_client(tmp_path) for _ in range(count)]
    try:
        for client in clients:
            list(client.create(**REQUEST))
            list(client.create(**REQUEST))
    finally:
        for client in clients:
            client.close()
    return (tmp_path / "cwds.log").read_text().splitlines()


@posix_only
def test_clients_of_one_user_share_a_workspace_that_survives_close_and_prune(tmp_path, monkeypatch):
    monkeypatch.setattr(native.tempfile, "tempdir", str(tmp_path))
    shared = shared_path(tmp_path)
    assert set(run_clients(tmp_path, 2)) == {str(shared)}
    assert shared.is_dir(), "the shared workspace must survive client close"
    shutil.rmtree(shared)
    client = fake_client(tmp_path)
    try:
        list(client.create(**REQUEST))
        os.utime(shared, (0, 0))
        list(client.create(**REQUEST))
    finally:
        client.close()
    assert (tmp_path / "cwds.log").read_text().splitlines()[-2:] == [str(shared)] * 2
    assert shared.stat().st_mtime > 0, "every request must refresh the workspace mtime"


def unsafe(kind, tmp_path, monkeypatch):
    root = tmp_path / "tmp"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(native.tempfile, "tempdir", str(root))
    if kind == "no_uid":
        monkeypatch.delattr(native.os, "getuid", raising=False)
        return
    shared = shared_path(root)
    if kind == "file":
        shared.write_text("not a directory")
    elif kind == "symlink":
        (tmp_path / "elsewhere").mkdir(mode=0o700)
        shared.symlink_to(tmp_path / "elsewhere")
    elif kind == "open_mode":
        shared.mkdir()
        shared.chmod(0o777)
    elif kind == "open_parent":
        root.chmod(0o777)


@pytest.mark.parametrize("kind", ["no_uid", *(pytest.param(k, marks=posix_only) for k in ("file", "symlink", "open_mode", "open_parent"))])
def test_unsafe_shared_workspace_falls_back_to_one_private_workspace_per_client(kind, tmp_path, monkeypatch):
    unsafe(kind, tmp_path, monkeypatch)
    client = fake_client(tmp_path)
    try:
        list(client.create(**REQUEST))
        private = (tmp_path / "cwds.log").read_text().splitlines()[0]
        shutil.rmtree(private)
        list(client.create(**REQUEST))
    finally:
        client.close()
    cwds = (tmp_path / "cwds.log").read_text().splitlines()
    assert cwds == [private, private] and "claude-directsdk-cwd-" in private, cwds
    assert not private.endswith(f"-{getattr(os, 'getuid', lambda: None)()}"), "an unsafe shared path must never be used"
    assert not Path(private).exists(), "a private workspace must be removed with its client"
