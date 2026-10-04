"""Negative controls for the native qualification's comparison and isolation guards."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parallel_replay_qualification", ROOT / "evals" / "directsdk_parallel_replay.py")
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def wire():
    return {"system": [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "tools": [{"name": "probe", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "output",
                                                        "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}


def test_prefix_comparison_detects_added_default_error_flag():
    before = wire()
    after = copy.deepcopy(before)
    after["messages"][0]["content"][0]["is_error"] = False
    assert qualification.prefix(before, (0, 0)) != qualification.prefix(after, (0, 0))


def test_prefix_comparison_detects_string_to_text_block_conversion():
    before = wire()
    after = copy.deepcopy(before)
    after["messages"][0]["content"][0]["content"] = [{"type": "text", "text": "output"}]
    assert qualification.prefix(before, (0, 0)) != qualification.prefix(after, (0, 0))


def test_prefix_ends_at_previous_marker_and_ignores_moving_directive_only():
    before = wire()
    after = copy.deepcopy(before)
    after["messages"][0]["content"][0].pop("cache_control")
    after["messages"][0]["content"].append({"type": "text", "text": "later", "cache_control": {"type": "ephemeral", "ttl": "1h"}})
    assert qualification.prefix(before, (0, 0)) == qualification.prefix(after, (0, 0))


def test_static_marker_comparison_detects_changed_settings_and_locations():
    before = wire()
    changed_ttl = copy.deepcopy(before)
    changed_ttl["system"][0]["cache_control"]["ttl"] = "5m"
    moved = copy.deepcopy(before)
    moved["system"].append({"type": "text", "text": "other", "cache_control": moved["system"][0].pop("cache_control")})
    assert qualification.static_directives(before) != qualification.static_directives(changed_ttl)
    assert qualification.static_directives(before) != qualification.static_directives(moved)


def test_diagnostics_do_not_include_prompt_values():
    before = {"text": "private-original-sentinel"}
    after = {"text": "private-changed-sentinel"}
    differences = qualification.changed_paths(before, after)
    assert differences == [{"path": "$.text", "change": "value"}]
    assert "private-" not in json.dumps(differences)


def test_network_guard_rejects_regular_host(monkeypatch):
    monkeypatch.setattr(qualification.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    with pytest.raises(RuntimeError, match="loopback-only"):
        qualification.require_loopback_namespace()


def test_network_guard_accepts_loopback_only(monkeypatch):
    monkeypatch.setattr(qualification.socket, "if_nameindex", lambda: [(1, "lo")])
    qualification.require_loopback_namespace()


def test_queried_frame_is_not_confused_with_separate_native_context():
    request = wire()
    host = request["messages"][0]["content"]
    request["messages"].append({"role": "user", "content": [{"type": "text", "text": "native context"}]})
    assert qualification.queried_frame(request, host) == (0, host)


def test_queried_frame_rejects_duplicate_identity():
    request = wire()
    host = request["messages"][0]["content"]
    request["messages"].append(copy.deepcopy(request["messages"][0]))
    with pytest.raises(AssertionError, match="not unique"):
        qualification.queried_frame(request, host)
