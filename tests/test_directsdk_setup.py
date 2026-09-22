"""Setup for the provider is driven by the Claude CLI itself: its auth status gates the flow and its
own model picker (initialize handshake) supplies the list, with the pinned catalog as fallback."""
import json
import os
import sys
import textwrap


FAKE_CLI = textwrap.dedent('''
    import json, os, sys
    state = json.loads(os.environ["FAKE_STATE"])
    if sys.argv[1:3] == ["auth", "status"]:
        print(json.dumps(state["auth"])); sys.exit(0 if state["auth"]["loggedIn"] else 1)
    assert "-p" in sys.argv and "--input-format" in sys.argv, sys.argv
    req = json.loads(sys.stdin.readline())
    assert req["request"]["subtype"] == "initialize"
    if state.get("hang_upstream"):
        import urllib.request
        urllib.request.urlopen(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages", data=b"{}")
    print(json.dumps({"type": "control_response", "response": {"subtype": "success", "request_id": req["request_id"],
          "response": {"models": state["models"], "account": state["account"]}}}))
''')


def _cli(tmp_path, state):
    path = tmp_path / "claude.py"
    path.write_text(FAKE_CLI)
    return [sys.executable, str(path)], {**os.environ, "FAKE_STATE": json.dumps(state), "PATH": os.defpath}


def test_setup_status_reports_login_and_models_from_the_cli(profile, tmp_path):
    state = {"auth": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"},
             "account": {"subscriptionType": "Claude Pro"},
             "models": [
                 {"value": "sonnet[1m]", "resolvedModel": "claude-sonnet-5[1m]", "displayName": "Sonnet 5 (1M context)", "description": "Sonnet 5 for long sessions"},
                 {"value": "opus", "resolvedModel": "claude-opus-5", "displayName": "Opus", "description": "Opus 5 · Best for everyday, complex tasks"},
                 {"value": "opus[1m]", "resolvedModel": "claude-opus-5[1m]", "displayName": "Opus (1M context)", "description": "Opus 5 with 1M context · Draws from usage credits · $5/$25 per Mtok"},
                 {"value": "haiku", "resolvedModel": "claude-haiku-4-5-20251001", "displayName": "Haiku", "description": "Haiku 4.5 · Fastest for quick answers"},
             ]}
    command, env = _cli(tmp_path, state)
    status = profile.setup_status(command=command, env=env)
    assert status["available"] and status["logged_in"] and status["plan"] == "Claude Pro"
    assert status["login_command"] == command + ["auth", "login"]

    models = profile.discover_models(command=command, env=env)
    ids = [m["id"] for m in models]
    # Every native picker row stays its own route: `opus` (included 200K) and `opus[1m]` (metered 1M)
    # are two rows, and only the row the CLI marks "Draws from usage credits" carries the note.
    assert ids == ["claude-sonnet-5[1m]", "claude-opus-5", "claude-opus-5[1m]", "claude-haiku-4-5-20251001"]
    assert [m["label"] for m in models] == ["Sonnet 5 for long sessions", "Opus 5", "Opus 5 with 1M context", "Haiku 4.5"]
    assert [m["note"] for m in models] == ["", "", "usage credits", ""]
    # Discovery goes through the admission relay with zero upstream requests
    assert all(m["upstream_requests"] == 0 for m in models)


def test_discovery_keeps_base_and_1m_rows_distinct_with_plan_rule_notes(profile, tmp_path):
    """A Pro picker with both Fable rows: the plan rule marks each Fable row, the Sonnet rows stay
    unmarked, a duplicate alias row merges into its route instead of adding a fourth Sonnet entry,
    and a `[1m]` row for a 200K-only model is dropped rather than invented."""
    state = {"auth": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"},
             "account": {"subscriptionType": "Claude Pro"},
             "models": [
                 {"value": "sonnet", "resolvedModel": "claude-sonnet-5", "displayName": "Sonnet", "description": "Sonnet 5 · Best balance"},
                 {"value": "sonnet[1m]", "resolvedModel": "claude-sonnet-5[1m]", "displayName": "Sonnet (1M context)", "description": "Sonnet 5 with 1M context · Long sessions"},
                 {"value": "default", "resolvedModel": "claude-sonnet-5", "displayName": "Default (recommended)", "description": "Sonnet 5 · Recommended"},
                 {"value": "fable", "resolvedModel": "claude-fable-5-1", "displayName": "Fable", "description": "Fable 5.1 · Most capable"},
                 {"value": "fable[1m]", "resolvedModel": "claude-fable-5-1[1m]", "displayName": "Fable (1M context)", "description": "Fable 5.1 with 1M context"},
                 {"value": "haiku[1m]", "resolvedModel": "claude-haiku-4-5-20251001[1m]", "displayName": "Haiku (1M)", "description": "not a real route"},
                 {"value": "unknown", "resolvedModel": "claude-future-9", "displayName": "Future", "description": "Future"},
             ]}
    command, env = _cli(tmp_path, state)
    # Request routing preferences affect calls, not the account offers displayed in the picker.
    env["CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING"] = "always-200k"
    models = profile.discover_models(command=command, env=env)
    assert [(m["id"], m["note"]) for m in models] == [
        ("claude-sonnet-5", ""), ("claude-sonnet-5[1m]", ""),
        ("claude-fable-5-1", "usage credits"), ("claude-fable-5-1[1m]", "usage credits")]
    assert [m["label"] for m in models] == ["Sonnet 5", "Sonnet 5 with 1M context", "Fable 5.1", "Fable 5.1 with 1M context"]
    assert profile.fetch_models(command=command, env=env) == [m["id"] for m in models]


def test_logged_out_or_missing_cli_degrades_to_pinned_catalog(profile, tmp_path):
    command, env = _cli(tmp_path, {"auth": {"loggedIn": False, "authMethod": "none"}, "account": {}, "models": []})
    status = profile.setup_status(command=command, env=env)
    assert status["available"] and not status["logged_in"]
    assert profile.discover_models(command=command, env=env) is None

    missing = profile.setup_status(command=[str(tmp_path / "nope")], env=env)
    assert not missing["available"] and not missing["logged_in"]
    assert "install" in missing["detail"].lower()
