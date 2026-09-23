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


PRO = {"auth": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"},
       "account": {"subscriptionType": "Claude Pro"}}
PINNED_PICKER = [
    {"value": "sonnet[1m]", "resolvedModel": "claude-sonnet-5[1m]", "displayName": "Sonnet 5 (1M context)", "description": "Sonnet 5 for long sessions"},
    {"value": "opus", "resolvedModel": "claude-opus-5-5", "displayName": "Opus", "description": "Opus 5.5 · Best for everyday, complex tasks"},
    {"value": "opus[1m]", "resolvedModel": "claude-opus-5-5[1m]", "displayName": "Opus (1M context)", "description": "Opus 5.5 with 1M context · Draws from usage credits · $4/$20 per Mtok"},
    {"value": "haiku", "resolvedModel": "claude-haiku-4-5-20251001", "displayName": "Haiku", "description": "Haiku 4.5 · Fastest for quick answers"},
]
# Models the pinned table has never heard of (fictitious on purpose): a plain + [1m] pair behind the
# CLI's own `opus` alias, a plain-only id in a family Hermes guesses at 1M, and a [1m]-only id.
UNPINNED_PICKER = [
    {"value": "opus", "resolvedModel": "claude-opus-9", "displayName": "Opus", "description": "Opus 9 · Best for everyday, complex tasks"},
    {"value": "opus[1m]", "resolvedModel": "claude-opus-9[1m]", "displayName": "Opus (1M context)", "description": "Opus 9 with 1M context · Draws from usage credits · $5/$25 per Mtok"},
    {"value": "claude-sonnet-5-9", "resolvedModel": "claude-sonnet-5-9", "displayName": "Sonnet", "description": "Sonnet 5.9 · Efficient for routine tasks"},
    {"value": "claude-fable-9[1m]", "resolvedModel": "claude-fable-9[1m]", "displayName": "Fable", "description": "Fable 9 · Most capable for your hardest tasks"},
    # An alias row the CLI leaves unresolved is not a model and must not become a route.
    {"value": "default", "displayName": "Default (recommended)", "description": "Opus 9 · Best for everyday, complex tasks"},
]


def _discover(profile, tmp_path, models):
    command, env = _cli(tmp_path, {**PRO, "models": models})
    return profile.discover_models(command=command, env=env)


def test_setup_status_reports_login_and_models_from_the_cli(profile, tmp_path):
    command, env = _cli(tmp_path, {**PRO, "models": PINNED_PICKER})
    status = profile.setup_status(command=command, env=env)
    assert status["available"] and status["logged_in"] and status["plan"] == "Claude Pro"
    assert status["login_command"] == command + ["auth", "login"]

    models = profile.discover_models(command=command, env=env)
    ids = [m["id"] for m in models]
    # Native picker rows are deduplicated to their Hermes route ids (opus and opus[1m] both -> opus 1M)
    assert ids == ["claude-sonnet-5[1m]", "claude-opus-5-5[1m]", "claude-haiku-4-5-20251001"]
    assert [m["label"] for m in models] == ["Sonnet 5 for long sessions", "Opus 5.5", "Haiku 4.5"]
    assert models[1]["note"] == "usage credits"
    assert models[2]["note"] == ""
    # Discovery goes through the admission relay with zero upstream requests
    assert all(m["upstream_requests"] == 0 for m in models)


def test_every_model_the_cli_advertises_is_selectable(profile, tmp_path):
    """The pinned table adds metadata (1M route, window, aliases); it never decides visibility. A model
    it does not know keeps the CLI's own id and label, gains no [1m], and is marked unpinned."""
    pinned = _discover(profile, tmp_path, PINNED_PICKER)
    models = _discover(profile, tmp_path, PINNED_PICKER + UNPINNED_PICKER)
    # Newcomers leave the pinned rows exactly as they were.
    assert models[:len(pinned)] == pinned
    fresh = {m["id"]: m for m in models[len(pinned):]}
    # Every advertised model is listed under an id the CLI announced: the plain-only id stays plain,
    # and a plain + [1m] pair collapses onto its [1m] form, as pinned 1M models do.
    assert sorted(fresh) == ["claude-fable-9[1m]", "claude-opus-9[1m]", "claude-sonnet-5-9"]
    assert {m["id"]: m["label"] for m in fresh.values()} == {
        "claude-opus-9[1m]": "Opus 9", "claude-sonnet-5-9": "Sonnet 5.9", "claude-fable-9[1m]": "Fable 9"}
    # Unpinned is visible, never at the cost of the CLI's own billing warning, which reads first.
    assert fresh["claude-opus-9[1m]"]["note"] == "usage credits · unpinned"
    assert fresh["claude-sonnet-5-9"]["note"] == fresh["claude-fable-9[1m]"]["note"] == "unpinned"
    assert all(m["upstream_requests"] == 0 for m in fresh.values())


def test_hermes_never_budgets_a_discovered_row_past_the_native_window(profile, tmp_path):
    """Behind the relay native Claude Code runs a plain id within its 200K default and a [1m] id
    within 1M. Left to itself Hermes sizes an id by family substring (claude-sonnet-5-9 -> 1M), so a
    plain unpinned row would let a conversation outgrow the window the native client enforces."""
    from agent.model_metadata import get_model_context_length
    models = _discover(profile, tmp_path, PINNED_PICKER + UNPINNED_PICKER)
    assert "claude-sonnet-5-9" in {m["id"] for m in models}
    for m in models:
        native = 1_000_000 if m["id"].endswith("[1m]") else 200_000
        assert get_model_context_length(m["id"], provider=profile.name, base_url=profile.base_url) <= native, m["id"]


def test_logged_out_or_missing_cli_degrades_to_pinned_catalog(profile, tmp_path):
    command, env = _cli(tmp_path, {"auth": {"loggedIn": False, "authMethod": "none"}, "account": {}, "models": []})
    status = profile.setup_status(command=command, env=env)
    assert status["available"] and not status["logged_in"]
    assert profile.discover_models(command=command, env=env) is None

    missing = profile.setup_status(command=[str(tmp_path / "nope")], env=env)
    assert not missing["available"] and not missing["logged_in"]
    assert "install" in missing["detail"].lower()
