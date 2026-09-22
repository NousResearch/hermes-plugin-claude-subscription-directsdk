"""Catalog capacity must agree with the actual native --model selection under each routing policy."""
import json
import os
import sys

from test_directsdk import FAKE

# One picker row per explicit native route, like the CLI's own picker: the base row is the included
# 200K window, the [1m] row the metered 1M one.
EXPECTED = {
    'claude-sonnet-5': 200_000,
    'claude-sonnet-5[1m]': 1_000_000,
    'claude-haiku-4-5-20251001': 200_000,
    'claude-opus-5': 200_000,
    'claude-opus-5[1m]': 1_000_000,
    'claude-opus-4-8': 200_000,
    'claude-opus-4-8[1m]': 1_000_000,
    'claude-fable-5-1': 200_000,
    'claude-fable-5-1[1m]': 1_000_000,
}


def test_catalog_windows_match_explicit_native_routes(profile, monkeypatch):
    from agent.model_metadata import get_model_context_length
    monkeypatch.delenv('CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING', raising=False)
    assert tuple(profile.fallback_models) == tuple(EXPECTED)
    assert profile.default_aux_model == 'claude-sonnet-5'
    assert profile.model_aliases == {'sonnet': 'claude-sonnet-5', 'haiku': 'claude-haiku-4-5-20251001',
                                     'claude-haiku-4-5': 'claude-haiku-4-5-20251001', 'opus': 'claude-opus-5', 'fable': 'claude-fable-5-1'}
    for model, window in EXPECTED.items():
        assert profile.get_model_context_length(model) == window
        assert get_model_context_length(model, provider=profile.name) == window
        assert get_model_context_length(model, provider=profile.name, config_context_length=200000) == 200000
    assert profile.get_model_context_length('sonnet') == 200_000
    assert profile.get_model_context_length('unqualified-future-model') is None
    # The budget follows the policy: always-1m raises ordinary ids to 1M, always-200k pins even [1m] ids.
    monkeypatch.setenv('CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING', 'always-1m')
    assert profile.get_model_context_length('sonnet') == 1_000_000
    assert profile.get_model_context_length('haiku') == 200_000
    monkeypatch.setenv('CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING', 'always-200k')
    assert profile.get_model_context_length('claude-opus-5[1m]') == 200_000
    monkeypatch.setenv('CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING', 'bogus')
    assert profile.get_model_context_length('sonnet') is None


def test_native_argv_never_adds_1m_silently(profile, tmp_path):
    """Default policy: every picker row selects exactly itself; aliases and ordinary ids stay on the
    included 200K route, only an id that carries [1m] selects the metered 1M route."""
    capture = tmp_path / 'argv.json'
    native = tmp_path / 'native.py'
    native.write_text(FAKE.replace('rows=[]', "pathlib.Path(os.environ['ARGV_CAPTURE']).write_text(json.dumps(sys.argv))\nrows=[]"))
    aliases = {'sonnet':'claude-sonnet-5', 'opus':'claude-opus-5', 'haiku':'claude-haiku-4-5-20251001',
               'fable':'claude-fable-5-1', 'claude-haiku-4-5':'claude-haiku-4-5-20251001',
               'unqualified-future-model':'unqualified-future-model'}
    with_client = profile.create_client(command=[sys.executable,str(native)], env={'PATH':os.defpath,'HOME':str(tmp_path),'ARGV_CAPTURE':str(capture)})
    try:
        for requested, expected in {**{m:m for m in EXPECTED}, **aliases}.items():
            result = with_client.create(model=requested, messages=[{'role':'user','content':'fixture'}],
                                        tools=[{'type':'function','function':{'name':'probe','description':'TAIL','parameters':{'type':'object','properties':{'value':{'type':'string'}}}}}])
            argv = json.loads(capture.read_text())
            assert argv[argv.index('--model')+1] == expected, requested
            assert result.usage.model_dump()['native_admission']['routes'] == [expected]
    finally:
        with_client.close()
