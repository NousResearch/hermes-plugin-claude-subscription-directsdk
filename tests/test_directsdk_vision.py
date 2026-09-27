"""Per-model vision declarations and the transport that has to back them."""
import directsdk as native
from model_catalog import ALIASES, CONTEXT_WINDOWS, native_model

PNG = 'data:image/png;base64,iVBORw0KGgo='


def test_every_catalog_id_declares_vision(profile):
    from agent.image_routing import decide_image_input_mode
    from agent.models_dev import get_model_capabilities

    # Every id a session may carry: native route, canonical id, alias and 1M alias.
    long_aliases = [a + '[1m]' for a in ALIASES if native_model(a).endswith('[1m]')]
    for model in (*profile.fallback_models, *CONTEXT_WINDOWS, *ALIASES, *long_aliases):
        caps = get_model_capabilities(profile.name, model)
        assert caps.supports_vision is True
        # Same window the profile reports, not core's 200K unknown-model default.
        assert caps.context_window == profile.get_model_context_length(model)
        assert decide_image_input_mode(profile.name, model, {}) == 'native'


def test_vision_is_not_declared_provider_wide(profile):
    # Profile-wide supports_vision would route computer_use screenshots of unpinned ids natively.
    from agent.image_routing import decide_image_input_mode

    assert not profile.supports_vision
    assert decide_image_input_mode(profile.name, 'unpinned-future-model', {}) == 'text'


def test_a_config_override_still_turns_it_off(profile):
    from agent.image_routing import decide_image_input_mode

    cfg = {'model': {'supports_vision': False}}
    assert decide_image_input_mode(profile.name, profile.default_aux_model, cfg) == 'text'


def _tool_history(url):
    return [
        {'role': 'user', 'content': 'look'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'look', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': [
            {'type': 'text', 'text': 'image:'}, {'type': 'image_url', 'image_url': {'url': url}}]},
    ]


def test_routing_consumers_follow_the_declaration(profile):
    from tools.computer_use.vision_routing import should_route_capture_to_aux_vision

    # computer_use screenshots stay on the main model only for declared ids.
    assert should_route_capture_to_aux_vision(profile.name, 'claude-opus-5-5[1m]', {}) is False
    assert should_route_capture_to_aux_vision(profile.name, 'unpinned-future-model', {}) is True


def test_tool_result_images_reach_native_as_image_blocks():
    _, frames = native.prepare_history(_tool_history(PNG))
    result = frames[-1]['message']['content'][0]
    assert result['type'] == 'tool_result' and result['tool_use_id'] == 'c1'
    assert result['content'][1] == {
        'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'iVBORw0KGgo='}}


def test_remote_image_url_becomes_a_hint_instead_of_failing_the_turn():
    # Native routing passes remote URLs through (delegate_task images, one-shot task bodies).
    user = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'look'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}}]}]
    _, frames = native.prepare_history(user)
    hint = frames[-1]['message']['content'][1]
    assert hint['type'] == 'text'
    assert 'https://example.com/a.png' in hint['text'] and 'vision_analyze' in hint['text']


def test_remote_image_url_in_a_tool_result_becomes_a_hint():
    _, frames = native.prepare_history(_tool_history('HTTPS://example.com/b.png'))
    hint = frames[-1]['message']['content'][0]['content'][1]
    assert hint['type'] == 'text' and 'HTTPS://example.com/b.png' in hint['text']
