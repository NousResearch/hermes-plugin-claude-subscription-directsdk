"""Pinned native routes; a loopback gateway needs explicit long-context selection."""
CONTEXT_WINDOWS = {
    'claude-sonnet-5-5': 1_000_000,
    'claude-sonnet-5': 1_000_000,
    'claude-haiku-4-5-20251001': 200_000,
    'claude-opus-5-5': 1_000_000,
    'claude-opus-5': 1_000_000,
    'claude-opus-4-8': 1_000_000,
    'claude-fable-5-1': 1_000_000,
}
# Families that 400 on ``thinking: {"type": "disabled"}`` (the same contract Hermes core keeps
# in agent/anthropic_adapter.py). A caller's disable is omitted for them, which beats a dead
# request. That does not turn thinking down: a reasoning-off request on these routes carries no
# `thinking` and no effort (neither `output_config.effort` nor `--effort`), so the model thinks
# at native's default effort. Opus 5.5 joined Fable: the docs say thinking cannot be turned off
# there and the API answers the disable with the same 400 (#22). Plain Opus 5 still accepts it,
# so the entry is the full `claude-opus-5-5` prefix. Sonnet 5.5 rejects the disable too (its lowest
# setting is `between_tools`); plain Sonnet 5 still accepts it.
MANDATORY_THINKING = ('claude-fable', 'claude-opus-5-5', 'claude-sonnet-5-5')
ALIASES = {
    'sonnet': 'claude-sonnet-5-5',
    'haiku': 'claude-haiku-4-5-20251001',
    'claude-haiku-4-5': 'claude-haiku-4-5-20251001',
    'opus': 'claude-opus-5-5',
    'fable': 'claude-fable-5-1',
}
# Haiku 4.5 rejects `thinking: {'type': 'adaptive'}` with a 400 upstream; its effort signal
# still applies. Unknown routes keep adaptive so future models are not silently downgraded.
NO_ADAPTIVE_THINKING = frozenset({'claude-haiku-4-5-20251001'})


def native_model(model):
    base = model.removesuffix('[1m]')
    canonical = ALIASES.get(base, base)
    window = CONTEXT_WINDOWS.get(canonical)
    if window == 1_000_000:
        return canonical + '[1m]'
    if window == 200_000:
        if model.endswith('[1m]'):
            raise ValueError('Haiku 4.5 does not support a 1M context window')
        return canonical
    return model


def accepts_thinking_disable(model):
    # Same tolerance as supports_adaptive_thinking: an absent model still reaches `model is required`.
    base = model.removesuffix('[1m]') if isinstance(model, str) else ''
    return not ALIASES.get(base, base).startswith(MANDATORY_THINKING)


def supports_adaptive_thinking(model):
    # Body assembly precedes model validation, so an absent or invalid route answers True
    # and still reaches the existing `model is required` error.
    base = model.removesuffix('[1m]') if isinstance(model, str) else ''
    return ALIASES.get(base, base) not in NO_ADAPTIVE_THINKING


MODEL_METADATA = {
    native_model(model): {'canonical_model': model, 'context_window': window}
    for model, window in CONTEXT_WINDOWS.items()
}


# Core looks capabilities up by exact id, so key the canonical id, the native `[1m]` route and
# every alias (each targets a pinned id). Unpinned ids keep the catalog path.
VISION_CAPABILITIES = {
    model_id: {'supports_vision': True} for model_id in (*CONTEXT_WINDOWS, *MODEL_METADATA, *ALIASES)
}
