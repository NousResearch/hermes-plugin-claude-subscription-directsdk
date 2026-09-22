"""Pinned native routes: the plain id is the plan-included 200K route, `[1m]` is explicit.

Claude Code applies its gateway defaults whenever ``ANTHROPIC_BASE_URL`` is set, and the request path
always sets it (the admission relay): a 1M-capable model runs its included 200K window unless ``[1m]``
is selected. The account picker offers ``opus`` and ``opus[1m]`` as separate rows because the ``[1m]``
route is metered as usage credits on most plans, so the suffix is never appended silently; the
advertised window follows the route actually selected, so Hermes compacts to the window it is billed for.
"""
INCLUDED_WINDOW = 200_000

# Native capability ceiling per model; the route selected may be the included window below it.
CONTEXT_WINDOWS = {
    'claude-sonnet-5': 1_000_000,
    'claude-haiku-4-5-20251001': 200_000,
    'claude-opus-5': 1_000_000,
    'claude-opus-4-8': 1_000_000,
    'claude-fable-5-1': 1_000_000,
}
ALIASES = {
    'sonnet': 'claude-sonnet-5',
    'haiku': 'claude-haiku-4-5-20251001',
    'claude-haiku-4-5': 'claude-haiku-4-5-20251001',
    'opus': 'claude-opus-5',
    'fable': 'claude-fable-5-1',
}


def native_model(model):
    base = model.removesuffix('[1m]')
    canonical = ALIASES.get(base, base)
    window = CONTEXT_WINDOWS.get(canonical)
    if window is None:
        return model
    if not model.endswith('[1m]'):
        return canonical
    if window <= INCLUDED_WINDOW:
        raise ValueError(f'{canonical} does not support a 1M context window')
    return canonical + '[1m]'


# One row per explicit native route, base and [1m] alike, like the CLI's own picker.
MODEL_METADATA = {}
for _model, _window in CONTEXT_WINDOWS.items():
    MODEL_METADATA[_model] = {'canonical_model': _model, 'context_window': min(_window, INCLUDED_WINDOW)}
    if _window > INCLUDED_WINDOW:
        MODEL_METADATA[_model + '[1m]'] = {'canonical_model': _model, 'context_window': _window}
del _model, _window
