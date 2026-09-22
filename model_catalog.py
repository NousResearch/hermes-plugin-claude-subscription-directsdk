"""Pinned native routes and the explicit context-routing policy.

Claude Code applies its gateway defaults whenever ``ANTHROPIC_BASE_URL`` is set, and the request path
always sets it (the local admission relay). Behind a gateway a 1M-capable model runs its included
200K window unless ``[1m]`` is selected on the command line. That ``[1m]`` route is metered
differently from included usage on most plans (the native picker marks Opus 1M and Fable as drawing
from usage credits), so the plugin never appends the suffix silently. The policy below decides the
route; a genuine context-window failure is the only automatic escalation, and only under ``auto``.

Policies (``CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING`` or ``Client(context_routing=...)``):

``auto`` (default)
    Ordinary ids and short aliases run on the included 200K route. An id that already carries
    ``[1m]`` runs on the 1M route from the first request. When the 200K route rejects a request as
    over its context window before anything has been delivered, the request is retried once on the
    ``[1m]`` route, so a long request completes instead of failing or truncating.
``always-1m``
    Every 1M-capable model runs on its ``[1m]`` route; Haiku stays 200K.
``always-200k``
    Never select a ``[1m]`` route, even for an id that carries the suffix. A request that does not
    fit fails as the API reports it.
"""
import os

INCLUDED_WINDOW = 200_000
LONG_WINDOW = 1_000_000

# Native capability ceiling per model; the route actually selected may be smaller.
CONTEXT_WINDOWS = {
    'claude-sonnet-5': LONG_WINDOW,
    'claude-haiku-4-5-20251001': INCLUDED_WINDOW,
    'claude-opus-5': LONG_WINDOW,
    'claude-opus-4-8': LONG_WINDOW,
    'claude-fable-5-1': LONG_WINDOW,
}
ALIASES = {
    'sonnet': 'claude-sonnet-5',
    'haiku': 'claude-haiku-4-5-20251001',
    'claude-haiku-4-5': 'claude-haiku-4-5-20251001',
    'opus': 'claude-opus-5',
    'fable': 'claude-fable-5-1',
}

ROUTING_ENV = 'CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING'
AUTO, ALWAYS_1M, ALWAYS_200K = 'auto', 'always-1m', 'always-200k'
POLICIES = (AUTO, ALWAYS_1M, ALWAYS_200K)
_POLICY_SPELLINGS = {AUTO: AUTO, ALWAYS_1M: ALWAYS_1M, '1m': ALWAYS_1M, ALWAYS_200K: ALWAYS_200K, '200k': ALWAYS_200K}


def context_routing_policy(value=None, env=None):
    """Resolve the policy: an explicit ``value`` wins, then ``$CLAUDE_SUBSCRIPTION_DIRECTSDK_CONTEXT_ROUTING``,
    then ``auto``. Unknown spellings raise rather than silently picking a billing route."""
    if value is None:
        value = (env if env is not None else os.environ).get(ROUTING_ENV)
    policy = _POLICY_SPELLINGS.get(str(value or AUTO).strip().lower())
    if policy is None:
        raise ValueError(f'{ROUTING_ENV} must be one of {", ".join(POLICIES)}, not {value!r}')
    return policy


def canonical_model(model):
    """Version-pinned model id behind an alias or a ``[1m]`` route; unknown ids pass through."""
    base = str(model).removesuffix('[1m]')
    return ALIASES.get(base, base)


def route_plan(model, policy=AUTO):
    """Native ``--model`` selections to try, in order, for ``model`` under ``policy``.

    Unknown ids pass through untouched (no invented window, no suffix). A 200K-only model never
    gets ``[1m]``; asking for it explicitly is an error rather than a silent downgrade.
    """
    policy = context_routing_policy(policy)
    model = str(model)
    explicit = model.endswith('[1m]')
    canonical = canonical_model(model)
    window = CONTEXT_WINDOWS.get(canonical)
    if window is None:
        return (model,)
    if window < LONG_WINDOW:
        if explicit:
            raise ValueError(f'{canonical} does not support a 1M context window')
        return (canonical,)
    long_route = canonical + '[1m]'
    if policy == ALWAYS_200K:
        return (canonical,)
    if policy == ALWAYS_1M or explicit:
        return (long_route,)
    return (canonical, long_route)


def native_route(model, policy=AUTO):
    """The first native selection for ``model``; the one every request starts on."""
    return route_plan(model, policy)[0]


def picker_route(model):
    """Canonicalize one native picker row without applying a request-routing policy.

    Discovery reports the account's offers, not this process's preferred request route.  In
    particular, an ``always-200k`` preference must not make an offered 1M row disappear from
    the picker or merge it with the included row.
    """
    model = str(model)
    explicit = model.endswith('[1m]')
    canonical = canonical_model(model)
    window = CONTEXT_WINDOWS.get(canonical)
    if window is None:
        return model
    if explicit:
        if window < LONG_WINDOW:
            raise ValueError(f'{canonical} does not support a 1M context window')
        return canonical + '[1m]'
    return canonical


def context_window(model, policy=AUTO):
    """The window Hermes should budget for ``model`` under ``policy``, or ``None`` when unknown.

    Under ``auto`` an ordinary id budgets the included 200K window: Hermes compacts to fit it, and
    the ``[1m]`` retry is a safety net for the request that overflows anyway, not a standing budget.
    """
    canonical = canonical_model(model)
    if canonical not in CONTEXT_WINDOWS:
        return None
    try:
        route = native_route(model, policy)
    except ValueError:
        return None
    return LONG_WINDOW if route.endswith('[1m]') else min(INCLUDED_WINDOW, CONTEXT_WINDOWS[canonical])


# The pinned picker: one row per explicit native route, base and [1m] alike, like the CLI's own picker.
MODEL_METADATA = {}
for _model, _window in CONTEXT_WINDOWS.items():
    MODEL_METADATA[_model] = {'canonical_model': _model, 'context_window': min(_window, INCLUDED_WINDOW)}
    if _window >= LONG_WINDOW:
        MODEL_METADATA[_model + '[1m]'] = {'canonical_model': _model, 'context_window': _window}
del _model, _window
