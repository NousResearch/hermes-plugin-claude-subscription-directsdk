# hermes-plugin-claude-subscription-directsdk — agent rules

Standalone Hermes model-provider plugin: Hermes talks to a native Claude Code child over a
loopback relay, on the user's Claude subscription. `model_catalog.py` is the pinned route table,
`__init__.py` is the provider profile Hermes core queries, `directsdk.py` is the transport,
`admission.py` is the single-request gate. Tests import Hermes core from a checkout
(`HERMES_AGENT_REPO`, default `~/.hermes/hermes-agent`).

## Invariant: Hermes core never guesses a window for a route we serve

Native Claude Code runs a plain model id inside its 200K gateway default and selects the 1M
window only for an explicit `[1m]` route. Hermes core, when a provider profile returns no
window, falls back to a family-substring guess (`claude-opus-5` matched `claude-opus-5-5` → 1M).
Those two disagreeing is the failure class this file exists for: Hermes packs a 1M
conversation, the CLI rejects it at 200K, and the user reads a "server rejected this request
as too large, but this conversation is only 128K" error that blames the server.

- `Profile.get_model_context_length` answers for every plain id: the pinned window, else
  `200_000`. Returning `None` for a plain id is a regression, whatever the reason.
  `tests/test_directsdk_models.py::test_catalog_windows_match_explicit_native_routes` guards it;
  do not weaken the unpinned assertions.
- `native_model()` appends `[1m]` only for ids pinned at 1M. An unpinned id is sent bare and
  reported as 200K. A user typing a brand-new id gets a working 200K session, never a broken
  1M one.
- Discovery lists every model the CLI advertises; the pinned table adds metadata only. Never
  filter the picker down to the table.

## Adding a Claude model (one PR, same day it ships)

Anthropic ships a model → this PR lands the same day, and the catalog pin bump follows within
the hour (below). Do not wait for hermes-agent's own static Anthropic list; it lagged Opus 5.5
too. Vendor truth is https://code.claude.com/docs/en/model-config ("Extended context" for the
window and which plans include it; the model table for the minimum CLI version).

1. `model_catalog.py::CONTEXT_WINDOWS` — canonical id (hyphenated, as the CLI spells it) and
   window from the docs. Do not copy a window from a sibling model.
2. `ALIASES` — move the family alias (`opus`, `sonnet`, `fable`, `haiku`) when the docs move it.
3. `MANDATORY_THINKING` / `NO_ADAPTIVE_THINKING` if the docs say thinking cannot be turned off
   or adaptive thinking 400s (Fable, Opus 5.5 and Haiku 4.5 today). Mirror what
   `agent/anthropic_adapter.py` in hermes-agent encodes; if the two disagree, fix both. Probe a
   new family before adding it: Opus 5.5 was confirmed by a live 400
   (`"thinking.type.disabled" is not supported for this model`) on `claude-opus-5-5[1m]`.
4. `tests/test_directsdk_models.py::EXPECTED` — one row per route, window included. The test
   drives the fake CLI and checks the argv Hermes actually sends.
5. `README.md` → Requirements: the minimum Claude Code version the model needs.
6. Bump `plugin.yaml` `version` when a route changes; a pin bump alone does not surface in
   `hermes plugins list`.

## Pin bump: a fix on main reaches nobody until the catalog moves

`hermes plugins update claude-subscription-directsdk` follows the pin in
`NousResearch/hermes-agent:plugin-catalog/claude-subscription-directsdk.yaml`, not this repo's
main. Only `hermes plugins install NousResearch/hermes-plugin-claude-subscription-directsdk`
tracks main. So after every merge here that changes routing, windows, thinking policy or
admission:

1. Open the hermes-agent PR moving the pin to the new main SHA (the card `image:` URL follows
   the pin); `scripts/validate_plugin_catalog.py plugin-catalog/claude-subscription-directsdk.yaml`
   must print OK.
2. Once merged, confirm the live catalog serves it:
   `curl -sL https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json` → the
   `claude-subscription-directsdk` entry's `sha`.
3. Then tell reporters to `hermes plugins update`; before that, the update does nothing and the
   correct instruction is the `install …` form above.

A user report of a window mismatch is first a pin question: ask which SHA
(`hermes plugins list` shows the version; `git -C ~/.hermes/plugins/<name> rev-parse HEAD` the
commit) before debugging the table.

## Verification

```
HERMES_AGENT_REPO=~/.hermes/hermes-agent python -m pytest -q
```

CI runs the suite on ubuntu, macOS and Windows. Gate a merge on
`gh pr checks <n> --repo NousResearch/hermes-plugin-claude-subscription-directsdk` exiting 0
(8 = still pending), never on a `statusCheckRollup` grep: lanes that have not started have no
row, so "no pending rows" reads ubuntu-passed as all-green while macOS and Windows are queued.
After the train, `gh run watch <id> --exit-status` on main.

## Contributor PRs

Teknium is the only merge arbiter. Verify a PR's premise against the vendor docs and a real
red-on-main / green-on-head run before anything else; salvage (cherry-pick + fixup, or a slim
redo with `Co-authored-by`) rather than iterate on the contributor's branch; never merge a PR
that re-opens a ruled question (`[1m]` routing and the 200K-vs-1M billing question are settled
by #8/#11 and the docs) without new evidence. The `ANTHROPIC_BASE_URL` guard is fail-closed and
plugin-owned; a knob that forwards the subscription bearer to a third-party host is a policy
change, not a feature.
