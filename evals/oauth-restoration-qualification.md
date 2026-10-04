# Subscription OAuth restoration qualification

## Scope and tested sources

Observed on Linux, October 4, 2026, using official Claude Code 2.1.286 and requested model `claude-sonnet-5`, with an existing `claude.ai` / `firstParty` subscription login. These are real vendor responses, not the dummy-key loopback fixtures. No API-key fallback was used.

- Baseline: `ef73726cfaf2fa0ee041e55572f406e2c24fed83`.
- Candidate: `916f6fea9374cae31923ff56d43bd97ac1204402`.
- Native binary SHA-256: `fe503f65c6289d59c23e5b21ae44f03583f997dd33a2cbfc75ab4f96fb8fc73f`.

Both arms used immutable Git-archive snapshots, the same tool inventory and a stable synthetic workload directory, including across client recreation. The local qualification runner used the production `Client(env=None)` auth guards and canonical vendor validation. Request limits were enforced before constructing single-use admissions, not only counted after completion.

## Baseline versus candidate

Each arm completed six requests: four rounds of two inert parallel tools, a final continuation after client recreation, and an ordinary-user control. Result representations rotated through strings, one text block and split text blocks; omitted and explicit-false `is_error` were exercised.

| Observation | Baseline | Candidate |
|---|---|---|
| Completed requests / HTTP 200 | 6 / 6 | 6 / 6 |
| Exact queried host tool-result representations | No | Yes |
| Previous-breakpoint prefix comparisons | `[false, true, true, true, true]` | `[false, true, true, true, true]` |
| Marker settings and static directive locations unchanged | Yes | Yes |
| Client recreation exercised | Yes | Yes |
| Nonempty signed history observed | No | No |
| Full-chain acceptance | False | False |

The first comparison is an inherited ordinary-user/native-context projection difference present on both versions, not a demonstrated candidate regression. The remaining four comparisons preserve exact previous-breakpoint prefixes, excluding only cache directives checked separately. Real `1h` TTL settings remained intact. Empty signed-history equality is not counted as coverage.

## Real signed-thinking follow-up

A separate candidate-only probe completed three bounded OAuth requests at adaptive/high effort, with a 4,000-output-token ceiling per request. It exercised two rounds of parallel tools and client recreation before the final continuation.

- All three admissions returned HTTP 200; zero blocked requests.
- Historical signed-block counts at request time: `[0, 1, 1]`.
- Nonempty real signed history was preserved exactly, including across recreation; the service accepted both continuation requests.
- Queried host tool-result restoration remained exact.
- Previous-breakpoint comparisons were `[false, true]`: the inherited initial difference remained; the later signed-history/recreated-client comparison was identical.
- Marker settings and static directives stayed identical, including `1h` TTLs.
- Scoped `signed_tool_replay_passed` is true. Full `acceptance_passed` remains false; the scoped result does not waive that mismatch.

One earlier attempt before allowance reset stopped on its first HTTP 429 with no model response. That unsuccessful attempt is not qualification evidence. There was no automatic retry loop; the successful follow-up was a separately authorized bounded run after reset.

## Evidence and limits

The three accompanying receipts are byte-for-byte copies of the sanitized successful-run receipts:

- `receipts/claude-2.1.286-oauth-baseline.json`
- `receipts/claude-2.1.286-oauth-candidate.json`
- `receipts/claude-2.1.286-oauth-signed-candidate.json`

They retain structural facts, native counters and hashes, not raw prompts/replies, actual signatures, headers, credentials or account identifiers. The OAuth runner is currently local rather than included in this contribution; these receipts document observations, not a repository-contained reproduction command. The existing loopback qualification remains reproducible from this repository without model allowance.

Local checks: 33 offline OAuth/signed-probe harness tests and 100 candidate plugin tests passed. These Linux checks do not replace upstream cross-platform CI, which still requires maintainer workflow approval.

This qualifies the scoped tool-result restoration and actual signed-history continuation on this exact candidate and CLI binary. It does not claim globally identical OAuth context, resolution of ordinary-user/account-context behavior in #33, compatibility with every CLI version/platform/model, statistically established cache performance or subscription allowance savings. The initial mismatch remains visible for upstream assessment; it is outside the tool-result-only restoration patch.
