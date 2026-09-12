# S11 — Demo bring-up script (`start_demo.sh`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S11 (salesperson-ui)

## Scope & verdict

Reviewed: `falkor-chat/scripts/start_demo.sh` (new, 272 lines, executable) and the one-row diff to
`falkor-chat/AGENTS.md`, both currently uncommitted, against the S11 row in
`docs/plans/salesperson-ui.md` (search `| **S11** |`) as the spec/done-condition. Checked
statically: read every line of the script, traced every exported env var against
`falkor-chat/server/falkorchat/config.py`'s actual `os.environ.get(...)` reads (all match by
name and default), confirmed `bash -n` syntax, confirmed the file is executable and genuinely
untracked (`git status --porcelain`), read `verify_catalog.sh`/`verify_salesperson.sh`/
`salesperson/build.sh`/`storefront.py`/`storefront_api.py` for the claims that touch them, and
read plan §4.3/§4.9/R6 for the two judgment calls the brief flagged. Not exercised live (no
FalkorDB/Node toolchain driven by this review) — devops's own live run is taken as reported,
not re-run.

**Verdict: approve with suggestions.** No blockers. The script is well-grounded in the plan and
the real codebase, every documented env var is real, and both judgment calls devops flagged hold
up under scrutiny — one of them (`--host 0.0.0.0`) turns out to be not just reasonable but the
*only* choice consistent with the plan's own explicit rejection of a loopback-binding design
(§4.3, R6). One minor ordering deviation from the plan's literal step sequence, and one nit on
the Cypher-parameter convention.

**CPG:** not applicable — bring-up shell script, no source dependency graph this task calls for
(the AGENTS.md docstring cross-checks were done by reading `config.py` directly, which is the
faster and more precise check for a handful of named env vars than a graph query).

## Findings

### Minor — preflight verification is interleaved per-artifact, not clustered as the plan's row literally sequences it

The plan row's chain is `seed_demo.sh → seed_catalog.sh → seed_salesperson.sh → preflight
verify_salesperson.sh + verify_catalog.sh → build the SPA` — read literally, all three seeds run
first, then both preflight checks run together as one step. The delivered script instead pairs
each artifact's seed with its own verify immediately (`start_demo.sh:193-215`): seed_catalog +
verify_catalog as step 5, seed_salesperson + verify_salesperson as step 6, plus an inline Agent
check right after seed_demo in step 4 (justified separately — no standing `verify_demo.sh`
exists, and the plan's done-condition names the Agent as a failure mode, so a check has to live
somewhere).

This doesn't violate the done-condition (still fails loudly, before uvicorn, with a specific
diagnosis per artifact) and arguably improves fail-fast locality — a broken catalog seed is
reported before spending time seeding salesperson defs. But it's a real deviation from the row's
written sequencing, and the plan's own review-gate convention treats a plan row as the spec an
implementer executes without re-deriving; a reviewer should not have to infer that "interleaved"
and "clustered" are equivalent here. Suggested fix: none required for correctness, but the plan
row (next time it's revised) or a one-line script comment should note that preflight checks run
per-artifact rather than clustered, so a future reader isn't left reconciling the two orderings
themselves.

### Nit — the Agent-existence check interpolates an env var into the `CYPHER` parameter preamble rather than binding it structurally

`start_demo.sh:179-180` builds `"CYPHER agentId='${FALKORCHAT_AGENT_ID}' MATCH (a:Agent
{agentId: \$agentId}) RETURN a.agentId"` — the query body itself correctly uses `$agentId` as a
bound parameter, but the *value* is spliced into the `CYPHER key=val` preamble via shell string
interpolation, which is textual, not a driver-level bind. Root `AGENTS.md` rule 1 is "always
parameterise Cypher, never interpolate variables into query strings." In practice this is inert
here — `FALKORCHAT_AGENT_ID` is operator-supplied dev/demo config (default `assistant`), not
participant input, and a stray single quote would just break the query loudly rather than open
an injection path. Still, it's the one place in the script that touches Cypher text directly, and
the pattern is worth flagging so it isn't copied into a script that *does* take untrusted input.
No suggested change to this script; worth a one-line comment if this pattern is reused elsewhere.

## What's solid

- **Env-var truth matches `config.py` exactly.** Every one of the ~14 exported vars
  (`FALKORCHAT_EMBEDDING_DIM`, `FALKORCHAT_STOREFRONT_DIR`, `FALKORCHAT_TRIGGER_DEF_KEY`/
  `_VERSION`, `FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH`, `FALKORCHAT_STOREFRONT_ENABLED`, etc.)
  was checked against `config.py`'s actual `os.environ.get`/`_env_flag` calls and matches by name
  and default. This is the part most likely to silently rot and it's right.
- **`--host 0.0.0.0` as `UVICORN_ARGS`'s default is correct, not merely defensible.** The plan
  explicitly rejected a loopback-binding variant (§4.3: "the loopback-binding variant this plan
  briefly carried") specifically because it's incompatible with the key-based presenter design
  and a TLS-terminating reverse proxy, and R6 records LAN exposure as an accepted Medium risk
  bounded by FR-1's controlled-demo scope. A demo meant to be joined by participants' own
  devices, with a documented mobile-viewport Playwright pass (S15) still ahead, cannot be
  loopback-only. `127.0.0.1` would have been the *wrong* default here.
- **The un-defaulted `FALKORCHAT_STOREFRONT_PRESENTER_KEY` is safe, traced end-to-end.**
  `config.STOREFRONT_PRESENTER_KEY` defaults to `""` when unset; `storefront.py`'s
  `presenter_login` checks `presenter_configured` *before* any key comparison specifically
  because `hmac.compare_digest("", "")` is `True`, and refuses with the same `403` either way.
  Nothing downstream assumes a key exists — an unset key means the presenter surface is
  permanently (safely) unauthenticatable, logged, never a crash.
- **The catalog-missing failure mode not being exercised live is a reasonable call, not a gap**
  — and the parallel the brief suggested (a throwaway workspace, the same technique used for
  "a def missing") does not actually transfer. `salesperson`/`order-fulfillment` defs and the
  demo Agent are workspace-scoped (materialized per `ws:<id>`), so an unseeded throwaway
  workspace genuinely reproduces "missing" for those. The product catalog is not: `seed_catalog.sh`
  writes only to the global `reference` graph (`<wsId>` is accepted but unused — FR-6, confirmed
  in both the script and `falkor-chat/AGENTS.md`), and `verify_catalog.sh` takes no workspace
  argument at all — it always reads `reference` directly. A throwaway workspace would see whatever
  `reference`'s catalog state already is; the only way to genuinely reproduce "catalog missing" is
  to corrupt the shared `reference` graph (destructive to every other workspace) or stand up a
  second, from-scratch FalkorDB instance (disproportionate for validating a 3-line `if ! verify_
  catalog.sh; then exit 1; fi` wrapper around a script whose exit contract is already documented
  and, per `falkor-chat/AGENTS.md`, predates S11). Declining to test it live and relying on
  `verify_catalog.sh`'s pre-existing contract was the right call.
- **Idempotency and defence-in-depth match the plan's own reasoning.** Every seed/verify call
  passes `FALKORCHAT_WS_ID` explicitly even where the pin already makes the default correct,
  exactly as §4.9 move 2 asks for ("defence in depth, not a load-bearing requirement").
- **The startup banner satisfies the done-condition's explicit ask** — it prints `Workspace:
  ws:${FALKORCHAT_WS_ID} (pinned — never config.py's 'acme' default)`, which is the "assert the
  resolved workspace in the startup banner" clause verbatim.
- **`build.sh`'s Node-detection claim holds** — read `salesperson/build.sh`: it does fail loudly
  and specifically (`die_no_node`) with an actionable remediation, so `start_demo.sh` not
  duplicating that check (comment at `start_demo.sh:229-231`) is correct, not a shortcut.
- **The AGENTS.md row fits the file's own conventions** — cites the plan (`docs/plans/
  salesperson-ui.md` S11) rather than restating rationale, stays under the ~700-char smell
  threshold (unlike a pre-existing neighbor row, `verify_salesperson.sh` at 980 chars — not
  introduced by this change), and the table format matches its siblings.

## Open questions

None — both judgment calls the brief raised resolved cleanly against the plan's own text rather
than needing a stakeholder call.
