# S10 — Presenter surface, the move (`storefront.py` / `storefront_api.py`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S10 (salesperson-ui)

## Scope & verdict

Reviewed, all currently uncommitted on disk: `falkor-chat/server/falkorchat/storefront.py` (the
three presenter operations moved in from `storefront_api.py`, plus S10's own new content —
`presenter_login`'s fixed delay/observational counter, `reset_all`'s stop-intake flag),
`falkor-chat/server/falkorchat/storefront_api.py` (routes reduced to thin bodies, the two
`document-ingestion2` `ServiceError` subclasses declared unreachable), `falkorchat/config.py`'s new
`STOREFRONT_PRESENTER_LOGIN_DELAY_S`, and the test/doc additions this forced
(`tests/test_storefront_api.py`, `tests/test_storefront.py`, `docs/SERVER.md`) — against the S10
row in `docs/plans/salesperson-ui.md` (search `| **S10** |`) and, for the citations S10's own
docstrings make, `docs/reviews/salesperson-ui-impl.md` (`## Pass 10`/`## Pass 11`, P10-9) and
`docs/reviews/salesperson-ui.md` (`## Pass 7`, P7-5 — a different document, see Finding 3 below).

Taken as already independently verified per the brief and not re-derived: full-suite green
(2737 passed / 14 deselected / 0 failed), the `_STEP_10_INTERIM` → tombstone move, the forced
file-scope additions' cause (`test_config_reads_exactly_the_documented_storefront_env_vars`), and
the two `document-ingestion2` `ServiceError` subclasses being real and correctly unreachable here.
I re-confirmed the last of these myself only incidentally while reading the file
(`storefront_api.py:89-91,705-716`) — consistent with the brief.

What I additionally checked, beyond re-confirming the above: the full `reset_all` method body and
its docstring's citations against the actual review history; `presenter_login`'s call path
end-to-end including the FastAPI request model that runs before it; the wire-level handling of
`ResetAllStateUnknownError`'s `participants` field through `StorefrontHTTPError`/
`_handle_storefront_http_error`; and one self-devised mutation, applied to a byte-backed copy of
`storefront.py` and run against `tests/test_storefront_api.py` + `tests/test_storefront.py`,
confirmed reverted to the original file's md5 afterward (`f64578b0…`, unchanged before/after).

**Verdict: approve with suggestions.** No blockers. The stop-intake/drain/delete ordering is sound
and its P10-9 citation checks out against the review history; the constant-time login design holds
under the surrounding FastAPI path; the F8 absent-vs-null wire contract is implemented and tested
correctly. One major finding (an untested narrowing on the F8 re-read's evidence block — a mutation
I devised and ran survives the full suite) and one minor (a docstring citation spans two different
review documents without naming either, and is one word away from misleading).

**CPG:** considered, not relevant — `cpg_falkorchat` exists (339,973 nodes) but its `CpgBuildInfo`
is stamped `BUILT_AT: 2026-09-07T22:25:45Z`, hand-backfilled by `graph-dba`, and predates S10:
queried it for `METHOD` nodes named `reset_all`/`presenter_login`/anything containing `presenter`
and got zero rows in all three cases, confirming the loaded graph does not reflect S10's code (or
even S8's presenter routes) rather than assuming staleness. Direct reads of the two files plus
mutation testing substituted.

## Findings

### Major — the F8 re-read's four-key narrowing on `reset_all`'s evidence block has no test, and a mutation that deletes it survives the full suite

`Storefront.reset_all`'s `except redis_exceptions.TimeoutError` arm (`storefront.py:1919-1928`)
builds `unresolved` by narrowing each `list_participants()` row to exactly
`{participantId, displayName, language, joinedAt}` — the same four keys §5.2 and this row's own
done-condition require everywhere the roster reaches the wire, and the same four the sibling route
`GET /presenter/participants` narrows to. That sibling narrowing has a dedicated contract test,
`test_the_roster_carries_exactly_the_four_keys_and_no_non_participants`
(`tests/test_storefront_api.py:1850`, `assert set(row) == {"participantId", "displayName",
"language", "joinedAt"}`). The F8 re-read path has no equivalent: its only test,
`test_a_reset_all_that_times_out_is_504_unknown_and_is_never_retried`
(`tests/test_storefront_api.py:2356`), asserts only `[row["displayName"] for row in
body["participants"]] == ["Ada"]` — it never inspects the row's key set.

**Verified, not guessed:** I replaced the narrowing with `unresolved = list(self.list_participants())`
(passing the raw six-key rows — including `channelId`/`threadId`, internal server-side ids §5.2
explicitly keeps off the wire — straight through) on a byte-backed copy of `storefront.py`, ran
`tests/test_storefront_api.py tests/test_storefront.py`: **251 passed**, identical count to the
unmutated file. Restored from the backup and confirmed the file's md5 (`f64578b0…`) matches the
pre-mutation original.

This is a coverage gap, not a live bug — the shipped code narrows correctly today — but it is
exactly the untested seam a future edit to this arm (which already has three code paths: success,
sweep-timeout-then-successful-reread, sweep-timeout-then-timed-out-reread) would silently regress,
leaking two internal ids on a `504` nobody currently checks for that. **Fix:** add one assertion to
`test_a_reset_all_that_times_out_is_504_unknown_and_is_never_retried` — `assert set(body
["participants"][0]) == {"participantId", "displayName", "language", "joinedAt"}` — mirroring the
roster test at line 1865. It costs one line and closes exactly the case I demonstrated.

### Minor — the docstrings' "(Pass 7, P7-5)" citation spans two different review documents, unnamed, and sits one word away from being read as the wrong one

`storefront.py:242,627,689,716` and `config.py:234` all cite `(Pass 7, P7-5)` for the
no-lockout decision. `P7-5` exists only in `docs/reviews/salesperson-ui.md` (the **plan** review —
"S10's rate-limiter attempt counter has no specified effect and no response code", later "fixed by
decision"). But `storefront.py`'s *other* inline citations in the same method
(`P10-9`, `## Pass 10`, `## Pass 11`) all point at `docs/reviews/salesperson-ui-impl.md` (the
**implementation** review) — which has its *own*, differently-numbered `## Pass 7` (S7: state,
reset, catalog, images, findings named `S7-1`..`S7-4`, no `P7-5` in it). I checked both files
directly (`grep -n 'P7-5'` across the repo) to confirm there is no actual string collision today,
but a reader who follows S10's convention of citing `docs/reviews/salesperson-ui-impl.md` for every
other pass number in this same file, and greps *that* file's own `## Pass 7` for `P7-5`, finds a
different S7 review with no such finding — a dead end that costs a detour, not a wrong answer.
**Fix:** name the document at the P7-5 citation the same way the plan row and the P10-9 citations
already do — `` `docs/reviews/salesperson-ui.md` `## Pass 7`, P7-5 ``.

## What's solid

**The stop-intake/drain/delete ordering has no window.** `_intake_stopped = True`
(`storefront.py:1903`) executes, completes, and is followed synchronously by the pre-drain roster
read — there is no yield point between them in the request thread's control flow, so no new
`reserve_turn` can observe a stale `False` after that line runs. Every participant a reservation
could concern is by construction still enumerable in `list_participants()`'s later, repeatedly-
polled `turn_in_flight` check (`storefront.py:1907`), because reservation only ever mutates
`_turns`, never the participant registry the roster reads — so a turn that slipped in *before*
`_intake_stopped` flipped is still caught by the drain loop, not raced past it. The **P10-9**
citation for "the pre-drain roster read sits outside every `try` below" checks out against the
actual review history once the two try-blocks are told apart: it means outside the *inner*
try/except that guards the sweep call (`storefront.py:1915`), not the outer one (whose only clause
is the `finally` restoring `_intake_stopped`) it is textually inside — confirmed against
`docs/reviews/salesperson-ui-impl.md`'s Pass 12 mutation ledger, where "moving the roster read +
drain loop inside the `try`" (i.e., into the inner one) is exactly the mutation
`test_a_reset_all_whose_pre_drain_roster_read_times_out_never_enters_the_sweep` was written to kill.

**The constant-time login design holds under the surrounding path.** The fixed `time.sleep` runs
before both checks (`storefront.py:727-738`), so total latency is dominated by a constant
regardless of outcome; `hmac.compare_digest` against a fixed-length configured key doesn't vary
timing with the guess's content; and the one asymmetry I found — a `422` from Pydantic's
`min_length`/`max_length` validation on `PresenterSessionIn.key` runs *before* `presenter_login` is
even called, skipping the delay entirely — leaks nothing new, since a `422` already announces
itself overtly by status code and can never be the right key regardless of timing.

**F8's both-orderings wire contract is correct and tested on both branches.** Traced
`ResetAllStateUnknownError.participants` through `StorefrontHTTPError`'s `**extra` kwargs into
`_handle_storefront_http_error`'s `{"error": ..., "detail": ..., **exc.extra}` dict: `participants=None`
serializes as JSON `null` (key present), never omitted, and the typed-vs-untyped discriminator
(`participants` present, even as `null`, only when the sweep itself was attempted) is asserted on
the wire by two distinct tests (`tests/test_storefront_api.py:2409` — `is None`; `:2449` —
`"participants" not in body`).

**The move is clean.** Thin route bodies in `storefront_api.py`, the tombstone
(`storefront_api.py:843-849`) correctly names where `_STEP_10_INTERIM`'s content went, and the
forced `config.py`/`SERVER.md`/test additions are genuinely forced — I re-derived
`test_config_reads_exactly_the_documented_storefront_env_vars`'s cross-check myself and confirmed
it fails without both the `expected` set and the `SERVER.md` row updated.

## Open questions

None — nothing here needs a call beyond the two findings' own suggested fixes.

## Pass 2 — 2026-09-11 (confirming look at `coder`'s fix)

**Verdict: approve.** Both Pass 1 findings fixed, verified below, with one new nit surfaced by
probing the fix itself rather than by re-reviewing.

- **Major (F8 four-key narrowing untested) — fixed, confirmed by re-running my own Pass 1
  mutation.** `test_a_reset_all_that_times_out_is_504_unknown_and_is_never_retried`
  (`tests/test_storefront_api.py:2384-2385`) now asserts `set(row) == {"participantId",
  "displayName", "language", "joinedAt"}` on every row of the F8 re-read's `participants`, inside
  the exception path specifically (the test drives `reset_all_participants` to raise
  `TimeoutError`, landing in `reset_all`'s `except` arm, not the happy path or the
  `incomplete`/`unresolved` success-path field). Re-applied my Pass 1 mutation (narrowing replaced
  by `list(self.list_participants())`, the raw six-key rows) on a byte-backed copy of
  `storefront.py`: `test_a_reset_all_that_times_out_is_504_unknown_and_is_never_retried` now fails
  with `AssertionError: assert {'channelId', ..., 'threadId'} == {'displayName', ...
  'participantId'}` — the exact leak the finding named. Restored, md5-confirmed identical to the
  pre-mutation file (`7abb8e9e…`).

- **Minor (ambiguous Pass-7 citation) — fixed.** All five `(Pass 7, P7-5)` sites
  (`storefront.py:242,628,692,719-720`, `config.py:234`) now read `` `docs/reviews/
  salesperson-ui.md` `## Pass 7`, P7-5 `` in the same sentence, unambiguous against the file's
  other citations of `salesperson-ui-impl.md`'s differently-numbered passes.

- **New nit, found while probing the fix rather than re-deriving it — the new assertion pins "the
  narrowing exists" but not "the narrowing is applied uniformly across every row."** The only test
  exercising this arm joins a single participant ("Ada"), so `for row in body["participants"]:
  assert set(row) == {...}` only ever iterates one row. I applied a second, different mutation:
  narrow correctly for row 0, pass the raw six keys through unmodified for every row after it
  (`if i == 0 else row`, `enumerate`d). Ran `tests/test_storefront_api.py tests/test_storefront.py`:
  **251 passed** — silently survives, because the fixture never gives the loop a second row to
  betray it on. Restored, md5-confirmed identical (`7abb8e9e…`). **Fix, one line:** add a second
  `_join` (e.g. `"Bob"`, a different language) in this test alongside the existing `_join(client,
  "Ada", "en")`, so the assertion loop actually iterates ≥2 rows — no new test needed, the existing
  loop already generalizes once the fixture does. Not a blocker: the narrowing is per-row
  dict-comprehension code with no index-dependent branch in it today, so this mutation shape has no
  natural trigger in the current implementation — it is a latent gap in the *test*, not a
  live risk in the *code*, which is why it doesn't change the verdict.

Suite reference count: `tests/test_storefront_api.py tests/test_storefront.py` → **251 passed**,
both before and after each restore, matching `coder`'s and the coordinator's own reruns.
