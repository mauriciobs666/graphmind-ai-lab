# `mention-reply-delivery` — plan-gate review (K-039 item 3)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-039 (M3.5)

## Scope & verdict

**Reviewed:** `docs/plans/mention-reply-delivery.md` (full text), statically, pre-implementation —
no code exists yet. Checked against: the RCA it traces back to
(`docs/reviews/mention-reply-delivery-rca.md`), the K-039 backlog entry and its shared
done-condition (`docs/BACKLOG.md` ~line 1120-1188, K-027 addendum ~line 431-433), both 2026-07-31
`docs/HISTORY.md` entries (the K-039 immediate-mitigation fix and the same-day `ws:acme` cleanup),
`falkor-chat/AGENTS.md`, and the source the plan cites: `server/falkorchat/services.py`
(`check_demo_readiness`, `DEMO_EXPECTED_DEFS`, `RAG_QUERY_TIMEOUT_MS` precedent), `repository.py`
(`find_runs_for_thread`, `complete_run`/`fail_run`/`link_step_emission`, `start_run`), `api.py` (the
readiness route + `response_model` usage), `config.py` (`TRIGGER_DEF_KEY`/`_VERSION`),
`web/app.js`/`web/index.html` (readiness banner), `scripts/bootstrap_schema.sh` (`WorkflowRun`
indexes), `server/pyproject.toml` (`addopts`), and the test files `test_executor_agent.py`,
`test_executor_produced.py`, `test_repository.py`, `test_services.py`, `test_api.py`,
`test_workflow_live.py`.

Beyond static reading, I ran three independent verifications this gate doesn't get for free:
1. Ran `cd server && .venv/bin/python -m pytest -q` myself — **647 passed, 1 deselected**, matching
   the plan's/HISTORY.md's claimed baseline exactly.
2. Started `falkordb-dev` and, against the project's actual pinned engine (`v4.18.11`, matching
   `claude/graph-dba/falkordb-quirks.md`'s verified version) and the project's actual `falkordb-py`
   client, ran the plan's proposed §12.15 query verbatim (both zero-matching-run and
   some-runs-didn't-post shapes) to check its stated `sum()`-over-empty-group edge case.
3. Queried the live `ws:acme` graph (same persistent Docker volume the RCA used) to check whether
   the specific run the plan cites as QA evidence in step 4 is still there.
   (FalkorDB was left running afterward — a `docker stop`/`rm falkordb-dev` attempt was blocked by
   the sandbox's command classifier; no project data was mutated, only two of my own throwaway
   graph keys, created and deleted within this review.)

**Verdict: approve with suggestions.** The core decision — decline promoting any `@pytest.mark.live`
test into the default loop; build a lagging, informational post-success signal on the existing
readiness route instead — is sound, correctly grounded in the repo's own recorded intent, and
genuinely closes the RCA's contributing-factor-2 gap for the specific mechanism that broke. No
blockers. Five findings below, all independently verified (not just re-asserted from the plan's own
text), worth fixing before or during implementation; none requires a design change.

## Findings

### Minor — the plan's stated `sum()`-over-zero-rows edge case is factually wrong; the real gotcha is a float, not a `None`

§4 Step 1 states as fact: *"`sum(...)` over zero matched rows returns Cypher `NULL`, not `0`"* and
instructs `graph-dba` to "confirm this live." §4 Step 2 then tells `coder` to "coalesce a `None`
`postedCount` to `0`" on that basis. I ran the exact proposed query, live, against the project's
pinned FalkorDB version through the project's actual `falkordb-py` client, in both the zero-matching-
`WorkflowRun` case and a "some ran, none posted" case:

```
sampleSize = 1 (int), postedCount = 0.0 (float)      # zero-row aggregation
sampleSize = 2 (int), postedCount = 1.0 (float)      # non-empty case
```

`sum(CASE WHEN … THEN 1 ELSE 0 END)` never returns `None`/`NULL` here — it returns `0.0`. The
`None`-coalesce §4 Step 2 asks `coder` to write is therefore a no-op for a case that doesn't arise.
The **actual** gotcha, which the plan doesn't mention at all, is a **type** mismatch: `postedCount`
comes back as a Python `float` (`0.0`/`1.0`), not an `int`, from every `sum()` call over this `CASE`
expression, in both the empty and non-empty case — `sampleSize` (a `count()`) stays a clean `int`.
Left uncast, `check_demo_readiness`'s JSON response would carry `"postedCount": 1.0` instead of `1`,
and §4 Step 3's proposed banner text (`"{postedCount}/{sampleSize} replied"`) would literally render
`"1.0/2 replied"`.

*Why it matters:* low risk on its own (an `int()` cast is a one-line fix, and `graph-dba`'s own "confirm
this live" instruction in step 1 would likely surface *some* discrepancy) — but the plan's specific,
asserted claim about *which* edge case to guard against is wrong, and pointing `coder` at the wrong
mechanism (`None`-coalescing) risks a fix that "looks done" (code added, tests pass since `0.0 == 0`
in Python) while the real int/float wrinkle ships untouched into the JSON contract and the banner text.

*Suggested fix:* correct §4 Step 1's edge-case note to describe the actual behavior (`sum()` returns a
`float`, never `NULL`/`None`, over this query shape) and change §4 Step 2's repository/service guidance
from "coalesce `None` to `0`" to "cast `postedCount` to `int`" (e.g. `int(posted_count)` in
`Repository.read_recent_post_success`, alongside the `sampleSize`/`postedCount` dict construction).

### Minor — plan cites a repository method that does not exist

`docs/plans/mention-reply-delivery.md` §2.3 and §4 Step 1 both cite `Repository.read_thread_workflow_runs`
(`repository.py:1529-1531`) as the precedent for the `r.startedAt >= 0` index-anchor idiom. I read
`repository.py` directly (`grep -n "def read_thread_workflow_runs"` returns nothing): no such method
exists anywhere in the file. The method at that location — the one that actually carries the cited
`WHERE r.startedAt >= 0 AND m.threadId = $threadId` clause and the QUERIES.md §12.14 PROFILE finding
the plan is drawing on — is `find_runs_for_thread` (`repository.py:1511-1544`, doc comment
1514-1526). The line numbers the plan cites are correct; only the method name is wrong, in two places.

Relatedly (a smaller nuance, not worth a separate entry): `find_runs_for_thread`'s doc comment
explains the `startedAt >= 0` idiom is load-bearing there specifically because its *other* filter
(`m.threadId = …`) sits on a **different pattern variable** (`m`, pulling the scan anchor off `r`).
The new §12.15 query's filters (`defKey`/`defVersion`/`status`) are all on `r` itself, and
`WorkflowRun.status` is independently indexed too (`bootstrap_schema.sh:145-146`) — a meaningfully
different planner situation, so "copies the established idiom" slightly overstates the transfer. The
plan already correctly defers final verification to `graph-dba`'s `GRAPH.PROFILE` in step 1, so this
doesn't change the outcome — just worth graph-dba recording *which* index (`startedAt` vs. `status`)
the plan actually anchors on and why, not only "an index scan, not a label scan."

*Why it matters:* small — an implementer or `graph-dba` searching the codebase for
`read_thread_workflow_runs` to check the precedent won't find it, and may propagate the wrong name
onward into `docs/QUERIES.md` §12.15's own writeup.

*Suggested fix:* correct both occurrences to `find_runs_for_thread` before/while `graph-dba` executes
step 1; have the step 1 PROFILE note in QUERIES.md record which index the query actually lands on.

### Minor — an existing test breaks the moment `postSuccess` lands, and the plan doesn't say so

§4 Step 2's test list says to extend `test_api.py`'s `test_readiness_route_*` tests "to assert the
response body now also carries `postSuccess`." I read the actual test file: `test_api.py:757` defines
`_READINESS_KEYS = {"ready", "defs"}`, and `test_readiness_route_not_ready_when_nothing_seeded`
(`test_api.py:768`) asserts `set(body) == _READINESS_KEYS` — an **exact** top-level key-set equality,
not a subset check. I confirmed `check_demo_readiness`'s current return statement
(`services.py:1068`) is literally `return {"ready": ready, "defs": results}` — exactly those two keys.
Adding a third key (`postSuccess`, as §4 Step 2 specifies) breaks this assertion the moment the field
lands, independent of and before any new assertion an implementer adds.

*Why it matters:* this is precisely the "existing consumer of the route the plan didn't check" class
of risk a plan gate exists to catch (brief item 5). It's not a blocker — `pytest -q` fails loudly and
the fix is a one-line widening of `_READINESS_KEYS` — but leaving it unnamed means `coder` rediscovers
it from a red test rather than the plan flagging it up front, and it's the kind of thing worth
stating explicitly given the plan otherwise claims (correctly, for the Pydantic/`response_model` side)
that the route change is "backward-compatible."

*Suggested fix:* add one sentence to §4 Step 2's test list naming `_READINESS_KEYS` (`test_api.py:757`)
as needing to widen to `{"ready", "defs", "postSuccess"}`.

### Minor — §4 Step 4's cited QA evidence run was deleted by a same-day cleanup the plan doesn't reference

§4 Step 4 tells `qa-engineer`: *"`ws:acme` conveniently already has real historical data covering
'some runs didn't post' … `runId 00d95a27ac2a4dc8b74a86ed117b5c95`, produced nothing) alongside runs
that did post via item 1's fix."* I queried live `ws:acme` (same persistent volume the RCA used) for
that exact `runId`: **zero rows returned** — it does not exist. Reading `docs/HISTORY.md`'s
2026-07-31 "Cleanup" entry (the one dated the same day but listed *more recently* than the immediate
mitigation, i.e. it happened after) explains why: that entry explicitly deletes this exact `WorkflowRun`
(`00d95a27ac2a4dc8b74a86ed117b5c95`) and its triggering message as part of removing the RCA's own
live-repro artifacts. The plan's §2.3/§4 sourcing predates that cleanup and wasn't refreshed against it.

What *is* still live in `ws:acme` (I queried directly): three `triage@v1` `WorkflowRun`s total — one
`done` with a `PRODUCED` message (`6dea1ba3c5d543cebf5f5a578ad07073`, the RCA's separately-noted
corroborating run, explicitly left untouched by the cleanup), one `done` with zero `PRODUCED`
messages, and one still `waiting` (correctly excluded by the plan's own terminal-status filter). That
actually gives a real, live "degraded" (1/2 posted) case to exercise — but no live "ok"-only case,
which the plan already separately anticipates needing a throwaway workspace for.

*Why it matters:* low — `qa-engineer` will discover the citation is stale the moment they query for
it (an empty result, not a wrong result), and the plan's own hedge ("a synthetic 'ok'-only or 'no-data'
case may still need a throwaway workspace") already covers the gap in practice. But a plan citing a
specific, verifiable-but-now-false piece of live evidence is exactly the "grounding matches the real
codebase/state" class of check this gate exists for, and it costs `qa-engineer` a dead-end lookup.

*Suggested fix:* update §4 Step 4 to cite the still-live corroborating run
(`6dea1ba3c5d543cebf5f5a578ad07073`) instead of the deleted one, and note plainly that a throwaway
workspace is needed for the "ok" state (no live "ok" example currently exists in `ws:acme`).

### Nit — `postedCount`'s float type is worth a one-line callout in QUERIES.md itself, not just the service layer

Follows from the first finding: since `sum()` here always returns a `float`, `docs/QUERIES.md` §12.15
(authored in step 1, before step 2's service code exists) is the right place to record the observed
Python-side result types (`sampleSize: int`, `postedCount: float`) so `coder` doesn't have to
rediscover it by inspecting `res.result_set` types by hand. Low stakes — the finding above already
covers the actionable fix — but worth folding into the same QUERIES.md edit graph-dba is already
doing in step 1.

## What's solid

- **The core (a)/(b) decision is well-reasoned and correctly grounded.** I read `server/pyproject.toml`'s
  `addopts` comment directly — its wording matches the plan's §2.2/§3.1 quotation exactly — and the
  plan's argument that promoting any live-gated test into the default loop reintroduces exactly the
  property that comment was written to prevent holds up.
- **The "already covered by offline tests" claim is true, live-verified twice over.** I read
  `test_executor_produced.py:126-155`
  (`test_implicit_post_when_tool_not_called_still_creates_produced_edge_live`) and the four tests in
  `test_executor_agent.py` directly — they construct exactly the RCA's live-reproduced failure shape
  and assert a real `Message` + `PRODUCED` edge. I then ran `pytest -q` myself:
  **647 passed, 1 deselected**, matching the plan's/HISTORY.md's claimed count exactly — not merely
  read as claimed.
- **The proposed §12.15 Cypher's `LIMIT`-before-`OPTIONAL MATCH` shape is correct** — `LIMIT $limit`
  sits in the `WITH` clause before the `OPTIONAL MATCH` fan-out, so the sample is bounded to N
  `WorkflowRun`s, not N `StepRun`-`Message` pairs (a classic pagination-after-join bug the plan avoids).
  Live-verified this holds for both the empty and non-empty case above.
- **RAM/index claim verified against `bootstrap_schema.sh`.** No `WorkflowRun.defKey` index exists
  today (confirmed by grep); the plan doesn't add one, reusing the existing `startedAt` index (and,
  as noted above, the independently-existing `status` index) with `defKey`/`defVersion` as residual
  filters — consistent with the file's own stated tiny-cardinality rationale.
- **Scope discipline holds.** K-027 item 2 and unrelated K-036 follow-ups are correctly fenced out in
  both §1 and §3.4; the signal is correctly narrowed to `config.TRIGGER_DEF_KEY`/`_VERSION` only —
  verified against `services.py:357-360`, which shows `access-request@v1` is a structurally different
  def, plus its full `DEMO_EXPECTED_DEFS` tuple confirmed by direct read.
- **Unit sequencing/ownership is sound.** `coder` (not `tdd-engineer`) for step 2 matches this repo's
  own stated routing distinction — the repository method, service constant, and JSON dict shape are
  all fully pinned by the plan (module out to the float/int wrinkle above), with no ambiguous behavior
  a red/green cycle would need to discover. The `graph-dba → coder → frontend-engineer → qa-engineer →
  doc-close-out` sequence has no missing dependency; step 4's two pieces are correctly independent of
  each other.
- **The `_start`/`_start_at` + `complete_run`/`fail_run`/`link_step_emission` test-seeding precedent
  §4 Step 2 points `coder` at is real** — confirmed these helpers exist at the cited location
  (`test_repository.py:1138-1474`) in exactly the composable shape the plan describes.
- **The open question in §6 (should `postSuccess` ever flip `ready`?) is correctly left open** — it's
  a product-scope call that doesn't block implementation (the design is additive and stays easy to
  extend later), and flagging it here rather than guessing is the right move at this gate.

## Open questions

None that block this gate. The one open product-scope question the plan itself flags (§6: whether a
persistently degraded post-success rate should ever flip `ready`) is appropriately left to whoever
owns the demo-readiness product surface, not decided here.

---

## Pass 2 — diff-scoped re-gate (2026-07-31)

> **Scope:** implementation review of `docs/plans/mention-reply-delivery.md` v2 §4 Steps 1–3 —
> `graph-dba` (query), `coder` (repository/service wiring), `frontend-engineer` (web banner). Diff
> against `main` on the 9 files: `docs/QUERIES.md`, `scripts/test_queries.sh`,
> `server/falkorchat/repository.py`, `server/falkorchat/services.py`, `server/tests/test_api.py`,
> `server/tests/test_repository.py`, `server/tests/test_services.py`, `web/app.js`,
> `web/index.html`. Steps 4 (`qa-engineer` acceptance + the live-verification action) and 5
> (doc close-out) are not part of this diff and were not reviewed — confirmed still open in
> `docs/BACKLOG.md` K-039 (item 3 not yet marked delivered). Does not re-review the plan itself
> (Pass 1 above still stands).

**Verdict: approve.** No blockers, no majors. The implementation matches the v2 plan's design
precisely, including every correction the Pass 1 review forced into the plan (the `int()` cast
instead of a `None`-coalesce, the `_READINESS_KEYS` widening, `postSuccess` kept separate from
`ready`). Independently verified, not taken on any implementer's self-report:

1. **Ran the full pytest suite myself:** `cd server && .venv/bin/python -m pytest -q` →
   **658 passed, 1 deselected** — exactly the claimed 647 → 658 (+10 new test functions across
   `test_repository.py`/`test_services.py`, +1 new regression-pin test, +2 existing `test_api.py`
   assertions extended in place — 647 + 11 new test functions = 658).
2. **Ran the query suite myself:** `./scripts/test_queries.sh` → **282/282 passed**, matching
   `graph-dba`'s claimed count and `QUERIES.md`'s own header (`282/282, 2026-07-31`).
3. **Spot-checked all three of `coder`'s claimed mutation-testing catches, not just one** — each by
   deliberately reintroducing the bug, running the targeted tests, observing the failure, then
   reverting via the original file content I'd already `Read` (confirmed byte-identical after via
   `diff`, then re-ran the full suite to confirm 658/658 clean):
   - Dropped `int(row[1])` back to raw `row[1]` in `repository.py` →
     `test_read_recent_post_success_all_posted` fails on `assert isinstance(res["postedCount"], int)`
     (`assert False = isinstance(2.0, int)`). Real catch, not a tautology — the assertion targets the
     exact float/int wrinkle the plan called out.
   - Swapped the `status` branch order in `services.py` (`"ok" if posted==sample else "no-data" if
     sample==0 else "degraded"`) → `test_check_demo_readiness_post_success_no_data_when_sample_empty`
     fails (`'ok' == 'no-data'`, since `0 == 0` satisfies the now-first `posted_count == sample_size`
     branch before the zero-sample check ever runs). Real catch.
   - Reverted `test_api.py`'s `_READINESS_KEYS` back to `{"ready", "defs"}` →
     `test_readiness_route_not_ready_when_nothing_seeded` fails with `Extra items in the left set:
     'postSuccess'`. Real catch.
   All three reversions confirmed clean (`diff` against the pre-mutation file, then a final
   `pytest -q` at 658 passed) before moving on — the tree was never left in a broken state.
4. **Read every line of the actual diff** (`git diff` on all 9 files) — Cypher, Python, and JS — not
   just the plan's prose:
   - **Query (`QUERIES.md` §12.15 / `repository.py`):** the Cypher is a verbatim match to the plan's
     pinned shape — `startedAt >= 0` anchor, `defKey`/`defVersion`/`status IN ['done','failed']`
     filter, `ORDER BY startedAt DESC LIMIT $limit` before the `OPTIONAL MATCH` fan-out (correctly
     avoids the pagination-after-join trap Pass 1 already praised), `sum(CASE …)`. Fully
     parameterized (`{"defKey": def_key, "defVersion": def_version, "limit": limit}`), consistent
     with `AGENTS.md` rule 1. `int(row[1])` cast present exactly where Pass 1's Minor 1 asked for it
     — `sampleSize`/`postedCount` are both clean `int`s in the returned dict.
   - **New planner fact, genuinely earned, not asserted:** §12.15's PROFILE write-up documents that
     two independently-indexed `AND`-ed `WHERE` predicates on the same label fold into **one**
     `Node By Index Scan`, not "pick one, filter the other" — backed by a real isolation test (a
     probe row failing exactly one of the two predicates, shown excluded at the scan step, before
     `Filter`). This is promoted into `claude/graph-dba/falkordb-quirks.md` ("Query tuning"
     section) — I read both the QUERIES.md write-up and the KB entry; they agree and the KB entry
     is correctly framed as a general engine fact, not schema-specific. Real, useful finding, not
     boilerplate.
   - **`services.py`:** `POST_SUCCESS_SAMPLE_SIZE = 20` as a plain module constant next to
     `RAG_QUERY_TIMEOUT_MS`, exactly as specified. `postSuccess` dict shape
     (`defKey`/`defVersion`/`sampleSize`/`postedCount`/`rate`/`status`) matches §4 Step 2's pinned
     code block field-for-field, including `rate = None` when `sampleSize == 0` and the
     `"no-data"`/`"ok"`/`"degraded"` derivation order. `postSuccess` is added as a fourth top-level
     key alongside the pre-existing `return {"ready": ready, "defs": results}` — genuinely additive,
     not folded into `ready`'s computation (confirmed by reading the full `check_demo_readiness`
     method, not just the diff hunk).
   - **`app.js`/`index.html`:** `renderReadinessBadge` (`app.js:615-620`) is untouched — still
     `report.ready ? "readiness--ready" : "readiness--not-ready"`, no `postSuccess` reference
     anywhere in it. `renderPostSuccess` is a new, separate function composed into
     `renderReadinessPanel`'s output, not merged into the badge path. The degraded-state CSS
     (`.post-success--degraded`, `index.html:105-106`) reuses the pre-existing `--mention`/
     `--mention-line` custom properties (already defined at `index.html:10`, used for the chat
     mention-highlight styling) — confirmed by grep, not a new color invented, per the plan's
     explicit ask.
5. **Regression safety:** the pre-existing `test_check_demo_readiness_*` tests are untouched in this
   diff (`git diff` shows zero hunks inside their bodies), and a new test
   (`test_check_demo_readiness_existing_ready_and_defs_assertions_are_regression_pinned`) re-asserts
   the same `ready`/`defs` shape as an explicit pin. `FakeRepo.read_recent_post_success`'s default
   (`{"sampleSize": 0, "postedCount": 0}`) means the older tests, which never script
   `post_success_result`, exercise the real "no-data" code path rather than an unset/`None` sentinel
   — a deliberate, correct choice, not an oversight.
6. **Scope discipline:** `git status --porcelain -- falkor-chat/` shows modifications on exactly the
   9 listed files, nothing else. No K-027 item 2 engine-contract code, no unrelated K-036 changes.
7. **`ws:acme` shared-state hygiene:** `test_queries.sh` was run as part of this review (per the
   brief) and, as documented, wiped the shared `reference` graph — `verify_workflows.sh acme`
   reported **FAIL** immediately before my run too, for reasons unrelated to this diff (the
   pre-existing, already-tracked `reference`/`ws:acme` drift documented in `docs/HISTORY.md`'s
   2026-07-31 cleanup entry, not something this review or this diff caused). I ran
   `./scripts/seed_workflows.sh acme` afterward; `./scripts/verify_workflows.sh acme` now reports
   **OK — 2 defs in sync**.

### Minor findings

None. This diff earned an unqualified **approve** — every Pass 1 correction landed exactly as
specified, the new planner fact is a genuine, well-evidenced contribution to the shared KB, and all
three of `coder`'s mutation-testing claims held up under independent reproduction.

### What's solid

- **Every one of Pass 1's five findings is visibly and correctly resolved in the code**, not just in
  the plan text: the `int()` cast (not a coalesce), the corrected planner-fact framing (now backed by
  a real isolation test rather than asserted), the `_READINESS_KEYS` widening, and the `ws:acme`
  citation correction (§4 Step 4, not reviewed here since it's `qa-engineer`'s step, but the still-live
  corroborating run cited in the plan is unaffected by this diff).
- **Test layering matches the plan's own prescription exactly** — repository-layer tests for the six
  named cases (all-posted, some-posted, none-posted, zero-runs, `limit` truncation, non-terminal
  exclusion), service-layer tests for the three `status` states plus a call-args assertion plus the
  regression pin, API-layer tests for the route's JSON shape including the exact-key-set fix.
- **The `test_queries.sh` fixture is well-isolated** — a dedicated `defKey` (`post_success_probe`)
  not used anywhere else in the suite, explicit cleanup at the end of the section, and a
  collision-resistant string-composed assertion (`"sampleSize=… postedCount=…"`) rather than bare
  digit matching that could collide with PROFILE/redis-cli noise — consistent with the suite's own
  existing convention (§12.1/§12.2).
- **The new FalkorDB planner fact is real, not filler** — verified by reading both the QUERIES.md
  write-up and the promoted `claude/graph-dba/falkordb-quirks.md` entry; the isolation methodology
  (drop-each-predicate variants plus a probe row that fails exactly one predicate) is sound evidence
  for the "both predicates fold into one index scan" claim.

### Open questions

None. Steps 4 (qa acceptance + the live-verification action) and 5 (doc close-out) remain open per
`docs/BACKLOG.md` K-039 and are outside this diff's scope — routing them to `qa-engineer` and the
plan's own step-5 owners is unchanged from the plan's sequencing, not a new open question raised by
this review.
