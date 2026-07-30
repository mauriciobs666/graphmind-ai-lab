# `wf-trigger-def-key-collision` (K-037) — review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-037 (M3.5 follow-up)

## Scope & verdict

Static review of `tdd-engineer`'s uncommitted K-037 delivery: the env-var collision between
`FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` (the app's real `@mention`-trigger override,
`config.py`) and `scripts/seed_workflows.sh`'s reuse of that same pair to pick the publish
identity of its own inline `triage`-literal def. Reviewed against `falkor-chat/docs/BACKLOG.md`'s
K-037 entry (root cause + candidate fix + test-strategy note) and
`falkor-chat/docs/test-reports/web-api-coverage-report.md` Finding 1 (major) + Finding 2 (minor),
which is where K-037 was filed from.

Files read in full: `falkor-chat/scripts/seed_workflows.sh`, `falkor-chat/scripts/start_server.sh`
(both live, post-edit), `falkor-chat/scripts/verify_workflows.sh` (unchanged — checked for
scope-discipline drift), `falkor-chat/server/falkorchat/config.py` (`TRIGGER_DEF_KEY`/`_VERSION`
definitions), `falkor-chat/server/falkorchat/services.py` (`DEMO_EXPECTED_DEFS`),
`falkor-chat/server/.env.example`, `falkor-chat/server/tests/test_workflow_live.py` (the one place
a test shells out to `seed_workflows.sh`), and the diff against working tree via
`git diff falkor-chat/scripts/seed_workflows.sh falkor-chat/scripts/start_server.sh
falkor-chat/docs/BACKLOG.md falkor-chat/docs/HISTORY.md`. Ran `bash -n` on both changed scripts
(clean). Did **not** re-run the pytest suite or `test_queries.sh` — both are documented to wipe the
shared `reference` graph on the live instance, which is exactly the destructive side effect
`AGENTS.md` already flags and HISTORY.md's follow-up section already accounts for; re-triggering it
gratuitously during a static review would add nothing checkable statically and would mutate shared
state I have no mandate to touch.

**Verdict: approve with suggestions.** The decoupling is correct and complete — traced end-to-end,
not just grepped — and the doc updates are accurate. One major-leaning gap (no regression guard for
this class of bug) and a couple of minor/nit polish items, none of which block landing.

## Findings

### Major — no automated regression guard for the exact bug class this item fixes

The backlog item's own test-strategy note (`docs/BACKLOG.md`, K-037, "Test strategy") proposed "a
script-level check (or a `verify_workflows.sh`-style assertion) that seeding with
`FALKORCHAT_TRIGGER_DEF_KEY=access-request` set does **not** add any `Step`/`START` edge to
`access-request@v1` beyond its canonical 6/1 shape." HISTORY.md's "Verified" section documents that
this exact scenario *was* exercised manually (RED against pre-fix code with a throwaway
`wf037decoy@v1`, GREEN after the fix) — good, real verification, not a claimed-green rubber stamp —
but the throwaway data was deleted afterward and no repeatable check was committed. Nothing in this
diff would stop the next person from reintroducing the collision (e.g., "simplifying" by pointing
`seed_workflows.sh` back at `FALKORCHAT_TRIGGER_DEF_KEY` for consistency with `config.py`'s naming)
without the mistake surfacing until the next accidental graft.

`server/tests/test_workflow_live.py` already establishes the precedent of a pytest module shelling
out to `scripts/seed_workflows.sh` via `subprocess.run` against a throwaway workspace (`ws:live`,
gated `@pytest.mark.live`) — the exact shape this needs, minus the `live` LLM dependency. Suggest a
narrow, network-free pytest test (no `live` marker needed, since it only exercises the seed script +
Cypher structure reads, not the LLM): run `seed_workflows.sh` once against a throwaway workspace/def
pair with `FALKORCHAT_TRIGGER_DEF_KEY` (and, for belt-and-suspenders, `FALKORCHAT_TRIAGE_DEF_KEY`
too) set to the *other* def's key/version, then assert via `services.get_workflow_def_structure`
that neither def's step/edge count moved from its canonical shape. This is the regression test the
backlog item itself scoped and is the natural next unit for `tdd-engineer` to pick up, not something
this review is asking to block the current change on.

### Minor — header-comment ordering in `seed_workflows.sh` states the invariant before the rationale

`scripts/seed_workflows.sh:66-70` ("The TRIAGE def key/version MUST match the trigger config
(`config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION`, defaults `triage`/`v1`)...") reads, in isolation, as
if the two pairs are meant to be kept manually in sync — which was exactly the (wrong) assumption
that caused K-037. The correct explanation — that they're two independent pairs that merely *default*
to the same values — doesn't appear until the K-037 block two paragraphs later (lines 72-81). A
reader skimming top-to-bottom hits the confusing framing first. Not wrong, just ordered
against the grain of the fix. Suggest moving the K-037 block immediately after this paragraph (or
folding a one-line forward-reference into it, e.g. "— but as two independent pairs that happen to
share defaults; see K-037 below for why they must never be merged back into one").

### Nit — `HISTORY.md`'s "Files touched" list drops the backtick styling used everywhere else in the same entry

`docs/HISTORY.md:89` lists `Files touched: falkor-chat/scripts/seed_workflows.sh,
falkor-chat/scripts/start_server.sh, falkor-chat/docs/BACKLOG.md.` without the backticks the rest of
the entry (and the rest of the file's convention) uses for file paths. Cosmetic only.

### Nit — `start_server.sh` doesn't document that `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` are reachable through it via env inheritance

`start_server.sh`'s stage-5 invocation of `seed_workflows.sh` (line 138-142) only explicitly forwards
`FALKORCHAT_WS_ID`/`FALKORDB_HOST`/`FALKORDB_PORT`; it never references
`FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` at all. They still work if exported by the caller before
invoking `start_server.sh` (ordinary bash export inheritance — the same mechanism that caused the
original bug), but that path is undocumented in `start_server.sh`'s own header/usage text (unlike
every other overridable var, including the new `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`). Low value
to fix (nobody has an obvious reason to override the triage-literal's own publish key via
`start_server.sh` rather than `seed_workflows.sh` directly), but worth a one-line mention if the
header comment is touched again.

## What's solid

- **The decoupling is complete and correctly wired**, verified by tracing the actual data flow, not
  just grepping: `seed_workflows.sh` computes `KEY`/`VERSION` for the triage-literal exclusively from
  `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` (`seed_workflows.sh:108-109,158-159,277-278`); the Python
  heredoc imports `config` only for `config.get_context()`, never touches
  `config.TRIGGER_DEF_KEY`/`_VERSION`. `grep -n "TRIGGER_DEF" scripts/seed_workflows.sh` returns only
  comments explaining the *former* bug and the *deliberate* non-reuse — zero remaining functional
  reads of that pair. Confirmed this holds even though the parent process (a demo restart) still has
  `FALKORCHAT_TRIGGER_DEF_KEY` exported and thus inherited into the Python subprocess's environment —
  it's simply never read there anymore.
- **Defaults match by construction and are documented as doing so on purpose**: `TRIAGE_DEF_KEY`
  defaults to `triage`/`v1`, identical to `config.TRIGGER_DEF_KEY`/`_VERSION`'s own defaults, so an
  unoverridden `./scripts/start_server.sh` run is provably unaffected — traced end-to-end rather than
  assumed.
- **`start_server.sh`'s new banner fix is correct**: defaults declared with the standard
  `${VAR:-default}` env-override-then-default pattern used by every other var in the script, exported
  unconditionally (consistent with `FALKORCHAT_WORKFLOW_ENABLED` et al.), and the banner line now
  interpolates the live values instead of a hardcoded literal.
- **Bash hygiene is clean**: `set -euo pipefail` intact in both scripts, all new/changed variable
  expansions are double-quoted, no word-splitting or glob risk introduced, and the new
  `TRIAGE_DEF_KEY`/`TRIAGE_DEF_VERSION` local-var naming now mirrors the existing
  `PROCESS_DEF_KEY`/`PROCESS_DEF_VERSION` convention it's modeled on (previously it was the generic,
  collision-prone `DEF_KEY`/`DEF_VERSION` — the rename is itself a small clarity win).
- **Scope discipline held.** `verify_workflows.sh` is untouched by the diff, and I confirmed the
  implementer's reasoning independently: `services.DEMO_EXPECTED_DEFS` is built directly from
  `(config.TRIGGER_DEF_KEY, config.TRIGGER_DEF_VERSION)` plus `proof_defs.ACCESS_REQUEST_DEF`'s own
  key/version (`services.py:357-359`) — it never reads anything `seed_workflows.sh` defines locally,
  so it was structurally incapable of being affected by this change either way. No destructive
  cleanup of the pre-existing `ws:acme` snapshot contamination was attempted or silently masked
  anywhere in the diff — confirmed by reading both scripts fully; neither contains a delete/republish
  path, and HISTORY.md's follow-up section documents the divergence candidly and stops per
  `verify_workflows.sh`'s own "if they diverge, do NOT re-seed" instruction rather than working around
  it.
- **Doc accuracy is good.** Checked the updated header comments in both scripts, the BACKLOG.md
  status-line flip, and the full HISTORY.md entry (including the same-day follow-up) against what the
  diff actually does — all match. Also checked adjacent, unedited references that could plausibly have
  gone stale (`config.py`'s `TRIGGER_DEF_KEY` docstring, `server/.env.example`,
  `verify_workflows.sh`'s own header) and confirmed none of them assume the two env-var pairs are
  still shared — they were never wrong in the first place, since they only ever documented
  `config.py`'s pair, not `seed_workflows.sh`'s internal one.
- **Regression risk on the two paths this review was asked to check is low.** The unoverridden
  default path (`./scripts/start_server.sh`, no env vars) is unaffected — traced through both defaults
  matching. The legitimate trigger-override demo/QA path (`config.py`'s own `@mention` resolution,
  exercised via `FALKORCHAT_TRIGGER_DEF_KEY`) is untouched — `config.py` itself wasn't edited, and
  `test_app.py`'s `TRIGGER_DEF_KEY`/`_VERSION` monkeypatches are independent of anything in this diff.
  The one live test that shells out to `seed_workflows.sh` (`test_workflow_live.py`, `@pytest.mark.live`)
  invokes it with no def-key overrides at all, so it exercises only the matching-defaults path.

## Open questions

- The `ws:acme` snapshot divergence on `access-request@v1` (pre-existing, out of scope for this diff
  per the backlog item's own text) is still open as a separate stakeholder-gated decision — noted
  here as context per the task brief, not held against this change.
- Whether the regression-test gap (major finding above) should be filed as its own backlog follow-up
  or folded into closing out K-037 is a call for whoever owns K-037's close-out, not something this
  review can decide unilaterally.

## Pass 2 — 2026-07-30, narrow re-review of follow-ups

Scope: not a full re-review — confirming the four Pass 1 follow-ups landed correctly. Read
`server/tests/test_seed_workflows_script.py` in full, the reordered header block in
`scripts/seed_workflows.sh`, `docs/HISTORY.md`'s "Files touched" line (`cat -A`), and the new
env-inheritance comment in `scripts/start_server.sh`. Did not re-run the implementer's RED/GREEN
regression proof or the pytest suite myself (not required — see below); ran `bash -n` on both
scripts and `python3 -m py_compile` on the new test file (all clean).

1. **New regression test — confirmed, correctly built.**
   `server/tests/test_seed_workflows_script.py::test_trigger_def_key_override_does_not_graft_onto_the_other_def`:
   - Shells out to the real script (`subprocess.run(["bash", str(_SEED_WORKFLOWS), "test"], ...)`,
     `_SEED_WORKFLOWS = _REPO_ROOT / "scripts" / "seed_workflows.sh"`) — not a reimplementation.
   - Throwaway everything: `ws="test"` (never `acme`), throwaway def identities
     `k037-triage-guard`/`k037-target-guard` (never `triage`/`access-request`); grepped the file for
     `acme` — zero hits.
   - `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` are set explicitly in `_run_seed`'s `env` dict on every
     call, never left to the script's own default — the docstring correctly explains why (a
     regression that silently stopped reading that pair would otherwise hide behind matching
     defaults).
   - Replays the collision correctly: baseline `_run_seed()` publishes both throwaway defs at
     canonical shape (asserted: `stepCount == 6`/`3`, no `startKeys`), then
     `_run_seed(trigger_key=TARGET_KEY, trigger_version=TARGET_VERSION)` points
     `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` at the *other* throwaway def's identity — the exact
     shape of K-037's `FALKORCHAT_TRIGGER_DEF_KEY=access-request` demo override, replayed against
     disposable data.
   - Asserts via `services.get_workflow_def_structure` that neither def's structure moved
     (`target_after == target_before`, `triage_after == triage_before` — full structural equality,
     not just a step count, so it would also catch a stray extra `START` edge).
   - Cleans up at teardown: `_cleanup_throwaway_defs` (fixture, `yield`-based, so teardown runs
     regardless of test outcome) `DETACH DELETE`s both throwaway defs from `reference` and
     `ws:test`.
   - The reported RED/GREEN proof holds up on a static read: pre-fix, `TRIAGE_DEF_KEY` falls back to
     reading `FALKORCHAT_TRIGGER_DEF_KEY` (unset on the *baseline* call, since `trigger_key=None`
     there), which defaults to `"triage"` — so the triage-literal would publish under `triage@v1`
     instead of `k037-triage-guard@v1`, and the very next line
     (`services.get_workflow_def_structure(CTX, key=TRIAGE_KEY, ...)`) would raise
     `WorkflowDefNotFoundError` exactly as reported. I did not re-run this myself (not required per
     the brief), but the logic is sound and the failure mode is exactly what reverting the fix would
     produce.

2. **Polish item 1 (header reorder) — confirmed, resolves the concern.**
   `scripts/seed_workflows.sh:66-83`: the K-037 decoupling explanation (why the two pairs are
   independent, sharing only *defaults*) now precedes the "TRIAGE def key/version MUST MATCH the
   trigger config" paragraph, joined by an explicit bridge sentence ("Read the next paragraph in
   light of the above... but that's a match of DEFAULTS between two independent pairs, never a
   shared variable to keep manually in sync — the exact wrong assumption K-037 fixes"). A
   top-to-bottom reader no longer hits the easy-to-misread framing first. Concern resolved.

3. **Polish item 2 (HISTORY.md backtick styling) — confirmed false alarm, and it was my error, not
   a miss on the implementer's part.** Re-reading my own Pass 1 evidence: the diff I quoted at the
   time already showed the line as `` Files touched: `falkor-chat/scripts/seed_workflows.sh`,
   `falkor-chat/scripts/start_server.sh`, `falkor-chat/docs/BACKLOG.md`. `` — backticked throughout.
   My Pass 1 nit misread that line. Spot-checked the current file myself
   (`docs/HISTORY.md:198-199`, `cat -A`): every path is backticked, no stray characters, nothing to
   fix. The implementer's "no change made, confirmed via `cat -A`" report is accurate.

4. **Polish item 3 (start_server.sh env-inheritance note) — confirmed, correctly placed.**
   `scripts/start_server.sh:134-137`, immediately above the stage-5 seed invocation: "Only
   `FALKORCHAT_WS_ID`/`FALKORDB_HOST`/`FALKORDB_PORT` are explicitly forwarded below, but
   `seed_workflows.sh`'s own `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` (K-037) still reach it if the
   caller exported them before invoking this script — ordinary bash env inheritance, the same
   mechanism the original K-037 bug relied on, now pointed at the harmless var." Accurate and well
   placed.

**`ws:acme` snapshot cleanup note:** `graph-dba`'s concurrent surgical cleanup (separate unit,
independently verified) is visible in `docs/HISTORY.md` as its own dated entry ("K-037 follow-up:
surgical cleanup of `ws:acme`'s contaminated `access-request@v1` snapshot") plus a mention in this
diff's own "Follow-up 2" section explaining why `ws:acme` now reads clean. Not re-verified here —
out of this pass's scope, and not a source of confusion once read.

**Updated verdict: approve.** All four follow-ups landed correctly; the previously-open major
(no regression guard) is now closed with a well-built, correctly-scoped test. No new issues found
in this pass.
