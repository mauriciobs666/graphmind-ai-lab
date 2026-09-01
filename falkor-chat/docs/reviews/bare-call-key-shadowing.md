# Review — bare-call argument-key shadowing (K-035, implementation gate)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-035 (M4)

Reviewer: `analyst` · Type: static diff review vs `architect`'s plan, plus executed offline test
runs. **Baseline reviewed:** the uncommitted working-tree diff, `cd falkor-chat && git diff --
server/falkorchat/llm.py server/tests/test_llm.py docs/HISTORY.md docs/BACKLOG.md`, checked
against `docs/plans/bare-call-key-shadowing.md` (§3/§4/§5) as the spec.

## Scope & verdict

**Scope.** The implementation diff for K-035 — `_parse_content_tool_calls`'s guard, the
`_BARE_CALL_OPEN` relocation, six new tests in `test_llm.py`, and the `BACKLOG.md`/`HISTORY.md`
housekeeping — checked for: plan conformance, whether the new tests actually exercise the fix,
whether all three repro rows close and both rejected-alternative hazards are avoided, regression
risk on named existing tests, quality of the landed residual documentation, and doc housekeeping.
Did not review `docs/plans/bare-call-key-shadowing.md` or `-coordination.md` themselves (out of
scope per the brief) beyond using them as the spec to check against.

**Verdict: approve.**

**CPG:** considered, not relevant — re-verified the plan's own grounding rather than trusting it:
`grep -rn "_normalize_tool_call" server/falkorchat/*.py server/tests/*.py` confirms exactly one
call site (`llm.py:319`, inside `_parse_content_tool_calls` itself); this is a single-file,
two-named-function change with no cross-module call-graph question a CPG would answer faster.

## Findings

No blockers, no majors.

**1 (minor, plan hygiene, not a diff defect).** Plan §1 scope item 4 says "**Five** new tests,"
but enumerates three repro pins + one regression pin + one negative pin + one optional residual
pin = **six**, which is exactly what §5 spells out and exactly what landed
(`test_llm.py:600-676`, confirmed by reading all six bodies against §5's code blocks — byte-for-
byte match). The implementer correctly followed the authoritative §5 test-strategy section, so
this is a latent inconsistency in the plan document itself, not an implementation problem.
Suggested improvement: `architect` corrects "Five" to "Six" in a future revision of the plan (the
plan is still `active`, so this is a same-slug in-place fix per the collision rules, not a
successor document).

**2 (informational, self-disclosed).** Verifying the plan's "no regression, traced individually"
claim, I ran the default offline suite (`cd server && .venv/bin/python -m pytest -q`), which — per
`falkor-chat/AGENTS.md`'s own documented hazard — wipes the shared `reference` graph at teardown.
This was my own action, not a diff defect; I repaired it in full before finishing (`bootstrap_schema.sh
acme` → `seed_demo.sh acme` → `seed_workflows.sh acme` → `seed_catalog.sh acme` →
`seed_salesperson.sh acme`) and re-confirmed with all three read-only verify scripts, each exit 0
(`verify_workflows.sh acme`, `verify_catalog.sh`, `verify_salesperson.sh acme` — all "in sync").
Flagging this so the record is honest about what touched shared state during this review, and as a
reminder for the next reviewer to prefer the file-scoped run (`pytest -q tests/test_llm.py`, which
needs no live FalkorDB at all for this file) over the full default suite when only `llm.py`/
`test_llm.py` are in scope.

## Verification performed

- **Guard code matches plan §3 exactly**, including the added comment
  (`llm.py:314-317` vs plan §3's code block) — confirmed by direct diff comparison, not just
  reading the plan's own claim.
- **`_BARE_CALL_OPEN` relocation** — pure move, identical regex/flags/comment, now sits at
  `llm.py:274-281`, immediately above `_parse_content_tool_calls` (`llm.py:284`), referenced at
  three sites (`llm.py:296`, `:318`, `:405`) plus one comment mention (`:535`) — `grep -n
  "_BARE_CALL_OPEN" falkorchat/llm.py` shows no duplicate definition left behind.
- **Docstring/comment rewrite** (`llm.py:295-306`) reads correctly on its own: states the fixed
  behavior, gives the same repro example the plan uses, and names the residual — a future reader
  does not need the plan doc to understand the trade-off. The stale "harmless only while…" framing
  is gone, as the plan required.
- **Tool_calls-list branch untouched** — read the final code (`llm.py:308-322`), not just the
  plan's reasoning: the guard wraps only the single-object `_normalize_tool_call` branch; the
  `tool_calls`-list check (`:310-313`) runs unconditionally before the guard, exactly preserving
  candidate 1's claimed property that no input is ever routed away from that branch (closing
  candidate 2's collision hazard by construction, not by assertion).
- **Layering hazard (candidate 3)** — confirmed the fix reads nothing from a granted-tool-name
  set; `executor.py`/`guards.py` do not appear in the diff.
- **All three repro rows** — hand-traced through the actual (not plan-described) code:
  `create_user({"name": "bob"})` opens its line with `create_user(`, so `_BARE_CALL_OPEN.search`
  matches, the guard suppresses `_normalize_tool_call`, and `_parse_bare_call_syntax` (unchanged)
  recovers `ToolCall(name='create_user', arguments={"name": "bob"})`; same trace pattern for the
  `action`/`tool` rows. Also ran the tests: `.venv/bin/python -m pytest -q tests/test_llm.py` →
  **55 passed** (this file needs no live FalkorDB, 0.04s).
- **Regression traces spot-checked beyond the required 2-3** — five of the six plan-named
  existing tests plus the full K-027 bare-call section (`test_llm.py:282-576`, ~19 tests): every
  content string that opens a line with `identifier(` (the "echoed call" test, the "quoted call"
  tests, `OBSERVED_BARE_CALL`, etc.) only ever carries `text`/`query` argument keys, so
  `_normalize_tool_call` already returned `None` for them pre-fix — the new guard cannot change
  their outcome regardless of whether it engages. The two JSON-envelope tests
  (`test_chat_parses_content_embedded_json_fallback`, `test_chat_falls_through_an_empty_tool_calls
  _envelope_to_the_sibling_call`) open with `{`, never matching `_BARE_CALL_OPEN`, so the guard is
  false for them by construction. Also ran the full offline suite (`pytest -q`, minus `live`):
  **2291 passed, 0 failed** — no regressions anywhere in the suite, not just the traced subset
  (count/deselect split differs from HISTORY's reported 2315/14, plausibly an environment-marker
  difference unrelated to this diff — `test_llm.py` itself is 100% green both isolated and
  in-suite, which is the part under review).
- **Mutation-testing claim plausibility** — traced by hand rather than re-executed (per the
  brief): reverting the guard restores the pre-fix path for all three repro-row tests (content
  matches `_BARE_CALL_OPEN`, so without the guard `_normalize_tool_call(obj)` runs and returns
  `ToolCall(name='bob', arguments={})` etc., failing the tests' `call.name == "create_user"`
  assertions for the exact shadowing reason) and for test 5 (the negative-direction pin — without
  the guard, the shadowed name resolves the same way, wrongly returning `is_tool_call=True`). The
  claim in `HISTORY.md` is plausible and specific enough (three tests, "for the original reason")
  to trust.
- **Test 6's behavior is a real, intentional regression on an unobserved shape** — traced: pre-fix,
  a genuine envelope followed by an unrelated trailing `foo(bar)` line would have parsed correctly
  (the envelope's own `name` key resolves it, the trailing line is irrelevant to the old code);
  post-fix, the content-wide guard suppresses it and the whole message falls through to text. This
  is exactly the "Residual, by design" trade-off both the plan and the landed comment name — not a
  silent side effect.
- **Doc housekeeping** — `grep -n "K-035" docs/BACKLOG.md` returns nothing; the removed hunk is
  clean (no dangling headers/blank-line artifacts). `HISTORY.md` gained one dated entry
  (2026-09-01, most-recent-first position) with a "What"/"Tests" structure comparable in shape and
  detail to its neighbor (2026-08-31 K-061 entry) — appropriately shorter, since K-035's fix and
  test surface are genuinely smaller.
- **`ruff check falkorchat/llm.py tests/test_llm.py`** — all checks passed; formatting (blank-line
  spacing between the new test defs) is correct on inspection too.

## What's solid

- The diff is exactly what the plan promised: one guard, one relocation, one docstring rewrite,
  six tests, two doc edits — no scope creep, no touched files beyond what §1/§4 named.
- The residual trade-off is documented in three places consistently (plan §3, `llm.py`'s docstring,
  and a dedicated test) with the same framing each time — a future reader hitting any one of the
  three gets the full picture.
- Test placement, naming, and structure follow the file's existing conventions exactly (labeled
  section comment, `_chat_content` helper reuse, one assertion block per test).

## Open questions

None requiring the caller's input — this gate is straightforward: the diff matches its spec, the
tests are real (not vacuous), and doc housekeeping is complete.

## Appendix — repair commands run after the full-suite pytest run

```
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh acme
./scripts/seed_demo.sh acme
./scripts/seed_workflows.sh acme
./scripts/seed_catalog.sh acme
./scripts/seed_salesperson.sh acme
```

All three verify scripts (`verify_workflows.sh acme`, `verify_catalog.sh`, `verify_salesperson.sh
acme`) returned exit 0 / "in sync" / "OK" afterward.
