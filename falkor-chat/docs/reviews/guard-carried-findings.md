# Guard carried findings — diff review (K-027 six carried findings)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-027 (carried findings)

## Scope & verdict

Reviewed `tdd-engineer`'s diff (unit U1 of `docs/plans/guard-reliability-followups-coordination.md`)
against the six still-open "carried findings from the analyst gate"
(`docs/archive/reviews/m3-guard-thread-context-impl.md`) recorded under `docs/BACKLOG.md`'s
`### K-027`: **m-1** (false-advance negation bug), **m-2** (`_recent_turns` filter-before-slice),
**m-3** (evidence-tier trace visibility), **n-1** (stray local `import json as _json`), **n-2**
(O(n²) judge-prompt cap), and the **doc-drift** byte-count figure. Baseline: `git diff` against the
working tree (`HEAD` vs. uncommitted changes) on `server/falkorchat/{guards,executor,app}.py`, their
tests (`server/tests/test_{guards,executor,app}.py`), and `docs/{BACKLOG,HISTORY}.md`. I did not
redo the SHA/byte-count/suite-count verification the coordinator already ran independently and
recorded exact numbers for — I focused on code correctness and test quality, then independently
re-derived the same numbers as an incidental byproduct of tracing the logic, and they match.

**Verdict: approve.**

**CPG:** not applicable — the brief states `cpg_falkorchat` is stale relative to this diff's own
files (built 2026-08-17T00:40:42Z, several commits since, including this diff) and directs reading
`guards.py`/`executor.py`/`app.py` directly rather than leaning on it for structural claims; I did
so, and had no need for call-graph/impact-analysis questions the CPG would otherwise answer.

## Findings

No blockers, no majors. One minor, one nit.

### Minor — `_CLAUSE_BOUNDARY` including `,` can also cut off a *same*-clause negation

`server/falkorchat/guards.py:100-101,527-535`. The clause-boundary truncation added for m-1 treats
a bare comma the same as `;`/`.` — any negator sitting the far side of *any* comma within the
12-char lookback stops counting, even when the comma is a mid-clause parenthetical rather than a
genuine clause break. Verified live:

```python
>>> r = "The information is not, strictly speaking, unclear"
>>> _coerce_verdict({"decision": True, "rationale": r})
GuardVerdict(decision=False, rationale='rationale contradicts advance ...', tier=None)
```

Here `"not"` and `"unclear"` are grammatically the same clause (a parenthetical aside sits between
them), so this rationale should read as an *affirmation* ("not unclear" ≈ "clear") and let the
verdict advance — instead the new code reads it as a contradiction and force-suspends. This is real
overcorrection, but **it lands on the safe side**: `_rationale_contradicts`'s whole design (and the
corrected comment) is explicit that erring toward "not negated → contradiction → suspend" is the
accepted-safe direction under DS Q1, exactly symmetric to the false-advance bug this finding fixes.
So this is not a regression of the kind m-1 was chasing (it never produces a false *advance*), just
a narrower true-negative window than strictly necessary, and the scenario (a comma-delimited
aside sandwiched between negator and cue in an LLM-produced rationale) is uncommon phrasing.
Suggested improvement, non-blocking: if this shows up in live calibration data, consider narrowing
`_CLAUSE_BOUNDARY` to `;`/`.` only (drop `,`) and re-run the m-1 probe table to confirm the three
pinned `SUSPENDING_RATIONALES` still contradict correctly (they all use `;`, so dropping `,` would
not un-pin them) — or leave as-is and note the tradeoff explicitly in the `_NEGATOR_WINDOW` comment
block. Not filed as a backlog item; flagging here is enough given the safe-direction landing.

### Nit — `n-2`'s "no mutation-test applies" framing undersells what was actually verified

`docs/BACKLOG.md`/`docs/HISTORY.md` describe n-2 as "a refactor under green... no new RED/GREEN
cycle applies," which is accurate for TDD process but slightly undersells the verification that
*was* done: a 5000-trial (I re-ran independently at 5000, not the claimed 2000) differential
fuzz test against the literal old algorithm, covering `n_turns ∈ {0,1,2,3,6,10,50,300}`,
`understanding` absent/falsy/populated, and randomized text lengths — 0 mismatches. That is
stronger evidence than the prose conveys. No action needed; noting only because a future reader
skimming BACKLOG might undervalue the confidence behind "verified byte-identical... across 2000
randomized trials."

## What's solid

- **m-1 direction and fix, independently reproduced.** I traced `_coerce_verdict` →
  `_rationale_contradicts` → `_is_negated` by hand for the brief's exact example
  (`"The user did not say; more info is needed."`): the *old* 12-char window puts `"not "` inside
  the lookback (`'id not say; '`), reads it as negating the `"more info"` cue, so
  `_rationale_contradicts` returns `False` and a `decision:true` verdict stays `True` — a genuine
  **false advance**, confirming the bug's claimed direction is correct (not false-suspend). The
  *new* code truncates the window at the `;`, `_is_negated` returns `False` (not negated), the
  cue now correctly contradicts, and the verdict correctly flips to `False` (suspend). Both traces
  reproduced by direct execution against the actual `guards.py`, not by reading alone.
- **The three pinned `SUSPENDING_RATIONALES`** (`"The user did not say; more info is needed."`,
  `"Alice gave no version; still need the logs."`, `"She said 'I do not know'; more info is
  needed."`) match the archived gate's probe table (`m3-guard-thread-context-impl.md:236-238`)
  verbatim — not approximations.
- **m-2's new test targets the right shape.** `test_malformed_rows_in_the_tail_do_not_shrink_the_
  evidence_window` puts the 3 malformed rows *after* 6 valid ones specifically — a filter-order bug
  would be invisible if the malformed rows were elsewhere in the list, and this test would not have
  passed under the old slice-then-filter code (6 valid + 3 malformed tail, `n=6` window, old code
  would return only 3).
- **m-3's additivity claim holds.** `GuardVerdict.tier: str | None = None` — grepped every
  `GuardVerdict(...)` construction site in `falkorchat/`; all are 0–2 positional/keyword args, none
  would collide with the new third field. Full suite green (1098/3, see below) corroborates no
  breakage. `_select_transition` (`executor.py:943`) and `_trace_step` (`:1043`) independently
  re-derived to sit outside the SHA-locked `_drive_loop` region: `awk` extraction bounded by
  `/^    def _drive_loop/` (line 451) to `/^    # ── seams/` (line 512) reproduces
  SHA `71055f756280`, 2860 bytes — matching both the delegate's and the coordinator's reported
  figures exactly, and matching `_drive_loop`'s actual line range (451–511), well short of 943/1043.
  The no-tier trace shape (`"{label} -> {decision}: {rationale}"`, no `[tier]` segment) is preserved
  byte-for-byte for `cmp`/unconditional guards, verified by reading `_trace_step`'s
  `tier_note = f" [{verdict.tier}]" if verdict.tier else ""` — empty string when `tier` is falsy.
- **n-1 is a real one-command confirmation**, not a hand-wave: `grep -rn "import json as _json"
  server/falkorchat/` → zero matches; `git log` confirms `1dd48a0` (2026-07-24, K-027 slice A,
  already delivered and unrelated to this run) hoisted the top-level `import json` in the same
  `app.py` functions this finding named.
- **n-2 is genuinely output-identical**, independently re-derived by writing a standalone
  differential harness reproducing both the literal pre-diff loop and the new arithmetic side by
  side (not reusing the delegate's harness) and running 5000 randomized trials across turn counts
  0–300, `understanding` present/absent/empty, and randomized text lengths: 0 mismatches. I also
  traced the arithmetic manually (the `total -= len(lines[n-kept]) + (1 if kept > 1 else 0)`
  separator bookkeeping) and confirmed it matches the `"\n".join`/`"\n\n".join` semantics exactly,
  including the `kept == 0` fallthrough to `base[:JUDGE_USER_MAX_CHARS]` matching the old code's
  empty-`turns` fallthrough.
- **Doc-drift resolved honestly.** `grep -rn '2844\|2839' docs/` confirms every hit where 2844/2839
  are asserted as *live* byte-count figures is inside `docs/archive/` (`m3-executor-coordination.md`,
  `m3-guard-thread-context-impl.md`); the hits inside `docs/BACKLOG.md`/`docs/HISTORY.md`/
  `docs/plans/guard-reliability-followups-coordination.md` are this run's own prose *describing* the
  historical wrong figures as wrong, not asserting them as current fact. Nothing live needed
  correcting.
- **Diff scope is exactly what's claimed.** `git diff --stat` touches only the six named files plus
  `docs/BACKLOG.md`/`docs/HISTORY.md` — no `scripts/`, no `QUERIES.md` executable Cypher, no
  DDL/index changes. K-027 items 1–3 (already-delivered subsections) are untouched; items 4/5
  (owned by sibling U2/U3 in this run) are untouched; the unrelated carried finding
  `m-A/n-1` (missing `node_note` in the trace-kind enumeration) is explicitly left alone with an
  honest "not part of this run's scope" annotation rather than silently dropped or folded in.
- **Full offline suite reproduced independently:** `.venv/bin/python -m pytest -q` from `server/` →
  **1098 passed, 3 deselected**, matching the coordinator's own baseline exactly.
- **Shared-state hygiene:** my own pytest run wiped `reference` (the known offline-suite hazard
  documented in `falkor-chat/AGENTS.md`); re-seeded via `./scripts/seed_workflows.sh acme` and
  confirmed `./scripts/verify_workflows.sh acme` → exit 0, "2 defs in sync" before finishing.

## Open questions

None — all six findings are grounded, correctly scoped, and test-backed. The one minor
(`_CLAUSE_BOUNDARY` comma sensitivity) is worth a mental note for whoever next touches this
heuristic during live calibration, but does not block this delivery.
