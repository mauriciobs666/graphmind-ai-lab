# CPG agent adoption — Test Plan (successor 2)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-409 (M4)

## 1. Scope & objective

Targeted, small follow-up to the archived `cpg-agent-adoption` acceptance pass — **not** a
re-run of the original suite. Both prior passes (`docs/test-reports/cpg-agent-adoption-report.md`
Pass 1 and Pass 2, both archived) only ever observed a dispatched agent querying a live CPG's
freshness marker and getting **zero rows back** — the "no marker at all" edge of AC-3/AC-4. That
gap is `docs/BACKLOG.md` item **C-409**, which names `qa-engineer` as trigger owner for a
follow-up once either live graph picks up a real rebuild.

`graph-dba` has now rebuilt `cpg_falkorchat` on request. It carries a real, populated
`:CpgBuildInfo` row for the first time in this feature's test history. This document plans and
records a single targeted dispatch that exercises that previously-unobserved condition: an agent
consulting a loaded CPG that actually returns a populated freshness marker, on a graph whose
build pattern is known (per `docs/plans/cpg-agent-adoption-graph.md` §6) to lack
`SOURCE_COMMIT`/`SOURCE_DIRTY` — so the dispatch also exercises the freshness recipe's documented
raw-age-only fallback for this specific graph.

This is instrumentation of an existing, already-shipped mechanism, not feature delivery. One
dispatch, one small real code-level task, scoped to stay low-risk and proportionate to the size
of the gap it closes.

## 2. References

- `docs/BACKLOG.md` — item C-409 (the trigger for this follow-up).
- `docs/test-plans/cpg-agent-adoption.md` and `docs/test-reports/cpg-agent-adoption-report.md`
  (both archived) — the original acceptance pass (U6) and live-dispatch re-pass (U9); this
  document extends their coverage, not repeats it. See report Pass 2 D1′ in particular (`coder`
  on `Repository.materialize_snapshot`) — this dispatch is deliberately the same shape
  (`coder`, impact-analysis question, `falkor-chat/server`, `cpg_falkorchat`) so the only material
  variable is the freshness marker's content, not the dispatch discipline.
- `docs/requirements/cpg-agent-adoption.md` — AC-3 (freshness signal on report-back) and AC-4
  (refresh suggestion when the signal indicates staleness), both archived, both still the
  contract under test here.
- `skills/cpg-analysis/references/freshness.md` — the read recipe, its two escalating staleness
  checks, and its own documented limitation for a `.git`-less scratch-copy build (exactly
  `cpg_falkorchat`'s build pattern, per its "Limits" section).
- `docs/plans/cpg-agent-adoption-graph.md` §6 ("Staged-source builds lose `SOURCE_COMMIT`") — the
  design-level acknowledgment that this graph's `SOURCE_COMMIT`/`SOURCE_DIRTY` will come back
  absent by design.

## 3. CPG relevance check (own orientation)

This document's own subject *is* `cpg_falkorchat`'s freshness marker, so this pass uses the
graph directly — not for code-under-test impact analysis, but to independently confirm the
marker's live content before writing test items (see §4). `CPG: used cpg_falkorchat` — see §4
for the query and result.

## 4. Live-environment grounding (confirmed before writing test items)

Independently re-verified, not taken on the dispatching prompt's word:

```
MATCH (b:CpgBuildInfo)
RETURN b.BUILT_AT, b.SOURCE_PATH, b.SOURCE_COMMIT, b.SOURCE_DIRTY
```
against `cpg_falkorchat` →

| BUILT_AT | SOURCE_PATH | SOURCE_COMMIT | SOURCE_DIRTY |
|---|---|---|---|
| `2026-08-17T00:40:42Z` | `/tmp/cpg-src/falkor-chat-server` | `null` | `null` |

Confirmed at query time (`2026-08-17T00:46:33Z`, ~6 minutes after `BUILT_AT`) via
`mcp__cypher__query`. `GRAPH.LIST` confirms both `cpg_falkorchat` and `cpg_salesperson` still
loaded. `SOURCE_COMMIT`/`SOURCE_DIRTY` are absent, as predicted by
`cpg-agent-adoption-graph.md` §6 for this graph's `.git`-less scratch-copy build pattern — the
dispatched agent has no way to run the stronger git-log staleness check and must fall back to
raw-age reasoning only, per `freshness.md`'s own documented limitation.

`cpg_salesperson` is out of scope for this follow-up — the rebuild that unblocks it (per C-409)
has not happened; it is not re-checked here.

## 5. Risk assessment & coverage strategy

**In scope:**
- Does a dispatched agent, given a task that plausibly warrants consulting `cpg_falkorchat`,
  actually query the freshness marker (AC-3), on a graph now returning a real, non-empty row?
- Does it correctly read that row — recognizing `SOURCE_COMMIT`/`SOURCE_DIRTY` as absent (not
  erroring, not treating absence as some other signal, e.g. "commit unknown = definitely stale")
  and falling back to `BUILT_AT` raw-age reasoning only?
- Given `BUILT_AT` is ~minutes old at dispatch time, does it correctly conclude "fresh enough,
  no refresh suggestion needed" — the **positive** branch of the staleness judgment, i.e. does
  the mechanism avoid a false-positive stale claim on a graph that is genuinely fresh? This is
  the mirror case of AC-4's "suggest a refresh" branch: proving the check doesn't cry wolf is
  itself evidence the mechanism works.
- Does the agent still emit the literal `CPG:` evidence line per the plan's three-shape
  convention (`docs/plans/cpg-agent-adoption.md` §3)?

**Deliberately out of scope, and why:**
- **A genuinely stale, populated marker.** `cpg_falkorchat`'s marker is minutes old at dispatch
  time — there is no organic source drift to observe staleness against, and fabricating a stale
  timestamp or backdating a commit would be gaming the observation, which the task explicitly
  rules out. This edge of AC-4 (a real stale-marker refresh suggestion) remains genuinely
  unobserved after this pass; carried forward honestly in the report, not closed by proxy.
- **`cpg_salesperson`.** Not rebuilt; still zero rows; already covered by the archived passes'
  "no marker" case.
- **The stronger git-log staleness check.** Structurally unavailable for this graph's build
  pattern (§4) — nothing to observe here regardless of dispatch design.
- **Re-testing DEF-1/DEF-2/DEF-3/DEF-4 (already closed/tracked).** Those are closed or tracked
  as C-408 in `docs/BACKLOG.md`; not this follow-up's job to re-litigate.
- **`architect`/`tdd-engineer`/`frontend-engineer`/`qa-engineer`(self) dispatch variety.** One
  dispatch is proportionate to a targeted instrumentation follow-up; `coder` is chosen because
  the task shape (impact analysis on a signature change) is squarely its normal orientation and
  its dispatch shape is a direct, comparable extension of D1′ (same agent, same repo, same
  graph, different target function) — isolating the marker's content as the one new variable.

## 6. Test items

### TP-001 — AC-3: freshness marker query returns a real populated row; agent reads it correctly

**Preconditions:** `cpg_falkorchat` loaded with a populated `:CpgBuildInfo` row (confirmed §4).

**Steps:** Dispatch `coder` on a small, real, investigation-shaped task against
`falkor-chat/server` that plausibly warrants a call-graph/impact-analysis check — no CPG
mention, no request to edit files. Exact prompt: impact analysis of adding a new required
parameter to `Repository.advance_cursor` (`falkorchat/repository.py`) — what call sites (production
code, not tests) would need to change.

**Expected result:** The agent's normal orientation surfaces `cpg_falkorchat`; it runs the
freshness recipe unprompted and gets back the real row from §4 (not zero rows).

**Priority:** High. **Type:** Acceptance / live-dispatch.

### TP-002 — AC-3: correct handling of absent `SOURCE_COMMIT`/`SOURCE_DIRTY` on a populated row

**Preconditions:** Same dispatch as TP-001.

**Expected result:** The agent recognizes `SOURCE_COMMIT`/`SOURCE_DIRTY` are `null` and falls
back to `BUILT_AT` raw-age reasoning only — it does not attempt `git log` against
`/tmp/cpg-src/falkor-chat-server` (a nonexistent scratch path outside the repo), does not error,
and does not misread the absence as an unrelated signal (e.g. treating `null` commit as itself
proof of staleness).

**Priority:** High. **Type:** Acceptance / live-dispatch.

### TP-003 — AC-4 mirror: no false-positive staleness claim on a genuinely fresh marker

**Preconditions:** Same dispatch; `BUILT_AT` is ~minutes old at dispatch time (§4).

**Expected result:** The agent concludes the graph is fresh enough to trust and does **not**
emit an unwarranted refresh suggestion. (The converse — a genuinely stale populated marker
triggering a correct refresh suggestion — remains out of scope per §5; not testable without
fabricating drift.)

**Priority:** Medium. **Type:** Acceptance / live-dispatch.

### TP-004 — AC-2: literal `CPG:` evidence line present and correctly shaped

**Preconditions:** Same dispatch.

**Expected result:** The response's closing line is a literal `CPG:` line matching one of the
plan's three defined shapes (`used` is expected here, since the CPG is loaded, has a marker, and
is plausibly relevant to the task).

**Priority:** Medium. **Type:** Acceptance / live-dispatch.

## 7. Environment & data setup

Live dispatch against the real `falkor-chat/server` repo tree and the real `cpg_falkorchat`
graph in the shared FalkorDB instance — no fixtures, no mocks, matching the original pass's
"drive the real thing" discipline. The dispatched task is investigation-only and explicitly
does not request file edits; an incidental code change is acceptable per the task brief but not
the point of this pass, and none is required for TP-001…TP-004 to resolve.

## 8. Entry/exit criteria

**Entry:** `cpg_falkorchat` confirmed loaded with a populated `:CpgBuildInfo` row (§4, done).

**Exit:** All four test items resolved to pass/fail/blocked with quoted evidence from the
dispatch transcript; C-409 in `docs/BACKLOG.md` updated to reflect the outcome (closed, or
narrowed to the specific remaining gap).

## 9. Explicitly out of scope

See §5. Summarized: `cpg_salesperson` (not rebuilt), the git-log staleness check (structurally
unavailable for this graph), a genuinely stale populated marker (not organically observable
without fabrication), and re-litigating DEF-1…DEF-4 (already closed/tracked elsewhere).
