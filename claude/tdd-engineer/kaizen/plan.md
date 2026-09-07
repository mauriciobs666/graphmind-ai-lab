# Kaizen — Improvement Plan: tdd-engineer

> Forward-looking backlog for the `tdd-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07 (`kaizen_team` distillation, chunk A — see history.md)

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-007 | 2026-09-07 | med | 🔵 | Document the shared-`falkordb-dev` concurrency flake in `falkor-chat/docs/SERVER.md` §1.7 |
| K-008 | 2026-09-07 | med | 🔵 | Document the bare-workspace-id contract of the seed scripts in `falkor-chat/AGENTS.md` |
| K-009 | 2026-09-07 | low | 🔵 | Document `DatasetSchema`'s second, non-obvious construction site in `falkor-chat/docs/SERVER.md` §1.7 |
| K-010 | 2026-09-07 | med | 🔵 | Document that `trace=True` alone traces nothing without `tracer=GraphTracer(repo)` — `falkor-chat/docs/SERVER.md` §1.7 |

> All four are `falkor-chat` project-doc writes, **outside `cobb`'s write remit** — that is why
> they are kept open here rather than promoted during the distillation pass. Each belongs to
> whoever next works the relevant `falkor-chat` area, or routes through `teco` to the doc's owner.

### K-007 — the shared `falkordb-dev` instance makes `pytest -q` flake across unrelated files
- **Status:** 🔵 proposed · **Priority:** medium
- **Source:** `kaizen_team` entry `f3a8c9d2-…` (2026-08-25), kept open at the 2026-09-07 distillation.
- **Fact:** when another agent session is concurrently using `falkordb-dev`, `pytest -q` on
  `falkor-chat/server` shows 21–53 transient failures scattered across unrelated test files
  (`test_workflow_timers.py`, `test_repository.py`, `test_graphrag.py`, `test_tools.py`), with a
  **different failure set on each rerun** and no relation to your own change. Reproduced identically
  on a `git stash`-clean HEAD (ruling the change out), then fully green (1766 passed) on an
  immediate rerun with no code change.
- **Why it matters:** the diagnostic move is "re-run once and compare the failure *set*, then check
  for a concurrent session" — without it the natural response is to bisect one's own change against
  noise. `SERVER.md` §1.7 already lists four/five hazards "a green `pytest` run does not surface";
  this is the same kind and the section is the established home.
- **Proposed change:** one bullet in `falkor-chat/docs/SERVER.md` §1.7 ("Testing hazards specific
  to `server/`"), and bump that section's "Four gotchas" lead count.

### K-008 — seed scripts take a **bare** workspace id; an already-prefixed one silently creates `ws:ws:<id>`
- **Status:** 🔵 proposed · **Priority:** medium
- **Source:** `kaizen_team` entry `a7f3c1d2-…` (2026-08-28), kept open at the 2026-09-07 distillation.
- **Fact (re-verified 2026-09-07):** `bootstrap_schema.sh`, `seed_demo.sh`, `seed_workflows.sh`,
  `seed_catalog.sh` and `seed_salesperson.sh` all build the graph key themselves
  (`bootstrap_schema.sh:111`, `local g="ws:${wid}"`), with **no guard** against an input that is
  already prefixed. Passing the value `GRAPH.LIST` displays (`ws:acme`) therefore creates a real,
  bogus `ws:ws:acme` graph key, silently and with exit 0, leaving the intended workspace untouched.
- **Why it matters:** `GRAPH.LIST` is the natural place to look up the id to pass, and it shows the
  prefixed form — so the wrong value is the one in front of you. `AGENTS.md`'s "Key scripts" table
  writes the parameter as `<wsId>`, which does not disambiguate.
- **Proposed change:** say **bare id, no `ws:` prefix** in the `falkor-chat/AGENTS.md` "Key scripts"
  table (once, on the `bootstrap_schema.sh` row, since it heads the seed sequence). A `case "$wid" in
  ws:*) die ...` guard in the shared script preamble would be stronger, but that is a code change,
  not a doc one — worth proposing separately.

### K-009 — `querygen.DatasetSchema` is hand-constructed in `test_repository.py` too, not only `test_querygen.py`
- **Status:** 🔵 proposed · **Priority:** low
- **Source:** `kaizen_team` entry `d8f0c1e2-…` (2026-08-30), kept open at the 2026-09-07 distillation.
- **Fact (re-verified 2026-09-07):** `server/tests/test_repository.py` builds its own ad hoc
  `DatasetSchema` (lines ~3498 and ~3523) to probe `run_readonly_query` end to end. Changing
  `DatasetSchema.labels`' shape therefore breaks a second test file that a grep scoped to
  `test_querygen.py` would miss — which is exactly how it was found (a `TypeError: frozenset object
  is not subscriptable` from `test_repository.py` after a `labels` shape change).
- **Why it matters:** low frequency, but the failure lands in a file the author has no reason to be
  looking at. The §18 comment block in `test_repository.py` explains *why* that schema is ad hoc; the
  missing half is the reverse pointer for whoever edits `querygen.py`.
- **Proposed change:** cheapest correct fix is a one-line note on `querygen.DatasetSchema`'s own
  docstring naming both construction sites — a code comment, not a doc. Alternatively a `SERVER.md`
  §1.7 bullet. Whoever next touches `querygen.py` should take the docstring option.

### K-010 — `trace=True` alone writes zero `TraceEvent`s from an ad-hoc in-process executor
- **Status:** 🔵 proposed · **Priority:** medium
- **Source:** `kaizen_team` entry `f3a1e6b2-…` (2026-08-30), kept open at the 2026-09-07 distillation.
- **Fact (re-verified 2026-09-07):** tracing needs **both** halves. `executor.py:605` reads
  `tracer = self._tracer if run["trace"] else _NULL_TRACER`, and `__init__` sets
  `self._tracer = tracer or _NULL_TRACER` — so a script that constructs
  `WorkflowExecutor(services, repo, llm=…, tool_registry=…)` without `tracer=` and then passes
  `trace=True` on the run gets `_NULL_TRACER` regardless, and `repo.read_trace(...)` returns `[]`
  with no error anywhere. Adding `tracer=GraphTracer(repo)` fixes it immediately.
- **Scope — read this before acting on it.** This is **not** the (false) claim that REST-driven runs
  write no trace events: `app.py:540-543` builds the production executor **with**
  `tracer=GraphTracer(repo)`, `POST /workflow-runs` has accepted `trace: true` since `670474a`, and
  `GET /workflow-runs/{runId}/trace` reads the events back black-box. The trap is confined to a
  hand-rolled in-process harness that wires its own executor — the shape `test_workflow_live.py`'s
  `_build_live_stack` gets right and an ad-hoc regression script easily gets wrong. Any doc bullet
  must say so explicitly, or it will be read as the false general claim.
- **Proposed change:** one bullet in `falkor-chat/docs/SERVER.md` §1.7, alongside the structurally
  identical `_default_clock` bullet already there ("a test that injects `Services(clock=…)` silently
  fails to control `StepRun.startedAt`") — same failure shape, same section, and worth phrasing to
  match it. `executor.py:447`'s docstring covers the `run["trace"]` half only.

### K-003 — Tool permissions decision  ⚪ DEFERRED (2026-06-05)
- **Status:** ⚪ deferred — user chose to keep `tools` unconstrained for now.
- **Decision:** No `tools` key; the agent continues to inherit all tools, preserving flexibility to spawn subagents and fetch docs mid-task. The focused-set restriction (`Read, Edit, Write, Bash, Grep, Glob`) was considered and declined.
- **Revisit if:** the agent's broad tool access causes surprise or unwanted actions in practice.

## Parking lot / ideas
- State explicitly that the agent does **not** auto-commit (the harness rule is "commit only when asked") — avoids surprise commits given the "commit-sized increments" language.
- Add a one-liner that coverage % is a guide, not a goal — pin behaviors, don't chase numbers.
- Note on flaky tests: quarantine + diagnose root cause rather than re-run until green.
- Optional enrichment: a brief nod to advanced test techniques where they fit — table-driven/parameterized tests for boundary sweeps, property-based testing for invariants, and mutation testing as a *coverage-quality* check (does the suite actually catch injected faults?). Low priority; the prompt is deliberately lean, so only add if it earns its keep.
