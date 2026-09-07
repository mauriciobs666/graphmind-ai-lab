# Kaizen — Improvement Plan: coder

> Forward-looking backlog for the `coder` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Validate the architect→coder handoff end-to-end and confirm the coder can execute an architect plan without re-investigating. |
| K-006 | 2026-09-07 | medium | 🔵 | Four verified `falkor-chat` behaviour facts distilled out of `kaizen_team` have no home yet — each needs one addition to a component doc outside cobb's write remit. |

### K-006 — Four distilled `falkor-chat` facts need a home in that component's docs
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** `agent-maintenance` §5 distillation of `coder`'s `kaizen_team` capture
  (2026-08-27..29, unit U10) verified four facts that are true, durable, and **not** published
  anywhere a reader would find them. All four are facts about `falkor-chat`, so they belong in that
  component's docs where every agent sees them — not in `coder`'s prompt and not in an agent
  knowledge base. All four land outside `cobb`'s hook-enforced write remit.
- **Proposed change:** route to `teco` to assign each to the doc's owner. One row per fact:

  | Source entry | Fact (verified by re-derivation) | Target |
  |---|---|---|
  | `29b6274a…` | A Cypher-level regression in a `repository.py` write is invisible to `test_services.py`: its `FakeRepo` re-implements the same semantics in plain Python (`tests/test_services.py:606-615` says so in its own comment — "Faithful to `repository.py` §17's `coalesce()`-per-field semantics"). Mutation-testing a repository write must target `test_repository.py`; a green `test_services.py` is not evidence the Cypher is load-bearing. | `docs/SERVER.md` §1.7 (testing hazards) |
  | `55364b9a…` | `tests/test_salesperson_scaffold.py` hand-mirrors the def's tool list in a module-level `_SCHEMAS` dict, and its `StubRegistry.schema()` is a bare `_SCHEMAS[name]` lookup — so adding a tool to `proof_defs.SALESPERSON_DEF["config"]["tools"]` without adding its schema fails the offline scaffold tests at runtime with `KeyError`, not at collection. Confirmed live: the def's 11 tools and `_SCHEMAS` are kept in lockstep by hand today. | `docs/SERVER.md` §1.7 (testing hazards) |
  | `9050f193…` | This box's single LM Studio instance JIT-loads one model at a time, so concurrent embed (embedding model) + extract (generation model) background jobs from *different* documents thrash the swap and fail with `ProviderCallError … {"error":"Model is unloaded."}` — the document reaches a terminal `failed` status with zero entities. Any live batch driver must settle one item to a terminal status before starting the next. Already documented **at the point of use** (`scripts/seed_nlq_eval_corpus.py:342-353`, a 16-line comment); what is missing is the generalized constraint where the next author of a live driver would look. | `docs/SERVER.md` §1.7 (the QA/acceptance gotchas half) |
  | `c1f2a8b4…` | `executor._drive_loop`'s OUTCOME B keys on `config.waitsForHuman` alone and never on `step.type` — so a step of **any** type parks when it declares the flag (a shipped precedent already relies on this: `SALESPERSON_DEF`'s `agent`-typed assistant step). The undocumented half is the validation asymmetry: `services._validate_def_spec` *requires* the flag only for `WAITING_STEP_TYPES` (`human`/`wait`) and neither requires nor forbids it elsewhere, so a `decision`/`agent` step that needs to park but omits it **publishes clean** and fails only at runtime, self-looping to a `maxSteps` budget failure. `services.py:85` documents the flag's primacy; nothing documents the publish-clean/fail-at-runtime gap. | `docs/DESIGN.md`, the rule-8 / `waitsForHuman` paragraph (~line 344) |

- **Notes:** source entries cleared from `kaizen_team` after this disposition was logged
  (2026-09-07) — this item, not the raw graph nodes, is the durable record. Do **not** re-open a
  per-entry item if a later pass reads these ids fresh: check this table first.

### K-002 — End-to-end handoff validation
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The architect→coder contract is designed but unproven; the coder should be able to pick up an architect plan cold (isolated context) and build it.
- **Proposed change:** Run a real feature through architect→coder; capture what the plan was missing; feed back into both prompts.
- **Notes:** The transport is settled — the implementer receives the plan as a document path (`<component>/docs/plans/<slug>.md`) and reads the file itself — and the *contract* is proven, but by the wrong agent: `architect` K-002 closed on a teco K-001 run where **`tdd-engineer`** executed an architect plan cold with no re-investigation. What remains is coder-specific. The `coder` has since run as the implementer half repeatedly in falkor-chat (K-022 Landing 2 M-2, K-024 U2/U4/U4b), reading plan docs by path and reporting blockers rather than guessing (the zero-transition `IndexError` was *"not fixed (out of unit scope); reported to teco"* — correct scope discipline). That is strong circumstantial evidence and not a review: those run reports were never read against the plan. **Close on one deliberate read of a completed architect→coder run.**

## Parking lot / ideas
- **Judged and kept, do not re-litigate (2026-08-24, C6 lint).** Three restatements will read as class-7 duplicates to a future dedup sweep; all are keeps under finding 5 ("needed twice", not "said twice"):
  - **"Don't claim what you didn't run" vs. step 5's report procedure** — prohibition vs. the mechanics of an honest report (show output; report `passed`/`skipped`/`deselected`). Same pair `qa-engineer` certified at C5.
  - **"Ask before destructive or environment-changing actions" vs. step 2's bootstrap ask** — two decision points, each carrying its own subagent carve-out. The carve-out duplication is an `agent-maintenance` §4 check-3 **certification requirement**, not style.
  - **"Minimal blast radius" vs. "Don't silently exceed scope"** — scope of edits vs. reporting obligation. (The *third* member of that trio, `:11`, is a genuine contradiction — see K-003.)
- **A mutant must be proven to change behavior before its survival is read as a coverage gap.**
  From distilled entry `a1b2c3d4-e5f6-…` (2026-08-29, re-verified on CPython 3.12.3): stripping
  `^`/`$` from a regex whose call site is `re.fullmatch()` is a **no-op mutant** — `fullmatch`
  enforces whole-string matching regardless of anchors in the pattern text, so the intended
  "unanchored regex" regression only appears once the call is also switched to `.match()`. The
  general rule is worth stating somewhere the implementers read; there is no owning knowledge base
  for testing technique today (`skills/python-web-quirks/` is scoped to web/async + pytest
  import-timing, and `claude/qa-engineer/qa-testing-techniques.md` is scoped to black-box QA
  mechanics), which is why the entry was discarded rather than promoted. Revisit if a
  mutation-testing knowledge base ever earns its own file.
- A "definition of done" checklist (suite green, behavior covered, no scope creep, honest run report) the coder self-checks before reporting completion.
- Consider whether the coder should delegate the test-writing step to `tdd-engineer` when strict TDD is required, rather than doing it itself.
