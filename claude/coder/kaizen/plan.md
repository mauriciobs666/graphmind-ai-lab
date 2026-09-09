# Kaizen — Improvement Plan: coder

> Forward-looking backlog for the `coder` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07 (U12)

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Validate the architect→coder handoff end-to-end and confirm the coder can execute an architect plan without re-investigating. |
| K-006 | 2026-09-07 | medium | 🔵 | Six verified `falkor-chat` behaviour facts distilled out of `kaizen_team` have no home yet — each needs one addition to a component doc outside cobb's write remit. |

### K-006 — Six distilled `falkor-chat` facts need a home in that component's docs
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** `agent-maintenance` §5 distillation of `coder`'s `kaizen_team` capture
  (2026-08-27..29 unit U10, then 2026-09-03 unit U12) verified six facts that are true, durable,
  and **not** published anywhere a reader would find them. All six are facts about `falkor-chat`,
  so they belong in that component's docs where every agent sees them — not in `coder`'s prompt and
  not in an agent knowledge base. All six land outside `cobb`'s hook-enforced write remit. The
  shared shape of the last two: the mechanism *is* already written down at the point of use, and
  the point of use is a place nobody consults before trusting the gate.
- **Proposed change:** route to `teco` to assign each to the doc's owner. One row per fact:

  | Source entry | Fact (verified by re-derivation) | Target |
  |---|---|---|
  | `29b6274a…` | A Cypher-level regression in a `repository.py` write is invisible to `test_services.py`: its `FakeRepo` re-implements the same semantics in plain Python (`tests/test_services.py:606-615` says so in its own comment — "Faithful to `repository.py` §17's `coalesce()`-per-field semantics"). Mutation-testing a repository write must target `test_repository.py`; a green `test_services.py` is not evidence the Cypher is load-bearing. | `docs/SERVER.md` §1.7 (testing hazards) |
  | `55364b9a…` | `tests/test_salesperson_scaffold.py` hand-mirrors the def's tool list in a module-level `_SCHEMAS` dict, and its `StubRegistry.schema()` is a bare `_SCHEMAS[name]` lookup — so adding a tool to `proof_defs.SALESPERSON_DEF["config"]["tools"]` without adding its schema fails the offline scaffold tests at runtime with `KeyError`, not at collection. Confirmed live: the def's 11 tools and `_SCHEMAS` are kept in lockstep by hand today. | `docs/SERVER.md` §1.7 (testing hazards) |
  | `9050f193…` | This box's single LM Studio instance JIT-loads one model at a time, so concurrent embed (embedding model) + extract (generation model) background jobs from *different* documents thrash the swap and fail with `ProviderCallError … {"error":"Model is unloaded."}` — the document reaches a terminal `failed` status with zero entities. Any live batch driver must settle one item to a terminal status before starting the next. Already documented **at the point of use** (`scripts/seed_nlq_eval_corpus.py:342-353`, a 16-line comment); what is missing is the generalized constraint where the next author of a live driver would look. | `docs/SERVER.md` §1.7 (the QA/acceptance gotchas half) |
  | `c1f2a8b4…` | `executor._drive_loop`'s OUTCOME B keys on `config.waitsForHuman` alone and never on `step.type` — so a step of **any** type parks when it declares the flag (a shipped precedent already relies on this: `SALESPERSON_DEF`'s `agent`-typed assistant step). The undocumented half is the validation asymmetry: `services._validate_def_spec` *requires* the flag only for `WAITING_STEP_TYPES` (`human`/`wait`) and neither requires nor forbids it elsewhere, so a `decision`/`agent` step that needs to park but omits it **publishes clean** and fails only at runtime, self-looping to a `maxSteps` budget failure. `services.py:85` documents the flag's primacy; nothing documents the publish-clean/fail-at-runtime gap. | `docs/DESIGN.md`, the rule-8 / `waitsForHuman` paragraph (~line 344) |
  | `62bc71d6…` (U12) | §1.7's first bullet already says `conftest.wf_repo` wipes `reference` at fixture **setup** only, and stops at the consequence for workflow *defs* ("re-run `seed_workflows.sh`"). The catalog half is not there and is not fixed by that remedy: a `reference`-touching test also leaves its fixture `Product` rows behind, and `scripts/seed_catalog.sh` `MERGE`s by `productId` (`:124`), so the documented post-pytest re-seed **cannot** remove a stray `widget-…` — `scripts/verify_catalog.sh` then fails against its exact `EXPECTED_COUNT=15` (`:33`). The general rule to state: a test that writes to `reference` needs its own **yield-fixture teardown** wipe, not just `wf_repo`. Already solved in code and documented at the point of use — `tests/test_storefront.py`'s `catalog_repo` fixture (`:839-851`) is exactly that teardown and its docstring carries the whole chain — but a docstring on one fixture is not where the next author of a `reference`-touching test looks. | `docs/SERVER.md` §1.7, extending the existing first bullet rather than adding a second |
  | `16ab10b7…` (U12) | `scripts/test_queries.sh` never executes the code under test: it re-types each `QUERIES.md` query as its own shell constant (27 of them; zero invocations of `python`, `pytest`, `falkorchat` or `.venv` anywhere in the script) and runs it through `redis-cli`. So it is a live-dialect check of a **transcription**, not a code-vs-doc fidelity gate — change a query in `repository.py` and `QUERIES.md` together and the script keeps asserting the old text and still reports 408/408. The lesson is written out in full at `docs/QUERIES.md` §15.1's `productId` note ("a transcription gate goes green on a wrong transcription… checked against the code, never against each other"), but it sits inside one query's sub-note, while `AGENTS.md:200` states the standing rule "the full suite (`./scripts/test_queries.sh`) must pass before any schema or **query** change is committed" — which is what makes a reader treat a green run as fidelity evidence. One bullet, in the same "a green gate is not evidence of X" family as §1.7's existing `ruff check` and green-exit-code bullets. | `docs/SERVER.md` §1.7 |

- **Notes:** source entries cleared from `kaizen_team` after this disposition was logged
  (2026-09-07) — this item, not the raw graph nodes, is the durable record. Do **not** re-open a
  per-entry item if a later pass reads these ids fresh: check this table first. Rows 5 and 6 were
  folded in here rather than opened as their own item, since the ask is identical in kind and
  target: one addition each to `docs/SERVER.md` §1.7, routed through `teco` to that doc's owner.

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
  general rule is worth stating somewhere the implementers read. **A home now exists** — U11
  (2026-09-07) promoted a sibling mutation technique (mutating a class-level Cypher constant via a
  pytest `-p` plugin, no source edit) into `claude/analyst/review-techniques.md`, whose stated scope
  is verification technique; that file, not a new mutation-testing knowledge base, is where this
  lesson should land. U11 did not move it, to stay inside its own eight-entry scope. The homes ruled
  out at U10 still are: `skills/python-web-quirks/` (web/async + pytest import-timing) and
  `claude/qa-engineer/qa-testing-techniques.md` (black-box QA mechanics). **What actually remains
  is smaller than this item says (U39, 2026-09-09):** the *general* rule — a surviving mutant is
  not evidence of a coverage gap until the mutant is shown to change behaviour — has stood in
  `claude/tdd-engineer/tdd-engineer.md`'s mutation bullet all along, as the equivalent-by-construction
  case, and U39 extended that same bullet again. Only the `re.fullmatch`/anchor **instance** is
  unhomed, and it is an instance, not a rule.
- A "definition of done" checklist (suite green, behavior covered, no scope creep, honest run report) the coder self-checks before reporting completion.
- Consider whether the coder should delegate the test-writing step to `tdd-engineer` when strict TDD is required, rather than doing it itself.
