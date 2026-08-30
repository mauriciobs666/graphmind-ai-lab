# Structured catalog/reference lookup for workflows — Implementation Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-052 (M6) · **Extends:** `docs/reviews/workflow-catalog-lookup.md`

**Routing note.** The coordinator's brief for this unit asked for a `## Pass 2` section appended
to the design-phase plan review at the bare slug. Per root `AGENTS.md`'s own closed role set
(`(none)` · `-coordination` · `-ml` · `-graph` · `-rca` · `-impl` · `-report`) and this agent's own
operating rule ("a review of an implementation... takes the `-impl` role suffix on the same slug
... the bare slug is the review of the plan, and implementation findings never grow the plan
review"), a plan review (role `(none)`) and an implementation re-gate (role `-impl`) are two
different roles, not two passes of the same role — the `reviews/`-document collision exception
("revises in place... regardless of the selector's answer") applies to a second document of the
*same* kind/topic/role, which this isn't. Filed here instead, as a new `-impl` document
`Extend`-ing the plan review (which is unaffected — `Status:` stays `active`, header now carries
`Extended by:` pointing here). Flagging this deviation explicitly for the coordinator to reconcile
if a single combined file was actually intended.

## Scope & verdict

Diff-scoped re-gate of the **uncommitted** K-052 implementation (structured catalog/reference
lookup for workflows) against the already-approved plan `docs/plans/workflow-catalog-lookup.md`
and its Pass-1 gate `docs/reviews/workflow-catalog-lookup.md`. Reviewed every file the brief named:
`scripts/bootstrap_schema.sh` (`Product` DDL), `docs/QUERIES.md` §15, `scripts/test_queries.sh`
§15, `server/falkorchat/{repository,services,tools,proof_defs}.py`, the four new scripts
(`seed_catalog.sh`/`verify_catalog.sh`/`seed_salesperson.sh`/`verify_salesperson.sh`),
`server/tests/{test_repository,test_services,test_tools,test_salesperson_scaffold}.py`, and
`falkor-chat/AGENTS.md`'s "Key scripts" table additions. Did not re-litigate the plan's own design
choices (Pass 1 already approved those) — this pass is about what only exists in the diff: bugs,
plan deviations, missed edges, and whether the two live-discovered/documented Cypher deviations
(`QUERIES.md` §15's `coalesce`-sentinel shape and the `categoryNormalized` fix) actually hold up.

Ran the real suites myself rather than trusting the implementers' reported numbers: full offline
`pytest` (`server/`, `.venv/bin/python -m pytest -q`) — **1811 passed, 4 deselected** — and
`./scripts/test_queries.sh` — **346/346 passed**, including the new `▶ §15` block (26 assertions,
3 of them `assert_index_scan` pairs). Both destructive to shared dev-instance state as the brief
warned; restored the standard baseline afterward (`bootstrap_schema.sh acme` →
`seed_demo.sh acme` → `seed_workflows.sh acme` → `seed_catalog.sh acme` →
`seed_salesperson.sh acme`) and confirmed with `verify_workflows.sh acme` /
`verify_catalog.sh` / `verify_salesperson.sh acme` — all three report `OK`. One self-inflicted
cleanup note: independently re-verifying the mutation-testing claims below (`prod1`..`prod5`
fixture rows briefly written by a scratch-copy `pytest` run against the *same* live FalkorDB
instance, since `test_repository.py`'s `wf_repo` fixture has no isolation from the shared
`reference` graph) left 5 stray `Product` nodes in `reference`; caught via
`seed_catalog.sh`'s own "20 Product node(s), 15 declared" sanity print, `DETACH DELETE`d before
final restoration — `verify_catalog.sh` now reports the correct 15.

**Verdict: approve with suggestions.** No blocker. Every AC is genuinely implemented and covered;
both documented Cypher deviations are independently re-derived as correct for the data this system
actually seeds; the AC-6 fence and the scaffold's safety-critical guard property are verified live,
not just read. Two minor findings (a latent NULL-handling edge in `filter_products`, and a stale
suite-count header) and one nit, none blocking.

**CPG:** considered, not relevant — this is new-code review for `falkor-chat/server`, not an
impact-analysis question; the brief additionally confirmed `cpg_falkorchat` is stale as of this
session. All structural claims below were verified by reading the actual source and, where
possible, executing it.

## Findings

### MINOR — `filter_products`'s `categoryNormalized` self-coalesce silently drops any `Product` missing that property, even from the "list everything" unfiltered call

`repository.py`'s `WHERE p.categoryNormalized = coalesce($categoryNormalized, p.categoryNormalized)`
is tautologically true only when `p.categoryNormalized` is non-`NULL` — FalkorDB's Cypher, like
standard SQL/Cypher three-valued logic, evaluates `NULL = NULL` to `NULL` (not `true`), so a
`Product` node with no `categoryNormalized` property is filtered out of **every** call, including
an all-omitted "list the whole catalog" call, which the design (`QUERIES.md` §15.2, `tools.py`'s
`FilterProductsTool` docstring) explicitly intends to always succeed at listing everything up to
the limit. Live-verified: created a throwaway `Product` in `reference` with no `categoryNormalized`
(and no `category`) and ran the exact §15.2 Cypher with all three filters `NULL` — the probe row
was silently absent from the result set (cleaned up immediately after). Today this never manifests
— `seed_catalog.sh` and every test fixture always set `categoryNormalized` — but there is no
`NOT NULL`-shaped enforcement (no constraint requires it, nothing validates it at write time), so
a future direct write that forgets the property (a manual fixture, a follow-up script) would have
its rows silently vanish from every catalog query, not just category-filtered ones — a surprising
failure mode for a property that isn't even being filtered on in that case. Suggest either an
explicit `p.categoryNormalized IS NULL OR ...` fallback branch, or — cheaper — a one-line comment
at the DDL/repository call site flagging that `categoryNormalized` is a de facto required field
despite carrying no constraint, so a future writer doesn't reintroduce the gap blind.

### MINOR — `docs/QUERIES.md`'s suite-green header claims `343/343`, but the code as it stands (including the `categoryNormalized` fix layered on afterward) produces `346/346`

The header line (`docs/QUERIES.md:3-4`) reads "**343/343, 2026-08-27**... 320/320 immediately
before the K-052 §15 product-catalog gate." 320 + the 26 new §15 assertions = 346, which is what an
independent run actually reports (confirmed above) — not 343. Cross-checked against the
coordination ledger (`docs/plans/workflow-salesperson-demo-coordination.md`'s implementation-phase
table): 343/343 was the real, correct count at the U13/U14/U15 checkpoint, **before** U15b's
`categoryNormalized` live-discovered fix added 3 more `test_queries.sh` assertions (the
case-insensitive-category checks) on top — U15b's own row even states "346/346 `test_queries.sh`"
— but the `QUERIES.md` header comment was never bumped to match the code as it now sits in the
working tree. Low-stakes on its own, but this header is the line `falkor-chat/AGENTS.md`/root
`AGENTS.md` treat as evidence a query-suite gate actually ran clean before a schema/query change is
committed; a stale count here is exactly the kind of small drift that erodes that evidentiary
value for the next reader. Suggest bumping the header to `346/346` (and folding the corrected math
into the existing "343/343... 320/320" chain) before this lands.

### NIT — `name`/`category` normalization happens at two different layers

`services.lookup_product` normalizes the caller's `name` via `extraction.normalize_name` before
calling `repository.lookup_product` (which takes an already-normalized `name_normalized` kwarg),
but `services.filter_products` passes `category` through unnormalized, and
`repository.filter_products` does the `normalize_name` call itself. Both are correct (verified by
the case-insensitivity tests and the live `test_queries.sh` §15.2 block) and neither layer
double-normalizes, but the asymmetry (service-layer vs. repository-layer normalization for two
sibling lookups added in the same change) is a minor consistency smell for a future reader
comparing the two methods side by side. Not blocking — no behavior is wrong.

## What's solid

- **Every plan-required piece landed and matches the plan's design intent**: `Product` DDL
  (index-then-constraint order verified — `bootstrap_schema.sh`'s new block runs the four
  `CREATE INDEX`s before the `UNIQUE` constraint), `lookup_product`/`filter_products` at all three
  layers (`repository`/`services`/`tools`), the `salesperson@v1` scaffold, and all four new
  scripts, each mirroring their named precedent (`seed_workflows.sh`/`verify_workflows.sh`) closely
  enough to be maintainable by the same playbook.
- **Both documented Cypher deviations are correct, not just plausible.** Independently re-derived
  the `coalesce`-sentinel shape (`-1.0`/`1e9` bounds are safely outside every seeded product's
  price, and `p.price` is confirmed via live `GRAPH.PROFILE` to anchor every filter combination,
  including the fully-omitted case, on `Node By Index Scan`) and the `categoryNormalized` fix
  (case/whitespace-insensitive category matching, verified against 5 case variants in
  `test_repository.py` and against a live-run `GRAPH.PROFILE` in `test_queries.sh`, still
  index-anchored after the fix — the NULL-coalesce edge above is the one gap this deviation
  doesn't cover, not a flaw in the deviation's own reasoning).
- **AC-6 (ungranted-tool fence) is intact.** Confirmed exactly one process-wide `ToolRegistry` is
  constructed (`app.py:389`, inside the single `WORKFLOW_ENABLED` branch) and that
  `executor._handle_tool_call`'s AC-6 rejection (`executor.py:835`) gates on the *node's*
  `config.tools`/`granted_set`, never on registry membership — registering `lookup_product_fact`/
  `filter_products` into the same shared registry as `triage`'s tools cannot let `triage` (or any
  other def) call them, since `triage`'s own `config.tools` never lists them.
- **The scaffold's single safety-critical property is verified live, not just read.** Reproduced
  the coordination ledger's mutation-testing claim independently, via a zero-working-tree-touch
  scratch copy (`cp -r` + in-place mutation only in the scratch tree, `PYTHONPATH`-shadowed against
  the real editable install): flipping `SALESPERSON_DEF`'s `ctx.endConversation` guard to
  unconditional (`"guard": ""`) makes
  `test_ordinary_multi_turn_conversation_never_fires_the_end_transition` fail with the run
  finishing `done` instead of `waiting` on its very first turn — exactly the failure mode the test
  exists to catch — and 3 of the 4 scaffold tests fail overall, matching the ledger's "3/4 new
  tests failed as expected" claim exactly. Same technique against `repository.filter_products`'s
  `categoryNormalized` fix (reverted to the pre-fix `p.category = $category` exact-match shape in
  a scratch copy) reproduces exactly 3 failing tests, matching U15b's own claim.
- **Test coverage is real, not name-matching.** Every test file reviewed makes assertions on
  actual returned values/call arguments (`repo.calls == [...]`, exact row-set equality, abstention
  shapes), not just "did not raise." `test_salesperson_scaffold.py`'s guard-safety test drives 10
  real `resume_workflow_run` turns through the real executor and asserts `ctx.endConversation` stays
  unset every turn, with an explicit `llm.calls == 22` check proving the loop genuinely ran rather
  than trivially never-firing.
- **Abstention shapes are internally consistent** and match `GraphragRetrieveTool`'s existing
  idiom: `filter_products` → `{"items": [], "finding": "..."}"`, same shape as
  `graphrag_retrieve`'s `{"seeds": [], "finding": "no relevant context found"}`;
  `lookup_product_fact` → `{"found": false}`, the right shape for a single-item (not list) lookup.
  Plan §3.5's claim holds.
- **Suites are genuinely green**, independently re-run: 1811 passed / 4 deselected (offline
  pytest), 346/346 (`test_queries.sh`, corrected count — see the finding above).

## Open questions

- Confirm with the coordinator whether this unit was actually meant to land as a `## Pass 2`
  section inside the bare-slug plan review (as briefed) rather than as this separate `-impl`
  document — see the routing note at the top. If a single combined file is genuinely wanted despite
  the `-impl` role-suffix rule, that's a one-time merge, not a re-review.
