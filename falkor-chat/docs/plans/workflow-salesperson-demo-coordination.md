# Salesperson-demo capabilities — Design Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-052…(TBD) (M6, proposed)

Coordinating the **design phase** (requirements → gated plans, no implementation in this unit) for
four sibling, stakeholder-confirmed `Ready for design` requirements documents, all proven inside
one combined "salesperson"-style demo agent (one orchestrating `agent` step, many tools — mirrors
falkor-chat's existing `salesperson`/`triage` pattern):

- `falkor-chat/docs/requirements/workflow-cart-and-totals.md`
- `falkor-chat/docs/requirements/workflow-catalog-lookup.md`
- `falkor-chat/docs/requirements/workflow-durable-profile.md`
- `falkor-chat/docs/requirements/workflow-nl-query-generation.md`

**Why one unit, not four independent handoffs.** All four are proven inside the same demo agent.
A design decision made for one — especially `workflow-cart-and-totals.md` FR-8's
step-type-vs-tool fork for deterministic computation — has to hold consistently across all four
(e.g. if catalog-lookup or durable-profile end up needing their own deterministic/tool-dispatch
step, they should use whatever pattern FR-8 settles on, not invent a second one).

A fifth sibling, `workflow-composition.md`, is deliberately held at `Interviewing` by the
stakeholder and is explicitly not required by or blocking any of these four — no action on it in
this coordination.

## Context

- **Design forks left open, by document** (from the brief):
  - `workflow-cart-and-totals.md` FR-8 — deterministic computation (price × quantity) as a new
    step-type vs. a tool. Likely sets the pattern for the others.
  - `workflow-catalog-lookup.md` — mechanism for the fixed-shape (exact-name/category/price-range)
    lookup.
  - `workflow-durable-profile.md` — workspace-scoped durable-write mechanism (deliberately not
    touching `identity`; that write-path question is out of scope by the stakeholder's own
    decision — see the doc's own "Problem & current state").
  - `workflow-nl-query-generation.md` — the query-generation technology itself, **plus** two
    specialist contributions beyond `architect`, both explicitly part of this capability's own
    acceptance bar (not deferred): `data-scientist` (golden-set metric + passing threshold, FR-4)
    and `security-expert` (adversarial test-case set proving the mechanism is structurally
    non-mutating, FR-3/FR-3a).
- **`workflow-cart-and-totals.md`/`workflow-durable-profile.md` also introduce new durable,
  workspace-scoped graph state** (cart/order, profile) that didn't exist before — genuine graph
  data-modeling work (new labels, indexes/constraints, workspace-scoped write shape), so
  `graph-dba` is in this coordination's design track too, alongside `architect`, even though the
  brief didn't name it explicitly — same shape as `document-ingestion`'s U1/U2/U3 split
  (`docs/plans/document-ingestion-coordination.md`), which is the closest precedent in this repo
  for a multi-specialist design phase.
- **Cross-reference checked, and corrected:** `workflow-nl-query-generation.md`'s own "Related
  work" section and 2026-08-22 decision-log entry describe `docs/requirements/document-ingestion.md`
  as "Ready for design, active `teco` coordination in flight." **That is now stale** — the
  ingestion coordination closed 2026-08-25 (`docs/plans/document-ingestion-coordination.md`,
  `docs/requirements/document-ingestion.md`, and `docs/plans/document-ingestion.md` are all
  `Status: archived`, M5 delivered per `docs/HISTORY.md`). The cross-reference itself is still
  valid and, if anything, stronger now — ingestion's extracted-entity schema is a **shipped,
  stable** second-schema candidate for AC-2, not a moving target. Flagged to `architect`/
  `data-scientist` as current fact; the stale "in flight" wording in
  `workflow-nl-query-generation.md` is a documentation-drift nit for `tico` to fix opportunistically,
  not blocking this coordination.
- **Sibling in-flight work, not part of this coordination:** a separate, currently-`active` `teco`
  coordination (`docs/plans/assemble-messages-alternation-coordination.md`, K-048) has uncommitted
  changes to `falkor-chat/server/falkorchat/executor.py` and
  `falkor-chat/server/tests/test_executor_agent.py` right now. Unrelated topic (message-assembly
  alternation, not business entities/workflow steps) — flagged so `architect` reads `executor.py`'s
  actual on-disk state (which already differs from the last commit) rather than being surprised by
  an unfamiliar diff, not because it changes this coordination's scope.
- **CPG freshness** (checked 2026-08-27, per `skills/cpg-analysis/references/freshness.md`):
  `cpg_falkorchat`, built `2026-08-26T22:27:22Z`, scratch-copy build (no `sourceCommit`,
  `sourcePath` a `/tmp/.../cpg-src/falkor-chat-server` scratch copy whose real counterpart is
  `falkor-chat/server`) — **stale**: one commit (`da10d57`, K-049) has touched `falkor-chat/server`
  since build, **and** the K-048 in-flight work above has uncommitted changes to `executor.py` on
  top of that. Flagged to `architect`/`graph-dba`: read `executor.py`/`tools.py` directly for any
  structural claim about the current step-type/tool-dispatch machinery; don't lean on
  `cpg-analysis` for it here.
- **Backlog/milestone:** M5 closed 2026-08-25; no milestone open yet (`docs/BACKLOG.md`). Next
  free backlog id: **K-052**. Next milestone slot: **M6** (proposed). Left to `architect` to
  actually propose the K-item(s)/M6 framing in the `BACKLOG.md` diff that ships with U1 (mirroring
  the document-ingestion/K-050/M5 precedent) — `teco` verifies the diff, doesn't dictate its shape.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `ae8b24f0595f327cb` | delivered | 4 plans: `docs/plans/workflow-{cart-and-totals,catalog-lookup,durable-profile,nl-query-generation}.md` | `analyst` → — | 305k tok / 48 tools |
| U2 | `graph-dba` | `a65bb2f47ea7a86b4` | delivered | `docs/plans/workflow-cart-and-totals-graph.md`, `docs/plans/workflow-durable-profile-graph.md` | `analyst` → — | 206k tok / 53 tools |
| U3 | `data-scientist` | `a277477d79ce069c6` | delivered | `docs/plans/workflow-nl-query-generation-ml.md` | `analyst` → — | 134k tok / 17 tools |
| U9 | `analyst` | `aefab24e1845b5deb` | delivered | 4 review docs, `docs/reviews/workflow-{cart-and-totals,catalog-lookup,durable-profile,nl-query-generation}.md` | plan gate → catalog-lookup **approve**, cart-and-totals **approve w/ suggestions**, durable-profile **needs changes** (1 BLOCKER), nl-query-generation **approve w/ suggestions** | 192k tok / 31 tools |
| U10 | `architect` (resumed) | `ae8b24f0595f327cb` | accepted | fix MAJOR+MINOR, `workflow-cart-and-totals.md` (v2) | `analyst` (U12) → — | 432k tok / 18 tools |
| U11 | `graph-dba` (resumed) | `a65bb2f47ea7a86b4` | accepted | fix BLOCKER (`coalesce`), `workflow-durable-profile-graph.md` (v2) | `analyst` (U12) → — | 249k tok / 9 tools |
| U12 | `analyst` (resumed) | `aefab24e1845b5deb` | accepted | re-gate `workflow-{cart-and-totals,durable-profile}.md` | plan re-gate → cart-and-totals **approve**, durable-profile **approve w/ suggestions** | 221k tok / 6 tools |

## Design phase closed 2026-08-27 — final verdicts

| Document | Plan | Companion note(s) | Verdict |
|---|---|---|---|
| `workflow-catalog-lookup.md` | v1 | — | **approve** |
| `workflow-cart-and-totals.md` | v2 | `workflow-cart-and-totals-graph.md` v2 | **approve** |
| `workflow-durable-profile.md` | v1 | `workflow-durable-profile-graph.md` v2 | **approve with suggestions** (1 MINOR: `docs/BACKLOG.md` wording — fixed by `teco` directly) |
| `workflow-nl-query-generation.md` | v1.1 | `workflow-nl-query-generation-ml.md` v1, `docs/reviews/workflow-nl-query-generation-security.md` (approve, Pass 2) | **approve with suggestions** (1 MINOR: `docs/BACKLOG.md` wording — fixed by `teco` directly) |

Both remaining MINORs (stale `docs/BACKLOG.md` wording, not a design defect) are closed —
`docs/BACKLOG.md`'s "Active"/K-054/K-055 sections were corrected directly by `teco` after the
re-gate confirmed they were the only open items. No open BLOCKER/MAJOR anywhere in the set.

`docs/BACKLOG.md` M6 section reflects the closed design phase and that K-052 (catalog lookup) is
next up for implementation — not yet dispatched; out of this coordination's original scope
("coordinate design work"). This coordination doc stays `active` pending a future implementation
dispatch, rather than closing to `archived`, since M6 itself is still open.

## Implementation phase (dispatched 2026-08-27)

**Scope:** build K-052..K-055 per their gated plans, in strict dependency order — all four bump
the same `SALESPERSON_DEF["version"]` + edit the same `scripts/seed_salesperson.sh` (a shared-file
axis every plan itself calls out), so capabilities are **serialized**, not parallelized:
K-052 (v1, scaffold) → K-053 (v2) → K-054 (v3) → K-055 (v4). Within a capability, each plan's own
step table is split into dependent clusters per the step-table sizing rule (>3 steps / >5 files),
dispatched as checkpoints to the **same** `coder` agent (resumed via `SendMessage`, not
re-briefed cold) unless its recorded cost crosses the large-context threshold, in which case the
next cluster goes to a fresh agent with a state-recovery brief. Every capability gets two gates
after its implementation lands: `analyst` (diff-scoped static re-gate — reusing the same reviewer,
`aefab24e1845b5deb`, that gated all four plans) and `qa-engineer` (live acceptance pass driving the
running server — state-machine/version-bump/guard logic needs to actually run, not just be read).
K-055 additionally gets `data-scientist` (golden-set harness verification against their own
methodology note) and `security-expert` (live re-run of the Groups A-E adversarial set against the
real `query_graph_data` tool, not just the static check already done at design time). A final
combined `qa-engineer` pass proves all four capabilities live together in one `salesperson@v4`
conversation, per M6's own closing condition.

**CPG:** `cpg_falkorchat` re-checked 2026-08-27 — still built `2026-08-26T22:27:22Z`, still stale
(2 commits since: `da10d57` K-049, `f30c378` K-048). Not relevant here regardless — every plan's
own §"CPG" note already assessed this as new-code / non-impact-analysis work; flagged to
`analyst`/`qa-engineer` so neither leans on it for a structural claim about `executor.py`/
`tools.py`, same posture as the design phase.

### Implementation ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U13 | `coder` | `ae60aa821d81e40d7` | delivered | K-052 cluster 1: `bootstrap_schema.sh` (Product DDL), `docs/QUERIES.md` §15, `scripts/test_queries.sh` (343/343 green, incl. 22 new §15 assertions, `PROFILE`-verified index-scans) | `analyst`/`qa-engineer` (U16/U17, batched) → — | 193k tok / 64 tools |
| U14 | `coder` (resume `ae60aa821d81e40d7`) | `ae60aa821d81e40d7` | delivered | K-052 cluster 2: `repository.py`/`services.py`/`tools.py` (lookup/filter), registry choice: extended `build_builtin_registry` (no `app.py` seam for a second builder) — 1806 passed/4 deselected, +21 net-new tests, mutation-tested | `analyst`/`qa-engineer` (batched w/ U13,U15) → — | 307k tok / 84 tools |
| U15 | `coder` (fresh — U13/U14's agent now over the large-context threshold) | `a497f8167ced0c75f` | delivered | K-052 cluster 3: `seed_catalog.sh`, `verify_catalog.sh`, `proof_defs.py` (`SALESPERSON_DEF` v1), `seed_salesperson.sh`, `verify_salesperson.sh`, `AGENTS.md`, `test_salesperson_scaffold.py` (4 tests: publish validity, republish-no-op, `ctx.endConversation` safety regression + guard-sanity companion). **Live-proved against real LM Studio** (`qwen/qwen3-4b-2507`) on a throwaway `ws:live-salesperson`: AC-1/AC-3/AC-4 all correct, run stayed parked across 4 real turns. Mutation-tested (unconditional guard → 3/4 new tests failed as expected). 1810/4 pytest, 343/343 `test_queries.sh`. | `analyst`/`qa-engineer` (U16/U17) → — | 273k tok / 133 tools |
| U15-finding | — | — | — | **Real bug, live-discovered, out of U15's own scope**: `filter_products`' category match is exact/case-sensitive; a real LLM lowercased "audio" against seeded `"Audio"`, silently returning zero rows. Routed to a fresh `coder` fix, U15b, before the K-052 gates (cheaper to fix once than have both `analyst` and `qa-engineer` independently rediscover it). | — | — |
| U15b | `coder` (fresh) | `a3d25046b87549518` | delivered | fix: `Product.categoryNormalized` (mirrors `nameNormalized` precedent, not runtime `toLower()`), `repository.filter_products` normalizes+compares against it; `GRAPH.PROFILE`-reverified still index-scan, not label-scan; 346/346 `test_queries.sh`, 1811/4 pytest, mutation-tested (3 tests correctly fail reverted) | `analyst`/`qa-engineer` (U16/U17) → — | 154k tok / 78 tools |
| U16 | `analyst` (fresh) | `ab27291958c6b7672` | delivered | code review, K-052 diff (incl. U15b) → **new** `docs/reviews/workflow-catalog-lookup-impl.md` (not a Pass 2 in the bare-slug file — see routing note below), `Extends:`/`Extended by:` pointers added to both review docs | self → **approve with suggestions** (2 MINOR, 1 nit, no blocker) | 193k tok / 82 tools |
| U17 | `qa-engineer` (fresh) | `a2de4430707c3bb70` | accepted | `docs/test-plans/workflow-catalog-lookup.md`, `docs/test-reports/workflow-catalog-lookup-report.md` | self → **PASS WITH DEFECTS** (D-1, see below) | 224k tok / 96 tools |

**U17 delivered — K-052 all 5 ACs hold live; K-052 itself ships as-is (no code blocker).** One
real, reproducible defect (D-1, MAJOR by user impact) found specifically by driving the live
system: within one extended `@mention` conversation, the local model (`qwen/qwen3-4b-2507`)
fabricates catalog facts (invented products, a wrong price) on later tool-calling turns, while the
identical question in a fresh conversation succeeds every time. Ground-truth Cypher + graph-level
`Message` content confirm the repository/service/tool/def layer is correct in every case —
root-caused to live model/conversation-length behavior, not K-052's code. Also found and disclosed:
a session-local LM Studio connectivity gotcha (shared `opencode.json`'s gateway-IP `baseURL`
unreachable from this WSL2 box; worked around via `FALKORCHAT_OPENCODE_CONFIG`, shared file
untouched) — worth a `devops` follow-up (fix the shared config or add a `start_server.sh` preflight
check), not blocking.

**User decision (2026-08-28):** given three of the answer choices explicitly, the user picked
"investigate D-1 first, then proceed" — a bounded diagnostic pass before dispatching K-053 (U18),
not a full program pause and not proceeding uninvestigated. Dispatched as **U36**.

| U36 | `data-scientist` (fresh) | `a3b396e82d988e713` | accepted | `docs/reviews/salesperson-tool-reliability-ml.md` | self → **proceed with K-053+, gated on a cheap mitigation** | 164k tok / 49 tools |

**U36 delivered — root cause resolved, no fix yet.** Two independent live repros with
`trace=True` (bypassing the `@mention` path's default-off trace, no shipped code touched) show the
model's very first LLM turn skips tool invocation entirely on the fabricating turns (zero
`tool_calls`) — not bad arguments, not an overridden correct result. Mechanism: `_assemble_messages`
replays only prior turns' *final text* into each new prompt, never the tool-call scaffolding, so by
turn 3-4 the model's own in-context precedent looks like "this gets answered directly," which
dominates the system prompt's explicit tool-use instruction for a 4B-class model — instruction-vs.
-in-context-precedent robustness, not context-window pressure (collapse point is 6-8 short
messages). `tool_choice: "required"` forcing was tested directly against the exact failing prompt
and **falsified** — didn't force a tool call, and separately triggered a runaway-repetition
failure. Verdict text says both "gated on" a mitigation and "not a hold" — read together as: ship a
cheap fix, but the note's own severity call (§4.4) is what actually matters for sequencing — this
**gets worse, not better, with more tools/turns**, and K-053/K-054's tools are write/mutating
(cart, profile), turning a fabricated *reply* into risk of a fabricated *state-mutation narration*
if the same skip pattern recurs on a write tool. Filed as **K-056** (`docs/BACKLOG.md`), in-progress.

**User instruction (2026-08-28, mid-turn): "stop after the defect is fixed."** Overrides the
default "proceed to K-053" reading of U36's verdict — K-053 (U18) stays `queued`, not dispatched.
Scope for the fix pass: implement the note's targeted mitigation (§4.3 — a tool-use breadcrumb in
the replayed history, the one candidate that targets the diagnosed mechanism directly, not sampling
params) plus its cheap observability signal (§4.1), live-verify against D-1's repro sequence same
as U36's own method, gate with `analyst`, then **stop — no K-053 dispatch this session** regardless
of gate outcome (report back either way).

| U37 | `tdd-engineer` (fresh) | `ab5e9e9d5e1819c78` | delivered | breadcrumb + `Message.toolsUsed` + observability signal (`executor.py`/`repository.py`/`services.py`, +18 tests); **D-1 NOT resolved** — see note below | `analyst` (U38) → — | 304k tok / 174 tools |
| U38 | `analyst` (fresh) | `a6347053812f722e2` | accepted | `docs/reviews/salesperson-tool-reliability-impl.md` | self → **approve with suggestions** (1 MAJOR, 1 MINOR) | 121k tok / 31 tools |
| U39 | `tdd-engineer` (fresh) | `a13492d09a4d2148c` | in-flight | revert breadcrumb tagging only (MAJOR 1), keep `toolsUsed`/observability signal | teco (direct diff read) → — | — |

**U38 delivered.** No blocker, but **1 MAJOR**: the breadcrumb-imitation risk (found live by U37)
is not neutral — traced to `executor.py:1025-1030`, a fabricated reply's own `Message.toolsUsed`
stays empty, so the fake `"[verified via <tool>]"` text is free-text only, and once posted it
becomes part of the *next* turn's replayed history as a self-authored false-verification example —
same instruction-vs-precedent mechanism as the underlying bug, now reinforcing a more deceptive
pattern. Recommends reverting *only* the tagging code path, keeping `toolsUsed`/`link_step_emission`
/`read_thread`/the observability signal (pure audit/logging, no prompt feedback, none of this risk).
1 MINOR: `_looks_fact_bearing`'s bare two-decimal regex flags non-price numbers (e.g. "version
3.14") — low severity, log-only. Everything else: solid (parameterized Cypher, no index needed,
tests pin real behavior not just paths, docs candid about the negative result). Analyst explicitly
left the revert-or-carry-as-known-risk call to "the team," not decided unilaterally — **`teco`
decision: take the revert.** Shipping a code path that actively worsens an already-open,
live-confirmed defect isn't a close call, and the fix is small/contained per the reviewer's own
description. Dispatched as U39 (fresh — U37's agent was over the large-context threshold, and this
follow-up is small/self-contained per the reviewer's precise instructions, doesn't need U37's own
undocumented reasoning).

**U37 delivered — the breadcrumb mitigation is falsified; D-1 stays open.** Live-verified 2/2
independent 9-turn runs against a fresh `ws:tdd-d1-fix`: both collapsed at turn 3, never recovered,
same shape as the original diagnostic (including the identical fabricated $149.99). **New,
more concerning finding**: in both passes the model's own customer-visible reply started
**verbatim imitating the breadcrumb's surface format** (`"... [verified via <tool>]"`) **without
ever calling the tool** — a false-verification claim layered on top of the wrong fact, not just a
wrong fact. The observability signal (`_note_possible_fabrication`, generalized off each step's own
tool-grant set) worked correctly in both passes. `docs/BACKLOG.md`'s K-056 item rewritten in place
by the implementer to record this outcome (not closed). Offline: full suite 1829/4 (was 1811/4,
+18 tests), `test_queries.sh` 346/346, mutation-tested (reverted → 12 new tests failed for the
right reason). Everything left **uncommitted** for `teco` to review/commit per convention.

**Correction to the report's own cross-reference:** the report calls D-1 "already flagged as a
known open epic (K-027)" — checked, and that's stale/wrong: `docs/BACKLOG.md` shows **K-027 closed
2026-08-21**, and it was about tool-call *parsing* precedence/robustness (bare-call vs. JSON-envelope
ambiguity), not live model reliability degrading over conversation length. No existing backlog item
covers D-1's actual failure mode — `grep` for reliability/degrad/hallucin/fabricat/"long
conversation" across `BACKLOG.md` returns nothing. D-1 needs its **own** new K-item, not a citation
to K-027. Left open below as a decision for the user rather than filed unilaterally, since it
affects whether K-053/K-054/K-055 proceed as planned or pause for investigation first.

**U16 routing note, resolved by `teco`:** the analyst filed a *separate* `-impl`-role review
document rather than a `## Pass 2` section in the bare-slug plan review, citing root `AGENTS.md`'s
closed role set (`-impl` is a listed role) and its own reading of collision rule 5 — the
"`reviews/` revises in place" exception applies to a second document of the *same* role, and a
plan review (role `(none)`) vs. an implementation re-gate (role `-impl`) are different roles, not
two passes of one. **Accepted as correct** — a more careful application of the convention than my
own brief assumed; the family-chain example in `AGENTS.md` shows the *typical* one-review-per-topic
case, it doesn't preclude the `-impl` role that's separately listed in the same closed set. No
merge needed; both documents' `Extends:`/`Extended by:` header pointers are already correctly set.

**U16 findings, disposition:** 1 MINOR (`docs/QUERIES.md`'s suite-count header stale at `343/343`
vs. the actual `346/346` post-U15b) — **fixed directly by `teco`** (mechanical, single-line,
already-verified fact, same posture as the design-phase BACKLOG.md wording fixes). 1 MINOR
(`filter_products`'s `categoryNormalized` self-coalesce silently drops any `Product` missing that
property from every call, including the unfiltered one — three-valued `NULL = NULL` logic; never
manifests today since every seed/fixture sets the field, no constraint enforces it) — **not fixed,
logged as a non-blocking follow-up** (needs a repository-layer code judgment call, not a mechanical
edit; genuinely low-risk since nothing in this milestone writes a `Product` without the field). 1
NIT (asymmetric service-vs-repository normalization layering between `lookup_product`/
`filter_products`) — no action, cosmetic only. **No blocker anywhere — K-052 implementation is
gate-clean enough to commit.**

## Session-boundary pause (2026-08-28) — user requested stop-and-commit before a reboot

Per explicit user instruction, stopping here rather than proceeding to U17 (qa-engineer live
acceptance pass). **K-052's code + docs are implemented and analyst-gated (approve with
suggestions, no blocker)** — committing this as one coherent, verified unit. U17 (qa-engineer live
acceptance, AC-1..AC-5) and all of K-053/K-054/K-055 remain queued, unchanged from the ledger
above — resume by dispatching U17 next. **Per explicit user direction, do not resume any
prior-session agent id going forward — every unit from here on dispatches a fresh agent**, even
where an earlier row in this ledger predates that direction and shows a resume.
| U18 | `coder` | — | queued | K-053 cluster 1: `pricing.py`, `repository.py` (cart/order) | — | — |
| U19 | `coder` (resume U18) | — | queued | K-053 cluster 2: `services.py`, `tools.py` (5 cart/order tools) | — | — |
| U20 | `coder` (resume U18) | — | queued | K-053 cluster 3: `proof_defs.py` (v2 + `ORDER_FULFILLMENT_DEF`), seed/verify scripts, `QUERIES.md`/`test_queries.sh`, `AGENTS.md` | — | — |
| U21 | `analyst` (resume) | — | queued | code review, K-053 diff | — | — |
| U22 | `qa-engineer` (resume) | — | queued | live acceptance, K-053 (cart/order/fulfillment) | — | — |
| U23 | `coder` | — | queued | K-054 cluster 1: `repository.py`, `services.py` (profile) | — | — |
| U24 | `coder` (resume U23) | — | queued | K-054 cluster 2: `tools.py`, `proof_defs.py` (v3), seed/verify scripts, `QUERIES.md`/`test_queries.sh` | — | — |
| U25 | `analyst` (resume) | — | queued | code review, K-054 diff | — | — |
| U26 | `qa-engineer` (resume) | — | queued | live acceptance, K-054 (profile persistence) | — | — |
| U27 | `coder` | — | queued | K-055 cluster 1: `querygen.py` (DSL + unit tests incl. reviewer's escape fixtures) | — | — |
| U28 | `coder` (resume U27) | — | queued | K-055 cluster 2: `repository.py` (`run_readonly_query`), `services.py` (`run_structured_query`), `tools.py` (`QueryGraphDataTool`) | — | — |
| U29 | `coder` (resume U27) | — | queued | K-055 cluster 3: golden-set harness + fresh corpus, per `data-scientist`'s note | — | — |
| U30 | `coder` (resume U27) | — | queued | K-055 cluster 4: `proof_defs.py` (v4), seed/verify scripts | — | — |
| U31 | `analyst` (resume) | — | queued | code review, K-055 diff | — | — |
| U32 | `data-scientist` (resume `a277477d79ce069c6`) | — | queued | golden-set harness verification, thresholds report | — | — |
| U33 | `security-expert` (resume `ae4185d22350610f7`) | — | queued | live Groups A-E re-run against real `query_graph_data` | — | — |
| U34 | `qa-engineer` (resume) | — | queued | live acceptance, K-055 (AC-2/AC-5) | — | — |
| U35 | `qa-engineer` (resume) | — | queued | combined e2e pass, all four capabilities in one `salesperson@v4` conversation | — | — |

**Notes from U13:** one deliberate, documented deviation from the plan's illustrative
`filter_products` Cypher — a live-verified `Node By Label Scan` regression with the plan's
`$param IS NULL OR prop = $param` null-guard shape (same quirk already documented for
`list_matches`, `claude/graph-dba/falkordb-quirks.md`); fixed with a `coalesce($minPrice, -1.0)`/
`coalesce($maxPrice, 1e9)` sentinel shape instead, `PROFILE`-verified to keep every filter
combination on an index scan. Explicitly the implementer's call per the plan's own "exact param
names... are the implementer's to finalize" — reversible, not a scope change. Canonical Cypher now
lives in `docs/QUERIES.md` §15. **Unrelated pre-existing drift flagged, not acted on** (out of
this coordination's scope): `ws:qa-tico-workflows-manual`'s `triage@v1` snapshot diverges from
`reference` (2 config differences); `ws:eval` has no `triage@v1`/`access-request@v1` materialized
at all. Follow-up for whoever owns those workspaces.

| U4 | `security-expert` | `ae4185d22350610f7` | delivered | `docs/reviews/workflow-nl-query-generation-security.md` | — (is itself the security gate) → **approve w/ suggestions** | 139k tok / 31 tools |
| U6 | `architect` (resumed) | `ae8b24f0595f327cb` | delivered | fix 2 MAJORs + 2 minors in `workflow-nl-query-generation.md` (v1.1) | `security-expert` (U8) → — | 366k tok / 16 tools |
| U7 | `graph-dba` (resumed) | `a65bb2f47ea7a86b4` | accepted | `sku`→`productId` rename, `workflow-cart-and-totals-graph.md` (v2) | (verified by teco) | 225k tok / 33 tools |
| U8 | `security-expert` (resumed) | `ae4185d22350610f7` | accepted | Pass 2 confirmation, `docs/reviews/workflow-nl-query-generation-security.md` | security re-gate → **approve** | 164k tok / 7 tools |
| U5 | `teco` | — | accepted | applied M6/K-052..K-055 diff to `docs/BACKLOG.md` | (self-verified: reviewed diff before applying) | — |

**Planned next (not yet dispatched):**
- `analyst` plan gate over the full, reconciled set (4 plans + 2 graph notes + 1 ml note) —
  4 separate review docs, one per topic-slug family (`docs/reviews/workflow-{cart-and-totals,
  catalog-lookup,durable-profile,nl-query-generation}.md`). **Dispatched as U9.**
- Fix/re-gate cycles as needed.

- **U8 (`security-expert` Pass 2) delivered: verdict upgraded to plain `approve`.** All 4 Pass 1
  findings independently re-verified as fixed (regexes/Pydantic constraints run for real against
  the actual escape strings, not just read). Reconciliation is now fully closed — both items in
  the "Reconciliation needed" section above are resolved. Ready for the `analyst` plan gate.

## Reconciliation needed before the analyst gate

1. **`sku` → `productId` rename in `graph-dba`'s two notes.** Confirmed: `workflow-catalog-lookup.md`
   (architect, U1) defines the catalog item's key property as `productId` (`(:Product {productId,
   name, nameNormalized, category, price})`), not `sku`. `graph-dba`'s `workflow-cart-and-totals-graph.md`
   used `sku` as a placeholder pending exactly this confirmation (self-flagged in its own report).
   **Resolved (U7, `graph-dba` resumed).** Renamed throughout `workflow-cart-and-totals-graph.md`
   (schema/DDL/Cypher/prose; `workflow-durable-profile-graph.md` had no `sku` references). Bumped
   to `Version: 2`, dated revision note, per the revise-in-place convention. Re-verified the one
   structurally-distinct write shape (the 2-property composite `UNIQUE` constraint + its
   `MERGE`-and-increment) live against a fresh disposable probe graph — `OPERATIONAL`, identical
   result to the original; other occurrences were straight identifier substitutions, explicitly
   noted as not individually re-run rather than silently assumed.
2. **2 MAJOR security findings on the nl-query-generation DSL**, from `security-expert`'s
   `docs/reviews/workflow-nl-query-generation-security.md` (verdict: approve w/ suggestions, no
   blocker — but these are load-bearing on the FR-3 structural-safety property, worth closing
   before the `analyst` gate rather than carrying them as suggestions):
   - MAJOR 1 — `returns`/`order_by` are compound expressions unlike the flat-token
     `label`/`property`/`var` allowlist checks; no specified decomposition-before-validation step,
     and an unanchored regex (`re.match` vs `re.fullmatch`) there would silently accept a crafted
     injection string. Independently confirmed exploitable in principle (a completed, syntactically
     valid injection was constructed and shown caught only by the independent `GRAPH.RO_QUERY`
     Layer-2 backstop, not Layer 1).
   - MAJOR 2 — `QueryMatch.var`'s regex (`^[a-z][a-z0-9]{0,7}$`) is stated only as a code comment,
     not a specified/enforced Pydantic constraint, and has no allowlist backstop at all.
   - 2 minors: pin an explicit conservative timeout on `run_readonly_query`; harden the
     "only `querygen.compile` calls this" invariant with `extra="forbid"` Pydantic models + a
     nominal `CompiledQuery` type rather than a grep-based test alone.
   - Routing to `architect` (U1's owner) to fix in `workflow-nl-query-generation.md`, then a quick
     confirmation pass by `security-expert` (resumed) before the `analyst` gate.

## Analyst plan gate (U9) — findings routed

- **`workflow-catalog-lookup.md` — approve.** No action needed.
- **`workflow-cart-and-totals.md` — approve with suggestions.** 1 MAJOR: `ensure_customer`/
  `ensure_cart` calls are never explicitly assigned to a listed service method, even though
  `graph-dba`'s Cypher requires them before a brand-new customer's first `add_to_cart` succeeds —
  risk of a silent no-op on the demo's most basic path. 1 MINOR: `place_order`'s in-flight
  price-change race needs a scoping clarification (already self-flagged in the plan). Routing both
  to `architect` (U1/U6's owner) to fix — real correctness issue, worth closing now even though the
  verdict doesn't strictly require it.
- **`workflow-durable-profile.md` — needs changes (1 BLOCKER).** `graph-dba`'s `write_profile`
  Cypher does an unconditional `SET` on both `name`/`deliveryAddress` fields, copying a precedent
  (`write_model_overrides`) whose NULL-means-clear semantics don't fit `SaveProfileTool`'s
  genuinely-optional-args calling convention — a partial update (e.g. "just update the address")
  would silently null out the customer's previously-stored name, exactly the scenario AC-2
  requires to work. Fix supplied by `analyst`: `SET c.name = coalesce($name, c.name), ...`. Also 1
  MINOR: `docs/BACKLOG.md`'s K-054 entry references a "`Profile` schema" that was deliberately
  never built (properties live on the shared `Customer` node). Routing the BLOCKER to `graph-dba`
  (U2/U7's owner) — mandatory fix + re-gate. The BACKLOG.md wording fix is `teco`'s own to make
  (mechanical, already-verified fact).
- **`workflow-nl-query-generation.md` — approve with suggestions.** 1 MINOR: `docs/BACKLOG.md`'s
  K-055 entry still reads the security review as "in progress," but Pass 2 (same date) shows it
  approved with zero open findings. `teco`'s own fix to make.
   - **U6 (`architect`, resumed) delivered.** `workflow-nl-query-generation.md` bumped to
     `Version: 1.1`, revised in place (correct per the collision rule — not yet approved/gated).
     Both MAJORs closed with fully-anchored (`.fullmatch()`) `_PROJECTION_RE`/`_AGGREGATE_RE`
     validators routed through the existing label/property allowlist (not a second,
     independently-written check, per the reviewer's own instruction); `QueryMatch.var` is now an
     enforced `Field(pattern=...)` plus a `compile()`-side re-check that a var reference resolves to
     the declared match variable. Both minors closed: explicit `DEFAULT_QUERY_TIMEOUT_MS = 2500`
     with the batch-granular caveat documented; `extra="forbid"` on all 3 DSL models + a nominal
     frozen `CompiledQuery` type replacing the bare tuple, upgrading the "only `compile()` calls
     this" invariant from grep-only to type-checker-visible. No design change to the two-layer
     architecture itself — gaps closed in Layer 1's specification only. Also propagated the
     reviewer's adversarial test groups (A-E) and Layer-2 re-verification into the plan's own
     §3.2/§4/§5/§6. Ready for `security-expert`'s confirmation pass.

## Notes from delivered units

- **U3 (`workflow-nl-query-generation-ml.md`) delivered.** Two-layer design: Layer 1 (FR-4/AC-4
  gate) is execution-accuracy against the mechanism's *raw structured result* (pre-NL-rendering),
  exact match after canonicalization — deliberately Spider-style, not LLM-as-judge, since ground
  truth here is a discrete fact rather than an open-ended faithfulness question. Thresholds:
  overall ≥ 85% (Wilson 95% CI reported alongside), second-schema (document-ingestion) subset
  gated separately at ≥ 75% so AC-2 can't hide behind a pooled average, abstention/not-found
  false-answer-rate ≤ 10% (asymmetric on purpose). **Flags a real risk to `architect`'s mechanism
  choice**: Layer 1 requires the mechanism to expose its raw structured result, not just a
  rendered sentence — needs checking once U1 lands. If the mechanism is LLM-generated Cypher,
  recommends a harness-level write-clause scan as defense-in-depth for the *eval's own runs*
  only — explicitly **not** a substitute for `security-expert`'s FR-3a adversarial set.
  **Verdict on AC-2's second-schema candidate**: yes for the document-ingestion *schema* (a real
  generalization test — graph traversal, free-text-predicate edges, conflicting-facts handling),
  **no for its existing `ws:acme` data** (K-050's QA fixture is too thin/skewed — ~10-12 entities,
  almost all `Organization`/`Other`, no `Person`/`Location`/`Product`/`Event`/`Concept` instances).
  Recommends reusing the schema but ingesting a fresh, purpose-built 10-15 document corpus into a
  dedicated workspace, with golden answers verified against actual post-extraction graph content
  (not source text, given extraction's known non-determinism). No blocking fork hit. Not yet
  gated — holding the `analyst` plan gate until U1 (architect) lands, so the gate covers the full,
  reconciled set in one pass rather than re-reviewing this note twice.

- **U1 (4 architect plans) delivered.** K-052 (catalog-lookup) ships the shared `salesperson`
  `WorkflowDef` scaffold (bumped `v1→v4` across the four plans rather than four separate defs, per
  `docs/DESIGN.md` §4's create-only-properties rule) and is sequenced first; K-053/K-054 each
  extend it and depend on their respective `graph-dba` note; K-055 depends on both
  `data-scientist`'s note (delivered, U3) and `security-expert`'s review (not yet dispatched).
  **FR-8 decision: a plain Python function called directly from tool bodies, not a new engine step
  type** — reasoned from reading `executor.py`/`guards.py`/`services.py` directly: a `Tool.run()`
  body is exactly as LLM-free as a typed step handler (no extra guarantee a step type would add), a
  new step type would be unreachable given the fixed one-`agent`-step topology all four docs share,
  and what AC-9 actually rules out is an *avoidable second LLM call* layered on top of arithmetic —
  exactly what `salesperson/cart.py`'s existing `_extract_quantity_from_flavor` does today. Flags
  the dormant `kind:'process'` + `tool`-step-type combination in `services.STEP_TYPES`/
  `executor._execute_step` (`NotImplementedError`) as the right future extension point if a
  zero-LLM-call process def is ever needed — not built now. Propagates as a *principle*
  (a tool's body does exact work directly, no second LLM hop) rather than a literal mechanism reuse,
  since catalog-lookup/durable-profile are already exact by construction.
  **nl-query-generation mechanism**: a constrained query-builder DSL (structured model output
  populates typed filter/match/request fields, compiled to a fixed handful of clause templates,
  every value bound-parameterized, every label/property checked against a closed allowlist before
  splicing) — mutation is *inexpressible* in the compiler, not filtered. Second, independent,
  engine-enforced backstop: execution goes exclusively through `graph.ro_query(...)`, live-verified
  in `claude/graph-dba/falkordb-quirks.md` to refuse write queries on this build. This is exactly
  the concrete mechanism `security-expert`'s next review needs to engage with.
  **BACKLOG.md**: architect drafted the M6/K-052..K-055 diff but did not apply it, reading
  `docs/BACKLOG.md` as `teco`-owned and outside its plan's write scope. Diff reviewed by `teco` —
  correct shape (mirrors the K-042/M4, K-050/M5 precedents), one K-item per capability (not one
  umbrella item, since these are four independently-interviewed/confirmed requirements docs with
  their own FR/AC sets — a reasonable, explained call, not a default). Resuming the architect agent
  to apply it directly (it authored the content and holds full context on the dependency chain
  between items; a Write/Edit to `docs/BACKLOG.md` is a normal in-scope action for it, unlike for
  `teco`).
  **Correction (U5):** the architect declined — its own operating rules scope its Write/Edit to
  `docs/plans/` only, and it correctly treated my resume message as *not* authorization to expand
  its own permission scope (it did, helpfully, confirm the diff's 4 plan paths were byte-accurate
  first). Right call on its part — re-reading root `AGENTS.md`'s doc-kind ownership table,
  `BACKLOG.md` has no by-kind specialist owner there; its own header names `teco` as owner, and
  it's explicitly a `teco`-curated forward-looking living document (Documentation curation
  section), unlike `plans/`/`reviews/`/etc. which are specialist-owned. Applied the diff myself —
  content was already fully authored and reasoned by `architect`, verified by `teco` before
  applying, not fresh judgment exercised on `teco`'s part.
