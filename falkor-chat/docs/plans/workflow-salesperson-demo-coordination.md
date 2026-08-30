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
| U39 | `tdd-engineer` (fresh) | `a13492d09a4d2148c` | accepted | revert breadcrumb tagging only (MAJOR 1), keep `toolsUsed`/observability signal | teco (direct diff read + independent suite re-run) → clean | 123k tok / 60 tools |

## Session resume (2026-08-29) — Ministral re-point phase dispatched

New session. Reconciled against `git log`/`git status` first: HEAD (`1f3d7e7`) matches the ledger
exactly through U41/U42, working tree clean for this coordination (only an unrelated, untracked,
`Interviewing`-status `docs/requirements/salesperson-ui.md` from `tico`, out of scope). Per the
pending decision left open at the prior session's end, asked the user via `AskUserQuestion`
whether to proceed with the Ministral re-point now, hold, or something else. **User chose: proceed
now.**

Plan (per U40/U41's own §9.5 recommendation): (1) re-point `salesperson@v2` at
`mistralai/ministral-3-3b` — (2) live-reverify K-052/K-053's ACs hold on the new model — (3) resume
K-054 (U23/U24, already queued) → K-055. Dispatching (1) now as **U43**; (2)/(3) wait on U43.

**Mechanism chosen** (read `docs/SERVER.md` §1.8 directly before dispatching, not guessed): the
per-step `config.model` field is the correct, narrowly-scoped override — it resolves through
`ModelGateway` at the "consumer's own requested choice" precedence rung (below the workspace hard
cap, above the role default), checked resolvable at publish time (FR-9). This re-points
`salesperson` only, leaving `triage`/`access-request`'s shared `agent`/`step` role defaults (still
`qwen/qwen3-4b-2507`) untouched — deliberately narrower than editing `config/models.json`'s global
defaults or adding a `ws:acme`-scoped `WorkspaceConfig` override, either of which would also flip
`triage` onto an unevaluated-for-that-role model as an unplanned side effect.

| U43 | `coder` | `af2b84ceb50bb2fac` | accepted | re-point `SALESPERSON_DEF`'s `assistant` step at `mistralai/ministral-3-3b` via `config.model` (create-only, needed its own version — `v2.1`), `proof_defs.py`/`seed_salesperson.sh`/`verify_salesperson.sh`/`AGENTS.md`/`test_salesperson_scaffold.py` (+2 tests, mutation-tested) | `qa-engineer` (U44, live re-run) → — | 241k tok / 110 tools |

**U43 delivered and committed.** Verified independently by `teco`, not on the delegate's word alone:
diff read directly, matches the report exactly (create-only reasoning for `config.model`, the
`v2.1` labeling choice and its rationale, `config/models.json` deliberately left untouched since
the validated Ministral eval ran with empty sampling params — adding an unvalidated
`temperature: 0`-style entry would diverge from the validated condition). Independently re-ran the
full offline suite (**1920 passed/4 deselected**, matches — the +1 over U42's 1919 baseline is the
new regression-pin test) and `./scripts/test_queries.sh` (**387/387**, matches); reseeded
`reference` per the documented sequence and re-verified `verify_workflows.sh`/`verify_catalog.sh`/
`verify_salesperson.sh acme` all `OK`, `salesperson@v2.1` in sync with correct 2-step/1-transition
topology, `triage`/`access-request` confirmed untouched (still `qwen/qwen3-4b-2507`). Confirmed the
delegate's `kaizen_team` entry for the LM-Studio `opencode.json` baseURL gotcha is actually present.
**Note:** my own reseed hit a malformed `bootstrap_schema.sh` invocation on my part (a stray
`ws:EMBEDDING_DIM=1024` graph from a bad arg) — caught via `GRAPH.LIST` before verifying sync,
`GRAPH.DELETE`d immediately; no data at risk (empty scratch graph), not the delegate's error.
Committed (`03a3c8c`).

| U44 | `qa-engineer` (fresh — standing direction) | `a433bbf8bbe3826bf` | accepted | `docs/test-plans/workflow-{catalog-lookup2,cart-and-totals2}.md`, `docs/test-reports/workflow-{catalog-lookup2,cart-and-totals2}-report.md` | self → **K-052 PASS WITH DEFECTS, K-053 PASS WITH DEFECTS** (both qwen-era D-1 mechanisms confirmed resolved) | 218k tok / 150 tools |

**U44 delivered — both qwen-era K-056 D-1 findings confirmed resolved on `ministral-3-3b`, verified
independently by `teco`, not on the delegate's word alone.** Read both new reports in full: the
exact repro sequences from the original qwen-era reports were replayed verbatim, every claim
cross-checked against `Message.toolsUsed` and ground-truth Cypher, not trusted from reply text. **New
non-blocking findings** (both correctly judged non-gating, consistent with the brief): K-052 — a
cosmetic `$`/`€` currency-symbol inconsistency (MINOR, cosmetic only, no AC impact). K-053 — (D-1)
a cart-mutation reply-completeness gap on `remove_from_cart`/`add_to_cart` that extends the
original report's own D-2 finding and confirms it's model-independent, exactly as predicted (write
always correct, only the reply summary sometimes incomplete); (D-2, informational, explicitly not a
gate per the brief) Ministral's own known duplicate-instruction defect reproduced once live on the
real `@mention` path, exactly matching the ml-note's own characterization (honestly-grounded,
self-disclosing) — a second independent confirmation, not new information, and correctly not
weighed against the verdict. Independently re-ran both the offline suite (**1920 passed/4
deselected**, matches) and `./scripts/test_queries.sh` (**387/387**, matches); reseeded and
re-verified `ws:acme`/`reference` in sync (`verify_workflows.sh`/`verify_catalog.sh`/
`verify_salesperson.sh acme` all `OK`, `salesperson@v2.1` correct topology, no stray graphs this
time). Confirmed the delegate's `kaizen_team` entry (the `POST .../input` body-nesting gotcha) is
present. `Extended by:`/parent-pointer edits on both original test plans read correctly, following
the `llm-provider-config`/`llm-provider-config2` precedent as instructed.

**Out-of-scope observation, flagged not acted on:** `git status` at this point also shows
`.claude/settings.json` modified (repo root, outside `falkor-chat/` and outside anything asked of
any delegate in this coordination) — `defaultMode` changed from `acceptEdits` to
`bypassPermissions`, plus a new `ask:` allowlist for specific destructive ops (`FLUSHALL`,
`GRAPH.DELETE`, `docker system prune`, etc.) and `docs/BACKLOG.md` edits. Not committed, not
reverted, not otherwise acted on by `teco` — outside this coordination's scope and outside `teco`'s
own write grant, and a permission-mode change is exactly the class of thing that needs the actual
user's own eyes, not a coordinator's guess at intent. Surfaced in the session report.

**K-054 unblocked.** Both capabilities hold on `mistralai/ministral-3-3b` with no regression and no
blocking defect — per the already-agreed plan, resuming K-054 (U23/U24, already queued/speced)
next. **User confirmed (2026-08-29, asked via `AskUserQuestion`):** proceed to K-054 now; the
settings.json anomaly is the user's own to handle, not blocking, `teco` leaves it untouched.

**U23 brief note — carries context neither `workflow-durable-profile.md` nor its graph note has**
(both predate K-056/the Ministral re-point): `SALESPERSON_DEF["version"]` is currently `"v2.1"`
(K-056's re-point, just shipped, `03a3c8c`), not the plain `"v2"` both plans were written against
— bumping to `"v3"` per the plan is still correct (next in sequence), but **the `assistant` step's
`config.model: "lmstudio/mistralai/ministral-3-3b"` must be carried forward into `v3`'s config**,
same as every other cumulative property (`config.tools`/`systemPrompt`) already is per the
module's own documented convention — each version publishes a full, fresh `config`, so omitting
`model` on `v3` would silently revert the assistant to the `step`-role default
(`qwen/qwen3-4b-2507`), undoing U43/U44's whole point. Flagged explicitly in U24's brief (the unit
that touches `proof_defs.py`), since U23 itself doesn't touch that file.

**U44 dispatched.** Re-run K-052's (`docs/test-plans/workflow-catalog-lookup.md`) and K-053's
(`docs/test-plans/workflow-cart-and-totals.md`) full AC lists live against the now-repointed
`salesperson@v2.1` (`mistralai/ministral-3-3b`), specifically to confirm the prior qwen-era D-1
findings (skip-and-fabricate in both reports; the write-tool escalation on `remove_from_cart` found
just before this pilot decision) do not reproduce, and to watch for (not gate on) Ministral's own
known duplicate-instruction defect. Naming convention: pointed at the `llm-provider-config.md` /
`llm-provider-config2.md` (+ matching `-report`) precedent for a second QA cycle against a changed
underlying dependency, same test-plan content reused where unchanged, new `-report` per collision
rule 5 (ordinal on slug) since the earlier reports are already executed-against — left the exact
filenames to the reviewer's own judgment per that precedent.

**U43 brief given:** goal — re-point the shared `salesperson` demo agent's LLM from
`qwen/qwen3-4b-2507` to `mistralai/ministral-3-3b`, scoped to this def only. Read
`docs/SERVER.md` §1.8 (the model-resolution seam) and `server/falkorchat/modelconfig.py` directly
for the exact mechanics — don't take this brief's paraphrase as the spec. Concretely: add
`"model": "lmstudio/mistralai/ministral-3-3b"` to `SALESPERSON_DEF`'s `assistant` step `config` in
`server/falkorchat/proof_defs.py`; add a `config/models.json` `"models"` entry for that ref only if
you determine one is needed (mirror the existing `qwen/qwen3-4b-2507` entry's `temperature: 0`
precedent if an equivalent setting is warranted for Ministral — check
`docs/reviews/salesperson-tool-reliability-ml.md` §8/§9 first for anything the eval passes already
established about how this model was queried, but note that harness used its own scratch script,
not necessarily the production `ModelGateway` path, so don't assume it mirrors production config
without checking). **Verify directly, don't assume, whether `config.model` is a create-only
property on an already-materialized version** (same class as `config.tools`/`systemPrompt`, per
this file's own module docstring) — if so, this needs its own version bump on `SALESPERSON_DEF`
(currently `v2`) to take effect; decide the exact version label yourself, reasoning about how to
keep it legible against the existing `v1`→`v2`→`v3`→`v4` per-capability convention (a labeling
choice, reversible, your call to make and document — not a fork worth stopping for). Update
`seed_salesperson.sh`/`verify_salesperson.sh` and `falkor-chat/AGENTS.md`'s script-table entry for
the new version if the version label changes. Live-verify: republish/re-materialize against a
throwaway workspace (or `ws:acme` if you confirm it's safe per the shared-instance conventions
already documented in this coordination doc), confirm LM Studio actually has
`mistralai/ministral-3-3b` loaded and the publish-time model-resolvability check
(`_check_models_resolvable`, FR-9) accepts it, then restore `ws:acme`/`reference` to their prior
in-sync state and re-verify with `verify_salesperson.sh acme`. Mutation-test any new/changed test.
Do not touch `triage`/`access-request` or `config/models.json`'s global `defaults` block — this is
scoped to `salesperson` alone. Do not edit `docs/BACKLOG.md`/`docs/HISTORY.md` (teco-owned; will
fold in the residual-Ministral-risk disclosure the ml-note's §9.5 recommends after verifying this
unit). **CPG:** not relevant — this is a small, localized constant/config edit, not a
call-graph/impact-analysis question. Standard subagent-awareness clause applies (stop-and-ask only
for a genuine high-stakes fork; this unit's decisions are all routine/reversible per above).
Deliverable: the code+config+doc diff, by path, plus what you verified live.

**Gate note:** deliberately skipping a separate `analyst` static-review gate for this specific
unit — the very next unit (`qa-engineer`'s live re-run of K-052/K-053's existing test plans against
the new model) is a stronger, execution-level check for exactly this kind of change (does the live
system still behave correctly on the new model) than a static read would add. Explicitly the
justified skip per the standing "review is the default, not ceremony for its own sake" guardrail,
not an oversight.

**U39 delivered and committed (`15a9749`).** Verified directly before committing, not just on the
implementer's word: `grep`-confirmed no `[verified via` interpolation remains in `executor.py`
(only in comments/docstrings recording the revert); independently re-ran the full offline suite
(**1828 passed/4 deselected**, matches) and `./scripts/test_queries.sh` (**346/346**, matches);
both wiped `reference`/`ws:acme` per the documented teardown hazard, re-seeded
(`bootstrap_schema.sh` → `seed_demo.sh` → `seed_catalog.sh` → `seed_workflows.sh` →
`seed_salesperson.sh`) and re-verified in sync (`verify_workflows.sh`/`verify_catalog.sh`/
`verify_salesperson.sh`, all OK) before handing back. `Message.toolsUsed`/`link_step_emission`/
`read_thread`/the observability signal all ship as designed; the model-facing breadcrumb text is
fully gone. **K-056 stays open** — D-1 itself (the model skipping tool calls) is unresolved; only
the newly-discovered aggravating side effect from the failed mitigation attempt is closed.

## Session stop (2026-08-28) — per explicit user instruction ("stop after the defect is fixed")

**D-1/K-056 is not fixed** — it remains open, with two candidate mitigations now falsified
(`tool_choice: "required"` forcing, and the replayed-history breadcrumb) and no confirmed
scaffold-level fix. Per the user's own instruction, stopping here rather than continuing to iterate
on a fix or dispatching K-053. **Everything through U39 is committed** — nothing left uncommitted in
the working tree. `docs/BACKLOG.md`'s K-056 item and `docs/HISTORY.md`'s 2026-08-28 entry are the
durable record of exactly what was tried, what worked (the audit property, the observability
signal), what didn't (both mitigations), and what's still open (a controlled eval; the
larger-model-fallback decision) — read those first when this resumes. U18 (K-053, `coder`) and
everything after it in the implementation ledger above stays `queued`, unchanged.

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
## Session resume (2026-08-28/29) — K-056 stays open, user directs proceeding to K-053

New session. Asked the user explicitly whether to (a) proceed to K-053 accepting D-1/K-056 as a
known, disclosed, open risk, (b) pursue the ml note's remaining levers (controlled eval,
larger-model fallback) before touching any write-mutating tool, or (c) something else. **User
chose (a) — proceed to K-053.** Dispatching U18 per the already-queued ledger below.

| U18 | `coder` | `a48905847f09736a2` | delivered | `server/falkorchat/pricing.py` (new), `repository.py` §16 Cart/Order methods, `bootstrap_schema.sh` DDL (added, not originally scoped — mechanical, per graph note's own §7 DDL), 41 new tests | teco (independent re-run) → 1869 passed/4 deselected, matches | 210k tok / 128 tools |
| U19 | `coder` (resume U18) | `a48905847f09736a2` | abandoned | transient platform failure (API timeout, user reports a power outage) mid-run; left one legitimate, incomplete, uncommitted `repository.py` diff (`lookup_product` productId extension + new `lookup_products_by_id`) — not redone, picked up by U19b | — | — |
| U19b | `coder` (fresh, state-recovery brief) | `af09366ad8de5196e` | delivered | `services.py` §16 (add_cart_item/get_cart/remove_cart_item/clear_cart/place_order/get_order_status/advance_order), `tools.py` (5 cart/order tools), `repository.py` `lookup_products_by_id` retained + test fix, 42 new/changed tests | teco (independent re-run + ensure_customer/ensure_cart wiring spot-check) → 1910 passed/4 deselected, matches; MAJOR-finding closure confirmed in code | 242k tok / 163 tools |
| U20 | `coder` (fresh — U19b over large-context threshold) | `af5d1fb344b03b29a` | delivered | `proof_defs.py` (v2 + `ORDER_FULFILLMENT_DEF`), `seed_salesperson.sh`/`verify_salesperson.sh` (two-def), `QUERIES.md` §16, `test_queries.sh` (+41), `AGENTS.md`, `test_order_fulfillment.py` (new, 7), `test_executor_agent.py`/`test_salesperson_scaffold.py` (+2) | teco (independent re-run) → 1919 passed/4 deselected offline (matches), 387/387 `test_queries.sh` (matches), live `verify_workflows`/`verify_catalog`/`verify_salesperson acme` all OK after final restore | 331k tok / 152 tools |
## Session stop (checkpoint, this session) — pausing for a fresh restart, K-053 implementation clusters 1-3 done and committed

All three K-053 implementation clusters (U18, U19/U19b, U20) are delivered, independently
re-verified by `teco`, and committed (`f020f90`, `bcd2dcc`, and this checkpoint commit). **Not yet
done: U21 (`analyst` code review) and U22 (`qa-engineer` live acceptance)** — K-053 is
implementation-complete but **not gated**, same posture K-052 was in at its own equivalent
checkpoint. Next session should dispatch U21 next, unchanged from the ledger below.

**Shared-instance note for whoever resumes.** Mid-session, `teco`'s own verification pytest runs
(which wipe the global `reference` graph at teardown, per the documented hazard) ran concurrently
with U20's own live-verification cycle while U20 was still mid-edit on `proof_defs.py` —
`verify_salesperson.sh acme` briefly reported `salesperson@v1` **diverged** between `reference`
and `ws:acme` (stale pre-bump content republished into a freshly-wiped `reference` while the
workspace snapshot still held the original). This resolved itself without any surgical fix once
`reference` was fully recreated fresh with U20's final, completed `proof_defs.py` (the def moved
to `v2` anyway, obsoleting the divergent `v1` entry) — `verify_workflows.sh`/`verify_catalog.sh`/
`verify_salesperson.sh acme` all report `OK`/in-sync as of this checkpoint. No production data was
at risk (this is `reference`/`ws:acme`, dev/demo state, not anything with a real-user consequence)
but it's a concrete instance of the "one agent owns a shared graph key when suites run
concurrently" hazard the coordination rules warn about — worth remembering next time a `teco`-side
verification run and a delegate's own live-verification cycle could overlap in time.

| U21 | `analyst` (fresh — per the standing "no prior-session agentId resume" direction above) | `a02d8f0b401ac4ac1` | delivered | `docs/reviews/workflow-cart-and-totals-impl.md` | self → **approve with suggestions** (0 BLOCKER/MAJOR, 2 MINOR, 1 nit) | 222k tok / 59 tools |
| U22 | `qa-engineer` (fresh — same standing direction) | `ab9afe8d07d36b74d` | accepted | `docs/test-plans/workflow-cart-and-totals.md`, `docs/test-reports/workflow-cart-and-totals-report.md` | self → **PASS WITH DEFECTS** (D-1 MAJOR, D-2 MINOR) | 242k tok / 104 tools |

**U22 delivered — K-053's own code holds on every graph-state assertion across all 10 ACs**
(brand-new-customer cart write, cross-conversation persistence, deterministic/repeatable totals,
frozen order snapshot surviving a live price mutation, full order-fulfillment lifecycle incl. both
invalid-transition guards) — live, against a fresh `ws:qa-cart-totals`, real LM Studio
(`qwen/qwen3-4b-2507`), real `order-fulfillment@v1` (run-side REST, `Order`-side CAS via direct
`Services.advance_order` calls per U21's flagged no-REST-route gap). Every mutating action
cross-checked against ground-truth Cypher, not trusted from reply text. Final offline regression
(run last, separately, per brief): 1919 passed/4 deselected, 387/387 `test_queries.sh` — both match
the last recorded baseline exactly; `ws:acme` restored and re-verified in sync. Independently
re-verified (teco): grepped the report's own D-1 repro steps against its ground-truth-Cypher
methodology — consistent with the delegate's summary; deliverables present on disk as claimed;
working tree otherwise clean apart from the two new docs + this coordination doc + U21's Extends/
Extended-by edit to `docs/reviews/workflow-cart-and-totals.md`.

**User decision (2026-08-29):** given the D-1 write-tool escalation, the user chose **"pause
K-054, invest in K-056 first"** over proceeding as planned. Per the ml-note's own §5 remaining
scope (items 2-3 — item 1, the generalized observability signal, already shipped in U37): dispatch
`data-scientist` to run a proper controlled eval (repeated live conversations varying turn/tool
count, replacing the n=2 anecdote with an actual rate estimate) and decide, on that data, whether
a scaffold-level fix exists or this is a capability ceiling calling for a larger-model fallback for
this role. Dispatched as **U40**. K-054 (U23/U24) stays `queued`, not dispatched.

| U40 | `data-scientist` (fresh — standing direction) | `ad8bba92133fb4b50` | delivered | `docs/reviews/salesperson-tool-reliability-ml.md` §8 (Pass 2, revised in place) | self → **confirmed capability ceiling for `qwen3-4b-2507`; recommend piloting `ministral-3-3b`, conditional on its own duplicate-instruction defect** | 288k tok / 149 tools |

**U40 delivered — K-056 root cause fully quantified, no longer an anecdote.** Controlled eval
(n=40 conversations/280 turns baseline): skip-and-fabricate is a **near-deterministic collapse at
exactly the 4th user turn** (39/40, 97.5%, CI 87.1-99.6%), 0 recoveries in 121 post-onset turns
(100% persistent), and **100% reproduction rate (15/15) on write-mutating cart conversations** —
this is not a rare edge case, it is what happens on essentially every real conversation reaching
turn 4. Detector (`_note_possible_fabrication`) holds up at scale: 0% false-positive, 98.1% recall,
one confirmed blind spot (bare price-less write confirmations).

**Alternative-model check (expanded mid-run per relayed user steer to include same-size models,
not just larger), queried live from this environment's actual LM Studio instance:**
`openai/gpt-oss-20b` (larger) is directionally clean where it ran (0 skips) but crashes on 75% of
probe conversations on an LM-Studio-side serving-stack bug (harmony-format incompatibility) —
**not usable as-is**, `devops` follow-up, not actionable here. `mistralai/ministral-3-3b`
(same/smaller size class) shows **0/176 skip-and-fabricate instances** (CI 0-2.1%) — directly
contradicts a "bigger fixes it" framing — **but has its own distinct, real defect**: silent
duplicate re-execution of an earlier cart instruction in 3/10 (30%, CI 10.8-60.3%) write-mutating
runs (every tool call real/dispatched, no fabrication — a different failure class the shipped
detector correctly doesn't catch).

**Recommendation:** confirmed capability ceiling for `qwen3-4b-2507` in this role — not a scaffold
defect (both scaffold-level mitigations this lineage produced are now independently falsified).
Pilot `ministral-3-3b` as the same-cost, immediately-available replacement, **conditional** on
closing or explicitly accepting its own duplicate-instruction defect first (a small, scoped
follow-up eval, not a full re-run). Alternatively, proceed on the current model accepting the
near-certain (87-100% CI) fabrication risk, with the observability signal as an after-the-fact
logging backstop only (does not prevent it). Decision point for the user — routed via
`AskUserQuestion`, not decided unilaterally, since it determines what model K-054/K-055 (and
potentially K-052/K-053 retroactively) run on.

Verified by `teco`: deliverable present, `git status` clean apart from this doc + the coordination
doc, all five throwaway eval workspaces confirmed deleted by the delegate, `ws:acme`/`reference`
re-verified in sync per the report's own artifacts section.

**User decision (2026-08-29):** given U40's recommendation, the user chose **"pilot Ministral,
scope its defect first"** — a scoped follow-up eval targeting specifically the duplicate-
instruction pattern (not a full re-run of §8's baseline), before any decision to re-point
`salesperson@v2` at `mistralai/ministral-3-3b`. If that comes back clean/acceptable, the plan is:
re-point the def at `ministral-3-3b`, re-run K-052/K-053 QA live on it, then resume K-054.
Dispatched as **U41**.

| U41 | `data-scientist` (fresh — standing direction) | `a04f15e782ecc57c7` | delivered | `salesperson-tool-reliability-ml.md` §9 (Ministral duplicate-instruction eval) | self → **Go — pilot `ministral-3-3b`, not blocked** | 167k tok / 67 tools (2 mid-run status-line check-ins before the real result) |
| U42 | `data-scientist` (fresh — standing direction) | `a635e0df17c6a9d1c` | delivered | scratch handoff, merged by `teco` into `salesperson-tool-reliability-ml.md` §10 (model landscape survey), scratch file deleted after merge | self → informational, no gate | 109k tok / 19 tools |

**U41 delivered — Ministral's duplicate-instruction defect is real but not blocking.** 32 fresh
live conversations (6 conditions varying instruction similarity/spacing/tool), ground-truth
double-checked via `TraceEvent` + Cypher: 1/32 confirmed duplicate dispatch (pooled with §8.4's
3/10, combined estimate somewhere in the high-single-digits-to-twenties-percent range, wide CIs,
none anywhere near certainty). **Categorically less severe than K-056**: every occurrence is
honestly-grounded (real dispatched call, real state) and self-disclosing (the doubled quantity
is stated plainly in the reply) — unlike K-056's silent, near-certain fabrication. Named an
untested candidate mitigation (a dispatch-time check that a write tool's target appears in the
current turn's own raw text) as a near-term follow-up, not a pre-pilot gate. **Verdict: Go —
proceed with re-pointing `salesperson@v2` at `ministral-3-3b`, per the already-agreed
contingency plan**, with the residual risk disclosed in `docs/BACKLOG.md`/`docs/HISTORY.md`
once the re-point lands. Two of U41's mid-run notifications were stale status lines ("waiting
for the background eval"), not real results — `teco` sent two check-ins before the actual
finding arrived; noted as a real recurring cost of this delegate's polling pattern, not
something to read into the finding's reliability.

**U42 delivered — model landscape survey merged cleanly, no conflict materialized.** Headline:
no published tool-calling benchmark tests K-056's exact mechanism (spontaneous skip-after-N-
turns with only replayed plain text, no tool scaffolding) — closest analogue is a general
multi-turn-degradation paper (Laban et al., ICLR 2026) showing even frontier models lose ~39%
accuracy and never recover from an early misstep once conversations are split across turns,
suggesting this is a general LLM trait, not unique to this stack. Shortlist for a future
live-test, if wanted: `Salesforce/xLAM-2-3b-fc-r` (top priority — best multi-turn BFCL among
small models, purpose-built agentic training, CC-BY-NC-4.0), `ibm-granite/granite-4.1-3b`
(only Apache-2.0 candidate, thinner evidence), `microsoft/phi-4-mini` (training-lineage
diversity), `MadeAgents/Hammer2.1-3b` (lower priority). Deprioritized: Qwen2.5 variants, further
Gemma 3 sizes (family-wide — no native tool-call tokens), older 8B BFCL-leaderboard-toppers
(stale-generation scores). Purely informational — no gate, no action required now.

**Self-correction, logged:** U41/U42 were dispatched in parallel against the same file
(`salesperson-tool-reliability-ml.md`) — a same-file conflict this coordination's own rules
exist to prevent (`teco` error, not either delegate's). Caught before any edit collision
materialized (U41 hadn't started editing yet); U42 was redirected mid-run to a scratch file,
merged in by `teco` after U41's §9 landed, scratch file deleted post-merge. No data lost, but
the dispatch itself should have been serialized from the start.

Merged §10 in by `teco` (mechanical content merge + heading renumber `9`→`10`, `9.1-9.3`→
`10.1-10.3`, no content change) and made one small consistency fix to the top `## Verdict`
block (Ministral's defect status was stale relative to §9's actual "not a blocker" verdict).

**D-1 (MAJOR) — first confirmed K-056 recurrence on a write-mutating tool.** Mid-conversation
(turn 4 of one thread), the model fabricated a "successfully removed" reply for `remove_from_cart`
with **zero tool call and zero graph change**; a retry in a fresh thread worked correctly. This is
the same K-056 mechanism (already open, already accepted as a known risk when the user chose to
proceed to K-053), but it is the **first instance reaching a mutating tool** rather than a
read-only fact — exactly the escalation `data-scientist`'s ml-note (U36) warned proceeding past
K-052 risked, and worse than previously known: a customer can now be told a cart mutation
succeeded when it silently didn't, with no warning either party has stale state. K-054 is another
mutating tool (`save_profile`) directly in the escalation's path. **D-2 (MINOR, new)** —
`add_to_cart` onto a non-empty cart sometimes reports only the new line's price as "Total,"
omitting pre-existing lines in the reply text (checkout correctness unaffected; a follow-up
`view_cart`/`place_order` always shows the true total) — logged, not fixed, low severity.
**Decision point for the user, per the standing pause-on-genuine-decision rule**: this changes the
risk calculus the user's 2026-08-28 "proceed to K-053" call was made under — surfacing before any
K-054 dispatch rather than proceeding on the prior decision's premise.

**U21 delivered — K-053 code review clean, no blocker.** Both prior plan-gate findings (the
`ensure_customer`/`ensure_cart` MAJOR, the price-change-race MINOR) independently confirmed closed
in code, not just claimed — traced through `services.py` directly plus a call-ordering test that
would fail on a regression. Three new findings, all non-blocking: **MINOR** — a checkout where some
(not all) cart lines reference a since-deleted catalog product silently drops the vanished line
with no signal in the tool response, untested (no product-deletion flow exists in this milestone,
so **logged as a non-blocking follow-up, not fixed now** — same posture as K-052's `categoryNormalized`
follow-up); **MINOR** — cluster-3's commit message overstates its test-count delta ("19" vs the
actual 9 new test functions, though the before/after totals it quotes are exact) — **noted for
`teco` to get right in the eventual `HISTORY.md` entry**, no code action; **nit** — `AddToCartTool`
treats an explicit `quantity: 0` the same as omitted (defaults to 1) — cosmetic, no action. The
reviewer deliberately skipped the offline pytest/`test_queries.sh` runs (both wipe `reference`) to
avoid disturbing the live `ws:acme` state it had just re-verified in sync — independently confirmed
`verify_salesperson.sh` in-sync and the new `Customer`/`Cart`/`CartItem`/`Order` constraints
`OPERATIONAL` instead. **Flagged for U22:** `services.advance_order` has no REST route this
milestone (order-fulfillment's `Order.status` lifecycle is a direct Services-layer call, not
reachable via HTTP by an operator) — `qa-engineer` needs to drive that half of the acceptance pass
accordingly, not assume an HTTP path exists.
| U23 | `coder` (fresh) | `a792dfe24a7601443` | accepted | K-054 cluster 1: `repository.py`/`services.py` §17 (profile), `test_repository.py`/`test_services.py` (+12 tests) | `analyst`/`qa-engineer` (U25/U26, batched w/ U24) → — | 121k tok / 45 tools |

**U23 delivered — verified independently by `teco`, not on the delegate's word alone.** Diff read
directly: `repository.upsert_profile` is a faithful transcription of the graph note's v2
`coalesce()`-guarded `SET` (never an unconditional one — the exact BLOCKER the graph note's own
revision closed); `get_profile`/`save_profile` in `services.py` correctly return the
always-populated shape (not a `{"found": false}` abstention), per plan §3.2. Independently re-ran
the full offline suite (**1931 passed/4 deselected**, matches). **Independently re-mutation-tested
myself** (not just trusting the report): reverted `upsert_profile`'s `coalesce()` `SET`s to an
unconditional `SET` on a copied-aside file, re-ran `-k profile` — both AC-2 partial-update tests
failed exactly as the delegate reported, restored, full suite green again. Confirmed the delegate's
`kaizen_team` entry (FakeRepo doesn't catch a repository-Cypher regression, a good methodology note
for future mutation-testing in this suite) is present. **Out-of-scope drift observed, not
touched:** `docs/requirements/cross-initiative-coordination.md` (repo root) was modified during this
unit's run — a separate, unrelated `tico`-authored requirements doc (dated today, well-formed,
clearly a different concurrent interview), not something asked of this delegate and not part of
this coordination; left completely alone.
| U24 | `coder` (resume U23) | `a792dfe24a7601443` | accepted | K-054 cluster 2: `GetProfileTool`/`SaveProfileTool` (`tools.py`), `SALESPERSON_DEF` v3 (`proof_defs.py`), `seed_salesperson.sh`/`verify_salesperson.sh`/`AGENTS.md`, `docs/QUERIES.md` §17, `test_queries.sh` (+20), `test_tools.py`/`test_salesperson_scaffold.py` | `analyst`/`qa-engineer` (U25/U26, batched w/ U23) → — | 268k tok / 137 tools |

**U24 delivered — verified independently by `teco`, not on the delegate's word alone.** Diff read
directly: the `config.model` carry-forward into `v3` is exactly as briefed, documented inline;
`config.tools`/`systemPrompt` extended correctly; `GetProfileTool`/`SaveProfileTool` are thin,
correct dispatches per plan §3.2. Independently re-ran the full offline suite (**1935 passed/4
deselected**, matches) and `./scripts/test_queries.sh` (**408/408**, matches). **Independently
re-mutation-tested the highest-risk spot myself**, not just trusting the report: removed
`config.model`'s line from `proof_defs.py` on a copied-aside file, re-ran
`test_salesperson_scaffold.py` — both flagged tests failed with `KeyError: 'model'` exactly as
claimed, restored, full suite green again. Reseeded `reference` and re-verified `ws:acme` in sync
(`verify_workflows.sh`/`verify_catalog.sh`/`verify_salesperson.sh acme` all `OK`, `salesperson@v3`
correct 2-step/1-transition topology, no stray graphs). Confirmed `docs/QUERIES.md` §17 exists as
claimed. **K-054 implementation-complete, not yet gated** — proceeding to the batched `analyst`
(U25) + `qa-engineer` (U26) gates next, covering both clusters (U23+U24) together as one diff, same
posture K-052/K-053 used at their equivalent checkpoint.
| U25 | `analyst` (fresh — standing direction) | `a81b4331031ba6b70` | accepted | `docs/reviews/workflow-durable-profile-impl.md` | self → **approve** (1 nit, no blocker) | 119k tok / 39 tools |
| U26 | `qa-engineer` (fresh — standing direction) | `ae6397b0a69156ca6` | accepted | `docs/test-plans/workflow-durable-profile.md`, `docs/test-reports/workflow-durable-profile-report.md` | self → **PASS** (0 defects against K-054's own scope) | 174k tok / 102 tools |

**U26 delivered — K-054's AC-1..AC-3 all hold live, including both directions of the historical
BLOCKER axis, verified independently by `teco`, not on the delegate's word alone.** Read the report
in full: TP-04/TP-05 ground-truth both partial-update directions against the real `Customer` node.
Independently re-ran the offline suite (**1935 passed/4 deselected**, matches) and
`./scripts/test_queries.sh` (**408/408**, matches); reseeded and re-verified `ws:acme` in sync
(`verify_workflows.sh`/`verify_catalog.sh`/`verify_salesperson.sh acme` all `OK`, `salesperson@v3`
correct topology). **Spot-checked the report's final ground-truth claim directly**: `GRAPH.QUERY
ws:qa-durable-profile "MATCH (c:Customer {customerId:'u1'}) RETURN c.name, c.deliveryAddress,
c.profileUpdatedAt"` → `Jane Smith` / `456 Oak Ave, Springfield` / `1788034841415` — matches the
report's stated final state exactly. Confirmed the delegate's `kaizen_team` entry is present (an
unrelated `@mention`-dispatch mechanics note, from this pass's own live setup work). Two
non-blocking observations noted, neither gating: Ministral doesn't always call `get_profile` on a
turn that doesn't itself implicate the profile (prompt-discipline note, not a persistence bug —
the very next relevant turn checked correctly); a pre-existing, already-tracked K-052 catalog MINOR
recurred opportunistically (`filter_products` category-mismatch), out of this pass's scope,
correctly not re-investigated. **K-054 fully gated — analyst approve, qa-engineer PASS.**
Committing both gates together.

**U25/U26 dispatched in parallel** (same precedent as U21/U22 at K-053's equivalent checkpoint):
`analyst` briefed to skip any suite run that wipes `reference` (`pytest` offline, `test_queries.sh`)
to avoid colliding with `qa-engineer`'s concurrent live session against `ws:acme`/a fresh
workspace — static review only, `verify_*.sh` (read-only) fine to use for its own before/after
check.

**U25 delivered — approve, 1 nit, no blocker.** Verified independently by `teco`: read the review
doc in full, spot-checked its one nit (`git show 663093d ... | grep -c '^+def test_'` → **11**,
confirmed, the commit message's "12" claim is off by one, cosmetic only) and confirmed the
`Extended by:` pointer landed correctly on the parent plan-review doc. Findings on `config.model`
survival and the three-altitude AC-2 test coverage read as genuinely independently re-derived
(cites specific line numbers/query text/live `GRAPH.RO_QUERY` output), not restated from the
coordination doc's own notes. Waiting on U26 before committing both together, same as K-053's
equivalent gate pair.
| U27 | `coder` (fresh) | `a4b5eb52bafbd0085` | delivered | K-055 cluster 1: `querygen.py` (DSL + unit tests incl. reviewer's escape fixtures). Teco-verified: diff read in full, offline suite independently re-run (24/24 `test_querygen.py`, 1959 passed/4 deselected overall — matches claim), delegate's own two mutation tests (`var` pattern removal, anchor-stripping+`.match()` swap) spot-checked as plausible from the diff. Teco's own independent mutation test (removing the explicit `label in schema.labels` `_require`, unrelated to the delegate's two) found a real test-coverage gap: for a bare-aggregate `returns` entry with no property (e.g. `count(p)`), the property-allowlist check never fires, so the label-registration guard is the *sole* defense — and no shipped test exercises exactly that shape (the existing `test_compile_rejects_unregistered_label` uses `returns=["p.name"]`, which is also caught by the property check, masking this). Shipped code is safe today (verified: restoring the original file correctly raises `ValueError` on this exact case) — this is a coverage gap, not a live defect. Folded into U28's brief for the same delegate to close. `querygen.py` restored byte-identical (`diff` empty) after both mutation tests. | `analyst` (U31, batched w/ U27-U30) → — | ~122k tok / 33 tool uses |
| U28 | `coder` (resume U27) | `a4b5eb52bafbd0085` | delivered | K-055 cluster 2: `repository.py` (`run_readonly_query` + `_graph_by_key`), `services.py` (`run_structured_query`, deliberate `timeout=None`-not-forwarded logic), `tools.py` (`QueryGraphDataTool` + schema-description generator + internal structured-completion call) + U27's coverage-gap nit closed (`test_compile_rejects_unregistered_label_for_bare_aggregate_with_no_property`, confirmed present and correctly targeted). Teco-verified independently: full diff read (7 files, 665 lines), full offline suite re-run myself (**1978 passed, 4 deselected** — matches claim exactly), delegate's own two mutation tests (`services.py` unconditional-timeout-forward, `tools.py` narrowed except-clause) read as sound from the diff/tests. Teco's own independent mutation test (forcing `dataset` into the internal `QueryRequest` removed) broke 2/10 `query_graph_data` tests — confirms the forcing is load-bearing, restored byte-identical after (`diff` empty). Verified against actual repo conventions, not just the plan sketch: `self._models.llm("step", ws=ctx.ws)` matches the existing `ModelGateway` call pattern used elsewhere in `tools.py`; `extract_own_line_json_object(reply, require_key="matches")` matches `llm.py`'s real signature; the live `test_run_readonly_query_executes_a_compiled_query_and_returns_dict_rows` test proves the un-aliased-column row shape (`{"c.name": ..., "c.channelId": ...}`) is real, not assumed. Static AST regressions for Layer 2 (`run_readonly_query` never calls `.query()`/`.profile()`; `CompiledQuery` accepted by exactly one method) read and confirmed sound. Kaizen entry confirmed present (`b7e1f0a2` — FalkorDB un-aliased-RETURN column naming). **Note for later, not a gate:** delegate's own usage carried a very large context this round (~297k tokens / 181 tool uses per the completion notification) — U29 (golden-set harness) should dispatch to a **fresh** `coder` rather than resuming this agent further, per the cost-based fresh-dispatch guideline. | `analyst` (U31, batched w/ U27-U30) → — | ~297k tok / 181 tool uses |
| U29 | `coder` (fresh — per U28's own note) | `a8ee3e4945fc16139` | delivered | K-055 cluster 3a: `scripts/seed_nlq_eval_corpus.{py,sh}` (12-doc "Marlowe Robotics" synthetic corpus, sequential-post idempotent seed script), `server/tests/eval/nlq_corpus_provenance.json`, into a dedicated `ws:nlq-eval` workspace. Teco-verified independently: `git status` matches claim exactly (3 new files, nothing else touched); live graph re-queried directly — 12/12 `Document{status:'ready'}`, `Entity.type` distribution matches claim exactly (Organization 17/Other 11/Location 11/Event 8/Product 7/Person 5/Concept 3), conflicting-fact pair independently confirmed persisted (`Marlowe Robotics -[has]-> "62"` and `-[has]-> "140 employees"`, both present), absentee confirmed (`revenue`/`arr` appear only in the reserved not-found question/forbidden-terms check, never as a stated fact); offline suite independently re-run (**1978 passed, 4 deselected** — matches claim exactly); kaizen entry confirmed present (LM Studio model-swap-thrashing-on-concurrent-ingestion fact). **Note:** my own independent suite re-run re-wiped `reference` (the same documented hazard the delegate already hit and fixed once) — re-ran `bootstrap_schema.sh`/`seed_demo.sh`/`seed_workflows.sh`/`seed_catalog.sh`/`seed_salesperson.sh` against `ws:acme` myself; `verify_workflows.sh`/`verify_catalog.sh`/`verify_salesperson.sh` all report `OK` again. `ws:nlq-eval` unaffected by any of this (confirmed 12 documents still present after). Two garbled extraction entities flagged by the delegate for U29b's author to avoid building precise pairs against, not re-verified line-by-line by teco (low-stakes, self-disclosed). | `analyst` (U29-gate) → — | 202k tok / 68 tools |
| U29-gate | `analyst` (fresh) | `aabe27efc9888cf21` | accepted | `docs/reviews/workflow-nl-query-generation-corpus.md` | self → **approve with suggestions** (1 MAJOR, 1 MINOR, 2 nits) | 103k tok / 28 tools |

**U29-gate MAJOR finding — teco-verified independently, both sub-claims confirmed live:** a third,
previously-undisclosed extraction defect beyond U29's own two — `NLQ-EVAL-08` (one of the two FR-6
conflict documents) produced a semantically **reversed** `Marlowe Robotics -[acquired by]->
Brightline Systems` edge (the correct direction, `-[acquired]->`, is in `NLQ-EVAL-06`); a second
garbled date entity, `NovaGrid -[was released in]-> "September than"` (parallel to the disclosed
`"January 1026"`). Not a corpus defect requiring re-ingestion (verdict: approve with suggestions,
non-blocking) — carried forward as an explicit exclusion list into U29b's brief instead, so the
golden-set author doesn't have to rediscover it. MINOR (script's conflict self-check filters by
subject only, not the `has` predicate — real data unaffected, future-regression coverage only) and
2 nits (no incremental provenance write, `'failed'`==`'ready'` skip conditions) logged in the review
itself, non-blocking, not chased further this session.

| U29b | `tdd-engineer` (fresh) | `a861eba1703f2282d` | delivered | K-055 cluster 3b: `server/tests/eval/nlq_golden_set.jsonl` (39 pairs: catalog 20, knowledge_base 19) + `test_nlq_golden_set_integrity.py` (offline, DB-free, 249 generated tests). Teco-verified independently: `git status` matches claim (2 new files; two unrelated stray changes from a concurrent `tico` session — `kiro/docs/requirements/kiro-vision-followups.md` + new `falkor-chat/docs/requirements/deliverable-provenance.md` — correctly not touched or committed by this unit); per-dataset/per-shape counts re-tallied from the file directly, match claim exactly; offline suite independently re-run (**2227 passed, 4 deselected** — matches claim); spot-checked 3 pairs live against the actual graphs (a compound-filter catalog pair, a knowledge_base aggregation pair, the conflicting-facts pair) — all exact matches. **Note:** my own suite re-run re-wiped `reference` again (delegate's brief didn't carry the restore reminder — a gap in my own brief, not the delegate's fault) — restored `ws:acme` myself (bootstrap/demo/workflows/catalog/salesperson), all three verify scripts `OK` again after. Kaizen entry re: `ws:nlq-eval`'s un-fused/un-deduplicated entities (Marlowe Robotics = 9 raw nodes) confirmed present and consistent with the golden set's own count-vs-dedup-set handling. | `analyst` (U29b-gate) → — | 155k tok / 47 tools |

**Brief carried a load-bearing finding teco derived by reading `querygen.py` directly (not in either
upstream doc as stated):** `KNOWLEDGE_BASE_SCHEMA` allows only `Entity`/`Document`/`Chunk` node
properties, no `RELATES_TO` traversal at all (plan §3.6's v1 non-goal) — so any question reachable
only via a relationship edge (including the corpus's own conflicting-fact pair, which is
edge-modeled, not a node property) is structurally unanswerable by the shipped mechanism. Per the ml
note's own §6 guidance this doesn't mean dropping those golden-set categories — author them anyway,
expect ~0%, label it a known v1 scope limit, not a defect. Also flagged: golden-set `dataset` field
must use the actual `DATASET_REGISTRY` keys (`catalog`/`knowledge_base`), not the ml note's
illustrative labels; `Entity` has no numeric property (aggregation there is count-only).
| U29b-gate | `analyst` (fresh) | `a90d20dd5760c0577` | accepted | `docs/reviews/workflow-nl-query-generation-golden-set.md` | self → **needs changes** (1 MAJOR) | 103k tok / 18 tools |
| U29b-fix | `tdd-engineer` (resume U29b) | `a861eba1703f2282d` | delivered | reworded 18/39 `question` fields (all flagged template-clone groups now fully distinct within their `(dataset, shape)` group). Teco-verified: diff is `question`-field-only (confirmed line-by-line, no `expected`/`shape`/`dataset`/`id` touched); re-tallied distinct-question-count per group myself — every group now has as many distinct questions as pairs (5/5, 4/4, 3/3, 3/3, 5/5, 4/4, 3/3, 3/3, 3/3, 4/4, 2/2); offline suite independently re-run (**2227 passed, 4 deselected** — matches, unchanged from before the fix as expected for a pure text edit). Restored `reference`/`ws:acme` again after (same pytest-teardown hazard) — all three verify scripts `OK`. | `analyst` (re-gate) → — | 192k tok / 71 tools |
| U29b-regate | `analyst` (resume `a90d20dd5760c0577`) | `a90d20dd5760c0577` | accepted | `docs/reviews/workflow-nl-query-generation-golden-set.md` `## Pass 2` | self → **approve with suggestions** (finding #1 downgraded Major→Minor: `nlq-21`/`nlq-24` still byte-identical, `nlq-22`/`nlq-29` synonym-only) | 147k tok / 10 tools |

**U29b closed.** Teco-verified the residual Minor directly (`nlq-21`/`nlq-24` confirmed byte-identical, `nlq-22`/`nlq-29` confirmed `type`→`kind`-only) — accurate, and low-value enough (1 template pair among 39, verdict already non-blocking) not to spend a third fix/re-gate cycle on; left as a standing suggestion in the review, not re-dispatched.
| U29c | `tdd-engineer` (fresh) | `aebfed22ae08ee1f2` | delivered | `server/tests/eval/nlq_scoring.py`+tests (32, mutation-tested), `run_nlq_golden_set_eval.py` (live harness), `nlq_eval_results.json` (raw per-pair results), `docs/test-reports/workflow-nl-query-generation-report.md`. **Verdict: Layer 1 gate FAILS decisively — overall 18/39=46.2% (CI [31.6,61.4]) vs ≥85%; knowledge_base subset 8/19=42.1% (CI [23.1,63.7]) vs ≥75%; not-found false-answer rate 0/6=0% vs ≤10% PASSES.** Teco-verified independently: `git status` matches claim (5 new files); re-tallied `nlq_eval_results.json`'s 39 records directly — 18 correct overall, 8/19 knowledge_base, 0/6 false-answers — exact match; spot-checked 5 illustrative-failure records (`nlq-18`, `nlq-21`, `nlq-25`, `nlq-34`, `nlq-38`) byte-for-byte against the report's own quoted examples — all accurate; offline suite independently re-run (**2259 passed, 4 deselected** — matches, +32 vs. pre-unit baseline). Restored `reference`/`ws:acme` again after (same pytest-teardown hazard, disclosed by the delegate itself this time) — all three verify scripts `OK`. Not a harness defect — the scorer module's own rules are unit/mutation-tested independently of this finding. | `analyst` (U31, batched w/ U27-U30) → — | 271k tok / 95 tools |

**STOP — this is a blocker that changes direction, not a routine gate result.** The already-`analyst`-approved K-055 mechanism (`querygen.py`/`tools.py`) does not clear its own accuracy bar against the real model — and not only on the two shapes designed to fail; `compound-filter` (0/3), `catalog aggregation` (1/5), and `knowledge_base single-fact` (1/4) all substantially underperform too. This is the same shape of problem as the still-open K-056 (`qwen/qwen3-4b-2507` local-model capability limits under this workload), where the user previously directed "no further mitigation iteration" mid-session. Paused and asked the user how to proceed via `AskUserQuestion`.

**User decision: root-cause it now.** Dispatching `data-scientist` RCA as U29d.

| U29d | `data-scientist` (fresh) | `ac9e47f85f5801d9a` | accepted | `docs/reviews/workflow-nl-query-generation-rca.md` — all 21 misses categorized (A: 8 quoted-numeric-string, B: 2 un-normalized filter value, C: 3 un-fused-duplicate no-DISTINCT, D: 2 wrong projected property, 6 structurally-expected). **Refutes the report's own leading hypothesis** (live-probed `nlq-21`: model's filter/projection were already correct; failure was C, not a prompt/filter-inversion bug). Recommends DSL fixes (A/B/C, 13/21 misses, zero model dependency) first, prompt hardening (D, drop-in replacement text supplied) second, defers model-routing (not supported by evidence). Teco-verified: raw-JSON shapes for `nlq-08/16/18/21/25` independently re-checked, all consistent with the claimed categorization; kaizen entry confirmed present. Live-probe disambiguation (7 probes) not independently re-run by teco (would need live LM Studio access mid-review; internally consistent with everything independently checkable). | self → — | 140k tok / 16 tools |
| U29e | `graph-dba` | `a30a62b591c18c6ba` | accepted | `claude/graph-dba/falkordb-quirks.md` (+21 lines, new dialect fact). **Confirmed: `RETURN DISTINCT <col> ORDER BY <col not in RETURN>` silently succeeds on this build but with non-deterministic, insertion-order-dependent ordering** (live-proven by flipping node-creation order and getting different sort results for the identical logical data) — worse than an error, since it looks deterministic and isn't. **RCA's C fix is not safe to ship as specified**; recommends rejecting such a `QueryRequest` at DSL validation (`order_by ⊆ returns` whenever `DISTINCT` will apply) rather than silently dropping `ORDER BY` or auto-widening `RETURN`. Teco-verified: quirks-file diff is a clean +21-line addition; probe graphs confirmed deleted (queried one, absent from the loaded-graphs list); `reference`/`ws:nlq-eval`/`ws:acme` unaffected. | — | 62k tok / 15 tools |
| U29f | `tdd-engineer` (fresh) | `a1a0c4b3b0c42fecf` | delivered | `querygen.py` (typed `DatasetSchema`, fixes A/B/C), `tools.py` (fix D prompt), +tests, new opt-in `test_querygen_live.py`. Teco-verified: diff read in full; fix A/B independently spot-checked via direct `compile()` calls (numeric coercion + error, `*Normalized` value normalization) — both correct; offline suite independently re-run (**2283 passed, 8 deselected** — matches); live opt-in suite independently re-run (**4 passed** — matches); restored `reference`/`ws:acme` after, all 3 verify scripts `OK`. **Teco found a real regression fix C introduces, not caught by the delegate's own tests**: `compile()` now raises `ValueError` on the "top-1-by-sort-key returning a different column" pattern (`returns=["p.name"], order_by="p.price", limit=1`) — repro'd directly. This is exactly the RCA's own drop-in prompt example #4 ("which product is the cheapest") and the *only* way this single-`MATCH` DSL can express "name of the extreme-valued item" (no aggregate function returns an associated non-aggregated column) — so fix C as shipped makes every min/max-by-name superlative question (`nlq-16/17/20`) permanently abstain, a full regression on a designed, golden-set-required shape, worse than pre-fix (where it depended on fix-A's bug rather than failing unconditionally). Not something to silently accept or fix myself — routing to `analyst` with this finding front-loaded, `graph-dba` likely needed for the correct Cypher-shape fix. | `analyst` (U29f-gate) → — | 197k tok / 75 tools |
| U29f-1 | `graph-dba` (fresh) | `a1c612ca76049f555` | accepted | `claude/graph-dba/falkordb-quirks.md` (+37 lines). **Confirmed the reject-outright guard was not just over-broad but hid a worse fact: the original buggy shape can return a flat-out WRONG answer, not just a nondeterministic one** (live repro: 3 products, `RETURN DISTINCT name ORDER BY price LIMIT 1` returned the wrong cheapest item in one creation order). **Fix: `WITH DISTINCT <returns ∪ {order_by}> ORDER BY <order_by> LIMIT n RETURN <returns>`** — dedup on the full tuple, not just the returned columns — live-verified correct in both creation orders and on genuine duplicates. One residual (real ties under `LIMIT 1` resolve insertion-order-dependently) is ordinary engine behavior, not a defect, explicitly out of scope. Teco-verified: quirks-file diff reviewed in full (well-evidenced); probe graph confirmed deleted. | — | 86k tok / 10 tools |
| U29f-2 | `tdd-engineer` (resume U29f) | `a1a0c4b3b0c42fecf` | delivered | `querygen.py` (tuple-`DISTINCT` `WITH` fix, re-aliased back to preserve the "column key = raw expression text" contract — a detail the delegate correctly derived from `repository.run_readonly_query`'s docstring, not prompted), test updates. Teco-verified: diff read in full; independently re-compiled the exact superlative shape (`returns=["p.name"], order_by="p.price", limit=1`) — cypher matches claim byte-for-byte (`MATCH (p:Product) WITH DISTINCT p.name AS c0, p.price AS c1 ORDER BY c1 ASC LIMIT $limit RETURN c0 AS \`p.name\``); ran it live against real `reference` via `repository.run_readonly_query` myself — returned `Gaming Mouse Pad XL`, exactly `nlq-16`'s golden expected value; offline suite independently re-run (**2285 passed, 13 deselected** — matches); live opt-in suite independently re-run (**9 passed** — matches); restored `reference`/`ws:acme` after, all 3 verify scripts `OK`. K-055's DSL fix cycle (A/B/C/D + the C regression correction) is now solid. | `analyst` (U29f-gate) → — | 238k tok / 32 tools |
| U29g | `tdd-engineer` (fresh — U27's own context long past the fresh-dispatch threshold) | `ac2265eb82434a950` | in-flight | re-run `run_nlq_golden_set_eval.py` against the fixed mechanism, updated report | `data-scientist` (methodology) → — | — |
| U30 | `coder` (resume U27) | — | queued | K-055 cluster 4: `proof_defs.py` (v4), seed/verify scripts | — | — |
| U31 | `analyst` (fresh) | `adcf0fb00fe2b1081` | accepted | `docs/reviews/workflow-nl-query-generation-impl.md` | self → **needs changes** (2 MAJOR, 2 MINOR) | 153k tok / 65 tools |
| U31-regate | `analyst` (resume `adcf0fb00fe2b1081`) | `adcf0fb00fe2b1081` | accepted | `docs/reviews/workflow-nl-query-generation-impl.md` `## Pass 2` | self → **approve** | 161k tok / 23 tools |

**K-055 implementation-diff gate closed.** Teco-verified: git status matches claim (clean except the review doc itself — the reviewer's own incidental `judge_calibration.json`/stray-report side effect from an unrelated broader live-suite run was self-corrected via `git checkout`/delete before I even looked). Both MAJORs independently re-confirmed closed by the reviewer via live re-verification, not re-reading. The mechanism itself is done. Next: re-run the golden-set harness for the real, current numbers.
| U31-fix | `tdd-engineer` (resume U29f) | `a1a0c4b3b0c42fecf` | delivered | duplicate-`returns` uniqueness guard in `compile()` (MAJOR 2), 2-column tuple-distinct regression tests unit+live (MAJOR 1 — code was already correct, only coverage was missing), both MINORs (docstring reword, bool-filter-gap note). Teco-verified: diff read in full; independently reproduced both fixes live (`returns=["p.name","p.name"]` → raises exactly the claimed message; the 2-column pairing still correct); offline suite independently re-run (**2289 passed, 14 deselected** — matches); live opt-in suite independently re-run (**10 passed** — matches); restored `reference`/`ws:acme` after, all 3 verify scripts `OK`. Despite the large cumulative context (274k tok this round alone), the delegate's work held up cleanly on full independent verification — no fresh-dispatch needed after all. | `analyst` (re-gate) → — | 274k tok / 37 tools |

**U31 MAJOR findings, both teco-verified independently, live:** (1) tuple-`DISTINCT` re-aliasing has zero test coverage past 1 column — teco reproduced the reviewer's exact 2-column case live (`returns=["p.name","p.category"], order_by="p.price"` → correctly paired `{"p.name": "Gaming Mouse Pad XL", "p.category": "Peripherals"}`); a column-swap mutation would ship silently today. (2) a duplicate `returns` entry (`["p.name","p.name"]`) compiles successfully but crashes the *entire workflow run* at the engine (`redis.exceptions.ResponseError`) — teco reproduced this exact error live — contradicting `QueryGraphDataTool`'s own documented "never crashes" contract; `tools.py:993-995`'s `run_structured_query` call confirmed unwrapped by any try/except. Both cheap, targeted fixes.
| U32 | `data-scientist` (resume `a277477d79ce069c6`) | — | queued | golden-set harness verification, thresholds report | — | — |
| U33 | `security-expert` (resume `ae4185d22350610f7`) | — | queued | live Groups A-E re-run against real `query_graph_data` | — | — |
| U34 | `qa-engineer` (resume) | — | queued | live acceptance, K-055 (AC-2/AC-5) | — | — |
| U35 | `qa-engineer` (resume) | — | queued | combined e2e pass, all four capabilities in one `salesperson@v4` conversation | — | — |

**Re-plan (this session, before dispatch): the former single "U29" row (golden-set harness +
fresh corpus) is split into U29/U29-gate/U29b/U29b-gate/U29c**, mirroring `graphrag-eval`'s own
corpus→gate→golden-set→gate→metrics-harness sequencing (`docs/plans/graphrag-eval-coordination.md`
U3a/U3a-gate/U3b/U3c) — the step-table sizing rule applies (>3 steps, several files: fixture docs,
a new seed script, a JSONL golden set, a scorer module + tests) and each of the three artifacts
(corpus, golden set, harness code) has a materially different owner/verification shape, same as
the precedent. CPG check for this cluster: `cpg_falkorchat`, built `2026-08-26T22:27:22Z`, no
`sourceCommit` (scratch-copy build, real source `falkor-chat/server`) — **stale**, 12 commits to
`falkor-chat/server` since that build (includes U20/U23/U24/U27/U28). Noted for `analyst`'s U31
gate and `qa-engineer`'s U34/U35 passes if either leans on it for impact analysis; not relevant to
U29's own corpus-authoring work.

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
