# Backlog — falkor-chat

> **Status:** active · **Owner:** `teco` · **Tracks:** K-016…K-050

> **How to read this.** Forward-looking only — what is proposed but unbuilt. *When* something
> changed and *what* it involved live in [`HISTORY.md`](./HISTORY.md), one dated entry per
> delivered item; `## Delivered` below is the index into it, not a summary of it. Design lives in
> [`DESIGN.md`](./DESIGN.md), the query library in [`QUERIES.md`](./QUERIES.md). Completed plan
> documents stay where they are, marked `Status: archived` (root `AGENTS.md`); `archive/` holds
> frozen documents from the previous convention.
>
> **Milestone status is authoritative here**, not in DESIGN.md or any README.
>
> Status markers: 🔵 proposed · 🟡 in-progress · ✅ delivered (→ `HISTORY.md`) · ⚪ rejected/deferred.
> Item IDs keep the `K-` prefix from the former `kaizen/plan.md`.

## Active

**M5 — Ingestion pipeline & entity fusion** 🟡, the only item under construction, is **K-050**
below. Its six implementation stages run out of `docs/plans/document-ingestion.md`, tracked in
`docs/plans/document-ingestion-coordination.md`: Stages 1–4 (chunking, chunk embeddings +
standalone search, extraction, fusion) are delivered and `analyst`-gated; **Stage 5**
(chat-grounding integration, FR-2) is delivered with its gate in flight; **Stage 6** (the
deferred hardening set — thread fan-out, batching/pooling) is the last one open. M5 reaches ✅ on
a `qa-engineer` acceptance pass over the whole pipeline.

Everything else open is a follow-up filed out of a closed milestone (`## Open follow-ups`) or the
deferred M2.5 hardening track — neither is on the M5 path.

### K-050 — Ingestion pipeline & entity fusion: chunk, extract, fuse, and serve as both chat grounding and a standalone knowledge base (🟡 in-progress — requirements `docs/requirements/document-ingestion.md`, plan `docs/plans/document-ingestion.md`, 2026-08-22)

> **Why it exists.** Today falkor-chat's GraphRAG has exactly one knowledge source: chat messages,
> embedded as they're posted. There is no path for ingesting knowledge from outside the chat itself,
> even though the schema has carried a dormant, never-populated shape for exactly this
> (`Document`-`[:HAS_CHUNK]`->`Chunk`-`[:ABOUT]`->`Entity`, plus a `Chunk.embedding` vector index,
> bootstrapped since M2 — `docs/DESIGN.md` §5.1/§7.1, `docs/QUERIES.md:472`). K-050 finally
> populates that scaffolding: documents (and agent-generated text, treated identically) are chunked,
> entities/relationships are extracted from chunk text into real graph nodes/edges, and each
> extracted entity is fused against what the graph already knows at one of three confidence tiers —
> auto-merge, suggested-pending, or confirm/reject (with rejection reversible) — while conflicting
> facts from different sources are always kept side by side with their own provenance, never
> silently overwritten.
- **Scope (FR-1..FR-14/AC-1..AC-10, plan §4 six stages).** Chunking (FR-13, a deterministic
  size/overlap/boundary splitter — no LLM); extraction (FR-7a, LLM-based entity/relationship
  extraction into `Entity` nodes + a new `RELATES_TO` fact edge, predicate carried as an opaque
  property rather than an open-ended relationship-type vocabulary); fusion (FR-6/FR-8/FR-9/FR-10 —
  a recommended `MatchSuggestion` node per candidate pair, mirroring the `WorkflowRun.status`
  index-anchored pattern, rather than ever physically merging `Entity` nodes — FalkorDB has no
  APOC-style node-merge procedure, and physical merge would also make FR-6's "keep both conflicting
  facts" a separate mechanism instead of a structural guarantee); a new MCP/REST write+read surface
  (FR-5: `ingest_document`/`ingest_documents`/`get_document`/`search_documents`/
  `list_pending_matches`/`confirm_match`/`reject_match`/`recheck_match`); bulk ingestion (FR-11) and
  full-source retention (FR-12, `Document.text` verbatim); and chat-grounding integration (FR-2 —
  extending the existing `AgentResponder`/`EMITTED`-provenance retrieval path to also seed from
  `Chunk` vectors, app-layer fan-out+merge, per the requirements doc's own decision log) alongside a
  standalone `Chunk`-only search capability (FR-3), deliberately **not** unified into one search
  index (FR-14 — the requirements doc explicitly does not require that).
- **OQ-1/OQ-2/OQ-3 (requirements doc, explicitly left open there for design):** OQ-2 (where a
  pending match surfaces) resolved to a **dedicated review surface** (`list_pending_matches`, MCP +
  REST), not a chat post — a pending fusion decision has no natural channel/thread anchor and FR-14
  already keeps ingested-content concerns separate from chat. OQ-3 (re-evaluating a rejected match)
  resolved to **two** paths — automatic reopen to `pending` (never straight to `confirmed`) when a
  later ingestion independently re-derives the same candidate pair, plus an explicit
  `recheck_match` tool for an on-demand human/agent-forced recheck. OQ-1 (what "very-high
  confidence" means) gets a **recommended default** (exact normalized-name+type match, zero
  ML-confidence numbers, chosen because this pipeline has no calibration data yet — unlike the K-027
  guard judge, which was calibrated against a golden set before being trusted) — flagged to
  `data-scientist` to confirm or replace, not locked here.
- **Two design axes delegated, not decided in the main plan (plan §0):**
  1. **`docs/plans/document-ingestion-ml.md` (`data-scientist`)** — the extraction
     technique/prompt/schema (FR-7a) and whether the OQ-1 default above is defensible for v1 or
     needs semantic (embedding) matching to catch non-lexical synonyms fuzzy string matching can't.
  2. **`docs/plans/document-ingestion-graph.md` (`graph-dba`)** — final schema for `MatchSuggestion`
     (node vs. edge-property, indexes/constraints, RAM), the exact `Document`/`Chunk`/`Entity`/
     `RELATES_TO` Cypher, the `Entity.name` full-text index DDL, and generalizing the `EMITTED`
     provenance write/read (today `Message`→`Message` only, `QUERIES.md` §10.1) to also target
     `Chunk` for FR-2.
- **Owner chain:** `tico` (requirements ✅) → `architect` (plan ✅) + `graph-dba`/`data-scientist`
  (the two notes above) → `analyst` (plan gate) → implementers per stage (plan §4: chunking/write
  path → chunk embeddings/standalone search → extraction → fusion → chat-grounding integration →
  batch hardening) → `analyst` re-gate → `qa-engineer`
  (`docs/test-plans/document-ingestion.md` + `-report.md`). Coordinated by `teco`
  (`docs/plans/document-ingestion-coordination.md`, not yet authored).
- **Risks/RAM (rule 6):** `Chunk.embedding`'s vector index is the dominant new RAM line (same
  empirical ~12.4 KB/vector-at-1024-dim shape as `Message.embedding`, `docs/DESIGN.md` §11) — no new
  DDL needed (the index already exists, bootstrapped since M2), but ingestion is a materially new,
  corpus-size-driven growth axis the existing per-workspace RAM budget did not account for. The
  recommended fusion default deliberately adds **no** second vector index (`Entity.embedding`) —
  reuses the existing `Message.text`-style RediSearch full-text mechanism instead — to avoid
  doubling that growth axis; if data-scientist's note argues for semantic matching instead, that
  RAM trade-off must be made visibly, not silently. Per-chunk extraction is capped (recommended 20
  entities/relationships per chunk) to bound both LLM output and graph growth, mirroring the
  existing `docs/DESIGN.md` §5.4 entity-fan-out mitigation.
- **Test strategy:** full AC-1..AC-10 → test-altitude map in plan §5, plus chunking boundary-rule
  unit tests, extraction-parser robustness tests (reusing the K-027-proven fence-tolerant JSON
  parser rather than a bare `json.loads`), background-job failure isolation
  (`Document.status` reflects a failed/partial pipeline rather than silently sticking at
  `'processing'`), and `graph-dba`'s `test_queries.sh` baseline raise for every new Cypher shape.
- **Done-condition:** all six implementation stages delivered and `analyst`-gated, `qa-engineer`
  acceptance PASS (or PASS-with-parked-defects) on green baselines, DESIGN §5.1/§7 and this
  component's docs updated in the same changes ⇒ **M5 ✅**.
## Milestone-to-green map

| Milestone | Reaches ✅ when | Items |
|---|---|---|
| **M1 — Chat core** ✅ | Reached 2026-07-06 | K-011, K-012 |
| **M2 — GraphRAG** ✅ | Reached 2026-07-08 — embeddings + vector index @1024 + hybrid retrieval + AI agent participant with `EMITTED` provenance; QA-accepted, zero defects | K-008, K-013, K-014, K-015 |
| **M3 — Workflows** ✅ | Reached 2026-07-21 — def model + snapshot + executor + chat linkage, proven by one conversational and one business-process flow; QA-accepted (PASS with parked, model-gated limitations) | K-020, K-021, K-022, K-023, K-024, K-025 |
| **M3.5 — Web API Coverage** ✅ | Reached 2026-07-29 — FR-1..FR-10/AC-1..AC-6 wired into `web/`; QA-accepted (PASS with parked limitations) | K-036 |
| **M4 — LLM provider & model configuration** ✅ | Reached 2026-08-11 — providers/models declared once in two hand-edited files, every consumer resolving through one `ModelGateway` seam, per-consumer model or role, resolved model on the run trace, the four legacy env vars replaced; QA-accepted on both landings | K-042 |
| **M5 — Ingestion pipeline & entity fusion** 🟡 | Documents (and agent-generated text) chunked, entity/relationship-extracted, and fused against existing knowledge at three confidence tiers (auto-merge / suggested-pending / confirm-reject-reconsiderable); ingested knowledge retrievable via the existing chat-grounding path **and** as a standalone knowledge base; a connected MCP agent can write ingested content as persistent memory | **K-050** 🟡. Requirements `docs/requirements/document-ingestion.md`; plan `docs/plans/document-ingestion.md`; graph-side `docs/plans/document-ingestion-graph.md`; ML `docs/plans/document-ingestion-ml.md`; coordination `docs/plans/document-ingestion-coordination.md`. |
| **M2.5 — Hardening** ⚪ *(deferred)* | Real auth, transport-level agent path, real-time push | K-016 → K-017, K-018 |

Follow-ups filed out of a closed milestone are **not** green-gates for it; they are listed under
`## Open follow-ups`.

## Standing decisions

- **"M2 green" = functional GraphRAG** (user, 2026-07-05) — embeddings + vector index + hybrid
  retrieval + agent participant + `EMITTED`. Real auth and real-time push are deferred to the M2.5
  track; rationale: "long road before production." Safe because the AI participant is a
  **server-side responder** posting as a configured Agent, needing no per-request auth to function.
- **The `identity` graph is authoritative (standalone)**, not an external-IdP projection (user,
  2026-07-05; DESIGN §1.2). K-016 implements auth *per* that decision — no user input pending.
- **The M2 model stack** (embedding model/dim, agent LLM, runtime, VRAM, upgrade path) is locked in
  `docs/DESIGN.md` §1.3.
- **`bootstrap_schema.sh` still defaults to `EMBEDDING_DIM=1536`** and **must** be run with
  `EMBEDDING_DIM=1024` for any new workspace (`scripts/bootstrap_schema.sh:255-264` documents this
  inline). `start_server.sh` already defaults to 1024.

## Open follow-ups

Each was filed out of a closed milestone's gates or a later investigation; none gates M5.

### K-029 — Converge the seed def sources into `proof_defs.py` (+ the symmetric `decision` publish invariant) (🔵 proposed — filed out of K-024, open item O-5 / gate m-9 / nit n-3)

> **Why it exists.** The two seeded defs use **two different source conventions**, deliberately for
> the K-024 slice: `access-request@v1`'s spec is imported from `server/falkorchat/proof_defs.py`
> (so the seed script and the offline acceptance test provably cannot drift), while **`triage@v1`'s
> literal is still inline in `scripts/seed_workflows.sh`**. Moving `triage`'s def *during* K-024 was
> declined with a reason, since corrected by **K-034**: at the time, published defs were believed
> **create-only** (`MERGE … ON CREATE SET`) end to end, so a byte-diff introduced while relocating a
> **live** def was assumed silently swallowed. As of K-034 that is only true for a **property**-only
> byte-diff (e.g. a `config` field reformatted during the move) — it still silently no-ops, so K-029's
> planned before/after equality check remains load-bearing for that half. A **topology**-changing
> byte-diff (e.g. a retargeted transition introduced while relocating the literal) is now **rejected**
> (`409 WorkflowDefConflictError`, nothing written) rather than swallowed — safer, but a `409` mid-
> deploy is still worse than catching it in a pre-flight check, so the equality check stays this item's
> load-bearing safeguard either way. `reference`/`ws:<id>` can still go stale independently whenever
> one side was never re-published/re-materialized at all. That is a split-brain risk to take on its
> own, with its own verification, not as a rider on a feature slice.
- **Owner:** **`coder`**, with an explicit before/after equality check on the published def subgraph
  (not just "the script ran").
- **Scope:** (1) move `triage@v1`'s inline literal into `proof_defs.py` beside `ACCESS_REQUEST_DEF`,
  leaving `seed_workflows.sh` a pure driver over the service layer for **both** defs; (2) prove the
  move is byte-identical *in the graph*, which given create-only semantics means either verifying
  against a freshly published `reference` or bumping `triage`'s version in lockstep with
  `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` (note `start_server.sh` neither forwards nor exports
  those two vars today — a version bump also needs a script change); (3) fold the `n-A` warning
  (`ACCESS_REQUEST_DEF`'s key set **is** `publish_workflow_def`'s keyword signature) into whatever
  shape both defs end up sharing.
- **Also carries nit n-3 — the symmetric `decision` publish invariant.** K-024 enforces
  "a `human`/`wait` step must declare `config.waitsForHuman: true`" at publish, but **not** its
  mirror: **a `decision` step whose outgoing transitions are *all* conditional and which does not
  declare `waitsForHuman` self-loops until the step budget fails the run.** It is documented as a
  warning in `falkor-chat/AGENTS.md` and deliberately left unenforced because the symmetric check
  would **retro-reject existing test fixtures** (`server/tests/test_services.py`) — the same
  blast-radius problem B-2 caused in K-024, which is precisely why it belongs in an item that can
  budget for the fixture edits.
- **Risks:** touching a live published def is the risk; there is no new graph surface and no RAM cost.
- **Test strategy:** a test that both defs come from importable constants; a publish-equality check
  over the def subgraph; if n-3 is implemented, one ordering pin (it must run **last**, like the other
  three invariants) plus the fixture edits it forces.

### K-030 — Allow zero-transition (single-step) workflow defs; guard the `UNWIND` instead of rejecting (🔵 proposed — filed out of K-024 re-gate findings r-1/r-2)

> **Why it exists.** `repository._PUBLISH_CYPHER` ends in a bare `UNWIND $transitions AS tr …
> RETURN …`. With `$transitions = []` the row stream **collapses** — after the `WorkflowDef`, its
> `Step`s and the `START` edge have already been written — so the caller's `res.result_set[0]` raises
> `IndexError` on a **partially written** def. Because publish is `MERGE … ON CREATE SET`, retrying
> the corrected spec on the same `(key, version)` is a **silent no-op on the half-written def**: the
> version is permanently wrong and cannot be repaired by re-publishing. This is the same empty-`UNWIND`
> class that `AGENTS.md` documents as *guarded* for the §4 mention write-block; this path was not
> guarded. K-024 U4b **closed the reachable route** with a `_validate_def_spec` rule (running last)
> that rejects a transition-less spec **before any repository call** — prevention, not a nicer
> exception.
- **What is still open (re-gate r-1):** the fix is **publish-only**. `services.materialize_def` →
  `repository.materialize_snapshot` (`repository.py:1397`) **reuses the same query shape** and performs
  **no** spec validation, so a def poisoned before U4b — or any zero-transition subgraph read back by
  `read_def_subgraph`, which returns `transitions: []` rather than `None` — is still an unguarded
  `IndexError`/500 on materialize. Low likelihood (materialize is fed by publish, now guarded), but the
  guard is **asymmetric**, and the docs/tests currently imply it isn't:
  `server/tests/test_services.py:916` seeds a `FakeRepo` def with `"transitions": []` and asserts
  materialize **succeeds** — true of the fake, and exactly the shape the real query rejects.
- **Accepted limitation to remove (re-gate r-2):** the U4b rule also **rejects a legitimate shape** —
  a genuine single-step def. All four doc sites state the workaround ("a terminal outcome is a step
  with no *outgoing* transition, never a def with none") but none records it as **debt**. Without a
  `K-` number the next person needing a one-step def will either fight the rule or bypass validation.
- **The known cheap remedy:** guard the trailing `UNWIND` in `_PUBLISH_CYPHER` (and therefore
  `materialize_snapshot`, which reuses it) with the **§4 empty-`UNWIND` `CASE` pattern** this codebase
  already relies on and documents as load-bearing — `UNWIND (CASE WHEN $transitions = [] THEN [null]
  ELSE $transitions END) AS tr` with a `FOREACH` that never filters — then **relax**
  `_validate_def_spec`'s rule, and drop the `transitions=[]` mitigation comments in `proof_defs.py`
  and `tests/test_process_input.py`.
- **Owner:** **`graph-dba`** (the query change needs a gate + a re-PROFILE: the guard must not turn the
  index-anchored publish plan into a scan) → **`coder`** for the service-layer relaxation and the
  fixture/doc cleanup.
- **Risks/RAM:** none — no new node, index or property; a query-shape change only. The risk is plan
  regression, which is what the re-PROFILE is for.
- **Test strategy:** a publish and a materialize of a genuine single-step, zero-transition def, both
  asserted to succeed *and* to leave a complete subgraph (steps + `START` + the returned row); the
  existing ordering pins for the other publish invariants must stay green.

### K-032 — Materialize the workflow def's **data-dependence overlay** (CPG-style READS/WRITES) for publish-time static analysis (🔵 proposed — from a design conversation, 2026-07-22)

> **The framing (Code Property Graph lens).** The def graph is already a control-flow graph:
> `(:Step)-[:TRANSITION {guard, order}]->(:Step)` is a guarded CFG, `HAS_STEP` is one-level AST
> containment, and `(:StepRun)-[:NEXT]->` is an executed CFG path. **What's missing is the
> data-dependence layer (DDG):** which `ctx`/`output` keys each step *reads* (via its `cmp`/`llm`
> guards and, for a `decision`/`human`, the keys it branches on) and which it *writes* (a `human`/
> `wait` step's `config.expects`, a step's declared outputs). That information is **not missing —
> it's trapped inside the opaque `guard`/`config` strings**, and `services._validate_def_spec` +
> `guards.validate_cmp` already walk the `cmp` guard tree at publish, so ~90% of the extraction pass
> exists and is currently thrown away. Materialize it as real edges and three otherwise-impossible
> checks become one-hop Cypher.
- **Why it's worth doing (the payoff).** Publish-time (not live-run) detection of:
  1. **Dangling read** — a guard reads a `ctx`/`output` key no upstream step writes. This is exactly
     the **un-enforced n-3 hazard** AGENTS.md documents (a `decision` step with all-conditional
     outgoing transitions and no `waitsForHuman` **self-loops to budget exhaustion**) — today a live
     discovery, turned into a `WorkflowDefSpecError` at seed time. Overlaps K-029's symmetric-invariant
     proposal; this is the graph-shaped way to get there.
  2. **Unreachable step / dead branch** — plain CFG reachability from `START`.
  3. **Change-impact / blast radius** — "I changed `submit`'s output shape; which downstream guards
     read it?" This matters **specifically because published defs are topology-immutable (K-034) and
     property-create-only**: a def edit costs a version bump + snapshot republish + a
     `reference`↔`ws:{id}` split-brain risk, so knowing the blast radius *before* the bump has real
     value here.
- **Hard constraints (fall out of locked decisions — non-negotiable in any plan):**
  - **Derive at publish, never parse in Cypher.** Rule 8 (`ctx`/`config`/`guard` opaque, never
    filtered in Cypher) holds *iff* publish is treated as a compile step — `joern-parse` builds
    overlays once, queries traverse edges; same contract. Extraction runs app-side in the publish
    validator; only the resulting edges hit the graph.
  - **Overlay edges built inside `_PUBLISH_CYPHER` and `materialize_snapshot`, same query.** A
    separately-written overlay on a `MERGE … ON CREATE SET` def is a **new split-brain axis** on top
    of the `reference`-vs-snapshot one — and per the K-030 note the materialize path still `IndexError`s
    after a partial write, so the overlay must ride the existing atomic publish, not a follow-up write.
  - **Static-only, on the def — never on `StepRun`.** The def graph is tens of nodes (overlay edges
    are single-digit multiples → RAM non-issue, the inverse of a repo CPG where AST/CFG/REACHING_DEF
    fan-out dominates). The run graph is thousands of nodes and RAM-bound; "why did *this run* branch
    here" is a join through `RAN`, not a second copy of the layer.
  - **Honest `READS_UNKNOWN` for what can't be derived statically** — an `agent`/`llm`-guard node
    whose reads aren't extractable gets an explicit marker (Joern marks indirect calls the same way).
    A **feature**: it says precisely which parts of a flow are analyzable vs. trust-the-model. Do
    **not** attempt a probabilistic DDG — an unsound dependence edge produces confident-wrong impact
    answers, worse than none.
- **Owner:** **`graph-dba`** gates the FalkorDB model first (the overlay labels/edge types, whether a
  `CtxKey`-style node is per-def or shared, indexes) → **`architect`** designs the publish-time
  extraction pass + the three validations → **`coder`**/**`tdd-engineer`**. A CPG-model design note
  would land at `falkor-chat/docs/plans/<slug>-graph.md` (graph-dba convention).
- **Scope sketch (to be designed, not decided here):** first slice = extract read/write sets from
  `cmp` guard paths (`ctx.`/`output.` roots) + `config.expects` at publish → materialize
  `(:Step)-[:READS]->` / `(:Step)-[:WRITES]->` a key node → add the **dangling-read** and
  **unreachable-step** publish validations (closes n-3 the graph way). `llm`-guard reads and the
  change-impact query are follow-on slices. No DDL beyond one node label + its index; no rule-8
  violation; no run-side cost.
- **Relationship to neighbours:** complements **K-031** (that exposes def *structure* for reading;
  this *analyzes* it), and overlaps **K-029**'s symmetric-`decision` invariant (K-029 proposes the
  rule; K-032 proposes the graph mechanism that could enforce it). Not an M3-green gate — M3 is ✅.
- **Risks/RAM:** negligible on the def side (see the static-only constraint). Real risk is *scope
  creep* toward a general expression/data-flow engine — the `expr` seam stays a `NotImplementedError`;
  this rides the existing closed `cmp` family only.
- **Test strategy:** offline contract tests — a published def reads back with the exact READS/WRITES
  overlay for its guards/`expects`; a def with a guard reading an unwritten key is **rejected at
  publish**; an unreachable step is **rejected at publish**; a step whose reads can't be derived
  carries `READS_UNKNOWN` rather than silently claiming zero reads.

### K-033 — Make `maxSteps` an exact cap (`>` → `>=` in `_drive_loop`) (🔵 proposed — filed out of K-031, stakeholder decision OQ-1 "document now, fix later", 2026-07-24)

> **Why it exists.** `maxSteps` does not mean what its name says. `executor._drive_loop` records a
> step and *then* checks `rec["stepCount"] > max_steps` — at `executor.py:410` (OUTCOME A, a guard
> fired) and `:427` (OUTCOME C, a legitimate self-loop). With `maxSteps: 2`: step 1 → `1 > 2` false;
> step 2 → `2 > 2` false; **step 3 runs**, `3 > 2` → fail. So the budget means *"at least N, then one
> more"*, and a run executes at most **`maxSteps + 1`** steps. Confirmed by reading and **pinned by a
> passing test** — `tests/test_executor.py:158`, `assert len(trail) == 4  # maxSteps=3 → the 4th
> advance trips the guard`. Harmless for the two proof defs (8/6/6 steps against `maxSteps: 24`), but
> it makes `maxSteps` unusable as an SLA or a cost budget, which is exactly what a caller reaches for
> it for. K-031 shipped the **documentation** of the real semantics at six sites (DESIGN §6,
> QUERIES §12.5 + the two `$maxSteps` comments, `schemas.py`, `AGENTS.md`'s executor-invariants
> block) per the binding stakeholder decision; this item is the **fix**.
- **The change is two characters, in two places** — `>` → `>=` at `executor.py:410` and `:427`.
  Everything else about this item is ceremony, and the ceremony is the reason it was deferred out of
  an observability slice rather than the difficulty of the edit.
- **Both sites are *inside* the SHA-locked `_drive_loop`** (`71055f756280`, still live on the
  tree — recompute with the DESIGN §6.2 recipe). Landing it therefore costs:
  - a lock break + **re-lock ceremony**: recompute the SHA, then re-quote it everywhere it is
    asserted. Scope it with `grep -rn 71055f756280` rather than a count here — the number only
    grows, and it is now well past forty sites across plans, reviews and test plans;
  - **records that assert the SHA was unchanged during their own work must not be rewritten** —
    every `archived` plan/review that quotes it is a historical claim about its own delivery, true
    when written. The re-lock has to be expressed as *"as of K-033 the lock is `<new>`; earlier
    records quote the pre-K-033 value"* — i.e. the lock stops being a single grep-able constant.
    **Decide that framing before editing**; it is the item's only real design question.
  - **test edits**: `tests/test_executor.py:142-158` (the pinned count 4 → 3, and its explanatory
    comment), plus a sweep of `tests/test_process_flow.py`'s step accounting and the
    `access-request@v1` `maxSteps: 24` headroom;
  - **behavioural blast radius**: every existing run's effective budget shrinks by one.
- **There is no bundling opportunity left — settled 2026-08-25.** This item was filed hoping to
  ride the next change that legitimately broke the `_drive_loop` lock, plausibly K-027 item 2 (the
  terminal-node-must-post engine contract). K-027 closed 2026-08-21 and **item 2 shipped without
  breaking the lock** — it was implemented at `_run_agent_node`, outside it, exactly as this item's
  original scepticism predicted. K-033 now pays the full re-lock ceremony on its own, or the honest
  `maxSteps + 1` prose stays in six documents indefinitely. That is the decision to make; nothing is
  waiting on anything else.
- **Also decide (part of the item, not a separate one):** whether the *park* path (OUTCOME B,
  `executor.py:415-421`) and the terminal path stay deliberately unchecked. They are unchecked today
  by design — a parked run cannot self-drive — and K-031's documentation says so explicitly, so
  changing that is a second semantic decision, not a consequence of `>=`.
- **Owner:** **`tdd-engineer`** (a behaviour change pinned by an existing passing test is the
  test-first shape: flip the assertion red, then the operator) — with an **`architect`** call first
  **only** on the re-lock framing above, if the coordinator wants the archive-document question
  settled before code moves. No `graph-dba` gate: no Cypher, no DDL, no index.
- **Risks/RAM:** **none** — no new node type, label, property, index or vector dimension; no query
  changes. The risk is purely behavioural (every run's budget shrinks by one) and procedural (the
  lock/archive framing).
- **Test strategy:** flip `tests/test_executor.py:142-158` to assert exactly `maxSteps` advances
  (3, not 4) and watch it go red before the fix; add the boundary case at `maxSteps = 1`; assert the
  park path is still **not** budget-checked (a parked run at `stepCount == maxSteps` must stay
  `waiting`, not fail); re-run `tests/test_process_flow.py` and the `access-request@v1` acceptance
  flow to confirm the `maxSteps: 24` headroom still covers it.

### K-035 — An argument key named `name`/`action`/`tool` shadows the bare call's own name (🔵 proposed — filed out of the K-027 slice A analyst gate, finding M-2, 2026-07-24)

> **Why it exists.** In `llm._parse_content_tool_calls` the **JSON probe runs before** the bare-call
> probe, and `_normalize_tool_call` maps `name` / `action` / `tool` loosely. So for a bare call whose
> *argument object* happens to carry one of those keys, the argument is mistaken for the call
> envelope and the real call name is never seen:
>
> | model emits | parsed as |
> |---|---|
> | `create_user({"name": "bob"})` | `ToolCall(name='bob', arguments={})` |
> | `run_tool({"action": "delete"})` | `ToolCall(name='delete', arguments={})` |
> | `x({"tool": "y", "args": {"a": 1}})` | `ToolCall(name='y', arguments={"a": 1})` |
>
> Reproduced by `analyst` at the K-027 slice A gate; the three rows above are verbatim from
> [`docs/reviews/k027-parse-robustness.md`](reviews/k027-parse-robustness.md) **M-2**.
- **Not currently reachable, and that is the whole risk.** No registered tool declares such a
  parameter — `post_message` takes `text`/`mentions`, `graphrag_retrieve` takes `query`,
  `human_handoff` takes `reason` (`server/falkorchat/tools.py`). The premise was verified at the
  gate and **deferring the fix was judged correct** for a slice whose point was not to widen
  precedence. It is filed as its own item because the *shape* of the failure is bad, not because the
  likelihood is high: it manufactures a tool call **named after a user-supplied value**, so it fails
  **silently and misleadingly** — `executor._handle_tool_call`'s AC-6 check rejects `'bob'` as an
  ungranted tool, burns a re-prompt iteration, and tells the model its own argument is not a tool.
  That trace is close to undebuggable, and a bullet under "candidate follow-ups, small" would not
  have surfaced at the moment someone registers `create_user(name)`.
- **Tripwire in place (do not remove).** `server/falkorchat/llm.py` carries a comment at the
  probe-order site naming this item, because that is where the next author will be looking. **Read
  this item before registering any tool with a `name`, `action` or `tool` parameter.**
- **Owner:** **`architect`** to pick the remedy (it is a precedence decision, not a bug fix), then
  **`tdd-engineer`**. Candidates, cheapest first: (1) in `_normalize_tool_call`, skip the loose
  `name`/`action`/`tool` mapping when the surrounding content also matches `_BARE_CALL_OPEN` — a
  partial hardening available today; (2) run the bare-call probe **first**, which reorders the
  content fallback and needs its own regression pass over the JSON shapes; (3) pass the granted tool
  names down as a **recognition filter** — this also closes most of M-1's residual and the
  `Summary ({"a": 1})` class, but it is a real layering decision (`llm.py`'s note that name
  validation belongs to the agent loop is deliberate, and worth keeping unless consciously reversed).
  Open question 3 of the K-027 gate review is exactly this choice.
- **Risks/RAM (rule 6):** none — parse layer only, no node type, index, property or vector dimension.
  No Cypher, no schema, no script.
- **Test strategy:** offline pins in `server/tests/test_llm.py` driving the public `llm.chat(...)`
  seam (never the private probes). Pin all three rows in the table above to the **call's own**
  identifier; pin that a genuine `{"name": …, "arguments": …}` envelope with no surrounding call
  expression still parses as a call (the shape this must not regress); and pin the negative
  direction — an argument object carrying `name` must not resurrect a call the M-1 final-content rule
  rejected.

### K-038 — `refreshRunPanel` has no mutex against overlapping poll-tick/submit-response invocations (🔵 proposed — filed out of K-036's Wave 3+4 analyst re-review gate, `docs/reviews/web-api-coverage-impl.md` Pass 3, findings m6/m7, 2026-07-29)

> **Why it exists.** Pass 3 fixed M1 (the destructive every-tick `renderWaitingForm` rebuild) and,
> while deep-tracing the fix per the task's own request, found two narrower, non-blocking races in
> the same run-panel poll/submit machinery — recorded there rather than chased at gate time, and
> filed here as this item per the K-036 close-out plan.
- **m6 — unordered concurrent `refreshRunPanel` calls; a stale response can transiently overwrite
  a fresher one.** `refreshRunPanel` (`web/app.js`) is not mutex-protected against overlapping
  invocations, and both `startRunPolling`'s periodic tick and `submitRunInput`'s post-submit call
  invoke it independently. If a tick's `refreshRunPanel` is still in flight
  (`Promise.all([GET run, GET step-runs])` awaiting) at the moment a submit resolves and triggers
  its own `refreshRunPanel`, both fetches race against the same `runId`, and whichever's
  `Promise.all` resolves *last* wins the render — regardless of which is fresher. A stale
  (pre-submit) response landing after the fresh one briefly shows an already-superseded step.
- **m7 — the same-step-key rebuild guard can't distinguish "still the same wait" from "revisited
  after an externally-driven round-trip this session's poll never observed."** The
  `state.runWaitingKey` guard (M1's fix) only resets when *this session's own* render observes a
  non-`waiting` status. If some other actor completes a full `waiting → running → waiting`
  round-trip on the *same* `atStepKey` entirely between two of this session's poll ticks, the key
  still matches on the next tick and the box stays hidden behind the early-return guard even
  though it's a genuinely new visit to that step.
- **Non-blocking, self-healing — carry this framing accurately.** Both require timing conditions
  well outside normal single-operator, human-typing-speed form use (a sub-`POLL_MS`, i.e. <3s,
  window for m6; an external actor plus a round-trip faster than one `POLL_MS` for m7), and both
  self-heal within one more `POLL_MS` (≤3s) tick — the next poll fetches true current state again
  and forces a correct rebuild either way. Neither loses user-entered input (unlike M1); neither is
  a permanent inconsistency.
- **Owner:** **`frontend-engineer`**, `web/app.js`.
- **Scope sketch (to be designed, not decided here):** m6 — a request-sequence token (stamp each
  `refreshRunPanel` call with an incrementing counter, ignore a response whose stamp is behind the
  latest issued) would close it; m7 — needs a server-supplied, monotonically-changing identifier
  per wait-instance (not just `atStepKey`) to distinguish "unchanged wait" from "revisited wait",
  which is a server-side surface change, not JS-only.
- **Risks/RAM:** none — web-only, no server/graph surface touched by m6; m7's full fix would need a
  new/changed field on the run-detail response, scoped by whoever picks this up, not decided here.
- **Test strategy:** a DOM-stub harness (the same shape Pass 2's fix verification used) driving two
  overlapping `refreshRunPanel` promises resolving out of order, asserting the later-issued one
  always wins regardless of resolution order (m6); an injected-clock or mocked-response test
  simulating an external round-trip between two ticks, asserting the panel still rebuilds (m7).

### K-040 — `POST /workflow-runs`'s request field is `version`, not `defVersion` — decide whether to rename for consistency (🔵 proposed — found during a `tico` manual-verification pass, 2026-07-31)

> **Why it exists.** `StartWorkflowRunIn` (`server/falkorchat/schemas.py:198-202`) declares the
> field as `version`, while the rest of the def/snapshot vocabulary — `WorkflowRun.defVersion`
> (DESIGN §6.2), the `GET /workflow-defs/{key}/versions/{version}` path segment name notwithstanding,
> and the general "def key + version" phrasing throughout `DESIGN.md` §14.4/§12 — uses `defVersion`.
> The mismatch is easy to get wrong by pattern-matching the graph model rather than the actual
> schema: it produced a real `422 Unprocessable Entity` in a first draft of
> `falkor-chat/docs/manuals/workflows.md`'s API walkthrough, caught only because a `qa-engineer`
> verification pass actually called the endpoint rather than composing the example from the schema
> next to the design doc. Already fixed in the manual (`docs/manuals/workflows.md`, 2026-07-31); this
> item is about the underlying inconsistency, not the doc.
- **The actual decision, not pre-judged here:** either (a) rename `StartWorkflowRunIn.version` →
  `defVersion` for consistency with the rest of the surface — a live API contract change, so it
  needs an assessment of what currently depends on `version` (the web UI does not call this route
  today per `docs/manuals/workflows.md`'s Walkthrough 4 note that it's API-only; MCP tools don't
  cover workflow runs either per DESIGN §15.2 — so the blast radius may be small, but that needs
  confirming, not assuming) — or (b) leave the field as-is and treat this as a documented,
  intentional naming divergence (a one-line callout at `DESIGN.md` §14.4 / `QUERIES.md` §12.12 would
  suffice). Both are legitimate; this item exists so the choice is made deliberately rather than by
  the next person guessing again.
- **Owner:** **`architect`** — assess the rename's blast radius and decide (a) vs (b); if (a),
  **`coder`**/`tdd-engineer` implements + updates every call site and doc reference.
- **Scope:** grep every caller of `POST /workflow-runs` (tests, scripts, docs, the manual) before
  deciding; if renaming, land the schema change, the `test_process_flow.py`/`test_services.py` call
  sites, and the `docs/manuals/workflows.md` example in the same change.
- **Risks/RAM:** none — no graph/DDL surface; purely a request-schema field name. The only risk is a
  silently-missed caller if the rename is chosen without the grep above.
- **Test strategy:** if renamed, flip the existing `test_process_flow.py` start-run assertions to the
  new field name (should go red first, confirming the old name is actually gone) plus a negative test
  that the old `version` name is rejected, not silently ignored.

### K-043 — `compose.yaml`/`Dockerfile` never verified against a real `docker build`/`docker compose` (🔵 proposed — filed out of K-042 close, 2026-08-11)

> **Why it exists.** K-042 Landing 1's L1-5 unit updated `compose.yaml` (the two config-file paths,
> a read-only bind mount of the shared overlay file, `host.docker.internal:host-gateway`) and
> `Dockerfile`-adjacent run instructions on the strength of static review only — no Docker toolchain
> was available anywhere in the coordination pipeline (agents, gates, or the QA acceptance pass), so
> the change was never exercised by an actual `docker build` / `docker compose up`. The risk is
> narrow (a bind-mount path typo or a missing `host.docker.internal` extra_hosts entry would only
> surface in a real container run) but real, and it's the one surface in K-042 that shipped unverified.
- **Owner:** **`devops`** — build the image, bring the stack up via `compose.yaml`, and confirm the
  server inside the container can resolve both config-file paths and reach LM Studio on the host via
  `host.docker.internal`.
- **Scope:** `docker build` against `falkor-chat/Dockerfile`; `docker compose up` against
  `falkor-chat/compose.yaml`; confirm the bind-mounted shared overlay file is readable at the path
  the container expects, and that a chat request round-trips to the host LM Studio instance.
- **Risks/RAM:** none — verification only, no design change expected unless the run surfaces a defect.
- **Test strategy:** a live manual run is the test; if it surfaces a defect, file a fix as its own
  follow-up rather than folding it into this item.

### K-044 — Decide whether an admin manual (`docs/manuals/llm-provider-config.md`) is wanted (🔵 proposed — filed out of K-042 close, 2026-08-11)

> **Why it exists.** `tico` flagged, while archiving K-042's requirements document, that no
> end-user-facing manual was written for LLM provider/model configuration — the two config files,
> per-kind defaults, roles/fallback chains, workspace override precedence, and how to read the
> resolved model off a run's trace are all documented at requirements/plan altitude
> (`docs/requirements/llm-provider-config.md`, `docs/plans/llm-provider-config.md`) but not at the
> operator-facing altitude `docs/manuals/` is for (per root `AGENTS.md`'s documentation convention).
> This was never raised to the stakeholder as a decision point — it's an open call, not a commitment.
- **Owner:** **`tico`** — decide whether the audience (whoever hand-edits the two config files) needs
  a manual, or whether the existing `README.md` config section is sufficient; if yes, author
  `docs/manuals/llm-provider-config.md` per the family-slug convention (this feature's slug already
  spans requirements/plans/reviews/test-plans/test-reports, so a manual would join that family).
- **Scope:** a stakeholder check-in on whether this is wanted, then (if yes) write the manual,
  illustrated with a diagram of the precedence chain (workspace → step/agent/guard's own choice →
  per-kind default) where a picture beats prose.
- **Risks/RAM:** none — documentation only.
- **Test strategy:** N/A (docs); if written, a `qa-engineer` walkthrough of the manual's steps against
  the running system per the standard manual-verification pattern.

### K-045 — FR-10's requirements text ("the run suspends") is stale against the shipped `failed`-with-cause behavior (🔵 proposed — filed out of K-042 close, 2026-08-11)

> **Why it exists.** `docs/requirements/llm-provider-config.md` FR-10 reads "An unresolvable model
> encountered at **use time** fails loudly — the run suspends…". The shipped Landing 2 behavior
> (confirmed by U7's QA acceptance pass and D-2's fix) is that a use-time resolution/provider failure
> fails the run with a recorded cause (`status='failed'`), not a suspend/park state — the same
> failure vocabulary as the executor's other terminal faults, not the human/wait "suspend and wait for
> a signal" semantics FR-10's wording evokes. `tico` flagged the drift while flipping the requirements
> document's header to `Status: archived`, but an archived document's metadata-only edit (the one
> kind of edit `archived` permits) cannot fix stale body text.
- **Owner:** **`tico`** — since the source document is `archived`, this needs a deliberate choice per
  root `AGENTS.md`'s collision rules: either a successor requirements document (new document,
  `Supersedes:`/`Superseded by:` pointers) correcting the wording, or a deliberate un-archive (only if
  the original owner chain agrees the fix is a trivial in-place correction rather than a substantive
  change) — not a silent edit to the archived file.
- **Scope:** correct FR-10's language to match the shipped `failed`-with-cause behavior; no code
  change implied — this is a documentation-accuracy item only.
- **Risks/RAM:** none — documentation only.
- **Test strategy:** N/A (docs).

### K-048 — `executor._assemble_messages`'s unconditional trailing `user`-role `CONTEXT` block breaks strict-alternation chat templates (🔵 proposed — found during the K-027 item 5 Ministral re-probe, `data-scientist`, 2026-08-20)

> **Why it exists.** `WorkflowExecutor._assemble_messages` (`server/falkorchat/executor.py:910-931`)
> builds an `agent`-typed node's opening message list as: the node's `systemPrompt` (`role: system`),
> then the recent thread turns role-mapped `user`/`assistant`, then **one more `role: user` message
> appended unconditionally** — a `"CONTEXT:\n{...}"` block carrying the run's serialized state. If
> the thread's last turn before this block is itself `user`-authored, the request ends with two
> consecutive `user`-role messages. This is **structural, not incidental**: the message that starts
> a run is a `user`-role trigger turn, so `intake`'s very first call already has this shape by
> construction; `research` (granted only `graphrag_retrieve`, never `post_message`) never posts an
> assistant-visible turn before `answer` runs in the same drive, so the same shape recurs there too.
> **Confirmed live**, not assumed: replaying `intake`'s exact system prompt + trigger message +
> `CONTEXT` block against LM Studio's `mistralai/ministral-3-3b` (and its alias
> `mistralai_ministral-3-3b-instruct-2512`) returns a hard `HTTP 400`, with the underlying Jinja
> template error surfaced verbatim: *"After the optional system message, conversation roles must
> alternate user and assistant roles except for tool calls and results."* A minimal 3-message repro
> (`system`, `user`, `user`) reproduces the same failure on both catalog ids; the identical shape
> against `qwen/qwen3-4b-2507` succeeds cleanly (`finish_reason: tool_calls`). Traced through the
> code, not just observed at the boundary: `falkorchat/transport.py`'s `urllib.error.HTTPError` rung
> wraps the 400 into a `ProviderCallError`; `executor._drive`'s `except Exception` (`executor.py:
> 447-449`) catches it, calls `_fail_with_note` (`fail_run`), and **re-raises** — so a live run on an
> affected model fails loudly on its very first LLM call, not gracefully. Full evidence trail:
> `docs/plans/ministral-reprobe-ml.md` §4.2 (the live-verified crash, both catalog ids, the exact
> error text, the traced fault-net path) and §4.4 (why K-039's implicit-`post_message`-dispatch
> fallback, shipped 2026-07-31, makes this matter less for Qwen today — Qwen's template tolerates
> the shape, so it never needs the fallback for *this* reason — but would matter for any future
> multi-vendor model-portability goal). Independently confirmed at the `analyst` gate on that note
> (`docs/reviews/ministral-reprobe.md`, verdict approve).
- **Model-agnostic, independent of the Ministral decision.** The trigger was evaluating Ministral as
  a candidate model (declined — see K-027 item 5), but the defect is in the message-assembly
  convention itself: *any* strict-alternation chat template (confirmed here for LM Studio's
  Mistral-family serving; plausibly other vendors/templates too, unverified) would hit the same
  crash on `intake`'s first call, regardless of why that model was chosen.
- **Not inside the SHA-locked `_drive_loop`** — verified directly, not assumed, since the K-033
  precedent established the lock covers `_drive_loop` only and named `_execute_step`/
  `_select_transition`/`_trace_step`/`resume` as outside it. `_assemble_messages` (executor.py:910)
  and its caller `_run_agent_node` (executor.py:615) sit well outside the locked span
  (`_drive_loop` is executor.py:451-514); re-ran the lock's own reproducible recipe
  (`docs/DESIGN.md` §9's awk one-liner) on the current tree and confirmed the hash is still
  `71055f756280`, unchanged. A fix here needs no re-lock ceremony.
- **Owner:** **`architect`** to pick the remedy (a message-shape decision, not a one-line bug fix —
  candidates include merging the `CONTEXT` block into the prior turn when it would otherwise be
  same-role, or folding thread turns and `CONTEXT` into a single trailing `user` message
  unconditionally), then **`tdd-engineer`**.
- **Scope:** `server/falkorchat/executor.py`'s `_assemble_messages` only; no Cypher, no schema, no
  index, no vector dimension (rule 6) — pure message-list construction.
- **Risks/RAM:** none structural. Behavioural risk: changing how thread context is folded into the
  opening message list could alter what a *tolerant* model (Qwen) sees too, so a fix needs a
  regression pass against the existing live triage flow, not just a Ministral-shaped test case.
- **Test strategy:** an offline unit test asserting `_assemble_messages` never emits two consecutive
  same-role messages for a thread whose last turn is `user`-authored (the exact shape that crashes
  today); a live/replay-level regression check (mirroring `docs/plans/ministral-reprobe-ml.md` §4.2's
  own repro) that the fixed shape no longer 400s against a strict-alternation template, without
  requiring a live Ministral instance to be part of the standing suite.

### K-049 — An oversized value on an *indexed* graph property crashes the shared FalkorDB instance outright (🔵 proposed — found during the K-028 workflow-timers implementation, `coder`, 2026-08-21)

> **Why it exists.** While building K-028's ctx-merge length-bound test (`test_workflow_timers.py`),
> `coder`'s first attempt used a deliberately oversized **`Step.key`** — an indexed/constrained graph
> property (`falkor-chat/AGENTS.md`'s schema convention: "every entity node has a stable
> `{label}Id` property, a range index, and a uniqueness constraint") — to trigger the length check.
> Publishing that def **crashed the shared dev `falkordb-dev` container outright**: the connection
> dropped mid-request and the `--rm` container vanished entirely from `docker ps -a`, reproduced
> **twice** across independent restarts. Root cause not established — `--rm` meant no logs survived
> the crash instant. `coder` worked around it for K-028's own purposes by switching the trigger to
> an oversized **ctx** value instead (opaque, unindexed, confirmed safe) — K-028 itself never writes
> an unbounded value to an indexed property, so this is not a K-028 defect, but the underlying engine
> behavior is real and unresolved.
- **Why it matters beyond K-028.** This is a **shared-instance** reliability risk: the same
  `falkordb-dev` container also hosts `cpg_falkorchat`, `kaizen_team`, and every workspace graph any
  agent is actively using. Any future def/data with an oversized value on *any* indexed/constrained
  property (not just `Step.key`) could reproduce this — a crash-on-write footgun with a blast radius
  well outside whatever feature happens to trigger it.
- **Owner:** **`graph-dba`** — root-cause (reproduce deliberately in an isolated/throwaway container,
  not the shared dev instance; capture logs by dropping `--rm` for the repro run; identify whether
  this is a FalkorDB engine limit, a resource-exhaustion crash, or something else) → harden (an
  app-side length guard before any indexed-property write reaches the graph, and/or an upstream
  FalkorDB issue filing per the existing `K-007 OQ6`-style precedent for confirmed engine anomalies).
- **Scope sketch (to be designed, not decided here):** confirm reproducibility in isolation; bound
  the actual failure mode (a specific length threshold? any oversized value? specific to
  indexed/constrained properties or broader?); decide the fix layer — likely an app-side
  `MAX_*_LEN`-style guard on every indexed/constrained-property write path (`services.py`'s existing
  `MAX_ID_LEN`/`MAX_CONFIG_LEN` precedent), mirrored to cover step keys, def keys, and any other
  indexed identifier, not just workflow `ctx`/`config`.
- **Risks:** the shared dev instance itself — reproduce only in a disposable/isolated container, never
  against `falkordb-dev` while other agents may depend on it (`kaizen_team`, `cpg_falkorchat`, live
  workspaces).
- **Test strategy:** an isolated-container repro proving the crash and capturing logs (no `--rm`);
  once root-caused, a regression test proving the chosen guard rejects the offending shape before it
  ever reaches a graph write.

## Deferred — M2.5 hardening track

Auth + real-time. Not on the M5 path; no scheduled start.

### K-016 — Real auth/tenancy replacing the hardcoded `get_context` seam (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design pass — designs the auth mechanism *per* the authoritative-identity decision, now
  resolved: the `identity` graph is authoritative/standalone, DESIGN §1.2) → **`tdd-engineer`** (implement the resolved `get_context`).
- **Inputs/prereqs:** the identity source-of-truth is **decided** (identity graph authoritative/standalone; DESIGN §1.2) —
  K-016 no longer needs the user for that axis; it implements per that decision. Localized by design — only
  `config.get_context` changes (`config.py:43`); everything below already parameterized on `ws`/`actor`.
- **Scope:** token → (user, workspace claim) resolution replacing hardcoded `ws=acme/user=u1`; wire the `identity`
  graph per the §1.2 authoritative-identity decision; keep or replace MCP's `frm`-ignoring rule with authenticated agent identity.
- **Done-condition:** `get_context` resolves a real principal from a credential; multi-tenant isolation test; pytest green.
- **Risks/RAM:** `identity` graph nodes (small). First real trust boundary — MCP endpoint is currently unauthenticated (§15.3).
- **Test strategy:** service/api tests with injected auth contexts; a cross-tenant isolation test.

### K-017 — Transport-level agent-actor path (K-007 QA carry-over) (🔵 proposed — M2.5, deferred · depends on K-016)

- **Owner:** `qa-engineer` (+ small `tdd-engineer`/`coder` fold-in if MCP must express an authenticated agent actor).
- **Scope:** with auth able to express an *agent* principal, drive an external agent authoring over MCP/REST (the M1
  hardcoded seam couldn't) and verify authorship/role/provenance end-to-end.
- **Done-condition:** the K-007 QA carry-over closed — a report showing an externally-authenticated agent authoring
  first-class over the transport.
- **Risks/RAM:** none new. **Test strategy:** black-box over MCP with an agent credential.

### K-018 — Real-time push (Redis Pub/Sub → WebSocket/SSE) (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design: Pub/Sub fan-out topology; resolve the DESIGN §13 Bolt-vs-RESP gateway question
  here since it touches the transport) → **`coder`/`tdd-engineer`**.
- **Inputs/prereqs:** K-012/K-014 web client (swap polling → push).
- **Scope:** Redis Pub/Sub on message write → WebSocket/SSE endpoint on the same FastAPI process (§14.1: "slots onto
  the same service layer, no schema change") → web client subscribes instead of polling.
- **Done-condition:** a posted message appears in another client without a poll; graceful fallback to polling.
- **Risks/RAM:** no graph RAM; Pub/Sub is transient. Publish *after* the guarded §4 write commits, never inside it (atomicity rule).
- **Test strategy:** integration test of publish-on-write + a WebSocket client receiving it.
- **Related work (client-side polling alternative):** `mcp-monitor/` (`mcp-monitor/docs/requirements/mcp-monitor.md`) has shipped as a separate, polling-based watcher that detects MCP tool-result changes and launches commands — a distinct, complementary approach to K-018's server-side push. K-018 remains its own open item.

## Delivered

> Full write-ups — what changed, when, which gates, which baselines — are in
> [`HISTORY.md`](./HISTORY.md), one dated entry per item. This table is the index only; a
> delivered item's body is not kept here.

| Item | Title | Delivered | M |
|---|---|---|---|
| **K-008** | GraphRAG retrieval core — embedding worker, vector index @1024, hybrid retrieval read path | 2026-07-08 | M2 |
| **K-011** | M1 DoD closeout — append-path load harness, hot-read PROFILE, per-workspace RAM budget | 2026-07-06 | M1 |
| **K-012** | Web request/response UX polish ⇒ **M1 ✅** | 2026-07-06 | M1 |
| **K-013** | AI `Agent` participant with `EMITTED` provenance | 2026-07-08 | M2 |
| **K-014** | Web M2 — render agent replies + reader `isMention` highlighting | 2026-07-08 | M2 |
| **K-015** | QA acceptance pass on M2 GraphRAG — PASS, zero defects ⇒ **M2 ✅** | 2026-07-08 | M2 |
| **K-019** | Documentation-inconsistency sweep (stale test counts, §13 embedding drift, real-time wording) | 2026-07-05 | — |
| **K-020** | Workflow definition model in `reference` | 2026-07-09 | M3 |
| **K-021** | Snapshot materialization into `ws:{id}` on publish | 2026-07-09 | M3 |
| **K-022** | Run + StepRun executor core — Landing 1 (offline executor + capabilities), Landing 2 (trigger, `PRODUCED` linking, fault net) | 2026-07-12 / 2026-07-19 | M3 |
| **K-023** | Workflow ↔ chat linkage — `TRIGGERED_BY` + `PRODUCED` (D2); closed inside K-022 Landing 2 | 2026-07-19 | M3 |
| **K-024** | Proof flows — one conversational (`triage@v1`), one business-process (`access-request@v1`) | 2026-07-19 / 2026-07-21 | M3 |
| **K-025** | QA acceptance pass on M3 — PASS with parked, model-gated limitations ⇒ **M3 ✅** | 2026-07-21 | M3 |
| **K-026** | GraphRAG retrieval + generation evaluation harness — QA-accepted, PASS | 2026-08-16 | M2.5-quality |
| **K-027** | Live triage reliability — judge-parse robustness, must-post engine contract, judge calibration, golden-set expansion (26→85), Ministral re-probe, six carried gate findings | 2026-07-24 → 2026-08-21 | M3 follow-up |
| **K-028** | Workflow timers / scheduled wakeups — QA-accepted | 2026-08-21 | M3 follow-up |
| **K-031** | Def/snapshot structure read surface — makes the create-only split-brain detectable | 2026-07-24 | M3 follow-up |
| **K-034** | Create-only re-publish is additive, not a no-op — topology-conflict gate + the doc-site sweep | 2026-08-01 | M3 follow-up |
| **K-036** | Web API Coverage — defs viewer, run cue + detail panel, structured-input resume, participants, readiness banner ⇒ **M3.5 ✅** | 2026-07-29 | M3.5 |
| **K-037** | `FALKORCHAT_TRIGGER_DEF_KEY` grafted `triage`'s steps onto `access-request@v1` — decoupled | 2026-07-30 | M3.5 follow-up |
| **K-039** | `@mention`→`triage@v1` completed `done` posting zero replies — implicit `post_message` dispatch fallback + CI readiness signal | 2026-07-31 | M3.5 follow-up |
| **K-041** | MCP `send_message` never scheduled the responder/workflow trigger | 2026-08-01 | M3.5 follow-up |
| **K-042** | LLM provider & model configuration — two config files, one `ModelGateway` seam, per-consumer model choice ⇒ **M4 ✅** | 2026-08-11 | M4 |
| **K-046** | Root `conftest.py`'s `_falkordb_reachable()` write-mode `GRAPH.QUERY` bug | 2026-08-16 | — |
| **K-047** | `generate_report.py` rendering/branching test coverage | 2026-08-16 | — |

## Plan docs still to author

Every other plan named by a delivered item now exists; this is what is left.

| Path | For | Scope |
|---|---|---|
| `docs/plans/auth-tenancy.md` | K-016 (deferred) | Real auth replacing the `get_context` seam, per the §1.2 identity-authoritative decision. |
| `docs/plans/realtime-push.md` | K-018 (deferred) | Pub/Sub → WebSocket/SSE, resolving DESIGN §13's Bolt-vs-RESP question. |

> Both slugs were previously listed as `m2-auth-tenancy.md` / `m2-realtime.md`. Renamed here before
> creation: root `AGENTS.md`'s filename grammar forbids a new document's basename beginning with
> `m<digit>`, and neither topic *is* a milestone.

## Parking lot / ideas

- Verify the K-009 GitHub Action goes green on first push (path-filtered `.github/workflows/falkor-chat.yml`; FalkorDB
  service container). Note the CI baseline echoes in its comments (75/92) predate K-007/K-010's 110/126 — the suites
  themselves are the source of truth. (K-019 fixes the README/DESIGN body numbers; the CI comments are separate.)
- File upstream FalkorDB issues (K-007 OQ6, recommended to the user): `GRAPH.MEMORY USAGE` under-reports vector-index
  memory; one-shot instant-timeout anomaly after a long override run.
- Per-endpoint response schemas (QA, recommended three times now): full-thread / since-reads / search each carry a
  different field subset (all documented/intentional) — a declared schema per endpoint would make the contract testable
  and stop accretion. **Status update (K-031, 2026-07-24):** the three new §11 structure/diff routes **do** declare
  `response_model=` (`WorkflowDefStructureOut` / `WorkflowDiffOut`, `schemas.py`) with exact-key-set contract tests; every
  pre-existing route still does not. That is a **deliberate** non-retrofit — FastAPI's `response_model` *filters*
  undeclared fields, so a wrong model silently drops a field the web client reads — and it leaves the repo with a **mixed
  convention**. This entry stays open; the new routes are the worked precedent for the eventual retrofit, not the retrofit.
- **Opportunistic nit — re-slug the K-031 implementation review** (recorded, **not** scheduled work).
  It is filed under the slug `k031-structure-read-impl`, while the rest of its family — the plan and
  the plan review — uses `workflow-def-structure-read`. The filename grammar's family rule (*the same
  slug across several kinds **is** the family; a downstream document inventing a new slug is a
  defect*) is therefore broken by one member. Correcting it to `workflow-def-structure-read-impl`
  spans **four files** as of 2026-08-25 (this backlog, `docs/HISTORY.md`,
  `docs/plans/m3-followups-coordination.md`, `docs/plans/workflow-republish-semantics.md`) — check
  with `grep -rln k031-structure-read-impl docs/` rather than trusting that count. Fold it into a
  change that already opens them; it does not earn a change of its own, and renames in this repo are
  forward-only by ruling.
- DESIGN §13 remaining open questions — resolve as their milestones arrive: real auth (K-016),
  message/embedding retention, cross-workspace analytics, Bolt vs RESP for the gateway (K-018).
  (The workflow guard expression language is **no longer** among them — resolved at DESIGN §6.1,
  and §13 records it struck through.)
- **WSL2 memory cap for the 16GB host** (parked, not applied per user 2026-07-18) — WSL2 runs uncapped at its 8GB
  default (50% of the 16GB host) with `autoMemoryReclaim` off, overcommitting host RAM alongside Windows-side LM Studio;
  likely root cause of the recent memory-overload crashes. Parked fix: set `memory=6GB` + `swap=4GB` +
  `autoMemoryReclaim=gradual` in `C:\Users\mauri\.wslconfig` (keep `networkingMode=mirrored`), then `wsl --shutdown`.
  Full diagnostic + apply procedure: `docs/plans/wsl2-memory-diagnostic.md`. Un-park (apply) if the crashes recur —
  verdict was confirmed-by-defaults, not reproduced live (FalkorDB was down during the diagnostic).
