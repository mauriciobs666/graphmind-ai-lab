# LLM Provider & Model Configuration — Graph Design

> **Status:** archived · **Owner:** `graph-dba` · **Tracks:** K-042 (M4) · **Version:** 4

Graph-side design note for `docs/requirements/llm-provider-config.md`. Scope is the three
graph questions in that feature: **FR-8** (record the resolved concrete model on the execution
trace), **FR-16/FR-17** (workspace-level model override + precedence), **FR-19** (embedding
dimension guard) — acceptance criteria **AC-4, AC-6, AC-9, AC-10, AC-11**.

**This note owns graph storage and query mechanics only.** The resolver seam, config-file
layering/parsing, precedence *implementation* and publish-time validation are
`docs/plans/llm-provider-config.md` (`architect`). §6 states the interface contract between the
two, and **§6.5 answers that plan's §8 point by point** in its own numbering. The two documents
agree on every substantive decision; the single divergence is a property name (§1.3). *(v1 also
flagged the override's shape, §7-Q1, as an open disagreement candidate; the `analyst` gate upheld
the per-kind design as a faithful reading of FR-16, and v2 withdraws the question — §2.1, §7-Q1.)*

**Sequencing:** every decision here lands in **Landing 2** (FR-7..FR-10, FR-16..FR-19). This is
**design-ahead**, not immediately-implemented — Landing 1 ships the resolver seam and per-kind
defaults with **no graph change at all**. Where a Landing-1 choice would foreclose a Landing-2
option, it is called out inline (§6.4).

---

## 0. Verification standing

Everything marked **[verified]** was executed on **2026-08-10** against the live pinned instance
(`falkordb/falkordb:v4.18.11`, graph module reporting `41811`, `vectorset` also loaded), using
`GRAPH.RO_QUERY` against shared graphs and a throwaway `ws:probe-gdba-*` for anything mutating —
the pattern in `falkor-chat/AGENTS.md` "Probing shared graph state without mutating it". The probe
graphs were torn down with `GRAPH.DELETE`; `./scripts/verify_workflows.sh acme` reports
`OK — 2 defs in sync` after the run, and `reference` / `ws:acme` were only ever read.
Everything marked **[proposed]** is design not yet executed. The full command log is §8.

One correction landed in `claude/graph-dba/falkordb-quirks.md` as part of this work: that file
claimed `db.indexes()` **does not** expose a vector index's dimension. That was recorded against
the *edge* build (module 999999) and is **false on the pinned build** — §3 depends on it being
true, so it was re-probed and the entry rewritten with the 2026-08-10 evidence.

**v2 revision (2026-08-10)**, responding to `docs/reviews/llm-provider-config.md`:

- **B-1 / A-5 (blocker, accepted in full):** §2.6's read-site guidance was wrong — `Executor.run`/
  `resume` never read a snapshot; the actual convergence point is `_drive`, outside the SHA lock.
  §2.6 now names the correct call site, the correct carrier (`run["ws"]`/`run["modelOverrides"]`
  stamped on the `run` dict), and why it is the *only* lock-free way the `guard` kind — which has
  no `CallContext` — can see a workspace override at all.
- **A-2 (accepted, reasoning only):** §2.1's per-kind conclusion stands, but the v1 justification
  ("a blanket override breaks FR-19 by construction") was overstated and is withdrawn; the
  correct, narrower argument is in §2.1, and §7-Q1 is withdrawn as a stakeholder question with it.
- **m-4 (accepted):** §3.3's `Chunk` check moved to layer 1 (startup assertion, operator warning
  only) and each write-time layer now gates on the label actually being written — `grep -rn
  'Chunk' server/falkorchat/*.py` finds no real usage, so gating a `Message` write on `Chunk`'s
  index was a refusal with nothing behind it.
- **m-5 (accepted):** `modelSource` cannot show that an FR-18 fallback occurred — §1.3/§6.2 now add
  a fourth, boolean `StepRun` property (`modelFallback`) for that orthogonal fact, rather than
  overloading `modelSource`.
- **m-6 (accepted):** a single `StepRun.resolvedModel` cannot represent a step whose iterations
  answered on two different models (FR-18 chain mid-loop). §1.6 gains an addendum recording the
  same rule already used for guards — **last answering model wins** — next to the guard
  discussion it mirrors.
- New **§8 Live verification log** re-runs the load-bearing checks (trace properties, the
  `WorkspaceConfig` node, `db.indexes()` dimension exposure) against the pinned build and records
  the commands, so the [verified] claims above are independently re-checkable.

Two things re-checked and confirmed still correct after the above: the per-kind (not blanket)
override design (§2.1, corrected reasoning, same conclusion), and `StepRun.resolvedModel`/
`modelSource` naming used consistently throughout (the A-1 ruling in this note's favor).

**v3 revision (2026-08-10)**, responding to the Pass 2 `analyst` re-gate (`docs/reviews/llm-provider-config.md`
`## Pass 2`):

- **Minor (routed to `graph-dba`) — one stale restatement in §6.5, now fixed.** §2.1 and §7-Q1
  correctly withdrew the overstated "a blanket override breaks FR-19 by construction... the two
  requirements would be in direct contradiction" framing in v2, but §6.5's own §8.2 answer (the
  wildcard-rejection bullet) was not touched by that revision and still carried the withdrawn
  sentence verbatim — while citing §2.1 as though it still supported the claim §2.1 now
  contradicts. That bullet is corrected to state §2.1's actual argument (a blanket override makes
  a permanently-unembeddable configuration expressible, not a requirements contradiction) and now
  points to §2.1 for the full reasoning instead of restating a version of it that had drifted from
  the source it cites.
- **Not mine:** Pass 2's one blocker, P2-B (`-graph.md`'s `modelFallback` field from v2/m-5 was
  not yet adopted into `docs/plans/llm-provider-config.md`'s `StepRun` schema), is `architect`'s
  side of a cross-document disagreement and is being closed there in parallel. No change to this
  document was needed or made for P2-B — `modelFallback`'s design here (§1.3, §1.4, §6.2) is
  unchanged and was reconfirmed sound by the same Pass 2 pass.

**v4 revision (2026-08-11)**, responding to a finding from K-042 Landing 2's U11 unit
(`coder`, implementing FR-19's guard, and the independent `analyst` review gate — both reproduced
the same result live against `ws:test`, in the same session):

- **§3.2's edge-behaviour table, `User` row, corrected.** The row read "one row, `dim = NULL`" for
  a label with only `RANGE` indexes; the actual result is **zero rows**, identical in shape to the
  row below it ("label unknown to the graph"). Cause: `types[$prop]` is a map access on a property
  name (`embedding`) `User` has no index of any kind on — `User`'s only index is `userId`
  (`RANGE`) — so the lookup is a missing key, evaluating to `NULL`; `WHERE ... AND NULL = ['VECTOR']`
  is `NULL`, not `true`, so the `WHERE` filters the row out rather than passing it through with a
  `NULL` dimension. Re-verified independently for this revision: `GRAPH.RO_QUERY ws:test` against
  the §3.2 query filtered to `label = 'User'` returns zero rows; `CALL db.indexes() ... WHERE label
  = 'User' RETURN label, properties, types` confirms `User`'s only index is `{userId: [RANGE]}` —
  no `embedding` entry exists in its `types` map at all. This does not change the design or any
  downstream code: `repository.read_index_dimension` (Landing 2, U11) already collapses "zero rows"
  and "one row, `dim = NULL`" to the same `None`, so the guard's behavior was never in question —
  only this document's worked example.
- The genuinely-`NULL`-row shape is real but was mis-attributed to the wrong example; §3.2 now adds
  a footnote naming its actual precondition instead of claiming it for `User`.

---

## 1. FR-8 / AC-4, AC-6, AC-9, AC-10 — the resolved model on the execution trace

### 1.1 Decision

Record the resolved concrete model as **first-class scalar properties on `StepRun`**, written
inside the existing atomic advance. No new label, no new edge type, no new index.

```
(:StepRun {stepRunId, stepKey, status, startedAt, endedAt, input, output,
           resolvedModel,        // NEW — "<provider>/<model-id>", the model that ACTUALLY answered
           modelSource,          // NEW — 'workspace' | 'step' | 'default', which precedence rung won
           modelFallback})       // NEW in v2 (§1.3, m-5) — true iff resolvedModel is not the
                                  // kind's primary/first choice; absent when no fallback occurred
```

All three are **nullable and absent by default**. A step that made no LLM call (`decision`,
`human`, `wait`, and the `agent`-without-LLM offline stub) carries none of them.

**These are not inside `ctx` / `input` / `output`.** Rule 8 of `falkor-chat/AGENTS.md` forbids
designing queries that filter inside those three serialised strings; `resolvedModel` and
`modelSource` are ordinary graph properties, so they may be projected, `WHERE`-filtered,
aggregated and (if ever justified) indexed like any other scalar. **Rule 8 does not apply to
them.** This is the whole reason for putting the model on a property rather than folding it into
the `output` envelope — an audit field you cannot query is not an audit field.

### 1.2 Why a property — not a `TraceEvent`, and not a node

**Rejected first and hardest: `TraceEvent`.** The requirement's own wording ("recorded on the
workflow run's **execution trace**") reads as an instruction to use the `TraceEvent` /
`TRACED` machinery. It is not one, and building it there would silently fail four acceptance
criteria.

**[verified] by reading `server/falkorchat/executor.py:390`:**

```python
tracer = self._tracer if run["trace"] else _NULL_TRACER
```

`TraceEvent`s are **debug-only by construction** — DESIGN §6.2 states it as an invariant ("a
non-debug run writes **zero**"), `bootstrap_schema.sh` calls the `TraceEvent` DDL "inert RAM
until a debug run writes to it", and `read_trace` returns `[]` for a non-debug run (which AC-5
of the M3 executor feature actively *asserts*). An ordinary run has `run["trace"]` false, so a
`TraceEvent`-borne resolved model would exist **only for runs someone thought to debug** — and
the runs you need to audit are precisely the ones nobody flagged in advance. AC-4, AC-6, AC-9
and AC-10 would each pass in a debug test and fail in production.

FR-8's operative words are "**for any given execution**" and "remains accurate even after the
configuration is later changed". Both demand a **durable, unconditional** record. So the field
must live on something every run writes exactly once per step — which is `StepRun`, created by
`record_step_and_advance` on every run regardless of the trace flag.

*(This note reached that conclusion independently before `architect` raised it; the two designs
agree. `architect`'s plan §2.6 and L2-2 carry the same finding. A `model_resolved` `TraceEvent`
for debug runs, as its L2-2 proposes, is a fine **addition** — richer diagnostics for a flagged
run — provided it is never the mechanism AC-4/AC-6/AC-9/AC-10 depend on.)*

**Also rejected: a node.** `(:StepRun)-[:USED_MODEL]->(:ModelUse {model, source, at})`, or a
`(:Model)` catalog node in `reference` with a per-run edge.

- In FalkorDB every **label is a diagonal matrix** and every **relationship type is its own
  adjacency matrix**. A `ModelUse` node adds a label matrix, an adjacency matrix, a node block
  entry, an identity index and a uniqueness constraint per workspace — all to store one string.
- Nothing ever **traverses** from a model. Every question the requirements actually ask is a
  projection over step-runs already reached by `runId` ("which model ran for this run's steps",
  AC-4/AC-6/AC-9/AC-10). A traversal-shaped model for a projection-shaped question is the classic
  over-modelling mistake — *model for the questions, not the entities*.
- A `(:Model)` catalog in `reference` would also be a **cross-graph reference** (edges cannot
  cross graphs), so it degrades to a property key anyway.

The one thing a node buys — "list every step-run in the workspace that used model X" without a
label scan — is an admin query over a label whose per-workspace cardinality is bounded by
`maxSteps × runs` (78 `StepRun` nodes in `ws:acme` today). Not worth a matrix.

### 1.3 Naming — and one amendment for `architect`

`resolvedModel` and `modelSource`, `camelCase` per the schema conventions. `resolvedModel`
deliberately says *resolved*, not `model`: with FR-7 roles and FR-18 fallback chains the value is
the **outcome** of resolution, never the thing the author typed. `modelSource` names the
precedence rung that won, which is what makes FR-17's hard cap auditable — the requirement leans
on exactly this ("the trace is what keeps that visible", FR-17/AC-10).

> **Amendment requested.** `docs/plans/llm-provider-config.md` L2-2 names the property
> `StepRun.model` and the parameter `record_step_and_advance(..., model=…)`. Please rename to
> **`resolvedModel`** / `resolved_model=`. Beyond the "resolved is the point" argument above,
> `model` collides in review with the *requested* ref that a step's `config.model` carries — the
> two differ in exactly the cases AC-9 and AC-10 exist to catch, so a reader who conflates them
> reads the trace backwards. That plan's own vocabulary already leans this way
> (`ResolvedModel`, `Resolution`, `last_used`). This is the only naming divergence between the
> two documents; everything else is agreement.
>
> `modelSource` is an **addition** L2-2 does not currently have. It is optional for the stated
> ACs and costs ~10 bytes per LLM step, but it is what turns AC-10 from "the model differs from
> what the step asked for" into "a workspace override overruled the step" — the difference
> between a reader inferring the cause and reading it. Recommended, not required.

**`modelFallback` — new in v2 (analyst m-5; finding accepted).** `modelSource ∈ {workspace, step,
default}` names *which precedence rung won*; it cannot say whether, within that rung, an FR-18
fallback chain had to move past the primary model. An operator reading the trace after an AC-9
event sees `modelSource = 'step'` and a model the step never named, with nothing on the row
saying a degradation occurred — the two facts (which rung, and whether that rung's answer came
from a fallback) are orthogonal, and the review is right that overloading `modelSource` with a
`'fallback'` value would conflate them (a fallback can happen *at any rung* — a workspace
override can itself be a role with a fallback chain).

**Decision: a fourth, boolean `StepRun` property, `modelFallback`**, set `true` only when
`resolvedModel` is not the first model in whichever chain the winning rung named, and **omitted
(not written) otherwise** — following the same "nullable, absent by default" contract as
`resolvedModel`/`modelSource` (§1.1, §1.4's `NULL`-omits-the-property proof). This is chosen over
the review's documentation-only alternative (leaving fallback visible only in logs/debug
`TraceEvent`s) because AC-9 is a **formal acceptance criterion**, not a debugging concern, and
`TraceEvent`s are debug-only by construction (§1.2 — `run["trace"]` false for an ordinary run, so
a `TraceEvent`-borne signal would exist only for runs someone thought to flag in advance, the
exact failure mode §1.2 already rejected `TraceEvent` over for `resolvedModel` itself). The same
argument applies unchanged to whether a fallback occurred: it must be on every run's durable
record, not gated behind an opt-in flag.

Cost: one boolean, written only on the (expected-rare) rows where a fallback actually fired — near
enough to the existing ~50-bytes/row estimate (§5) to not move it. The resolver sets it; see §6.2.

### 1.4 Write path — the one real constraint

`record_step_and_advance` (`repository.py:1301`, `QUERIES.md` §12.2) is a **single atomic query**
that creates the `StepRun`, hangs `NEXT` off the `LAST_STEP_RUN` tail, moves `AT_STEP` and bumps
`stepCount`. The three properties ride the existing `CREATE`:

```cypher
// [proposed] — QUERIES.md §12.2, additions marked ⊕ (modelFallback added in v2, m-5)
MATCH (r:WorkflowRun {runId: $runId})-[atRel:AT_STEP]->(cur:Step)
MATCH (to:Step {stepUid: $toStepUid})
OPTIONAL MATCH (r)-[lastRel:LAST_STEP_RUN]->(prevSR:StepRun)
CREATE (sr:StepRun {stepRunId: $stepRunId, stepKey: cur.key,
                    status: $stepStatus, startedAt: $startedAt,
                    endedAt: $endedAt, input: $input, output: $output,
                    resolvedModel: $resolvedModel,          // ⊕
                    modelSource: $modelSource,              // ⊕
                    modelFallback: $modelFallback})         // ⊕ v2 — NULL unless a fallback fired
CREATE (r)-[:HAS_STEP_RUN]->(sr)
CREATE (sr)-[:RAN]->(cur)
FOREACH (p  IN CASE WHEN prevSR  IS NULL THEN [] ELSE [prevSR]  END |
  CREATE (p)-[:NEXT]->(sr))
FOREACH (lr IN CASE WHEN lastRel IS NULL THEN [] ELSE [lastRel] END |
  DELETE lr)
CREATE (r)-[:LAST_STEP_RUN]->(sr)
DELETE atRel
CREATE (r)-[:AT_STEP]->(to)
SET r.stepCount = r.stepCount + 1
RETURN r.stepCount AS stepCount, sr.stepRunId AS stepRunId, cur.key AS ranStepKey
```

**No extra round trip, no atomicity change, no new failure mode.** Rule 4 ("all writes that touch
HEAD/TAIL must be a single `GRAPH.QUERY`") is preserved because nothing was split out.

**[verified] A `NULL` parameter omits the property entirely.** `CREATE (s:StepRun {... resolvedModel: $rm})`
with `rm=NULL` reports `Properties set: 2` (not 4) and `keys(s)` returns `[stepRunId, stepKey]`.
So **one query shape serves LLM and non-LLM steps alike** — no branching, no `FOREACH` trick, and a
`decision`/`human`/`wait` StepRun costs **zero** extra bytes. This is what makes the "nullable and
absent by default" contract in §1.1 free rather than a compromise. `modelFallback` (v2) follows
the identical rule — `$modelFallback=NULL` on the non-fallback (common) path omits it exactly like
the original two. This is a generic engine behavior (row 12, §8), not specific to a two-property
`CREATE`, so it was not re-run for a third property; nothing in FalkorDB's `CREATE {..: NULL}`
handling is arity-dependent.

**[verified] Backward compatible with no migration.** A `StepRun` written before Landing 2 simply
lacks the properties; the read below returns `NULL` for it. Proved on the probe with a mixed trace
(two annotated step-runs plus one "legacy" one) — the legacy row came back with empty
`resolvedModel`/`modelSource` and correct `NEXT` ordering. **No backfill script is needed**, and
none should be written: a historical run's model is genuinely unknown, and inventing one from
today's config is precisely the re-derivation FR-8 exists to prevent. The same applies to
`modelFallback`: absent means "unknown/not applicable", never "confirmed no fallback".

### 1.5 The `_drive_loop` SHA lock — the sharp edge for the implementer

`_drive_loop` is **SHA-locked** at `71055f756280` (DESIGN §6.2). The call site
`rec = self._record(ctx, run, current_key, to_key, result)` is **inside** the lock; `_record`,
`_execute_step`, `_select_transition` and `StepResult` are all **outside** it.

Consequence, and it constrains the implementation shape:

> **The resolved model must reach `_record` on the `StepResult`** (a new field), because
> `StepResult` is the only value the locked call site already passes. Any other carrier —
> adding a `decision` argument to `_record`, or a new positional parameter — edits the locked
> body and forces a deliberate lock reopen plus a SHA recompute.

`StepResult` is a frozen dataclass outside the lock, so `resolvedModel: str | None = None`,
`modelSource: str | None = None` and `modelFallback: bool | None = None` (v2, §1.3) are all
additive and default-safe. **Recommended path: zero lock impact.** The value itself must be
produced by whichever client actually made the call, so a fallback chain reports the model that
*answered*, not the one that was tried first (AC-9) — see the interface contract, §6.2.

### 1.6 Deliberately **not** recorded: the guard judge's model

An `llm`-kind guard is a distinct FR-5 consumer, and a single `StepRun` can span the step's own
LLM call **plus** several outgoing guard judgments, potentially on different models. One scalar
cannot hold that.

**Decision: do not record guard models on `StepRun` in Landing 2.** Rationale:

- **No acceptance criterion asks for it.** AC-4/AC-6/AC-9/AC-10 are all about the *step's* model;
  AC-5 (a guard with no named model uses the guard default) is verified by which endpoint receives
  the call, not by the trace.
- It cannot be done cheaply. Guard verdicts live on `_TransitionDecision`, which `_drive_loop`
  passes to `_trace_step` but **not** to `_record` — so it needs either a **SHA-lock reopen** or a
  **second `SET` query per step**, trading rule-4 single-query atomicity for an audit annotation.
- The debug path already covers it: `TraceEvent` records guard judgments for a `trace = true` run,
  and the judgment payload can carry the model with no schema change at all (`payload` is an opaque
  string — rule 8 applies to it, so it stays diagnostic-only and un-queryable).

**If it is later wanted**, the cheapest correct shape is `StepRun.guardModels` as a **deduped list
of strings** (a first-class list property, not a serialised blob — so `ANY(m IN sr.guardModels …)`
works). Recorded here so the successor doesn't re-litigate it. Flagged to `architect` as an
explicit scope boundary in §7.

**Addendum — v2 (analyst m-6; finding accepted).** The same "one scalar, many calls" shape
recurs **one level down**, inside a single step's *own* execution, and needs the same explicit
rule. `_run_agent_node` (`executor.py:585`) loops up to `config.maxIterations`, calling
`self._llm.chat` on each pass; with an FR-18 fallback chain live, iteration 1 can answer on model
A and a later iteration (after the chain re-engages mid-loop) can answer on model B.
`StepRun.resolvedModel`/`.modelSource` are one scalar apiece, same as above — but this time it is
the step's *own* calls colliding, not a set of guards.

**Decision: last answering model wins**, same rule §1.6 would use for guards if it recorded them.
`resolvedModel`/`modelSource`/`modelFallback` record whichever `chat` call **produced the output
the loop actually returned** — the terminal iteration, not the first, not an aggregate. Rationale:

- Matches the contract §1.1/§6.2 already state: "`resolvedModel` is the model that answered, not
  the one that was chosen." For a multi-iteration step, the call whose output the step actually
  emits is unambiguously "the model that answered"; an earlier iteration's model is causally
  superseded by it and answers a question ("what generated *this* step's output") the requirement
  doesn't ask.
- Costs nothing beyond what §1.4/§1.5 already specify: `StepResult.resolvedModel` is stamped once
  from whichever client made the last successful call — a multi-iteration step is a resolver-loop
  detail invisible to the write path, exactly as designed.
- If per-iteration history is ever wanted, it is a **`StepRun.iterationModels`** deduped list
  property — the same shape recommendation as `guardModels` above, and, like it, not something any
  Landing-2 AC asks for.

**Implementer note, so "last wins" is not silently violated by construction:** the resolver/
executor must let each iteration's result **overwrite** the pending `resolvedModel`/`modelSource`/
`modelFallback` values inside `_run_agent_node`'s loop, so whatever is current when the loop exits
is what reaches `StepResult`. This falls out for free if the loop just reassigns local variables
per iteration and reads them once after the loop — no special-casing required — but it is worth
flagging because a "set once, on first resolution" implementation would silently record model A's
values when B is what actually answered, and nothing downstream would catch it.

### 1.7 Read path — "which model ran for this run's steps"

Extend the existing `read_step_runs` projection (`QUERIES.md` §12.8, `repository.py:1497`). This
is already the `NEXT`-ordered audit trail behind `GET /workflow-runs/{id}/step-runs`, so AC-4 and
AC-10 become observable through an endpoint that already exists.

```cypher
// [verified] — §12.8 + ⊕. Anchors on Node By Index Scan (r:WorkflowRun); route via GRAPH.RO_QUERY.
// modelFallback column added in v2 (m-5).
// $runId
MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (pv:StepRun)-[:NEXT]->(sr)
WITH sr, pv WHERE pv IS NULL                    // chain head — never the broken exists()-in-pattern
MATCH (sr)-[:NEXT*0..]->(x:StepRun)
RETURN x.stepRunId AS stepRunId, x.stepKey AS stepKey, x.status AS status,
       x.startedAt AS startedAt, x.endedAt AS endedAt, x.input AS input, x.output AS output,
       x.resolvedModel AS resolvedModel,        // ⊕
       x.modelSource   AS modelSource,          // ⊕
       x.modelFallback AS modelFallback         // ⊕ v2
ORDER BY x.startedAt
```

The `OPTIONAL MATCH` + `IS NULL` head-finding stays exactly as-is — `exists()` inside a pattern is
broken on this build (quirks KB) and this projection change must not tempt anyone to "simplify" it.

And the compact audit answer, for a human asking "what did this run actually use":

```cypher
// [verified] — the FR-8 one-liner. GRAPH.RO_QUERY. modelFallback column added in v2 (m-5).
// $runId
MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun)
WHERE sr.resolvedModel IS NOT NULL
RETURN sr.resolvedModel AS model, sr.modelSource AS source,
       coalesce(sr.modelFallback, false) AS fellBack, count(sr) AS steps
ORDER BY model
```

AC-4 reads as two rows with different `model`. AC-6 reads as the *new* concrete model appearing
after a config edit + restart with no republish. AC-9 reads as the fallback model, not the first
choice — and now, explicitly, as `fellBack = true` on that row rather than requiring the reader to
infer degradation from the model name alone (v2, m-5). AC-10 reads as the workspace's model with
`source = 'workspace'` — which is strictly more than AC-10 asks, and is what turns "an explicit
step choice was overruled" from invisible into a single-column fact.

### 1.8 No index on `resolvedModel`

Deliberate. Every read above is anchored on `runId` (already index-backed) followed by a bounded
`HAS_STEP_RUN` / `NEXT` walk — `resolvedModel` is a **projection, never a scan anchor**, so an
index on it would be pure RAM with no plan benefit (DESIGN §7.2: "an index that isn't hit is just
RAM"). The only query that would want one — "every step-run in this workspace that used model X" —
is a rare admin/audit read over a small per-workspace label.

**Revisit only if both** become true: per-workspace `StepRun` cardinality exceeds ~10⁵, **and**
the model-audit query becomes routine rather than incidental. Confirm with `GRAPH.PROFILE` before
adding it, not from this note.

---

## 2. FR-16 / FR-17 / AC-10 — the workspace model override

### 2.1 The problem beneath the requirement

FR-16 says a workspace "may override model choice for **everything** running in it" and FR-17
makes it a **hard cap** beating an explicit per-step choice. The design question is whether
"everything" means *one string governs all four consumers* (arity) or *the scope is the whole
workspace rather than one step* (scope).

**Decision: per consumer kind**, not a single blanket value.

> **Reasoning corrected in v2 (analyst A-2).** The **conclusion was upheld** — per-kind is a
> *faithful* reading of FR-16, not a silent narrowing. But v1 justified it by claiming a blanket
> override "breaks FR-19 by construction" and that "FR-16 and FR-19 would contradict each other".
> **That was overstated, and I withdraw it.** FR-19 would work exactly as specified: it would
> read the workspace's frozen index dimension, find that a chat model's declared dimension does
> not match, and refuse to embed, loudly — which is precisely its job. Nothing breaks and no two
> requirements contradict. The correct objection is narrower, and it is *stronger for being
> narrower*.

**The argument that actually holds**, in ascending weight:

1. **"Everything" is a scope quantifier, not an arity claim.** It contrasts *the whole workspace*
   with *one step / one agent / one guard* — the granularities FR-5 enumerates. It does not assert
   that one string is type-correct against all of them.
2. **FR-17's own chain is already kind-indexed at two of its three rungs** — the middle rung is
   "the **step/agent/guard**'s own choice", the last is "the **per-kind** default". A kind-blind
   top rung would be the odd one out in a precedence chain the requirement itself writes as
   kind-aware throughout.
3. **A blanket override makes an incoherent configuration *expressible*, and its only reachable
   outcome is a workspace that can never embed again.** A chat ref forced onto the embedding
   consumer either has no embeddings endpoint or produces the wrong width; FR-19 then correctly
   refuses every embed, permanently, until someone edits the override. The administrator gets a
   working hard cap on three consumers and a silently dead GraphRAG corpus on the fourth. Per-kind
   can express every *coherent* thing a blanket can (set the three chat kinds) and only loses that
   one.

So this is a **refinement of FR-16, not a departure from it** — no requirement change is needed,
and none is requested (§7-Q1, restated in v2).

### 2.2 Decision

Store the override as a **singleton `WorkspaceConfig` node inside `ws:{workspaceId}`**, with one
nullable scalar property per consumer kind.

```
(:WorkspaceConfig {workspaceConfigId: 'default',      // fixed singleton key; MERGE identity
                   agentModelOverride,                 // nullable "<provider>/<model-id>" or role
                   guardModelOverride,
                   embeddingModelOverride,
                   responderModelOverride,
                   modelOverrideUpdatedAt,             // ms epoch, provenance
                   modelOverrideUpdatedBy})            // member id, provenance
```

One node per workspace. No edges — it is a workspace-scoped singleton, not a participant in any
traversal, so it adds **no adjacency matrix**.

**Why a singleton node with named properties**, rather than a generic
`(:WorkspaceSetting {settingKey, settingValue})` key/value bag: the bag is stringly-typed and
turns every future setting into a parse, while the singleton gives each setting a typed, directly
projectable property. Crucially the extension path is **free** — a second workspace-level setting
is a new *property* on the existing node: **no new label, no new index, no new constraint, no DDL
change at all.** That makes this the cheapest possible design for "the first of N workspace
settings", which is what it is.

There is deliberately **no `Workspace` node** to hang this on — the workspace *is* the graph key
(DESIGN §3), so no existing node is a natural home, and inventing a `Workspace` node purely as a
property carrier would be a worse version of the same thing.

### 2.3 Alternatives weighed

| Option | Verdict |
|---|---|
| **A. Singleton `WorkspaceConfig` node in `ws:{id}`** | **Chosen.** Per-tenant state in the tenant's own graph; survives workspace creation without a file edit; one shared store across processes; rule-7 clean (no `workspaceId` filter anywhere — the graph key *is* the scope). |
| **B. Property on an existing node** | Rejected — there is no workspace-singleton node to use. `Channel`/`Thread` are the wrong cardinality. |
| **C. `(:WorkspaceModelOverride {workspaceId, …})` in `reference`** | **Rejected outright.** This is exactly the shape rule 7 forbids: per-workspace rows in a shared graph, filtered by a `workspaceId` property. It also puts a hot-path read on the globally-shared graph, and it scales the `reference` graph with tenant count. |
| **D. A `workspace.<wsId>` block in the falkor-chat config file** | **Rejected, but it is the genuine runner-up — see below.** |

**Option D deserves the honest treatment**, because it is arguably simpler and the requirements
lean its way: config is read at startup (FR-15), editing config through an API is explicitly out of
scope, so *nothing writes this at runtime*; and it would keep all model configuration in one
hand-edited file, which is the stakeholder's stated pain ("one place to look"). It costs zero RAM,
zero DDL and zero reads, and it gets **startup validation against the declared models for free** —
whereas a graph-stored override can name a model the config does not declare, deferring the failure
to run time (FR-10).

It is rejected on three points that outweigh that:

1. **Workspace lifecycle.** Workspaces are created at runtime (`bootstrap_schema.sh <wsId>`). A
   *global* file cannot govern a workspace that does not exist when the file is written; every new
   workspace would need a file edit **and a server restart** before it could be governed. Option A
   lets provisioning set the override at creation.
2. **Single-store philosophy.** DESIGN §1/§3: per-tenant state lives in `ws:{id}`. This is
   per-tenant state.
3. **Multi-process divergence.** FR-15 mandates restart semantics but says nothing about a single
   process. Two server processes with drifted files would disagree about a *hard cap* — the one
   control that must not be circumventable. One shared store cannot drift.

The residual risk from D's advantage (an override naming an undeclared model) is handled in §6.3:
the resolver validates the graph-stored override against the loaded config **at read time** and
fails loudly per FR-10 — never a silent fallback.

**Condition under which `architect` should flip this decision:** if Landing 2 ships against a
single process with statically-provisioned workspaces *and* no near-term intent to expose an admin
write path, option D is simpler and this note does not object. State the flip in
`docs/plans/llm-provider-config.md` if taken; the graph side then needs nothing at all.

### 2.4 Write path

```cypher
// [verified] MERGE on the singleton, backed by the UNIQUE constraint (§4). GRAPH.QUERY.
// $agent/$guard/$embedding/$responder may each be NULL (= leave unset / clear)
MERGE (c:WorkspaceConfig {workspaceConfigId: 'default'})
SET c.agentModelOverride     = $agent,
    c.guardModelOverride     = $guard,
    c.embeddingModelOverride = $embedding,
    c.responderModelOverride = $responder,
    c.modelOverrideUpdatedAt = $at,
    c.modelOverrideUpdatedBy = $by
RETURN c.agentModelOverride AS agent, c.guardModelOverride AS guard,
       c.embeddingModelOverride AS embedding, c.responderModelOverride AS responder
```

`MERGE` here is **constraint-backed**, per the standing rule ("every `MERGE` must be backed by a
uniqueness constraint — no exceptions"). Without the `UNIQUE` on `workspaceConfigId`, two
concurrent writers would each create a node and the read in §2.5 would start returning two rows —
a duplicate-node bug the constraint makes impossible.

Setting a property to `NULL` in `SET` **clears** it (unlike `CREATE`, where a `NULL` param simply
omits the key). Both paths land on "the property is absent", which is the single "no override at
this kind" representation — there is exactly one, and it is never the empty string. **An empty
string must never be written**: `''` is a *value*, and a resolver that treats it as "no override"
is one refactor away from treating it as "a model named empty".

### 2.5 Read path

```cypher
// [verified] Node By Index Scan | (c:WorkspaceConfig) — 0.008 ms. GRAPH.RO_QUERY (replica-safe).
MATCH (c:WorkspaceConfig {workspaceConfigId: 'default'})
RETURN c.agentModelOverride     AS agentModel,
       c.guardModelOverride     AS guardModel,
       c.embeddingModelOverride AS embeddingModel,
       c.responderModelOverride AS responderModel
```

**The absent case — [verified] both halves:**

- **Node never written → zero rows.** The resolver must read "zero rows" as *no overrides at any
  kind*, and fall through to the step/agent/guard's own choice. This is the default state of every
  existing workspace, so it must be the cheap, silent, non-exceptional path.
- **Node present, some kinds unset → one row with `NULL` in those columns.** Verified with only
  `agentModelOverride` written: the other three came back null.

Zero rows and an all-null row are therefore **equivalent** and must be handled by one code path.

### 2.6 When to read it — and where, exactly

> **Revised in v2 (analyst B-1 / A-5).** The v1 text said to read the overrides "at
> `Executor.run` / `resume` entry, alongside the snapshot read that already happens there".
> **That was wrong, and wrong in the specific way this note warns about elsewhere.** The snapshot
> read is at `executor.py:376-378` — *inside* `_drive_loop`, *inside* the SHA lock. `run()` and
> `resume()` do not read the snapshot. Sending an implementer there would have cost a lock reopen
> for the **inbound** read, the mirror image of the outbound hazard §1.5 correctly flagged.
> Finding accepted in full; the corrected placement follows.

**Read once per drive, in `_drive` (`executor.py:339`) — outside the lock — and carry it on the
`run` dict.**

```python
# executor._drive, before `return self._drive_loop(ctx, run)` at :366  [proposed]
run["ws"] = ctx.ws                                    # B-1: the guard kind's only carrier
run["modelOverrides"] = self._repo.read_model_overrides(ctx.ws)   # §2.5, one RO_QUERY
```

**[verified] why this placement and no other:**

| Fact | Evidence |
|---|---|
| The lock covers `_drive_loop` only | `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f'` — body spans `:375`–`:436` |
| The lock is live on this tree | recomputed → **`71055f756280`**, matching DESIGN §6.2 |
| `_drive` (`:339`) is outside it, and has both `ctx` and `run` | `def _drive(self, ctx: CallContext, run: dict[str, Any])` |
| `_drive` is the **single** convergence point of both entry paths | `run()` `:307` and `resume()` `:335` both call `self._drive(ctx, run)`; `repo.get_run` is at `:304`/`:332` |
| The `run` dict reaches every consumer inside the loop, untouched | `_drive_loop` forwards it to `_execute_step` (`:404`) and `_select_transition` (`:405-407`) — **no locked line changes** |
| …including the guard, which has no other route | `_select_transition` calls `evaluate_guard(guard, ctx=run_ctx, run=run, …)` at `:806-808` — **`run` is already passed**; `ctx=` there is the run-ctx dict, not a `CallContext` |

That last row is the whole of B-1's graph-side half. The `guard` kind is the one consumer with
**no `CallContext` in scope** — `repository.get_run`'s projection cannot carry `ws`, because in
this design *the workspace is the graph key*, not a field. Stamping `run["ws"]` in `_drive` is
therefore not a convenience; **it is the only lock-free way FR-17's hard cap reaches
`guardModelOverride` at all.** Without it, `guardModelOverride` (§2.2) is a property nothing can
read, and the failure is invisible — a guard with no declared model resolves to the kind default
whether or not the cap applied, so AC-5 still passes while the cap silently does not.

**Rejected: storing the overrides on `self`.** The `Executor` is a process-wide singleton and
drives run concurrently — sync routes and sync `BackgroundTasks` on Starlette's anyio worker
threadpool (`api.py:105`), plus a bare daemon thread per posted message (`mcp.py:71`). Per-drive
state on `self` is a straightforward data race between two runs in the same process, and the
symptom would be an audit field that lies about which workspace's cap applied. The `run` dict is
per-drive by construction. *(Same hazard the review raises against a cached `FallbackClient` —
the fix is the same: carry per-drive values on per-drive objects.)*

For the two consumers with no run — the `@mention` responder and the embedding worker — read per
call. Both are already network-bound on an LLM round trip, so a sub-millisecond local read is
noise. A short-TTL per-`ws` process cache is a legitimate optimisation but is **not** needed for
Landing 2 and should not be built speculatively; if it is added, the TTL must be documented,
because it silently changes "takes effect on the next run" into "takes effect within N seconds".

Per-run reading gives a graph-stored override **next-run** semantics rather than FR-15's
next-restart semantics. That is a deliberate improvement, not a violation: FR-15 governs the
*config files*, and this value is not in one. Say so in the manual (§7-Q3) so an administrator
isn't surprised in either direction.

---

## 3. FR-19 / AC-11 — the embedding-dimension guard

### 3.1 The failure being prevented (re-confirmed live, today)

**[verified] on module `41811`**, against a throwaway dim-4 index:

| Step | Result |
|---|---|
| `SET m.embedding = vecf32([0.1,0.2,0.3])` (3 dims into a dim-4 index) | `Properties set: 2` — **accepted, no error** |
| `MATCH (m:Message) RETURN m.msgId` | both nodes present |
| `CALL db.idx.vector.queryNodes('Message','embedding',10, vecf32([…4 dims…]))` | returns **only** the correctly-sized node |

So the wrong-dim node is written, is visible to ordinary Cypher, and is **permanently invisible to
ANN** with nothing anywhere reporting a problem. That is the silent corruption FR-19 exists to
stop, and it is still live on the pinned build.

**[verified] The second half of the trap:** `CREATE VECTOR INDEX` on an already-indexed property
is **rejected, never re-applied** — re-creating with `dimension:8` (and separately with a different
`similarityFunction`) returns `Attribute 'embedding' is already indexed`, and introspection still
reports the **original** options. Because `redis-cli` exits 0 on Redis-level errors (documented in
`bootstrap_schema.sh`'s own header), **re-running `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh <ws>`
against a workspace already bootstrapped at 1536 does NOT change the dimension** — it prints
`Attribute 'embedding' is already indexed` amid ~30 other lines and continues. An operator who
believes they have re-dimensioned a workspace this way has not. **This is the single most likely
way a real mismatch gets created**, which makes FR-19 a guard against an operator error, not
against a hypothetical.

### 3.2 The index dimension **is** introspectable — decision

**[verified]** `CALL db.indexes()` exposes it. The `options` column is a map keyed by property
name; a `VECTOR`-typed property's entry carries
`{dimension, similarityFunction, M, efConstruction, efRuntime}`.

```cypher
// [verified] GRAPH.RO_QUERY ws:{id} — replica-routable, zero write risk.
// $label ∈ {'Message','Chunk'}, $prop = 'embedding'
CALL db.indexes() YIELD label, types, options
WHERE label = $label AND types[$prop] = ['VECTOR']
RETURN options[$prop].dimension AS dim
```

Both dynamic map-key access (`options[$prop].dimension`) and a post-`YIELD` `WHERE` work. Against
`ws:acme` this returns **1024** for both `Message` and `Chunk` — matching the M2 GraphRAG stack
(Qwen3-Embedding-0.6B) and **not** the `bootstrap_schema.sh` default of 1536, which is precisely
why a guard reading the *actual* index rather than a config constant is the requirement.

Sweeping both labels at once:

```cypher
// [verified] returns 2 rows for ws:acme — Chunk 1024, Message 1024
CALL db.indexes() YIELD label, types, options
WHERE types['embedding'] = ['VECTOR']
RETURN label, options['embedding'].dimension AS dim
ORDER BY label
```

**[verified] edge behaviours the guard must handle**, all four distinct:

| Situation | Result | Guard must |
|---|---|---|
| Vector index exists, **zero vectors written** | dimension reported normally (it is index metadata, not data) | proceed — an empty workspace is fully checkable |
| Label has only `RANGE` indexes on other properties, none named `embedding` (e.g. `User`, indexed only on `userId`) | **zero rows**¹ | **refuse** — no vector index means no place to put an embedding |
| Label unknown to the graph | **zero rows** | **refuse** |
| Graph key does not exist at all | `ERR Invalid graph operation on empty key` | **refuse**, and do not let a write create the graph implicitly |

The last one matters: an un-bootstrapped workspace **errors** rather than returning zero rows, so
the guard's error handling cannot be a bare `if not rows`. And under no circumstances may the
guard "recover" by issuing a write — that would create the graph with no indexes and no
constraints, which is worse than the mismatch.

¹ **[verified], re-confirmed 2026-08-11 against `ws:test`.** `types[$prop]` is a map lookup keyed
by property *name*; when the label has no index — of any type — on a property literally named
`embedding`, the lookup is a missing key, which evaluates to `NULL`. `WHERE ... AND NULL =
['VECTOR']` evaluates to `NULL`, not `true`, so the row is filtered out — the query returns **zero
rows**, not a row with `dim = NULL`. A genuine "one row, `dim = NULL`" *is* producible, but only
from a **different** precondition than "has only `RANGE` indexes": a label carrying a `RANGE`
(non-vector) index **specifically on the property named `embedding`**. Nothing in this schema's
current DDL does that — every indexed property in the bootstrap DDL that isn't `embedding` is
named something else, and the `Message`/`Chunk` `embedding` indexes are `VECTOR`, not `RANGE` — so
this shape has no live example in the current schema; it is included here only so a future reader
who does add such an index understands why they'd see a `NULL` row instead of the empty result
this table documents for every case observed against the real schema today.

Rejected alternative: **probing by ANN query with a deliberately mismatched vector** and parsing
the `Vector dimension mismatch, expected N but got M` error text. This was the previously-documented
technique (and the quirks KB recommended it, wrongly). Rejected — it depends on error-message
parsing, it needs a `k`-sized ANN traversal per probe, it cannot distinguish "no index" from "wrong
dim", and it is strictly worse than a metadata read now that the metadata read is proven.

### 3.3 When the check runs

Three layers, each catching what the one before cannot:

1. **Startup assertion, per known workspace — fail fast and loud.** For each workspace the process
   is configured for, read `Message.embedding` **and** `Chunk.embedding` dimensions and compare
   against the resolved embedding model's declared dimension. A mismatch is a startup error naming
   the workspace, the label, the index dimension, the model and its dimension. This is the layer
   that satisfies AC-11's "clear message" for the ordinary case.
   *Caveat:* today the process knows only `config.WS_ID`, so this covers one workspace. It is a
   fail-fast convenience, **not** the correctness boundary.
2. **First-embed resolution per `(ws, label)`, cached for the process lifetime — the correctness
   boundary.** Covers workspaces created after startup, which layer 1 structurally cannot. Caching
   for the process lifetime is sound precisely because §3.1 proves the dimension **cannot change in
   place** — changing it requires dropping and recreating the index, an out-of-band admin action.
   Cache the *introspected dimension*, keyed by `(ws, label)`; never cache a *failure* (an
   un-bootstrapped workspace can become bootstrapped without a restart).
3. **Post-hoc length check before the `SET` — keep the existing one, unchanged.**
   `EmbeddingWorker.embed_message` and `Repository.set_embedding` already raise
   `EmbeddingDimensionError` on a length mismatch. Keep both. They now compare against the
   **introspected** dimension instead of `config.EMBEDDING_DIM` (which FR-20 removes anyway).

**Which labels get checked, and at which layer** — *revised in v2 (analyst m-4; finding accepted,
with one addition).*

v1 said "both `Message` and `Chunk` must be checked, independently" without saying *where*, which
read as "check both on every write". That is wrong, and the review is right about why:

- **[verified] nothing in the application writes `Chunk.embedding` today.**
  `grep -rn 'Chunk' server/falkorchat/*.py` returns **exactly one hit, and it is a comment**
  (`config.py:32`). `Chunk`/`Document` are bootstrapped DDL and a DESIGN §3 corpus concept; no
  module populates them.
- So gating a **`Message` write** on the `Chunk` index's dimension would let a divergent-but-
  **unused** `Chunk` index block message embedding that the workspace would accept perfectly
  well — a refusal with no corresponding corruption to prevent. FR-19 says refuse *"when the
  configured embedding model's vector dimension does not match the target workspace's vector
  index"*; for a `Message` write, the target index is `Message.embedding`. Nothing else.

**Corrected rule — gate each write on the label being written:**

| Layer | Checks | Blocking? |
|---|---|---|
| 1 — startup assertion (§3.3) | **both** `Message` and `Chunk` | yes, at startup — an operator-facing warning that the workspace is internally inconsistent |
| 2 — first-embed cache (§3.3) | **the label being written**, only | yes — this is the write gate |
| 3 — post-hoc length check | **the label being written**, only | yes |

The `Chunk` check keeps its place in layer 1, where it belongs: `bootstrap_schema.sh` creates both
indexes from one `EMBEDDING_DIM`, but nothing *enforces* that they agree, and §3.1's "already
indexed" rejection means a re-bootstrapped workspace can end up genuinely divergent. That is worth
telling an operator about at startup. It is not worth blocking an unrelated write over.

**Addition the review does not make, and it matters for sequencing:** when a `Chunk` writer *is*
built, it inherits layers 2 and 3 for `Chunk` automatically — the introspection query (§3.2) is
already label-parameterised (`$label ∈ {'Message','Chunk'}`) and the cache is keyed `(ws, label)`.
No FR-19 rework is needed then. That is the reason to keep the design label-generic even though
only one label is live today.

### 3.4 The failure surface

Reuse the existing `EmbeddingDimensionError` (`repository.py:19`) — it already exists, is already
raised on both paths, and is already understood by callers. Do not introduce a second exception.

The behavioural change FR-19 requires is **where the check fires**:

- **Pre-flight, before any HTTP call [new].** Compare the *configured embedding model's declared
  dimension* (from the falkor-chat config file, per FR-14's per-model settings) against the
  introspected index dimension. On mismatch: raise, **no embedding is computed and no vector is
  written**. This is AC-11's "fails with a clear message and no vector is written" in its
  strongest form — the model is never even called.
- **Post-hoc, before the `SET` [existing].** Actual returned vector length vs. index dimension.
  Last line of defence against a model that lies about its dimension.

The error text must name **all five** facts, because each one alone sends the operator to the wrong
place: workspace, label, index dimension, model id, model dimension. Add the remedy explicitly —
*the index dimension cannot be changed in place; re-bootstrapping does not change it (§3.1); either
configure a matching model or create a new workspace* — since the natural operator instinct
(re-run bootstrap with a new `EMBEDDING_DIM`) is exactly the thing that silently does nothing.

**Dependency on `architect` — already satisfied.** The pre-flight check needs the config to
declare a dimension per embedding model; `docs/plans/llm-provider-config.md` §4.5 provides it as
overlay `models."<ref>".dim`, authoritative when present, with `FALKORCHAT_EMBEDDING_DIM` as the
fallback. That is exactly what §3.4 needs, and it keeps the **`ws:test` dim-4 estate**
(DESIGN §14.7) working through the declared-dim path.

Note the **three distinct numbers** this feature now juggles, and keep them distinct in code and
in error text: (a) `bootstrap_schema.sh`'s `EMBEDDING_DIM` — *DDL-time input*, consumed once at
index creation and thereafter meaningless; (b) the model's declared `dim` — *what the model will
produce*; (c) the workspace's **frozen index dimension** — *what the graph will accept*, and the
only one of the three that is authoritative at embed time. FR-19 is precisely the assertion
`b == c`. A guard that compares against (a) is checking a stale constant and would have missed the
real `ws:acme` case, where (a) is 1536 and (c) is 1024.

---

## 4. Proposed DDL (consolidated)

All **[proposed]** — `bootstrap_schema.sh` is not edited by this unit. Add to
`bootstrap_workspace()`, in this order (index before constraint — the live-verified ordering rule):

```bash
# ── workspace-level configuration singleton (K-042 / FR-16) ──
echo "[index] WorkspaceConfig.workspaceConfigId"
gquery "$g" "CREATE INDEX FOR (n:WorkspaceConfig) ON (n.workspaceConfigId)"

# … alongside the other constraints, after all indexes:
echo "[constraint] WorkspaceConfig unique {workspaceConfigId}"
gconstraint "$g" UNIQUE NODE WorkspaceConfig PROPERTIES 1 workspaceConfigId
```

**[verified] on the probe**: the index creates (`Indices created: 1`), the constraint returns
`PENDING` and reaches `OPERATIONAL` (confirmed via `CALL db.constraints()`), and the subsequent
`MERGE` + `RO_QUERY` behave as §2.4/§2.5 describe.

**No DDL for FR-8.** `resolvedModel` / `modelSource` / `modelFallback` (v2) are unindexed
properties on an existing label (§1.8). **No DDL for FR-19** — it is a read of existing index
metadata.

`reference` is untouched by this design. So is the `identity` graph.

---

## 5. RAM implications (rule 6)

RAM is the binding constraint; every item is called out.

| Addition | Cost | Basis |
|---|---|---|
| `StepRun.resolvedModel` + `.modelSource` | **≈50 bytes per LLM-executing StepRun**; ≈1 MB per 20 000 | **[verified]** `GRAPH.MEMORY USAGE` delta between two 20 000-node probe graphs with and without the two properties: 2 MB → 3 MB (`amortized_node_attributes_by_label_sz_mb` for `StepRun`, 1 → 2). MB granularity — treat as an upper bound of that order, not a precise figure. |
| …`.modelFallback` (v2, §1.3) | **effectively zero** | one boolean, written only on the rows where a fallback fired (expected rare); on the common (non-fallback) row it is omitted, same as the other two — not separately measured, below the ≈50 B/row figure's own MB-granularity noise floor |
| …on non-LLM step-runs (`decision`/`human`/`wait`/offline stub) | **zero** | **[verified]** a `NULL` param omits the property entirely (§1.4) |
| `WorkspaceConfig` node | **one node per workspace** + one label matrix + one range index + one constraint over 1 row | effectively zero; below `GRAPH.MEMORY USAGE`'s 1 MB resolution |
| Index on `resolvedModel` | **not added** (§1.8) | avoided cost |
| FR-19 introspection | **zero** | reads existing index metadata; creates nothing |
| New relationship types | **none** | no new adjacency matrix in any graph |

For scale: `ws:acme` holds 78 `StepRun` nodes today (`db.indexes()` `numDocuments`). Even a
100 000-step-run workspace adds ≈5 MB — against the vector index line of ≈1.25 GB per
100k-message workspace at dim 1024 (DESIGN §11). **These additions do not move the sizing model
and do not change shard packing.**

The `ModelUse`-node alternative rejected in §1.2 would have cost, per LLM-executing step: a node
block entry, an identity property, a label matrix row, an adjacency matrix entry, plus an index and
constraint — conservatively 4–6× the property approach, for an audit field nothing traverses.

---

## 6. Resolver-facing interface contract

The seam between this note and `docs/plans/llm-provider-config.md`. **`architect` owns the
resolver; this note owns what it reads and writes and when.**

### 6.1 What the resolver must READ from the graph

One call, at the start of a drive (or per call for the run-less consumers):

```
read_model_overrides(ws) -> {agentModel, guardModel, embeddingModel, responderModel}
                            # every field Optional[str]; ALL-None when the node is absent
```

- Backed by §2.5 (`GRAPH.RO_QUERY`, `Node By Index Scan`, 0.008 ms).
- **Zero rows and an all-null row are the same answer.** One code path.
- A value may be a concrete `"<provider>/<model-id>"` **or** a role name (FR-7) — the resolver
  resolves it through the same path as any other named model, so a role works as an override.
- The graph never enforces precedence. **Precedence is entirely the resolver's**
  (workspace → own choice → per-kind default, FR-17). The graph is a store, not a policy engine.

### 6.2 What the resolver must WRITE back, and when

Per LLM call, the resolver must return alongside the client:

```
resolvedModel:  str    # the concrete "<provider>/<model-id>" that ACTUALLY answered
modelSource:    str    # 'workspace' | 'step' | 'default' — which rung won
modelFallback:  bool   # NEW in v2 (§1.3, m-5) — True iff resolvedModel is not the winning
                        # rung's primary/first-choice model; omit (None) when it is
```

Four binding requirements:

1. **`resolvedModel` is the model that answered, not the one that was chosen.** With an FR-18
   fallback chain the two differ, and AC-9 is specifically about the difference. The value must
   therefore be produced by the client that made the successful call — not computed up-front by the
   resolver. **Naming it `resolvedModel` rather than `model` is load-bearing for this reason.**
   Suggested carrier: a `model` field on `ChatResult` (`llm.py:47`), which the successful client
   populates.
2. **It must reach `_record` on the `StepResult`** — see §1.5. `StepResult` is the only value the
   SHA-locked `_drive_loop` call site already passes to `_record`; anything else forces a lock
   reopen. All three new fields default to `None` so non-LLM step types are unaffected.
3. **`modelSource` must be set by whichever rung actually won**, including `'workspace'` when a
   hard cap overruled an explicit step choice. This is the field AC-10 turns on.
4. **`modelFallback` is set by comparing the answering call against the chain the winning rung
   named, not against `modelSource`.** The resolver already holds the ordered chain it is walking
   (FR-18) at the moment a call succeeds; `modelFallback = (index of the successful entry > 0)`.
   It is **orthogonal to `modelSource`** — a workspace override can itself resolve to a role with
   its own fallback chain, so `('workspace', True)` is a valid, meaningful combination, not a
   contradiction. Leave it unset (`None`) rather than `False` on the non-fallback path, matching
   the omission contract in §1.1/§1.4 — the property's *presence* is the signal an operator scans
   for, not its value.

### 6.3 What the resolver must VALIDATE

- **Graph-stored overrides are not config-validated.** A `WorkspaceConfig` override may name a
  model or role the config file does not declare (the acknowledged cost of choosing option A over
  option D, §2.3). The resolver **must** resolve it through the normal path and, on failure, fail
  loudly per FR-10 — the run suspends with an error naming the workspace override and the
  unresolvable identifier. **Never** fall through to the step's choice or the per-kind default: a
  hard cap that silently degrades to the thing it was capping is worse than no cap.
- **Publish-time validation (FR-9/AC-7) cannot see workspace overrides** and must not try to. A def
  is published to `reference`, which is global; overrides are per workspace. Publish validates the
  *declared* identifiers only. Stated here so nobody attempts a cross-graph check — edges cannot
  cross graphs, and per-workspace state has no business in `reference`.
- **The embedding config must declare a `dimension` per embedding model** (§3.4). Without it the
  FR-19 pre-flight check cannot run before the HTTP call.

### 6.4 Landing 1 → Landing 2 sequencing

Landing 1 needs **no graph change**. Two things Landing 1 should nonetheless get right so Landing 2
is additive rather than a rework:

- **Keep the resolver's return a value object, not a bare string.** Landing 2 needs `(client,
  resolvedModel, modelSource)` from the same call. A Landing-1 signature of `resolve(kind, name)
  -> LLM` forces a breaking change; `-> ResolvedModel` does not.
- **Do not let the resolver hard-code its inputs to the config file.** The workspace override is a
  second input arriving in Landing 2. A resolver constructed with `(config, ws_override_reader)`
  absorbs it; one constructed with `(config)` and reaching into module state does not.

Neither costs Landing 1 anything. Both are cheap now and expensive later.

### 6.5 Direct answers to `docs/plans/llm-provider-config.md` §8

That plan's §8 asks four numbered questions. Answered here in its numbering, so the two documents
can be read against each other without translation.

**§8.1 — FR-8, a durable always-written resolved-model field.**
**Answer: two unindexed scalar properties on `StepRun`, written by the existing atomic
`record_step_and_advance`.** Agreed on the mechanism and on the `TraceEvent` rejection (§1.2 —
reached independently, same evidence, `executor.py:390`).

| What §8.1 asked for | Where |
|---|---|
| The query change | §1.4 — the additions marked ⊕ inside `QUERIES.md` §12.2. One query, no extra round trip, rule 4 intact |
| The read surface | §1.7 — `QUERIES.md` §12.8 gains two projected columns, so `GET /workflow-runs/{id}/step-runs` carries it with no new query; plus a one-line aggregate audit read |
| The RAM note (rule 6) | §5 — **≈50 bytes per LLM-executing StepRun**, measured; **zero** on non-LLM steps; ≈5 MB at 10⁵ step-runs against a ≈1.25 GB vector line. No index (§1.8) |

Three implementation facts that will otherwise cost the implementer a debugging cycle each:

1. **The value must ride on `StepResult`.** `_drive_loop` is SHA-locked and its
   `self._record(ctx, run, current_key, to_key, result)` call site is *inside* the lock. `result`
   is the only carrier already crossing it. See §1.5 — L2-2 as written ("`executor._record`
   passes the concrete label") is correct about the destination but silent about the lock.
2. **A `NULL` parameter omits the property** ([verified], §1.4) — so one query shape serves
   `agent` and `decision`/`human`/`wait` alike, with no branching and no cost on non-LLM steps.
3. **No backfill, ever.** Pre-Landing-2 `StepRun`s read back `NULL` ([verified]) and must stay
   that way; synthesising a historical model from today's config is the exact re-derivation FR-8
   exists to forbid.

Naming amendment (`model` → `resolvedModel`) in §1.3.

**§8.2 — where the workspace override lives and how it is read.**
**Answer: a singleton `(:WorkspaceConfig {workspaceConfigId: 'default'})` node in `ws:{id}` —
not `reference`** (§2.2, §2.3). Node/property choice §2.2; write path §2.4; read path §2.5;
alternatives incl. the config-file option and the condition to flip §2.3.

- **Shape: `{kind -> ref}`, and explicitly *not* `{kind|"*" -> ref}`.** §8.2 offered the wildcard
  as an option; reject it. *(Corrected in v3 — this passage still carried the overstated framing
  §2.1 withdrew in v2; see §2.1 for the full argument.)* A `"*"` blanket that includes the
  embedding kind does not break anything by construction — FR-19 would do exactly its job, reading
  the frozen index dimension and refusing every embed once it finds a chat model's dimension
  doesn't match. The objection is narrower than "contradiction": that refusal is **permanent**,
  because nothing else can ever satisfy it, so a blanket override makes an incoherent
  configuration *expressible* whose only reachable outcome is a workspace that can never embed
  again. An administrator who wants "everything chat" sets the three chat kinds; that is one extra
  key in a hand-written admin action, against a wildcard that makes a permanently-broken
  configuration expressible for no expressive gain (§2.1 point 3). If a wildcard is added anyway
  it **must** exclude `embedding`, and that exclusion must live in the resolver, not in a comment.
- **Write path: a one-shot parameterised `MERGE`** (§2.4) — a seed script or an admin query, as
  §8.2 anticipated. Constraint-backed, per the standing `MERGE` rule; the constraint is the only
  new DDL in this design (§4).
- **Read cost**: `Node By Index Scan`, 0.008 ms [verified] — comfortably cheap enough for L2-3's
  "once per drive / per responder call", and it would in fact survive per-step; L2-3's instinct to
  hoist it is still right, since it also makes the override **stable for the duration of a run**,
  which is the more valuable property.
- **Absent = zero rows**, and a partially-set node returns `NULL` per unset kind ([verified],
  §2.5). One code path for both.

**§8.3 — the workspace's frozen vector-index dimension.**
**Answer: yes, it is introspectable on this build, via `CALL db.indexes()`** — §3.2 carries the
exact query, live-verified today, with `ws:acme` returning **1024** for both `Message` and
`Chunk`.

§8.3 was right to demand live verification rather than assumption: **my own knowledge base
(`claude/graph-dba/falkordb-quirks.md`) asserted that `db.indexes()` does *not* expose the
dimension.** That was recorded against the edge build (module 999999) and is **false on the pinned
`v4.18.11` / module `41811`**. The entry has been corrected with today's evidence. Had this note
been written from the cached fact, L2-6 would have been designed onto the far worse
error-message-parsing probe rejected in §3.2.

Four edge behaviours the L2-6 implementation must branch on — no vector index, unknown label,
nonexistent graph key (which **errors**, it does not return zero rows), and index-with-no-vectors
(which works fine) — are tabulated in §3.2. Check **both** `Message` and `Chunk` (§3.3). Caching
per `(ws, process)` as L2-6 proposes is sound, and §3.1 proves *why*: the dimension cannot change
in place. Cache the dimension; never cache a failure.

**§8.4 — should the chat agent's model live on the `Agent` node instead of the overlay file?**
**Answer: no. Keep it in the overlay `agents` map. `architect`'s call is right, and the
graph-side alternative is not materially better.**

The tempting argument is that `Agent` already carries an `agentId` index and a UNIQUE constraint,
so `Agent.model` needs no DDL either — the usual objection doesn't apply. It still loses:

- **An `Agent` node is a chat-participant identity, not a configuration record.** It is created by
  `seed_demo.sh` / `ensure_agent` as a *member* of a workspace, alongside `User`. Hanging provider
  configuration on it mixes a tenancy projection with an operations concern, and the next person
  wanting a per-agent setting finds a precedent for putting it there too.
- **It is per-workspace, so it is N places, not one.** The same logical agent is a separate node in
  every workspace it participates in. `agents["assistant"]` is one line; `Agent.model` is one write
  per workspace, kept in sync by hand — which is a second copy of exactly the duplication driver 2
  of this feature exists to eliminate.
- **The graph-side need it would serve is already met.** The real requirement behind "this
  workspace's assistant should use a different model" is FR-16, and §2.2's
  `responderModelOverride` covers it — with hard-cap semantics, which a per-agent property would
  not have. There is no residual case needing a second graph-side mechanism.
- **Offline testability** — the overlay is a dict in a test; an `Agent.model` read puts a graph
  round trip inside responder unit tests that currently have none.

One consequence to record, since it is the honest cost: `responderModelOverride` is **per
workspace, not per agent** — a workspace running several agents caps them all to one model. That
is consistent with FR-16's wording ("everything running in it") and with FR-17's hard cap, and it
is the correct semantics for a cap. If per-agent granularity is ever genuinely needed, it belongs
in the overlay's `agents` map (where it already fits), **not** on the node.

---

## 7. Open questions, assumptions, and things I could not verify

Flagged to `architect` (resolver) and `tico` (requirements), each with the assumption taken.

**Q1 — ~~Is the workspace override per-kind or blanket?~~ WITHDRAWN in v2. No stakeholder
question is raised.** The `analyst` gate (A-2) upheld per-kind as a **faithful reading of FR-16**,
not a narrowing: "everything running in it" is a scope quantifier, and FR-17's chain is already
kind-indexed at two of three rungs. It is a design refinement inside the approved requirement, so
it needs no requirements amendment and no stakeholder adjudication.

v1 escalated this to `tico` on the premise that a blanket override would make FR-16 and FR-19
*contradict*. That premise was wrong (§2.1 carries the correction and the argument that does
hold), and the escalation went with it. **Nothing to route.**

**One residual worth a sentence in the manual, not a question** *(analyst A-2, and I agree)*:
with per-kind overrides, "everything" is exactly as complete as the kind set. Plan §3.1 declares
that set **fixed and closed**, which closes the hole today — but adding a fifth consumer kind
later means adding its override property in §2.2, or the new kind silently escapes a control the
requirement calls a *hard cap*. That is a **note for the admin manual and a comment on the
`WorkspaceConfig` DDL**, owned by `tico` and whoever adds the fifth kind. Not a blocker, and not
a question needing an answer now.

**Q2 — The `@mention` responder has no execution trace.** FR-8 records the resolved model on "the
workflow run's execution trace". The responder (FR-5/consumer 1) runs **outside** any workflow when
`WORKFLOW_ENABLED` is off; there is no `WorkflowRun` and therefore nowhere for FR-8's field to land.
**Assumption:** out of FR-8's literal scope; not designed here.
**Recommendation if wanted:** `Message.resolvedModel` on agent-authored messages — an unindexed
property on an existing label, no DDL, ~40 bytes per agent message, and it costs nothing to add
later. It is the same shape as §1 and should reuse the same property name.

**Q3 — Override change semantics.** §2.6 gives a graph-stored override **next-run** effect, while
FR-15 gives config files **next-restart** effect. Deliberate (the override is not in a config file)
but user-visible.
**Assumption:** next-run is desirable. Needs one line in the admin manual so an administrator is
not surprised in either direction.

**Q4 — Guard-judge model traceability is deliberately out of Landing 2** (§1.6). No AC requires it;
recording it needs a SHA-lock reopen or a second write per step. Recorded so it is a decision, not
an oversight; the shape to use if it is ever wanted is in §1.6.

**Not verified live — and why:**

- **The FR-8 write inside the real `record_step_and_advance`.** Verified as an equivalent query
  against a probe graph with a hand-built run/step-run trace, not through `executor.py` — running
  the real executor means driving a real run against shared state. The added `CREATE` properties
  are ordinary parameters in an already-verified query, so the risk is low, but a `qa-engineer`
  pass on `test_queries.sh` at Landing 2 is the real proof.
- **The `_drive_loop` SHA lock (§1.5)** was read from DESIGN §6.2 and confirmed against the current
  `executor.py` call site by reading the source. The SHA itself was **not** recomputed — the
  documented recompute command is in DESIGN §6.2 and the implementer should run it before and after
  any change to confirm the lock is intact.
- **AC-2 / AC-3 (cloud provider, `{env:}` secret substitution)** are model-gated per the
  coordination record and are not graph-side concerns.
- **Multi-workspace behaviour of the FR-19 startup check** — this deployment has one real workspace
  (`ws:acme`), so "per known workspace" (§3.3 layer 1) is structurally, not empirically, verified.

---

## 8. Live verification log (2026-08-10)

Instance: `falkordb/falkordb:v4.18.11`, module `41811`, plus `vectorset`. Started detached via
`./scripts/start_falkordb.sh -d` (it was not running); the `falkordb-data` volume was reused, not
recreated. Shared graphs read with `GRAPH.RO_QUERY` only. All mutations ran against throwaway
`ws:probe-gdba-*` graphs, deleted with `GRAPH.DELETE` at the end.
`./scripts/verify_workflows.sh acme` → `OK — 2 defs in sync` **after** the session.
`./scripts/test_queries.sh` was **not** run (it deletes `reference` at teardown).

| # | What | Result |
|---|---|---|
| 1 | `MODULE LIST` | `graph` `41811`, `vectorset` |
| 2 | `db.indexes()` on `ws:acme`, `options` column | exposes `{dimension: 1024, similarityFunction: cosine, M: 16, efConstruction: 200, efRuntime: 10}` — **corrects the quirks KB** |
| 3 | `options[$prop].dimension` with a `CYPHER` param + post-`YIELD` `WHERE` | works; `ws:acme` `Message`/`Chunk` both **1024** |
| 4 | Same query for `User` (range-only) | one row, `dim` null |
| 5 | Same query for an unknown label | zero rows |
| 6 | `GRAPH.RO_QUERY` against a nonexistent graph | `ERR Invalid graph operation on empty key`; graph **not** created |
| 7 | dim-4 index + 3-dim `vecf32` write | `Properties set: 2`, no error; node `MATCH`-able; **absent from ANN** |
| 8 | dimension readable with **zero** vectors written | yes — index metadata |
| 9 | `CREATE VECTOR INDEX` re-create at dim 8 / different similarity | `Attribute 'embedding' is already indexed`; **original options retained** |
| 10 | `StepRun` + `resolvedModel`/`modelSource`, §12.8 read extended | correct values; a property-less "legacy" StepRun returns nulls, `NEXT` order intact |
| 11 | FR-8 aggregate audit query (§1.7) | two distinct models with their sources |
| 12 | `CREATE` with a `NULL` property param | property **omitted**; `Properties set: 2`, `keys(s) = [stepRunId, stepKey]` |
| 13 | `WorkspaceConfig` index → constraint → `db.constraints()` | `PENDING` → `OPERATIONAL` |
| 14 | §2.5 read with node absent | **zero rows** |
| 15 | §2.4 `MERGE` write, then §2.5 read with only `agentModelOverride` set | one row, other three null |
| 16 | `GRAPH.PROFILE` of §2.5 | `Node By Index Scan \| (c:WorkspaceConfig)`, 0.008 ms |
| 17 | `GRAPH.MEMORY USAGE`, 20 000 StepRuns with vs. without the two properties | `StepRun` attributes 1 MB → 2 MB (≈50 B/node) |
| 18 | `GRAPH.LIST` after teardown | `cpg_falkorchat`, `cpg_salesperson`, `reference`, `ws:acme` — probes gone, shared graphs intact |
| 19 | `executor.py:390` read directly (source, not inference) | `tracer = self._tracer if run["trace"] else _NULL_TRACER` — confirms `TraceEvent`s are debug-only, grounding §1.2 |
