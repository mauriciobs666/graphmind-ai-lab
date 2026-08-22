# Graph Ontology Reference — manual verification test report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (—)

## Summary

Executed every runnable example query and schema table in `docs/manuals/graph-ontology.md`
(commit `ad5fd93`, 2026-08-21) against the live `falkordb-dev` instance: `cpg_falkorchat`,
`cpg_salesperson`, `kaizen_team`, `reference`, `ws:acme`. 20 test items run, per
`docs/test-plans/graph-ontology.md`. **18 pass, 2 fail** (both in §4, `ws:<workspaceId>`).
No query errored; every divergence found is a documentation-accuracy gap, not a broken
example. Overall verdict: **the manual is substantively accurate and its "Try it" queries
all work as written** — two real but narrow defects below should be fixed before the next
edit.

CPG: considered, not relevant — the check is a live-data verification of a hand-authored
manual's Cypher examples and schema tables against running FalkorDB graphs, not a
code-property-graph question (no source-code call-graph/impact-analysis angle); a CPG would
not have changed how this was tested.

## Results table

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-001 | `cpg_falkorchat` labels match | **PASS** | `db.labels()` returned exactly the 21 labels the manual's §1 table implies (19 real + `CpgNode` + `CpgBuildInfo`) |
| TP-002 | `cpg_falkorchat` relationship types match | **PASS** | `db.relationshipTypes()` returned exactly the 19 types listed in §1 |
| TP-003 | `cpg_salesperson` labels match minus `CpgBuildInfo` | **PASS** | `db.labels()` returned the same 21-label set minus `CpgBuildInfo` = 20 labels, exactly as claimed |
| TP-004 | `cpg_salesperson` relationship types match | **PASS** | `db.relationshipTypes()` returned the identical 19-type set (manual only claims verified on `cpg_falkorchat`; confirmed no divergence) |
| TP-005 | §1 Try-it query 1 (find method by name) | **PASS** | `MATCH (m:METHOD {NAME:'post_message'})...` on `cpg_falkorchat` returned 3 rows (`api.py`, `services.py`, a `<returnValue>` phantom) |
| TP-006 | §1 Try-it query 2 (callees) | **PASS** | Returned `_dispatch_write`, `_validate_and_derive_role`, `_next_ts` |
| TP-007 | §1 Try-it query 3 (callers) | **PASS** | Returned 21 callers across tests, `mcp.py`'s `send_message`, and `api.py`'s `post_message` itself |
| TP-008 | Same 3 queries on `cpg_salesperson` | **PASS** | Substituted `get_customer_profile` (real method, `customer_profile.py:54`) — find/callee(`_get_profile`)/caller(`render_sidebar` in `chatbot.py`) all returned sensible non-error results |
| TP-009 | §2 Try-it query 1 (notes by agent) | **PASS** | `author:'graph-dba'` returned 1 real `KaizenEntry` row |
| TP-010 | §2 Try-it query 2 (pending per agent) | **PASS** | Returned 6 agents, counts 4/4/2/2/1/1 (coder, analyst, architect, teco, qa-engineer, graph-dba) |
| TP-011 | `kaizen_team` zero relationship types | **PASS** | `db.relationshipTypes()` returned 0 rows; `db.labels()` returned only `KaizenEntry` |
| TP-012 | `reference` labels/relationship types match | **PASS** | Labels: `Entity`, `Step`, `WorkflowDef` (exact match); relationship types: `HAS_STEP`, `START`, `TRANSITION` (exact match) |
| TP-013 | §3 Try-it query 1 (`access-request`/`v1` START) | **PASS** | No substitution needed — `access-request`/`v1` is live; returned `submit`/`human` |
| TP-014 | §3 Try-it query 2 (steps+transitions) | **PASS** | Returned 6 transition pairs (`submit→route`, `route→approval`, `route→provision`, `approval→provision`, `approval→rejected`, `provision→activate`) |
| TP-015 | `Entity` zero live nodes in `reference` | **PASS** | `MATCH (n:Entity) RETURN count(n)` → `0` |
| TP-016 | `ws:acme` labels match §4a/4b/4c union | **PASS** | `db.labels()` returned exactly the 15 expected labels (`Agent`,`Channel`,`Chunk`,`Document`,`Entity`,`Message`,`ReadCursor`,`Step`,`StepRun`,`Thread`,`TraceEvent`,`User`,`WorkflowDefSnapshot`,`WorkflowRun`,`WorkspaceConfig`) |
| TP-017 | `ws:acme` relationship types match §4a/4b/4c union | **FAIL** | `db.relationshipTypes()` returned `TRANSITION` live (confirmed real, 8 edges within a `WorkflowDefSnapshot`) — **not listed anywhere in §4c's relationship-type table**. See Defect 1 |
| TP-018 | §4a Try-it query 1 (thread walk) | **PASS** | `threadId:'demo-welcome'`, `HEAD`→`NEXT*0..` returned 50 ordered messages, `user`/`assistant` roles, ascending `createdAt` |
| TP-019 | §4c Try-it query 2 (`AT_STEP` lookup) | **PASS** | Sampled 10 `WorkflowRun`s: 3 status values present (`done`×10, `waiting`×7, `failed`×4 across the full set); picked a `waiting` run (`runId b6465871…`) → returned `waiting`/`submit`/`human` |
| TP-020 | `ws:acme` zero-node claims: `Document`/`Chunk`/`Entity`/`WorkspaceConfig` | **PARTIAL FAIL** | `Document`=0, `Chunk`=0, `Entity`=0 (matches the §4b "dormant corpus" claim exactly); `WorkspaceConfig`=0 (label present via `db.labels()`, zero live nodes) — **contradicts §4c's separate, unhedged "one node per workspace" claim**. See Defect 2 |

## Defects

### Defect 1 — `TRANSITION` relationship type missing from §4c's schema table (Medium)

**Where:** `docs/manuals/graph-ontology.md`, §4c "Workflow run model," relationship-type table
(lines ~393–404).

**Claim:** The table lists `HAS_STEP`/`START`, `OF_DEF`, `AT_STEP`, `TRIGGERED_BY`,
`HAS_STEP_RUN`, `LAST_STEP_RUN`, `RAN`, `NEXT`, `PRODUCED`, `TRACED` as the complete set of
relationship types in a workspace's workflow-run model.

**Actual:** `CALL db.relationshipTypes()` against `ws:acme` also returns `TRANSITION`, and it
is real live data, not an artifact:

```cypher
MATCH (:WorkflowDefSnapshot)-[:HAS_STEP]->(s:Step)-[:TRANSITION]->(next:Step)
RETURN s.key, next.key LIMIT 10
```
returns 8 rows (`intake→research`, `research→answer`, `provision→activate`,
`approval→rejected`, `approval→provision`, `route→provision`, `route→approval`,
`submit→route`) — i.e. the materialized `Step` graph inside a `WorkflowDefSnapshot` carries
the same `TRANSITION` edges §3 documents for `reference`, exactly as the manual's own prose
promises ("the server copies... that version's step subgraph into the workspace's own graph
... real local edges"). The edge type itself is just never listed in §4c's table.

**Impact:** A reader who wants to walk a *workspace's own* materialized transition graph
(rather than reach back into `reference`, which the manual correctly says you shouldn't do)
has no way to learn `TRANSITION` exists there from §4c alone — they'd have to already know it
from §3 and guess it carries over. Low severity in practice (the edge works exactly like its
`reference` counterpart, and an alert reader can infer it), but it's a real gap in an
otherwise exhaustively-verified table.

**Reproduce:** `mcp__cypher__query(graph="ws:acme", cypher="CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")` — `TRANSITION` is in the result set; it is absent from the manual's §4c table.

**Suggested fix:** Add a `TRANSITION` row to §4c's relationship table, e.g. `Step → Step`,
"Same meaning as in `reference` — materialized along with the rest of the snapshot," and
optionally add the edge to the §4c mermaid diagram.

### Defect 2 — `WorkspaceConfig` existence claim contradicts live data and the manual's own hedge two paragraphs later (Low-Medium)

**Where:** `docs/manuals/graph-ontology.md`, §4c, the "Also present, singleton config" line
(≈lines 422–424), versus the §4 gotchas bullet "Not every label with a schema has live data"
(≈lines 441–444).

**Claim (unhedged):** "**Also present, singleton config:** `WorkspaceConfig` — one
`{workspaceConfigId: 'default'}` node per workspace holding optional per-kind LLM/embedding
model overrides." Read on its own, this asserts every workspace has exactly one such node.

**Claim (hedged, same section, later):** The gotchas bullet lists `WorkspaceConfig` as an
example of a label that "can report... currently zero nodes in a given workspace... on a
workspace that hasn't exercised those paths yet" — directly contradicting the "Also present"
framing just above it.

**Actual (`ws:acme`):**
```cypher
MATCH (n:WorkspaceConfig) RETURN n LIMIT 5   -- 0 rows
```
`db.labels()` lists `WorkspaceConfig` (the label/index exists), but there is no live node —
confirming the *hedged* claim, not the "Also present... one... node per workspace" claim.
`Document`/`Chunk`/`Entity` are correctly documented as zero (§4b's "dormant corpus" callout
is explicit and unhedged in the right direction, and matches live data exactly — no defect
there).

**Impact:** A reader who takes the "Also present, singleton config" line at face value and
writes `MATCH (c:WorkspaceConfig {workspaceConfigId:'default'}) RETURN c.embeddingModel` (or
similar) against `ws:acme` today gets an empty result and no signal that this is expected —
the one sentence that would tell them so (the gotchas bullet) doesn't call out that the
*primary* description above overstates presence.

**Reproduce:** `mcp__cypher__query(graph="ws:acme", cypher="MATCH (n:WorkspaceConfig) RETURN count(n)")` → `0`.

**Suggested fix:** Soften "Also present, singleton config... one node per workspace" to
something like "created lazily on first per-kind model override — zero or one node per
workspace" (or confirm with `falkor-chat`'s actual create-on-first-use behavior in
`falkor-chat/docs/DESIGN.md` and word it precisely); the two claims should say the same
thing.

## Coverage & gaps

**Covered:** every schema table (`db.labels()`/`db.relationshipTypes()`) in all four
sections, every "Try it" example query in all four sections (10 queries total, one set
substituted per the task brief for `cpg_salesperson`), and all three explicit "zero live
nodes" claims (`reference:Entity`, `ws:acme:Document`/`Chunk`/`Entity`, plus the
`WorkspaceConfig` claim this run added scrutiny to).

**Not covered (deliberately, per test plan):** the manual's non-runnable narrative claims
(design rationale, "who writes/reads" columns, the FAQ prose) — that's `analyst`'s half of a
full manual review, not a black-box behavioral check. Also not covered: `identity` and
`kaizen_analyst` (explicitly out of scope in the manual itself) and the other four `ws:*`
graphs (`ws:eval`/`ws:qa028`/`ws:test`/`ws:qa-tico-workflows-manual`) — the manual's claims
are stated to hold "identically to every `ws:<id>` graph," and `ws:acme` was the designated
sample; a full audit would spot-check at least one more to confirm the "identically" claim
empirically, which this run did not do (residual risk: low, since the schema itself is
enforced by the same server code path for every workspace).

**Residual risk:** low. Both defects are documentation-only (nothing in the manual's example
queries is broken, mis-cased, or wrong-directioned), and both are narrow, well-localized
fixes in §4c.

## Feedback & recommendations

- The manual's live-verified schema tables (§1, §2, §3, and 14 of 15 labels + 19 of 20
  relationship types in §4) are excellent — every one matched the running data exactly on
  first query, which is a strong signal the "verified live" claims in the doc are genuine and
  not aspirational.
- Both defects sit in §4c specifically (the workflow-run model), which is also the newest and
  most recently-changed sub-model per the repo's `HISTORY.md` (K-028 workflow timers landed
  2026-08-21, same day as this manual). Worth a quick pass over §4c specifically the next time
  `falkor-chat`'s workflow-run schema changes, since it's the section most likely to drift.
- No flakiness or environmental issues encountered — the shared `falkordb-dev` instance
  answered every one of the ~24 queries run in this session in under 160ms, and no query
  needed a retry.

## Artifacts

- Test plan: `docs/test-plans/graph-ontology.md`
- Test report (this document): `docs/test-reports/graph-ontology-report.md`
- Subject under test (not modified): `docs/manuals/graph-ontology.md`
