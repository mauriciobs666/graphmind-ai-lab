# Graph Ontology Reference — manual verification test plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (—)

## Scope & objective

Verify, by executing live Cypher against the actual FalkorDB instance, that the walkthroughs
and factual claims in `docs/manuals/graph-ontology.md` (authored by `tico`) hold up against
the real data. This is a black-box behavioral check of a user manual: every "Try it" query is
run for real, and every schema table (`db.labels()` / `db.relationshipTypes()`) and "zero
live nodes" claim is checked against the running graphs, not read statically. The manual's
factual/architectural claims that are *not* runnable checks (e.g. design rationale, "who
writes it") are out of scope here — that's `analyst`'s half of a manual review, not this one.

## References

- `docs/manuals/graph-ontology.md` (subject under test)
- Live FalkorDB instance (`falkordb-dev`, port 6379) via `mcp__cypher__query`
- Graphs in scope: `cpg_falkorchat`, `cpg_salesperson`, `kaizen_team`, `reference`, `ws:acme`

## Risk assessment

Highest risk: schema tables and "Try it" queries are the parts a reader will copy-paste
verbatim — a wrong label, wrong property casing, or a query that silently returns nothing
wastes a reader's time and erodes trust in the whole document. "Zero live nodes" claims are
lower-risk to get wrong (informational) but easy to falsify with a single `count()` query, so
cheap to check. Out of scope: performance of the queries, the design rationale prose, and
graphs the manual explicitly marks out of scope (`identity`, `kaizen_analyst`).

## Test items

| ID | Section | Title | Steps | Expected | Priority | Type |
|---|---|---|---|---|---|---|
| TP-001 | §1 | `cpg_falkorchat` labels match manual table | `CALL db.labels()` | Matches manual's §1 label list exactly | High | contract |
| TP-002 | §1 | `cpg_falkorchat` relationship types match manual table | `CALL db.relationshipTypes()` | Matches manual's §1 relationship-type list exactly | High | contract |
| TP-003 | §1 | `cpg_salesperson` labels match manual table minus `CpgBuildInfo` | `CALL db.labels()` | Matches §1 label list minus `CpgBuildInfo` | High | contract |
| TP-004 | §1 | `cpg_salesperson` relationship types | `CALL db.relationshipTypes()` | Matches §1 relationship-type list (manual only claims verified on `cpg_falkorchat`; check for material divergence) | Medium | contract |
| TP-005 | §1 | Try-it query 1 (find method by name) | Run against `cpg_falkorchat` with `post_message` | Returns a row with FULL_NAME/FILENAME/LINE_NUMBER | High | functional |
| TP-006 | §1 | Try-it query 2 (callees) | Run against `cpg_falkorchat` with `post_message` | Returns sensible callee rows, no error | High | functional |
| TP-007 | §1 | Try-it query 3 (callers) | Run against `cpg_falkorchat` with `post_message` | Returns sensible caller rows, no error | High | functional |
| TP-008 | §1 | Same 3 queries against `cpg_salesperson` | Substitute a real method name found in that CPG | Returns sensible non-error results | Medium | functional |
| TP-009 | §2 | Try-it query 1 (notes by agent) | Run with a real `author` value | Returns rows, no error | High | functional |
| TP-010 | §2 | Try-it query 2 (pending notes per agent) | Run as-is | Returns aggregated rows, no error | High | functional |
| TP-011 | §2 | `kaizen_team` has zero relationship types | `CALL db.relationshipTypes()` | Empty result | High | contract |
| TP-012 | §3 | `reference` labels/relationship types match manual tables | `CALL db.labels()` / `CALL db.relationshipTypes()` | Matches §3 tables | High | contract |
| TP-013 | §3 | Try-it query 1 (`access-request`/`v1` START) | Run as-is; substitute if absent | Returns entry step, or documented substitution | High | functional |
| TP-014 | §3 | Try-it query 2 (steps + transitions) | Run as-is; substitute if absent | Returns step pairs, or documented substitution | High | functional |
| TP-015 | §3 | `Entity` has zero live nodes in `reference` | `MATCH (n:Entity) RETURN count(n)` | `0` | Medium | functional |
| TP-016 | §4 | `ws:acme` labels match manual's 4a/4b/4c tables | `CALL db.labels()` | Matches union of §4a/4b/4c label tables + `WorkflowDefSnapshot`/`StepRun`/`TraceEvent`/`WorkspaceConfig` | High | contract |
| TP-017 | §4 | `ws:acme` relationship types match manual's 4a/4b/4c tables | `CALL db.relationshipTypes()` | Matches union of §4a/4b/4c relationship tables | High | contract |
| TP-018 | §4a | Try-it query 1 (thread walk via HEAD/NEXT) | Find a real `threadId`, run the query | Returns ordered messages, no error | High | functional |
| TP-019 | §4c | Try-it query 2 (`AT_STEP` lookup) | Find a real `runId`, prefer `running`/`waiting` | Returns status+step, or documented non-terminal-run caveat | High | functional |
| TP-020 | §4b/4c | `Document`/`Chunk`/`Entity`/`WorkspaceConfig` node counts in `ws:acme` | `MATCH (n:<Label>) RETURN count(n)` per label | Report actual counts against manual's claims (note: manual's own text is internally inconsistent about `WorkspaceConfig` — see report) | Medium | functional |

## Environment & data setup

No setup required — all graphs are already loaded on the shared `falkordb-dev` instance.
All queries are read-only (`MATCH`/`RETURN`/`CALL db.*`); nothing is created, modified, or
deleted.

## Entry / exit criteria

**Entry:** FalkorDB reachable, all five in-scope graphs loaded (confirmed by the task brief).
**Exit:** all 20 items executed and recorded pass/fail/blocked with evidence; any divergence
from the manual's claims written up as a defect in the report.

## Out of scope

- The manual's non-runnable factual/architectural claims (design rationale, "who writes it"
  columns, the FAQ's narrative claims) — `analyst`'s territory for a full manual review.
- `identity` and `kaizen_analyst` graphs — manual explicitly marks both out of scope.
- `ws:eval`/`ws:qa028`/`ws:test`/`ws:qa-tico-workflows-manual` — task brief marks out of scope; only `ws:acme` is checked for §4.
- `cpg-getting-started.md` (a separate manual, separate family per the repo's document-family convention).
