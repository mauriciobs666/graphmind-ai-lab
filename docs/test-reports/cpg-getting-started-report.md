# Test Report — CPG Getting Started manual

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-201…C-208, C-301…C-307 (M1–M3)

## Summary

Behaviorally verified `docs/manuals/cpg-getting-started.md` against the live system on
2026-07-30: FalkorDB (`falkordb-dev`, port 6379) up and reachable, `cpg` MCP tool connected
(`claude mcp list` → `✔ Connected`), two CPGs already loaded (`cpg_falkorchat`, `cpg_salesperson`)
— so the "asking a question" walkthrough was exercised live without triggering a rebuild, per the
plan's explicit scoping. The "building a CPG" walkthrough was **not executed** (destructive, slow,
unnecessary given loaded CPGs); its factual claims were spot-checked statically against
`skills/joern-cpg/SKILL.md` and `cpg/mcp/README.md`.

**Overall verdict: mostly accurate, one real defect found.** Every "checking you're ready" and
"asking a question" step worked exactly as described, including the read-only guarantee and the
"unknown graph lists what's loaded" behavior. One specific factual claim — that the reported
`rows=` figure "is always the true total" — is **measurably false** once the true result count
exceeds FalkorDB's own server-level default cap (10,000 rows), which neither the manual nor its
source (`cpg/mcp/README.md`) accounts for. See TP-008 / DEF-001.

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-001 | **Pass** | `redis-cli -p 6379 PING` → `PONG` |
| TP-002 | **Pass** | `claude mcp list` → `cpg: ... - ✔ Connected` |
| TP-003 | **Pass** | `GRAPH.LIST` → `cpg_falkorchat`, `cpg_salesperson`, `ws:acme`, `reference`, `ws:test` |
| TP-004 | **Pass** | `mcp__cpg__query(cpg_falkorchat, "MATCH (m:METHOD) RETURN count(m) AS n")` → `graph=cpg_falkorchat · rows=1 · 73.9ms`, `n=1968` |
| TP-005 | **Pass** | Callers-of-`post_message` recipe → 21 rows incl. `falkorchat/api.py:...post_message` (line 144) and `falkorchat/mcp.py:...send_message` (line 53) — matches the exact example in `cpg/mcp/README.md`'s sample output (same two callers, close line numbers — expected drift since the README's is a snapshot from an earlier build) |
| TP-006 | **Pass** | `CREATE (n:QaProbe {x:1}) RETURN n` → rejected with curated message ("This tool is read-only (GRAPH.RO_QUERY)..."); follow-up `MATCH (n:QaProbe) RETURN count(n)` → `0`, confirming nothing was written |
| TP-007 | **Pass** | Query for a nonexistent method name → `rows=0`, column header `m.FULL_NAME`, `(no rows)` — matches manual/README's "ran, matched nothing" vs "failed" distinction |
| TP-008 | **Fail (defect, see DEF-001)** | `MATCH (n) RETURN n` on `cpg_falkorchat` → notice mechanism (dual first/last line, cap named, shown-of-total) works exactly as documented, **but** `rows=10000` while the true node count (verified via `MATCH (n) RETURN count(n)`) is **110048** — the reported total is not true above FalkorDB's `RESULTSET_SIZE` (confirmed `10000` via `GRAPH.CONFIG GET RESULTSET_SIZE`) |
| TP-009 | **Pass** | Query against `cpg_does_not_exist` → `Graph 'cpg_does_not_exist' does not exist. Loaded graphs: cpg_falkorchat, cpg_salesperson, ws:acme, reference, ws:test...`; `GRAPH.LIST` afterward confirms the typo'd name was **not** materialized |
| TP-010 | **Pass** | Loaded CPGs are `cpg_falkorchat`, `cpg_salesperson` — both follow `cpg_<component>` |
| TP-011 | **Pass (static)** | `skills/joern-cpg/SKILL.md` §"Reloading is deliberate (destructive)": loader refuses a non-empty graph, reset is an explicit `redis-cli GRAPH.DELETE` marked "destructive, shared-state — escalates via graph-dba's guard"; corroborated live by the existence of `claude/graph-dba/hooks/guard-destructive-ops.sh` (not executed — no rebuild was performed) |
| TP-012 | **Pass** | `EXPLAIN MATCH (m:METHOD) RETURN m.NAME LIMIT 5` → returned a query plan, no rows, prefixed "plan only — nothing was executed"; `PROFILE MATCH (m:METHOD) RETURN m.NAME LIMIT 5` → refused with "PROFILE is not available through this tool: GRAPH.PROFILE executes the query, including writes..." — matches `cpg/mcp/README.md` exactly |

Bonus cross-check (not a plan item, surfaced incidentally): `skills/joern-cpg/SKILL.md`'s own
scale example ("41 files → 110k nodes / 735k edges") matches `cpg_falkorchat` almost exactly —
live count came back **110048 nodes / 734929 edges** — good evidence the currently-loaded CPG is
the same one the docs were written against, and that the manual's scale-estimate claim (~2,700
nodes / ~18,000 edges per Python file) is grounded in a real, still-live measurement rather than a
stale guess.

## Defects

### DEF-001 — "the underlying count is always the true one" is false above FalkorDB's 10,000-row default cap

**Severity:** Medium (documentation/trust defect — not data loss or a security issue, but it
undermines a specific, explicitly-stated safety property the manual sells to build reader trust
in truncated results).

**Where the claim appears:** `docs/manuals/cpg-getting-started.md`, "Walkthrough — asking the CPG
a question": *"Long answers are only shown trimmed, never silently wrong. ... the underlying count
is always the true one."* The same claim, worded almost identically, is in `cpg/mcp/README.md`:
*"the `rows=` figure is always the **true** total... `MATCH (n) RETURN n` on a CPG still fetches
tens of thousands of rows."*

**Steps to reproduce:**
1. `redis-cli -p 6379 GRAPH.CONFIG GET RESULTSET_SIZE` → `10000` (FalkorDB's own server-level
   default cap on any query's result set, independent of the `cpg` MCP tool).
2. `mcp__cpg__query(graph="cpg_falkorchat", cypher="MATCH (n) RETURN count(n) AS total")` →
   `total = 110048` (the true node count).
3. `mcp__cpg__query(graph="cpg_falkorchat", cypher="MATCH (n) RETURN n")` → notice line says
   `showing 132 of 10000 rows`, and the stats line says `rows=10000`.
4. Confirmed the FalkorDB-level cap, not the tool's own `CPG_MCP_MAX_ROWS`, is what binds: an
   explicit `MATCH (n) RETURN n.id LIMIT 50000` via raw `redis-cli GRAPH.RO_QUERY` still returns
   only ~10,000 rows.

**Expected:** Per the manual and the README, the `rows=` figure reported alongside a truncation
notice is the true, exact total — readers are told explicitly they can trust it even when the
*rendered* rows are an arbitrary sample.

**Actual:** When the true result count exceeds FalkorDB's own `RESULTSET_SIZE` (10000 by default,
a layer *beneath* the `cpg` MCP tool and not mentioned in either document), the reported `rows=`
is **also capped at 10000** and is silently wrong — it understates the true total by roughly 10×
in this observed case (110048 actual vs. 10000 reported), with no signal distinguishing "this is
the exact count" from "this is itself a truncated count."

**Recommendation for the manual:** Soften the claim — something like *"the reported total is
accurate for results up to a few thousand rows; for very large unfiltered queries (tens of
thousands of rows or more), use `count()`/an aggregate to get a trustworthy total, since the
displayed `rows=` figure can itself be capped by FalkorDB's own result-set limit."* Since
`cpg/mcp/README.md` carries the identical overclaim and is the manual's own cited source, this is
worth flagging upstream too (not fixed here, per the QA role's no-edit boundary) — whoever owns
that document should decide whether to correct the claim or make the tool surface the
`RESULTSET_SIZE` cap explicitly in its own notice.

No other defects found among the executed items.

## Coverage & gaps

**Covered live:** FalkorDB reachability, MCP tool connection status, graph discovery, basic query
execution, the impact-analysis ("callers of X") recipe end-to-end against a real method, read-only
write rejection (with residue check), clean-empty-result formatting, truncation-notice mechanics
(and the defect above), unknown-graph error messaging (with a materialization check), `EXPLAIN`
and `PROFILE` behavior.

**Covered statically only (by design — see test plan's scope):**
- The full "building a CPG" walkthrough. Not executed: destructive to a live shared graph, minutes
  of runtime, and unnecessary since two real CPGs were already available to verify the query-side
  claims. Spot-checked instead: "reload refuses a non-empty graph" and "delete is guarded" both
  confirmed against `skills/joern-cpg/SKILL.md`'s explicit prose, and the guard's existence
  confirmed via `claude/graph-dba/hooks/guard-destructive-ops.sh` on disk.
- Scale claims ("~2,700 nodes / ~18,000 edges per Python file", "multi-million-edge territory for
  a large repo") — corroborated by the bonus cross-check above rather than a fresh measurement.

**Not checked at all:**
- JS/TS-frontend CPG behavior — no JS/TS CPG is loaded, and the manual makes no language-specific
  claim, so this is a non-gap, not a blocker.
- The `redis-cli` fallback path end-to-end for the query walkthrough (used only for the
  cross-verification of DEF-001 and for `GRAPH.LIST`/`GRAPH.CONFIG` checks) — the manual's
  querying walkthrough is written around `mcp__cpg__query`, which is what was exercised as the
  primary path.
- `graph-dba`'s actual build pipeline runtime characteristics (the "minutes, not seconds" claim,
  JVM startup cost) — not measured this run; would require executing the build.

**Residual risk:** Low outside of DEF-001. The verified surface (readiness checks, read-only
enforcement, error messaging, EXPLAIN/PROFILE divergence) is exactly the part a stakeholder is
most likely to rely on for trust, and it held up. DEF-001 is the one place the manual's language
could lead a reader to over-trust a number on a genuinely large, unfiltered query — which is
plausible reader behavior given the manual actively invites "ask any question."

## Feedback & recommendations

1. **Fix DEF-001's wording** in the manual (see recommendation above) — this is the main
   actionable finding.
2. **Consider flagging the same overclaim in `cpg/mcp/README.md`** to whoever owns that document;
   the manual inherited it faithfully from its cited source, so the manual can't fully self-correct
   without a compensating caveat unless the source also gets one.
3. **No testability issues encountered.** Both CPGs needed for verification were already loaded,
   the MCP tool was connected with no setup friction, and the manual's own "checking you're ready"
   section is sufficient in practice — a stakeholder following it literally (`PING`, `claude mcp
   list`, `GRAPH.LIST`) would reach exactly the state this test run found.
4. **The manual's FAQ and read-only guarantees are unusually well-grounded** — every specific,
   checkable claim in those sections (error messaging, no accidental graph creation, `EXPLAIN`
   vs `PROFILE` divergence) matched observed behavior exactly, including phrasing close enough to
   suggest it was written by reading the actual tool output rather than guessed from the design
   docs alone.

## Artifacts

- Test plan: `docs/test-plans/cpg-getting-started.md`
- Test report: `docs/test-reports/cpg-getting-started-report.md` (this document)
