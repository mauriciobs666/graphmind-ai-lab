# Test Plan — CPG Getting Started manual

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-201…C-208, C-301…C-307 (M1–M3)

## Scope & objective

Behaviorally verify `docs/manuals/cpg-getting-started.md` — a `tico`-authored, stakeholder-facing
manual for the Joern-CPG-in-FalkorDB capability — against the actual running system. The manual's
own walkthroughs are the spec: each step is a test item, and the "expected result" is exactly what
the manual claims. This is a manual-verification pass (smaller than a full feature QA pass), scoped
to the manual's three walkthroughs ("checking you're ready", "asking the CPG a question", "building
a CPG") plus the specific behavioral guarantees the manual asserts (read-only enforcement,
truncation notice format, graph-not-found messaging).

Out of scope, and why:
- **Actually building a CPG.** Per the requester's instruction, this is slow (minutes) and
  destructive (wipes an existing graph). Two CPGs are already loaded (`cpg_falkorchat`,
  `cpg_salesperson`), sufficient to verify the query walkthrough without triggering a rebuild.
  The "building a CPG" walkthrough's factual claims are spot-checked statically against
  `skills/joern-cpg/SKILL.md` and `cypher-mcp/README.md` instead.
- **The manual's factual/architectural claims** (e.g. "Joern reads source and builds nodes for
  methods/call sites/parameters/literals") — that's `analyst`'s half of a `tico`-authored-manual
  review, routed separately by `teco`. This plan covers only claims that are checkable by
  *executing* the system.
- **JS/TS CPG behavior.** Both loaded CPGs are Python-only (per `cpg-analysis/SKILL.md`'s own
  documented coverage boundary); the manual makes no language-specific claim, so this is a
  non-issue, not a gap.

## References

- `docs/manuals/cpg-getting-started.md` (document under test)
- `cypher-mcp/README.md` (tool contract: read-only, truncation, EXPLAIN/PROFILE, error behavior)
- `skills/cpg-analysis/SKILL.md` §1–2 (query surface, the five gotchas)
- `skills/joern-cpg/SKILL.md` (build pipeline, reload-is-destructive claim)

## Risk assessment

| Risk | Why it matters | Priority |
|---|---|---|
| "Checking you're ready" steps don't actually confirm readiness | Stakeholder follows the manual, gets false confidence or a dead end | High |
| Read-only guarantee doesn't hold | Manual explicitly sells this as a safety property ("even by mistake") — if false, it's a trust-breaking defect | High |
| Truncation notice doesn't match the manual's description | Manual tells readers how to recognize a truncated answer; if the real notice differs, readers misread results | High |
| "No CPG loaded" messaging doesn't list what's loaded | Manual's FAQ makes a specific claim about error quality | Medium |
| Build-walkthrough claims (destructive reload, guarded delete) are stale | High blast radius if wrong, but not exercisable here — mitigated by static fact-check only | Medium |
| Graph naming convention (`cpg_<component>`) doesn't hold in practice | Cosmetic/informational claim | Low |

## Test items

| ID | Title | Preconditions | Steps | Expected result | Priority | Type |
|---|---|---|---|---|---|---|
| TP-001 | FalkorDB reachable | none | `redis-cli -p 6379 PING` | `PONG` | High | functional |
| TP-002 | `cypher` MCP tool connected | Claude Code session | `claude mcp list` | `cypher` listed, `✔ Connected` | High | functional |
| TP-003 | A CPG is already loaded (avoid a needless rebuild) | FalkorDB up | `redis-cli -p 6379 GRAPH.LIST` | At least one `cpg_<name>` graph present | High | functional |
| TP-004 | Basic query against a loaded CPG | TP-003 passes | `mcp__cypher__query(graph="cpg_falkorchat", cypher="MATCH (m:METHOD) RETURN count(m) AS n")` | Plain-text result with a stats line (`graph=… rows=1 …`) and a count | High | functional |
| TP-005 | Impact-analysis query shape ("who calls X") | TP-004 passes | Run the `cpg-analysis` "callers of a method" recipe against a real method found in `cpg_falkorchat` | Rows listing caller `FULL_NAME`/`FILENAME`/`LINE_NUMBER`, or a clean empty result | High | functional |
| TP-006 | Read-only enforcement | TP-004 passes | `mcp__cypher__query(graph="cpg_falkorchat", cypher="CREATE (n:QaProbe {x:1}) RETURN n")` | Rejected server-side (error), no node created | High | functional |
| TP-007 | Unmatched query returns clean empty result | TP-004 passes | Query with a condition matching nothing | Stats line + column names + `(no rows)`, not an error | Medium | functional |
| TP-008 | Large/unfiltered query triggers truncation notice | TP-004 passes | `MATCH (n) RETURN n` (or similar) on a CPG with more rows than the cap | Notice naming the cap, shown-vs-true row counts, and the true `rows=` figure in the stats line | High | functional |
| TP-009 | Unknown graph name lists what is loaded | TP-002 passes | `mcp__cypher__query(graph="cpg_does_not_exist", cypher="MATCH (n) RETURN n LIMIT 1")` | Error message enumerating the graphs that *are* loaded (per manual FAQ: "no CPG loaded for that name lists what is loaded") | Medium | functional |
| TP-010 | Graph naming convention | TP-003 | Inspect `GRAPH.LIST` output | Loaded CPGs follow `cpg_<component>` | Low | functional |
| TP-011 | Build-walkthrough claims (static) | none — not executed live | Read `skills/joern-cpg/SKILL.md` §"Reloading is deliberate (destructive)" and `cypher-mcp/README.md` | Confirms: loader refuses non-empty graph; reset is an explicit guard-gated `GRAPH.DELETE`; destructive FalkorDB ops are approval-gated | Medium | static fact-check |
| TP-012 | `EXPLAIN` yes / `PROFILE` no (static, optionally spot-live) | TP-004 passes | Fact-check against `cypher-mcp/README.md`; optionally try `EXPLAIN` through the tool | Manual's claim ("read-only, enforced server-side... not just convention") is consistent with README's documented `PROFILE` refusal and `GRAPH.RO_QUERY` behavior | Low | static + optional functional |

## Environment & data setup

No setup required beyond what's already running: shared `falkordb-dev` container (started by
`falkor-chat/scripts/start_falkordb.sh`), already-loaded `cpg_falkorchat` / `cpg_salesperson`
graphs, and the `cypher` MCP server wired via the repo-root `.mcp.json`. No test data is created;
TP-006's write attempt is designed to fail closed and leave no residue (verified by re-querying
afterward).

## Entry / exit criteria

- **Entry:** FalkorDB up, `cypher` MCP tool connected, at least one CPG loaded (TP-001…TP-003 must
  pass before continuing — if any is blocked, the rest of the live items are blocked too, and the
  blocker is reported plainly rather than worked around).
- **Exit:** All items executed or explicitly marked blocked/skipped with a reason; every defect
  reproduced and recorded; report written and cross-referenced by `TP-NNN` ID.

## Out of scope (restated)

- Executing the "building a CPG" walkthrough end-to-end (destructive, slow, not necessary to
  verify the querying claims — two CPGs are already loaded).
- The manual's static/architectural claims (Joern internals, node/edge semantics) — `analyst`'s
  half of this review.
- JS/TS-frontend CPG behavior (out of scope for the manual itself).
