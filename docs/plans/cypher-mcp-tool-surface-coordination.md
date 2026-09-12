# Cypher MCP tool surface — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** C-901, C-902 (M9)

## Goal & definition of done

Deliver `docs/requirements/cypher-mcp-tool-surface.md` (Status: Ready for design, FR-1…FR-4,
AC-1…AC-3): give any caller of `mcp__cypher__query` a **direct** way to list every FalkorDB graph
currently loaded, without a second tool, a required new parameter, or a widened blast radius.
`docs/plans/cypher-mcp-tool-surface.md` (architect, v1, Status: active) is already written —
a fourth `GRAPHS` directive alongside `EXPLAIN`/`PROFILE`, no schema change, one implementer/one
pass per the plan's own step table (`server.py`, its tests, `cypher-mcp/README.md`,
`docs/requirements/cpg-query-access.md`'s header-note, a `docs/HISTORY.md` closeout entry).

This coordination opens because the chain carries a plan-gate review, an implementation unit, and
a post-implementation re-gate (3 units, one gate-bearing) — over the "hold it in the report"
threshold.

**Environment:** FalkorDB reachable (`redis-cli -p 6379 ping` → `PONG`, checked at coordination
open). No CPG relevant to this delivery — the plan's own §0 CPG-freshness note (this is a design
task over `cypher-mcp/server.py` and two markdown docs; no Joern CPG covers this repo's own MCP
tooling).

## Unit ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `analyst` | `a585f7ceb38e1ee2e` | accepted | `docs/reviews/cypher-mcp-tool-surface.md` — no blockers. Major: plan §2's cited README test-gate pass counts (74/7 offline, 7/74 live) are stale — real re-run is 113 passed/10 deselected offline, 10/113 live; self-correcting via the plan's own step-3 "recount, don't trust the plan" done-condition. Minor: `GRAPHS` directive tolerates only leading trivia, not trailing (e.g. `GRAPHS // note` falls through to a raw FalkorDB syntax error) — left to implementer judgment. Nit: §6 miscites `cpg-query-access.md`'s "D5" label (substance correct, citation wrong) — non-blocking, not fixed. Central collision-safety and FR-3 blast-radius claims independently traced and confirmed sound; `split_directive()` diff confirmed genuinely additive. | plan gate → **approve with suggestions** | 200.5k tok / 26 tools |
| U2 | `coder` | `a393b6820196ae196` | delivered | `cypher-mcp/server.py` (fourth `GRAPHS` directive, `format_graph_list()`, docstrings/instructions), `cypher-mcp/tests/test_server.py` (+13 tests incl. 1 live), `cypher-mcp/README.md` (Graph discovery section + freshly-recounted gate numbers), `docs/requirements/cpg-query-access.md` (header-note only, `git diff`-confirmed), `docs/HISTORY.md` (closeout entry) — all uncommitted. Offline 125/11 (was 113/10), live 11/125, container gate 116/11 + 11/116, no `_cypher_mcp_selftest_*` residue. FR-2 pins (`test_input_schema_has_two_required_params_and_one_optional_agent`, `test_exactly_one_tool_named_query`) re-run unedited, pass. `graph`-never-read asserted structurally. 2 mutations run one-at-a-time with restore-and-reverify between (drop `\s*\Z` anchor → named-alternative test failed as expected; delete `"graphs"` branch → all 4 new tests failed as expected). Trailing-trivia finding: chose doc-caveat over a regex fix (reasoned in result), stated as a judgment call. | re-gate (`analyst`) → — | 167.7k tok / 52 tools |
| U3 | `analyst` | `a585f7ceb38e1ee2e` | accepted | `docs/reviews/cypher-mcp-tool-surface.md` `## Pass 2` — diff matches plan §4 exactly; own Major finding re-derived independently (125/11 offline, 11/125 live — matches teco's own independent re-run too), container-gate arithmetic (125−9=116 host-only items) verified; trailing-trivia doc-caveat disposition judged acceptable (stays fail-safe); FR-2 pin tests confirmed byte-for-byte absent from the diff; `cpg-query-access.md` diff confirmed header-only; no mutation/debug residue. | re-gate → **approve** | 221.3k tok / 14 tools |

**All gates closed.** teco independently re-ran the offline suite (`125 passed, 11 deselected`, matching both the implementer's and analyst's figures) and confirmed via `git diff` that `docs/requirements/cpg-query-access.md` changed only its header block. Committed `<see commit below>`. Milestone M9 closed in the same pass: `docs/requirements/cypher-mcp-tool-surface.md`, `docs/plans/cypher-mcp-tool-surface.md`, `docs/reviews/cypher-mcp-tool-surface.md`, and this coordination doc all flipped to `Status: archived` (mechanical Status-only edits).

## Notes

- Sequencing: U1 (plan gate) → U2 (`coder`, implementation, gated on U1's verdict) → U3 (`analyst`,
  diff-scoped re-gate) → teco integration/commit.
- The plan is explicitly sized "small enough for one implementer, one pass" (5 tightly-coupled
  steps, one coherent feature diff) — implementation stays one dispatch, not split per step.
