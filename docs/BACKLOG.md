# Backlog — CPG code-graph component

> Forward-looking backlog for the repo-root **CPG / code-graph** component (Joern → FalkorDB;
> requirements in [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md)).
> Delivered work is logged in [`HISTORY.md`](./HISTORY.md).
> Item IDs use the `C-` prefix (distinct from falkor-chat's `K-`); the hundreds digit tracks the
> milestone (C-2xx = M2).
> Status: 🔵 proposed · 🟡 in-progress · ✅ done · ⚪ deferred
> Last reviewed: 2026-07-18.

## Handoff — `teco` drives M2 (2026-07-18)

`teco` coordinates M2 from here. This section is the cold-start brief; everything below is the detail.

**Read first (entry points):**
1. [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md) — WHAT/WHY; M2 = FR-9…FR-14, AC-6…AC-8.
2. This backlog — the C-201…C-208 units, ownership, and sequencing.
3. [`../skills/joern-cpg/references/cpg-model.md`](../skills/joern-cpg/SKILL.md) — the **single** CPG
   schema/label/property contract the recipes cite (FR-14); do not duplicate it.
4. [`../skills/agent-standards`](../skills/README.md) — cobb's skill-authoring/lint standards the new
   skill must pass.

**Already decided — do not re-litigate:**
- **Shape:** one `cpg-analysis` skill = lean `SKILL.md` core + four bundled recipes (not four sibling skills).
- **Ownership:** `graph-dba` builds/owns the skill (Cypher over a loaded FalkorDB graph); `cobb`
  vets it against skill standards; `teco` coordinates. `analyst` is the independent reviewer of the
  graph-dba deliverable (producer ≠ reviewer, per the team's review-gate convention).
- **Scope:** all four recipes are in (impact, RCA, code-review, test-gap). `qa-engineer` is a named
  consumer. Runtime coverage is **excluded** — test-gap is structural reachability only.
- **Naming:** `cpg-test-gap`, not `cpg-test-coverage`.

**Open items to route during planning (don't block C-201…C-204):**
- **OQ2** — component structure/naming (a `code-graph/` dir vs. living as the `joern` agent + skills) → `architect`.
- **OQ3** — Joern Python + JS/TS frontend coverage adequacy → design-time verification.

**Done-condition reminders (per repo `AGENTS.md`):** skill source + `skills/README.md` + agent-description
wiring + this backlog→`HISTORY.md` land in the **same** change (C-208); the skill is **live-verified**
against a real loaded CPG before M2 is called ✅, not just authored.

## Milestone map

| Milestone | Reaches ✅ when | Items |
|---|---|---|
| **M1 — Producer pipeline** ✅ | CPG builds from source and loads into FalkorDB, live-verified | `joern` agent + `joern-cpg` skill — delivered 2026-07-17, commit `b2b9a6e` (see [`HISTORY.md`](./HISTORY.md)) |
| **M2 — CPG consumer skill** 🔵 | One `cpg-analysis` skill (FR-9…FR-14) lets `analyst`/`architect`/`qa-engineer` run impact / RCA / code-review / test-gap recipes against a loaded CPG via Cypher, cobb-vetted, catalogs updated | **C-201 → C-208** |

### Decision — skill is the access mechanism (user, 2026-07-18)

Approved shape (resolves requirements **OQ1**): **one `cpg-analysis` skill**, not four sibling
skills. A lean `SKILL.md` core (connection + shared traversal idioms) plus four bundled
`references/*.md` recipes loaded on demand — the same core-plus-references pattern
`agent-maintenance`/`agent-standards` use. This keeps the CPG schema/label contract in **one**
place (cited from `skills/joern-cpg/references/cpg-model.md`) so it can't drift four ways.

**Ownership:** the skill queries a *loaded FalkorDB graph with Cypher*, so **`graph-dba`** owns it
(not `joern`, which owns build→export→load); **`cobb`** vets it against skill standards;
**`teco`** coordinates the multi-agent build. Renamed `cpg-test-coverage` → **`cpg-test-gap`**: a
static CPG has structure/reachability, not runtime line/branch coverage.

## M2 — CPG consumer skill (`cpg-analysis`)

- **C-201 — Adopt the schema contract.** 🔵 Confirm `skills/joern-cpg/references/cpg-model.md` is
  the canonical node/edge/property reference the recipes cite; fill any *consumer-query* gap
  (label → Cypher idiom mapping, the UPPER_CASE-property + `id`-lowercase + real-boolean gotchas).
  *No new schema doc — reuse.* Owner: graph-dba.
- **C-202 — Skill core (`SKILL.md`).** 🔵 FalkorDB connection via `redis-cli GRAPH.QUERY`; the
  `CpgNode(id)` model + per-label index reality; shared traversal idioms (callers/callees over
  `CALL`, transitive reach, data-flow over `REACHING_DEF`, symbol def/ref). Owner: graph-dba.
- **C-203 — Recipe: impact-analysis.** 🔵 Callers/callees + transitive up/downstream reach —
  **FR-2, FR-3 / AC-2, AC-3**. Consumers: `analyst`, `architect`. *In-scope, no reqs change.*
- **C-204 — Recipe: rca.** 🔵 Data-flow back from a symptom (`REACHING_DEF`) + cross-file symbol
  def/ref — **FR-4, FR-5 / AC-4, AC-5**. Consumer: `analyst`. *In-scope, no reqs change.*
- **C-205 — Recipe: code-review.** 🔵 Taint/sink & suspicious-pattern queries (data-flow to risky
  calls) — **FR-12 / AC-7**. Consumer: `analyst`. *(Was a scope extension; approved into scope
  2026-07-18 — no longer gated.)*
- **C-206 — Recipe: test-gap.** 🔵 Reachability from prod entrypoints vs. test entrypoints
  (code reachable in prod but from no test) — **FR-13 / AC-8**. Consumer: `qa-engineer`. *(Was a
  scope extension; approved into scope 2026-07-18 — no longer gated.)*
- **C-207 — Agent wiring.** 🔵 Add CPG-capability lines to `analyst` / `architect` / `qa-engineer`
  descriptions (their live routing contract); note graph-dba ownership. cobb reviews. Owner: cobb.
- **C-208 — Catalog & doc sync.** 🔵 `skills/README.md` (new skill), root `AGENTS.md`, `claude/README.md`
  if agent descriptions change, and this backlog → `HISTORY.md` on delivery. Per AGENTS.md, skill +
  catalog + agent wiring land in the **same** change.

### Requirements coverage

- **C-200 — Requirements pass for the two scope extensions.** ✅ **Resolved 2026-07-18 (user)** —
  code-review and test-gap folded into the requirements as **FR-12/FR-13 (AC-7/AC-8)** with
  `qa-engineer` named as a consumer; OQ1/OQ4 closed. All of M2 (C-201…C-208) now has requirements
  backing.

## Sequencing

```
C-201 (schema contract) ─▶ C-202 (skill core) ─┬─▶ C-203 (impact)    ─┐
                                                ├─▶ C-204 (rca)       ─┤
                                                ├─▶ C-205 (code-review)┼─▶ C-207 (wiring) ─▶ C-208 (catalogs) ⇒ M2 ✅
                                                └─▶ C-206 (test-gap)  ─┘
Critical path: C-201 → C-202 → C-203/C-204 → C-207 → C-208 (recipes C-205/C-206 parallel after C-202).
```
