# Backlog — CPG code-graph component

> Forward-looking backlog for the repo-root **CPG / code-graph** component (Joern → FalkorDB;
> requirements in [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md) and,
> for the read path, [`requirements/cpg-query-access.md`](./requirements/cpg-query-access.md)).
> Delivered work is logged in [`HISTORY.md`](./HISTORY.md).
> Item IDs use the `C-` prefix (distinct from falkor-chat's `K-`); the hundreds digit tracks the
> milestone (C-2xx = M2, C-3xx = M3).
> Status: 🔵 proposed · 🟡 in-progress · ✅ done · ⚪ deferred
> Last reviewed: 2026-07-25.

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
| **M2 — CPG consumer skill** ✅ | One `cpg-analysis` skill (FR-9…FR-14) lets `analyst`/`architect`/`qa-engineer` run impact / RCA / code-review / test-gap recipes against a loaded CPG via Cypher, cobb-vetted, catalogs updated — delivered 2026-07-19 | **C-201 → C-208** |
| **M3 — CPG query access (MCP)** ✅ | The read path is a single MCP tool `mcp__cpg__query(graph, cypher)` (`cpg/mcp/`) instead of a hand-assembled `redis-cli GRAPH.QUERY` command line; wired for Claude Code, skill + agents + requirements reconciled, CPG rebuilt, AC-1…AC-4 acceptance-tested — delivered 2026-07-25. DEF-1 / **C-313** closed the same day by stakeholder ruling **D5** (AC-3 reconciled, no code change) | **C-301 → C-307** |

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

- **C-201 — Adopt the schema contract.** ✅ Confirm `skills/joern-cpg/references/cpg-model.md` is
  the canonical node/edge/property reference the recipes cite; fill any *consumer-query* gap
  (label → Cypher idiom mapping, the UPPER_CASE-property + `id`-lowercase + real-boolean gotchas).
  *No new schema doc — reuse.* Owner: graph-dba.
- **C-202 — Skill core (`SKILL.md`).** ✅ FalkorDB connection via `redis-cli GRAPH.QUERY`; the
  `CpgNode(id)` model + per-label index reality; shared traversal idioms (callers/callees over
  `CALL`, transitive reach, data-flow over `REACHING_DEF`, symbol def/ref). Owner: graph-dba.
- **C-203 — Recipe: impact-analysis.** ✅ Callers/callees + transitive up/downstream reach —
  **FR-2, FR-3 / AC-2, AC-3**. Consumers: `analyst`, `architect`. *In-scope, no reqs change.*
- **C-204 — Recipe: rca.** ✅ Data-flow back from a symptom (`REACHING_DEF`) + cross-file symbol
  def/ref — **FR-4, FR-5 / AC-4, AC-5**. Consumer: `analyst`. *In-scope, no reqs change.*
- **C-205 — Recipe: code-review.** ✅ Taint/sink & suspicious-pattern queries (data-flow to risky
  calls) — **FR-12 / AC-7**. Consumer: `analyst`. *(Was a scope extension; approved into scope
  2026-07-18 — no longer gated.)*
- **C-206 — Recipe: test-gap.** ✅ Reachability from prod entrypoints vs. test entrypoints
  (code reachable in prod but from no test) — **FR-13 / AC-8**. Consumer: `qa-engineer`. *(Was a
  scope extension; approved into scope 2026-07-18 — no longer gated.)*
- **C-207 — Agent wiring.** ✅ Add CPG-capability lines to `analyst` / `architect` / `qa-engineer`
  descriptions (their live routing contract); note graph-dba ownership. cobb reviews. Owner: cobb.
- **C-208 — Catalog & doc sync.** ✅ `skills/README.md` (new skill), root `AGENTS.md`, `claude/README.md`
  if agent descriptions change, and this backlog → `HISTORY.md` on delivery. Per AGENTS.md, skill +
  catalog + agent wiring land in the **same** change.

### Requirements coverage

- **C-200 — Requirements pass for the two scope extensions.** ✅ **Resolved 2026-07-18 (user)** —
  code-review and test-gap folded into the requirements as **FR-12/FR-13 (AC-7/AC-8)** with
  `qa-engineer` named as a consumer; OQ1/OQ4 closed. All of M2 (C-201…C-208) now has requirements
  backing.

## Sequencing — M2

```
C-201 (schema contract) ─▶ C-202 (skill core) ─┬─▶ C-203 (impact)    ─┐
                                                ├─▶ C-204 (rca)       ─┤
                                                ├─▶ C-205 (code-review)┼─▶ C-207 (wiring) ─▶ C-208 (catalogs) ⇒ M2 ✅
                                                └─▶ C-206 (test-gap)  ─┘
Critical path: C-201 → C-202 → C-203/C-204 → C-207 → C-208 (recipes C-205/C-206 parallel after C-202).
```

## M3 — CPG query access (MCP read path)

Replaces the hand-assembled `redis-cli GRAPH.QUERY` command line on the CPG **read** path with one
MCP tool, `mcp__cpg__query(graph, cypher)`. Requirements:
[`requirements/cpg-query-access.md`](./requirements/cpg-query-access.md) (FR-1…FR-6 / AC-1…AC-4, as
amended 2026-07-25) · plan: [`plans/cpg-query-access.md`](./plans/cpg-query-access.md) v2.2
(steps S1–S10) · reviews: [`reviews/cpg-query-access.md`](./reviews/cpg-query-access.md) ·
acceptance: [`archive/test-reports/cpg-query-access-report.md`](./archive/test-reports/cpg-query-access-report.md)
(**PASS WITH DEFECTS**) · coordination log:
[`plans/cpg-query-access-coordination.md`](./plans/cpg-query-access-coordination.md).

### Decisions ruled by the stakeholder (2026-07-25) — do not re-litigate

- **D1 — destructive rebuild approved.** `cpg_falkorchat` was deleted and rebuilt from
  `falkor-chat/server/{falkorchat,tests}`; the M2 figures describe source that has since moved 8
  commits and are **not** reproducible, so the fresh numbers are the new baseline.
- **D2 — the stale AC-3 figure is corrected** (30 → 39 rows / 32 distinct names), then superseded
  in practice by the D1 rebaseline (50 rows / 43 distinct names).
- **D3 — AC-1 is demonstrated with the direct-caller question.** This feature changes *how Cypher
  is transmitted*, not how powerful Cypher is; the bounded transitive upward-closure query is
  deferred, not discarded → **C-308**.
- **D4 — `EXPLAIN`-only; `PROFILE` removed.** `GRAPH.PROFILE` *executes* the query including
  writes (reproduced live), so routing to it from a `readOnlyHint=True` tool was a read-only hole.
  `GRAPH.RO_QUERY` is what makes the main path safe; `graph-dba` keeps `PROFILE` via `redis-cli`.
- **D5 — AC-3 is narrowed to values + row counts + ordering** (Option A), excluding the display
  rendering of non-scalar cells, for which plan §4.4 is the authority. Option B (re-rendering lists
  `redis-cli`-style) rejected; **no source change**. Closes DEF-1 → **C-313**.

Settled at design time: **build, not buy** — the official `@falkordb/mcpserver` v1.3.0 exposes
**7** tools including `delete_graph` with no tool filtering (a flat FR-2 violation) and needs
Node ≥18, which is not on PATH on the Linux side. Reversal trigger: an upstream server that can be
filtered down to one read-only tool.

### Items

- **C-301 — MCP server (`cpg/mcp/`).** ✅ 2026-07-25 (commit `f2d55f7`) — `requirements*.txt`,
  idempotent `setup.sh`, `run.sh`, and `server.py`: a Python **FastMCP** stdio server exposing
  **exactly one** read-only tool `mcp__cpg__query(graph, cypher)` over `GRAPH.RO_QUERY`, with the
  `PROFILE` refusal (comment-blind), the `GRAPH.LIST` pre-check that keeps a typo'd graph name from
  materialising an empty key, display-only truncation (row/cell/char caps, notice at both ends of
  the payload) and a server-level `instructions=` string. Pytest suite: **53 offline / 7 live**.
  Steps S1–S2. Owner: devops (deps) / coder (server).
- **C-302 — Harness wiring.** ✅ 2026-07-25 — repo-root `.mcp.json` (the
  `bash -c 'exec "$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh"'` form, no absolute paths) plus
  `enabledMcpjsonServers: ["cpg"]` in `.claude/settings.json`; `claude mcp list` → `✔ Connected`,
  and the contract was proven at protocol level (1 tool, 2 required params, `readOnlyHint`).
  The repo's **first MCP wiring, and Claude-Code-only** (see C-310). Step S3. Owner: devops.
- **C-303 — Skill surface.** ✅ 2026-07-25 — `skills/cpg-analysis/SKILL.md` frontmatter
  (`allowed-tools: mcp__cpg__query, Bash, Read`) + §1 rewritten around the tool (discovery,
  `EXPLAIN`-only, truncation, `redis-cli` fallback block), the `impact-analysis` recipe preamble,
  and the `skills/README.md` catalog row. Step S4. Owner: cobb.
- **C-304 — Agent wiring & catalogs.** ✅ 2026-07-25 — `mcp__cpg__query` added to the
  `claude/analyst/analyst.md` and `claude/architect/architect.md` `tools:` allowlists (without
  which the tool is **invisible** to them; `qa-engineer` declares none and inherits),
  `claude/README.md` rows 9/16/17, root `AGENTS.md`, the three kaizen histories and the `tico`
  inbox close-out. Step S5. Owner: cobb.
- **C-305 — Requirements reconciliation.** ✅ 2026-07-25 — `joern-cpg-pipeline.md` **FR-9
  reversed** (redis-cli → `mcp__cpg__query`, `redis-cli` documented as the fallback, marked
  deliberately reversed and pointing at `cpg-query-access.md`); `cpg-query-access.md` AC-1 → direct
  callers (D3) and AC-3 → tool ≡ `redis-cli` equivalence on the fresh baseline (D1/D2). Satisfies
  **AC-4**. Step S6. Owner: coder.
- **C-306 — CPG rebuild, fresh baseline, live acceptance.** ✅ 2026-07-25 — `cpg_falkorchat` fully
  rebuilt (D1) to **110,048 nodes · 734,929 edges · 1,968 METHODs · 1,019 test-file METHODs (512
  `test_*`) · direct callers of `post_message` = 21 · test-gap = 50 rows / 43 distinct names**
  (do not collapse the last pair to one number), then acceptance-tested against it: 23 cases,
  **22 pass / 1 fail**, **AC-1 · AC-2 · AC-4 PASS, AC-3 pass-with-defect** (DEF-1 → C-313).
  Steps S8–S9. Owners: joern (rebuild) / qa-engineer (acceptance).
- **C-307 — Knowledge capture into `skills/agent-standards/`.** ✅ 2026-07-25 —
  `claude-code.md` §MCP rewritten from the verified mechanics (config scopes, `enabledMcpjsonServers`,
  deferred MCP tools + tool search, output limits) plus a **new OpenCode MCP section** recording the
  divergences (`<server>_<tool>` naming, `command` as an array, `env` → `environment`) and the
  cross-tool rule that **MCP wiring does not port** — a shared skill routing through MCP needs a
  documented non-MCP fallback or it works in exactly one harness. Step S7. Owner: cobb.

### Sequencing — M3 (as delivered)

```
S1/S2 C-301 (server + tests) ─▶ S3 C-302 (wiring, 1 human approval) ─┐
S4    C-303 (skill surface) ─────────────────────────────────────────┤
S5    C-304 (agent wiring)  ─────────────────────────────────────────┼─▶ S9 C-306 (acceptance)
S6    C-305 (requirements)  ─────────────────────────────────────────┤        │
S7    C-307 (agent-standards) ───────────────────────────────────────┤        ▼
S8    C-306 (CPG rebuild + fresh baseline) ──────────────────────────┘   S10 BACKLOG + HISTORY ⇒ M3 ✅
```

Docs (S4–S7) ran in parallel with code (S1–S3): the tool contract was frozen by the plan, so no
doc edit waited on code. S8 was independent and started early (Joern parse + load is the longest
latency).

### Status at close

**Delivered and acceptance-tested; AC-1…AC-4 all met.** **DEF-1 (C-313) was ruled the same day
(D5): AC-3 reconciled to values + counts + ordering, no code change** — the only finding that
touched an AC's wording, and **AC-3 passes** under it. DEF-2/DEF-3/DEF-5 are low-severity cleanups
(C-314/C-315/C-316) and DEF-4 is closed by S10 itself (C-317). Two residuals recorded by the
acceptance run and *not* carried as backlog items: no `mcp__cpg__query` call has been made *from
inside* an `analyst`/`architect` subagent (their allowlists are proven to resolve, which was the
actual risk), and the FalkorDB stop/restart connection-pool recovery path is untested because the
container is shared.

**Known limits of what shipped:** Claude-Code-only wiring (C-310); read-only (`GRAPH.RO_QUERY`);
`EXPLAIN`-only, no `PROFILE` (D4); display-only truncation at 200 rows / 300-char cells / 30,000
chars; cell rendering diverges from `redis-cli` for non-scalars — **by design** per plan §4.4 and
accepted by D5 (C-313 closed); the residual cleanups are C-314/C-315.

## Follow-ups (post-M3)

- **C-308 — Bounded transitive upward call-closure query.** 🔵 Give `impact-analysis.md` a single
  query for *"who calls `X`, **transitively**"* — the L1/L2/L3 shape `test-gap.md` already uses
  downward, with the `WITH`-splitting idiom and the name-collision caveat — and live-verify it. A
  naive composition returned **0 rows** on the live graph, so this is real work, not a copy-paste.
  Deferred by **D3**; until it lands, the recipe answers the transitive question by iterating.
  Owner: `graph-dba`. *(Cited as a forward reference from
  [`requirements/cpg-query-access.md`](./requirements/cpg-query-access.md) and
  [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md) — do not renumber.)*
- **C-309 — `audit-team.sh` gate was red, and blind to untracked files.** ✅ **Resolved
  2026-08-08 by `cobb`.** Two parts. **(a)** The gate previously returned `RESULT: FAIL` on
  **two pre-existing** check-7 home-path and username leaks — hit in `.claude/settings.json`
  (2 lines), `claude/devops/kaizen/inbox.md`, `claude/joern/kaizen/inbox.md`,
  `docs/plans/m2-cpg-analysis-skill.md`, plus a username in
  `falkor-chat/docs/requirements/workflow-dependence-overlay.md`. These predated M3, which is why
  every M3 step used *"no **new** failures"* (a before/after diff) as its done-condition. Already
  genericized as fallout from later, unrelated work and never reflected back into this backlog
  entry — `claude/joern/kaizen/inbox.md` doesn't even exist anymore (the joern agent was folded
  into `graph-dba`, commit `cbf26c4`). Confirmed clean by grepping all five paths directly plus a
  green `audit-team.sh` run; **no code change needed for (a)**. **(b)** Check 7 used `git grep`,
  so it saw **tracked files only** — every new untracked artifact was invisible to it until
  committed, making the differential audit a **post-commit-only** signal for new files.
  **Fixed:** the scan now unions `git ls-files --cached` with `git ls-files --others
  --exclude-standard` before grepping, so a brand-new untracked (non-ignored) file leaking an
  identifier now fails the gate too, no `git add` required first. Verified by planting an
  untracked file containing `$HOME` under `claude/`, confirming the gate FAILed on it, then
  removing it and confirming `RESULT: PASS` returned.
  Owner: `cobb` / `devops`.
- **C-310 — OpenCode + Kiro MCP wiring for the `cpg` server.** 🔵 `.mcp.json` and
  `enabledMcpjsonServers` are **Claude Code only**; OpenCode and Kiro configure MCP through their
  own files and neither is wired. `skills/cpg-analysis` is a *shared* skill, so today it reaches
  the MCP path in exactly one harness and `redis-cli GRAPH.QUERY` remains the **only** path under
  OpenCode/Kiro. Includes the `allowed-tools` portability result from S4 (an unknown entry is
  ignored, not rejected — spot-checked, not exercised by a real OpenCode invocation).
  **Updated 2026-07-26 (C-320):** the launch command is now `cpg/mcp/docker-run.sh` rather than
  `cpg/mcp/run.sh`. The property this item depends on is preserved exactly — the launch surface is
  still *a single command*, and a script ports where a JSON `args` array does not; it also replaces
  the per-host question "is there a working Python 3.12 venv there" with "is there a Docker daemon".
  Two new obligations for this item: Docker becomes a prerequisite on any harness host (`run.sh` is
  what ports to a Docker-less one), and **`MCP_TIMEOUT` is a Claude-Code knob** — OpenCode's and
  Kiro's own startup budgets must be established here. Owner: `cobb` / `devops`.
- **C-311 — `guard-destructive-ops.sh` was blind to destructive commands wrapped in scripts.** ✅
  **Resolved 2026-08-08 by `cobb`.** The guard matched the Bash *command string*, so
  `pipeline.sh --reset` deleted a graph with **no prompt** — the approval S8 originally leaned on
  could not fire (S8 was restructured to run an explicit `redis-cli GRAPH.DELETE`, which does trip
  the guard with the graph name in the text the human approves). **Fixed:** added a wrapper-match
  branch to `guard-destructive-ops.sh` (alongside the existing `docker`/`FLUSHALL`/`GRAPH.DELETE`
  branches) that catches a `pipeline.sh` invocation carrying `--reset`, in either token order, with
  a reason string naming it as a wrapped `GRAPH.DELETE`. Re-grepped `skills/*/scripts/` and
  confirmed `pipeline.sh` is still the **only** wrapper in the repo with a destructive flag, so the
  ad-hoc pattern match (rather than a general wrapper-registry mechanism) stays appropriately
  scoped, per the code comment left in place: if a second such wrapper appears, replace this
  one-off matching with a documented wrapper-registry convention rather than accreting more
  special cases. Verified with manual PreToolUse-payload tests: both `--reset` token orderings
  trigger `permissionDecision: "ask"`; `pipeline.sh` without `--reset`, an unrelated benign
  command, and all pre-existing patterns (`GRAPH.DELETE`, `FLUSHALL`, `docker rm -f`) behave
  unchanged.
  Owner: `cobb` / `devops`.
- **C-312 — `FILENAME` is relative to the Joern parse root: add a post-load verification step.** 🔵
  The parse root alone silently decides whether every `STARTS WITH 'tests/'` filter in the
  `cpg-analysis` recipes matches anything, **and the failure is invisible in node/edge counts** — a
  graph can look healthy and answer test-gap questions wrongly. This, not the missing test sources,
  is why the pre-rebuild `cpg_falkorchat` was useless. Add an explicit post-load check (assert the
  expected `FILENAME` prefixes resolve, e.g. a non-zero count of `METHOD`s under `tests/`) to
  `skills/joern-cpg/SKILL.md`. Producer-path work, out of scope for M3. Owner: `joern`.
- **C-313 — DEF-1: AC-3's "byte-identical value sets" is unmeetable as written.** ✅ **Resolved
  2026-07-25 by stakeholder ruling D5 — Option A.** AC-3 asked for byte-identical value sets
  between the tool and `redis-cli`, while plan §4.4 mandates Python `repr` for list/map cells; the
  two approved specs could not both hold for any query projecting a non-scalar. 5 of 6 equivalence
  pairs were byte-identical; the sixth (the RCA data-flow recipe, which projects `labels()`) had
  the **same 44 rows in the same order with identical values** and differed only in list syntax.
  **Option A taken:** `requirements/cpg-query-access.md` **AC-3 is narrowed to values + row counts
  + ordering**, with the display rendering of non-scalar cells excluded and plan §4.4 named as the
  authority; **AC-3 now passes**. **Option B — re-rendering lists `redis-cli`-style — was
  rejected; no source change**, the server is correct as built. A specification reconciliation, not
  a defect concession. Ruled by the stakeholder; recorded in the requirements decision log.
- **C-314 — DEF-2: map-valued cells leak the client's Python type.** 🔵 Low. A map renders as
  `OrderedDict({'a': 1, 'b': 'x'})` because `falkordb 1.6.2` returns an `OrderedDict` whose `repr`
  carries the class name — plan §4.4's *"map → `repr`"* did not anticipate that. Agent-facing noise
  and a non-round-trippable cell. No CPG recipe projects a map today, so exposure is currently nil;
  cheap to fix before one does. Owner: `coder`.
- **C-315 — DEF-3: booleans render Python-style `True`/`False`.** 🔵 Low. Plan §4.4's rendering
  table omits booleans, so they fall through to `str()` — contradicting the `SKILL.md` gotcha that
  CPG booleans are real booleans (`WHERE m.IS_EXTERNAL = false`). Verified **cosmetic**: FalkorDB
  accepts boolean literals case-insensitively, so the rendered form still round-trips.
  Owner: `coder`.
- **C-316 — DEF-5: plan §7.3's char-cap probe does not bind the char cap.** 🔵 Low (test design).
  `MATCH (m:METHOD) RETURN m.CODE` yields ~1,951 chars on this graph — the **row** cap binds first,
  so anyone following the plan literally leaves the char-cap path untested. A genuine binder:
  `MATCH (n:LITERAL) WHERE size(n.CODE) > 400 RETURN size(n.CODE) AS len, n.CODE AS code`
  (29,890 chars, 92 of 111 rows). Correct the probe in the plan. Owner: `qa-engineer` / `architect`.
- **C-317 — DEF-4: dangling `C-308` citation in the requirements.** ✅ 2026-07-25 — both
  `requirements/cpg-query-access.md` and `requirements/joern-cpg-pipeline.md` deferred the
  transitive upward-closure query to *"backlog item C-308"* before this backlog carried one.
  Closed by creating **C-308** above under M3's follow-ups; the citations now resolve. Owner: S10.
- **C-318 — Pin the server `instructions=` string in the test suite.** 🔵 `architect`'s
  recommendation. `cpg/mcp/tests/test_server.py` asserts the whole tool contract, but nothing
  asserts that `mcp.instructions` is non-empty and ≤ 2,000 chars — it is the only unguarded part of
  the contract, and it is what tool search reads (MCP tools are deferred by default), so a cold
  session depends on it. Owner: `coder`.
- **C-319 — Document `enabledMcpjsonServers` approval scoping.** 🔵 `.mcp.json` **discovery** walks
  up to the git root, but pre-approval is keyed on the **session's cwd** — a session started in a
  subdirectory (e.g. `falkor-chat/`) reports `⏸ Pending approval` where the repo root reports
  `✔ Connected`, costing one extra approval prompt per subdirectory. That is approval scoping, not
  path expansion, and the `$CLAUDE_PROJECT_DIR` form is otherwise cwd-independent. Documentation
  home: `skills/agent-standards/claude-code.md` §MCP. Already filed in
  `claude/devops/kaizen/inbox.md`. Owner: `cobb` / `devops`.
- **C-320 — Containerize the `cpg` MCP server.** ✅ **Delivered 2026-07-26.** The server runs as a
  container instead of a host venv, so a clone needs **Docker** rather than a correctly built local
  Python 3.12 venv to answer CPG queries. The tool contract is unchanged (one tool, two parameters,
  read-only, same output). Shipped: `cpg/mcp/{Dockerfile,.dockerignore,image-tag.sh,build.sh,
  docker-run.sh}` plus a two-line `.mcp.json` edit. The launch tag is a **content hash of the build
  inputs** gated by `docker image inspect`, so **on a hit** the launch path makes **no registry
  contact** and a stale image is unrepresentable. A **miss builds**, and a build does need the
  network (base-image pull + BuildKit's `FROM` metadata resolution) — see C-321. The host venv path
  is **retained** as the test loop and the fallback. Design, measurements and rejected alternatives:
  `docs/plans/cpg-mcp-containerization.md` (v3); review:
  `docs/reviews/cpg-mcp-containerization.md`. Owner: `devops`.
- **C-321 — Make the live suite's scratch-graph name unique inside a container.** 🔵
  `cpg/mcp/tests/test_server.py:472` derives the scratch graph from `os.getpid()`, which is **`1`**
  in a container's PID namespace — so every containerized `-m live` run uses the same key
  `_cpg_mcp_selftest_1` on the **shared** FalkorDB. The suite is still self-contained with respect to
  `cpg_*`/`ws:*`/`reference`, but the uniqueness that made it safe *against itself* is gone: two
  concurrent container live runs corrupt each other, and an interrupted one leaves residue on a
  shared instance. Fix: `uuid4().hex[:8]` instead of `os.getpid()`. Worked around meanwhile by
  documenting "do not run the live gate concurrently" plus a `GRAPH.LIST` residue check
  (`cpg/mcp/README.md`). Found during C-320; out of that change's scope because it is test code.
  Owner: `tdd-engineer` / `coder`.
  - **Also do this here (deferred from the C-320 review, M-8): make the autobuild not pull.**
    `cpg/mcp/docker-run.sh:84` calls `build.sh --runtime-only` on a hash miss without setting
    `CPG_MCP_NO_PULL`, so **every autobuild performs an unbounded Docker Hub `docker pull` inside
    Claude Code's 30 s MCP startup budget** (`build.sh:191-196`), followed by a build that also
    resolves `FROM` metadata over the network when the base is not in the local image store. That is
    a degradation on the exact axis the design was chosen for — the launch path being local and
    offline — and it is bounded (non-blocking MCP startup, a curated fallback message, `MCP_TIMEOUT`)
    rather than a break, which is why it was not fixed in C-320.
    **Why it belongs with this item specifically:** the content hash covers every file under
    `cpg/mcp/tests/` (plus `pytest.ini` and `requirements-dev.txt`), none of which the **runtime**
    stage COPYs — so this item's one-line edit to `tests/test_server.py` invalidates the launch image
    and forces the miss branch to rebuild a **byte-identical** runtime image. The first session start
    after it lands pays for a pull and a build unless `cpg/mcp/build.sh` is run first.
    **Cheapest known fix:** `CPG_MCP_NO_PULL=1 "$HERE/build.sh" --runtime-only …` at
    `docker-run.sh:84` (the base is already in the local store in the steady state, and the explicit
    `cpg/mcp/build.sh` stays the thing that refreshes it), plus one line in the existing curated
    build-failure message: "…or run `cpg/mcp/build.sh` once with a network connection". Rejected as
    over-reach for now: splitting the digest into runtime-inputs and test-inputs halves, which would
    cost the "same hash on both tags" property that makes a stale gate image unreachable.
    Also worth folding in while in this file, from the same review (all safe-direction, all one-line):
    `image-tag.sh`'s walk excludes `**/__pycache__` and `*.pyc` but **not** `.pytest_cache`, which
    three places claim it mirrors from `.dockerignore` (m-21); a file-mode change (m-18) and a symlink
    under a walked directory (m-19) do not move the hash; a missing walked directory is silently
    skipped where a missing file is a hard error (m-20); a failed `find` is unobserved (m-22); and
    under `--no-cache` the two `docker build` invocations can resolve different dependency versions
    (m-23). Full evidence and suggested fixes: `docs/reviews/cpg-mcp-containerization.md` §17–§18.

- **C-322 — Documentation reference & naming convention.** ✅ **Delivered 2026-07-27.** The repo had
  **two silently competing anchoring conventions** for citing a document and **no stated rule** for
  naming one. Ruled and landed: a citation is a **backticked path from the repo root** (a markdown
  link is permitted, never required, and never `/docs/…` — a leading slash is unresolvable for an
  agent); a document that freezes **stays in place** and gets `Status: archived` in a three-field
  header block (`Status:` · `Owner:` · `Tracks:`) instead of moving to `archive/`, which also
  abolishes the inbound-link repair that move required; and a new document follows the grammar
  `<component>/docs/<kind>/<topic-slug>[-<role>].md` with a closed role set and no `m<digit>`/
  `k<digit>`/date **prefix**. Applied **forward-only — zero renames, zero hook edits, no CI gate**.
  Shipped in three commits: the convention into root `AGENTS.md`, `falkor-chat/AGENTS.md`,
  `qa-engineer` and `analyst`; the header contract into the producing prompts; and the `Status:`
  backfill across 25 active documents (25 → 0 nonconforming). Design, measurements and rejected
  alternatives: `docs/plans/doc-reference-convention.md` (v1.4); review:
  `docs/reviews/doc-reference-convention.md`; entry: `docs/HISTORY.md` 2026-07-27.
  Owner: `architect` (design) / `coder` + `cobb` (execution).
- **C-323 — Bulk repath of the remaining module-anchored references to full root-anchoring (S5).**
  ⚪ **Deliberately deferred — recorded, not scheduled.** C-322 normalised the **live guidance**
  files and left the module-anchored `` `docs/…` `` citations that sit inside **dated records and
  per-item ledgers**, where the module-relative spelling is arguably correct as written;
  `falkor-chat/docs/HISTORY.md` and `falkor-chat/docs/BACKLOG.md` account for most of them. A full
  conversion is a **~60-file, judgement-heavy sweep** — each citation must be resolved against its
  citing file before it can be rewritten — and the plan's cost decomposition puts the return at
  **≤4.5% of future archival churn**, a churn D4 has *already* removed by keeping frozen documents
  in place. **Do not schedule this.** Un-defer only on a measured, repeated failure to resolve one
  of these citations. Cost analysis: `docs/plans/doc-reference-convention.md` §1.2, §2.1 and §12
  *"Not scheduled"*. Owner: unassigned.

## Follow-ups (post-M2)

- **C-101 — Fix `joern-cpg` loader `MAX_ARG_STRLEN` failure + masked exit code.** 🔵 The M1
  `cpg-to-falkordb.py --load` passes each 500-node `UNWIND` batch as a single `redis-cli` argv;
  on large `CODE` properties this exceeds the Linux 128 KiB `MAX_ARG_STRLEN` limit →
  `OSError: [Errno 7] Argument list too long`, yet `pipeline.sh` still reports **exit 0**
  (the failure is masked). Discovered 2026-07-19 during the M2 CPG substrate build; worked
  around by streaming batches via stdin (`redis-cli -x`). Fix both defects: (a) stream each
  batch via stdin instead of argv, and (b) propagate the loader's real exit code so
  `pipeline.sh` fails loudly. Owner: `joern` (producer skill). Ref: `docs/HISTORY.md` M1;
  details in the M2 coordination doc.
