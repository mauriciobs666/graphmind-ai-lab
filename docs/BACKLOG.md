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
| **M3 — CPG query access (MCP)** ✅ | The read path is a single MCP tool `mcp__cypher__query(graph, cypher)` (`cypher-mcp/`) instead of a hand-assembled `redis-cli GRAPH.QUERY` command line; wired for Claude Code, skill + agents + requirements reconciled, CPG rebuilt, AC-1…AC-4 acceptance-tested — delivered 2026-07-25. DEF-1 / **C-313** closed the same day by stakeholder ruling **D5** (AC-3 reconciled, no code change) | **C-301 → C-307** |
| **M4 — CPG agent adoption** ✅ | Six agents (`analyst`/`architect`/`qa-engineer`/`coder`/`tdd-engineer`/`frontend-engineer`) default-orient on CPG discovery, freshness is knowable via `:CpgBuildInfo`, and a spot-checked transcript shows `CPG:` evidence either way — extends, does not override, M2/M3. Implementation (C-401…C-407) complete; both gates closed — U5 `analyst` diff-gate (approve), U6 `qa-engineer` acceptance pass (FAIL, DEF-1/2/3), U7+U7-fix `cobb` wording fix, U8 `analyst` re-gate (approve w/ suggestions), U9 `qa-engineer` live re-pass (PASS, DEF-4 minor residual — see Follow-ups) | **C-401 → C-407** |
| **M5 — Generic Cypher MCP** ✅ | `mcp__cypher__query` gains write capability (an optional `agent` param, two enforced write shapes) and is piloted end to end on `graph-dba`'s kaizen working memory: the graph replaces `inbox.md` as the raw-capture layer, `history.md` is unchanged, `cobb`'s distillation workflow runs against the graph. Implementation (C-501…C-505) complete; all gates closed — U3 plan gate (`analyst`, 3 passes, needs changes → needs changes → approve), U4/U6 code re-gates (`analyst`, both approve with suggestions, fixed at U4-fix/U6-fix), U7 acceptance (`qa-engineer`, PASS, 8/8 ACs, no defects) | **C-501 → C-506** |
| **M6 — MCP tool rename** | The MCP server/tool is renamed `cpg`/`mcp__cpg__query` → `cypher`/`mcp__cypher__query`, relocated `cpg/mcp/` → `cypher-mcp/`; every active reference repo-wide updated, genuinely CPG-specific naming (`cpg-analysis`, `joern-cpg`, `cpg_<component>` graphs, top-level `cpg/`) untouched; AC-1…AC-6 acceptance-tested. | **C-601 → C-605** |

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
MCP tool, `mcp__cypher__query(graph, cypher)`. Requirements:
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

- **C-301 — MCP server (`cypher-mcp/`).** ✅ 2026-07-25 (commit `f2d55f7`) — `requirements*.txt`,
  idempotent `setup.sh`, `run.sh`, and `server.py`: a Python **FastMCP** stdio server exposing
  **exactly one** read-only tool `mcp__cypher__query(graph, cypher)` over `GRAPH.RO_QUERY`, with the
  `PROFILE` refusal (comment-blind), the `GRAPH.LIST` pre-check that keeps a typo'd graph name from
  materialising an empty key, display-only truncation (row/cell/char caps, notice at both ends of
  the payload) and a server-level `instructions=` string. Pytest suite: **53 offline / 7 live**.
  Steps S1–S2. Owner: devops (deps) / coder (server).
- **C-302 — Harness wiring.** ✅ 2026-07-25 — repo-root `.mcp.json` (the
  `bash -c 'exec "$CLAUDE_PROJECT_DIR/cypher-mcp/run.sh"'` form, no absolute paths) plus
  `enabledMcpjsonServers: ["cypher"]` in `.claude/settings.json`; `claude mcp list` → `✔ Connected`,
  and the contract was proven at protocol level (1 tool, 2 required params, `readOnlyHint`).
  The repo's **first MCP wiring, and Claude-Code-only** (see C-310). Step S3. Owner: devops.
- **C-303 — Skill surface.** ✅ 2026-07-25 — `skills/cpg-analysis/SKILL.md` frontmatter
  (`allowed-tools: mcp__cypher__query, Bash, Read`) + §1 rewritten around the tool (discovery,
  `EXPLAIN`-only, truncation, `redis-cli` fallback block), the `impact-analysis` recipe preamble,
  and the `skills/README.md` catalog row. Step S4. Owner: cobb.
- **C-304 — Agent wiring & catalogs.** ✅ 2026-07-25 — `mcp__cypher__query` added to the
  `claude/analyst/analyst.md` and `claude/architect/architect.md` `tools:` allowlists (without
  which the tool is **invisible** to them; `qa-engineer` declares none and inherits),
  `claude/README.md` rows 9/16/17, root `AGENTS.md`, the three kaizen histories and the `tico`
  inbox close-out. Step S5. Owner: cobb.
- **C-305 — Requirements reconciliation.** ✅ 2026-07-25 — `joern-cpg-pipeline.md` **FR-9
  reversed** (redis-cli → `mcp__cypher__query`, `redis-cli` documented as the fallback, marked
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
acceptance run and *not* carried as backlog items: no `mcp__cypher__query` call has been made *from
inside* an `analyst`/`architect` subagent (their allowlists are proven to resolve, which was the
actual risk), and the FalkorDB stop/restart connection-pool recovery path is untested because the
container is shared.

**Known limits of what shipped:** Claude-Code-only wiring (C-310); read-only (`GRAPH.RO_QUERY`);
`EXPLAIN`-only, no `PROFILE` (D4); display-only truncation at 200 rows / 300-char cells / 30,000
chars; cell rendering diverges from `redis-cli` for non-scalars — **by design** per plan §4.4 and
accepted by D5 (C-313 closed); the residual cleanups are C-314/C-315.

## Follow-ups (post-M3)

- **C-308 — Bounded transitive upward call-closure query.** ✅ Added Q4 to `skills/cpg-analysis/references/impact-analysis.md`:
  bounded (L1/L2/L3) `CALL.NAME`-based upward closure reusing `test-gap.md`'s `WITH`-splitting idiom
  in reverse. **Live-verified** against `cpg_falkorchat` (target `post_message`): 24 rows vs. Q1's 21
  direct callers — 1 genuine transitive addition (`test_triage_flow_runs_end_to_end_against_live_llm`,
  reached only at L2) plus 2 expected name-collision artifacts, fully explained with a filter
  recommendation. During verification a first-draft self-recursion guard was found to silently drop a
  legitimate same-named caller and was removed — fix documented inline in the recipe. Reviewed by
  `cobb`: approve, no findings. Owner: `graph-dba`.
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
- **C-310 — OpenCode + Kiro MCP wiring for the `cypher` server.** 🔵 `.mcp.json` and
  `enabledMcpjsonServers` are **Claude Code only**; OpenCode and Kiro configure MCP through their
  own files and neither is wired. `skills/cpg-analysis` is a *shared* skill, so today it reaches
  the MCP path in exactly one harness and `redis-cli GRAPH.QUERY` remains the **only** path under
  OpenCode/Kiro. Includes the `allowed-tools` portability result from S4 (an unknown entry is
  ignored, not rejected — spot-checked, not exercised by a real OpenCode invocation).
  **Updated 2026-07-26 (C-320):** the launch command is now `cypher-mcp/docker-run.sh` rather than
  `cypher-mcp/run.sh`. The property this item depends on is preserved exactly — the launch surface is
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
  unchanged. **Follow-up, same day:** `analyst`'s independent review
  (`docs/reviews/safety-net-guard-fixes.md`, verdict approve) flagged as non-blocking that the
  match was unanchored on the left — a bare substring like `mypipeline.sh --reset` also tripped
  it. Stakeholder asked for it tightened. **Fixed:** added a left token-boundary requirement on
  the `pipeline.sh` basename (start-of-string or non-alphanumeric immediately before it) —
  deliberately **not** anchored to the full `skills/joern-cpg/scripts/` path, because the
  skill's own documented usage (`scripts/pipeline.sh <source> ...`) is written cwd-relative, so a
  real invocation may legitimately appear as `scripts/pipeline.sh`, `./pipeline.sh`, or bare
  `pipeline.sh` depending on the caller's cwd — anchoring on the full path risked a false
  negative on exactly the case C-311 exists to catch. Re-verified with synthetic payloads: every
  realistic invocation shape (full repo-root path, `bash`/`sh`-prefixed, cwd-relative per
  SKILL.md, bare basename, absolute path) still asks; `mypipeline.sh --reset` (the reviewer's
  concrete finding) now passes through clean; a prose/argument mention of the real path (e.g.
  inside a `grep`/`echo`) still asks, same as the pre-existing `GRAPH.DELETE`/`FLUSHALL` branches
  already did before this change — accepted as inherent to command-text pattern matching, not a
  regression, and the safe failure direction for a destructive-ops guard. **Pass-2 correction,
  same day:** the re-review (`docs/reviews/safety-net-guard-fixes.md`, revised — verdict *needs
  changes*) caught that the tightened regex, as first written, was a single alternation
  (`pipeline\.sh.*--reset|--reset.*pipeline\.sh`) whose two boundary groups could need to consume
  the *same* separator character when only one space stood between the tokens — so `--reset
  pipeline.sh` (bare basename, flag before the name) silently stopped matching, a real regression
  against the already-approved `6ab4ffe`, and falsified this entry's own "before or after the
  path" claim. Rated major, not blocker (no realistic single command puts `--reset` textually
  before a *bare* `pipeline.sh`, since the executable has to precede its own flags — but the
  written claim was still wrong). **Fixed:** replaced the one intertwined alternation with two
  independent `grep` tests ANDed together — `pipeline.sh` present (basename-anchored) AND
  `--reset` present as its own token — so each boundary consumes its own separator regardless of
  which token comes first or how far apart they are. Re-verified through the actual script
  (`bash claude/scripts/guard-destructive-ops.sh`, not a standalone shell `grep -qiE` — this
  sandbox's bare `grep` is shadowed by `ugrep` with different ERE semantics than the GNU grep the
  script subprocess actually runs) against the full matrix: `pipeline.sh --reset`, `--reset
  pipeline.sh` (the regression case — now asks), `scripts/pipeline.sh --reset`,
  `bash .../pipeline.sh --reset`, absolute path, `sh`-prefixed, and the negative
  `mypipeline.sh --reset` (still does not ask); all pre-existing branches and the fail-open
  malformed-stdin contract re-verified unchanged.
  Owner: `cobb` / `devops`.
- **C-312 — `FILENAME` post-load verification.** ✅ `skills/joern-cpg/scripts/pipeline.sh` gained a
  repeatable `--verify-prefix PREFIX` flag, run after `--load`: asserts `MATCH (m:METHOD) WHERE
  m.FILENAME STARTS WITH PREFIX RETURN count(m)` is nonzero for each prefix, exits non-zero
  (reporting every prefix checked, not short-circuiting) with a fix-it message if any fails.
  `SKILL.md`'s Gotchas entry documents both the scripted check and a manual fallback. **Live-verified**
  against `cpg_falkorchat`: happy path (`tests/`) → 1067, failure path (`nonexistent/`) → 0. Reviewed
  by `cobb`: approve, no findings (one optional low-severity note not acted on: unescaped `"` in a
  `--verify-prefix` value could break the Cypher literal — not blocking). Owner: `graph-dba`.
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
- **C-314 — map-valued cells leaking client type.** ✅ Fixed in `cypher-mcp/server.py`: a new
  `_normalize_for_repr()` helper recursively walks dict/list/tuple values (not just top-level) before
  `repr()`, rebuilding any `Mapping` (e.g. `falkordb`'s `OrderedDict`) as a plain `dict` at every
  nesting depth. Pinned by two tests: the original flat-case test plus
  `test_render_cell_normalizes_booleans_and_maps_at_any_nesting_depth` (added after `analyst`'s
  Pass-1 review caught that the first draft only handled the top-level case). Reviewed by `analyst`:
  Pass 1 found a Major (nested values still leaked) → fixed → Pass 2 **approve**, independently
  re-verified including edge cases (apostrophe-in-key, 10-level nesting, tuples, empty containers,
  nested `set`). Owner: `coder`.
- **C-315 — booleans rendering Python-style.** ✅ Fixed in the same `_normalize_for_repr()` pass as
  C-314 (same commit-worthy diff, same review cycle) — booleans anywhere in a nested structure now
  render lowercase `true`/`false` via a `_ReprAsIs` repr-substitution sentinel, not just at the top
  level. Same review history as C-314 (Pass 1 Major → fixed → Pass 2 approve). Owner: `coder`.
- **C-316 — DEF-5: plan §7.3's char-cap probe does not bind the char cap.** ✅ **Resolved
  2026-08-09 — corrected probe recorded here, plan left unedited.** `docs/plans/cpg-query-access.md`
  is `Status: archived`; per the repo's doc-reference convention (root `AGENTS.md`), an archived
  document takes no substantive edits, only a header-pointer edit — so §7.3 is **intentionally not
  corrected in place**. This entry is the authoritative record of the fix going forward: the
  original probe, `MATCH (m:METHOD) RETURN m.CODE`, yields ~1,951 chars on the reference graph — the
  **row** cap (200 rows) binds before the char cap (30,000 chars) does, so anyone following the plan
  literally leaves the char-cap path untested. The genuine binder is
  `MATCH (n:LITERAL) WHERE size(n.CODE) > 400 RETURN size(n.CODE) AS len, n.CODE AS code`
  (29,890 chars, 92 of 111 rows on the graph this defect was originally filed against). **Live
  re-verification 2026-08-09, `mcp__cypher__query` against `cpg_falkorchat`** (`falkordb-dev` was
  briefly down mid-session; `teco` restarted it and confirmed `cpg_falkorchat` survived in the
  persisted volume): the probe returns `rows=120` in 9.6ms, and the tool truncates its own reply
  with `"showing 92 of 120 rows (char cap 30000)"` — the **char cap**, not the 200-row cap, is what
  cuts the response off, which is exactly what this defect required demonstrating. Individual `len`
  values observed range from 405 up to 4,314 chars (e.g. a 4,314-char `LITERAL` in the transition-guard
  evaluation module), comfortably past the 300-char single-cell truncation threshold, confirming the
  probe binds a meaningful char cap. The row/char counts differ slightly from the original filing
  (120 rows here vs. 111 then, reflecting normal drift in `cpg_falkorchat`'s content since); the
  binding **shape** — char cap trips before row cap, on real oversized `LITERAL.CODE` values — is
  confirmed live, not merely reviewed statically. Owner: `qa-engineer`.
- **C-317 — DEF-4: dangling `C-308` citation in the requirements.** ✅ 2026-07-25 — both
  `requirements/cpg-query-access.md` and `requirements/joern-cpg-pipeline.md` deferred the
  transitive upward-closure query to *"backlog item C-308"* before this backlog carried one.
  Closed by creating **C-308** above under M3's follow-ups; the citations now resolve. Owner: S10.
- **C-318 — pin the server `instructions=` string.** ✅ `cypher-mcp/tests/test_server.py` gained
  `test_server_instructions_are_present_and_bounded`, asserting `mcp.instructions` is non-empty and
  ≤2000 chars (currently 408 chars). Verified as a real pin (not a tautology) by transiently
  blanking the string and observing the assertion fail before restoring. Reviewed by `analyst`:
  approve. Owner: `coder`.
- **C-319 — document `enabledMcpjsonServers` approval scoping.** ✅ Added to
  `skills/agent-standards/claude-code.md` §MCP → "Scopes, precedence, and the approval gate":
  `.mcp.json` discovery walks up to the git root (cwd-independent), but project-scope approval is
  keyed to the session's cwd, costing one extra interactive approval per subdirectory a session
  starts in — stated as a fact parallel to (not caused by) `${CLAUDE_PROJECT_DIR}` path expansion,
  which is a separate, unrelated cwd-independent mechanism (server-launch env var, not `.mcp.json`
  discovery). Sourced from and distilled out of `claude/devops/kaizen/inbox.md`'s 2026-07-25 entry.
  Reviewed by `cobb` (self-review, applying real scrutiny per the round's brief): caught and fixed a
  Major finding — the first draft asserted an unsupported causal link between the two facts;
  corrected to state them as parallel, verified against the official MCP docs. Owner: `cobb`.
- **C-320 — Containerize the `cypher` MCP server.** ✅ **Delivered 2026-07-26.** The server runs as a
  container instead of a host venv, so a clone needs **Docker** rather than a correctly built local
  Python 3.12 venv to answer CPG queries. The tool contract is unchanged (one tool, two parameters,
  read-only, same output). Shipped: `cypher-mcp/{Dockerfile,.dockerignore,image-tag.sh,build.sh,
  docker-run.sh}` plus a two-line `.mcp.json` edit. The launch tag is a **content hash of the build
  inputs** gated by `docker image inspect`, so **on a hit** the launch path makes **no registry
  contact** and a stale image is unrepresentable. A **miss builds**, and a build does need the
  network (base-image pull + BuildKit's `FROM` metadata resolution) — see C-321. The host venv path
  is **retained** as the test loop and the fallback. Design, measurements and rejected alternatives:
  `docs/plans/cpg-mcp-containerization.md` (v3); review:
  `docs/reviews/cpg-mcp-containerization.md`. Owner: `devops`.
- **C-321 — both halves now done, close the whole item.** ✅ **Core** (the item's main body):
  `cypher-mcp/tests/test_server.py`'s live-suite scratch-graph key now derives from `uuid4().hex[:8]`
  instead of `os.getpid()` (extracted into a `_scratch_graph_name()` helper), fixing the PID-1
  collision risk when the server runs containerized. TDD red/green: pinned the bug with
  `test_scratch_graph_name_is_unique_across_calls` (red under the old `os.getpid()` behavior, green
  after), then sanity-checked live (7/7 passed, no residue). **"Also do this here" sub-item**
  (deferred from the C-320 review, M-8): `docker-run.sh`'s autobuild path now sets `CYPHER_MCP_NO_PULL=1`
  on its `build.sh --runtime-only` call (plus a one-line addition to the build-failure message
  pointing at a manual `cypher-mcp/build.sh` run) — smoke-tested end to end, confirmed no `docker pull`
  step, ~5s cold build, well under the 30s MCP startup budget. `image-tag.sh`'s hash-walk now
  excludes `.pytest_cache` (matching `.dockerignore`) and hard-fails (rather than silently
  skipping/succeeding) on a missing walked directory, a symlink under a walked directory, or a failed
  `find` — judged by captured `find` stderr content rather than exit code, to correctly distinguish a
  real failure from an unrelated SIGPIPE under this script's `set -euo pipefail`. A file-mode change
  is deliberately left out of the hash, with an inline comment explaining why (inert today; revisit if
  the image's `CMD` ever execs a COPYed file directly). `build.sh` was reordered so the `test` image
  builds first (respecting `--no-cache` if given) and `runtime` builds second always without
  `--no-cache`, so `runtime` deterministically reuses the dependency layer `test` just populated —
  closing a version-drift risk between the two images under `--no-cache`. Verified as a no-op
  reordering when `--no-cache` isn't requested at all. Reviewed by `analyst`: approve (Pass 2
  confirmed U4/U5 diffs untouched by the C-314/315 re-fix cycle). One non-blocking Informational
  note from the review: the `.pytest_cache` exclusion is currently a no-op given the real directory
  layout (pytest's cache lands at `cypher-mcp/.pytest_cache`, never under a walked `tests/` subtree) —
  correct and defensive, not a defect, no action needed. Owner: `coder` / `devops`.

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

## M4 — CPG agent adoption

Widens which agents discover and use a loaded CPG (roster: `coder`, `tdd-engineer`,
`frontend-engineer` added to `analyst`/`architect`/`qa-engineer`), makes discovery a default
orientation step instead of a conditional one, and lets a consulting agent judge/flag graph
staleness via graph-dba's new `:CpgBuildInfo` freshness marker. Requirements:
[`requirements/cpg-agent-adoption.md`](./requirements/cpg-agent-adoption.md) (FR-1…FR-9 /
AC-1…AC-6) · plans: [`plans/cpg-agent-adoption-graph.md`](./plans/cpg-agent-adoption-graph.md)
(freshness mechanics, graph-dba) + [`plans/cpg-agent-adoption.md`](./plans/cpg-agent-adoption.md)
(roster/discovery/evidence-trail, cobb) · coordination:
[`plans/cpg-agent-adoption-coordination.md`](./plans/cpg-agent-adoption-coordination.md).

**Extends, does not override, M2/M3.** See `plans/cpg-agent-adoption.md` §4 — the MCP read path
and the skill's four recipes are unchanged; only the consumer list and the default-ness of
discovery widen.

### Items

- **C-401 — Freshness marker mechanics.** ✅ `:CpgBuildInfo` singleton node (BUILT_AT/SOURCE_PATH/
  SOURCE_COMMIT/SOURCE_DIRTY), stamped at the end of `skills/joern-cpg/scripts/pipeline.sh`'s
  `--load` branch after verification passes; new `skills/cpg-analysis/references/freshness.md`
  recipe; one nav-table row in `SKILL.md`. FR-5/FR-6 (mechanical)/FR-7/FR-8. Owner: `graph-dba`
  (unit U4a).
- **C-402 — `cpg-analysis` SKILL.md: broaden + discovery mechanic.** ✅ Frontmatter `description`
  widened to six consumers; `cpg_<component>` naming-convention paragraph (the one-query,
  no-noise-on-a-miss discovery mechanic) added to §1; Navigation-table Consumer column (§4)
  updated — the impact-analysis row now lists `analyst, architect, coder, tdd-engineer,
  frontend-engineer` (rca/code-review/test-gap rows left as-is, no stated reason to widen them).
  FR-1 (skill-side), **FR-4 / AC-5**. Owner: `cobb` (unit U4b-1).
- **C-403 — Wire `analyst`/`architect`/`qa-engineer`: default-orientation reword.** ✅ Description +
  orientation-step + evidence-trail-line edits on the three already-wired agents, including the
  freshness-check bundling (run the freshness recipe as part of the same default step, note it,
  surface a refresh suggestion if stale). FR-2, FR-6 (surfacing integration) / AC-1, **AC-3, AC-4**.
  Owner: `cobb` (unit U4b-2).
- **C-404 — Wire `coder`/`tdd-engineer` as new consumers.** ✅ Description clause + orientation
  sentence (including the freshness-check bundling) + evidence-trail line. FR-1, FR-2, FR-3 /
  AC-1, **AC-3, AC-4**. Owner: `cobb` (unit U4b-3).
- **C-405 — Wire `frontend-engineer` as a new consumer.** ✅ Same shape as C-404, grounded in
  `cpg_salesperson`/`chatbot.py`. FR-1, FR-2, FR-3 / AC-1, **AC-3, AC-4**. Owner: `cobb`
  (unit U4b-4).
- **C-406 — Evidence-trail convention.** ✅ The `CPG: used | considered, not relevant | not
  applicable` one-line convention landed in all six wired agents' deliverable skeletons, including
  the freshness signal the deliverable reports when the CPG was actually consulted. **AC-2**, and
  reinforces **AC-3** (the freshness signal surfaces in the deliverable, not just inside the
  agent's reasoning). Owner: `cobb` (folded into units U4b-2…U4b-4).
- **C-407 — Catalog & doc sync.** ✅ `claude/README.md` (all six agent rows reworded to
  default-orientation framing; `coder`/`tdd-engineer`/`frontend-engineer` rows gained their first
  CPG mention), `skills/README.md` (`cpg-analysis` row widened from three to six named consumers).
  Root `AGENTS.md`'s `skills/` bullet checked — already consumer-agnostic ("the consumer side",
  no named-consumer list), so left unchanged. This backlog → `HISTORY.md`. Owner: `cobb`
  (unit U4b-5).

### Requirements coverage

FR-1…FR-8 and AC-1…AC-5 each carry an explicit tag on at least one of C-401…C-406 above (see each
item's FR/AC line — C-401 for FR-5/FR-7/FR-8 and FR-6's mechanical half, C-402 for FR-1/FR-4/AC-5,
C-403 for FR-2/FR-6's surfacing half/AC-1/AC-3/AC-4, C-404/C-405 for FR-1/FR-2/FR-3/AC-1/AC-3/AC-4,
C-406 for AC-2/AC-3. **FR-9 and AC-6 are the one deliberate exception**, carried by no backlog
item because they aren't implementation work — AC-6's "states explicitly that it extends" is a
property of `plans/cpg-agent-adoption.md` §4 itself (already written, already satisfied), not a
task C-407 or any other item performs. `C-407` (catalog/doc sync) is untagged for the same reason
each catalog-sync item was in M2/M3's own backlog sections — it is process bookkeeping, not a
requirement-bearing deliverable.

**Gate status (per the coordination ledger — full unit-by-unit detail there, not restated here).**
C-401…C-407 are implementation-complete. **U5** (`analyst` diff-scoped re-gate, distinct from the
design-level U3 plan-gate): **approve**. **U6** (`qa-engineer` acceptance pass against AC-1…AC-6,
three live subagent dispatches): **FAIL** — AC-1/AC-6 held, but AC-2/AC-3/AC-4 broke a different
way across all three dispatches (DEF-1 `coder`, DEF-2 `architect`, DEF-3 `tdd-engineer`). **U7 +
U7-fix** (`cobb`): two wording-tightening rounds on the same six agent files, closing DEF-1/DEF-2/
DEF-3; diff-gated by `analyst` at **U8** — **approve with suggestions** (two minors + a nit,
folded into U7-fix). **U9** (`qa-engineer` live-dispatch re-pass, different target
functions/components than the original dispatches): **PASS** — DEF-1 and DEF-2 confirmed closed
with directly opposite behavior, DEF-3's silence failure closed, but a new minor, **DEF-4**
(`CPG:` shape-selection nit), surfaced and was accepted as a low-severity residual rather than
triggering a fourth fix-and-regate round (see Follow-ups below, C-408). Milestone **closed** — see
[`plans/cpg-agent-adoption-coordination.md`](./plans/cpg-agent-adoption-coordination.md) for the
full U1…U9 ledger.

## Follow-ups (post-M4)

- **C-408 — `CPG:` shape-selection ambiguity (DEF-4).** 🔵 The three-shape `CPG:` convention
  (`docs/plans/cpg-agent-adoption.md` §3) gives one worked example each for `used` and
  `considered, not relevant`, but none for `not applicable` — and U9's live re-pass found a real
  dispatch (`tdd-engineer`, D3′) pick `not applicable` for a code-level task on a component with
  no loaded CPG, where the plan's own definition and its `considered, not relevant` worked
  example both point the other way (`docs/test-reports/cpg-agent-adoption-report.md` Pass 2,
  DEF-4). Severity minor — AC-2's anti-silence guarantee is intact; only a shape-specific
  spot-check (e.g. `grep "considered, not relevant"`) would miss it. Fix direction (per the
  report's own feedback #2): either add a worked counter-example distinguishing `not applicable`
  from `considered, not relevant`, or explicitly accept this as a low-severity, rare edge case not
  worth further prompt surface. Owner: `cobb` (next time this doc's wiring is touched).
- **C-409 — No live dispatch had observed a populated `:CpgBuildInfo` marker.** ⚪ **Narrowed,
  not fully closed** — `graph-dba` rebuilt `cpg_falkorchat` on request; `qa-engineer`'s targeted
  follow-up (`docs/test-plans/cpg-agent-adoption2.md`, `docs/test-reports/
  cpg-agent-adoption2-report.md`, 2026-08-17) dispatched `coder` against it and confirmed, live:
  the freshness marker query now returns a real populated row (not zero rows); the agent correctly
  falls back on `SOURCE_COMMIT`/`SOURCE_DIRTY` being absent (this graph's known `.git`-less
  scratch-copy build pattern, `docs/plans/cpg-agent-adoption-graph.md` §6) without erroring or
  misreading the absence; and it correctly avoids a false-positive stale claim on a genuinely
  fresh marker (the mirror of AC-4's positive branch) — PASS, 4/4, zero defects. That closes two
  of the three edges this item named: "no marker at all" (covered since Pass 1/Pass 2) and "fresh,
  populated marker" (covered now). **What remains open, and why it's deferred rather than
  re-triggered:** a *genuinely stale, populated* marker actually producing a concrete refresh
  suggestion has still never been observed live — `cpg_falkorchat`'s marker was minutes old at
  this follow-up's dispatch time, and there's no organic source drift to observe that branch
  against without fabricating a stale timestamp/commit, which the follow-up correctly declined to
  do. This edge is inherently time-dependent (needs either real elapsed time + independent commits
  on a rebuilt graph, or a future rebuild that happens to land already behind current source) —
  not something to chase with another proactive rebuild ping. Deferred as an accepted residual
  risk; re-open only if a future dispatch happens to hit this condition organically, or if a
  stakeholder decides the branch is worth deliberately engineering a real (not fabricated)
  drift scenario for. Owner: `qa-engineer` (this pass, closed); no active trigger owner while
  deferred.

## M5 — Generic Cypher MCP

`mcp__cypher__query` gains write capability — an optional `agent` parameter and two enforced write
shapes (author-write, curator-clear) — and is piloted end to end on `graph-dba`'s kaizen working
memory: the graph (`kaizen_graph_dba`) replaces `inbox.md` as the raw-capture layer, `history.md`
stays unchanged, and `cobb`'s distillation workflow runs against the graph. Requirements:
[`requirements/generic-cypher-mcp.md`](./requirements/generic-cypher-mcp.md) (FR-1…FR-11 /
AC-1…AC-8) · plans: [`plans/generic-cypher-mcp-graph.md`](./plans/generic-cypher-mcp-graph.md)
(data model, `graph-dba`) + [`plans/generic-cypher-mcp.md`](./plans/generic-cypher-mcp.md) (tool
mechanism, `architect`) · coordination:
[`plans/generic-cypher-mcp-coordination.md`](./plans/generic-cypher-mcp-coordination.md).

### Items

- **C-501 — MCP server write path.** ✅ `cypher-mcp/server.py` gains the optional `agent` parameter
  and the write-detection/authorization branch (author-write vs. curator-clear);
  `cypher-mcp/README.md` updated. Step 1. Owner: `coder`.
- **C-502 — Requirements pointer.** ✅ `docs/requirements/cpg-query-access.md`'s header gains a
  `**Note:**` pointing at this feature's AC-8 supersession of its "Non-CPG graphs / general agent
  access to FalkorDB" out-of-scope line. Step 2. Owner: `coder`.
- **C-503 — Migration + inbox freeze.** ✅ One-time import of `claude/graph-dba/kaizen/inbox.md`'s
  six entries into `kaizen_graph_dba`, `entryId` index + uniqueness constraint, frozen-inbox note
  prepended to `inbox.md`. Step 3. Owner: `graph-dba`.
- **C-504 — Repo-wide catalog/backlog docs.** ✅ `claude/AGENTS.md`, `claude/README.md`, and this
  backlog's M5 section describe `graph-dba`'s actual post-migration behavior — no remaining
  unconditional claim that every agent appends to `inbox.md`. Step 4a. Owner: `cobb`.
- **C-505 — Agents' operative-prompt + distillation-workflow docs.** ✅
  `claude/graph-dba/graph-dba.md`'s Learning capture section and `claude/cobb/cobb.md`'s
  distillation-duties bullet updated to match; `skills/agent-maintenance/SKILL.md` §5 gains the
  graph-read-then-distill sequence for `graph-dba`, including the append-before-delete ordering
  constraint (documented there only). Step 4b. Owner: `cobb`.
- **C-506 — Acceptance pass.** ✅ AC-1…AC-8 each exercised live; delivers
  `docs/test-plans/generic-cypher-mcp.md` + `docs/test-reports/generic-cypher-mcp-report.md`.
  Step 5. Owner: `qa-engineer`.

**Gate status (per the coordination ledger — full unit-by-unit detail there, not restated here).**
C-501…C-505 are implementation-complete. **U3 plan gate** (`analyst`, on `graph-dba`'s U1
data-model plan and `architect`'s U2 tool-mechanism plan): three passes — **needs changes** (Pass
1: blocker B1 on the close-out file list + majors M1/M2/M3 on the write-detection/authorization
regex), **needs changes** again (Pass 2, `U3-regate`, after `U2-fix` closed Pass 1's findings: a
new major, M1-residual, on the CREATE-clause-location step not being string-literal-aware),
**approve** (Pass 3, `U3-regate2`, after `U2-fix2` closed M1-residual) — plan gate closed. **U4**
(`coder`, steps 1+2 — server write path + tests), code re-gate (`analyst`) → **approve with
suggestions**, fixed at **U4-fix** (spot-checked and accepted by `teco`). **U5** (`graph-dba`, step
3 — live migration to `kaizen_graph_dba`) — no formal gate, independently verified by `teco`
directly against the running graph. **U6** (`cobb`, steps 4a+4b — doc sync), code re-gate
(`analyst`) → **approve with suggestions**, fixed at **U6-fix** (spot-checked and accepted by
`teco`). **U7** (`qa-engineer`, step 5 — acceptance pass): **PASS** — all 8 acceptance criteria
(AC-1…AC-8) hold under live exercise, no defects found. Milestone **closed** — see
[`plans/generic-cypher-mcp-coordination.md`](./plans/generic-cypher-mcp-coordination.md) for the
full U1…U7 ledger.

## Follow-ups (post-M5)

- **C-507 — AC-5's append-before-delete ordering is enforced procedurally, not mechanically.** 🔵
  `cobb`'s 4-step distillation sequence (append to `history.md`/knowledge base, confirm, only then
  curator-clear) is a documented discipline, not a tool-enforced invariant — `mcp__cypher__query` has
  no way to require or check the ordering of two independent write calls
  (`docs/plans/generic-cypher-mcp.md` §9 names this explicitly as procedural, not mechanical,
  enforcement). U7's acceptance pass could confirm only end-state consistency, not the raw sequence
  of API-call timestamps (`docs/test-reports/generic-cypher-mcp-report.md`, "AC-5 detail" section
  and Feedback & recommendations #1). U7's one real dispatch behaved correctly, but a single
  successful run is weaker long-run assurance than a mechanically-enforced invariant would be. No
  action needed for this delivery (the trade-off was already named and accepted at plan-gate time),
  but if this pattern extends to a second curator agent or a higher-volume distillation cadence,
  consider a tool-side "last write timestamp" queryable independently of the dispatched agent's own
  narration, rather than relying on end-state consistency plus self-report. Owner: `architect`
  (next time this tool's write path is revisited).

## M6 — MCP tool rename

- **C-601 — Relocate + rebuild `cypher-mcp/`.** 🔵 `git mv cpg/mcp cypher-mcp`; every identity
  string inside the moved files renamed (server name, tool name, Docker image/label, env-var
  prefix, internal shell-function names, log-line prefix); relative links shortened; offline
  suite and in-container gates green from the new path. Step 1. Owner: `coder`.
- **C-602 — Harness + agent-tool-surface wiring.** 🔵 `.mcp.json`, `.claude/settings.json`, both
  `analyst`/`architect` `tools:` lines, and `skills/cpg-analysis/SKILL.md`'s `allowed-tools:` +
  body mentions updated to `mcp__cypher__query`; `claude mcp list` shows `cypher` connected, no
  `cpg` entry. Step 2. Owner: `coder`.
- **C-603 — `claude/` + `skills/` sweep.** 🔵 Every active agent prompt, kaizen log, and skill
  body under `claude/`/`skills/` with a live hit renamed; CPG-domain vocabulary (`cpg-analysis`,
  `joern-cpg`, `cpg_<component>` graphs) left untouched. Step 3a. Owner: `cobb`.
- **C-604 — `docs/` + `mcp-monitor/` + `falkor-chat/` sweep.** 🔵 Every active
  `docs/{plans,requirements,reviews,test-plans,test-reports}/*.md` with a live hit,
  `docs/manuals/cpg-getting-started.md`, `docs/BACKLOG.md`/`docs/HISTORY.md` (surgical),
  `mcp-monitor/{AGENTS,README}.md` + `mcp-monitor/docs/{BACKLOG,HISTORY}.md` (surgical), and
  `falkor-chat/compose.yaml` renamed; includes the `docs/requirements/generic-cypher-mcp2.md`
  `(M6)`→`(M7)` header bump. Step 3b. Owner: `cobb`.
- **C-605 — Acceptance pass.** 🔵 AC-1…AC-6 exercised live; zero unexplained `cpg`-identity hits
  survive the widened sweep outside archived/family/domain-vocabulary categories. Step 4. Owner:
  `qa-engineer`.

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
