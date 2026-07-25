# CPG query access — test plan

> **Version 1.0 · 2026-07-25 · author `qa-engineer` · step S9 of
> [`../plans/cpg-query-access.md`](../plans/cpg-query-access.md) (v2.1).**
> Requirements under test: [`../requirements/cpg-query-access.md`](../requirements/cpg-query-access.md)
> (AC-1…AC-4 **as amended 2026-07-25**, decisions D1–D3).
> Coordination: [`../plans/cpg-query-access-coordination.md`](../plans/cpg-query-access-coordination.md).
> Results: [`../test-reports/cpg-query-access-report.md`](../test-reports/cpg-query-access-report.md).

---

## 1. Scope & objective

Verify, by **executing the delivered system**, that the `cpg` MCP server
(`cpg/mcp/`, tool `mcp__cpg__query`) satisfies **AC-1…AC-4** of the CPG-query-access
requirements, and that the server's declared read-path semantics hold live against the
rebuilt `cpg_falkorchat` graph.

**In scope**

- The MCP read path end-to-end: `mcp__cpg__query(graph, cypher)` as an agent actually calls it.
- Equivalence of that path with the documented `redis-cli GRAPH.QUERY` fallback (AC-3).
- The server's frozen contract semantics — read-only, `EXPLAIN`-only / `PROFILE`-refused,
  the empty-key guarantee, curated errors, display-only truncation (plan §4.4, §7.3).
- Documentation coherence between the two requirements documents and the consumer skill (AC-4).
- The component's own regression signal (`cpg/mcp/tests`, offline + `-m live`).

**Out of scope (and why)**

| Not tested | Reason |
|---|---|
| Unit-level coverage of `server.py` internals | Owned by S2's pytest suite; this plan **runs** that suite rather than re-litigating it. |
| Building / loading a CPG (the `joern` producer path) | Out of the feature's scope; the graph is a fixed precondition (S8). |
| FalkorDB stop/restart resilience (plan §7.3 row 10) | Requires stopping the **shared** `falkordb-dev` container, which `falkor-chat` and `salesperson` also use. Environment-mutating, needs stakeholder approval, and a subagent cannot ask mid-run. **Deliberately omitted — residual risk recorded in the report.** |
| Server-killed-mid-session recovery (`/mcp` reconnect) | Interactive, human-only action; the plan itself labels it a known limitation, not a defect. |
| 30 s query-timeout path (deep `*1..12` traversal) | Long-running and load-generating on a shared instance for a low-risk, unit-testable branch. Omitted; residual risk recorded. |
| OpenCode / Kiro MCP wiring | Explicitly deferred to backlog **C-310**. |
| Auth / grants / read-only *enforcement at the server* | Explicitly out of requirements scope. |
| Transitive upward call closure | Deferred to **C-308** by decision **D3**. |

**Read-only discipline.** No graph is created, written to, deleted or rebuilt.
`cpg_falkorchat` is the acceptance baseline. The one write-shaped case (TP-019) is a
*negative* test whose expected outcome is server-side rejection; it is aimed at a
throwaway graph name that must **not** come into existence, and `GRAPH.LIST` is
re-asserted afterwards.

---

## 2. Risk assessment (drives priority)

| # | Risk | Likelihood | Impact | Mitigating cases |
|---|---|---|---|---|
| R-a | **The tool renders results differently from `redis-cli`**, so an agent silently reasons on a different value set than the fallback path produces. This is the risk AC-3 exists to catch. | Medium — the renderer is new, hand-written, and lossy by design (cell caps, `\n` escaping, unescaped pipes) | High | TP-006…TP-011 |
| R-b | **`PROFILE` falls through and returns results.** FalkorDB silently ignores the prefix, so a naive sniff hands back *results where a profile was asked for* — a wrong answer, not an error. Comment-prefixed spellings are the live-verified bypass. | Medium (sniff is bespoke) | High | TP-015, TP-016 |
| R-c | **A typo'd graph name materialises an empty key**, polluting a shared instance. FalkorDB's `GRAPH.QUERY` does this natively; only the `RO_QUERY`/pre-check design prevents it. | Medium | High (shared DB) | TP-017, TP-018 |
| R-d | **Truncation notice is lost** to Claude Code's own output limits (>25 k tokens → persisted to disk, replaced by a file reference), leaving an agent reading a silently truncated table. | Medium | High (silent wrong conclusions) | TP-021, TP-022 |
| R-e | **The tool is invisible to `analyst` / `architect`** because their `tools:` frontmatter is an allowlist (plan R-3, m-4). Ships a feature that works for one of three consumers. | Low (S5 delivered) | High | TP-003 |
| R-f | **Multi-line / quote-hostile Cypher is mangled** in transit, defeating the feature's whole purpose (FR-3). | Low | High | TP-004, TP-005 |
| R-g | **Read-only guarantee is not real** — a write reaches the graph. | Low | Critical | TP-019 |
| R-h | **Docs disagree** about the access mechanism; a reader follows the reversed FR-9. | Low (S6 delivered) | Medium | TP-013 |
| R-i | **Baseline drift** — the graph is no longer the S8 build, so every count in this run is measured against a different substrate. The baseline lives only in a Docker volume. | Medium | Medium | TP-012 |
| R-j | Component regression signal broken (no root test runner ⇒ silent rot). | Low | Medium | TP-023 |
| R-k | Curated errors degrade to tracebacks / crash the stdio server (not auto-reconnected). | Low | Medium | TP-020 |

Prioritisation: **P1** = an AC cannot be signed off without it, or the risk is High/Critical.
**P2** = contract semantics with a real wrong-answer or shared-state mode. **P3** = quality signal.

---

## 3. Environment

| Item | Value |
|---|---|
| Repo / commit | `graphmind-ai-lab` @ `f2d55f7` (feature commit) — working tree carries the uncommitted S3 follow-ups (`.mcp.json` untracked, `.claude/settings.json`, `cpg/mcp/README.md`, coordination doc modified) |
| FalkorDB | container `falkordb-dev`, image `falkordb/falkordb:v4.18.11`, `redis_version 8.6.3`, `127.0.0.1:6379`, no auth |
| Graphs loaded (must stay exactly these five) | `cpg_falkorchat`, `cpg_salesperson`, `ws:acme`, `ws:test`, `reference` |
| Server venv | `cpg/mcp/.venv` — Python 3.12.3, `mcp 1.28.1`, `FalkorDB 1.6.2`, `redis 8.0.1`, `pytest 9.1.1` |
| MCP wiring | repo-root `.mcp.json`, project scope, `bash -c 'exec "$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh"'`, `timeout 60000`; `.claude/settings.json` → `enabledMcpjsonServers: ["cpg"]` |
| Tool caps in force | `CPG_MCP_MAX_ROWS=200`, `CPG_MCP_MAX_CELL=300`, `CPG_MCP_MAX_CHARS=30000`, `CPG_MCP_TIMEOUT_MS=30000` (defaults; `.mcp.json` sets only host/port) |
| Fallback path | `redis-cli -p 6379 GRAPH.QUERY <graph> '<cypher>' --no-raw` |
| Query source | `skills/cpg-analysis/SKILL.md` §3 idioms and `skills/cpg-analysis/references/{impact-analysis,rca,code-review,test-gap}.md` |

### 3.1 Recorded baseline carried into this run (S8, `joern`, 2026-07-25)

These are the figures S9 re-confirms; per **D1/D2** they **supersede** the M2 figures
(79,581 / 522,182 / 336 / 21 / 39 / 32), which describe source that has since moved 8 commits.

| ID | Measure | S8 value |
|---|---|---|
| B1 | nodes / edges in `cpg_falkorchat` | **110,048 / 734,929** |
| B1b | `METHOD` nodes | **1,968** |
| B2 | `METHOD`s with `FILENAME STARTS WITH 'tests/'` (must be > 0) | **1,019** (of which **512** named `test_*`) |
| B3 | direct callers of `post_message` | **21** |
| B4 | test-gap closure | **50 rows / 43 distinct names** (do **not** collapse to one number) |

### 3.2 Entry criteria

1. FalkorDB reachable; `GRAPH.LIST` returns exactly the five graphs above.
2. `claude mcp list` shows `cpg … ✔ Connected` in the workspace running the tests.
3. `mcp__cpg__query` is callable in this session (deferred-tool schema loaded).
4. Both pytest suites green **before** any acceptance case runs (TP-023 first).

### 3.3 Exit criteria

- Every P1 case executed with recorded evidence; AC-1…AC-4 each carry an explicit verdict.
- No case leaves the graph inventory changed (`GRAPH.LIST` = the same five keys, end of run).
- Fresh baseline recorded in the report; any deviation from §3.1 explained, not silently absorbed.
- Every failure filed as a reproducible defect with severity — **no defect is fixed by this step**;
  fixes route back to `coder` / `devops`.

---

## 4. Test items

Notation: **T** = the query went through `mcp__cpg__query`; **R** = through `redis-cli`.

### AC-1 — one tool call, no shell escaping, direct callers

| ID | Title | Pri | Type |
|---|---|---|---|
| **TP-001** | Direct callers of `post_message` answered in **one** `mcp__cpg__query` call | P1 | acceptance |
| **TP-002** | Tool surface is exactly one tool with exactly two required parameters (`graph`, `cypher`) | P1 | contract |
| **TP-003** | `mcp__cpg__query` is reachable by `analyst` and `architect` (the two `tools:`-allowlisted consumers) | P1 | integration |

**TP-001** — *Preconditions:* entry criteria met. *Steps:* issue the impact-analysis **Q1**
recipe query (`MATCH (caller:METHOD)-[:CONTAINS]->(c:CALL {NAME:'post_message'}) RETURN DISTINCT
caller.FULL_NAME, caller.FILENAME, caller.LINE_NUMBER ORDER BY …`) as the `cypher` parameter with
`graph = "cpg_falkorchat"`, in a single tool call. *Expected:* a rendered table of **21** distinct
callers; the transcript shows one `mcp__cpg__query` call, **no `Bash`** event, and **no shell
quoting or escaping anywhere**.
*Permitted transcript events (amended per cobb's live doc verification):* `ToolSearch` (tool search
is **on by default** and MCP tools are **deferred**, so a cold session *must* resolve the schema
first — its presence is a **good** run), `Skill`, and `Read`s of `SKILL.md` / the recipe.
A `ToolSearch` event must **not** fail this criterion.

**TP-002** — *Steps:* inspect the resolved tool schema as the harness presents it.
*Expected:* one tool named `mcp__cpg__query`; `required: [graph, cypher]`; no third parameter;
no `params`, `readOnly` or `limit` knob (FR-2).

**TP-003** — *Steps:* read the harness-resolved tool allowlist for the `analyst` and `architect`
agent types, and the `tools:` frontmatter of both agent definitions.
*Expected:* `mcp__cpg__query` present in both. (m-4 / plan R-3.)

### AC-2 — multi-line Cypher accepted verbatim

| ID | Title | Pri | Type |
|---|---|---|---|
| **TP-004** | Multi-line form of a query returns a result body byte-identical to its single-line form | P1 | acceptance |
| **TP-005** | Cypher containing `'`, `"` and `$` inside string literals passes through unescaped | P2 | acceptance |

**TP-004** — *Steps:* send the TP-001 query as one line; send it again with newlines and
indentation between clauses. *Expected:* the two payloads are identical apart from the timing
field on the stats line; the row sets are byte-identical.

**TP-005** — *Steps:* send a query whose string literal contains a double quote, a `$` and an
apostrophe-escaped form. *Expected:* the query runs and returns the correct answer, with nothing
escaped by the caller. Contrast: state what the same string would require on the `redis-cli` path.

### AC-3 — tool ≡ `redis-cli` equivalence (+ fresh baseline)

Restated by **D1/D2**: AC-3 is an **equivalence proof**, not a number-matching exercise.
The two paths format output differently, so comparison is on **normalised value sets**, and the
normalisation is stated explicitly in the report.

| ID | Title | Pri | Type |
|---|---|---|---|
| **TP-006** | B3 direct callers (impact-analysis Q1): T ≡ R | P1 | contract |
| **TP-007** | B4 test-gap L1/L2/L3 closure: T ≡ R (50 rows) | P1 | contract |
| **TP-008** | B4 variant `count(DISTINCT g.NAME)`: T ≡ R (43) | P1 | contract |
| **TP-009** | Impact-analysis Q2 callees (resolved `CALL` edge): T ≡ R | P2 | contract |
| **TP-010** | RCA data-flow slice (`REACHING_DEF*1..12` backward from a `RETURN`): T ≡ R — the multi-column / `labels()`-bearing shape | P2 | contract |
| **TP-011** | Code-review taint Pattern A (`Repository.post_first_message`): T ≡ R | P2 | contract |
| **TP-012** | B1/B1b/B2 fresh baseline re-measured and reconciled against §3.1 | P1 | acceptance |

**Normalisation rule for TP-006…TP-011.** Both payloads are reduced to a set of rows, each row a
tuple of cell values, by: (a) dropping the tool's stats/header lines and any line beginning
`… truncated:` (display metadata, per plan §4.4); (b) dropping `redis-cli`'s `--no-raw` decoration
(numbered result lines, column-name header, the trailing `Cached execution` / `Query internal
execution time` lines); (c) splitting the tool's rows on ` | ` and `redis-cli`'s on its cell
separator; (d) comparing as **multisets, order-insensitive** where the query has no `ORDER BY`,
**order-sensitive** where it does. Any difference is a defect in the tool's rendering.
Row cap is 200 — a query exceeding it must be narrowed (projection/aggregate) so both paths
return the full set, not silently compared on a truncated sample.

### AC-4 — documents agree on the access mechanism

| ID | Title | Pri | Type |
|---|---|---|---|
| **TP-013** | `joern-cpg-pipeline.md` FR-9 points at `cpg-query-access.md`; no reader can find the two documents (or `SKILL.md`) disagreeing | P1 | documentation |

*Steps:* `grep -n "redis-cli\|MCP\|GRAPH.QUERY"` across `docs/requirements/joern-cpg-pipeline.md`,
`docs/requirements/cpg-query-access.md`, `skills/cpg-analysis/SKILL.md`,
`skills/cpg-analysis/references/*.md`, `skills/README.md`, root `AGENTS.md`; read every hit.
Also confirm `claude/tico/kaizen/inbox.md` no longer records the FR-9-vs-MCP contradiction as open,
and that the **C-308** forward reference cited by the requirements resolves (or is flagged).
*Expected:* one mechanism (MCP tool), one documented fallback (`redis-cli`), scoped honestly to
Claude Code; no residual sentence asserting `redis-cli` was *chosen over* an MCP tool.

### Regression / declared-semantics spot-checks

| ID | Title | Pri | Type | Expected |
|---|---|---|---|---|
| **TP-014** | `EXPLAIN MATCH …` on a real graph | P2 | contract | Plan text only — **no result rows** |
| **TP-015** | `PROFILE MATCH …` | P1 | contract | Refusal message naming the `redis-cli GRAPH.PROFILE` fallback; **no results**, no server call |
| **TP-016** | `/* c */ PROFILE …` and `// c⏎PROFILE …` | P1 | contract | The **same** refusal (comment-blind sniff, D5a). Raw `GRAPH.RO_QUERY` is live-verified to *accept* these and return rows, so results here = wrong answer |
| **TP-017** | Typo'd graph name, plain query | P1 | contract | Curated "does not exist", **lists the loaded graphs**, routes to `joern`; `GRAPH.LIST` **unchanged at five** afterwards |
| **TP-018** | Typo'd graph name with `EXPLAIN ` prefix | P1 | contract | Same error via the pre-check; `GRAPH.LIST` **still unchanged at five** |
| **TP-019** | `CREATE (:X)` write query | P1 | contract | Read-only rejection message; nothing written; graph inventory unchanged |
| **TP-020** | Syntax error | P2 | contract | FalkorDB's line/column/context message verbatim + the schema pointer; server survives |
| **TP-021** | Row cap binding (`MATCH (m:METHOD) RETURN m.FULL_NAME`) | P2 | contract | Exactly 200 rows shown, `rows=` reports the **true** total, truncation notice present as **both first and last line**, "unordered" clause present |
| **TP-022** | Char cap binding (`MATCH (m:METHOD) RETURN m.CODE`) | P2 | contract | Payload ≤ ~30 000 chars, notice names the **char cap**, arrives **in the conversation** with the notice intact — *not* replaced by a file reference. A file reference here is a defect; do **not** raise the cap |
| **TP-023** | `cpg/mcp/tests` offline + `-m live` | P3 | regression | Both green; pass counts recorded |

---

## 5. Data & fixtures

No fixtures are created. All queries are read-only `MATCH … RETURN` against the existing
`cpg_falkorchat`, taken verbatim (with the one documented literal substitution) from the
`cpg-analysis` recipes. Named probe values used by the negative cases:

- Typo'd graph name: `cpg_falkorchat_typo_qa_s9` — must **never** appear in `GRAPH.LIST`.
- Write probe (TP-019): aimed at `cpg_falkorchat`; expected outcome is server-side rejection,
  so no throwaway graph is needed.

`GRAPH.LIST` is snapshotted at entry and re-asserted after TP-017, TP-018, TP-019 and at exit.

---

## 6. Traceability

| AC / concern | Cases |
|---|---|
| AC-1 (one call, no escaping, direct callers) | TP-001, TP-002, TP-003 |
| AC-2 (multi-line verbatim) | TP-004, TP-005 |
| AC-3 (T ≡ R equivalence + fresh baseline) | TP-006, TP-007, TP-008, TP-009, TP-010, TP-011, TP-012 |
| AC-4 (documents agree) | TP-013 |
| FR-2 (one tool, two parameters) | TP-002 |
| FR-3 (verbatim, multi-line) | TP-004, TP-005 |
| FR-4 (graph caller-supplied) | TP-002, TP-017 |
| Plan §4.4 D4a (read-only + empty-key) | TP-017, TP-018, TP-019 |
| Plan §4.4 D5 / D5a (`EXPLAIN` only, comment-blind sniff) | TP-014, TP-015, TP-016 |
| Plan §4.4 truncation / N-2 | TP-021, TP-022 |
| Plan §4.4 error table | TP-017, TP-019, TP-020 |
| Component regression signal (m-6) | TP-023 |

---

## 7. Reporting

Results, evidence, the recorded fresh baseline, defects (severity by user impact) and the overall
verdict go to [`../test-reports/cpg-query-access-report.md`](../test-reports/cpg-query-access-report.md),
referencing these IDs. This step **finds and documents** defects; it does not fix them.
