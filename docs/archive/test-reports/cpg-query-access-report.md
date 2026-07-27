# CPG query access — test report

> **Run 1 · 2026-07-25 · executed by `qa-engineer` · step S9** of
> [`../../plans/cpg-query-access.md`](../../plans/cpg-query-access.md).
> Plan executed: [`../test-plans/cpg-query-access.md`](../test-plans/cpg-query-access.md) v1.0.
> Requirements: [`../../requirements/cpg-query-access.md`](../../requirements/cpg-query-access.md)
> (AC-1…AC-4 as amended 2026-07-25, decisions D1–D3).

---

## 1. Verdict

## **PASS WITH DEFECTS**

The feature does what it was built to do. An agent asks the CPG a question in **one tool call**,
passing the graph key and multi-line Cypher as parameters, **with no shell layer anywhere** — and
gets the right answer. Every declared safety semantic held live: read-only enforcement, the
`PROFILE` refusal (including the comment-prefixed bypass), the empty-key guarantee on a typo'd
graph name, curated errors, and display-only truncation with its notice intact at both ends of the
payload. The shared FalkorDB instance ended the run byte-for-byte as it started.

The defects are **rendering-fidelity and documentation-hygiene issues, not functional failures**.
The one that matters is **DEF-1**: AC-3's literal demand for *byte-identical value sets* between
the tool and `redis-cli` **cannot be met as written**, because plan §4.4 deliberately specifies a
*different* (and mostly better) rendering for non-scalar cells. Five of the six equivalence pairs
are byte-identical; the sixth — the RCA data-flow recipe, which projects `labels()` — differs
purely in list syntax while carrying identical values in identical order. This is a **conflict
between two approved specs**, and it needs an owner's ruling rather than a code change.

| AC | Result | One-line basis |
|---|---|---|
| **AC-1** | ✅ **PASS** | Direct callers of `post_message` = **21**, returned in a single `mcp__cpg__query(graph, cypher)` call, zero shell quoting/escaping. Tool surface verified at protocol level: **1 tool, exactly 2 parameters**. |
| **AC-2** | ✅ **PASS** | Multi-line and single-line forms returned byte-identical row bodies. A query carrying `"`, `'`, `$HOME`, `$(whoami)` and a leading `//` comment passed through verbatim and unescaped. |
| **AC-3** | ⚠️ **PASS WITH DEFECT** | **5 of 6** equivalence pairs byte-identical after the stated normalisation (incl. all of impact-analysis, test-gap and code-review). RCA data-flow diverges on list-valued cells only — same 44 rows, same order, same values (**DEF-1**). Fresh baseline re-measured and **matches S8 exactly**. |
| **AC-4** | ✅ **PASS** (one hygiene defect) | FR-9 now routes through `mcp__cpg__query` and points at `cpg-query-access.md`; no document asserts the reversed mechanism. Dangling `C-308` reference filed as **DEF-4** (owner S10). |

**No defect blocks S10.** DEF-1 needs a ruling; the rest are cleanups.

---

## 2. Environment as executed

| Item | Value |
|---|---|
| Commit | `f2d55f7`; working tree carried the uncommitted S3 follow-ups |
| FalkorDB | `falkordb-dev`, `falkordb/falkordb:v4.18.11`, `redis_version 8.6.3`, `127.0.0.1:6379`, no auth |
| Graphs (entry **and** exit) | `cpg_falkorchat`, `cpg_salesperson`, `reference`, `ws:acme`, `ws:test` — **5, unchanged** |
| Server venv | Python 3.12.3, `mcp 1.28.1`, `FalkorDB 1.6.2`, `redis 8.0.1`, `pytest 9.1.1` |
| MCP status | `claude mcp list` → `cpg: bash -c exec "$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh" - ✔ Connected` |
| Caps in force | defaults: rows 200 · cell 300 · chars 30 000 · timeout 30 000 ms |

**Concurrency note.** Mid-run, an `architect` session bumped `docs/plans/cpg-query-access.md`
to **v2.2** (audit-trail correction recording re-gate finding **n-4** as implemented) and appended
to `claude/architect/kaizen/inbox.md`. This closes teco's "n-4 still open" carry-forward. It is
an audit correction with **no design change**, so it does not invalidate this run, which was
planned against v2.1. Corroborated independently here: the server's `instructions=` string is
present in this session's own context as an **"MCP Server Instructions"** block for `## cpg`.

### 2.1 Recorded baseline — CONFIRMED, supersedes the M2 figures

Re-measured through `mcp__cpg__query` in this run. **Every figure matches S8 exactly**, so the
rebuilt graph has not drifted since 2026-07-25.

| ID | Measure | S8 | **This run** | |
|---|---|---|---|---|
| B1 | nodes | 110,048 | **110,048** | ✅ |
| B1 | edges | 734,929 | **734,929** | ✅ |
| B1b | `METHOD` nodes | 1,968 | **1,968** | ✅ |
| B2 | `METHOD`s under `tests/` | 1,019 | **1,019** | ✅ |
| B2b | of those, named `test_*` | 512 | **512** | ✅ |
| B3 | direct callers of `post_message` | 21 | **21** (2 prod + 19 test) | ✅ |
| B4 | test-gap | 50 rows / 43 names | **50 rows / 43 distinct names** | ✅ |

> These figures **supersede** the M2 baseline (79,581 / 522,182 / 336 / 21 / 39 / 32) per **D1/D2**.
> The M2 numbers describe source that has since moved 8 commits and are **not reproducible**;
> they are not a target and must not be iterated toward.
> **Do not collapse B4 to one number** — 50 is the method-*site* count, 43 the distinct-*name* count.

**Semantic sanity (recipe anchors, indicative not required):** all four hold on the rebuilt graph —
`ping`, `_safe_respond`, `_safe_run_workflow` flagged as gaps; `_serialize_opaque` correctly **not**
flagged (it is transitively test-reached). The closure is doing real transitive work, not
one-hop matching.

---

## 3. Results by case

23 of 23 executed. **22 pass · 1 fail · 0 blocked.**

| ID | Case | Result | Evidence |
|---|---|---|---|
| **TP-001** | Direct callers in one tool call | ✅ PASS | `graph=cpg_falkorchat · rows=21 · 62.3ms`; 21 rows = `api.py:…post_message` L144, `mcp.py:…send_message` L53, + 19 `tests/…`. One tool call, no `Bash`, **no quoting or escaping anywhere**. A `ToolSearch` event preceded it (MCP tools are deferred) — **permitted and expected** per the amended criterion. |
| **TP-002** | FR-2 tool surface | ✅ PASS | Raw JSON-RPC `tools/list`: `serverInfo {'name':'cpg','version':'1.28.1'}`, **TOOL COUNT 1**, `properties ['graph','cypher']`, `required ['graph','cypher']`, `annotations {'readOnlyHint': True}`, `_meta {'anthropic/maxResultSizeChars': 60000}`, **`outputSchema: None`** (no double payload). |
| **TP-003** | analyst/architect reach the tool | ✅ PASS¹ | `analyst.md:5` and `architect.md:5` both end `…, Agent, mcp__cpg__query`; the harness's own resolved agent listing renders both with `mcp__cpg__query` in their tool set. `qa-engineer` declares no `tools:` and inherits (this run is the proof). |
| **TP-004** | Multi-line ≡ single-line | ✅ PASS | Both forms → `rows=21`, **row bodies byte-identical**; only the stats timing differs (62.3 ms vs 3.5 ms). |
| **TP-005** | Quote-hostile literals | ✅ PASS | Query carrying `"`, `\"`, `$HOME`, `$(whoami)`, `'quoted'` and a leading `//` comment → `rows=1`, correct answer. Nothing escaped by the caller; the `//` comment did **not** misclassify as a directive. On the `redis-cli` path the same string needs shell defence. |
| **TP-006** | B3 callers: T ≡ R | ✅ PASS | 21 vs 21 rows, headers equal, **ordered lists identical**. |
| **TP-007** | Test-gap closure: T ≡ R | ✅ PASS | 50 vs 50 rows, headers equal, **ordered lists identical**. |
| **TP-008** | Test-gap distinct-name count: T ≡ R | ✅ PASS | Both `43`. Combined probe: `gapRows 50 | distinctNames 43`. |
| **TP-009** | Resolved callees: T ≡ R | ✅ PASS | 3 vs 3 rows identical. |
| **TP-010** | RCA data-flow slice: T ≡ R | ❌ **FAIL** | 44 vs 44 rows, same order, same `line`/`code` values — but `labels()` cells differ: tool `['CpgNode', 'IDENTIFIER']` vs redis-cli `[CpgNode, IDENTIFIER]`. → **DEF-1** |
| **TP-011** | Taint pattern A: T ≡ R | ✅ PASS | 9 vs 9 rows identical (the documented over-reporting `taintedParam` set). |
| **TP-012** | Fresh baseline | ✅ PASS | `110048 | 734929 | 1968 | 1019 | 512` — exact S8 match (§2.1). |
| **TP-013** | AC-4 documents agree | ✅ PASS | FR-9 rewritten (`joern-cpg-pipeline.md:76-82`) routing through `mcp__cpg__query`, `redis-cli` as fallback, explicitly marked *"deliberately reversed on 2026-07-25"* and pointing at `cpg-query-access.md`. `skills/README.md:19`, root `AGENTS.md:21,82-83` and `SKILL.md` all state the same mechanism + fallback + Claude-Code-only scoping. `grep "30 untested"` → none. The two surviving "chosen over MCP tool" hits are **historical citations** (a problem-statement quote and a supersession entry), correctly framed. `tico/kaizen/inbox.md:22-24` carries cobb's close-out. → hygiene issue **DEF-4** |
| **TP-014** | `EXPLAIN` = plan only | ✅ PASS | `graph=cpg_falkorchat · EXPLAIN (plan only — nothing was executed)` + the operator tree. **No result rows.** |
| **TP-015** | `PROFILE` refused | ✅ PASS | Exact refusal text, pointing at `redis-cli … GRAPH.PROFILE`. No results. |
| **TP-016** | Comment-prefixed `PROFILE` refused | ✅ PASS | `/* sneaky comment */ PROFILE …` **and** `// sneaky line comment⏎PROFILE …` → **the same refusal**. Load-bearing, proven: raw `redis-cli GRAPH.RO_QUERY` with the identical string returns **`count(m) 1968` — results**. The sniff converts a silent wrong answer into an honest refusal. |
| **TP-017** | Typo'd graph, plain | ✅ PASS | `Graph 'cpg_falkorchat_typo_qa_s9' does not exist. Loaded graphs: … If no CPG is loaded, building and loading one is the joern agent's job …`. `GRAPH.LIST` **unchanged at 5**; `EXISTS` → `0`. |
| **TP-018** | Typo'd graph, `EXPLAIN` | ✅ PASS | Identical curated error via the pre-check; `GRAPH.LIST` **still 5**, no key materialised. |
| **TP-019** | Write rejected | ✅ PASS | `This tool is read-only (GRAPH.RO_QUERY). …`. Post-check `MATCH (n:QaS9WriteProbe) RETURN count(n)` → **0**. |
| **TP-020** | Syntax error | ✅ PASS | FalkorDB's message verbatim (`errMsg: Invalid input 'R': expected a label … line: 1, column: 17 … errCtx:`) + the schema pointer line. Server survived. |
| **TP-021** | Row cap | ✅ PASS | `showing 200 of 1968 rows (row cap)` as **first and last line, byte-identical**; `rows=1968` = true total; "unordered" clause present; 200 rows rendered; delivered inline. |
| **TP-022** | Char cap | ✅ PASS | `showing 92 of 111 rows (char cap 30000)`, first **and** last line; payload 29,890 chars (≤ 30,000); whole rows dropped from the tail; 94 cell cuts `…(+N chars)`; embedded newlines escaped so one row = one line. **Arrived in the conversation — not replaced by a file reference.** R-d closed. |
| **TP-023** | Component suites | ✅ PASS | Run twice, before and after: **53 passed / 7 deselected** offline; **7 passed / 53 deselected** live. |

¹ **Residual on TP-003** — see §6.

### 3.1 How AC-3 equivalence was normalised (stated explicitly, per the criterion)

The two paths render differently by design, so comparison is on **value sets**, not display bytes.
Harness: `cpg/mcp/.venv/bin/python <scratchpad>/equiv.py`.

- **Tool side** — drop the stats line (`graph=… · rows=N · …ms`), drop the header line, drop any
  line beginning `… truncated:` (display metadata, plan §4.4); split remaining rows on ` | `.
- **`redis-cli` side** — `--no-raw` emits one scalar per line: N header names, then rows flattened
  N cells at a time, then exactly 2 trailing lines (`Cached execution:` /
  `Query internal execution time:`). Drop the header and the 2 stats lines; regroup into N-tuples.
- **Compare** — as an **ordered list** where the query has `ORDER BY`, else as a **multiset**.
  Column headers compared separately and matched in all six cases.

No query exceeded the 200-row cap, so no comparison ran on a truncated sample.

---

## 4. Defects

### DEF-1 — Non-scalar cells are not byte-identical between the tool and `redis-cli`; AC-3 as written is unmeetable · **Medium**

**This is a conflict between two approved specifications, not an implementation bug.**
AC-3 requires *"byte-identical value sets"* against `redis-cli`. Plan §4.4 requires `None → null`
and *"list/map → `repr`"*. Python `repr` and FalkorDB's own display syntax differ, so the two specs
cannot both be satisfied for any query projecting a non-scalar. The server implements §4.4 faithfully.

**Reproduce**

```bash
# the RCA recipe query, both paths — 44 rows each, same order, same values
cpg/mcp/.venv/bin/python <scratchpad>/equiv.py     # → "TP-010 … VALUE SETS EQUAL: False"
```

**Expected vs actual** — expected byte-identical cells; actual:

| Value type | Tool | `redis-cli` | Equal |
|---|---|---|---|
| integer, float, string, empty string, `list<int>` | `42` / `1.5` / `abc` / `` / `[1, 2]` | same | ✅ |
| **`null`** | `null` | *(empty)* | ❌ — **tool is better** (plan §4.4 intends this: "ran and matched nothing" stays distinguishable) |
| **boolean** | `True` / `False` | `true` / `false` | ❌ (**DEF-3**) |
| **`list<string>` / `labels()`** | `['CpgNode', 'METHOD']` | `[CpgNode, METHOD]` | ❌ — **tool is arguably better** (its form is valid Cypher; `redis-cli`'s unquoted form is not) |
| **nested list** | `[['a', 1], None]` | `[[a, 1], NULL]` | ❌ (`None` is not valid Cypher) |
| **map** | `OrderedDict({'a': 1, 'b': 'x'})` | `{a: 1, b: x}` | ❌ (**DEF-2**) |
| **node** | `(:CpgNode:METHOD{AST_PARENT_FULL_NAME:"<empty>",…` | `id/labels/properties` breakdown | ❌ — spec'd (`str(Node)`), both lossy in different ways |

**Impact.** Low for analysis correctness — values, row counts and ordering are identical, and no
conclusion from any of the four recipes changes. It matters for (a) AC-3's literal sign-off,
(b) anyone building an automated tool-vs-fallback diff, and (c) an agent copying a rendered cell
back into Cypher.

**Recommendation (owner ruling needed — `architect` + `tico`, not fixable by QA).** Either:
1. **Narrow AC-3** to *"identical values, row counts and ordering; cell *rendering* may differ by
   type per plan §4.4"* — the honest description of what shipped, and the option this run
   recommends; or
2. **Change the renderer** to emit Cypher literal syntax for non-scalars (`true`/`false`, `null`
   inside lists, `{a: 1}` for maps). Cheap, and it makes rendered cells round-trip into Cypher.

Option 1 plus the DEF-2/DEF-3 cleanups is the smaller, better-value change.

---

### DEF-2 — Map-valued cells leak the client's Python type · **Low**

A map renders as **`OrderedDict({'a': 1, 'b': 'x'})`**, not `{'a': 1, 'b': 'x'}`.
Plan §4.4 says "map → `repr`" and did not anticipate that `falkordb 1.6.2` returns an
`OrderedDict`, whose `repr` carries the class name.

**Reproduce:** `mcp__cpg__query(graph="cpg_falkorchat", cypher="RETURN {a:1, b:'x'} AS v")`
**Expected:** a map literal. **Actual:** `OrderedDict({'a': 1, 'b': 'x'})`.
**Impact:** agent-facing noise and a non-round-trippable cell. No CPG recipe projects a map today,
so exposure is currently nil — cheap to fix before one does.

---

### DEF-3 — Booleans render Python-style `True`/`False` · **Low**

Plan §4.4's rendering table covers `None`, `str`, `Node`/`Edge`, list/map — **not booleans**, which
fall through to `str()`. `SKILL.md` gotcha #2 stresses that CPG booleans are real booleans
(`WHERE m.IS_EXTERNAL = false`), so the rendering contradicts the guidance an agent just read.

**Reproduce:** `RETURN m.NAME, m.IS_EXTERNAL` → `post_message | False`.
**Impact: cosmetic — verified, not assumed.** I checked whether this breaks a round-trip and it does
**not**: FalkorDB accepts boolean literals case-insensitively —
`… AND m.IS_EXTERNAL = False` and `… = false` both return `2`. Consistency fix only.

---

### DEF-4 — `C-308` is cited by the requirements but does not exist in `docs/BACKLOG.md` · **Low**

`docs/requirements/cpg-query-access.md:72` and `:116` both defer the transitive upward-closure
query to *"backlog item **C-308**, owner `graph-dba`"*. `docs/BACKLOG.md` exists but ends at the
M2 items — **C-301…C-311 are absent**, because **S10 has not run**.

This confirms teco's **nn-2** carry-forward as a live dangling reference. It is a forward-reference
gap, not a contradiction, so **AC-4 still passes**. **Owner: S10 (`coder`).** Verify after S10 that
C-308 (and C-309/C-310/C-311) exist under milestone M3 and are not renumbered.

---

### DEF-5 — Plan §7.3's char-cap probe does not bind the char cap · **Low (test-design)**

Plan §7.3 suggests `MATCH (m:METHOD) RETURN m.CODE` to exercise the char cap. On this graph
`METHOD.CODE` holds short signatures, so the payload is **1,951 chars** and the **row** cap binds
first — the char-cap path would go untested by anyone following the plan literally.

A genuine binder on this graph (used here): `MATCH (n:LITERAL) WHERE size(n.CODE) > 400 RETURN
size(n.CODE) AS len, n.CODE AS code` → 29,890 chars, `char cap 30000`, 92 of 111 rows.
Widest `CODE` by label: `LITERAL` 4,314 · `BLOCK` 2,715 · `CALL` 2,552 · `METHOD` short.
**Recommendation:** correct the probe in plan §7.3 so the next run exercises the path.

---

## 5. Coverage

**Covered.** All four ACs; FR-2/FR-3/FR-4 at protocol level; all four `cpg-analysis` recipes
executed through **both** paths; the full §7.3 semantics table except the three rows listed below;
both component suites; shared-state integrity (graph inventory + no key materialisation + no write).

**Not covered — deliberate, with residual risk.**

| Gap | Why | Residual risk |
|---|---|---|
| **Live `mcp__cpg__query` call from an `analyst` / `architect` subagent** (plan S9 done-condition m-4) | I ran as a subagent and did not spawn further agents. TP-003 proves the allowlist **resolves** — the harness renders `mcp__cpg__query` in both agents' tool sets, which is exactly the R-3 failure mode — but no call was *made* by them. | **Low.** The allowlist was the risk; it is closed. Recommend teco close it with one throwaway `analyst` query. |
| FalkorDB stop/restart → connection-pool recovery | Requires stopping the **shared** `falkordb-dev` container (`falkor-chat` + `salesperson` depend on it). Environment-mutating; a subagent cannot ask for approval mid-run. | **Medium.** Untested recovery path. Worth a supervised run. |
| Server killed mid-session (`/mcp` reconnect) | Interactive human-only action; plan labels it a known limitation. | Low. |
| 30 s query-timeout path (`*1..12`) | Long-running load on a shared instance for a unit-testable branch. | Low. |
| OpenCode / Kiro | Deferred — **C-310**. | Known and documented. |

---

## 6. Feedback & recommendations

1. **Rule on DEF-1 before S10 closes.** It is the only finding that touches an AC's wording.
   Recommend narrowing AC-3 to *values + row counts + ordering*, and recording the per-type
   rendering divergence in `cpg/mcp/README.md` next to the existing "ignore `… truncated:` lines"
   note — a future reader building a diff will hit this immediately.
2. **The `PROFILE` sniff earns its complexity.** Live proof: `/* c */ PROFILE …` through raw
   `GRAPH.RO_QUERY` returns **1968 results**. Without the comment-blind sniff an agent asking for a
   profile receives results and cannot tell. Keep the D5a normalisation, and keep its unit tests.
3. **The truncation double-notice works.** Both caps bound live and the notice survived as the
   first *and* last line; a ~30 k payload arrived inline, not as a file reference. The `_meta`
   +lowered-default combination (N-2) is doing its job — don't let a future change raise
   `CPG_MCP_MAX_CHARS` without re-running TP-022.
4. **Fix the plan's char-cap probe (DEF-5)** so the path is actually exercised next time.
5. **Testability, genuinely good:** `server.run_query()` is importable and side-effect-free at
   import, which made the whole equivalence harness possible without protocol plumbing. The
   `pytest.ini` choice to **deselect** `live` rather than skip-on-reachability is the right call —
   a running database cannot silently change what the default command covers. Keep both.
6. **Baseline fragility (unchanged, worth repeating).** The S8 baseline lives only in the
   `falkordb-data` Docker volume. It survived this session and matched exactly, but *"is FalkorDB
   up, and does `MATCH (m:METHOD) RETURN count(m)` still say 1968?"* remains the correct first
   check before any future CPG acceptance work.

---

## 7. Artifacts & state

- Test plan: `docs/archive/test-plans/cpg-query-access.md` (new, v1.0)
- This report: `docs/archive/test-reports/cpg-query-access-report.md` (new)
- Harnesses (scratchpad, not committed): `equiv.py` (AC-3 equivalence), `celltypes.py` (DEF-1…3 characterisation)

**No source, plan, requirement or skill file was modified by this run.** No graph was created,
written to or deleted. `GRAPH.LIST` at exit = the same five keys as at entry. Nothing was committed.

---

## Addendum — 2026-07-25 · DEF-1 ruled (added after the run)

> Appended after the fact. **Sections 1–7 above are the dated execution record and stand exactly as
> written** — they were accurate when the run happened. Nothing above has been re-scored.

The stakeholder ruled on **DEF-1** (backlog **C-313**) the same day, taking **Option A**:

- **`docs/requirements/cpg-query-access.md` AC-3 is narrowed** to *values + row counts + ordering*,
  explicitly excluding the display rendering of non-scalar (list/map) cells, for which
  [`../../plans/cpg-query-access.md`](../../plans/cpg-query-access.md) **§4.4** is the authority. The
  amendment and the ruling are recorded in that document's AC-3 and decision log.
- **Option B — changing the server to render lists `redis-cli`-style — was rejected. No source
  changed**; the server is correct as built.
- This is a **specification reconciliation, not a product fix**. **TP-010's ❌ was a
  criterion-wording artifact, not a defect in the tool**: it returned **44 rows vs 44, in the same
  order, with the same `line`/`code` values**, differing solely in how a `labels()` list is printed
  (`['CpgNode', 'IDENTIFIER']` vs `[CpgNode, IDENTIFIER]`).
- Under the reconciled wording **AC-3 passes**, so **AC-1…AC-4 are all met** and **C-313 is
  closed**. DEF-2/DEF-3/DEF-5 (C-314/C-315/C-316) are unaffected and remain open cleanups.
