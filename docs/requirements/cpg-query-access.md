# CPG query access — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** C-301…C-307 (M3) ·
> **Delivered ✅** — AC-1…AC-4 met and accepted (M3, 2026-07-25); follow-ups tracked in
> [`../BACKLOG.md`](../BACKLOG.md) · **Last updated:** 2026-07-25
>
> **Note:** the "Non-CPG graphs / general agent access to FalkorDB" and "Authentication, per-user
> grants, and read-only enforcement" lines below are widened by
> [`generic-cypher-mcp.md`](./generic-cypher-mcp.md) FR-1 — read that document for the current
> scope; this archived document's body is left exactly as originally written.

## Intent
Agents that read a loaded Joern CPG in FalkorDB should be able to ask the graph a
question with low friction. Today every query is hand-assembled as a `redis-cli
GRAPH.QUERY` command line, and the stakeholder wants that ceremony to stop getting in
the way of the analysis itself.

## Problem & current state
`redis-cli GRAPH.QUERY` is the only access path in the repo. For CPG work it is
prescribed by `skills/cpg-analysis/SKILL.md` §1 and was a recorded decision —
`docs/requirements/joern-cpg-pipeline.md` **FR-9**, *"chosen over MCP tool / raw
Cypher."* Three costs surfaced in use:

1. **Quoting/escaping pain** — Cypher lives inside a shell argument; quotes, `$`
   substitutions and multi-line queries have to be defended against the shell.
2. **Connection rediscovery** — host, port and the graph key are re-derived by each
   agent in each session (the skill deliberately refuses to hardcode the graph name).
3. **Process overhead** — one `redis-cli` process per query, one shell round-trip per
   question, in workflows that ask many small questions.

Known related scar (not itself in scope): the M1 loader passed a 500-node batch as a
single `redis-cli` argv, hit the Linux 128 KiB `MAX_ARG_STRLEN`, and failed while
`pipeline.sh` still reported exit 0 (M2 coordination log, 2026-07-19).

## Scope
**In:** the CPG read path — the agents querying an already-loaded CPG (`analyst`,
`architect`, `qa-engineer`, and the `cpg-analysis` skill they use).

## Out of scope
- `falkor-chat` and `salesperson` component scripts (`bootstrap_schema.sh`,
  `seed_demo.sh`, `test_queries.sh`, …) — they stay on `redis-cli`.
- Non-CPG graphs / general agent access to FalkorDB.
- **Authentication, per-user grants, and read-only enforcement.** Considered and
  explicitly deferred: FalkorDB stays open on `:6379` with no auth, as it is today.
- The `joern-cpg` **load** path and its `MAX_ARG_STRLEN` bug — tracked separately.

## User stories
- As an **analyst/architect/qa-engineer**, I want to run a Cypher query against the
  loaded CPG without shell-escaping it, so that I spend my effort on the traversal, not
  the quoting.
- As an **agent starting a fresh session**, I want the CPG connection and graph name to
  be available without re-deriving them, so that I reach the first answer sooner.

## Functional requirements
- **FR-1** — Agents query a loaded CPG through an **MCP tool**, not by assembling a
  `redis-cli` command line.
- **FR-2** — The MCP surface is **a single tool** taking exactly **two parameters**: the
  **graph name** and the **Cypher query**. No second tool, no per-recipe tools.
- **FR-3** — A query is passed as the Cypher text itself — no shell quoting/escaping
  applies, and multi-line queries are accepted verbatim.
- **FR-4** — The graph name stays **caller-supplied** (a parameter, per FR-2); it is not
  hardcoded anywhere.
- **FR-5** — Asking the CPG a question costs no `redis-cli` process per query.
- **FR-6** — This **supersedes FR-9 of `joern-cpg-pipeline.md`**, which chose
  `redis-cli GRAPH.QUERY` over an MCP tool. The reversal is deliberate and must be
  recorded there, not left as a contradiction between the two documents.

*Context for the architect (not requirements):* the `cpg-analysis` skill's recipes and
its §1 connection section are written around `redis-cli`; whether they are rewritten,
wrapped, or left as a fallback is a design decision.

## Acceptance criteria
- **AC-1** — Given a CPG loaded in FalkorDB and a **cold agent session**, when the agent
  is asked "who calls `post_message`" (its **direct** callers), then it obtains the answer
  in **one tool call**, passing the graph name and the Cypher as parameters, having written
  no shell quoting or escaping. *(Amended 2026-07-25 per stakeholder ruling **D3** — the
  original wording asked for the callers "transitively". This feature changes **how Cypher
  is transmitted**, not how powerful Cypher is, and no single query answers the transitive
  form today. The bounded transitive upward-closure query is **deferred, not dropped** —
  backlog item **C-308**, owner `graph-dba`.)*
- **AC-2** — A **multi-line** Cypher query is accepted verbatim and returns the same
  result as its single-line equivalent.
- **AC-3** — The M2 acceptance queries, re-run through the tool against a **freshly built**
  `cpg_falkorchat`, return the **same values, the same row counts and the same row ordering** as
  the same queries run through `redis-cli GRAPH.QUERY` on the same graph; the resulting counts are
  recorded as the new baseline. The **display rendering of non-scalar (list/map) cells is excluded**
  from this equivalence: how a cell is rendered is governed by
  [`../plans/cpg-query-access.md`](../plans/cpg-query-access.md) **§4.4**, which is the authority
  (list/map → Python `repr`), so the same list may read `['CpgNode', 'IDENTIFIER']` through the tool
  and `[CpgNode, IDENTIFIER]` through `redis-cli` while carrying identical values in identical
  order. *(Amended 2026-07-25 per stakeholder rulings **D1** + **D2**. The figures this
  criterion originally cited were **AC-2 callers = 21** and **AC-8 test-gap = 39 rows / 32
  distinct method names** — the test-gap figure recorded here was **30**, which was wrong
  (source: `docs/plans/m2-cpg-analysis-coordination.md`, 2026-07-19). Both are now
  **superseded**: D1 authorised a full CPG rebuild, the source they were measured on has
  moved on, and `joern` records a **fresh baseline** on the rebuilt graph. The equivalence
  proof above is what AC-3 is met by — it tests this feature and is graph-independent.)*
  *(Further amended 2026-07-25 per stakeholder ruling **D5**, backlog item **C-313** — a
  **specification reconciliation**, not a loosening to accommodate a bug. The original wording
  demanded "byte-identical value sets", which no implementation could satisfy alongside plan §4.4;
  the acceptance run measured 5 of 6 equivalence pairs byte-identical and the sixth — the RCA
  data-flow slice — returning the **same 44 rows in the same order with the same values**, differing
  only in list syntax. Values, counts and ordering are what this criterion protects, and they held.)*
- **AC-4** — `joern-cpg-pipeline.md` FR-9 is updated to point at this document; no reader
  can find the two documents disagreeing about the access mechanism.

## Open questions
*(none)*

## Decision log
- 2026-07-19 — Trigger for the request? → Escaping/quoting pain, agents re-deriving the
  connection each session, per-query process overhead. Not graph correctness.
- 2026-07-19 — Blast radius? → **CPG analysis only**; component shell scripts and
  non-CPG FalkorDB access stay as they are.
- 2026-07-19 — Access mechanism? → **MCP tool**, deliberately reversing `joern-cpg-pipeline.md`
  FR-9. Shape fixed by the stakeholder: **one tool, two parameters (graph name, Cypher)**.
- 2026-07-19 — Per-user grants / read-only enforcement? → Raised, then **withdrawn**:
  "let's not change the auth, keep everything open for now." Out of scope.
- 2026-07-19 — Recipes reshaped? → **No.** `cpg-analysis` recipes keep handing agents raw
  Cypher (already copy-adapt-one-parameter); the tool only removes the shell layer.
- 2026-07-19 — `redis-cli` forbidden? → **No.** It stays usable; the skill documents the
  MCP tool as the path. The `joern` write/load side keeps `redis-cli` (out of scope).
- 2026-07-19 — Definition of "solved" → AC-1…AC-4 accepted as written.
- 2026-07-25 — AC-3's baseline (stakeholder rulings **D1** + **D2**, recorded verbatim in
  [`../plans/cpg-query-access-coordination.md`](../plans/cpg-query-access-coordination.md))
  → the M2 numbers are **not reproducible** — the source they were measured on has moved 8
  commits — and a destructive rebuild of `cpg_falkorchat` is **approved** ("dont worry about
  data in it, you can delete and recreate"). **AC-3 is restated** as *tool ≡ `redis-cli`
  equivalence* on the rebuilt graph, whose fresh counts become the new baseline (D1); the
  stale test-gap figure (**30**) is **corrected to 39 rows / 32 distinct method names** and
  then superseded by that baseline (D2).
- 2026-07-25 — AC-1's demonstrated question (stakeholder ruling **D3**, same source) →
  **direct** callers of `post_message`, not transitive. This feature changes how Cypher is
  transmitted, not how powerful Cypher is. **AC-1 is amended** accordingly; the bounded
  transitive upward-closure query is **deferred to backlog item C-308** (owner `graph-dba`),
  not discarded.
- 2026-07-25 — AC-3's *"byte-identical value sets"* vs plan §4.4 (stakeholder ruling **D5**,
  raised as defect **DEF-1** in
  [`../archive/test-reports/cpg-query-access-report.md`](../archive/test-reports/cpg-query-access-report.md) and
  carried as backlog item **C-313**) → the two approved specs could not both hold: AC-3 demanded
  byte-identical values while plan §4.4 mandates Python `repr` for list/map cells, so no
  implementation could satisfy both for any query projecting a non-scalar. **Option A chosen: AC-3
  is narrowed to values + row counts + ordering**, with non-scalar *rendering* excluded and plan
  §4.4 named as its authority. Option B — changing the server to render lists `redis-cli`-style —
  was **rejected**; **no source change**, the server is correct as built. This is a reconciliation
  of the specification, not a concession to a defect: the tool returned the same 44 rows in the
  same order with the same `line`/`code` values, differing only in how a `labels()` list is printed
  (case **TP-010**). **AC-3 passes** under the reconciled wording and C-313 is closed.
- 2026-07-25 — AC-4 executed → `joern-cpg-pipeline.md` FR-9 rewritten to point here, with a
  matching dated entry in that document's decision log. The two documents now agree: the MCP
  tool is the access mechanism, `redis-cli GRAPH.QUERY` the documented fallback.
