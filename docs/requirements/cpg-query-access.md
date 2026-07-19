# CPG query access — Feature Requirements
> Status: Ready for design · Last updated: 2026-07-19

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
  is asked "who calls `post_message`, transitively", then it obtains the answer in **one
  tool call**, passing the graph name and the Cypher as parameters, having written no
  shell quoting or escaping.
- **AC-2** — A **multi-line** Cypher query is accepted verbatim and returns the same
  result as its single-line equivalent.
- **AC-3** — The M2 acceptance queries re-run through the tool against `cpg_falkorchat`
  reproduce the recorded numbers — **AC-2 callers = 21**, **AC-8 test-gap = 30 untested
  methods** (per `docs/plans/m2-cpg-analysis-coordination.md`, 2026-07-19).
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
