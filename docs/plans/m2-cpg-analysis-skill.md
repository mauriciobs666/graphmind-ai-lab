# M2 — `cpg-analysis` skill: implementation plan

> Component: repo-root CPG / code-graph (Joern → FalkorDB). Milestone **M2** (consumer skill).
> Requirements: [`../requirements/joern-cpg-pipeline.md`](../requirements/joern-cpg-pipeline.md) (FR-9…FR-14, AC-6…AC-8).
> Backlog: [`../BACKLOG.md`](../BACKLOG.md) C-201…C-208.
> Implementer: **`graph-dba`** (authors + live-verifies the Cypher). Vetter: **`cobb`** (skill standards).
> Reviewer: **`analyst`** (independent). Coordinator: **`teco`**.
> Author: architect · 2026-07-18.

## 1. Goal & scope

Deliver **one** Claude Code / Agent-Skills skill, `cpg-analysis`, that teaches the existing
agent team to query an already-loaded Code Property Graph in FalkorDB via
`redis-cli GRAPH.QUERY`. Shape is **locked**: a lean `SKILL.md` core (connection + shared
traversal idioms) plus **four** bundled `references/*.md` recipes — impact-analysis, RCA,
code-review, test-gap — progressively disclosed. This mirrors the core-plus-references pattern
of `skills/agent-standards/` and `skills/agent-maintenance/`.

**In scope:** the skill (core + 4 recipes), citing `skills/joern-cpg/references/cpg-model.md` as
the single schema source; agent-description wiring for `analyst`/`architect`/`qa-engineer`; catalog
+ doc sync. Everything is **skill + docs only — no application code**.

**Out of scope** (from the requirements, do not drift into): RAG/vector indexing of code;
continuous auto-sync; runtime line/branch coverage (test-gap is *structural reachability* only);
wiring the CPG into falkor-chat workflows; four sibling skills (explicitly rejected); a new schema
doc (reuse `cpg-model.md`).

## 2. Context & findings

- **The schema contract is already canonical and complete for authoring.**
  `skills/joern-cpg/references/cpg-model.md` fixes: shared `:CpgNode` label + per-node Joern type
  label; `CpgNode(id)` indexed; the `:ID → id` (lowercase, integer) rule; **UPPER_CASE property
  keys** (`m.NAME`, `m.CODE`, `m.FULL_NAME`, `m.IS_EXTERNAL`, `m.LINE_NUMBER`) — a lowercase key
  silently returns null; **real booleans** (`WHERE m.IS_EXTERNAL = false`, not `'false'`); edge
  relationship types verbatim (`AST`, `CFG`, `CALL`, `ARGUMENT`, `REACHING_DEF`, `CONTAINS`,
  `RECEIVER`, `DOMINATE`, …). Node-label vocabulary (`METHOD`, `METHOD_PARAMETER_IN`, `CALL`,
  `IDENTIFIER`, `LITERAL`, `LOCAL`, `FILE`, `TYPE_DECL`, …) is enumerated there. **The recipes cite
  this file; they must not restate the schema (FR-14).**
- **The `joern-cpg` skill is the producer counterpart** — same repo home (`skills/`), same
  connection substrate (`redis-cli GRAPH.QUERY`, `FALKORDB_HOST`/`FALKORDB_PORT`, started via
  `falkor-chat/scripts/start_falkordb.sh`). Its `SKILL.md` already carries a **CPGQL** cheat-sheet
  (Scala DSL, in the Joern REPL). Note the altitude split: CPGQL is for the *binary CPG in the Joern
  REPL* (lowercase `.name`/`.code`); `cpg-analysis` is **Cypher over the loaded FalkorDB graph**
  (UPPER_CASE keys). The recipes are Cypher, not CPGQL — this is exactly the gap M1 left open.
- **Skills in this repo do not carry a `kaizen/` dir.** `skills/joern-cpg/` has none, and root
  `AGENTS.md` states kaizen is an *agent-folder* convention only. So `cpg-analysis` gets **no**
  `kaizen/` dir — the component's `BACKLOG.md`/`HISTORY.md` is its change ledger. (This overrides the
  generic "every skill carries kaizen" line in the `agent-maintenance` skill §1, which is written
  for other projects.)
- **Consumer agents today** (`claude/analyst/analyst.md`, `claude/architect/architect.md`,
  `claude/qa-engineer/qa-engineer.md`) have no CPG-capability line in their `description` (their live
  routing contract). `analyst` and `architect` carry `Bash`; `qa-engineer` carries all tools. All can
  run `redis-cli`. C-207 adds the capability line; cobb owns it.
- **Doc-sync surface (AGENTS.md "Module documentation convention" + skill-maintenance §2):** a skill
  change lands with `skills/README.md` (catalog), root `AGENTS.md` (Skills section), `claude/README.md`
  (only if agent descriptions change — C-207 does change them), and this component's
  `BACKLOG.md` → `HISTORY.md` — **all in the same change**.
- **Existing reference-bundle precedent:** `agent-standards/` = lean `SKILL.md` + a *Navigation*
  table pointing at `claude-code.md`/`kiro.md`/`opencode.md`; `agent-maintenance/` = one fat `SKILL.md`
  with `## N.`-numbered sections. `cpg-analysis` follows the `agent-standards` shape (core + a routing
  table to four sibling reference files), because the four recipes are independent task-shaped units
  loaded on demand.

## 3. Decision on OQ2 (component structure / naming)

**Decision: no new `code-graph/` code directory. M2 lives entirely as (a) the new
`skills/cpg-analysis/` skill, (b) the existing `joern`/`graph-dba` agents that own it, and (c) the
existing repo-root `docs/` (requirements/BACKLOG/HISTORY/plans).**

**Rationale.** A top-level component directory in this monorepo earns its place by holding *runnable
artifacts* — `salesperson/` and `falkor-chat/` each carry application code, scripts, and their own
`AGENTS.md`. M2 ships **no application code**: it is a `SKILL.md` package (which must live under
`skills/` to be symlink-deployed to all three harnesses — see `skills/README.md` deployment table)
plus Markdown docs (which already live at repo-root `docs/` by the 2026-07-12 placement decision).
A `code-graph/` dir would be an **empty shell** — it could hold nothing that isn't already correctly
homed, and it would fragment the CPG story across `skills/joern-cpg/` (producer),
`skills/cpg-analysis/` (consumer), `docs/` (specs), and `code-graph/` (nothing), *worsening*
navigability rather than improving it. The component's identity is already carried coherently by the
repo-root `docs/` triplet + the two `skills/` packages + two agents.

**Naming.** Skill folder = `cpg-analysis` (folder name **must** equal frontmatter `name`; ≤ 64
chars — OK). The four recipes are files inside it, not `cpg-`-prefixed sibling skills; the
`cpg-test-coverage → cpg-test-gap` rename in the backlog resolves to the recipe filename
`test-gap.md`.

**Reconsider only if** a future milestone adds runnable code that is neither Joern-pipeline (belongs
in `joern-cpg`) nor a skill (e.g. a standalone CPG query service, or the falkor-chat wiring). At that
point a `code-graph/` component with its own `AGENTS.md` becomes justified. Flag this in `BACKLOG.md`
as the trigger; do not pre-build it now.

## 4. Skill file layout

```
skills/cpg-analysis/
├── SKILL.md                          # C-202 — lean core (graph-dba)
└── references/
    ├── impact-analysis.md            # C-203 — FR-10 / AC-2, AC-3  (analyst, architect)
    ├── rca.md                        # C-204 — FR-11 / AC-4, AC-5  (analyst)
    ├── code-review.md                # C-205 — FR-12 / AC-7        (analyst)
    └── test-gap.md                   # C-206 — FR-13 / AC-8        (qa-engineer)
```

No `kaizen/`, no `scripts/` (recipes are Cypher run via the consuming agent's own `redis-cli`; if a
recipe grows a helper it can add `scripts/` later, but M2 needs none).

### `SKILL.md` frontmatter

```yaml
---
name: cpg-analysis                    # MUST equal the folder name
description: <≤ 1024 chars — see below>
allowed-tools: Bash, Read             # recipes run `redis-cli GRAPH.QUERY` (Bash) and read the schema ref (Read)
---
```

- `description` (the always-on trigger; ≤ 1024 chars) must name **what** (query a loaded Joern CPG in
  FalkorDB), the **four task shapes** (impact analysis, root-cause analysis, code review / taint,
  test-gap), the **consumers** (analyst, architect, qa-engineer), and the **access** (Cypher via
  `redis-cli GRAPH.QUERY`, schema per `joern-cpg`'s `cpg-model.md`). Draft for cobb to lint:
  > "Query an already-loaded Joern Code Property Graph in FalkorDB with Cypher (`redis-cli
  > GRAPH.QUERY`) to do impact analysis (callers/callees + transitive reach), root-cause analysis
  > (data-flow + symbol def/ref), code review (input→risky-sink taint), and test-gap analysis (code
  > reachable from prod but no test entrypoint). Use when analyst/architect/qa-engineer need
  > structured call-graph / data-flow answers instead of reading files. Recipes cite the single CPG
  > schema in `skills/joern-cpg/references/cpg-model.md`. Requires a CPG already loaded by the
  > `joern` pipeline; building/loading a CPG routes to `joern`."
- `allowed-tools`: `Bash, Read`. Rationale: the recipes execute `redis-cli` (Bash) and the core
  points readers at the schema file (Read). Do **not** add Write/Edit — the skill only reads the
  graph. (Note per `skills/README.md`: `allowed-tools` gating *behavior* is Claude-Code-specific; the
  field is harmless in other tools. cobb verifies via `agent-standards`.)

### How the core references the recipes and cites the schema

The `SKILL.md` core carries a **Navigation** table (agent-standards shape) — one row per recipe:
"You need to… → open `references/<file>.md`". It states **once**, prominently, that the node/edge
labels and property contract live in **`skills/joern-cpg/references/cpg-model.md`** (relative link
`../joern-cpg/references/cpg-model.md`) and are **not** repeated here or in any recipe (FR-14). Each
recipe file opens with a one-line back-pointer to that schema file and to the core, then contains only
its task-specific query patterns.

## 5. Recipe → requirement → consumer → graph concept map

For each recipe, `graph-dba` **authors and live-verifies the actual Cypher**. This plan frames the
*concepts* each must cover; it deliberately does **not** write the Cypher (graph-dba owns dialect,
edge topology, and index reality). Each recipe file should carry: purpose, inputs (what the caller
supplies), the query pattern(s), how to read the result, and known limits.

### `impact-analysis.md` — C-203 · FR-10 (packages FR-2, FR-3) · AC-2, AC-3 · analyst, architect
Concepts to cover:
- **Callers & callees of a function** over `CALL`. Cover the edge-topology subtlety graph-dba must
  resolve: in Joern a `CALL` node is the call *site*; the callee `METHOD` is reached from it, and the
  *caller* method is the `METHOD` that contains the `CALL` node (via `AST`/`CONTAINS`). "Callers of
  M" and "callees of M" are therefore not a single symmetric edge — spell out the resolved traversal
  once verified. Match the target method by `FULL_NAME`/`NAME` (UPPER_CASE keys).
- **Transitive up/downstream reach** — variable-length traversal over the resolved call relation
  ("what could break if I change X"), both directions, with a bound to keep it tractable at repo
  scale. This is the AC-3 query.
- Note `IS_EXTERNAL = false` (real boolean) to exclude library stubs when the caller wants
  first-party reach only.

### `rca.md` — C-204 · FR-11 (packages FR-4, FR-5) · AC-4, AC-5 · analyst
Concepts to cover:
- **Data-flow back from a symptom** — reverse traversal over `REACHING_DEF` from a use (an
  `IDENTIFIER`/`CALL` argument) to the definitions that reach it, transitively (AC-4). Frame the
  "given a value/parameter, return propagation path(s)" shape.
- **Cross-file symbol definition & references** (AC-5) — locate where a symbol is defined
  (`LOCAL` / `METHOD_PARAMETER_IN` / `METHOD` / `MEMBER`) and every reference (`IDENTIFIER` /
  `FIELD_IDENTIFIER` / `CALL`) to it, spanning `FILE`s. graph-dba resolves how "same symbol across
  files" is expressed (by `NAME`/`FULL_NAME`, associating nodes to their `FILE` via containment).

### `code-review.md` — C-205 · FR-12 · AC-7 · analyst
Concepts to cover:
- **Input → risky-sink taint**: source set = input-like calls / external parameters; sink set = risky
  calls matched by `NAME`/`METHOD_FULL_NAME` regex (e.g. exec/system/query/eval families — recipe
  ships a starter list the caller edits); path = `REACHING_DEF`(+`ARGUMENT`) reachability from a
  source expression into a sink argument. **AC-7 both directions:** tainted code reports the path(s);
  clean code reports **none**.
- State the fidelity limit plainly: this is *static graph reachability*, an approximation of Joern's
  `reachableBy` taint engine (which the `joern` agent can run in the REPL for higher fidelity). The
  recipe should say "for deep taint, escalate to `joern`/CPGQL" as the failure path.

### `test-gap.md` — C-206 · FR-13 · AC-8 · qa-engineer
Concepts to cover:
- **Prod-vs-test entrypoint reachability** (AC-8): classify `METHOD`s as production vs. test by a
  path/name heuristic on their owning `FILE`/`FULL_NAME` (test dirs, `test_`/`_test`/`spec`
  conventions — recipe documents the heuristic and makes it overridable). Compute the set reachable
  over `CALL` from prod entrypoints, subtract the set reachable from any test entrypoint; return the
  difference. graph-dba decides how to express set-difference reachability in FalkorDB OpenCypher
  (no GDS) — possibly two reachability queries diffed, or a single neg: pattern.
- Reinforce the scope line in the recipe: **structural reachability, not runtime coverage** — a
  method with a test path may still be functionally untested; that's out of scope.

## 6. Live-verification strategy (AC-6 is the gate)

A joern-built CPG will be loaded in a running FalkorDB; the **graph key is reported separately** to
graph-dba (do not hardcode it in the skill — the skill takes the graph key as caller input; verify
against the provided key). AC-6 requires that an agent invoking the skill runs the recipe Cypher and
gets **correct** results *without hand-knowing the schema*. "Correct" per recipe:

1. **Establish ground truth independently.** For the verification target, pick concrete anchors
   (a known function with known callers, a known symbol, a known input→sink path, a known
   prod-only method). Establish the expected answer **outside** the recipe — either by the `joern`
   agent running the equivalent **CPGQL** in the REPL (its `cpg.method.name(...).caller`,
   `reachableBy`, etc.) or by direct source inspection. This is the oracle.
2. **Run the recipe Cypher** via `redis-cli GRAPH.QUERY <graphkey> "<cypher>"` and compare to the
   oracle. Per recipe:
   - **impact** (AC-2/AC-3): callers set and callees set match the CPGQL `.caller`/`.callee` sets for
     a chosen method; transitive reach at depth *n* matches a hand-traced or CPGQL-traced chain.
   - **rca** (AC-4/AC-5): the `REACHING_DEF` back-trace for a chosen use returns the definition(s)
     CPGQL's data-flow reports; def/ref for a chosen symbol returns every file location found by
     inspection.
   - **code-review** (AC-7): **two** cases — a seeded/known tainted path returns non-empty **and** a
     known-clean function returns empty. Both must hold (the false-positive guard is as important as
     the true-positive).
   - **test-gap** (AC-8): a method known to be reachable only from production returns in the list; a
     method reachable from a test entrypoint is absent. Sanity-check the counts against the
     prod/test file split.
3. **Record evidence.** graph-dba captures the exact `GRAPH.QUERY` command + trimmed output for each
   AC in the delivery note (feeds `HISTORY.md` and analyst's review). M2 is "live-verified," not
   merely authored — this evidence is the proof.
4. **Language coverage caveat (ties to OQ3, §8).** Record **which language(s)** the verification CPG
   actually covered. The recipes are schema-driven and language-agnostic (they traverse CPG labels,
   not language syntax), so verifying on a Python CPG establishes correctness of the *queries*; note
   explicitly if JS/TS was not exercised, so the AC claims are honest about coverage.

## 7. Step-by-step sequence (C-201 → C-208)

Follows the backlog critical path: C-201 → C-202 → {C-203, C-204, C-205, C-206} → C-207 → C-208.
Recipes are parallelizable after the core exists; keep each recipe individually verifiable.

1. **C-201 — Adopt the schema contract (graph-dba).**
   Read `skills/joern-cpg/references/cpg-model.md` and confirm it is the canonical node/edge/property
   reference the recipes will cite. **No new schema doc.** Identify any genuinely *missing schema
   fact* that consumer queries need but the file lacks (e.g. the exact `CALL`→callee vs.
   caller-containment topology). **Consumer-query idioms (label → Cypher pattern mapping) belong in
   the new `SKILL.md` core, not in `cpg-model.md`.** Only if a *schema* fact (not a query idiom) is
   missing, add it to `cpg-model.md` — which is part of the `joern-cpg` skill, so that edit triggers
   `joern-cpg` doc-sync too (flag to teco; coordinate with the `joern` agent). **Done:** written
   confirmation of the contract + a list of the label→idiom mappings the core will carry.

2. **C-202 — Skill core `SKILL.md` (graph-dba).**
   Create `skills/cpg-analysis/SKILL.md` with the §4 frontmatter and a lean body: (a) FalkorDB
   connection via `redis-cli GRAPH.QUERY <graphkey> "…"` incl. `FALKORDB_HOST`/`PORT`, graph key as
   caller input; (b) the `:CpgNode` + `CpgNode(id)` model and the three gotchas (UPPER_CASE keys,
   lowercase `id`, real booleans) — **linking** to `cpg-model.md`, restating only the minimum a query
   author trips on; (c) the shared traversal idioms (callers/callees over `CALL`, bounded transitive
   reach, data-flow over `REACHING_DEF`, symbol def/ref); (d) the Navigation table to the four
   recipes. **Done:** core reads clean, cites the one schema source, passes a self-check that no schema
   table is duplicated.

3. **C-203 — `references/impact-analysis.md` (graph-dba).** Author + live-verify per §5/§6.
   **Done:** AC-2 and AC-3 verified against the loaded CPG with captured evidence.

4. **C-204 — `references/rca.md` (graph-dba).** Author + live-verify. **Done:** AC-4, AC-5 verified.

5. **C-205 — `references/code-review.md` (graph-dba).** Author + live-verify. **Done:** AC-7 verified
   in **both** directions (tainted → non-empty, clean → empty).

6. **C-206 — `references/test-gap.md` (graph-dba).** Author + live-verify. **Done:** AC-8 verified;
   structural-only scope stated in the recipe.

   *(Steps 3–6 parallelizable; each is independently reviewable — keep them separate commits/edits so
   analyst can review recipe-by-recipe.)*

7. **C-207 — Agent wiring (cobb).** Add a concise CPG-capability line to the `description`
   frontmatter of `claude/analyst/analyst.md`, `claude/architect/architect.md`, and
   `claude/qa-engineer/qa-engineer.md` (their live routing contract), naming the `cpg-analysis` skill
   and graph-dba ownership. cobb runs the §7 prompt-quality lint on each changed description and the
   new `SKILL.md`. Because agent descriptions change, `claude/README.md` entries update in the same
   change. **Done:** three descriptions carry the capability line; cobb's lint clean; boundary/roster
   parity holds (`audit-team.sh` still green).

8. **C-208 — Catalog & doc sync (cobb + graph-dba, one change).** Land together, per AGENTS.md:
   - `skills/README.md` — new `cpg-analysis` catalog row (what / when / origin).
   - root `AGENTS.md` — add `cpg-analysis` to the Skills section (and the skill count if enumerated —
     prefer not to hardcode counts; if the "all 7 skills" phrasing exists, update it or delete the
     enumerated fact per agent-maintenance §2.3).
   - `claude/README.md` — reflects the C-207 description changes (if not already done in step 7).
   - `docs/BACKLOG.md` → mark C-201…C-208 ✅; append a dated `docs/HISTORY.md` entry (skill delivered,
     ACs verified with the evidence pointers, commit).
   - Requirements doc status line: flip M2 from "in progress" to delivered, close OQ2 (this decision)
     and note OQ3's outcome.
   **Done:** skill source + `skills/README.md` + `AGENTS.md` + agent wiring + BACKLOG→HISTORY all in
   the same change; analyst signs off as independent reviewer; cobb certifies.

## 8. OQ3 — Joern Python + JS/TS frontend coverage (parallel, `joern`-owned)

OQ3 (do the Python + JS/TS Joern frontends adequately cover the monorepo targets?) is a **design-time
verification the `joern` agent handles in parallel** — it owns the frontends and the CPG build. This
plan's **only dependency** on OQ3 is the **verification substrate**: §6 needs a loaded CPG whose
language(s) are known.

- **Not a blocker for authoring.** The recipes traverse CPG *labels/edges*, which are frontend-agnostic
  — the Cypher is identical whether the CPG came from `pysrc2cpg` or `jssrc2cpg`. graph-dba can author
  and verify against **whatever CPG is loaded** (Python is sufficient to prove query correctness).
- **The dependency to honor:** record in the C-208 evidence which language(s) were verified (§6.4). If
  OQ3 concludes a frontend under-covers a target (e.g. JS/TS misses call resolution for a construct),
  that limits *what the recipe can find on that language*, not the recipe's correctness — capture it as
  a known limitation in the affected recipe and a `BACKLOG.md` follow-up, and let the `joern` agent's
  OQ3 finding drive any frontend remediation. **Do not gate M2 on JS/TS**; note the coverage honestly.

## 9. Risks & open questions

- **Edge topology is the main authoring risk (impact + rca).** Joern's `CALL` node = call site, not a
  method-to-method edge; caller resolution goes through AST containment. graph-dba must resolve the
  exact traversal against the live graph before trusting the impact recipe — verification (§6) catches
  a wrong assumption, so gate the recipe on it.
- **Repo-scale traversal cost.** `AST`/`CFG`/`REACHING_DEF` dominate edge count; unbounded
  variable-length traversals can blow up. Recipes must bound depth and prefer `IS_EXTERNAL = false`
  filters; graph-dba should `GRAPH.EXPLAIN`/`PROFILE` the heavier queries and note any recipe that
  needs a supporting index (a per-label index is graph-dba's call, out of the skill's scope but worth
  a `BACKLOG.md` note if one materially helps).
- **Taint fidelity (code-review).** Static Cypher reachability ≠ Joern's `reachableBy`. Accept the
  approximation for the skill; the recipe's failure path is "escalate to `joern`/CPGQL for deep
  taint." Ensure AC-7's clean-code-returns-none case is verified, not just the positive.
- **`cpg-model.md` edits ripple into the `joern-cpg` skill.** If C-201 must add a schema fact there,
  that is a second skill's doc-sync — coordinate through teco with the `joern` agent so the two skills
  don't drift.
- **Graph key handling.** The skill must take the graph key as caller input (never hardcode the
  verification graph's key), or every consumer breaks on the next CPG. Verify this is explicit in the
  core.
- **No open question blocks the build.** OQ2 is decided here; OQ3 is parallel and non-blocking. The one
  input graph-dba needs before §6 is the **live graph key**, reported separately per the brief.

## 10. Ready to implement

Plan: **`/home/mauricio/prg/graphmind-ai-lab/docs/plans/m2-cpg-analysis-skill.md`**

- **OQ2 decided:** no `code-graph/` code dir — M2 is skill + docs only; it lives as
  `skills/cpg-analysis/` + the `joern`/`graph-dba` agents + the existing repo-root `docs/`. A
  component dir would be an empty shell fragmenting the CPG story; revisit only when runnable non-skill
  code appears.
- **Build order:** C-201 (adopt `cpg-model.md`, no new schema doc) → C-202 (`SKILL.md` core, cites the
  one schema, `allowed-tools: Bash, Read`) → C-203/204/205/206 (four `references/*.md` recipes,
  each **live-verified** against the loaded CPG per its ACs) → C-207 (cobb wires
  analyst/architect/qa-engineer descriptions) → C-208 (catalog + AGENTS.md + BACKLOG→HISTORY, one
  change; analyst reviews, cobb certifies).
- **Owner:** graph-dba authors + verifies the Cypher (this plan frames concepts, not Cypher); cobb
  vets standards + does the wiring; analyst is the independent reviewer.
- **Open risks:** `CALL`-node caller topology (verify before trusting impact/rca), repo-scale
  traversal cost (bound depth, PROFILE), static-taint fidelity (approximation + escalate-to-joern
  path), and honest per-language coverage tied to `joern`'s parallel OQ3. graph-dba needs the **live
  graph key** (reported separately) before verification.
