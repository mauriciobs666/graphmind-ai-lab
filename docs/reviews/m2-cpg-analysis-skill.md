# Review — M2 `cpg-analysis` skill (Gate-2a: correctness + cold AC-6 invocation)

> Artifact: [`../../skills/cpg-analysis/`](../../skills/cpg-analysis/) — `SKILL.md` + `references/{impact-analysis,rca,code-review,test-gap}.md`
> (graph-dba, delivered 2026-07-19), plus the additive C-201 "Consumer-query facts" section in
> [`../../skills/joern-cpg/references/cpg-model.md`](../../skills/joern-cpg/references/cpg-model.md).
> Baseline: requirements [`../requirements/joern-cpg-pipeline.md`](../requirements/joern-cpg-pipeline.md) (FR-9…FR-14, AC-2…AC-8),
> plan [`../plans/m2-cpg-analysis-skill.md`](../plans/m2-cpg-analysis-skill.md), Gate-1 review
> [`m2-cpg-analysis.md`](m2-cpg-analysis.md), coordination [`../plans/m2-cpg-analysis-coordination.md`](../plans/m2-cpg-analysis-coordination.md).
> Live substrate: `cpg_falkorchat` on FalkorDB :6379 — **79,581 nodes / 522,182 edges** (byte-identical to graph-dba's verification graph), falkordb v4.18.11.
> Reviewer: analyst (independent; did NOT author the skill) · 2026-07-19.

## Scope & verdict

Two-part Gate-2a: (1) a static correctness review of the delivered `cpg-analysis` skill's Cypher against the
real schema/topology, and (2) a **cold AC-6 invocation** — I ran every recipe's queries against the live loaded
CPG following **only** what the skill and its cited references tell a schema-naive caller, and judged whether
each recipe yields a correct result without hand-knowing the schema. I ran read-only `GRAPH.QUERY` only; I did
not modify the graph or any artifact. Skill-format lint, frontmatter standards, and agent-description wiring are
cobb's Gate-2b and are out of my scope.

**Verdict: approve with suggestions.** All four recipes produce correct results for a schema-naive caller — AC-2,
AC-3, AC-4, AC-5, AC-7, and AC-8 all reproduce live, and AC-6 is met on every recipe. No blocker. One **major**
finding: the test-gap recipe's recorded verification count (30) does not reproduce on the identical graph (it is
a stable **39 rows / 32 distinct names**) — the recipe is functionally correct, but its "live-verified" evidence
number is wrong and must be corrected before it lands in `HISTORY.md`. The rest are minor precision/completeness
notes. Owner for all query-side items: **graph-dba**.

## AC-6 cold-invocation outcome (per recipe)

I opened each recipe as a schema-naive agent would (the SKILL core + the recipe + the one cited `cpg-model.md`
pointer), ran the queries verbatim, and compared to the recipe's stated "Verified" result. All four **pass** the
"correct results without hand-knowing the schema" bar; the substrate handling flagged in the brief is honored:

| Recipe | AC | Cold result | AC-6 |
|---|---|---|---|
| impact-analysis | AC-2, AC-3 | Q1 → 21 distinct callers (2 prod, 19 test); Q2a → `_dispatch_write`,`_next_ts`,`_validate_and_derive_role`; Q3 → those + `_dedup`. **Exact match** to recipe. | ✅ pass |
| rca | AC-4, AC-5 | Backward slice from `build_router.post_message` RETURN → `posted`@143 + feeders (`services`,`ctx`,`body.text`, `_safe_*` refs); `get_context` def → both api.py:29 & config.py:99 (METHOD+TYPE_DECL), refs → both files. **Match.** | ✅ pass |
| code-review | AC-7 | Positive `Repository.post_first_message` → `query`@320 non-empty (incl. `text`); negative `Services.ping` → empty. Interproc bridge crosses real edges; no prod entrypoint reaches a `query` sink over the resolved graph — **reproduced and honestly documented.** | ✅ pass |
| test-gap | AC-8 | Query runs; seed counts (336/17/10) match; anchors hold (`ping`,`_safe_respond`@48,`_safe_run_workflow`@71 flagged; `_serialize_opaque` correctly excluded via `publish_workflow_def`). **But headline count is 39 rows / 32 names, not the stated 30** (see Major-1). | ✅ pass (with count caveat) |

A schema-naive caller who copies the recipe query and changes the one documented parameter gets a correct answer
in all four cases. The gotcha restatement in `SKILL.md §2` (UPPER_CASE keys, real booleans, CALL-site vs CALL-edge,
FILENAME only on METHOD/TYPE_DECL, intraprocedural REACHING_DEF) is exactly the set a caller trips on, and each
recipe carries the boundary/limit notes needed to not over-claim. **AC-6: satisfied.**

## Findings

### Major

**M-1 — test-gap recipe's recorded verification count (30) does not reproduce; live graph yields a stable 39 rows / 32 distinct names.**
Evidence — the recipe's exact query (`references/test-gap.md` lines 29–51) run verbatim against `cpg_falkorchat`:
```
RETURN count(DISTINCT g.NAME) AS distinctNames, count(*) AS rows  →  distinctNames=32, rows=39
```
Stable across three runs. The recipe (`test-gap.md:72`) and the coordination log both state "**30** methods flagged."
The graph is byte-identical to graph-dba's verification substrate (79,581 nodes / **522,182 edges** confirmed this
session), so this is not a graph drift — the stated evidence number is simply wrong. The 39-vs-32 gap is legitimate:
same-`NAME` methods at different sites (e.g. `record` ×3 in `executor.py` at L126/134/150, `ping` in
`repository.py`+`services.py`, `_default_clock` in `executor.py`+`services.py`) — and the recipe's own "expected
shape" says *"one row per untested production method (file + line)"*, which makes **39 rows** the intended output.
Why it matters: M2's acceptance standard is "live-verified," and this is the one recorded number that a reviewer or
consumer would sanity-check first; a figure that reproduces as 39/32 undermines confidence in the verification
evidence, even though the recipe itself is functionally correct (all four documented anchors hold).
*Suggested improvement (graph-dba):* re-run the recipe, correct the stated count to the reproduced value, and add one
clarifying line distinguishing **rows (39, one per file:line)** from **distinct names (32)** so the number is
self-checking. Do this before the count is copied into `HISTORY.md` at C-208.

### Minor

**m-2 — code-review Pattern A treats *every* `METHOD_PARAMETER_IN` as a taint source, so the positive case over-reports.**
Evidence — the positive query returns 9 "taintedParam" rows for `Repository.post_first_message` (`thread_id`, `role`,
`msg_id`, `self`, `mentions`, `created_at`, `author_id`, `ws`, `text`), not just the externally-influenced `text` the
recipe's narrative singles out (`code-review.md:33-36`). `self` and internally-derived params are flagged purely
because they are arguments to the `query` call. AC-7 still holds (tainted → non-empty, clean → empty), so this is
precision, not correctness — but a schema-naive analyst reading the output would over-report "9 tainted parameters."
*Suggested improvement:* add a line to Pattern A noting the source set is *all* parameters, and that the analyst must
judge which are genuinely externally-influenced (or narrow the source set, e.g. exclude `self`).

**m-3 — AC-5 cross-file references are demonstrated on a symbol (`get_context`) that has essentially no cross-file usage, and the ref query misses `LOCAL` bindings.**
Evidence — `MATCH (n) WHERE n.NAME='get_context'` returns 2 IDENTIFIER, 2 LOCAL, 2 METHOD, 2 TYPE_DECL; there are
**zero** `CALL` sites named `get_context`, so the recipe's "references across files" query resolves to only the two
IDENTIFIERs *at the definition sites themselves* (api.py:29, config.py:99) — not genuine downstream cross-file uses.
The ref query (`rca.md:68-69`) matches `IDENTIFIER/CALL/METHOD_REF/FIELD_IDENTIFIER` but not `LOCAL`, so the 2 `LOCAL`
`get_context` bindings are silently dropped. The recipe returns "all that exist," so AC-5 is technically satisfied,
but the chosen anchor is a weak demonstration of "cross-file references" and the `LOCAL` omission is undocumented.
*Suggested improvement:* pick an AC-5 anchor with real multi-file call usage (e.g. a widely-called helper) so the
"references across files" claim is actually exercised, and either add `LOCAL` to the ref match or note its exclusion.

**m-4 — test-gap satisfies AC-8 by the *complement* of the test-reach closure, not by forward-reach from prod entrypoints; worth stating the deviation explicitly.**
The plan (§5) framed AC-8 as "set reachable from prod entrypoints minus set reachable from test entrypoints." The
delivered recipe instead returns *all first-party `falkorchat/` methods not in the test-`NAME`-closure* — a superset
that also includes prod code reachable from no entrypoint at all (dead code). Given the brief's substrate fact that
framework entrypoints (17 routes + 9 MCP tools) are **not statically connected**, a forward-reach-from-prod-entrypoint
design would have been crippled, so the complement approach is the right adaptation, and the recipe honestly frames
itself as "a lower bound on untested code." This is a sound deviation, not a defect — but it is a deviation from the
plan's stated method and reframes what "reachable from production" means. *Suggested improvement:* one sentence in the
recipe's Purpose noting it computes "prod code no test reaches" (complement), not "prod-entrypoint-forward minus
test," so the AC-8 mapping is explicit rather than implicit. (The "Seed the entrypoints" block is a count sanity-check,
not an actual seeding of the closure — the heading slightly oversells; consider "Sanity-check the prod/test split.")

### Nit

**n-5 — SKILL vs. rca def-query inconsistency for symbol definitions.**
`SKILL.md §3` "Cross-file symbol def & references" matches `d:METHOD OR d:TYPE_DECL OR d:LOCAL` over `[:CONTAINS|AST]`,
while `rca.md`'s def query matches only `d:METHOD OR d:TYPE_DECL`. Both work; the mismatch is cosmetic. Align them (or
note that `rca.md` intentionally drops `LOCAL` to avoid scope-collision noise, which ties to m-3).

## What's solid

- **The two hardest substrate traps are handled correctly and honestly.** The CALL-site-vs-CALL-edge topology
  (callers by `CONTAINS`+`NAME`, callees by the `CALL` edge) reproduces exactly, and the code-review recipe's central
  honesty claim — *no production entrypoint reaches a `query` sink over the resolved call graph, so a clean Pattern-B
  result across a cross-object boundary is inconclusive, not proof of safety* — reproduces live (empty result) and is
  documented as a limit, not hidden. This is exactly the AC-7 false-negative trap the Gate-1 review (M1) worried about,
  and it is defused correctly: the recipe reports Pattern-A positives **and** the coverage caveat, and routes deep
  cross-object taint to `joern`/`reachableBy`.
- **The intraprocedural `REACHING_DEF` boundary is named in every place it matters** — `SKILL.md §2` gotcha 5, rca's
  "Interprocedural boundary (must state in findings)", and code-review's "Fidelity limit." No over-claiming.
- **FR-14 (single schema source) is honored.** The SKILL and all four recipes cite `joern-cpg/references/cpg-model.md`
  and do not restate the schema; the only in-skill restatement is the sanctioned "minimum you trip on" gotcha list.
  The additive C-201 "Consumer-query facts" section lands in the one canonical file, correctly scoped as topology
  facts (not producer semantics).
- **impact and rca recipes reproduce their documented results exactly** — callers/callees/transitive-reach and the
  data-flow slice + def/ref all match, cold, with no schema knowledge beyond the skill's pointer.
- **Determinism.** The heavy test-gap closure query is stable across repeated runs (the FalkorDB `WITH`-splitting
  idiom note in the recipe is real and correctly applied).

## Open questions (for the caller / teco)

1. **M-1 disposition:** correcting the test-gap count is a graph-dba doc fix (re-run + update the number). Confirm it
   is done before C-208 copies the "AC-8 verified = N" evidence into `HISTORY.md` — otherwise the wrong number
   propagates into the permanent change ledger.
2. **Scope caveat to record (not a defect):** all live verification here and in W2 was **Python-only** (`pysrc2cpg`);
   JS/TS frontends were not exercised. The recipes are label-driven and language-agnostic, so this bounds *coverage
   claims*, not query correctness — but it should be stated in the C-208 `HISTORY.md` entry and any recipe that a
   consumer might run against a JS/TS CPG.
</content>
</invoke>
