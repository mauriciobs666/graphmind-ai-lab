# Review — M2 `cpg-analysis` skill implementation plan

> Artifact: [`../plans/m2-cpg-analysis-skill.md`](../plans/m2-cpg-analysis-skill.md) (architect, 2026-07-18)
> Baseline: requirements [`../requirements/joern-cpg-pipeline.md`](../requirements/joern-cpg-pipeline.md) (FR-9…FR-14, AC-2…AC-8),
> backlog [`../BACKLOG.md`](../BACKLOG.md) (C-201…C-208), schema contract
> [`../../skills/joern-cpg/references/cpg-model.md`](../../skills/joern-cpg/references/cpg-model.md),
> skill standards [`../../skills/agent-standards/SKILL.md`](../../skills/agent-standards/SKILL.md), root `AGENTS.md`.
> Reviewer: analyst (independent) · 2026-07-18.

## Scope & verdict

Static review of the M2 build plan against its requirements, backlog, the canonical CPG schema, and the
repo's skill/doc conventions. I verified the plan's grounding claims against the real files (schema-property
enumeration, agent descriptions, the AGENTS.md skill count) but did **not** author or run any Cypher — that is
graph-dba's live-verification work in W2, gated by this review.

**Verdict: approve with suggestions.** The plan is well-grounded, scope-disciplined, and correctly maps every
FR/AC to a recipe and consumer. No blockers. Three **major** items should be tightened before/within
implementation — they concern verification *validity* (whether a "pass" actually proves the AC), not the design.
The rest are minor/nit hardening.

## Findings

### Major

**M1 — Data-flow recipes conflate intraprocedural `REACHING_DEF` with interprocedural taint (§5 rca/code-review, §9).**
Joern's `REACHING_DEF` edges are **intraprocedural** — they connect def→use *within a single method*. Interprocedural
flow (a value entering method A that reaches a sink in method B) is what CPGQL's `reachableBy` reconstructs by
combining `REACHING_DEF` with `CALL`/`ARGUMENT`/parameter linking. The plan frames rca (AC-4) and code-review (AC-7)
as traversal "over `REACHING_DEF` (+`ARGUMENT`)" and acknowledges "static reachability ≠ `reachableBy`," but never
names *this specific* boundary. The risk is in verification: if the §6 AC-7/AC-4 anchors are interprocedural but the
Cypher only walks `REACHING_DEF`, the "clean code returns none" case can pass **as a false negative** — the query
finds nothing because it *can't cross the call boundary*, not because the code is clean. That silently satisfies
AC-7's negative direction while the recipe is actually broken.
*Suggested improvement:* in §5/§9 state the intra/inter-procedural boundary explicitly; have graph-dba decide per
recipe whether traversal crosses call boundaries (argument→`METHOD_PARAMETER_IN`, return→call-site); and require §6
anchors to include **at least one interprocedural case** for rca and code-review so the negative direction is a real
clean-code test, not an artifact of scope.

**M2 — AC-6 is verified by the schema author, which does not test what AC-6 asserts (§6).**
AC-6: "when an agent invokes the `cpg-analysis` skill, then it can run the recipe's Cypher and get correct results
**without hand-knowing the schema**." That is a *skill-usability* claim. §6's strategy has **graph-dba** — who knows
the schema cold and wrote the queries — run them against an oracle. That proves query *correctness* (AC-2/3/4/5/7/8),
but it is the weakest possible test of AC-6, because the one property AC-6 cares about (self-sufficiency for a
schema-naive caller) is exactly what the author cannot falsify.
*Suggested improvement:* add one independent cold-invocation check as the AC-6 gate — a schema-naive agent (analyst
is already the independent reviewer at Gate 2a and does not own the schema) opens only `SKILL.md` + the recipe,
runs it, and confirms a correct answer without consulting `cpg-model.md` beyond the skill's own pointer. Record it as
the AC-6 evidence.

**M3 — The verification substrate must contain the anchors each AC needs; §6 assumes them without a handoff checklist.**
§6 requires "a known input→sink path," "a known-clean function," "a known prod-only method," "a known symbol." The
coordination doc's W1b tasks `joern` to build a "representative" CPG, but the plan hands `joern`/`teco` no explicit
anchor manifest as a *precondition*. If the loaded CPG happens to contain no seeded tainted path, AC-7's positive
direction is unverifiable (or worse, silently skipped) and M2 ships "live-verified" on a vacuous substrate.
*Suggested improvement:* §6 should enumerate the required anchors as an explicit precondition returned to teco/joern
(one tainted src→sink path, one clean method, one prod-only method, one cross-file symbol, one method with known
callers/callees), and W2 verification should assert their presence before running — turning a hidden assumption into
a gated dependency.

### Minor

**m4 — Two properties the recipes lean on are under-documented in the canonical schema (§5, C-201).**
Verified against `cpg-model.md`: `METHOD_FULL_NAME` has **zero** occurrences, and `IS_EXTERNAL` appears **only** as a
boolean example (lines 84–85), not in the node-property enumeration (lines 31–35). The plan's code-review recipe
matches sinks by `METHOD_FULL_NAME` and impact filters on `IS_EXTERNAL = false`. C-201 is chartered to catch "missing
schema facts" and names the `CALL` topology as its example — it should *also* explicitly confirm `METHOD_FULL_NAME`
(on `CALL` nodes) and `IS_EXTERNAL` (on `METHOD` nodes) exist on the loaded graph and add them to `cpg-model.md` if
the recipes cite them (that edit ripples into `joern-cpg` doc-sync — the plan already flags this ripple in §9).

**m5 — `cpg-model.md`'s Call-graph row can mislead; the plan is more accurate and should feed a fix back (§5/§9, C-201).**
The schema doc's table (line 17) reads "Call graph | `CALL` | which method calls which," which sounds like a
method→method edge. The plan correctly treats `CALL` as a call-*site* node (callee via the `CALL` *edge*, caller via
AST/`CONTAINS` containment) and gates the recipe on live-verifying this. This is a genuine schema clarification the
consumers need — exactly C-201's remit. *Suggested improvement:* C-201 add a one-line `CALL`-node-vs-`CALL`-edge note
to `cpg-model.md` so the next author isn't misled.

**m6 — Symbol def/ref (AC-5) matched by `NAME` risks cross-scope false positives (§5 rca).**
The plan delegates "same symbol across files" to graph-dba but doesn't flag that matching a `LOCAL`/`IDENTIFIER` by
`NAME` alone over-returns — two unrelated locals named `x` collide. *Suggested improvement:* recipe should prefer
`FULL_NAME`/scope-aware association (identifier→enclosing `METHOD`/`TYPE_DECL`) and state the collision limitation.

**m7 — AC-3 is interpreted as call-graph reach only; the requirement says "call/**dependency** chains" (§5 impact).**
If "dependency" is meant to include type/inheritance/import edges (not just `CALL`), that surface is uncovered. The
call-graph reading is the standard one and probably intended, but the plan should make it an explicit decision, not a
silent narrowing — see Open Questions.

**m8 — Reconcile the OQ2 decision with the 2026-07-12 "new top-level component" placement decision (§3).**
The requirements decision log (line 137) records "Placement → **new top-level component**." The plan decides *not* to
create a `code-graph/` directory. This is defensible (the 2026-07-12 note was about relocating *docs* and asserting a
distinct *logical* component, not mandating a code dir) and the plan's rationale is sound — but it never cites/reconciles
that line, so it reads as a possible contradiction. *Suggested improvement:* one sentence in §3 stating the component's
identity is carried by repo-root `docs/` + the two `skills/` packages, honoring the 2026-07-12 decision without a dir.

### Nit

**n9 — AGENTS.md's "all 7 skills" is already stale; make C-208 a definite fix.**
Verified: `AGENTS.md:77` says "all 7 skills" but `skills/` already holds **8** folders (the count was not bumped when
`joern-cpg` landed). Adding `cpg-analysis` makes 9. The plan's instinct (§7 step 8: "prefer not to hardcode counts…
update or delete") is right and fixes a latent bug — but phrase it as a definite instruction: **delete the hardcoded
count** in C-208, don't leave it as a preference.

**n10 — The requirements milestone table under-scopes M2's ACs (informational).**
The requirements table (line 24) scopes M2 to "AC-6…AC-8," yet AC-2…AC-5 are phrased against "the impact recipe" /
"the RCA recipe" — artifacts that exist only in M2. The plan correctly maps AC-2…AC-5 to M2 recipes (a strength that
covers a gap in the requirements doc). Worth a one-line correction to the requirements status in C-208's doc-sync so
the two docs agree.

## What's solid

- **Grounding is excellent.** Every cited file/symbol checked out; the schema gotchas (UPPER_CASE keys, lowercase
  `id`, real booleans) are quoted correctly, and the `CALL`-node topology framing is *more accurate than the canonical
  schema doc itself* (m5).
- **Scope discipline is tight.** The out-of-scope list mirrors the requirements verbatim (no runtime coverage, no RAG,
  no falkor-chat wiring); the one-skill shape and graph-dba/cobb/analyst ownership are honored without drift.
- **AC-7's two-direction verification (tainted → non-empty AND clean → empty) is exemplary** — treating the
  false-positive guard as co-equal with the true-positive is the right rigor (subject to M1's interprocedural caveat).
- **Doc-sync / done-conditions match the AGENTS.md same-change rule** (C-208 lands skill + `skills/README.md` +
  `AGENTS.md` + `claude/README.md` + BACKLOG→HISTORY together), and the plan anticipates the skill-count fix.
- **OQ2 is decided with a sound rationale and a well-defined reconsideration trigger** (runnable non-skill code appears).
- **Sequencing follows the backlog critical path** and keeps recipes independently reviewable.

## Open questions (for the caller)

1. **AC-3 scope:** is "call/dependency chains" call-graph reach only, or does it include type/inheritance/import
   dependencies? (m7 — decide before graph-dba fixes the impact recipe's edge set.)
2. **AC-6 gate:** is an independent schema-naive cold invocation acceptable as the AC-6 evidence, run by analyst at
   Gate 2a? (M2 — confirms the falsifiable usability test is in the workflow.)
