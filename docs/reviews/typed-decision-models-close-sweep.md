# typed-decision-models: REVIEW.md close sweep

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (typed-decision-models close, step 4)

**Scope.** Every `### Proposed REVIEW.md lines` block (and the one inline form at `docs/reviews/typed-decision-models.md:2476`) in `docs/reviews/typed-decision-models{,2,3,4,5,6,-impl,4-impl,-ml,-refill-ml,-safeguard-rca}.md` and `decision-models/docs/reviews/*.md`, found by `grep -n -i 'REVIEW\.md'` and an awk extraction of each block, read in full. Compared against root `REVIEW.md` (59 lines, `wc -l`) and `model-bench/REVIEW.md` (21 lines). Both are unmodified in the working tree (`git status --short`). Nothing here edits them.

CPG: not applicable — a review-process document sweep, no code-level component.

## Result in brief

- 12 additions proposed, 0 prunes proposed anywhere (every review that touches the question says "no prune is proposed" or "None").
- 0 are already present in either target: a keyword grep of both files for each proposal's subject (fake clock, `compare_report`, transcript, round trip, field mutation, process tree, harness, byte-copied, single-flight, unattended, fail-closed, absence) returned no match.
- 1 is superseded by a later pass (row 7, by row 8). Live: 11 additions still to apply (row 7 withdrawn).
- 0 are narrowing by effect: all are additions of a check, none removes a check, caps severity, limits scope or relaxes evidence; there is no "Do not report" addition and no prune.

## Table of proposals

"Present?" is whether an equivalent line exists in the target now. The closest existing line is named where it overlaps.

| # | Source (file, pass, finding) | Target | Verbatim proposed text | Present? | Narrowing? |
|---|---|---|---|---|---|
| 1 | `docs/reviews/typed-decision-models-impl.md:1083`, Pass 9, P9-M1 | `model-bench/REVIEW.md`, `## Always check` | "A test of a measured duration (an item clock, a call clock) uses a fake clock that the stubbed calls advance by their own durations, and asserts the exact sum, so a mutant that starts or stops the clock at the wrong point changes the asserted value." | Absent (lines 11-21 cover boundary tests, selection records, load state, canonical vectors only) | No |
| 2 | `docs/reviews/typed-decision-models-impl.md:2273`, Pass 18, P18-M1 | `model-bench/REVIEW.md`, `## Always check` | "A change that routes runs around `compare_report`'s `runs` (as `replicates`, or as any new argument) shows that every guard the `runs` path applies still reaches them: the pack-version and content-hash banners, aggregate consistency, and the schema banner. Each one is pinned by a test on the rerouted runs." | Absent | No |
| 3 | `docs/reviews/typed-decision-models.md:1587`, Pass 9, P9-M3 | root `REVIEW.md`, *Always check* | "A method or plan that runs LLM work as Claude Code subagents names what its transcript audit checks beyond tool calls: the spawn prompt (against a hashed template), `agentType`, and the injected `instructions`/`gitStatus` context. A tool-call-only audit is a finding." | Absent | No |
| 4 | `docs/reviews/typed-decision-models.md:2475-2476`, Pass 50, P50-M1 (and P47-M1's late-failure class) | root `REVIEW.md`, *Always check* | "When a unit's code validates the shape of a record field that a later unit produces, the plan's record row states the shape, and the producer's done-condition feeds its output through that validator (a round trip over every role and call kind). A validator the producer is never run against fails only at the end of a live run." | Absent | No |
| 5 | `docs/reviews/typed-decision-models4-impl.md:1587`, Pass 10, P10-M1 (T1-T5, T24-T26) plus teco's two ledger survivors | root `REVIEW.md`, *Always check* | "For a record another unit consumes (a ledger event, a dataset row, a key binding), mutate each field's value at its write site. A field whose value no test pins, with the expected value computed independently, is a finding." | Absent. Root lines 20-23 (name the mutant, independent expected values) are the general rule; none requires per-field mutation of a consumed record | No |
| 6 | `docs/reviews/typed-decision-models4-impl.md:1844`, Pass 12, P12-m1 and P12-n1 | root `REVIEW.md`, *Always check* | "For code that stops a process tree (a group kill, a grace period, a timeout), plant a descendant that survives the first signal and assert it is gone afterwards; assert the grace by elapsed time. Run timing-dependent tests under CPU oversubscription before calling them stable." | Absent | No |
| 7 | `docs/reviews/typed-decision-models4.md:303`, Pass 1, F2/F4/F5 | root `REVIEW.md`, *Always check* | "A plan step that relies on agent-harness behaviour (frontmatter fields, hook firing, transcript fields, tool size limits) cites a live probe or transcript for each premise. It also names the smoke whose done-condition turns red if the premise is false, sized at the procedure's worst-case input." | Absent. **Superseded by row 8** (same source findings, reworded and extended) | No |
| 8 | `docs/reviews/typed-decision-models6.md:144-147`, Pass 1, restating plan4 review P1 F2/F4/F5, and M3 | root `REVIEW.md`, *Always check* | "A plan step that relies on agent-harness behaviour (frontmatter fields, hook firing, transcript fields, CLI flags, tool size limits) cites a live probe or transcript for each premise. The probe runs under the production invocation's exact flags and environment. Each premise has a named pass/fail route, and a worst-case-sized smoke that goes red if the premise is false." | Absent | No |
| 9 | `docs/reviews/typed-decision-models6.md:144-148`, Pass 1, M2 | root `REVIEW.md`, *Always check* | "A control command that shares a single-flight lock with the long-running process it controls (stop, pause, signal) is checked for reachability while that process holds the lock." | Absent | No |
| 10 | `docs/reviews/typed-decision-models6.md:276-279`, Pass 2, P2-M1 (with Pass 1 M2 and P2-m2 residue) | root `REVIEW.md`, *Always check* | "In a plan for unattended operation, every refusal or halt the plan adds is checked against what an ordinary user action on the same machine changes (config edits, logins, other sessions). Each refusal must also show up in a status command, not only in a run's exit line." | Absent | No |
| 11 | `docs/reviews/typed-decision-models6.md:415-418`, Pass 3, P3-M1 | root `REVIEW.md`, *Always check* | "A fail-closed path is traced to its consumer. The guard's own failure must reach a class that halts and is distinguishable from a policy decision. A fail-closed default that the consumer counts as the actor's fault turns a broken guard into silent attrition." | Absent | No |
| 12 | `docs/reviews/typed-decision-models6.md:415-419`, Pass 3, P3-m1, P3-m2, gate (c) in report §9.1 | root `REVIEW.md`, *Always check* | "A row that keys on a missing record (no hook entry, no result, no field) is checked against every producer of that absence: a kill, batched or parallel writes, and validation before the recorder runs. Checking only the producer that motivated the row is not enough." | Absent | No |

Rows 9-12 are all in `typed-decision-models6.md`, which is the plan6 review: the file's Pass 1/2/3 blocks name root `REVIEW.md`.

## Withdrawn or superseded

- Row 7 is superseded by row 8 (`typed-decision-models6.md:147` says the line was "proposed there and not yet applied" and restates it with the production-flags and pass/fail-route additions). Apply row 8 only.
- `typed-decision-models4-impl.md:1850` (Pass 12) says Pass 10's field-mutation line (row 5) "is still pending feature close; this pass applied it (R1-R9)". It applied the rule in review, not to `REVIEW.md`; row 5 stays live.
- Conditional, not a proposal: `typed-decision-models-impl.md:3294` (Pass 25) would propose "a loader's refusal must be a subset of `validate_pack`'s, pinned by one probe" if a second instance recurred. Pass 26 (`:3464`) saw a related but different second instance and still declined. Not a row.
- Not a `REVIEW.md` line: `typed-decision-models4-impl.md:3520`, Pass 38 "Stopping condition", a "proposed falsifiable rule" for gating plan4. It is a gating procedure for that document, outside the `### Proposed REVIEW.md lines` heading, and its own Pass 39 block (`:3522`) says "None".
- Passes whose block says "None" or "none" (no row): `typed-decision-models-impl.md` Passes 11, 12, 14, 16, 17, 19-24, 25-30; `typed-decision-models4-impl.md` Passes 24, 26-29, 39-44 (block read), plus the inline "Proposed `REVIEW.md` lines: none" lines at `:3237`, `:3276`, `:3307`, `:3349`, `:3412`, `:3439`, `:3480`; `typed-decision-models5.md` Pass 2; `typed-decision-models-refill-ml.md:63`. Pass 19 (`:2415`) says Pass 18's line (row 2) covers its finding.

## Files with no proposal block

`docs/reviews/typed-decision-models-ml.md`, `typed-decision-models-safeguard-rca.md`, `typed-decision-models2.md`, `typed-decision-models3.md`, `decision-models/docs/reviews/typed-decision-models-fit-diagnosis-ml.md`, `typed-decision-models-frontmatter.md`, and `decision-models/docs/reviews/typed-decision-models-ml.md` contain no `REVIEW.md` string at all (`grep -n -i 'REVIEW\.md'` returned nothing for them). The unproposed passes of `typed-decision-models.md` (all but 9 and 50) have no block.

## Caveats for the applier

- The root `REVIEW.md:38-44` rule says a proposed line is added only if it changes what a reviewer does and is not already enforced by a test, CI job or hook. I did not assess each row against that; no row names an enforcing hook. The applier decides.
- Rows 8, 9, 10, 11 and 12 all target the same "Always check" list (root, `REVIEW.md:8-48`); 5 and 6 are also general mutation/test advice and may merit a tighter home. Rows 1 and 2 are model-bench-specific and go to `model-bench/REVIEW.md`.
- Counting method: proposals counted from the awk extraction (195 lines) plus the grep for inline forms; the `4-impl` file has 43 `## Pass` headings, each checked for a block.
