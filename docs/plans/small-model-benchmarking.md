# Small-LLM benchmarking tool (`model-bench/`) — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.27 · **Reviews:** `docs/reviews/small-model-benchmarking.md` · `docs/reviews/small-model-benchmarking-impl.md` · `docs/reviews/small-model-benchmarking-ml.md`

2026-09-10 — v1.27 (M11-2): the plan gate's `## Pass 14` on v1.26 (`docs/reviews/small-model-benchmarking.md`, `2f5406b`) closed in full — **1 blocker, 2 majors, 3 minors, none carried** — together with **two** `-ml` note revisions folded in at once, **v1.20**'s §11.9 ask 7 and **v1.21**'s §4.3.1 (`998d13b`), and one contradiction between two of this plan's own sections that no gate pass reached. **P14-1 (blocker)** — §3.8.4's four-row table asserted column 4 (what scores a turn) as a function of column 1 (the mechanism), and it is not one: a turn emitting only undispatchable tool calls reaches the cap with `|E(t)| = 0`, which is §4.2(a)'s `no_attempt` — a **failure** — and is outside both denominators the `cap-hit` row routed it into. **The gate's two-row split is adopted in substance and refused in form: the column is deleted, not split.** `-ml` §4.3's funnel has routed these turns since v1.1 and §4.3 rule 4 now states the mapping outright, so the table was a *second home* for a mapping that already had one — which is the same defect class one level up, and §7 rule 2 gives *which denominator a mechanism lands in* to the note. Column 4 now carries only what the plan owns (the record fields the mechanism sets) and cites rule 4 for the rest; the *"only home of that mapping"* sentence is deleted rather than corrected. The gate's own fix sentence needed a correction as it landed and gets it: `iteration_cap_hit_rate` is not "the one count keyed on the disposition alone" — there are **three**, on **two different subsets**. **P14-2 (major) is closed on arrival**, verified against the tree rather than taken: `LMStudioCallFailed.status` shipped at `d5b549d` as a **required keyword argument with no default**, with twelve raise sites audited and a completeness pin that computes **both** sides (`test_every_raise_site_of_lmstudio_call_failed_is_covered_here`) — strictly stronger than the prescribed grep-with-a-count, and the reason the finding's own arithmetic slip (eleven named, twelve enumerated, thirteen matched by the grep) could not ship. §3.8.4's *"one-line adapter change"* is replaced by what was delivered. **P14-3 (major)**: §5 test 10c said three legs land now while §4 S2's stage row said two, and §4 S5's *Done when* never named the third — all three swept, and the third leg plus deleting its shipped tripwire is now **§4 S5 *Done when* item 1**, gated rather than referenced. The precursor's rationale is restated as **cross-unit** protection: v1.26's *"two sets authored in one unit agree by construction"* over-reached, since the transcript is authored there too. **P14-4 (minor)**: the `no-response` row leads with the discriminator — *the server did not answer, or answered unusably; no HTTP status is carried* — rather than a paraphrase four status-less sites contradict. **P14-5 (minor)**: the exception path is tested before the cap, so a call raising at the cap-th iteration is never `cap-hit`; stated under the table and added to test 10c. **P14-6 (minor)**: `iterations == len(chatResults)`, stated as the consequence of a pin the note already carries; its second half was fenced to `data-scientist` and is **now ruled** (§4.2(f): the `I(t)` summary is over `{replied, cap-hit}` and no others), so the seam closes as a citation. **The disposition set widens to five, `timed-out` split out of `no-response`** — reached here from three consumers (the scorer's `fail`/`unrunnable` split, `censoringExact`'s per-item distinction, and §3.6's re-probe asymmetry) and independently in `-ml` §4.3.1 item 2; the four-members-plus-`withheldFor` alternative is recorded as the two-vocabularies collision and refused on that ground. **`TurnDisposition`/`TURN_DISPOSITIONS` and their probe already shipped at `d5b549d` holding four**, so this is an edit to guarded code: legs 1 and 2 go red until the transcribed constant moves with them, **which is the guard working and not a regression**, and the plan says so where the rework unit will read it. **The contradiction, and it is the plan's**: §3.6's fourth disposition scored a non-2xx `fail` while §3.8.4 and §4 S5 scored the same turn `unrunnable`. Decided against §3.6's outcome clause on `-ml` §4.3 rule 4's discriminator — a turn scores `fail` only where the harness gave the model its **whole declared budget** and observed nothing come back, which is the timeout and nothing else; every other non-completion is an unattributable channel failure. **Judged, not adopted**: the objection §3.6 raised is real about a rate whose denominator silently shrinks, and §4.3 rules 1–3 are what stop that, with the one remaining escape closed by **`cleanThroughTurnH`'s third state** (an `unrunnable` turn at `t ≤ H` leaves the headline's denominator, never reads as clean). Everything P4-7 bought is untouched and said to be untouched. **Ask 7's twelve items all land** — §3.3 (`callCount` as a consequence, no manifest key), §3.5 (`callCount` column), §3.6 (five FR-11 rows moved to their real unit, one added, and a withholding bullet kept **deliberately separate** from the outcome clause above it), §3.8.4 (the runner's `ItemTiming`), §4 S1 (`ItemTiming` **smaller**, the new `CallTiming`, `unexplainedMs`/`callCount` derived), §4 S2 (`LatencyBlock` gains `callCount`, `statsCoveredCount` and (iv-c) move to the **call**, (iv-b)'s `Y` is `callCount` **not** `latencyItemCount`, `unexplainedMsMax` split out of rule (i), and (ii)/(iii)/(v)/(vi) stated as unmoved), §5 tests 10b and 15b, Appendix A, and the `lmstudio.py` docstring sweep pinned by symbol and a grep count. Ask 4's correction is swept with them: **`latencyMsMax` is `None` on every declared pack**, so v1.9's justification is withdrawn at §3.5 **and** §6 R-13 rather than softened. **Two §7 rule 3 raises opened** (R-1, the netted `Y_calls`; R-2, a drifted line cite), both consequences of rulings this revision adopts, neither blocking anything here.

2026-09-09 — v1.26 (M11-2): the plan gate's `## Pass 13` on v1.25 (`docs/reviews/small-model-benchmarking.md`, `05e449d`) closed except for its second blocker, which is routed and not mine — **1 blocker, 4 majors, 4 minors closed; P13-2 (`ItemTiming`/`unexplainedMs` under the loop) is `data-scientist`'s and this revision touches no latency, timing or withholding statement**, carrying one pointer so a rework unit does not build across the open seam. Item 1's ruling is unchanged: the gate judged it over-determined with reason 3 decisive alone. **P13-1 (blocker)** — `finalReplyText is None` **iff** `capHit` was false in the ← direction, and §4 S5's *absent-not-failed* rule keyed on it, laundering a §3.6 `fail` into an `n_a`. **The prescription is adopted — split the disposition off the field, never widen the `iff`** — with two corrections the gate's own set does not survive: the fourth member cannot be `-ml` §4.1's scoring word `unrunnable`, because `TurnTrace` records a **mechanism** and the count is the scorer's, the same two-vocabularies collision `BinaryMetric.unit`/`analysisUnit` and `unit_kind`/`analysis_unit_field` have already cost this plan twice, so it is `server-rejected` with the mapping to `-ml` §4.1's count written out; and the partition is **not derivable at `40a9bc8`**, since an HTTP 400 and a dropped connection both raise `LMStudioCallFailed` (`lmstudio.py:485,495`), so `LMStudioCallFailed` gains `status: int | None` and that is what separates them. The guard is honoured structurally: `TURN_DISPOSITIONS` and its **three-way** probe — the `Literal`'s members, the scorer's branch set, and a constant transcribed from this plan, each compared against the transcribed constant rather than against each other — land in a **precursor unit before** the rework unit, because a coverage probe authored beside the value it must reject contains it from birth. **P13-3** — `structured-replies-only` reproduces the executor's *replay policy*, not its shape; the speaker-name prefix, the per-assembly `CONTEXT:\n<json>` block and `_append_turn`'s same-role merging are named, judged, and added to R-3's candidate causes (verified at `executor.py:1243-1277`, read here rather than taken from the review). **P13-4** — the breadcrumb equation is deleted from §3.8.4 and R-3: `structured`'s native scaffolding is not U37/U38/U39's free-text suffix, and that suffix did **not** suppress onset — it was reverted as a severity increase (`salesperson-tool-reliability-impl.md` MAJOR 1, and the executor's own docstring). **P13-5** — the emission form is `-ml` §4.2's predicate over the turn's dispatch trace, never iteration 1, which diverges on an all-undispatchable first iteration; test 10b's clause pinned the divergence and is rewritten. **P13-6** — `maxIterationsPerTurn` is required **iff** the role runs a multi-call turn, held as a third role-table column with the same computed completeness assertion route (iii) uses, and forbidden elsewhere so a dead knob cannot accrete inside four content hashes. **P13-7** — claim 1's conclusion stands and its justification is narrowed: the basis was measured with the loop running, so the loop is not a *new* cost, but §4.5.2's minutes are a **floor** whose multiplier is bounded by `maxIterationsPerTurn`; the restatement of the figure is routed to `data-scientist`, the sizing decision is not reopened. **P13-8** — the enum spans **two** axes, role ownership and tool evidence, not one ladder. **P13-9** — a cap-hit prior turn under `structured` contributes its iterations and no trailing assistant message. **P13-10** — item 3 gains the U76 clause; item 2's provenance sentence is corrected to the shape that **already exists** rather than put in the imperative, since the gate read *"zero `_provenance` keys"* as per-entry keys while the fixture carries one `_provenance.perEntry` map citing all seven ids by name (parsed at `40a9bc8`) — what was missing was the assertion, which is now specified.

2026-09-09 — v1.25 (M11-2): five rulings routed to `architect`, four of them plan-semantics questions no implementer owned. **Item 1 — a prior turn is replayed from what the model *actually* produced, and U78's scripted-`expect` replay (`convo.py` at `40a9bc8`) is invalidated.** The clause it cited — §3.8.4's *"the harness never carries hidden state between turns beyond what the configuration says it carries"* — constrains the replay's **shape** and forbids **undeclared** state; no statelessness rule was ever written here, pinned rather than asserted — `git show 40a9bc8:docs/plans/small-model-benchmarking.md | grep -i -c stateless` → **0**, at the tip U78 built against, and pinned to that sha because this note and §3.8.4 now discuss the absent rule by name, which no live grep can distinguish from the rule itself — and a prior turn's real output replayed under a declared `historyReplay` is neither hidden nor undeclared, so the clause never decided whose content is replayed. What decides it is the prior art: `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §4.1 attributes the documented turn-4 collapse to **the model's own prior turns, replayed as ordinary chat with no visible evidence a tool was ever used** — a precedent a textbook replay makes identical for every conversation and every model, so §3.8.4's validation target and §5 test 19(b) stop measuring the documented phenomenon: §8.2's persistence half (zero recoveries in 121 post-onset turns) is unreproducible by construction, and its onset half survives or not according to which `historyReplay` value the pack happens to declare. **The claim is deliberately narrower than *the collapse cannot happen*** — a textbook prefix in a reply-text mode presents the same stimulus shape, so onset may well fire; what it would not be is evidence. Three further consequences, each independent of that one: the per-turn hazard's *clean through t−1* conditioning exists to hold accumulated contamination out of the denominator and has nothing to condition away once every turn sees the same textbook prefix, so the statistic stops separating the two hypotheses FR-9 names; `drive` dispatches real calls into a stateful `ToolEnvironment`, so a scripted context and the FR-10 ground-truth state disagree from the first failed turn onward, manufacturing failures that are the harness's; and U78's determinism-probe argument inverts — `-ml` §4.5.1(iii)'s probe object is the **conversation-level observation** that enters `n`, so a re-run that diverges mid-conversation is exactly the flakiness the probe must report, and scripted replay would return `identical` on a run whose observation did not reproduce, buying `basis: "by-construction"` for a design effect the evidence does not support. **What follows, all built from the plan alone:** §3.3's `historyReplay` gains a fourth value, `structured-replies-only` — native roles, the model's real final reply text, no tool scaffolding — because that is falkor-chat's executor shape and no existing value reaches it, which made §3.3's own *"reproducing falkor-chat's executor shape is one pack's settings"* false; `tool-caller-shop-assistant` declares it, and R-3's bisect ladder is now the enum; and `drive` gains the bounded per-turn iteration loop `-ml` §4.1's `I(t)` and §4.2(f)/(g) already presuppose, since one call per turn leaves a tool-calling turn with no final reply to replay or to score containment against. No conversation datum is re-authored and the 2026-09-02 sizing decision is untouched — §4.5.2's ~1.3 s/turn basis was measured on falkor-chat's own multi-step executor and already includes the loop. **Item 2 (F-S2-1)** — §4 S2's done-condition cited *"§2.5's captured 19-model response"* and no such artifact exists; §2.5 is a narrative probe record and now says so, the done-condition names the real, per-entry-provenanced `tests/fixtures/lmstudio/catalog.json`, and the verbatim capture rides the live session R-1 already requires. **Item 3** — §3.3's role-specific half is mechanised as `check_sampling_contract`'s **third** route against a role→analysis-unit-field table in `roles.py`, the existing closed-role home; P12-7's fixture passed both existing routes while violating the rule, which is what proves two routes were not two. **Item 4 (P13-7)** — *degenerates to `residencySource` alone* means **compare it**: outcome and result are different columns, `"unavailable"` names the comparands that were missing and `stale` carries the one that was not. **Item 5 (P13-9)** — §4 S2's `warm_up` sketch gains `was_resident_before`.

2026-09-09 — v1.24 (M11-1): §4 S1e Table H's landed `:444` row is corrected to write out the expression `f17efa2` actually shipped — the row had sketched `bound_by` as re-testing `u_lo`/`u_hi` against the support a second time, spelling `SUPPORT_DIFF_PROPORTIONS[0]`/`[1]` twice each against residuals 2 and 3's stated target of 1 apiece; it now names `lo`/`hi` from the clamp itself and derives `bound_by` from `lo != u_lo` / `hi != u_hi`, with the `max`/`min` identity stated as the reason that is Rule 4a's strict comparison and not a shortcut around it. Verified against the current tree: both residuals read 1. No residual, target, count or enumerating command moves, and Table H's `Landed:` commit stays `f17efa2`, per the landed-table convention. §7 rule 5(b) needed no new clause: the ninth instance of the specification-is-the-defective-party shape, and the rule's existing third-form clause already covers it — this row was written before it, not around it.

2026-09-08 — v1.23: the plan gate's `## Pass 12` (`docs/reviews/small-model-benchmarking.md`, `5a018a4`) closed in full — 0 blockers, 3 majors, 2 minors, 1 nit, none carried — in the same revision as the `e162ba9` re-baseline: **P12-1** returns **Table F's third residual to package-wide**, the one of the nine v1.22 narrowed whose reach *was* the check, since its `→ 0` is the `DistributionSummary` tag set's one-home claim and reads 0 file-scoped with a second home standing in `report.py`; **P12-2** bounds the scoping default — the enumerating commands are the completeness instrument, a residual's scope is its edit's, and a one-home or identity residual's scope is its purpose's; **P12-3** replaces that default's prospective-binding premise with convention 1's record argument, impl-gate F2 and F3 being landed residuals that did not hit their targets; **P12-4** retires *the narrowing costs no evidence* (a claim about *befores*, vacuous for a 0 → 0 pair) for the two-file probe and the 31-line-insertion survival, and states what the narrowing does trade; **P12-5** and **P12-6** give the scope claim and the *never re-widened* reversal one home each, across two and four sites. **Re-baseline to `e162ba9`** (the N4 fix; `stats.py` 1 418 → 1 449, 577 green): **Tables H and F** re-point, the **six** landed tables do not, Table H's commands move `bound_by` 14 → 15 and `envelope_arms` 21 → 22 and its five `stats.py` pins by 31, Table F lands on one commit at last, and **all eleven residuals re-run unchanged** — every `stats.py` line pin below the insertion broke and no exact-text residual did, which §7 rule 5(b) now records as that form's largest test.

2026-09-08 — v1.22: the plan gate's `## Pass 11` (`docs/reviews/small-model-benchmarking.md`, `68b0d14`) closed in full — 0 blockers, 1 major, 2 minors, none carried. **P11-1:** v1.20's repair of residuals 2 and 3 removed the introduced local and thereby widened their match to any subscript of `SUPPORT_DIFF_PROPORTIONS` **anywhere in the package** — in the same revision that mandated a second consumer of it, since P10-2 requires `report.py` to reach the constant and `-F` matches through the `stats.` qualifier, making the count **2** against a target of **1** on the explicit-branch rendering. Both are re-scoped to `modelbench/stats.py`, as Table C's first two are scoped a file each. Generalised at **§7 rule 5(b): a residual is scoped to the files its own site rows name; a wider scope is an exception the table states** — the reason being that **a pattern's specificity and its scope's breadth are substitutes**, so shortening a pattern obliges narrowing its scope. Written as a **default rather than a caution**, because this is the seventh residual to fail on a faithful edit and the first introduced by the fix for the sixth; it binds **prospectively**, so the two unlanded tables — **H and F** — are scoped to their rows' files in this revision, all nine reading the same *before* values under either scope. **P11-2:** Table H's residual block re-points to `93b0e42`, ending a two-baselined table that DC-12's own re-run claim had already contradicted. **P11-3:** the gloss names its four sites by **test-function name**, because this table inserts Rule 4a's assertions into that very file between the second and fourth of them — and convention 2's rationale is corrected with it: a site list is stable in a site's **identity**, not in its **line number**. Also placed: DC-12's gap paragraph is dated by design, and the mechanism that makes that safe — the round-end re-run — is now stated.

2026-09-08 — v1.21: **a mechanical re-baseline, plus the two conventions it forced.** `93b0e42` (the Pass 9 fix round) landed under v1.20 while it was being written and touched `stats.py` and `tests/test_stats.py`, staling exactly three things: DC-12's *byte-identical from `7f865e2` through HEAD* clause, §4 S1e Table H's second enumerating command (`envelope_arms` **20 → 21**, `tests/test_stats.py` 15 → 16), and P10-5's prose decomposition of that count. All three corrected; **nothing else moved** — Table H's six residuals, its first enumerating command (`bound_by` 14) and all ten `stats.py` lines pinned by Tables E, G and H are byte-identical across `93b0e42`, confirmed. **Two conventions now written into §7 rule 5, because this will recur when Table F's unit moves `results.py` and `report.py`.** First: **a stated baseline moves only when the revision moving it re-runs the commands**, never because another unit landed — a named commit is a property of the measurement, so re-pointing without re-running asserts a count nobody took; a **landed** table's baseline never moves (re-pointing a record falsifies it), an **unlanded** one's moves in the revision that re-measures it, and the gap between a table's baseline and the tree is carried once, in DC-12. Table H accordingly re-points to `93b0e42`; the seven landed tables do not. Second: **a gloss beside a count names *sites* and does not restate a total** — the gloss is not forbidden and is what rule 5(a) asks for, but a restated total is a second copy of the command's own number (§7 rule 4's one-home rule, applied to a number), and Table H's `envelope_arms` gloss failed twice in one day by two different mechanisms while the command was right all three times. That gloss is rewritten to name its four call sites and state no total.

2026-09-08 — v1.20: the plan gate's `## Pass 10` (`docs/reviews/small-model-benchmarking.md`, `913e159`) closed in full on §4 S1e **Table H** — 1 blocker, 2 majors, 1 minor, 1 nit, none carried, all five prose rather than arithmetic. **P10-1 (blocker):** the site list omitted the one **shipped test** the edit falsifies — `test_neither_printed_bound_is_ever_tighter_than_either_arm` compares the printed envelope against the arms **directly**, which is false once the arms come back unclamped (measured **0/38/78** failures at DEFF 1.0/1.2/2.0 against a 0/0/0 control) — so it gains a row, **pinned by test name and not by line** because a parallel unit is editing that file, moving the comparison to the **clamped** arms, which is Rule 4a's own restatement and not a weakening; command 2 already returned the line, so the enumeration was sound and only the hand-written row list was short. **P10-3:** the `stats.py:413` row claimed a prescribed post-edit spelling and never stated it — the only statement was inside residuals 2 and 3, which pinned two **locals the plan never fixed** while Rule 4a calls the same quantities `u_lo`/`u_hi`, so a faithful edit would have read 0 against a target of 1; the row now **writes the body out** and the residuals are restated over `SUPPORT_DIFF_PROPORTIONS[0]`/`[1]`, which depend on no introduced name. Generalised at **§7 rule 5(b): a third-form residual is only a residual if the table writes out the text it pins.** **P10-2:** a `support bound` token renders **with its boundary value** — `support bound (-1)`, the note's assertion 10 verbatim — not bare, and the renderer's source for that value is named as `SUPPORT_DIFF_PROPORTIONS` rather than left to become a second home for the support in `report.py`. **P10-4:** the docstring row extends from one paragraph to **three**, and v1.19's claim that the edit makes `envelope_arms`'s docstring sentence *true* is corrected — one clause becomes true and the other becomes false. **P10-5:** the fifteen-of-twenty sentence is re-derived — fifteen is the file's whole share and **four** are arm-value call sites. Table H is ten site rows; residuals stay six and DC-12 twenty-four.

2026-09-08 — v1.19: **note v1.19's §3.4 Rule 4a folded in as §4 S1e Table H** — the `(-1, 1)` support moves off the envelope's arms onto the composed interval, `bound_by` gains a third token on a strict comparison, and the `- decided by:` renderer stops attaching a level to a bound no level produced. Eighth table; two enumerating commands, nine site rows, **six** residuals, all counts at `7f865e2`. Closes impl-gate P8-1, with P8-5 as collateral. **Table E does not move**: its pair pins `_widen`'s body, which Rule 4a does not touch — checked, not assumed, and both read 1. §4 S1e is eight tables and DC-12 twenty-four residuals; both standing sweeps re-run over the six. §7 rule 5(b) gains the exact-text-over-line-pin virtue, now paid for twice. One §7 rule 3 raise opened: Rule 4a cites *plan-gate* P8-1 where it means *impl-gate* P8-1.

2026-09-08 — v1.18: **§7 rule 5 gains one clause — a residual must be portable.** These commands are re-run by different people in different shells, so a residual whose correctness turns on regex dialect is not a residual: prefer a fixed string, and verify any new *regex* residual under more than one `grep` before writing it down. The occasion is v1.17's `\bbootstrap_seed`, this plan's first word-boundary residual, checked under two implementations that agreed — nothing is owed retroactively. No table, target or count changes.

2026-09-08 — v1.17: the `stats.py` unit (`cc28d48`, §4 S1e Tables C, D, E and G) landed correct and raised **three findings against the residual *statements*, none against the code**, all three closed here, none carried: **F1 — Table E's two residuals go blind on the very half-application they exist to catch**, because they match the **shipped** text (`max(-1.0, point …`) while a faithful edit *parameterises the clamp* and therefore rewrites the expression the literal was fused into, so a half-application in the new spelling (`clamp[0]` wired, upper bound left as the literal) matches **neither** — reproduced here on the three states side by side, the pair reading 1/1 before, 0/0 after, and **0/0 on the half-application, indistinguishable from faithful**; that is a shape §7 rule 5(b) did not cover — a residual over text the edit **destroys**, not over a token that survives it — so the rule gains the **third-form residual**, stated over the text the edit **creates**, with the trigger named (**a parameterising edit**, the one kind that changes an expression's shape rather than a name or a value) and Table G kept as the contrast that shows the trigger is real: its literals sit at a call site the edit does not restructure, and its half-application **is** caught, verified by simulation; Table E's pair is **replaced** rather than supplemented, since rule 5(b) already forbids keeping a check that reads clean on a defect; **the standing sweep was then re-run over all eighteen residuals for this shape specifically** — the discriminator being *does the matched span include text the table's own edit rewrites* — and it partitions **16 robust / 2 fragile (E's pair, now closed) / 1 near-miss examined and cleared with its cover named** (Table B's `frozenset(FORBIDDEN`, whose only blind spot is an unprescribed respelling of the coupling, killed loudly by `ARM_KINDS`'s by-value pin); **F2 — Table C's third residual moves 1 → 2** the moment Table D lands, `-ml` §3.4 Rule 4's mandated `exact_paired_quantiles` matching its own pattern — a round-level interaction between two tables in one unit, not a defect, and **not** to be answered by renaming the function, since that residual's stated virtue is surviving a rename — so the target is restated with **both survivors named**, the plan says plainly that the **count no longer discriminates and the named line set is the check**, and rule 5(b) gains that clause too, `exact_paired_quantiles` being the same operator on the exact multinomial resample distribution rather than a second sample-quantile estimator (`-ml` §11.2 reason 2); and **F3 — Table D's first residual could never reach 0**, its target having been unreachable from the moment it was written: `def test_cluster_bootstrap_seed_…`, a **substring collision** (`cluster_bootstrap` + `_seed`) on a test for the function Table D explicitly **keeps**, was among the 29 lines the table counted at `5878014` — re-derived here, the collision is **two** lines and not one — so the residual becomes the whole-identifier form, **27 → 0**, with the two `def`-name lines named as non-sites and the arithmetic written out. Also: Tables C, D, E and G gain their **`Landed:`** lines (`cc28d48`), and v1.16's narrowing of Table C's third residual is **vindicated by the landed tree** — the seven `def test_*percentile*` names `cc28d48` added would have put the unnarrowed command at 9 against a target of 1.

2026-09-07 — v1.16: the implementation gate's `## Pass 5` (`docs/reviews/small-model-benchmarking-impl.md`, `cbfcda9`) closed on its three plan-side findings, none carried: **DC-1's residency-element rule is restated over the element's *key set*** rather than as a pair of refused names, and the retired `lms ps --json` element becomes an assertion the suite makes **by value** with both of its keys named — because impl-gate P5-3 built the counter-implementation whose extra-key rule carries a one-name tolerance and passes all 472 tests, and no test could close it while **§4 S1e Table A's second residual** counted that name across `tests/` as a whole: a residual counts the lines that *name* a token and cannot tell a live use from a mention that disowns it, so that residual is **re-scoped to `modelbench` plus `tests/conftest.py`** — its *before* value of **1** unchanged, its sole pre-edit line being the fixture element the row is about — and the assertion's literal is prescribed into `tests/test_fingerprint.py`, with **DC-12's note rewritten over both halves**, having reasoned the `modelKey` half correctly one paragraph away and never noticed that the other key's residual forbade the assertion it routed to; **§7 rule 5(b) gains the *disowning mention*** as a named trap shape, and DC-12 a standing sweep for it over the residual set, which found the **third** instance of the shape in the table the next unit implements — **Table C's package-wide percentile-definition check matches a test *function name*** (`def test_percentile_…`, verified by construction), so its stated target of 1 fails on a faithful edit that names `-ml` §11.10's acceptance tests naturally, and it is re-scoped to `modelbench`, which is what *package-wide* always meant; **Table C's `tests/test_results.py` row is re-pinned `:507` → `:543`** and every other Table C pin re-verified against the current tree, `results.py`/`stats.py`/`report.py` being byte-identical to `5878014` (impl-gate P5-6); **Appendix A's `unknown` becomes a *value* this build cannot interpret** rather than a discriminator, its three families written out (impl-gate P5-5); and §4 S1e gains a **`Landed:`** convention, so a table whose rows are now a record of completed work cannot be read as an instruction to apply it again — **Tables A and B carry the first two** (`8fc2341`).

2026-09-07 — v1.15: the plan gate's `## Pass 8 (narrow)` (`docs/reviews/small-model-benchmarking.md`, `4cd22b9`) closed in full — one blocker, two majors, none carried — and note **v1.18** (`bbbf18e`) folded in, the plan re-paired to it: **§4 S1e Tables C and G stop colliding on `stats.py:159`**, the line Table C moved to an integer-`permille` primitive while Table G required a family-derived level that is fractional for every `k ≥ 2` — the note **refuses** the exemption the gate offered (§11.2.2(3): what that line would keep is not a different *unit* but the estimator §11.2 rejects, at the one call site in the package whose level is not a literal), so the level becomes an **exact rational** — `percentile(values, *, level: Fraction)` with four `LEVEL_*` constants, `levels: tuple[Fraction, Fraction]` on both bootstraps — the collision is **named on both tables' rows** the way Tables D and E name each other on `stats.py:263`, with the order **fixed** (C before G on that line, neither order being faithful), and **Table G's two residuals are re-derived over `:159`'s post-Table-C spelling**, the pre-C pair having been driven to zero by Table C's edit alone — so DC-12 would have passed on an implementation that never applied Table G, which is the trap §7 rule 5(b) forbids sitting inside the plan that wrote the rule (plan-gate P8-1); **Table C's residual gains the `stats.py` half it never had** — its edit retires `_percentile` at **six** production sites across two files while its residual counted three in one, so the half-application §11.10(3) actually obliges left both bootstraps on the rejected estimator with the number reading zero — plus `-ml` §11.10(3)'s package-wide identity check as the residual no half-application passes (**2 → 1**, a stated target that is not zero, with the surviving line named) and a row for the seventh line its enumerating command returns, DC-12's *already satisfies* partition corrected from three sites to six (P8-3); and **§3.3 (iv)'s "nothing else in the verdict path changes" is replaced by an enumeration over the eleven emission sites of `report.py:606-780`** — the `Family-wise error control` block renders only where a Holm ladder actually ran, with a one-line replacement for each of the two conditions that reach it (a family refused whole, and an all-continuous `k > 1` family taking its correction in the interval), which makes `_decision`'s *no verdict — no paired data* **unreachable** for a refused member rather than relabelled, so no third input state is added and the trigger that would reverse that is stated (P8-2).

2026-09-07 — v1.14: the plan gate's `## Pass 7` (`docs/reviews/small-model-benchmarking.md`, `b6222c6`) closed in full — one blocker, two minors, none carried — and note **v1.17** (`1fbdb6f`) folded in, the plan re-paired to it: **§4 S1e Table F decides `DistributionSummary`'s stored form** — the `"distribution"` tag, its six keys, `support` stored as a two-element array or `null` on both continuous types and read with **no `.get` fallback**, an unrecognised tag **raising** where a third shape silently returned a raw `dict`, and `benchSchemaVersion` **not** bumping with the trigger that would reverse that stated — together with the three shipped sites v1.13's six commands could not see (`results.py:354-359`, `:385`, `:584`), two commands that do see them, a corrected sixth, a third residual and DC-13(f) (plan-gate P7-1); **§7 rule 5(a) gains its attribute-based companion**, because a retype's site list is found by the **attribute the new type lacks** and not by the type's name — the fifth instance of *shipped code correct for its current caller and wrong for a caller this plan commits to adding*, and the first that the mechanism built to catch that class missed; **Table G's residual becomes symmetric** and its new test asserts that **both bounds move outward**, the width claim being satisfied by exactly the half-applied edit the one-sided residual missed — and **Table E, the one other table that retires two literals under one residual, gains the same second residual**, its half-application having left `_widen`'s upper clamp in place and shipped Table E's own defect intact (P7-2, generalised at §7 rule 5(b) and asserted per table at DC-12); **§3.3 (iv) names the site where a refused family's `exploratory` label prints** — `report.py:767`'s family filter, which structurally excludes the members that need it (P7-3); and note **v1.17**'s two plan-side deltas land — Rule 8 now **takes `support`** and derives the clamp inside itself, so §4 S1's loop **forwards `ContinuousMetric.support` and derives nothing** and v1.13's §7 rule 3 raise is closed, while §4 S1e Table E's engine `clamp` keeps its work on the one caller that still states it directly (§3.8.1's exploratory `sep_z` comparison — two callers, two surfaces), and the non-blocking classification is re-rested on `_widen`'s **scale-1.0 identity**, which holds for every metric, rather than on the pack census v1.13 used, which goes stale without anyone editing the sentence resting on it.

2026-09-07 — v1.13: **not a gate revision** — note **v1.16** (`e290148`) folded in and the plan re-paired to it, four deltas and their sweeps: §3.3 (iv)'s mixed-family refusal changes **granularity**, from excluding and naming the offending members to **refusing the whole family's verdicts** — `k` is `len(verdictMetrics)` and pre-registered, so dropping the minority kind shrinks it after the results exist and hands the survivors a weaker correction than the one declared — with the enforcement *point* v1.12 chose (`compare_report` pass 1) confirmed and the `validate` raise closed (`-ml` v1.16 §3.3); §4 S1's continuous producer stops being described and is **cited** — `-ml` §3.4 **Rule 8**'s `continuous_verdict()`, its `ContinuousVerdict` sibling return and the four parameters it deliberately does not take, with the one thing the note leaves here named as this plan's seam (`report.py` renders a `Verdict | ContinuousVerdict` union); new **§4 S1e Table G** lands Rule 8's other engine precondition — `paired_bootstrap`/`paired_cluster_bootstrap`'s quantile levels become required parameters, since a `k > 1` continuous family takes its correction in the interval and has nowhere else to put it — taking §4 S1e to **seven** tables and DC-12 to **thirteen** residuals; and §3.8.1 records `sep_raw`'s three published figures, **no mean for either quantity**, and the new **prohibition on printing any cross-model `sep_raw` difference** (`-ml` v1.16 §5.2). Also closed: §4 S1e's `-ml` §3.2f trip-hazard raise, by the note's own sweep of every site that named the retired instrument. One new §7 rule 3 raise, non-blocking and bounded: Rule 8's parameter list carries no `support`, so `continuous_verdict()` has nothing to forward as Table E's required `clamp` (at the embedder's `designEffect == 1.00` — it being the only pack in §3.8 with a continuous verdict metric — `clamp=None` and `(-1.0, 1.0)` return the same interval, so nothing built today turns on it).

2026-09-07 — v1.12: the plan gate's `## Pass 6` (`docs/reviews/small-model-benchmarking.md`, `afca8e0`) closed in full — one blocker, two majors, two minors, none carried — and note **v1.15** (`0ad0e7a`) folded in, the plan re-paired to it: the embedder pack's only verdict metric gains a **representable carrier**, because nothing on the record could hold a per-item continuous value and the shipped comparison path failed silently in two directions — `ItemResult.measures: Mapping[str, float]` beside `counts`, `scored_value` beside `scored_outcome`, and `scored_outcome` **raising** on a metric that lives in `measures` rather than booleanising it — landing as **§4 S1e Table F**, an **S1** edit on S1e's own *free only now* argument, with `report.py`'s continuous verdict path, `ContinuousMetric.support`, the new `DistributionSummary` carrier for `sep_z` and DC-10's third arithmetic beside it (plan-gate P6-1; note v1.15 §3.2d, §3.2e's two new strings, §3.3's homogeneous-family rule); §4 S1e **Table B's fourth residual is narrowed** to the mapping key set it actually means, because the wider form fails on the `fingerprint.py:137` row's own literal branch — the trap §7 rule 5(b) forbids, written into the table that closed the finding about it (P6-2); **Table E is re-enumerated** over a second command, its four name-matched non-sites named as non-sites, its three real call sites given the clamp each passes, and Table D's row scoped to *not touched **by this table*** (P6-3); §4 S1e's preamble scopes its tables to **retiring and re-keying** edits and names DC-11's `latencyMs` change as the type-system-enforced remainder (P6-4); Table B's second residual becomes the command it already was (P6-5); and, **reversing v1.11**, §4 S3 done-condition 2 no longer gates on Table E — `separationZ` is reported rather than verdicted, so the clamp is due with the `sep_z` comparison and not with S3 (note v1.15 §3.2d) — while S3's meaningful gate becomes **Table F**, without which S3 done-condition 1 cannot store a result at all.

2026-09-07 — v1.11: the plan gate's `## Pass 5` (`docs/reviews/small-model-benchmarking.md`, `b9964d1`) closed in full — all ten findings, none carried — and note **v1.14** (`ca69cb1`) folded in, the plan re-paired to it: §3.4.4a step 3a and §3.6 **scope the two pre-load refusals separately**, the `callSurface`-versus-catalog-`type` cross-check on every `model` run and the tool-calling gate on a `tool-caller` pack alone, because v1.10's unscoped sentence refused every `model:embeddings` arm and made S3 unrunnable (plan-gate P5-1); §7 **rule 5's property is restated honestly** — a *no-forgetting guarantee over token-carrying sites*, not completeness — with a second named command required for every token-free site and a **second-form residual** wherever the token survives, which gives §4 S1e Table B three more commands, eleven site rows and four residuals, and reduces DC-12 to the direction a residual actually proves (P5-2); §3.4.2's capture-ordering paragraph is rewritten to **cite** §3.4.4a rather than contradict it (P5-4); §4 S2 gains rule **(iv-c)**, co-presence as a runtime **disposition that raises nothing**, with rule (iv)'s identity becoming an inequality (P5-5, note §11.4) and rule (i) scoped to the wall-clock figures (P5-7); `ItemTiming.withheldFor` gains a third value, **`"timeout"`**, so `-ml` §11.5.1's `censoringExact` is evaluable, while the counter stays one (P5-6, note §11.5.1); §3.6's unit boundary states that a derived field is `None`-never-`0` and **`ChatResult` never raises on a missing `stats`** (P5-8) and that a no-response item carries an `ItemTiming` rather than no timing at all (P5-9); §3.4.2 answers the `sampling.seed`-in-the-fingerprint gap **transitively through `packContentHash`**, before S3 rather than after (P5-10); §4 S1e Table D dispositions `paired_cluster_bootstrap` as **kept** — §3.2d's entry point for every continuous verdict, `paired_bootstrap` its engine (P5-3, note §3.4 Rule 4) — and new **Table E** parameterises `_widen`'s `[-1, 1]` clamp, a live defect in shipped code that is wrong for `sep_z`, gated on §4 S3's done-condition 2.

2026-09-06 — v1.10: the plan gate's `## Pass 4` (`docs/reviews/small-model-benchmarking.md`, `bb0cacf`) closed in full — all fourteen findings, none carried — and notes **v1.11** (`a5f42f6`) and **v1.12** (`fc2fcf6`) folded in: §3.6 gains **the unit boundary**, the one place the seconds→milliseconds conversion is stated, so an ms-named field is never sourced from a seconds-valued one (plan-gate P4-1); §4 S1's `ItemResult` and §4 S2's `LatencyBlock` gain the carriers every FR-11 figure the report is committed to printing needs (P4-3); §3.4.2's single edit table becomes **§4 S1e's four grep-pinned tables** under §7's new rule 5 — *an edit list over shipped code is a grep with a count* — which is what makes the `armProfile` re-key and the two shipped `_percentile` copies enumerable rather than remembered (P4-2); plus the capture order's missing `catalog()` step and both refusals (P4-5), `attest`'s unobservable runtime comparand and the first-observation back-fill (P4-6), a fourth failure disposition with `latencyWithheldForTimeout` renamed `latencyWithheldForNoResponse` (P4-7), DC-10's pooled-metric contradiction **closed** by a `validate_pack` refusal rather than disclosed (P4-8), R-14 swept to match §3.6 (plan-gate P4-4), and plan-gate P4-9…P4-14; and from note v1.11, §3.4 Rule 4's closed form is binding, so the paired **binary** interval resamples nothing and takes no seed, `sampling.seed`'s object moves to §3.2d's continuous-metric bootstrap (§3.3, §3.9), `verdict`'s `bootstrap_seed` and `conservative_envelope`'s `diffs`/`B`/`seed` retire, and the `DecidedBy` token `cluster-bootstrap` is renamed `conservative-envelope` (§4 S1e Table D); and from note v1.12, `ItemResult` gains an **`ItemTiming`** record — `latencyMs` becoming a derivation over it rather than a second home for the same number — which is where the withheld wall clock stays readable for §11.5.1's computed `censoringExact` (§11.9 ask 2b), and `statsCoveredCount` becomes `None`-never-`0` **by call surface** (§11.9 ask 5, plan-gate P4-13).

2026-09-03 — v1.9: the plan gate's Pass 3 (`docs/reviews/small-model-benchmarking.md`, `ff499d5`) closed in full — §3.4.4a stops being a section about one source and becomes one about all five, gaining the source table, the capture-order sequence that pins where the staleness trip-wire fires, and **`callSurface` as a second discriminator** so an embeddings arm is required to carry what it can observe and forbidden to carry what it cannot (G3-1, G3-2, sweeping §3.4.1's forbidden-set derivation, §3.4.2, §3.6, §4 S1's signature block, Appendix A); the contamination guard withholds **`latencyMs` alone** and gives the three `stats`-derived figures their own coverage — G3-3's fix is the missing denominator, not a nulling, settled by `-ml` §11.4's measurement that LM-Studio-side TTFT *excludes* the JIT load — with `unexplainedMs` closing R-14's in-call residual (§11.5.1), the item-1 baseline moved to the post-warm-up probe, and a timed-out scored call given a disposition (G3-3, G3-4, G3-5); DC-10's predicate becomes `roles.unit_kind(pack.role)` and treats `IncompleteItemRecord` as a mismatch (G3-6, G3-7); plus R-15's three costs, §3.4.2's fourth edit site with its DC-1 assertion, `--no-cold-load` deleted, test 15b moved out of the `-m live` block, and §3.4.2's tier complement rule (G3-8…G3-13); and, from note v1.10, **R-13 is closed** with the plan carrying its four consequences — the `LatencyBlock`, `latencyMsMax` and the `index.csv` columns, the two withholding dispositions and the detector — while R-4's second κ restatement is withdrawn (n-ML-9) and no load-cost figure is left anywhere the design is sized against, the two measured cold loads on this box differing by ~6×.

2026-09-03 — v1.8: three changes forced by evidence gathered since v1.7 — the auto-captured fingerprint fields move off the `lms` CLI onto LM Studio's native `GET /api/v0/models`, with no substitute source, no fallback that can populate them and no default (new §2.5 and §3.4.4a, sweeping §3.2, §3.3, §3.4.2, §3.4.4, §3.4.5, §3.6, §3.6a, §3.7, §3.10, §4 S2, §5 test 15, §6 R-1/R-2, new R-15 and Appendix A); the runner gains an explicit un-timed warm-up call, a two-budget timeout and a per-item load-contamination guard for the measured 21.068 s JIT first call (§3.6, §4 S2, §5 test 15b, new R-14); and the `run.aggregates`-versus-`run.items` cross-check returns to **S1**, where the Pass 4 gate placed it (impl-gate P4-4 — §4 S1 done-condition 10, §4 S2, §5's table and new test 11c).

2026-09-03 — v1.7: the two Pass 3 findings routed here — §5 gains the stage-attribution table three gates had to re-derive (engineering Pass 3, "carried a third time"), and every remaining restatement of something `docs/plans/small-model-benchmarking-ml.md` owns is replaced by a citation: the α in §4 S1's `holm_steps`/`resolving_power` sketch (statistics Pass 3, n-ML-7), the `z` literal, the verdict string, the judge-gate thresholds and κ figures, §6 R-9's worked-case numbers, and the note's own rule count; plan and note re-pair at v1.7.

2026-09-03 — v1.6: the S1 fix round's seam changes (`3ad27d3`, engineering re-gate `8b7b60a`) — `PackRef.contentHash` becomes `str | None` with `Pack.ref()` as the totality boundary (§3.3, §4 S2, Appendix A), `stats.holm_steps`/`HolmStep`/`verdict(holm_tested=)` and `compare_report`'s two-pass structure replace `holm_thresholds` (§4 S1), §7 records S1's real state (not closed — the statistics re-gate has one open blocker), and the three places that restated the note's family-adjusted α (§3.3(ii), §3.8.2, §3.9) now cite it instead, ahead of the note's v1.6 ruling.

2026-09-03 — v1.5: the S1 implementation gate's plan-side corrections (`docs/reviews/small-model-benchmarking-impl.md` §4) — §3.4.1's forbidden-field list becomes a derivation over §3.4.2's now-complete field set, Appendix A's `PackRef` and `FieldProblem` catch up with §3.3 and the shipped code, §4 S1 records `RunResult.designEffect`/`basis` as required-with-no-default, §6 opens R-13 (`_percentile`'s definition), and §7 gains the intra-document half of its staleness rule plus S1's delivery status. (v1.4 closed Pass 2's N-1/N-2/N-4 and narrowed §7's precedence rule; v1.3 aligned the vocabulary to the `-ml` note; v1.2 closed the Pass 1 gate.)

Design for [`../requirements/small-model-benchmarking.md`](../requirements/small-model-benchmarking.md)
(Status: *Ready for design*, 23 FRs / 5 ACs). This plan covers the whole feature: the new top-level
component `model-bench/`, its harness, its task-pack format, and all five roles FR-21 requires at
first delivery — role five, `chat-responder`, honestly partial per FR-21a.

The statistical instruments are **not** designed here. They are settled once in the companion
method note, [`small-model-benchmarking-ml.md`](./small-model-benchmarking-ml.md) (`data-scientist`),
and cited by section from §3.9 below.

**CPG:** used `cpg_falkorchat` — confirmed the blast radius of the "extract falkor-chat's eval
helpers" option before rejecting it (§3.1 D1): every caller of `recall_at_k`, `mrr`,
`check_regression`, `score_pair`, `wilson_interval`, `layer2_contains`, `judge_triple`,
`build_judge_prompt` lives inside `falkor-chat/server/tests/eval/` — 8 callees, 16 caller/callee
pairs, 0 outside that directory. The graph covers `tests/` as well as `falkorchat/` (2 862 vs
1 222 `METHOD` nodes), so that is a real absence, not an unindexed one. **The graph is stale**
(built 2026-09-02T12:38:21Z at `4bb96e1`, `SOURCE_DIRTY = true`, several `falkor-chat/server`
commits since), but the claim is directory-scoped and the `analyst` gate independently re-ran it
against the current tree and reproduced it exactly. No CPG exists for `model-bench/` — it is a new
component — so every structural claim about `model-bench` in this plan is a claim about what is to
be built, not a graph answer.

---

## 1. Goal & scope

Build `model-bench/` — a standalone, human-started harness that measures one local model at a time
against one named task pack, stores the result with a full environment fingerprint, and compares it
against previously stored results **for the same role**, with confidence intervals and visible
flags whenever a comparison is not apples-to-apples.

**In scope (first delivery):**

- The component skeleton: `model-bench/` at the repo root, sibling of `falkor-chat/`, with its own
  `docs/` per root `AGENTS.md`'s module-documentation convention.
- The harness core: task-pack loading + versioning, run orchestration, fingerprint capture and
  enforcement, result store, comparison report.
- The LM Studio adapter (inference + embeddings + model catalog + load control).
- **Five task packs**, one per role (FR-21): `tool-caller`, `guard-judge`, `nlq-generator`,
  `chat-responder`, `embedder`.
- Golden data **copied into the tool and versioned by it** (§3.1), including the one genuinely new
  asset: the tool-caller's fixed multi-turn conversation scripts (FR-22).

**Explicitly out of scope** (from the requirements' own out-of-scope section — restated because
each one is a thing an implementer might otherwise build):

- No CI hook, no scheduler, no timer. `model-bench` only ever runs because a person typed a command.
- No pass/fail gate, no non-zero exit on a "bad" score. The only non-zero exits are operational
  (bad arguments, unreachable LM Studio, invalid pack, missing fingerprint).
- No writing to any other component's configuration, ever, and **no run-time code path that reads
  anything outside `model-bench/`** — the one-way `refresh_golden.py` importer (§3.1) is a
  human-invoked maintenance script and is never reachable from a run.
- No global leaderboard and no cross-role aggregate number (FR-20) — enforced structurally in §3.5.
- No reproduction of published benchmark scores.

**Also out of scope for this plan, deliberately:** cloud/hosted model support, and **any LLM judge
at all** — FR-21a defers judged reply quality, so no pack in this delivery contains one (§3.8.5
records the deferred design rather than building it). Both are "deliberately not ruled out" in the
requirements; the pack format in §3.3 leaves room for the first (a pack names a *provider*, and
LM Studio is one) without building it.

---

## 2. Context & findings

### 2.1 What `falkor-chat/server/tests/eval/` actually is

Read in full. 32 files, ~2 400 lines of Python plus the data. Structure:

| Artifact | Size | Nature |
|---|---|---|
| `metrics.py` | 96 lines | **Pure**, zero imports: `recall_at_k`, `mrr`, `check_regression`. |
| `nlq_scoring.py` | 267 lines | **Pure** (stdlib only): `score_pair`, `layer2_contains`, `wilson_interval`. |
| `guard_calibration.py` | 327 lines | Imports `falkorchat.guards.evaluate_guard`. |
| `judge.py` | 159 lines | Imports `falkorchat.llm`. LLM-as-judge, faithfulness/relevance binary. |
| `conftest.py` | 137 lines | Probes `ws:eval` in FalkorDB, reads `config/models.json` for the expected dim. |
| `run_nlq_golden_set_eval.py` | 278 lines | Drives `falkorchat.tools.QueryGraphDataTool` against live FalkorDB. |
| `golden_retrieval.jsonl` | 38 items | `{id, query, relevant_msgIds, topic, target_text, rationale}`. |
| `retrieval_baseline.json` | — | Pinned: recall@10 = 0.9737, recall@5 = 0.8947, MRR = 0.6259, n = 38. |
| `golden_guards.jsonl` | 85 items | tiers: 30 clear_advance / 40 clear_suspend / 15 boundary; 30 `expected:true` / 55 `false`; evidence path 51 understanding / 34 turns. |
| `golden_judge_calibration.jsonl` | 10 items | `{question, context[], answer, expected_faithfulness, expected_relevance}`. |
| `nlq_golden_set.jsonl` | 40 items | 21 catalog / 19 knowledge_base; shapes: 9 single-fact, 8 aggregation, 7 filter-list, 6 not-found, 4 compound-filter, 4 relationship-traversal, 2 conflicting-facts. |
| `corpus_provenance.json` | — | 121 messages, 12 threads, `text-embedding-qwen3-embedding-0.6b`, dim 1024. |
| `judge_calibration.json` | — | n=10, faithfulness agreement 0.90 (κ = 0.83), relevance agreement 0.70 (**κ = 0.21** — recomputed, `-ml` §6.1), `sameModelAsAgentUnderTest: true`. |

Three things this inventory settles:

1. **The golden *data* is portable; the golden *mechanisms* are not.** Every scored item is text
   plus a label. But the retrieval items key relevance on `msgId`s that only exist in the seeded
   `ws:eval` graph, the guard items are scored through `falkorchat.guards`, and the NLQ items are
   scored by executing `QueryGraphDataTool` against a seeded `reference`/`ws:nlq-eval`.
2. **Two of the three source corpora are inline literals; the third is derived.**
   `falkor-chat/scripts/seed_eval_corpus.py` carries the whole 121-message retrieval corpus as a
   `_CORPUS` list (line 80 ff.) and `falkor-chat/scripts/seed_catalog.sh` carries the 15-product
   catalog as a `CATALOG` list of `(name, category, price)` tuples (line ~78 ff.) — both copy out to
   JSON fixtures directly. `falkor-chat/scripts/seed_nlq_eval_corpus.py`'s `_CORPUS` (line 127 ff.)
   is only the *input documents*: the queryable form is produced by falkor-chat's ingestion +
   entity-extraction pipeline, and currently lives in the `ws:nlq-eval` graph (62 `Entity`, 12
   `Document`, 12 `Chunk` — read-only count, 2026-09-02). So that one is copied as a **snapshot of
   the derived rows**, not re-derived (§3.8.3).
3. **The NL-query mechanism does not ask a model for Cypher.** `falkorchat/querygen.py`'s
   `DatasetSchema`/`CATALOG_SCHEMA` plus the prompt in `falkorchat/tools.py` (~line 882) have the
   model emit a **structured JSON query spec** (`[{"property": "name", "op": "=", "value": …}]`)
   which falkor-chat then compiles. That is what makes an in-process, database-free `nlq-generator`
   pack possible (§3.8.3) — worth knowing before assuming a FalkorDB dependency.

### 2.2 The prior tool-calling experiment (FR-22's starting point)

`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8 is the empirical basis for FR-9.
Its §8.1 documents three fixed conversation scripts — **Condition A** (9 turns, read-only catalog
lookups: exact-name lookup, category filter, price-range filter and repeat, an abstention pair,
then a repeated/rephrased exact-name lookup), **Condition B** (7 turns, write-mutating: add 2×
Wireless Mouse Pro, add 1× Portable SSD 1TB, view cart, remove 1×, remove the SSD entirely, view
cart, add 1× Bluetooth Speaker Mini), **Condition C** (4 turns, read-only, a length-sensitivity
probe) — run at n=40 conversations / 280 turns.

Two consequences:

- **The scripts were never committed** (§8.6: "two throwaway scripts … left in this session's
  scratchpad"), so FR-22's asset genuinely has to be built. But §8.1's prose specifies A and B
  turn by turn, so it is a **reconstruction from a written spec**, not a blank page.
- **It hands the new harness a known-answer validation target.** §8.2 records that
  `qwen/qwen3-4b-2507` does not degrade gradually but collapses near-deterministically at a fixed
  turn position, while `mistralai/ministral-3-3b` behaves differently on the same scripts. A
  freshly built harness that cannot reproduce that contrast is broken. §5 makes this an explicit
  acceptance step rather than a hope.

### 2.3 The LM Studio surface — live-probed on this box, 2026-09-02

Verified by direct calls, not from documentation:

- `GET http://localhost:1234/api/v0/models` returns, per model: `id`, `type`
  (`llm`/`vlm`/`embeddings`), `publisher`, `arch`, `compatibility_type`, `quantization`, `state`,
  `max_context_length`, `capabilities` (e.g. `["tool_use"]`), and `loaded_context_length` for a
  loaded model. 19 models are installed on this box.
- **The requirements' "two catalog ids for the same weights" is visible in that list**:
  `mistralai/ministral-3-3b` and `mistralai_ministral-3-3b-instruct-2512` are separate entries with
  the same `Q8_0` quantization and different `publisher` fields. The fingerprint must record the
  literal key used, never a normalized alias.
- `POST /api/v0/chat/completions` (the LM Studio-specific route, **not** `/v1/`) returns, beside
  the OpenAI-shaped body: `stats{tokens_per_second, time_to_first_token, generation_time,
  stop_reason}`, `model_info{arch, quant, format, context_length}`, `runtime{name, version,
  supported_formats}`. That is FR-11's TTFT and FR-7's runtime identity, for free, per call.
- `lms.exe` **is** reachable from WSL at `/mnt/c/Users/<user>/.lmstudio/bin/lms.exe` (LM Studio runs
  on Windows; the memory note about mirrored networking applies). `lms ps --json` works and returns
  `[]` when nothing is loaded — that is FR-7's "what else was resident". `lms server status --json`
  returns `{"running":true,"port":1234}`. `lms load` exposes `--context-length`, `--gpu`, `--ttl`,
  `--identifier`, `-y`, and `--estimate-only` (resource estimate **without** loading).
  **Still true on 2026-09-03, and no longer a design input** — §2.5 re-probed it and §3.4.4a decides
  against depending on it. Read this bullet as what the CLI *can* do, not as what the harness uses.
- **Two FR-7 fields have no programmatic source** (§6 R-1): `lms version` prints only a *CLI commit*
  (`07b7252`), not the LM Studio application version; and no `lms load` flag or API field exposes
  the **KV-cache setting**, which is a GUI-side load option on this build.

### 2.4 Repo grain the component must follow

- `mcp-monitor/` is the structural template for a small standalone Python component: `pyproject.toml`
  (`requires-python = ">=3.12"`, ruff `select = ["E","F","W","I"]`, `line-length = 100`, pytest
  `testpaths`), an idempotent `setup.sh` creating a component-local `.venv`, a thin `run.sh`,
  `AGENTS.md` + `README.md`, and `docs/{BACKLOG,HISTORY,requirements,plans,reviews,test-plans,test-reports}`.
- `falkor-chat/server/pyproject.toml` supplies the **live-test convention** to copy verbatim:
  `addopts = '-ra -m "not live"'` plus a `live` marker, so the default suite is network-free and
  real-model tests are opt-in via `pytest -m live`.
- `falkor-chat/server/falkorchat/transport.py` reaches the OpenAI-compatible API with **stdlib
  `urllib.request`**, no `httpx` at runtime. `numpy` appears nowhere in the repo.
- `cypher-mcp/` establishes the **content-hash-as-identity** idea (its image tag is a hash of build
  inputs, "so a stale image is unrepresentable"). §3.3 applies the same idea to pack versions.

### 2.5 The same surface, re-probed 2026-09-03 — and the CLI stops being a dependency

§2.3 is the 2026-09-02 probe and stays as written. This is the second probe, run because the
stakeholder's own 2026-09-03 pass found `lms` missing from `PATH`; every number below was
re-measured here rather than taken on report. The design consequences are §3.4.4a's.

- **`lms` is not on `PATH`** (`command -v lms` → exit 1). The Windows binary *is* still there and
  still works when invoked by absolute path: `/mnt/c/Users/*/.lmstudio/bin/lms.exe ps --json`
  returns `[]`, exit 0. So the honest statement is **not** "the source disappeared" — it is that the
  only way to reach it is a globbed Windows path, which is a host-layout accident, not a contract.
- **The CLI is ~150× more expensive than the HTTP surface for the same fact.** `lms.exe ps --json`
  measured **0.30 s** on three consecutive runs; `GET http://localhost:1234/api/v0/models` measured
  **1.6–2.3 ms** on five (4.2 ms on the first, cold-TCP call). The CLI figure is the same
  WSL→Windows subprocess cost the gate measured for `powershell.exe` (0.18–0.54 s, gate Appendix
  A.3) — it is the boundary, not the tool. That difference is what makes a *between-item* residency
  probe affordable (§3.6) where a between-item `lms ps` would not be.
- **`GET /api/v0/models` answers with everything the fingerprint's auto half needs**: 19 models,
  each carrying `id`, `object`, `type`, `publisher`, `arch`, `compatibility_type`, `quantization`,
  `state`, `max_context_length`, `capabilities` (plus `loaded_context_length` once a model is
  loaded — §2.3). All 19 read `state: "not-loaded"` at probe time, so the clean-box case is real and
  observable: the residency list is genuinely `[]`, from a probe that succeeded. **This bullet is a
  narrative record, not a payload** *(v1.25, F-S2-1)*: it names the field set, the count and the
  aggregate state, and no verbatim response body was ever saved anywhere in this repo. No
  done-condition, test or fixture may cite it as a capture — §4 S2 did, and is corrected.
- **`GET /v1/models` cannot substitute for it.** The OpenAI-compatible route returns exactly
  `{id, object, owned_by}` per model — no `state`, no `quantization`, no `arch`, no `type`, no
  `capabilities`. It can populate `modelKey` and nothing else in §3.4.2, which is why §3.4.4a uses
  it as a **reachability discriminator only** and never as a degraded source.
- **Auto-load (JIT) is on, and the first call to a model pays for it.** A cold
  `POST /v1/chat/completions` against `mistralai/ministral-3-3b` at `temperature: 0`,
  `max_tokens: 10` returned correctly in **21.068 s**, essentially all of it model load. *(Read this
  as one model's load cost, never as the load cost: `-ml` §11.4 later measured `qwen/qwen3-4b-2507`
  (Q4_K_M) cold at 3.625 s on this same box. **The two are not the same call surface, and v1.9 said
  they were** — plan-gate P4-11. They differ in model, in quantization **and in route**: this one is
  `POST /v1/chat/completions`, while §11.4's carries a `stats` object and so was taken on
  `POST /api/v0/chat/completions`. The note names **page-cache state** as the obvious difference
  between them; what follows for this plan is only that the ~6× spread has more than one cause and
  none of them is predictable per model, so nothing in the design may be sized against either
  figure — §3.6's budgets are sized by magnitude. **The note has since ruled on the consequence for
  its own threshold** (`-ml` v1.12 §11.5.1): the *"~3.5× below the smallest cold load"* margin is
  **withdrawn** — a pair of unattributed single observations bounds no load from below — while the
  1 000 ms value stands on a different basis. Cited, not restated; §6 R-14 carries what follows for
  this plan.)* The
  harness
  pays this on its first call to **each arm**, which is a runner-design constraint, not a slow box:
  see §3.6's warm-up rule. It also means the harness has **no way to force a cold state** — nothing
  on either HTTP surface unloads a model.

---

## 3. Design & rationale

### 3.1 D1 — Relationship to `falkor-chat/server/tests/eval/`: **copy the data, clean-build the code**

This is the design constraint the requirements deliberately left open. The decision:

> **Copy the golden *data* into `model-bench`, under `model-bench`'s own versioning and with a
> provenance record naming its origin. Re-implement the *code* — metrics, scoring, judge — inside
> `model-bench` from scratch. Change nothing in `falkor-chat`. Zero imports in either direction.**

**Why not extract-and-generalize** (move `metrics.py`/`nlq_scoring.py` into `model-bench` and have
falkor-chat import them):

- The CPG says the mechanical blast radius is small and contained (§ preamble: 16 call sites, all
  inside `tests/eval/`) — so the objection is not "it's too big to do". It is that the *result* is
  bad: falkor-chat's regression gate would acquire a runtime dependency on a brand-new sibling
  component, meaning `falkor-chat/server/.venv` must have `model-bench` installed for
  `pytest -q` to collect. Root `AGENTS.md` opens with "a monorepo of **independent, self-contained
  components**"; this would make the oldest, most locked-down component depend on the newest and
  least stable, on exactly the axis (statistics) where the new one will churn most.
- The genuinely shared surface is **~40 lines of textbook formulas** (`recall_at_k`, `mrr`,
  `wilson_interval`). Everything else does not transfer: `check_regression` implements a
  zero-tolerance *gate*, which the requirements put explicitly out of scope for this tool;
  `score_pair`/`layer2_contains` are shaped around `QueryGraphDataTool`'s return payload;
  `guard_calibration.py` calls `falkorchat.guards`. Extracting 40 lines is not worth a
  cross-component dependency.
- FR-23 permits the falkor-chat → model-bench direction. It permits it; it does not recommend it.

**Why not a clean build that also re-drafts the data:** the expensive, irreplaceable part of a
golden set is not the text, it is **the human verification of every label** (FR-19). 38 + 85 + 10 +
40 = 173 human-verified items already exist. Re-drafting them would be destroying value to avoid a
duplication that §"the honest answer" below shows is not a duplication at all.

**The duplication risk, answered head-on.** The stakeholder's concern is "two golden sets and two
metric implementations to keep honest". Three points:

1. **The two golden sets *should* diverge, and copying is what makes that safe.** falkor-chat's set
   is a regression gate pinned to *its* corpus and *its* embedding model; it must be updated
   whenever that corpus changes. `model-bench`'s set must **freeze**, because the entire stated
   value of the tool is that "a new model's result lines up against models I tested months ago".
   A shared set would mean any edit made for falkor-chat's benefit silently invalidates every
   stored `model-bench` result. Divergence is the correct behavior, not drift to be prevented.
2. **The metric duplication is discharged by a transcribed behavioural fixture, and the residual
   is stated rather than papered over.** *(Revised in v1.2. v1.1 claimed a cross-check "against the
   values in `retrieval_baseline.json` when fed the same ranked lists"; the gate established that
   **the ranked lists do not exist as an artifact** — `retrieval_baseline.json` holds four aggregate
   numbers, and the ranked lists behind them are produced live by `services.hybrid_search` against
   the seeded `ws:eval` graph, which FR-23 forbids `model-bench` from touching and which is
   ANN-approximate anyway. That mechanism was not constructible. This is its replacement.)*

   **(a) The fixture.** `model-bench/tests/fixtures/metrics_agreement.json` is a one-time,
   hand-transcribed capture of **every assertion in
   `falkor-chat/server/tests/eval/test_metrics.py` that exercises `recall_at_k` or `mrr`** — read
   from the file, counted: **20 cases (18 value assertions + 2 `ValueError` cases)**, 13 for
   `recall_at_k` and 7 for `mrr`. The count is stated here so an implementer who ships three cannot
   call it done. Per case:

   ```json
   {"function": "recall_at_k", "case": "hit_outside_top_k_window",
    "args": {"retrieved": ["x","y","z","w","v","a"], "relevant": ["a"], "k": 5},
    "expected": 0.0}
   ```

   with `"expectedError": "ValueError"` in place of `expected` for the two raise cases. File header:
   `sourcePath` (repo-root-relative), `sourceGitSha`
   (`9650a3858b9d5c4e7e934f977839fc1a61c84b1b` at the time of writing), `sourceSha256` of the origin
   file's bytes, `copiedAt`, `transcribedBy`, `verifiedBy`, and an `excluded` list. Transcription is
   **manual, not extracted**: only 6 of the 20 cases live in `@pytest.mark.parametrize` tables; the
   other 14 are literals inside individual test bodies, so a mechanical extractor would silently
   capture a third of the surface and pass. `check_regression`'s 6 cases go in `excluded` with their
   reason — `model-bench` deliberately does not implement a regression gate (the requirements put
   zero-tolerance gating out of scope), so there is nothing to agree with.

   **(b) The test.** `model-bench/tests/test_metrics_agreement.py` runs `model-bench`'s
   implementation over every non-excluded case and requires equality to within `1e-12` absolute
   (the values are small exact binary fractions plus `1/2` and `1/3`, which both implementations
   reach by the same division), and `pytest.raises` for the two error cases. It reads only
   `model-bench/tests/fixtures/`, so the default suite still passes with `falkor-chat/` renamed away
   (§5 test 20).

   **(c) The drift detector lives on the maintenance path, not in the suite.**
   `scripts/refresh_golden.py --check-origins` re-reads and re-hashes every recorded origin file,
   including `test_metrics.py`, and reports each one as unchanged or drifted against its
   `sourceSha256`. It is the one component that may read `falkor-chat` (§1 out-of-scope, third
   bullet), it is human-invoked, and it is never on a run path.

   **The honest residual.** This proves behavioural agreement **on the cases falkor-chat itself
   tests, as of the transcription** — not on untested inputs, and not automatically when the origin
   moves. That is a weaker guarantee than v1.1 claimed, and it is stated here rather than implied
   away. What carries the rest of the weight is **(i)** the shared numeric surface is deliberately
   tiny — `recall_at_k`, `mrr`, `wilson_interval`, three textbook formulas — and each is *separately*
   pinned to a reference outside falkor-chat: `wilson_interval` and the paired instruments to the
   `-ml` note's own worked regression fixtures (§5 test 6), `recall_at_k`/`mrr` to the transcribed
   cases plus their textbook definitions; and **(ii)** the two implementations' outputs are **never
   compared to each other as numbers, by design** — `retrieval_baseline.json`'s pinned figures come
   from a hybrid-ANN pipeline and `-ml` §5.4 already forbids reading a difference against them in
   either direction (§3.8.1, S3's done-condition). So the duplication cost is bounded at "two places
   to fix a formula bug", which a 20-case fixture detects, and never at "two numbers that silently
   disagree in a report".
3. **Copying is one-way and deliberate.** `model-bench/scripts/refresh_golden.py` (§4 S3) is a
   human-invoked, read-only importer that re-copies from a given `falkor-chat` path, rewrites the
   pack's `PROVENANCE.md` and **forces a pack-version bump**. It is never run automatically, and
   `model-bench` never reads `falkor-chat` at run time — the importer is a maintenance script, not
   a code path of any run.

**What gets copied, and to where:** see the per-pack table in §3.8.

### 3.2 D2 — Component shape and layout

```
model-bench/
  AGENTS.md  README.md  pyproject.toml  setup.sh  run.sh  .gitignore
  docs/{BACKLOG.md,HISTORY.md,requirements/,plans/,reviews/,test-plans/,test-reports/}
  modelbench/
    __init__.py  __main__.py  cli.py
    lmstudio.py        # LM Studio adapter: catalog, residency, chat, embeddings, warm-up, per-call stats
    hostinfo.py        # residency/catalog probe (§3.4.4a), host RSS sampling, attestation file
    fingerprint.py     # Fingerprint dataclass, capture(), validate() — FR-7/AC-2
    packs.py           # manifest load, content hash, version compare — FR-5/FR-6/FR-9a
    convo.py           # multi-turn driver + pack-configured prompt assembly — FR-9a
    tooling.py         # simulated-tool protocol, dispatch trace — FR-10
    runner.py          # one run = one model × one pack × n items
    results.py         # RunResult/ItemResult schemas, store, index, quarantine — FR-2/AC-2
    stats.py           # intervals + paired comparison — FR-15/FR-16, per the -ml note
    report.py          # markdown comparison output — FR-3/FR-6/AC-3/AC-4
    scoring/{toolcalls,retrieval,classification,extraction,grounding}.py
  packs/<pack-id>/     # one directory per pack, see §3.3
  results/runs/<runId>.json        # committed
  results/index.csv                # committed, derived, regenerable
  results/transcripts/<runId>.jsonl  # gitignored — raw model output, not needed for comparison
  reports/<pack-id>-<date>-<n>.md  # committed
  tests/
```

**`results/` and `reports/` do not exist at S0.** They are created on first write by S1–S3, so the
S0 tree is the top three lines plus `modelbench/` and `tests/`. S0's `.gitignore` nevertheless names
`results/transcripts/` from the start, deliberately: a `.gitignore` entry for a path that does not
exist yet is inert and free, whereas adding it only when the first transcript lands is one commit
away from committing a megabyte of raw model output by accident.

**Dependencies: none at runtime.** stdlib `urllib.request` for HTTP (falkor-chat's own precedent),
stdlib `json`/`math`/`statistics`/`hashlib`/`subprocess`. Dev extras: `pytest`, `ruff` only. The
largest numeric job is 38 queries × 121 documents × 1024 dims of cosine (~4.7 M multiply-adds),
which is a few seconds of pure Python — acceptable, and the reversal trigger is explicit: **add
`numpy` if a pack's corpus exceeds ~1 000 documents or scoring exceeds ~5 s**. A benchmarking tool
whose own dependency tree can rot is a benchmarking tool whose old results stop being reproducible;
zero runtime dependencies is worth a few seconds.

**Python 3.12**, matching every other component.

### 3.3 D3 — The task pack: a directory, a manifest, and a content hash

A pack is a directory under `model-bench/packs/`. FR-5 requires that adding a scenario not change
the harness; FR-9a requires prompt assembly to be *pack* configuration; FR-6/AC-3 require version
mismatches to be detectable after the fact.

`packs/<pack-id>/pack.json`:

```json
{
  "packId": "tool-caller-shop-assistant",
  "packVersion": "1.0.0",
  "role": "tool-caller",
  "schemaVersion": 1,
  "description": "Multi-turn catalog + cart tool-calling against a simulated storefront.",
  "scorer": "toolcalls",
  "environment": {"requires": ["lmstudio-chat"]},
  "prompt": {
    "systemPrompt": "prompts/system.md",
    "toolSchemas": "tools/schemas.json",
    "historyReplay": "structured-replies-only",
    "representToolSchemasEachTurn": true,
    "historyTurns": 0,
    "maxIterationsPerTurn": 8,
    "temperature": 0.0,
    "maxTokens": 1024
  },
  "data": {"conversations": "conversations.jsonl", "catalog": "catalog.json"},
  "tools": {"module": "tools/sim.py", "entrypoint": "build_environment"},
  "sampling": {"scripts": 12, "replicatesPerScript": 1, "seed": 20260902,
               "pairingKey": ["scriptId", "replicate", "turnIndex"],
               "analysisUnit": "scriptId",
               "determinismProbeScripts": ["A-01", "B-03"]},
  "metrics": {
    "verdictMetrics": ["cleanThroughTurnH"],
    "headlineMetric": "cleanThroughTurnH",
    "cleanThroughTurnH": {"H": 4}
  },
  "provenance": "PROVENANCE.md"
}
```

Key decisions:

- **`prompt` is the FR-9a carrier, and it declares the replay's *shape*. Whose content is replayed
  is not a knob: it is always the model's own, from this run** *(v1.25 — the ruling is §3.8.4's
  "Prompt assembly" bullet and is not restated here)*. `representToolSchemasEachTurn`,
  `historyTurns` (0 = unbounded) and `maxIterationsPerTurn` are separate knobs;
  `historyReplay` selects how prior turns are re-presented, over **four** values spanning **two**
  axes — **role ownership** (are prior turns presented as the model's own turns, or quoted inside
  someone else's message?) and **tool evidence** (is the tool scaffolding visible?). *(v1.26,
  P13-8: v1.25 called this "a ladder in tool evidence", which its own table contradicts —
  `structured-replies-only` and `plaintext` carry identical tool evidence, namely none, and differ
  only on ownership. Which axis moved is what the §6 R-3 bisect's inference rule turns on, so the
  two-axis reading is canonical and R-3's wording is where it is stated for the bisect.)*

  | value | what a prior turn contributes to the message list |
  |---|---|
  | `structured` | the whole real exchange in native roles: per in-turn iteration, the model's own `assistant` message with its `tool_calls` verbatim, then one `tool` message per call carrying the environment's real return value; then the final `assistant` reply |
  | `structured-replies-only` | native roles, one `assistant` message per prior turn carrying that turn's **final reply text only** — no `tool_calls`, no `tool` messages, no breadcrumb |
  | `plaintext` | one flattened `user` message, `User: …` / `Assistant: <final reply text>` per prior turn, likewise with no tool evidence |
  | `none` | nothing |

  The scripted `user` text is always the script's, at every value — the harness drives the full
  script and never rewrites a user turn (`-ml` §4.1).

  **`structured-replies-only` is v1.25's, and it exists because this bullet's own
  *reproducing-the-executor-shape* claim below was false without it.** falkor-chat's executor replays *native* `user`/`assistant` roles carrying only
  each prior turn's final text, discarding the tool scaffolding used inside that turn
  (`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §4.1) — a point the three original
  values do not reach: `structured` shows full tool evidence and `plaintext` quotes the history
  inside someone else's message rather than presenting it as the model's own turns. Since §4.1
  attributes the documented collapse to precisely the absence of visible tool evidence in the
  model's own replayed turns, the difference is the mechanism, not a rendering detail.

  `convo.py` implements exactly these axes and reads them from the manifest — nothing about prompt
  shape is hardcoded, and nothing is imported from any product. Reproducing falkor-chat's executor
  shape is then *one pack's settings* — `structured-replies-only`, which is what
  `tool-caller-shop-assistant` declares — and the settings themselves become a testable variable
  (two packs differing only in `historyReplay` answer "is the replay style what breaks at turn 4?",
  and the four values now span both axes the executor differs on rather than leaving a hole on the
  ownership one; §6 R-3's bisect moves one axis at a time).

  **`maxIterationsPerTurn` is v1.25's, it is pack data with no default, and v1.26 says which packs
  must carry it** — it is `-ml` §4.1's `I(t)` cap, which §4.2(f) reports `iteration_cap_hit_rate`
  against, so a harness-side constant would put a number the report analyses outside the content
  hash. `8` for this pack is falkor-chat's own budget, the one §8.4's `gpt-oss-20b` message-spam
  defect exhausted (independently verified at `falkor-chat/server/falkorchat/proof_defs.py:415`,
  whose `salesperson@v2` node declares `"maxIterations": 8`).
  **Required *iff* the role runs a multi-call turn, and forbidden otherwise** *(v1.26, P13-6:
  v1.25 said "required, no default" while carrying one manifest example, leaving two readings with
  opposite costs — a dead knob inside four packs' content hashes, where a later edit re-identifies
  the pack under AC-3, or an optional field contradicting the stated rule)*. The condition is a
  **third column on the role table**, `roles.MULTI_CALL_TURN_BY_ROLE` — `True` for `tool-caller`,
  `False` for the four item-level roles, which are single-call by construction — held exactly the
  way route (iii) below is held: a computed `set(MULTI_CALL_TURN_BY_ROLE) == set(roles.ROLES)`, and
  both refusals driven by **executing** `validate_pack` per role, never a prose sentence.
  `validate_pack` refuses a `tool-caller` manifest without the field and refuses any other role's
  manifest with it. `PromptConfig.maxIterationsPerTurn` is `int | None`, `None` **iff** the role's
  column is `False`; **`drive` raises on `None`** rather than substituting a value, which is what
  keeps the no-default rule intact — the one consumer refuses, and no number is invented anywhere.
  The mechanical consequences are the rework unit's, named so they are not discovered: the four
  item-level packs' manifests gain nothing, and `tests/fixtures/packs/valid/pack.json` — the only
  fixture with a `prompt` block, declaring `historyReplay: "structured"` and no
  `maxIterationsPerTurn` at `40a9bc8` — needs both fields.
  **One consequence of this field, and it is a consequence rather than a second declaration**
  *(v1.27, `-ml` §11.9 ask 7)*: an item's **`callCount`** — the number of model calls the item
  comprises, and the unit `-ml` §11.4's three `stats`-derived figures are denominated in — is
  `I(t)` on a `tool-caller` and **`1` for every other role**, because
  `MULTI_CALL_TURN_BY_ROLE[role]` is `False` there and a single-call turn is one call by
  construction. **No new manifest key**: `callCount` is recorded per item (§4 S1's `ItemTiming`),
  never declared, and the two are bound by assertion rather than by convention (§4 S2 rule (ii)).
- **`metrics` pre-registers the verdict family, and a pack may legitimately have no headline.**
  Two separable fields, because they control different things — `-ml` §3.3 states the split and
  this plan owns the names:
  - **`verdictMetrics`** — the closed, pre-registered list (1..k) of metrics that may receive a
    better / not-distinguishable **verdict**. It controls **inference**: its length *is* the
    multiplicity correction's `k`.
  - **`headlineMetric`** — either exactly one member of that list, or an **explicit `null`**. It
    controls **presentation**: what a reader is entitled to read as "the" number.

  Everything not in `verdictMetrics` prints with `exploratory — no significance claim`.
  `validate_pack` fails when `verdictMetrics` is absent or empty, when the `headlineMetric` **key**
  is absent (omission is not the same statement as `null`, and only the latter is a decision), or
  when a non-null `headlineMetric` is not a member of `verdictMetrics`. **Four** consequences the
  implementer must build rather than infer *(the fourth is v1.12's)*:

  **(i)** When `headlineMetric` is `null`, `report.py` has **no code path that synthesises a
  headline** from `verdictMetrics` — the same structural refusal FR-20 gets in §3.5 — and the report
  prints the verdict metrics side by side **in the manifest's declared order**, with no summary line
  above them and no arithmetic combining them.

  **(ii)** When `len(verdictMetrics) > 1`, family-wise error control is **mandatory, not optional**
  (`-ml` §3.3): Holm–Bonferroni across the declared family, with each member's adjusted threshold
  printed beside its p-value and the step-down stop applied to the decision, never only to the
  printout. That much is the harness surface and this plan owns it. **Which α each figure is
  computed at — the ladder's steps, the resolving-power line, and the observable floor — is the
  note's and is not restated here** (`-ml` §3.3 and Rule 7; the α/k arithmetic v1.5 spelled out in
  this paragraph is withdrawn in v1.6 rather than re-stated, because the note's v1.6 revision moves
  it). What the plan does require is that the numbers a report *prints* and the rule it *applies*
  are the same rule: `verdict()` refuses rather than warns when its preconditions do not hold, so a
  family verdict cannot be printed at a threshold other than the one that decided it. That
  guarantee is not a matter of taste — printing one rule and applying another was the S1 gate's
  blocker (impl review Pass 1 B-1), and the two-pass `compare_report` in §4 S1 is its fix.

  **(iii)** Pre-registration in a content-hashed manifest is what stops a headline being chosen
  after the results exist.

  **(iv) A `verdictMetrics` family is homogeneous in kind, and a mixed one is refused** *(v1.12,
  `-ml` v1.15 §3.3)*. Holm orders a family **by p-value** and a continuous verdict has none — for a
  continuous metric the interval *is* the test (`-ml` §3.2d) — so a list mixing a binary member with
  a continuous one has no ordering, hence no ladder, and the correction silently fails to happen for
  one of them. An **all-continuous** family with `k > 1` takes its correction **in the interval**
  instead of in a ladder; the percentiles it is taken at are the note's and are not restated here.
  Every pack declared in §3.8 is homogeneous, so this binds nothing today and forbids the silent
  case later.

  **Where the refusal lives is this plan's, and it is not `validate_pack`** *(§7 rule 2 — the note
  owns the rule, the harness surface owns its enforcement point; **confirmed** by `-ml` v1.16 §3.3,
  which moved its own wording rather than this plan's, closing v1.12's raise)*. A metric's kind is
  declared by **which per-item map it lives in** (`counts` or `measures`, §4 S1) and by the type its
  arm's aggregate carries for it — neither of which a manifest contains. `validate_pack` sees a
  manifest and no records, so it structurally cannot decide a family's kinds, and giving the
  manifest a `kind` field would be exactly the *infer the instrument from a pack field* the note
  refuses. So the family's kind is resolved in `compare_report`'s **pass 1**, from each member's
  aggregate type.

  **What the refusal refuses is the whole family's verdicts, not the offending members** *(v1.13;
  `-ml` v1.16 §3.3 corrects this plan's v1.12 wording, and the correction's reasoning is the point)*.
  Excluding the minority kind and verdicting the rest **shrinks `k` after the results exist**: `k` is
  `len(verdictMetrics)` and it is pre-registered, so the survivors would be corrected at a threshold
  weaker than the one declared — the fishing artefact pre-registration exists to prevent, arriving as
  a repair — and choosing *which* kind survives is itself a post-hoc instrument choice. So
  `compare_report` **names each member with the kind resolved for it, prints no verdict for any
  member of that family, and lets every one of their numbers through labelled `exploratory — no
  significance claim`** under this section's standing rule. **No arm is excluded and nothing is
  raised.** That is what distinguishes this from DC-10's per-member kind *disagreement* — an
  aggregate type contradicting the map the same metric's per-item values arrive in — where the arm's
  scorer has been shown to disagree with itself and the arm is excluded and named in the
  `INVALID RESULTS EXCLUDED` block (§4 S1, DC-10). A mixed family is a pack-authoring defect with a
  one-line fix; the arms' own numbers are not in question.

  **Where the label prints, named because the shipped home structurally excludes the members that
  need it** *(v1.14, plan-gate P7-3)*. `report.py:763-777` builds the Exploratory section from
  `m.name not in family` (`:767`) and emits one **name-only** line per metric (`:776`), so a refused
  family's members — which are *in* `family` — reach it today and print nothing. Two decisions, so
  the implementer is not left to make them and then write §5 test 11d against whichever they made.
  **(1)** The filter widens to `m.name not in family` **or** *this family's verdicts were refused
  whole* — and to that, not to "was not verdicted", because a member with no paired data is also
  unverdicted and must keep rendering `_NO_PAIRED_DATA` in place rather than migrating into a
  section that claims something else about it. **(2)** The line stays **name-only**: every member's
  per-arm figures are already printed unconditionally by the Arms table, and a second copy beside
  the label is a second home for one number (§7 rule 4). So "every one of their numbers prints"
  above is a statement about the Arms table, and the Exploratory line is what withholds the claim.
  One consequence, because it is the same question one screen down: the **headline** block
  (`report.py:744-751`) reads the headline member's verdict and falls back to `_NO_PAIRED_DATA` when
  it is absent — false for a refused family, which has paired data and a withheld claim. It prints
  the same `exploratory — no significance claim` label instead.

  **Two more sites in that path print a false claim for a refused family, and v1.14's closing
  sentence — *nothing else in the verdict path changes* — was wrong about both** *(v1.15, plan-gate
  P8-2)*. They were found by **enumerating** the path's renderers rather than by reading around the
  two already named, which is the only form of that sentence worth writing:
  `awk 'NR>=606 && NR<=780 && (/lines \+=/ || /lines\.append/)' modelbench/report.py` returns
  **11** emission sites at `5878014`, and **five of them, in four blocks, state something a
  whole-family refusal falsifies** — `:719` and `:738` (below), `:751` (the headline, named above)
  and `:770`/`:776` (the Exploratory section, named above). The other six are checked and named so
  the claim is over a set rather than an impression: `:666` is the **empty-intersection** member's
  own block, which a refused member does not reach because it *has* paired data; `:690` is the
  verdicted member's block, not rendered for a member with no verdict; `:756` is the no-headline
  note, which claims co-equality and never a verdict; `:741` and `:777` emit a blank line; and
  `:779` is the marginal-overlap footnote, a diagnostic about a line this path did not print.

  **(3) The `Family-wise error control` block renders on `len(family) > 1` and prints
  "Holm–Bonferroni across the {k} pre-registered verdict metrics, applied" (`report.py:711`, the
  prose at `:722-724`) — false whenever no ladder ran. Decision: the block renders only where a Holm
  ladder actually ran, and one line stands in its place where it did not.** *Rejected: rendering it
  with the prose and the decision cells relabelled.* Every column in that table — McNemar *p*,
  Holm-adjusted threshold, decision — is a **ladder artefact**, so a relabelled block is three
  columns of em-dash under a heading that still claims family-wise control, which is exactly the
  *print one rule, apply another* shape (ii) above refuses one screen up. **Two conditions reach the
  replacement line and they are different facts, so each gets its own**: a family **refused whole**
  (this paragraph's case) — no correction was applied and no member was verdicted — and an
  **all-continuous** family with `k > 1`, which takes its correction **in the interval** and never
  in a ladder, the case this section's own second paragraph authorises and §4 S1's loop already
  states (`holm_steps` is not called on a continuous family). The second is not new ground: it is
  the same shipped line, under the other condition that reaches it, and a fix scoped to the refusal
  alone would leave it printing the same false sentence for the family Table G exists to serve.

  **(4) `_decision(None, step)` returns `"no verdict — no paired data"` (`report.py:327-330`) —
  false for a refused member for the reason (1) gives. Decision: `_decision` keeps its two states
  and gains no third input, because (3) makes the false string *unreachable* rather than
  relabelled.** `_decision` has exactly **one** call site — `grep -rFn '_decision(' modelbench
  --include='*.py'` → **2** lines, `report.py:327` the definition and `:739` the call inside the
  family table's loop — so suppressing the block for a refused family removes the only path on which
  `v is None` means *withheld*. The `v is None` that survives is `:660-669`'s empty paired
  intersection, for which the string is true. **The reversal trigger is (3):** if a later revision
  renders that block for a refused family after all, the third input state — *verdict withheld*,
  distinct from *no paired data* — comes back with it, and this is the one place that says so.

  With those four named, the rest of the verdict path is unchanged: the member's `### <metric>`
  block still names its resolved kind and still prints its pairing tally.

  *(Naming, settled in v1.3 and aligned with the note. The retired field is the singular
  `primaryMetric`, and it is retired rather than redefined: re-pointing an
  established name at new semantics — "may now be `null`" — is its own trap, and pluralising it to
  distinguish the list would leave two fields one character apart in every manifest and every diff.
  `verdictMetrics` names what the field controls and collides with nothing; `headlineMetric` does
  the same for the presentation half.)*
- **`sampling` declares the analysis unit, and the harness never chooses one.** *(New in v1.4,
  closing gate finding N-1.)* `-ml` §3.4 Rule 1 makes `PairedOutcomes.from_units` raise on a repeated
  analysis-unit id — but **that guard only fires if the id passed in is the *cluster* key.** 48
  conversations drawn from 12 scripts have 48 *distinct conversation ids*, so a caller who passes
  conversation ids sees no error and gets an anti-conservative verdict from correlated rows. Rule 1
  is a backstop; the contract that actually closes it is here, in the pack:
  - **`sampling.pairingKey`** — the ordered component names of `ItemResult.pairingKey`
    (`["scriptId", "replicate", "turnIndex"]` for the tool-caller; `["itemId"]` for the four
    item-level packs). The pairing key stops being a scorer convention and becomes pack data.
  - **`sampling.analysisUnit`** — the independent unit of analysis, and it is fixed by a **rule**,
    not chosen per pack: **the analysis unit is the *outermost* component of `pairingKey` — the one
    whose *repetition* is what would indicate correlated rows.** For the tool-caller that is
    **`scriptId`, never a conversation id**; for the four item-level packs it is `itemId`. A rule
    rather than a per-pack list because a list goes stale the moment a pack is added, and because
    the rule is what a reviewer can check by reading. `report.py` resolves the unit id from this
    field and passes it to `from_units`; **no call site chooses it, and there is no parameter
    through which a caller could.** It is one declaration per pack rather than per metric because
    the rule is structural: every verdict metric in a pack shares the same independent unit.
  - **`sampling.seed` — the continuous-metric bootstrap seed, and nothing else** *(v1.10, from note
    v1.11)*. It has appeared in the manifest example above since v1.1 and no section said what it
    seeds. Under `-ml` §3.4 Rule 4's now-**binding** closed form the paired **binary** interval
    resamples nothing and takes no seed, so the field's object is `-ml` §3.2d's continuous-metric
    paired bootstrap — the embedder pack's `mrr` verdict, score separation, and any future
    continuous verdict metric. **No manifest changes and no field is added or removed:** the seed
    stays required, stays pre-registered in the content-hashed manifest for point (iii)'s reason,
    and is carried as `PackRef.seed` with no default, `validate_pack` refusing a manifest without
    it. Why a required field whose only consumer arrives with §3.2d's path is nonetheless the right
    call, and the discriminator under which it would be removed, are the note's (`-ml` v1.11's
    *"what retires, what stays"*) and are cited rather than restated.
  - **`validate_pack` enforces the rule by three independent routes** *(the third is v1.25's; two
    was the claim and P12-7 falsified it — a `tool-caller` fixture declaring
    `pairingKey: ["conversationId", "turnIndex"]`, `analysisUnit: "conversationId"` over 12
    conversation rows passed **both** existing routes while violating the half this bullet states in
    prose, and shipped as the suite's positive control. **The fixture itself was corrected at U76**,
    re-keyed to `["scriptId", "turnIndex"]` / `"scriptId"` with a pinning test — so a rework unit
    must not re-fix it and read the finding as closed. What route (iii) adds is the **rule**, which
    is what was missing: without it the next such fixture passes the same way)*.
    **(i) Structural:** `analysisUnit` must equal `pairingKey[0]`, and `pairingKey` is ordered
    outermost → innermost. This catches the pack that declares a correct unit against a
    wrongly-ordered key.
    **(ii) By the data — the check that catches a *consistently* wrong choice, which (i) cannot.**
    The row-count identity, computed over `analysisUnit`'s own values: the data file holds exactly
    `scripts × replicatesPerScript` rows, the number of distinct `analysisUnit` values is exactly
    `scripts`, and **each appears exactly `replicatesPerScript` times.** A pack that declared
    `analysisUnit: "conversationId"` under `scripts: 12, replicatesPerScript: 4` fails immediately —
    48 distinct values where 12 are required — which is precisely the N-1 shortcut, caught by
    arithmetic rather than by intent. It also catches the pack that declares
    `replicatesPerScript: 1` and ships four conversations per script, the case that slips past
    Rule 6's declaration check *and* past Rule 1 at once.
    **(iii) By the role — the check that catches the *consistently* wrong id, which neither (i) nor
    (ii) can** *(v1.25, impl-gate Pass 12 OQ-1)*. Routes (i) and (ii) are both satisfied by any
    self-consistent naming, so the *role-specific* half of this bullet — "for the tool-caller that
    is `scriptId`, never a conversation id; for the four item-level packs it is `itemId`" — was
    prose with no mechanism, and the defect it describes shipped. It is mechanised as a **second
    column on the existing closed role table**: `roles.ANALYSIS_UNIT_FIELD_BY_ROLE`, beside
    `UNIT_KIND_BY_ROLE`, read through `roles.analysis_unit_field(role)`, and `pairingKey[0]` must
    equal it.

    | role | `analysis_unit_field` | `unit_kind` (unchanged, `-ml` §3.3) |
    |---|---|---|
    | `tool-caller` | `scriptId` | `conversation` |
    | `guard-judge` | `itemId` | `item` |
    | `nlq-generator` | `itemId` | `item` |
    | `chat-responder` | `itemId` | `item` |
    | `embedder` | `itemId` | `query` |

    **Three things make this a rule and not the per-pack list this bullet rejects.** It is keyed by
    **role**, and the role set is closed and already lives in `roles.py` — adding a role is a harness
    change today, adding a *pack* is not, so FR-5 is untouched. It goes in `roles.py` rather than
    `packs.py` because a role's unit is a property of the role, which is the premise
    `UNIT_KIND_BY_ROLE` already rests on. And the two columns are **different vocabularies** — a
    `pairingKey` **component name** against a denominator **noun** — which is why the embedder's row
    reads `itemId` / `query` and why deriving either from the other is wrong; the function is named
    `analysis_unit_field`, with `field`, so a call site cannot read it as the noun.
    **Two assertions keep the table honest rather than decorative:**
    `set(ANALYSIS_UNIT_FIELD_BY_ROLE) == set(roles.ROLES)`, computed, so a sixth role cannot be
    added without a unit field; and the negative case is driven **over every role**, each with a
    `pairingKey[0]` borrowed from another row, by executing `check_sampling_contract` — never by
    reading the constant back.
- **Identity is `(packId, packVersion, contentHash)`.** `contentHash` = SHA-256 over the sorted
  relative paths and bytes of every file in the pack directory except `PROVENANCE.md`. Declared
  versions get forgotten; a hash cannot. `compare` flags a mismatch in **either** (AC-3), which
  also catches the nastier case: same declared version, different bytes. This mirrors
  `cypher-mcp`'s content-hash image tag — a stale pack should be unrepresentable.

  **The triple is total for a *loaded* pack, and only for one** *(v1.6, following the S1 fix round)*.
  Hashing the triple's third component means reading every file in the pack directory, which only
  `load_pack` does (§4 S2). The two carriers therefore differ deliberately: **`Pack.contentHash: str`
  is total** — a loaded pack always knows its bytes — while **`PackRef.contentHash: str | None`**
  carries `None` for a reference built from a manifest read alone, which is all S1's `compare` has.
  `None` means *"not loaded"*; it never means *"hashed to empty"*, and the two must stay
  distinguishable in the one field whose entire job is identity — the same absent-versus-empty rule
  §3.4.2 applies to fingerprints, one level up. Three consequences:
  - **`Pack.ref()` is the totality boundary** (§4 S2). A `PackRef` obtained from a loaded pack never
    has `contentHash is None`; that is `load_pack`'s postcondition and an S2 test, and it is what
    keeps the identity triple total exactly where this bullet claims it.
  - **Test `is None`, never truthiness.** `if ref.contentHash:` re-merges `None` and `""` and
    silently undoes the distinction the type was widened to express.
  - **No report path may substitute `PackRef.contentHash` for the AC-3 banner.** The banner's
    question is *what did these runs actually execute against*, which only each run's recorded
    `fingerprint.packContentHash` answers; the currently-loaded pack cannot, whatever it hashes to.
    (Judged rather than adopted: this is the engineering re-gate's reading, `docs/reviews/small-model-benchmarking-impl.md`
    Pass 2. It holds, and the plan already implied it — S2's `Pack` carried a total `contentHash: str`
    from v1.1 while `PackRef` is what `compare_report` takes. Dropping the field from `PackRef`
    entirely was the alternative, and it is rejected: it would make the report's type depend on
    whether a pack happens to be loaded, for a field the report is forbidden to read anyway.)
- **A pack may ship executable Python** (`tools/sim.py`), loaded via `importlib` from the pack
  directory. That is a deliberate plugin seam, not an accident: FR-10 requires ground truth from a
  dispatched-call trace and resulting state, which needs a real (simulated) tool implementation.
  The alternative — a declarative mini-language for tool behavior — would be a worse, buggier
  Python. Constraint: pack modules import only from stdlib and `modelbench.tooling`, enforced by
  **one mechanism, `validate_pack`'s AST walk** over every `Import`/`ImportFrom` node in the pack's
  Python files, checked against an allowlist. *(v1.1 also claimed "a `ruff` check … enforces it";
  that is not constructible under this component's own `select = ["E","F","W","I"]` — banned-import
  rules live in `TID`, which is unselected, and are a denylist rather than an allowlist. The claim
  is withdrawn.)* Two rules that make the AST check real rather than decorative: **`run` calls
  `validate_pack` first and fails closed**, so the check fires on a normal run and not only when
  someone remembers to type `model-bench validate`; and every subprocess launch in the tool
  (after §3.4.4a, `powershell.exe` is the only one left) is `subprocess.run([...], shell=False)`
  with an argv list — model
  keys reach the argv verbatim (`mistralai/ministral-3-3b`) and a `shell=True` slip is the only
  injection surface in the tool. Pack code is part of the content hash, so a behavior change to a
  simulated tool is a version change like any other.
- **A pack declares its environment.** `environment.requires` lists capability tokens
  (`lmstudio-chat`, `lmstudio-embeddings`). A run against a pack whose requirements are unmet
  fails fast with a named reason. No pack in this delivery requires FalkorDB (§3.8).
- **Packs are versioned in place.** Bump `packVersion`, edit files, hash changes; git history holds
  the old bytes. Recovering an old pack version to re-run it is a `git checkout` of that path —
  documented in `README.md`, not a feature.

### 3.4 D4 — The environment fingerprint, made enforceable (FR-7 / AC-2)

FR-7 says a result missing any part of its fingerprint is invalid and not merged into history. To
make that mechanical rather than aspirational, the fingerprint is a **required-field set that is a
function of `(benchSchemaVersion, armKind)`, checked on write and again on read**. Both keys are
new in v1.2 and each closes a gate finding; take them in order.

#### 3.4.1 `armKind` — a run is not always a model run (B-3)

FR-13/AC-5 require **every** embedding run to carry a keyword-only reference arm, reported as a full
paired arm with an interval (§3.8.1). BM25 has no model, no quantization and no runtime, so as a
`RunResult` it fails a model-shaped validation on write — and §3.4.2's rule is that there is no
"save anyway" flag. The resolution is a **discriminator, not an exemption**:

- `Fingerprint.armKind: "model" | "deterministic"`, required on every record, and
  the required and forbidden sets are both keyed by it — `REQUIRED_BY_SCHEMA[schema][armProfile]`
  (§3.4.3 adds the outer key) and `FORBIDDEN_BY_ARM_PROFILE[armProfile]`. `validate()` branches on
  the profile and never on field presence. **This is a rewrite of shipped, green S1 code and its
  sites are §4 S1e Table B** *(v1.10, plan-gate P4-2)* — in particular `ARM_KINDS` must stop being
  derived from the forbidden mapping in the same edit, or a mechanical re-key makes its members
  `{model:chat, model:embeddings, deterministic}`, `armKind == "model"` fails the membership test at
  `fingerprint.py:162`, and **every model record returns `FieldProblem("armKind", "unknown")` and
  refuses on write**.

  **The discriminator gained a second component in v1.9, and the key is now a *profile*** (G3-1).
  `armKind` keeps its two values and its meaning — every `armKind == "model"` filter in this plan is
  unchanged, including `models --tested`'s — and a second required field, **`callSurface: "chat" |
  "embeddings"`**, joins it on `model` records. The mapping key is the derived token
  `armProfile = armKind if armKind == "deterministic" else f"{armKind}:{callSurface}"`, so schema 1
  has exactly three profiles: `model:chat`, `model:embeddings`, `deterministic`. Like `armKind`,
  `callSurface` is a **discriminator and therefore a member of no required set** — present on every
  `model` record, checked before any mapping is consulted, and so incapable of appearing in a derived
  forbidden set. §3.4.4a owns where its value comes from (declared by the pack, cross-checked against
  the catalog) and what each profile requires; this section owns only the shape.
- **`model`** requires the full auto-captured set below plus the four operator-attested fields.
- **`deterministic`** requires exactly eleven fields: `armId` (`"bm25"`), `armParametersHash`
  (SHA-256 over the pack's declared arm parameters — tokenizer, stopword list, `k1`, `b`, IDF
  variant), `packId`, `packVersion`, `packContentHash`, `benchVersion`, `benchSchemaVersion`,
  `pythonVersion`, `hostOs`, `startedAt`, `endedAt`. (`armKind` is the discriminator itself: present
  on every record and checked before either mapping is consulted, so it is a member of **neither**
  required set and can never appear in a derived forbidden set.)
- **The forbidden set is a derivation, never a list.** *(v1.5; generalised in v1.9 from two kinds to
  three profiles — the rule is the same rule, and it had to stop being a pairwise difference the
  moment there were more than two sides.)* For each profile, at each schema version:

  > **`FORBIDDEN_BY_ARM_PROFILE[p] = (⋃ required(q) for every other profile q at that schema) −
  > required(p at that schema)`** — a record may not carry a field that only *some other* profile is
  > required to have.

  The implementer writes the set operation, not the resulting names —
  `frozenset().union(*(REQUIRED_BY_SCHEMA[v][q] for q in profiles if q != p)) - frozenset(REQUIRED_BY_SCHEMA[v][p])`
  — so adding a field to any profile at schema 2 forbids it on the others for free. What it resolves
  to at schema 1:

  | Profile | Forbidden |
  |---|---|
  | `deterministic` | every model field, the sampling settings and all four operator-attested fields — everything in §3.4.2 that is not one of the eleven above |
  | `model:chat` | exactly `{armId, armParametersHash}` — unchanged from v1.5, because `model:embeddings`' required set is a strict subset of `model:chat`'s and contributes nothing new to the union |
  | `model:embeddings` | `{armId, armParametersHash, runtimeName, runtimeVersion, temperature, maxTokens}` (§3.4.4a) |

  The last row is the derivation earning its keep: nobody wrote those four names down, and forbidding
  them is exactly right — an embeddings call has no `runtime` object to observe and no sampling
  parameters to obey, so a record carrying either is claiming something it cannot have measured. The
  **test** pins all three sets against independently written literals, so a set that *shrinks* fails
  loudly rather than silently shrinking the suite (S1 impl-gate M-4).

  The forbid half is the point of the whole discriminator: it is what makes
  `{"modelKey": "bm25", "quantization": "n/a"}` — the shortcut a time-pressed implementer reaches
  for, and exactly the "invalid result reaching a report unlabelled" this design exists to prevent —
  fail loudly on write instead of quietly becoming a sixth model in the history.

  *(Why a derivation. Through v1.4 this bullet carried a hand-typed list of fourteen model fields
  beside the words "forbids every model field". The two never agreed: §3.6's eligibility-gate
  decision put `modelType`, `modelCapabilities` and `modelCapabilitiesPresent` into the model set
  (§3.4.2) and nobody swept the list here. The S1 implementer forbade all three anyway, following
  the stated intent, and the implementation gate confirmed the plan was the stale side (§4 item 3).
  Two review passes had certified the list before that — both read it against itself rather than
  against the set it claims to complement, which is what a hand-maintained enumeration invites. A
  set difference cannot drift from its own intent, so the intent is now the only thing written
  down.)*
- A `deterministic` arm is **reproducible from `(packContentHash, armParametersHash, benchVersion)`
  alone**, which is why host state is not merely optional for it but forbidden: recording a KV-cache
  setting beside a BM25 score would imply the score depends on it.
- Consequences downstream, all decided here in S1 because S3 consumes them: `runId` for a
  deterministic arm is `<packId>-<armId>-<UTC-timestamp>`; the arm is computed and stored on **every**
  embedding run (FR-13 says "every"), sharing the model run's `sessionId` so the pairing label in
  §3.7 still applies; `compare_report` accepts model and deterministic runs in one
  `Sequence[RunResult]` and labels the deterministic one `reference arm (deterministic given pack
  version)`; two deterministic arms are **never** the subject of a verdict against each other; and
  `models --tested` (FR-17a) filters to `armKind == "model"`, so BM25 can never be offered as a
  reference model.

#### 3.4.2 The fields, and *absent* versus *empty* (M-3)

**Auto-captured** (`fingerprint.capture()`, no human input, cannot be wrong without the tool being
wrong): `modelKey` (the literal LM Studio id, never normalized), `modelPublisher`, `arch`,
`quantization`, `compatibilityType`, `maxContextLength`, `loadedContextLength`, `runtimeName`,
`runtimeVersion`, `residencySource`, `residentModelsAtStart[]`, `residentModelsAtEnd[]` (the last
three from §3.4.4a's residency probe: the source token the probe answered on, and the two snapshots
it produced), `modelType`, `modelCapabilities` and `modelCapabilitiesPresent` (the catalog's raw
`type` and `capabilities` verbatim, plus the absent-vs-empty bit below — §3.6's gate decision must
be auditable after the fact), `temperature`, `maxTokens`,
`packId`, `packVersion`, `packContentHash`, `benchVersion`, `benchSchemaVersion`, `pythonVersion`,
`hostOs`, `startedAt`, `endedAt`.

**This section owns the model field set.** The 26 names above plus the four below **are**
`REQUIRED_BY_SCHEMA[1]["model:chat"]`, 30 in total; `REQUIRED_BY_SCHEMA[1]["model:embeddings"]` is
**those 30 minus `runtimeName`, `runtimeVersion`, `temperature` and `maxTokens`**, 26 in total, for
the reasons §3.4.4a gives — and §3.4.1's forbidden sets are derived from both — so
a field added here is forbidden on a deterministic arm without a second edit anywhere, and no list
elsewhere in this plan needs sweeping to keep up.

**The pack's `sampling.seed` reaches the fingerprint transitively, through `packContentHash`, and no
seed field is added** *(v1.11, plan-gate P5-10)*. `-ml` §3.2d says the seed *"goes into the
environment fingerprint (FR-7)"* and no seed field exists — a gap `-ml` §3.4 Rule 4 flags as
pre-existing and unaffected by its own ruling. It is unaffected by that ruling; it is **not**
unaffected by S3, which is why it is answered here. The answer is the mechanism already in place
rather than a thirty-first field: `sampling.seed` is a manifest field, the manifest is inside
`content_hash(root)`'s input (§3.3 — SHA-256 over every path in the pack but `PROVENANCE.md`), and
`packContentHash` is `REQUIRED_NONEMPTY` on all three profiles. So a run whose seed differed carries
a **different `packContentHash`**, a record that lost it is refused on write, and AC-3's banner
already declares the two runs not apples-to-apples. What the transitive route does not buy is a seed
a reader can see without the pack in hand; that is the cost, recorded rather than paid for with a
field. **It is resolved now and not after S3 for the same reason everything else in this section is:**
the set is closed at 30/26 and free only while `results/runs/` does not exist — a thirty-first field
added after S3 costs a `migrate`. The wording of the note's own §3.2d sentence is
`data-scientist`'s; the plan side is this paragraph.

**`lmsCliCommit` → `residencySource`: a one-for-one swap, not an addition** *(v1.8, §3.4.4a)*. The
count is unchanged at 26 + 4 = 30. `lmsCliCommit` recorded *which `lms` build produced the residency
snapshot*; after §3.4.4a nothing in the harness runs `lms`, so the field has no source and — under
this section's own rule — could only be kept by defaulting it to `""`, which is precisely the
silently-defaulted fingerprint field FR-7 exists to refuse. It is therefore **removed from
`REQUIRED_BY_SCHEMA[1]["model:chat"]` and `["model:embeddings"]`**, not blanked. `residencySource` replaces it with the same
question honestly answered: a `REQUIRED_NONEMPTY` token naming the surface the residency and catalog
data actually came from (`"lmstudio-api-v0"` is the only value this build emits), so a record read a
year from now says how it was fingerprinted rather than leaving a reader to assume.

**It is free only now**, because `results/runs/` does not yet exist, so no stored record is
being invalidated and no `migrate` step is owed (the same argument §7 makes for
`BinaryMetric.unit`). Doing it one stage later would not be free. **It is an S1-local edit to a
shipped module, and its sites are enumerated in §4 S1e Table A** *(v1.10. v1.8 named two sites;
v1.9's G3-9 table named four and was still incomplete elsewhere in the same revision — plan-gate
P4-2. §7 rule 5 now requires an edit table to carry the grep that enumerates it, which is what
Table A carries. This section states the field change; where it lands is Table A's.)*

**One of those sites fails *silently*, and that is the reason a completeness property matters here
rather than care.** `tests/conftest.py`'s `residentModelsAtEnd` fixture declares a residency element
in the retired `lms ps --json` shape — `{modelKey, sizeBytes}`, whose only source v1.8 removed —
where §3.4.4a's shape is `{id, state}`. The field is `REQUIRED_PRESENT`, which checks presence and
**never element shape**, so the stale shape validates, ships green, and travels into S2, where
`residency()` emits `{id, state}` and the two disagree with nothing to catch them. The other three
sites fail loudly the moment they are missed. The structural fix is therefore **not** the fixture
edit but the missing assertion — **S1 done-condition 1's element-shape check** — which holds whether
or not anyone remembered the fixture. *(v1.10, plan-gate P4-14: v1.9 put that emphasis in a
"fails loudly if missed?" **column**, answering "no — and this is the one that matters", where a
reader scanning the column reads three yeses and one no. The emphasis belongs to the row, so the row
is now a paragraph and the column is gone.)*

**Operator-attested** (§6 R-1 — no programmatic source exists): `lmStudioAppVersion`,
`kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads`. These live in a local, gitignored
`model-bench/host.json` (§3.4.4) and are **copied into every run record**, so a record is
self-contained and history stays readable when the box changes.

**Validation distinguishes three states per field, not two.** v1.1 said `validate()` raises on
"missing, empty, or `null`", which rejects legitimate values: the residency probe returns `[]` when
nothing is loaded — verified on this box, all 19 catalog entries at `state: "not-loaded"` (§2.5) —
so `residentModelsAtStart: []` is the correct and informative value on a clean
box; and the catalog omits the `capabilities` key entirely for several models rather than sending an
empty list. So each required field is declared in one of two tiers:

- **`REQUIRED_PRESENT` — the key must be present; `[]`, `0`, `false` and `""` are all valid values.
  This list is closed** *(v1.9, G3-13)*: `residentModelsAtStart`, `residentModelsAtEnd`,
  `modelCapabilities`, `modelCapabilitiesPresent` (`false` is a real answer, and the whole reason the
  field exists), `temperature` (0.0 is the pinned value for four of the five packs),
  `otherResidentWorkloads`. Six fields, matching what S1 shipped — **and the six are `model:chat`'s
  closed list, not a global one** *(v1.10, plan-gate P4-10)*. The tier is already per-profile by
  construction: it is `FieldSpec.tier` inside `REQUIRED_BY_SCHEMA[schema][armProfile][field]` (§4
  S1), so each profile carries its own column of the same mapping. What each resolves to at schema 1:
  `model:chat` the six above; **`model:embeddings` five** — the six minus `temperature`, which that
  profile forbids; **`deterministic` none** — its eleven fields are all `REQUIRED_NONEMPTY`. A
  per-profile tier is therefore expressible, and §3.4.4a's open `loadedContextLength` question
  depends on its being so.
- **`REQUIRED_NONEMPTY` — the key must be present *and* the value truthy — is the complement, by
  rule and not by list:** every required field not named in the six above is `REQUIRED_NONEMPTY`.
  So `modelKey`, `quantization`, `runtimeName`, `runtimeVersion`, `residencySource`, `packId`,
  `packVersion`, `packContentHash`, `benchVersion`, `lmStudioAppVersion`, `kvCacheSetting` and every
  other field in this section. **A field added to any profile states its tier in the same edit**, and
  states it by joining the closed list or by saying nothing. *(Written as a complement for §7 rule
  4's reason: two hand-maintained lists beside each other drift, and a tier now decides whether a run
  refuses rather than merely how a record reads.)*

**Schema 1 has no *optional* tier, and that is a design decision rather than an omission**
*(v1.10, plan-gate P4-10)*. Every field is required at some tier on every profile that carries it,
and §3.4.1's union-of-others-minus-mine derivation then makes *not required by me* mean **forbidden
on me** — which is exactly the property that stops `{"modelKey": "bm25"}` reaching a report. The
cost is that "permitted but not required" cannot be said, and the resolution when a field turns out
to be conditionally absent is **not** to drop it from a profile's required set (which forbids it,
so a future LM Studio build that *does* return it would refuse a correct capture) but to move it to
that profile's `REQUIRED_PRESENT` column, captured `""` — the same route `modelCapabilities` already
takes for a key the catalog omits. §3.4.4a applies this to `loadedContextLength`. Adding an optional
tier is rejected: a fourth state per field buys one case and costs the derivation its meaning.

`null` is invalid in **both** tiers — it is the shape of "we did not capture this", which is the one
thing FR-7 refuses. A field the catalog genuinely omits is captured as `[]` or `""` by `capture()`,
never passed through as `null`, and `modelCapabilities` records the distinction it needs in
`modelCapabilitiesPresent: bool` beside it.

**Capture ordering is part of the contract, because two fields only exist at one point in time —
and the order itself is §3.4.4a's, cited here rather than stated a third time** *(v1.11, plan-gate
P5-4; §7 rule 4)*. `loadedContextLength` appears in the catalog only once a model is loaded, and
residency is the thing that changes across a run. The consequence this section owns is the field
one: **the catalog is read twice, and only the second read can populate `loadedContextLength`** —
step 3a, before the warm-up, supplies the twelve catalog fields that exist on a cold model and
carries the pre-load refusals; step 8, after the warm-up returns, re-reads the same `catalog()` call
for `loadedContextLength` **alone**. `residentModelsAtStart` is snapshotted before the warm-up (§3.6
— under JIT auto-load the warm-up *is* the load) and `residentModelsAtEnd` after the last scored
call. A `capture()` that takes `loadedContextLength` from the **step 3a** read produces a record with
a null value for it and fails its own validation, which is correct behaviour and is why the read is
split rather than left to the implementer.

*(v1.11 rewrites this paragraph. Through v1.10 it stated a **single** post-warm-up catalog read and
added that "a `capture()` that reads the catalog first … fails its own validation — which is correct
behaviour" — which forbids §3.4.4a's step 3a outright. The two sections then gave opposite
instructions on the one sequencing question plan-gate P4-5 was raised about, and an implementer
following this one would have written a single post-warm-up read and lost **both** pre-load refusals,
paying a full JIT load to refuse a model that cannot serve the pack. It is §7 rule 4's own failure
shape — an owning section changed, its derived surface not swept — occurring in the revision that
added rule 5, which is why the order is now cited and only its field consequence restated.)*

#### 3.4.3 `benchSchemaVersion`, `benchVersion`, and not deleting history (M-4)

**`benchVersion` = `modelbench.__version__`**, which S0 pinned to the installed distribution's
metadata (`project.version` in `pyproject.toml`, asserted by `tests/test_package.py`). That
assertion is load-bearing, not filler: a skew between the two would silently mislabel `benchVersion`
in every stored record.

**`benchSchemaVersion` is a separate integer constant in `modelbench/results.py`, starting at `1`**,
never derived from `benchVersion` and never bumped by a release. It increments only when the
required-field set or the on-disk record shape changes in a way a *reader* must branch on.

v1.1 said an "older-schema" record is excluded on read. That is wrong as stated, and against FR-3
directly: the first time a required field is added, every result stored before it would be
quarantined out of every comparison, and the tool's entire value is that "a new model's result lines
up against models I tested months ago". The rule instead:

- `REQUIRED_BY_SCHEMA: Mapping[int, Mapping[str, FieldSpec]]` — an older record is validated against
  **the contract it was written under**, and passes if it satisfied it.
- `invalid` is reserved for a record that fails **its own** schema (hand-edited, truncated,
  blanked field) or declares a `benchSchemaVersion` this build does not know — a record from the
  *future*, which is the genuinely uninterpretable case.
- A report comparing records across schema versions prints a `SCHEMA VERSIONS IN THIS COMPARISON`
  line naming them, in the same spirit as AC-3's pack-version banner: visible, never silent, and
  never a reason to drop the record.
- `model-bench migrate` exists for a genuinely breaking change: it rewrites stored records forward,
  in place, recording `migratedFrom` on each. `README.md` states that a schema bump is a deliberate
  act — a new `REQUIRED_BY_SCHEMA` entry, a `HISTORY.md` line, and a decision about whether a
  migration is needed — not a side effect of adding a field.

#### 3.4.4 `host.json` — the operator-attested file (M-8)

Local, gitignored, written by `model-bench attest` (§3.6a), read at the start of every `model` run:

```json
{
  "schemaVersion": 1,
  "apiBaseUrl": "http://localhost:1234",
  "attested": {
    "lmStudioAppVersion": "0.3.31",
    "kvCacheSetting": "f16",
    "hostRamGb": 16,
    "otherResidentWorkloads": ["docker: falkordb-dev", "windows desktop session"]
  },
  "attestedAt": "2026-09-02T14:05:00Z",
  "observedAtAttestation": {"residencySource": "lmstudio-api-v0"}
}
```

`attested` is exactly the four FR-7 fields with no programmatic source; `observedAtAttestation` is
what the staleness trip-wire compares against; `apiBaseUrl` is §3.4.4a's one endpoint setting and
replaces v1.7's `lmsPath` — the file no longer names a filesystem path at all.

**`observedAtAttestation` carries only what `attest` can actually observe, and `runtimeName` /
`runtimeVersion` are not among them** *(v1.10, plan-gate P4-6 — v1.9 wrote all three into this
example and S2's done-condition asserted a trip-wire that could not be built)*. `attest` probes
`GET /api/v0/models` and `GET /v1/models` (§3.6a); **no catalog entry carries a `runtime` key** — the
2026-09-03 probe's key union is §2.5's ten names and the gate re-probed it independently — and
`runtime` exists only on a chat-completions response. `attest` takes no model argument, so there is
no model to issue such a call against, and adding one would JIT-load an arbitrary model at
attestation time (3.6–21 s, §2.5) for a field the run itself observes for free at capture-order
step 5. So:

- **At attestation the two runtime keys are *absent*** — the key is not written, never `""`. This
  file is outside the fingerprint, so §3.4.2's "absence is not representable" rule does not reach it;
  `host.json` has its own three states and this is the one that says *never observed*.
- **The first `model:chat` run that reaches capture-order step 5 back-fills them**, writing
  `runtimeName`, `runtimeVersion` and a sibling `runtimeObservedAt` timestamp into
  `observedAtAttestation` and touching nothing else — never the `attested` block, never `attestedAt`,
  which attest nothing new. `run` writing to `host.json` is the one write it makes outside
  `results/`, it is inside `model-bench/` (FR-23), and it is confined to those three keys.
- **The trip-wire compares from the second such run onward**, and the first run's state is **named in
  the record rather than silently equal**: `RunResult.attestationTripWire` (§4 S1) is
  `"first-observation"` on that run, `"compared"` thereafter, and `"unavailable"` on a
  `model:embeddings` arm, which has no runtime to observe at all. A deterministic arm carries `None`
  — it attests nothing and skips the check (§3.4.5 point 3).
- **Rejected: writing the two keys empty at attestation.** `""` collides with *observed and empty*,
  and it makes the trip-wire either never fire (if `""` matches everything) or fire on every run (if
  it does not) — the same absent-versus-empty defect §3.4.2 exists to prevent, one file over.

#### 3.4.4a The source of truth for the auto-captured fields — and what happens when it is absent

*(New in v1.8, forced by §2.5. This section owns the **source**; §3.4.2 owns the **field set**; §3.6
owns the adapter that calls it.)*

> **`GET {apiBaseUrl}/api/v0/models` — LM Studio's native catalog — is the single source of truth
> for every auto-captured catalog and residency field. Nothing falls back to it and nothing falls
> back from it. `GET /v1/models` is a *reachability discriminator only*: it decides which error the
> harness prints, never the value of a field. When the native endpoint does not answer, a `model`
> run refuses before its first model call and writes nothing.**

**But the catalog is one of five sources, and this section governs all five** *(v1.9, G3-2)*. v1.8
wrote the rule above as though the catalog supplied the auto-captured half; it supplies **thirteen
of the twenty-six**. Both of the plan gate's Pass 3 blockers were instances of that single gap rather
than disagreements with the rule, so the map comes first and the rule applies to each row:

| Source | Fields | Exists from |
|---|---|---|
| **`GET /api/v0/models`** — the catalog | `modelKey`, `modelPublisher`, `arch`, `quantization`, `compatibilityType`, `maxContextLength`, `loadedContextLength`, `modelType`, `modelCapabilities`, `modelCapabilitiesPresent`, `residencySource`, `residentModelsAtStart`, `residentModelsAtEnd` (13) | any time — **except `loadedContextLength`**, which appears only once the model is loaded |
| **the chat response's `runtime` object** | `runtimeName`, `runtimeVersion` (2) | only after a **chat-surface** call has returned — hence `model:chat` only |
| **the run's own configuration** — pack manifest + CLI | `temperature`, `maxTokens` (2) | before the run, and meaningful only on a chat-surface call — hence `model:chat` only |
| **`host.json`** (§3.4.4) | the four operator-attested fields (4) | before the run |
| **the process and the pack loader** | `benchVersion`, `benchSchemaVersion`, `pythonVersion`, `hostOs`, `startedAt`, `endedAt`, `packId`, `packVersion`, `packContentHash` (9) | before the run; `endedAt` after the last scored call |

13 + 2 + 2 + 4 + 9 = 30, which is §3.4.2's `model:chat` set exactly. **The converse of the refusal
rule is false and an implementer will otherwise assume it**: "the catalog answered, therefore the
fingerprint can be completed" does not follow, because four other sources have to answer too. §3.6's
adapter table says "the catalog half" for the same reason.

**Capture order, pinned** *(v1.9, G3-2; it was implicit across §3.4.2, §3.4.4 and §3.6, and where the
staleness trip-wire fires depends on it)*. One `model` run, in order:

0. **`startedAt` is stamped** — UTC, at the instant `run` begins, before step 1, so it brackets every
   step below including the refusals *(v1.10, plan-gate P4-5: v1.9's list named no step for it)*.
   `endedAt` is stamped at step 10's completion, after the last scored call and before `store()`.
1. **Read `host.json`** — the four attested fields, `apiBaseUrl`, `observedAtAttestation`. Absent or
   schema-invalid → exit `5`, nothing run.
2. **`probe()`** → `api-v0` / `v1-only` / `unreachable`. The last two → exit `3` with their two
   distinct messages, **before any model call**.
3. **`residentModelsAtStart` ← `residency()`**, and `residencySource` is set to the surface that
   answered.
3a. **`catalog()`** — the twelve remaining catalog fields (`modelKey`, `modelPublisher`, `arch`,
   `quantization`, `compatibilityType`, `maxContextLength`, `modelType`, `modelCapabilities`,
   `modelCapabilitiesPresent` and the rest of the source table's first row, all but
   `loadedContextLength`), **plus the refusals that must precede the load, each with its own
   scope**: the `callSurface`-versus-catalog-`type` cross-check below, **on every `model` run**, and
   §3.6's tool-calling eligibility gate, **on a `tool-caller` pack only**. Whichever refusal is in
   scope exits before step 4 — exit `4` — so a model that cannot serve
   the pack is refused **without paying a JIT load**, which is the expensive refusal this list exists
   to avoid. *(v1.11, plan-gate P5-1: v1.10 wrote "both refusals" unscoped, which applied §3.6's
   `type ∈ {"llm","vlm"}` predicate to every model run and so refused **every** `model:embeddings`
   arm here — including `text-embedding-qwen3-embedding-0.6b`, the model §4 S3's first
   done-condition names as the plan's first real run. The universal pre-load refusal is the
   cross-check; the tool-use rule is a role-specific addition on top of it, and §3.6 now says so in
   its own words. This is G3-1's defect — the embedder unbuildable — arriving a second time through
   the sentence written to close plan-gate P4-5, which is why the scope is now stated on both
   sides.)* *(v1.10, plan-gate P4-5. v1.9's list ran `residency()` and then jumped to a "catalog
   re-read" at step 8: `residency()` is the catalog **filtered on** `state != "not-loaded"`, so on a
   cold start the model under test is absent from it, twelve of the thirteen catalog fields had no
   step at all, step 8's word "re-read" referred to a read the list did not contain, and both
   refusals landed after the load. The cross-check's cost — "one comparison against data step 3
   already fetched" — is true of this step's response and was false of `residency()`'s return type.)*
4. **The warm-up call**, call-surface aware (below), under `firstCallTimeoutSeconds`.
5. **Read `runtimeName`/`runtimeVersion` from the warm-up response's `runtime` object** — chat
   surface only. This is the earliest moment they exist, which is the whole reason this list exists.
6. **The staleness trip-wire fires here** (§3.4.5 point 3) — the first instant *both* its comparands
   are available, the observed one at step 5 and the attested one at step 1. Mismatch → exit `5`,
   **before a single scored item is consumed**. On the first `model:chat` run against a freshly
   attested `host.json` the attested comparand is **absent** rather than stale, so this step
   back-fills it instead of comparing and records `attestationTripWire: "first-observation"` (§3.4.4)
   — v1.9 asserted this step was "the first instant its comparands are available" having checked only
   one of the two (plan-gate P4-6). The cost of putting it here is one warm-up call; the cost of
   putting it anywhere later is a run's worth of items.
7. **`residency()` again.** This snapshot — *not* `residentModelsAtStart` — is the contamination
   guard's baseline for item 1 (§3.6, G3-4).
8. **Catalog re-read** — the same `catalog()` call as step 3a — for `loadedContextLength` **alone**,
   which does not exist before step 4.
9. **The scored items**, with the guard's between-item probe from step 7 onward.
10. **`residentModelsAtEnd`** after the last scored call, with the three retries below.

**`callSurface` — the second discriminator, and where its value comes from** *(v1.9, G3-1)*. §3.4.1
owns the profile key's shape; this owns its provenance and its consequences.

- **It is declared, never observed.** The value comes from the pack's `environment.requires` (§3.3):
  `lmstudio-chat` → `"chat"`, `lmstudio-embeddings` → `"embeddings"`. `validate_pack` refuses a pack
  declaring **neither or both** — a pack needing both surfaces would need a per-call profile, which
  is out of scope, and the refusal is what keeps it out. Declared rather than observed for the same
  reason R-15 gives one level up: a contract selected by what a probe found is a contract that cannot
  be checked before probing.
- **It is cross-checked against the catalog, and the mismatch is refused before any call. This is
  the one pre-load refusal that runs on *every* `model` run** *(the scope is v1.11's, plan-gate
  P5-1)*. The predicate, written out because it is now load-bearing rather than illustrative:
  `callSurface == "chat"` requires the catalog `type` to be in `{"llm", "vlm"}`;
  `callSurface == "embeddings"` requires it to be `"embeddings"`. Anything else — including a
  `type` this build has never seen — fails at run start with both values named, exit `4`, the
  pack/environment code. It costs one comparison against data **step 3a** already fetched. A pack
  declaring `lmstudio-chat` against an embeddings model is the case the rule was written for; a
  future fourth `type` is refused rather than guessed at, which is §3.4.2's absent-versus-empty
  discipline applied to a discriminator. §3.6's tool-calling gate is a **narrower, role-specific
  addition** on top of this one, not its sibling: it asks a second question (*can this model emit
  tool calls?*) of the models this check has already admitted to the chat surface, and it runs only
  on a `tool-caller` pack.
- **What each profile requires and forbids** is §3.4.2's and §3.4.1's respectively. The substance:
  `model:embeddings` is `model:chat` minus `runtimeName`, `runtimeVersion`, `temperature`,
  `maxTokens`, and those four are **forbidden** on it rather than merely absent. That is the
  difference the blocker turned on — v1.8 left them required with no source, so
  `POST /api/v0/embeddings` (which returns `object`, `data`, `model` and `usage`, and **no**
  `runtime`, `model_info` or `stats` — LM Studio's own v0 REST documentation) could not produce a
  storable record and S3, the plan's "first end-to-end result", was unbuildable.
- **The warm-up is call-surface aware too**, or the mandatory per-arm warm-up has no instantiation
  for an embeddings pack: see §3.6.
- **One field stays open, deliberately: `loadedContextLength` on an embeddings model.** It is
  `REQUIRED_NONEMPTY` and §2.3's evidence that it appears on load is from a *chat* model. It stays in
  the `model:embeddings` required set provisionally, and **S2's R-1 probe reads a loaded embeddings
  model as well as a loaded chat one**. **If the key is absent there the field moves to that
  profile's `REQUIRED_PRESENT` column — captured `""` — and *not* out of the required set**
  *(v1.10, plan-gate P4-10; v1.9 said "the field moves out of that set", which under §3.4.1's
  union-of-others-minus-mine derivation makes it **forbidden** on `model:embeddings`, so a later
  LM Studio build that did return it would refuse a correct capture)*. The tier is per-profile
  (§3.4.2), so the move costs one `FieldSpec` and leaves `model:chat` untouched; it is recorded with
  a `HISTORY.md` line. Free either way, because no embedder record exists before S3. Guessing it now
  is the one thing that would not be free.
- **The trip-wire is chat-surface only, and that is a real narrowing rather than an oversight.** Step
  6 compares `runtimeName`/`runtimeVersion`/`residencySource` — from the second `model:chat` run
  onward, the first having back-filled the attested side (§3.4.4); on an embeddings arm the first two do
  not exist, so **only `residencySource` is comparable there — and it *is* compared** (§3.4.5 point 3,
  v1.25), which is a real check on a field that today takes one value and a real one on the day it
  takes two, but no check at all on the runtime. The alternative — issuing a chat-surface probe call
  to obtain a runtime — would JIT-load a second model on a 16 GB box, evicting the one under test at
  a whole model load a side, which trades a stronger check for a corrupted measurement. Recorded as a residual in
  R-1 rather than papered over.

**`residentModelsAtStart` / `residentModelsAtEnd`** are that catalog filtered on
`state != "not-loaded"`, each surviving entry recorded as `{id, state}` with the **literal `state`
string** kept. Not `state == "loaded"`, and not normalised to a boolean: the 2026-09-03 probe
observed only the one value (§2.5), so any other state LM Studio can report is a value this plan has
not seen, and a filter written against the value it *has* seen would silently reclassify it. `[]`
remains the correct, informative clean-box answer — and it is `[]` *because a probe succeeded and
found nothing*, which is a different fact from "no probe ran" and must stay distinguishable exactly
as §3.4.2 requires (the same absent-versus-empty rule, one level up again).

**Why this and not the three alternatives.** All three were considered explicitly, because this is
the one decision in v1.8 with a real trade-off.

- **Keep the `lms` CLI.** Rejected. It still works (§2.5), so this is not a provisioning gap and it
  does not route to `devops` — it is that the *dependency* is bad on three axes at once: it is
  reachable only through a globbed Windows path (`/mnt/c/Users/*/…`), which is a host-layout
  accident rather than a contract; it costs **0.30 s against 1.7 ms** for the same fact, which is
  what makes a between-item residency probe (§3.6) affordable in one design and not the other; and
  it is a **second surface to keep honest** beside the HTTP one the adapter already targets, on a
  box where the two can disagree. A benchmarking tool whose fingerprint depends on which machine's
  `PATH` it runs on has a fingerprint that does not travel.
- **Use only `/v1/models`, staying on the OpenAI-compatible surface.** Rejected, and this is the
  one that looks principled until it is costed. `/v1/models` returns `{id, object, owned_by}` and
  nothing else (§2.5). Choosing it means either **dropping nine of §3.4.2's model fields** —
  `modelPublisher`, `arch`, `quantization`, `compatibilityType`, `maxContextLength`,
  `loadedContextLength`, `modelType`, `modelCapabilities`, `modelCapabilitiesPresent`; `modelKey` is
  the only one `/v1` can answer — which is most of FR-7's model identity and would also take §3.6's
  tool-calling eligibility gate with it, or **defaulting them**, which is the exact failure this
  component exists to refuse. Portability bought at the price of the value claim is not a
  trade-off; it is the wrong side of it.
- **`/api/v0/models` with a `/v1/models` fallback that populates what it can.** Rejected as a
  *source*, kept as a *diagnosis*. As a source it buys nothing on the success path — a record
  carrying `modelKey` and nothing else fails `validate()` and is refused anyway (§3.4.5 point 1) —
  while inviting the one edit that would break the component: "the fallback already fills half the
  record, let it fill the rest with reasonable values." As a diagnosis it is worth its ten lines,
  because the two failure modes need different messages and only this call distinguishes them:
  - **neither endpoint answers** → exit `3`, *"LM Studio is not reachable at `<apiBaseUrl>`"*;
  - **`/v1/models` answers and `/api/v0/models` does not** → exit `3`, *"`<apiBaseUrl>` serves an
    OpenAI-compatible API but not LM Studio's native `/api/v0` catalog. `model-bench` fingerprints
    from that catalog and will not record a run it cannot fingerprint (§3.4.4a)."* — the message a
    person pointing the tool at an older LM Studio, a proxy, or another server actually needs.

**The proprietary-dependency question, answered rather than dodged.** `/api/v0` is
LM-Studio-specific and this makes the harness depend on it. Three things make that the right call
rather than a reluctant one. **(i)** The adapter is *already* on that surface for its hottest path:
`POST /api/v0/chat/completions` has been the chat route since v1.1 (§3.6) because `stats`,
`model_info` and `runtime` — FR-11's TTFT and FR-7's runtime identity — exist nowhere else. v1.8 adds
no new *class* of dependency; it removes one (the CLI) and leaves a single surface instead of two.
**(ii)** What the plan promises is one provider. §1 keeps cloud/hosted models out of scope and §3.3
leaves the seam for a second provider in the pack's `environment.requires`, not in a
lowest-common-denominator adapter. **(iii)** The reversal is designed for rather than hoped for: a
second provider does **not** arrive by making these fields nullable — it arrives by making the
required-field set a function of provider as well as `armKind`, which is §3.4.1's discriminator
pattern applied a second time, with `residencySource` the field that keeps today's records readable
once there is more than one answer. That is why `residencySource` is required now, while there is
only one value it can take.

**Absence, stated once for both halves of the record** — because the rule inverts at the fingerprint
boundary and an implementer who carries one convention across it will get the other backwards:

| Where | Representation of "not captured" | Consequence |
|---|---|---|
| **Inside the fingerprint** (§3.4.2's required set) | **Not representable.** `null` is invalid in both tiers, `""`/`[]` mean *captured and empty* | The run **refuses**: no partial record reaches `results/runs/`, because `store()` has no bypass (§3.4.5 point 1). Refusing to run is how this component records an absent fingerprint field |
| **Outside it** (FR-11's speed block, `ItemResult.latencyMs`) | **`None`/`null`, mandatory** — never `0`, never omitted | The figure is excluded from its aggregate **and that aggregate's coverage is printed** — `-ml` §11.7's slots, never a percentile over an unstated subset |

**One asymmetry the implementer must handle rather than discover.** The start-of-run probe fails
*before* any model call, so refusing costs nothing. `residentModelsAtEnd`'s probe runs *after* the
last scored call, where refusing costs the whole run. The rule does not bend for that — a record
missing a required field is not stored — but the failure is made unlikely rather than merely
accepted: the end-of-run probe **retries three times at one-second intervals** before the run is
declared unfingerprintable, and the refusal message names what was lost and points at the retained
transcript (`results/transcripts/<runId>.jsonl`, written as the run proceeds). A transient blip on a
localhost GET should not cost twenty minutes of model calls; a genuinely gone endpoint should cost
exactly that, rather than buying a record nobody can stand behind.

**The one field that loses its source outright is `modelSizeBytes`** (loaded-weights size, previously
`lms ps --json`): `/api/v0/models` does not carry a size, so it is recorded as absent under the
second row above and never as `0`. It is not a fingerprint field, so nothing else follows; R-2 is
amended to say so. `peakHostRssBytes` is unaffected — it was never a CLI reading.

#### 3.4.5 Three enforcement points, all cheap

1. **Write refuses.** `results.store(run)` calls `fingerprint.validate()` and raises on any field
   that fails its tier, on any field forbidden for its `armKind`, and on a `null` anywhere in the
   required set. There is no "save anyway" flag.
2. **Read quarantines.** `results.load_history()` re-validates every record against its own
   `benchSchemaVersion` and returns `(valid, invalid)`. `compare` merges only `valid` and prints an
   `INVALID RESULTS EXCLUDED` block naming each excluded record and why. This is AC-2's actual test
   surface — a hand-edited record must be excluded on *read*, not merely rejected on write.
3. **Attestation staleness is detected, not trusted.** `host.json` records the `runtimeName`,
   `runtimeVersion` and `residencySource` observed when it was last attested. If any differs at run
   time, the run stops
   with "LM Studio changed since you last attested `host.json` — re-check the app version and KV
   cache setting, then `model-bench attest`." That converts the weakest link (a human typing a
   value once and never revisiting it) into a loud failure at the moment it goes stale. A
   `deterministic` arm skips this check entirely — it attests nothing.

   **When it fires is §3.4.4a's capture-order step 6** *(v1.9, G3-2)*: immediately after the warm-up
   returns, which is the first moment `runtimeName`/`runtimeVersion` exist and the last moment before
   any scored item is consumed. `host.json` is still read at step 1; what step 6 adds is that the
   *comparison* cannot happen earlier, and must not happen later.

   **`residencySource` is compared on every path the check runs at all, and "degenerates to
   `residencySource` alone" means *compare it*, never *give up*** *(v1.25, impl-gate P13-7 / Pass 13
   OQ-1 — the code at `40a9bc8` read it as give-up and returned `"unavailable"` without comparing
   anything)*. Two reasons, and the second is the plan's own scope statement one paragraph down:
   the field is present on both sides of every non-deterministic arm — `attest` always writes
   `observedAtAttestation.residencySource`, and §3.4.4a's step 3 observes one **before** the
   warm-up, so an embeddings run has it too — and the trip-wire's stated triggers are an
   inference-runtime change **or a change of capture surface**, of which `residencySource` is the
   only comparand that carries the second. Not comparing it makes that half of the stated scope
   uninstantiated on the embeddings surface and half-uninstantiated on the first chat run, opening
   exactly when §3.4.4a says the field starts earning its keep — "once there is more than one
   answer".

   **The check has three outcomes, not two, because `attest` cannot observe a runtime** *(v1.10,
   plan-gate P4-6 — §3.4.4 carries the schema and the reasoning)*, **and the outcome names the
   check's *coverage* while `stale` carries its *result*: they are different columns and neither
   substitutes for the other** *(v1.25)*. `"first-observation"`: the attested runtime side is
   absent, so step 6 **back-fills** `runtimeName`/`runtimeVersion`/`runtimeObservedAt` into
   `host.json` and the run proceeds — the two runtime keys are being established and are not
   evidence of anything, but `residencySource` **is** compared, and on a difference the run exits
   `5` **and `host.json` is not written**: a baseline runtime must not be recorded against attested
   values the operator is about to redo. `"compared"`: all three present — equal proceeds, a
   difference in any one exits `5`. `"unavailable"`: a `model:embeddings` arm, where the two runtime
   comparands do not exist; the string names *those two*, `residencySource` is still compared, and a
   difference exits `5` on this path as on the others. So the stored outcome is unchanged from
   v1.10 on all three paths and the *reach* is not: what a reader of a record learns from
   `"unavailable"` is which comparands were missing, not that nothing was checked. The outcome is
   stored on the run as `RunResult.attestationTripWire`, so a reader of a stored record can tell a
   run that was checked from one that established the baseline — a first-observation run is **not**
   evidence that the runtime was unchanged, and a record that did not say so would imply it was.

   *(v1.8: `lmsCliCommit` was the third comparand and is gone with the CLI (§3.4.4a). The trip-wire
   is therefore **narrower and stated as such**: it fires on an inference-runtime change or a
   change of capture surface, and no longer on an `lms` CLI update. That is a smaller net, and
   arguably a better-aimed one — the KV-cache setting is a property of the loaded runtime, which
   `runtimeVersion` tracks directly, whereas the CLI commit only ever correlated with it. The
   residual — an app update that changes the KV-cache default while keeping the same runtime
   version — is now undetected, and R-1 says so rather than implying the trip-wire covers it.)*

### 3.5 D5 — Storage: JSON per run is the truth, CSV and markdown are derived

FR-2 wants durable, human-readable, comparable; the stakeholder's stated preference is "CSV or
markdown, either is fine". A single flat file cannot hold per-turn × per-failure-kind breakdowns
without becoming unreadable, so:

- **`results/runs/<runId>.json` — the record of truth.** One file per run: fingerprint, pack
  identity, per-item and per-turn scored records, aggregate counts, timings. Append-only in
  practice: a run file is never rewritten (`model-bench migrate` is the one exception, §3.4.3).
  `runId` = `<packId>-<modelSlug>-<UTC-timestamp>`, where **`modelSlug` is the `modelKey` with every
  character outside `[A-Za-z0-9._-]` replaced by `-`** — real keys contain `/`
  (`qwen/qwen3-4b-2507` → `qwen-qwen3-4b-2507`) and a raw key would create a directory. The slug is
  a filename convenience and is never the identity: `fingerprint.modelKey` keeps the literal id and
  is what every comparison reads (§2.3, R-8). Timestamp is `YYYYMMDDTHHMMSSZ`. For a deterministic
  arm the middle segment is `armId` (§3.4.1).
- **`results/index.csv` — one row per run**, the human-openable summary: runId, date, role, packId,
  packVersion, packContentHash(8), modelKey, quantization, n, headline metric(s),
  `latencyMsP50`, `latencyMsP95`, `latencyMsMax`, `latencyTimedCount`, `latencyItemCount`,
  **`callCount`**, valid/invalid. **Every latency cell is copied from the run's own `LatencyBlock` and never
  recomputed here** *(v1.10)*: the block is where `-ml` §11's estimator and both floors are applied,
  so a percentile computed in the index builder bypasses both — which is what the shipped
  `_index_row` does today, and §4 S1e Table C is the edit. **A latency cell is empty exactly when the
  record's field is `None`** — no
  qualifier, no zero, no placeholder — which is what makes a populated cell readable without its
  prose (`-ml` §11.6, §11.8). `latencyMsMax` is a v1.9 column carrying the tail figure on any run
  where `-ml` §11.3's identity floor **renames** the p95 rather than refusing it; **it is `None` on
  every run of every pack this plan declares**, because that rename is reachable only at `Y ≤ 21`
  and the smallest declared `Y` is 30 (`-ml` §11.3, swept there — the bound is the note's and is
  not re-derived here). The column still earns its home on the narrower argument: it is what keeps
  the `p95` **label** honest the day a pack with `Y ≤ 21` is declared, and the alternative is
  discovering at that point that the maximum has nowhere to live. *(v1.27 withdraws v1.9's
  justification rather than softening it — "the tail figure is the sample maximum at every
  tool-caller run's sample size" substituted the pack's analysis-unit count for its item count,
  which `-ml` v1.20 corrects and which is this component's signature defect; the same sentence is
  withdrawn at §6 R-13.)* **Three count columns join the figures, and the third is a correctness
  requirement rather than a readability one** *(v1.27, `-ml` §11.8)*: `latencyTimedCount` and
  `latencyItemCount` are coverage a CSV reader cannot otherwise see, while **`callCount`** is what
  tells that reader whether the `latencyMsP50` cell in front of them is a per-call figure or a
  per-turn one. Without it one column silently holds two estimands — the same defect the gate
  closes at the value, one column over. **Derived and fully regenerable** (`model-bench index rebuild`), so it is never a
  second source of truth to keep honest — the same argument §3.1 makes about golden sets, applied
  internally.
- **`reports/<pack-id>-<date>-<n>.md` — generated comparisons**, committed because they are the
  artifact a human actually reads six months later. Never hand-edited. `<n>` is a two-digit
  same-day sequence starting at `01`, chosen by scanning existing files — a same-day re-run is the
  normal case while a pack is being developed, and silently overwriting the earlier comparison is
  the one behaviour a tool built around durable history must not have.
- **`results/transcripts/<runId>.jsonl` — raw model output**, gitignored. Useful for a post-hoc
  "why did it fail", not needed for any comparison, and large.

**FR-20 is enforced structurally, not by convention.** `results.load_history()` takes a `packId`
and there is no API to load across packs; `report.py` has no code path that aggregates two roles;
`compare` requires `--pack`. There is deliberately no `model-bench leaderboard` command, and
`README.md` says why. The same shape of refusal carries the two other "the report must not be able
to say this" rules: no path emits a pooled 85-item guard accuracy (§3.8.2), no path emits a blended
tool-calling percentage (§3.8.4), and no path synthesises a headline for a pack whose
`headlineMetric` is `null` (§3.3). Each is a **missing function**, not a guarded one — §3.5's
argument about `index.csv` applied to the report: a rule you cannot express is a rule you cannot
break under deadline pressure.

### 3.6 D6 — Model access: LM Studio adapter

`lmstudio.py`, stdlib `urllib.request`, four operations:

| Operation | Endpoint / command | Notes |
|---|---|---|
| `catalog()` | `GET /api/v0/models` | Full per-model metadata (§2.3, re-probed §2.5) — **the catalog half** of the fingerprint's auto-captured fields, 13 of 26, and the only source for those 13 (§3.4.4a's five-source table names the rest). |
| `chat(messages, tools=…, temperature=…)` | `POST /api/v0/chat/completions` | **`/api/v0`, not `/v1`** — the `stats`/`model_info`/`runtime` objects only exist on the v0 route, and they are FR-11's TTFT and FR-7's runtime identity. |
| `embed(texts)` | `POST /api/v0/embeddings` | Batched; records dimension from the first vector (AC-5). |
| `residency()` | `GET /api/v0/models`, filtered `state != "not-loaded"` | §3.4.4a. Measured at **1.7 ms**, which is what makes a between-item probe affordable — see the contamination guard below. |
| `warm_up(model)` | **call-surface aware** (§3.4.4a): `POST /api/v0/chat/completions` for a `model:chat` arm, `POST /api/v0/embeddings` for a `model:embeddings` one | Under JIT auto-load this **is** the load (§2.5). There is no unload on either HTTP surface, so the harness cannot force a cold state. On the chat surface its response also carries `runtime` — capture-order step 5. |

- **Endpoint resolution** is one explicit setting (`host.json: apiBaseUrl`, default
  `http://localhost:1234`), and §3.4.4a's two-step probe decides the error when it does not answer.
  Never silently degrade to "no residency data" — that would put a hole in the fingerprint, and
  §3.4.4a makes the hole unrepresentable by refusing the run instead.
- **The unit boundary — stated once, here, and inherited by every `…Ms` field in this plan**
  *(v1.10, plan-gate P4-1)*. **LM Studio's `stats` object reports seconds**: its own v0 REST
  documentation gives `"time_to_first_token": 0.111`, `"generation_time": 0.954`. Every timing field
  this plan names ends in `Ms` and is **milliseconds** (`-ml` §11.7 — *"the unit is milliseconds and
  the block never prints seconds"*). The conversion is this document's to state, because the harness
  surface is the plan's (§7 rule 2), and it is stated **at the transport boundary so no caller ever
  sees a raw seconds value**: `lmstudio.ChatResult` normalises on construction.
  - `ChatResult.ttftMs = 1000 × stats.time_to_first_token`
  - `ChatResult.generationMs = 1000 × stats.generation_time`
  - `ChatResult.tokensPerSecond = stats.tokens_per_second` — **the one figure that is not
    converted**, because a per-second rate already is what its name says.
  - `ChatResult.wallClockMs` — the client wall clock, measured by the harness, never by LM Studio.
  - **Each derived field is `None` when its source key is absent, never `0`, and `ChatResult`
    construction never raises on a missing or partial `stats`** *(v1.11, plan-gate P5-8)*. The four
    conversions above are stated as equations because that is what a unit boundary is; read as
    *unconditional* they make a `stats`-less chat response a crash. It is not: §4 S2's rule (iv-a)
    says `statsCoveredCount == 0` **keeps its meaning on the chat surface, where it says every
    response lacked `stats` and is a real signal** — an expected state the report is committed to
    printing, so the transport boundary must be able to produce it. Same rule for `promptTokens`
    (`usage.prompt_tokens`) and for a `stats` object present but missing one key. This is §3.4.4a's
    absent-versus-empty discipline on the far side of the fingerprint boundary: outside the
    fingerprint, absence is `None`/`null` and mandatory (§3.4.4a's second row).
  - The raw `stats` mapping is carried verbatim beside them for auditability, and **no runner,
    scorer or report path reads a timing figure out of it** — the derived fields are the only source.
  - `coldLoadSeconds` is the single seconds-valued figure in the tool: a warm-up wall clock, reported
    separately, never inside the FR-11 block (`-ml` §11.7).

  **Why this is a blocker and not a naming preference.** `-ml` §11.5.1's gap computed on unconverted
  operands is `latencyMs` minus about 1.1, so **every call slower than about one second crosses its
  1 000 ms threshold**. §2.2's measured pack turns are ~1.3 s, so a tool-caller run would withhold
  100 % of its latencies, reach `X = 0`, and print §11.7's `X == 0` slot — *"no item's timing
  survived"* — under the model-load cause: a units bug reported to the operator as contamination.
  It also fails **selectively**, which is what makes it survive a test suite: the minimal warm chat
  calls `-ml` §11.4 measured at 55–115 ms stay far below the threshold, so any fixture or fast pack
  built at that scale looks correct while the long-pole pack — the one the figure exists for —
  withholds everything. v1.9's own FR-11 table carried both readings in adjacent rows. §5 test 15b asserts the conversion against a stubbed `stats`, which is the only
  thing that catches this class.
- **The first call to each arm is an explicit warm-up: never timing data, never a scored item.**
  *(v1.8. The measurement that forces it: a cold `POST /v1/chat/completions` at `max_tokens: 10`
  took **21.068 s**, essentially all of it JIT load, against sub-second warm calls — §2.5. Any
  single per-request timeout tuned to warm latency fires on the first call of every run and kills
  it.)* Seven parts *(five at v1.8; v1.9 adds the timeout disposition and `unexplainedMs`)*, and
  everything from the fourth on exists so the fix cannot contaminate what the benchmark reports:
  - **The warm-up is outside the item set, and it is call-surface aware** *(v1.9, G3-1)*. `runner`
    issues exactly one warm-up request per arm before the first scored call. On a **`model:chat`**
    arm that is the pack's system prompt plus a fixed, pack-independent probe message at
    `temperature: 0`, `max_tokens: 10`; on a **`model:embeddings`** arm it is one
    `POST /api/v0/embeddings` over a single fixed short string, timed identically — an embeddings
    model serves no chat route and the embedder pack declares no system prompt, so without this the
    mandatory warm-up has no instantiation for the very pack S2 exists to unblock. **Its *content* is
    discarded, not its metadata**: it is not an `ItemResult`, it is not written to
    `results/transcripts/`, and it enters no aggregate — but on the chat surface the response's
    `runtime` object **is** read, because it is the only source of `runtimeName`/`runtimeVersion`
    (§3.4.4a step 5), and its `stats` are recorded for the R-1 measurement below. v1.8's "the
    response is discarded" was the sentence that made the runtime identity sourceless.
  - **Two timeout budgets, not one.** `firstCallTimeoutSeconds` (default **300**) governs the
    warm-up; `requestTimeoutSeconds` (default **120**) governs every scored call. The first is an
    order of magnitude above the *slowest* load measured rather than a tight fit around it: two
    cold calls on this box came in at 3.6 s and 21.1 s for two different models (§2.5, `-ml` §11.4),
    a 27B at a long prompt takes the same load path, and the spread is the point — the budget is
    sized against a load cost that **varies by model and by page-cache state** and is predictable
    from neither (`-ml` §11.5, §11.4; plan-gate P4-11 — v1.9 wrote "per model" alone and §2.5 wrongly
    called the two measurements the same call surface), never against either figure; the
    second is sized to warm generation and **is** meant to fire on a hung call. Neither is a
    fingerprint field: they decide whether a run completes, never what it measures, and §3.4.2's
    set is not extended for them.
  - **`coldLoadSeconds` is the warm-up's wall clock, recorded only when the model was not resident
    at start** (`residentModelsAtStart`, §3.4.4a). When the model was already loaded it is
    **absent** — `None`, never `0` and never the warm number — because the harness can no longer
    unload, so such a run has genuinely not measured a cold load. Cold-load and steady-state are
    never averaged together.
  - **The contamination guard, because a warm-up alone is not enough.** JIT can reload *mid-run* —
    TTL expiry, or the operator loading another model — and a reload inside a timed call adds a
    whole model load to that item's latency (seconds, and how many varies by model: 3.6 s and 21.1 s
    are the two measured on this box) while leaving its correctness untouched. So `runner` calls `residency()`
    **between items and never during a timed call** (1.7 ms; the same discipline the
    `powershell.exe` sampler already follows for the same reason), and for any item whose preceding
    snapshot did not show the model resident it **withholds that item's `latencyMs`**. The
    item is still scored — only that one measurement is absent, which is exactly what §3.4.4a's
    second row is for. Two clauses make that precise, and both were review findings:

    **(a) The baseline for item 1 is the residency probe taken *after* the warm-up returns**
    (§3.4.4a capture-order step 7), **never `residentModelsAtStart`** *(v1.9, G3-4)*. The start
    snapshot is taken *before* the warm-up, so on any cold-start run it is `[]` by construction —
    the warm-up is the load. Read the other way, every cold-start run would withhold item 1 and
    report a contamination event that did not occur. The figures would still print (`-ml` §11.6
    sweeps this and the floors hold either way), which is precisely why it needed fixing rather than
    catching: a true number beside a false cause is this project's signature defect.

    **(b) Withholding is per *item*, and it takes `latencyMs` — the client wall clock — and nothing
    else** (`-ml` §11.4, which answers G3-3 and owns the rule). `ttftMs`, the prefill input and
    `tokensPerSecond` come off the same response but measure generation that happened *after* the
    load, so they are **kept** and printed with **their own denominator**, normally `Y of Y` beside
    the wall clock's short one. G3-3's defect is real and its fix is the missing denominator, not a
    nulling.

    *(This reverses what v1.9 first wrote, and the reversal is the process working. G3-3 proposed a
    free measurement — record the warm-up's own `stats` beside its wall clock on a known-cold load —
    and until it existed the conservative rule (withhold all four) was the right one. It was then
    taken: `-ml` §11.4 tabulates a cold call whose load is entirely invisible to LM Studio's own
    `stats`, against 50 warm calls. **LM-Studio-side TTFT excludes the load.** Keeping the
    conservative rule after that would discard good measurements to honour a caveat that no longer
    describes reality. The figures are the note's and are not repeated here.)*
  - **What is printed, and where the rule lives.** The figures, the denominator, the cause split,
    the two floors and the refusal cases are `-ml` §11's — **§11.7's slot grammar verbatim**, cited
    here and not restated, the same way §3.9 cites §3.2e's verdict strings. *(v1.8's own sketch,
    `p95 = … (latency n = 34 of 38)`, is withdrawn rather than corrected: under `-ml` §11.6's level floor
    that run prints no figure at all, so the illustration was wrong in exactly the way a restatement
    goes wrong — §7 rule 2.)* What this plan owns is that the numbers have a home in the record:
    see the latency block in §4 S2 and §3.5's `index.csv` columns.
  - **A scored call that hits `requestTimeoutSeconds` is a censored observation, not a
    measurement** *(v1.9, G3-5; `-ml` §11.5 depends on this disposition)*. Four clauses:
    **(i)** the item is **scored per the pack's rule** and the run continues — the call produced no
    output, which is evidence about the model, so its outcome is `fail`, never `n_a`; scoring it
    `n_a` would drop it from its denominator and let a model that hangs out-score one that answers
    wrongly, which is §5 test 7's laundering rule pointed at the clock. **From v1.27 this is the
    *only* non-completion that scores `fail`**, and the reason is the one in this clause: a timeout
    is the single mechanism where the harness gave the model its whole declared budget and observed
    nothing come back. The fourth disposition's outcome clause moves to `unrunnable` below (`-ml`
    §4.3 rule 4). **(ii)** Its `latencyMs` is
    **withheld**: storing `latencyMs = 120000` would put the *timeout
    constant* into the tail, where it is by construction the largest value, so the report would
    print a figure about the configuration rather than about the model. Its three siblings are
    absent too, for a different reason worth keeping distinct — a timed-out call returned no
    response, so no `stats` object was ever produced to keep. In (b) a figure exists and is dirty;
    here nothing exists. The item still carries an **`ItemTiming`** with
    `withheldFor: "timeout"` (§4 S1) — absence of a measurement is not absence of a record. On a
    single-call item that is its only populated field; on a **turn** it also carries the
    `CallTiming`s of the iterations that *did* return before the timing-out one, which are real
    per-call measurements and stay (v1.27) — what does not exist is the turn's own `wallClockMs`,
    which is `None` because the turn is incomplete.
    **(iii)** The count is
    carried separately from the load-contamination count, in `latencyWithheldForNoResponse` (§4 S2),
    because the two demand different operator actions (`-ml` §11.7 slot 2 prints both).
    **(iv)** After a timeout the runner
    re-probes: `probe()` returning anything but `api-v0` means the server went away, so the run
    exits `3` rather than scoring a server outage as a model failure. **The warm-up's own timeout**
    is not a scored call at all — exit `3`, nothing written, message naming `--first-call-timeout`.
    *(The alternative the note also accepts — abort the run, `MT` pinned at 0 — is rejected here: it
    discards every item already scored, and a hung model at item 30 of 38 is a result worth keeping,
    not an operational failure.)*
  - **A scored call that fails *without* timing out — the fourth disposition, and without it both
    `LatencyBlock` invariants are falsifiable** *(v1.10, plan-gate P4-7)*. A non-2xx response, a
    dropped connection or an unparseable body — an HTTP 500 on context overflow, or LM Studio
    restarting, over a twenty-minute run on a 16 GB box — is not a timeout, and v1.9 dispositioned only
    timeouts. Such an item has no `stats` and no trustworthy wall clock, so under v1.9 it was withheld
    under *neither* named cause: §4 S2's invariants (iii) and (iv) both fail on it and `-ml` §11.7's
    two-cause split stops summing to `M`. An implementer writing those invariants as `assert`s
    crashes the run on the first 500; writing them as computed values miscounts silently and prints a
    false cause line. **For the *timing record* the disposition is the timeout's, verbatim**, because
    from the record's point of view the two are one situation — **the call returned no response**:
    it carries **no timing figure at all**, neither
    wall clock nor `stats`-derived siblings — but it does carry an **`ItemTiming` whose only
    populated field is `withheldFor: "no_response"`** *(v1.11, plan-gate P5-9: v1.10 wrote "no timing
    at all", which reads against §4 S1's `timing is None` **iff** the arm produces no timings at all
    and against §4 S2 rule (vi), which derives both withheld counts from `timing.withheldFor` values
    and so needs the record to exist. The two were reconcilable and an implementer should not have to
    reconcile them — absence of a measurement is not absence of a record)*; and it is counted in
    **`latencyWithheldForNoResponse`**, which is
    what the timeout count was already measuring and is renamed to say so (§4 S2, Appendix A). Then
    invariants (iii) and (iv) hold unchanged and §11.7's grammar stays exhaustive with no third cause.
    **Unlike a timeout it does not re-probe and does not exit `3`**: an error response is evidence
    the server is answering, so the run continues; a *connection* failure re-probes exactly as clause
    (iv) specifies, because that is the case where the server may be gone. **The printed label is the note's and it has ruled**: `-ml`
    v1.12 §11.7 states that widening this counter to any scored call that returned no response makes
    slot 2's cause read `no response <MT>` and moves nothing else in the grammar.

    **The *scored outcome* is `unrunnable`, not `fail`, and that one clause is what v1.27 reverses**
    *(`-ml` v1.21 §4.3 rule 4, and §4.3.1 item 3 naming this sentence)*. v1.10 wrote *"the
    disposition is the timeout's, verbatim"* and carried the timeout's `fail` across with the
    record-keeping — but the two are one situation only for the **record**, and two for the
    **scoring**. A timeout is the one non-completion where the harness gave the model its whole
    declared budget and observed nothing come back; a non-2xx, a dropped connection and an
    unparseable body are channel failures the harness cannot attribute — a `400` from a model's
    runaway message list and a `400` from a malformed harness payload are the same status code —
    so all three are `-ml` §4.1's **`unrunnable`**: out of every scoring denominator, counted,
    printed. §3.8.4's disposition block is where the discriminator and the *why it is not an escape
    hatch* argument live, and `cleanThroughTurnH`'s third state (§3.8.4) is the one hole that had
    to close with it. **Everything else P4-7 bought is untouched and must not be reopened by a
    reader who finds this reversal**: the third `withheldFor` value, the single
    `latencyWithheldForNoResponse` counter, invariants (iii) and (iv), the `ItemTiming` that exists
    without a figure, and the re-probe/exit-`3` asymmetry above. P4-7 was a `LatencyBlock`
    accounting fix and the outcome sentence rode along with it.
    **This paragraph and the withholding bullet below answer two different questions about the
    same turn, and they are deliberately not merged** — one says what the turn *scores*, the other
    what its *timing record* holds. Collapsing them is how the contradiction this reverses was
    made.

    **One counter, three item states — the merge was right for the total and wrong for the item**
    *(v1.11, plan-gate P5-6, ruled by `-ml` v1.14 §11.5.1)*. v1.10 folded the timeout into
    `no_response` in **both** places, and the second fold cost the record a distinction §11.5.1's
    `censoringExact` depends on: its first clause is satisfied by a withheld item that is a
    **timeout**, and `report.py` could no longer tell a 120 s timeout from a 40 ms HTTP 500. The
    consequence was silent and always in the same direction — a run whose only withholding was a
    timeout rendered slot 3's **weaker** string where the note says the stronger one is owed, and
    neither document recorded that it had happened. The note's ruling makes the reason substantive
    rather than mechanical: **a timeout is a *censored* observation and a call that failed at 40 ms
    is a *missing* one**, so the predicate also gains an explicit **false** branch for the second,
    which v1.12 had no branch for at all. So:
    **`ItemTiming.withheldFor` becomes `Literal["load", "timeout", "no_response"] | None`** (§4 S1,
    Appendix A) — the item state is three — while **`latencyWithheldForNoResponse` stays one
    counter** over the last two values, because §11.7 slot 2 prints one `no response` label either
    way and the note has already ruled that. §4 S2 rule (vi) derives that one count from **two**
    `withheldFor` values, which is the only arithmetic this costs. The predicate and both slot-3
    strings are the note's and are not restated here.
  - **The in-call reload detector — `unexplainedMs` — which closes R-14's acknowledged residual.**
    The between-item probe structurally cannot see a reload that begins *and* ends inside one timed
    call. The same measurement that reversed (b) supplies the detector, because the field the load
    hides in is exactly the field `stats` does not see: the runner computes **`-ml` §11.5.1's gap**
    **per call** — over that call's `wallClockMs`, `ttftMs` and `generationMs` as the unit boundary
    above normalises them — **sums the gaps over the item's calls** *(v1.27: at `callCount == 1`
    that is v1.9's expression unchanged, and a single-call role cannot tell the two rules apart,
    which is why the choice is not left to an implementer)*, and **withholds `latencyMs` when the
    sum exceeds the threshold §11.5.1 sets**,
    counting the item under §11.7's
    **model-load** cause — the same cause, a second way of detecting it, which is why it opens no
    third entry in the published cause split. Three things this plan owns and the note does not:
    **placement** (here, in the runner, beside the residency guard rather than instead of
    it — the probe sees a reload *before* an item and the gap sees one *inside* any of its calls;
    **the gap is measured per call and the withholding decided per item**, v1.27, because the
    subtrahends are properties of one generation and the thing withheld is the item's admitted
    wall clock); the **field
    name**; and that `unexplainedMs` is **stored per item and its maximum reported** even when it is
    below the threshold, so the threshold — which the note names as a starting value on two cold
    observations, not a derived constant — can be re-checked against the first real pack run rather
    than re-derived from scratch. The metric, the threshold and its basis are `-ml` §11.5.1's and
    are not restated here. *(v1.10, plan-gate P4-9: v1.9 wrote the formula out in bold inside the very
    sentence that says it is not restated, and in a third spelling — `generationMs`, a name that
    until v1.10 appeared nowhere else. The three ownership claims and the field name are what this
    plan owns; the arithmetic is the note's, and its operands are named once by the unit boundary
    above.)*

    **The detector is chat-surface only, and the narrowing is real** *(v1.10; the sibling of
    plan-gate P4-13)*. `POST /api/v0/embeddings` returns no `stats`, so `ttftMs` and `generationMs`
    do not exist on a `model:embeddings` arm and the gap cannot be computed. On that arm the
    between-item residency probe is the **only** contamination guard, `latencyWithheldForLoad` counts
    only what the probe saw, and an in-call reload is undetected exactly as R-14 describes for the
    general case. Recorded here rather than left for an implementer to discover when the subtraction
    meets a `None`.
  - **Withholding is at the unit of the figure, and the loop makes that a real distinction rather
    than a restatement** *(v1.27, `-ml` §11.9 ask 7)*. `latencyMs` is withheld **per item**; the
    three `stats`-derived figures lose or keep coverage **per call**; and **no call is ever
    withheld for load** — §11.4's measurement is that the load sits outside `ttftMs` and
    `generationMs`, and nothing about how many calls an item has changes that. Four consequences,
    all of which an implementer would otherwise have to derive:
    - **A reload *between* two iterations of one turn costs nothing.** It is invisible to the
      between-item probe (which runs between items) and sits outside every call's wall clock, so
      the sum above would miss it — but LM Studio loads **JIT on the request**, so whatever was
      unloaded is re-loaded inside the *next* call, where that call's own gap sees it (`-ml`
      §11.5.1). No guard is owed here, and this sentence exists so nobody builds one.
    - **What does sit between the calls is the harness's own tool dispatch and message assembly,
      and it is not foreign time.** It is real time the operator waits through and it belongs
      inside the turn's latency; the identity worth asserting rather than assuming is
      `ItemTiming.wallClockMs ≥ Σᵢ wallClockMsᵢ` (§5 test 10b).
    - **A turn whose loop ended on a call that timed out or returned nothing is *not timed*.** Its
      wall clock measures only the iterations that happened to succeed, so it is **incomplete**:
      `ItemTiming.wallClockMs` is `None`, the partial turn wall clock is never stored as a
      measurement, and `latencyMs` is withheld under `timeout` / `no_response` exactly as a
      single-call item's is. §3.8.4's table gives the map from the three failing dispositions onto
      those two `withheldFor` values.
    - **A `cap-hit` turn is the opposite case and must not be folded into it.** Every one of its
      `maxIterationsPerTurn` calls returned, its wall clock is complete, and it is timed and ranked
      like any other item. The two are one word apart in prose and opposite in the record, which is
      why they are two bullets.

  Two rejected alternatives, both of which contaminate the data they are meant to protect. **One
  widened timeout (300 s everywhere)**: it removes the only signal that a warm call has hung, and it
  lets a mid-run reload pass unremarked into the percentiles — the contamination the guard exists
  for. **v1.1's `warmupTurns` (drop the first *n* scored latencies)**: it discards real measurements
  chosen by *position* rather than by evidence, and leaves *n* as a knob that moves the reported p50.
  `--warmup <n>` survives only as *additional warm-up calls before the first item*, never as items
  removed after the fact.
- **Peak RAM (FR-11)** — see §6 R-2 for the honesty caveat. Captured as two separate, named fields:
  `modelSizeBytes` (**no source as of v1.8** — it was `lms ps --json`'s loaded-weights figure and
  `/api/v0/models` carries no size, so it is recorded absent per §3.4.4a and never as `0`) and
  `peakHostRssBytes` (sampled from
  `powershell.exe Get-Process`, best-effort). Neither is presented as "the model's RAM cost" without
  its method label. **Sampling is suspended for the duration of every timed call**: the runner
  samples *between* turns, never during one, and each sample carries its timestamp so a reader can
  confirm no sample overlaps a timed window. This is not fastidiousness — a `powershell.exe`
  round-trip across the WSL boundary was measured at **0.18–0.54 s** (gate Appendix A.3) against
  ~1.3 s per turn in the prior experiment, so an interval sampler running beside the call would
  perturb the very latency it sits next to by a double-digit percentage.
- **The full FR-11 surface, with each field's source named.** FR-11 lists five things and v1.1
  accounted for three; all five, plus the one thing FR-11 demotes and the two diagnostics this plan
  adds:

  | Field | Source | Status |
  |---|---|---|
  | `latencyMsP50` / `latencyMsP95` / `latencyMsMax` | **client wall clock**, measured around the HTTP call from just before the request to the last byte of the body, over items whose timing block survived; estimator, denominator, both floors and the printed grammar are `-ml` §11's | **headline**; an **item** figure, and on a `tool-caller` an item is a *turn* — so it is the whole turn's wall clock, every iteration and the tool dispatches between them included (v1.27: "client wall clock" no longer disambiguates it, `-ml` §11.4) |
  | `ttftMs` | **`ChatResult.ttftMs`** = `1000 × stats.time_to_first_token` — the unit boundary above, and the only place the conversion is written | reported; a **call** figure, pooled over calls with a **call** denominator (v1.27, `-ml` §11.4); **kept on a load-contaminated item** — it measures generation after the load — its median taking the p50 gate against its own coverage |
  | `generationMs` | **`ChatResult.generationMs`** = `1000 × stats.generation_time` | stored per **call**, on `CallTiming` (v1.27, §4 S1 — v1.10 put it on the item, which the loop makes wrong); it is the second operand of `-ml` §11.5.1's per-call gap and is printed nowhere on its own. **New row in v1.10** — v1.9 used the name without defining it (plan-gate P4-9) |
  | `prefillMsPer1kPromptTokens` | `ttftMs ÷ (usage.prompt_tokens ÷ 1000)`, per call, aggregated as median. **No second conversion:** `ttftMs` is already milliseconds, and v1.9's `1000 × stats.time_to_first_token ÷ …` was the same arithmetic written from the raw seconds — correct in itself, and the row above it read the same field as milliseconds (plan-gate P4-1) | reported (**new in v1.2 — FR-11 names it and v1.1 omitted it**); a **call** figure with a call denominator, same treatment and same gate as `ttftMs` |
  | `coldLoadSeconds` | the warm-up call's wall clock, once per run, **only** when the model was not resident at start | reported separately, never averaged into steady state; **absent** (not `0`) otherwise |
  | `tokensPerSecond` | `stats.tokens_per_second`, **unconverted** — a per-second rate is already in the units its name claims | **diagnostic only** — FR-11 says so in words, and the report labels it so; a **call** figure like its two siblings, kept on a contaminated item and still printed with its (call) denominator, because a diagnostic over an unstated subset is the same defect one severity down (`-ml` §11.4) |
  | `unexplainedMs` | `-ml` §11.5.1's gap **summed over the item's own calls**, each call's gap taken over that call's `wallClockMs`, `ttftMs` and `generationMs` as normalised above — **`None` unless every call of the item yields a readable gap**; the metric, the sum and the threshold are the note's | an **item** figure: the in-call reload detector, **withholding `latencyMs`** above `-ml` §11.5.1's threshold, stored and its maximum reported either way; **chat surface only** |
  | `callCount` | `int`, `len(ItemTiming.calls)` — the calls the item comprises, asserted equal to `TurnTrace.iterations` (§4 S2 rule (ii)) | **new in v1.27** (`-ml` §11.9 ask 7, §11.8): `Y_calls` is its sum over items and is the denominator the three `stats`-derived figures print against; it is also what tells an `index.csv` reader whether a latency cell is per call or per turn |
  | `modelSizeBytes`, `peakHostRssBytes` | **no source (v1.8, R-2)**; sampled `Get-Process` | first recorded absent, never `0`; second best-effort, method-labelled (R-2) |

  **Every figure in that table needs a typed home, and four of them had none until v1.10**
  *(plan-gate P4-3)*. v1.9 committed the report to printing `ttftMs`, prefill, `tokensPerSecond` and
  `unexplainedMs` — with their own denominator, their medians taking `-ml` §11.6's p50 gate, and
  `unexplainedMs` "stored per item and its maximum reported" — while `ItemResult` carried
  `latencyMs` and nothing else timing-related and `LatencyBlock` carried only wall-clock figures and
  counts. With no typed home those figures land in `ItemResult.detail`, which this plan's own
  contract forbids `report.py` to read. The shape is **one record per item, `ItemTiming`, holding
  the item's own wall clock and an ordered `CallTiming` per call** (§4 S1 — v1.27 splits the two
  levels apart, and the item record gets *smaller*: the five scalars v1.10 put on it move onto
  `CallTiming` and have no second home), and the **per-run** medians and maximum on
  `LatencyBlock` (§4 S2). Two things about that shape are decisions rather than packaging:
  - **`latencyMs` becomes a derivation over `ItemTiming`, not a second field holding the same
    number.** `ItemTiming.wallClockMs` is what the harness measured; `latencyMs` — the *admitted*
    figure, and the only one any aggregate reads — is `wallClockMs` when nothing was withheld and
    `None` when something was, which is one line of derivation over the record's own
    `withheldFor`. Two stored fields for one measurement would need an asserted invariant to stop
    them drifting, and §7 rule 4 prefers the derivation to the invariant wherever one is available.
  - **Prefill is derived per call, never stored** — it is `ttftMs` and `promptTokens`, both of which
    are stored — so there is one fewer number that can disagree with itself.

  **This is also where a withheld item's wall clock stays readable** (`-ml` v1.12 §11.9 ask 2b).
  §11.5.1's `censoringExact` is a comparison between withheld and timed wall clocks, and §11.6
  promises the reader that *the summary is withheld, not the data* — so the number must survive
  somewhere `latencyMs`'s absence does not reach, which is `ItemTiming.wallClockMs`. The note offers
  reconstruction from `unexplainedMs + ttftMs + generationMs` as the alternative to a new field; this
  plan takes the field, for a reason the reconstruction cannot cover: on a `model:embeddings` arm
  there is no `stats` object, so the three reconstruction operands are all `None` while the wall
  clock was measured perfectly well — and that arm's only withholding producer is the between-item
  probe, which *is* right-censoring, so it is exactly the arm on which the predicate should come out
  true. Reconstruction would make it fail safe to false on every embedder render. **`censoringExact`
  itself is computed at render time from `run.items` and never stored** — the record carries its
  inputs, so §11.8's reconstructibility holds without a second home for a derivable boolean.

  The headline being **client wall clock** and not `stats.generation_time` is a decision, not an
  oversight: FR-11 asks for *end-to-end turn latency*, which includes request assembly, transport
  and the server's own queueing — everything the LM-Studio-side number excludes. Both are stored, so
  the difference between them is itself readable, but only one is the headline.
- **Tool-calling eligibility is gated before a tool-caller run**, so that "this model cannot do
  tool calls at all" is never recorded as "this model got them wrong". The gate is **not** a
  `"tool_use" in capabilities` test — that was v1.1's rule and it is wrong in both directions on
  this box's actual catalog, which the gate re-probed: `text-embedding-qwen3-embedding-0.6b`, a
  `"type": "embeddings"` model, advertises `"capabilities": ["tool_use"]` and would pass; while
  `google/gemma-3-4b` and `gemma-3-4b-vl-it-…` have **no `capabilities` key at all** and would be
  refused, though nothing establishes they cannot emit tool calls. The rule:

  > eligible ⟺ `type ∈ {"llm", "vlm"}` **and** (`capabilities` is absent **or** contains
  > `"tool_use"`).

  A refusal names which half failed. The raw `type` and `capabilities` go into the fingerprint
  (`modelType`, `modelCapabilities`, `modelCapabilitiesPresent` — §3.4.2), so the gate's decision on
  any past run is auditable from the stored record rather than re-derived from a catalog that has
  since moved.

  **This gate runs on a `tool-caller` pack and on nothing else, and the scope is load-bearing rather
  than tidy** *(v1.11, plan-gate P5-1)*. Its predicate refuses `type == "embeddings"`, and
  `text-embedding-qwen3-embedding-0.6b` — the model §4 S3's first done-condition names — is exactly
  that. Run unscoped at §3.4.4a capture-order step 3a it therefore refuses **every**
  `model:embeddings` arm before the adapter is ever called, which is G3-1's defect (the embedder
  unbuildable) in a second sentence; v1.10 wrote it unscoped and the plan gate caught it. **The
  universal pre-load refusal is §3.4.4a's `callSurface`-versus-catalog-`type` cross-check**, which
  already asks the question every arm owes an answer to — *does this model serve the surface this
  pack declares?* — and admits `type == "embeddings"` on an embeddings pack. This gate is a
  **role-specific addition on top of it**: a second question (*can a model this cross-check has
  already admitted to the chat surface emit tool calls?*) asked only where the pack's items are tool
  calls. Its `type ∈ {"llm","vlm"}` half is then redundant with the cross-check on a `tool-caller`
  pack and is kept anyway, because the refusal message that names *which half failed* is the point
  and the two halves fail for different reasons.

### 3.6a D6a — The CLI surface, consolidated (M-8)

The CLI is the tool's entire user surface. v1.1 mentioned six commands across five sections and
specified none of them, which left `model-bench attest` — required before S3's done-condition can be
met — owned by no stage. One table, and each command is assigned to the stage that must ship it:

| Command | Flags | Effect | Stage |
|---|---|---|---|
| `compare --pack <id>` | `--models a,b` · `--session <id>` · `--negative-control` · `--out <path>` | Reads `results/runs/`, renders the markdown comparison to `reports/` and stdout | **S1** |
| `index rebuild` | — | Regenerates `results/index.csv` from `results/runs/` | **S1** |
| `models --tested` | `--pack <id>` · `--role <role>` | Lists models with stored results (`armKind == "model"`); from S2 also intersects with the installed catalog | **S1**, catalog half **S2** |
| `attest` | `--api-base-url <url>` · non-interactive `--set k=v` | Prompts for the four operator-attested fields, probes LM Studio (§3.4.4a's two-step probe), writes `host.json` (§3.4.4) — with `observedAtAttestation` carrying **`residencySource` only**, since neither probed endpoint exposes a `runtime` and `attest` has no model to call (v1.10, plan-gate P4-6); the two runtime keys are back-filled by the first `model:chat` run | **S2** |
| `validate --pack <path>` | `--strict` | Runs `validate_pack`: manifest schema, `metrics` block, `sampling` contract (§3.3's three routes — `analysisUnit == pairingKey[0]`, the row-count identity, and `pairingKey[0] == roles.analysis_unit_field(role)`; plus `replicatesPerScript` and, v1.26, the `maxIterationsPerTurn` role rule), ids, provenance, paraphrase rule, pack-module import allowlist, `H ≤ min(script length)` | **S2** |
| `run --pack <id> --model <key>` | `--session <id>` · `--reference <key>` · `--warmup <n>` · `--first-call-timeout <s>` · `--request-timeout <s>` | One model × one pack; calls `validate` first and fails closed | **S2** plumbing, first usable **S3** |

**`--no-cold-load` is deleted in v1.9** *(G3-10)*. It survived from v1.7, where it meant "skip the
unload-then-timed-`lms load` step"; under v1.8 there is no unload, the warm-up is unconditional, and
`coldLoadSeconds` is recorded iff the model was not resident at start. The only meaning left to
invent was "suppress the `coldLoadSeconds` record" — a flag that hides a measurement, which this
plan refuses everywhere else. The opposite flag (refuse to run unless the model is non-resident, for
a deliberate cold-load measurement) is not added either: R-14 already routes that to a documented
human action, and a flag that can only *fail* the run adds no capability.

**Exit codes.** `0` whenever the tool ran and reported, *whatever the scores* — the requirements rule
out pass/fail gating and §1 states that the only non-zero exits are operational. The closed set:
`2` bad arguments or usage · `3` LM Studio unreachable, reachable without its native `/api/v0`
catalog (§3.4.4a's two messages share this code), not answering the warm-up within
`--first-call-timeout`, or gone when re-probed after a scored call timed out (§3.6) · `4` invalid
pack (`validate` failure, load error, unmet `environment.requires`, or a `callSurface` the model's
catalog `type` contradicts — §3.4.4a) · `5` fingerprint incomplete or
`host.json` stale/absent. Nothing else. A `compare` that finds every stored record invalid still
exits `0` and prints the `INVALID RESULTS EXCLUDED` block — that is a report, not an operational
failure.

**Output.** Every command writes its human-readable output to stdout; `run` and `compare` also write
their artifacts (`results/runs/<runId>.json`, `reports/<pack-id>-<date>-<n>.md`) and print the paths.
No command writes outside `model-bench/`, ever (FR-23, §5 test 20).

### 3.7 D7 — Runs, sessions, pairing, and the reference arm (FR-1 / FR-16 / FR-17)

FR-1 (one model per run) and FR-16 (paired, same session) read as a tension. They are reconciled by
separating three ideas that the requirements use in one breath:

- **A run** is one model × one pack × the pack's declared sampling (`sampling.scripts ×
  sampling.replicatesPerScript` for a conversation pack, one call per item otherwise). Always
  exactly one model (FR-1).
- **A session** (`--session <id>`, recorded in every run record) groups runs executed back to back
  on an undisturbed box. It is a *label with a meaning the tool enforces*: `runner` refuses to
  reuse a session id if `residency()` (§3.4.4a) shows a different residency profile than the session's first run
  saw, or if more than a configurable gap has elapsed.
- **Pairing is item-level and comes from the pack, not from the clock.** Both models see the same
  items in the same order with the same seed, because the pack pins them. So a paired comparison is
  always *computable*; what a shared session adds is the guarantee that box conditions did not
  change between the arms. `compare` therefore reports **which kind of comparison it is doing** —
  `paired, same session`, `paired, cross-session`, or `unpaired (different pack version)` — and
  never silently mixes them.
- **FR-17's reference arm** is `model-bench run --pack P --model CANDIDATE --reference INCUMBENT`,
  which runs both models in one session, sequentially (**block-level**, not item-interleaved: a
  16 GB box cannot hold two models, and per-item load/unload would dominate runtime — under JIT
  auto-load, item-interleaving forces a whole model load per item, per side, §2.5). The residual
  is named rather than hidden: within-session drift is bounded and recorded (both arms' timestamps
  and residency snapshots are in the records), not eliminated.
- **FR-17a**: the candidate list offered by `model-bench models --tested` comes from
  `results/index.csv` — the tool's own record — intersected with what LM Studio currently has
  installed. No other component's configuration is read, ever.

### 3.8 D8 — The five packs

Common to all: golden items are copied in as JSONL with a per-item `provenance` object
(`{origin, originPath, originGitSha, copiedAt, draftedBy, verifiedBy, corpusVersion}` — the git SHA
is required, not optional: without it FR-6/FR-19 provenance dead-ends at the copy and no diff
against the origin is ever possible again, `-ml` §5.5) per FR-19, and each pack
carries `PROVENANCE.md` naming the origin file, the origin commit, the copy date, and what was
changed on copy. `model-bench validate` re-checks: unique ids, required fields present, and — for
retrieval-style packs — the FR-19 **paraphrase rule** (a query must not be a verbatim substring of
its own target text), re-implementing what
`falkor-chat/server/tests/eval/test_golden_set_integrity.py::test_query_is_not_verbatim_self_retrieval`
checks today.

#### 3.8.1 `embedder` — pack `embedder-graphrag-retrieval`

- **Data copied:** `golden_retrieval.jsonl` (38 queries → `queries.jsonl`) and the 121-message
  corpus extracted from `falkor-chat/scripts/seed_eval_corpus.py`'s `_CORPUS` literal →
  `corpus.jsonl` (`{docId, text, topic}`; `docId` keeps the original `msgId` so
  `relevant_msgIds` resolve inside the pack). Also copied: `retrieval_baseline.json` and
  `golden_retrieval.embeddings.json` — the latter to isolate "is my ranking code right" from "is my
  embedding call right" (`-ml` §5.4). **On its own it does not isolate that**, and v1.1 claimed it
  did: inspected, the file holds **38 query vectors only** (`{gr-NN: {model, vector}}`), so the 121
  corpus vectors are still computed live and a wrong `documentPrefix` or a truncated corpus
  contaminates the self-test anyway. So S3 additionally writes the **121 corpus vectors** into the
  pack once, from the same deterministic embed pass, as `corpus.embeddings.json` — giving the
  ranking path a fully fixed input on both sides. Both files are inside the content hash, so a
  re-embed is a pack version bump.
- **Mechanism:** embed the 121 documents and 38 queries through `/api/v0/embeddings`, applying the
  pack's per-model `queryPrefix`/`documentPrefix` (FR-14 — a configuration field, never an
  assumption), **L2-normalize every vector and record the raw norm distribution** (`-ml` §5.2 — an
  endpoint returning unnormalized vectors corrupts separation without touching ranking, so it is
  invisible unless measured), then **brute-force exact cosine** in-process. No ANN index, no
  FalkorDB. Rationale: the object of measurement is the *model*, and an approximate index injects
  pipeline noise into a model comparison; exact search also gives the irrelevant-document scores
  that FR-12's score-separation metric needs. The scope boundary is stated in `README.md`: a model
  that wins here has not thereby been shown to win *through* falkor-chat's hybrid ANN pipeline.
- **The embedding cache key is `(model, quantization, docPrefix, corpusVersion)`** and the corpus is
  re-embedded whenever any part changes. Getting this wrong produces a plausible, invalid
  comparison with no visible trace (`-ml` §5.5) — so the cache key is asserted in a unit test, not
  just implemented.
- **Keyword arm (FR-13/AC-5):** BM25 over the same corpus, in-process, on every embedding run —
  reported as a **full paired arm with a confidence interval**, labelled `reference arm
  (deterministic given pack version)`. Tokenization, stopwords, `k1`/`b`, and the always-positive
  IDF variant required at N=121 are specified in `-ml` §5.3 and are pack configuration, not
  hardcoded. **It is stored as its own `RunResult` with `armKind: "deterministic"`** (§3.4.1) —
  that is the data shape that carries it to `compare_report`, and the reason `armKind` is decided in
  S1 rather than improvised in S3. The pack's declared BM25 parameters are hashed into
  `armParametersHash`, so changing `k1` invalidates the arm's comparability the same way changing a
  golden item invalidates the pack's.
- **Reported (FR-12/FR-14):** recall@k, MRR, **P@1** plus precision@k, and score separation as both
  `sep_raw` (within-model, actionable) and the corpus-sd-normalized `sep_z` (the cross-model
  comparable one), aggregated as median + p10 + fraction > 0 (`-ml` §5.2). **That aggregation is a
  distribution summary and not a mean**, so `sep_z` and `sep_raw` are carried by
  `DistributionSummary` and not by `ContinuousMetric`, whose single `mean` can hold none of the
  three (§4 S1, §4 S1e Table F, v1.12). Both `sep_raw` and `sep_z` take that carrier, being the same
  per-query quantity on two scales; **which** figures print for each is `-ml` §5.2's, and v1.16
  answers it for both: a **median, a p10 and the fraction above zero**, and **no mean for either** —
  which is the concrete reason `ContinuousMetric`, whose one aggregate field is a mean, is the wrong
  carrier here (§7 rule 2 — the carrier is the plan's, the publication is the note's). The
  **fraction of queries with `sep_raw > 0` is not a third
  stored figure** — `-ml` §5.2's own identity makes it exactly P@1, so it renders as P@1 with that
  identity footnoted, and a second field would be a second home for one number and would let the two
  drift.

  > **No difference between two models' `sep_raw` figures is printed, on any path** *(v1.13, `-ml`
  > v1.16 §5.2)*. `sep_raw` is scale-dependent per model — that is the entire reason `sep_z` exists —
  > so a difference of two models' raw separations differences two quantities measured on different
  > scales. Giving `sep_raw` the same per-item carrier as `sep_z` (§4 S1e Table F) is right and makes
  > such a difference **computable**, which is what turns a reader's inference into a prohibition
  > worth writing down. It is the same shape as `-ml` §11.7 slot 6's latency rule: a per-item
  > continuous quantity reported per arm and never differenced. Structurally, two things already
  > enforce it and are named here so neither is undone by accident — `separationRaw` is not in
  > `verdictMetrics` and never will be, and the Arms table renders a `DistributionSummary` row with
  > **no interval** (§4 S1e Table F) — so the only path that could compute the difference is the
  > exploratory comparison below, which is entered for `sep_z` alone.

  **`sep_z` is reported, not verdicted** (`verdictMetrics = ["mrr"]`), and §5.2's cross-model
  comparison of it — a paired bootstrap on per-query `sep_z` differences, wired through
  `paired_cluster_bootstrap` with `clamp=None` because a difference of z-scores is not bounded by 1
  and the shipped widening clamps it, and with `levels=(LEVEL_CI95_LO, LEVEL_CI95_HI)` stated at the
  call site (§4 S1e Table G; the element type is `Fraction` — `-ml` §11.2.2 — v1.15) — goes
  **through the engine entry point and not through `-ml` §3.4 Rule 8's
  `continuous_verdict()`**, which produces a verdict, and this comparison is not one: a number
  outside `verdictMetrics` prints its interval with no significance claim and takes no family
  correction (§3.3). The comparison is **exploratory and is no stage's done-condition**; §4 S1e
  Tables E and G are that comparison's two preconditions and land with the rest of S1e *(v1.12,
  `-ml` v1.15 §3.2d: v1.11 read the comparison as part of S3 and gated §4 S3 done-condition 2 on the
  clamp, which the note reverses; Table G is v1.13's and gates nothing either)*. **Also reported:**
  output dimension,
  RAM cost, embedding throughput (texts/s and tokens/s), max input length and observed truncation
  behavior, and the prefix convention used.
- **`verdictMetrics = ["mrr"]`, `headlineMetric = "mrr"`**, and the report carries two honesty lines
  the note requires
  (`-ml` §5.1, §7.4): precision@k is an exact rescaling of recall@k on this set (|R| = 1 for 36 of
  38 items) and is printed for FR-12 compliance with that footnote; and **recall@10 = 37/38 leaves
  one winnable item**, so this pack can detect a materially *worse* embedder but can never certify
  a *better* one on recall. Both are printed in the report, not just documented here.
- **Cost:** low. Existing verified data, no new golden asset. The follow-up that would remove the
  recall ceiling (+22 harder queries, several with |R| ≥ 3) goes to `docs/BACKLOG.md`, not into
  first delivery.

#### 3.8.2 `guard-judge` — pack `guard-judge-understanding`

- **Data copied:** `golden_guards.jsonl` (85 items, distribution in §2.1) → `items.jsonl`.
- **Mechanism:** the pack carries the judge prompt as **text** (`prompts/judge.md`), transcribed
  from `falkorchat/app.py::_LlmGuardJudge`'s prompt construction — data, not an import. One
  single-turn call per item; the reply is parsed with the pack's declared parse mode
  (`ownLineJsonObject`, re-implemented in `modelbench/scoring/classification.py` — the conservative
  reading: an unparseable reply is a **parse failure counted in the denominator**, never a
  fabricated verdict).
- **Reported — class-conditional, and deliberately *not* a pooled 85-item accuracy figure**
  (`-ml` §7.3): **`falseAdvanceRate`** = P(judge advances | gold says suspend) on the 40
  `clear_suspend` items, **`falseSuspendRate`** = P(judge suspends | gold says advance) on the 30
  `clear_advance` items, and the 15 `boundary` items **descriptive only** at that n and explicitly
  not a verdict metric. Split by evidence path (`understanding` / `turns`) as a diagnostic, plus a
  parse-failure count in the denominator. `report.py` has no code path that emits a pooled 85-item
  accuracy number.
- **This pack has no headline number, deliberately** *(stakeholder decision, 2026-09-02)*:
  `verdictMetrics = ["falseAdvanceRate", "falseSuspendRate"]`, `headlineMetric = null`. Both
  class-conditional rates get a verdict, with equal weight and no ranking between them — the
  stakeholder declined to declare which error is costlier in the product, which is precisely the
  judgement `-ml` §10's open question 2 said could not be made by the analyst.
- **Both co-primaries are error rates pointing the same way, and that is a requirement rather than a
  style choice** (`-ml` §7.3, authoritative). v1.2 paired `falseAdvanceRate` with **advance-recall**,
  which is `1 − falseSuspendRate` — the same quantity read backwards. Two co-equal verdict metrics
  where one reads "better is lower" and the other "better is higher" makes "worse on one, better on
  the other" unreadable at a glance, in the one pack where the stakeholder deliberately declined to
  rank the two errors. So both verdicts render on error rates; **`advanceRecall` stays printed as
  the labelled complement** of `falseSuspendRate`, so a reader looking for recall still finds it,
  but it carries no verdict.
- **The two-member family costs resolving power, and the report says so.** Holm–Bonferroni is
  mandatory here (§3.3(ii)), so each class's resolving-power line is computed at the αs the note
  assigns to its bounds; `-ml` §7.3 carries them and the recomputed figures for both slices, and
  the plan restates neither. `-ml` §7.3 also records a costed **reversal trigger**: if the
  stakeholder ever ranks the two errors, the loser moves out of `verdictMetrics` into the
  exploratory block, the family collapses to k=1, and resolving power improves measurably. That
  trade is the stakeholder's to
  make later, not the implementer's to make quietly.
- This is the pack that makes §3.3's `headlineMetric: null` a first-class case rather than a special
  case; a future pack that wants no headline needs no further harness work.
- **Cost:** low.

#### 3.8.3 `nlq-generator` — pack `nlq-structured-query`

- **Data copied:** `nlq_golden_set.jsonl` (40 items) → `items.jsonl`; the 15-product catalog from
  `falkor-chat/scripts/seed_catalog.sh`'s `CATALOG` literal — which is a **Python list of
  `(name, category, price)` tuples inside a `<<'PY'` heredoc**, not a shell array, so copying it is
  a Python parse rather than a shell one — plus a **read-only snapshot** of
  `ws:nlq-eval`'s `Entity`/`Document`/`Chunk` rows (§2.1 finding 2) → `tables.json`; the dataset
  schema from `falkorchat/querygen.py`'s `CATALOG_SCHEMA`/`KNOWLEDGE_BASE_SCHEMA` → `schema.json`;
  the generation prompt from `falkorchat/tools.py` → `prompts/querygen.md`. The snapshot is taken
  once by `refresh_golden.py` via a read-only Cypher read, recorded in `PROVENANCE.md` with its
  date and row counts, and never read again at run time.
- **Mechanism, and why no database is needed:** the model emits a **structured JSON query spec**
  (§2.1 finding 3), not Cypher. The pack ships a small in-process executor (`tools/exec.py`) that
  applies a spec to the in-memory tables, implementing exactly the surface
  `querygen.DatasetSchema` declares — a label, property filters with the declared types and the
  same string→number coercion rule, returns, and the `count`/`avg`/`min`/`max` aggregates. Scoring
  is **Layer 1** — exact match after canonicalization against the item's `expected`, with the same
  numeric epsilon (`_NUMERIC_EPSILON = 0.01`) and the same scalar/set shape rules
  `falkor-chat/server/tests/eval/nlq_scoring.py` documents (re-implemented, cross-checked in tests
  against a copied sample of `nlq_eval_results.json` records). **Layer 1 is not uniformly
  exact-match, and the exception must be built rather than discovered:**
  `nlq_scoring.score_pair` scores `shape == "conflicting-facts"` by **subset containment**
  (`expected_set.issubset(actual_set)`, `nlq_scoring.py:190`), not set equality — it affects 2 of
  the 40 items, and a scorer that applies equality uniformly marks both wrong for every model,
  turning them into a second silent floor beside the unanswerable bucket. `nlq_scoring`'s **Layer
  2** (`layer2_contains`, containment against the *rendered* natural-language answer) is a
  non-gating sanity signal in falkor-chat and is **not** copied: this pack scores the structured
  result, which is what the model actually produced.
- **Answerability must be established at pack-build time, not assumed.** The declared KB schema
  exposes node *properties* only — no relationship types — so the golden set's 4
  `relationship-traversal` items ("Who did Marlowe Robotics acquire?") are not answerable by any
  valid spec against these tables. That is not speculation: falkor-chat's own stored
  `nlq_eval_results.json` scores `nlq-34` incorrect with `{"items": [], "finding": "no matching
  data found"}`. An item no model can get right is a floor, not a discriminator, and silently
  including it deflates every arm equally while making the headline number mean less. So
  `refresh_golden.py` runs a **hand-written reference spec per item** through the executor at copy
  time and stamps each item `answerable: true|false`; `validate` fails a pack that has unstamped
  items; and the report puts unanswerable items in a **separate, named bucket** excluded from the
  accuracy denominator. Keeping them is still worthwhile — they measure whether a model correctly
  abstains instead of fabricating — but as their own count, not as accuracy.
- **Reported:** `verdictMetrics = ["layer1ExactMatchRate"]`, `headlineMetric =
  "layer1ExactMatchRate"` — the note's choice for this role (`-ml` §3.3, §7.2), and v1.1 declared
  none at all, which under §3.3's own rule would have labelled every number in this pack's report
  `exploratory` and made a verdict impossible. Beside it, exploratory: correct/incorrect split by
  `shape` (single-fact, filter-list, compound-filter, not-found, aggregation,
  relationship-traversal, conflicting-facts) — the shape split is the informative part at n=40;
  plus malformed-spec and schema-violation counts separately from wrong answers, and the
  abstain-vs-fabricate count on the unanswerable bucket.
- **Cost: medium-high, and v1.1 under-costed it.** Execution *is* small — `QueryRequest.matches` is
  `min_length=1, max_length=1`, so there are no joins — but "malformed-spec and schema-violation
  counts **separately** from wrong answers" means re-implementing `querygen`'s **validation**
  surface too, in **stdlib only** (§3.3 forbids a pack module from importing pydantic): the
  `QueryFilter`/`QueryMatch`/`QueryRequest` constraints (`extra="forbid"`, the six-operator
  whitelist, `filters` max 4, `returns` 1–6 against the projection/aggregate regexes, the `order_by`
  shape, `limit` 1–50) plus `compile()`'s allowlist and string→number coercion. That is the larger
  half of `tools/exec.py` and it is what makes the three failure classes distinguishable rather
  than pooled into "wrong". Still pure and unit-testable against the golden set's own expected
  values — but sized as its own piece of work in S4, not as a rounding error.
- **Scope note:** this measures *structured query generation against a declared schema*. It does
  not measure raw Cypher authorship; nothing in the requirements asks for that, and adding it would
  need a database and a much larger golden set.

#### 3.8.4 `tool-caller` — pack `tool-caller-shop-assistant` (the long pole, FR-22)

- **Data: new, and the main cost of this feature.** `conversations.jsonl` — fixed, versioned,
  multi-turn scripts. Each row:

  ```json
  {"scriptId": "A-02", "shape": "A", "replicate": 1,
   "description": "read-only catalog lookups",
   "turns": [{"seq": 1, "user": "...",
              "expect": {"toolRequired": true, "tool": "lookup_product_fact",
                         "args": {"name": "Wireless Charging Pad"},
                         "argChecks": [{"kind": "boundary", "arg": "maxPrice", "value": 50}],
                         "terminal": false,
                         "finalReplyMustContain": ["24.99"]}}],
   "provenance": {"draftedBy": "...", "verifiedBy": "...", "basedOn": "..."}}
  ```

  `scriptId` is the **sampling unit** and the first component of `ItemResult.pairingKey` (S1);
  `shape` ∈ `{A, B, C}` is the reporting stratum; `replicate` is `1` throughout under the current
  sizing and exists so that raising `replicatesPerScript` later does not change the record shape.

  Three conversation **shapes**, reconstructed from
  `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.1 (§2.2): **A** 9 turns
  read-only, **B** 7 turns write-mutating, **C** 4 turns short. Then extended — §8.1's set was
  designed to characterize one defect, and this pack needs coverage of all seven FR-8 failure
  kinds, including "stopping when done" and "final reply matches what the tool returned".
- **Sizing: 12 distinct scripts (4 per shape) × 1 run each = 12 conversations, at
  `temperature: 0.0`** — *stakeholder decision, 2026-09-02, replacing v1.1's 4 scripts × 4
  replicates × 3 shapes.* The authoring cost lands in S6. **Correction, v1.3:** this decision was
  put to the stakeholder as "the same total run budget", and that was wrong — 12 × 1 is roughly a
  **quarter** of the previous design's inference budget, not a re-allocation of it (`-ml` §4.5.2
  carries the measured basis, §4.5.3 the honest consequence, and a costed reversal trigger). The
  correction is being relayed to the stakeholder separately; the design is **not** revisited on it
  here, because the resolving power it buys is unchanged — the old nominal 48 never supported more,
  which is the whole reason the decision was taken. It satisfies **FR-22a**, and it is taken on
  `-ml` §4.5's
  own argument: at temperature 0, replicates of an identical prompt are near-duplicates that add
  almost no information, so 48 conversations clustered in 12 scripts carried an effective *n* the
  note put "closer to 12 than 48". Buying the 12 outright makes the honest number the real one
  instead of a design effect the bootstrap has to discover. Two consequences that must be built,
  not assumed:
  - **The sampling unit is the script and there is exactly one observation per script**, so the
    two-level cluster resample `-ml` v1.1 §4.5 specified no longer has an inner level. **The
    clustering treatment under this design — what `stats.py` must implement for
    `min_detectable_difference`, `verdict()` and any cluster-aware interval — is settled in the
    `-ml` note, not here** (see §3.9). This plan does not restate it, and `stats.py` implements the
    note's signatures verbatim.
  - **The pack declares `replicatesPerScript: 1` in its manifest and the report prints it beside
    every conversation-level *n***, with `temperature`, as `-ml` §4.5 requires. **`validate` fails
    any pack declaring `replicatesPerScript > 1`** while `stats.py` carries only the one-level
    `cluster_bootstrap` (`-ml` §3.4 Rule 6): a replicated pack needs the two-level resample, and its
    absence must be an error rather than an approximation nobody notices. So the field cannot be
    raised invisibly *or* silently mis-analysed — raising it is a deliberate act that first requires
    the note's two-level function to exist.
- **The determinism probe — the evidence behind `basis: "by-construction"`** *(new in v1.4, gate
  finding N-2)*. `-ml` §3.4 Rule 4 lets McNemar decide **only** when `design_effect == 1.0` **and**
  `basis == "by-construction"`, and §4.5.1 grants that basis to 12 × 1 on the grounds that each
  script contributes one observation. But §4.5.1(iii) also states that this design makes run-to-run
  variability *unmeasurable*, that LM Studio at temperature 0 is "near-deterministic but not
  guaranteed bit-deterministic", and prescribes the evidence: **re-run 2 of the 12 scripts a second
  time, once per model, and report whether the outcome vector is identical.** v1.3 asserted the
  basis and built no probe, so the one input deciding whether McNemar may decide the flagship metric
  was an attestation. Now:
  - **The pack names the two scripts** (`sampling.determinismProbeScripts`, one shape-A and one
    shape-B script — the long and the write-mutating shapes, where non-determinism is most likely to
    bite), so which two is pack data and cannot be chosen after seeing a result.
  - **Budget: 2 extra conversations per model** (~14 turns, one shape-A + one shape-B), stated here
    because it is a real cost that must appear in the run plan rather than surprise S6. It runs in
    the same session, after the 12 scored conversations.
  - **Diagnostic, outside `n`, never pooled into it** (`-ml` §4.5.1(iii)). The probe's conversations
    are excluded from every denominator and appear only in the report's own probe line.
  - **The outcome is wired to `basis`, and absence of evidence degrades it.** `runner` sets the
    `basis` passed to `resolving_power()` to `"by-construction"` **only if** all three hold:
    `replicatesPerScript == 1`, the probe **ran**, and both probe scripts produced outcome vectors
    identical to their scored runs. Otherwise `basis = "assumed"` — including the case where the
    probe simply was not run. Via `-ml` §3.4 Rule 4 that automatically moves McNemar out of the
    decision seat and **the conservative envelope** into it, with McNemar's p still printed and
    labelled `anti-conservative under clustering — not the decision`. *(v1.10: the instrument on that
    path is `-ml` §3.4 Rule 4's bound-by-bound envelope of two arms, and naming one arm —
    "the cluster-bootstrap CI" — is the defect note v1.11 corrects in its own four printed strings.
    The plan's prose follows it.)* **The fail-safe default is the point**:
    an unrun probe can never silently buy the stronger instrument, so N-2 cannot recur by omission
    the way it arose.
  - **Carried in the record:** `ToolCallAggregates.determinismProbe =
    {scriptIds, ran: bool, identical: bool, differingTurns: [...]}`, so a stored run's `basis` is
    re-derivable years later rather than trusted.
- **Honest consequence, printed not buried:** twelve conversations is a small *n*, and the
  resolving-power line computed from it (§3.9 point 2) will be a large number. That is the true
  precision of the previous design too — it was simply hidden inside a design effect. The effects
  this pack exists to catch are the near-total ones (`-ml` §7.1's reassurance: the qwen3-4b turn-4
  collapse, the ministral duplicate-instruction defect), and the report says in words which
  magnitudes it can and cannot resolve rather than leaving a reader to assume.
- **Simulated tool environment (`tools/sim.py`)**, deterministic and stateful within a
  conversation: `lookup_product_fact`, `filter_products`, `view_cart`, `add_to_cart`,
  `remove_from_cart`, `clear_cart`, `place_order` — schemas re-declared in the pack as JSON Schema
  (transcribed from `falkorchat/tools.py`, which is where the boundary/unit-translation wording
  that FR-8(d) targets lives). The environment records a **dispatch trace** (every call: name,
  raw arguments, parsed arguments, return value, timestamp) and exposes its **final state** (cart
  contents, orders). FR-10's "system ground truth" is exactly these two — never the model's reply
  text. Reply text is used only for FR-8(g), and only as a containment check against what the tool
  actually returned.
- **Scoring (`scoring/toolcalls.py`)** produces the FR-8 counts **per turn position**, with the
  exact denominators in `-ml` §4.2. Never a single blended percentage; `report.py` has no code path
  that produces one (AC-1 enforced structurally, not by discipline). Four shape changes the note
  requires, all of which the implementer must build rather than infer:
  - FR-8(a) and (b) **collapse into one three-way partition** over the same denominator —
    `native` / `prose_pseudo_call` / `no_attempt` — which satisfies both sub-requirements without
    the harness having to guess whether the model "intended" a call. The prose detector is a
    heuristic, so the pack ships ~20 labelled replies and the report prints **the detector's own
    precision and recall** (`-ml` §4.2).
  - FR-8(d)'s unit is the **call**, not the turn (`n_calls` printed explicitly), and
    `boundary_unit` is a **named subset of `wrong_value`**, not a sibling — as amended FR-8(d) now
    requires, with the **per-argument boundary rule supplied by the pack's tool schemas**
    (`boundaryRule`), never inferred by a regex in the scorer.
  - An **eighth count the requirements omit: restraint** — the rate of correctly *not* calling a
    tool on turns where none was required. Without it a trigger-happy model scores perfectly
    (`-ml` §4.2).
  - **Precondition failures are never silently excluded.** Every rate prints `k/n` inline, every
    count carries an `n/a` tally, and the report **opens with a funnel table**. This is the
    specific mechanism that stops a model which collapses early from scoring *better* on every
    conditional count downstream — the note names it as the harness's most likely way of lying
    (`-ml` §4.3). Paired *n* is the intersection of both arms' scoreable items, printed, with a
    one-model-only `asymmetry` count.
- **Headline and diagnostic (`-ml` §4.6).** `verdictMetrics = ["cleanThroughTurnH"]`,
  `headlineMetric = "cleanThroughTurnH"` — the fraction of conversations with no failure through
  turn *H*: one observation per conversation, length-independent by construction, and the statistic
  that would have caught the incumbent model. ***H* is declared in `pack.json`
  (`metrics.cleanThroughTurnH.H`, `4` for the A/B/C set), never derived from the data.** v1.1
  derived it as the shortest script length, which means adding one 3-turn script to a future pack
  version silently redefines the headline from `cleanThroughTurn4` to `cleanThroughTurn3` — the
  primary metric changing meaning under a fixed name, which is the exact failure this tool exists
  to prevent, and one AC-3's version banner would report as "the data changed" rather than "the
  headline now measures something else". So `H` is pack data, and three consequences follow:
  - **`validate` *fails* a pack where `H > min(script length)` — it never clamps `H`, and never
    warns.** The methodological reason, and it is stronger than tidiness (`-ml` §4.6, v1.4): a
    headline computed at `H > min` is **selection-conditioned**, because short scripts cannot reach
    turn `H` at all, so the metric quietly becomes a rate over long conversations only. That is
    `-ml` §4.3's laundering failure landing in the one number the report calls its headline —
    precisely the thing §3.8.4's funnel table exists to prevent everywhere else. A clamp would
    silently produce a *different, valid* metric under the declared name, which is M-11 again by
    another route; a warning would be ignored. Fail closed.
  - **`H` is printed beside the metric name in every report** — `cleanThroughTurn4`, never
    `cleanThroughTurnH`.
  - **The metric has three states, not two, and the third is what stops `unrunnable` becoming an
    escape hatch at the headline** *(v1.27; `-ml` v1.21 §4.3 rule 4's closing clause and §4.3.1
    item 5)*. *"No failure through turn `H`"* reads a turn the harness could not drive as **clean**,
    which is `-ml` §4.3's laundering arriving at the one number the report calls its headline —
    and it arrives there precisely *because* rule 4 rules such a turn `unrunnable` rather than
    `fail` (§3.8.4's disposition block). So: a conversation carrying an `unrunnable` turn at any
    `t ≤ H` is **neither clean nor failed**. It leaves the headline's denominator —
    `ItemResult.scoreable["cleanThroughTurnH"] is False`, the machinery §4 S1 already has — is
    counted in the headline's own `n/a` tally and printed beside it (rule 2), and drops out of the
    paired intersection for **both** arms, surfacing as an `asymmetry` count. **No new gate keeps
    the shrunken `n` honest**: `verdict()` already refuses below `resolving.observable_floor`
    (`-ml` §3.4 Rule 7) and that floor is computed from the surviving `n`, so a collapsed
    denominator refuses itself.
  - **`H` strictly below `min(script length)` is legitimate, and the gap is reported.** Declaring
    `H < min` keeps the headline's meaning stable across pack versions, at the price that turns
    `H+1 … min` are scored and then excluded from the headline — the metric becomes **coarser, not
    wrong** (`-ml` §4.6, v1.4). So when the gap is non-zero the report carries a line naming it as
    discriminating information deliberately left on the table. It is **zero today** (`H = 4 =
    min(script length)`), which is why the line renders only when it is not. The **per-turn hazard** — P(first failure at *t* | clean
  through *t*−1) — is the required diagnostic, because it is the only statistic that separates
  "gradual degradation" from "deterministic collapse", which is the entire reason FR-9 exists.
  Turn-pooled rates are never the headline.
- **Prompt assembly** is `convo.py` reading the pack's `prompt` block (§3.3). Turn *n* is built by
  replaying turns 1..*n*−1 in the configured shape; the harness never carries hidden state between
  turns beyond what the configuration says it carries.

  **What is replayed is what the model actually produced in *this* run — never the script's
  `expect`** *(v1.25; this sentence is the ruling, and every other section cites it rather than
  restating it)*. A turn's `expect` block is a **scoring oracle**: it names what a correct agent
  should have done, and `scoring/toolcalls.py` is its only reader. A prior turn's replayed content
  comes from that turn's own record — the model's `ChatResult`s and the environment's
  `DispatchRecord`s — and the scripted `user` text, which is the script's at every turn because the
  harness drives the full script (`-ml` §4.1). The clause above decides the replay's **shape** and
  forbids **undeclared** state; a prior turn's real output is neither hidden (it is in the message
  list) nor undeclared (`historyReplay` is what declares it), so it never decided this — and no other
  rule here did either, which the v1.25 revision line pins to `40a9bc8` rather than asserting.
  **Four reasons, each sufficient alone:**
  - **The validation target below stops being a measurement of the documented phenomenon.**
    `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §4.1 attributes the turn-4 collapse
    to the model's **own prior turns, replayed as ordinary chat with no visible evidence a tool was
    ever used** — "exactly three prior 'user asks, assistant answers' text exchanges" flipping the
    completion pattern on the next turn. Under a textbook replay that precedent is **identical for
    every conversation and every model**, decoupled from what the model under test did. **The claim
    here is narrower than "the collapse cannot happen", and deliberately so:** a textbook prefix in
    one of the reply-text modes presents a stimulus of the same *shape*, so onset might well fire.
    What cannot happen is the finding. §8.2's result is onset **and** persistence — zero recoveries
    in 121 post-onset turns — and persistence is unreproducible by construction, since the model's
    own collapsed turn never enters any later context. Worse, onset's reachability becomes a
    function of the *mode*, and in a direction nobody has measured: a reply-text replay presents the
    same stimulus shape, while `structured`'s textbook tool evidence presents a different one whose
    effect at this replay shape is **unknown**. *(v1.26, P13-4: v1.25 asserted that onset is absent
    under `structured` on the grounds that its scaffolding is the U37/U38/U39 breadcrumb condition.
    Both halves are wrong. The breadcrumb was a free-text suffix folded into a replayed **assistant**
    message, not native `tool_calls` plus `tool` results; and it did **not** suppress the failure —
    `salesperson-tool-reliability-impl.md` MAJOR 1 records it live-verified 2/2 not to reduce
    fabrication and reverted as a **severity increase**, the model imitating the breadcrumb's surface
    text while calling no tool, and `executor.py:1259-1266`'s own docstring says so. The clause is
    deleted rather than repaired; nothing in this ruling rested on it.)* So a reproduced contrast
    would be a coincidence of stimulus shape rather than evidence that the harness measures what
    §8.2 measured — and §3.8.4's own words for that outcome are that the harness, not the models, is
    what has been measured.
  - **The per-turn hazard stops separating what FR-9 needs separated.** P(first failure at *t* |
    clean through *t*−1) conditions on cleanliness to keep conversations whose context is already
    contaminated out of the turn-*t* denominator. Under a textbook replay the context at turn *t* is
    identical whether the model failed at *t*−1 or not, so the conditioning removes nothing about
    the trial, the accumulation channel is gone, and both "gradual degradation" and "deterministic
    collapse" reduce to position sensitivity under a fixed clean prefix.
  - **The context would contradict FR-10's ground truth.** `drive` dispatches the model's real calls
    into a stateful `ToolEnvironment`, so from the first failed write-mutating turn onward a
    textbook context asserts a cart the environment does not have, and the next `view_cart` returns
    the real one. Failures produced that way are the harness's, which is precisely what the
    validation target is worded to catch.
  - **It would corrupt the determinism probe rather than protect it.** `-ml` §4.5.1(iii)'s probe
    object is the **conversation-level observation that enters `n`** — "this model is flaky" versus
    "this script is hard". A re-run that diverges mid-conversation *is* that flakiness, and must be
    reported. A textbook replay makes every turn an independent trial against a fixed prompt, so a
    run whose observation did not reproduce can still return `identical` — buying
    `basis: "by-construction"` and McNemar's decision seat on evidence that was never taken. The
    fail-safe direction §3.8.4 built into `basis` runs the other way under this design.
- **One turn is a bounded iteration loop, not one call** *(v1.25 — a gap the plan carried and item
  1's ruling makes load-bearing)*. `-ml` §4.1's notation already has `I(t)`, "LLM iterations consumed
  in the turn, and whether the iteration cap was hit"; §4.2(f) scores stopping-when-done as *no call
  dispatched after `R(t)` was satisfied **and** the loop terminated below the cap*, and reports
  `iteration_cap_hit_rate` with mean and p95 of `I(t)`; §4.2(g) checks the turn's **final reply**
  against what the tool returned. One call per turn leaves a tool-calling turn with no final reply
  at all — nothing to score (g) against, nothing to replay under `structured-replies-only` or
  `plaintext`, and no source for `I(t)`. So `drive` runs, per scripted turn: call the model;
  while the response carries native `tool_calls` and the cap is not reached, dispatch each call
  against `env`, append the assistant message and one `tool` message per call to the **in-turn**
  working list, and call again; stop when a response carries no tool calls — that response's text is
  the turn's final reply — or when `maxIterationsPerTurn` is reached.

  **A turn ends in one of five ways, and the *mechanism* is recorded as `TurnTrace.turnDisposition`,
  separately from the reply field** *(v1.26, P13-1: v1.25 wrote `finalReplyText is None` **iff**
  `capHit`, which is false in the ← direction — a call that times out, drops or is refused also
  yields no final reply with `capHit == False`, and §4 S5's *absent-not-failed* rule keyed on that
  field converted a §3.6 `fail` into an `n_a`, in the paragraph that names laundering as the enemy.
  The `iff` was **not** widened; the disposition was split off the field instead. v1.27 widens the
  set from four members to five and **deletes the scoring claim the last column used to make** —
  P14-1, the `fail`/`unrunnable` ruling, and the timing ruling, all below.)*

  | `turnDisposition` | mechanism | `finalReplyText` | what the **record** then carries |
  |---|---|---|---|
  | `replied` | a response carrying no tool calls terminated the loop | `str(content or "")` — **never `None`**; a terminating response with `content: null` or `""` is *captured and empty* | `withheldFor`: `None`, or `"load"` if a guard fired |
  | `cap-hit` | `maxIterationsPerTurn` reached with tool calls still being emitted, **every one of those calls having returned** | `None` | the turn is **timed** — its wall clock is complete — so `withheldFor` is `None`, or `"load"` if a guard fired |
  | `timed-out` | a call hit `requestTimeoutSeconds`: `LMStudioCallTimeout` | `None` | `withheldFor: "timeout"`; the runner re-probes and may exit `3` (§3.6 clause (iv)) |
  | `no-response` | the server did not answer, or answered unusably — **no HTTP status is carried**: `LMStudioCallFailed` with `status is None` (dropped connection, socket error, or a 2xx whose body is not a usable response) | `None` | `withheldFor: "no_response"`; **no** re-probe and no exit `3` (§3.6's fourth disposition) |
  | `server-rejected` | the server answered and refused: `LMStudioCallFailed` carrying an HTTP status | `None` | `withheldFor: "no_response"`; no re-probe, as above |

  So `finalReplyText is None` **iff** `turnDisposition != "replied"`, and no consumer keys
  *absent-not-failed* on the reply field. **The set is a mechanism vocabulary and deliberately not
  the note's**: the gate proposed `unrunnable` as a member, and `unrunnable` is `-ml`
  §4.1's *count*. Naming a `TurnTrace` field after a scoring category is the collision that has
  already cost this plan twice (`BinaryMetric.unit` against `PackRef.analysisUnit`; `unit_kind`
  against `analysis_unit_field`), so the token names what happened and `-ml` §4.3 rule 4 is where
  it maps onto the counts.

  **What the mechanism *scores* is not in that table, and v1.27 deletes the column that said it
  was** *(P14-1, and `-ml` v1.21 §4.3.1 item 1 recommending the deletion over the row split)*.
  v1.26's fourth column asserted a scoring mapping and called this table *"the only home of that
  mapping"*. It was never the only home: `-ml` §4.3's funnel has routed these turns since v1.1, and
  §4.3 **rule 4** now states the mechanism-to-denominator mapping outright, in one place, as a
  function of the **pair** `(D(t), E(t))` rather than of the mechanism alone. **Two homes for one
  mapping is what produced P14-1**, and the ownership reason is §7 rule 2's: this plan owns
  `TurnTrace`'s mechanism vocabulary and the record fields above, while *which denominator a
  mechanism lands in* is a scoring question and the note's. So the column is a citation now, the
  sole-ownership sentence is deleted rather than corrected, and the plan restates none of rule 4's
  five rows. What P14-1 found is closed by that deletion: a `cap-hit` turn with `|E(t)| = 0` —
  which the loop reaches whenever a model emits only undispatchable tool calls, `drive` skipping a
  nameless call (`convo.py`'s `if not name: continue`, at `d5b549d`) — is `no_attempt` under
  §4.2(a), a **failure**, and is outside (f)'s `stopping_when_done` denominator and outside (g)'s
  entirely, so it cannot occupy (g)'s `unscoreable` bucket. v1.26's table routed it into both.
  *Two corrections the gate's own fix sentence needs and this deletion makes moot rather than
  inherits:* `iteration_cap_hit_rate` is **not** "the one count keyed on the disposition alone" —
  there are **three** so keyed (the cap-hit rate, and the mean and p95 of `I(t)`) and they use
  **two different subsets**, `all-less-unrunnable` for the first and `{replied, cap-hit}` for the
  other two; and a reader who takes the singular literally puts the `I(t)` summary back under
  `E(t)`. Both are `-ml` §4.2(f)'s and neither is restated here beyond that warning.

  **The outcome a non-completion scores is `-ml` §4.3 rule 4's, and it reverses one clause of
  §3.6** *(v1.27; `-ml` v1.21 §4.3 rule 4 and §4.3.1 item 3)*. Only the **timeout** scores `fail` —
  it is the one mechanism where the harness gave the model its whole declared budget and observed
  nothing come back. `no-response` and `server-rejected` are channel failures the harness cannot
  attribute (a `400` from a model's runaway message list and a `400` from a malformed harness
  payload are the same status code), so both are §4.1's **`unrunnable`**: out of every scoring
  denominator, counted, printed. §3.6's fourth disposition said `fail` and §3.8.4's v1.26 table
  said `unrunnable` for the same non-2xx turn; **the contradiction is real and is decided against
  §3.6's outcome clause**, which is amended there. *Judged rather than adopted: the discriminator
  is attributability, and it is the same distinction §11.5.1 already spends on `censoringExact` —
  a censored observation has a known bound and is evidence, a missing one is neither. The objection
  §3.6 raised — that `n_a` lets a model that hangs out-score one that answers wrongly — is real
  about a **rate whose denominator silently shrinks**, and `-ml` §4.3 rules 1–3 are what stop that:
  an `unrunnable` turn costs the model `n` visibly, in the funnel's head, in the paired
  intersection, and in the resolving-power line. The one place it could still escape is the
  headline, which is why `cleanThroughTurnH` takes a third state (above).*

  **Why the set is five and not four, and it is not P14-1's doing** *(v1.27; independently reached
  here and in `-ml` v1.21 §4.3.1 item 2)*. v1.26 folded the
  timeout and the status-less failure into one `no-response` row, and **three** consumers need them
  apart. **Scoring**: under rule 4 above a timeout is `fail` and a status-less failure is
  `unrunnable`, so a folded token leaves the S5 scorer unable to tell one from the other — the
  decisive reason, and the one v1.26 could not have had. **Timing**: `-ml` §11.5.1's
  `censoringExact` reads the distinction **per item** — a timeout is a *censored* observation and a
  call that failed at 40 ms is a *missing* one — which is exactly why §3.6 already carries
  **three** `withheldFor` item states. **Operations**: §3.6 clause (iv) re-probes after a
  timeout and deliberately does not after a non-timeout failure. Under the loop `drive` catches
  the exception, so the runner and the scorer see only the `TurnTrace` and can recover none of the
  three facts from a four-member set. The alternative — keeping four members and having the scorer
  re-derive the split from `ItemTiming.withheldFor` — is workable and is precisely the
  two-vocabularies collision this plan has paid for twice (`BinaryMetric.unit` against
  `PackRef.analysisUnit`; `unit_kind` against `analysis_unit_field`), now with a *timing* field
  deciding a *scoring* question; it is rejected on that ground and recorded here so the choice is
  not re-made by default. With `timed-out` split out the set is *two loop outcomes plus the
  transport boundary's own three*, and both `withheldFor` and rule 4's outcome column are total
  maps over it rather than second judgements.
  **The cost is one transcribed constant, and the probe going red is the guard working, not a
  regression.** `convo.TurnDisposition`, `convo.TURN_DISPOSITIONS` and
  `tests/test_convo.py`'s `_DISPOSITIONS_PER_PLAN_3_8_4` landed at `d5b549d` holding the four, and
  legs 1 and 2 of §4 S2's probe redden the moment either module declaration changes. That is the
  failure mode the precursor unit exists to catch; the rework unit updates the transcript to the
  five rows above **in the same change** as the two declarations, in files it is already editing,
  and must not read the red as a defect in its own work. *(`-ml` §4.3.1 item 2 asks for the widen
  to reach that unit "before `TURN_DISPOSITIONS` lands". It landed first — the note was written
  without sight of `d5b549d` — and the sequencing is fine either way: the probe is what makes a
  late widen visible rather than silent, which is the whole reason it was given a round of its
  own.)*

  **The partition of the last two rows is not derivable from the adapter as v1.25 found it, and
  closing that was part of P13-1's ruling — it has landed.** At `40a9bc8` an HTTP 400 — `-ml`
  §4.1's own example, and what cost §8.4 six of eight `gpt-oss-20b` conversations — and a dropped
  connection both raised `LMStudioCallFailed` (`lmstudio.py:485` and `:495`), so the two were
  indistinguishable. `LMStudioCallFailed` now carries **`status: int | None`** — the HTTP status
  when the server answered and refused, `None` when it did not — as a **required keyword argument
  with no default**, and that field, not the exception class, is what `drive` partitions on.
  Delivered at `d5b549d`: twelve raise sites audited, two carrying a status, and a completeness pin
  that computes **both** sides —
  `test_every_raise_site_of_lmstudio_call_failed_is_covered_here` compares the set of source lines
  that raise the class against the set the scenario table actually reaches — so a thirteenth site
  cannot arrive untested. *(v1.26 called this "a one-line adapter change", which under-stated it by
  the site count; the gate's correction, P14-2, named eleven sites while enumerating twelve, and a
  `grep -c` on the constructor returns thirteen because the class statement matches the same
  pattern — which is why the shipped pin derives both sides rather than transcribing a count. The
  precursor unit reached the same two decisions independently, before the pass existed, so P14-2 is
  closed on arrival.)* `timed-out` needs no equivalent field: `LMStudioCallTimeout` is already its
  own class.

  **`iterations` is the count of calls in the turn that *completed*, and it is one number with two
  others: `iterations == len(chatResults) == len(ItemTiming.calls)`** *(v1.27, P14-6's first half;
  the pin is `-ml` §11.9 ask 7's, in its `ItemTiming` item and its test-10b item, and §11.10 (7d)
  asserts it rather than leaving it a convention)*. A turn whose first call raised therefore has
  `iterations == 0` and `calls == ()`; one that raised on its third call has `iterations == 2`.
  **`0` there means *no iteration was observed*, never *the model used no iterations***, which is
  this component's absent-never-zero rule and is half of why `-ml` §4.2(f) keeps such a turn out
  of the `I(t)` summary below.
  *Raise **R-1** to `data-scientist` under §7 rule 3 — a consequence of that pin, not a
  disagreement with it.* Under the completed reading `-ml` §11.4's *"`callCount` is `1` for every
  role but `tool-caller`"* is false on an item whose only call returned nothing (it is `0` there),
  and ask 7's bound `statsCoveredCount ≤ Y_calls − (calls that returned no response)` has an
  **identically zero** subtrahend, since such a call is not in `Y_calls` at all. Two consequences,
  each stated where it is used: §4 S2 rule (iv) carries the bound in the form that holds under
  either reading, and `Y_calls` is a **netted** call denominator — the calls removed from it are
  exactly the ones that could not have carried `stats` — which is what §11.4's own `Y` argument
  refuses one unit up. The plan builds the note's pin; if the note unnets `Y_calls` the change is
  one field's definition plus one test, and nothing stored moves, because §4 S1's `ItemTiming` is
  unbuilt (`git grep -c ItemTiming` over `model-bench/**/*.py` at `d5b549d` → no match).

  **Which turns enter `-ml` §4.2(f)'s `I(t)` mean and p95 is settled, and it is not this
  document's** *(v1.27; P14-6's second half, fenced out of v1.26 and ruled in `-ml` v1.21
  §4.2(f))*. The summary is over `D(t) ∈ {replied, cap-hit}` and no others — `timed-out` included
  in the exclusion, because a budget bounds *how long*, not *how many*. The plan cites it and
  restates neither the reasoning nor the two censoring properties printed with it; what the plan
  fixes is only that **`iterations` is recorded on every turn regardless of disposition**, so the
  summary's population is a filter over the record rather than a decision made while writing it.

  **Precedence, which the loop's control flow determines and which no earlier revision stated**
  *(v1.27, P14-5)*: the exception path is tested **before** the cap, so a call that raises at the
  cap-th iteration is `timed-out` / `no-response` / `server-rejected`, never `cap-hit` — `cap-hit`
  requires a completed response still carrying tool calls.

  **`drive` catches `LMStudioError`, records the disposition, and continues the script** — `-ml`
  §4.1's hard rule is that a turn is never skipped because a previous turn failed, so an error that
  propagated out of `drive` would abandon the script and destroy every later turn's denominator.
  v1.25 left this unstated.

  **The turn's emission form is `-ml` §4.2's own predicate over the turn's dispatch trace —
  `native` iff `|E(t)| ≥ 1` — and is never read from iteration 1** *(v1.26, P13-5)*. The two
  usually coincide, and diverge exactly when iteration 1's tool calls are all **undispatchable**:
  the replay contract provides for that case, `drive` skips a nameless call, so nothing reaches
  `env.trace()`, `|E(t)| = 0`, and the note's partition is `no_attempt` or `prose_pseudo_call` where
  v1.25 said `native`. `E(t)` is the harness's own dispatch trace (`-ml` §4.1) and it is the only
  source. If intent-at-iteration-1 is later wanted, it is a separately named diagnostic field and
  never the partition.

  **No *new* cost, and §4.5.2's minutes are a floor rather than an estimate** *(v1.26, P13-7)*. The
  ~1.3 s/turn basis was measured on falkor-chat's real multi-step executor, so the loop is not a
  cost the loop introduces — but roughly half the turns behind that figure came from a model that
  collapses at turn 4 and were therefore single-iteration, so the derived minutes are a **lower
  bound** whose multiplier is bounded above by `maxIterationsPerTurn`. The 2026-09-02 sizing
  decision is **not** reopened: `-ml` §4.5.3 denominates its reversal trigger in *scripts* and
  states the binding constraint is FR-19 human verification, not compute. §4.5.2's figure is the
  note's and its restatement is routed to `data-scientist`, not made here.

  **How a multi-call turn's timing is taken — plan-gate P13-2, now ruled and carried** *(v1.27;
  the ruling is `-ml` v1.20 §11.4/§11.5.1 and its plan-side list is §11.9 ask 7, which this
  revision folds in section by section)*. The runner builds **one `ItemTiming` per turn** from
  `TurnTrace.chatResults`, in order — one `CallTiming` per completed call (§4 S1). Three
  consequences belong here rather than in §3.6, because they are properties of the turn:
  - The turn's **`wallClockMs` is the item figure**: every iteration and the tool dispatches
    between them, which is the response time an operator waits through, and the only latency an
    operator can act on. Redefining it as one call of the turn would make a model that loops eight
    times report a *smaller* latency than one that answers in a single call.
  - `-ml` §11.5.1's gap detector is **computed per call and summed over the item**; the item's
    `unexplainedMs` is `None` unless **every** one of its calls yields a readable gap. Subtracting
    one call's `ttftMs + generationMs` from the *turn's* wall clock would make the detector fire on
    iteration count — the quantity §4.2(f) exists to measure — rather than on a load.
  - **A turn that ended on a raise has an incomplete wall clock and is not timed**; a `cap-hit`
    turn is the opposite case and must not be folded into it. §3.6's withholding bullet is where
    both are stated, because withholding is a `LatencyBlock` concern.
- **Validation target (§2.2):** running this pack against `qwen/qwen3-4b-2507` and
  `mistralai/ministral-3-3b` must reproduce the documented per-turn contrast. If it does not, the
  harness — not the models — is what has been measured. **The pack declares
  `historyReplay: "structured-replies-only"`** (§3.3) because it reproduces the executor's **replay
  policy** — native roles, final reply text only, no tool evidence — which is the policy §8.2's
  numbers were produced under; asking a known-answer question under a different replay policy asks
  a different question.

  **It reproduces the policy, not the shape, and the difference is three named things** *(v1.26,
  P13-3 — v1.25 wrote "that is falkor-chat's executor shape", an identity claim over a mechanism
  that implements part of it; read at `falkor-chat/server/falkorchat/executor.py:1243-1277`)*.
  `_assemble_messages` additionally (i) **speaker-prefixes** every replayed turn's content,
  `f"{speaker}: {text}"`; (ii) appends a **`CONTEXT:\n<json>` `user` message** carrying the run's
  serialized `run_ctx` on every assembly; and (iii) routes every append through `_append_turn`,
  which **merges consecutive same-role turns** (K-048), so the trailing `CONTEXT` block coalesces
  into a preceding `user` turn. `convo.assemble` does none of the three, deliberately: (i) and (iii)
  are artefacts of a *multi-party* thread, and this pack's conversations are two-party and strictly
  alternating, where `_append_turn` is a no-op by its own docstring; (ii) carries workflow state
  this harness has no analogue of, and a serialized blob repeated every turn is a confound on the
  one covariate the pack is measuring against. What **does** verify as equivalent: native roles,
  final text only, no tool scaffolding, and `historyTurns: 0` matching the executor's
  `THREAD_CONTEXT_WINDOW = 20` on this pack, the 9-turn script being 18 messages. All three
  differences are carried into R-3's candidate causes, so a non-reproduction at S6 step 5 is not
  attributed to the scripts by elimination while two named prompt-shape differences stand.

  That the policy is *reproduced* is verified from the executor's source; that it *reproduces the
  contrast* is what S6 step 5 measures and R-3's bisect answers if it does not.

#### 3.8.5 `chat-responder` — pack `chat-responder-grounded-answers`

**Scope: deterministic layer only, per FR-21a** (amended 2026-09-02). Judged reply quality is
deferred, not cancelled — the deferred design is preserved below and in `-ml` §6.2 so it can be
picked up without re-deriving it.

- **Data: new. 30 items** derived from the copied 121-message corpus, LLM-drafted and **verified by
  a human, item by item** (FR-19). 30, not 20, because of the paired floor (`-ml` §7.1). **The
  10-item `golden_judge_calibration.jsonl` is not copied at first delivery** — it exists only to
  gate a judge, and there is no judge to gate. Item shape, aligned to `-ml` §6.2's **checklist
  ground truth** (v1.1 carried a `referenceAnswer` field that the note's design does not have and
  that nothing in this pack scores — an unscored free-text field beside a deterministic scorer is
  an invitation to a future judge, so it is **dropped**):

  ```json
  {"itemId": "cr-07", "question": "...", "context": ["..."],
   "mustContain": ["24.99"], "mustNotContain": ["19.99"], "mustAbstain": false,
   "format": {"maxWords": 120, "mustBeSingleParagraph": true,
              "forbiddenPatterns": ["^\\s*[-*]\\s", "```"]},
   "provenance": {"draftedBy": "...", "verifiedBy": "...", "corpusVersion": "..."}}
  ```

- **Scoring — three deterministic families, no judge anywhere in the pack.** Mapping FR-21a's
  "latency, format, faithfulness to what the tool actually returned" onto a role that calls no
  tools — **all three ship, where v1.1 shipped two**:
  - **latency** — the standard FR-11 block (§3.6).
  - **format** — checked against the item's own `format` block, with pack-level defaults in
    `pack.json` that an item may override. Three declared constraints, each a separate count and
    never pooled with grounding: word count within `maxWords`, single-paragraph discipline when
    `mustBeSingleParagraph`, and zero matches of `forbiddenPatterns`. The constraints are **pack
    data**, not scorer heuristics — the same rule FR-8(d)'s `boundaryRule` follows, and for the
    same reason: a format rule inferred by the harness is a rule two packs cannot agree on.
  - **grounding** — FR-21a's "faithfulness to what was actually returned", here the retrieved
    context: `mustContain` / `mustNotContain` / `mustAbstain` containment, using the same
    canonicalization as the `nlq-generator` scorer.
  - `verdictMetrics = ["groundingRate"]`, `headlineMetric = "groundingRate"`; the format counts and
    the latency block are exploratory beside it.
- **Cost: moderate, and it is now the *smallest* of the three data-bearing new assets** — 30
  human-verified items, with the expensive half (30 further calibration items plus a judge harness)
  deferred by FR-21a.
- **Deferred design, recorded so it is not re-derived** (backlog, `-ml` §6.1–§6.2): if judged
  quality is funded later, it is **faithfulness only** — the copied calibration record's relevance
  axis is near-worthless once chance agreement is accounted for, while its faithfulness axis is
  usable, and that contrast is what carried FR-21a. (`-ml` §6.1 recomputes both κ, the raw
  agreements and the marginal skew that inflates the relevance figure; the numbers are its, and
  §2.1's inventory row above is the only place this plan repeats one, attributed.) Two rules that
  must not be softened when it is built: the harness **errors out when
  `judgeModel == candidateModel`** and suppresses every judge-mediated number rather than caveating
  it; and the judge is gated on **class-conditional rates, not κ**, at the thresholds and
  calibration-set size `-ml` §6.2 computes — never at thresholds chosen here. A judge that fails
  the gate produces no numbers.

### 3.9 D9 — Measurement and statistics

Owned by [`small-model-benchmarking-ml.md`](./small-model-benchmarking-ml.md). **This section cites
that note; it no longer restates it.** *(Changed in v1.2. v1.1 paraphrased the note's formulas,
constants, thresholds and sample sizes here, and the gate found the two documents had already
silently diverged in three places — the `z` constant, the fixture tolerance, and
`min_detectable_difference`'s coefficient. Two copies of a formula is one copy and one bug. So:
every formula, constant, threshold, denominator, tolerance, bootstrap parameter, sample size and
verdict string lives **only** in the note; `modelbench/stats.py` implements the note's signatures
verbatim and cites it by section in its module docstring; and where this plan needs to name a
number, it names the function that computes it instead.)*

The five conclusions the rest of this plan is built on, as **statements of what is being built** —
each one's arithmetic is at the cited section:

1. **The comparison instrument is the paired difference — now what FR-15/AC-4 require** (amended
   2026-09-02; the amendment is §6 R-9). One instrument **decides** (an exact paired test on the
   discordant pairs) and a second **quantifies** (a confidence interval on the paired difference,
   derived from the same Wilson function used for per-arm reporting). They are never AND-ed into a
   single bloc, and when they disagree both component outcomes are printed in prose. **AC-4's
   not-distinguishable verdict fires exactly when the paired-difference interval includes zero**;
   the sentence it prints is `-ml` §3.2e's and is not quoted here. **The paired *binary* interval
   resamples nothing and takes no seed** *(v1.10, from note v1.11: §3.4 Rule 4's closed-form
   percentile is now **binding**, because the resample's seed — and, at a fixed seed, the row order —
   reached the **verdict** and not only the printed digits)*; **continuous metrics keep the seeded
   paired bootstrap** (`-ml` §3.2d), which is what `sampling.seed` seeds and all it seeds (§3.3). The
   measurements that made the closed form binding, the form itself and its acceptance tests are the
   note's. Per-arm Wilson intervals are
   still printed, labelled *descriptive, not the comparison instrument*, and the superseded
   marginal-overlap check is retained as a **diagnostic line** with a footnote saying why it is not
   the verdict. Test names, the exact constants, the bootstrap parameters and the three verdict
   strings: `-ml` §3.2. Why the old rule could not fire: `-ml` §3.1.
2. **Every report prints its own resolving power**, computed by `stats` from its own **effective**
   *n*, its analysis unit, its design effect and the α the note assigns to each bound it prints
   (`-ml` §3.4 Rules 2–4; the bounds do not share one α and this plan does not name either) — never
   quoted from the requirements, never a literal in the codebase, and never derivable from a bare
   observation count.
   The template, the mandatory sentences and every number in them are `-ml` §7.1/§7.2/§7.3's; the
   note carries the **tool-caller pack's rendered line verbatim** (`-ml` §7.2), and that string is
   what S1's report test asserts against. `min_detectable_difference` is the one function that must
   never be hardcoded, and S1's done-condition tests exactly that.
3. **Each pack pre-registers its verdict family** — `verdictMetrics` (1..k) plus an explicit
   `headlineMetric` that may be `null` (§3.3). Only a `verdictMetrics` member can receive a verdict;
   everything else prints `exploratory — no significance claim`. A family with k > 1 takes
   Holm–Bonferroni, and the α at which each printed figure is computed — the ladder's steps, the
   resolving-power line and the observable floor — is the note's, not this plan's. (`-ml` §3.3 and
   Rule 7; v1.5's "computes its resolving power at α/k" is withdrawn here as a **restatement**, not
   contradicted — see §3.3(ii). v1.1's "exactly one
   `primaryMetric`" is superseded by the 2026-09-02 stakeholder decision on
   `guard-judge`, and v1.2's field names are superseded by §3.3's.)
4. **Per-turn-position rates are computed over conversations, never over turns**, and no interval
   is ever printed over a turn-pooled count. **Under the 12 × 1 sampling design (§3.8.4) the
   clustering treatment is materially different from what `-ml` v1.1 assumed, and the note owns the
   difference**: what `stats.py` must implement for `min_detectable_difference`, `verdict()` and any
   cluster-aware interval — including whether a design-effect input is required and what
   `report.py` must refuse to print without it — is specified there and is not restated, guessed at
   or pinned by signature here. `stats.py` implements the note's surface as written.
5. **The negative control is a first-class feature, not a test fixture**: `compare` run with the
   same model in both arms — **two independent runs, not two copies of one record** — must report
   *not distinguishable*, with discordant counts roughly equal and the difference interval centred
   on zero. It is wired as a CLI-reachable mode (`--negative-control`) because it is the cheapest
   way to catch a whole class of harness bugs that otherwise present as plausible model differences.
   (`-ml` §9; the acceptance run is §5 test 19a.)

### 3.10 Alternatives considered and rejected

| Option | Rejected because |
|---|---|
| Extract `metrics.py`/`nlq_scoring.py` into `model-bench`, falkor-chat imports them | Couples a locked regression gate to a new component for ~40 lines of textbook formulas; the two golden sets must diverge anyway (§3.1). |
| Share one golden set between the two components | Any edit for falkor-chat's benefit silently invalidates every stored `model-bench` result — destroys the tool's stated purpose (§3.1). |
| Clean build, re-draft the golden data | Discards 173 human-verified labels, which are the expensive part (FR-19). |
| Score the embedder through FalkorDB's ANN index | Injects index approximation into a *model* comparison, and does not expose the irrelevant-document scores FR-12's score separation needs (§3.8.1). |
| Seed a FalkorDB graph for the `nlq-generator` pack | Unnecessary: the model emits a structured spec, not Cypher, so an in-process executor is exact and dependency-free (§3.8.3). |
| Item-level interleaving of paired arms | A 16 GB box cannot hold two models; under JIT auto-load, interleaving forces a whole model load per item, per side (§2.5, §3.7). |
| Keep the `lms` CLI as the host-info source | Reachable only through a globbed Windows path, 0.30 s against 1.7 ms for the same fact, and a second surface to keep honest beside the HTTP one the adapter already targets (§2.5, §3.4.4a). **v1.8.** |
| Fingerprint from `/v1/models`, staying on the OpenAI-compatible surface | Returns `{id, object, owned_by}` only: it would cost nine of §3.4.2's model fields and §3.6's eligibility gate, or force the defaults FR-7 exists to refuse (§3.4.4a). **v1.8.** |
| `/api/v0/models` with a `/v1/models` fallback that populates what it can | Buys nothing on the success path — a partial record fails `validate()` anyway — while inviting the "let it fill in the rest" edit. Kept as a *diagnosis* that selects the error message, never as a source (§3.4.4a). **v1.8.** |
| One widened request timeout (300 s everywhere) for the JIT first call | Removes the only signal that a warm call has hung, and lets a mid-run reload pass unremarked into the percentiles (§3.6). **v1.8.** |
| v1.1's `warmupTurns` — drop the first *n* scored latencies | Discards real measurements chosen by position rather than evidence, and leaves *n* as a knob that moves the reported p50. The warm-up call sits outside the item set instead (§3.6). **v1.8.** |
| Store results only as CSV (stakeholder preference read literally) | Cannot represent per-turn × per-failure-kind breakdowns without becoming unreadable; JSON is the truth, CSV is the derived human view (§3.5). |
| The original FR-15 marginal-CI-overlap rule as the decision instrument | Cannot fire at all at n ≤ 40 with baseline ≥ 0.90 — this lab's actual regime — and discards FR-16's pairing (`-ml` §3.1). **FR-15/AC-4 amended 2026-09-02** to the paired-difference interval (§6 R-9); the overlap check survives as a printed diagnostic. |
| Pooling turn-level outcomes into one accuracy rate | Turns within a conversation are not independent; the prior experiment's 280 turns (§2.2) carried roughly the information of its 40 conversations (`-ml` §4.4). Per-turn slices over conversations. |
| 4 distinct scripts × 4 replicates × 3 shapes for the tool-caller pack (v1.1's sizing) | At `temperature: 0` the replicates are near-duplicates, so 48 conversations carried an effective *n* the note put closer to 12 (`-ml` §4.5) — a real precision hidden inside a design effect. **Stakeholder decision, 2026-09-02:** 12 distinct scripts × 1 run (§3.8.4). |
| Replicates at `temperature > 0` instead of more scripts | Informative about run-to-run variance, but it adds a second variance source to a comparison whose subject is the *model*, and it weakens what FR-18's temperature pinning means. Declined with the same decision. |
| `numpy`/`scipy` for the numerics | Zero runtime dependencies keeps old results reproducible years later; the workload is seconds of pure Python. Reversal trigger stated in §3.2. |
| A declarative mini-language for simulated tools instead of pack Python | Would become a worse Python; pack code is content-hashed, so it is versioned data like everything else (§3.3). |
| Replaying each prior turn's scripted `expect` — the "textbook" conversation — instead of what the model produced | Decouples the in-context precedent from what the model under test did, so the validation target stops measuring the documented phenomenon (persistence unreproducible, onset a function of the declared mode); severs the per-turn hazard's conditioning from the accumulation it exists to handle; and puts the replayed context in contradiction with FR-10's stateful ground truth (§3.8.4, which carries all four reasons). **v1.25**, ruling on `convo.py` as built at `40a9bc8`. |
| Making scripted-vs-actual replay a pack knob, so a pack could choose teacher forcing | A genuine ablation, and rejected as a *setting*: `cleanThroughTurn4` under a textbook prefix is not the same quantity as under a real one, so one pre-registered metric name would cover two meanings — M-11 by another route (§3.8.4's own `H` argument). **Reversal trigger:** a pack that wants the ablation gets it as a manifest field **plus a distinct metric name**, never as a second setting under the existing one. **v1.25.** |

---

## 4. Step-by-step implementation

Eight stages. Each leaves the tree buildable and the suite green, and each has a done-condition that
does not depend on the next. Stages S1–S2 are the harness; S3–S7 are packs, ordered so the cheapest
end-to-end proof lands first and the long pole starts as early as its prerequisites allow.

**Every done-condition below runs with `model-bench/` as the working directory**, written here once
so no stage restates it. This is not a formality: the repo has **no root `pyproject.toml`,
`pytest.ini`, `setup.cfg` or `tox.ini`**, so invoking `model-bench/.venv/bin/python -m pytest -q`
from the repo root makes `rootdir` the monorepo, ignores this component's `testpaths`, and walks
into other components' suites — measured during S0 at 9 collected, 8 collection errors, exit 2. The
canonical form, used verbatim in every stage:

```bash
cd model-bench && ./setup.sh && .venv/bin/python -m pytest -q && .venv/bin/ruff check .
```

`ruff check` takes an explicit `.` for the same reason: without a target it resolves from the
current directory and its config discovery is not the same walk as pytest's.

### S0 — Component skeleton

**Create:** `model-bench/{pyproject.toml,setup.sh,run.sh,README.md,AGENTS.md,.gitignore}`,
`model-bench/docs/{BACKLOG.md,HISTORY.md}` + empty `requirements/ plans/ reviews/ test-plans/ test-reports/`,
`model-bench/modelbench/__init__.py`, `model-bench/tests/`.

- `pyproject.toml`: copy `mcp-monitor/pyproject.toml`'s shape — `requires-python = ">=3.12"`, no
  runtime dependencies, `dev = ["pytest>=9.1,<10", "ruff>=0.14,<0.15"]`, ruff
  `select = ["E","F","W","I"]`, `line-length = 100`; pytest `testpaths = ["tests"]`,
  `addopts = '-ra -m "not live"'` and a `live` marker (falkor-chat's convention, §2.4).
- `setup.sh`: adapt `mcp-monitor/setup.sh` (idempotent, `--recreate`, ends with an import smoke test).
- `.gitignore`: `.venv`, `host.json`, `results/transcripts/`.
- `README.md` states the three non-features up front: no CI, no gate, no leaderboard.
- Root `AGENTS.md` gains a `model-bench/` bullet in **Structure** and a row in **Component docs**;
  `docs/requirements/small-model-benchmarking.md` is left where it is (its own footnote says so).

**Done when:** the canonical command above passes from `model-bench/`, with **at least one test
collected**.

**S0 is delivered; this is what shipped** (commit `0522ffd`, `model-bench/docs/HISTORY.md`), and two
points differ from v1.1's text:

- **v1.1 asked for "zero tests collected", which is not a passing state.** pytest exits `5`
  (`EXIT_NOTESTSCOLLECTED`) when nothing is collected, so the `&&` chain could never return 0. S0
  ships **one real test** — `tests/test_package.py`, asserting `modelbench.__version__` equals the
  installed distribution's metadata version — rather than configuring the exit code away. That was
  the right call and this plan adopts it: a permanent "no tests ran is fine" setting would still be
  in place at S5, where it would hide a collection breakage; and the assertion is load-bearing
  rather than filler, because that version string is what stamps `benchVersion` into every run
  record (§3.4.3).
- **`.gitignore` names `results/transcripts/` before `results/` exists.** Deliberate — see §3.2.
- `docs/BACKLOG.md` was **seeded at S0** with the two items §7 carries forward (FR-21a's judged
  layer; the +22 harder retrieval queries). **Confirmed, not reversed:** a deferred item is recorded
  when it is decided, not when the milestone closes, or it is one forgotten commit from
  disappearing. S8 therefore *re-checks and extends* that file rather than creating it.

### S1 — Core: fingerprint, results, stats, report (no model calls at all)

**Create:** `modelbench/{fingerprint,results,stats,report,roles}.py`, `modelbench/cli.py`,
`modelbench/__main__.py`.

Key signatures:

```python
# fingerprint.py
class FieldSpec(NamedTuple):
    tier: Literal["nonempty", "present"]          # §3.4.2 — absent != empty
REQUIRED_BY_SCHEMA: Mapping[int, Mapping[str, Mapping[str, FieldSpec]]]   # {schemaVersion: {armProfile: {field: spec}}}
FORBIDDEN_BY_ARM_PROFILE: Mapping[str, frozenset[str]]                    # §3.4.1, a set operation not a list
BENCH_SCHEMA_VERSION: int = 1                                             # §3.4.3, lives in results.py
@dataclass(frozen=True)
class Fingerprint:
    armKind: Literal["model", "deterministic"]
    callSurface: Literal["chat", "embeddings"] | None   # v1.9 (§3.4.1, §3.4.4a); None iff deterministic
    ...  # every field in §3.4.2, per arm profile
    # armProfile = armKind if armKind == "deterministic" else f"{armKind}:{callSurface}" — the
    # mapping key, derived, never stored twice. Both discriminators are members of no required set.
    def validate(self) -> list[FieldProblem]: ...   # [] means valid; each problem names field + reason
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Fingerprint": ...
    def to_dict(self) -> dict[str, Any]: ...

# results.py
@dataclass(frozen=True)
class CallTiming:                   # v1.27 (-ml v1.20 §11.4/§11.5.1, §11.9 ask 7): ONE MODEL CALL.
    wallClockMs: float | None       # this call's own client wall clock
    ttftMs: float | None            # 1000 x stats.time_to_first_token — normalised in ChatResult
    generationMs: float | None      # 1000 x stats.generation_time
    promptTokens: int | None        # usage.prompt_tokens; prefill is derived, never stored
    tokensPerSecond: float | None   # stats.tokens_per_second, unconverted
    # Every field None-when-absent and never 0. These five are CALL figures: a `tool-caller` item
    # is a turn of I(t) calls, prefill is not even constant within a turn (the prompt grows with
    # each appended `tool` message), and -ml §11.4 pools them over calls rather than averaging up
    # to the item first. They lived on ItemTiming from v1.10 to v1.26, when an item was one call.

@dataclass(frozen=True)
class ItemTiming:                   # v1.10 (§3.6's unit boundary; plan-gate P4-3, -ml §11.9 2b);
                                    # v1.27: SMALLER — three fields, the five scalars moved down.
    wallClockMs: float | None       # the ITEM's wall clock: on a tool-caller the whole turn, every
                                    # iteration and the tool dispatches between them. None when the
                                    # turn is incomplete (it ended on a raise) — a partial turn's
                                    # wall clock is never stored as a measurement (§3.6).
    calls: tuple[CallTiming, ...]   # one per COMPLETED call, in order (-ml §11.9 ask 7). Empty for
                                    # an item whose only/first call returned nothing.
    withheldFor: Literal["load", "timeout", "no_response"] | None   # why latencyMs is absent
    # unexplainedMs and callCount are DERIVED PROPERTIES over `calls`, not stored fields — the same
    # §7 rule 4 move ItemResult.latencyMs already is over this record, applied to it a second time:
    #   callCount     == len(calls)                       # -ml §11.4's per-item Y_calls term
    #   unexplainedMs == sum of each call's own gap        # -ml §11.5.1, over calls in order,
    #                                                      # None unless EVERY call yields a gap
    # v1.11 (plan-gate P5-6, -ml v1.14 §11.5.1): THREE withheld states, ONE counter. `timeout` and
    # `no_response` are both counted in latencyWithheldForNoResponse (§4 S2 (vi)); they are separate
    # item states because §11.5.1's `censoringExact` is satisfied by a timeout (a censored
    # observation) and falsified by a non-timeout failure (a missing one). `None` = timed.
    # On a model:embeddings arm each CallTiming carries wallClockMs only: that surface returns no
    # `stats` (§3.4.4a). An item that returned no response still carries an ItemTiming — withheldFor
    # populated, wallClockMs None, calls () (plan-gate P5-9); `timing is None` means the ARM
    # produces no timings at all (deterministic).

class MetricKindError(ValueError): ...   # v1.12 (plan-gate P6-1, -ml v1.15 §3.2d) — a metric name
                                        # present in BOTH `counts` and `measures`, or
                                        # `scored_outcome`/`scored_value` called for the wrong map.
                                        # The map a name lives in IS the declaration of which
                                        # instrument decides it, so an ambiguity is refused, never
                                        # resolved by code order.
class NonFiniteMeasure(ValueError): ...  # v1.12 — a `measures` value that is not finite, refused at
                                        # construction: one NaN propagates through the mean and both
                                        # percentiles and arrives as a rendered interval rather than
                                        # as an error.

@dataclass(frozen=True)
class ItemResult:
    itemId: str                     # the pack's stable id — the pairing key's first component
    pairingKey: tuple[str, ...]     # components named by the pack's sampling.pairingKey (§3.3)
                                    # the analysis-unit id is pairingKey[pack.analysisUnitIndex],
                                    # resolved from sampling.analysisUnit — never chosen by a caller
    outcome: Literal["pass", "fail", "n_a", "parse_failure"]
    scoreable: Mapping[str, bool]   # per conditional count: was its precondition met? (-ml §4.3)
    counts: Mapping[str, int]       # per-count numerator contributions, plus a pooled metric's
                                    # denominator contribution under "<metric>#denominator" (DC-10)
    timing: ItemTiming | None       # None iff this arm produces no timings at all (deterministic)
    measures: Mapping[str, float]   # v1.12 (plan-gate P6-1; -ml v1.15 §3.2d) — the per-item
                                    # CONTINUOUS values: mrr, separationRaw, separationZ. A SECOND
                                    # map, never a widened `counts`: widening makes the
                                    # booleanisation type-legal without making it wrong, and puts a
                                    # count that §4.2's denominators count into the same key space
                                    # as a measurement they must not. Finite floats; the carrier
                                    # constrains NO domain (mrr is [0,1], sep_z is unbounded — a
                                    # domain here would be Table E's clamp mistake one layer up).
                                    # NOT `float | None`: absence stays `scoreable`'s job, so an MRR
                                    # of 0.0 is a MEASUREMENT and stays distinguishable from an
                                    # unjudged query. Defaults to `{}`, and that is not a
                                    # no-defaults violation (§4 S1's `designEffect` rule): the value
                                    # a forgetful scorer omits is refused loudly one call later by
                                    # `scored_value`, not silently accepted as a number.
    detail: Mapping[str, Any]       # scorer-specific, never read by report.py

    # v1.12 — the invariant that makes instrument selection TOTAL, checked in `__post_init__`:
    # a metric name is in `counts` OR in `measures`, never both (MetricKindError), and every
    # `measures` value is finite (NonFiniteMeasure). Nothing infers a kind from a name.
    def scored_value(self, metric: str) -> float | None: ...
                                    # `scored_outcome`'s sibling over `measures`, the SAME three
                                    # states: absent from `scoreable` -> None; scoreable False ->
                                    # None and -ml §4.3's asymmetry; scoreable True with no entry ->
                                    # IncompleteItemRecord, never 0.0.
    # And `scored_outcome` RAISES MetricKindError on a metric that lives in `measures`, instead of
    # returning `counts[metric] > 0`. That one line is what turns plan-gate P6-1's first silent
    # outcome — a booleanised MRR printed as a McNemar `+X pp` verdict, a different metric under the
    # same name — into a loud failure. `to_dict`/`from_dict` carry `measures`; `from_dict` reads a
    # missing key as `{}`, a reader's compatibility rule under §3.4.3 and not a constructor default.

    @property
    def latencyMs(self) -> float | None:
        """The *admitted* wall clock: the only timing any aggregate reads (§3.6).

        A derivation, not a stored field — v1.10. `wallClockMs` and a stored `latencyMs` would be
        two homes for one measurement, needing an asserted invariant to stop them drifting; §7
        rule 4 takes the derivation instead. `to_dict` still emits it, so a stored record reads the
        same; `from_dict` ignores it and re-derives.
        """
        t = self.timing
        return None if t is None or t.withheldFor is not None else t.wallClockMs

@dataclass(frozen=True) class ContinuousMetric:              # v1.12 gains `support`
    name: str; mean: float; n: int
    support: tuple[float, float] | None                     # the METRIC's support, REQUIRED, no
                                    # default — same discipline as `BinaryMetric.unit`. `report.py`
                                    # is generic over packs and cannot know that mrr is [0,1] and
                                    # sep_z unbounded, so the scorer that produced the figure states
                                    # it, and the paired bootstrap's clamp is derived from it. The
                                    # derivation itself is -ml §3.4 Rule 8's and is not written
                                    # here (§7 rule 2). This is what makes §4 S1e Table E's
                                    # required-with-no-default `clamp` reachable WITHOUT the
                                    # manifest field Table E rejects.
                                    # v1.14 (-ml v1.17): on a VERDICT path `report.py` forwards this
                                    # value as Rule 8's `support` and derives nothing — the producer
                                    # derives the clamp inside itself and exposes none. The one
                                    # caller that states a clamp directly is the EXPLORATORY sep_z
                                    # comparison, which reaches the engine with no metric to ask.
@dataclass(frozen=True) class DistributionSummary:          # v1.12 (plan-gate P6-1, -ml §5.2)
    name: str; median: float; p10: float; n: int; unit: str
    support: tuple[float, float] | None                     # as above
    # STORED FORM — the `"distribution"` type tag, the six keys, `support` as `[lo, hi]` or `null`,
    # and an unrecognised tag raising — is §4 S1e Table F's, decided at v1.14 (plan-gate P7-1).
    # `-ml` §5.2 publishes a MEDIAN and a p10 for sep_z — neither is a mean, so `ContinuousMetric`
    # cannot carry them and a bare `float | None` carries neither. v1.13 (-ml v1.16 §5.2): the same
    # two for sep_raw, and NO MEAN for either; the third figure, the fraction above zero, is P@1 by
    # §5.2's identity and gets no field (§3.8.1, which also carries the no-sep_raw-difference rule).
MetricValue = BinaryMetric | ContinuousMetric | DistributionSummary

@dataclass(frozen=True) class RetrievalAggregates: ...      # recall@k, mrr, p@1, sep_raw, sep_z, …
                                    # v1.12: `separationRaw`/`separationZ` become
                                    # `DistributionSummary | None` — today they are bare floats — and
                                    # `named_metrics()` RETURNS them, which it does not today, so
                                    # `sep_z` reaches a table at all (plan-gate P6-1). The fraction
                                    # of queries with sep_raw > 0 is NOT a field: §5.2's identity
                                    # makes it exactly P@1 (§3.8.1).
@dataclass(frozen=True) class ToolCallAggregates: ...       # per-turn table, hazard, funnel, restraint, …
@dataclass(frozen=True) class ClassificationAggregates: ... # per-class rates + parse failures
@dataclass(frozen=True) class ExtractionAggregates: ...
@dataclass(frozen=True) class GroundingAggregates: ...
Aggregates = RetrievalAggregates | ToolCallAggregates | ClassificationAggregates | ExtractionAggregates | GroundingAggregates

Basis = Literal["by-construction", "measured", "assumed"]   # -ml §3.4 Rule 4's input

@dataclass(frozen=True) class RunResult:
    runId: str; sessionId: str | None; role: str; armKind: Literal["model", "deterministic"]
    fingerprint: Fingerprint; items: tuple[ItemResult, ...]; aggregates: Aggregates
    designEffect: float; basis: Basis           # required, no defaults — see below
    attestationTripWire: Literal["compared", "first-observation", "unavailable"] | None
                                                # v1.10 (§3.4.4, §3.4.5 point 3); None iff
                                                # armKind == "deterministic"; no default

@dataclass(frozen=True) class InvalidRecord:
    path: Path; runId: str | None; benchSchemaVersion: int | None
    problems: list[FieldProblem]; reason: Literal["field", "unknown_schema", "unparseable"]

def store(run: RunResult, root: Path) -> Path: ...        # raises on invalid fingerprint
def load_history(root: Path, *, packId: str) -> tuple[list[RunResult], list[InvalidRecord]]: ...
def rebuild_index(root: Path) -> Path: ...

# stats.py  — implements docs/plans/small-model-benchmarking-ml.md; no other source
_Z_95: float          # pinned to the note's constant (nlq_scoring.py's `_Z_95`), module-level
def wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> tuple[float, float]: ...
def mcnemar_exact(b: int, c: int) -> float: ...              # conditional binomial, math.comb
def mover_d_interval(a: int, b: int, c: int, d: int) -> tuple[float, float]: ...   # Newcombe
def paired_bootstrap(diffs: Sequence[float], *, B: int, seed: int,
                     levels: tuple[Fraction, Fraction]) -> tuple[float, float]: ...
                     # The ENGINE, not an entry point (-ml v1.14 §3.4 Rule 4). Wiring a continuous
                     # verdict straight to it yields a correct interval that silently ignores the
                     # pack's declared design effect. v1.13: `levels` is required with no default —
                     # the two percentiles, hard-coded at 2.5/97.5 today (§4 S1e Table G).
                     # v1.15: its ELEMENT TYPE is `Fraction`, because a k-member family's level is
                     # `alpha/(2k)` and no fixed decimal unit expresses that (-ml §11.2.2).
def paired_cluster_bootstrap(diffs: Sequence[float], *, design_effect: float, B: int, seed: int,
                             clamp: tuple[float, float] | None,
                             levels: tuple[Fraction, Fraction]) -> tuple[float, float]: ...
                     # §3.2d's ENTRY POINT for every CONTINUOUS interval — mrr, sep_z — called with
                     # the pack's declared design_effect (identity widening at 1.00) and the pack's
                     # `sampling.seed` (§3.3), the only thing that field seeds. v1.11: `clamp` is
                     # required with no default — (-1.0, 1.0) for a difference of proportions,
                     # None for sep_z, which is unbounded (§4 S1e Table E). v1.13: `levels` too
                     # (§4 S1e Table G). On a VERDICT path the caller is -ml §3.4 Rule 8's
                     # `continuous_verdict()`, which computes `levels` from the family; only the
                     # exploratory `sep_z` comparison calls this directly (§3.8.1).
def conservative_envelope(table: tuple[int, int, int, int], *,
                          design_effect: float) -> tuple[float, float]: ...
                     # v1.10: the paired BINARY interval is -ml §3.4 Rule 4's closed form, so
                     # `diffs`, `B` and `seed` are gone from this signature and the
                     # `n != len(diffs)` guard goes with them — the error is unrepresentable.
                     # Its MOVER-D arm passes clamp=(-1.0, 1.0) explicitly (Table E).
def percentile(values: Iterable[float], *, level: Fraction) -> float: ...
LEVEL_P50; LEVEL_P95; LEVEL_CI95_LO; LEVEL_CI95_HI          # the whole literal level space
                     # -ml §11.2/§11.2.1/§11.2.2's: the estimator, the integer rank over the level's
                     # numerator and denominator, and its three refusals (empty input; a `float`
                     # level; a level outside (0, 1]). It lands HERE, replacing both shipped
                     # `_percentile` copies, by §4 S1e Table C. The four CONSTANT NAMES are this
                     # plan's, adopted from the note's recommendation; nothing else here is.
# PairedOutcomes, ResolvingPower, resolving_power(), min_detectable_difference(),
# observable_floor(), verdict(), cluster_bootstrap(), design_effect(), effective_n(),
# and — v1.13 — continuous_verdict() with its ContinuousVerdict return:
# signatures and semantics are -ml §3.4's binding rules (the last is Rule 8's). Not restated
# here — see below.

@dataclass(frozen=True) class HolmStep:      # one family member's rung on the ladder
    p: float; rank: int; threshold: float; tested: bool; rejected: bool
def holm_steps(p_values: Sequence[float], *, alpha: float) -> list[HolmStep]: ...  # len == k
def verdict(..., alpha_step: float | None = None, holm_tested: bool = True) -> Verdict: ...
                                             # remaining inputs and all semantics: -ml §3.4
                                             # v1.10: `bootstrap_seed` is REMOVED, with the raise
                                             # that demanded one on the clustered path.
DecidedBy = Literal["mcnemar-exact", "conservative-envelope", "paired-bootstrap"]
                                             # v1.10 renamed the second, §4 S1e Table D; v1.12 adds
                                             # the third for -ml §3.2d's continuous path, because a
                                             # token must name the instrument that actually ran
# No α is named anywhere in this block, and none of these parameters gets a literal default here:
# how many αs there are, which figure each governs, and where their single home is are -ml §3.3
# and §3.4 Rules 2-4's. `alpha_step`'s `None` is a shape (it is knowable only after ranking),
# not a value.

# report.py
def compare_report(runs: Sequence[RunResult], *, pack: PackRef,
                   invalid: Sequence[InvalidRecord] = ()) -> str: ...   # markdown
```

**Three deliberate changes from v1.1's signatures, each closing a gate finding:**

- **`z` is keyword-only and defaults to a module-level `_Z_95` taken from the note**, not to a
  literal `1.96`. v1.1 hardcoded the rounded form while the note settles the full-precision one
  (`-ml` §3.2(a), "the constant, settled"; `falkor-chat/server/tests/eval/nlq_scoring.py:59` is the
  existing pin it points at, and the note's paired interval is derived from Wilson) — so v1.1's
  default would have shifted every bound and failed the note's own regression fixtures. The lab's
  prose is genuinely split on how this constant is *written*, which is why it is pinned in one
  place and cited here rather than restated (v1.7: the digits themselves are the note's).
- **`ItemResult` and the aggregates are specified, not `…`.** They are the load-bearing shapes of
  two guarantees. Pairing (FR-16) needs a stable item identity across runs, which is `pairingKey`;
  `-ml` §4.3's paired-*n* intersection and `asymmetry` count are computed from `pairingKey` plus
  `scoreable`. And "`report.py` has no code path that produces a blended figure" (§3.5) is only
  *structural* if the aggregate is typed: with `dict[str, Any]` the report renders whatever a pack
  put in it and the enforcement is back to convention. The per-role variants are a **closed union**,
  so "no path emits a pooled figure" is a type fact a reviewer can check without running anything.
  S1 is built before any pack exists to constrain these shapes, which is exactly why they are fixed
  here.
- **`compare_report` takes `invalid`.** `load_history` returns two lists and v1.1's signature had a
  parameter for one of them, while AC-2 requires the excluded record to be *named in the report*.

**Two record fields added in v1.5, after S1 shipped them and the gate confirmed they were
required.** `RunResult` carries `designEffect: float` and `basis: Basis`, and done-condition 5b is
unsatisfiable without them: `-ml` §3.4 Rule 4 decides *which instrument may decide* from exactly
these two, only the runner sees the determinism probe that sets `basis` (§5 test 12b, gate finding
N-2), and `report.py` can recompute neither after the fact. Three rules go with them:

- **Neither carries a default on the dataclass.** `designEffect = 1.0` is the anti-conservative
  value, so a default rebuilds **plan-review B-1**'s "default by omission" at the record seam — the
  caller who forgets clustering is exactly the caller that gate found — and it makes DC-5's clause "`report.py`
  refuses to render one when the required input is absent" true only vacuously, because with a
  default the input can never *be* absent. S2's runner must state both.
- **`from_dict` is the one place a fallback belongs.** `d.get("designEffect", 1.0)` /
  `d.get("basis", "assumed")` there mean "a record written before these fields existed" — a
  *reader's* compatibility rule under §3.4.3, not a constructor's default.
- **`basis` degrades fail-safe, in one direction only.** A probe that did not run, or that
  disagreed, yields `"assumed"`, which moves the decision off McNemar and onto the cluster
  bootstrap; a comparison takes the *weaker* of its two arms (`by-construction` only when both arms
  are, and `max()` of the two design effects). That propagation is the mechanism N-2 asked for, so
  it is a test target rather than an incidental line (impl-gate M-3).

`items` is a `tuple`, not a `list`: `RunResult` is frozen, and a frozen record holding a mutable
sequence is frozen in name only.

**The multiplicity surface changed in the fix round, and S2 wires against the new one** *(v1.6,
commit `3ad27d3`)*. `stats.holm_thresholds(p_values, alpha) -> list[float]` is **removed**, not
deprecated; **`stats.holm_steps` returns one `HolmStep` per family member**, carrying `rank`,
`threshold`, `tested` and `rejected`, and `verdict()` gained `holm_tested`. Three things follow that
an implementer must not re-derive:

- **`rejected` and `tested` are separate fields, not one tri-state.** A member past the step-down
  stop still has a printed `threshold` — §3.3(ii) requires the adjusted threshold beside every
  p-value, which is what makes the correction auditable — but `tested=False` says the threshold did
  not decide anything. A `list[float]` could not express the stop at all, which is exactly how the
  Pass 1 blocker was possible: the ladder was printed and plain Bonferroni was applied.
- **`compare_report` runs two passes over the family, and that is structural rather than an
  optimisation.** Holm is a property of the *family*, so no member can be decided until every
  member's p-value exists: pass 1 computes each metric's paired rows and p-value, `holm_steps` then
  ranks the family, and pass 2 renders each metric with the `alpha_step`/`holm_tested` its rung
  gives it. A single-pass report cannot apply a step-down correction, whatever it prints.
- **The list is exactly `k` long, positionally aligned with `verdictMetrics`.** Consumers zip it
  against the family; a short return would silently drop a pre-registered metric from a report
  (impl-gate P2-3), so the length is a postcondition of `holm_steps` and the zip is
  `strict=True`. `k` is `len(pack.metrics.verdictMetrics)` and nothing else — §3.3's pre-registration
  is what makes the correction honest rather than chosen after the fact.

**The family loop is binary end-to-end, and the first real pack's only verdict metric is not**
*(new in v1.12, plan-gate P6-1(b))*. The shipped loop runs `_paired_rows` → `PairedOutcomes` →
`mcnemar_exact` → Holm → `verdict` for **every** `verdictMetrics` member, and its verdict strings
are percentage points throughout — so `mrr` either booleanises into *did this query retrieve
anything* or renders *"No verdict: no paired data"*. Table F gives the per-item value a home; this
is what reads it. Six things, and only the first two are new code paths:

- **Pass 1 resolves each member's kind, and that resolution is a type fact.** A member whose arm
  aggregate is a `BinaryMetric` is binary; a `ContinuousMetric` or `DistributionSummary` is
  continuous. The resolution is **cross-checked against the per-item map** — a binary member's
  values live in `counts`, a continuous member's in `measures` — and a disagreement between the two
  is DC-10's mismatch exactly: **exclude the arm and name it**, never guess. A family mixing kinds
  is refused here too, at a **different granularity and with nothing excluded**: no member of that
  family is verdicted, each is named with its resolved kind, and every one of their numbers prints
  as exploratory (§3.3 (iv), `-ml` v1.16 §3.3 — v1.13 corrects v1.12's *exclude the offending
  members*).
- **The binary branch is unchanged**, including the two-pass Holm structure above.
- **The continuous branch takes one difference per *analysis unit*, never per observation.** For
  every unit present in **both** arms, `value_A(u) − value_B(u)`, where a unit's value is its item's
  `scored_value(metric)` when the unit is one item — the embedder's case, unit ≡ query ≡ item — and
  the **mean over its items** when it is not. A unit scoreable in exactly one arm is **excluded from
  `diffs` and counted into the `-ml` §4.3 asymmetry tally the pairing tally already prints**; a
  silent drop is §4.3's laundering arriving on the continuous path. `diffs` is then handed to
  **`-ml` §3.4 Rule 8's `continuous_verdict()`**, and this loop calls nothing below it: the producer
  reaches `paired_cluster_bootstrap` internally — never `paired_bootstrap` (§4 S1e Table D) — and it
  is the party that computes the family's quantile levels (§4 S1e Table G). `pack.seed` is now
  genuinely consumed, as Rule 8's `seed`, which is what §4 S1e Table D kept it alive for. **The one
  quantity this loop hands the producer beyond `diffs` is the metric's support** *(v1.14, `-ml`
  v1.17)*: it **forwards `ContinuousMetric.support` and derives nothing** — Rule 8 now takes
  `support`, required with no default, derives the difference's own support inside itself and
  exposes no `clamp`, so the sign and order of that conversion are written once and not at each
  call site. A `support` whose `lo >= hi` raises **there**, which is the note's fifth refusal and
  not a check this loop repeats.
- **No p-value, so no rung — and Rule 8 makes that unrepresentable rather than guarded.**
  `holm_steps` is not called on a continuous family (the family-wise correction is taken in the
  interval, §3.3 (iv)) and `resolving_power()` is not called either. Neither is a discipline this
  loop has to keep: `continuous_verdict()` **does not take** a `ResolvingPower`, an `alpha_step`, a
  McNemar *p* or percentile levels, so no caller can supply one by omission or by habit. **Why each
  of the four is refused is Rule 8's argument and is not repeated here**; what this bullet owes an
  implementer is only that the loop must not reach for them.
- **The strings are `-ml` §3.2e's 4 and 5, taken from the note and quoted nowhere here** (§7 rule 2;
  v1.7's discipline about corrected copies). Their four mandatory properties — the metric's own unit
  and never `pp`, the absence of a significance test printed rather than implied, a descriptive
  half-width and never a power claim, and mandatory `B`/seed provenance — are the note's decisions,
  and the last of them is why §4 S1e Table D's retired seed parenthetical is not replaced by another
  one in `report.py`: the provenance is **inside** the string.
- **`DecidedBy` gains a third member, `"paired-bootstrap"`** *(the plan's decision, §7 rule 2 —
  the note left the v1.10 rename here for the same reason; **v1.13**: Rule 8 now carries the same
  token on `ContinuousVerdict.decided_by`, so the two documents name one instrument)*. A machine
  token must name the instrument that ran, and on this path neither `mcnemar-exact` nor
  `conservative-envelope` did. It is free while no stored record carries a verdict.

**The producer exists now, and this plan cites it rather than describing it** *(v1.13; `-ml` v1.16
§3.4 **Rule 8**, written at v1.12's ask under §7 rule 3 — that raise is closed)*. `stats.verdict()`
is shaped around `PairedOutcomes`, a McNemar *p* and a Holm rung, and a continuous verdict has none
of the three, so the note gives it its own producer, **`continuous_verdict()`**, returning a
**`ContinuousVerdict` sibling type** rather than a `Verdict` with six fields left empty. Its
parameters, its negative parameters, its four refusals and its three-decimal rendering precision are
Rule 8's and are **not restated here** (§7 rule 2; v1.7's discipline about corrected copies — v1.12
described this producer's behaviour because no specification existed, and one exists now). **What the note
leaves to this document is one seam, and it is named: `report.py` renders a union of the two verdict
types**, `Verdict | ContinuousVerdict`, so the family loop's two branches converge on one renderer
and neither type is asked to carry a field the other's string prints.

**The `support` seam is closed, and the two callers of the engine stay two** *(v1.14; `-ml` v1.17
rules v1.13's §7 rule 3 raise, adopting this plan's recommendation with a sharper shape)*. Rule 8
takes `support: tuple[float, float] | None`, keyword-only and required with no default, and still
takes **no `clamp`** — a third kind of parameter beside its four refusals, irreducible rather than
absent or derivable, on `design_effect`'s own principle. So this loop forwards the support and the
producer does the rest, as the bullet above says. **What that does not do is retire §4 S1e Table
E's required-with-no-default `clamp` on the engine, which still does real work:** `sep_z` is
reported and not verdicted (`verdictMetrics = ["mrr"]`), so §3.8.1's exploratory comparison calls
`paired_cluster_bootstrap` **directly** and states that engine's `clamp` itself, having no metric
aggregate to ask. Two callers, two surfaces, and neither is a simplification of the other.

*(v1.13 classified the raise non-blocking on a **pack census** — the embedder being the only pack in
§3.8 with a continuous verdict metric, and its `designEffect` 1.00 by construction. The
classification was right and the premise was not load-bearing enough to keep: nothing refuses a
future pack declaring `design_effect > 1.0` on a bounded continuous verdict metric, and a census
goes stale without anyone editing the sentence that rests on it. The note replaces it with the
identity, and this document follows: `_widen` scales half-widths by `sqrt(design_effect)`, so at
1.00 it returns its input unchanged and **no clamp can bind, for every metric** — an unwidened bound
being a bootstrap percentile of per-unit differences, which already lies inside the difference's own
support. The interim behaviour never changed; only the reason it was safe did.)*

The **semantics** these types carry — how many αs the module distinguishes, which one each rung,
bound and sentence is computed at, how the floor interacts with a rung, and what may veto a verdict
— are the note's, and its current revision governs them. This plan fixes the shape and the two-pass
ordering; it deliberately names no α and no α *count* (§3.3(ii), §7 rule 2). *(v1.7, statistics
review n-ML-7: this block previously printed a literal α default on `holm_steps` and described
`resolving_power`'s α in the singular. Both were the note's to state and both had gone stale — the
fix is the citation, not a corrected copy, because a corrected copy is what produced this finding
and the two before it.)*

**The clustering-aware and decision-making surface of `stats.py` is deliberately absent from this
block.** `-ml` §3.4 is now a **binding rule contract** for exactly that surface — written, in its
own words, so that "the anti-conservative version does not typecheck, and the honest one is the only
one that runs" — and it is the single source of truth for `PairedOutcomes` (whose `from_units` is the
*only* constructor and raises on a repeated analysis-unit id), `ResolvingPower`/`resolving_power()`
(whose `design_effect`, `basis`, `unit_kind` and **every α parameter it takes** are keyword-only
**with no defaults** — how many there are and which printed bound each governs is §3.4 Rules 2–4's),
`min_detectable_difference` (which takes `n_effective: float`, never `n: int`), `verdict()` (which
asserts four preconditions and refuses rather than warns), `design_effect` (a **variance** ratio —
the width ratio *squared*), and `cluster_bootstrap` (one level only). Pinning any of these here is
what produced the v1.1 divergence — `min_detectable_difference(n)` against a note that requires the
sampling structure as an input — and this plan does not repeat it. What this plan *does* own is that
S1's done-condition can detect a non-conforming implementation; see below.

`compare_report` is where AC-2/AC-3/AC-4 become visible output: an excluded-invalid block naming
each record and its problems, a pack version/hash mismatch banner, a `SCHEMA VERSIONS IN THIS
COMPARISON` line when records span schema versions (§3.4.3), the resolving-power line (§3.9 point
2), and **the note's verdict-2 wording, taken from the note rather than quoted here** (`-ml` §3.2e),
wherever the decision rule says so. It also carries the two-instrument disagreement case in prose
(§3.9 point 1), and it
renders a `deterministic` arm (§3.4.1) beside model arms without ever ranking two of them against
each other.

**CLI:** S1 ships `compare` (including `--negative-control`), `index rebuild`, and the
stored-records half of `models --tested` (§3.6a). `attest`, `validate` and `run` are S2's.

**Done when**, with unit tests over hand-built `RunResult` fixtures and no LM Studio involved:

1. **AC-2** — a record with a blanked attested field is excluded on read and named, with its
   problem, in the report. Plus the three states of §3.4.2: `residentModelsAtStart: []` is **valid**
   (`REQUIRED_PRESENT`), an empty `modelKey` is **invalid** (`REQUIRED_NONEMPTY`), and a `null` in
   either tier is invalid. **Plus, new in v1.9 (G3-9), the residency *element* shape — stated over
   the element's key set** *(v1.16, impl-gate P5-3; v1.9 stated it as a pair of refused names, and a
   rule written as a list of names is one a tolerance list can be written against)*: an element's key
   set is **exactly** `{id, state}`, both values non-empty strings, and **any other key set is
   invalid**. The rule is key-set-exact, not a denylist, so a key nobody anticipated is refused by
   construction rather than by having been named — and, on the same argument, so is either key of the
   retired `lms ps --json` element, the shape the shipped `conftest.py` fixture declared until
   `8fc2341`. **That retired element is the instance the suite must assert by value, both of its keys
   named in the assertion.** Without it the rule is unpinned in exactly the direction that matters:
   an extra-key rule carrying a one-name tolerance for one retired key passes the **entire** suite
   (impl-gate P5-3's mutation M1, **472 passed**), so the rule is right in the code and held there by
   nothing. **Where that assertion's literal may live is prescribed at §4 S1e Table A's second
   residual and is binding** — the residual counts the lines that *name* the retired key and cannot
   tell a live use from a disowning one, which is why the two are stated together rather than
   separately. The rule holds on **both** snapshots, `residentModelsAtStart` and
   `residentModelsAtEnd` alike (§3.4.2, §3.4.4a); Table A's row names only the second because that is
   the *fixture* site, not because the rule is one-sided. `REQUIRED_PRESENT` checks presence and
   never element shape, so without this assertion a stale fixture validates, ships green and travels
   into S2, where `residency()` emits `{id, state}` and the two disagree with nothing to catch them.
   The assertion is the fix; the fixture edit is a consequence of it.
2. **AC-3** — two runs differing in `packVersion`, and separately in `packContentHash` only, both
   produce the mismatch banner and still render the comparison.
3. **AC-4** — a paired-difference interval that includes zero renders the note's verdict-2 wording
   (`-ml` §3.2e, asserted against the note's string rather than a fragment quoted here), **and the
   40/40 vs 34/40 case does not** — that case's paired difference excludes
   zero and the correct verdict is *distinguishable* (`-ml` §3.1/§3.2; it is the worked example that
   carried the FR-15 amendment, and the one the *old* marginal-overlap rule got backwards). §5 test
   6 states the same thing; these two must not diverge again.
4. **`stats` reproduces the note's `(a,b,c,d)` regression fixtures to the tolerance the note
   states** — the tolerance is the note's to set, and this plan does not restate it. Plus the two
   contract assertions `-ml` §3.4 calls out by name: `PairedOutcomes.from_units` **raises** on a
   repeated analysis-unit id (Rule 1), and the ρ = 1 identity holds (Rule 5 — when within-cluster
   correlation is 1, effective *n* must equal the cluster count; it is the one assertion that
   catches a squaring error in either direction). **Rule 1 is a backstop, not the mechanism**, and
   v1.3 overstated it: `from_units` only raises if the id it is handed is the *cluster* key, so 48
   conversation ids drawn from 12 scripts are unique and would be accepted (gate finding N-1). What
   closes the clustered-design case is Rule 6 plus §3.3's `sampling` contract, at pack validation —
   which is why DC-5(c) below tests *which key is used*, not merely that something raised.
5. **`min_detectable_difference` is verifiably not a constant, and verifiably not naive.** Two
   tests: it returns different values for different *n*; and — the B-1 detector — **calling it with
   only an item count, for a pack whose sampling unit is clustered, must fail rather than return a
   number.** Whatever shape the note gives that input, the assertion is the same: a resolving-power
   line for a clustered pack cannot be produced from a bare *n*, and `report.py` refuses to render
   one when the required input is absent. This is the test that would have caught v1.1's
   `min_detectable_difference(48)` printing a materially over-optimistic figure for the tool-caller
   pack. **It needs a synthetic clustered fixture**, because under §3.8.4's sizing no shipped pack
   is clustered any more — which is exactly why the guard must be tested rather than assumed
   unreachable: the next conversation pack that raises `replicatesPerScript` reintroduces the case,
   and by then S1 is long closed.

   **DC-5(c) — the fixture must assert *which key* is used as the unit id, not merely that something
   raised.** *(Gate finding N-1. Written to be lifted verbatim into the S1 brief.)* The fixture is an
   in-memory `Sequence[ItemResult]` plus a `PackRef` — S1 has no pack loader, that is S2 — built as
   **12 clusters × 4 rows = 48 rows**, each row's `pairingKey` being a **unique** `(scriptId,
   replicate)` pair, with `PackRef` declaring `pairingKey = ["scriptId", "replicate"]` and
   `analysisUnit = "scriptId"`. The test asserts all three of:

   1. **Identity of the unit ids.** Capture the `unit_ids` argument actually passed to
      `PairedOutcomes.from_units` and assert it is the rows' **`scriptId`** values — the *outermost*
      component of `pairingKey`, 12 distinct values each appearing 4 times — resolved from
      `PackRef.analysisUnit`. Assert the captured argument itself, not a property inferred from the
      outcome.
   2. **Consequence.** `from_units` therefore raises on the repeated unit id.
   3. **A negative control on the guard itself.** Passing the 48 *unique conversation ids*
      (`f"{scriptId}-{replicate}"`) to `from_units` directly is **accepted without error** — which
      proves `from_units` alone does not close this, and that assertion 1 is what does.

   **A test that asserts only 2 passes while testing nothing**: 48 conversation ids are unique, so
   the wrong unit-id choice raises nothing and the fixture goes green on a harness that would
   silently produce an anti-conservative verdict. Assertions 1 and 3 are what make the test real.
5b. **The rendered resolving-power line matches the note's verbatim string.** `-ml` §7.2 carries the
   tool-caller pack's line as four mandatory sentences; the report test asserts against that string,
   parameterised only by `<packId>@<packVersion>`. This is the acceptance surface for §3.9 point 2:
   a line missing the unit, the design effect, the best-case caveat or the conditionality clause
   fails, whatever number it prints.
6. **`armProfile` (B-3, G3-1) — three profiles, and the acceptance surface asserts all three.**
   *(Rewritten in v1.10, plan-gate P4-2: v1.9 replaced the two-kind contract in §3.4.1 and left this
   done-condition and §5 test 1 asserting the retired one.)* A `deterministic` fingerprint with no
   model fields **validates**; the same record with `modelKey: "bm25"` added **fails on write** as a
   forbidden field. A **`model:embeddings`** record **without** `runtimeName` **validates** — that
   field is not in its required set — while the same record **carrying** any of `runtimeName`,
   `runtimeVersion`, `temperature` or `maxTokens` **fails as forbidden**; a **`model:chat`** record
   missing `runtimeName` fails as **absent**. (v1.9's *"a `model` record missing `runtimeName`"* is
   false for `model:embeddings`, where the field is forbidden rather than required — the sentence
   asserted the contract the same revision replaced.) A record whose `armKind` is `model` and whose
   `callSurface` is absent fails with `FieldProblem("callSurface", "absent")` **before** any mapping
   is consulted, and `armKind == "model"` remains a valid membership answer after §4 S1e Table B's
   re-key — the assertion that catches the mechanical failure plan-gate P4-2 names. All three forbidden sets
   are pinned against independently written literals (§3.4.1), and `compare_report` renders a model
   arm and a deterministic arm in one report without ranking two deterministic arms against each
   other.
7. **Schema versioning (M-4)** — a record written under `benchSchemaVersion: 1` still validates
   after a hypothetical field is added at version 2, and a record declaring version 99 lands in
   `invalid` with `reason == "unknown_schema"`.
8. **`headlineMetric: null` (§3.3)** — a pack fixture with two `verdictMetrics` and a null
   `headlineMetric` renders both verdicts and **no headline**; a fixture omitting the
   `headlineMetric` key entirely fails validation.
9. **`--negative-control` smoke check** — two copies of the same stored run report
   not-distinguishable. Labelled a smoke check in the test's own docstring, because with two copies
   `b = c = 0` by construction and it **cannot fail**: it proves the mode is wired, not that the
   harness is sound. The real negative control is two *independent* runs of the same model and is
   §5 test 19a, an acceptance step.
10. **The `aggregates`-versus-`items` cross-check.** *(New in v1.8. The engineering gate's Pass 4
   finding **impl-gate P4-4** overturned the S1 fix round's deferral of this to S2, and the deferral's two
   grounds were re-checked and do not hold: `_DESCRIPTIVE_NOTE` caveats the per-arm **interval**
   ("descriptive, not the comparison instrument") and says nothing about the **rate**, which is the
   thing that misreports; and `RunResult` carries `items` and `aggregates` as required fields side
   by side, so the check needs nothing S2 produces. It is S1-local, and it is the net that catches
   the first real scorer's first mistake — which is why it must exist before S3 produces any.)*

   **The check** *(predicate corrected in v1.9 — G3-6; v1.8's wording named two disjoint
   vocabularies and could never be true. In v1.10 the same predicate stops being a **filter** and
   becomes a **selector between two arithmetics** — plan-gate P4-8, below.)* In `compare_report`,
   for each arm and each `BinaryMetric` the arm declares, the predicate

   > **`metric.unit == roles.unit_kind(pack.role)`** — imported in `report.py` as
   > `unit_kind_for_role`, and the predicate the shipped report already uses —

   chooses the arithmetic. Where it holds, `metric.n` must equal the
   number of that arm's items for which `scored_outcome(metric) is not None`. **Not
   `metric.unit == pack.analysisUnit`**: those are two namespaces that never intersect —
   `BinaryMetric.unit` is a denominator noun (`item`/`conversation`/`query`/`turn`/`call`, `-ml`
   §3.4's `unit_kind` column) while `PackRef.analysisUnit` is a `pairingKey` **component name**
   (`scriptId`), pinned by §3.3's contract to `pairingKey[0]`. A cross-check whose selector is never
   true is a cross-check that never fires, which is the worse failure of the two: it passes.

   **Pooled metrics are cross-checked too, and v1.9's residual is closed rather than disclosed**
   *(v1.10, plan-gate P4-8)*. v1.9 said a `BinaryMetric` whose `unit` is `turn` or `call` is **not**
   cross-checked "because its denominator is not an item count and no arithmetic over `items` can
   confirm it", while §4 S2's scorer contract said **every** `BinaryMetric.n` is computed as an item
   count. Both cannot be true — under the unqualified contract a pooled metric is unconstructible by
   a conforming scorer, so the residual was unreachable rather than open — and the gate was right
   that fixing it needs no S2 design work. Two changes, and the residual disappears:

   **(i) A pooled metric can never be a *verdict* metric.** `validate_pack` refuses a pack whose
   `verdictMetrics` contains a member whose declared `unit` is finer than
   `roles.unit_kind(pack.role)`. A pre-registered member receives a verdict, a verdict needs an
   interval, and `-ml` §4.4 forbids an interval over a turn-pooled count — so such a member is
   unprintable by the note's own rule and refusing it at validation is naming that, not adding a
   policy. It is a **missing capability rather than a guarded one**, the shape §3.5 gives FR-20.
   Pooled metrics stay fully reportable as **exploratory**, without an interval and without a
   verdict, which is what they already were.

   **(ii) A pooled denominator is checkable by the same arithmetic, because the scorer declares it
   per item.** Every pooled `BinaryMetric` records its per-item denominator contribution in
   `ItemResult.counts` under the reserved key `"<metric>#denominator"` (§4 S2's scorer contract;
   `validate_pack` refuses a metric name containing `#`, so the namespace cannot collide). The
   check is `metric.n == sum(item.counts["<metric>#denominator"] for item in arm.items)`, and a
   pooled metric with no contribution on an item that marks it scoreable is the sibling
   malformation — `IncompleteItemRecord`, treated as a mismatch exactly as below.

   So DC-10 ranges over **every** `BinaryMetric` an arm declares, with one arithmetic per unit kind,
   and impl-gate P4-4's defect — a rate printed for a metric no item declares scoreable — is no longer
   printable for any of them. **The alternative the gate offered was to scope §4 S2's sentence and
   leave the residual standing.** Rejected: the residual would then be *deferred by choice* rather
   than blocked on anything, since (i) is a validation rule and (ii) is a dictionary key, neither
   needs a scorer that does not exist yet, and both are testable at S1 against synthetic fixtures —
   the argument DC-5(c) already makes about a guard no shipped pack currently reaches.

   **A third arithmetic, for a continuous member** *(v1.12, plan-gate P6-1)*. The shipped selector
   at `report.py:211` reads `if not isinstance(metric, BinaryMetric) …: continue`, so a
   `ContinuousMetric` or `DistributionSummary` is cross-checked by **nothing** today. It widens: for
   a continuous member, `metric.n` must equal the number of that arm's items for which
   `scored_value(metric) is not None`. **The same check is where a kind disagreement surfaces** — a
   member whose aggregate is continuous while its per-item values live in `counts`, or the reverse,
   is a mismatch of exactly DC-10's class, because the arm's aggregate path and its per-item path
   disagree about what was measured. Excluded and named, never reconciled. `scored_value` raises
   `IncompleteItemRecord` on the sibling malformation for the same reason `scored_outcome` does, and
   a `MetricKindError` from either is caught and reported on the same route, so neither escapes as a
   traceback (G3-7's rule, applied to the new call).

   **The counting call has a raise path, and the check must own it** *(v1.9, G3-7)*.
   `ItemResult.scored_outcome` does not return `None` for the sibling malformation: a metric
   declared `scoreable: True` with no entry in `counts` **raises `IncompleteItemRecord`**, and
   `cli.py`'s `_cmd_compare` catches only `PackConfigError` — so an uncaught raise from inside this
   very check would escape as a traceback at exit 1, outside §3.6a's closed set, which is exactly
   the impl-gate P4-5 shape DC-10 rejects "raising" for. So: **the cross-check treats `IncompleteItemRecord`
   as a mismatch** — catch it, exclude the arm, and name it in the `INVALID RESULTS EXCLUDED` block
   with the offending item id and metric. (That is DC-10's own path only. impl-gate P4-5's separate route
   through the paired intersection is the implementer's fix, not this one.)

   Today's failing case
   is an arm declaring `BinaryMetric(successes=0, n=10)` for a metric **no item declares
   scoreable**, which still prints `0/10 = 0.000` in the Arms table beside a paired-rows section
   saying *"No verdict: no paired data"* — one report making two mutually exclusive statements about
   the same metric.

   **On mismatch the arm is excluded from the comparison and named in the existing
   `INVALID RESULTS EXCLUDED` block**, with the declared `n` and the counted one printed, and the
   remaining arms still render. Two rejected alternatives, and each is rejected for a reason the
   reviews already paid for: **raising** reproduces impl-gate P4-5's shape — an exception thrown at report
   time, outside `cli.py`'s closed exit-code set, taking the valid arm down with the invalid one —
   whereas exclude-and-name is AC-2's own mechanism, already built and already tested; and
   **suppressing just the offending metric's row** leaves a partially-trusted arm inside a
   comparison, when what the mismatch has actually demonstrated is that this scorer's per-item and
   aggregate paths disagree, which is no basis for trusting its other denominators. If exclusion
   leaves fewer than two arms, the report prints the no-comparison case it already has.

   **The test** builds two arms of ten items each with every item `scoreable={m: False}` and each
   arm's stored aggregate declaring `BinaryMetric(m, successes=0, n=10, unit="item")` — the gate's
   own reproduction — and asserts that no `0/10` row is rendered, that both arms are named in the
   excluded block with `declared n=10, counted 0`, and that the report is still produced. **The
   fixture must pin a `PackRef.role` whose unit kind is `item`** — `guard-judge`, `nlq-generator` or
   `chat-responder`; the embedder's is `query` and the tool-caller's is `conversation` — or the
   fixture's `unit="item"` metric falls outside the selector and the test passes while testing
   nothing, which is DC-5(c)'s failure shape a second time. A **third arm** carries the
   `scoreable: True` / no-`counts` malformation and must be excluded and named rather than raising.
   A **fourth arm** carries a **pooled** metric (`unit="turn"` under an `item`-unit role) whose
   declared `n` disagrees with the sum of its `"<metric>#denominator"` contributions, and is
   excluded and named the same way — the assertion that makes (ii) real rather than declarative.
   §5 test 11c is the same requirement stated from the test side.

11. **The timing carriers exist and the derivation is the only route to `latencyMs`** *(v1.10,
   plan-gate P4-3 and `-ml` v1.12 §11.9 ask 2b)*. `ItemResult.timing` is an `ItemTiming` or `None`,
   and `ItemResult.latencyMs` is a **property**, not a field: an item whose `timing.withheldFor` is
   set reports `latencyMs is None` **while `timing.wallClockMs` still holds the measurement**, which
   is what makes `-ml` §11.6's promise — *the summary is withheld, not the data* — true of the record
   rather than only of the prose, and what `censoringExact` compares. Three assertions: the property
   returns `None` for each of the **four** absent cases (`timing is None`, `withheldFor == "load"`,
   `withheldFor == "timeout"`, `withheldFor == "no_response"` — v1.11, plan-gate P5-6) and the wall
   clock for a timed item; a round-trip through
   `to_dict`/`from_dict` preserves `timing` and re-derives `latencyMs` rather than reading the
   stored copy; and **no printed *number* is computed from `timing.wallClockMs`** — every aggregate
   in `results.py` and `report.py` reads the admitted `latencyMs`, and the one permitted reader of
   the raw wall clock is `censoringExact`'s comparison (§4 S2), which selects a **string** and never
   contributes to a figure. The test captures the values each aggregate is handed, the way DC-5(c)
   captures the unit-id argument, rather than inferring it from the output.

12. **Every residual §4 S1e states is re-run and hits its stated target** *(v1.10, §7 rule 5;
   reworded at v1.11 to assert what a residual actually proves — plan-gate P5-2; the Table B and
   Table F rows are v1.12's, the Table G row v1.13's — plan-gate P6-2, P6-5, P6-1; **the target is
   zero for every one of them except **five** — v1.19**: Table C's third, whose target is **two named
   lines** and where the count is expressly *not* the check (impl-gate F2); Table E's pair, whose
   targets are **1** each because they are third-form and stated over the text the edit creates
   (impl-gate F1); and Table H's second and third, **1** each for the same reason. v1.15 stated this
   exception as "Table C's third, target 1" and v1.17 as four; every part of that has moved at least
   once, which is why this condition is worded over *a stated target* and never over zero — and why
   the exceptions are counted here rather than described)*. Each of §4 S1e's **eight**
   tables re-runs its enumerating commands and asserts its stated residual: zero for
   `lmsCliCommit` and `sizeBytes` (A); zero for `FORBIDDEN_BY_ARM_KIND`, zero for
   `frozenset(FORBIDDEN`, zero for `REQUIRED_BY_SCHEMA[1]["model"]` and zero for the
   `REQUIRED_BY_SCHEMA[1]` **key-set assertion** the third profile makes unwritable (B); zero for
   `_percentile` in `modelbench/results.py`, **zero for `_percentile` in `modelbench/stats.py`, and
   — v1.17 — *two named lines, not a number*, for the package-wide percentile-definition check
   `-ml` §11.10(3) states** (C);
   zero for the **whole-identifier** `bootstrap_seed` and for the `cluster-bootstrap` token (D);
   **one — not zero — for each of `_widen`'s two prescribed post-edit clamp expressions, the
   third-form pair v1.17 puts in place of the two that read clean on the half-application** (E); zero
   for each of the two bare-`float` separation annotations **and for `_decode`'s
   `{"binary", "continuous"}` tag literal** (F); zero for
   **`percentile(means, level=LEVEL_CI95_LO)` and for `percentile(means, level=LEVEL_CI95_HI)`** (G);
   and — v1.19 — zero for `clamp=(-1.0, 1.0)`, **one each for the composer's two prescribed
   post-edit clamp expressions**, zero for `verdict()`'s inline two-token attribution, zero for the
   renderer's one-token `p=` condition and zero for the two-token `bound_by` annotation (H).
   **That is twenty-four — A 2, B 4, C 3, D 2, E 2, F 3, G 2, H 6 — each stated on its own
   table as a command with a count** *(v1.12 wrote out the five that were prose — plan-gate P6-5's
   lesson applied to every table rather than only to the row it was raised against)*, and each was re-run against `5878014` — twelve at v1.12, Table G's at v1.13, three more at v1.14,
   and **the sixteen that are stated over the shipped tree again at v1.15**, several of them having
   changed spelling in that revision *(v1.16: `8fc2341` landed Tables A and B, so the four files it
   touched — `fingerprint.py`, `conftest.py`, `test_fingerprint.py`, `test_results.py` — no longer
   match `5878014`. Every figure re-run at v1.16 is stated against **`612888c`** and named there;
   every pin into `stats.py`, `results.py` and `report.py` still resolves at `5878014`, those three
   being **byte-identical** across the two commits — verified by hash, not assumed)* *(v1.17:
   `cc28d48` then landed Tables C, D, E and G, so `stats.py`, `results.py` and `report.py` have
   moved too and **no pin into any file under `model-bench/` resolves at `5878014` any more**.
   Every figure re-run at v1.17 is stated against **`cc28d48`**; a *before* value is re-derived from
   the `5878014` blob with `git show`, which is now the only way to reach it. **All eighteen were
   re-run at `cc28d48`, and each of the three commands v1.17 changes was scored on three states —
   before, faithful, and the half-application it must catch** — because F1's whole finding is that a
   residual can read its target on all three)* *(v1.19: `7f865e2` landed the impl-gate Pass 8 fix
   round, so `stats.py` and `report.py` moved again. **All twenty-four were re-run at `7f865e2`** —
   the eighteen at their stated targets and Table H's six at their stated *before* values — and
   `model-bench/` was byte-identical from `7f865e2` through HEAD when v1.19 was written, and has moved
   twice since. **The component's current commit is `e162ba9`** *(v1.23, replacing v1.21's `93b0e42`
   statement rather than stacking on it)* — the N4 fix, which put two `math.isfinite` guards into
   `stats.py` and took it from **1 418 to 1 449 lines**, in two blocks — **+12** after `:186` and
   **+19** after `:262`, in `93b0e42` numbering — so a pin above `:186` does not move, one between
   the blocks moves by 12, and one below `:262` moves by **31**; and added 123 lines to
   `tests/test_stats.py`, all of them after `:1423`. Suite 577 green, ruff clean, tree clean at that
   commit.
   **What re-points and what does not** — the rule that settles it is §7 rule 5's, stated there
   rather than here. The **two unlanded tables, H and F, are re-measured and re-pointed** to
   `e162ba9` in this revision, which is what licenses moving their baselines; the **six landed
   tables keep the commits their authors measured at** — A and B at `8fc2341`, C, D, E and G at
   `cc28d48`, each named on its own `Landed:` line — because re-pointing a record falsifies what it
   says was checked. *(Six, and the derivation is the check: eight tables, six `Landed:`
   lines. §7 rule 5's v1.21 occasion note counted one more than that, having included Table F, which
   has not landed; corrected there at v1.23.)*
   **What moved:** Table H's two enumerating commands — `bound_by` **14 → 15** and `envelope_arms`
   **21 → 22**, both the same added line in `tests/test_stats.py` — and its five `stats.py` site-row
   pins, each by exactly 31 (`:382`/`:387` → `:413`/`:418`, `:413` → `:444`, `:896` → `:927`,
   `:1168` → `:1199`, `:1184-1187` → `:1215-1218`). **What did not move: any residual count in this
   section that was re-run.** Table H's six read **2 / 0 / 0 / 1 / 1 / 2**, Table F's three read
   **1 / 1 / 1**, and Table E's landed pair — which pins `_widen`'s body, the function the N4 fix
   edited — reads **1 / 1**; eleven for eleven, re-run at `e162ba9`. Table F's eight enumerating
   commands were re-run in the same pass and their counts and per-file shares are **identical to
   `5878014`**; only the line numbers inside their glosses moved, and those are re-derived on that
   table. Neither Table H nor Table F inherits `5878014` any more.
   **The gap this paragraph carries forward, since carrying it once is the whole point:** the six
   landed tables' line pins are `8fc2341`- and `cc28d48`-era records and are **not re-pointed**,
   which is correct and is not a defect to fix. The ten `stats.py` lines Tables E, G and H pinned had
   held unchanged from `cc28d48` through `93b0e42` (v1.21, confirmed there); `e162ba9` is the first
   commit to move any of them, and it does not move all: Table E's `:261`, its third-form pair's
   anchor, now sits at `:292`, while Tables C and G's low pins sit above the first inserted block and
   are unmoved (`:159` and `:162` re-read at `e162ba9`, byte-identical to `cc28d48`). What carries
   the check where a pin did move is the residual's **count**, which did not, exactly as Table E's
   own cost paragraph predicted. Also outstanding: §3.3 (iv)'s two
   `report.py` pins for the refused-family label are `5878014`-era and belong to §3.3, which no
   revision has re-measured — §4 S1e Table F cites them and, from v1.23, no longer restates their
   numbers.
   **This paragraph is itself dated, and that is deliberate rather than overlooked** *(v1.22, raised
   by plan-gate Pass 11 as an observation rather than a finding)*: it names a commit, so it goes
   stale the next time an unrelated unit lands, exactly as the gloss it replaced did. What makes it
   safe where the gloss was not is that **this done-condition re-runs at the end of the round** — the
   gap statement is re-derived by the same pass that re-runs the twenty-four, so it cannot outlive
   one revision unnoticed. It is the one part of convention 1 with a mechanism behind it rather than
   a rule, and it is written here rather than on a table for that reason.)* — and shown
   both non-zero now **and** at its target after a faithful edit — the second half is the check v1.10 and
   v1.11 each shipped one residual without (plan-gate P5-2's rejected residual, then P6-2).
   **Table G's two are the exception, and the exception is stated on its own table**: their *before*
   is the intermediate state after Table C, not the shipped tree — both return **0** against
   `5878014`, measured — so they are re-run at the **end** of the round, when both tables have
   landed, and it is the fixed C-then-G order that makes their *before* reachable at all
   *(v1.15, plan-gate P8-1(d))*.

   **A third property joins them at v1.14, and it is a property of a table's residual *set*:** where a
   table retires more than one literal, its residuals distinguish them, so no *half*-applied edit of
   that table can pass (§7 rule 5(b)). **All eight satisfy it. The sweep is re-run per table at each
   revision** *(v1.15: two of the last three revisions shipped a table that broke the property the
   same revision wrote, so this is a standing pass and not a reviewer's catch)*, **and the eight
   partition four ways** *(the fourth is v1.19's, and it is one table)*:
   **H** is the first table designed against this property rather than corrected into it — its
   composer edit clamps **two** bounds and its residual pair is stated one per bound, so the
   one-sided half-application reads 1 and 0 and the numbers say which half was skipped; its arm-clamp
   residual carries the same distinguishing property in a single command whose count is 2, which is
   rule 5(b)'s own named alternative. The other three ways are the pre-existing:
   **E** (the two `_widen` bounds) and **G** (the two quantile levels) each retired two literals
   under one residual and each gained its second at v1.14; **C** gained its second and third at
   v1.15 — its edit retires `_percentile` at **six** production sites across **two** files and its
   residual counted three in one of them, so the `stats.py` half was unmeasured and its survival
   silent (plan-gate P8-3; v1.14 recorded this table as *already satisfying* the property on a count
   of three where there are six, which is what a hand-written count beside the invariant it
   instantiates does); and **A**, **B**,
   **D** and **F** already state one residual per retired token, which the enumeration above shows
   on its face. *(**A** was re-derived rather than inherited at v1.15, because it is the one of the
   four whose answer is not on the enumeration's face — and **corrected at v1.16, because that
   derivation carried only one of its two halves** (impl-gate P5-3). Its residency-**element** row
   retires two keys, `modelKey` and `sizeBytes`, under a residual over the second alone. **The
   `modelKey` half, which v1.15 reasoned correctly:** that key keeps its meaning everywhere else in
   the fingerprint — `grep -rFn modelKey modelbench tests --include='*.py'` → **97** lines at
   `cc28d48`, 94 at `612888c`, 90 at `5878014`, all three re-derived — so a residual over it would
   fail on a faithful edit, and the number rising across three commits with nobody editing it is
   the point rather than an aside; which is the
   trap rule 5(b) forbids; rule 5(b)'s named alternative applies and the row names it, **DC-1's
   element-shape assertion**. **The `sizeBytes` half, which v1.15 never reached, one paragraph
   away:** the residual the row *does* state was scoped across `tests/` as a whole, and the very
   assertion the first half routes to has to **name** that key in order to make it. So v1.15 held
   *the residual forbids the assertion* on one key and *the assertion covers the residual's gap* on
   the other, and the consequence was that DC-1's `sizeBytes` half was asserted nowhere and could
   not be — which impl-gate P5-3 demonstrated by building it. **Both halves now rest on one
   discharge:** the rule is key-set-exact (DC-1), the assertion names both retired keys by value,
   and Table A's second residual is scoped to `modelbench` and `tests/conftest.py` so that it cannot
   see that assertion. The half-application that swaps one key and keeps the other fails at DC-1's
   assertion, and for the fixture element itself at the residual.)*

   *(Plan-gate P7-2 found it in Table G — wiring one level and
   leaving the other passed the check and printed an uncorrected bound; Table E had the identical
   shape, and its half-application would have shipped that table's own defect intact. Plan-gate
   P8-1 found the **cross**-table form: each of C's and G's residual sets was sound read alone, and
   C's edit alone drove both of G's to zero.)*

   **A second standing sweep joins it at v1.16, and it runs over the residual *set* rather than over
   the tables: no residual may be raised by a *disowning mention* — a line that names a retired
   token in order to record its retirement** (§7 rule 5(b)). Derived from the eighteen rather than
   listed beside them, over the scopes **as v1.15 stated them**: the **ten** scoped to `modelbench`
   or to one file under it (B 2; C 1, 2; E 1, 2; F 1, 2, 3; G 1, 2) are unreachable from a test at
   all and need checking only against a production comment or docstring; the **eight** whose scope
   reaches `tests/` (A 1, 2; B 1, 3, 4; C 3; D 1, 2) are the ones an assertion can raise. **Six of
   those eight are clear, on direct evidence rather than on inference:** A 1 and B 1, 3 and 4 each
   read their stated target **0** at `612888c`, on the faithful, landed implementation of the two
   tables that own them; D 1 and D 2 are named by no done-condition in this plan and by no `-ml`
   acceptance item, so nothing obliges a test to spell either token. **The other two were traps and
   both close here** — **A 2**, which impl-gate P5-3 found by building the counter-implementation
   the residual made unassertable, and **C 3**, which this sweep found: its pattern matches a test
   *function* name, and `-ml` §11.10's acceptance items land as exactly such tests with Table C.
   After both re-scopings six residuals reach `tests/`, and the count of eighteen is unchanged —
   neither correction adds or removes a command. *(v1.19 re-runs the sweep over Table H's six and
   adds nothing to either column: **none of them reaches `tests/`**, for the scopes stated in that
   table's residual block and not restated here, so no assertion can raise one and the question does
   not arise *(v1.23, plan-gate P12-5 — v1.22 re-scoped all six to a single file each and left this
   parenthetical saying `modelbench` alone; the conclusion survives *a fortiori*, a narrower scope
   seeing strictly less, but a scope claim gets one home like any other fact, §7 rule 4)*. The one
   candidate that would have been exposed was caught at design time and is recorded on the table — a
   residual over the bare `(-1.0, 1.0)` rather than over `clamp=(-1.0, 1.0)` reads **3**, the extra
   line being a sentence in `_widen`'s own docstring explaining why that default is refused; the
   count and its commit are that table's. Sixteen of the twenty-four are
   now `modelbench`-scoped and eight reach `tests/`, unchanged.)*

   **A third standing sweep joins them at v1.17, and it is the one that catches a residual which is
   blind rather than merely narrow: does the span a residual matches include text that its own
   table's edit rewrites?** (§7 rule 5(b)'s third form). Derived over the eighteen, not listed
   beside them: **sixteen are robust**, and the reason is structural rather than lucky — the span is
   either the retired token entire (A 1, 2; B 1; C 1, 2; D 1, 2; F 1, 2, 3), or a construct whose
   *form* the edit leaves alone while changing a name, a key or a value inside it (B 3's subscript,
   B 4's set display, C 3's `def` line, G 1 and 2's call site). **Two were blind — Table E's pair**
   — and they are the whole of F1: that table is the only one in §4 S1e whose edit
   **parameterises**, turning a literal into an argument and so rewriting the expression the literal
   sat in. Both are replaced with third-form residuals at the table. **One near-miss was examined and
   cleared rather than waved through: Table B's `frozenset(FORBIDDEN`.** Its span does include call
   syntax, but the case it exists to catch — the coupling left in place — leaves that span
   untouched, so the residual fires; it goes blind only for a coupling *respelled* (`set(…)` for
   `frozenset(…)`), which is neither the prescribed edit nor a half of it, and which is killed
   loudly and independently by `ARM_KINDS`'s by-value pin in `test_fingerprint.py` — the mutation
   that couples `ARM_KINDS` back to the forbidden mapping fails **201** tests. That is rule 5's own
   *residuals plus the suite*, named at the residual rather than assumed. **16 + 2 = 18**, and the
   count is unchanged: E's pair is replaced, not supplemented, because rule 5(b) forbids keeping a
   check that reads clean on the defect it was written for. **v1.19 runs the same sweep over Table
   H's six, and for the first time it runs *forward*:** four are robust — the retired keyword
   argument, the deleted inline attribution, the renderer's one-token condition and the two-token
   annotation are each the retired thing entire — and **two are third-form by construction**,
   because `_compose`'s edit trips the trigger exactly as Table E's did: the body stops being a
   single composed return and becomes a clamp over a composition, so the text a retiring count would
   be stated over is destroyed by the faithful edit. That is the rule being applied where it was
   meant to be applied, at design time, rather than after a unit lands and a reviewer finds the
   blind pair. **Twenty of twenty-four robust, four third-form (E 2, H 2), none fragile.**

   **What that proves, and what it does not.** A residual that is **not** zero is a site the table
   missed — the check is sound in that direction and is the whole reason the tables carry commands.
   The converse does **not** hold: *residual zero* does not imply *nothing was missed*, because a
   site carrying none of a table's tokens is invisible to every one of its commands and to its
   residual alike. That is why §7 rule 5(a) requires such sites to be covered by a **named second
   command** — Table B's commands 4, 5 and 6 are that coverage, and without them 15 of its 18
   `arm_kind` sites sit outside every check while this done-condition still passes. So the
   done-condition is **the residuals plus the suite**: green tests over the profile contract
   (DC-6, §5 test 1) are what stands behind the sites no grep reaches, and neither half is
   sufficient alone. *(v1.10 called the tables "complete by construction" and this condition
   inherited the claim; the claim was false in both of §7 rule 5's two ways, and a mechanism adopted
   to stop a third incomplete edit list is the last place an over-claim belongs.)*

13. **The continuous carrier is proven, and every proof is offline** *(new in v1.12, plan-gate
   P6-1; `-ml` v1.15 §3.2d publishes four obligations and this is where they land)*. **(a)**
   `scored_outcome` called on a metric that lives in `measures` **raises** `MetricKindError`,
   asserted from both maps — the test that fails if the booleanisation is ever reintroduced.
   **(b)** A measure of `0.0` on a `scoreable: True` item survives a `to_dict`/`from_dict` round
   trip **as `0.0` and not as absent**, while the same item with the key missing raises
   `IncompleteItemRecord`: the absent-never-zero boundary asserted on both sides, which is the one
   that stops a query nobody judged from being read as a query that retrieved nothing relevant.
   **(c)** A non-finite measure is refused at construction (`NonFiniteMeasure`), and a metric name
   present in **both** maps is refused there too (`MetricKindError`). **(d)** Over two arms where one
   unit is scoreable in exactly one of them, `diffs` has length `n_units − 1` **and** the dropped
   unit appears in the `-ml` §4.3 asymmetry tally — **asserted together, because either assertion
   alone passes on a silent drop.** Plus this plan's own two, which are the carrier's other half:
   `named_metrics()` returns `separationRaw` and `separationZ`, so `sep_z` reaches the Arms table at
   all (it reaches none today), and that table renders a `DistributionSummary` row **without reading
   `mean`** — the shipped `else` branch reads it and would raise `AttributeError` on the new type.
   **(e)** *(v1.13, `-ml` v1.16 §3.3)* A `verdictMetrics` family mixing a binary member with a
   continuous one is refused **whole**: over a two-arm fixture declaring such a family, **no member
   is verdicted**, each is named with its resolved kind, every member's number still prints under
   `exploratory — no significance claim`, **both arms still render**, and the
   `INVALID RESULTS EXCLUDED` block is empty — the last two being what distinguish this from DC-10's
   exclusion and what fail if an implementer reaches for DC-10's mechanism (§3.3 (iv), which also
   names the **four** sites the verdict path renders this at and is where those decisions live —
   v1.14, extended at v1.15). **Two more assertions, one per site §3.3 (iv)(3) and (4) add**
   *(v1.15, plan-gate P8-2)*, and both are on an **absence**, which is what a test that only checks
   the lines that should render cannot see: the rendered report contains **no
   `### Family-wise error control` heading and no `Holm–Bonferroni` substring**, and it contains the
   one-line replacement naming the refusal; and **no decision cell anywhere in it reads
   *no verdict — no paired data***, which is the string that becomes unreachable rather than
   relabelled. **A second fixture covers the other condition that reaches the same shipped line**: an
   **all-continuous `k = 2`** family renders the *correction taken in the interval* replacement and
   likewise no ladder — the case a fix scoped to the mixed family alone would have left printing
   `applied`.
   **(f)** *(v1.14, plan-gate P7-1(d))* The stored form §4 S1e Table F decides is asserted, all four
   offline: a `DistributionSummary` survives `to_dict`/`from_dict` **as a `DistributionSummary`**
   — the type asserted and not merely its fields, which is the assertion that fails on `_decode`'s
   silent pass-through of an unknown tag; a metric dict carrying a `"type"` this build does not know
   **raises** rather than returning a `dict`, and a record containing one is surfaced as
   `unparseable` by `load_history` rather than half-decoded; `support` round-trips on **both**
   continuous types with `None` surviving as `None` and a bounded pair as a `tuple`, while a stored
   metric dict with **no `support` key** raises rather than defaulting; and `_index_row`'s metrics
   cell renders a `DistributionSummary` — its `p50`-labelled median, no `mean` read — so
   `index.csv` is writable for an embedder run.
   §5 test 11d is the same requirement stated from the test side.

This stage encodes the **amended** FR-15/AC-4 decision rule (§3.9 point 1) — the paired-difference
interval, not marginal overlap.

#### S1e — The edit set over shipped code, enumerated by grep

*(New in v1.10. Several revisions since S1 shipped have specified edits to the shipped tree, and two
of those edit lists were incomplete — v1.8 named two of four sites for the `residencySource` swap, and
v1.9 fixed that table while omitting the two shipped `_percentile` copies from the estimator ruling
it adopted in the same revision (plan-gate P4-2). §7 rule 5 is the response: an edit list over
shipped code carries **the commands that enumerate its own sites**, their **counts at a
named commit**, and a **done-condition that re-runs them and asserts a stated residual**. All eight
tables
below are stated against the shipped tree `5878014`, working directory `model-bench/`, and the
counts are **matching lines** from `grep -rFc`, not occurrences — for `armKind` those are 50 and 57,
and a table that says only "59" has already lost the distinction that lets a reader reproduce it.
Every count and line number below was re-run against `5878014` at v1.11; two line numbers v1.10
carried were wrong by two and are corrected in place — `ARM_KINDS` is `fingerprint.py:137`, the
membership test is `:162`. **The two unlanded tables are the exception and each states its own
baseline, which is now `e162ba9` for both** *(Table H's since v1.19, Table F's since v1.23)*.
Table H was written after Tables A–E and G had landed, so `5878014` was never reachable for it in
any useful sense: it was stated against `7f865e2`, re-pointed to `93b0e42` at v1.21 and to
`e162ba9` at v1.23. **Table F re-points at v1.23** — its residual block had moved to `93b0e42` at
v1.22 while its enumerating commands still named `5878014`, and v1.23 re-runs all eight of those
commands and lands the whole table on one commit. Each move is made by the revision that re-ran the
commands, which is the only thing that licenses it (§7 rule 5). **The six landed tables keep their
own commits** and are not re-pointed.)*

All eight are **S1-local**: they change `modelbench/` and `tests/` only, and
all eight are free **only now**, because `results/runs/` does not exist so no stored record is
invalidated and no `migrate` step is owed (§3.4.2, §7). **Table F is v1.12's, and it is here rather
than at S3 on exactly that argument** *(plan-gate P6-1, and the gate's open question 1, answered)*:
it changes the stored record's shape, so deferring it means changing a record schema **after**
records exist — the one thing this section's deadline exists to prevent. Its renderer half is §4
S1's `compare_report` block and is S1-local too; what genuinely belongs to S3 is only the first run
that exercises them. **Table G is v1.13's**, on the same argument and more cheaply: it changes two
`stats.py` signatures and the test call sites that break under them, touches no stored record, and
retires the two fixed levels that make `-ml` v1.16 §3.4 Rule 8's family correction unreachable.

**Seven of the eight touch nothing S2 constructs. Table F is the exception, and it is why the S1 fix
round comes *before* S2 rather than beside it** *(v1.14, plan-gate P7-1(e); v1.12 wrote "nothing S2
constructs" of all of them, which was true of Tables A–E and false of the table added in the same
revision)*. S2's runner constructs the `ItemResult` Table F changes, and every S2 scorer must state
a `ContinuousMetric.support` that is **required with no default** — so an S2 written against the
shipped shapes is rework of exactly the kind this section's deadline exists to prevent. The order
is therefore fixed: Tables A–G with DC-11 and DC-13, **then** S2.

**And within the round one pair is ordered** *(v1.15, plan-gate P8-1)*. Tables C and G both edit
`stats.py:159`, and unlike Tables D and E on `stats.py:263` only one order is faithful: **C, then
G**. The collision is named on both tables' rows and the reason is on Table C's, which is the one
that moves first.

**No table now carries a stage gate, and Table E's is withdrawn** *(v1.12, `-ml` v1.15 §3.2d)*.
v1.11 gated §4 S3 done-condition 2 on Table E; the note reverses it — `separationZ` is **reported
rather than verdicted**, so the `_widen` clamp is due with the `sep_z` **comparison**, which is
exploratory and is no stage's done-condition (§3.8.1). What §4 S3 is gated on instead is **Table F**,
and not as an added condition: without the carrier, S3 done-condition 1 cannot store a result at
all, because the embedder's only verdict metric has nowhere on the record to live.

**What these tables enumerate, and what they deliberately do not** *(v1.12, plan-gate P6-4; the count
follows Table G at v1.13, Table H at v1.19)*. The eight
are the **retiring and re-keying** edits — the ones where a site can be missed in silence, because
the old spelling still compiles and still means something. An edit that **adds** a required field is
enforced by the type system instead (§7 rule 5's closing paragraph) and gets no table, because a
table says less than a failing construction site already does. That remainder is therefore not an
omission, and it is named here so nobody reads *the* edit set and goes looking for an eighth: it is
**§4 S1's signature block plus DC-11** — `ItemResult.timing`, `latencyMs` becoming a property over
it, and `LatencyBlock`'s four v1.10 figures. `latencyMs` is a required positional today
(`grep -rFc latencyMs modelbench tests --include='*.py'` → **26** lines across five files:
`modelbench/results.py` 8, `tests/test_results.py` 8, `tests/test_report.py` 8, `tests/conftest.py`
1, `tests/test_cli.py` 1), and `ItemTiming`, `LatencyBlock`, `withheldFor` and `wallClockMs` appear
**0** times in the tree at `5878014` — so every construction site breaks loudly and there is nothing
for a residual to prove. **Table F is the one *adding* edit that is still tabled**, and it states on
itself why: its new field is defaulted, so no construction site breaks, and the behaviour it changes
is reached from sites that keep compiling either way.

**A table that has landed says so, and its rows stop being an instruction** *(new in v1.16;
impl-gate P5-3 and P5-6 were its occasion rather than its finding)*. This section is executed in
sequenced units, so from the first unit onward some of these tables describe **completed work**
while the rest describe work still owed — and a revision that re-pins a landed table's line numbers,
as v1.16 does for Table C, must not read as a fresh instruction to apply it again. The convention is
one line under a table's opening paragraph:

> **Landed:** `<commit>` (`<date>`) — this table's site rows and counts are a **record**, not an
> instruction. Its residuals remain DC-12's and are re-run at the **end** of the round, against the
> tree as it then stands.

Three things it deliberately does not do. It does not close DC-12 for that table: rule 5(b)'s last
clause makes the residual property one of the **round**, so a later table's edit can move a landed
table's number and the re-run at the end is what catches it. It flips no `Status:` token — the
document's status is the whole plan's. And it is **not** a claim that the delivering unit's review
closed: the commit is named, the gate is not. Where a landed table's row is later corrected, the
correction says so and the `Landed:` commit is unchanged.

**Table A — `lmsCliCommit` → `residencySource`** (§3.4.2; opened by G3-9).
Enumerate: `grep -rFn lmsCliCommit modelbench tests --include='*.py'` → **3 lines**:
`modelbench/fingerprint.py` 1, `tests/conftest.py` 1, `tests/test_fingerprint.py` 1. The fourth site
carries no token and is why the table is not the grep alone: `tests/conftest.py`'s
`residentModelsAtEnd` element, found by `grep -rFn sizeBytes …` → **1 line**.

> **Landed:** `8fc2341` (2026-09-07) — this table's site rows and counts are a **record**, not an
> instruction. Its residuals remain DC-12's and are re-run at the **end** of the round. *(The second
> residual's **scope** changed at v1.16 and its target did not; see below. The row list, the
> enumerating commands and their per-file counts are unchanged.)*

| Site | Edit | Fails loudly if missed? |
|---|---|---|
| `modelbench/fingerprint.py` — the schema-1 model mappings | `lmsCliCommit` out, `residencySource` in, on **both** `model:chat` and `model:embeddings` | yes |
| `tests/test_fingerprint.py` — the independently-written literal pinning the set | same swap | yes |
| `tests/conftest.py` — the shared `MODEL_FIELDS` fixture | same swap | yes — a missing `residencySource` fails `validate()` |
| `tests/conftest.py` — the `residentModelsAtEnd` **element** shape | `{modelKey, sizeBytes}` → `{id, state}` (§3.4.4a) | **no** — see §3.4.2; the fix is DC-1's assertion, not this row |

**Residual after the edit** — the table's own two enumerating commands, re-run
*(written out at v1.12 so DC-12 re-runs a command rather than re-deriving one — plan-gate P6-5's
lesson applied to every table, not only to the one it was raised against; the **second command's
scope** is v1.16's, impl-gate P5-3)*:

- `grep -rFc lmsCliCommit modelbench tests --include='*.py'` → **3 → 0** *(observed **0** at
  `612888c`, re-run)*
- `grep -rFn sizeBytes modelbench tests/conftest.py --include='*.py'` → **1 → 0** *(the single
  pre-edit line is `tests/conftest.py:40` at `5878014`, the sole match anywhere in either half of
  the scope; observed **no match** at `612888c`, re-run. `--include` does not suppress a file named
  explicitly on the command line — checked, because this command depends on it)*

**Why the second residual is scoped to `modelbench` and `tests/conftest.py`, and the one thing the
implementer may therefore not write** *(v1.16, impl-gate P5-3)*. A residual counts the lines that
**name** a token, and grep cannot tell a live use from a mention that disowns it. DC-1 requires the
retired `lms ps --json` element to be refused, and the only honest way to assert that is a test which
**constructs** it — which names both retired keys in a `.py` file and puts v1.15's unscoped command
at 1 on a faithful implementation. That is §7 rule 5(b)'s trap, and here it was load-bearing rather
than merely annoying: with the assertion unwritable, a counter-implementation whose extra-key rule
carries a one-name tolerance passes the whole suite (impl-gate P5-3's mutation M1, **472 passed**),
so DC-1's second half was pinned by nothing at all. The scope leaves the residual's *before* value
exactly where v1.10 put it — the sole pre-edit occurrence is `tests/conftest.py:40`'s fixture
element, which is precisely the site this table's fourth row is about — so it still proves what it
was added to prove, and it is **not** the narrowing to `modelbench/` alone that impl-gate P5-3 rules
out, which would read **0** *before* the edit and prove nothing. What it costs is one prescription,
and the prescription is binding: **the retired element's literal lives inline in
`tests/test_fingerprint.py` and nowhere else** — not in `tests/conftest.py`, and not in a
`modelbench/` comment or docstring, in neither case even as prose recording the retirement. A scope
without that half is a guess about where a comment will land.

**Table B — `armKind` → `armProfile`, and `ARM_KINDS` decoupled** (§3.4.1; plan-gate P4-2, its
command set and residual corrected at v1.11 by plan-gate P5-2).

> **Landed:** `8fc2341` (2026-09-07) — this table's site rows and counts are a **record**, not an
> instruction. Its residuals remain DC-12's and are re-run at the **end** of the round. *(All four
> re-run at `612888c` and observed at **0**; nothing in this table changed at v1.16.)*

**Enumerate — six commands, because three of them reach sites the other three structurally cannot.**
All re-run against `5878014`:

| # | Command (under `model-bench/`) | Lines | Per file |
|---|---|---|---|
| 1 | `grep -rFn armKind modelbench tests --include='*.py'` | **50** / 57 occurrences | `tests/test_fingerprint.py` 18, `modelbench/fingerprint.py` 16, `modelbench/results.py` 8, `tests/test_results.py` 4, `modelbench/report.py` 2, `tests/conftest.py` 2 |
| 2 | `grep -rFn FORBIDDEN_BY_ARM_KIND modelbench tests --include='*.py'` | **10** | `tests/test_fingerprint.py` 7, `modelbench/fingerprint.py` 3 |
| 3 | `grep -rFn ARM_KINDS modelbench tests --include='*.py'` | **2** | `modelbench/fingerprint.py` 2 (`:137`, `:162`) |
| 4 | `grep -rFn arm_kind modelbench tests --include='*.py'` | **18** | `tests/test_fingerprint.py` 9, `tests/conftest.py` 4, `tests/test_results.py` 3, `tests/test_report.py` 2 |
| 5 | `grep -rFn REQUIRED_BY_SCHEMA modelbench tests --include='*.py'` | **22** | `tests/test_fingerprint.py` 9, `modelbench/fingerprint.py` 4, `tests/test_results.py` 4, `modelbench/results.py` 3, `tests/test_report.py` 2 |
| 6 | `grep -rFn EXPECTED_MODEL_SCHEMA_1 modelbench tests --include='*.py'` | **3** | `tests/test_fingerprint.py` 3 (`:149` the literal, `:202`, `:228`) |

**Commands 4–6 are v1.11's and they are the finding, not a flourish.** Command 1's token is
camelCase and the **test** vocabulary is snake_case: of command 4's 18 `arm_kind` lines only **3**
also carry `armKind` (`test_fingerprint.py:22`, `conftest.py:169`, `:170`), so **15 edit sites — the
parametrized required-field contract, the two-kind set assertions, the shared fixture's branch and
five construction sites — appear in none of v1.10's three commands** and the residual v1.10 asserted
was satisfied without them (plan-gate P5-2, verified). Command 5 reaches the mapping this whole table
is *about*, including the re-key line (`fingerprint.py:124`, the body under the declaration at
`:123`) and the two `parametrize` decorators over `REQUIRED_BY_SCHEMA[1]["model"]`
(`test_fingerprint.py:34`, `:41`). Command 6 reaches the hand-transcribed literal, which is
independent of the module under test by design (impl-gate M-4) and therefore outside every
name-based grep — §7 rule 5(a)'s *contract restated by value* shape, exactly.

| Site | Edit | Found by |
|---|---|---|
| `fingerprint.py:123-125` — `REQUIRED_BY_SCHEMA[1]` | re-key `"model"` → `"model:chat"` and add `"model:embeddings"` (26 fields, §3.4.2), `"deterministic"` unchanged. The entries are the body under the declaration line | 5 |
| `fingerprint.py:128-135` — `FORBIDDEN_BY_ARM_KIND` → `FORBIDDEN_BY_ARM_PROFILE` | the union-minus-mine **set operation** over the three profiles, never a list (§3.4.1). Again the entries (`:131`, `:133`) are the body under the declaration | 2 |
| `fingerprint.py:137` — `ARM_KINDS = frozenset(FORBIDDEN_BY_ARM_KIND)` | **decouple it.** `ARM_KINDS` stays `{"model", "deterministic"}` — a literal or a derivation over `REQUIRED_BY_SCHEMA`'s profile keys split on `:` — because `armKind` keeps its two values (§3.4.1). Re-keying the mapping and leaving this derivation makes `ARM_KINDS` `{model:chat, model:embeddings, deterministic}`, so `armKind == "model"` fails membership at `:162` and **every model record returns `FieldProblem("armKind", "unknown")` and refuses on write**. **Both branches survive v1.12** — narrowing residual 4 is what makes the literal branch legal again (plan-gate P6-2, below) | 1, 3 |
| `fingerprint.py:184` — `REQUIRED_BY_SCHEMA[schema][self.armKind]` | the lookup key becomes the derived `armProfile`, which is the single line that makes the re-key real | 1, 5 |
| `fingerprint.py` — `Fingerprint` | `callSurface` joins `armKind` as a constructor argument, `armProfile` is derived from the two, and `validate()` checks both discriminators **before** consulting any mapping | 1 |
| `fingerprint.py` — `from_dict` / `to_dict` | `from_dict` strips `"armKind"` today (`:200`); it must strip `"callSurface"` too and pass it through, or the discriminator lands in `fields` and appears as a forbidden field | 1 |
| `results.py`, `report.py` | `armKind` filters are **unchanged by design** (§3.4.1) — the edit is to confirm each one still means the two-valued discriminator and not the profile | 1, 5 |
| `test_fingerprint.py:34`, `:41` | the two `@pytest.mark.parametrize("field", sorted(REQUIRED_BY_SCHEMA[1]["model"]))` decorators, which drive one test per required field: re-key, and decide whether the `model:embeddings` set gets the same treatment (it should — 26 fields, one test each). **Carries no `armKind` and no `arm_kind`** | 5 |
| `test_fingerprint.py:144` | the comment naming `REQUIRED_BY_SCHEMA[1]["model"]`. The **key spelling** changes; the impl-gate M-4 finding it records does not | 5 |
| `test_fingerprint.py:149` — `EXPECTED_MODEL_SCHEMA_1` | the hand-transcribed 30-name literal gains a `model:embeddings` sibling of 26 names, transcribed the same way and **not** derived from it — deriving one from the other is the M-4 defect this literal exists to prevent | 6 |
| `test_fingerprint.py:200-207` | the `parametrize` over `[("model", …), ("deterministic", …)]` and the `REQUIRED_BY_SCHEMA[1][arm_kind]` lookup inside it: three cases, keyed by profile. **Carries no `armKind`** | 4, 5 |
| `test_fingerprint.py:211-213` | `assert set(REQUIRED_BY_SCHEMA[1]) == {"model", "deterministic"}` — the two-kind contract asserted **by value**; it becomes the three profiles. **The assertion line carries no `armKind` and no `arm_kind`** | 4, 5 |
| `test_fingerprint.py:227-234` | the forbidden-set assertions, including `set(FORBIDDEN_BY_ARM_KIND) == {"model", "deterministic"}`: the literals become three (§3.4.1's table) | 2, 6 |
| `test_fingerprint.py:21`, `:97`, `:102`, `:267` | `_problems(arm_kind, …)` and the three discriminator tests — their semantics change once there are two discriminators and one derived key | 4 |
| `conftest.py:148`, `:160` | the shared fixture's `arm_kind: str = "model"` default and its `model_fields() if arm_kind == "model"` branch: the branch must become **profile**-aware so an embeddings fixture is expressible at all. `arm_kind` itself may well stay — `armKind` keeps its two values — so this is the site where a residual over `"model"` would fail on a *correct* edit (§7 rule 5(b)) | 4 |
| `conftest.py:169`, `:170`; `test_results.py:62`, `:71`, `:239`; `test_report.py:471`, `:486` | `Fingerprint(…)` / `_run(…)` construction sites: fixtures gain `callSurface`, and every one of them breaks loudly the moment `callSurface` is a required no-default argument | 1, 4 |
| `test_fingerprint.py`, `test_results.py` | DC-6's three profile cases replace the two-kind ones | 1, 4 |

**Residual after the edit — and `armKind` itself has no zero residual, which the table states rather
than papers over** *(v1.11, §7 rule 5(b))*. `armKind` **survives by design** with both its values, so
its count moves upward from 50 and no number over it is assertable; v1.10's two residuals covered
only auxiliary tokens that go to zero from the mapping rename alone and said nothing about the sites
the table is about. Four residuals, each zero after a faithful edit and non-zero before it:

| Residual command | Today | After |
|---|---|---|
| `grep -rFc FORBIDDEN_BY_ARM_KIND modelbench tests --include='*.py'` | 10 | **0** |
| `grep -rFn 'frozenset(FORBIDDEN' modelbench --include='*.py'` | **1** | **0** — `fingerprint.py:137`, the only match. This is the row that proves `ARM_KINDS` is no longer derived from the forbidden mapping, and residual 1 does not: a rename that leaves the derivation coupled to the *renamed* mapping satisfies residual 1 and leaves this one at 1 *(v1.12, plan-gate P6-5 — v1.11 stated it as prose, in a table whose thesis is that a residual is a command with a count)* |
| `grep -rFn 'REQUIRED_BY_SCHEMA[1]["model"]' modelbench tests --include='*.py'` | **3** | **0** — the two decorators and the comment; the profile key makes the old spelling unwritable |
| `grep -rFn 'set(REQUIRED_BY_SCHEMA[1]) == {"model",' modelbench tests --include='*.py'` | **1** | **0** — `test_fingerprint.py:213`, the mapping's key set asserted by value, and the only mechanical check that reaches that line at all *(narrowed at v1.12 — plan-gate P6-2, below)* |

**Residual 4 is narrowed to the mapping's key set, because v1.11's wider form failed on an
implementation this very table authorises** *(v1.12, plan-gate P6-2)*. v1.11 asserted the bare
two-name set display, unscoped, across `modelbench` **and** `tests` → 0. The `fingerprint.py:137`
row above permits `ARM_KINDS` to be decoupled **as a literal**, and that literal *is* the same
two-name set display, in production code — so under half the implementations this table permits the
wider residual is 1 after a faithful edit, DC-12 fails on a correct implementation, and the
implementer's only exit is to override a done-condition. That is precisely the shape §7 rule 5(b)'s
closing sentence forbids, and precisely the shape this document rejected Pass 5's own prescribed
residual for, one table earlier. Scoping it to `tests/` does not fix it either: a test pinning the
decoupled `ARM_KINDS` by value — the natural way to assert the `:137` row's whole point — puts the
same display back, in `tests/`. The narrowed command names the site that genuinely **retires**: the
key set of `REQUIRED_BY_SCHEMA[1]`, which becomes three profiles, so the old spelling is unwritable
and no decoupled `ARM_KINDS` can match it under either branch. `test_fingerprint.py:234`'s display
needs no residual of its own — the retired name `FORBIDDEN_BY_ARM_KIND` is **on that same line**, so
residual 1 reaches it and takes it to zero with the rename.

Residuals 2, 3 and 4 are the **second-form** residuals §7 rule 5(b) requires: each retires a
spelling this edit genuinely makes unwritable, so zero is earned rather than assumed. What stands in place of a residual
over `armKind` is the **type system**: `callSurface` as a required, no-default constructor argument
breaks every `Fingerprint(` construction site loudly — `grep -rFn 'Fingerprint(' modelbench tests
--include='*.py'` → **12 lines** (`test_fingerprint.py` 6, `test_results.py` 3, `results.py` 2,
`conftest.py` 1), 10 of them real construction sites — which is §7 rule 5's *adds rather than
retires* half, asserted by the compiler and not by a count.

**Table C — one percentile implementation** (`-ml` §11.2; plan-gate P4-2's second instance, and the
code half of §6 R-13's closure).
Enumerate: `grep -rFn _percentile modelbench tests --include='*.py'` → **7 lines**:
`modelbench/results.py` 3 (`:573` the definition, `:599`/`:600` the `index.csv` call sites),
`modelbench/stats.py` 3 (`:296` the definition, `:159`/`:292` the bootstrap call sites),
`tests/test_results.py` 1 (a comment naming R-13).

**Re-pinned against the current tree at v1.16, because `8fc2341` moved one of these lines**
*(impl-gate P5-6)*. `modelbench/results.py` and `modelbench/stats.py` are **byte-identical** to
`5878014` — verified by hash against `git show 5878014:…`, not assumed — so all six production pins
below resolve unchanged. The seventh moved: the `tests/test_results.py` comment is at **`:543`**,
not the `:507` v1.15 carried, `8fc2341`'s two added test blocks having shifted it by 36 lines. The
enumerating command was re-run at `612888c` and returns the same **7** lines with the same per-file
split, so the line drift is the whole delta and this is a re-pin rather than a re-enumeration.
*(v1.17: it has landed since, at `cc28d48`, and all seven pins above are `8fc2341`-era — they name
where each site **was**, which is what an enumerating command's counts are for. `stats.py`'s public
`percentile` is now at `:464`.)*

> **Landed:** `cc28d48` (2026-09-08) — this table's site rows and counts are a **record**, not an
> instruction. Its residuals remain DC-12's and are re-run at the **end** of the round. *(The third
> residual's **target** changed at v1.17 — impl-gate F2, below — because Table D landed in the same
> unit and its mandated closed form matches this residual's pattern. The row list and the
> enumerating command are unchanged.)*

| Site | Edit |
|---|---|
| `modelbench/stats.py:296` | replaced by the note's public **`percentile(values, *, level: Fraction)`** *(signature corrected at v1.15 — `-ml` v1.18 §11.2.2 replaces v1.17's `permille: int`)* — Hyndman–Fan type 1, the integer rank taken over the level's numerator and denominator, sorting a copy of its input, and raising on empty input, on a **`float` level** and on a level outside `(0, 1]`. The four module constants **`LEVEL_P50`, `LEVEL_P95`, `LEVEL_CI95_LO`, `LEVEL_CI95_HI`** land beside it and are the whole literal level space. The estimator, the rank expression, the level's type and all three refusals are `-ml` §11.2/§11.2.1/§11.2.2's and are not restated here; **the four constant names are this plan's**, adopted from the note's recommendation because naming is the architect's, as `DecidedBy`'s tokens were. **Its acceptance tests are the note's and land with this table, at S1** *(v1.15)*: `-ml` §11.10 items **1** (the rank fixtures), **2a**/**2b** (the two bin-edge guards, whose scopes the note states) and **10** (empty input plus the level's three refusals); item **3** is this table's third residual. §11.10's other items are `latency_summary` and rendering behaviour and are S2's, per §5's stage table |
| `modelbench/results.py:573` | **deleted.** `results.py` imports `stats.percentile`; `-ml` §11.10(3) asserts *identity*, not equal behaviour, so the module may keep no private helper |
| `results.py:599-600` | `_index_row` currently computes p50/p95 inline from `run.items`. They come from the run's own `LatencyBlock` instead (§4 S2), which is where `-ml` §11's two floors are applied — a percentile computed here bypasses both |
| `stats.py:292` | `cluster_bootstrap`'s pair moves to the new signature and **stays literal**: `percentile(rates, level=LEVEL_CI95_LO)` and `percentile(rates, level=LEVEL_CI95_HI)`. Rule 6's one-level resample of one arm's own rate is on no verdict path, so no `k` reaches it — Table G's command 1 states the same non-site from the other side. It survives the v1.11 ruling: only the *paired binary* path stops resampling (§3.9 point 1, Table D) |
| `stats.py:159` — **and Table G edits this same line** *(v1.15, plan-gate P8-1)* | moves to the new signature as `percentile(means, level=LEVEL_CI95_LO)` and `percentile(means, level=LEVEL_CI95_HI)`. **That post-edit spelling is prescribed rather than left open, because Table G's two residuals are stated over it** and a residual over a spelling the implementer may vary is a trap. **Table G then replaces those two levels with its own `levels[0]` / `levels[1]` on this line**, so the line is edited twice in one round. **The order is fixed — this table first, then Table G — and unlike Tables D and E on `stats.py:263`, neither order is faithful here:** Table G first would hand the shipped `_percentile(ordered, pct: float)` a `Fraction` level (`Fraction(1, 40)` is `0.025`, not `2.5`), which is a unit error one substitution away from a plausible number. `:159` is §3.2d's continuous `paired_bootstrap` and survives the v1.11 ruling for the same reason `:292` does |
| `tests/test_results.py:543` *(`:507` before `8fc2341`; re-pinned at v1.16, impl-gate P5-6)* | the comment recording R-13 as open and `_percentile` as having **two copies**: both halves are false after this edit, and it is rewritten to name `stats.percentile` as the one implementation. It is a row because the enumerating command returns the line, and a returned line with no row is how a site is forgotten — §7 rule 5's own diagnosis, applied to the seventh line rather than only to the six that execute |

Both shipped copies are `int(round(p/100·(X−1)))`, the estimator `-ml` §11.2 explicitly **rejects**
— `round` is half-to-even, so the tie-break direction alternates with the sample size. Neither
appeared in v1.9's edit table.

**This table is not scoped away from `:159`, and the refusal is the note's** *(v1.15, plan-gate
P8-1(b)/(c); `-ml` v1.18 §11.2.2(3), §11.9 item 6(a))*. The gate's cheapest resolution was to exempt
`:159` — the one percentile call in the package whose level is **family-dependent** — from this
table and leave it to Table G, on the reading that `-ml` §11.10(3) obliged only
`modelbench.results`. The reading of the v1.17 wording was right and the exemption is **refused
anyway**, on a measured cost rather than on scope: what that line would keep is not a different
*unit* but the **estimator** §11.2 rejects, at the only call site in the package whose level is not
a literal, on the one path where §3.2d rules the interval **is** the test. §11.10(3) is package-wide
as of v1.18, and its command is this table's third residual. *(The question was raised there rather
than answered here, under §7 rule 3: the plan could state an exemption on its own authority, but it
could not change the estimator's signature, and it is the signature that moved.)*

**Residual after the edit — three commands and one stated absence, the first two commands being
the two halves of one token** *(the
`stats.py` half and the identity check are v1.15's — plan-gate P8-3 and `-ml` §11.9 item 6(e); the
first was written out at v1.12)*:

- `grep -rFc _percentile modelbench/results.py` → **3 → 0**. All three go: the definition at `:573`
  is deleted, and `:599`/`:600` take p50/p95 from the run's own `LatencyBlock`. `results.py` then
  imports the note's **public** `stats.percentile`, which carries no leading underscore — so zero is
  reachable by the prescribed edit and is not an accident of naming.
- `grep -rFc _percentile modelbench/stats.py` → **3 → 0** — the definition at `:296` and the two
  bootstrap call sites at `:159`/`:292`, by the same argument: the replacement is the **public**
  name. **This half had no residual until v1.15 and its absence was the finding**: the half
  §11.10(3) actually obliged is the `results.py` one, so an implementer could do exactly that half,
  leave both bootstraps on the rejected estimator, and read a residual of zero.
- `grep -rEn 'def [A-Za-z_]*(percentile|quantile)' modelbench --include='*.py'` → **2 → 2**, and
  **the two are named, because the count is not the check** *(target restated at v1.17 — impl-gate
  F2; it read `2 → 1` from v1.15)*. `-ml` §11.10(3)'s package-wide check. Before the edit the two
  are `results.py:573` and `stats.py:296`, **both the rejected estimator**; after it they are
  `stats.py:464`'s public **`percentile`** — the one sample-quantile estimator the rule requires to
  exist — and `stats.py:264`'s **`exact_paired_quantiles`**, which **Table D's own row mandates**
  (`-ml` §3.4 Rule 4's closed form). **DC-12 re-reads the two names; a third definition moves the
  number and fails the check, which is what still earns this command its place** (§7 rule 5(b)'s
  named-line-set clause).
  **`exact_paired_quantiles` is not a second sample-quantile estimator**, which is why the answer
  is to restate the target and not to rename the function: it is the *same* operator applied to the
  **exact multinomial resample distribution** rather than to a sample (`-ml` §11.2 reason 2), it
  takes a `table` and not a list of values, and it exists precisely so the paired binary interval
  resamples nothing. Renaming it to slip the pattern would also forfeit this residual's stated
  virtue — that it **survives a rename** of the private helper, which is the shape of review M27's
  defect. **Two tables in one unit is why the number moved**, so this is a round-level interaction
  of the kind §7 rule 5(b)'s last clause already governs, not an implementation defect.
  **What no half-application passes is therefore the first two commands plus the assertion**, not
  this count: the two file-scoped residuals **partition** the six production sites, and `-ml`
  §11.10(3)'s identity assertion — `results.percentile is stats.percentile`, plus a `results.py`-
  scoped re-run of this very pattern inside the test — is the behavioural half. *(v1.15 attached the
  no-half-application claim to this command; with the target at 2 the count alone proves nothing
  about which lines they are, and the claim moves to where it was always true.)* **Scoped to `modelbench` at v1.16, which is what *package-wide* always meant, because
  v1.15's widening to `tests` made it a trap** *(the same shape as impl-gate P5-3, third instance,
  found by DC-12's new per-residual sweep)*: the pattern matches a test **function name** — `def
  test_percentile_rejects_a_float_level` and `def test_the_percentile_rank_fixtures` both match it,
  verified by construction this session — and `-ml` §11.10 items 1, 2a, 2b and 10 land as **this
  table's own** acceptance tests at S1, so the natural spelling of the very tests this table's first
  row prescribes would have pushed the count above its target on a faithful edit. Nothing was lost
  by the narrowing: at `612888c` both lines the command returned were in `modelbench`
  (`results.py:573`, `stats.py:296`) and `tests/` defined no percentile or quantile function at all.
  **The landed tree settles it beyond argument** *(v1.17)*: `cc28d48` added **seven** such test
  names — `test_the_percentile_rank_fixtures`, `test_percentile_rejects_a_float_level` and five
  more across `tests/test_stats.py` and `tests/test_results.py`, re-run — so the unnarrowed command
  would now read **9** where the table asks for a stated target of 2. That is a residual failing on
  the faithful implementation of its own table, one revision after the narrowing that prevented it.
  The **implementer's half of the
  bargain**, as at Table A: the retired `_percentile` spelling may not survive anywhere under
  `modelbench/`, comments and docstrings included, since residuals 1 and 2 are scoped there and
  count a mention exactly as they count a call. Under `tests/` it may — which is what the next
  bullet is about.
- **The seventh line gets a row and deliberately no residual, and rule 5(b) requires that be said.**
  A command over `_percentile` scoped to `tests/` would fail on a faithful edit that names the
  retired helper while recording its retirement — a comment is prose, and a residual that can fail
  on a correct edit is a trap rather than a check. What stands in its place is the site row.

**Why three commands and not one:** `3 + 3 + 1 = 7`, which is this table's enumerating command's own
count, so every line it returns has a row; the two file-scoped residuals **partition** the six
production sites, so a half-application is non-zero on the half it skipped and the number says which
half; and the identity check is over a different construct entirely, so it survives any renaming of
the private helper that the first two would miss.

**Table D — the seed retires from the paired *binary* path** (`-ml` v1.11 §3.4 Rule 4).
Enumerate: `grep -rFn bootstrap_seed modelbench tests --include='*.py'` → **29 lines**
(`tests/test_stats.py` 22, `tests/test_report.py` 3, `modelbench/stats.py` 3,
`modelbench/report.py` 1); `grep -rFn conservative_envelope …` → **8 lines**;

> **Landed:** `cc28d48` (2026-09-08) — this table's site rows and counts are a **record**, not an
> instruction. Its residuals remain DC-12's and are re-run at the **end** of the round. *(The first
> residual's **command** changed at v1.17 — impl-gate F3, below — because its stated target of 0 was
> unreachable from the moment it was written. The row list and the enumerating commands are
> unchanged.)*
`grep -rFn cluster-bootstrap …` → **27 lines** (`modelbench/stats.py` 11, `tests/test_stats.py` 11,
`tests/test_report.py` 4, `modelbench/report.py` 1); `grep -rFn DecidedBy …` → **3 lines**.

| Site | Edit |
|---|---|
| `stats.py` — `verdict(..., bootstrap_seed=…)` | parameter removed, with the raise that demanded one on the clustered path |
| `stats.py` — `conservative_envelope(diffs, table, *, design_effect, B, seed)` | collapses to `conservative_envelope(table, *, design_effect)`; the `n != len(diffs)` guard goes with the argument that made its error representable |
| `stats.py` — the closed form itself | the exact multinomial quantile at **`LEVEL_CI95_LO` / `LEVEL_CI95_HI`** in integer arithmetic *(v1.15: the two levels are the same numbers and only their representation moved, with the estimator's — `-ml` v1.18 §11.2.2's closing paragraph, whose atom selector stays one exact integer comparison. Swept here because this row named the retired unit; the paired **binary** path takes its `k` correction in the Holm ladder, so no family level reaches it)*. Pinned in full by `-ml` §3.4 Rule 4 and **not restated here**; its five acceptance tests are the note's too |
| `stats.py:62` — `DecidedBy` | `"cluster-bootstrap"` → **`"conservative-envelope"`**. The note leaves this token to the architect and recommends renaming; **renamed**, because a machine token naming a resample that no longer runs is the same defect as the prose that named one arm of an envelope, and it is free while no stored record carries it. *(The same literal gains a **third** member, `"paired-bootstrap"`, for §4 S1e Table F's continuous path. Both edits land on this line and **this row owns it** — Table F cites this row rather than duplicating it, so one line has one owner — v1.12.)* |
| `report.py:701` | the `- decided by: … (seed N, from the pack's sampling.seed)` parenthetical goes; the note publishes what replaces it (which arm bound each bound) and this plan does not restate the string |
| `report.py:687` | `bootstrap_seed=pack.seed` goes. **`PackRef.seed` stays** — its consumer moves to `-ml` §3.2d's continuous bootstrap (§3.3) |
| `packs.py:88-95` | the docstring's justification moves from the paired table to §3.2d; the field, the no-default rule and `validate_pack`'s refusal are unchanged |
| `test_stats.py`, `test_report.py` | the four tests pinning the seed parenthetical and the `bootstrap_seed=None` precondition-ordering test go; every fixture asserting a rendered interval on this path is **re-derived, not adjusted** (`-ml` v1.11) |
| `stats.py` — **`paired_cluster_bootstrap` and `paired_bootstrap`: kept, and neither is touched *by this table*** *(v1.11, plan-gate P5-3; scoped at v1.12, plan-gate P6-3)*. Enumerate: `grep -rFn paired_cluster_bootstrap modelbench tests --include='*.py'` → **13 lines** (`modelbench/stats.py` 5, `tests/test_stats.py` 8); `grep -rFn paired_bootstrap …` → **8 lines** (`tests/test_stats.py` 5, `modelbench/stats.py` 3) | **No edit — and the row exists because "no edit" is a decision here rather than an omission.** Collapsing `conservative_envelope` removes `stats.py:263`, the chain's only *production* call site, leaving `conservative_envelope → paired_cluster_bootstrap → paired_bootstrap` unreachable from anything that runs today. It is kept because it lost its **current** caller and not its **designed** consumer: `-ml` §3.4 Rule 4 (v1.14) names `paired_cluster_bootstrap` as §3.2d's **entry point for every continuous verdict** — called with the pack's declared `design_effect`, the identity widening at 1.00 — and `paired_bootstrap` as its **engine, not a second entry point**. Its **three** direct tests — `test_stats.py:1259`, `:1317` and `:1328` by `def` line, calling it at `:1270`, `:1323` and `:1330` (§4 S1e Table E's command 2) — are the only executable statement of the `√DEFF` exactness argument and stay with it. *(v1.11 wrote "four", mixing two `def` lines with two call lines; corrected at v1.12 alongside plan-gate P6-3, which is the same reading error one table over.)* **Deleting them is not authorised by this table and would be the wrong tree.** **Table E *does* change `paired_cluster_bootstrap`'s signature** — `clamp`, required with no default — which is a different edit from this table's *retire the seed* one; **no edit** here means no edit **by Table D**, and v1.11's unqualified wording contradicted Table E one table over *(v1.12, plan-gate P6-3)*. The keep takes the **same discriminator** already attached to `sampling.seed` (§3.3): both go if the embedder pack's continuous verdict is ever cut |

**One consequence for whoever wires §3.2d's continuous path** *(S1's `report.py`, first exercised at
S3 — stated here because this table is where the chain's reachability was decided)*. MRR and `sep_z`
are wired to **`paired_cluster_bootstrap`**, never straight to `paired_bootstrap`: the second is a
correct interval that **silently ignores a declared design effect**, which is the one failure
direction the note refuses everywhere. *(v1.13: on the **verdict** path the caller is `-ml` §3.4
Rule 8's `continuous_verdict()` rather than `report.py` itself, which changes who calls the entry
point and not which entry point is called — §4 S1. The exploratory `sep_z` comparison still calls it
directly, §3.8.1.)* **Table F carries the record shape that wiring reads and the
renderer that calls it** *(v1.12)*, and Table E the clamp the `sep_z` comparison depends on, Table G
its quantile levels *(v1.13)*.

**Residual after the edit** *(commands written out at v1.12)*:

- `grep -rEn '\bbootstrap_seed' modelbench tests --include='*.py'` → **27 → 0** *(the command is
  v1.17's — impl-gate F3. It read `grep -rFn bootstrap_seed …` → **29 → 0** from v1.12, and **that
  target was unreachable on a correct edit from the moment it was written**, which is worse than the
  traps rule 5(b) already names: those fail on a faithful edit, this one could not be reached by
  any.)*
- `grep -rFn 'cluster-bootstrap' modelbench tests --include='*.py'` → **27 → 0**

**Why the first residual takes the whole-identifier form, and the arithmetic that finds the two
lines it drops** *(v1.17, impl-gate F3)*. The enumerating command is `-F`, and `bootstrap_seed`
occurs as a **substring of longer identifiers**: `29` matching lines at `5878014`, of which `27`
carry it as a whole identifier — the two that do not are both `def test_…` names, and they are named
here so nobody re-checks them. `tests/test_stats.py:882`'s
`test_cluster_bootstrap_seed_is_keyword_only_with_no_default` is the one that makes the old target
unreachable: it is `cluster_bootstrap` + `_seed`, a test for **`cluster_bootstrap`** — Rule 6's
one-level resample of one arm's own rate, which this table explicitly **keeps** (§4 S1e Table C's
`stats.py:292` row states the same disposition from the other side) — so a faithful edit leaves it
standing and the `-F` command reads 1 forever. `tests/test_report.py:1678`'s
`test_the_bootstrap_seed_comes_from_the_pack_and_is_printed_beside_the_instrument` is the second, and
it is a **genuine site rather than a collision** — a test *about* the retired parameter. It is not
an unrowed line: it is one of "the four tests pinning the seed parenthetical" in this table's
`test_stats.py`, `test_report.py` row above, and it breaks loudly at its own assertion the moment
`verdict()` stops taking the argument. What the whole-identifier form costs is therefore one site's
*mechanical* coverage, on a site the type system and the row both already reach — which is the trade
rule 5(a) describes when it says a command's output is a superset of the sites and the site list is
read, not transcribed. **`27 + 2 = 29`**, which is the enumerating command's own count, so every line it
returns is accounted for — the derivation is written out rather than the two lines listed beside the
number, because a hand-written list beside the invariant it instantiates is what DC-12 already
records going wrong twice. *(Re-derived at `5878014` with `git show`, per file: `report.py` 1,
`stats.py` 3, `test_report.py` 3 → 2, `test_stats.py` 22 → 21.)* **The implementer's half of the
bargain**, since this residual reaches `tests/` (§7 rule 5(b)): the retired parameter name may not
survive as a **whole identifier** anywhere under `modelbench/` or `tests/`; carrying it inside a
longer `def test_…` name is what the two dropped lines show is tolerated, and is the only thing that
is.

Both are first-form residuals over genuinely retired tokens (§7 rule 5(b)); `paired_cluster_bootstrap`
has none and owes none, because its disposition is *no edit*.

**One trip hazard on the second residual — raised at v1.12, closed at v1.13** *(§7 rule 1)*. Eleven
of those 27 lines are in `stats.py`, and two of them render the trailing *"Decided by the …"*
clause's two variants, which `-ml` §3.2f published in its pre-v1.11 wording while §3.4 Rule 4
required the **conservative envelope** instead — so an implementer copying §3.2f verbatim left the
retired token behind and this residual could not reach zero. **Note v1.16 swept it**, at §3.2f and
at every other site that named the same instrument — six in all, where v1.12 had reported two — so
the surface an implementer copies and the rule that governs it now agree. Copy either; the residual
reaches zero.

**Table E — `_widen`'s clamp becomes an argument, because `[-1, 1]` is false for `sep_z`**
*(new in v1.11, from `-ml` v1.14 §3.4 Rule 4's condition on Table D's keep; a live defect in shipped
code that neither gate had reached)*.

> **Landed:** `cc28d48` (2026-09-08) — this table's site rows and counts are a **record**, not an
> instruction. Its residuals remain DC-12's and are re-run at the **end** of the round. *(Both
> residuals were **replaced** at v1.17 — impl-gate F1, below — because the pair v1.14 stated reads
> clean on the half-application it exists to catch. The row list and the enumerating commands are
> unchanged.)* *(**v1.19 changes nothing here, and that is stated rather than left to be inferred**,
> because Table H moves the clamp this table parameterised: see *Table H meets this table*, below.
> Both residuals re-run at `7f865e2` and read **1** and **1**.)*

**Enumerate — two commands, because `_widen` is private and its own name reaches none of the sites
the edit breaks** *(both re-run against `5878014`; the second is v1.12's, plan-gate P6-3)*:

| # | Command (under `model-bench/`) | Lines | Per file |
|---|---|---|---|
| 1 | `grep -rFn _widen modelbench tests --include='*.py'` | **7** | `tests/test_stats.py` 4, `modelbench/stats.py` 3 |
| 2 | `grep -rFn 'paired_cluster_bootstrap(' modelbench tests --include='*.py'` | **5** | `modelbench/stats.py` 2, `tests/test_stats.py` 3 |

Command 1's three `stats.py` lines are the code sites: `:188` (`paired_cluster_bootstrap`'s call),
`:191` (the definition) and `:264` (`conservative_envelope`'s MOVER-D arm). **Its four
`test_stats.py` lines are not sites, and they are named here so that nobody re-checks them**:
`:743`, `:773`, `:915` and `:1110` are `def` lines matched on a **test name** containing a widening
word, and every one is a `verdict()`-level test that calls neither `_widen` — which is private — nor
`paired_cluster_bootstrap`. v1.11 read them as this table's test row, which made that row
unexecutable as written. The lesson is §7 rule 5(a)'s other edge, worth stating once: **a command's
output is a superset of the sites, never the site list itself**, so a table that transcribes the
output without reading each line has enumerated names rather than sites.

Command 2 reaches the sites the edit actually breaks. Its two `modelbench/stats.py` lines are `:162`
(the definition) and `:263` (`conservative_envelope`'s resample call, which **Table D retires**) —
so if Table D lands first that site is already gone, and if Table E lands first it passes
`clamp=(-1.0, 1.0)` until it does; **either order is faithful and neither leaves a site unedited.**
Its three `tests/test_stats.py` lines are `:1270`, `:1323` and `:1330`, the direct call sites, each
of which stops compiling the moment `clamp` is required with no default.

**The defect, in one sentence:** `_widen` clamps both bounds to `[-1.0, 1.0]` — correct for the
difference of proportions the envelope was written for, and **wrong for `sep_z`** (`-ml` §5.2), whose
per-query differences are differences of z-scores and are not bounded by 1. Wired as it stands, a
`sep_z` interval whose upper bound exceeds 1 is silently clamped to it and the point estimate can
land **outside its own interval**. The note is explicit that the *verdict* survives — a positive
difference's exclusion of zero is decided by the lower bound — so what is false is the **printed
interval**, which is this project's signature defect shape: a true decision beside a false number.
MRR and every rate difference pass through unchanged.

| Site | Edit |
|---|---|
| `stats.py:191` — `_widen(interval, point, scale)` | gains a fourth parameter, keyword-only and **required with no default**: `clamp: tuple[float, float] \| None`, `None` meaning *do not clamp*. The two literals at `:200`/`:201` become its components |
| `stats.py:162` — `paired_cluster_bootstrap(diffs, *, design_effect, B, seed)` | gains the same keyword-only `clamp`, **required with no default**, and forwards it. **Two callers state it, on two surfaces that must not be collapsed** *(v1.14, `-ml` v1.17)*: on the **verdict** path `-ml` §3.4 Rule 8's `continuous_verdict()` derives the clamp from the metric's `support` inside itself and passes it here, so no verdict caller states one; the caller that does state one is §3.8.1's **exploratory** `sep_z` comparison, which reaches this entry point directly, has no metric aggregate to ask, and states `clamp=None` |
| `stats.py:264` — `conservative_envelope`'s MOVER-D arm | passes `clamp=(-1.0, 1.0)` **explicitly**. Both arms of the envelope keep the identical clamp — Rule 4 requires them widened the same way, and a clamp is part of "the same way" |
| `test_stats.py:1270` — the envelope's two-arms test | passes `clamp=(-1.0, 1.0)`: its `diffs` are ±1/0 differences of proportions, so the clamp *is* the present behaviour and this table changes none of its assertions. The same test is **also Table D's** — `conservative_envelope`'s signature collapses under it — so the two edits meet on this line and neither owns it alone |
| `test_stats.py:1323` — `test_paired_cluster_bootstrap_scales_the_half_widths_by_sqrt_deff` | passes `clamp=(-1.0, 1.0)`; its bounds lie inside `[-1, 1]`, so the clamp is inert there and the √DEFF exactness assertions hold verbatim. **This test is the executable statement of the exactness argument Table D says must survive**, so it is adjusted and never deleted |
| `test_stats.py:1330` — the `design_effect < 1.0` refusal | passes `clamp=(-1.0, 1.0)`; the raise precedes any widening, so the argument is needed only to make the call constructible |
| `test_stats.py` — **one new test, written at the public surface** | `paired_cluster_bootstrap(…, clamp=None)` over differences whose widened upper bound exceeds 1: the bound comes back unclamped and the point estimate lies inside its own interval — the assertion that reproduces the defect rather than the fix. **At `paired_cluster_bootstrap` and not at `_widen`**, because `_widen` is private and the defect is one a caller can reach: §3.8.1 wires the `sep_z` comparison through exactly this entry point |

**Why required rather than defaulted to `(-1.0, 1.0)`.** A default that is right for proportions and
wrong for z-scores fails **silently**, in the direction that prints, which is exactly the failure the
note names — and this plan already refuses that shape for `designEffect`, `BinaryMetric.unit` and
`sampling.seed`. Requiring it makes wiring `sep_z` a decision the call site must state rather than
one it can inherit. The cost is two production call sites and four test lines. *Rejected alternative:*
declaring each metric's support in the pack manifest — a new required field for one metric, which
§3.4.2's "no fourth state" reasoning refuses; the reversal trigger is a third unbounded verdict
metric, at which point the support belongs with the metric rather than at the call site.

**Residual after the edit — two, one per bound, and both are *third-form*: stated over the text this
edit **creates**, because it destroys the text a residual would otherwise be stated over**
*(replaced at v1.17 — impl-gate F1; §7 rule 5(b)'s third form is written from this finding)*:

- `grep -rFn 'max(clamp[0], widened[0])' modelbench --include='*.py'` → **0 → 1** (`stats.py:261`)
- `grep -rFn 'min(clamp[1], widened[1])' modelbench --include='*.py'` → **0 → 1** (`stats.py:261`)

**Read the arrow as *absent before* → *present after*, and the reason is the finding.** v1.14's pair
was stated over the shipped text — `max(-1.0, point …)` and `min(1.0, point …)`, `1 → 0` each, both
re-derived at `5878014` — and it is **blind**, not merely weak. This is the one table in §4 S1e whose edit
**parameterises**: the clamp stops being a literal fused into the return expression and becomes an
argument, so the expression itself is rewritten, and a half-application written in the *new* shape
(`clamp[0]` wired, the upper bound left as the literal `min(1.0, widened[1])`) spells neither of the
old strings. Scored on the three states side by side, which is how it was found rather than argued:

| state of `_widen` | `max(clamp[0], widened[0])` | `min(clamp[1], widened[1])` | v1.14's `max(-1.0, point` | v1.14's `min(1.0, point` |
|---|---|---|---|---|
| shipped (`5878014`) | 0 | 0 | 1 | 1 |
| faithful (`cc28d48`) | **1** | **1** | 0 | 0 |
| half-applied — upper bound left literal | **1** | **0** ← caught | 0 | 0 ← **reads clean** |

So the old pair answers *the same number* on the faithful edit and on this table's own defect
shipped intact — a `sep_z` upper bound silently clamped to 1, in the printing direction — which is
precisely what rule 5(b) calls worse than no residual at all. The pair is therefore **replaced and
not supplemented**: keeping a check that reads clean on the defect it was written for is the thing
the rule forbids, and DC-12's count of eighteen is unchanged.

**The cost, stated because a third-form residual is the only kind that pins a spelling.** These two
commands match `_widen`'s return expression as it actually shipped, so an unrelated refactor of that
expression breaks them — the same bargain Table C's `:159` row strikes for Table G's pair, and the
same reversal: when the expression is rewritten for another reason the residuals are **re-derived**
over the new spelling, never re-**widened** (§7 rule 5(b)'s third-form trigger, which is where the
word and its reason live — v1.23, plan-gate P12-6). What they do **not** rest on is either bound's *value*: the
same two numbers clamp four **unrelated** intervals elsewhere in `modelbench/stats.py`, so the
obvious narrowing — dropping the operand and matching `max(-1.0,` and `min(1.0,` — is itself a trap:
at `cc28d48`, on the faithful edit, those two read **1** (`:137`) and **4** (`:100`, `:114`, `:137`,
`:702`), re-run. It was rejected on that measurement rather than on taste. The pair also complements the new test above, which reproduces the
defect on the **upper** bound only: the test catches a surviving upper clamp, and the second residual
is the mechanical check that catches the same thing without being run.

**Table H meets this table, and this table does not move** *(v1.19; §7 rule 5(b)'s round property,
which requires two tables meeting on a line to name each other on both rows)*. Table H retires the
`clamp=(-1.0, 1.0)` **argument** at the two `_widen` call sites this table created — the MOVER-D arm
and the exact arm, both now inside `envelope_arms` — and replaces it with `clamp=None`, because
`-ml` v1.19 §3.4 Rule 4a rules the support a property of the estimand and moves it onto the composed
interval. **What that does *not* touch is `_widen` itself**, whose parameter, body and both bounds
are exactly what this table specified; Table H changes what a caller *passes*, not what the function
*does*. So this table's two residuals — stated over `_widen`'s body at `stats.py:261` — are
untouched by Rule 4a and read **1** and **1** at `7f865e2`, verified, and its rows stand as written.
**This is the third-form pair's first live test and it passed**: an exact-text residual pinning one
function's body survived an edit to two call sites thirty lines above it in the same file, where a
line-pinned residual would have been broken outright by the Pass 8 fix round's insertions. That is
the evidence for §7 rule 5(b)'s *exact text over line pin* clause, and it is why the clause is stated
there rather than here.

**One row of this table is superseded in substance and kept in place** *(v1.19)*. The row reading
*"`conservative_envelope`'s MOVER-D arm passes `clamp=(-1.0, 1.0)` **explicitly**. Both arms of the
envelope keep the identical clamp"* was right when written and is now Table H's to change: after
Rule 4a both arms keep the identical clamp and that clamp is `None`. The row is **not** rewritten,
because this table has landed and its rows are a record of what `cc28d48` did (§4 S1e's `Landed:`
convention); Table H's row for the same two call sites is where the current instruction lives. The
three `clamp=(-1.0, 1.0)` call sites in `tests/test_stats.py` (`:1578`, `:1609`, `:1718`) are
**non-sites for Table H** and are named here so nobody re-checks them: all three call
`paired_cluster_bootstrap`, the **continuous** path, which Rule 4a leaves unchanged by its own point
3 — which is also why Table H's first residual is scoped to `modelbench` and would have been a trap
unscoped.

**Deadline — withdrawn, and the withdrawal is the note's** *(v1.12, `-ml` v1.15 §3.2d; plan-gate
P6-1's last clause)*. v1.11 wrote that this table was the only one in §4 S1e carrying a deadline and
gated **§4 S3 done-condition 2** on it. That reads as though the clamp were the one obstacle between
S3 and a rendered `sep_z` interval, and it was not: the interval also had **no producer and no
renderer** — no per-item carrier for a `sep_z` difference and no `named_metrics()` entry for
`separationZ` — which is the larger defect Table F now closes. The note settles the ordering:
`separationZ` is **reported, not verdicted** (`verdictMetrics = ["mrr"]`), so the clamp is due with
the **`sep_z` comparison**, which §3.8.1 records as exploratory and as no stage's done-condition.
So this table lands with the rest of S1e — it is `stats.py`, S1-local, blocked on nothing, and a
live defect in shipped code — **and it gates no stage.** The stage gate that replaces it is
**Table F on §4 S3 done-condition 1**, which is not a gate anyone has to remember: without the
carrier there is no storable result to sign off.

**Table F — the continuous carrier: `measures`, `scored_value`, and the aggregates `sep_z` reaches**
*(new in v1.12, from `-ml` v1.15 §3.2d; plan-gate P6-1(a) and (b) — a live defect in shipped code
that four gate passes did not reach)*.

**The defect, in one sentence:** nothing on the record can carry a per-item **continuous** value —
`ItemResult.counts` is `Mapping[str, int]` and `scored_outcome` returns `self.counts[metric] > 0` —
so the embedder pack's only verdict metric and its headline, `mrr`, either **booleanises** into *did
this query retrieve anything* and prints a McNemar `+X pp` verdict for a metric the note gives no
significance test, or is left unscoreable and renders **"No verdict: no paired data"**. Both are
silent, and both happen on a green run. `RetrievalAggregates.named_metrics()` omits `separationZ`
and `separationRaw`, so `sep_z` reaches no table either.

> **Landed:** `b5dab1f` (2026-09-08) — this table's site rows and counts are a **record**,
> not an instruction, **with one stated exception**. The `report.py:623-789` row's continuous
> branch — resolving each member's kind and routing a continuous verdict metric through `-ml` §3.4
> Rule 8's `continuous_verdict()` — is **not** this commit's: `continuous_verdict()` and its
> `ContinuousVerdict` sibling exist nowhere in the tree (checked; only citations in comments and
> docstrings), Table F's own two-file scope statement and its eight enumerating commands never
> touch `modelbench/stats.py`, and building the producer means authoring it from `-ml` §3.4 Rule 8
> — a ~3,600-line note this unit was not scoped against. **What lands here** is the carrier
> (`ItemResult.measures`/`scored_value`, `ContinuousMetric.support`, `DistributionSummary` and its
> stored form, `RetrievalAggregates.named_metrics()`, the Arms table and `_index_row` renderers)
> plus DC-10's third arithmetic, and the refusal that keeps the gap **loud rather than
> safe-by-absence**: `scored_outcome` raises `MetricKindError` on a `measures`-resident metric, so
> the still-binary family loop fails immediately, uncaught, the moment a pack declares a continuous
> `verdictMetrics` member — converting *silently wrong when that pack finally arrives* into
> *refuses right now* — rather than booleanising it, which is the property this table exists to
> guarantee ahead of that pack existing. `continuous_verdict()`/`ContinuousVerdict`, the branch that
> calls it, and §3.3(iv)'s mixed-kind-family refusal are a separate, properly-sized unit landing
> before S1 closes (this scope split was put to the stakeholder and confirmed before
> implementation). Its residuals remain DC-12's and are re-run at the **end** of the round.

**Enumerate — eight commands** *(six at v1.12; 7 and 8 are v1.14's and 6 is corrected there)*. All
re-run against **`e162ba9`**, working directory `model-bench/` — **re-pointed from `5878014` at
v1.23, in the revision that re-ran them** (§7 rule 5), which also ends this table's two-baselining:
its residual block had moved to `93b0e42` at v1.22 while these commands still named the shipped tree
three landed rounds behind it. **Every count and every per-file share below is unchanged from
`5878014`** — the eight are 17, 14, 5, 2, 12, 6, 3, 2 at both commits — and what moved is the line
numbers inside the glosses, which is what a re-baseline is for:

| # | Command (under `model-bench/`) | Lines | Per file |
|---|---|---|---|
| 1 | `grep -rFn scored_outcome modelbench tests --include='*.py'` | **17** | `modelbench/report.py` 6, `tests/test_results.py` 5, `modelbench/results.py` 4, `tests/test_report.py` 2 |
| 2 | `grep -rFn 'ItemResult(' modelbench tests --include='*.py'` | **14** | `tests/test_report.py` 7, `tests/test_results.py` 5, `tests/conftest.py` 1, `tests/test_cli.py` 1 |
| 3 | `grep -rFn ContinuousMetric modelbench tests --include='*.py'` | **5** | `modelbench/results.py` 5 (`:80`, `:86`, `:185`, `:370`, `:374`) |
| 4 | `grep -rn separation modelbench tests --include='*.py'` | **2** | `modelbench/results.py` 2 (`:187`, `:188`) |
| 5 | `grep -rFn named_metrics modelbench tests --include='*.py'` | **12** | `modelbench/results.py` 6, `modelbench/report.py` 3, `tests/test_report.py` 3 |
| 6 | `grep -rn 'isinstance(.*BinaryMetric' modelbench tests --include='*.py'` | **6** | `modelbench/report.py` 3 (`:211`, `:570`, `:581`), `modelbench/results.py` 3 (`:356`, `:374`, `:577`) |
| 7 | `grep -rn '\.mean' modelbench --include='*.py'` | **3** | `modelbench/results.py` 2 (`:360`, `:577`), `modelbench/report.py` 1 (`:600`) |
| 8 | `grep -rn '"continuous"' modelbench --include='*.py'` | **2** | `modelbench/results.py` 2 (`:360`, `:386`) |

**Commands 4, 6, 7 and 8 are §7 rule 5(a)'s token-free coverage; 7 and 8 are v1.14's and 6 is
v1.14's correction of its own earlier form** *(plan-gate P7-1; the general rule they instance is §7
rule 5(a)'s attribute-based companion)*. The shipped code that breaks under a third `MetricValue`
member spells **neither** type name: it reads an **attribute** a sibling has (`.mean`) or tests the
**string** that tags that sibling in storage (`"continuous"`). A name-based sweep cannot reach
either, which is why v1.13's six commands found one of the three bare-`else` `.mean` readers
(`report.py:600`) and missed two. **Neither 7 nor 8 alone suffices and their union is exactly the
four sites:** 7 returns the three `.mean` readers and not `_decode`'s tag gate; 8 returns the tag
gate and the encoder and neither `_index_row` nor the Arms table. Both are scoped to `modelbench`
because both return **0** lines under `tests/` — re-run today, and stated so the scope reads as a
measurement rather than an oversight. Command 6 pinned a **variable name** at v1.12
(`isinstance(metric, BinaryMetric)`), and `results.py` spells the same construct with `m`, so three
of its six sites were invisible to it; the regex form above is the repair and the lesson is §7 rule
5(a)'s — *a command pins the type, never the variable that holds it*. Command 2 is listed even though **no site it names breaks**, and that is
the point: `measures` is defaulted, so the compiler says nothing, and the behaviour that changes
(`scored_outcome`'s) is reached from sites that keep compiling either way. That is why this table
exists at all while DC-11's larger, louder edit does not.

**The stored form of a `DistributionSummary`, decided here** *(v1.14, plan-gate P7-1(a))*. v1.12
retyped two published figures to a new type and left that type's JSON to whoever implemented the
table — at the one moment §4 S1e says a record shape is free, which is the moment it stops being
free to leave open.

- **Tag and keys.** `{"type": "distribution", "name", "median", "p10", "n", "unit", "support"}` —
  the dataclass's six fields plus the tag, one key each, no nesting. `"distribution"` is a **third
  value of the same `"type"` discriminator** `"binary"` and `"continuous"` already carry, and not a
  `"continuous"` with extra keys: a reader that took one for the other would read a **median as a
  mean**, which is the one confusion `DistributionSummary` exists to make unrepresentable.
- **`support` is stored, on both continuous types**, as a two-element array `[lo, hi]` or `null`,
  and is decoded back to a `tuple` or to `None`. `_metric_from_dict` reads `d["support"]` with **no
  `.get` fallback** — the rule `results.py:365-366` already states for `BinaryMetric.unit`, in the
  same function and for the same reason. It is not recomputed by a reader: the scorer that knows
  the metric runs at S2 and the reader runs at compare time, and a support re-derived on read is a
  second home for a declaration. `null` is a **stated value** meaning *unbounded* (`-ml` §3.4 Rule
  8, v1.17), which is only true while the key is required — an absent key and a stated `null` must
  not decode to the same thing.
- **An unrecognised `"type"` tag raises, and never falls through.** Today `_decode` admits
  `value.get("type") in {"binary", "continuous"}` and returns anything else **as a raw `dict`**.
  The tag set therefore gets **one home** — a module-level mapping from tag to decoder that
  `_metric_from_dict` dispatches on and `_decode` gates on — so the two functions cannot disagree
  about how many metric types exist (§7 rule 4: where a derived surface can be a derivation, it
  must be). A dict tagged with a `"type"` this build does not know is a record written by a build
  that knew a metric type this one does not, which §3.4.3 already names the genuinely
  uninterpretable case; the raise surfaces as `unparseable` in `load_history`, exactly as
  `results.py:327`'s own comment says a `KeyError` in `from_dict` does.
- **`benchSchemaVersion` does not bump** *(the plan gate's open question 1, ruled — the one part of
  this decision that is a judgement rather than a transcription)*. The `measures` row below rules
  no bump on an **additive and absent-safe** ground, and that ground does not reach this: retyping
  `separationRaw`/`separationZ` from a number to an object changes an **existing** key's stored
  type. The ground that does reach it is §3.4.3's own criterion — the version increments when the
  record shape changes "in a way a *reader* must branch on" — and **no record carrying the numeric
  form exists or ever will**, because `results/runs/` does not exist, which is this section's whole
  deadline. A bump would add a `REQUIRED_BY_SCHEMA[2]` differing from `[1]` in nothing any reader
  could act on, and §3.4.3 is explicit that a bump is a deliberate act rather than a side effect of
  changing a field. **The reversal trigger is this section's deadline restated:** land the retype
  after the first embedder run has stored a record and it becomes a schema bump **and** a `migrate`
  step, because a stored numeric `separationZ` would then have to be read by a build expecting an
  object.

| Site | Edit | Found by |
|---|---|---|
| `results.py` — `ItemResult` | gains **`measures: Mapping[str, float]`**, defaulted `{}` and placed before `detail`; a `__post_init__` refusing a metric name present in **both** maps (`MetricKindError`) and any non-finite value (`NonFiniteMeasure`); a **`scored_value(metric) -> float \| None`** sibling with `scored_outcome`'s three states; and **`scored_outcome` raising `MetricKindError`** on a `measures`-resident metric instead of returning `counts[metric] > 0`. Shape, domain, absence rule and the one-map invariant are `-ml` §3.2d's and are not restated here | 1, 2 |
| `results.py` — `ItemResult.to_dict` / `from_dict` | `measures` round-trips, and a `0.0` round-trips **as `0.0`**; `from_dict` reads a missing key as `{}` — a *reader's* compatibility rule under §3.4.3, never a constructor default. **`benchSchemaVersion` does not bump**: the field is additive and absent-safe, and no stored record exists to migrate | 1 |
| `results.py:80`, `:86` — `ContinuousMetric`, `MetricValue` | `support: tuple[float, float] \| None` joins `(name, mean, n)`, **required with no default** — `BinaryMetric.unit`'s discipline for `BinaryMetric.unit`'s reason; `MetricValue` gains `DistributionSummary` | 3 |
| `results.py` — new `DistributionSummary` | `(name, median, p10, n, unit, support)`, frozen. `-ml` §5.2 publishes a **median and a p10** and neither is a mean, so `ContinuousMetric` cannot carry them and one bare float carries neither | 3, 4 |
| `results.py:185-188` — `RetrievalAggregates` | `separationRaw` and `separationZ` change from `float \| None` to `DistributionSummary \| None`; `mrr` keeps its type and its construction site states `support=(0.0, 1.0)` | 3, 4 |
| `results.py` — `RetrievalAggregates.named_metrics()` | returns `separationRaw` and `separationZ` as well. **This one line is what makes `sep_z` reach a table at all** — today it reaches none, whatever the scorer computes | 5 |
| `results.py:370`, `:374` — the metric (de)serialisers | `ContinuousMetric(name=…, mean=…, n=…)` gains `support`, reads it with no `.get` (above) and breaks **loudly** on construction, being required; `_metric_from_dict`'s two-branch `if` on the tag (`:364`) becomes a dispatch through the **tag mapping**, so a `"distribution"` record decodes as one; the `isinstance(value, (BinaryMetric, ContinuousMetric))` dispatch at `:374` gains `DistributionSummary`, and **that line carries no `separation` token** | 3, 6, 8 |
| `results.py:355-360` — `_metric_to_dict`, the encoder *(v1.14)* | the bare **`else`** splits: a `DistributionSummary` encodes to the six-key `"distribution"` dict above, and the surviving `"continuous"` branch gains `"support"`. `:360` reads `m.mean` on whatever arrives, so today it **raises `AttributeError`** the first time a `DistributionSummary` is stored — loud, but not until the first store, which is S3's | 7, 8 |
| `results.py:386` — `_decode`'s tag gate *(v1.14)* | the literal `{"binary", "continuous"}` becomes the tag mapping's keys and an unknown tag raises. **This is the site that fails silently**: a third tag falls through both `if`s today and is returned as a raw `dict`, so a field typed `DistributionSummary \| None` holds a `dict` and every later reader is wrong about a type nothing checked | 8 |
| `results.py:577` — `_index_row`'s metrics cell *(v1.14)* | the same bare `else`, `f"{m.name}={m.mean:.4f}"` — **`AttributeError`**, so `index.csv` cannot be written for an embedder run at all. A `DistributionSummary` renders **`{name}=p50 {median:.4f}`**; the `p50` label is not decoration, because the cell's continuous form is a bare number and a median printed like a mean is §3.5's defect one column over. **p10 is not in this cell** — the index is a per-run locator and the Arms table is where a distribution prints — and the cell is a **per-arm** figure, so §3.8.1's no-`sep_raw`-difference prohibition is untouched | 6, 7 |
| `report.py:211` — DC-10's selector | widens past `isinstance(metric, BinaryMetric)` to DC-10's third arithmetic and its kind cross-check | 6 |
| `report.py:581-601` — the Arms table | the `else` splits: a `ContinuousMetric` renders `n=…, mean, —` as it does today; a `DistributionSummary` renders its median and p10 and **no interval**. **Carries no type name**, reached only through `:581`'s head | 6 |
| `report.py:623-789` — the family loop and the two renderers downstream of it | pass 1 resolves each member's kind and branches, per §4 S1's `compare_report` block; `scored_outcome` is not called on a continuous member, and now raises if it is. **The refused-family label's two sites — a family filter and a headline fallback — are §3.3 (iv)'s, decided there and cited here, and their line numbers live there and not here** *(v1.14, plan-gate P7-3; the numbers dropped at v1.23, since restating a pin §3.3 owns is what would have put this row and that section on two different trees under this revision's re-baseline — §7 rule 4)* | 1, 6 |
| `tests/test_results.py`, `tests/test_report.py`, `tests/conftest.py`, `tests/test_cli.py` | the 14 `ItemResult(` sites **do not break** and mostly do not change; what changes is the `scored_outcome` assertions, which gain the raising case, and the fixtures DC-13 needs | 1, 2 |

**Why a second map and not a widened `counts`** — `-ml` v1.15 §3.2d rules it, and the reason is worth
one clause here because it is the tempting one-word fix: widening `counts` to `float` makes the
booleanisation **type-legal without making it wrong**, and it puts a count that §4.2's denominators
count into the same key space as a measurement they must not.

**Residual after the edit.** Three, all second-form over a literal or an annotation this edit
rewrites — stated as commands rather than as a table so they can be copied and run verbatim:

- `grep -nF 'separationRaw: float | None' modelbench/results.py` → **1 → 0**
- `grep -nF 'separationZ: float | None' modelbench/results.py` → **1 → 0**
- `grep -rFn '{"binary", "continuous"}' modelbench --include='*.py'` → **1 → 0** *(v1.14; package-wide, and the breadth is the check — v1.23)*

*(**Residuals 1 and 2 are scoped to the one file their site rows name; residual 3 is package-wide,
and it is the exception §7 rule 5(b)'s scoping default provides for** — v1.23, plan-gate P12-1,
correcting v1.22, which narrowed all three together and inverted the third one's purpose. All three
*before* values are re-run at **`e162ba9`** and read **1**. **Why the third is different, in the
table's own terms.** For residuals 1 and 2 a `report.py` or `stats.py` line spelling
`separationRaw: float | None` would be **noise** — the annotation can exist nowhere but
`results.py`, so the wide scope bought nothing and could only misfire, which is what the default is
for. For residual 3 such a line is **the defect this table is closing**: the `DistributionSummary`
decision above gives the tag set **one home** — the module-level mapping `_metric_from_dict`
dispatches on and `_decode` gates on — precisely so the two functions cannot disagree about how many
metric types exist (§7 rule 4), and the `report.py:211` row widens a **kind cross-check**, which is
where a second transcription of the tag set would plausibly land. Package-scoped, a `→ 0` is a
statement that the tag set has one home; `results.py`-scoped, it reads 0 with a second home standing
one file over. Measured at `e162ba9`: `{"binary", "continuous"}` occurs **exactly once** in
`modelbench` **and** in `tests` — `modelbench/results.py:386`, the site row's own line — so the wide
form costs nothing today and buys the one-home check. The alternative that also works and was
rejected for costing a row: keep all three narrow and add a fourth, package-wide residual for the
same result.)*

*(**What the narrowing on 1 and 2 actually trades, stated because v1.22 asserted it cost nothing and
that was established of the wrong half** — v1.23, plan-gate P12-4. Equal *before* values under
either scope say only that no **stated** number moved; a residual's claim is its **after** value,
and after the narrowing that assertion ranges over one file instead of the package. For these two it
is not a loss, on the reason above and not on the equality. What backs the **form** — an exact-text
residual scoped to a file — is two measurements rather than an argument, and neither is restated
here: the gate's synthetic two-file probe, stated on Table H, and the `e162ba9` insertion, stated at
§7 rule 5(b)'s exact-text-versus-line-pin paragraph.)*

The first two are `modelbench/results.py:187` and `:188` today and are unwritable after the retype,
so zero is earned. The third is `:386`'s tag gate — the silent site — and it is the one that
**cannot be passed by the half-application that ships the silence**: an implementer who adds a
`"distribution"` branch to both (de)serialisers and leaves `_decode` alone still has that
two-element literal, and the residual is 1. **Its reach and its limit, both stated because a
residual that is trusted past its reach is worse than none** *(the reach half is v1.23's, plan-gate
P12-1)*. What the package-wide scope reaches, and a `results.py`-scoped one did not, is a **second
copy of this two-tag literal in another file** — `report.py`'s kind cross-check being the plausible
one — so the `→ 0` is what makes the one-home decision above *checkable* rather than merely stated.
What it does not reach is a *third transcribed tag* written in place of the mapping
(`{"binary", "continuous", "distribution"}` contains no substring the command matches, so that reads
zero too), which is behaviourally identical to the derivation once an unknown tag raises — and the
raise is what **DC-13(f)** asserts. **Command 7 is not a residual and `.mean` never becomes zero:**
`ContinuousMetric` keeps its `mean` and the encoder keeps reading it, so the token survives by
design (§7 rule 5(b)) — a residual over it would be a trap that fails on a faithful edit. What
stands in place of a residual over `measures` — which **adds** and retires nothing — is
the **type system on one side and DC-13 on the other**: `ContinuousMetric.support` is required with
no default, so `results.py:370` breaks loudly; and `scored_outcome`'s refusal is a *behaviour*, which
no count can see, so DC-13(a) asserts it directly. Saying so is §7 rule 5(b)'s requirement, not an
apology for a missing number.

**One edit this table needs is deliberately not in its rows.** `DecidedBy` gains a third member,
`"paired-bootstrap"`, and that literal is the `DecidedBy` alias's declaration — a line **Table D
owns and enumerates**, with the command, the count and the three line numbers on that table and not
restated here *(the numbers dropped at v1.23: Table D is landed and its pins are a record at
`cc28d48`, while this table now states its counts at `e162ba9`, so a copy here would have put one
row on two trees)*. Cited rather than duplicated, so one line has one owner and the two tables
cannot drift on it (§7 rule 4).

**Table G — the bootstrap's quantile levels become parameters, because a `k > 1` continuous family
has nowhere else to take its correction** *(new in v1.13, from `-ml` v1.16 §3.4 Rule 8's two engine
preconditions; the other one is `_widen`'s clamp and is already Table E)*.

> **Landed:** `cc28d48` (2026-09-08) — this table's site rows and counts are a **record**, not an
> instruction. Its residuals remain DC-12's and are re-run at the **end** of the round. *(Unchanged
> at v1.17. This table is the **contrast** that made impl-gate F1's trigger nameable: its residuals
> were re-scored on the half-application and **do** catch it — see below.)*

**The defect, in one sentence:** `paired_bootstrap` hard-codes the 2.5th and 97.5th percentiles, so
the interval it returns is taken at one fixed, uncorrected pair whatever the family size — while `-ml` §3.3 rules
that an all-continuous family with `k > 1` takes its Bonferroni correction **in the interval**, at
percentiles that section fixes and this one does not restate (§3.3 (iv)). Rule 8's
`continuous_verdict()` computes those two levels from
`alpha_family` and `len(family)` and has to pass them down; with the levels fixed one function below
it, a `k = 3` family renders at 2.5/97.5 and the correction silently fails to happen. At `k = 1` the
note's rule returns exactly the two levels the code hard-codes, so **nothing renders wrongly today
and nothing will until a pack declares a second continuous verdict metric** — which is what makes
this invisible rather than urgent, and free rather than deferred. (No α is named in this table, for
§4 S1's reason: how many there are and which figure each governs is the note's.)

**Enumerate — three commands, because the literals and the call sites the signature change breaks
share no token** (all re-run against `5878014`, working directory `model-bench/`):

| # | Command (under `model-bench/`) | Lines | Per file |
|---|---|---|---|
| 1 | `grep -rFn 97.5 modelbench tests --include='*.py'` | **2** | `modelbench/stats.py` 2 (`:159`, `:292`) |
| 2 | `grep -rFn 'paired_bootstrap(' modelbench tests --include='*.py'` | **5** | `modelbench/stats.py` 2, `tests/test_stats.py` 3 |
| 3 | `grep -rFn 'paired_cluster_bootstrap(' modelbench tests --include='*.py'` | **5** | `modelbench/stats.py` 2, `tests/test_stats.py` 3 |

**Command 1 enumerates against the shipped tree and Table C retires the spelling it matches**
*(v1.15, plan-gate P8-1)*. `97.5` identifies this table's site at `5878014`; after Table C that line
reads `percentile(means, level=LEVEL_CI95_HI)` and carries no such literal, so **after Table C the
site is identified by the constant, not by the number** — which is also why this table's residuals
are stated over the post-Table-C spelling rather than over the pre-Table-C one (below). The
enumeration itself is unaffected: the two tables edit the same one line and Table C's `:159` row
prescribes the spelling this one then rewrites.

**Command 1's second line is a non-site, and it is named here so nobody re-checks it** — Table E's
lesson, that a command's output is a superset of the sites and never the site list. `stats.py:292` is
`cluster_bootstrap`'s pair: Rule 6's one-level resample of **one arm's own rate**, not a paired
difference and on no verdict path, so no `k` reaches it — a binary family takes its correction in the
Holm ladder, and a number outside `verdictMetrics` prints its interval as exploratory with no
correction at all (§3.3). **Command 3 is Table E's command 2 verbatim** and reaches the same five
lines: the two tables' edits meet on every one of them and neither owns them alone, exactly as
`test_stats.py:1270` already meets Table D. Command 2 is this table's alone, and it is the one that
reaches `paired_bootstrap`'s own callers, which neither `_widen`'s name nor
`paired_cluster_bootstrap`'s ever touches.

**There is no token-free shipped site, and §7 rule 5(a) is satisfied by exhaustion rather than by a
fourth command.** Both functions are module-level names that every caller must spell, and the three
commands above return every line in `modelbench/` and `tests/` that spells either — `test_stats.py`'s
two `from modelbench.stats import` lines carry the names without a call and need no edit. The only
caller that will pass a value other than the fixed `(LEVEL_CI95_LO, LEVEL_CI95_HI)` pair does not
exist yet: it is inside Rule 8's
`continuous_verdict()`, which this plan does not write, and the required-with-no-default parameter is
what makes it impossible to write without deciding the levels.

| Site | Edit |
|---|---|
| `stats.py:148` — `paired_bootstrap(diffs, *, B, seed)` | gains **`levels: tuple[Fraction, Fraction]`**, keyword-only and **required with no default**, replacing the two constant levels Table C's edit leaves at `:159`. *(Element type corrected at v1.15 from `tuple[float, float]` — `-ml` v1.18 §11.2.2: a `k`-member family's level is `alpha/(2k)`, which no fixed decimal unit expresses.)* **And `levels[0] >= levels[1]` raises** — the transposed pair is the one error that otherwise returns a plausible **inverted** interval that no other check sees (`-ml` §11.2.2), and it is a refusal this table could not have had while the pair was two literals |
| `stats.py:159` — **Table C edits this same line first** *(v1.15, plan-gate P8-1)* | the two levels Table C's row prescribes there — `level=LEVEL_CI95_LO` and `level=LEVEL_CI95_HI` — become `level=levels[0]` and `level=levels[1]`. **The collision is named on both rows and the order is fixed, C then G**, for the reason Table C's row gives: unlike Tables D and E on `stats.py:263`, neither order is faithful here. This is the line the whole table exists for — the one percentile call in the package whose level is not a literal |
| `stats.py:162` — `paired_cluster_bootstrap(diffs, *, design_effect, B, seed)` | gains the same keyword-only `levels`, same element type, required with no default, and forwards it at `:187`. This is the surface Rule 8's producer calls, so it is where the family-adjusted levels arrive. **Table E adds `clamp` to this same signature**: one line, two tables, and neither owns it alone |
| `stats.py:263` — `conservative_envelope`'s resample call | passes `levels=(LEVEL_CI95_LO, LEVEL_CI95_HI)` explicitly — the paired *binary* path takes its `k` correction in the Holm ladder and never in the interval. **Table D retires this call site**, so if Table D lands first the site is already gone and if this table lands first it passes the pair until it does; either order is faithful and neither leaves a site unedited |
| `test_stats.py:890`, `:891` — `test_paired_bootstrap_is_seeded_and_reproducible` | pass `levels=(LEVEL_CI95_LO, LEVEL_CI95_HI)`; the levels are inert to a determinism assertion, so no assertion in that test changes |
| `test_stats.py:1321` — the unwidened baseline inside the √DEFF test | passes the same pair. It is compared against `:1323`'s widened interval, and the comparison is only meaningful if both are taken at the same levels |
| `test_stats.py:1270`, `:1323`, `:1330` | pass `levels=(LEVEL_CI95_LO, LEVEL_CI95_HI)`; each stops compiling the moment the parameter is required, which is the point of requiring it. `:1323`'s √DEFF exactness assertions hold verbatim — Table D rules that test survives — and `:1330`'s raise precedes any resampling, so the argument is needed only to make the call constructible |
| `test_stats.py` — **one new test** | over one fixed `diffs` of distinct floats and one fixed seed, `paired_bootstrap` at the pair `-ml` §3.3's rule yields for a **`k = 2`** family — `(Fraction(1, 80), Fraction(79, 80))`, a pair the retired integer unit could not express and this one can *(v1.15, `-ml` §11.9 item 6(c))* — returns an interval **both of whose bounds have moved outward** from the pair the code takes today, the lower strictly below and the upper strictly above, **asserted as two assertions and never as one about the width** *(v1.14, plan-gate P7-2)*. A width assertion passes on the half-applied edit that wires the lower level and leaves the upper one on `LEVEL_CI95_HI`, whose interval is genuinely wider while its upper bound takes no family correction at all — the printing direction, silently. Both strict inequalities are checkable **before** the test is written — `diffs`, `B` and the seed all being fixed, and the note having measured the rank movement at `B = 10 000` — so the fixture is chosen to make them hold rather than asserted in hope. The two levels are computed from the note's rule in the test, never transcribed |
| `test_stats.py` — **one more new test** *(v1.15, `-ml` §11.10(10))* | `paired_bootstrap` with `levels[0] >= levels[1]` **raises**, one line, at the same boundary as the estimator's own three refusals |

**Why required rather than defaulted to `(LEVEL_CI95_LO, LEVEL_CI95_HI)`.** *(Nothing in note
v1.18 reopens this argument, which is unaffected by the element type — §11.9 item 6(f); only the
spelling of the pair moved, at v1.15.)* A default that is right at `k = 1` and
silently wrong at `k > 1` is the shape this plan already refuses for `designEffect`,
`BinaryMetric.unit`, `sampling.seed` and Table E's `clamp` — and here it is worse than usual, because
the wrong value is the *conventional* one and prints a plausible interval beside a family that was
never corrected. Requiring it makes the family size something a caller states.
*Rejected alternative:* passing `alpha_family` and `k` down and letting the engine derive the levels
— it would put §3.3's correction arithmetic in two places, and Rule 8 already fixes it in one.

**Rule 8's producer takes no percentile parameter, and this table does not give it one.** `levels`
is the **engine's** parameter, computed inside `continuous_verdict()` from `alpha_family` and
`len(family)` and never reachable from `report.py` (§4 S1) — which is exactly what makes a `k = 3`
family rendered at 2.5/97.5 unrepresentable rather than guarded. The one place `report.py` names the
levels itself is the **exploratory** `sep_z` comparison (§3.8.1), which is not a verdict, does not go
through the producer, and takes no family correction.

**Residual after the edit — two, one per retired level, and both are re-derived over `:159`'s
post-Table-C spelling** *(v1.15, plan-gate P8-1(d); `-ml` §11.9 item 6(d). The symmetry is v1.14's,
plan-gate P7-2: this edit retires **two** levels and v1.13 stated a residual over one, so a
half-applied edit passed it)*:

- `grep -rFn 'percentile(means, level=LEVEL_CI95_LO)' modelbench --include='*.py'` → **1 → 0**
- `grep -rFn 'percentile(means, level=LEVEL_CI95_HI)' modelbench --include='*.py'` → **1 → 0**

**Read the arrow as *after Table C, before this table* → *after this table*, and the reason is the
finding.** Both commands return **0** against the shipped tree today, because the spelling they
match does not exist until Table C lands; the pair v1.14 stated — over `_percentile(means, 2.5)` and
`_percentile(means, 97.5)`, **1** each at `5878014`, re-run and reproduced this session — was
meaningful only against the pre-Table-C form and is **driven to zero by Table C's edit alone**, so
DC-12 would have passed on an implementation that never applied this table. That is precisely the
half-application §7 rule 5(b) forbids, and it was invisible because it is a *cross*-table one: each
table's residuals were sound read alone. Re-scoping could not fix it — the retired token is gone
either way — so the pair is **re-derived** over the surviving spelling instead, which is why Table
C's `:159` row prescribes that spelling and why the order is fixed. **What guarantees the
intermediate state exists** is that fixed order plus the enumerating command's own count: `:159` is
one line, both tables name it, and DC-12 re-runs these two at the **end** of the round, when both
tables have landed.

Both are second-form, both functions surviving so neither has a zero of its own (§7 rule 5(b)) —
**and second-form is sufficient here, which is the finding this table contributes rather than
suffers** *(v1.17, impl-gate F1)*. This edit replaces a keyword argument's **value**
(`level=LEVEL_CI95_HI` → `level=levels[1]`) and leaves the call's shape alone, so the text these two
commands match survives into the half-applied state and still identifies the level left behind.
Re-scored on the landed line, the way Table E's pair was: on the faithful edit both read **0**; on
the half-application that wires `levels[0]` and leaves the upper level a constant, the second reads
**1** and the first **0** — the pair says which half was skipped. That is the contrast that makes
rule 5(b)'s third-form **trigger** nameable rather than a warning to sprinkle everywhere: Table E
parameterises and so rewrites its expression; this table substitutes one operand inside an
expression it preserves. **Stated symmetrically because the edit is symmetric:** either level left behind is a bound taking
no family correction, and the pair is what makes each half-application non-zero on the half it
skipped. Each is written over `means` rather than over the constant alone so that
`cluster_bootstrap`'s `percentile(rates, level=LEVEL_CI95_LO)` / `(rates, level=LEVEL_CI95_HI)` at
`:292` — which Table C rewrites and this table does not touch — cannot hold either above zero on a
faithful implementation. **What stands behind them, for the parameter this table *adds*, is the type
system** (§7 rule 5's adding half): `levels` required with no default breaks **eight** call sites
loudly — commands 2 and 3 return ten lines and two of them are the `def` lines — so the parameter
cannot be skipped, only left **unused** at `:159`, which is exactly what these two residuals
measure.

**Table H — the support clamp moves off the envelope's arms and onto the interval that is
printed** (`-ml` v1.19 §3.4 **Rule 4a**; impl-gate P8-1, with P8-5 closed as collateral).
*(New in v1.19. This is the eighth table and the first added since v1.13's Table G. It is the
implementation spec for a ruling this plan did **not** make: `P8-1` was routed to `data-scientist`
because "which arm bound this bound" is a statement about an estimand, and the note took it —
§7 rule 3 working as designed. The rule, its ten assertions and its two exhaustive sweeps are the
note's and are **not restated here**; what is this plan's is where the edit lands, what it retires,
and the six commands that prove it landed.)*

> **Landed:** `f17efa2` (2026-09-09) — this table's site rows and counts are a **record**,
> not an instruction. Its residuals remain DC-12's and are re-run at the **end** of the round,
> against the tree as it then stands.

**What the note ruled, in the four lines this table exists to enumerate** — cited, not re-derived
(`-ml` §3.4 Rule 4a points 1–4): a support is the parameter space of the **estimand**, so it is
applied **once, to the printed interval**, never to an input of a composition. Hence
**(i)** `envelope_arms` widens both arms with `clamp=None` and returns them unclamped;
**(ii)** the private composer clamps its own **result** and returns `(interval, bound_by)`;
**(iii)** `bound_by` becomes a **three**-token closed set with `"support bound"` added, computed
from the composed *unclamped* value against the support on a **strict** comparison; and
**(iv)** the `- decided by:` renderer must not attach a `p=` clause to a `support bound` token,
because no level produced it — and must render that token **with the support's boundary value**
instead, `support bound (-1)`, which is the string the note's assertion 10 pins verbatim
*(v1.20, plan-gate P10-2: "no `p=` clause" is true and was incomplete)*. The note's headline scopes the work and is worth carrying here in one
clause: **arms-versus-composed is immaterial to the statistics and material only to the audit
bullet** — the printed interval, its coverage, its width and every verdict are bit-identical either
way, verified over 173 472 combinations — so this is a **reporting** correction, and no number in
any §3.8 pack moves when it lands.

**Two names are this plan's, as `DecidedBy`'s tokens and the four `LEVEL_*` constants were**
(§7 rule 4 — the note fixes the arithmetic, this document fixes what things are called):
`BoundBy = Literal["MOVER-D", "exact paired bootstrap", "support bound"]`, a named alias beside
`DecidedBy` rather than a bare `tuple[str, str]`; and `SUPPORT_DIFF_PROPORTIONS: tuple[float, float]
= (-1.0, 1.0)`, the support as **one constant in one place**, which is the note's own
*derivable-from-what-the-function-already-holds* category and not a new parameter. Both names are
free at `e162ba9` — `grep -rFn SUPPORT_DIFF_PROPORTIONS modelbench tests --include='*.py'` → **0**,
`grep -rFn BoundBy …` → **0**, re-run.

**Enumerate — two commands, because the token the edit is *about* reaches none of the tests it
breaks** *(both re-run at **`e162ba9`** — re-pointed from `93b0e42` at v1.23, and from `7f865e2`
before that at v1.21, **each time by the revision that re-ran them**, which is the only thing that
licenses moving a stated baseline; §7 rule 5)*:

| # | Command (under `model-bench/`) | Lines | Per file |
|---|---|---|---|
| 1 | `grep -rFn bound_by modelbench tests --include='*.py'` | **15** | `modelbench/stats.py` 5, `tests/test_stats.py` 7, `modelbench/report.py` 2, `tests/test_report.py` 1 |
| 2 | `grep -rFn envelope_arms modelbench tests --include='*.py'` | **22** | `tests/test_stats.py` 17, `modelbench/stats.py` 5 |

Command 2 is rule 5(a)'s token-free coverage and is not a flourish: the arm clamp lives at two
`_widen` call sites that carry **no** `bound_by`, and command 2's `tests/test_stats.py` lines are
where the arm-level edits are — none of which command 1 reaches. **The four that are arm-value call
sites are named by test function, not by line** *(v1.22, plan-gate P11-3)*:
`test_the_envelope_takes_each_bound_from_whichever_arm_is_more_conservative`,
`test_the_first_published_envelope_anchor_is_unmoved_by_the_closed_form`,
`test_neither_printed_bound_is_ever_tighter_than_either_arm` and
`test_both_arms_are_widened_about_the_same_point_by_the_same_factor`. Every other line the command
returns in that file is an import, a `parametrize` id or source entry, docstring prose, or a call
made to assert a **refusal** — a precondition raise or N4's non-finite guard — none of which Rule 4a
touches. *(Plan-gate P10-1 hid in this sentence — the third of the four is its
row. And the line numbers had to go for the reason that row is name-pinned: **this table inserts
into this file**, landing Rule 4a's ten assertions beside the envelope tests they extend, which is
between the second and fourth of these four — so a line-pinned gloss is stale-by-construction the
moment its own edit lands.)*

*(**This gloss names the sites and deliberately states no total** — v1.21, and the reason is §7 rule
5's new clause rather than fastidiousness. v1.19's version said "fifteen of twenty are the arm-level
assertions", which was wrong twice; plan-gate P10-5 corrected it to "fifteen of the command's
twenty, of which four are call sites", which was right and went stale **within the hour**, because
`93b0e42` added one matching line and the pair became sixteen of twenty-one. The command was right
on all three occasions. What kept breaking was the **count restated beside it**, which is §7 rule 4's
one-home rule applied to a number. The four call sites have now survived **two** unrelated units, and each time the check was
whether that was merit or luck rather than an assumption. `93b0e42` added a line of docstring prose;
`e162ba9` added `_WIDENING_SURFACES`' `envelope_arms` entry, a lambda in a `parametrize` source that
drives the N4 refusal sweep. Both are genuine non-sites, named rather than counted — *(v1.23)*. **But the
claim needs its correction, which plan-gate P11-3 supplies** *(v1.22)*: a site list is stable in a
site's **identity** and not in its **line number** — additions elsewhere leave *which tests are call
sites* untouched while moving *where they are*. So a gloss names sites by a stable handle, which for
a test is its function name, and the four above are named that way.)*

| Site | Edit | Found by |
|---|---|---|
| `stats.py:413`, `:418` — `envelope_arms`'s two `_widen` calls | `clamp=(-1.0, 1.0)` → **`clamp=None`** on both arms. The comment above them ("Both arms keep the identical clamp") goes with them; Rule 4 still requires both arms widened the same way, and `None` on both is the same way | 2 |
| `stats.py:444` — `_compose`'s body | clamps its own result to `SUPPORT_DIFF_PROPORTIONS` and returns **`(interval, bound_by)`**. **The post-edit spelling is prescribed, and prescribing it means writing it here** *(v1.20, plan-gate P10-3 — v1.19 claimed the prescription and left the text only inside two residual commands, which is not a prescription at all)*: <br>`u_lo, u_hi = min(mover[0], exact[0]), max(mover[1], exact[1])` <br>`lo = max(SUPPORT_DIFF_PROPORTIONS[0], u_lo)` <br>`hi = min(SUPPORT_DIFF_PROPORTIONS[1], u_hi)` <br>`bound_by = (…)` — Rule 4a point 4's two branches, decided by `lo != u_lo` / `hi != u_hi`, **not** by a second test of `u_lo`/`u_hi` against the support <br>`return (lo, hi), bound_by` <br>**The local names are the note's** — `u_lo` and `u_hi` are Rule 4a point 4's own notation for the composed unclamped bounds, so an implementer reading the rule this table says it does not restate writes the same names the residuals expect. **The support is subscripted once per bound, never unpacked, and never re-tested** *(v1.24, M11-1 — corrects a row inconsistent with its own residuals since v1.20: the prior sketch re-tested `u_lo`/`u_hi` against the support a second time to compute `bound_by`, spelling each subscript twice against residuals 2 and 3's stated target of **1** each; `lo`/`hi` are now named once, from the clamp itself, and `bound_by` reads their inequality against `u_lo`/`u_hi` instead — the `Landed:` commit above is unchanged, per this section's own rule: *where a landed table's row is later corrected, the correction says so and the `Landed:` commit is unchanged*)*: `lo_b, hi_b = SUPPORT_DIFF_PROPORTIONS` would be a faithful edit that residuals 2 and 3 cannot see, and so would a `bound_by` that spelled either subscript a second time. **`lo != u_lo` is Rule 4a's strict comparison, not a shortcut around it**: `max(SUPPORT_DIFF_PROPORTIONS[0], u_lo) != u_lo` iff `SUPPORT_DIFF_PROPORTIONS[0] > u_lo`, i.e. iff `u_lo` ran strictly below the support — a bound sitting *at* the support because both arms genuinely produced it leaves `lo == u_lo` and stays an arm's (the note's assertion 5), the `max`/`min` identity being what makes that read-off exact. Symmetrically, `hi != u_hi` iff `u_hi` ran strictly above the support. Table C's `:159` row strikes the same bargain for Table G, and writes its spelling out in the row, which is why Table G's residuals over it are safe | 1 |
| `stats.py` — **three** docstring paragraphs, in the two functions Rule 4a restructures *(extended from one at v1.20, plan-gate P10-4)* | `_compose`'s **last** paragraph — the one that says "**it is not applied here yet**" and forward-references this table — is rewritten to describe what the function now does; `7f865e2` deliberately left it as the seam. Two more go stale with the same edit and v1.19 named neither. `_compose`'s **first** paragraph says `verdict()` "needs the arms themselves **for the attribution** and so cannot go through `conservative_envelope`": the conclusion survives, the reason does not — after Rule 4a the attribution comes back *from* `_compose`. And `envelope_arms`'s docstring says "`verdict()` **reads the attribution off the same pair**": after Rule 4a `verdict()` reads no attribution, it **receives** one, and the attribution is no longer a function of the arms pair alone — the third token comes from the composed value against the support. A docstring asserting what its body no longer does is impl-gate P8-5's finding, which this table closes as collateral, so it does not get to reintroduce it | 1, 2 |
| `stats.py:927` — `Verdict.bound_by` | `tuple[str, str] \| None` → **`BoundBy` pair \| None**; the comment gains the third token's meaning — *the `√DEFF` widening ran off the parameter space, so the interval carries no information in that direction* (`-ml` Rule 4a), which is the one thing a reader of the bullet cannot infer | 1 |
| `stats.py:1199` — `verdict()`'s local declaration | the same retype | 1 |
| `stats.py:1215-1218` — `verdict()`'s inline attribution | **deleted.** The attribution is computed once, in `_compose`, from the composed **unclamped** value against the support, on a **strict** comparison. `verdict()` takes both halves of `_compose`'s return and recomputes nothing — which is what finally makes `envelope_arms`'s **no-caller-recomputes-another's-arithmetic** clause true (impl-gate P8-5). *(v1.20, plan-gate P10-4: v1.19 said "docstring sentence", which over-claimed — the sentence's **other** clause, that `verdict()` reads the attribution off the arms pair, is **falsified** by the same edit and is rewritten by the docstring row above. One clause becomes true and one becomes false; only the first is this row's.)* | 1, 2 |
| `report.py:338` — the `- decided by:` bullet | the `p=` clause attaches to the exact-bootstrap arm **only**, and a `support bound` token renders **with its boundary value and never with a level** — `support bound (-1)` on the lower bound, `(1)` on the upper *(v1.20, plan-gate P10-2 — v1.19 said "renders bare", which the note's assertion 10 forbids; the note's sentence is that a support token never carries a **level**, not that it carries nothing)*. **So the renderer needs the support *value*, and its source is named here rather than re-spelled**: `SUPPORT_DIFF_PROPORTIONS`, imported from `stats` exactly as `report.py` already imports `LEVEL_CI95_LO`/`LEVEL_CI95_HI` for this same expression. Writing `-1` as a literal in `report.py` would be a second home for the support, which is the one-arithmetic-two-homes shape this table is closing impl-gate P8-5 for. Today's condition tests one token against one name, so a third token falls into the `p=` branch. **The note's assertion 10 pins the *correct* bullet verbatim** — `- decided by: conservative envelope (lower bound: support bound (-1); upper bound: MOVER-D)` — and `support bound, p=0.025` is what that assertion **kills**, not what it is | 1 |
| `stats.py` — `conservative_envelope` | returns `_compose(...)`'s **first element**. No public signature changes and no new required parameter (`-ml` Rule 4a) | 2 |
| `tests/test_stats.py` — **`test_neither_printed_bound_is_ever_tighter_than_either_arm`**, pinned by **test name and not by line**, because a parallel unit is editing this file *(v1.20, plan-gate P10-1 — the blocker, and the one row v1.19 omitted)* | **This is the one shipped test the edit falsifies, and it is not an addition.** It asserts `lo <= mover[0] and hi >= mover[1]` (and the same for `exact`) against `envelope_arms`' return **directly**; once the arms come back unclamped and the composer clamps, that is false wherever an arm escapes the support — `-1.0 <= -1.5` is False. Measured by the gate against the `7f865e2` blob over the test's own sweep set: **0 / 38 / 78** failures at DEFF 1.0 / 1.2 / 2.0, against a **0 / 0 / 0** control on today's arm-clamped code; first failing tables `(0, 0, 9, 3)`, `(0, 0, 10, 2)`, `(0, 0, 11, 1)`. **The comparison moves to the *clamped* arms** — `lo <= max(SUPPORT_DIFF_PROPORTIONS[0], mover[0])` and `hi >= min(SUPPORT_DIFF_PROPORTIONS[1], mover[1])`, and the matching pair for `exact` — which is **Rule 4a's own restatement of the property** (*"Rule 4's own conservatism property survives verbatim: the envelope is never tighter than the **clamped** MOVER-D arm"*) and **not** a weakening. Relaxing the assertion instead would delete the executable statement of `-ml` §3.4 Rule 4 acceptance 4, which is what that test's docstring says it exists to defend. **No residual can see this** — no residual of this table reaches `tests/` at all, for the scopes stated in the residual block below and not restated here *(v1.23, plan-gate P12-5: v1.22 re-scoped every one of the six to a single file — `modelbench/report.py` for residual 5, `modelbench/stats.py` for the rest — and left this sentence saying `modelbench`; the conclusion held *a fortiori*, a narrower scope seeing strictly less, but a scope claim with two homes is what produced the blocker this row exists for)* — and that is why it is a row: command 2 **does** return the line (`:1361`), so the enumeration was sound and only the site list was short, which is the two-command design working and the hand-written row list not | 2 |
| `tests/test_stats.py`, `tests/test_report.py` | the note's **ten** named assertions land here, with its computed witness values — the two witness tables are `(5, 0, 7, 0)` and `(0, 0, 38, 2)`. Assertions 1–7 and 10 are single-table; 8 and 9 are the two sweeps and both run in seconds at n=12. **Values, tolerances and witnesses are the note's and are not restated here** | 1, 2 |

**Residual after the edit — six, and the split is the point** *(all *before* values re-run at
**`e162ba9`** — the block is re-pointed from `93b0e42` at v1.23, in the revision that re-ran them,
and the enumerating commands above move with it, so the table is single-baselined; it was
two-baselined between v1.21 and v1.22, which plan-gate P11-2 closed)*. **Every one is scoped to the single
file its own site rows name — `modelbench/stats.py` for all of them except residual 5, whose site is
in `report.py` — so none reaches `tests/`** — v1.22, plan-gate P11-1, and §7 rule 5(b)'s scoping default, of which this
table is the occasion:

| # | Residual command (under `model-bench/`) | Form | Before → after |
|---|---|---|---|
| 1 | `grep -nF 'clamp=(-1.0, 1.0)' modelbench/stats.py` | first | **2 → 0** (`:413`, `:418`) |
| 2 | `grep -nF 'SUPPORT_DIFF_PROPORTIONS[0]' modelbench/stats.py` | **third** | **0 → 1** |
| 3 | `grep -nF 'SUPPORT_DIFF_PROPORTIONS[1]' modelbench/stats.py` | **third** | **0 → 1** |
| 4 | `grep -nF '"MOVER-D" if mover_arm[0] <= exact_arm[0]' modelbench/stats.py` | first | **1 → 0** (`:1216`) |
| 5 | `grep -nF 'arm if arm == "MOVER-D" else' modelbench/report.py` | first | **1 → 0** (`:338`) |
| 6 | `grep -nF 'tuple[str, str] \| None' modelbench/stats.py` | first | **2 → 0** (`:927`, `:1199`) |

**Why the scope narrowed, and it is the third consecutive revision to restate this pair** *(v1.22,
plan-gate P11-1)*. v1.19 pinned the whole clamp expression, which exists only in `_compose`, and was
immune to a second consumer but vulnerable to an implementer varying the local; v1.20's repair
removed the introduced local and **widened the match to any subscript of the constant anywhere in
the package** — in the very revision that also mandated a second consumer of it, since the
`report.py:338` row requires the renderer to reach `SUPPORT_DIFF_PROPORTIONS`, `report.py` does
`from modelbench import stats` and `-F` matches straight through the `stats.` qualifier. On the
explicit-branch rendering the package-scoped count is **2** against a stated target of **1**:
DC-12 failing on a faithful edit, arriving by a different mechanism than the one v1.20 fixed. The
gate settled it on a synthetic two-file probe rather than by argument — both spellings planted, the
package-scoped command reads 2 and the `stats.py`-scoped one reads 1 — and the file scope is immune
to whichever rendering the implementer picks, so the renderer's expression stays unprescribed and
this table stays out of `report.py`'s business. **Nothing stated moved**: all six read the same
*before* values file-scoped as package-scoped (**2 / 0 / 0 / 1 / 1 / 2**), re-run at `e162ba9`.
*(v1.23, plan-gate P12-4 — v1.22 read that equality as *the narrowing costs no evidence*, which it
cannot establish. A *before* is not a residual's claim; the **after** is, and after the narrowing
that assertion ranges over one file instead of the package. For residuals 2 and 3 the equality is
worse than weak, being **0 = 0** under any scope including an empty one — and those two are the pair
plan-gate `P11-1` was about. What the narrowing costs is therefore argued residual by residual above, and the
**form** is backed by two measurements instead: the gate's synthetic two-file probe in this
paragraph, and the `e162ba9` insertion — which broke every line pin in this table and moved none of
its six counts — stated at §7 rule 5(b)'s exact-text-versus-line-pin paragraph and not restated
here.)*

**Why residual 1 carries `clamp=` and not the bare tuple, which is the measurement that decided it.**
`grep -rFn '(-1.0, 1.0)' modelbench --include='*.py'` returns **3** lines at `e162ba9`, and the one
that is not a call site is `stats.py:265` — a sentence *inside `_widen`'s docstring* explaining why a
default of `(-1.0, 1.0)` is refused. That is a **disowning mention** (§7 rule 5(b)), and a residual over the bare tuple
would read 1 on a faithful edit and be a trap. Scoping to the keyword argument retires exactly the
two call sites the edit changes and nothing else. Residual 1 also **distinguishes**: an implementer
who changes one arm and not the other reads **1**, and the number says one is left.

**Why residuals 2 and 3 are third-form, decided by §7 rule 5(b)'s trigger rather than discovered
afterwards** — the first time that rule has been applied **forward**, which is what it was written
for. `_compose`'s edit is a *parameterising* one in the rule's sense: the body stops being a single
composed return and becomes a clamp over a composition plus a second return value, so the text a
first-form residual would be stated over — `return min(exact[0], mover[0]), max(exact[1], mover[1])`
— is **destroyed by the faithful edit**, and a half-application that clamps the lower bound and not
the upper spells neither the old line nor anything a retiring count could see. That is Table E's F1
shape exactly, one table over, and it is stated over the text the edit **creates** for the same
reason. **The pair is stated separately, one per bound, because the half-application this table
must catch is one-sided**: clamping only the lower bound reads 1 and 0, and the numbers say which
half was skipped. The cost is rule 5(b)'s stated one — these two pin a spelling, so the spelling is
written out in the `:444` row rather than left to the implementer, and if the expression is later
rewritten for an unrelated reason they are **re-derived over the new spelling, never re-widened**
(§7 rule 5(b)'s third-form trigger — v1.23, plan-gate P12-6; the scoping default of the paragraph
below *narrows*, and this reversal forbids the opposite move, not that one).
**Both are stated over the support constant's *subscript* and not over the whole clamp expression**
*(v1.20, plan-gate P10-3)*: v1.19 wrote them as `max(SUPPORT_DIFF_PROPORTIONS[0], lo)` and
`min(SUPPORT_DIFF_PROPORTIONS[1], hi)`, which pinned two **local names the plan never fixed** —
while Rule 4a point 4 calls those quantities `u_lo` and `u_hi` in the very expressions the
implementer is told to implement. An implementer following the note would have read **0** against a
target of **1**: a residual failing on a faithful edit, in the direction rule 5(b) forbids, inside
the table this document held up as the first written *against* that rule rather than corrected into
it. The subscript form depends on no local name, still reads **0** at `e162ba9` and **1** after the
prescribed edit, and still splits one per bound.

**What no residual can see, named because rule 5(b) requires it.** Three of Rule 4a's decisions are
**behaviour** and no count reaches them: that the comparison is **strict** (a bound sitting *at* the
support because both arms genuinely produced it is an arm's bound — the note's assertion 5, witness
`(0, 0, 12, 0)` at DEFF 1.00); that the **tie-break stays** `<=`/`>=` and is now pinned (assertion 6,
which kills impl-gate Pass 8's surviving mutation 6); and that the printed numbers **do not move**
(assertion 2). What stands in their place is the note's ten assertions, landed by the row above —
residuals prove the edit reached every site, the assertions prove it was the right edit, and neither
half is sufficient alone.

**This table meets Table F on `report.py`, and neither order is constrained** *(§7 rule 5(b)'s round
property)*. Table F's `report.py` sites are `:211`, `:564-584` and `:606-777`; this table's is
`:338`, inside `_decided_by`. They do not overlap, no residual of either is stated over text the
other rewrites, and `_decided_by` is not on Table F's continuous-verdict path — so unlike Tables C
and G on `stats.py:159`, the two are independent and may land in either order. Stated rather than
left to be re-derived, because "two tables touch one file" is where the round property has bitten
twice.

**Free only now, on §4 S1e's own argument, and more cheaply than most.** `bound_by` is computed at
report time and is **not** stored: `grep -rFn bound_by modelbench/results.py` → **0**, re-run, so no
run record carries the two-token set and no `migrate` step is owed even in principle. What makes it
*urgent* rather than merely free is the note's own measurement — the false sentence is not confined
to the not-distinguishable path, and at the design effect a determinism probe is most likely to
return, all of it lands beside a published, positive verdict.

### S2 — Packs, LM Studio adapter, host info, runner

**Create:** `modelbench/{packs,lmstudio,hostinfo,runner,convo,tooling}.py`.

```python
# packs.py
@dataclass(frozen=True) class Pack:
    packId: str; packVersion: str; role: str; contentHash: str   # total: a loaded pack knows its bytes
    manifest: Mapping[str, Any]; root: Path
    def data_path(self, key: str) -> Path: ...
    def load_tool_module(self) -> ModuleType: ...   # importlib from pack root
    def ref(self) -> PackRef: ...                   # §3.3: contentHash never None on this path
def load_pack(root: Path) -> Pack: ...
def content_hash(root: Path) -> str: ...            # SHA-256, sorted paths, excludes PROVENANCE.md
def validate_pack(pack: Pack) -> list[str]: ...

# lmstudio.py
class LMStudio:                                     # base_url from host.json apiBaseUrl (§3.4.4a)
    def catalog(self) -> list[ModelInfo]: ...       # GET /api/v0/models — the only fingerprint source
    def residency(self) -> list[ResidentModel]: ... # catalog filtered state != "not-loaded"; {id, state}
    def chat(self, messages, *, model, tools=None, temperature, max_tokens,
             timeout_s: float) -> ChatResult: ...   # timeout_s required, no default (§3.6's two budgets)
    def embed(self, texts: Sequence[str], *, model, timeout_s: float) -> EmbedResult: ...
    def warm_up(self, model: str, *, call_surface: Literal["chat", "embeddings"],
                system_prompt: str | None, was_resident_before: bool,
                timeout_s: float) -> LoadResult: ...
                                                    # was_resident_before: required, no default;
                                                    # the caller's §3.4.4a step-3 snapshot, echoed
                                                    # onto LoadResult so coldLoadSeconds' precondition
                                                    # is explicit at the type (v1.25, impl-gate P13-9)
                                                    # under JIT this is the load; content discarded,
                                                    # `runtime`/`stats` kept on the chat surface only
    def probe(self) -> Literal["api-v0", "v1-only", "unreachable"]: ...   # §3.4.4a's two-step probe
class LMStudioCallFailed(LMStudioError):
    status: int | None                              # v1.26; DELIVERED at d5b549d as a REQUIRED
                                                    # keyword arg with no default. The HTTP status
                                                    # when the server ANSWERED and refused, None
                                                    # when the call never completed OR answered
                                                    # unusably (a 2xx whose body is not a usable
                                                    # response carries None deliberately: the
                                                    # server did not refuse). Absent-vs-empty
                                                    # again, and the only thing separating
                                                    # §3.8.4's `server-rejected` from
                                                    # `no-response` — at 40a9bc8 an HTTP 400 and a
                                                    # dropped connection raised the same class.
                                                    # `timed-out` needs no field: LMStudioCallTimeout
                                                    # is its own class.
# No load/unload/ps: v1.7's three lms.exe operations are gone with the CLI (§2.5, §3.4.4a), and
# nothing on either HTTP surface can unload — the harness cannot force a cold state.

# tooling.py
class ToolEnvironment(Protocol):
    def schemas(self) -> list[dict]: ...
    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any: ...
    def trace(self) -> list[DispatchRecord]: ...
    def state(self) -> dict[str, Any]: ...

# convo.py
def assemble(turn_index: int, script: Sequence[Turn], observed: Sequence[TurnTrace],
             cfg: PromptConfig) -> list[ChatMessage]: ...
             # v1.25: `observed` is what this run actually produced for turns 0..turn_index-1;
             # PRECONDITION len(observed) == turn_index, raise otherwise. That equality is the
             # structural guarantee — you cannot assemble turn n from anything but n observations,
             # so the §3.8.4 ruling cannot be undone by a caller. `script` supplies user text only.
def drive(env: ToolEnvironment, script: Conversation, llm, cfg: PromptConfig) -> ConversationTrace: ...
```

`ChatResult` carries the raw `stats` (ttft, generation_time, tokens_per_second, stop_reason),
`model_info`, `runtime`, `usage`, and the parsed native `tool_calls` — plus a
`toolCallForm` field distinguishing **native tool-call** from **prose that looks like a call**,
which is FR-8(b) and must be decided at the transport boundary where the evidence is, not later.

**It also normalises the units, and that is the same argument one field over** *(v1.10, plan-gate
P4-1 — §3.6's unit boundary)*. LM Studio reports `time_to_first_token` and `generation_time` in
**seconds**; every `…Ms` field in this plan is milliseconds. `ChatResult` therefore exposes
`ttftMs`, `generationMs`, `tokensPerSecond` (unconverted — already a rate) and `wallClockMs`
**derived on construction**, and the raw `stats` mapping stays beside them for auditability only:
no runner, scorer or report path reads a timing figure out of it. Deciding the unit at the transport
boundary is what stops a seconds value reaching `-ml` §11.5.1's millisecond threshold, where it
withholds every call slower than about a second. **Each of the three is `None` when its source key is
absent, never `0`, and construction never raises on a missing or partial `stats`** *(v1.11,
plan-gate P5-8)* — a chat response without `stats` is an expected state, not an error, and rule
(iv-a) below commits the report to printing it.

**CLI:** S2 ships `attest`, `validate`, `run`'s plumbing, and the installed-catalog half of
`models --tested` (§3.6a). `attest` is assigned here and not later because S3's done-condition —
"a stored result with a **complete** fingerprint" — is unreachable until `host.json` exists.

**Done when:** `packs`/`convo`/`tooling` are unit-tested offline against a stub LLM and a fixture
pack; **`validate_pack` enforces the §3.3 `sampling` contract** — a fixture pack declaring
`analysisUnit: "conversationId"` under `scripts: 12, replicatesPerScript: 4` is rejected on the
row-count identity (48 distinct values where 12 are required), one declaring
`analysisUnit` outside its own `pairingKey[0]` is rejected structurally, one declaring
`replicatesPerScript > 1` is rejected per `-ml` §3.4 Rule 6, and — **route (iii), v1.25** — a pack
whose `pairingKey[0]` is not `roles.analysis_unit_field(role)` is rejected for **every** role in
`roles.ROLES`, driven by executing `check_sampling_contract` rather than by reading the constant,
with `set(ANALYSIS_UNIT_FIELD_BY_ROLE) == set(roles.ROLES)` computed in its own assertion; the
`tool-caller` case of that sweep is P12-7's own fixture shape (`pairingKey[0] == "conversationId"`),
which passes routes (i) and (ii); **`validate_pack` enforces §3.3's `maxIterationsPerTurn` role
rule** — v1.26 — refusing a `tool-caller` manifest without the field and any other role's manifest
with it, both driven per role through `roles.MULTI_CALL_TURN_BY_ROLE` with its own computed
`set(...) == set(roles.ROLES)`; **`validate_pack` derives `callSurface`
from `environment.requires` and rejects a pack declaring neither or both of `lmstudio-chat` /
`lmstudio-embeddings`** (§3.4.4a), and `run` rejects, before any model call, a pack whose derived
surface contradicts the model's catalog `type`; `validate_pack`'s AST import check
rejects a fixture pack module that imports outside the
allowlist, and `run` is shown to call it and fail closed (§3.3); **`load_pack(...).ref().contentHash`
is not `None`** and equals `content_hash(root)`, while a `PackRef` from `pack_ref_from_manifest`
is `None` there — the §3.3 totality boundary, asserted rather than assumed, and `validate_pack`
reuses `packs.check_sampling_contract` rather than re-implementing the rule (impl review Pass 1,
§4 item 6); `lmstudio` is unit-tested against
recorded JSON payloads, including the §3.6 eligibility gate **on a `tool-caller` pack** over the
three real catalog entries that
break the naive rule (an `embeddings` model advertising `tool_use` → refused; an entry with **no**
`capabilities` key → admitted; an `llm` with `tool_use` → admitted) — **and the same
`embeddings` entry on an `embedder` pack, where the gate does not run and the run proceeds**
*(v1.11, plan-gate P5-1: the negative assertion is the one that would have caught an unscoped gate,
because every positive one passes with the scope missing)*; `attest` writes a `host.json`
matching §3.4.4's schema — with `observedAtAttestation` carrying `residencySource` and **omitting**
`runtimeName`/`runtimeVersion`, since neither probed endpoint exposes a `runtime` (v1.10, plan-gate
P4-6: v1.9's done-condition was unbuildable as written) — and the trip-wire's three outcomes are
each asserted: a first `model:chat` run **back-fills** both keys and records
`attestationTripWire: "first-observation"`, a second run with the same runtime records
`"compared"` and proceeds, a second run with a changed `runtimeVersion` exits `5`, and an
embeddings arm records `"unavailable"`; and
**one** `-m live` test confirms `catalog()` returns the real installed models and that `chat()`
surfaces `stats.time_to_first_token`.

**`convo.py`'s replay contract, spelled out because v1.24 left it to be inferred and it was inferred
wrongly** *(v1.25 — the ruling and its reasons are §3.8.4's "Prompt assembly" bullet; what follows is
only what to build)*. `assemble(turn_index, script, observed, cfg)` returns the **whole** message
list for turn `turn_index` — the harness resends the conversation from scratch every turn — in this
order: system prompt if any · the tool-schema text block on turn 0, or every turn when
`representToolSchemasEachTurn` · the replayed history · `{"role": "user", "content":
script[turn_index].user}`, always last. `historyTurns > 0` windows the replayed prefix from the
tail; `0` replays all of it.

- **Preconditions, both raising:** `0 <= turn_index < len(script)`, and `len(observed) ==
  turn_index`. The second is the whole mechanism — a caller cannot assemble turn *n* without *n*
  observations, so there is no argument through which a textbook prefix could re-enter.
- **`structured`** — per replayed prior turn *i*: `{"role": "user", "content": script[i].user}`;
  then, per in-turn iteration, the model's own assistant message with `content` exactly as returned
  (`None` permitted) and `tool_calls` **verbatim, ids included**, followed by one
  `{"role": "tool", "tool_call_id": …, "name": …, "content": …}` per call in emission order, the
  content being that call's real `DispatchRecord.returnValue` JSON-encoded; then the turn's final
  assistant reply. **Every `tool_calls` entry gets exactly one `tool` message, without exception** —
  for one that could not be dispatched (no name, or the dispatch raised) the content is a JSON
  object naming the failure. An unanswered tool call is rejected by OpenAI-shaped servers, so
  omitting it would convert a model failure into a transport failure at the next turn.
- **`structured-replies-only`** — `{"role": "user", …}` then one `{"role": "assistant", "content":
  <that turn's final reply text>}`. No `tool_calls`, no `tool` message, no breadcrumb.
- **`plaintext`** — one `{"role": "user", …}` carrying a flattened transcript, `User: …` /
  `Assistant: <final reply text>` per prior turn, likewise with no tool evidence.
- **`none`** — no history messages.
- **A prior turn with no final reply — any `turnDisposition` other than `replied` — is replayed and
  never omitted, in every mode.** In the two reply-text modes it contributes an assistant message
  with `content: ""` (*captured and empty*, which is what the conversation contained); **under
  `structured` it contributes its iterations and no trailing assistant message** *(v1.26, P13-9 —
  v1.25 ruled only the reply-text modes, leaving `structured`'s recipe ending on "then the turn's
  final assistant reply" for a turn that has none, and a prefix ending on a `tool` message
  immediately before the next `user` turn)*. Omitting the turn in any mode would shorten the visible
  history, and history length is the covariate this whole pack is measuring against.
- **`drive`** runs §3.8.4's bounded per-turn loop, `assemble`s each turn from the `TurnTrace`s it has
  accumulated, catches `LMStudioError` and **continues the script** (§3.8.4; `-ml` §4.1), and
  returns a `ConversationTrace`. `TurnTrace` carries `messagesSent`, `chatResults` (**all**
  completed iterations, in order — empty for a turn whose first call raised), `dispatches` (that
  turn's own slice of `env.trace()`), `envState`, `iterations`, `turnDisposition`,
  `finalReplyText: str | None`
  — `None` **iff** `turnDisposition != "replied"`, per §3.8.4's table, which is the only
  home of that mapping — and `wallClockMs`. **`iterations == len(chatResults)`**, the calls that
  *completed*, so a turn that raised on its third call records `2` (§3.8.4, P14-6). **`wallClockMs`
  brackets the whole turn** — every iteration and the tool dispatches between them — and the runner
  stores it as the item's measured figure only when the turn completed; on a turn that ended on a
  raise it is incomplete and `ItemTiming.wallClockMs` is `None` (§3.6's withholding bullet, §3.8.4).
  *(`capHit` is retired as a field: it was one bit standing where five states are, and
  `turnDisposition == "cap-hit"` is the same fact without the false `iff`.)*
- **The disposition set is plan data, and its guard lands in a precursor unit** *(v1.26; the plan
  gate's structural condition. **Delivered at `d5b549d`**, holding v1.26's four members — the two
  module declarations and legs 1 and 2 of the probe — so from v1.27 the rows below are an **edit**
  to shipped code, not a fresh build.)* `convo.TURN_DISPOSITIONS: frozenset[str]` holds exactly
  the members of §3.8.4's table — **five from v1.27**, `{"replied", "cap-hit", "timed-out",
  "no-response", "server-rejected"}` — and a **three-way** probe asserts that
  `set(get_args(TurnDisposition))`, `TURN_DISPOSITIONS`, and the set of dispositions the S5 scorer
  branches on **each equal a constant transcribed into the test from §3.8.4's table** — each
  compared against the transcript, never against each other. **The transcript's source is §3.8.4's
  table in all three places**; two of them named §4 S2 at v1.26, which pointed a reader at the
  bullet holding the set literal rather than at the mapping's declared only home.
  **What the precursor buys is cross-unit protection, and v1.26 over-claimed it** *(v1.27, P14-3's
  second half)*. The claim was that *"two sets authored in one unit agree by construction and a
  probe that cannot redden is not a guard"* — but the **transcript** is authored in that same unit
  too, so in round 1 all three artefacts share one author and the probe cannot redden then either.
  What it genuinely buys is real and is the reason it was worth a round of its own: it reddens when
  a **later** unit widens the enum without touching the transcript — which is exactly what v1.27
  does, and which is how the two-state reading arrived in the first place. The shipped module
  docstring on `TURN_DISPOSITIONS` carries the same over-claim in its own words; the rework unit
  corrects it in the same pass, in a file it is already editing.
- **The v1.27 edit to that shipped guard, stated so the rework unit does not read red as
  regression.** `convo.TurnDisposition` gains `"timed-out"`; `convo.TURN_DISPOSITIONS` gains the
  same string; `tests/test_convo.py`'s transcribed constant gains the fifth row's token, and legs
  1 and 2 go **red until it does**. That reddening is the guard working. All three sites are in the
  two files the rework unit already owns, and the third leg is S5's (below).
- **The tests that make it real** are §5 tests 10 and 10b; the one that would have caught v1.24's
  reading is its negative: a `structured`-mode fixture whose `observed` assistant text differs from
  every string in the script's `expect` blocks, asserting the **observed** text appears in the
  assembled list and no `expect` string does.

**Three additions in v1.8, all in the adapter and the runner:**

- **The host-info seam is §3.4.4a's, and `hostinfo` launches no subprocess for it.** `probe()`
  returns each of its three values against stubbed HTTP — a v0 catalog; a server answering
  `/v1/models` but 404 on `/api/v0/models`; nothing listening — `run` exits `3` with the
  *distinguishing* message in the second case, and **no `model` record is written in either failure**;
  `residency()` maps a stubbed catalog to `{id, state}` on `state != "not-loaded"` and returns `[]`
  for an all-not-loaded payload. **The fixture is `tests/fixtures/lmstudio/catalog.json`, and this
  done-condition no longer cites a capture that does not exist** *(v1.25, F-S2-1: v1.8 wrote
  "§2.5's captured 19-model response is the fixture" and §2.5 is a narrative probe record — it names
  the field set and states all 19 entries read `state: "not-loaded"`, and no response body was ever
  saved)*. Two halves, and the second is the one that closes it:
  - **Now, offline:** the shipped fixture holds only entries this repo's documentation records a
    field for, and it **does not pad to 19 with invented models** — the count is whatever the
    citations support. Provenance is carried as a **single top-level `_provenance` object with a
    `perEntry` map keyed by model id**, one citation per entry, not as a `_provenance` key inside
    each entry; any field with no citation is marked a placeholder there, and no placeholder may sit
    in `type` or `capabilities`, the two fields the §3.6 eligibility gate and `residency()` actually
    read. A fixture that padded would make the all-not-loaded assertion true of a catalog nobody
    observed. *(v1.26, P13-10: v1.25 wrote "each carrying a per-entry `_provenance` citation", which
    reads as an entry-level key and describes a shape the fixture does not have — the gate parsed it
    that way and reported zero. Parsed at `40a9bc8`, the fixture has one `_provenance.perEntry` map
    citing all **seven** ids by name, so the substance was already true and the description was not.
    Corrected in place rather than turned into an imperative: there is nothing to re-author.)*
    **What was genuinely missing is the assertion**, and it is owed here rather than trusted:
    `set(catalog["_provenance"]["perEntry"]) == {e["id"] for e in catalog["data"]}`, computed, so an
    entry added without a citation reddens instead of shipping.
  - **On the live session this stage already requires** (R-1's probe below, which needs a human):
    save the verbatim `GET /api/v0/models` body to `tests/fixtures/lmstudio/catalog-live-<date>.json`,
    re-point these assertions and
    `test_catalog_parses_every_fixture_entry_into_model_info` at it, and record the capture in
    `model-bench/docs/HISTORY.md`. It is nearly free — the call needs LM Studio **running**, not a
    model **loaded**, which is exactly §2.5's clean-box state — and it is the only thing that turns
    "documented" back into "captured". Until it runs, the honest framing above is what ships.
- **The timing discipline is tested offline against a stub clock and a stub LLM** (§5 test 15b): the
  warm-up is issued exactly once per arm, on the right surface for the arm's profile, and produces
  no `ItemResult`; a stubbed 21 s first response
  followed by sub-second ones completes under the two budgets where a single warm-sized budget would
  not; `coldLoadSeconds` is set when start-residency was empty and **absent, not `0`,** when it was
  not; an item whose preceding residency snapshot shows the model not resident has **`latencyMs`
  withheld and its `stats`-derived siblings kept**, and is **still scored**; an item whose
  `unexplainedMs` exceeds `-ml` §11.5.1's threshold is withheld the same way and counted under the
  same cause; a scored call that hits `requestTimeoutSeconds`
  is scored `fail` with no timing **figure** — an `ItemTiming` whose only populated field is
  `withheldFor: "timeout"` (v1.11, plan-gate P5-6/P5-9) — and the run continues; and item 1's
  baseline is the **post-warm-up** probe, so a clean cold run withholds nothing. **Three
  multi-call cases join them at v1.27** and are written out at §5 test 15b, because every timing
  fixture named above is single-call and the regression they catch — a per-call reading of
  §11.5.1's gap firing on iteration count rather than on a load — is invisible without one.
- **One shipped docstring is stale from the moment the loop lands, and it is a one-word sweep
  rather than a rewrite** *(v1.27, `-ml` §11.9 ask 7's last item)*. `modelbench/lmstudio.py`'s
  `_coerce_finite_float` docstring quotes the detector's gap as `latencyMs - (ttftMs +
  generationMs)`. The operand at that level is the **call's** `wallClockMs`; `latencyMs` is the
  *item's* admitted figure and after this ruling is not an operand of the gap at all. The
  docstring's actual claim — that a non-finite value would send the gap to `-inf` so the detector
  could never fire — is unaffected and correct, so only the expression changes. Pinned by symbol
  and not by line, because the note cites `:225` and the same text sits at `:247` at `d5b549d`:
  `grep -n 'latencyMs' modelbench/lmstudio.py` returns **exactly one** line today, and **zero**
  after the sweep — the module has no other business with an item-level field name.
- **The latency block is S2's to produce**, because S2 is the first stage at which a timing exists.
  `RunResult.latency: LatencyBlock | None`, a frozen dataclass carrying `latencyMsP50`,
  `latencyMsP95`, `latencyMsMax` (each `float | None`), `latencyTimedCount`, `latencyItemCount`,
  `latencyWithheldForLoad`, `latencyWithheldForNoResponse`, **`callCount`** (`-ml` §11.4's
  `Y_calls`, v1.27) and — because the `stats`-derived figures
  are **kept** on a load-contaminated item and so have a *different* coverage from the wall clock
  (`-ml` §11.4) — `statsCoveredCount`, the number of **calls** carrying a `stats` object
  **and a usable `promptTokens`** (rule (iv-c), v1.11; **calls, not items, from v1.27** — the three
  figures it denominates are call figures and an item is `callCount` of them).
  **Four aggregate figures join them in v1.10** *(plan-gate P4-3: v1.9 committed the report to
  printing these and gave them nowhere to live)* — `ttftMsMedian`, `prefillMsPer1kMedian`,
  `tokensPerSecondMedian` and `unexplainedMsMax`, each `float | None` — and
  `latencyWithheldForTimeout` is **renamed `latencyWithheldForNoResponse`** *(plan-gate P4-7)*,
  which is what it was always counting.
  One coverage number per *field group*, not one per run and not one per field: the wall clock has
  its own — **in items** — its three siblings share `statsCoveredCount` — **in calls** — and each
  is printed beside the figures it governs. **The two units are printed side by side and never
  substituted for one another** (`-ml` §11.4, §11.7 slot 2's two lines); on every role but
  `tool-caller` they coincide, which is precisely why a substitution would ship green. **Every figure and every count is computed from `run.items` in one pass and asserted
  against a recomputation from them** — the block is stored for the record's self-containment, and a
  stored number whose inputs are also stored is one that can be checked rather than trusted. **Nine**
  rules, none of them optional *(eight at v1.10; (iv-c) is v1.11's — plan-gate P5-5)*:
  **(i)** the three **wall-clock** figures — `latencyMsP50`, `latencyMsP95`, `latencyMsMax` — are
  `None` exactly when `-ml` §11's gates refuse them, never `0`,
  never a number carrying a prose qualifier. **The three `stats`-derived medians**
  (`ttftMsMedian`, `prefillMsPer1kMedian`, `tokensPerSecondMedian`) are `None`
  under **either** of two causes — a gate refusal against their own coverage (iv-b), or
  absence-of-input under (iv-a) — and **the block does not distinguish them**; a reader who needs to
  can, because `statsCoveredCount` is `None` in the second case and a number in the first.
  **`unexplainedMsMax` is split out of that sentence at v1.27** (`-ml` §11.9 ask 7): it is a
  **maximum over the items that had a reading**, it takes **no** coverage gate, and it is `None`
  only under absence-of-input — no item yielded a readable gap, or (iv-a)'s surface produces no
  `stats` at all. §11.6 gates *medians* because a median over a selected subset misrepresents the
  run; a maximum over a subset is a **lower bound** on the run's largest gap, which is the useful
  direction for the one job that figure has. It is printed with the count of items that had a
  reading, because a bound whose base is unstated reads as a point estimate;
  *(v1.11, plan-gate P5-7. At v1.9 this block held exactly three `float | None` figures and "the
  three figures" was unambiguous; v1.10 added four more without sweeping the clause, and the
  sentence directly above it uses "its three **siblings**" for the other trio — so under the sibling
  reading (i) forbade exactly the `None`s (iv-a) mandates. One word, and it is §7 rule 4's sweep
  missed inside the revision that wrote rule 5.)* **(ii)** `latencyItemCount == len(run.items)` **and** `callCount == Σ_items len(timing.calls)`,
  both asserted,
  because they are stored for the record's self-containment and a stored duplicate that can drift is
  worse than a derived one. **`callCount` is v1.27's and carries a third assertion** the note asks
  for by name (`-ml` §11.4, §11.10 (7d)): each item's `len(timing.calls)` equals that item's
  `TurnTrace.iterations`, because two independently maintained counts of the same thing is how
  several of this component's defects started; **(iii)** `latencyWithheldForLoad + latencyWithheldForNoResponse ==
  latencyItemCount − latencyTimedCount`, asserted, so the cause split can never fail to account for
  the gap — and it now holds for **every** withheld item, because §3.6's fourth disposition gives the
  scored call that fails without timing out a named cause, where v1.9's two causes left it under
  neither and falsified this very line (plan-gate P4-7); **(iv)** `statsCoveredCount` is the count of **calls** carrying **both** a usable `stats` object
  **and** a usable `promptTokens` (iv-c), computed in the same one pass as everything else and
  **asserted against a recomputation from `run.items`** — now a **sum over each item's `calls`**
  (v1.27) — and that recomputation is the equality worth
  asserting. On a `stats`-bearing surface it is bounded by
  **`statsCoveredCount ≤ callCount`**, an **inequality** whose
  gap is exactly (iv-c)'s co-presence exclusions; a load-contaminated call still returned a response
  and still contributes, which is the whole content of the note's v1.10 reversal and the thing an
  implementer who withholds the siblings anyway will break. *(v1.11, plan-gate P5-5: v1.10 wrote this
  as an equality against `latencyItemCount − latencyWithheldForNoResponse`, which (iv-c)'s
  disposition falsifies — an excluded item returned a response and
  **was** timed, so it sits on neither side of that subtraction. **v1.27 re-bases it on the call
  count**, because the left side counts calls and the old right side counted items, which differ by
  construction on any pack whose `maxIterationsPerTurn` exceeds 1 — the one-word substitution
  `-ml` §11.9 ask 7 warns will otherwise ship. The bound is written in the form above rather than
  as that ask's `Y_calls − (calls that returned no response)` because, under the same note's pin
  that `callCount == len(chatResults)`, a call that returned no response is not in `callCount` at
  all and the subtrahend is identically zero; §3.8.4's raise **R-1** carries that to
  `data-scientist`, and the form here is true under either resolution.)*;
  **(iv-a)** on a call surface that returns **no** `stats` — today `POST /api/v0/embeddings`, and
  every `deterministic` arm — `statsCoveredCount` is **`None`, never `0`**, rule (iv) does not apply,
  and `-ml` §11.7's second denominator line is not rendered *(v1.10, plan-gate P4-13, confirmed by
  `-ml` v1.12 §11.9 ask 5)*. **The condition is the call surface, not the arm profile**, which is the
  wider and therefore the correct one: `0` keeps its meaning on the chat surface, where it says every
  response lacked `stats` and is a real signal, while a profile-shaped condition would miss the
  deterministic arm. The four sibling figures are `None` there for the same reason, and so is
  `unexplainedMsMax` — with no `stats` there is no gap, so §11.5.1's detector does not run and the
  between-item probe is that arm's only load producer (§3.6);
  **(iv-b)** the three sibling **medians** take `-ml` §11.6's p50 gate against **their own** coverage
  — **`X = statsCoveredCount`, `Y = callCount`** — never against the wall clock's, which is the
  whole point of giving them a separate denominator (§11.4). The gate, its constant and its integer
  form are the note's. ***`Y` is `callCount` and not `latencyItemCount`, and v1.27 says so in bold
  because it is the one-word substitution `-ml` §11.9 ask 7 predicts will otherwise ship*** — the
  two are equal on every role but `tool-caller`, so the wrong one passes every test that does not
  use a multi-call fixture;
  **(iv-c)** **co-presence — one coverage number is honest only while the three figures are
  co-present, and where they are not the *call* leaves *both* the count and the medians** *(v1.11,
  plan-gate P5-5; ruled by `-ml` v1.14 §11.4; **restated per call at v1.27**, since v1.25 made an
  item several of them)*. `ttftMs` and `tokensPerSecond` need a `stats` object;
  the prefill figure **additionally** needs a usable `usage.prompt_tokens`, and
  `CallTiming.promptTokens` is `int | None` — so the plan's own type admits the case a single
  denominator denies. The rule: a **call** carrying `stats` whose `promptTokens` is **absent or `≤ 0`**
  is excluded from `statsCoveredCount` **and from all three sibling medians**. **The exclusion is
  per call and never per item**: on a multi-call item the sibling figures of its *other* calls are
  good measurements and nothing about the excluded one contaminates them (`-ml` §11.4). The item's
  `unexplainedMs` is the separate question with the opposite answer — §11.5.1 forms **no partial
  sum**, so one unreadable call makes the whole item's gap `None`. **Both halves of that
  sentence are load-bearing** — excluding it from the count while still letting it into the `ttftMs`
  and `tokensPerSecond` medians prints a denominator that does not describe its own numerator, which
  is worse than either clean option. **And it is a *disposition*, not an assertion: nothing raises.**
  An implementer who writes it as an `assert` crashes a real run on a rarity — plan-gate P4-7's shape
  a second time — and one who applies it to the count but not to all three medians reintroduces the
  defect it closes. The four places are one edit: the count, and each median's input set.
  **Rules (ii), (iii), (v) and (vi) are item-level and do not move** — v1.27 says so explicitly,
  because *"one unit changed"* is how the other five get changed by accident (`-ml` §11.9 ask 7).
  *The
  alternative the note weighed and rejected — a fourth `prefillCoveredCount`, exact but a permanent
  second gate evaluation, second stored count and second printed line — carries a reversal trigger
  that costs nothing to watch, since the exclusions are countable from `run.items`: the first run in
  which they are not a rarity is the run that buys it. Why one count rather than two is §11.4's
  reasoning and is not restated here.*
  **(v)** `None` on the block itself means *this run produced no timings at all* —
  correct for S1's fixtures and for a `deterministic` arm — and is the only permitted default, which
  is not a violation of the no-defaults rule (`designEffect`, §4 S1) because the anti-conservative
  value there was a *number*; here absence is the honest state;
  **(vi)** the counts are **derived from the items, never reported alongside them**:
  `latencyTimedCount` is the count of items whose derived `latencyMs` is not `None`, and the two
  withheld counts are counts of `timing.withheldFor` values (§4 S1's `ItemTiming`) — which is what
  makes (iii) an assertion about the record rather than an identity a miscounting runner satisfies by
  construction. **The mapping is three item states onto two counts** *(v1.11, plan-gate P5-6)*:
  `latencyWithheldForLoad` counts `"load"`, and `latencyWithheldForNoResponse` counts **`"timeout"`
  and `"no_response"` together** — one counter, because `-ml` §11.7 slot 2 prints one `no response`
  label for both, and two item states, because §11.5.1's `censoringExact` reads the distinction per
  item. Every item whose `timing.withheldFor` is not `None` lands in exactly one of the two, so (iii)
  is unchanged. The estimator, both floors, the
  denominator and the printed grammar are `-ml` §11's and are cited, never restated.
- **`censoringExact` is computed at render time and never stored** *(v1.10, from `-ml` v1.12
  §11.5.1)*. §11.7's slot 3 now has two variants selected on a **computed** predicate, because the
  `unexplainedMs` detector withholds on a covariate rather than on the wall clock and so is not
  right-censoring. The predicate is the note's. What this plan owns is that **both its inputs live on
  the record** — `ItemTiming.wallClockMs` survives on a withheld item and `timing.withheldFor` names
  the producer (§4 S1) — and that `report.py` evaluates it per render from `run.items` rather than
  storing a boolean the items already determine. **Naming the producer is what makes the predicate
  evaluable, and it takes three values rather than two** *(v1.11, plan-gate P5-6; `-ml` v1.14
  §11.5.1)*: a `"timeout"` item satisfies the predicate's first clause with no comparison at all, a
  `"no_response"` item that is not a timeout forces it **false** on the note's new explicit branch,
  and a `"load"` item is compared on its wall clock and fails safe to false where that wall clock is
  unreadable. Under v1.10's merged two-value `withheldFor` the first two were indistinguishable, so a
  timeout-only run rendered the weaker string with nothing recording that it had — the safe
  direction, but not the true one. The predicate and both slot-3 strings remain the note's.
- **S2's scorers derive `aggregates` from the same `items` they emit, in one pass** — this settles
  the impl-gate Pass 4's open question 1, which routed the shape here. Every `BinaryMetric.n` a
  scorer emits is *computed* in that pass and never counted along a second path — **and v1.10 scopes
  how**, because the unqualified form contradicted DC-10 (plan-gate P4-8). For a metric whose `unit`
  is `roles.unit_kind(pack.role)`, `n` is the count of items the scorer marked
  `scored_outcome(metric) is not None`. For a **pooled** metric — `unit` finer than the role's, so
  `turn` or `call` — `n` is the **sum of the per-item denominator contributions the scorer records
  in the same pass**, under the reserved `ItemResult.counts` key `"<metric>#denominator"` (§4 S1
  DC-10). Neither form is an item count for a pooled metric and neither is a second path: one pass,
  two arithmetics, both confirmable from `items`. **A continuous metric is the third** *(v1.12,
  plan-gate P6-1)*: its per-item value goes into `ItemResult.measures` in that same pass, its
  aggregate's `n` is the count of items the scorer marked `scored_value(metric) is not None`, and
  the scorer states the metric's **`support`** — it is the only party that knows whether the
  quantity is bounded. A scorer never writes one metric name into both maps; `ItemResult` refuses
  that at construction (§4 S1e Table F), so the kind is a fact about the record rather than a
  convention between two files. That makes S1's cross-check (S1 done-condition 10) unfalsifiable from
  inside a correct scorer, which is precisely its job: it is the net under the seam, not a substitute
  for getting the seam right, and it earns its place because the next scorer will be written by
  someone who has not read this bullet. **S2 does not re-implement the check** — it is S1's, and it
  has already landed by the time S2 runs.

**R-1's probe is part of this done-condition, not a note** (§6 R-1 promised it "during S2" and v1.1
gated nothing on it). *(v1.8: same probe, new instrument — there is no `lms ps` any more.)* With a
model actually loaded — issue one warm-up call, which under JIT auto-load *is* the load — re-read
`GET /api/v0/models` and record in `model-bench/docs/HISTORY.md` whether the loaded entry exposes the
KV-cache or load configuration. The 2026-09-03 probe saw only `not-loaded` entries and their ten
keys (§2.5); `loaded_context_length` is known to appear on load (§2.3), and whether anything else
does is exactly what this probe settles. **If it does, `kvCacheSetting` moves from operator-attested
to auto-captured** and §3.4.2/§3.4.4 are updated in
the same change; if it does not, the recorded negative result is what closes R-1. Either outcome
satisfies the condition; silence does not.

**The same probe now answers two more questions, and both are one extra assertion on a call it
already makes** *(v1.9)*:

- **Does LM-Studio-reported `time_to_first_token` include the JIT load? Answered — it does not**
  (`-ml` §11.4, measured after this bullet first asked for it; §11.5.1 turns the same gap into R-14's
  detector). What S2 still owes is the *re-measurement*, not the question: record the warm-up's own
  `stats` beside its wall clock on each known-cold load, because §11.5.1's threshold is a starting
  value resting on two cold observations and the first real pack run is where it gets re-checked.
- **Does `loadedContextLength` appear on a loaded *embeddings* model?** §2.3's evidence is from a
  chat model, and the field is `REQUIRED_NONEMPTY` in the `model:embeddings` set. Read a loaded
  embeddings model as well as a loaded chat one; if the key is absent, the field moves **to that
  profile's `REQUIRED_PRESENT` column, captured `""`** — not out of the required set, which under
  §3.4.1's derivation would forbid it (§3.4.4a, plan-gate P4-10). Free either way — no embedder
  record exists before S3.

### S3 — `embedder` pack + `refresh_golden.py` (first end-to-end result)

**Create:** `model-bench/scripts/refresh_golden.py` (one-way, human-invoked importer),
`packs/embedder-graphrag-retrieval/` (§3.8.1), `modelbench/scoring/retrieval.py`.

Sequenced third on purpose: it is the cheapest path to a complete real run — the golden data
exists, the mechanism is pure arithmetic, and there is a pinned prior figure to sanity-check
against. It proves the S1/S2 core end to end before the expensive packs are built on it.

**Done when:**

1. A real run against `text-embedding-qwen3-embedding-0.6b` produces a **stored result with a
   complete `model` fingerprint** (which requires `attest` from S2 to have written `host.json`).
   **§4 S1e Table F must have landed** *(v1.12, plan-gate P6-1)* — not as an added gate but as a
   precondition nobody has to remember: this pack's only verdict metric is `mrr`, a per-item
   **continuous** value, and until `ItemResult.measures` exists there is nowhere on the record for it,
   so no result of this pack is storable at all. The verdict this run renders is `-ml` §3.2e's string
   **4 or 5**, never one of the three that carry `pp` and a McNemar clause.
2. The BM25 arm is stored as its own `armKind: "deterministic"` record sharing that run's
   `sessionId`, and `compare` renders the two arms in one report with the deterministic one
   labelled (§3.4.1, §3.8.1), each arm's `sep_z` appearing as a **`DistributionSummary` row —
   median and p10, and no interval** (§4 S1e Table F). **The Table E gate v1.11 attached here is
   withdrawn** *(v1.12, `-ml` v1.15 §3.2d)*: `separationZ` is **reported, not verdicted**, so this
   report renders no `sep_z` interval at all, and `_widen`'s `[-1, 1]` clamp is due with the `sep_z`
   **comparison** — exploratory, and no stage's done-condition (§3.8.1, §4 S1e Table E). v1.11 read
   the clamp as the one obstacle between S3 and a `sep_z` interval; the interval had **no producer
   and no renderer** either, which is the larger defect and is Table F's.
3. `test_metrics_agreement.py` passes over all 20 transcribed cases (§3.1 point 2). *(v1.1's
   done-condition asked for "byte-identical output" from running falkor-chat's own `test_metrics.py`
   fixtures through the copied implementation — that mechanism does not exist; §3.1 point 2 explains
   what replaced it and why.)* `scripts/refresh_golden.py --check-origins` runs clean, and
   `metrics_agreement.json` carries its `sourceGitSha` and `sourceSha256`.
4. `corpus.embeddings.json` (the 121 corpus vectors) is written into the pack and the ranking path
   is shown to reproduce identical rankings from the two fixed vector files with no live embedding
   call at all (§3.8.1) — that is what actually isolates "is my ranking code right".
5. **The harness self-check is run and its outcome recorded — it is a diagnostic, never a gate**
   *(stakeholder decision, 2026-09-02)*. The same model on the same corpus and queries is expected
   to land near the copied baseline; **a below-expectation result does not block S3.** What is
   required is that the deviation and the investigation of it are written into
   `model-bench/docs/test-reports/`, naming which of the known causes (wrong prefix, unnormalized
   vectors, truncated corpus — `-ml` §5.4) were checked and what was found. It is not a metric
   target in either direction: hybrid-ANN vs exact-vector-only differ in two directions at once, so
   a *disagreement* of either sign is uninterpretable and must not be "explained". v1.1 left this
   silent, which is a done-condition that can be argued either way at the moment it matters most.

### S4 — `guard-judge` and `nlq-generator` packs

**Create:** `packs/guard-judge-understanding/`, `packs/nlq-structured-query/`,
`modelbench/scoring/{classification,extraction}.py`, the pack-local `tools/exec.py` spec executor.

Both are single-call-per-item roles that reuse the S1 statistics unchanged; grouping them keeps the
new surface to two scorers plus one small executor. Do `guard-judge` first (no executor at all).

**Done when:** both packs run end to end against one model; **every item is stamped
`answerable: true|false`** by running its hand-written reference spec through the executor
(§3.8.3 — this tests the executor, not the model, and is the step that keeps unanswerable items out
of the accuracy denominator); the executor's **validation** half is covered too — a malformed spec,
a schema violation and a wrong answer each land in their own count, in stdlib only (§3.8.3's cost
note); the two `conflicting-facts` items score by subset containment, not set equality; `validate`
passes on both packs; and the guard-judge report shows the two class-conditional verdict metrics
side by side with **no headline number and no pooled 85-item figure anywhere in it** (§3.8.2).

### S5 — `tool-caller` pack, part 1: environment and scoring

**Create:** `packs/tool-caller-shop-assistant/{pack.json,catalog.json,tools/sim.py,tools/schemas.json,prompts/system.md}`,
`modelbench/scoring/toolcalls.py`.

Build the simulated storefront and the per-turn scorer **before** the conversation scripts, and test
both against hand-built synthetic traces (a trace where the model called nothing, emitted a
prose pseudo-call, called the wrong tool, omitted a required argument, mis-translated a boundary,
duplicated a call within a turn, re-issued one across turns, kept going after done, contradicted
the tool result, and called a tool when none was required). This is the piece where a scoring bug is
invisible and expensive, and synthetic traces are the only way to test it deterministically.

**Two of those traces are v1.27's and neither is produced by accident, which is why they are named
rather than left to "one per disposition"** *(P14-1; `-ml` §4.3.1 item 4)*. **(1) A `cap-hit` turn
with an *empty* dispatch trace** — the model emitted only undispatchable tool calls every iteration
— asserting all three consequences at once: it partitions as `no_attempt`, it **does** enter
`iteration_cap_hit_rate`, and it **does** enter the `I(t)` summary. A list of five dispositions is
not coverage over a mapping keyed on the pair `(D(t), E(t))`, which is exactly how P14-1 got in.
**(2) A `timed-out` turn beside a `no-response` turn, identical in every respect but the
mechanism**, asserting `fail` against `unrunnable` — the pair that pins `-ml` §4.3 rule 4's
discriminator, and the pair a four-member disposition set could not express at all.

**Done when:** each FR-8 count (including the restraint count and the three-way call-form partition)
has at least one synthetic trace that moves it and one that does not; the **outcome-vector
comparison the determinism probe needs** exists as a pure function over two `ConversationTrace`s and
is unit-tested on identical and one-turn-differing pairs (§3.8.4 — no new statistics, and S6 must
not be the first place it runs); denominators and `n/a` tallies
behave per `-ml` §4.2–§4.3, verified by test, including the laundering case — a model that collapses
at turn 2 must not score *better* than one that reaches turn 8; the funnel table renders; and
`report.py` renders both the per-turn-position table and the hazard curve.
**Seven items v1.27 adds to this list, gated here rather than referenced in prose** *(P14-3's own
finding — a contract owed by no done-condition stays unbuilt — applied to `-ml` §4.3.1's tests as
well as to the probe)*: **(1)** the **third leg** of §4 S2's disposition probe is written — the set
of dispositions this scorer branches on, asserted equal to the same constant transcribed from
§3.8.4's table that legs 1 and 2 use, never to either of them — **and
`test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5` is deleted in the same change**;
that tripwire reddens the moment `modelbench/scoring/` exists, which is this stage's own `Create`
line, so S5 cannot start without meeting it. **(2)** The two discriminating synthetic traces above.
**(3)** `cleanThroughTurnH`'s third state, asserted on a conversation carrying one `unrunnable`
turn at `t ≤ H`: not clean, not failed, out of the headline's denominator, into its `n/a` tally,
and out of the paired intersection for both arms. **(4)** The **cross-module union**:
`ITERATION_SUMMARY_DISPOSITIONS | ITERATION_SUMMARY_EXCLUDED == convo.TURN_DISPOSITIONS`, and the
two disjoint — binding two independently authored declarations across two modules, so a sixth
mechanism belongs to neither set and reddens before it can be silently included or dropped.
**(5)** **A distinct behavioural consequence per member**, five cases, each observable and each
different, so that moving any one member between the two sets reddens exactly one of them —
`-ml` §4.2(f) enumerates the five and this list does not restate them. **(6)** The **exactness
rule swept in both directions** over `X ≤ 200` and every `c ≤ X` (`-ml` §4.2(f)): raising each
cap-hit observation leaves the `r`-th order statistic unchanged **iff** `r ≤ X − c` — a test
asserting only the forward half would bless a bare p95 where a `>=` was owed. **(7)** The `I(t)`
summary and `Y_calls / Y` are asserted to **differ** on a run containing a non-`replied` turn; a
report that shows them equal there has substituted one for the other.
*(Items 4 and 6 need only `TURN_DISPOSITIONS` and `stats.percentile`, both of which exist; they
land here rather than in the precursor unit because their other comparand is this stage's own
scoring constant. `-ml` §4.3.1 item 9 asks for the union in the precursor unit, which had already
landed when that was written — and landing it here is the stronger placement anyway, since the two
declarations then have two authors as well as two modules.)*
Per amended FR-8(d),
one of those synthetic traces asserts that a `boundary_unit` error increments **both**
`wrong_value` and its `boundary_unit` subset, driven by the pack's declared `boundaryRule` — never
double-counted as two sibling failures, never classified by scorer heuristics.

**And a turn with no final reply is dispositioned on `TurnTrace.turnDisposition`, never on
`finalReplyText`** *(v1.25, corrected in v1.26 per plan-gate P13-1; rewritten at v1.27 — the
mapping is `-ml` §4.3 rule 4's and this bullet no longer restates it, because restating it is what
made §3.8.4's table a second home and produced P14-1)*. **Keying the rule on the reply field
instead was `fail` laundered into `n_a` in the paragraph that names laundering as the enemy.** What
S5 owes, stated as work rather than as a mapping:

- **The scorer branches on all five `turnDisposition` members and reads every scoring consequence
  out of `-ml` §4.3 rule 4** — including that a turn's admission to a conditional count is a
  function of the **pair** `(D(t), E(t))` and not of the mechanism alone, and that `no-response`
  and `server-rejected` are `unrunnable` while `timed-out` alone is `fail`.
- **`cleanThroughTurnH`'s third state** (§3.8.4): a conversation with an `unrunnable` turn at any
  `t ≤ H` is neither clean nor failed — `scoreable` is `False` for that metric, it leaves the
  headline's denominator into that metric's own `n/a` tally, and it drops out of the paired
  intersection for both arms as an `asymmetry`.
- **The two scoring constants, and they do not live in `convo.py`** *(`-ml` §4.3.1 item 8)*.
  `ITERATION_SUMMARY_DISPOSITIONS = frozenset({"replied", "cap-hit"})` and
  `ITERATION_SUMMARY_EXCLUDED = frozenset({"timed-out", "no-response", "server-rejected"})` are a
  **scoring** vocabulary and sit beside the scorer in `modelbench/scoring/toolcalls.py`;
  `TURN_DISPOSITIONS` is a **mechanism** vocabulary and stays in `convo`. Each is written out
  literally; the second is **not** derived as the first's complement, because a derived complement
  makes the union assertion below a tautology.
- **The `I(t)` summary's report surface** *(`-ml` §4.2(f), §4.3.1 item 7)*: mean and p95 over
  `ITERATION_SUMMARY_DISPOSITIONS` only, printing their own denominator inline and the excluded
  count beside it; the mean rendered `>= <value>` whenever the cap-hit count is non-zero and the
  p95 `>= <value>` whenever `r > X − c`; the p95 computed through `stats.percentile`, never
  hand-rolled. **`Y_calls / Y` and the mean of `I(t)` are both printed and never substituted for
  one another** — the first is the unrestricted *cost* figure and the second the restricted
  *behaviour* figure, and they differ on any run with a non-`replied` turn.

### S6 — `tool-caller` pack, part 2: the conversation scripts (FR-22)

**Create:** `packs/tool-caller-shop-assistant/conversations.jsonl` + `PROVENANCE.md`.

1. Reconstruct shapes A / B / C from
   `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.1 (§2.2), turn by turn.
2. Extend to **4 distinct scripts per shape, 12 in total, each run once** (§3.8.4's sizing, revised
   by the 2026-09-02 stakeholder decision) — and cover the FR-8 failure kinds §8.1's set was not
   designed to exercise: in particular "stopping when done", "final reply matches the tool result",
   and turns where **no** tool is required (the restraint count). **This is where the sampling
   decision's cost lands** — 12 scripts to author and human-verify instead of 12 replicated runs of
   4 — and it is why S6 remains the long pole. Also draft the ~20 labelled replies the
   prose-vs-native detector is scored against.
3. **Human verification of every turn's expectations** (FR-19): each `expect` block is checked by a
   person against the simulated environment's actual behavior, not against a model's output.
   `provenance.verifiedBy` is filled per conversation.
4. Declare `sampling.determinismProbeScripts` (one shape-A, one shape-B) and **run the determinism
   probe** — those two scripts a second time, once per model, in the same session, outside `n`
   (§3.8.4). Record `determinismProbe` in each run and let it set `basis`.
5. Run the known-answer validation: `qwen/qwen3-4b-2507` vs `mistralai/ministral-3-3b`, and compare
   the per-turn profile to §8.2's recorded finding.

**Done when:** step 5 is **run and its outcome recorded** in `model-bench/docs/test-reports/`;
**step 4's determinism probe has run for both models and its result is recorded, with `basis` set
from it** — an identical outcome vector on both probe scripts leaves `basis: "by-construction"` and
McNemar deciding; anything else records `basis: "assumed"`, and the report must then show the
**conservative envelope** as the decision with McNemar labelled anti-conservative (`-ml` §3.4 Rule 4).
Both outcomes satisfy the condition — what does not is the probe not having run. AC-1 holds on real
output (per-failure-kind **and** per-turn-position, with no blended headline anywhere in the
report); and the pack validates, including `H ≤ min(script length)`, the `sampling` row-count
identity and `analysisUnit` membership (§3.3).

**If the contrast does not appear**, the stage is still completable: R-3's bisect is executed and
its result recorded, and the pack ships flagged `known-answer validation: not reproduced` in every
report it generates until it is. *(v1.1 made the done-condition "step 4 reproduces the documented
contrast" while R-3 itself states the reconstruction "will not be turn-for-turn identical" and the
contrast may not appear — a gate on an empirical outcome the plan says may not occur, which under
deadline pressure is resolved by editing the reconstruction until the contrast appears. That is
fitting the instrument to the expected answer, and it is the one thing a measuring instrument must
not be built by.)*

**This is the long pole.** It is last among the data-bearing stages not because it is least
important but because it is the only one whose *scoring* correctness cannot be checked against
anything except itself — S5's synthetic traces have to exist first, or a script defect and a scorer
defect are indistinguishable.

### S7 — `chat-responder` pack

**Create:** `packs/chat-responder-grounded-answers/`, `modelbench/scoring/grounding.py`.

**Deterministic layer only (FR-21a).** No judge, no calibration set, no `judged.py` — the deferred
design stays in §3.8.5 and `-ml` §6.2 until it is funded. Ungated: the amendment removed the
decision this stage used to wait on, so S7 could now run any time after S1, and stays last only
because it is the least valuable of the five packs until the judged half exists.

**Done when:** the 30 items are human-verified and stamped with provenance; the grounding scorer
covers containment, exclusion and correct abstention; **the format scorer covers all three declared
constraints** — `maxWords`, `mustBeSingleParagraph`, `forbiddenPatterns` — read from pack data with
per-item override, each a separate count never pooled with grounding (§3.8.5; v1.1 shipped two of
FR-21a's three layers and this closes the third); the report prints `groundingRate` as
`headlineMetric` alongside the format counts and the standard latency block; and the report says in
words that reply *quality* is not measured by this pack.

### S8 — Documentation and close

`model-bench/README.md` (how to run, what a pack is, how to add one, the three non-features, the
exact-cosine scope note from §3.8.1), `model-bench/AGENTS.md` (working context: layout, live-test
convention, the FR-23 rule stated as a hard rule for future agents, the operator-attested
fingerprint fields and why), `model-bench/docs/HISTORY.md` (an entry per stage, first written at S0),
`model-bench/docs/BACKLOG.md` **re-checked and extended** — it was already seeded at S0 with the two
deferred items named in §7 (the judged reply-quality layer, FR-21a; the +22 harder retrieval
queries), so S8 adds whatever R-1's S2 probe and the S3/S6 test reports left open and removes
anything since delivered. Root `AGENTS.md` rows added in S0 are re-checked against what actually
shipped. `README.md` additionally states the two things §3.4.3 and §3.6a make into user-visible
contracts: that a `benchSchemaVersion` bump is a deliberate act with a migration decision attached,
and the closed exit-code set.

---

## 5. Test strategy

The harness is a measuring instrument, so the test strategy has an unusual centre of gravity: **the
tests that matter most are the ones that prove the instrument reports honestly when the data is
bad**, not the ones that prove it reports a number when the data is good.

### Which stage owes which test

**Read this table before using the numbered list below.** The numbering is **stable and cited by
number** from all three review documents, so the list is deliberately *not* re-ordered into stage
order; this table is the mapping instead, and it exists because all three gate passes of this
component had to derive the same split independently from §4 (engineering review Pass 3, "carried a
third time"). **§4's stage blocks are its source: where the two disagree, §4 governs and this table
is the stale one** (§7 rule 4). A stage is done when the items in its row pass **and** its own §4
done-conditions hold — the two lists overlap on purpose and neither replaces the other.

| Stage | Items it owes | Where an item splits across stages |
|---|---|---|
| **S1** — core | **1, 2, 3, 5, 6, 7b, 11b, 11c, 11d** | **7b** is a `stats.py` test over a synthetic clustered fixture and needs no pack loader — S1 done-conditions 4 and 5 already require it. **11b**'s `validate_pack` clause is S2's; at S1 those same refusals go through `metrics_from_manifest`, which raises `PackConfigError`. **12**'s `metrics`-block rule (a non-null `headlineMetric` outside `verdictMetrics`) is S1's too, at that same seam. **11c** is S1's whole and does not split: `RunResult` carries `items` and `aggregates` side by side, so the cross-check needs nothing S2 produces (impl-gate P4-4, v1.8 — S1 done-condition 10). **11d** is S1's whole for the same reason: the continuous carrier is a record shape and a renderer branch, both S1-local, and the first *run* that exercises them is S3's (v1.12, S1 done-condition 13). |
| **S2** — packs, adapter, host info, runner | **4, 10, 10b, 10c, 12, 12b, 13, 14, 15, 15b** | **10b** is v1.25's, the per-turn iteration loop, and is S2's whole. **10c** is v1.26's and **splits**: its `drive` disposition cases — **five** from v1.27, plus the precedence case — are S2's, while its three-way coverage probe reaches the S5 scorer's branch set. So the probe lands **two of its three legs** with `TURN_DISPOSITIONS` (which it did, at `d5b549d`), and **S5's *Done when* item 1 completes it** when the scorer exists — gated there, not merely referenced (§4 S5) — `drive` is S2's and a stub LLM is all it needs. **4** is `packs.content_hash`, which S1 does not have: S1 ships `PackRef` / `metrics_from_manifest` / `check_sampling_contract` only. **12**'s rule machinery — the `sampling` contract, the AST import allowlist, `replicatesPerScript > 1` — is `validate_pack`'s, tested here against fixture packs and re-run against each real pack at that pack's own stage. **12b** splits three ways: the four `basis` cases are `runner`'s and land here, the outcome-vector comparison is S5's, and "`assumed` moves the decision off McNemar" is already S1's. **13–15** are the `-m live` adapter tests S2's done-condition names. **15b** is offline (stub clock, stub LLM) and is filed in the unit block for that reason (v1.9, G3-12); its report half — the latency block and `-ml` §11.7's rendered slots — lands here rather than at S1 for the same reason `basis` does: S2 is the first stage at which a latency exists at all. S2 does **not** own **11c**; it owns the scorer contract that makes 11c's failure unreachable (§4 S2). |
| **S3** — embedder pack | **8, 11, 16, 18** | **16** is one arm of a per-pack obligation: each of S3–S7 owes the end-to-end run for the pack it builds. |
| **S4** — guard-judge, nlq-generator | **9, 16** | — |
| **S5** — tool-caller scoring | **7**, **10c** (third leg only), **12b** (outcome-vector half), **16** | S5's done-condition requires the outcome-vector comparison to exist and be unit-tested here, precisely so S6 is not the first place it runs. **10c** appears twice on purpose: its third probe leg is the half S2 cannot write, and §4 S5's *Done when* item 1 is what gates it (v1.27, P14-3). §4 S5's own list carries six further v1.27 pins — the two discriminating traces, `cleanThroughTurnH`'s third state, the cross-module union, the five per-member cases and the exactness sweep — which are done-conditions rather than numbered items here because they are scorer behaviour, not a module's test surface. |
| **S6** — tool-caller scripts | **12** (the `H ≤ min(script length)` clause), **16**, **17**, **19** | **19a** must pass before **19b** is interpreted at all. §4 S6 gates on 19b being *run and recorded*, never on the contrast appearing. |
| **S7** — chat-responder | **16** | — |
| **S8** — close | **20** | The FR-23 audit is only meaningful once every stage has shipped; §4 S8's done-condition does not currently name it, and should record it at close. |

**Unit (default suite, network-free, `pytest -q`)**

1. `fingerprint.validate()` — one test per required field, per profile: blank it, assert it is
   named. Plus the three tier cases (§3.4.2): `residentModelsAtStart: []` valid, `modelKey: ""`
   invalid, `null` invalid in either tier; and the **three `armProfile` cases** (§3.4.1) *(v1.10,
   plan-gate P4-2 — v1.9 left this item stating the retired two-kind contract)*: a `deterministic`
   record with no model fields validates and the same record with `modelKey` added fails as
   *forbidden*; a `model:embeddings` record **without** `runtimeName` validates and the same record
   **with** any of `runtimeName`/`runtimeVersion`/`temperature`/`maxTokens` fails as *forbidden*; a
   `model:chat` record missing `runtimeName` fails as *absent*. Plus the discriminators themselves:
   `armKind == "model"` is a valid membership answer after §4 S1e Table B's re-key, and a `model`
   record with no `callSurface` fails before any mapping is consulted. §4 S1 DC-6 states the same
   requirement from the done-condition side; these two must not diverge again.
2. `results.store()` refuses an invalid fingerprint; there is no bypass flag (assert the absence by
   API surface, not by comment).
3. `results.load_history()` quarantines: a hand-edited record with a missing attested field, a
   record declaring a **future** `benchSchemaVersion`, a truncated JSON file → all three appear in
   `invalid`, none in `valid`; **and a record written under an older, known `benchSchemaVersion`
   validates against its own contract and appears in `valid`** (§3.4.3 — the FR-3 case v1.1's
   "older-schema records are excluded" would have silently deleted). **(AC-2)**
4. `packs.content_hash()` — stable across path order, changes when any byte in any pack file
   changes, unchanged when `PROVENANCE.md` changes.
5. `compare_report` — version mismatch banner on differing `packVersion`; **also** on identical
   `packVersion` with differing `contentHash`; comparison still rendered in both cases. **(AC-3)**
6. `stats` — Wilson against published worked examples, at the note's pinned `_Z_95` and **not**
   `1.96`; the paired instruments against the note's `(a,b,c,d)` regression fixtures, to the
   tolerance the note states; the exact test's small-sample floor as the note tabulates it; a
   paired-difference interval containing zero produces the note's verdict-2 wording (`-ml` §3.2e,
   asserted against the note's string, never against a fragment quoted in this plan), and the
   40/40 vs 34/40 case does **not** — the regression test that pins the amended rule; the
   instruments-disagree case renders the both-components prose; and on the non-`by-construction`
   path the interval is `-ml` §3.4 Rule 4's **conservative envelope**, computed in closed form and
   taking **no seed** — asserted by construction (the function has no seed parameter) and by
   rendering the same table under twenty row permutations (`-ml` v1.11's own acceptance step 1).
   **The widening clamp is asserted on both supports** *(v1.11, §4 S1e Table E)*: a difference of
   proportions widened with `clamp=(-1.0, 1.0)` still clamps exactly as today, and an **unbounded**
   difference widened with `clamp=None` comes back with an upper bound above 1 and a point estimate
   **inside its own interval** — the assertion that reproduces the `sep_z` defect rather than the
   fix, and the one that fails if the clamp is ever re-hardcoded. **Both are written at
   `paired_cluster_bootstrap`, the public surface, never at `_widen`, which is private** *(v1.12,
   plan-gate P6-3)*. The resolving-power
   functions are computed from the run's own sampling structure and asserted never to return a
   constant, **and asserted to refuse a bare item count for a clustered pack** (S1 done-condition 5,
   the B-1 detector), and the rendered line is asserted against `-ml` §7.2's verbatim four-sentence
   string. `PairedOutcomes.from_units` raises on a duplicate unit id, and the ρ = 1 / effective-n
   identity holds (`-ml` §3.4 Rules 1 and 5). Every number in this test comes from the `-ml` note;
   none is a literal in this plan. **(AC-4)**
7. `scoring/toolcalls` — the synthetic-trace matrix from S5, one per failure kind, plus denominator
   edge cases (turn where no tool was required — the restraint count; turn where the model emitted
   nothing at all; conversation that ended early) and the **laundering test**: a trace that
   collapses at turn 2 must not out-score one that reaches turn 8 on any conditional count.
7b. The note's cluster-aware surface — a fixture where all observations within a cluster are
   identical must resolve to a far smaller effective sample than the raw count, and independent
   observations to roughly the raw count. The exact functions and their expected values are the
   note's; this test exists to prove `stats.py` implements them rather than a naive substitute.
8. `scoring/retrieval` — recall@k/MRR/precision@k/score-separation on hand-built ranked lists with
   known answers, including multi-relevant items, fewer than *k* results, and zero relevant found.
9. `scoring/extraction` — the scalar/set shape rules and numeric epsilon.
10. `convo.assemble` — the **four** `historyReplay` modes produce the documented message sequences
    (§4 S2's replay contract), `representToolSchemasEachTurn=false` really does drop the schemas
    after turn 1, and `historyTurns=k` windows the replayed prefix to the last `k`. **Four
    assertions are v1.25's and none of them passes under a scripted-`expect` replay:** (a) the
    negative — a fixture whose `observed` assistant text shares no string with any `expect` block
    asserts the observed text is present in the assembled list and no `expect` string is; (b)
    `len(observed) != turn_index` raises, on both sides; (c) `structured` emits exactly one `tool`
    message per `tool_calls` entry, including for an entry the environment could not dispatch, with
    ids matching pairwise; (d) `structured-replies-only` and `plaintext` emit **no** `tool_calls` and
    no `tool` message for a prior turn that really did call a tool — the tool-evidence axis §3.3's
    table declares, asserted as an absence rather than assumed. **(e), v1.26 (P13-9):** a prior turn
    with `turnDisposition != "replied"` is present in every mode — `content: ""` in the two
    reply-text modes, and under `structured` its iterations with **no** trailing assistant message —
    and the assembled list never has one fewer turn than the observed prefix.
10b. `convo.drive`'s per-turn loop and its dispositions (§3.8.4) — a stub LLM that returns tool
    calls twice and then text records `iterations == 3`, `turnDisposition "replied"` and the third
    response's text as `finalReplyText`; one that never stops records
    `iterations == cfg.maxIterationsPerTurn`, `"cap-hit"` and `finalReplyText is None`; one that
    terminates with `content: None` records `"replied"` and `finalReplyText == ""`, **not `None`**;
    and `cfg.maxIterationsPerTurn is None` **raises** rather than defaulting (§3.3).
    **Two timing assertions on a multi-call fixture are v1.27's** (`-ml` §11.9 ask 7, §11.10 (7d)):
    `callCount == len(chatResults) == TurnTrace.iterations` — one number, asserted rather than
    assumed, because two independently maintained counts of the same thing is how several of this
    component's defects started — and `ItemTiming.wallClockMs >= Σᵢ wallClockMsᵢ`, the difference
    being the harness's own dispatch and message assembly, which is real time the operator waits
    through and belongs inside the turn's latency (§3.6). **The emission
    form is `-ml` §4.2's predicate over the turn's dispatch trace**, asserted on the case that
    discriminates it: a turn whose only `tool_calls` were undispatchable partitions as
    `no_attempt`, never `native`. *(v1.26, P13-5: v1.25 asserted the form is "read from
    `chatResults[0]` and unchanged by what later iterations return", which pinned exactly that
    divergence as a green test.)*
10c. **The guard, and it lands before the rework unit** *(v1.26, P13-1; the dispositions widened
    and the leg count corrected at v1.27, P14-3)* — a stub LLM raising
    `LMStudioCallFailed` with no `status` at turn 3 of a 5-turn script yields **5** `TurnTrace`s
    with the third `"no-response"`; the same raising it with `status=400` yields
    `"server-rejected"`; a `LMStudioCallTimeout` yields **`"timed-out"`**; and in every case the
    script runs to completion, because `-ml` §4.1 forbids abandoning it. **A fourth case is the
    precedence one** (v1.27, P14-5): a stub whose `maxIterationsPerTurn`-th call raises records
    `"timed-out"`/`"no-response"`/`"server-rejected"` and **never `"cap-hit"`** — one more stub-LLM
    script, and the assertion that pins the exception path being tested before the cap.
    Plus the **coverage probe** of §3.8.4's table — the `Literal`'s members, `TURN_DISPOSITIONS`,
    and the S5 scorer's branch set, each asserted equal to a constant transcribed from **§3.8.4's
    table**, never to one another. **Two of its three legs land here and the third at S5**, which
    is what §4 S2's stage-split row says and what shipped at `d5b549d`; §4 S5's *Done when* item 1
    is what completes it. *(v1.26 wrote this item as though all three landed together, which
    contradicted the stage table in the same breath and left the third leg owed by no
    done-condition — the shape that stays two-legged forever.)*
11. `test_metrics_agreement.py` — all 20 transcribed cases from §3.1 point 2, reading only
    `model-bench/tests/fixtures/`, including the two `ValueError` cases. A test that skips or
    xfails any case is a failing test: the case count is the guarantee.
11b. `report.py` structural refusals — a pack fixture with `headlineMetric: null` renders both
    verdict metrics and no headline; a manifest omitting the `headlineMetric` key fails
    `validate_pack`; a metric outside `verdictMetrics` always renders with the `exploratory` label.
11c. **The `aggregates`-versus-`items` cross-check** *(new in v1.8; S1's, per impl-gate P4-4 and S1
    done-condition 10; predicate and third arm corrected in v1.9 per G3-6/G3-7)*. Two arms of ten
    items, every item `scoreable={m: False}`, each arm's stored
    aggregate declaring `BinaryMetric(m, successes=0, n=10, unit="item")`: no `0/10` row is
    rendered, both arms appear in the `INVALID RESULTS EXCLUDED` block with their declared and
    counted `n`, and the report is still produced rather than raising. **The fixture pins a
    `PackRef.role` whose `roles.unit_kind` is `item`** (`guard-judge`, `nlq-generator` or
    `chat-responder`), without which the `unit="item"` metric is outside the selector and the test
    passes while testing nothing. A **third arm** carries an item with `scoreable={m: True}` and no
    `counts` entry: `IncompleteItemRecord` must be caught and reported as a mismatch naming that
    item and metric, **never** escape as a traceback. A **fourth arm** carries a **pooled** metric
    (`unit="turn"` under an `item`-unit role) whose declared `n` disagrees with the sum of its
    `"<metric>#denominator"` contributions across the arm's items, and is excluded and named the same
    way *(v1.10, plan-gate P4-8 — the assertion that makes DC-10's pooled arithmetic real rather than
    declarative; without it the selector silently skips the case it was widened to cover, which is
    DC-5(c)'s failure shape a third time)*. Plus the positive case — an
    arm whose declared `n` matches its scoreable-item count renders normally, and a pooled metric
    whose declared `n` matches its summed contributions renders normally — so the check is shown
    to discriminate rather than to exclude everything.
11d. **The continuous carrier** *(new in v1.12; S1's, per S1 done-condition 13 and `-ml` v1.15
    §3.2d)*. `scored_outcome` on a metric that lives in `measures` **raises** `MetricKindError`,
    asserted from both maps; a measure of `0.0` on a `scoreable: True` item round-trips through
    `to_dict`/`from_dict` **as `0.0`**, while the same item with the key missing raises
    `IncompleteItemRecord`; a non-finite measure and a name present in both maps are both refused at
    construction; and over two arms where one unit is scoreable in exactly one of them, `diffs` is
    `n_units − 1` long **and** the dropped unit shows in the `-ml` §4.3 asymmetry tally — **the two
    asserted together, because either alone passes on a silent drop.** Plus the aggregate half:
    `named_metrics()` returns `separationRaw` and `separationZ`, and the Arms table renders a
    `DistributionSummary` row **without reading `mean`**, which the shipped `else` branch does.
    **And the family half** *(v1.13)*: over a fixture declaring a **mixed** `verdictMetrics` family,
    no member is verdicted, each is named with its kind, every member's number still prints as
    exploratory — **at the four sites §3.3 (iv) names**: the widened Exploratory filter, the
    headline block, the suppressed `Family-wise error control` block and the decision cell that
    goes with it *(v1.15)*, so the test is written against the plan rather than against whatever the
    implementer chose (v1.14) — both arms render, and the `INVALID RESULTS EXCLUDED` block is
    **empty** — the two
    negative assertions being the ones that fail if the refusal is built as a DC-10 exclusion
    (§3.3 (iv), S1 done-condition 13(e)). **Two of the four are asserted on an absence** and a
    second fixture carries the **all-continuous `k = 2`** family, which reaches the same shipped
    line by the other condition (S1 done-condition 13(e) states both — v1.15).
    **And the stored half** *(v1.14)*: S1 done-condition 13(f)'s four assertions — a
    `DistributionSummary` round-tripping **as its own type**, an unknown `"type"` tag raising rather
    than decoding to a `dict`, `support` round-tripping on both continuous types with an absent key
    raising, and `_index_row`'s metrics cell rendering a distribution without reading `mean` — over
    hand-built dicts and one `RunResult` fixture, with no pack loader and no LM Studio.

12. **Pack integrity, per pack:** unique ids, required fields, per-item provenance present (with
    `originGitSha`), paraphrase rule for retrieval-style packs, every `expect` block referring to a
    tool the pack's own `schemas.json` declares, the `metrics` block well-formed (§3.3 — including
    a non-null `headlineMetric` that is not a `verdictMetrics` member being rejected; and **a
    `verdictMetrics` member whose declared `unit` is finer than `roles.unit_kind(pack.role)` being
    rejected**, with a pack declaring the same metric outside `verdictMetrics` **accepted** — v1.10,
    plan-gate P4-8: a pooled metric cannot receive a verdict because `-ml` §4.4 forbids the interval
    a verdict needs, and it stays fully reportable as exploratory; plus a metric name containing `#`
    being rejected, which is what keeps DC-10's reserved `"<metric>#denominator"` key from
    colliding), `H ≤ min(script length)` for the tool-caller pack, **`replicatesPerScript > 1` rejected while only the
    one-level `cluster_bootstrap` exists** (`-ml` §3.4 Rule 6), the **`sampling` contract** (§3.3 —
    `analysisUnit == pairingKey[0]`; the row-count identity over `analysisUnit`'s values, with a
    fixture declaring `analysisUnit: "conversationId"` under `scripts: 12` rejected for having 48
    distinct values where 12 are required; and **route (iii)**, `pairingKey[0] ==
    roles.analysis_unit_field(role)`, swept over every member of `roles.ROLES` with a borrowed unit
    field and rejected in each, plus the computed
    `set(ANALYSIS_UNIT_FIELD_BY_ROLE) == set(roles.ROLES)` — v1.25), the **`maxIterationsPerTurn`
    role rule** (v1.26 — a `tool-caller` manifest without it rejected and every other role's
    manifest with it rejected, swept per role with the same computed completeness assertion over
    `MULTI_CALL_TURN_BY_ROLE`), and `validate_pack`'s AST import check rejecting a pack module that
    imports outside stdlib + `modelbench.tooling`.
12b. **The determinism probe's wiring** (§3.8.4) — the outcome-vector comparison is exact on
    identical traces and localises the first differing turn; and `runner` sets `basis` correctly in
    all four cases: probe ran and identical → `"by-construction"`; probe ran and differed →
    `"assumed"`; **probe did not run → `"assumed"`** (the fail-safe, asserted explicitly); and
    `replicatesPerScript > 1` → `"assumed"` regardless. A `basis` of `"assumed"` must be shown to
    move the decision off McNemar and onto **the conservative envelope** (`-ml` §3.4 Rule 4), with
    McNemar's p still printed under its anti-conservative label *(v1.10: the instrument is an
    envelope of two arms and naming one of them was the note's own v1.11 correction)*.
15b. **The runner's timing discipline** — offline, stub clock and stub LLM, no network.
    *(Numbered `15b` since v1.8 and **kept** at that number because it is cited by number from three
    documents — this plan gate's review, the method note (`-ml` §11.9) and the coordination ledger;
    **filed here, at the end of the unit block, from v1.9** — G3-12. v1.9 said "three reviews", which
    is wrong: the other `15b` hits under `docs/` belong to `kaizen-agent-ontology`, a different
    component's test — plan-gate P4-14. Under the `-m live` heading a `live`
    marker gets applied by adjacency, and §2.4's `addopts = '-ra -m "not live"'` would then remove
    from the default suite the two assertions this list calls load-bearing.)* The warm-up is issued
    exactly once per arm, **on the surface the arm's profile names** (§3.4.4a), and produces no
    `ItemResult`; a stubbed 21 s first response with sub-second successors completes under the two
    budgets and fails under a single warm-sized one; `coldLoadSeconds` present when start-residency
    was empty and **absent, not `0`,** otherwise; an item preceded by a not-resident snapshot has
    **`latencyMs` withheld while `ttftMs`, prefill and `tokensPerSecond` are kept** — the assertion
    that pins `-ml` §11.4's measured ruling, and the one a conservative implementer will get
    backwards — and is **still scored**; an item whose `unexplainedMs` crosses §11.5.1's threshold
    is withheld identically and counted under the same cause, while one below it is not; a call
    hitting `requestTimeoutSeconds` is scored `fail` with **no timing figure but an `ItemTiming`
    whose only populated field is `withheldFor: "timeout"`**, the run continuing, and
    its count carried separately from the load-contamination count;
    **the units are asserted against a stubbed `stats`** — `{"time_to_first_token": 0.111,
    "generation_time": 0.954}` gives `ttftMs == 111.0`, `generationMs == 954.0` and an
    `unexplainedMs` of `wallClockMs − 1065.0`, so a seconds value reaching a millisecond field fails
    here rather than silently withholding every turn of the tool-caller pack (v1.10, plan-gate P4-1;
    a unit assertion on a stub costs nothing and is the only thing that catches this class);
    **a scored call that returns no response without timing out** — a stubbed HTTP 500 — is scored
    **`unrunnable`** (v1.27, `-ml` §4.3 rule 4 — an unattributable channel failure, where the
    timeout above scores `fail`; **the two assertions sit side by side here on purpose**, since the
    only thing separating them is the mechanism), carries no timing figure and an `ItemTiming`
    whose only populated field is
    **`withheldFor: "no_response"`** — the *timing* answer, which is unchanged and is deliberately
    not the same answer as the scoring one — does **not** re-probe or exit `3`, and lands in
    `latencyWithheldForNoResponse` — **the same counter the timeout landed in, from a different item
    state**, which is the assertion that pins v1.11's three-states-two-counts mapping (§4 S2 (vi),
    plan-gate P5-6) — with invariants (iii) and (iv) asserted to still hold on that run
    (plan-gate P4-7);
    **the three sibling figures share one count and lose a *call* together** — a call carrying a
    `stats` object whose `promptTokens` is absent or `≤ 0` is absent from `statsCoveredCount`
    **and** from all three medians, **and the run does not raise**; the same run's (iv) bound holds
    as the inequality it now is (v1.11, plan-gate P5-5, `-ml` §11.10 test 7; **per call from
    v1.27**, and case (c) below is the multi-call fixture that distinguishes the two units);
    **a chat response carrying no `stats` at all constructs a `ChatResult` whose `ttftMs`,
    `generationMs` and `tokensPerSecond` are `None` — not `0` — and raises nothing**, the assertion
    that keeps rule (iv-a)'s `statsCoveredCount == 0` reachable on the chat surface (v1.11,
    plan-gate P5-8);
    **a withheld item's `timing.wallClockMs` is still readable** while its `latencyMs` is `None`, and
    `censoringExact` is asserted on **all** of `-ml` §11.10 test 7b's branches — **false** where a
    fast call was withheld on its gap, **true** where only the slowest were withheld, **true** on a
    run whose only withholding is a **timeout**, with no withheld wall clock read at all, and
    **false** on a run whose only withholding is a **non-timeout no-response** (`-ml` v1.14 §11.5.1;
    the last two are v1.11's, and they are the branches v1.10's merged `withheldFor` could not
    express — plan-gate P5-6); and **a clean cold run withholds nothing**
    — `latencyTimedCount == latencyItemCount` — which is the assertion that pins G3-4's baseline fix
    and would have gone green on the wrong mechanism under v1.8's wording.
    **Three multi-call cases are v1.27's, and every timing fixture above is single-call, so the
    regression they catch is invisible without them** *(`-ml` §11.9 ask 7)*: **(a)** a warm
    3-iteration turn, and one at the cap of 8, report **no** model-load contamination and **keep**
    their `latencyMs` — the regression a per-call reading of §11.5.1's gap produces, where the
    detector fires on iteration count rather than on a load; **(b)** a 3-iteration turn one of
    whose calls carries a 3 485.6 ms gap **is** withheld and counted under model load, while the
    other two calls' sibling figures are **kept**; **(c)** a 3-iteration turn one of whose calls
    has no `stats` has `unexplainedMs is None`, is **not** withheld, and contributes its two
    readable calls to `statsCoveredCount` — the partial-sum refusal and the per-call exclusion in
    one fixture, which are the two rules an implementer is most likely to apply at the wrong unit.
    The rendered figures,
    their floors and their refusal cases are asserted against `-ml` §11.7's slots and §11.10's
    fixtures, never against a string written in this plan.

**Integration (`-m live`, opt-in, real LM Studio)**

13. `catalog()` returns installed models with `quantization`/`max_context_length` populated.
14. `chat()` surfaces `stats.time_to_first_token` and a native `tool_calls` array on a
    tool-capable model, and `toolCallForm` correctly distinguishes native from prose.
15. **JIT residency round-trip** *(rewritten in v1.8 — there is no `load`/`unload` to round-trip)*:
    against a model `residency()` reports as not resident, one `warm_up()` call returns and
    `residency()` then names it with a non-`"not-loaded"` `state`; `coldLoadSeconds` is recorded and
    is **materially above the same model's warm-call latency** — asserted as a magnitude, never
    against a figure: the two cold loads measured on this box differ by nearly 6× between models
    (§2.5, `-ml` §11.4), so any pinned number would be false for the next model. The unload half is **not** testable and
    is not asserted — nothing on either HTTP surface unloads (§3.4.4a) — so the honest complement is
    the second run: repeating `warm_up()` against the now-resident model records **no**
    `coldLoadSeconds`.
*(**15b** was listed here at v1.8. It is offline and now appears at the end of the unit block above,
at the same number — G3-12.)*

16. One full run per pack, end to end, producing a stored, valid result.

**Acceptance (human-run, once, recorded in `docs/test-reports/`)**

17. **AC-1** on real output: a tool-caller report shows per-failure-kind **and** per-turn-position,
    and no blended "tool-calling accuracy" figure appears anywhere in it.
18. **AC-5**: an embedding report shows the keyword-only arm and the model's output dimension.
19. **The instrument-validation pair.** (a) **Negative control — two *independent* runs of the same
    model** in the two arms, on the tool-caller pack (not two copies of one record: that is S1's
    smoke check and cannot fail). It must report *not distinguishable*, with discordant counts
    roughly equal and the difference interval centred on zero — the note names this the highest-value single test in the harness, because it
    catches a whole class of bugs that would otherwise present as plausible model differences
    (`-ml` §9). (b) **Known-answer:** `qwen/qwen3-4b-2507` vs `mistralai/ministral-3-3b` on the same
    pack reproduces the per-turn contrast recorded in
    `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.2. Together these are the only
    tests that distinguish "the harness works" from "the harness produces plausible-looking
    numbers"; (a) must pass before (b) is interpreted at all.
20. **The FR-23 audit:** `grep` the shipped tree for any path reference outside `model-bench/` in
    runtime code (`scripts/refresh_golden.py` is the one permitted exception and is not on any run
    path), and confirm `model-bench` runs correctly with `falkor-chat/` renamed away.

If this plan is executed by `tdd-engineer`, the red→green sequence is **per stage, not across the
whole list**: take the stage's row from the table above, drive its items in numeric order, then
close the stage against its §4 done-conditions. Items **13, 14, 15 and 16** follow the implementation
they cover rather than driving it; **`15b` drives it** — it is offline, it is the acceptance surface
for the runner's timing design, and it is written first. *(v1.10, plan-gate P4-12: v1.9 moved `15b`
into the unit block and deleted the closing carve-out that had said exactly this, leaving a range —
"items 13–16" — that `15b` falls inside by number, so the one sentence protecting it excluded it. The
range is now an explicit list, which is what a number kept for citation stability costs.)*

---

## 6. Risks, open questions, and requirement frictions

**R-1 — Two FR-7 fingerprint fields have no programmatic source (medium; resolved in design, worth
the stakeholder knowing).** `lms version` yields only a CLI commit hash, and neither the REST API
nor `lms load` exposes the **KV-cache setting** on this build (§2.3, live-probed). `lmStudioAppVersion`
and `kvCacheSetting` are therefore **operator-attested** in `host.json` (§3.4), not measured. FR-7's
invalidity rule still holds mechanically — a missing value is refused — but a *wrong* attested value
is undetectable. Mitigations built in: the staleness trip-wire (§3.4 point 3) catches the common
failure of attesting once and never revisiting; `runtime.version` is auto-captured from every call
and is arguably the more reproducibility-relevant version anyway. **The probe is in S2's
done-condition**, not left as prose here: read `GET /api/v0/models` for a *loaded* model, record the
outcome in `model-bench/docs/HISTORY.md`, and if the load configuration is exposed, move
`kvCacheSetting` from attested to auto-captured in the same change. A mitigation nobody is gated on
executing is a mitigation nobody executes.

*(v1.8 — this risk is **unchanged in substance and narrowed in one mitigation**. Dropping the `lms`
CLI (§3.4.4a) does not touch the claim: neither field had a programmatic source through the CLI
either, and `lms version`'s CLI commit was never the app version. What does change is the staleness
trip-wire, which loses `lmsCliCommit` as a comparand and now fires on a runtime or capture-surface
change only (§3.4.5 point 3). The residual this leaves is specific and stated rather than implied
away: **an LM Studio app update that changes the KV-cache default while keeping the same inference
runtime version goes undetected.** The probe above is what would retire the whole risk; until it
runs, this is the gap.)*

*(v1.9 — a second, larger residual, and it follows from §3.4.4a's `callSurface` discriminator rather
than from anything about attestation. **On a `model:embeddings` arm the trip-wire loses its two runtime
comparands**: `runtimeName`/`runtimeVersion` come only from a chat response and an embeddings run
never makes one, leaving `residencySource` — which **is** compared, and exits `5` on a difference
like any other comparand (§3.4.5 point 3, v1.25), but which takes one value today, so the *runtime*
half of the check is genuinely unavailable there. Obtaining a runtime would mean JIT-loading a chat
model beside the embedder on a 16 GB box — a whole model load a side, and an eviction of the model
under test — buying a staleness check by corrupting the measurement it protects. So an embedding run
trusts `host.json` on everything the runtime pair would have caught, and that is stated here rather
than left for a reader to infer from the absence of a failure.)*

**R-2 — FR-11's "peak RAM at the measured settings" is best-effort, not exact (medium).** There is
no RAM endpoint. What was available through the CLI: `lms ps --json`'s loaded-weights size and
`lms load --estimate-only`'s pre-load estimate; what remains after §3.4.4a is Windows-side process
working set sampled through `powershell.exe`. **This risk got worse in v1.8 and the plan says so:**
`modelSizeBytes` has no source at all now — `/api/v0/models` carries no size — so it is recorded
absent, never `0`, and FR-11's RAM answer rests on the single most fragile measurement in the tool.
The reversal is cheap and named, not hypothetical: if the size figure turns out to matter, the
options are the loaded entry's own keys (whatever S2's R-1 probe finds on a *loaded* model) or a
deliberate, documented re-admission of one CLI call for that one field — not a silent default. The
design still
records what it has as separately named fields rather than one authoritative "peak RAM" (§3.6), and
`README.md` states the method. A cross-WSL `powershell.exe` sample is also the most fragile thing in
the tool; it degrades to "not captured" without failing the run — **but note that makes it the one
FR-7-adjacent field that is not enforceable**, which is why it lives in FR-11's speed result rather
than in the fingerprint. It is also the most *invasive*: measured at 0.18–0.54 s per invocation
across the WSL boundary against ~1.3 s per turn, so §3.6 confines sampling to the gaps between
turns and stamps every sample with a timestamp. An implementer who "simplifies" this into an
interval timer silently inflates the headline latency of every run.

**R-3 — Reconstructed conversation scripts are a reconstruction (medium).** FR-22's asset is being
rebuilt from a prose description (§2.2) because the originals were never committed. The reconstruction
will not be turn-for-turn identical to what produced §8.2's numbers, so the known-answer validation
(§5 test 19) is a **qualitative** check — "does the same contrast appear" — not a numeric
reproduction. If the contrast fails to appear, the ambiguity between "the harness is wrong" and "the
scripts differ" is real; the fallback is to bisect on prompt-assembly settings, which is exactly the
FR-9a side benefit the requirements anticipate. **v1.25 makes that bisect concrete, and narrows this
risk; v1.26 corrects what it may conclude.** The pack now starts at falkor-chat's **replay policy**
rather than reaching for it — `historyReplay: "structured-replies-only"`, native roles carrying real
reply text and no tool evidence (§3.3, §3.8.4) — where v1.1 through v1.24 declared `structured`.
The bisect moves **one axis at a time**, which is what makes it inferential: `structured` moves the
**tool-evidence** axis (add scaffolding, keep ownership); `plaintext` moves the **ownership** axis
(same absent evidence, quoted rather than owned). A contrast that appears at one value and not
another localises the cause to the axis that moved.

**Three named prompt-shape differences remain even at the default value, and they are candidate
causes on equal footing with the scripts** *(v1.26, P13-3)*: the executor speaker-prefixes every
replayed turn, appends a per-assembly `CONTEXT:\n<json>` block, and merges consecutive same-role
turns. §3.8.4 judges each immaterial to §4.1's mechanism and says why; if the contrast does not
appear, that judgement is a hypothesis to test, not a premise — so a non-reproduction is **not**
attributed to the scripts by elimination while these three stand.

**What the bisect may not conclude** *(v1.26, P13-4)*: nothing about the *expected* result at
`structured`. v1.25 predicted onset would be absent there, equating native tool scaffolding with the
U37/U38/U39 breadcrumb. That equation is deleted — the breadcrumb was a free-text suffix, not native
scaffolding, and it did **not** reduce fabrication (live-verified 2/2, reverted as a severity
increase: `salesperson-tool-reliability-impl.md` MAJOR 1). The honest statement is that the effect
of adding tool evidence at this replay shape is **unknown**, the one adjacent probe found no
reduction, and that is precisely what makes `structured` worth a bisect rung rather than a
prediction. What no rung can separate is "these scripts differ from §8.1's" from "this model changed
since 2026-08-29", and nothing in this harness can.

**R-4 — `chat-responder` measures grounding, not quality (resolved by FR-21a; residual is a
reader-expectation risk).** Raised as a funding decision — 30 golden items plus 30 calibration
items and a judge harness, against a judge whose recomputed relevance agreement is near-worthless
once chance agreement is accounted for (the figures are `-ml` §6.1's; §2.1's inventory row is where
this plan repeats one, attributed — *v1.9, n-ML-9: this risk repeated it a second time and the
number is withdrawn rather than re-stated*) — and **settled: FR-21a ships the deterministic layer
only** and defers judged
reply quality. §3.8.5 builds that and records the deferred design. The residual risk is that a
reader six months from now sees a `chat-responder` score and reads it as "reply quality", which it
is not; the mitigation is in the pack's own report text (S7's done-condition) and in
`docs/BACKLOG.md`, which carries the deferred judged layer as an open item rather than letting it
disappear.

**R-5 — Copied golden data can silently go stale relative to its origin (low, accepted).** By
design (§3.1 point 1) the copies freeze. The accepted consequence is that a fix made to
falkor-chat's golden set — a mislabelled item, say — does not reach `model-bench` until someone runs
`refresh_golden.py`, which forces a pack version bump and thus flags every prior result as
version-mismatched (AC-3). That is the correct trade: comparability over time is the tool's purpose,
and a version bump is loud. `PROVENANCE.md` records the origin commit so a diff is always possible.

**R-6 — Pack Python is a plugin seam (low).** Packs execute code (`tools/sim.py`, `tools/exec.py`).
This tool runs locally, on packs written in this repo, started by a person — so the threat model is
"a mistake", not "an attacker". Mitigations: pack modules may import only stdlib and
`modelbench.tooling`; pack code is inside the content hash; `validate` reports a pack whose module
imports outside that set. A stricter sandbox would be disproportionate.

**R-7 — Item-level pairing across sessions is weaker than it looks (low, disclosed in output).** The
pack pins items and order, so a cross-session paired comparison is arithmetically valid, but the box
is shared and its state is not controlled between sessions. §3.7's report label
(`paired, same session` / `paired, cross-session`) is the disclosure; FR-17's reference arm is the
remedy the tool *offers*, and the report says so wherever it prints a cross-session comparison.

**R-12 — The tool-caller pack's *n* is small, and the report must say so rather than imply
otherwise (medium, accepted by decision).** The 2026-09-02 sampling decision buys 12 independent
conversations where v1.1 bought 48 clustered ones (§3.8.4). The precision is not worse — `-ml` §4.5
put the old design's effective *n* near 12 anyway — but it is now *visible*, and the
resolving-power line will print a large number. Accepted deliberately: an honest wide interval is
worth more than a narrow one that a design effect would have had to walk back. The mitigations are
all in output rather than in design — the resolving-power line computed from the real structure
(§3.9 point 2), `replicatesPerScript` and `temperature` printed beside every conversation-level *n*,
and the report naming the magnitudes it can and cannot resolve. The reversal trigger is explicit:
if a real comparison lands inside the unresolvable band and the answer actually matters, the fix is
**more distinct scripts** (a pack version bump, S6's authoring cost again) — never more replicates
at temperature 0, which is what produced the problem.

**R-13 — the latency percentile: estimator, denominator and floors. CLOSED by the note
(2026-09-03).** *(Raised in v1.5 from the S1 gate's open question 2; a second input added at v1.8
and a third by the plan gate's G3-11; all three answered together in `-ml` **§11**, which is now the
single source for this and is cited rather than summarised.)* The risk was real and is worth keeping
on the record because its shape recurs: S1 shipped **two** copies of a nearest-rank `_percentile`
(`stats.py`, `results.py`), invisible at the bootstrap's draw count and decisive over a handful of
items, and FR-11's p50/p95 are compared across runs stored months apart — so the definition freezes
the instant the first latency figure is stored. What the note settles, and what this plan therefore
carries as design rather than as an open question:

- the estimator, and that there is **one** implementation, not one per module (`-ml` §11.2, §11.10);
- the denominator, and **which fields the withholding governs** — `latencyMs` alone, with the
  `stats`-derived figures kept under their own coverage, settled by measurement rather than by a
  conservative default (`-ml` §11.4), plus the in-call reload detector built from the same gap
  (§11.5.1) — which is §3.6's guard and G3-3's scope;
- **two floors**, one renaming the tail figure and one refusing a figure outright, both evaluated at
  the value so that `index.csv` needs no qualifier (`-ml` §11.3, §11.6) — which is why §3.5's
  latency cells are empty exactly when the record's fields are `None`;
- the published block, as a slot grammar this plan cites and never restates (`-ml` §11.7).

**What the closure obliges the plan to carry** is in §3.5 (`index.csv` columns), §3.6 (the guard,
the timeout disposition, `unexplainedMs`) and §4 S2 (the `LatencyBlock` and its rules) — **and, from
v1.10, an edit to shipped S1 code that v1.9's own prose implied and no table named**: both
`_percentile` copies are `int(round(p/100·(X−1)))`, the estimator `-ml` §11.2 explicitly rejects, and
`results.py` computes `index.csv`'s p50/p95 inline from `run.items`, bypassing both of §11's floors.
**§4 S1e Table C enumerates it** (plan-gate P4-2's second instance) — plus
the one field the note asked this plan to name: **`latencyMsMax`**, the home for the tail figure on
any run where `-ml` §11.3's identity floor **renames** the p95 rather than refusing it. *(v1.27
withdraws v1.9's justification for it — "the tail figure is the sample maximum at every tool-caller
run's sample size" — rather than softening it. That substituted the tool-caller pack's
analysis-unit count for its item count, which is this component's signature defect and which
`-ml` v1.20 corrects: the rename is reachable only at `Y ≤ 21`, the smallest declared `Y` is 30,
and `latencyMsMax` is therefore `None` on every run of every pack this plan declares. The field
still earns its home on the narrower argument §3.5 now carries — a pack with `Y ≤ 21` is buildable
and the `p95` label would be false without it. The same sentence is withdrawn at §3.5; `-ml`
§4.3.1 item 10 flags it as the fourth instance of the substitution.)*

**The residual, and it is a reader risk rather than a method one:** a run can now legitimately print
no latency figure at all. That is the design working — an absent number beats a number whose label
has drifted off it — but it is surprising the first time, so the refusal clause says why in the
report itself rather than leaving a blank the reader has to interpret.

**R-8 — Two LM Studio catalog ids for the same weights (low, designed for).** Confirmed on this box
(§2.3). The fingerprint records the literal key and never normalizes, so two runs of "the same
model" under different ids compare as two models — visibly, with both ids printed. That is the
correct conservative behavior; a `notes` field lets a human record that they are the same weights.

**R-14 — The harness can no longer force a cold state, and JIT can reload underneath it (medium,
mitigated in output).** *(New in v1.8.)* Nothing on either HTTP surface unloads a model (§2.5), so
two things follow. **(i)** `coldLoadSeconds` is only measurable when the operator happens to start a
run against a non-resident model; otherwise it is absent, and FR-11's cold-load figure becomes an
*opportunistic* measurement rather than a guaranteed one. Producing it deliberately is now a human
action — stop the model in the LM Studio GUI, or wait out its TTL — documented in `README.md`, not a
harness capability. **(ii)** A JIT reload can land *inside* a run: TTL expiry between items, or the
operator loading something else. §3.6's between-item residency probe is the mitigation and it is
honest rather than complete — it detects a reload that happened before an item and marks that item's
latency absent, but it cannot detect one that begins and ends inside a single timed call. The
residual corrupts **timing only** — the scored outcome of a slow call is the same scored outcome —
and it is now *detectable* rather than merely bounded: the wall-clock-minus-`stats` gap separates a
load from a warm call by more than two orders of magnitude (`-ml` §11.5.1), which is a far sharper
discriminator than the raw latency, whose cold values differ by ~6× between models on this box. The accepted position is that
the tool reports fewer latency samples honestly rather than all of them optimistically, which is why
`-ml` §11.7's block prints the coverage and its cause split instead of quietly shrinking the sample.

*(v1.10 — the in-call residual has a detector **and a rule**, and this entry now says the same thing
§3.6 does.* `unexplainedMs` *is computed **per call and summed over the item** (v1.27, `-ml`
§11.5.1 under the loop — at `callCount == 1` that is v1.9's expression unchanged), stored per item,
its maximum reported, and it
**withholds `latencyMs` above the threshold `-ml` §11.5.1 sets**, counting the item under §11.7's
**model-load** cause rather than opening a third — an in-call reload* is *a model load, so a third
entry would name a detector rather than a cause. **v1.9's parenthetical here said the opposite** —
"no cut-off is set and nothing is withheld on it" — while §3.6 already carried the withholding: the
note does set the threshold (§11.5.1) and asks for the withholding as a plan dependency (§11.9 item
2a), and this entry was swept from an earlier draft of the same revision. Plan-gate P4-4, and it
matters because R-14 is where a reader consults the residual when the guard behaves unexpectedly; the
two readings differ in `latencyTimedCount`, in the withheld counts, and therefore in whether §11.6's
level floor refuses a figure at all. What is genuinely R-14's and survives: the threshold is a
**starting value with a named basis, not a derived constant**, and the first real pack run is where
it is re-checked — which is why `unexplainedMs` is stored even below it. The note revised that basis
at v1.12, withdrawing the "~3.5× below the smallest cold load" margin after plan-gate P4-11 showed
the two cold loads differ in model, quantization and route and so bound no load from below; the
1 000 ms value stands on the asymmetry of the detector's two errors. That reasoning is the note's and
is cited, never restated. **And the cost of a withheld item is real**: under §11.6's level floor a
run can withhold a small number of items and print no latency figure at all — the design working as
intended, and it means an operator who leaves a TTL short enough to reload mid-run does not get a
degraded latency number, they get none.)*

**R-15 — The fingerprint now depends on a vendor-proprietary endpoint (low, accepted with a named
reversal).** *(New in v1.8.)* `GET /api/v0/models` is LM Studio's own API, not the OpenAI-compatible
surface, and §3.4.4a makes it the only source for most of FR-7's model identity. Accepted for the
reasons argued there — the chat path has been on `/api/v0` since v1.1 for the same reason, so this
consolidates onto one proprietary surface rather than adding a second — and because the alternative
costs nine of §3.4.2's model fields. The exposure is real: an LM Studio release that changes or removes
`/api/v0/models` stops the tool, loudly (exit `3`, no record written), and stored records remain
readable because `residencySource` says what produced them. The reversal trigger is a **second
provider**, and its shape is decided in advance rather than improvised: the required-field set
becomes a function of provider as well as `armKind` (§3.4.1's discriminator pattern, applied again —
and v1.9 applies exactly that pattern one level earlier, for `callSurface`, so the reversal is no
longer hypothetical machinery), never a set of nullable fields.

**Three costs that reversal carries, named so the pattern is not mistaken for a free one** *(v1.9,
G3-8)*. **(i) The discriminator must be *declared*, not observed.** `residencySource` is captured —
it is the token the probe answered on — so it cannot select the contract a record is validated
against, because that validation would depend on having already probed the provider. A second
provider needs a **declared** `provider` field (from `host.json` or the arm), with `residencySource`
staying the observation; `callSurface` is already built that way, from the pack's
`environment.requires` (§3.4.4a). **(ii) Provider B's required set is a collapse, not a variation.**
Eight of the nine fields §3.4.4a costs out — `arch`, `quantization`, `compatibilityType`,
`maxContextLength`, `loadedContextLength`, `modelType`, `modelCapabilities`, `modelPublisher` — have
no analogue on a hosted API. So the reversal does **not** preserve fingerprint strength across
providers; it **segregates** strongly-fingerprinted records from weakly-fingerprinted ones. That is
defensible and it is a different claim from "the reversal is designed for". **(iii) Nothing guards
the comparison yet.** `compare_report` would place a provider-A and a provider-B arm in one table
with no banner and no refusal, while `-ml`'s instruments assume the arms differ only in the model.
§3.4.3's `SCHEMA VERSIONS IN THIS COMPARISON` line is the precedent sitting one paragraph away. No
design work is owed now, because the trigger has not fired; what is owed is the correction that
**the reversal's cost is a report-surface change, not merely a schema one**.

### Requirement defects raised by this design pass — all amended and closed

Three requirements were unbuildable as written. All three amendments were accepted on 2026-09-02
and are in `docs/requirements/small-model-benchmarking.md` (commit `afe4aef`). They are kept here
because the *reason* each was wrong is what an implementer needs when they meet the code — the
requirement now states the rule, not why the earlier one failed.

- **R-9 — FR-15/AC-4's marginal-overlap rule could never fire. Amended.** At this tool's sample
  sizes and baselines, **no result whatsoever separates two marginal Wilson intervals** (`-ml` §3.1
  shows the region is empty), and the rule discarded exactly the covariance FR-16's paired design
  pays for. Worked case: 40/40 vs 34/40, perfectly nested, marginal intervals overlap → old rule
  prints not-distinguishable, while both paired instruments separate the two arms. *(`-ml` §3.1
  carries the case with its exact p, difference and interval; v1.7 withdraws this bullet's copy of
  those three figures rather than re-deriving them — the numbers are the note's and the reason the
  requirement was wrong survives without them.)* FR-15 and AC-4 now read the interval as the one on
  the **paired difference**; §3.9 point 1 is the build. The superseded overlap check remains a
  printed diagnostic.
- **R-10 — FR-8(d)'s three-way split was not a partition. Amended.** *boundary/unit translation* is
  a **subset** of *wrong value*, not a sibling; coding them disjoint forces double-counting or an
  arbitrary priority rule, and two runs under different priority rules stop being comparable
  (`-ml` §8). FR-8(d) now states the nesting, and the per-argument boundary rule comes from pack
  data (`boundaryRule` in the pack's tool schemas) rather than a regex in the scorer — §3.8.4.
- **R-11 — FR-9/FR-22 did not require distinct scripts per shape. Added as FR-22a.** Replicates of
  one script yield an interval describing "this script again", not "a script of this kind"
  (`-ml` §4.5). §3.8.4's sizing satisfies it — **12 distinct scripts, 4 per shape, one run each**,
  per the 2026-09-02 stakeholder decision. *(FR-22a's own text still cites the plan's earlier
  4 × 4 × 3 sizing as an illustration; the requirement it states — several distinct scripts per
  shape — is satisfied a fortiori, and the illustrative clause is `tico`'s to refresh, not this
  plan's to contradict silently. Flagged, not amended.)*

**Nothing in this plan is blocked on a stakeholder answer.** The gate's three open questions were
all closed on 2026-09-02 — tool-caller sampling (§3.8.4), `guard-judge`'s absent headline (§3.8.2),
and what S3 counts as done when the self-check fires (S3 done-condition 5). Two items remain, both
verify-during-implementation and both now gated in a done-condition rather than left as prose:
R-1's S2 probe of whether a **loaded** model's `/api/v0/models` entry exposes the KV-cache setting
(v1.8 — same probe, new instrument), and S6's
known-answer validation, which is recorded either way (S6, M-13's exit).

**One thing this plan no longer claims.** §3.1 point 2's duplication mitigation is weaker than v1.1
stated it, and it says so in its own text rather than here. D1 — copy the data, clean-build the
code — still stands, and on a stronger argument than the mitigation: the two golden sets *must*
diverge, and the two implementations' numbers are never compared to each other by design.

---

## 7. Ready to implement

**Plan:** `docs/plans/small-model-benchmarking.md` (this document).
**Method note:** `docs/plans/small-model-benchmarking-ml.md` (`data-scientist`) — **the single
source of truth for every formula, constant, threshold, tolerance, denominator, sample size and
verdict string in this feature**, plus the deferred judge design. It is not optional background:
`modelbench/stats.py` implements it and this plan cites it. Read it before starting S1 — in
particular **§3.4, the binding rules that are `stats.py`'s contract** (their number is the note's
too — v1.7 stops restating it), and **§7.2's verbatim resolving-power string**, which is a test
target.

**Version pairing:** this plan **v1.27** is aligned to the note **v1.21** (`998d13b`), folding in
**two** note revisions at once. **v1.20** rules plan-gate P13-2 — `ItemTiming` under the loop — and
its plan-side edit list is §11.9 **ask 7**, landed here at §3.3, §3.5, §3.6 (the FR-11 table and the
withholding bullet), §3.8.4, §4 S1, §4 S2, §5 tests 10b and 15b, and Appendix A. **v1.21** rules
P14-6's second half (§4.2(f)'s three denominators), adds **§4.3 rule 4** — the
mechanism-to-denominator mapping, in one place — and reports a **live contradiction between two
sections of this plan** that no gate pass reached; its plan-side list is §4.3.1, landed here at
§3.6's third and fourth dispositions, §3.8.4, §3.8.4's `cleanThroughTurnH` bullet, §4 S5 and §5.
*(Plan v1.19–v1.26 were paired to note v1.19 (`a707d09`), which added §3.4 Rule 4a and landed as §4
S1e Table H; v1.20 through v1.26 folded in no note delta.)*

**One §7 rule 3 raise carried forward, and two opened at v1.27.** The carried one is a citation
defect rather than a method one *(v1.19)*: Rule 4a's opening attributes itself to *"plan-gate Pass
8's `P8-1`"*. The finding it
closes is **impl-gate** `P8-1` — the `- decided by:` audit naming the arm that did not bind, in
`docs/reviews/small-model-benchmarking-impl.md` — whereas **plan-gate `P8-1` exists and is a
different finding**: Tables C and G colliding on `stats.py:159`
(`docs/reviews/small-model-benchmarking.md`). §7's own prefix convention was adopted precisely
because two reviews number `P8-*` independently, and it says an unprefixed or mis-prefixed number
"names nothing". Nothing in this plan depends on the resolution and Table H lands unaffected; the
note is the `data-scientist`'s and this document does not edit it (§7 rule 3). *(v1.10 paired
itself to v1.12 and was one revision stale by the time the gate read it — plan-gate P5-5. The pairing
is a claim about a named commit and it is re-checked in the revision that makes it.)*

**The two v1.27 raises, both consequences of note rulings this revision adopts rather than
disagreements with them, and neither blocking anything here.**
**R-1 — the `Y_calls` netting** (stated at §3.8.4 where it is used, and carried into §4 S2 rule
(iv)). `-ml` §11.4 and §11.9 ask 7 pin `callCount == len(ItemTiming.calls) == len(chatResults)`,
i.e. **completed** calls. Two sentences of §11.4 do not survive that pin: *"`callCount` is `1` for
every role but `tool-caller`"* is false on an item whose only call returned nothing (it is `0`),
and the bound `statsCoveredCount ≤ Y_calls − (calls that returned no response)` has an
**identically zero** subtrahend, because such a call is not in `Y_calls` at all. The substantive
half is that `Y_calls` is then a **netted** call denominator whose removed members are exactly the
calls that could not have carried `stats` — the pattern §11.4's own `Y` argument refuses one unit
up. The plan builds the pin and writes rule (iv)'s bound in the form true under either resolution;
if the note unnets `Y_calls`, the cost is one field's definition and one test, and nothing stored
moves.
**R-2 — a line cite that has drifted.** `-ml` §11.9 ask 7's last item cites the stale gap
expression at `modelbench/lmstudio.py:225`; at `d5b549d` that text is `_coerce_finite_float`'s
docstring at `:247`. §4 S2 pins the sweep by symbol and by a grep with a count rather than by the
line, so nothing here waits on it.

The two note
revisions v1.10 folded in are separable and are recorded first: **v1.11** makes §3.4 Rule 4's
closed-form percentile **binding** — so the paired *binary* interval resamples nothing, takes no
seed, and the four printed strings naming one arm of the envelope are corrected — with the plan-side
consequences at §3.3 (`sampling.seed`'s object), §3.9 point 1, §4 S1's signature block and §4 S1e
Table D, including the one decision the note left to this document: **`DecidedBy`'s
`cluster-bootstrap` token is renamed `conservative-envelope`**, for the note's own reason with the
sign reversed — a machine token naming a resample that no longer runs is the same defect as prose
naming one arm of an envelope, and it is free while no stored record carries it. **v1.12** rules that
§11.5's exactness argument does not extend to §11.5.1's detector, so §11.7's slot 3 selects on a
**computed** `censoringExact`; it asks this plan for two things and both land — a withheld item's
wall clock stays readable (`ItemTiming.wallClockMs`, §4 S1; the plan takes the field rather than the
offered reconstruction, because on an embeddings arm the reconstruction's three operands are all
absent while the wall clock was measured) and `statsCoveredCount` is `None`-never-`0` **by call
surface** (§4 S2 rule (iv-a)). §11.6's floor, its 5-point constant and §11.5's table are untouched by
that ruling, so nothing downstream of them moves.

**Notes v1.13 and v1.14, folded in at plan v1.11.** **v1.13** *confirmed* §4 S2 rule (iv-b) by
correcting the note rather than the plan — it withdrew its own `tokensPerSecond` exemption, so all
three sibling medians take §11.6's p50 gate, which is what (iv-b) already said. **No plan change was
owed, and the plan's rule was right before the note's argument for it was**; what v1.13 added is the
**co-presence invariant**, which the plan did owe and now carries as rule **(iv-c)**. **v1.14**
answers the three method questions plan gate Pass 5 routed to `data-scientist`, and all three land
here: **co-presence** resolves as the *conservative single count* — an item with `stats` but no
usable `promptTokens` leaves `statsCoveredCount` **and** all three medians, as a **disposition that
raises nothing**, which turns rule (iv)'s identity into an inequality (§4 S2 (iv), (iv-c));
**`censoringExact` clause 1 survives** and needs a third `withheldFor` value, `"timeout"`, because a
timeout is a *censored* observation and a 40 ms failure is a *missing* one — three item states, one
counter (§3.6, §4 S1, §4 S2 (vi)); and **`paired_cluster_bootstrap` is kept**, named as §3.2d's entry
point for every continuous verdict with `paired_bootstrap` as its engine (§4 S1e Table D). v1.14 also
carries **one condition neither gate had reached** — `_widen`'s `[-1, 1]` clamp is a
difference-of-proportions assumption and is wrong for `sep_z` — which is a live defect in shipped
S1 code and lands as **§4 S1e Table E**. *(v1.11 gated that table on §4 S3 done-condition 2; note
v1.15 withdraws the gate — next paragraph.)*

**Note v1.15, folded in at plan v1.12 — one blocker's method half, and one reversal.** v1.15 answers
plan-gate **P6-1**, whose finding was that the embedder pack's only verdict metric had no
representable carrier and that the shipped comparison path therefore failed **silently, in two
directions**. Four rulings land here. **(1) The carrier is a second per-item map**,
`measures: Mapping[str, float]`, never a widened `counts`; finite floats with **no domain constraint
at the carrier** (a domain here would be Table E's clamp mistake one layer up); a metric name in
`counts` **or** `measures` and never both, which is what makes instrument selection total; absence
stays `scoreable`'s job, so an MRR of `0.0` is a *measurement* and `measures` is not
`Mapping[str, float | None]`; `scored_value` as `scored_outcome`'s sibling, and **`scored_outcome`
raising** on a continuous metric — the one line that turns a booleanised MRR into a loud failure.
It lands as **§4 S1e Table F**, at **S1**, on S1e's own *free only now* argument. **(2) §3.2e gains
two strings, not one** — the decision stays binary (the interval excludes zero), so *distinguishable*
and *not distinguishable* both need renderings, and *instruments disagree* cannot arise where there
is one instrument; their four mandatory properties are the note's and §4 S1 cites them rather than
quoting them. **(3) §3.3 rules a `verdictMetrics` family homogeneous in kind** — Holm orders by
p-value and a continuous verdict has none — which lands at §3.3 (iv), together with this plan's
decision about **where** that refusal is enforced, since a manifest carries no records. **(4) The
reversal:** `separationZ` needs the same per-item carrier but a **different aggregate** one (§5.2
publishes a median, a p10 and a fraction; none is a mean), and it is **reported rather than
verdicted** — so `_widen`'s clamp is due with the `sep_z` comparison and **not** as a precondition
of S3. §4 S3 done-condition 2's gate on Table E, added at v1.11, is withdrawn, and §4 S1e's
stage gate becomes Table F on done-condition 1. Latency is deliberately **not** a `measures` member:
it is continuous and per-item and reads like one, and it has its own carrier (`ItemTiming`) and is
verdict-ineligible by §11.7.

**Note v1.16, folded in at plan v1.13 — three raises answered, and one correction to this plan.**
**(1) The continuous producer is specified**: new §3.4 **Rule 8** gives `continuous_verdict()` its
parameter list, its four refusals, its rendering precision, a **`ContinuousVerdict` sibling return**
rather than a `Verdict` with six fields left empty, and four **negative** parameters — no
`ResolvingPower`, no `alpha_step`, no McNemar *p*, no percentile levels — which make the
multiplicity correction *unrepresentable* rather than guarded. §4 S1 replaces its v1.12 restatement
with a citation, and takes the one thing the note leaves here: `report.py` renders a
`Verdict | ContinuousVerdict` union. **(2) Rule 8 names two engine preconditions**, and the plan
lands both: `_widen`'s conditional clamp was already §4 S1e Table E, and the bootstrap's **quantile
levels become required parameters** as new **Table G** — without them a `k > 1` continuous family's
correction, which §3.3 puts *in the interval*, has nowhere to land. **(3) The mixed-family refusal
is corrected in this plan's direction on the enforcement point and against it on the granularity**:
`compare_report` pass 1 is confirmed (a manifest carries no records, and a manifest `kind` field
would be the second declaration §3.2d refuses), while the refusal is of **the whole family's
verdicts** rather than of the offending members — dropping the minority kind shrinks a
pre-registered `k` after the results exist. §3.3 (iv) and §4 S1 carry it; DC-13(e) and §5 test 11d
assert it, including the two negatives that fail if it is built as a DC-10 exclusion. **(4) `sep_raw`
gets its published figures and one prohibition** (§5.2): a median, a p10 and the fraction above zero,
**no mean for either quantity**, and **no cross-model `sep_raw` difference on any path** — §3.8.1
carries it, in the shape §11.7 slot 6 already uses for latency. Separately, the note **swept the
retired instrument name across all six of its sites**, which closes §4 S1e Table D's trip-hazard
raise: §3.2f is now safe to copy verbatim and Table D's second residual can reach zero.

**Note v1.17, folded in at plan v1.14 — one raise ruled, and no open raise left** *(plan gate Pass 7
deliberately excluded these two from its findings; they ride in the same revision so the pair does
not go two revisions out of step)*. v1.13 raised, under §7 rule 3, that Rule 8's parameter list
carried no `support` and so had no `clamp` to forward to the `paired_cluster_bootstrap` whose
`clamp` §4 S1e Table E makes required with no default. **v1.17 rules it in, with a sharper shape
than the recommendation:** `support: tuple[float, float] | None`, keyword-only and required with no
default, its `None` a **stated** value meaning *unbounded*; **no `clamp` parameter**, the difference
support being derived inside the function so its sign and order are written once; and a fifth
refusal, a `support` with `lo >= hi`. Two plan-side consequences, both at §4 S1: the family loop
**forwards `ContinuousMetric.support` and derives nothing** — less than v1.13 described — and the
non-blocking classification is re-rested on `_widen`'s **scale-1.0 identity** rather than on the
pack census v1.13 used, the note accepting the census as true today and rejecting it as
load-bearing. §4 S1e Table E is untouched by the ruling and its engine `clamp` still does real work,
on the one caller that states it directly: §3.8.1's exploratory `sep_z` comparison, which has no
metric aggregate to ask. *(v1.12's two raises were closed by note v1.16: the producer's signature is
Rule 8, and §3.3's word "validate" moved to `compare_report` pass 1 — the note's sentence moved, not
this plan's. With v1.13's closed here, **no §7 rule 3 raise is open** — plan-gate P8-1 opened one again at
Pass 8 and note v1.18 closed it, next paragraph. *(v1.19 opens one: a **citation** defect in note
v1.19's Rule 4a, recorded in the* Version pairing *block above. It blocks nothing.)*)*

**Note v1.18, folded in at plan v1.15 — the §7 rule 3 raise plan-gate P8-1 opened, ruled, and one
scope corrected** *(the raise was opened by the gate rather than by this plan, and it is the
mechanism working: the plan could state an exemption on its own authority and could not change the
estimator's signature)*. **v1.18 rules the level an exact rational** — `percentile(values, *, level:
Fraction)`, `levels: tuple[Fraction, Fraction]` on both bootstraps — **and refuses the exemption**
the gate offered for `stats.py:159`. Its two rejected alternatives are recorded there and neither is
reopened here: a **wider integer unit** fails at `k = 3` rather than late, because the denominator
is `40k` and 3 divides one of them; **rounding the level outward** works and is rejected on price —
a second number on every continuous verdict whose entire content is that the tool cannot represent
its own level, and a second meaning for *attained level* in one report. The note also **corrects
§11.10(3)'s scope to package-wide**, its v1.17 wording having named `modelbench.results` only, which
is the gap the exemption read as licence. Four plan-side consequences, all in §4 S1e and all listed
as the note's own edit list at §11.9 item 6: Table C keeps `:159`, its signature moves to the
`Fraction` level with the four `LEVEL_*` constants, Table G's element type moves with it, and
**Table G's two residuals are re-derived over `:159`'s post-Table-C spelling**. Two things the note
leaves here and this revision takes: the **four constant names**, and the **order** of the two
tables on that one line. *(Also from v1.18 and cheap: `paired_bootstrap` refuses a transposed
`levels` pair — §4 S1e Table G's last row.)*

*(The v1.9↔v1.10 pairing, recorded here because its reversal is the reason §3.6 reads as it does.)*
The note's **§11** closes R-13 and, in doing so,
constrains four plan-owned decisions it depends on (the timeout disposition, which fields a
contaminated item withholds, the guard's first comparand, and where the in-call reload detector
sits) and asks the plan to name one new field
(`latencyMsMax`) — all five land in this revision, at §3.5, §3.6, §4 S2 and §6 R-13. **One of them
reversed inside this revision**: §11.4 first ruled the whole timing block withheld, then took the
measurement G3-3 had proposed and reversed to `latencyMs` alone. The plan follows the measurement,
and the sequence is recorded in §3.6 rather than smoothed away — a conservative default is right
while a question is open and wrong the moment it closes. In the other
direction the plan's v1.9 changes touch no method: `callSurface`, the capture order and DC-10's
predicate are the harness surface and the result schema, which §7 rule 2 gives to this document.
They share one vocabulary
(`verdictMetrics` + `headlineMetric`, the plan's names), one `guard-judge` metric pair
(`falseAdvanceRate` + `falseSuspendRate`, the note's), one `H` contract (pack-declared and validated
`H ≤ min(script length)`, the plan's), and one analysis-unit rule (the outermost component of
`pairingKey`, the note's). A future revision of either that changes a name, a number, a signature or
a contract must sweep the other in the same pass — the Pass 1 gate's finding was silent divergence
between exactly these two documents, and the fix is not a one-time correction but a standing
obligation.

**How to resolve an apparent disagreement — and why this is not a simple precedence rule.** v1.3
said "where the two ever appear to disagree the note is right". That sentence caused a defect. When
the plan moved `H` from derived to declared-and-bounded (M-11) and the note's §4.6 still carried the
old derived definition, the precedence rule did not resolve a conflict — it **propagated the stale
clause**, converting a fixed finding back into a live one (Pass 2, N-3). A blanket precedence rule
launders staleness with exactly the authority it was given to settle disputes, and the more
trustworthy the senior document, the more efficiently it does so. So:

1. **A disagreement is presumed to be staleness, not a conflict.** Find which document changed last
   on that point and reconcile both; do not apply precedence to a clause one side has already
   superseded. Both documents carry version numbers and dated revision lines for this purpose.
2. **Precedence applies only when both sides are current, and it is split by ownership, not by
   seniority.** **The note owns method** — formulas, constants, thresholds, tolerances,
   denominators, sample sizes, instrument selection, `stats.py`'s signatures. **This plan owns the
   pack-manifest contract and the harness surface** — field names, validation rules, `H`'s
   declaration semantics, the CLI, the result schema. `H` is the worked example: the *statistics* of
   `cleanThroughTurnH` are the note's, the *rule that `H` is declared and bounded* is the plan's.
3. **Neither document may resolve a disagreement by editing the other.** Raise it; the owner fixes
   it in their own file. That is what kept N-3 a one-clause fix rather than a merge conflict between
   two agents editing in parallel.
4. **The same discipline applies *inside* each document — v1.4's three rules did not reach there,
   and both of v1.5's defects lived in that gap.** *(New in v1.5.)* Appendices, recap tables and
   enumerations are **derived surfaces**: where one disagrees with the section that owns the
   contract, the owning section is right by construction and the derived surface is stale — never a
   second opinion to be weighed, and never a reason for an implementer to build the weaker of the
   two. §3.3 owns the `sampling` contract, so Appendix A's `PackRef` was simply behind it; §3.4.2
   owns the model field set, so §3.4.1's hand-typed forbidden list was behind it the moment §3.6's
   audit fields joined that set. Two consequences, both applied in v1.5:
   - **A change to an owning section sweeps its derived surfaces in the same pass** — rule 1's
     standing obligation, turned inward. The sweep is cheap and the omission is invisible: both
     defects passed two review passes, because a reader checks an enumeration against its own prose
     rather than against the set it claims to complement.
   - **Where a derived surface can be a *derivation* rather than a transcription, it must be.**
     §3.4.1's forbidden set is now a set difference over §3.4.2's names, and `PackRef`'s
     `analysisUnitIndex` is a property over `pairingKey`, precisely so there is nothing left to keep
     in sync. A list maintained by hand beside its own stated intent has already drifted once here;
     state the rule and let the list follow from it.

5. **An edit list over shipped code is a grep with a count, not a list of remembered sites — and
   what that buys is a *no-forgetting* guarantee over token-carrying sites, not completeness.**
   *(New in v1.10, plan-gate P4-2; its property corrected in v1.11, plan-gate P5-2.)* Twice a
   revision has specified S1 work with an incomplete
   edit table — v1.8 named two of four sites for the `residencySource` swap, and v1.9, in the very
   revision that adopted `-ml` §11.2's estimator ruling, omitted the two shipped `_percentile` copies
   that ruling rewrites. The failure is not carelessness: a hand-written list of edit sites has **no
   completeness property**, and a reader checks it against its own prose rather than against the tree
   it claims to enumerate — which is rule 4's diagnosis, pointed at code instead of at a derived
   surface. So any table telling an implementer which shipped files to change carries three things:
   **(a)** the command or commands that enumerate its sites; **(b)** their per-file counts at a named
   commit; **(c)** a done-condition that re-runs them and asserts a stated residual. The counts are
   reproducible, which a prose number is not (`grep -rFc` gives `armKind` 50 matching
   lines where `grep -rFo` gives 57 occurrences; a table saying only "59" has already lost the
   distinction). **And reproducible means portable: these commands are re-run by different people
   in different shells, so a residual whose correctness depends on regex dialect is not a residual —
   state one as a fixed string (`-F`) wherever a fixed string will do, and verify any new *regex*
   residual under more than one `grep` implementation before writing it down** *(v1.18. The occasion
   is v1.17's `\bbootstrap_seed`, this plan's first word-boundary residual: checked under two
   implementations, which agreed, so nothing is owed retroactively — the exposure is the next one.)*

   **What those three guarantee, stated exactly — because v1.10 claimed more and the claim is the
   dangerous part.** v1.10 wrote that the table is then *"complete by construction — a site the
   author forgot is still in the command's output and the residual assertion fails until it is
   gone"*. That is **false in two ways**, and the converse it invites — *residual zero, therefore
   nothing was missed* — does not hold. What is true is narrower and still worth the machinery: **a
   site that carries one of the table's tokens cannot be forgotten**, because it is in the command's
   output whether or not the author saw it. The two gaps and what closes each:

   - **(a) A site that carries no token is invisible to the command, and no residual reveals it.**
     So **every such site must be covered by a second named command, or the table is not complete** —
     Table A's `sizeBytes` row was the pattern and is now the rule. Two clarifications that keep the
     search bounded rather than infinite. *A command reaches a construct's **body** through its
     head*: `grep REQUIRED_BY_SCHEMA` names the mapping's declaration line and the implementer edits
     the entries under it, `grep arm_kind` names a `parametrize` decorator and the implementer edits
     its cases — the unit of an edit site is the construct, not the line, and only a construct **no
     command names at all** is a gap. And the gaps have a recognisable shape, because they are all
     one thing: **a contract restated by value rather than by name** — a hand-transcribed literal, a
     test asserting a set with a set display, a fixture branching on a string. That is deliberate
     (impl-gate M-4 requires those literals to be independent of the module under test), so the same
     discipline that makes the suite honest is what puts these sites outside every grep. Search for
     them by that shape; grep cannot prove their absence and this rule does not claim it does.

     **One family of token-free site has a mechanical command form, and it is named because it
     recurred five times before anyone wrote it down: a table that adds a *new type to an existing
     union* enumerates its sites by the *attribute the new type lacks*, never by the type's name**
     *(v1.14, plan-gate P7-1)*. Shipped code that breaks under a retype spells **neither** type. It
     reads an attribute a sibling has — a bare `else` branch reaching for `.mean` — or it tests the
     string literal that tags that sibling in storage, which is a lower-cased word and not a class
     name. Both are structurally invisible to a name-based command, and the attribute form is the
     sharper of the two because it enumerates *the defect* rather than a spelling:
     `grep -rn '\.mean' modelbench` returns exactly the three bare-`else` readers §4 S1e Table F
     missed at v1.13. So such a table carries, beyond its type-name commands, **one command per
     attribute the union's other members carry and the new one does not, and one per string literal
     that tags a member in storage or in a comparison** — and states each command's scope, since
     these forms often return nothing under `tests/` and a silent scope reads as an oversight.
     A narrower rule falls out of the same finding and is worth its one sentence: **a command pins
     the type, never the variable that holds it.** Table F's v1.12 token-free command pinned the
     variable name inside its `isinstance` call, and half the tree spells that variable with a
     single letter, so three of six sites were invisible to the command written specifically to
     reach them.
   - **(b) A token that survives the edit has no zero residual, so asserting one is meaningless.**
     Residual zero is a real check only where the edit **retires** a token. Where the token
     **survives by design** — `armKind` keeps its two values while the *mapping key* becomes a
     profile — the table must name a **second-form residual**: a different command, over a token the
     edit does retire or a literal it rewrites, **whose count is zero after the edit**. Where no such
     command exists, the table **says so** and names what stands in its place. A residual over a
     surviving token is worse than none: it reads as a guarantee, is satisfied by construction, and
     the next revision will trust it exactly as v1.8's and v1.9's lists were trusted.
     Symmetrically, a residual must not fail on a **correct** edit — a command that would still match
     after a faithful implementation is not a residual, it is a trap that trains the implementer to
     override the done-condition. **And a table's residuals must not be satisfiable by a *half*-applied
     edit of that same table** *(v1.14, plan-gate P7-2)*: where an edit retires *n* distinct literals
     its residuals must **distinguish** them — one residual per literal, or one command whose count
     is *n* and whose target is zero — because a single residual over *one* of two is passed by the
     implementation that retires that one and leaves the other standing, and the surviving half is
     a number that still prints.

     **A residual's target need not be zero, and where it is not, the table states the target and
     names the surviving line** *(v1.15)*. `-ml` §11.10(3)'s package-wide percentile-definition
     check went **2 → 1** when the clause was written, the survivor being the one implementation the
     rule requires to exist.
     What makes a number a residual is that it is **stated in advance and re-run**, not that it is
     zero; a rule written over zero alone would have excluded the one check in §4 S1e that no
     half-application can pass. *(v1.17: that check now goes **2 → 2** and its survivors are named
     — see the named-line-set clause below — and it is no longer the only non-zero target, Table E's
     third-form pair being **0 → 1** each. The clause is unchanged; what it governs has grown.)*

     **A residual counts the lines that *name* a token, and grep cannot tell a live use from a
     mention that disowns it — so a residual is scoped away from wherever this plan's own
     done-conditions require the retired spelling to be written** *(v1.16, impl-gate P5-3)*. Call
     the shape the **disowning mention**: a comment recording a retirement, a docstring naming what
     a function replaced, a test that must **construct** the retired shape in order to assert it is
     refused, or a test *function name* carrying the retired word. Each is a line the residual
     counts and none of them is a site the edit missed. What makes it worse than a false alarm is
     that the residual and the assertion become **mutually exclusive**: the implementer's only exits
     are to drop the assertion or to override the done-condition, and the behaviour the assertion
     existed to pin is then held by nothing at all — which is exactly what impl-gate P5-3 found, and
     proved by building the counter-implementation that survives. So a table in that position does
     one of three things and **says which**: scope the residual to the files where the token
     genuinely retires and **name the file the assertion lives in**, so the prescription is binding
     rather than incidental (§4 S1e Table A's second residual, Table C's third); or state a
     **non-zero target** and name each disowning line (the clause above); or state that no residual
     is available and name what stands in its place (Table C's seventh line). A table that scopes
     owes the other half of the bargain — **the retired spelling may not be written in the scoped
     files at all, not even as prose recording the retirement** — because a scope without that is a
     guess about where a comment will land. **Two of §4 S1e's eighteen residuals were in this
     position**, in the document that wrote this rule, and neither was found by reading the table:
     one by building the counter-implementation the residual made unassertable, one by the sweep
     DC-12 now runs per residual rather than per table.

     **A residual stated over text the edit *destroys* is blind, not weak — so where the edit
     rewrites the expression its retired literal is fused into, the residual is stated over the text
     the edit *creates*. Call that the third form** *(v1.17, impl-gate F1)*. The two forms above both
     assume the residual's match survives to be counted: the **first** counts a token the edit
     retires, the **second** a literal the edit rewrites while its construct survives. Neither holds
     when the edit **restructures the expression**, because then a defective implementation written
     in the *new* shape matches nothing the old text described, and the residual reads **exactly what
     a faithful edit reads**. §4 S1e Table E was the instance: its pair matched `max(-1.0, point …)`
     and `min(1.0, point …)` — 1 and 1 before the edit, 0 and 0 after it, and **0 and 0 on the
     half-application that wires the lower clamp and leaves the upper bound literal**, which is that
     table's own defect shipped intact. A third-form residual is stated over the **prescribed
     post-edit spelling**, its *before* is 0 and its target is 1 (or *n*), and it fires on the
     half-application because the half-application never writes the text.

     **The trigger is nameable, which is what keeps this from being a warning to apply everywhere: a
     *parameterising* edit** — one that turns a literal into an argument, so the call's shape changes
     and not merely a name or a value. Contrast the three edits that look similar and are not.
     §4 S1e **Table G** replaces `LEVEL_CI95_HI` with `levels[1]` at a call site the edit leaves
     otherwise intact, so `percentile(means, level=LEVEL_CI95_HI)` still matches under the
     half-application and the second-form residual works — confirmed by simulation, not assumed.
     **Table B** re-keys a subscript and rewrites a set display's *contents*, leaving both
     constructs' form alone. **Table F** retypes an annotation, which is the retired thing entire.
     So the sweep asks one question of each residual — **does the span it matches include text this
     table's own edit rewrites?** — and only a parameterising edit answers yes. Its cost is stated
     with it: a third-form residual **pins a spelling**, so it must be pinned to the spelling that
     actually shipped and it fails on a later refactor of that expression. That is the same bargain
     Table C's `:159` row already strikes, and the reversal trigger is the same — when the expression
     is rewritten for an unrelated reason, the residual is re-derived over the new spelling, never
     re-**widened**. *(v1.23, plan-gate P12-6. Before this revision the word here and at the
     three other sites stating this reversal named a **scope change** rather than a widening, which
     read as an absolute — and one the scoping default below contradicts on its face, that default
     being a **narrowing**. What this reversal forbids
     is the other direction — widening a residual's scope so that a spelling which moved is caught
     again somewhere else — and the repair is to re-derive the residual over the text that now
     exists. All four sites carry the corrected word; the reason is stated here and cited there.)*

     **A stated baseline moves only when the revision moving it *re-runs the commands*, and never
     because some other unit landed** *(v1.21)*. Clause (b) says a table carries its counts **at a
     named commit**, and the named commit is a property of the **measurement**, not of the tree: to
     re-point a baseline without re-running is to assert a count at a commit nobody ran it at, which
     is the one thing this rule exists to stop. So the treadmill alternative — re-point every table
     to the tip whenever any unit lands — is only available to someone willing to re-measure every
     table every time, and this document is not, nor should the next author be. **A landed table's
     baseline never moves at all**: its rows are a record of what a commit did (§4 S1e's `Landed:`
     convention), and re-pointing a record falsifies it. **An unlanded table's baseline moves in the
     revision that re-runs it, and states which commit that was.** Where the two diverge — the tree
     has moved and no revision has re-measured — the gap is carried **once**, in DC-12, as *what has
     changed since*, and not by silently editing a number nobody re-derived. *(v1.21's occasion:
     `93b0e42` landed under v1.20 while it was being written; Table H is re-measured and re-points
     to it, the landed tables do not move, and DC-12 names what changed. *(v1.21's count here was one too many: there are **six** landed
     tables of eight, one `Landed:` line each, and the extra one it counted was Table F — which had
     not landed and was simply not re-measured that revision. Corrected at v1.23, the revision that
     re-measures it.)* Table F's unit will
     move `results.py` and `report.py` under exactly these tables next, so the rule is written down
     before it is needed rather than after.)*

     **A table states its command and its count; a gloss beside the count names *sites*, and does
     not restate a total** *(v1.21)*. The gloss itself is not the problem and is not forbidden — it
     is what rule 5(a) asks for in spirit, the step from *"a command's output is a superset of the
     sites"* to a list a reader can act on, and §4 S1e Table E's naming of its four non-sites is the
     model. What breaks is a gloss that **re-states the total** in order to partition it: that total
     is a second copy of the command's own number, and §7 rule 4's one-home rule applies to a number
     exactly as it does to a fact. §4 S1e Table H's `envelope_arms` gloss is the case, and it failed
     **twice by two different mechanisms** in one day: v1.19 mis-partitioned the total, plan-gate
     P10-5 re-derived it correctly, and an unrelated unit added one matching line within the hour and
     staled it again. The command was right all three times. A **site list** — four call sites,
     named — survived both events untouched, because sites are stable against additions and totals
     are not. So: name the sites, cite the command for the count, and let the count live in one
     place.

     **A residual is scoped to the files its own site rows name. A wider scope is an exception the
     table states and justifies** *(v1.22, plan-gate P11-1)*. The reason is a trade-off worth stating
     because it governs every residual, not only the ones that go wrong: **a pattern's specificity
     and its scope's breadth are substitutes.** A long, unique pattern tolerates a wide scope —
     which is why the residuals over retired names like `FORBIDDEN_BY_ARM_KIND` or `bootstrap_seed`
     are safely stated across `modelbench` and `tests` — and a short one does not. So **shortening a
     pattern obliges narrowing its scope**, and the case that produced this clause is precisely
     that: §4 S1e Table H's residual pair was repaired at v1.20 by removing an introduced local
     (rule 5(b)'s second repair, below), which shortened it to a bare constant subscript and thereby
     widened its match to any use of that constant **anywhere in the package** — in the revision
     that also mandated a second consumer of it in `report.py`. The v1.19 form was immune to that and
     vulnerable to the other; each repair traded one exposure for the other, and only the scope
     closes both.
     **Why a default rather than a caution.** A caution — *"take care when you shorten"* — does not
     fire, and this document has now watched a residual fail on a faithful edit **seven** times, the
     seventh introduced by the fix for the sixth. A default is checkable: read the table's row list,
     read the residual's scope, and they either agree or the table says why not. It is also not new
     practice — Table C's first two residuals are scoped one file each and **partition** that table's
     six production sites, which is the discipline this clause generalises. **Partition is the
     property to aim for wherever a table's residuals can supply it, and it is strictly stronger than
     narrowness** *(v1.23, plan-gate P12-2)*: it makes a half-application non-zero on the half it
     skipped and the number says which half. Narrowness does not imply it — §4 S1e Table H's six do
     not partition its ten site rows, nothing reaching the two `tests/` rows — which is fine and is
     why the two properties are named separately rather than one being read off the other.
     Tables B and D reach `tests/` because their site rows do.

     **Where the default is bounded, and it is bounded by what each instrument is for**
     *(v1.23, plan-gate P12-2)*. **The enumerating commands are the completeness instrument and a
     residual's scope is its *edit's*** — that division of labour is what licenses narrowing at all,
     and without it the scope silently inherits the site list's completeness. Plan-gate `P10-1` is the proof
     that this is structural rather than hypothetical: an incomplete site list in Table H, caught by
     an enumerating command and by no residual under either scope, because *"command 2 **does**
     return the line … the enumeration was sound and only the site list was short"*. So the
     mitigation is named rather than assumed — **wide enumerating commands plus DC-12's
     per-residual sweep**, not the residual's own scope. **And where a residual's job is instead to
     prove that a text exists nowhere wider than the edit — a one-home or an identity check — the
     scope is that *purpose's*, and the table says so.** Two tables instance it and both state their
     reason on themselves: **Table C's third** is `-ml` §11.10(3)'s package-wide percentile-definition
     identity, and **Table F's third** is the `{"binary", "continuous"}` tag set, whose `→ 0` is a
     claim that the set has one home and is therefore worth nothing file-scoped (plan-gate P12-1 —
     the case that showed a specificity argument alone points the wrong way here, since that pattern
     is short and its purpose is wide).
     **This clause binds prospectively, and the reason is convention 1's and not a claim about
     landed residuals** *(the premise corrected at v1.23, plan-gate P12-3)*. A landed table's rows
     are a **record of what a commit did**, and re-scoping a record falsifies what it says was
     checked — which is the same argument that stops a landed table's baseline moving, stated above, and
     it is sufficient on its own. **What must not be said, because this document holds
     two counter-examples to it, is that a landed table's residuals were run against the faithful
     implementation and hit their targets.** Impl-gate **F3**: Table D's `bootstrap_seed` residual
     *"was unreachable on a correct edit **from the moment it was written**"* and was repaired at
     v1.17 to the whole-identifier form. Impl-gate **F2**: Table C's third residual's target moves
     1 → 2 the moment Table D lands, and was **restated** rather than met. So *landed* certifies
     that a residual's **statement** was repaired after being run — the opposite direction — and a
     later reader must not treat a landed residual as validated and skip re-checking it. That
     re-check is DC-12's, at the end of the round.
     The two tables that had not landed when this clause was written, **H and F**, are scoped to
     their rows' files in the same revision that writes it — **except Table F's third, which v1.23
     returns to package-wide under the purpose clause above**. *(v1.23, plan-gate P12-4: v1.22
     justified that compliance with *every one of the six and the three reads the same* before
     *value under either scope, so nothing was traded*. Equal *befores* are silent about the
     **after**, which is what a residual claims, and for Table H's third-form pair they are **0 = 0**
     — true under any scope including an empty one, and vacuous for exactly the pair plan-gate `P11-1`
     was about. What the narrowing trades is that the after assertion now ranges over one file, argued
     per residual on each table; what backs the **form** is the gate's synthetic two-file probe, stated
     on Table H, and the `e162ba9` insertion, measured and stated once in the exact-text-versus-line-pin
     paragraph below.)*

     **A third-form residual is only a residual if the table *writes out* the text it pins, in the
     row, and not only inside the residual command** *(v1.20, plan-gate P10-3)*. The two forms above
     are stated over text that already exists, so a reader can always go and look at it; the third
     is stated over text **that does not exist yet**, and a command is not a specification. Where
     the pinned text contains any name the edit must *introduce* — a local, a helper, a constant's
     spelling — that name is fixed **in the site row**, because the implementer reads the row and
     the done-condition reads the command, and nothing reconciles them if they disagree. §4 S1e
     Table H is the case that produced this sentence: it claimed a prescription, left the only
     statement of it inside two residual commands that pinned two locals the plan never fixed, and
     those locals differed from the notation `-ml` Rule 4a uses for the same two quantities — so an
     implementer following the rule the table says it does not restate would have read **0** against
     a stated target of **1**. That is a residual failing on a faithful edit, the thing this rule
     forbids, arrived at in the inverted direction and inside the first table written *against* the
     third form rather than corrected into it. Two repairs, and a table may take either: write the
     text out, or state the residual over a fragment that contains **no** introduced name. Table H
     takes both.

     **And the form has a third virtue, which is the reason to prefer it even where a line number
     would be shorter: an exact-text residual is robust to edits *elsewhere in the same file*, and a
     line pin is not** *(v1.19)*. This document has now watched both halves of that happen. A line
     pin broke: `tests/test_results.py:507` became `:543` when a *different* table's unit added two
     test blocks above it, and §4 S1e Table C's row was false until impl-gate P5-6 caught it — one
     of two such corrections this plan has had to make. An exact-text residual held: Table E's pair
     matches `_widen`'s body, and it survived both the Pass 8 fix round inserting a guard thirty
     lines above it **and** Table H changing what its two call sites pass — reading **1** and **1**
     throughout, where a line pin would have been broken twice. A residual is a claim that has to
     survive other people's edits to be worth stating, so it is written over text that only the edit
     it describes can change.
     **The form's largest test so far, and it is the reason to keep paying the third form's cost**
     *(v1.23)*. `e162ba9` inserted **31 lines** into `modelbench/stats.py`, above **every line pin
     any table of this section states below `:262`**: all five of Table H's moved
     (`:382`/`:387` → `:413`/`:418`, `:413` → `:444`, `:896` → `:927`, `:1168` → `:1199`,
     `:1184-1187` → `:1215-1218`), and so did Table E's third-form anchor, `:261` → `:292`. **Not one
     residual count moved** — Table H's six
     read `2 / 0 / 0 / 1 / 1 / 2`, Table F's three read `1 / 1 / 1` and Table E's landed pair reads
     `1 / 1`, all eleven re-run at that commit. Every line pin below the insertion broke and every
     exact-text residual held, in one commit, which is the cleanest evidence this document is going to get for a
     convention it argued for over three revisions.

     **And where a count stops discriminating, the residual's assertion is the *named line set*, not
     the number** *(v1.17, impl-gate F2)*. A count is a proxy for an identity claim, and a second
     construct that the plan **mandates** can match the same pattern and restore the number by
     coincidence: §4 S1e Table C's package-wide percentile-definition check reads 2 before the edit
     and 2 after it, the two lines being entirely different functions. The command keeps its place —
     a *third* definition still moves the number and fails the check — but the table states the
     survivors **by name**, DC-12 re-reads them rather than the count, and the discriminating work is
     named where it actually lives. A table that lets a coincidental count stand as the assertion has
     written the same over-claim v1.10 wrote about completeness.

     **And the property is a property of the *round*, not only of a table** *(v1.15, plan-gate
     P8-1)*. Where two tables edit one line, one table's edit can retire the very literal the
     other's residual is stated over — so **both** of the second table's residuals go to zero from
     the first table's work alone, and the done-condition passes on an implementation that never
     applied the second table. Each table's set was sound read alone, which is why six passes did
     not see it. So: two tables meeting on a line **name each other on both rows**, state whether
     both orders are faithful, and the residual of whichever lands second is stated over the
     **surviving** spelling — re-derived, never re-**widened** (the third form's reversal trigger,
     above), since the token it was written over is gone either way.

   Where an edit **adds** rather than retires, the enumerating command is over the
   type's **construction sites** and the residual is asserted by the type system — a required field
   with no default — never by a count. §4 S1e is where this plan's **eight** tables live, and its
   preamble scopes them: the eight are the *retiring and re-keying* edits, and the adding ones are
   named there as the type-system-enforced remainder rather than left for a reader to miss
   *(v1.12, plan-gate P6-4; the seventh is v1.13's Table G, which retires two literals; the eighth
   is v1.19's Table H, which lands `-ml` Rule 4a and is the first written against rule 5(b)'s
   third form rather than corrected into it)*.

**Finding IDs are prefixed by their review, because two reviews independently number `P4-*`**
*(v1.10, at the plan gate's request)*. This plan cites `docs/reviews/small-model-benchmarking.md`'s
findings from Pass 4 onward as **`plan-gate P4-n`** / **`plan-gate P5-n`**, and
`docs/reviews/small-model-benchmarking-impl.md`'s as
**`impl-gate P4-n`**, with the same prefix applied to that document's earlier passes (`impl-gate
P3-1`, `impl-gate P2-3`). The plan gate's Pass 5 adopted the convention in its own text. The plan gate's Passes 1–3 used `B/M/m/N/G3-`, which collide with nothing
and are unchanged. An unprefixed `P4-4` in a document that cites both series names nothing.

New standalone component `model-bench/`, zero runtime dependencies, Python 3.12, eight stages:
S0 skeleton → S1 core (fingerprint/results/stats/report, no model calls; delivers AC-2/AC-3/AC-4) →
S2 LM Studio adapter + pack loader + conversation driver → S3 `embedder` pack (first real
end-to-end run; AC-5) → S4 `guard-judge` + `nlq-generator` packs → S5 tool-caller simulated
environment + per-turn scorer (synthetic traces) → S6 the FR-22 conversation scripts and the
known-answer validation (AC-1) → S7 `chat-responder` (deterministic layer only, FR-21a) → S8 docs.

The eval-harness question is resolved as **copy the golden data, clean-build the code, change
nothing in falkor-chat** (§3.1), with the duplication risk discharged by a numeric agreement test
rather than by shared code — and on the observation that the two golden sets *must* diverge, because
one tracks a live corpus and the other must freeze for results to stay comparable.

**Nothing is blocked.** The four requirement defects this design pass raised were settled on
2026-09-02 (requirements commit `afe4aef`) and the plan is built to the amended text throughout
(§6). The gate's three open questions were settled the same day and are folded in: **12 distinct
tool-caller scripts × 1 run at temperature 0** (§3.8.4), **`guard-judge` has no `headlineMetric`**
(§3.8.2, with §3.3 making a headline-less pack a first-class case), and **S3's self-check is a
diagnostic, never a gate** (S3 done-condition 5).

**S0 is delivered** (commit `0522ffd`). **S1 is built and not yet closed**, which is a different
state and the plan says so deliberately. Engineering gates:
`docs/reviews/small-model-benchmarking-impl.md`; statistics gates:
`docs/reviews/small-model-benchmarking-ml.md`.

| S1 event | commit | outcome |
|---|---|---|
| first delivery | `ab91419` | 233 tests |
| engineering gate, pass 1 | `d6c4997` | *needs changes* — 1 blocker, 6 majors |
| statistics gate, pass 1 | `6cfafaa` | *needs changes* |
| fix round | `3ad27d3` | **296 tests**; all 18 engineering findings fixed, both gates addressed |
| engineering re-gate, pass 2 | `8b7b60a` | **approve with suggestions** — 0 blockers, 1 major, 2 minors, 2 nits; **S2 may be dispatched** |
| statistics re-gate, pass 2 | `d79acca` | **needs changes** — one blocker still open |
| second fix round | `95b4c88` | **314 tests**; the floor moves to the unadjusted α, McNemar becomes a veto |
| statistics re-gate, pass 3 | `95b4c88` | **needs changes** — 0 blockers, 1 major, 2 minors, 4 nits; note revised to **v1.7** in the same pass |
| engineering re-gate, pass 3 | `95b4c88` | **needs changes** — 1 blocker (impl-gate P3-1), 6 majors, 5 minors, 3 nits |
| third fix round | `d55f4d8` | **353 tests**; both gates' Pass 3 findings closed |
| engineering re-gate, pass 4 | `e8bedce` | **approve with suggestions** — 0 blockers, 5 majors |
| statistics re-gate, pass 4 | `27501c9` | **approve with suggestions** — 1 major (M-ML-8), 4 minors, 2 nits; note revised to **v1.8** |

So: **S2 may start, S1 may not be marked done.** The statistics blocker that was open at pass 2 is
closed — the note ruled in v1.7 — and both pass-3 rounds route to `tdd-engineer`, except the two
items routed here and closed by this revision (statistics n-ML-7; the engineering pass's
stage-attribution item, now §5's table). Every open finding lands inside `stats.py`, `report.py`,
`cli.py` and their tests — none of which S2 constructs — so S2's runner, packs and adapter are
unaffected; if a fix moves a `verdict()` input, that is the `tdd-engineer` round's work, not a
change to what S2 hands the report (§7 rule 2). Read S0's and S1's own text before starting: they record
what shipped rather than what v1.1 asked for, including the one real smoke test and the
working-directory rule that every done-condition in §4 depends on.

**Pass 4, and what it changed here** *(v1.8)*. Both gates returned **approve with suggestions** with
**no blocker**, having each been asked directly whether their residue blocks S2; both said it does
not, and both independently named the same constraint — land the two nets before **S3**, the first
stage that produces real scored data. Exactly one Pass 4 finding was routed to this document:
**impl-gate P4-4**, which overturned the S1 fix round's deferral of the `aggregates`-versus-`items` cross-check
to S2. It is closed here as **S1 done-condition 10** and **§5 test 11c**, and the gate's own open
question 1 — whether S2's scorers compute `aggregates` from `items` in one pass — is answered in
**§4 S2**: they do, and the S1 check stays anyway, as the net under that seam rather than a
substitute for it. Everything else in Pass 4 is code, and none of it changes a signature S2 wires
against.

**What S2 inherits from the two gates**, stated here rather than left in the reviews:

- **`RunResult.designEffect` / `.basis` are required with no default** — the runner sets both (§4 S1).
- **`BinaryMetric.unit` is required with no default**, so every S2 scorer states its denominator
  unit. There are no stored records to migrate (`results/runs/` does not exist yet); that is free
  now and would not have been one stage later.
- **The multiplicity surface is `holm_steps`/`HolmStep`/`verdict(holm_tested=)`**, consumed by a
  two-pass `compare_report` (§4 S1). `holm_thresholds` is gone; nothing may call it.
- **`load_pack(...).ref().contentHash` is never `None`** — §3.3's totality boundary, asserted in S2's
  done-condition; and `validate_pack` reuses `packs.check_sampling_contract` rather than
  re-implementing the rule.
- **The latency percentile is decided** — `-ml` **§11**, which closes §6 R-13 and settles the
  estimator, the single implementation, the denominator, both floors and the published block. S2
  implements it and cites it; **no figure, floor or string from §11 is restated in this plan**.
  What S2 owes on the plan's side is the `LatencyBlock` and its nine rules (§4 S2), §3.5's
  `index.csv` columns including **`latencyMsMax`** and **`callCount`**, and the three dispositions
  §11 depends on: a
  load-contaminated item loses **`latencyMs` only** and keeps its `stats`-derived siblings under
  their own coverage (§11.4 — measured, and the reverse of what v1.9 first wrote); **a scored call
  that returns no response — timed out or failed — stores no latency figure at all** and is counted
  in `latencyWithheldForNoResponse`, from **two** distinct `withheldFor` item states (v1.11,
  plan-gate P5-6); and `unexplainedMs` withholds above §11.5.1's threshold, which
  closes R-14's residual. **From v1.27 all of that is denominated in two units, not one** — the
  wall clock per item, the three `stats`-derived figures per call — because a `tool-caller` item is
  a turn of `I(t)` calls (§3.6, §4 S2). **`-ml` §11.2's estimator is a change to shipped S1 code, not only to S2's**
  — §4 S1e Table C.

**Plan gate Pass 3, and what it changed here** *(v1.9)*. `docs/reviews/small-model-benchmarking.md`
`## Pass 3` (`ff499d5`) returned **needs changes** against v1.8 — 2 blockers, 6 majors, 4 minors,
1 nit — and **all thirteen are closed in this revision**, none carried, per the stakeholder's
standing principle that defects are not passed to later stages. Two shaped the document rather than
patching it: the blockers were both instances of one gap — **§3.4.4a was written as a source-of-truth
section while governing one of five sources** — so v1.9 fixes the general case (the source table, the
capture order, `callSurface`) rather than the two symptoms. Two more, **G3-6** and **G3-7**, are
defects in DC-10 itself and were relayed to the `tdd-engineer` building it while this revision was
being written; §4 S1 DC-10 now states the corrected predicate and the caught raise path, and the
implementer's deviations are reconciled against it rather than the reverse.

**Plan gate Pass 4, and what it changed here** *(v1.10)*.
`docs/reviews/small-model-benchmarking.md` `## Pass 4` (`bb0cacf`) returned **needs changes** against
v1.9 — 3 blockers, 5 majors, 5 minors, 1 nit — and **all fourteen are closed in this revision, none
carried**, per the stakeholder's standing principle that defects are not passed to later stages. Two
shaped the document rather than patching it. **`plan-gate P4-1`** — every ms-named field sourced from
a seconds-valued one, in a table that contradicted itself between adjacent rows — is answered by a
**unit boundary stated once** (§3.6) and normalised at the transport boundary, rather than by
correcting the two rows. **`plan-gate P4-2`** named a *pattern*, not only its instance: an edit list
over shipped code with no completeness property, wrong twice. The answer is §7 rule 5 and §4 S1e's
four grep-pinned tables, which is why this revision's own edit lists are commands rather than lists.
`plan-gate P4-8` is the one finding whose fix went **further than the gate asked**: rather than
scoping §4 S2's sentence and leaving DC-10's pooled-metric residual standing, the plan closes it —
a pooled metric cannot be a verdict metric (`-ml` §4.4 forbids its interval) and a pooled
denominator is declared per item, so DC-10's selector becomes total. **No finding is carried, and no
residual is disclosed in its place.**

**Plan gate Pass 5, and what it changed here** *(v1.11)*.
`docs/reviews/small-model-benchmarking.md` `## Pass 5` (`b9964d1`) returned **needs changes** against
v1.10 — 2 blockers, 4 majors, 4 minors — and **all ten are closed in this revision, none carried**,
per the stakeholder's standing principle. Three needed a method ruling and got one (note **v1.14**,
`ca69cb1`); the other seven were plan edits available the day the gate was written, which is what the
gate said. Two shaped the document rather than patching it. **`plan-gate P5-1`** is this project's
own warning made concrete — *the fix that reopens a closed finding*: the sentence written to close
`P4-5` applied §3.6's tool-calling gate to every model run and so refused every embedder arm, which
is G3-1 arriving a second time, so the fix is a **scope stated on both sides** (§3.4.4a step 3a and
§3.6) rather than a word changed in one. **`plan-gate P5-2`** is the more consequential: §7 rule 5,
adopted to stop a *third* incomplete edit list, claimed a completeness property it does not have.
The rule is kept — it is the right instrument — and its property is **restated honestly** as a
no-forgetting guarantee over token-carrying sites, with two obligations that follow from the
restatement (a second named command for every token-free site; a second-form residual wherever the
token survives) and one prohibition (a residual must not fail on a *correct* edit). Table B gained
three commands and four residuals under the corrected rule, and DC-12 now asserts what a residual
proves rather than its converse.

| Finding | Sev. | Closed by |
|---|---|---|
| **P5-1** — step 3a's unscoped eligibility gate refuses every `model:embeddings` arm | blocker | §3.4.4a step 3a scopes the two refusals separately — the cross-check on **every** `model` run, §3.6's gate **on a `tool-caller` pack only**; §3.4.4a's `callSurface` bullet writes the cross-check's predicate out and names it the universal pre-load refusal; §3.6 states the gate's scope and why its redundant half is kept anyway; §4 S2's done-condition gains the **negative** assertion — the same `embeddings` entry on an `embedder` pack is admitted — which is the only one that fails when the scope is missing |
| **P5-2** — rule 5's completeness property is false and DC-12 asserts it; Table B is incomplete | blocker | §7 rule 5 restated: **no-forgetting over token-carrying sites**, not completeness, with (a) every token-free site covered by a second named command, (b) a second-form residual wherever the token survives, plus two bounds — a command reaches a construct's **body** through its head, and a residual that fails on a correct edit is a trap rather than a check. §4 S1e Table B: **six** enumerating commands (three new — `arm_kind` 18 lines, `REQUIRED_BY_SCHEMA` 22, `EXPECTED_MODEL_SCHEMA_1` 3), eleven new site rows with a *found by* column, and **four** residuals, two of them second-form (`REQUIRED_BY_SCHEMA[1]["model"]` 3 → 0; `{"model", "deterministic"}` 2 → 0), with `armKind`'s **absence** of a residual stated rather than faked. DC-12 reworded to the sound direction only. *(Judged rather than applied: the review's second proposed residual — `arm_kind` filtered on `"model"`, 2 → 0 — is **rejected**, because `armKind` keeps the value `"model"` and `conftest.py:148`'s `arm_kind: str = "model"` can survive a faithful edit, so that residual would fail on a correct one. The two adopted in its place were verified against `5878014`.)* Also corrected: two line numbers v1.10 pinned wrong — `ARM_KINDS` is `fingerprint.py:137`, the membership test `:162` — a *counts at a named commit* claim that did not reproduce |
| **P5-3** — Table D leaves `paired_cluster_bootstrap` unreachable and undispositioned | major | Routed to `data-scientist`, because a public statistical function is not deleted on the architect's judgement, and ruled in note **v1.14** §3.4 Rule 4: **keep**. Table D gains a row with its own two enumerating commands (13 and 8 lines) and the disposition **no edit**, naming `paired_cluster_bootstrap` as §3.2d's **entry point** for every continuous verdict and `paired_bootstrap` as its **engine**, under the same discriminator `sampling.seed` carries. §4 S1's signature block carries the entry point explicitly, because wiring a continuous verdict straight to the engine yields a correct interval that silently ignores a declared design effect |
| **P5-4** — §3.4.2 still states v1.8's single catalog read and calls v1.10's sequence a validation failure | major | §3.4.2's capture-ordering paragraph rewritten: the order is **cited** to §3.4.4a (§7 rule 4) and only its field consequence restated — the catalog is read twice and `loadedContextLength` belongs to the **step 8** read — with the superseded sentence recorded as what it was, rule 4's own failure shape inside the revision that wrote rule 5 |
| **P5-5** — the pairing is one note revision stale and v1.13's co-presence invariant is uncarried | major | Ruled in note **v1.14** §11.4: the **conservative single count**. §4 S2 gains rule **(iv-c)** — an item carrying `stats` whose `promptTokens` is absent or `≤ 0` leaves `statsCoveredCount` **and all three sibling medians**, both halves load-bearing — stated as a **disposition that raises nothing**, with the note's rejected alternative and its reversal trigger named. Rule (iv)'s identity becomes an **inequality**, the one-pass recomputation being the equality worth asserting. Swept: §4 S2's `statsCoveredCount` definition, Appendix A's `LatencyBlock` row, the rule count (eight → nine) and §5 test 15b. §7 re-paired to **v1.14** |
| **P5-6** — `censoringExact` clause 1 is unevaluable, so a timeout-only run prints the weaker string | major | Ruled in note **v1.14** §11.5.1: clause 1 survives, and the reason is substantive rather than mechanical — a timeout is a *censored* observation, a 40 ms failure is a *missing* one, and the predicate gains an explicit **false** branch for the second. `ItemTiming.withheldFor` becomes `"load" \| "timeout" \| "no_response"`, while **`latencyWithheldForNoResponse` stays one counter** over the last two: three item states, two counts (§4 S2 (vi)). Swept: §3.6's timeout clause (ii) and fourth disposition, §4 S1's `ItemTiming` and DC-11 (three absent cases → four), §4 S2's `censoringExact` bullet, Appendix A, and §5 test 15b against all four of `-ml` §11.10 test 7b's branches |
| **P5-7** — rule (i)'s "the three figures" is under-determined and reads against (iv-a) | minor | §4 S2 (i) names the three **wall-clock** figures, and adds the clause: the four `stats`-derived figures are `None` under **either** a gate refusal or (iv-a)'s absence-of-input, the block does not distinguish the two, and `statsCoveredCount` is what tells them apart |
| **P5-8** — §3.6's unit boundary raises on an absent `stats`, which (iv-a) requires to be reachable | minor | §3.6's unit boundary gains a bullet: each derived field is `None` when its source key is absent, **never `0`**, and `ChatResult` construction **never raises** on a missing or partial `stats`; same for `promptTokens`. Swept into §4 S2's `ChatResult` paragraph and Appendix A's row, and asserted in §5 test 15b |
| **P5-9** — §3.6 says a no-response item carries "no timing at all" while (vi) reads its `withheldFor` | minor | §3.6's fourth disposition and its timeout clause now read **no timing figure**, but an `ItemTiming` whose only populated field is `withheldFor`. `timing is None` keeps its `iff` meaning — the **arm** produces no timings at all. §4 S1's `ItemTiming` comment, Appendix A and §5 test 15b swept |
| **P5-10** — the `sampling.seed`-in-the-fingerprint gap has a deadline the note does not carry | minor | §3.4.2 states the plan-side answer: the seed reaches the fingerprint **transitively through `packContentHash`** — it is a manifest field, the manifest is inside `content_hash(root)`'s input (§3.3), and `packContentHash` is `REQUIRED_NONEMPTY` on all three profiles — so a differing seed yields a differing hash and AC-3's banner already fires. No thirty-first field; the cost (a reader needs the pack in hand) is recorded; **resolved before S3**, while the set is still free. The note's own §3.2d wording stays `data-scientist`'s |

**One defect neither gate reached, closed in the same pass.** Note v1.14 attached a condition to
P5-3's keep: `stats._widen` clamps to `[-1, 1]`, which is right for the difference of proportions it
was written for and **wrong for `sep_z`**, whose per-query differences are unbounded — the verdict
survives the clamp, the printed interval does not, and the point estimate can land outside its own
interval. It lands as **§4 S1e Table E**: an enumerating command, a second-form residual over the shipped
clamp expression *(replaced at v1.17 by a third-form pair — impl-gate F1 — because the form v1.11
chose reads clean on the half-application; the table carries the current pair and the three-state
scoring)*, the clamp **required with no default** rather than defaulted to the
value that is wrong for one metric, a test in §5 item 6 that reproduces the defect rather than the
fix, and — at v1.11 — the only deadline in §4 S1e, gating §4 S3 done-condition 2. *(**That
deadline is withdrawn at v1.12.** Note v1.15 rules `separationZ` **reported rather than verdicted**,
so the clamp is due with the `sep_z` *comparison* and not with S3; the gate that replaces it is
Table F on S3 done-condition 1 — see Pass 6 below.)*

**Plan gate Pass 6, and what it changed here** *(v1.12)*.
`docs/reviews/small-model-benchmarking.md` `## Pass 6` (`afca8e0`) returned **needs changes** against
v1.11 — 1 blocker, 2 majors, 2 minors — and **all five are closed in this revision, none carried**.
The pass confirmed all ten Pass 5 dispositions against the tree, and **upheld both of v1.11's
adjudications**: the rejection of `P5-2`'s second prescribed residual was correct, and §7 rule 5's
second formulation is sound as stated. Neither is reopened here. One finding needed a method ruling
and got one (note **v1.15**, `0ad0e7a`); the other four were plan edits available the day the gate
was written.

**Two of the five are the same lesson and are worth naming as one.** `P6-2` and `P6-3` are not
disagreements with rule 5 — they are v1.11 **breaking rule 5 inside the revision that wrote it**.
Table B's fourth residual is the trap 5(b) forbids, published one table after this document rejected
Pass 5's residual for being exactly that; Table E's site list transcribed a command's output without
reading it, which is 5(a)'s *superset, not a site list* edge, published in the table added to
demonstrate 5(a). The counter this revision adopts is the gate's own and is cheap enough to be
standing practice: **after writing a residual, name the authorised implementation that would make it
non-zero; before publishing a table's site rows, run each of its commands once and read every line
it returns.** Both were run for all six tables at v1.12 — and for Table G's three commands and one
residual at v1.13 — so every residual in §4 S1e is shown
**non-zero today and zero after a faithful edit** — the second half being the check the last two
revisions each shipped one residual without.

| Finding | Sev. | Closed by |
|---|---|---|
| **P6-1** — no field on the record can carry a per-item continuous value, so the embedder's only verdict metric is unbuildable and the shipped comparison path fails silently in two directions | blocker | Split as the gate asked, and the record half ruled **S1** rather than S3 on §4 S1e's own *free only now* argument (the gate's open question 1). **(a)** Method ruled in note **v1.15** §3.2d and landed as **§4 S1e Table F**: `ItemResult.measures: Mapping[str, float]` beside `counts` — never a widened `counts` — with `scored_value` beside `scored_outcome`, `scored_outcome` **raising** on a `measures`-resident metric, one map per metric name, no domain constraint at the carrier, and `0.0` as a measurement. Six enumerating commands, eleven site rows, two second-form residuals. **(b)** §4 S1's `compare_report` block states the continuous verdict path end to end — kind resolved from the aggregate type and cross-checked against the item map, one difference per **analysis unit**, one-arm-only units into §4.3's asymmetry tally, `paired_cluster_bootstrap` with `clamp` derived from a new required `ContinuousMetric.support` *(v1.14: on the verdict path that derivation moved inside `-ml` §3.4 Rule 8 — see §4 S1)*, no Holm rung, `-ml` §3.2e's strings 4 and 5 cited not quoted, `DecidedBy` gaining `"paired-bootstrap"`; `named_metrics()` returns `separationRaw`/`separationZ` and a new `DistributionSummary` carries §5.2's median and p10, so `sep_z` reaches a table at all. **(c)** Raised and answered: §3.2e now publishes **two** strings, not one, and §3.3 rules a family homogeneous in kind. DC-10 gains a third arithmetic and the kind cross-check; DC-13 and §5 test 11d carry the note's four proof obligations plus the plan's two. **And Table E's deadline is corrected by reversal** — `separationZ` is reported, not verdicted, so §4 S3 done-condition 2's gate on the clamp is **withdrawn** and the gate that means something is Table F on done-condition 1, which nobody has to remember because without the carrier no result of that pack is storable |
| **P6-2** — Table B's fourth residual fails on an implementation Table B itself authorises | major | The residual is **narrowed to the mapping's key set** — `grep -rFn 'set(REQUIRED_BY_SCHEMA[1]) == {"model",' modelbench tests --include='*.py'` → **1 → 0** — which is the site the edit genuinely retires and which no decoupled `ARM_KINDS` can match under **either** branch of the `fingerprint.py:137` row. Scoping the wider form to `tests/` was considered and **rejected**: a test pinning the decoupled `ARM_KINDS` by value, which is that row's whole point, puts the same display back inside `tests/`. `test_fingerprint.py:234` needs no residual of its own — the retired name `FORBIDDEN_BY_ARM_KIND` is on that line, so residual 1 reaches it. Both branches of the `:137` row survive, and the row says so |
| **P6-3** — Table E names four tests that do not call `_widen`, misses the three sites its edit breaks, and contradicts Table D | major | Table E gains a **second enumerating command** (`paired_cluster_bootstrap(` → **5** lines: `stats.py` `:162`, `:263`; `test_stats.py` `:1270`, `:1323`, `:1330`) with per-file counts. Its four `_widen` matches in `test_stats.py` are **named as non-sites** — `def` lines matched on a test *name* — with the general lesson stated: a command's output is a superset of the sites, never the site list. The three real call sites get a row each naming the clamp each passes (`(-1.0, 1.0)`, all three being differences of proportions), `:1323` is marked as the √DEFF exactness test Table D says must survive, and the new defect-reproducing test is placed at **`paired_cluster_bootstrap`, the public surface**, because `_widen` is private. Table D's row is scoped to *neither is touched **by this table***, its own three-versus-four test count corrected, and the Table D/Table E ordering on `stats.py:263` is stated as faithful either way |
| **P6-4** — §4 S1e is titled "**the** edit set" while DC-11's 26-line `latencyMs` change sits outside all of its tables | minor | §4 S1e's preamble scopes the then-six tables to the **retiring and re-keying** edits — the ones a site can be missed in silence from — and names the remainder outright: §4 S1's signature block plus DC-11, where `latencyMs` is a required positional at 26 lines across five files and `ItemTiming`/`LatencyBlock`/`withheldFor`/`wallClockMs` appear 0 times, so every construction site breaks loudly and no residual has anything to prove. §7 rule 5's closing paragraph carries the same scoping. **Table F is the one adding edit still tabled**, and it states on itself why: its field is defaulted, so nothing breaks, and the behaviour that changes is `scored_outcome`'s |
| **P6-5** — Table B's second residual is prose in a table whose thesis is that a residual is a command with a count | minor | Written as the command it already was: `grep -rFn 'frozenset(FORBIDDEN' modelbench --include='*.py'` → **1 → 0** (`fingerprint.py:137`, the only match, verified at `5878014`). The row also records what it adds over residual 1, which is the reason it is not redundant: a rename that leaves `ARM_KINDS` coupled to the *renamed* mapping satisfies residual 1 and leaves this one at 1 |

**Nothing in Pass 6 is carried, and nothing is blocked on unbuilt work.** `P6-1` is the only one that
needed anything outside this document, and what it needed was a **method ruling**, which §7 rule 3
routes and note v1.15 delivered in the same session. Two items are raised to `data-scientist` and
neither blocks an implementer: `stats.py`'s continuous-verdict producer needs a §3.4 signature the
way `verdict()` has one, and §3.3's word *"validate"* needs to say which surface enforces the
homogeneity rule. Both are named in the *Version pairing* block above. *(**Both are closed at
v1.13** by note v1.16 — Rule 8 is the producer's signature, and §3.3's own sentence moved to
`compare_report` pass 1. Two clauses of the P6-1 row above are superseded there rather than rewritten
here, this table being a record of what Pass 6 closed: the loop calls Rule 8's producer, not
`paired_cluster_bootstrap` directly, and the mixed family is refused **whole** rather than by
excluding its offending members.)*

**Plan gate Pass 7, and what it changed here** *(v1.14)*.
`docs/reviews/small-model-benchmarking.md` `## Pass 7` (`b6222c6`) returned **needs changes** against
v1.13 — 1 blocker, 0 majors, 2 minors, 0 nits, the smallest set of the seven passes and the first in
which no finding is a defect *of* the mechanism — and **all three are closed in this revision, none
carried**. It ruled the plan **implementable** and S2 **not yet dispatchable**, for `P7-1` alone; it
re-checked all five Pass 6 dispositions and every residual against the tree and found them holding;
and it settled four questions this revision does not reopen — Table G's `means` scoping is
principled and Table G belongs as its own table, delta 1's two negative assertions catch the DC-10
mis-build against all four plausible ones, rule 5's second formulation and the earlier residual
adjudication stand, and the `support` seam's non-blocking classification holds on the note's
replacement argument.

| Finding | Sev. | Closed by |
|---|---|---|
| **P7-1** — `DistributionSummary` has no stored form; Table F retypes two published figures to it and, of the **four** shipped sites that break under the retype, names **one** — the three it misses including the one that fails **silently** | blocker | §4 S1e Table F decides the stored form — the `"distribution"` tag, its six keys, `support` stored on both continuous types and read with no `.get`, an unrecognised tag **raising** through one tag-set home, and `benchSchemaVersion` **not** bumping with its reversal trigger. Three site rows (`results.py:354-359`, `:385`, `:584`) and the tag half of the `:369`/`:373` row; commands **7** (`\.mean` → 3) and **8** (`"continuous"` → 2), whose union is exactly the four sites, plus command 6 corrected from a variable pin to a type pin (3 → 6); a third residual (`{"binary", "continuous"}` → 1 → 0), which the half-application that ships the silence cannot pass; **DC-13(f)** and §5 test 11d's stored half; and §4 S1e's preamble corrected — Table F **is** constructed by S2, which is why the S1 fix round precedes S2 |
| **P7-2** — Table G retires two literals under one residual, and its new test passes on the half-applied edit | minor | The symmetric residual (`_percentile(means, 97.5)` → **1 → 0**, verified) and a test asserting **both bounds move outward** rather than that the width grows. Generalised rather than patched: §7 rule 5(b) now requires a table's residuals to distinguish each retired literal, DC-12 asserts it per table, and **Table E — the one other table with that shape — gains the same second residual** (`min(1.0, point` → **1 → 0**; *both of Table E's are replaced at v1.17 — impl-gate F1 — the shape being right and the **form** blind on the half-application*), its half-application having left the upper clamp in place and shipped Table E's own defect intact |
| **P7-3** — the `exploratory — no significance claim` label has no shipped home that admits family members | minor | §3.3 (iv) names both sites — `report.py:767`'s family filter and `:744-751`'s headline fallback — and decides both questions the implementer would otherwise have decided: the filter widens to *not in family **or** this family's verdicts were refused whole* (never to "not verdicted", which would move a no-paired-data member out of `_NO_PAIRED_DATA`), and the line stays **name-only**, the Arms table already printing every member's figure. §5 test 11d asserts at those sites |

**Nothing in Pass 7 is carried and nothing is blocked on unbuilt work** — the gate states that of all
three, and all three were plan edits available the day it was written. The one part that is a
judgement rather than a transcription, `benchSchemaVersion`, is ruled in Table F with the exact
condition that would reverse it. **Pass 7 asks for a narrow re-check of this revision rather than an
eighth full gate**, and fixes its scope: these three findings plus note **v1.17**'s two plan-side
deltas, both of which land here (*Version pairing* above).

**Plan gate Pass 8, and what it changed here** *(v1.15)*.
`docs/reviews/small-model-benchmarking.md` `## Pass 8 (narrow)` (`4cd22b9`) was the narrow re-check
Pass 7 asked for — the v1.14 delta only, 25 hunks, against note v1.17 and the shipped tree — and it
returned **needs changes** against v1.14: **1 blocker, 2 majors, 0 minors, 0 nits**, and **all three
are closed in this revision, none carried**, per the stakeholder's standing principle. It closed all
three Pass 7 findings, verified all eleven counts v1.14 cites and reproduced every one, and judged
both of the questions it was set — commands 7 and 8 enumerate the class, and rule 5(a)'s new
companion is the right statement of it. **What it found instead was a defect no pass had swept
for**, and its shape is the lesson: **a cross-table collision**. Each of Tables C and G was sound
read alone; read together, C's edit alone drove both of G's residuals to zero, so DC-12 would have
passed on an implementation that never applied G — the trap §7 rule 5(b) forbids, inside the plan
that wrote the rule. §7 rule 5(b) is therefore **generalised from a property of a table to a
property of the round**, which is the same move P7-2's finding forced one revision earlier.

| Finding | Sev. | Closed by |
|---|---|---|
| **P8-1** — Tables C and G both edit `stats.py:159` and prescribe incompatible forms for it, neither names the other, and **both of Table G's residuals go to zero from Table C's edit alone** | blocker | The statistics half was raised to `data-scientist` under §7 rule 3 and **ruled in note v1.18 §11.2.2**: the level is an exact `Fraction`, `stats.py:159` is **not** exempted, and §11.10(3) becomes package-wide. The plan half, all in §4 S1e: Table C's signature row moves to `percentile(values, *, level: Fraction)` with the four `LEVEL_*` constants; its `:159`/`:292` row **splits**, `:292` keeping literal levels and `:159` becoming the collision row that **prescribes its own post-edit spelling**, because Table G's residuals are stated over it; the collision is named on **both** tables' rows and the order **fixed — C then G**, unlike Tables D and E on `stats.py:263` where either order is faithful; Table G's element type becomes `tuple[Fraction, Fraction]`, its `k = 2` test's pair is `(Fraction(1, 80), Fraction(79, 80))` — a pair the retired unit could not express — and **its two residuals are re-derived, not re-scoped**, over the surviving spelling, with their *before* stated as the intermediate state and DC-12 re-running them at the end of the round. Swept for the retired representation: §4 S1's two signatures and its new `percentile` entry, Table D's closed-form row, §3.8.1 and Appendix A |
| **P8-2** — §3.3 (iv) ends "Nothing else in the verdict path changes" and two lines contradict it: `report.py:722-724` prints Holm–Bonferroni **applied** where no ladder ran, and `_decision(None, step)` at `:329-330` returns *no verdict — no paired data*, false for a refused family | major | §3.3 (iv) names both, takes both decisions, and replaces the claim with an **enumeration** over the eleven emission sites of `report.py:606-780` — five of them, in four blocks, state something a refusal falsifies, and the other six are named. **(3)** The `Family-wise error control` block renders only where a Holm ladder actually ran, with a one-line replacement per condition; relabelling its cells is rejected, every column in it being a ladder artefact. **(4)** `_decision` gains **no** third input state, because (3) makes the false string *unreachable* — it has exactly one call site, inside that block — with (3) named as the trigger that would reverse it. DC-13(e) and §5 test 11d gain one assertion per site, both on an **absence**, plus a second fixture |
| **P8-3** — Table C's residual is scoped to `results.py` while its edit retires three further `_percentile` sites in `stats.py`; the half-application leaves both bootstraps on the rejected estimator with the residual reading zero, and DC-12 records the table as already satisfying rule 5(b) on a count of three where there are six | major | Table C gains the `stats.py` half (**3 → 0**) and `-ml` §11.10(3)'s package-wide identity check (**2 → 1**, a stated target that is **not** zero, with the surviving line named), plus a **row** for the seventh line its enumerating command returns and an explicit statement that the seventh gets no residual, with the reason — rule 5(b)'s own named alternative. The derivation `3 + 3 + 1 = 7` is written out so the row list is checked against the command's count rather than against the prose beside it. DC-12's partition is corrected — C moves out of *already satisfies* into *gained its second at this revision* — and its wording moves from "is zero" to "hits its stated target", which the new residual required. §7 rule 5(b) gains the non-zero-target clause |

**Nothing in Pass 8 is carried and nothing is blocked on unbuilt work** — the gate states that of all
three, and the one part that needed a method ruling got one before this revision was written. **The
two open questions the gate raised are both answered**: `percentile` gains a rational level rather
than an exemption (note v1.18 §11.2.2, the §7 rule 3 raise closed — **no raise is open again**), and
the `Family-wise error control` block is **suppressed with a one-line replacement** rather than
relabelled, in §3.3 (iv), which is where this plan has been taking presentation calls with a
reader-facing consequence.

**Three changes beyond the three findings, disclosed rather than folded in** *(the discipline v1.14
used for Table E)*.

1. **§3.3 (iv)(3)'s decision covers two conditions, not the one P8-2 names.** It is stated over *no
   ladder ran*, so it reaches both a family refused whole and an **all-continuous `k > 1`** family,
   which takes its correction in the interval and never in a ladder. That is not new ground — it is
   the **same shipped line**, `report.py:722-724`, under the other condition that reaches it, and
   the condition is authorised by this section's own second paragraph and restated by §4 S1's loop
   (`holm_steps` is not called on a continuous family). A fix scoped to the refusal alone would
   have left the same line printing the same false sentence for exactly the family §4 S1e Table G
   exists to serve.
2. **Table C's row now says which of `-ml` §11.10's items land with it, at S1** — items 1, 2a, 2b
   and 10, item 3 being its own third residual, the rest being `latency_summary` and rendering and
   therefore S2's. Same finding applied: the signature this table lands **moved** in this revision
   and v1.18 added two refusals to it, so the tests that pin it moved with it; Table D's row already
   carries the same sentence for the closed form's five.
3. **DC-12's seven-table rule 5(b) sweep is re-run per table rather than inherited, and Table A's
   result is written out.** The brief for this revision made the sweep standing rather than a
   reviewer's catch; six tables reproduce their prior answer and **A** did not read off the
   enumeration — its residency-**element** row retires two keys under one residual, and what covers
   the second is DC-1's element-shape assertion, which is rule 5(b)'s own named alternative rather
   than a gap. Recorded in DC-12 because a sweep whose result is not written down is a sweep the
   next revision runs again.

**Implementation gate Pass 5, and what it changed here** *(v1.16)*.
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 5` (`cbfcda9`) is the **code** review of
`8fc2341`, the first of S1e's three implementation units (Tables A and B), and it returned **needs
changes** — 0 blockers, 4 majors, 2 minors, 1 nit. **Four of the seven are the implementer's** and
are closed in `model-bench/` code, not here: P5-1 (`callSurface` collapses *absent*, *empty* and
*null*), P5-2 (`to_dict`'s deterministic omission is unpinned), P5-4 (the half-swap test asserts
non-emptiness, which the missing key alone satisfies) and P5-7 (`model-bench/AGENTS.md`'s third
copy). **Three are this plan's and all three close in this revision, none carried**, per the
stakeholder's standing principle; none is blocked on unbuilt work, and the gate says so of all
seven. The unit's substance is **not** in dispute: the re-key is correct, the residuals were all
observed at their targets, and six of the gate's eight constructed counter-implementations died.

| Finding | Sev. | Closed by |
|---|---|---|
| **P5-3** — DC-1 names `sizeBytes` as a case that must be refused while Table A's second residual counts that name across `tests/`, so the assertion and the residual are mutually exclusive; the `sizeBytes` half is asserted nowhere and cannot be, and a counter-implementation whose extra-key loop excepts that one name passes all 472 tests | major | **DC-1 restated over the element's *key set*** — exactly `{id, state}`, any other key set invalid, a rule no tolerance list can be written against — with the retired `lms ps --json` element named as the instance the suite asserts **by value**, both keys named. **Table A's second residual re-scoped** to `modelbench` plus `tests/conftest.py`: *before* **1** (`tests/conftest.py:40` at `5878014`, the sole match in either scope) → **0**, both re-run, so it still proves the fixture edit landed and is not the `modelbench/`-only narrowing the gate rules out. The assertion's literal is **prescribed** into `tests/test_fingerprint.py`, with the scope's other half stated — the retired spelling may not appear in the scoped files even as prose. **DC-12's Table A note rewritten over both halves.** Generalised rather than patched: §7 rule 5(b) names the **disowning mention**, and DC-12 gains a standing sweep over the residual *set* |
| **P5-5** — Appendix A's `unknown` is narrower than the code: `8fc2341` widened it to residency-element type errors and swept the module docstring, not the plan | minor | Appendix A's `FieldProblem` row widened from *a **discriminator** this build cannot interpret* to *a **value** this build cannot interpret*, with its **three** families written out — unrecognised discriminator, future `benchSchemaVersion`, type error inside a structured value — so a fourth is a decision rather than a drift |
| **P5-6** — `8fc2341` shifted `tests/test_results.py` by 36 lines, so Table C's site row points at the wrong line and the next unit implements Table C | minor | Table C's row **re-pinned `:507` → `:543`**, and the gate's instruction taken wider than the one line it named: the enumerating command was **re-run** at `612888c` (same **7** lines, same per-file split) and `results.py`/`stats.py` confirmed **byte-identical** to `5878014` by hash, so all six production pins resolve unchanged and the drift is the whole delta |

**One change beyond the three findings, disclosed rather than folded in** *(the discipline v1.14 and
v1.15 used)*. **DC-12's new per-residual sweep found a third instance of P5-3's shape, in Table C —
the table the next unit implements.** Its package-wide percentile-definition check was scoped
`modelbench tests`, and the pattern matches a test **function name**: `-ml` §11.10's acceptance items
land with that table as tests of `percentile`, so the natural spelling of the tests the table itself
prescribes would push the count above its stated target of 1 on a faithful edit. Re-scoped to
`modelbench`, which is what *package-wide* meant; *before* and target both unchanged, and `tests/`
defines no percentile or quantile function today. This is disclosed rather than deferred because a
defect found in the table the next unit executes is not available to carry.

**Two things this revision deliberately did not do.** It did not adopt the gate's flagged option of
moving the retired element into a `tests/data/*.json` fixture to slip the residual's
`--include='*.py'` scope — the gate flagged it as reading like routing around a check and left the
call here; scoping the residual to the files where the token genuinely retires answers the same
problem without teaching that move. And it did not sweep §4 S1e's other six tables for the
*half-application* shape a second time — DC-12 already re-runs that sweep per table at every
revision, and the sweep this revision **adds** is the different one P5-3 actually exposed, over the
residual set rather than the tables *(the gate's open question 2, answered: the one-line DC-1 reword
was not sufficient, and what the second occurrence warranted was a mechanical check, not a
re-reading)*.

**A `Landed:` convention is introduced with this revision** and is proposed rather than assumed:
§4 S1e is executed in units, so from `8fc2341` onward some of its tables are a record of completed
work and the rest are instructions, and a revision that re-pins a landed table cannot rely on a
reader knowing which is which. Tables A and B carry the first two lines; the convention and the three
things it deliberately does not do are stated once, in §4 S1e's preamble.

**The `stats.py` unit, and what it changed here** *(v1.17)*.
`cc28d48` delivered §4 S1e **Tables C, D, E and G** — the second of S1e's three implementation
units, and the last before Table F. It is **not** a gate: the unit's own report raised **three
findings, every one of them against a residual's *statement* and none against the code**, which is
the first time this plan's grep machinery has been the whole of a unit's finding set. All three are
closed here, none carried; none is blocked on unbuilt work. The suite went 487 → **519** and the
implementer reported thirteen mutations with eleven killed and **both survivors disclosed rather
than hidden** — the discipline §7 asks for, applied by an implementer to its own work.

| Finding | Sev. | Closed by |
|---|---|---|
| **F1** — Table E's two residuals match the **shipped** text, which a faithful edit necessarily rewrites, so a half-application written in the new spelling matches neither and the pair reads **0/0 — the same as faithful** | major | **Both replaced by third-form residuals**, stated over the text the edit *creates*: `max(clamp[0], widened[0])` and `min(clamp[1], widened[1])`, **0 → 1** each at `stats.py:261`. Scored on three states side by side in the table — before 0/0, faithful 1/1, half-applied **1/0** — so the pair now says which half was skipped. Replaced and **not** supplemented, since rule 5(b) forbids keeping a check that reads clean on the defect it was written for; DC-12's eighteen is unchanged. Generalised at **§7 rule 5(b): the third form**, with the trigger named — **a parameterising edit**, the only kind that rewrites an expression rather than a name or a value — and its cost stated, since a third-form residual is the one kind that pins a spelling. The obvious cheaper fix was **measured and rejected**: `max(-1.0,` and `min(1.0,` read **1** and **4** on the faithful tree, four unrelated clamps in `stats.py` carrying the same numbers |
| **F2** — Table C's third residual moves **1 → 2** once Table D lands, `-ml` §3.4 Rule 4's mandated `exact_paired_quantiles` matching its own pattern | minor | Target restated as **2 → 2 with both survivors named**, and the plan says plainly that **the count is no longer the check — the named line set is**; a *third* definition still moves the number, which is what keeps the command. Why not rename the function to reach 1: `exact_paired_quantiles` is the same operator on the **exact multinomial resample distribution**, not a second sample-quantile estimator (`-ml` §11.2 reason 2), and renaming would forfeit this residual's stated virtue — surviving a rename. The *no half-application passes* claim moves off the count and onto the two file-scoped residuals plus §11.10(3)'s identity assertion, where it was always true. Generalised at **§7 rule 5(b)'s named-line-set clause** |
| **F3** — Table D's first residual could not reach 0 on any correct edit: `def test_cluster_bootstrap_seed_…` is a **substring collision** on a test for the function this table **keeps** | minor | Residual becomes the **whole-identifier** form, `grep -rEn '\bbootstrap_seed' …` → **27 → 0**. Re-derived at `5878014`: the collision is **two** `def test_…` lines, not one, and both are named — the second (`test_report.py:1678`) is a **genuine site** already carried by this table's test row, not a collision. **`27 + 2 = 29`**, the enumerating command's own count, written as a derivation rather than a list |

**One thing this revision states rather than fixes, because it is not a defect.** Tables A and B's
pins were re-verified as still resolving at `8fc2341` and Tables C, D, E and G's at `cc28d48`, but
**no pin into any file under `model-bench/` resolves at `5878014` any more** — `cc28d48` moved
`stats.py`, `results.py` and `report.py`, the last three that were byte-identical. Every *before*
value in §4 S1e is now reachable only through `git show 5878014:<path>`, which is how each one cited
at v1.17 was re-derived. DC-12 records this, and it is a fact about the tree rather than an error in
the tables: an enumerating command's counts are a claim about a named commit, and the commit is
still named.

**And one earlier decision the landed tree has now vindicated, recorded because the evidence only
exists once.** v1.16 narrowed Table C's third residual from `modelbench tests` to `modelbench`, on a
constructed example. `cc28d48` then added **seven** `def test_*percentile*` names, so the unnarrowed
command reads **9** on the faithful implementation of its own table against a stated target of 2.
The narrowing was worth its paragraph.

**The clamp ruling, and what it changed here** *(v1.19)*.
`-ml` **v1.19** (`a707d09`) answers the question impl-gate Pass 8's **P8-1** routed to
`data-scientist` — whether the `(-1, 1)` support belongs on the conservative envelope's **arms** or
on its **composed** interval — and rules **composed**, as new §3.4 **Rule 4a**. This plan folds it in
as §4 S1e **Table H**, the eighth table and the first added since v1.13. The ruling's headline is
what scopes the work and is worth repeating once: **arms-versus-composed is immaterial to the
statistics and material only to the audit bullet** — every printed interval, coverage figure, width
and verdict is bit-identical either way — so Table H is a **reporting** correction, and the reason it
is urgent rather than merely free is the note's own measurement that the false sentence lands beside
a *positive* verdict at the design effect a determinism probe is most likely to return.

**The handoff premise was wrong in the direction that shrinks the work, and it was checked rather
than assumed.** v1.18's carried trigger said a move to *composed* would rewrite the expression Table
E pins and send both of its residuals to 0, obliging a re-derivation. It does not. Table E's pair
matches `_widen`'s **body**; Rule 4a changes what `envelope_arms` **passes** at two call sites and
leaves `_widen`'s parameter and body verbatim — `diff` against `cc28d48` reports the body identical,
and both residuals read **1** and **1** at `7f865e2`. So Table E stands as written, which this
revision says explicitly on the table rather than leaving a reader of the next revision to wonder
why it did not move.

| Item | Closed by |
|---|---|
| **impl-gate P8-1** — the `- decided by:` audit names the arm that did not bind once the clamp binds both; the tie-break is unpinned | §4 S1e **Table H**, the implementation spec for `-ml` Rule 4a: `clamp=None` on both arms, the composer clamping its own result, `bound_by` as a **three**-token set on a **strict** comparison, and a renderer that attaches no `p=` clause to a `support bound` token. Two enumerating commands, **ten** site rows and **six** residuals *(nine at v1.19; plan-gate P10-1 added the tenth)*. *(v1.21: the two commands' counts were restated here and are not any more — one of them moved under `93b0e42` within the day, which is the case §7 rule 5's new gloss clause is written from. They live on the table, once.)* The ruling deliberately disagrees with `P8-1`'s own suggested fix — attributing from the unclamped arms prints the same false sentence on the separating case — and the note carries the witness |
| **impl-gate P8-5** — the composition rule is written twice and `envelope_arms`'s docstring says it is not | Closed as collateral. `7f865e2` already shipped the `_compose` helper with **three** deletions from Rule 4a's version recorded in its own docstring at the seam; all three are additive and Table H lands them, so nothing is reverted. Table H carries a row for that docstring paragraph, because a docstring asserting the opposite of its own body is P8-5's finding one revision later |
| **The `Landed:` convention's first real test** | Table E's rows are a record of what `cc28d48` did, and one of them — *"both arms keep the identical clamp `(-1.0, 1.0)`"* — is now Table H's to change. The row is **kept, not rewritten**, with the supersession stated beside it; the current instruction lives on Table H's row for the same two call sites |

**Two things measured at design time that would otherwise have been found by a reviewer.** First,
Table H's arm-clamp residual is scoped to the **keyword argument** (`clamp=(-1.0, 1.0)`) and not to
the bare tuple, because the bare form returns **3** lines at `7f865e2` and the third is a sentence
inside `_widen`'s own docstring explaining why that default is refused — a **disowning mention**,
v1.16's trap shape, caught before it was written down rather than after. Second, two of Table H's
six residuals are **third-form**, because `_compose`'s edit trips §7 rule 5(b)'s parameterising
trigger exactly as Table E's did. **That is the first time this plan has applied that rule forward
rather than been corrected into it**, which is what the rule was written for.

**And the exact-text form has now been paid for twice, so §7 rule 5(b) states its third virtue.**
Table E's pair survived the Pass 8 fix round inserting a guard thirty lines above it and Table H
changing what its call sites pass, reading 1 and 1 throughout; a line pin in the same position would
have broken twice, which is what happened to Table C's `tests/test_results.py` row at impl-gate P5-6.
A residual has to survive other people's edits to be worth stating.

**Plan gate Pass 10, and what it changed here** *(v1.20)*.
`docs/reviews/small-model-benchmarking.md` `## Pass 10` (`913e159`) gated the **v1.19 delta only** —
§4 S1e Table H against note v1.19 §3.4 Rule 4a and the tree at `7f865e2` — and returned **needs
changes**: 1 blocker, 2 majors, 1 minor, 1 nit, **all five closed here, none carried**. All five are
**prose**: not one is an arithmetic error, and the gate re-measured Table H's six residuals and both
enumerating commands and reproduced them exactly. What it found is that the table's *hand-written
row list* was short and three of its rows instructed something the note contradicts — which is the
same class of defect §7 rule 5 was written for, one level up: the commands were sound and the prose
beside them was not.

| Finding | Sev. | Closed by |
|---|---|---|
| **P10-1** — the site list omits the one **shipped test** the edit falsifies: `test_neither_printed_bound_is_ever_tighter_than_either_arm` compares the printed envelope against `envelope_arms`' return directly, which is false once the arms come back unclamped | blocker | A site row, **pinned by test name and not by line** — a parallel unit is editing `tests/test_stats.py` — stating that the comparison moves to the **clamped** arms, `-ml` Rule 4a's own restatement of the conservatism property rather than a weakening, with the gate's measurement carried (**0 / 38 / 78** failures at DEFF 1.0 / 1.2 / 2.0 against a **0 / 0 / 0** control, first failing tables named). The row also records *why* it is a row: **no residual can see it**, all six being scoped to `modelbench`, while command 2 **does** return the line — so the two-command design worked and the transcription did not |
| **P10-3** — the `stats.py:413` row claims the post-edit spelling is "prescribed rather than left open" and never states it; the only statement is inside residuals 2 and 3, which pin locals the plan never fixes and which differ from Rule 4a's own `u_lo`/`u_hi` | major | The row **writes the two-line body out**, local names included and taken from the note, with the unpacking variant explicitly forbidden; and residuals 2 and 3 are restated over `SUPPORT_DIFF_PROPORTIONS[0]` / `[1]`, which contain **no introduced name** — still **0 → 1** each, re-derived from the `7f865e2` blobs. Generalised at **§7 rule 5(b)**: a third-form residual is only a residual if the table writes out the text it pins, because it is stated over text that does not exist yet and a command is not a specification |
| **P10-2** — the `report.py:338` row prescribes a rendering assertion 10 forbids, and omits that the renderer now needs the support *value* | major | The row states the note's string — a `support bound` token renders **with its boundary value**, `support bound (-1)`, and never with a level — names `SUPPORT_DIFF_PROPORTIONS` as the source, imported as `report.py` already imports the two `LEVEL_*` constants for the same expression, and corrects the last clause: assertion 10 pins the **correct** bullet, and `support bound, p=0.025` is what it kills. The table's own four-line summary of the ruling, point **(iv)**, carried the same gap and is swept with it |
| **P10-4** — two further docstring paragraphs state the pre-Rule-4a attribution flow and get no row; one is the sentence the plan claims the edit makes *true* | minor | The docstring row extends from one paragraph to **three** — `_compose`'s first and last, and `envelope_arms`' — and the over-claim is corrected: after Rule 4a `verdict()` **receives** an attribution rather than reading one, so of that sentence's two clauses the edit makes one true and the other false, and only the first is the deletion row's |
| **P10-5** — "fifteen of `test_stats.py`'s twenty `envelope_arms` lines are the arm-level assertions" is wrong twice | nit | Re-derived: fifteen is the file's **whole share** of the command's twenty, and of those **four** are arm-value call sites, the rest being one import, two `parametrize` ids, five lines of docstring prose and two precondition-raise calls Rule 4a does not touch. The command's coverage claim was never in question; the sentence describing it was not reproducible — and it is where P10-1 hid |

**What the gate confirmed, recorded so it is not re-defended.** Every consumer of `_compose`'s
widened return type *is* enumerated — it has exactly two call sites at `7f865e2` and both have rows,
and the `Verdict` construction correctly needs none. The `bound_by is None` path is invariant under
Rule 4a and its absence from the renderer row is right; the `strict=True` zip stays safe. Residual 1
discriminates a one-arm application and says which arm. And the **third-form trigger for residuals 2
and 3 is correct** — the gate checked it rather than accepting it, and found that a first-form
residual there really would have been a trap, because a faithful edit can keep the old composed
expression intact as a *sub-expression* of the new one. The §7 rule 3 raise this plan opened against
note v1.19's citation is confirmed and is the one item here that this document cannot close: it is
**blocked on `data-scientist`**, and nothing in the plan depends on it.

**What S2 additionally inherits from v1.10**:

- **`ItemResult.timing: ItemTiming | None` is the timing carrier, and `latencyMs` is a property over
  it** (§4 S1). A load-withheld item keeps its `wallClockMs`; only the admitted figure disappears.
  Every aggregate reads `latencyMs` and nothing reads `wallClockMs`. **v1.27 splits the record in
  two**: `ItemTiming` keeps the item's wall clock, an ordered `CallTiming` per call and
  `withheldFor`, with `callCount` and `unexplainedMs` as derived properties over `calls`; the five
  per-call scalars live on `CallTiming` and nowhere else (§4 S1, Appendix A).
- **`LatencyBlock` carries four more figures and one renamed count** —`ttftMsMedian`,
  `prefillMsPer1kMedian`, `tokensPerSecondMedian`, `unexplainedMsMax`, and
  `latencyWithheldForTimeout` → `latencyWithheldForNoResponse` — with nine invariants, all asserted
  against a recomputation from `run.items` (§4 S2). **v1.27 adds `callCount`** (`-ml` §11.4's
  `Y_calls`) and moves `statsCoveredCount` and rule (iv-c) to the **call**; rules (ii), (iii), (v)
  and (vi) stay item-level, which the rule list now says out loud.
- **`RunResult.attestationTripWire` is required with no default on a model arm** (§3.4.4, §3.4.5).
  `attest` no longer writes a runtime it cannot observe; the first `model:chat` run back-fills it.
- **`stats.verdict` loses `bootstrap_seed`, `conservative_envelope` loses `diffs`/`B`/`seed`, and
  `DecidedBy`'s second member is `"conservative-envelope"`** (§4 S1e Table D). `PackRef.seed` stays;
  its consumer is `-ml` §3.2d's continuous bootstrap and nothing else.
- **A pooled `BinaryMetric` declares its denominator per item** under `"<metric>#denominator"`, and
  `validate_pack` refuses a pooled `verdictMetrics` member and a metric name containing `#`
  (§4 S1 DC-10, §4 S2).

**What S2 and S3 additionally inherit from v1.11** (plan gate Pass 5 + note v1.14):

- **`ItemTiming.withheldFor` has three values and `LatencyBlock` has two counts.** `"timeout"` and
  `"no_response"` are separate item states feeding one `latencyWithheldForNoResponse`, because
  `-ml` §11.5.1's `censoringExact` reads the distinction per item while §11.7 slot 2 prints one label
  (§3.6, §4 S1, §4 S2 (vi)).
- **`statsCoveredCount` counts *calls* carrying a usable `stats` *and* a usable `promptTokens`**
  (**calls, not items, from v1.27**), and a call failing the second leaves the count **and** all
  three sibling medians — a disposition that
  **raises nothing**, applied per call and never per item. Rule (iv)'s identity is now an
  **inequality** against `callCount`; the equality asserted is against
  the one-pass recomputation from `run.items`, a sum over each item's `calls` (§4 S2 (iv), (iv-c)).
- **`ChatResult` never raises on a missing or partial `stats`**, and each derived timing field is
  `None`-never-`0` when its source key is absent (§3.6's unit boundary) — which is what keeps rule
  (iv-a)'s `statsCoveredCount == 0` reachable on the chat surface.
- **§3.6's tool-calling gate runs on a `tool-caller` pack only.** The universal pre-load refusal is
  §3.4.4a's `callSurface`-versus-catalog-`type` cross-check, whose predicate is now written out
  (§3.4.4a, §3.6). Unscoped, the gate refuses every embedder arm and S3 is unrunnable.
- **§3.2d's continuous verdicts reach `paired_cluster_bootstrap`, never `paired_bootstrap`** — and
  after `-ml` v1.16 they reach it **through Rule 8's `continuous_verdict()`**, which `report.py`
  calls and which owns both the family's quantile levels and the interval (§4 S1, §4 S1e Table G;
  v1.13). The exploratory **`sep_z` comparison is the one exception and calls the entry point
  directly**, with `clamp=None` — the shipped `[-1, 1]` clamp is a difference-of-proportions
  assumption (§4 S1e Tables D and E, `-ml` v1.14 §3.4 Rule 4) — and
  `levels=(LEVEL_CI95_LO, LEVEL_CI95_HI)`, because it is not a verdict and takes no family
  correction *(v1.15: the two module constants, the levels being exact rationals — `-ml` §11.2.2)*.
  Table E is S1-local work; **v1.12 withdraws the S3
  gate v1.11 put on it** — `sep_z` is reported, not
  verdicted (`-ml` v1.15 §3.2d), so the clamp is due with the `sep_z` comparison, which is
  exploratory and is no stage's done-condition.
- **A continuous per-item value has a carrier, and S2's scorers write it** *(v1.12, plan-gate P6-1)*.
  `ItemResult.measures: Mapping[str, float]` sits beside `counts`; a metric name is in one map or
  the other and never both; `scored_outcome` **raises** on a `measures`-resident metric and
  `scored_value` is its sibling. S2's scorer contract gains the third arithmetic (§4 S2), and the
  scorer states each continuous metric's `support` because it is the only party that knows it. This
  is §4 S1e **Table F**, and §4 S3 done-condition 1 is gated on it — without it the embedder's only
  verdict metric has nowhere on the record to live.

**What S2 inherits from v1.8's own revision** (§2.5's probe, not a gate):

- **There is no `lms` CLI anywhere in the tool.** `GET /api/v0/models` is the single source for the
  auto-captured fingerprint fields, `/v1/models` decides only which error is printed, and a `model`
  run that cannot reach the native catalog **writes nothing** (§3.4.4a). `hostinfo` launches no
  subprocess for host info; `powershell.exe` is the only subprocess left in the tool.
- **`lmsCliCommit` is out of `REQUIRED_BY_SCHEMA[1]["model:chat"]` and `residencySource` is in** — a
  one-for-one swap that keeps the set at 30 and is an **edit to shipped S1 code**, enumerated with
  its grep and its counts at §4 S1e Table A, one row of which (`conftest.py`'s residency element
  shape) ships silently wrong unless S1 DC-1's element-shape assertion lands with it. Free only
  because `results/runs/` does not exist yet (§3.4.2).
- **`callSurface` is a required field and a second discriminator**, so the mapping is keyed by
  `armProfile` and there are three profiles at schema 1; `model:embeddings` requires 26 fields and
  **forbids** `runtimeName`, `runtimeVersion`, `temperature` and `maxTokens` (§3.4.1, §3.4.2,
  §3.4.4a). Without this, S3 — the first end-to-end run — cannot store a record at all.
- **The first call to each arm is a warm-up on the arm's own call surface, and is never timing
  data**, under a separate `firstCallTimeoutSeconds` budget, with a between-item residency probe and
  an `unexplainedMs` gap check — per call, summed over the item (v1.27) — withholding a
  contaminated item's **`latencyMs` only**; its
  `stats`-derived siblings are kept, under their own coverage (§3.6, §5 test 15b). The measured number
  this exists for is a JIT load whose cost is model-dependent — 3.6 s and 21.1 s on two models on
  this box (§2.5, `-ml` §11.4) — which is why the budget is sized by magnitude and no figure is
  written into anything the tool prints.
- **S1's `aggregates`-versus-`items` cross-check is not S2's to build** — S2 owes the scorer
  contract that makes it unreachable, not a second copy of the check (§4 S2).

Two items are already recorded in `model-bench/docs/BACKLOG.md` (seeded at S0, re-checked at S8):
the **deferred judged-quality layer** for `chat-responder` (design preserved in §3.8.5 and
`-ml` §6.2) and the **+22 harder retrieval queries** that would lift the embedder pack's recall
ceiling (§3.8.1).

---

## Appendix A — Types named in this plan

Named above and defined here so an implementer is not inferring them from usage. Everything in
`stats.py` beyond this list — in particular the cluster-aware surface — is the `-ml` note's.

**This appendix is a derived surface, not a second contract** (§7 rule 4). Where a row disagrees
with the section that owns the type, the section is right and the row is stale — that is exactly how
v1.4's five-field `PackRef` survived §3.3's `sampling` block. A change to an owning section sweeps
its rows here in the same pass.

| Type | Module | Shape |
|---|---|---|
| `FieldSpec` | `fingerprint` | `NamedTuple(tier: Literal["nonempty","present"])` — §3.4.2 |
| `FieldProblem` | `fingerprint` | `NamedTuple(field: str, reason: Literal["absent","empty","null","forbidden","unknown"])`. **`unknown` is v1.5's fifth value**, for a ***value* this build cannot interpret** *(widened at v1.16 from v1.5's "a **discriminator** this build cannot interpret" — impl-gate P5-5: the shipped code widened at `8fc2341` and the module docstring was swept with it, this row was not, which is the rule 4 staleness this appendix warns about in its own preamble)*. Three families, and the row names all three so a fourth is a decision rather than a drift: an **unrecognised discriminator** — an `armKind` or a `callSurface` this build does not know (§3.4.1); a **`benchSchemaVersion` from the future** (§3.4.3); and a **type error inside a structured value** — a residency snapshot that is not a list, an element that is not a mapping, an element key whose value is not a string (§3.4.4a, S1 DC-1). None of the three is absent, empty, null or forbidden, so the four-value set would have forced a mislabel, and for the third the closed five leave `unknown` as the only honest answer rather than the natural one; it is the field-level counterpart of `InvalidRecord.reason == "unknown_schema"`. **`field` carries an element path for the third family** — `<field>[<index>].<key>` — which is `results.py:466`'s already-shipped grammar one level down and not a new convention; `report.py:534` renders `field` as opaque text and parses nothing, so the grammar constrains no consumer. |
| `Fingerprint` | `fingerprint` | frozen dataclass, §3.4.1–§3.4.2. **Two discriminators (v1.9):** `armKind` and `callSurface`, combining into the derived `armProfile` key (`model:chat` / `model:embeddings` / `deterministic`). Both are members of no required set and are checked before any mapping is consulted; `callSurface` is `None` iff `armKind == "deterministic"`, and its value is *declared* by the pack, never observed (§3.4.4a) |
| `ItemResult`, `RunResult`, `InvalidRecord`, `Aggregates` | `results` | as given in S1. **v1.12:** `ItemResult` gains **`measures: Mapping[str, float]`** beside `counts` — the per-item *continuous* values (`-ml` v1.15 §3.2d), defaulted `{}`, finite, domain-unconstrained, and **not** `float \| None` because absence stays `scoreable`'s job — plus `scored_value()` beside `scored_outcome()`, which now **raises `MetricKindError`** on a `measures`-resident metric rather than booleanising it. A metric name is in one map or the other, never both (`MetricKindError` at construction); a non-finite value is `NonFiniteMeasure`. **v1.10:** `ItemResult.timing` replaces the stored `latencyMs`, which becomes a **property** over it; `RunResult` gains `attestationTripWire: Literal["compared","first-observation","unavailable"] \| None`, required with no default and `None` iff the arm is `deterministic` |
| `CallTiming` | `results` | **v1.27** (`-ml` v1.20 §11.4, §11.5.1, §11.9 ask 7): frozen dataclass `(wallClockMs, ttftMs, generationMs, promptTokens, tokensPerSecond)`, every field `\| None` and never `0` — **one model call**. These five lived on `ItemTiming` from v1.10 to v1.26, when a scored item *was* one call; they are **call** figures (a `tool-caller` item is a turn of `I(t)` calls, prefill is not even constant within one, and `-ml` §11.4 pools them over calls rather than averaging up to the item), so they have exactly one home and it is here. §4 S2 rule (iv-c)'s co-presence exclusion is evaluated **per call** over this record |
| `ItemTiming` | `results` | **v1.10** (plan-gate P4-3; `-ml` v1.12 §11.9 ask 2b), **and smaller at v1.27** (`-ml` §11.9 ask 7): frozen dataclass `(wallClockMs, calls: tuple[CallTiming, ...], withheldFor)` — three fields, the five scalars having moved down to `CallTiming`. `callCount == len(calls)` and `unexplainedMs` (`-ml` §11.5.1's per-call gap **summed** over `calls`, `None` unless every call yields one) are **derived properties**, not stored fields — §7 rule 4's preference for a derivation over the invariant two stored copies would need, applied to this record a second time. `wallClockMs` is the **item's** wall clock — on a `tool-caller` the whole turn, dispatches included — and is `None` on a turn that ended on a raise, whose wall clock is incomplete and is never stored as a measurement. `calls` is `()` exactly when the item's first call returned nothing. `wallClockMs` is what the harness measured and **survives on a withheld item** — which is what makes `-ml` §11.6's *"the summary is withheld, not the data"* true of the record, and what §11.5.1's `censoringExact` compares. `withheldFor: Literal["load","timeout","no_response"] \| None` names the producer, so §4 S2's cause split is a count over the items rather than a parallel tally. **v1.11 (plan-gate P5-6):** three item states, still **one** counter — `timeout` and `no_response` both feed `latencyWithheldForNoResponse`, and they are distinct item states because `-ml` §11.5.1's `censoringExact` is satisfied by the first and falsified by the second. An item that returned no response still carries an `ItemTiming`, its `withheldFor` populated, `wallClockMs` `None` and `calls` `()` (plan-gate P5-9, restated over the v1.27 shape); `timing is None` means the **arm** produces no timings at all. On a `model:embeddings` arm each `CallTiming` carries `wallClockMs` alone: that surface returns no `stats` (§3.4.4a). `ItemResult.latencyMs` — the **admitted** figure, the only timing any aggregate reads — is derived from this record and stored nowhere (§7 rule 4 prefers the derivation to the invariant two stored copies would need) |
| `LatencyBlock` | `results` | **v1.9, extended v1.10**, produced by S2 and carried as `RunResult.latency: LatencyBlock \| None`: `latencyMsP50` / `latencyMsP95` / `latencyMsMax` (`float \| None`, `None` exactly when `-ml` §11's gates refuse them), `latencyTimedCount`, `latencyItemCount`, `latencyWithheldForLoad`, **`latencyWithheldForNoResponse`** (renamed from `…ForTimeout` — plan-gate P4-7), `statsCoveredCount`, **v1.27's `callCount`** (`-ml` §11.4's `Y_calls`, the sum of each item's `len(timing.calls)`, and the denominator the three `stats`-derived medians are gated against — **not** `latencyItemCount`, §4 S2 (iv-b)), and **v1.10's four aggregate figures** `ttftMsMedian` / `prefillMsPer1kMedian` / `tokensPerSecondMedian` / `unexplainedMsMax` (each `float \| None` — plan-gate P4-3). The wall clock and the three `stats`-derived figures have **different** coverage **and different units** — items and calls respectively, v1.27 — and each prints its own (`-ml` §11.4); `statsCoveredCount` counts **calls** and is `None`-never-`0` on a call surface returning no `stats`. The block's **nine** invariants are in §4 S2 — v1.11 adds **(iv-c)**, co-presence, and turns (iv)'s identity into an inequality (plan-gate P5-5); v1.27 re-bases that inequality on `callCount`, moves (iv-b)'s `Y` and (iv-c)'s exclusion to the call, splits `unexplainedMsMax` out of (i), and leaves (ii), (iii), (v) and (vi) item-level — and the estimator, floors and printed grammar are the note's |
| `BinaryMetric` | `results` | frozen dataclass `(name, successes, n, unit: str)` — a count and **the unit its denominator is in**, `unit` required with no default (v1.6). It is what lets `report.py` honour `-ml` §4.4's "never print a Wilson interval over a turn-pooled count": without it a per-conversation rate and a turn-pooled one are the same type. Every S2 scorer therefore states its denominator unit; the permitted values are the note's denominators (`-ml` §4.2). |
| `ContinuousMetric`, `DistributionSummary` | `results` | `ContinuousMetric(name, mean, n, support)` and `DistributionSummary(name, median, p10, n, unit, support)` — **v1.12** (plan-gate P6-1). `support: tuple[float, float] \| None` is the **metric's own** support, required with no default on both, and is what the paired bootstrap's `clamp` is derived from — `(lo−hi, hi−lo)`, or `None` for an unbounded metric — without the manifest field §4 S1e Table E rejects. **v1.14** (`-ml` v1.17, closing v1.13's open seam): on the verdict path `report.py` **forwards** this value as Rule 8's `support` and derives nothing, the producer deriving the clamp inside itself; the one caller that states a clamp directly is the exploratory `sep_z` comparison (§4 S1, §4 S1e Table E). **Stored form** — the `"distribution"` tag, the keys, `support` as `[lo, hi]` or `null` — is §4 S1e Table F's (v1.14). `DistributionSummary` exists because `-ml` §5.2 publishes a **median and a p10** for `sep_z` and `sep_raw` alike and **no mean for either** (v1.16 §5.2); the third published figure, the fraction above zero, is P@1 by §5.2's own identity and is deliberately not a second field (§3.8.1); `RetrievalAggregates.separationRaw`/`separationZ` carry it in place of the bare `float \| None` they carry today, and `named_metrics()` returns them, which is what makes `sep_z` reach a table at all. |
| `HolmStep` | `stats` | frozen dataclass `(p, rank, threshold, tested, rejected)`, one per pre-registered family member, returned by `holm_steps` in the family's own order and always exactly `k` long. **v1.6 — it replaces `holm_thresholds`' `list[float]`, which could not express the step-down stop.** `tested=False` marks a member past the stop: its `threshold` is still printed (§3.3(ii)) and decided nothing. The α it is computed at is the note's (`-ml` §3.3). |
| `PairedOutcomes`, `ResolvingPower`, `Verdict`, `BootstrapResult` | `stats` | **`-ml` §3.4's, verbatim** — not restated here. `PairedOutcomes.from_units` is the only constructor and raises on a repeated analysis-unit id; `resolving_power()`'s inputs are keyword-only with no defaults. (v1.2's `PairedResult` was this plan's own invention and is withdrawn.) |
| `PackRef` | `packs` | `NamedTuple(packId, packVersion, contentHash: str \| None, role, metrics, pairingKey: tuple[str, ...], analysisUnit: str, seed: int)` — pack identity as a *report* sees it, plus §3.3's two `sampling` declarations. **`contentHash` is `str \| None` (v1.6)**, `None` meaning "not loaded" and never "hashed to empty"; it is total only on `Pack.ref()`, and no report path may read it in place of a run's own `fingerprint.packContentHash` — §3.3's identity bullet carries the rule and the reasoning. **`pairingKey`/`analysisUnit` are v1.5's**: `report.py` resolves the analysis-unit id from `analysisUnit` and **no call site chooses it** (§3.3, DC-5(c)), so without them the resolution has nowhere to come from; the five-field form predated §3.3's v1.4 `sampling` block. Derived, not stored: `analysisUnitIndex = pairingKey.index(analysisUnit)`, which is `0` whenever `check_sampling_contract` has passed. **`seed` is v1.5's shipped field and this row omitted it until v1.10** — a rule 4 staleness of exactly the kind this appendix warns about. It is `sampling.seed`, required with no default, and after note v1.11 its object is `-ml` §3.2d's **continuous-metric** bootstrap alone: the paired binary path takes no seed (§3.3, §4 S1e Table D). |
| `Pack` | `packs` | frozen dataclass, S2 |
| `ModelInfo` | `lmstudio` | one `/api/v0/models` entry, verbatim: `id, object, type, publisher, arch, compatibility_type, quantization, state, max_context_length, capabilities?, loaded_context_length?` — the ten keys the 2026-09-03 probe returned for every model, plus `loaded_context_length` once loaded (§2.5, §2.3) |
| `ChatResult` | `lmstudio` | `message, tool_calls, toolCallForm, stats, model_info, runtime, usage, wallClockMs` — plus **v1.10's normalised trio, derived on construction**: `ttftMs = 1000 × stats.time_to_first_token`, `generationMs = 1000 × stats.generation_time`, `tokensPerSecond = stats.tokens_per_second` (unconverted). LM Studio reports the first two in **seconds** and every `…Ms` field in this plan is milliseconds, so the conversion happens once, here, at the transport boundary (§3.6's unit boundary, plan-gate P4-1). **v1.11:** each of the three — and `usage.prompt_tokens` — is `None` when its source key is absent, **never `0`**, and construction **never raises** on a missing or partial `stats` (plan-gate P5-8). The raw `stats` mapping is retained for auditability and **no runner, scorer or report path reads a timing figure out of it** |
| `EmbedResult`, `LoadResult`, `ResidentModel` | `lmstudio` | vectors + dimension; **`LoadResult` is the warm-up call's outcome** — `(wallClockMs, wasResidentBefore: bool, runtime: Mapping \| None, stats: Mapping \| None)`, with `coldLoadSeconds` derived from it only when `wasResidentBefore` is `False`, and `runtime`/`stats` populated on the chat surface only — they are the sole source of `runtimeName`/`runtimeVersion` (§3.4.4a step 5), so v1.8's `discardedResponse` naming is retired: the *content* is discarded, the metadata is not (v1.9); **`ResidentModel` is one `/api/v0/models` row surviving `state != "not-loaded"`** — `(id, state)`, the literal `state` string kept, not a boolean (v1.8, §3.4.4a; v1.7's "one `lms ps --json` row" is gone with the CLI) |
| `PromptConfig` | `convo` | the manifest's `prompt` block, parsed: `systemPrompt, toolSchemas, historyReplay, representToolSchemasEachTurn, historyTurns, maxIterationsPerTurn, temperature, maxTokens`. **`historyReplay` is four-valued** (`structured`, `structured-replies-only`, `plaintext`, `none`) spanning §3.3's **two** axes, role ownership and tool evidence — not one ladder (v1.26, P13-8). **`maxIterationsPerTurn` is `int \| None`**: it is `-ml` §4.1's `I(t)` cap and §4.2(f) reports against it, so it is pack data inside the content hash, and **v1.26 scopes it by role** — required iff `roles.MULTI_CALL_TURN_BY_ROLE[role]`, forbidden otherwise, `None` here iff that column is `False`, and `drive` **raises** on `None` rather than substituting a value, which is how the no-default rule survives a field four packs must not carry (§3.3) |
| `Turn`, `Conversation` | `convo` | one scripted turn (`seq, user, expect`); one row of `conversations.jsonl`. **`expect` is a scoring oracle, and `scoring/toolcalls.py` is its only reader** — `assemble` never reads it, because a prior turn is replayed from what the model actually produced (§3.8.4, v1.25) |
| `TurnTrace`, `ConversationTrace` | `convo` | **`TurnTrace` is one turn's record** — `(messagesSent, chatResults: tuple[ChatResult, ...], dispatches, envState, iterations: int, turnDisposition: TurnDisposition, finalReplyText: str \| None, wallClockMs)`, `chatResults` holding **every completed** iteration of §3.8.4's per-turn loop in order (empty when the first call raised); `ConversationTrace` is `(scriptId, turns: tuple[TurnTrace, ...])`. **v1.27:** `iterations == len(chatResults)`, forced rather than chosen — `-ml` §11.4 binds `callCount == len(ItemTiming.calls) == TurnTrace.iterations` and §11.9 ask 7 builds `calls` from `chatResults` in order — so a turn whose first call raised records `0`, and that `0` is *not observed* rather than *no iterations used*, which is what keeps it out of §4.2(f)'s summary and on the record all the same. `wallClockMs` brackets the whole turn, dispatches included. **v1.25 replaced the singular `ChatResult` field**, since a one-call-per-turn shape leaves a tool-calling turn with no final reply to score `-ml` §4.2(g) against or to replay. **v1.26 retires `capHit: bool` for `turnDisposition`** (plan-gate P13-1): `finalReplyText is None` **iff** `turnDisposition != "replied"`, and the four members' mechanisms and scoring mappings are §3.8.4's table, which is their only home — one bit could not stand for four states, and the false `iff` was laundering a §3.6 `fail` into an `n_a` |
| `TurnDisposition`, `TURN_DISPOSITIONS` | `convo` | `Literal["replied", "cap-hit", "timed-out", "no-response", "server-rejected"]` and the `frozenset` of the same **five** — v1.26's four, **widened at v1.27** (`-ml` v1.21 §4.3.1 item 2) because `timed-out` scores `fail` while the other two non-completions score `unrunnable`, and a folded token leaves the S5 scorer unable to tell them apart (§3.8.4). A **mechanism** vocabulary, deliberately not `-ml` §4.1's scoring word `unrunnable` — two rows *map to* that count (`-ml` §4.3 rule 4) and naming the field after it would repeat the `BinaryMetric.unit`/`PackRef.analysisUnit` collision. Both landed in the **precursor unit** at `d5b549d` holding four members, so v1.27 is an **edit** to shipped code and legs 1 and 2 of §5 test 10c's probe go red until the transcript moves with them — the guard working. The transcript is taken from **§3.8.4's table** |
| `ITERATION_SUMMARY_DISPOSITIONS`, `ITERATION_SUMMARY_EXCLUDED` | `scoring.toolcalls` | **v1.27** (`-ml` v1.21 §4.2(f), §4.3.1 item 8): `frozenset({"replied", "cap-hit"})` and `frozenset({"timed-out", "no-response", "server-rejected"})`, the population of §4.2(f)'s `I(t)` mean and p95 and its complement. A **scoring** vocabulary, so it sits beside the scorer and **not** in `convo`, whose set is a mechanism vocabulary — that separation is what makes the union assertion (§4 S5 *Done when* item 4) bind two independently authored declarations across two modules rather than restate one of them. Each is written out literally; the second is **not** derived as the first's complement, which would make that assertion a tautology |
| `DispatchRecord` | `tooling` | `(name, rawArguments, parsedArguments, returnValue, timestamp)` — FR-10's ground truth |
| `DecidedBy` | `stats` | `Literal["mcnemar-exact", "conservative-envelope", "paired-bootstrap"]`. **v1.12 adds the third member** for `-ml` §3.2d's continuous path (§4 S1): a machine token must name the instrument that ran, and on that path neither of the other two did; free while no stored record carries a verdict. **v1.10 renames the second member** from `"cluster-bootstrap"`: after note v1.11 the paired binary interval is Rule 4's bound-by-bound **envelope** computed in closed form, so no cluster bootstrap runs on that path and a token naming one is the machine-readable form of the prose defect the note corrects in its own four strings. The note left the rename to this document and recommended it; §4 S1e Table D enumerates the sites |
