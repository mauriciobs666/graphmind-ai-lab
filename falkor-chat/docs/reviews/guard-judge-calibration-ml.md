# Guard-judge calibration — methodology review (K-027 item 3)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-027 item 3 (M3.5)

Reviews `docs/test-reports/guard-judge-calibration-2026-08-17.md` against the gate protocol at
`docs/archive/plans/m3-guard-calibration.md` (frozen 2026-07-16) and the current-code addendum at
`docs/plans/guard-judge-calibration-ml.md` (2026-08-17). Scope is methodology and statistical
presentation only — harness code correctness is a parallel `analyst` unit, not reviewed here.

**Verdict: approve with suggestions.** The gate math is correct, the verdict (`wire`) is the right
call under the archived protocol's own decision table, and the §6.1 statistical-honesty caveat is
present verbatim and correctly placed. One non-blocking revision is requested before this closes
out K-027 item 3: the report's "Passed" one-line summary of the materiality-probe check
undersells a real, individually-diagnostic signal that the protocol's own §8 item 8 says is worth
more than any aggregate at this N. Nothing here reopens G1/G2 or the wire decision.

## 1. Independent re-derivation (mechanical, not trust)

I re-derived every headline number directly from the report's own 26-row per-case table and from
`server/tests/eval/golden_guards.jsonl` (read in full), not from the report's summary lines:

- `sha256sum server/tests/eval/golden_guards.jsonl` = `35061c79aa9ae93f5e2350d30e4543a1a64f72b49a4cb0409319cd431d6776b4` — **matches the report's
  provenance header exactly**, and is a well-formed 64-hex-char digest.
- **G1 (FAR_strict):** 10 `clear_suspend` cases (cs-01..07, tn-04..06) × 3 replicates = 30 calls,
  all `False` in every replicate → **0/30 = 0.0%**. Matches.
- **G2 (advance-recall):** 11 `clear_advance` cases (ca-01..08, tn-01..03), per-case majority:
  9 correct (ca-04, ca-08 miss) → **9/11 = 81.8%**. Matches.
- **FAR_all:** 15 `expected:false` cases × 3 = 45 calls; sole failure tn-07, 3/3 → **3/45 = 6.7%**.
  Matches.
- **κ:** po = 19/21 = 0.905 (agreement count re-tallied row-by-row from the table); rater-A
  prevalence 11/21 = 52.4%; rater-B (judge majority) 9/21 = 42.9%; pe = (11·9 + 10·12)/21² =
  219/441 = 0.497; κ = (0.905−0.497)/(1−0.497) = **0.811**. Matches.
- **Per-path breakdown:** understanding n=15 (ca-01..08 + cs-01..07, boundary excluded), tp=6
  fp=0 tn=7 fn=2, accuracy 13/15=86.7%; turns n=6 (tn-01..06, tn-07 excluded as boundary), tp=3
  tn=3, accuracy 100%. Both match.
- **Boundary confusion / conservatism:** tp=0 fp=1 (tn-07) tn=4 (bd-01..04) fn=0, 4/5=80%.
  Matches.
- **coercion_flip_rate / flip_rate:** raw decisions equal final decisions in all 78 calls (no row
  in the table shows a raw≠final divergence), and no case shows replicate disagreement → 0/78,
  0/15 (r1_probe, 5 cases × 3), 0/26. Matches.

Every gate, diagnostic, and stratified number in the report is arithmetically correct against its
own underlying table and against the fixture. This corroborates `teco`'s independent derivation
(same numbers, same method) with a second, separately-executed pass.

## 2. §8 report-template compliance, checked item by item

| # | Requirement | Status |
|---|---|---|
| 1 | Provenance: model id, quant, temp, k, prompt revision, fixture sha256, date | **Present**, plus resolved `baseURL` (a `-ml.md` §5/§7 addition beyond the archived template — correctly incorporated) |
| 2 | Verdict line exact form | **Present, exact**: `G1 false-advance = 0.0% (n=10 cases / 30 calls) · G2 advance-recall = 81.8% (n=11 cases) · VERDICT: wire` |
| 3 | §6.1 caveat sentence, verbatim, adjacent to verdict | **Present, verbatim** (diffed character-for-character against archived §8 item 3), in the same `## Verdict` section as a blockquote directly under the verdict line — not a footnote |
| 4 | κ with both marginals + prevalence, "diagnostic — not a gate" heading | **Present** — heading reads "diagnostic, not a gate" (functionally identical wording), both raters' positive rates given (52.4% / 42.9%); rater-A's rate *is* the set prevalence here (11/21), so no separate line was needed |
| 5 | Per-path breakdown | **Present**, correctly restricted to the 21 clear cases (boundary excluded from both path cells) |
| 6 | `coercion_flip_rate` overall + `r1_probe` | **Present**, both correct |
| 7 | `flip_rate` | **Present**, correct |
| 8 | Full per-case table w/ raw rationales | **Present** — id, tier, path, expected, raw decisions, final decisions, rationales for all 26 cases |

One **nit**: item 8's table has separate "raw decisions" and "final decisions" columns but no
explicit per-row `coercion_flip` boolean. It is losslessly inferable (raw==final in every row, and
the aggregate rates are already reported and correct), so this is cosmetic, not a gap — not
required for approval.

Provenance header also correctly reflects the `-ml.md` addendum's guidance rather than the stale
archived text: "prompt revision" cites HEAD (`acda33d`), per `guard-judge-calibration-ml.md` §5's
explicit correction away from the archived note's "S4 commit" framing (the prompt hasn't moved
since K-042; citing HEAD is right).

## 3. Is "wire" the correct verdict?

Yes, unambiguously. G1 (0.0% ≤ 10%) and G2 (81.8% ≥ 0.80) both pass. Per archived §7: "Both gates
pass ⇒ wire it, and write the §6.1 sentence into the report." Both conditions are met, and the
verdict is placed adjacent to the gate numbers rather than buried. No override, no discretion
exercised beyond what the protocol specifies — this is a correct, mechanical application of the
decision table.

## 4. G2's two misses — ca-04 and ca-08

This is the substantive finding of the review. I read both cases against the fixture's own
`label_rationale` (not just the judge's output), because both turn out to be **the fixture's own
materiality probes** (§4.2 of the archived protocol), and the judge's rationale text is
diagnostic:

- **ca-04** (`missing: ["preferred summary length"]`, labeled advance because summary length is a
  *presentation* preference, not a research input). Judge's rationale, all 3 replicates: *"The
  user has not specified a preferred summary length, **which is necessary to research and tailor
  the summary appropriately**."* The judge's own words echo the fixture's `missing` entry almost
  verbatim and treat it as research-blocking — exactly the failure mode the case was built to
  detect.
- **ca-08** (`missing: ["the user's current session store configuration"]`, labeled advance
  because the general trade-off comparison is researchable without the user's personal config).
  Judge's rationale, all 3 replicates: *"The user has not provided their current session store
  configuration, **which is necessary to assess the relevance and applicability** of the
  trade-offs..."* Same pattern: the missing field's content is restated as a blocker for
  *researching*, when the fixture's design intent is that it only affects *tailoring the answer*.

Both misses are legitimate advance-recall errors, not defensible "arguable ambiguity" reads — the
fixture author's rationale for both is specific and the judge's own text tracks the omitted field
directly rather than reasoning about what "enough to *research*" requires. This is qualitatively
different from a boundary case (§4.2's third materiality probe, ca-05, correctly advanced, and
`cs-04` — the adversarial "empty-but-vacuous" probe — correctly suspended).

**Does this trip the protocol's own blocker rule? No, and I am not retroactively tightening it.**
Archived §7 is explicit and conjunctive: *"The materiality probes fail as a bloc (ca-04/05/08
suspended AND cs-04 advanced) ⇒ report as a blocker-grade finding."* ca-05 advanced correctly and
cs-04 correctly suspended, so the bloc condition is false. The report's "Passed" line is *correct*
under the literal rule I wrote, and I decline to move that goalpost post hoc — that would itself be
a statistical malpractice (fitting the threshold to the data after seeing it).

**But the report's one-line "Passed" summary is incomplete as written.** It reads as "no
materiality signal," when the actual result is "2 of 3 individually-designed materiality probes
were missed, with rationales that directly echo the omitted field rather than reasoning about
research-sufficiency." That is a real, non-gating, qualitatively important finding — exactly what
archived §8 item 8 calls "worth more than any aggregate at this sample size" and exists to
surface. Suppressing it behind a bare "Passed" undersells the one place in this run where the
judge showed a repeatable-looking pattern, even though n=2 is far too small to call it "systematic"
with confidence.

**Requested revision (non-blocking to the wire verdict):** amend the report's materiality-probe
section from a bare "Passed" to something that states both facts — the bloc criterion did not
trigger, *and* 2/3 individual probes missed in a pattern that tracks the `missing` field's content
rather than research-sufficiency reasoning. This is a report-completeness ask, not a gate dispute.

## 5. `tn-07` — the sole FAR_all contributor

`tn-07` (boundary, turns path) advanced 3/3. Its fixture `label_rationale` calls it explicitly a
*"GENUINE BOUNDARY on the fallback path: one clarifying question answered precisely, the other
answered only qualitatively... a real judgment call."* The transcript: user gives `production` for
the first sub-question and `"noticeably worse since Monday"` (no exact p95) for the second. The
judge's rationale — *"specified the environment as production and provided a clear observation
about the performance degradation since Monday, which allows for research"* — reads as a
defensible sufficiency call on genuinely partial-but-real evidence, not a fabrication or an
obviously wrong read. This looks like the design working as intended (a boundary case the judge
resolved toward advance, which archived §4.1 explicitly treats as non-safety-relevant and excludes
from G1's gated denominator), not a scoring artifact.

**On the turns-path-permissiveness question (brief item 4, DS-note risk #4):** the turns-path
clear stratum (n=6) scored 100% accuracy — *better* than the understanding path's 86.7%, which if
anything runs counter to the a priori "fallback path is degraded" prediction. `tn-07` is the one
turns case outside that clear set, and it is the sole boundary-stratum deviation toward advance
(vs. bd-01..04, which all correctly suspended on the understanding path). **What this one case
cannot establish, in either direction:** whether the turns/fallback path is more permissive than
the understanding path on boundary-adjacent inputs specifically. n=1 in the boundary∩turns cell is
not evidence of a defect, but it is also not evidence of the fallback path's safety — it is simply
uninformative at this sample size, and the report should not be read (nor does it currently read)
as settling risk #4 one way or the other. This is worth naming as an explicit open question for
future set-growth, not as a defect in the current report.

## 6. Statistical-honesty framing — preserved, not drifted

Checked the report's own prose (not just the presence of the caveat sentence) against the exact
failure mode §6.1 was written to prevent (reading a pass as "the judge is calibrated"). The report
does not do this. It states the gate numbers, cites the verbatim caveat adjacent to the verdict,
and elsewhere uses flat, unembellished language ("expected high; a low value is an early warning,"
"reported, never gated," "diagnostic, not a gate"). No instance of "calibrated ✅"-style framing,
no aggregate accuracy computed over all 26 cases (which archived §10 risk 3 explicitly forbids —
correctly absent), and the two case-level denominators (n=10, n=11) are stated next to their rates
in the verdict line exactly as §6 requires to prevent the call-level-CI-as-case-level-CI misread.
This item is clean.

## 7. Stale-BACKLOG pointer

The report's superseding note (line 5) cites all three of the `-ml.md` §3 reasons correctly
(pre-parse-fix, bypassed `evaluate_guard`, wrong G1 denominator) and points to
`docs/plans/guard-judge-calibration-ml.md` §3 rather than re-deriving the explanation inline —
exactly as that note's §7 step 5 asked. Correcting `docs/BACKLOG.md` itself is out of this report's
scope (flagged in the `-ml.md` note's own §8 risk 1 as a `teco`-coordination-close action) and is
correctly not attempted here.

## 8. Anything else

- **Golden-set integrity:** re-parsed the fixture directly; 26 rows, strata counts (11/10/5),
  path counts (19/7), and per-row invariants (turns cases carry `understanding: {}`, boundary
  cases all `expected:false`) all match the protocol's §4 table and `-ml.md`'s F4 mechanical
  count. No drift.
- **No leakage risk observed:** the fixture is synthetic-but-realistic per its own design (§5.3),
  and nothing in the report indicates it was echoed back into a prompt or corpus.
- **Single-labeler / boundary-as-policy caveat (archived §10 risk 3):** correctly not violated —
  the report never computes a blended accuracy across all 26 cases that would silently gate on the
  boundary labels' policy choice.

## Verdict

**Approve with suggestions.** K-027 item 3 is closeable on this report's numbers — both gates pass,
the verdict and its adjacent caveat are correctly derived and correctly presented, and every
headline statistic independently re-verifies against the report's own per-case table and the raw
fixture. The one requested change — expanding the materiality-probe line from "Passed" to state
the 2/3 individual-probe miss pattern on ca-04/ca-08 alongside the (correctly non-triggered) bloc
criterion — is a report-completeness improvement, not a reopening of the gate math or the wire
decision, and should not block closeout if the team judges the existing per-case table (which
already contains the raw rationales needed to see this) sufficient disclosure on its own.
