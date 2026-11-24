# Methodology review: P1 close-out plan (CO-5, CO-6, CO-9, Q2, Q4, Q5)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1 close-out)

Reviewed: `docs/plans/connected-code-graph-p1-closeout.md` against EP §12.1/§12.3, `docs/plans/connected-code-graph-p1-ml.md`, `docs/plans/connected-code-graph-p1-gate-marker-ml.md`, `cpg/eval/decisions/ccg-noise-recall-rule-v1.json`. No held-out data or custodian directory read. Rulings and pre-registration text are in the gate-marker note (last section) and `-p1-ml.md` §4.4.

**Verdict: needs changes** (two majors, both fixable in the plan text before CO-6).

## Findings

- **M1 (major), §3.4 / CO-9 entry criterion is too weak.** "Listed or triaged" lets engine misses be excused by triage, so the rehearsal is weaker than the gate. It also predicts little: dev items were developed against, so a dev pass is a fit statistic. Fix: every dev item listed, triage only for a stakeholder-confirmed `key-error`; mechanical no-workaround scan; rehearsal-count disclosure; rehearsal miss means new F and new record (gate-marker note, entry criterion 1-6).
- **M2 (major), the gate verdict is stated without its bound.** The pooled 159-item Wilson bound (0.976) assumes independent items; the cluster bound is 0.938 over 58 clusters (`-p1-ml.md` §6.1). A pass needs true cluster recall near 0.99 to be more likely than not (r^58: 0.56 at 0.99, 0.31 at 0.98, 0.17 at 0.97; arithmetic, not measured). Wiring 8/8 certifies nothing (lower bound 0.676); label "screened, not certified". Fix: the plan's §9 and CO-6 state the cluster bound and the wiring label.
- **m1 (minor), CO-5 dev probe "0 dev hits blocks" is too lax:** a stratum at 1/49 passes it. Tie it to the entry criterion (fewer dev hits than dev items blocks).
- **m2 (minor), CO-5 per-kind miss breakdown** (stratum x anchor type x impacted type) has cells of 1-2 items at wiring size; on a fail it is held-out-derived and reaches builders. Fix: report at stratum level to builders; the fine breakdown goes to custodian-only `work/`.
- **m3 (minor), S1 and the rehearsal:** dev noise runs in CO-1..CO-5 or the CO-9 rehearsal print variant reductions; Q2 must be answered and recorded before any is printed, or the earlier ones disclosed.
- **n1, HA-6 row in EP §5.2.1** is owed by me; its spec is gate-marker note §3 and I will write it with CO-5 (not edited here, scope).
- CO-6 additions: freeze-time `check-report` over `S47..record`; gated/withheld sets with counts; the S1 wording; the Q4 screen rule.

## Answers

**(a)** A held-out pass is valid despite dev-driven building because validity needs only untouched held-out anchors; the honest claim is the cluster-bounded one above, not generalization beyond the writers' shapes. The rehearsal is necessary, not predictive. Pre-register the six items of the gate-marker note's entry criterion.

**(b)** Achievable: `pass` (AC-5 over four strata, flow unavailable, wiring screened) or `fail`; with wiring withheld (Q3 W2) 151 items still clear 75. A first-spend miss moves missed clusters to dev, keeps the rest as regression only, and needs F = 35 fresh-cluster items; for wiring fresh supply is probably nil, so a new key-writer draw. Wiring's 8 items add risk to the pass probability and no certification, which argues for W2/W3 unless CO-4 delivers every shape. Q4's screen can avoid contamination only with a blind writer plus the custodian discarding overlaps and reporting a count, or a non-held-out source; it is a gross-failure detector (n = 25, zero misses: lower bound 0.87, catches true recall 0.80 at 99%, 0.97 at 53%), one read, items become dev afterwards. Recommend (b) narrowed so; otherwise (a).

**(c)** S1: (c) both. Key-scoped is a subset of kind-wide, so the 5x numeric bar binds on key-scoped; kind-wide carries the equal-recall comparison. Wording in the gate-marker note.

**(d)** Sound only if the target is not the dev measurement: winner's-curse and selection on the same set inflate it. Stakeholder sets a minimum a priori; dev lower bound under it means "not met" without spending held-out (`-p1-ml.md` §4.4).

## Stakeholder decisions needed

- **Q2 (S1):** ratify (c) (it adds a condition to the ratified 5x bar).
- **Q3 (wiring):** methodology favours W2/W3 (see b).
- **Q4:** choose (b) narrowed or (a); needs a blind key-writer.
- **Q5:** the a-priori minimum acceptable local AC-24 target (new question); model choice itself is a method call.
