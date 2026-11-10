# Review: connected code graph, P1 plan set consistency gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 (2026-11-10)

### Scope & verdict

**Reviewed:** the working-tree diff against `8e6bd0bb` of `docs/plans/connected-code-graph-p1.md` (index),
`-p1-extraction.md`, `-p1-linking.md`, `-p1-answers.md`, `-p1-gates.md` and `docs/plans/connected-code-graph.md`
(proposal), read against the cited deliverables: `-egress-spec3.md` (§7.4, §8.2, §10.1, §10.2), `-answer-spec2.md`
(§1.2, §3.2, §3.3, §5, §8), `-bundle-spec5.md` (§2, §8), `-p1-graph.md` (§1, §2, §5, §7, §8, §9) and `-p1-ml.md`
(§2.1, §3.1, §3.3–§3.5, §5.3, §6.1–§6.6, §8–§10). The cited deliverables were not re-reviewed. Run: `git diff`,
`git log`, unfiltered greps over the five plan files, a script over the index §7 table (49 rows, cycle and
reachability check), reads of `cpg/ccg/atrest.py`, `cpg/ccg/registry/reasons.yaml`, `cpg/infra/ccg-falkordb.sh`,
`cpg/pyproject.toml`. No tests or services were run. `cpg/REVIEW.md` does not exist; root `REVIEW.md` applies.

**Verdict: approve with suggestions.** No blocker. The plan set agrees with its cited sources on nearly every
point you listed (see *What's solid*). Five majors (F1–F5) are gaps where a deliverable's obligation has no step
that owns it, or the order that makes a gate valid is not written down; each is a one-or-two-line edit in one plan
file. Fix F1 before P1-17/P1-18 dispatch, F2/F3 before the `ac5-gate-1` record is scheduled, F4 before P1-26, F5
before P1-15.

CPG: considered, not relevant — the subject is plan documents for `cpg/`, and no `cpg_*` graph is loaded for it
(the plan's own CPG line says the same).

### Findings

**F1 (major, verified): an edit the graph note assigns to P1-17 is owned by no step, and the index row for P1-45
still names the file.** `-p1-graph.md` §2 (last bullet): "`reasons.yaml`'s `binding-confirmed` row gains `VIA` in
its `on` list (P1-17's edit)". `cpg/ccg/registry/reasons.yaml:19` is `on: [Link]` today. P1-17's Files in
`-p1-linking.md` list no `reasons.yaml`, and its behaviours state only the Link-level reason (4) and (9), not the
per-`VIA` rule of graph §2 (`VIA.reason == binding-confirmed` iff confirmed and a `BOUND_BY` with its `siteId`),
which P1-18's test (c) then asserts. `-bundle-spec5.md` §8 says "Nothing" for `reasons.yaml`, and
`-answer-spec2.md` §7 says P1-45 does not edit it, yet index §7's P1-45 row and P1-45's Files line
(`-p1-answers.md:32`) still name `reasons.yaml`. *Improve:* add `cpg/ccg/registry/reasons.yaml` and the per-`VIA`
reason rule to P1-17's Files and behaviours (and a mutant: dropping the `siteId` condition); strike `reasons.yaml`
from the P1-45 index row and its Files line.

**F2 (major, verified): the order that makes the first gate valid is not stated.** `-p1-ml.md` §6.1 pins the gate
harness at the record's commit (P1-32b compares `sha256` with `git show <recordCommit>:<path>`; "an edit after the
record makes the run 'after adaptation'"), and §6.2 requires the evaluation tooling (P1-32, P1-35, P1-36) to be
written in builder-safe dispatches **before** the record. Index §7 "Core freeze" lists the steps the record Needs
as "every step that edits a tracked file outside the `coreTreeHash` exclusions": 01, 02, 03–12, 14, 15, 16–26, 28,
29, 31, 34, 39, 44, 45, 48, 48b. `cpg/eval/**` is excluded from the hash, so 32, 35, 36 (and 47) are absent, and
P1-37's Needs lists them without ordering them against the record. A scheduler following the index can commit the
record first and land P1-32b after it, which loses first contact. *Improve:* add 32 (both dispatches), 35, 36 and 47
to the record's prerequisites with the stated reason (pinned, not hashed), as a sentence beside the hashed list.

**F3 (major, verified): pre-record and per-step obligations of the ML note have no owner.** The ML note's step-text
amendments and §8 table name actions no plan step carries (grep of the five files for each name returns nothing):
(a) the freeze-time `check-report` **run** by the custodian before the record (ML §6.2; P1-32a builds the verb but
nothing in the plan runs it, or says a hit moves a cluster to dev); (b) the stakeholder's 40-item redo (ML §9;
P1-37 treats it as a given prerequisite, with no owner, no "before the record" ordering and no `tico` hand-off:
N-TICO records only the answer); (c) records `ccg-p1-indexer-v1` (P1-15), `ccg-p1-flow-fallback-v1` (P1-39, "before
P1-39 runs") and `ccg-noise-ratify-v1` (cited by the next gate's record); (d) ML §9's "counts by stratum and class"
in the gate report (P1-37 does not list them); (e) the noise frame, sample and labels under the custodian root
(ML §6.4; P1-36's Files say it only for `oracle.py`). *Improve:* one line each in the owning step (P1-15, P1-39,
P1-36, P1-37) and a "before the record" list in index §7 naming the freeze-check run, the redo and the manifest
re-record, each with `teco`/`tico` as actor.

**F4 (major, verified): P1-26 does not carry spec3 §10.2's P1-26 consequences, and its test extension is stale.**
`-egress-spec3.md` §10.2 (P1-26 row) and §2.1 require: one `answer` run per question, opened at the first model
call from `interpret.py` or `summary.py` and reused for later calls, none for a question the parsers resolve; the
`local` entry's `paths` fixed; a test that a missing or latched broker fails the call with no direct connection
attempted. P1-26's behaviours (`-p1-answers.md:194-228`) name the broker call path only. Separately, P1-26 (6)
"extends `test_no_network.py` to `ccg/front`", but P1-25's test now scopes to **every module under `cpg/ccg/`**
(spec3 §10.1, applied at `-p1-answers.md:185`), so (6) is redundant, and P1-26 does not Need P1-25 although it
edits P1-25's file (script: `P1-26` and `P1-25` unreachable from each other). *Improve:* add the three spec3
behaviours to P1-26; replace (6) with "the I2 test passes unchanged with `ccg/front` present"; or add Needs 25.

**F5 (major, verified): P1-15's done-condition has no outcome for the stakeholder-decided operator provisioning,
and omits the ML measures.** The decision (ML §2.1, 2026-11-10) is that the operator installs dependencies and
writes `env.json` outside the system. P1-15's three outcomes (`-p1-extraction.md:355-363`) are adopted, "not adopted,
because" and "not run: runner"; an operator who has not yet provisioned fits none, yet the offline figure cannot
exist without it. Index R-2 (`-p1.md:438`) still says "unavailable ... unless pre-provisioned" with no owner. The
step also drops M0 (bare, informational), M1 (offline/online delta against `truth/<lang>.ref.jsonl`, the ML's primary
measure) and M2's named quantities, and the `ccg-p1-indexer-v1` record. *Improve:* add the outcome "not run:
dependencies not provisioned" (and say who is told: `teco`, who asks the operator), list M0/M1/M2 and the record in
Done by citing ML §2.1/§2.3/§8, and say "operator" in R-2.

**F6 (minor, verified): unordered shared files and two Needs that hold only because the steps shipped.**
(a) P1-14 and P1-45 both edit `cpg/tests/test_answer.py` (spec5 §8 row; spec2 §8's `REQUIRED`/`ENUMS` tables) and
neither Needs the other (script: no reachability either way); (b) P1-05 (7) requires `ccg.atrest.write_bundle`
(P1-02) with Needs 04 only; (c) P1-03's `test_sysname.py` "reads the script" of P1-48's status check, with Needs 01
only. (b) and (c) are moot for the order actually run (P1-02 and P1-48 are committed, `7eea68dd`), but the written
graph is wrong for a rebuild. *Improve:* add Needs 45 to P1-14 (or an additive-only note), 02 to P1-05, 48 to P1-03.
Nit: the plan quotes the status regex as `^(default|ccg_…)$` (`-p1-extraction.md`, P1-48 (2)); the shipped
`cpg/infra/ccg-falkordb.sh:122` is `ACL_USER_RE` without `default|` plus a separate length check. The same-verdict
test is unaffected.

**F7 (minor, verified): P1-23's labelled-row paraphrase differs from spec2 §5.** P1-23 says the labelled rows are
"`CALLS ←`, `CALLS_NAME ←` and `TO ←` continuing from a `Symbol`". Spec2 §5's table labels these only when
**leaving the `function` state**, adds the depends-on column (`HAS_CALL →`), and says a direct resolved caller of an
anchor `Symbol` is **not** labelled (P1-22 pins this: "`apiFetch` stays `confirmed`"). A reader of P1-23 alone could
label direct callers of the anchor. *Improve:* write "continuing from a `function`-state hop, both columns, the
closed table of `-answer-spec2.md` §5" and drop the state-name aside.

**F8 (minor, verified): P1-31 and index §6 restate spec3 §8.2 inexactly.** Spec3 §8.2: row 8.16 is "computed when
provisioned; otherwise **unrun**" (`strace` is not installed; provisioning is `devops`'s call; its trace parser needs
a positive control); 8.13's residue is computed; 8.15 (ii)/(iii) are vacuous. P1-31 says "each row computed or
reported vacuous" and the index says "rows 8.1–8.16" and lists the vacuous set as "8.2, 8.4, 8.12–8.13 and the
proxied halves of 8.3 and 8.5". A faithful P1-31 has no stated result for 8.16 on this host. *Improve:* state the
third status (`unrun`), the `strace` decision and the parser control in P1-31, and replace the index's list with the
spec3 §8.2 citation alone.

**F9 (minor, verified): P1-03's behaviours omit spec3 §10.1's P1-03 tests.** Spec3 §10.1 (P1-03): S1–S5 each refuse
a planted manifest; `policy.egress` is mandatory; its enum equals `$defs/egressMode`. P1-03's behaviours (1)–(8)
mention none, and neither the proving-system manifest (`cpg/ccg/system.yaml`) nor the fixture manifest is said to
carry `policy.egress` per repository. *Improve:* add the three tests as behaviours and the field to both manifests.

**F10 (minor, verified): stale index text.** (a) N-ML row (`-p1.md:420`) still asks about "ceiling 468/481 = 0.973";
ML §3.4 and index §5 say 467. (b) P1-36's index row says "ratification procedure"; the thresholds are ratified
(index §4, ML §5.3). (c) Index §2's contract list (line 35) was edited for `-egress-spec3.md` but still lists
`-answer-spec.md` and `-bundle-spec3/4.md` with no `-answer-spec2.md` or `-bundle-spec5.md`, the contracts P1-45
and P1-07/P1-14 implement. (d) P1-09's Done says "159 fixable misses"; index §5's text still says "149 of the 151".
(e) `P1-44` (1) says "through `write_bundle`" unqualified; it is `ccg.atrest.write_bundle`.

**F11 (minor, verified): restated facts have already drifted from their sources, and a few history phrases remain.**
Growth (words, base → now): extraction 3,368 → 4,343; answers 2,380 → 3,143; gates 3,109 → 4,012; linking +143;
index +104; proposal −181. Most of it is new scope. The rework-class growth is the long restatements of cited
notes, which have a home already: P1-09 (4b) (ML §3.4) omits the "no projected edge ends at a synthetic cond"
positive and the "point under a real intermediate cond keeps its direct edge" negative; P1-34 says "blind
`qa-engineer` labellers" where ML §3.3 says one fresh blind labeller; P1-32a restates ML §6.2 in full; P1-26 (3b)
restates ML §4.2. *Improve:* cite the section and keep only the done-condition and the named tests. History-shaped
phrases to turn into present-tense design: `-p1-gates.md:200` ("which replaces `cpg/.ccg-work/oracle/` and mode
0700"), `-p1-gates.md:164` ("not P0's `ccg-p0a-v1-flow|` nor EP §3.6's…"), `-p1-linking.md:145` and the proposal's
§3.5 ("the guarded-`EVAL` form being withdrawn" / "is withdrawn"), index D8 ("once more").

**F12 (nit): `E12` means two things in one bullet.** P1-32b writes "E12's floor (≥ 35 …)" (the EP decision) and
`~/.local/share/ccg-eval/<E12>/` (first 12 hex of `EVAL_COMMIT`) within five lines (`-p1-gates.md:111`, `:116`);
ML §6.4 itself warns they are unrelated. Rename the path token (`<EVAL12>`) in the plan.

### What's solid

All run or read unless marked.
- **Step graph.** 49 index rows (script) and 49 per-step headings, P1-48b included; no cycle; P1-30 appears only as
  "unused"; P1-48b's Needs (28, 48), P1-31's (29, 18), P1-29's (28, 03), P1-20's (…, 29) and P1-15's (…, 48b) agree
  with spec3 §10.2; P1-07 Needs 13; P1-18 owns `staleness.py`/`FAIL_REASONS` (eight codes equal `-p1-graph.md` G4),
  P1-20 adds `unit_staleness`; P1-03 owns `sysname.py`; D8 and P1-29 agree on the `broker` extra and the `dev` pins
  (`cpg/pyproject.toml:19-20` verified).
- **Sweeps (unfiltered, five plan files plus proposal).** No `.ccg-work/keys`; `.ccg-work/oracle` only in the
  replaced-by sentence (F11); no `system_key`; `write_bundle` hits all qualified `ccg.atrest` except P1-44 (F10e)
  and P1-07's `ccg.bundle` defence line; no role-only `ccg_build`/`ccg_raw`/`ccg_verdict`/`ccg_engine`; no `8.15`
  stale range; "join" hits are the S47 scripted-step join and the contains-join, not the AC-8 join; `bind_key_check`
  has one home (`atrest`, matches `cpg/ccg/atrest.py`).
- **Source agreement.** P1-19's `swap(build_conn, verdict_conn, …)` and the pre-`RENAME` replay credential match
  graph §7/§9; P1-23/P1-45/P1-21/P1-17(6) agree on `bound_of_link`, `Bound` without `siteId`, and the lowest
  `(siteId, ord)` pick (graph §5/§8(f), spec2 §3.2–§3.3); P1-25's allow-list equals spec3 §10.1's I2 test; P1-48b
  equals spec3 §10.1 (21)–(26) and §10.2; P1-34's 465/481 and seed prefix match ML §3.1/§3.3; index §5's 455/459/467
  match ML §1; the freeze-check verb's path set, exit codes and reverse map match ML §6.2; custodian root and
  `<E12>` match ML §6.4.
- **Stakeholder decisions** are reflected: Q1=C, `cpg/` excluded, flow-dependent hops only `possible`, per-system ACL
  user names, Q-C thresholds (gate 1 at Q ≤ 4), Q-D redo (subject to F3), operator provisioning (subject to F5).
- **Proposal.** The edits are replacements in place, the ACL bullet now points at graph §7, and I found no
  surviving contradiction with graph §7 or spec2 §5.

### Open questions

None blocks the verdict. For `teco`: F3's pre-record list needs an actor for the freeze-check run and the
stakeholder redo; the ML note names "the custodian" and "the stakeholder" but the plan has no step id for either.

### Proposed REVIEW.md lines

Under root `REVIEW.md`, "Always check" (source: F1, F3 of this review, a recurring class: a deliverable's
"obligations other documents owe" without an owner step): "For each edit, record or action a cited deliverable
says another document or step owes, name the plan step whose Files or Behaviours carry it; an edit that only the
deliverable's own text assigns is an orphan finding."
