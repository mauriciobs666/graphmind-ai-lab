# Review: connected code graph — P1 method note (unit N-ML)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 (2026-11-09)

### Scope & verdict

**Reviewed:** `docs/plans/connected-code-graph-p1-ml.md` (uncommitted, first version, 883 lines). I checked it
against index `docs/plans/connected-code-graph-p1.md` §5 and §9 N-ML (its six questions), the steps it unblocks
(P1-09, P1-15, P1-22/23, P1-26, P1-32, P1-34–P1-37, P1-39, P1-43, P1-47 in `-p1-{extraction,answers,gates}.md`),
the standing contracts (`-eval-protocol-ml.md` = EP, `-eval-protocol2-ml.md`, `-p0-ml.md` = P0-ML,
`-answer-spec2.md` §5/§9, `-graph.md` §6.2, `-p0-graph.md` §1.5/§3), the requirements
`docs/requirements/connected-code-graph.md`, and root `REVIEW.md`. `cpg/REVIEW.md` does not exist.
**Data read:** the P0 fitness artifacts (`cpg/spikes/frontend-fitness/truth/*.flow-final.jsonl`,
`adjudication/*-flow-{misses,spurious}.adj.jsonl`, `sample/`, `labels/`, `s28_metrics.py`), the source of the
four chain-credited methods at `0caf85e3`, the dev files `cpg/eval/dev/ac5-flow.jsonl` and `ac24.jsonl`, and
`cpg/eval/heldout-manifest.json` (counts only). I read no held-out file. I wrote nothing outside this review and
the session scratchpad, and made no FalkorDB writes.

**Verdict: needs changes.** There is no blocker. The arithmetic is right: every load-bearing figure reproduces,
including from an independent instrument (Appendix A). But the note has seven majors:
- the transitive-CDG ruling, which is what makes the entry value reachable at all, credits chains through conds
  that P0's own adjudication rules call spurious (M1);
- the chain scorer it relies on is owned by no step (M2);
- Q-D's options misstate the standing contracts (M3);
- the freeze-time contact scan contradicts EP §8.2 and cannot be run as written (M4);
- the oracle relocation is right, but its stated reason is wrong and it covers one artifact class out of several (M5);
- the offline `scip-python` protocol cannot pass its own argv check (M6);
- the interpreter's span design cannot express a qualifier that 5 of the 12 dev unambiguous questions use (M7).

CPG: considered, not relevant — no graph is loaded for `cpg/` (`GRAPHS` lists `cpg_falkorchat` and non-CPG graphs only); the artifact is a method note, and its figures were checked against the P0 data files directly.

### Findings

#### Major

**M1 — The chain ruling (§3.4) credits control pairs through conds that P0-ML §8.4's rules declare non-points.**
*Verified (source read at `0caf85e3`).*
- The intermediate cond of every one of the 9 `definition` misses is one of the following (Appendix B):
  - a frontend-synthetic `try` cond: `open_ = keep_going()` in `retry.py:134` (5 pairs), `json.loads(value)` in
    `services.py:449` (2 pairs);
  - a generator-loop or `for`-iterator cond (`iteratorNonEmptyOrException`: `retry.py:81`, `test_repository.py:4703`).
- P0-ML §8.4's control rules (v1.17) say *"synthetic … conds (`try`, …, generator loops) are not points … a pair
  touching one is spurious"*. In the `test_repository.py` pair the second link (loop header ← body `if` that
  returns) is the loop-carried control that the same rules call "not asked for".
- So 9 of the 9 credited pairs connect only through edges that the spurious-pair adjudication counts as wrong.
- The note's text says "through a projected cond". That is literally true, but it omits this. §3.4's rejection of
  closure ("contradicts the nearest-governing-cond ruling") applies equally to the chain it accepts.
- This matters because it is the hinge: the direct ceiling is 459, below every entry value derivable (461–465,
  Appendix A). Only 468 makes entry possible.
- Also, P1-09's join fix re-maps `iteratorNonEmptyOrException` to the `for` condition point. After that,
  `test_repository.py`'s chain runs through a *real* cond by a loop-carried edge, so the ceiling depends on P1-09's
  details, and §3.4's only guard is a P1-22 walk test.

**Fix (method owner's design call; I cannot name an assertion that would prove one variant right):** state the
fact in §3.4. Then choose between two rules:
- (a) a narrower rule, "a synthetic cond is transparent", the cond analogue of R3, fixed as a P1-09 projection
  rule with a golden. This credits the try/generator chains as direct edges and keeps the nearest-governing
  ruling for real conds. The dev ceiling is 467 or 468, depending on the `for` case;
- (b) keep the general rule, and state that the spurious rate and recall now disagree on the same edges.

Either way, add a P1-09 golden that pins the 9 pairs' connectivity, so that a projection change which drops a
synthetic cond turns red.

**M2 — Nobody owns the chain-definition scorer that §3.1's entry check and §3.3's decision run on.** *Verified.*
- `s28_metrics.py` scores a control pair as `ok = (tk, sk) in M.ctl`, a direct edge only (line 273).
- P1-34 step 1 edits `s28_run.py` only. P1-09 reuses `s28_metrics.py` "not copied".
- No step text adds a chain mode, so "P1-09 reports dev recall under both definitions" (§3.1) has no
  implementation.
- `cpg/spikes/**` is under `coreTreeHash` (index §7), so the change must also land before the freeze.

**Fix:** add it to P1-34 step 1 (or P1-09) as an amendment the note routes in §10. Name the behaviour and its
positive and negative controls:
- a two-step `CONTROLLED_BY` chain is connected under `chain` and missed under `direct`;
- a chain that mixes a `DATA` hop is not credited as control;
- the mutant `chain := direct` must fail on the 9 stored rows.

**M3 — Q-D (§9) offers as options two things the standing contracts already decide.** *Verified against text.*
- **(c) is already the rule.** EP §12.1 step 2 says: *"`key-error`: the item is false. The stakeholder must
  confirm this."* Recommending (c) makes it look as if the stakeholder could decline something EP already requires.
- **(b) is what decision S36 requires.** Requirements "Delivery preferences" and the 2026-10-26 S36 log entry say
  the blind keys are *"spot-checked by the stakeholder"*.
- P0's 40-item union spot-check (EP §7.3) was done by an AI analyst, which is the disclosed "deviation from
  decision S36" in the ground-truth report §P0 disclosures.
- So (a) "no audit" is not a neutral default. It continues a requirements deviation, and only the stakeholder can
  waive that.
- The note argues against (b) that 20 items cannot bound an error rate. EP §7.3 never asked it to: the sample is a
  screen with an escalation trigger (≥ 2/8 per stratum, ≥ 4/40 pooled).

**Fix:** reframe Q-D as one question: "redo EP §7.3's 40-item spot-check with the stakeholder, as S36 requires,
or waive S36 for gate 1, recorded in the decision log". Then state (c) as EP §12.1's existing rule, not as an
option. Index §4.3's row and Q-D default "no" need the same correction; route that to `architect`.

**M4 — The freeze-time contact scan (§6.2 item 2) contradicts EP §8.2 and is not executable as written.**
*Verified against text; the scan is not run.*
- **Conflict.** It moves the cluster of any held-out anchor whose locator appears in a committed builder artifact.
  But EP §8.2 rules that some appearances are *not* contact:
  - wrapper and chain hops (`apiFetch`, `buildUrl`, the `app.py` mount: "counting them would empty `http`");
  - "the same method appearing as an impacted element or chain hop in another stratum".
  A pure locator intersection moves exactly those, and builders' goldens and conformance tests will name them.
- **Not executable.** "Scans the committed tree for proving-repo locators" has three gaps:
  - no path set: the proving repo's own source trivially contains every locator;
  - no extractor from free text: `split_manifest.py contact` takes `{locator, source}` lines, not prose;
  - no positive control.

**Fix:** reuse the instrument that exists, `split_manifest.py --check-report` (EP §9.4). It already does
whole-token matching of held-out probes, prints hits by hash only, and drops probes that are public through dev or
C. Run it over a named builder-artifact path set: `cpg/` minus `cpg/eval/`, plus the P1 step reports. Then the
custodian triages each hit into contact or EP §8.2 mechanism before moving anything. The positive control is a
held-out probe planted in a scratch copy of one artifact, which must be reported as one hit.

**M5 — The oracle relocation (§6.4) is right, but its reason is wrong and its reach is too narrow.**
*Verified against EP §9.3 and the step text.*
- **The reason is wrong.** "Beside the held-out files and under the same guard" implies a read guard. There is
  none: EP §9.3 says the manifest *"detects modification, not reading"*. `~/.local/share/ccg-eval/` is exactly as
  readable by a same-user builder as `cpg/.ccg-work/`. The real reasons are these:
  - the location is outside every builder's working tree and sweep paths (an Explore/grep under `cpg/` never
    reaches it);
  - it is outside `cpg/.ccg-work/`, which builders must use (P1-04's slice bundles, D9);
  - it sits with the custodian's other files.
  State those reasons, because the "guard" wording would let someone skip the brief rule.
- **The reach is too narrow.** ML §4.1 and P1-36 send each oracle miss to the *next* held-out set. But nothing
  stops those misses from being printed in P1-37's committed report. EP §12.1 step 2 keeps miss content
  builder-invisible, and `--check-report` probes only `heldout/*.jsonl`, not `<E12>/oracle/`.
- §6.2 item 1 also names gate answers, the triage list and the noise sample as contact-making artifacts, but no
  step says where they are written. The same leak class applies to them.

**Steps it changes:**
- **P1-47:** the path, and drop "mode 0700, P1-02" as the protection;
- **P1-36:** `oracle.py` reads there and reports counts only;
- **P1-37:** the oracle misses are counts by triage class;
- **P1-32 (32b):** the gate answers and triage go under `<E12>/`, and `--check-report` takes the oracle edges as
  probes;
- **P1-35:** where the AC-24 run outputs live;
- **P1-43:** the fresh-item list reads the oracle misses from there.

**Fix:** write one rule: "every held-out-derived artifact lives under `~/.local/share/ccg-eval/<E12>/`". Then list
these six steps in §10.

**M6 — `scip-python` shells out to `pip` unless it is given `--environment`, so P1-15's argv check fails under
§2.2 as written.** *Checked against the tool's README via search; not run.*
- §2.2 mounts a provisioned venv and keeps "the argv check of P1-15 asserts no `npm`/`pip`/`uv` invocation".
- `scip-python` determines package names and versions by calling pip, unless `--environment <env.json>` (name,
  version, files per package) is passed.
- P0's online reference ran `scip-python index --project-name falkorchat` (`corpora.yaml:26`), so it used pip
  discovery.

**Fix:**
- At provisioning time, generate `env.json` from the venv (`pip inspect` or `uv pip list`, converted), record its
  sha256 in `ccg-p1-indexer-v1`, and pass `--environment`.
- Name the configuration difference from P0 as an M1 confound.
- Alternatively, narrow the argv check to package-*installing* commands, but that is weaker.

**M7 — The `InterpretDraft` cannot say "this span qualifies that one", and 5 of 12 dev unambiguous questions need
it.** *Verified on `cpg/eval/dev/ac24.jsonl`.*
- Five questions name one anchor by a symbol span plus a qualifying span:
  - QW-30 ("classe Runner … (decisionmodels/runner.py)"), QW-32, QW-5 and QW-7 add a file span;
  - QW-13 ("send_message tool on falkor-chat mcp") adds a server span.
- §4.2 gives `anchorSpans` as 1–3 independent `{text, kindHint}` entries. The resolver then either builds two
  anchors (wrong against a one-anchor gold) or must guess that one span qualifies the other, which is an
  unwritten rule.
- The dev entry check needs ≥ 11/12, so this alone can fail it.

**Fix:** make each anchor a list of spans (`anchors: [{spans: [{text, kindHint}]}]`), or add a `qualifies: <index>`
field. Then state the resolver's join rule: a file span plus a symbol span gives a qualified-name match, and a
partial path such as `falkorchat/api.py` matches by suffix. Add QW-30 and QW-27 as named resolver tests.

#### Minor

**m1 — Whether the three stakeholder decisions are forced (the brief's question).** *Verified against the
requirements text.*
- **S-1:** option (a) is excluded by FR-8, unless the stakeholder amends it. FR-8 says *"When an impact cannot be
  proven … it is still listed, marked as possible"*, and an impact through a flag-false unit is one that cannot be
  proven. The choice among (b), (c) and (d) is the method owner's and `architect`'s, made by the pre-registered
  rule. The proposal's "call-level only" wording is design text, not a requirement.
  - §3.5's decision rule ranks only the as-built walk, V-flow and V-both, so (d) is never measured. A union always
    has the highest s, so either compute (d) in P1-39 from the two listed sets, or drop it.
  - Also, (b) and (d) would make AC-25's control half listable as `possible`, which changes Q-B. §9 says Q-B is
    unchanged.
- **S-2:** not forced. AC-5's confirmed bar and AC-6's "none confirmed" both hold vacuously under (a). The note's
  case for (b) is sound. Under (a), P1-42's AC-6 status assertions on python/TS fixture units cannot fail, and
  root `REVIEW.md` makes that a finding against P1-42. Route it to `architect` with S-2.
- **Q-D:** see M3.
- **AC-24 under mixed systems (§4.5):** this one *is* forced. AC-24's text reads *"local in any system containing a
  local-only repository"*. Cite it as the source rather than deriving it.

**m2 — §6.1 pins `algorithm.version` "1.1", but the note's own recommendations would bump it.**
- answer-spec2 §1.2 puts the flow-fallback rule (§5) inside the walk definition.
- S-1(b)/(d) (new rows walked in flag-false units) and S-2(b) (new labelling) each change it, so the version would
  become `"1.2"` if adopted after 1.1 ships.
- **Fix:** say "the string the engine emits at the freeze commit", and name the S-1/S-2 bump.

**m3 — §6.2's builder definition contradicts itself.**
- "Writes under `cpg/` (other than custodian outputs under `cpg/eval/`)" makes P1-36's `data-scientist`
  (`noise.py`, `oracle.py`, `cpg/tests/test_noise.py`) and P1-32/P1-35's `qa-engineer` (`gate.py`,
  `ac24_smoke.py`) builders.
- The next bullet says the method owner and the gate runner are not builders.
- **Fix:** define the role per dispatch. Tooling written before the record is pinned (§6.1). An exposed dispatch
  never precedes a builder dispatch of that same tooling.

**m4 — §6.3's out-of-scope rule needs a named scope source.**
- If "out of scope" is read from the system manifest (P1-03), a scope bug in P1-03 turns a real extraction miss
  into a `key-error`. That raises "recall excluding key errors".
- **Fix:** take scope from `cpg/eval/eval-manifest.yaml`'s `exclusions` at `EVAL_COMMIT`, and report any
  disagreement with the system manifest as a defect.

**m5 — The §10 "Open, for architect" list names three amendments; the note makes about a dozen.** Also unrouted:
- P1-09: both definitions, the open-miss list, and M1's golden;
- P1-22: the two-step-chain guard test of §3.4;
- P1-15: M0, M1, env provisioning, M6;
- P1-32b: the harness-file sha256 pin, the out-of-scope `key-error`, storage;
- P1-34: the frame exclusion and the M2 scorer;
- P1-36: the scope filter, the two burdens and the baseline;
- P1-37: three recall units and chain length;
- P1-35: repeat 1 is scored;
- P1-43, P1-47: M5.

Without the list, the sub-plans stay stale while implementers follow the note. **Fix:** list each as a one-line
step-text amendment.

**m6 — Restated facts (growth-stops check; this is the note's first version, so nothing here is rework growth).**
Root `AGENTS.md`'s "a fact has one home" bullet applies. Restated:
- F5's held-out counts and §4.3's thresholds and Wilson intervals restate the manifest and EP §10.5, which already
  computes 23/24, 11/12 and their intervals;
- the §6.1 table restates EP §12.3's fields that the note does not change;
- §2.1 restates the fitness report §a1.2;
- §5.1's first paragraph lists EP §11's contents.

Key figures appear three times each: 465 in §0, §3.1 and §8; 0.546 in §0, F3 and §3.2; 23/24 in §0, F5 and
§4.3. §0's answer column (about 650 words) is a second copy of the whole note.

**Fix:** §0 gives the decision plus a section pointer. F5 cites the manifest and EP §10.5. §6.1 lists only the
fields it rules on. This cuts roughly 800–1,000 words with no loss.

**m7 — The oracle sources (§6.4 item 2) assume a Python call frame.** At `EVAL_COMMIT`, `falkor-chat/scripts/` has
23 `.sh` files and 6 `.py` files, and the HTTP callers are `curl` in shell. A PATH-shimmed `curl` can record the
parent script, via `/proc/$PPID/cmdline`, but not a function or line.
**Fix:** state file-grain attribution for shell consumers. Say that a script needing live services
(`start_agent_team.sh`, `load_test.sh`) runs against the throwaway or is listed unrunnable, never skipped silently.

**m8 — AC-24: a voided repeat 1 has no score.** §4.3 makes repeat 1 the scored draw, and EP §10.4 voids an item's
repeats on `cacheHit == true`. **Fix:** a voided repeat 1 scores "not understood" and stops the run as a C5 failure,
since the cache must be empty for the gate snapshot.

#### Nit

- **n1 (F3):** "All 15 are now development contact". `sample/javascript-flow-tier2.jsonl` holds 3 methods, and
  `labels/javascript-flow.jsonl` has 12 lines, so the tier-2 three were drawn but never labelled. Under EP §8.2
  they are not contact. The conclusion stands, since 3 methods cannot reach 120 pairs.
- **n2 (F3, §3.2):** "0.632 if the i18n method were recovered" is (87 + 9)/152, which uses chain credit. Under the
  direct definition it is (83 + 9)/152 = 0.605.
- **n3 (F6):** the count is at HEAD. I re-ran an unfiltered sweep (tests included) at `5064db3b` and at
  `EVAL_COMMIT` `cf7b2042`: 0 route-parameter catch-alls at both. Add the `EVAL_COMMIT` figure, since the current
  held-out keys were written there.
- **n4 (§3.5):** state that P1-39's snapshot is built at `EVAL_COMMIT`, where the dev flow items' locators were
  validated.
- **n5 (Appendix):** `flow.py` writes `m.json` into the working directory. Write it to an argument path or a
  tempdir.

### What's solid

- **Every load-bearing figure reproduces.** I checked them with my own scripts, not only the note's (Appendix A):
  481/308/173 and the cause split; 457/459/468; 149 of 151; the 22-in-8 and 13-in-5 residuals; ICC 0.04–0.17;
  JS 152/61/83/87; the 28-item signatures (10 pure `return-use`); H 159, 62 anchors, 58 clusters; every Wilson
  bound; 23/24, 11/12, 6/6; Q-C's algebra (pp ≥ 0.2 ⇒ Q ≤ 4); p90 = the 56th of 62; 0/20 → 0.161.
- **The entry value 465 is the conservative end of a 461–465 spread.** The direct ceiling 459 is below every entry
  value, so "unreachable" is robust. Under the chain ceiling, 3–7 open misses are allowed.
- **The plan corrections are real.**
  - Seed prefix: P0-ML §4.4 and `run_fitness.py:50` use `ccg-p0a-v1-flow|`, and `ccg-p0f-flow-v1|` is EP §3.6's.
  - 455 vs 459: the index's prose omits the contains-join that P1-09 ships.
  - Operating curve: the plan quotes P0-ML's independent n = 120 line correctly, but that is the wrong curve for
    python's structure.
- **The harness gap is real.** `cpg/eval/tools/` is excluded from `coreTreeHash`. The pin-by-`git show` remedy is
  executable.
- **The design choices are sound:**
  - the verbatim-span fabrication guard plus the deterministic resolver;
  - fixture-only few-shot examples;
  - repeat 1 pre-fixed as the scored draw;
  - prospective-only ratification, "never from the measured value";
  - the offline/online M1 delta as the primary new fact;
  - the merge rule that never drops a candidate;
  - F4 and S-1, which identify the caller-only fallback as predicted-unsafe (predicted 6/8 python, 4/20 TS/JS).
- **Builder-blindness:** the note names no held-out content. Its proving-repo locators are dev items, files, or
  EP §8.2 mechanism. Its held-out figures are manifest counts.

### Open questions

1. **For the stakeholder, via `tico` (M3):** redo the 40-item EP §7.3 spot-check yourself, as S36 requires, or
   waive S36 for gate 1?
2. **For the method owner (M1):** the narrow "synthetic cond is transparent" rule, or the general chain rule with
   the spurious-rate inconsistency disclosed?

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: M3 and m1 of this review, where two of three "stakeholder
options" were already decided or excluded by EP §12.1, S36 and FR-8.

- A stakeholder option a document recommends is checked against the requirements text, its decision log and the
  standing method contracts. An option they already decide or exclude is a finding, and the document must state it
  as a rule or a waiver request instead.

## Pass 2 (2026-11-09)

### Scope & verdict

**Reviewed:** the revised `docs/plans/connected-code-graph-p1-ml.md`, revised in place and uncommitted (6,755
words, down from 9,553). I rechecked it against the same contracts as Pass 1, plus:
- `cpg/eval/tools/split_manifest.py` and `match_keys.py` (the actual `check-report` interface);
- `cpg/sandbox/offline-run.sh`;
- `docs/plans/connected-code-graph-egress-spec3.md` §7.4;
- the revised `-answer-spec2.md` §5;
- the requirements AC-5, AC-25, FR-8, S8 and S36.

I read no held-out file.

**Verdict: needs changes.** There is still no blocker, and Pass 1's majors are all fixed or superseded. Five new
majors remain, each a local text edit:
- the new projection rule breaks P1-09's own superset test by construction (N1);
- the runner gap is routed to the wrong owner (N2);
- one dev ambiguous question is inexpressible (N3);
- the freeze-check path set is filtered (N4);
- Q-C omits the 4:1 ratio that the requirements make provisional (N5).

Q-D can go to the stakeholder now. Q-C and Q-B should be corrected first (N5, N6).

CPG: considered, not relevant — no graph is loaded for `cpg/`; the figures were rebuilt from the P0 data files directly.

### Dispositions of Pass 1 findings

- **M1:** fixed and superseded by §3.4's synthetic-cond transparency rule. I rebuilt 467 with my own instrument
  (Appendix C).
  - Classification from source at `0caf85e3`: the `try` conds `retry.py:134` ×5 and `services.py:449` ×2, the
    generator-expression iterator `retry.py:81` ×1, and the `for` iterator `test_repository.py:4703` ×1.
  - Result: 467/481 = 0.9709, cluster interval [0.945, 0.996]; 14 residual misses in 6 methods (4, 4, 3, 1, 1, 1).
    All of this matches the note.
  - Spread: 466–467 depending on whether the generator iterator is recognised as synthetic (Appendix C).
  - The transparency rule cannot connect any of the 14 residual misses, because none of their sinks is under a
    synthetic cond, so there is no upward spread.
  - The rule agrees with P0-ML §8.4's "synthetic conds are not points" rule.
  - The `test_repository.py` pair correctly stays a miss. Under P0-ML §8.4's early-exit and nearest-governing
    rules, the golden pair (4704 → 4706) is right and the projection's loop-header edge is wrong, so it is a
    genuine projection miss.
  - **Residual gap (minor):** the rule names its two shapes but not their *recognizers*. Both iterator conds carry
    the same code, `iteratorNonEmptyOrException`. The goldens `retry.py:81` (positive) and `test_repository.py`
    (negative) do discriminate them, which is good. Also unstated: what happens to a synthetic cond with no
    governing real cond (drop the edge?) and to a synthetic-under-synthetic case (walk up to the nearest real
    cond?).
- **M2:** fixed by construction. Scoring stays direct-edge, so `s28_metrics.py:273` is unchanged and no chain
  scorer is needed. See N1 for the cost this moves into P1-09.
- **M3:** fixed (§9 Q-D). It reframes the decision as "redo or waive", and EP §12.1's `key-error` confirmation is
  now stated as a rule. A procedure gap remains (N7).
- **M4:** partly fixed. `check-report` is verified to exist with the interface the note uses:
  `check-report --report F (--heldout-dir D | --heldout F…) [--dev F…] [--contact F]`, one report per call,
  printing hits as 12-hex probe hashes and exiting 1 on any hit. The planted-probe positive control and the
  contact/mechanism triage are present. The path set is filtered (N4).
- **M5:** fixed.
  - §6.4 states one custody rule, says there is no read guard, and limits committed reports to counts.
  - All six routed steps are in §10: P1-32a/b, P1-35, P1-36, P1-37, P1-43, P1-47.
  - `check-report --heldout` with an explicit list already accepts an extra key-schema file
    (`check_heldout_list` checks only that the required names are present). P1-32a therefore only needs to
    *write* the candidates in that schema, and must pass them by `--heldout`, because `--heldout-dir` globs
    `ac5-*.jsonl` only (nit).
- **M6:** the method part is fixed: `--environment` with `env.json`, its sha256 recorded, and the pip-discovery
  confound named. Confirmed: `SBX_FRONTENDS` (`offline-run.sh:51`) has no indexer entry. The routing is wrong (N2).
- **M7:** fixed.
  - The span groups and the join rule cover the dev questions I counted: QW-30 (symbol plus a path suffix),
    QW-27 (`falkorchat/api.py:603`, partial file:line), QW-13 (tool plus server), QW-7 (file before symbol), and
    QW-32, QW-5 and QW-3 under the same rules.
  - A different dev item, QW-47, is not covered (N3).
- **m1:** fixed.
  - S-1 is now a pre-registered design rule. The four walks are measured, and the rule picks the highest share,
    then the lowest burden. V-union ⊇ V-flow, so a tie at maximum share goes to V-flow, which is coherent.
  - FR-8 is read correctly: an impact through a flag-false unit cannot be proven, so it must still be listed as
    `possible`.
  - The rule cannot pick a walk that contradicts S-2 (§6.6). §6.6 labels "§3.5's added rows" `possible`, and an
    item's status is its first `possible` hop. Every item that reaches a flag-false unit through an added row is
    therefore `possible` under every walk.
  - One wording conflict remains (N8).
- **m2:** fixed (§6.1: "the string emitted at the freeze"; the bump is named).
- **m3:** fixed (§6.2: roles per dispatch).
- **m4:** fixed (§6.3: scope from `eval-manifest.yaml`).
- **m5:** mostly fixed. §10 lists 16 amendments; four owners are still missing (N9).
- **m6:** fixed. The note lost about 2,800 words, §0 is now pointers only, and §4.3 cites EP §10.5 instead of
  restating it.
- **m7, m8:** fixed (§6.5 shell attribution at file grain; §4.3 a repeat-1 `cacheHit` scores "not understood" and
  stops the run).
- **n1–n5:** fixed (F3 tier-2; 0.605; F5 at `EVAL_COMMIT`; §3.5 snapshot at `EVAL_COMMIT`; `flow.py` takes an
  output path).
- **Proposed `REVIEW.md` line:** stands, for `teco`.
- **Stakeholder answer folded in:** S-2 → "only the flow-dependent hops become `possible`" (2026-11-09) is now
  §6.6, and `-answer-spec2.md:281` matches. It is not yet in the requirements decision log (N9).

### New findings

#### Major

**N1 — §3.4's transparency rule makes P1-09 behaviour (5) red by construction.** *Verified against step text.*
- P1-09 (5) requires the shipped projection's `controlledBy` records to be "a subset of the fixed projection's".
- §3.4 emits `point → C` **in place of** `point → S`, so every rewritten shipped edge is absent from the fixed set.
- §10's P1-09 line does not amend (5).

**Fix:** restate (5) as "subset, modulo the rewritten synthetic-cond edges, each of which is listed with its
replacement". Add the pair-level guard "no golden pair connected by the shipped projection is unconnected by the
fixed one". `slice/test_projection_not_regressed.py` needs the same carve-out.

**N2 — The indexer runner change is a sandbox *contract* change, not only a `devops` image edit.** *Verified.*
- The closed frontend set, its `$JOERN_HOME` executable mapping and the flag rules are `-egress-spec3.md` §7.4
  (P1-28, `architect` with `security-expert` advisory). Anything outside the set is `unknown-frontend`.
- The `scip-*` tools are Node executables, outside `$JOERN_HOME`.
- P1-48's file list (`Dockerfile`, `build.sh`) does not include `offline-run.sh`.

**Fix:** route it in two parts:
- (i) a §7.4 revision through P1-28's spec: the two tools, their executables, the allowed flags, and `env.json`
  and the dependency trees as read-only mounts, with the AC-8 sandbox tests that pin it;
- (ii) the `offline-run.sh` implementation added to P1-48's file list.

Keep "not run: runner" as P1-15's third outcome.

**N3 — The draft cannot express QW-47's ambiguity, and 2/2 dev ambiguous items are required.** *Verified on
`cpg/eval/dev/ac24.jsonl`.*
- QW-47 ("relações da tool ingest_documents (MCP)") has `goldOptions` = the same anchor with `direction` `impact`
  and `depends-on`. The question names neither direction.
- §4.2 allows `readings` only "when the question names two use cases", and `direction` is a single required value.
- The model must therefore pick one direction, which is a silent single interpretation and fails EP §10.4.
- All 6 held-out ambiguous items must also pass.

**Fix:**
- Allow `direction: null`, or require readings when the question states no direction.
- The resolver expands an unstated direction into both `options`.
- Make QW-47 a named resolver test and cover the shape in one few-shot example.

**N4 — The freeze-check path set (§6.2) is filtered, so it is a bound, not a sweep (root `REVIEW.md`).**
*Verified by listing.*
- `docs/**/connected-code-graph-p1*` plus `cpg/` minus `eval/` and `spikes/` misses P1 builder-facing artifacts:
  - the P1 specs `-answer-spec2.md`, `-bundle-spec5.md`, `-egress-spec3.md` and their reviews (`answer-spec2`
    §5 cites AC-1's `PROVIDES ←` into `falkor-chat/server`);
  - P1-27's `skills/cpg-analysis/SKILL.md` and `claude/teco/teco.md` (usage recipes);
  - P1-34's edit under `cpg/spikes/`.
- Also, hits print as hashes, but the custodian's mapping from hash to probe is a procedure no verb provides.

**Fix:**
- Define the set mechanically: every file changed by a builder dispatch between the S47 commit and the record
  commit (`git diff --name-only <S47>..<record>`), minus proving-repo source outside `cpg/` and custodian outputs.
  Map each excluded path to a reason.
- Give the custodian the reverse map: recompute `sha(collapse(probe))[:12]` over the held-out probes, privately.
- Treat exit code 4 (unreadable file) as a failure, never a skip.

**N5 — Q-C (§5.3, §9) omits the one threshold the requirements make provisional, and its "never from the measured
value" contradicts AC-5's text.** *Verified against the requirements.*
- AC-5 sets the 4:1 cap "(provisional; ratified or adjusted once the first phase has measured the real ratio)",
  and decision S8 says "provisional until phase 1 measures it".
- §5.3 asks the stakeholder only about pp ≥ 0.20, p90 and 5×.
- §5.3's option (b) forbids a value derived from the measurement, but the requirement says adjustment *follows*
  the measured ratio.

**Fix:**
- Add Q ≤ 4 to the ratification, as a requirements obligation rather than an option.
- Keep "a floor set to the score certifies nothing" as the method owner's *recommendation*, stating that AC-5
  permits adjusting to the measurement.
- Prospective-only stays (gate 1 is gated at Q ≤ 4 as EP §11.3 says).

#### Minor

**N6 — Q-B (§9) is forced either way by AC-25's text, so it is not a stakeholder question.**
- AC-25 requires that "both effects are listed … confirmed before possible", and FR-8 marks an unprovable impact
  `possible`. So whether `possible` is acceptable is already answered: yes.
- If V-flow or V-union ships, the control half closes, given P1-42's fixture.
- If the as-built walk or V-both ships and does not list the guarded point, AC-25 is not closed. The only
  stakeholder act left is a waiver.

**Fix:** reword §9 Q-B that way, and update index Q-B (N9).

**N7 — Q-D(a) needs a post-split correction procedure.**
- EP §7.4 runs corrections and escalation *before* the split. The 40-item sample (EP §7.3) was drawn over the
  union, so its items are now in dev, held-out and reserve.
- A stakeholder "incorrect" on a held-out item changes a held-out file. The committed manifest's sha256 then fails
  `--verify` until the custodian re-records it.

**Fix:** state the procedure. An incorrect item is corrected or removed under EP §7.4. The manifest is re-recorded
and committed before the `ac5-gate-1` record. The escalation trigger applies as written. Counts go into the report.

**N8 — Wording conflict between §3.5 and §6.6.** §3.5 defines V-both and V-flow as "all `possible`", while §6.6
keeps contract and containment hops at their own status. The status is per hop, so say "every added-row hop
`possible`" in §3.5.

**N9 — Unrouted owners.** Missing from §10:
- index §5 option 5 ("Not decided here … method owner's rule") and §5's "method owner's and stakeholder's call"
  for the fallback, both now ruled;
- index Q-B and Q-C;
- the proposal's "every item is shown as *possible*" and "call-level reach only" (`connected-code-graph.md`
  "Flow availability"), which S-2 and §3.5 now contradict (`architect`);
- the S-2 answer, absent from the requirements decision log (`tico`; `grep 2026-11-09` finds nothing there);
- N1 and N2.

#### Nit

- §3.3 says "adjudicates … under P0-ML §8.4 (v1.17) plus §3.4", but §3.4 no longer changes adjudication; drop
  "plus §3.4".
- The revision carries no dated revision line (collision rule 5, revise-in-place form).

### Growth-stops check

- The revision is rework, so it must not grow, and it shrank by 2,800 words. The only additions are new scope:
  the runner prerequisites, the custody rule, span groups, the four-walk rule and §6.6, the folded-in answer.
- No finding→resolution table, no restated history, and no review-finding IDs.
- Two remaining restatements are small and tolerable: §2's opening four P0 figures, and §3.3's "Why 30" repeating
  F2's numbers.
- Same-revision fixes against each other: they are consistent, except N1 (§3.4 vs P1-09 (5)), N8 (§3.5 vs §6.6)
  and the stale §3.3 reference.

## Pass 3 (2026-11-10)

**Reviewed:** rev 3 of the note (7,551 words; "Revised 2026-11-10 in place after review Pass 2"). I checked it by
running things:
- the 467/466 spread, rebuilt again from the P0 artifacts;
- the recognizers, against the `0caf85e3` source using `ast`;
- `git diff --name-only 511b14cf HEAD`, against the files Pass 2 listed as missing;
- `cpg/eval/eval-manifest.yaml`'s scope exclusions.

I read no held-out file.

**Verdict: needs changes**, narrowly. One major remains (N12), and it is a one-clause fix. Everything else is fixed.
Re-gating N12 needs only that clause confirmed.

CPG: considered, not relevant — no graph is loaded for `cpg/`; the checks ran on the P0 data files, the source and git directly.

### Dispositions

- **N1:** fixed (§3.4). Behaviour (5) and `slice/test_projection_not_regressed.py` are restated as "subset, modulo
  the rewritten edges, each listed with its replacement", and the pair-level guard is added. One wording gap
  remains (N11).
- **N2:** routed to `architect` as intended. The requirement is stated and cited (§2.1): the closed set, the
  executables and the flag rules are named as `-egress-spec3.md` §7.4, plus P1-48's file list (§10).
- **N3:** fixed.
  - `direction: null` is expanded by the resolver into both directions as `options`, which matches QW-47's two
    `goldOptions` (same anchor, impact and depends-on).
  - QW-47 is a named test, and the few-shot examples cover a `null`.
  - The `readings` rule (two use cases) is unchanged and does not conflict.
  - The reverse error, `null` on a question that does state a direction, is counted as over-asking. That matches
    EP §10.4's k/24 measure and the ≥ 23/24 bound. A residual risk remains (N10).
- **N4:** mostly fixed, but its core is reopened by the exclusion clause (N12).
  - Verified: S47 is `511b14cf` ("S47: split, dev files, held-out manifest…"). The diff to HEAD has 204 files.
  - `-bundle-spec5.md` and `skills/cpg-analysis/SKILL.md` are already in it. `-answer-spec2.md` and
    `-egress-spec3.md` are untracked, and `teco.md` and `s28_run.py` are unedited, so all four enter at the
    record commit by construction.
  - Fixed: exit 2/4 fail rather than skip, the reverse map, and `--heldout` with the candidates.
- **N5:** fixed (§5.3, §9). Q ≤ 4 is now a requirements obligation; I checked that against the AC-5 and S8 wording.
  - (b) now says "AC-5 permits adjusting the cap to the measured ratio".
  - (c) is barred for the cap.
  - The cost-based reasoning is now a recommendation.
  - Gate 1 is still gated at Q ≤ 4, per EP §11.3.
  - This is consistent with §10's index line.
- **N6:** fixed (§9 Q-B: decided by AC-25 and FR-8; a waiver only if the guarded point is unlisted).
- **N7:** fixed. Under (a), held-out and reserve corrections are made, then `manifest` and `verify` are re-run and
  committed before the `ac5-gate-1` record.
- **N8:** fixed ("every added-row hop `possible`").
- **N9:** fixed. §10 routes the index items (§5 option 5, the fallback ownership, Q-B, Q-C), the proposal wording,
  and the S-2 decision-log entry for `tico`.
- **Nits:** fixed. "plus §3.4" is gone, and the dated revision line is at line 13.
- **M1's recognizer residual:** fixed.
  - The `try`-lowering and generator/comprehension recognizers are stated on the source construct.
  - I checked them with `ast`: `retry.py:134` and `services.py:449` are statements directly in a `try` body;
    `retry.py:81` is a generator expression; `test_repository.py:4703` is a `for` statement.
  - The nested and ungoverned cases are ruled.
  - The 467/466 spread re-ran unchanged: 467 with both recognizers, 466 with `try` only.

### New findings

**N12 (major) — §6.2's exclusion "proving-repo source in system scope outside `cpg/`" removes exactly the P1
builder artifacts that N4 asked to add.** *Verified.*
- `eval-manifest.yaml`'s exclusions are `cpg/`, `deprecated/`, `docs/**/connected-code-graph*` and
  `docs/BACKLOG.md`, plus the content-check files. So `skills/` and `claude/` are in system scope.
- P1-27's `skills/cpg-analysis/SKILL.md` (already in the diff) and `claude/teco/teco.md` are therefore excluded,
  even though §6.2 says the set "covers … skills, agent prompts".
- The rationale "its own locators are trivially present" also fails for them: a recipe file is not the source it
  names. It does not fully hold for real source either: bare tool-name `keyNorm` probes ≥ 8 characters, such as
  `send_message`, do hit source text.

**Fix:** carve back in every in-scope path changed by a P1 dispatch, at least P1-27's two files and P1-25's
`.mcp.json` proposal. Better: invert the rule. Include every changed path, and exclude only paths whose changes
all come from non-P1 commits in the range.

**N10 (minor) — The direction cue list (§4.2) could be read as closed.**
- Two dev unambiguous questions state their direction with cues that are not in the list: QW-3 ("blast radius")
  and QW-11 ("what calls this route").
- A prompt that treats the list as exhaustive returns `null` on them, which counts as over-asking. Two such
  answers fail ≥ 11/12.

**Fix:** mark the list "for example", and add QW-3 and QW-11 as negative resolver tests ("direction stated
without a listed cue → not `null`").

**N11 (minor) — The N1 carve-out covers only edges that are rewritten to one replacement.** §3.4 also *drops* an
ungoverned edge and emits *several* edges when there are several nearest real conds.
**Fix:** list each shipped edge with its replacement set, which may be empty or plural. The pair-level guard is
unaffected, because no golden pair can name a synthetic cond.

**N13 (minor) — Two P1-09 fixes act on one node.** `services.py:449` is both the `try`-lowering cond and the call
point of the "call that is itself a CDG source" miss. That miss is a golden pair (`isinstance(value, str)` (447) →
`loads` (449)), which the 459 ceiling credits to the unverified fix. My `ast` check found the other 9 of those 10
misses outside any `try` body, so there is no misclassification.
**Fix:** add one joint golden on `_normalize_opaque` asserting all three pairs (447 → `loads`, 447 → returns 449
and 451), so that neither fix can silently undo the other.

### Growth-stops check

- The note grew by 796 words (6,755 → 7,551). Almost all of it is newly specified behaviour that Pass 2 found
  missing:
  - the recognizers and the nested/ungoverned rule;
  - the behaviour (5) carve-out;
  - `direction: null` and its cues;
  - the path-set mechanics;
  - the 4:1 cap obligation;
  - the Q-D post-split procedure;
  - §10's new routing rows.
- I judge that new scope. No text was appended beside superseded text, and there is no history or disposition
  table.
- One small restatement: the 4:1 obligation is stated in §5.3 item 1, §9 Q-C and §10. §9 could point to §5.3
  (nit).
- Same-revision fixes against each other: consistent, apart from N11 (carve-out vs. the drop and multi-edge
  cases) and N12 (path set vs. exclusion). N3's `null` vs. `readings` and over-asking, and N5's wording vs.
  AC-5/S8, agree.

## Pass 4 (2026-11-10)

**Scope:** a narrow check of rev 4 (7,593 words): N10–N13, the 4:1 obligation, and the two folded stakeholder
decisions. I checked them by running `git diff --name-only 511b14cf HEAD` (now 229 files), the per-path commit
history, and the `0caf85e3` source. I read no held-out file.

**Verdict: needs changes**, narrowly. One major remains (N15), a text fix to the decision-log entry and to EP's
wording. Everything else is fixed or minor.

CPG: considered, not relevant — no graph is loaded for `cpg/`; the checks ran on git and the source directly.

### Dispositions

- **N12:** fixed (§6.2). The exclusion is inverted: drop only paths whose changes all come from non-P1 commits,
  plus custodian outputs. I ran the diff:
  - `skills/cpg-analysis/SKILL.md` and `claude/teco/teco.md` are in it, but today's changes to them are non-P1
    (the SKILL.md change is `578aa9e1`, a team-learnings sweep). Excluding them now is therefore correct, and they
    re-enter as soon as P1-27 edits them.
  - `.mcp.json` is not in the diff yet. P1-25 writes only a proposal in its report, and the file changes under
    P1-27. It enters then.
  - Residual (minor, N17): the "P1 commit" predicate.
- **N10:** fixed in substance. The cues are "examples, not a closed list", and QW-3 and QW-11 are named negative
  tests. One same-revision conflict remains (N14).
- **N11:** fixed. The replacement set is "one edge, several, or none".
- **N13:** fixed. I verified the source at `0caf85e3`, `services.py`: 447 `if isinstance(value, str):`,
  449 `return json.loads(value)` inside `try`, 451 `return value` in the `except`. The joint golden's three pairs
  (447 → `loads`@449, 447 → ret@449, 447 → ret@451) match the stored miss rows.
- **4:1 obligation:** stated once (§5.3), with §10 carrying only the decision-log routing.
- **S36 / Q-D decision ("redo the sample"):** consistent with S36, EP §7.4 and §6.2. The manifest re-record sits
  under `cpg/eval/`, outside `coreTreeHash`, so it does not break the freeze. A shortfall path is missing (N16).
- **Q-C decision ("ratified as proposed"):** it contradicts AC-5's own text (N15).

### New findings

**N15 (major) — The noise ratification is recorded as if AC-5's precondition were met, but gate 1 has not
measured anything.**
- AC-5 sets the cap as "provisional; ratified or adjusted **once the first phase has measured** the real ratio".
- S8 says "provisional until phase 1 measures it", and EP §11.3 says the three bars "are ratified after P1's first
  measurement".
- A 2026-11-10 ratification, taken before `ac5-gate-1` runs, is the stakeholder's to make. But as written, three
  documents still contradict it, and the proposed log entry says only "affects AC-5, S8".

**Fix:**
- The log entry (`tico`) names what it overturns: S8 and AC-5's "once the first phase has measured" clause, per
  the "a reversing entry names the entry it overturns" rule.
- `tico` rewords AC-5's parenthetical accordingly.
- §5.3 states that it replaces EP §11.3's "ratified after P1's first measurement" from P1 on.
- Optionally, state a re-open trigger: gate-1 values far outside the bars. Without one, the measurement that AC-5
  waited for cannot inform the bars.

**N14 (minor) — The fix for N10 now undercuts its own tests.**
- "blast radius" (QW-3) and "what calls" (QW-11) were *added* to the example cue list, so the two negative tests no
  longer exercise a direction that has no listed cue.
- Almost every cue is lifted from dev question text (QW-3, QW-11, QW-7, QW-17, QW-32, QW-37, QW-39, QW-42). If the
  list reaches `prompts/interpret.txt`, the dev entry check is tuned on its own items. That contradicts §4.2's
  "dev questions are kept for scoring".

**Fix:** keep dev phrasing out of the prompt's cue list, and keep QW-3 and QW-11 as unlisted-cue tests. Or state
that the dev check is then a development check, not a clean one.

**N16 (minor) — The Q-D procedure has no shortfall path.**
- `wiring` sits exactly at its floor of 8 held-out items, so a single "incorrect" removal leaves it unable to meet
  `H_s ≥ 8`.

**Fix:** name the existing route. Top up from the reserve by EP §9.1 step 4, or mark the stratum and report it
under SQ-P0-1. Do this before the manifest re-record.

**N17 (minor) — N12's predicate needs a mechanical form.**
- "Changes all come from non-P1 commits" needs a test for what counts as a P1 commit.
- `git log --grep='C-1001'` over the range also matches P0-close commits (`84ac62d5`, `262161e7`).
- `teco` commits can bundle P1 and non-P1 files.

**Fix:** define a P1 commit as a subject matching `C-1001 P1` (the convention in `fe0fdf96` and `5785fdc1`), and
treat a mixed commit as P1 for every file it touches. Also say that `cpg/eval/tools/` (P1-32's tooling, written in
a P1 dispatch) is *not* among the excluded "custodian outputs". Only the manifest, the dev files and the records
are excluded.

**Nit:** §5.1 item 4 still gives the status split's purpose as "so ratification is not confounded". Ratification
has happened, so the split now serves the next gate's bars.

## Pass 5 (2026-11-10)

**Verdict: approve with suggestions.** No major remains. I read no held-out file. CPG: considered, not relevant — no `cpg/` graph is loaded.

- **N15:** fixed. §5.3 and the §10 log entry name AC-5's clause, S8 and EP §11.3 as overturned, and `tico`
  rewords AC-5. Omitting the re-open trigger is acceptable; it was optional.
- **N14:** fixed. None of the 10 new cue phrases occurs in any `cpg/eval/dev/ac24.jsonl` text (substring check,
  casefolded). Only function words overlap. QW-3 and QW-11 are therefore genuine unlisted-cue tests.
- **N17:** fixed. `git log 511b14cf..HEAD | grep -E 'C-1001:? P1'` matches `8e6bd0bb`, `fe0fdf96` and
  `5785fdc1`, and excludes `84ac62d5` and `262161e7`. `cpg/eval/tools/` stays in the set.
- **N16:** fixed, with one suggestion. The reserve-cluster top-up is a sound route. But EP §9.1 step 6 defines the
  reserve as "never exposed at gate 1" and as the re-gate's first source of fresh items, so a move reduces the
  supply toward E12's 35. State that cost in the procedure.
- **Nit:** fixed (§5.1 item 4 now gives the status split's purpose as comparability across phases).

## Appendix

### C. Pass 2: the transparency ceiling, rebuilt (scratchpad `rv/`, inline script)

**Instrument.**
- Each `definition` miss is classified by its `detail.diag.sinkControlledBy` cond, read against the source at
  `0caf85e3` (Appendix B).
- The `try` conds (code is the try body's first statement) and the generator iterator connect; the `for`
  iterator does not.
- Residuals are those 1 + the 9 `frontend` + the 4 `undetermined` misses. I checked each of them: no sink is
  under a synthetic cond.
- The cluster interval uses P0-ML §7.3's formula over 29 methods.

| Definition | Connected | Recall | Cluster 95% | Entry 465 needs |
|---|---|---|---|---|
| note's rule (try ×7 + generator ×1) | 467/481 | 0.9709 | [0.945, 0.996] (t 2.048 or 2.045) | ≤ 2 open |
| generator iterator not recognised (try only) | 466/481 | 0.9688 | [0.942, 0.996] | ≤ 1 open |
| omission pair excluded (denominator 480) | 466/480 | 0.9708 | — | bar 456; entry ≈ 464 |
| recognizer fails (no transparency) | 459/481 | 0.954 | [0.920, 0.988] | unreachable |

Residual misses at 467, per method: 13: 4, 26: 4, 19: 3, 1: 1, 18: 1, 30: 1 (= 14), matching the note.

### A. Independent re-derivation and spread (scratchpad `rv/f1.py`, `rv/oc2.py`, `rv/oc4.py`)

**Instruments.**
- **Counts.** I joined the misses to the truth by `(ordinal, pairIndex)`: 0 duplicate keys, 0 kind mismatches,
  0 duplicate truth pairs.
- **Omissions.** The only omissions are python ordinal 22 (a control→call pair) and JS ordinal 1, from
  `*-flow-spurious.adj.jsonl` `verdict`.
- **Operating curve, two ways:**
  - (i) seeded Monte Carlo, beta-binomial per method, with sizes resampled from the 29 non-empty methods *or* all
    30, the zero-pair method included (4,000 reps per cell);
  - (ii) my own exact convolution over P0's *fixed* 30 sizes.
  Both agree with the note's `oc.py` within Monte Carlo error. The note's `flow.py` and `oc.py` were also run, and
  they reproduce its tables.

| Figure | Definition | Value |
|---|---|---|
| python ceiling (direct) | denominator 481, omission counted connected / denominator 480, omission excluded | 459/481 = 0.954 / 458/480 = 0.954 |
| python ceiling (chain) | same two | 468/481 = 0.973 / 467/480 = 0.973 |
| pass bar | ceil(0.95 n) | 457 (n 481) / 456 (n 480) |
| JS ceiling | direct / chain / chain + i18n / direct + i18n | 83/152 0.546 / 87 0.572 / 96 0.632 / 92 0.605 |
| entry r*: smallest true recall with P(pass) ≥ 0.80, 30 methods | fixed sizes, independent / ρ 0.1 / ρ 0.2 / ρ 0.3 | ≈ 0.957 / 0.962 / 0.964 / 0.966 → 461 / 463 / 464 / 465 |
| same | resampled sizes, ρ 0.2: 29-pool (note's) / 30-pool | 0.965 → 0.805 (note 0.804) / 0.794, so r* ≈ 0.9655 → 465 |
| same | resampled sizes, ρ 0.3 | r* ≈ 0.9665 → 465 |
| P(pass) at 0.97, 30 vs 12 methods (ρ 0.1/0.2) | MC | 0.909/0.875 vs 0.836/0.806 (note 0.87–0.91 vs 0.81–0.84) |
| P(pass) at 0.93, 30 vs 12 | MC | 0.168/0.253 vs 0.281/0.369 (note 0.17–0.25 vs 0.30–0.37) |
| fresh 30-method pairs | 29-pool / 30-pool | median 496 / 479; p10 420 / 401 |
| caller-only listed (F4) | pooled / by language / mean of 4 cluster shares | 10/28 = 0.36 / python 6/8, TS/JS 4/20 / 0.30 |

### B. The 9 `definition` misses (from `python-flow-misses.adj.jsonl`, source at `0caf85e3`)

| Method | Pairs | Golden source cond → sink | Projection's sink edge goes to | Intermediate cond is |
|---|---|---|---|---|
| `retry.py::_gate_open` | 5 | `keep_going is None` (131) → points after `try` | `cond:134 open_ = keep_going()` | the `try` synthetic cond |
| `services.py::_normalize_opaque` | 2 | `isinstance(value, str)` (447) → returns 449/451 | `cond:449 json.loads(value)` | the `try` synthetic cond |
| `retry.py::is_transient` | 1 | `bool(exc.causes)` (80) → `is_transient` call in the generator | `cond:81 iteratorNonEmptyOrException` | the generator-loop synthetic cond |
| `test_repository.py::_load_method` | 1 | `if … node.name == name` (4704) → `AssertionError` (4706) | `cond:4703 iteratorNonEmptyOrException` | the `for` iterator (P1-09 re-maps it to the `for` cond), reached loop-carried |
