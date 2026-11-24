# Connected code graph P1: whole-phase implementation review (P1-38)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-38)

## Scope & verdict

**Scope.** The committed tree at `231d5b35`, read from a clean clone of that sha (the working tree's other-session
files were neither read nor touched). Not redone: each unit's own gate (`docs/reviews/connected-code-graph-p1-*.md`) and
the security-gated sandbox/egress surface. Asked for: (1) does the AC-closure table
(`docs/test-reports/connected-code-graph-p1-closure-report.md`) say what its evidence supports; (2) the close docs
(`docs/HISTORY.md` P1 close and wave entries, `docs/BACKLOG.md` C-1001, `cpg/README.md`, root `AGENTS.md` `cpg/` bullet)
against the shipped code and reports; (3) leftovers. Read against root `REVIEW.md` (no `cpg/REVIEW.md` exists). No
held-out data and no custodian directory was read; I read the **dev** gold (`cpg/eval/dev/ac5-*.jsonl`), which the
closure already calls development data. No service was touched; I started no container.

**Verdict: needs changes.** One blocker (B1, a cited evidence file now says the opposite of the table on AC-8); five
majors, all documentation or attribution defects plus two stakeholder decisions (Q1, Q2). The delivered engineering is
not what fails: every offline check I re-ran is green and the closure's live figures match their result files. What
fails is the honesty of the exit record in six places, each cheap to correct before the HISTORY entry is final.

CPG: considered, not relevant — the subject is the closure bookkeeping, close documents and test evidence of the `ccg` P1 phase; no `cpg_<component>` Joern graph answers it, and the shipped code was exercised by execution instead.

## Findings

### B1 (blocker, verified): the AC-8 evidence file says "closes at P1"; the table says "reported not closed"

`docs/test-reports/connected-code-graph-p1-ac8-report.md:63` (edited by `231d5b35`, the close commit) ends: "AC-8 part 1
closes at P1 as not run beyond the harness's own test". The same file's line 9 still reads "AC-8 part 1 stays not closed",
the closure table's AC-8 row says "reported not closed", and HISTORY says only "harness live-tested". The closure cites
this file as AC-8's evidence, so a reader following the citation reads the opposite closure status.
**Fix:** replace the clause with "AC-8 part 1 is reported not closed at P1; the real-component run is carried to P2".

### M1 (major, verified by execution): two reds called "none P1's" are P1's own, one is a real packaging defect

HISTORY:60 ("the 9 are the known reds, none P1's") and BACKLOG:92 ("Known reds, not P1's") are wrong for two of the nine:
- **`test_p1_11b_the_repos_own_mcp_json_names_its_script`** (`cpg/tests/test_config_extractor.py:1180`) fails because P1-27
  (`e200a8a2`, 2026-11-19) added the `ccg` entry to `.mcp.json` with launcher `bash -c 'cd … && exec .venv/bin/python -m
  clients.mcp.server'`; the test, written by P1-11b, asserts no server in the repo's own file has a `-c` arg. I ran it at
  `231d5b35`: 1 failed, 270 passed in `-k p1_11b`.
- **`test_every_data_file_under_ccg_is_covered_by_a_package_data_glob`** (`cpg/tests/test_rule_pack_schema.py:1428`)
  fails on `front/prompts/interpret.txt`, `front/prompts/summary.txt`, `front/grammar/interpret.json`, added by P1-26
  (`8377c326`, 2026-11-19); `cpg/pyproject.toml:74` has no glob for them. A non-editable install of `ccg` would ship
  without the interpreter prompt and grammar. Not user-visible today (the venv is editable), but it is a defect the phase
  shipped and carried red for five days under a "known" label.
`test_units` golden (monorepo grew: it lists `coordination-graph` where the golden expects `cypher-mcp`) reproduced and is
not P1's; the leaf-harness survivor-tell reproduced (`survives(...)` is False on the current rule-pack spec text; cause
not traced, so its owner is unverified); `test_scip_parse` x4 did **not** reproduce in isolation (151 passed with
`test_units`), so it is environment- or order-dependent. **Fix:** HISTORY and BACKLOG name the two as P1's with owners
(the `.mcp.json` red: re-baseline or exempt the `ccg` server in the P1-11b test; the glob: one `pyproject.toml` line, then
delete the red), and the "not P1's" label shrinks to what was traced.

### M2 (major, verified by arithmetic): the AC-24 row says 8 of 12 "is not distinguishable from a 0.92 model"

Closure AC-24 row: "8 of 12 is not distinguishable from a 0.92 model at this n". The interval printed beside it is
Wilson [0.391, 0.862] (I recomputed it: same), whose upper bound is below 0.92; the exact binomial probability of 8 or
fewer in 12 from a true 0.92 model is 0.012. At 9 of 12 (the first dev check) it is 0.065 and Wilson [0.468, 0.911]. The
sentence originates from the data-scientist's pre-run note (`docs/plans/connected-code-graph-p1-interpreter-model2-ml.md`
l.98), which says a model *near 0.80* failing the entry check at **9 to 10** of 12 cannot be told from a 0.92 model with
bad luck; the ledger (2026-11-22) stretched it to 8 of 12 and the table repeated it. The verdict "not met at P1" stands;
what is false is the softening that follows ("not a measured ceiling"). The honest reading is that 8/12 (and 5/12 for
ministral) is evidence against the 0.92 target at this n, from 14 dev items that were used for development.
**Fix:** state that 9/12 is not distinguishable from 0.92 at 5% (P = 0.065) and 8/12 is (P = 0.012), keep "dev items,
prompt revised between runs, not a held-out measurement".

### M3 (major, verified): the closure rows "closed for a stated sub-scope" under-describe their evidence

- **AC-2 "MCP half 20 of 20".** `cpg/eval/dev/ac5-mcp.jsonl` has 20 items from 3 anchors (`get_document`,
  `delete_document`, `send_message`); **all 20 impacted files are Markdown** (`.md` counter 20/20), and
  `cpg/eval/results/p1t-20/p1t20.json` marks all 20 `possible`, `boundOnLink` false. The AC names `kiro/` and
  `mcp-monitor/` configuration callers; those are the wiring half (0/7). So the "MCP half" is prompt mentions of three
  tools. The row says "`possible`, distance 1" but not "all Markdown prompt mentions".
- **AC-1 "17 of 18".** Verified in the dev gold that the AC's literal scenario (a falkor-chat endpoint's `salesperson/`
  callers) is among the hits: the 6 `salesperson/` items are all hits; the one miss is the `claude/teco` `curl` line.
  This is a point **for** the row that the row omits.
- **AC-9b.** The live rows AB-1/AB-2 fail a **unit** (`fx-svc` `legacy`) inside one repository; the AC says "a repository".
  Of the four rows, only LNK-42 was mutated (killed, `-p1-report.md` "P1-41 dispatch C"); AB-1, AB-2, LNK-43 were not.
  The row's own residual (an unbound crossing into a never-built unit is not flagged) is disclosed.
- **AC-17.** Structured location anchors require `repo` (`-p1-report.md` O-2); only the plain-language path omits it,
  and that was run with injected fixture repositories. The row says the second and omits the first.
- **AC-25 data-flow.** The row lists DF-1, DF-2, DF-4 (`test_fixture_gate_p1_ac.py:292`); DF-3, the only effect that
  crosses a `Link` into another repository (so the only one where "grouped by repository and component" discriminates), is
  asserted by the row gate, not by the AC-25 test, and the row does not name it.
**Fix:** one clause each in the rows; none changes a status.

### M4 (major): the P1 carry-overs are filed against "P2", which the proposal schedules last; the BACKLOG states a phase order that contradicts its own heading

BACKLOG:32 reads "What remains: P2 onward, in the stakeholder's order P0 → P1 → P3 → P4 → P5 → P2". The plan the
close executed (`docs/plans/connected-code-graph-p1.md` §11) asked for C-1001 "rewritten in place to **P3 onward**". The
proposal (`docs/plans/connected-code-graph.md:29`, §4.1) puts P2 (Java/Kotlin/C#/Go/C++ extractors, gRPC/Kafka packs)
after P5, and its nine ACs that "close only at P2 exit" exclude AC-1, AC-2, AC-4, AC-5-scripts/wiring, AC-24 and
AC-25's control half: §4.1 says every other AC "closes at its row in §4", and P5 says "AC-5 ✓ (proving-repo strata re-run
on fresh held-out items)". Yet the carry-over list sends exactly those to P2: scripts/wiring linking ("trigger: the P2
linking plan"), the combined held-out AC-5 gate ("trigger: scripts and wiring linking exists"), DEF-1, AC-24's lower
target ("trigger: a P2 measurement"). Consequences read from the proposal itself: P3 lists a review operator over
"config/wiring" and P5 re-runs the AC-5 strata "on fresh items" and closes AC-24's full gate, both **before** the first
AC-5 gate and before scripts/wiring link anything; the proposal's P2 row builds no shell exec/URL/script-hop facts, so
"the P2 linking plan" is a plan for a phase that does not contain that work. See Q1.
**Fix once Q1 is answered:** BACKLOG:32 and the carry-over triggers, the proposal's P1 row, §4.1 and the P3/P5 exits.

### M5 (major, verified): decisions that change what an AC means are in the ledger and HISTORY, not in the requirements decision log

`docs/requirements/connected-code-graph.md` carries one P1-period entry (2026-11-19, flow fallback) after 2026-11-01.
Absent: the three entries the ledger lists as owed to `tico` at 2026-11-10 (only flow-dependent hops `possible`;
the S36 redo sample; noise thresholds ratified, `ccg-noise-ratify-v1`, which `-p1.md:426` calls "`tico`'s requirements-log
entry"), and the decisions of 2026-11-19..24: Q1 = C, Q-DYN B, Q-AC6 B', AC-6 example config at two instances, AC-18
"each step" excludes contract-point hops, AC-10 closes on review, AC-1 sub-scope, the narrowing and deferral of the
AC-5 gate, AC-24 not met. A reader of the requirements document sees AC text that the closure no longer applies as written.
**Fix:** `tico` appends one question → answer entry per decision, with the FR/AC it affects (append-only log).

### m1 (minor): disclosures the exit record does not carry

- **Oracle capture limits** (`docs/reviews/connected-code-graph-p1-47-impl.md` lines 114-120): no `mcp_tool` oracle edge
  exists at P1; SPA edges are at file grain; `salesperson/tests/e2e` was not run; `/order/advance` is one key for three
  transitions; the second `/session` join was not driven (S10 defect). HISTORY W4 gives "13 of 13 steps, 18 edges over 17
  keys" and no limit; the noise bar and the AC-5 per-kind breakdown inherit them.
- **The AC-8 host probes were run on the coordinator's reading of one ambiguous user message.** Ledger 2026-11-23: the
  user (after "no, later") wrote "sorry, actually i want to run", read as authorising `test_ac8.sh`'s host-address and
  `.invalid` probes, with "if the user meant something else, stop it". The ac8 report calls it "the stakeholder's
  authorisation"; the closure says "the user ran them after all"; HISTORY says "authorised". See Q2.
- **P1-34 flow-decision disclosures** in `cpg/eval/decisions/ccg-p1-flow-python-v2-decision.json` that HISTORY omits:
  labellers worked from sample source text alone (transcript check recorded as a deviation), the entry value had zero
  margin (465/481), and the two adjudicators (120 of 120 agreed) are one model family. The closure has only the last.
- **S36 redo:** items 40, 10, 17 count as correct only in one configuration the rubric counts (ledger 2026-11-22); not in HISTORY.
- The `raw_query` text says an OOM kill "drops the no-persistence snapshot, which `refresh` restores"
  (`cpg/README.md` "Running a build and an answer"; HISTORY, BACKLOG). The graph-dba note says the kill drops "graphs,
  ACL users, runtime config"; recovery is `ccg-falkordb.sh up`, `create_users` (which rotates secrets), then `refresh`
  (`cpg/infra/ccg-falkordb.sh:48`, `ccg/store/acl.py:14`). No affirmative "safe" wording (unfiltered `grep -w safe` over the closure, HISTORY, BACKLOG,
  README and `AGENTS.md`: the only memory-related hits are "not a safety claim", HISTORY:58 and the closure's AC-21 row); the root `AGENTS.md` "no memory cap" matches the note's "no memory cap" (a per-query cap).

### m2 (minor): live-tier evidence is cited without its commit gap

The AC-9/9b/16/17/18/21/25 live results are from one run at `a81a597e` + the then-uncommitted AC driver (2026-11-21;
`-p1-report.md` "P1-42"), with per-row ledgers written to a scratchpad path that was never committed (`p1t20.json` is the
only committed live result file). After it, `cpg/ccg/engine/anchors.py` changed (`cc53b3de` verbatim-key retry in
`resolve_contract`, 2026-11-23; `4cda0ed5`) and the interpreter prompt changed (`2bf875bd`). The P1T-20 run at `57be01a2`
(after `cc53b3de`) exercises contract anchors for AC-1/2, so the risk is small, but the closure's "cited, live" does not
say the AC-level driver was not repeated on the final engine. I could not re-run the live tier (it needs the store on a
forbidden port or a Joern build). **Fix:** one sentence naming the run's commit; consider committing the run ledger.

### m3 (minor): the `refresh` defect P5-1 is filed as gate-harness work

`docs/reviews/connected-code-graph-p1-32b-impl.md` P5-1: a failure inside `refresh` after the swap rename strands the
store with no remedy. That is product code (`ccg/store/swap.py`), reachable from the shipped MCP `refresh` tool and
`teco`'s CLI recipe. BACKLOG files it under "Gate harness before the list is spent" (`tdd-engineer`; trigger: the gate is
scheduled), a trigger that M4 shows may be phases away. **Fix:** its own item, trigger "before `refresh` is used
unattended".

### m4 (minor): unowned leftovers named only in HISTORY

HISTORY's close entry says the rule-pack spec rows, `skills/joern-cpg`, `skills/cpg-analysis`, the archive flips and the
whole-component spec are "left to their owners"; BACKLOG has no entry for the first three, and AC-10's "behaviour
unobserved" has no unit that ever observes it (the test plan's P1T-39 wanted a real brief on a covered system; none was
inspected). Also unlisted: AC-17's queue-name group (proposal §4.1 owns it at P2 exit), AC-5 gold drift (3 of 8 strict
spans; not repinned; closure §5 item 11), and the P1 close README's silence on D9a/D10 code now in `cpg/`: `cpg/README.md`
Layout lacks `ccg/gateway/` and `ccg/broker/holderclient.py`, and Setup lacks the `../llm-gateway` install the broker
tests need (`cpg/pyproject.toml` comment). Those belong to the llm-gateway feature's close; say so, or add the rows.

### m5 (minor): stale statements in plans and in `cpg/README.md` (unfiltered sweeps, every hit owned)

| Stale text | Where | Owner / fix |
|---|---|---|
| P1 row: "AC-1 ✓, AC-2 ✓, AC-4 ✓, AC-5 on the HTTP/MCP/scripts/flow strata, … AC-9b ✓, AC-16 ✓, AC-21 ✓, AC-25 ✓", and "git-remote fetch only" | `docs/plans/connected-code-graph.md:1035` | `architect`: AC-1 sub-scope, AC-2 MCP half, AC-4 not closed, AC-5 deferred, AC-9b/16/25 sub-scope; fetch deferred (Q1 = C) |
| "AC-1, AC-2, AC-4 closed; … AC-9b closed; AC-16 closed … AC-21 closed" | `docs/plans/connected-code-graph-p1-gates.md:301` | `architect` |
| "`tico` takes the 40-item redo … to the stakeholder" | `-p1.md:423` | `architect`: the redo was an AI reviewer, stakeholder decision 2026-11-22 |
| "87 P1 rows" (shipped set 81) | `-p1.md:48,396,436`; `-p1-gates.md:125,131,144`; `-p1-linking.md:583,598,602,705,744` | `architect`; `test-plans/…-p1.md:9` is dated history |
| "P1-40 runs it … until P1-40 supplies real ones" | `cpg/README.md:75` (sandbox/ac8 row) | close-doc writer: P1-40 is carried to P2 |
| `STRATA_GATED = ("http","mcp","scripts","wiring")` | `cpg/eval/tools/gate.py:56` | listed in BACKLOG; the stale `ac5-gate-1` record is absent (0 of 10 prereg lines) |
| "60 hits in 39 files" | HISTORY, spike report | the report's own command now gives 63/41; the close's own `AGENTS.md` and `cpg/README.md` add 3 (nit) |
| HISTORY: the whole-phase review "not yet written when this entry was drafted" | `docs/HISTORY.md` P1 close | rewrite on the verdict |
Already fixed (checked): `-p1.md:264` now reads "closes at P1 for its data-flow half only"; l.270 and l.487 agree.

### n1 (nit): the AC-18 offline checker accepts line 0

Mutant `and h["line"] >= 1` → `and True` in `_has_loc` (`cpg/tests/fixture_gate_p1_ac_helpers.py`) leaves the 8 offline
tests green (8 passed). Killed by that tier: the `isinstance(line, int)` check, the past-the-end check, the lifted-entry
check, the possible-needs-reason, ordering, bound, chain-start, directness checks (M2-M9, each 1 failed). Add a line-0 case
to `test_ac18_checker_mutants`.

## Growth-stops check (root `REVIEW.md`, documents the phase touched)

Documents not yet `archived` that the phase revised after the rule's date (2026-10-30), words at first commit → `231d5b35`:
`-p1.md` 8,601 → 10,626; `-p1-gates.md` 3,109 → 5,210; `-p1-linking.md` 2,402 → 19,176; `-p1-extraction.md` 3,368 → 10,794;
`-p1-answers.md` 2,380 → 6,057; `-p1-ml.md` 7,857 → 8,948; `-p1-graph.md` 5,476 → 6,697.
- **New scope:** the bulk of the growth is added unit sections (P1-46, P1-49..57 in `-p1-linking.md`; 5d/P1-50 in
  `-p1-extraction.md`), each a new step.
- **Rework / restated history (findings):** `-p1-linking.md:744` P1-57 opens with "Cause: … (76 of 87 rows green, 11
  red)", a result that lives in `-p1-report.md` (one home for a fact); the **87** that survives at the sites in m5 is the
  opposite failure, rework not propagated, so the plan states a design the build no longer has.
- **Review documents:** `-p1-32b-impl.md` is 11,672 words over five passes; Pass 5's "Dispositions of Pass 4" runs 231
  words for seven findings (rule: one line each plus the evidence rechecked), minor. The other reviews read are single-pass.
- **Requirements:** the append-only log is missing entries (M5), the reverse of growth.
No numeric threshold is applied. I did not find a finding→resolution table or a disposition list in the plans sampled.

## What's solid

- **Re-run at `231d5b35`, from `cpg/` with the repo venv, `-p no:cacheprovider`:** R1 `test_fixture_gate_p1_ac.py` +
  `test_fixture_gate_p1.py` 11 passed, 120 deselected, exit 0; R2 `test_pin_replay.py` 12 passed, 4 deselected, exit 0;
  R3 the four `test_ac8_*` files 360 passed, exit 0; R4 `test_ac24_smoke.py` 177 passed, exit 0; R5 `test_render.py` +
  `test_clients.py` + `test_front_clients.py` 290 passed, exit 0; R6 `test_acl_unit.py` + `test_atrest.py` 273 passed,
  exit 0; R7 `test_fixtures.py` 80 passed, 3 deselected, exit 0; R8 the four `test_gate*.py` 309 passed, exit 0. (A first
  run on a `git archive` copy failed 9 tests that call `git`; a real clone passed them: a method artefact, not a finding.)
  The closure's R1-R8 are reproduced exactly.
- **Figures reproduced from committed result files:** `p1t20.json` (AC-1 17/18, MCP 20/20, scripts 0/49 over 22 of 22
  anchors, wiring 0/7 with 4 unit-anchor items not asked, AC-4 filter 0/1, sentinel `held-out-opened` false), `build.json`
  (1335 fresh units, 934 Links = http 376 + mcp_tool 558, confirmed 104 / possible 830, 1772.3 s), `p1t19.json` (599 items,
  0 violations, 6 render mutants caught), the flow decision (296/334 = 0.8862, Wilson [0.8477, 0.9160], M = -389), the
  held-out counts (26/46/79/8/99; http+mcp 72), the prereg file (no `ac5-gate-1`, no `ccg-prereg-ac24-v1`), the AC-24
  attempts file. Each AC row's tally (4 + 8 + 4 = 16) is internally consistent.
- Offline mutants on the AC checkers: eight of nine killed (above); the AC-6/18/21/25 checker tests are real checks.
- The closure itself is the right shape: it names the AC-16 raw leg as sufficiency only, says which stakeholder
  decision produced each status, and discloses that AC-10 is unobserved and AC-5 unexposed. HISTORY's disclosures
  (custodian role, S36 AI-only at about 4 minutes with three `doubt:` items, AC-24 prompt revision, the denied `ls`)
  match the ledger and `s36-redo-report.md` (39 correct, 1 incorrect, items 1/9/37).
- `cpg/README.md` run recipe checked against code: CLI exit codes 0/1/2/3 and `--fail-on` (`clients/cli/main.py:8-10,27`),
  `CCG_SYSTEM`/`CCG_FALKORDB_HOST`/`_PORT` (`clients/common.py:28,139,161`), broker exits 64/69
  (`ccg/broker/server.py:70`), `create_users` (`ccg/store/acl.py:235`).

## Questions for the stakeholder (options, not decisions)

**Q1. What does "carried to P2" mean for the work P3-P5 depend on?** The proposal orders P0 → P1 → P3 → P4 → P5 → P2 and
gives P2 no scripts/wiring work. (A) Create a P1 close-out unit (scripts/wiring linking: shell exec, URL, script-hop
facts; DEF-1; the AC-5 gate and AC-24 lower target) scheduled before P3. (B) Keep it last, and re-baseline P3's
config/wiring review operator, P5's "AC-5 ✓" and AC-24 full gate, so none claims scripts/wiring/AC-24. (C) Another
order. Until answered, M4's fixes cannot be written. The AC-5 gate and "fresh held-out items at P5" also need a statement
of which list the first gate spends (http 26 + mcp 46 is 72, scripts 79, wiring 8).

**Q2. Was "sorry, actually i want to run" an authorisation of the AC-8 host-address probes?** (A) Yes: record it in
HISTORY and the ac8 report as the user's words plus the coordinator's reading. (B) Unsure: ask the user to confirm, and
disclose "run on an unconfirmed reading" until then.

**Q3. Who observes AC-10, and when?** (A) The first teco-coordinated architect or implementer brief on a covered system
is inspected against the three behaviours (a BACKLOG item with that trigger). (B) Accept review-only closure permanently.

**Q4 (routing, not a decision).** The two P1-caused reds in M1: close-out commit now, or BACKLOG items with owners?

## Proposed REVIEW.md lines

Target: root `REVIEW.md`, "Always check". Sources given per line; none duplicates a held line or a test/hook.
1. A phase-exit table row "closed for a stated sub-scope" names the composition of the evidence behind its figure (item
   kinds, status mix, anchor count, what was injected), and any statistic it states ("not distinguishable from X") is
   recomputed from the figures printed beside it. (M2, M3.)
2. A "known reds, not <this phase>'s" list is attributed per test by `git log -S` on the failing assertion's subject and by
   what the phase edited; a red caused by the phase's own change to a file another test reads as a golden is the phase's.
   (M1.)
3. A close-time edit to a cited evidence report is checked against that report's own verdict line and the table citing it;
   a sentence that flips closed / not closed is a finding. (B1.)
4. Live-tier evidence cited at an exit names the commit it ran at, and every commit since that touches the evidence path
   is listed as re-run or not re-run. (m2.)
5. A carry-over's trigger names a unit or phase that is scheduled, in the order the proposal now states, before the work
   that depends on it; a trigger naming a phase scheduled after its dependents is a finding. (M4.)
6. A stakeholder decision that changes how an AC is read or closed is present in the requirements decision log before the
   phase's exit entry cites it. (M5.)

## Appendix A: rows re-derived end to end

| Row | Opened | Ran / computed | Outcome |
|---|---|---|---|
| AC-9b | `test_fixture_gate_p1.py:504-541`, `-p1-report.md` dispatch C (l.164, 207-219, 240) | R1 exit 0 (offline tier proves only the manifest-mutation check; AB/LNK rows are live) | holds on cited live run; only LNK-42 mutated; unit not repository (M3) |
| AC-16 | `test_pin_replay.py`, `test_fixture_gate_p1_ac.py:383-520`, O-1 | R2 exit 0, R5 exit 0 | raw leg is sufficiency only, as the table says; plain-language leg is the parser path |
| AC-17 | `test_fixture_gate_p1_ac.py:173-258`, O-2 | R1 exit 0 (live legs cited) | two of four groups; structured anchors need `repo` (M3) |
| AC-25 data-flow | `:276-320`, `fixture_gate_p1_ac_helpers.py` `ac25_shape_violations` | mutants M2, M8, M9 killed | holds; DF-3 unnamed (M3) |
| AC-10 | `claude/teco/teco.md:120`, `-p1-27-impl.md`, test plan §13 Q-4 | `.mcp.json` ccg entry read; R6 exit 0 | review-only, as flagged; the entry also causes M1's red |
| AC-18 | `fixture_gate_p1_ac_helpers.py:40-120`, `p1t19.json` | mutants M1 (survives, n1), M3-M7 killed | holds on 51 fixture items and 599 real items |
| AC-21 | `test_fixture_gate_p1_ac.py:602-680,732`; `-p1-report.md` P1-42 | R6 exit 0 | holds; live 25/25 cited |
| AC-1 / AC-2 / AC-4 | `p1t20.json`, `eval/dev/ac5-{http,mcp,wiring}.jsonl` | counted | figures match; composition in M3 |
| AC-24 | `eval/results/ac24-dev2/p36_devAttempts.json`, `preregistration.jsonl` | Wilson and exact binomial | attempts match; M2 |
| AC-5 | counts report, `gate.py:50-56`, `preregistration.jsonl` | grep counts | figures and "no record" hold |

## Pass 2

Scope: the committed tree at `e7fdde48`, read from a clean clone; fix commits `d5782ff2` (B1), `8e07321a` (closure report),
`08991f6b` (plan text), `728fd8f8` (close docs), `a7be79e5` (requirements log), `e7fdde48` (the two reds). The other session's
broker/gateway files are not P1; two of its committed commits do show up in the suite (N2).

**Verdict: approve with suggestions.** No blocker remains; the close may proceed. N1 and the tidy items below are owed in the
close-out unit or one commit; none changes an AC status.

### Dispositions

- **B1 fixed.** `ac8-report.md:63` now reads "reported not closed at P1 (harness live-tested only)", agreeing with l.9 and the table.
- **M1 fixed, executed.** `e7fdde48`: `test_p1_11b_the_repos_own_mcp_json_names_its_script` now pins the `ccg` entry as
  `("bash", ["-c", REDACTED])` plus a plant (the `cypher` entry turned into a `-c` launcher must go red); `-k p1_11b` 53 passed.
  `pyproject.toml:74` gains `front/prompts/*.txt`, `front/grammar/*.json`; the glob test passed in the full run. Extractor untouched
  (`git diff 231d5b35 e7fdde48 -- cpg/extractors` empty). "Not P1's" now covers only what was traced.
- **M2 fixed.** Closure AC-24 row (l.122) states P(<=8 of 12 | 0.92) = 0.012, 9/12 = 0.065, "evidence against the 0.92 target", verdict unchanged.
- **M3 fixed.** Rows now carry: AC-2 MCP half = 20 Markdown files, 3 anchors, all `possible`; AC-1 six `salesperson/` hits; AC-9b unit-not-repository
  and only LNK-42 mutated; AC-17 structured anchors need `repo`; AC-25 names DF-3 (l.43, 48, 92, 109, 127).
- **M4 fixed in the close docs, residue in two places.** Q1 (close-out unit before P3) is in HISTORY, BACKLOG:32-40, root `AGENTS.md` and `cpg/README.md:75`;
  BACKLOG:32 no longer says "P2 onward". Residue: closure headings l.40, 55, 79 and §6 l.152-153 still say "carried/deferred to P2" (l.155 supersedes them in a
  sentence), and the proposal's P1 row (`docs/plans/connected-code-graph.md:1035`) still says AC-4/AC-5/AC-8 "P2". Both are the close-out plan's to repoint.
- **M5 fixed.** `a7be79e5`: 17 entries, one question to one answer each, AC wording untouched; the three owed 2026-11-09/10 entries and the 2026-11-19..24 decisions are there.
  The AC-9b entry says "reading recorded, not a separate user answer", which is honest. Unrecorded owed entries (per-system ACL names, indexer provisioning
  step and directory, Q-ENV, strace install, P1-34 fresh draw): **none blocks the close**. They change no AC reading or closure; they are recorded in plans, the
  README and the decision JSON. Q-ENV (env-var binding deferred) and the P1-34 spend are worth one log line each, non-blocking.
- **m1 fixed** (HISTORY:55-62 oracle limits, probe authorisation with the user's confirmation, P1-34 disclosures; README:154 OOM recovery via `up`, `create_users`, `refresh`).
- **m2 fixed** (closure l.11 names `a81a597e`, the later commits and that the driver was not repeated). **m3 fixed** (BACKLOG:71, trigger "before `refresh` is used unattended").
- **m4 mostly fixed.** BACKLOG:79 (AC-10 observed, qa, live model via broker; matches Q3), :84, :86 added. Not added: README rows for `ccg/gateway/` and `holderclient.py`
  and the `../llm-gateway` install; acceptable as that feature's close, but say so in HISTORY.
- **m5 fixed.** Unfiltered `git grep` for `87 (P1|of 134|rows)` at `e7fdde48`: only ledger lines and the dated `test-plans/...-p1.md:9` note remain (history, owned);
  proposal row and `-p1-gates.md` P1-38 text updated; `-p1.md:423` now says AI reviewer. **`-p1.md:371` "census 85" is acceptable**: it is the 57c step's intermediate
  state and the same row sends D10 on to Q-DYN; optionally append "(81 after Q-DYN B)".
- **n1 not fixed** (nit; no change to the checker or its mutant test).
- **Growth-stops: closed.** The staleness half (87) is repaired; no new restated-history found in the fix commits.
- **Q1 to Q4 answered:** Q1 close-out unit before P3 (plan uncommitted, out of scope); Q2 yes, recorded in the requirements log and HISTORY; Q3 in BACKLOG:79; Q4 done (`e7fdde48`).

### Fix-vs-fix interactions

- **AC-6 acceptance date.** Ledger and requirements log: 2026-11-23; HISTORY: 2026-11-23; closure l.77 still says "2026-11-24 ledger". The ledger is right; fix the closure line.
- **Spike hit count: both numbers are right, at different commits.** At `62d3ca48` (the report's commit) the report's own command gives 60 hits in 39 files; at `231d5b35` and `e7fdde48`
  it gives 63 in 41, the extra 3 being the close's own `AGENTS.md` (1) and `cpg/README.md` (2) mentions of the frozen directory. The report needs no correction; HISTORY's "60" should say "at the report's commit".
- AC rows agree across closure, HISTORY, BACKLOG and the requirements log on statuses (4 + 8 + 4), the AC-5 deferral, AC-10, AC-18, AC-1 and AC-24; the only divergence is the "P2" wording above.

### Full default suite on `e7fdde48` (clean clone, `cpg/`, host load about 1, live tiers deselected)

`13 failed, 17584 passed, 110 skipped, 531 deselected, exit 1` (827 s). Not the expected 7:
- **Fixed pair green:** `test_p1_11b_*` and the package-data glob pass. `test_units` golden, the leaf-harness survivor-tell and the `test_ac8_lib` flake did not fail here (they did at `231d5b35`, so they depend on tree state; not re-traced).
- **Known, still red (4):** `test_scip_parse` x4.
- **N1 (new, P1's, fresh-checkout red x4):** `test_config_extractor.py` `[.env]` cases fail on `KeyError: '.env'`: `cpg/tests/data/config/repo1/.env` is excluded by root `.gitignore:138` (`.env`, `git check-ignore` confirms) and untracked, so P1-11's golden corpus is incomplete in any clean checkout and the 531-test suite is green only where the file happens to exist. Fix: `!cpg/tests/data/config/repo1/.env` in `.gitignore` and add the file.
- **N2 (not P1's, a P1 guard reddened by another feature, x3):** `test_no_network.py` fails, also at `231d5b35` and in the live working tree, because `ccg/gateway/brokertransport.py:27` (llm-gateway D10, `cbab7cdd`) imports `ccg.broker.client.BrokerClient`, which P1-25b's import-boundary pin allows only to the front's interpret/summary modules. Owner: the llm-gateway feature (extend the pin with a granted import, as the test's "granted broker imports" case allows). The coordinator's "no new red" at `231d5b35` does not match what I observe; re-check on the real tree.
- **N3 (minor, clone-location dependence x2):** `test_broker_audit.py::test_the_default_audit_dir_is_the_codes_own_location_not_the_working_directory` and `test_eval_tools.py::test_derived_fixtures_match_builders` embed the checkout's absolute path or the editable-install location; they pass only at the original path. 110 skips (against 10) are the gitignored slice oracle, reported unrun as designed.

### Tidy items owed (non-blocking)

Closure l.77 date; closure headings and proposal row l.1035 to say close-out unit once its plan lands; N1 fixture; N2 pin; one-line HISTORY wording for the spike count.
