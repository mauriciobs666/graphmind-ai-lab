# Connected code graph — P1 test plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1) · **Reviews:** `docs/reviews/connected-code-graph-p1-test-plan.md`

*2026-11-10 (rev. after the analyst gate, F1 to F8): counts 92 of 133 after the third P1 example-config row (AC6-EX-4, now `http`, a Python `os.environ[...]` consumer bound in `deploy/compose.example.yaml`); DF-3 P1 form and locator/distance in the `p1Form` tie; P1T-20 `key-error` class; P1-23 `function`-state note; statuses asserted in index §5's P1 form; AC-10, AC-1/2/4 and the proving-system build get items; the resolved source disagreements are removed.*

## 1. Scope and objective

**Objective.** State how P1's acceptance criteria are verified, at which level, by which step, and what each
level can and cannot prove, so that P1-38's AC-closure table (`docs/plans/connected-code-graph-p1.md` §8) is
assembled from evidence and not from step completion. P1-33 is a document step: this plan writes no code and no
test.

**Who runs what.** `qa-engineer` executes P1-41, P1-42, P1-35, P1-37 and P1-43 and writes the report,
`docs/test-reports/connected-code-graph-p1-report.md`, one section per gate step (`-p1-gates.md` header). `devops`
builds and runs the AC-8 harness (P1-31, P1-40) and records the result; `qa-engineer` writes the report's AC-8
section from that record and checks it against §8.

**In scope.** The ACs of the proposal's P1 row (`docs/plans/connected-code-graph.md` §4, the "Closes" cell of the
P1 row) and the 92 fixture rows the fixtures plan gates at P1 (§6 below). **Out of scope:** §12.

**Item ids.** Items are `P1T-NN`. The fixtures plan already owns `TP-001..019`
(`docs/test-plans/connected-code-graph-fixtures.md` §9); this plan cites those by their own ids and never
renumbers them. Row ids (`LNK-nn`, `AB-n`) are the fixtures plan's.

## 2. References and how counts were derived

- Requirements (the ACs): `docs/requirements/connected-code-graph.md`, "Acceptance criteria".
- Proposal: `docs/plans/connected-code-graph.md` §4 (P1 row), §4.1 (closure points), §5.3 (AC-8).
- P1 index: `docs/plans/connected-code-graph-p1.md` §2 (fixture rows gated), §5 (flow, AC-25), §6 (option C),
  §7 (step table, critical path, core freeze), §8 (strategy, AC map), §10 (risks, Q-E).
- Gates: `docs/plans/connected-code-graph-p1-gates.md` P1-31, P1-40, P1-32, P1-33, P1-41, P1-42, P1-34 to P1-38, P1-43,
  P1-47.
- Specs: egress `docs/plans/connected-code-graph-egress-spec3.md` §8 (rows 8.1 to 8.16) and
  `-egress-spec4.md`; answers `-answer-spec2.md` §3.2, §3.3, §3.5, §5; bundle `-bundle-spec5.md` §9; evaluation
  `-p1-ml.md` §3.5, §4.3, §5.3, §6; fixtures `docs/test-plans/connected-code-graph-fixtures.md` §3, §5, §6, §8, §9.

**Counts cited in this plan, each recomputed in this run** (commands run from the repo root unless stated):

| Count | Value | Source and method |
|---|---|---|
| Fixture rows, all gates | 133 | distinct row ids over `cpg/fixtures/system1/expected/*.yaml` (the five manifests, ids deduplicated), by a `yaml.safe_load` script run with `cpg/.venv/bin/python` |
| Rows gated P1 / P2 / P3 / P4 / P5 | 92 / 16 / 21 / 1 / 3 | same script, grouped by each row's `gate`; no id carries two gates |
| P1 rows by family | AB 2, AC6 12, B1F 4, CD 4, DF 4, FT 2, KN 5, LNK 27, MR 2, NC 6, RV 3, SK 18, TW 3 (= 92) | same script, by id prefix |
| Same counts from the fixtures plan | 92 P1 of 133, matching | `docs/test-plans/connected-code-graph-fixtures.md` §6 "By earliest gate" (P1 92 · P2 16 · P3 21 · P4 1 · P5 3 = 133); `cpg/tests/test_fixtures.py` TP-004 recomputes the census from the manifests and compares it to that line (83 passed in `tests/test_fixtures.py` plus `tests/test_fixture_keynorm_mutants.py`, this run) |
| ACs in the proposal's P1 "Closes" cell | 16 | `grep -o 'AC-[0-9]*[ab]\?'` over that cell of `docs/plans/connected-code-graph.md` (line 1035): AC-1, 2, 4, 5, 6, 8, 9, 9b, 10, 11, 16, 17, 18, 21, 24, 25 |
| Rows of egress-spec3 §8.2 | 16 (8.1 to 8.16) | script over the table: 8 plain `computed`, 4 headed `vacuous` (8.13 with a computed residue), 3 mixed (8.3, 8.5, 8.15), 1 conditional (8.16) |
| Dev AC-24 items | 14 (12 unambiguous, 2 ambiguity; 6 + 6 unambiguous by language) | `cpg/eval/dev/ac24.jsonl`, `outcome` and `lang` fields |
| Dev AC-5 items by file | flow 28, http 18, mcp 20, scripts 49, wiring 11 | `wc -l cpg/eval/dev/ac5-*.jsonl` |
| Tests in `cpg/`, default suite (a snapshot; P1T-01 re-records it) | 6354 passed, 1 skipped (`CCG_LEAF_MUTATION` slow gate), 15 deselected = 6370 collected | `cd cpg && .venv/bin/python -m pytest -q -p no:cacheprovider`, 208 s, exit 0, this run, in a working tree with other uncommitted P1 work |

The held-out counts (30 AC-24 items, 159 counting AC-5 items and so on) are cited by the gates text only where a
step needs them; this run did not open held-out data and does not recompute them.

## 3. Test levels and how they relate

Five levels. Each answers a different question, and none substitutes for another. The levels are ordered by what
must be true before the next is worth running.

| Level | Question it answers | Where it runs | Owner steps | Data |
|---|---|---|---|---|
| **L0 Unit and contract suites** | Does each module meet its own contract, including its mutants? | `cpg/tests/**` (`.venv/bin/python -m pytest` from `cpg/`), `cpg/sandbox/tests/` (shell) | every build step, with a mutation check on its contract boundary (index §8) | goldens, `cpg/tests/golden/`, slice bundles (marker `slice`) |
| **L1 Fixture gate** | Does the built system emit the planted links, statuses, reasons and answer shapes? | the materialized fixture system, built through P1-20, throwaway instance | P1-41 (the 92 rows), P1-42 (AC-level) | `cpg/fixtures/system1/` and its manifests; dev anchors `cpg/eval/dev/` for the proving-repo asks |
| **L2 Policy and trust harness (AC-8)** | Does a full refresh plus queries of a mixed system make zero connections outside the allow-list? | one sandboxed scenario, no route out, a detector | P1-31 builds, P1-40 runs | the mixed fixture system (`cpg/fixtures/system1`) |
| **L3 Evaluation** | Does the system find the impacts of the proving repo it has never seen, and understand the questions? | the gate harness over frozen blind items | P1-32 (tooling and harness), P1-47 and P1-36 (oracle, noise), P1-34 (flow), P1-35 (AC-24 smoke), P1-37 (AC-5 first gate), P1-43 (re-gate) | held-out items (custodian root, harness only) and dev items |
| **L4 Review-only** | Is a behaviour that cannot be executed described correctly? | `analyst` and `cobb` reading | P1-27 (AC-10), the `analyst` gates, P1-38's implementation review | the prompt and skill text, the diff |

**Relations.**
1. **L0 is the baseline for everything else.** Before any L1 to L3 item runs, the `cpg` suite is green with
   `0 skipped` for every `slice`-marked gate (index §8, D9), collected counts recorded. A red or un-runnable
   baseline blocks the items above it and is reported, never layered on.
2. **L1 sits between L0 and L3 by design.** L1 is the cheapest place to find an extraction, linking or engine
   defect on ground truth the builders may see; the index's R-5 mitigation is exactly this ordering (P1-41 and
   P1-42 run, and are green, before P1-37). P1-37's prerequisites include P1-42 green. A fixture pass proves the
   mechanics on planted shapes; it does not prove recall on real code.
3. **L3 is blind and single-shot.** Held-out items are touched only by P1-37 and P1-43 (and P1-35's held-out
   AC-24 cells) through the harness; builders and this plan's executors work on dev items and fixtures. Fixture
   rows are never pooled with held-out items (fixtures plan §11), and the runtime oracle is reported beside seeded
   recall, never pooled (index §4.1).
4. **L2 is independent of L1's verdicts but shares its build.** P1-40 runs on a mixed fixture system through
   P1-20 and P1-25; it needs P1-26 for the interpreter-side rows (8.7 to 8.9). It reports rows computed, vacuous or
   unrun exactly as `-egress-spec3.md` §8.2 states; a vacuous row is a reported fact, not a pass (§8).
5. **L4 closes what L0 to L3 cannot execute.** AC-10 is "reviewable, not executable" per P1-27; its row in the
   closure table states that it is review-closed.
6. **Core freeze.** Steps that edit a hashed `cpg/` file after the `ac5-gate-1` record make the L3 run "after
   adaptation" (index §7, "Core freeze"). L1 findings that need a code fix are therefore fixed **before** the
   record or the first-gate result is lost; this plan's entry criteria (§11) order L1 before the record.
7. **Each level's defect goes to a step, not to a pooled bucket.** The report lists defects by the row, item or
   AC id and the step that supplies the fix.

**Tree state is not recorded here.** Which modules exist when an item runs is reported by P1T-01 (collected,
passed, skipped, deselected) and by the entry criteria of §11; a snapshot in this plan would be stale after one step.

## 4. Risk assessment

Prioritised by impact on a wrong "closed" claim times the likelihood the plan set already shows.

| # | Risk | Why it is likely | Mitigation in this plan | Items |
|---|---|---|---|---|
| R1 | A fixture pass is read as recall evidence | the fixtures are builder-visible by design | the AC table (§5) names, per AC, which level closes it; AC-5 closes only at L3 | §5, P1T-31 |
| R2 | The flow flag stays false (index R-1, §5), so the flow-dependent fixture rows cannot show their flag-true statuses | the manifests plant DF-1, DF-2, CD-1 to CD-3 and some RV dependencies in the flag-true form | the architect's P1 form (index §5) is the asserted status; the flag-true form is asserted only in a flag-true re-run (P1T-05, P1T-17, P1T-18) | P1T-05, P1T-17, P1T-18 |
| R3 | A check that cannot fail (REVIEW.md "Always check") | every item below names its planted failing input or positive control | §10, column "red when" | all |
| R4 | A skipped or vacuous result read as green | `slice` gates skip when the artifact is absent; AC-8 rows 8.2, 8.4, 8.12 are vacuous under option C; 8.16 is unrun without `strace` | §3 relation 1; §8 reports each as vacuous or unrun | P1T-02, P1T-28, P1T-29 |
| R5 | Held-out contact by a builder or by this plan's executors | single-shot blind gates | the briefing rule (`-p1-gates.md` header); this plan names no held-out path or content | P1T-30 to P1T-36 |
| R6 | Build host cost: full refresh on a 6 GB host | index R-4 | P1-20 reports wall time and peak RSS; L1 builds units one at a time and records both | P1T-04 |
| R7 | Cross-repo Compose context to unit (`../../fx-job`) is an inferred rule | `-p0-compose-binding.md` says it is unverified | LNK-15 asserted first; red means escalate, not patch the fixture | P1T-09 |
| R8 | Shared state: parallel live steps share fixed key names and ACL/`GRAPH.CONFIG` writes | index §8 "Live tests" | each live item uses its own throwaway instance; never `falkordb-dev` (6379) | §11 |
| R9 | AC-5's first run misses (proposal K12, index R-5) | likely even for a good system | L1 first; P1-43 is its own contingent step | P1T-31, P1T-36 |

Deliberately **not** tested: §12.

## 5. Acceptance criteria in P1's scope: level, owner, evidence

The source of this table is the proposal's P1 row "Closes" cell (16 ACs, §2), with each AC's wording from
`docs/requirements/connected-code-graph.md`, the executing step from the gates text (P1-31, P1-40, P1-35, P1-37,
P1-41, P1-42) and the closing and "reported not closed" rules from `docs/plans/connected-code-graph-p1-gates.md`
P1-38 and index §5, §8. "Supplier" is the step whose output the test exercises; "executor" is the step that runs the
test and owns the evidence. The report records, per AC, executor, evidence section, verdict.

| AC | Level | Executor (owning step) | Supplier steps (what is exercised) | Evidence and closing rule |
|---|---|---|---|---|
| AC-1 HTTP | L1 | P1-42, item P1T-20 | P1-04, P1-05, P1-08 (`salesperson/` i18n sources, `frontend-ignore` fallback), P1-17, P1-18, P1-21 to P1-23; the proving-system build (P1T-38) | **closed on P1-42 alone**, by P1T-20's pass rule. P1-37's HTTP stratum is AC-5 evidence, not AC-1's |
| AC-2 MCP and wiring | L1 | P1-42, P1T-20 | P1-11, P1-12, P1-17, P1-46, P1-14; P1T-38 | closed on P1-42 alone; wiring entries as `config` items (`-answer-spec2.md` §3.3) |
| AC-4 scripts | L1 | P1-42, P1T-20 | P1-10, P1-11, P1-14, P1-17, P1-46; P1T-38 | closed on P1-42 alone; the launch chain as `Link.bound[]` (`-answer-spec2.md` §3.2) |
| AC-5 never miss | L3 | P1-37 first run; P1-43 only on a miss | P1-32 (verbs, harness), P1-34 (flow), P1-36, P1-47, pre-registration, every build step before the freeze | strata HTTP, MCP, scripts; wiring "screened"; `flow` only if a flag is true after P1-34, otherwise "reported not closed, flow-unavailable" (index §5); the first run is the result |
| AC-6 doubt flagged (P1 kinds) | L1 | P1-41 (P1T-12, the 12 `AC6-*` rows gated P1) and P1-42 (P1T-25, per kind) | P1-05, P1-11, P1-12, P1-17, P1-46 | URL, dynamic import, example config and prompt mention, three instances each in distinct shapes, none `confirmed` |
| AC-8 part 1 | L2 | P1-40 run by `devops`; P1T-26 to P1T-29 verified and reported by `qa-engineer` | P1-29 (broker core), P1-18, P1-20, P1-25, P1-26, P1-48 | rows as `-egress-spec3.md` §8.2 classifies them; "part 1" only, vacuous rows "reported not closed" (index §8, Q-E) |
| AC-9 staleness (code-snapshot stamp) | L1 | P1-42, P1T-21 | P1-19 (stamp, `Snapshot`), P1-39 | stamp present and equal across repeats; the decision-state half is the empty log until P4 (`verdictLogOffset` 0), stated in the report |
| AC-9b partial failure | L1 | P1-41, P1T-16 | P1-20 (stale-kept, not-mapped), P1-22 and P1-39 (`TO_UNIT`, `staleness.units`, `-answer-spec2.md` §3.5) | closed on AB-1, AB-2, LNK-42, LNK-43; AB-3 and AB-4 gate P3 |
| AC-10 in the flow (impact stages) | L4 | P1-27 gate; P1T-39 | P1-27 | **pending stakeholder decision Q-4 (§13).** Until it is answered the report states "reviewed at prompt level, behaviour unobserved", and does not state AC-10 closed |
| AC-11 answer forms (impact) | L1 | P1-42, P1T-19 | P1-24, P1-26, P1-25 | grouped text and Markdown report on a question the parsers resolve (no model call) |
| AC-16 clients | L1 | P1-42, P1T-22 | P1-25 (MCP, CLI, `raw_query`, pin-and-replay), P1-26 | same `facts_bytes` through MCP, CLI and the engine; the plain-language client is tested on the parser path; the raw recipe reproduces the item set (the weaker leg, by decision) |
| AC-17 starting points | L1 | P1-42, P1T-23 | P1-21 (anchors), P1-26 (parsers) | code location and contract point; the other groups are out of scope (§12) |
| AC-18 evidence | L1 | P1-42, P1T-23 | P1-22, P1-23, P1-24 | every item of the impact answers carries chain, file and line |
| AC-21 read-only | L1 | P1-42, P1T-24 | P1-19 (ACL users), P1-25 (`raw_query`) | a write through `raw_query` and a direct `GRAPH.QUERY`/`GRAPH.PROFILE` with the `ccg_raw_<s>` credential are refused by the database, stamp unchanged |
| AC-24 smoke (impact cells, EN and PT) | L3 | P1-35, P1T-33 | P1-26, P1-29, P1-32a (the AC-24 split verb) | dev entry check, then one held-out run with repeat 1 scored; a pass shows the interpreter is not broken, it does not certify 0.92 (`-p1-gates.md` P1-35) |
| AC-25 impact shape | L1 | P1-41 (P1T-17, P1T-18) and P1-42 | P1-22, P1-23, P1-39 | asserted in the P1 form of index §5; the control half closes only if P1-39's shipped walk lists the guarded point, else "reported not closed" (index §5, `-p1-ml.md` §3.5) |

**ACs the requirements define that P1 does not close** (13 of 29 distinct ids, counted with
`grep -o '^- AC-[0-9]*[ab]\?'` over `docs/requirements/connected-code-graph.md`): AC-3, 7, 9a, 12, 13, 14, 15, 19, 20,
21a, 22, 23, 26. No item below targets them.

## 6. P1-41: the 92 P1 rows

**What the index and the gates require P1-41 to contain** (`docs/plans/connected-code-graph-p1.md` §2, §4.2, §7;
`-p1-gates.md` P1-41), mapped to rows. The 92 are the rows whose `gate` is P1 in the fixtures plan §6 and in
`cpg/fixtures/system1/expected/*.yaml` (§2). Rows asserted but red are defects, reported by id.

| Required content (source phrase) | Rows | Item |
|---|---|---|
| every link kind and status P1 builds (kinds with a P1 row: `http`, `mcp_tool`, `script_tool`, `launch`, `library_api`) | the P1 rows of the fixtures plan §7 matrix | P1T-05 (closed-world) and the family items |
| LNK-19 and LNK-20 (P1-46's pack) | LNK-19, LNK-20 | P1T-11 |
| cross-repo Compose context to unit (`-p0-compose-binding.md`) | LNK-15; the `JobPanel` binding is asserted through it, `OV-2` is a P3 row | P1T-09 |
| the 8000/8200 URL binding, and the example config `possible` | LNK-18 and LNK-21 (one unit, two launch configs); AC6-EX-1 (`File.role = example`) | P1T-09, P1T-12 |
| mount composition | LNK-04 (router `prefix` mounted at `/api`) | P1T-06 |
| the AC-1 lifting chain, field path, two levels | LNK-10 (`api.search.query` through `request()`) | P1T-08 |
| `ownedRegions` | no fixture row names it; P1-05's synthetic test is its only P1 test (index §4.2) | none |
| multi-candidate links (first real cases) | LNK-20 (one link per provider of the kind), LNK-30, LNK-39, LNK-40 (candidate sets) | P1T-10, P1T-11 |
| stale-kept and not-mapped | AB-1, AB-2 (AB-3 and AB-4 gate P3); with LNK-42, LNK-43 | P1T-16 |
| B1's four assertions, "those the fixtures plan gates at P1" | B1F-W, B1F-R, B1F-W2, B1F-RU (the `B1S-*` rows gate P3) | P1T-13 |
| contract walk from a provider-less contract point lists its consumers (index §4.2) | no row is labelled; this plan assigns LNK-42, LNK-43 (provider in an unbuilt unit) | P1T-16 |

**Rows by item, build state and supplier.** Build states are the fixtures plan §8 scenarios: `SC-STEADY` is a
fresh build at H4 (`svc-tools` never built, `svc-legacy` unbuildable); `SC-FAIL-UNIT` and `SC-NEVER-UNIT` as named.
Every id below is a P1 row; the union of this column is the 92, and P1T-05 asserts that equality. The supplier column
names the candidate steps; a red row is triaged to one step by the layer its evidence shows (extraction, linking,
assembly, engine), and the report records that layer with the row id.

| Item | Rows (explicit) | n | Build | Supplier steps |
|---|---|---|---|---|
| P1T-06 | LNK-01, 02, 04, 06, 07, 08, 36, 37, 38 | 9 | SC-STEADY | P1-05, P1-16, P1-17, P1-18, P1-19 |
| P1T-07 | KN-1, 2, 3, 4, 5 | 5 | SC-STEADY | P1-17 (`matchers.py`), P1-21 (anchors); `ccg.keynorm` |
| P1T-08 | LNK-10, 11, 12 | 3 | SC-STEADY | P1-05, P1-16, P1-17 |
| P1T-09 | LNK-13, 14, 15, 18, 21, 22 | 6 | SC-STEADY | P1-11, P1-14, P1-17, P1-18 (with P1-20's cross-repo unit index) |
| P1T-10 | LNK-16, 17, 30, 39, 40 | 5 | SC-STEADY | P1-16, P1-17, P1-18, P1-22 |
| P1T-11 | LNK-19, 20 | 2 | SC-STEADY | P1-10, P1-46, P1-17 |
| P1T-12 | AC6-URL-1, 2, 3; AC6-DYN-1, 2, 3; AC6-EX-1, 2, 4; AC6-PM-1, 2, 3 | 12 | SC-STEADY | P1-05, P1-11, P1-12, P1-17 |
| P1T-13 | B1F-W, B1F-R, B1F-W2, B1F-RU | 4 | SC-STEADY | P1-05, P1-07, P1-17, P1-22 |
| P1T-14 | TW-1, 2, 3; NC-1, 2, 3, 5, 6, 7; MR-PY, MR-TS; FT-1, FT-2 | 13 | SC-STEADY | P1-05 (resolution, field depth), P1-12 (docs snippets), P1-17 |
| P1T-15 | SK-1 to SK-18 | 18 | SC-STEADY, override ON and OFF | P1-08, P1-04, P1-44 |
| P1T-16 | AB-1; AB-2; LNK-42; LNK-43 | 4 | SC-FAIL-UNIT (AB-1); SC-NEVER-UNIT (AB-2); SC-STEADY (LNK-42, 43); built by P1T-04 | P1-20, P1-44, P1-22, P1-39 |
| P1T-17 | DF-1, 2, 3, 4; CD-1, 2, 3, 4 | 8 | SC-STEADY | P1-22, P1-23, P1-39; P1-09 if a flag turns true |
| P1T-18 | RV-1, 2, 3 | 3 | SC-STEADY | P1-23 |
| | **Total** | **92** | | |

The totals column recomputed: 9+5+3+6+5+2+12+4+13+18+4+8+3 = 92, and the per-family counts equal §2's family
counts (LNK 9+3+6+5+2+2 = 27 across items; AC6 12; SK 18; AB 2; B1F 4; KN 5; NC 6; TW 3; MR 2; FT 2; DF 4; CD 4;
RV 3).

**Prerequisites, checked first** (`-p1-gates.md` P1-41): P1-02 (F-1) and P1-06 (F-9) done, and the build ran with
`header.history` available (a shallow clone is a build error, index D1). Plus the fixtures plan's own exit
criteria, `TP-001..004`, `013`, `014`, `016` green with their planted failing inputs red (fixtures plan §10); a
row whose build or scenario precondition does not hold is `blocked`, not failed (fixtures `TP-015`).

## 7. P1-42: AC-level assertions

`-p1-gates.md` P1-42 lists, in its heading, AC-6, 9, 16, 17, 18, 21 and 25 and, in its body, the proving-repo asks for AC-1,
AC-2 and AC-4, AC-11 and the reverse question. Dev anchors only (`cpg/eval/dev/ac5-*.jsonl`); the held-out strata of
P1-37 are the AC-5 evidence and are never used here.

| Item | Asserts | Needs |
|---|---|---|
| P1T-19 | AC-11 | P1-24, P1-26 |
| P1T-20 | AC-1, AC-2, AC-4 on dev anchors, with the pass rule below | P1-23, P1-11, P1-46, P1T-38 |
| P1T-21 | AC-9 stamp and determinism | P1-19, P1-39 |
| P1T-22 | AC-16 | P1-25, P1-26 |
| P1T-23 | AC-17 and AC-18 | P1-21, P1-22, P1-24 |
| P1T-24 | AC-21 | P1-19, P1-25 |
| P1T-25 | AC-6 per kind | P1-41, P1-17 |
| P1T-17, P1T-18 | AC-25 in the P1 form (index §5) and the reverse question | P1-39 |

**P1T-20 sample and pass rule.** One structured impact request per dev anchor, sampled by the change the AC names:
AC-1 the anchors of `ac5-http.jsonl` whose changed symbol is a `falkor-chat` server endpoint; AC-2 the anchors of
`ac5-mcp.jsonl` and `ac5-wiring.jsonl` whose changed symbol is a `falkor-chat` MCP tool; AC-4 the anchors of
`ac5-scripts.jsonl` whose changed symbol lies under `cypher-mcp/`. Each filter is by the anchor's `path` or contract
kind, the matching count is recorded in the report, and a filter that matches no anchor makes the AC's item blocked,
never passed. (This run observed the distinct `anchorId` counts per file, 12, 8 for the mcp and wiring files together
and 22; the filtered counts are the executor's.) **Pass:** every dev item of a sampled anchor (matched by its
`impactedId`) is listed in the answer, as `confirmed` or `possible`, with a chain whose hops carry file and line;
AC-2's wiring items are `config` items and AC-4's launch chain is the `Link`'s `bound[]`. A missing item is a defect,
triaged to a class of `-eval-protocol-ml.md` §12.1 and not waived, except the P1-37 disposition class `key-error` (`-p1-gates.md` P1-42): an item whose gold is wrong, with the evidence, is reported and excluded; every other miss stays a defect, per anchor, with no recall floor. Dev anchors were used for development, so a pass
here is optimistic and is not AC-5 evidence; AC-1, AC-2 and AC-4 close on this item alone.

**Note for P1-23's gate (walk tests, P1T-17).** `answer.validate` (V18) accepts a chain that keeps `create_order` `confirmed` with `apply_discount` left in the `value` state, so the validator does not force the engine to enter the `function` state (`-answer-spec2.md` §5 `B-flow`: `HAS_CALL ←` is a `FUNCTION_ENTRY` row, the next `CALLS ←` a `FLOW_ROWS` row). P1-23's own walk tests must pin that a walk from the anchor `base_price` delivers the caller `Symbol` `apply_discount` in the `function` state, with a planted failing walk that leaves it in `value`; P1T-17 (DF-2 `possible`) fails the same walk only indirectly.

## 8. AC-8 harness (P1-31, P1-40)

Design under test is `docs/plans/connected-code-graph-egress-spec3.md` §8 (topology, preconditions P-a to P-g, the
controls, rows 8.1 to 8.16 and their computed, vacuous or unrun status in §8.2). This plan does not restate them.
`-egress-spec4.md` adds no row; it changes the runner grammar and record fields that rows 8.5 and 8.6 read, so the
harness recomputes row 8.6 against the lists at `listDigest` as built.

**Report rule.** For every row the report states the §8.2 class, the evidence, and, for a vacuous or unrun row, the
word "vacuous" or "unrun", never "pass". The L2 verdict is "AC-8 part 1" only (index §8, Q-E). `devops` runs the
scenario and records the result (P1-40); `qa-engineer` checks the record against P1T-26 to P1T-29 and writes the
report section.

**Checks shown to fail** are P1T-26 to P1T-29's red cells: the detector control, the preconditions' own positive
controls, and the in-run controls of spec3 §8.1.

## 9. AC-5 and AC-24 evaluation (P1-32 to P1-37, P1-43, P1-47)

Method and thresholds are `-p1-ml.md` §3.5, §4.3, §5.3, §6 and `-p1-gates.md` P1-32 to P1-37, P1-43, P1-47; not
restated. This plan fixes the order and what the executor verifies, and keeps the evaluation inside its own
fences.

**Order** (index §7, critical path and "Before the record"): P1-32a verbs; P1-32b harness (planted-input tests);
the freeze-time `check-report`, run by the custodian; P1-34 decision (entry check, expected not run); P1-35 dev
entry check; P1-47 capture on an isolated instance; P1-36 tooling; the five decision and pre-registration records
committed by `teco` before the acts they govern; then P1-37. P1-42 green precedes P1-37. P1-43 exists only on a miss.

**What is tested before any blind run** (P1T-30): the harness and verbs against **planted inputs**: a planted
held-out probe gives exactly one `check-report` hit; a fabricated-chain item fails C5's chain replay; a
`renormalize-check` difference stops the gate; a first-contact claim with any of the four pre-registration
fields unequal, or with an uncommitted change under `cpg/`, is refused; an out-of-scope item is flagged and never
scored; the routing rules and E12's floor under mutation (`-p1-gates.md` P1-32 gate). These use dev or synthetic
data only.

**Blind runs** (P1T-31 to P1T-36): executed through the harness only. This plan names no held-out path and the
executor does not open held-out data or the custodian root outside the harness (`-p1-ml.md` §6.4). Reports carry
counts, hashes and classes only. The first run is the result; a miss is triaged into exactly one class of
EP §12.1 and P1-43 starts; fixes go on dev items.

**AC-24 smoke** (P1T-33): dev entry check first; held-out run once; repeat 1 scored; `cacheHit` on repeat 1 stops
the run. A pass is reported as "not broken", never as the 0.92 certification.

## 10. Test items

Type: C contract, F functional, I integration, L live, E evaluation, review. Priority is risk times likelihood (§4). The executor is the step whose report section holds the evidence.

| ID | Title | Preconditions | Steps | Expected | Red when (planted failing input or positive control) | Prio | Type | Executor |
|---|---|---|---|---|---|---|---|---|
| P1T-01 | Baseline | none | from `cpg/`: `.venv/bin/python -m pytest -q`; from `cpg/sandbox/tests/` the shell suite; record collected, passed, skipped, deselected | green; counts recorded; any red stops L1 to L3 and is reported | a planted failing test in a scratch copy shows the run goes red; a skipped `slice` gate is reported unrun, never green | high | L0 | each gate step, before its items |
| P1T-02 | Slice gates not skipped | P1-04's regenerated slice artifact | run the `slice`-marked tests; record artifact sha256 | `0 skipped`, sha256 recorded (index D9) | absent artifact: the gate must report unrun | high | L0 | P1-41 |
| P1T-03 | P1-41 prerequisites and fixture hygiene | P1-02, P1-06, P1-08, P1-10, P1-12, P1-14 done | check F-1 and F-9 test evidence; run fixtures `TP-001..004, 013, 014, 016`; confirm `header.history` is available in the build | all true; else P1-41 is blocked and returned | a manifest copy without `AC6-EX-4` turns `TP-004` red (the AC-6 floor is read per gate); a row moved to gate P1 that needs Java turns `TP-014` red | high | C | P1-41 |
| P1T-04 | Fixture builds: SC-STEADY, SC-FAIL-UNIT, SC-NEVER-UNIT, with cost | P1T-03; a throwaway instance per build (`infra/ccg-falkordb.sh --throwaway`); fixture system materialized (`cpg/fixtures/materialize.py`) | `qa-engineer` runs, through P1-20: SC-STEADY (a fresh build at H4); SC-FAIL-UNIT (H1, then H4: `svc-legacy` fails with a previous refresh); SC-NEVER-UNIT (`svc-tools` excluded from the build set); record per-unit status, wall time, peak RSS | SC-STEADY: `svc-tools` not mapped, `svc-legacy` failed with no previous refresh, all other units built; SC-FAIL-UNIT and SC-NEVER-UNIT as fixtures plan §8; one unit at a time; times recorded (index R-4) | a fixture with the failure marker removed: `svc-legacy` must then build, so the AB expectations flip | high | I/L | P1-41 |
| P1T-05 | Closed-world comparison and P1-form statuses of the 92 | P1T-04 | emit every link, status and reason; compare with the 92 P1 rows: status and reason as the P1 form of index §5 where a row carries `p1Form` (fixtures plan §5.5), the manifest form otherwise; list the assertions run; then a flag-true re-run, when one exists, asserting the manifest form | the asserted set equals the 92 ids of §6 exactly; no *unplanted* link, where unplanted means matching no row of any manifest, whatever its gate; a link of a row gated P2 or later is reported "planted, not asserted at P1", never red; every `possible` has its reason | a stub emitting a link that matches no manifest row; one with a wrong reason; a row dropped from the asserted set (fixtures `TP-010`); DF-2 or DF-3 emitted `confirmed` under a false flag (`p1Form` says `possible`/`flow-unavailable:python`); a form with the wrong locator or distance (fixtures `p1_form_violations`) | high | I | P1-41 |
| P1T-06 | Key-match families, mount composition | P1T-04 | assert LNK-01, 02, 04, 06, 07, 08, 36, 37, 38: status, reason, provider and consumer locators, evidence chain | all `confirmed`/`key-match` as planted; LNK-04 resolves the `/api` mount prefix | a stub that drops the mount prefix; a consumer with an unresolved interpolation (must become `possible`/`partial-key`) | high | I | P1-41 |
| P1T-07 | Key normalization order (F-13) | P1T-04 | assert KN-1 to KN-5 through every P1 consumer (matcher, anchor, assembler) | expected `keyNorm` per fixtures §5.2; KN-5 masked and `possible`/`partial-key` | resolve-before-decode fails KN-1, KN-2; mask-before-decode fails KN-5 (the mutants of `cpg/tests/test_fixture_keynorm_mutants.py`) | high | F | P1-41 |
| P1T-08 | Wrappers and the lifting chain | P1T-04 | assert LNK-10, 11, 12; the ordered `LIFTED_VIA` entries of LNK-10 (two-level field path) | `confirmed`; `Link.reason` `key-match` (LNK-10, 11) and `binding-confirmed` (LNK-12), never `lifted-role` (the `CONSUMES` edge's reason); chain order and `siteId` as the graph note states | a linker that drops the second level (LNK-10 must then go `possible` or unresolved) | high | I | P1-41 |
| P1T-09 | Bindings, launch chain, cross-repo Compose | P1T-04 | assert LNK-13, 14, 15, 18, 21, 22; for LNK-15 the context `../../fx-job` resolves to unit `job`; LNK-18 one unit for 8000 and 8200 | `confirmed`/`binding-confirmed`; `BOUND_BY` chain in `ord` order | a build context pointing at an unmapped repo: LNK-15 must not stay confirmed; a red LNK-15 is an escalation (index §4.2), not a fixture patch | high | I | P1-41 |
| P1T-10 | Port-shared, partial key, multi-candidate | P1T-04 | assert LNK-16, 17, 30, 39, 40: status, reason, and that the candidate set contains the named true target (fixtures `TP-019`) | `possible` with the stated reason; candidates contain the target; none collapsed | a stub with an empty or wrong candidate set; one collapsing candidates to a single link | high | I | P1-41 |
| P1T-11 | Script to tool | P1T-04 | assert LNK-19 (bound, `key-match`) and LNK-20 (`shell-dynamic`, one link per `mcp_tool` provider, including `job:job/mcp_admin.py::reindex`, NC-5) | as planted | a pack lacking the dynamic-`mcp-call` negative: LNK-20 would become confirmed | high | I | P1-41 |
| P1T-12 | AC-6 rows | P1T-04 | assert the 12 P1 AC6 rows: `possible`, the stated edge or link reason, not `confirmed`; AC6-EX-4 is an `http` link from `svc:clients/peer.py::peer_health` (`os.environ["FX_PEER_BASE"]`) bound only through service `svc` of `deploy/compose.example.yaml`; EX-2 and EX-4 are two instances of one linker rule (P1-17 rules (5) and (7)), so a defect there turns both red, and fixtures `TP-004`, which counts distinct `item` strings, cannot tell the mechanism apart (accepted) | as planted | an engine stub that confirms one instance; one emitting `possible` with no reason (fixtures `TP-009`) | high | F | P1-41 |
| P1T-13 | B1 on the field | P1T-04 | assert, on the field `OrderCache.entries`: B1-1 (a change to W lists R), B1-4 (W does not list W2) and B1-3 (renaming the field lists R and W), the three that the B1F rows gate at P1 (fixtures plan §5.4); B1-2 (RCA from R lists W) needs P5 and is recorded deferred | B1-1, B1-3, B1-4 pass; B1F-RU is listed after the confirmed reader; B1-2 reported deferred, not passed | a stub listing W2 under W (B1-4); a rename that lists R but not W (B1-3); a B1-2 line reading "pass" | high | I | P1-41 |
| P1T-14 | Twins, negatives, mis-resolution, field depth | P1T-04 | assert TW-1 to 3 `confirmed`; NC rows produce no link and list nothing unaffected; MR-PY and MR-TS `possible`/`resolution-unverified`; FT-1 `field-truncated`, FT-2 `confirmed` | as planted | an engine downgrading every dynamic-looking site (TW fail); a stub linking NC-3 (wrong verb) or NC-7 (missing prefix) to `read_item` (fixtures `TP-012`, `TP-015`, `TP-017`) | medium | F | P1-41 |
| P1T-15 | Omitted files | P1T-04; P1-08's record of whether the `jssrc2cpg` ignore override exists | generate the probe table (language x reason x overlap x location x override) and run SK-1 to 18 in the mode P1-08 recorded: override ON if proven, otherwise OFF (then SK-5 is expected `frontend-ignore`); the report names the mode run | outcomes of fixtures §5.11 for that mode; sorted, unique, POSIX; no `file` record for a listed path | an extractor emitting `file` records for test files; a table cell with no row (fixtures `TP-007`); an OFF-mode run that expects a `file` record for SK-5 | high | I | P1-41 |
| P1T-16 | Stale-kept, not-mapped, provider-less walk | SC-FAIL-UNIT and SC-NEVER-UNIT builds of P1T-04 | assert AB-1 (an answer crossing `svc-legacy` says previous refresh H1, with the code), AB-2 (not mapped), LNK-42 and 43 (`possible`/`target-not-mapped`); a contract-point walk from `GET /legacy/fee` lists its consumer | as planted; other units updated | a stub reporting a failed unit as current (fixtures `TP-011`) | high | I | P1-41 |
| P1T-17 | AC-25 forward, P1 form | P1T-04 | ask the forward question for `CF-DF` (`base_price`) and `CF-CD` (`eligible`); assert the P1 form of fixtures plan §5.5: DF-1 `confirmed`/`resolved-call`, DF-2 and DF-3 `possible`/`flow-unavailable:python` (each with its locator and distance), DF-4 as the manifest, CD-1 to CD-3 not asserted, the caller `finalize` at distance 1 `possible`/`flow-unavailable:python`; A25-1, A25-2 for every asserted item; A25-3 only where a group holds both statuses (at P1 no forward group does, so every forward pair is vacuous and reported; the RV-1/RV-2 core/1 groups carry the check) | direct iff distance 1; groups ordered by repository, component, distance; confirmed before possible inside a group that holds both; vacuous A25-3 pairs listed as vacuous | DF-2 or DF-3 listed `confirmed`; a chain that keeps `create_order` `confirmed` with `apply_discount` left in the `value` state (the P1-23 note below); sorting with no status key; distance first; distance off by one; groups flattened (fixtures `TP-008`) | high | I | P1-41, P1-42 |
| P1T-18 | AC-25 reverse, P1 form; control-half decision | P1T-17 | ask the reverse question for the RV-1..3 functions; assert the dependency list in the P1 form (`connect` in RV-1, `load_rate` in RV-2, `base_price` in RV-3 `possible`/`flow-unavailable:python`, every other dependency as the manifest), and A25-4; record whether P1-39's shipped walk lists the guarded point | list equals the RV rows, nothing extra; the A25-3 pairs of the RV groups (`find_hook` after `is_member` or `tier_of`) hold both statuses and pass; the control half is closed or "reported not closed" by that fact | a stub adding an unplanted dependency | high | I | P1-41, P1-42 |
| P1T-19 | AC-11 grouped answer and Markdown report | P1T-04, P1-24, P1-26 | ask one question the deterministic parsers fully resolve (a `path::symbol` or `file:line`, no model call); render the grouped text and the Markdown report | grouped by repository, component, distance; each hop has file and line; interpretation shown at the top; every item in its group | a planted answer with one item outside its group, one with groups out of order, and one hop with no line: each must fail the render check (AC-11 grouping); a planted "Ignore previous instructions" in `Cond.code` must occur only inside the fence (F-3) | medium | F | P1-42 |
| P1T-20 | AC-1, AC-2, AC-4 on dev anchors | P1T-38; dev anchors only | per the sample and pass rule of §7: one structured impact request per sampled dev anchor; check every dev item is listed with a chain, the `config` items and `Link.bound[]` | the pass rule of §7 holds for AC-1, AC-2 and AC-4 separately | a dev item deleted from the answer set of one sampled anchor (an answer mutant, not an oracle edit) must be reported missing; a filter matching no anchor reports blocked; no held-out file is opened | high | I | P1-42 |
| P1T-21 | AC-9 stamp and determinism | P1T-04 | ask one request twice, then after a rebuild from the same commits; compare `facts_bytes` and the stamp | stamp present; identical facts; `verdictLogOffset` 0 stated | a rebuild from a changed commit must change the stamp | high | I | P1-42 |
| P1T-22 | AC-16 four clients | P1-25, P1-26, P1T-04 | submit one structured request through MCP and CLI; ask the same as a question the parsers resolve (no model call) and read its echoed request; re-run `recipe.queries` as a raw query | identical `facts_bytes` for MCP, CLI and engine; the echoed request equals the structured one; the raw leg reproduces the item set; the free-prose leg is AC-24's (P1T-33), not run here | a client altering `interpretation.request`; a raw recipe missing one read of carry-over C3 | high | I | P1-42 |
| P1T-23 | AC-17 and AC-18 | P1T-04 | resolve a file+line and a contract point without naming component, language or graph; walk every item of an impact answer | each starting point resolves; every item shows its chain with file and line | a file+line in a `.d.ts` or vendored path must answer "not found", never a symbol | high | I | P1-42 |
| P1T-24 | AC-21 read-only | throwaway instance with the four ACL users | a write via `raw_query`; a direct `GRAPH.QUERY` and `GRAPH.PROFILE` with the `ccg_raw_<s>` credential; compare the stamp before and after | each refused by the database; stamp unchanged | positive control: the same statement under the loader credential succeeds on the throwaway instance | high | I/L | P1-42 |
| P1T-25 | AC-6 per kind | P1T-12 | per P1 kind count instances, shapes, statuses and the edge-level reason of the planted consumer's item | URL, dynamic import, example config (AC6-EX-1 `launch`, AC6-EX-2 `http`, AC6-EX-4 `http`) and prompt mention: three instances each in distinct shapes, none `confirmed` | an engine stub that confirms one instance; one reporting `flow-unavailable` as the planted consumer's reason | high | F | P1-42 |
| P1T-26 | AC-8 detector control | P1-31 harness, throwaway instance | start the detector; connect once from a component container; plant a `conn` before the start marker and one after the flush marker | the verdict function fails on each | a verdict function that ignores the window | high | L | P1-40 (`devops` runs and records; `qa-engineer` verifies and reports) |
| P1T-27 | AC-8 preconditions P-a to P-g | P1T-26 | run the preconditions from each component container at the start and at the end of the run | all hold; a precondition that cannot be established is a failed run, never skipped | a container given a default route; a stub resolver answering one `.invalid` name | high | L | P1-40 (`devops` runs and records; `qa-engineer` verifies and reports) |
| P1T-28 | AC-8 main run, rows 8.1 to 8.16 | P1T-27 | full refresh plus queries of the mixed system; compute each row | per §8: computed rows pass, mixed rows' computed halves pass, zero `conn` in the window, every allow has exactly one `close` | the in-run controls (R4 body field, R5 hosted call, `--jar4import-url`, the two overrides) each must appear as stated | high | L | P1-40 (`devops` runs and records; `qa-engineer` verifies and reports) |
| P1T-29 | AC-8 report of vacuous and unrun rows | P1T-28 | list rows 8.2, 8.4, 8.12, 8.13 (residue computed), the proxied halves of 8.3 and 8.5, 8.15 (ii) and (iii), and 8.16 if `strace` is absent | each reported "vacuous" or "unrun" with the reason; allows with `privateAddress = true` listed (expected none) | a report line that says "pass" for a vacuous row | high | L | P1-40 (`devops` runs and records; `qa-engineer` verifies and reports) |
| P1T-30 | Eval tooling and harness on planted inputs | P1-32a and P1-32b delivered | run the verbs and `gate.py` against dev or synthetic inputs only (§9) | each planted failure is caught as §9 lists | the mutants on routing and E12's floor | high | C | P1-32 gate |
| P1T-31 | AC-5 first gate run | prerequisites of `-p1-gates.md` P1-37; records committed; P1-42 green | run the harness; report as run (misses by class, precision, wrong-possible, noise, oracle beside recall) | as P1-37 states; closing rule §5 | process item: no planted input. Its fail conditions are `-p1-gates.md` P1-37's: any miss (triaged, then P1-43), confirmed precision below 0.95, wrong-possible above 4 per real impact. The harness's own planted-input checks are P1T-30 | high | E | P1-37 |
| P1T-32 | Flow decision | P1-09, P1-34 entry check | apply the entry check; record the decision | flag false for both languages is the expected outcome; `flow` stratum reported "not closed, flow-unavailable" | process item: no planted input. The entry check is `-p1-ml.md` §3.1's value (465/481 on the dev figure, per `-p1-gates.md` P1-34); below it the fresh draw is not run, and a run that spends the draw below it is a protocol violation | high | E | P1-34 |
| P1T-33 | AC-24 smoke | `ccg-prereg-ac24-v1` committed; the local model of `-p1-ml.md` §4 | dev entry check; one held-out run; repeat 1 scored | thresholds of `-p1-gates.md` P1-35; reported as "not broken" | a repeat-1 `cacheHit` stops the run | high | E | P1-35 |
| P1T-34 | Runtime-oracle capture fence | `ccg-oracle-v1` committed; throwaway instance | run the capture wrapper | effective port is neither 6379 nor 6390; the instance carries the `ccg-falkordb=1` label | `FALKORDB_PORT` unset: the wrapper must refuse | high | L | P1-47 |
| P1T-35 | Noise measurement | P1-47, P1-32 | run the noise tooling on the gate answers | the ratified thresholds reported with intervals, split as P1-36 states; gate 1 gates at Q <= 4 only | process item: no planted input. The thresholds are `-p1-ml.md` §5.3's, reported with intervals; the labeller must be a fresh `qa-engineer`, not a pack author (`-p1-gates.md` P1-36) | medium | E | P1-36 |
| P1T-36 | Re-gate (contingent) | a P1-37 miss | its own brief from the triage; a new record before the new build | as `-p1-gates.md` P1-43 | a re-gate on the same list is invalid | high | E | P1-43 |
| P1T-37 | AC-closure inputs | all of the above | compile for P1-38: per AC the closing step, the report section, and "reported not closed, because" where so | one line per AC of §5; none closed by step completion alone; AC-10 is listed by name with its pending status (§13 Q-4) | an AC other than AC-10 marked closed with no executed item behind it | high | review | P1-38 |
| P1T-38 | Proving-system build | P1-03 system manifest, P1-20 refresh, P1-02; its own throwaway instance | `qa-engineer` runs a full refresh of the proving system (scope of index §2) through P1-20 before P1T-20; records per-unit status, wall time and peak RSS | a built, stamped snapshot of the monorepo; units, stale-kept and not-mapped listed; cost recorded (index R-4) | a unit that fails to build is listed stale-kept or not mapped, never absent; a peak RSS over the P1-20 brief budget is reported | high | L | P1-42 |
| P1T-39 | AC-10 review record | P1-27 delivered (`claude/teco/teco.md`, `skills/cpg-analysis/SKILL.md`) | read the P1-27 text for the three behaviours of `-p1-answers.md` P1-27: a brief for a covered system names the answer, the stamp and the staleness note; an uncovered system keeps the old `CPG:` line; a refresh failure reads "stale-kept", never omitted; if a real architect or implementer brief on a covered system exists by P1 close, inspect it for the same three | a written finding per behaviour; AC-10 reported "reviewed at prompt level, behaviour unobserved" until Q-4 is answered | a copy of the `teco` text with the stamp sentence deleted must be flagged; a copy that drops the stale-kept wording must be flagged | medium | review | P1-27 gate; P1-38 records it |

## 11. Environment, data, entry and exit criteria

**Environment.** Linux host, Python 3.12, `cpg/.venv`; run suites with `cpg/` as the working directory (root
`AGENTS.md`). Live items use a throwaway instance (`infra/ccg-falkordb.sh --throwaway`, P1-48) each, never the shared
`falkordb-dev` (6379), and `ccg-falkordb` (6390) only through a reservation `teco` grants (fixtures plan §10; index §8).
No item writes to any shared database. Docker is required for L2 and the sandboxed builds; `strace` is not installed
on this host (`-egress-spec3.md` §9), so row 8.16 is unrun unless provisioned (§13 Q-6).

**Data.** Fixtures: `cpg/fixtures/system1/` materialized under `cpg/.ccg-work/fixtures/` (gitignored), fixed dates,
stable SHAs (fixtures plan §8). Proving-repo asks: dev anchors `cpg/eval/dev/` only. Held-out data: reached only
through the P1-32b harness in P1-35, P1-37, P1-43.

**Entry.**
1. **Source amendments landed, in the order `-p1-gates.md` P1-41 states.** (a) The architect's ruling (index §5, P1-41
   and P1-42 text, P1-05 text): landed. (b) `qa-engineer`: the manifests (`p1Form` on DF-1 to DF-3, CD-1 to CD-3, RV-1 to
   RV-3, compared on status, reason, locator and distance; row AC6-EX-4 and its planted source), the fixtures plan v1.4 (§5.3, §5.5, §6, §7, the `OV-2` sentence), the
   per-gate AC-6 census in `cpg/tests/test_fixtures.py`, and this plan: landed in this revision. (c) The `analyst`
   re-gate of this plan: pending; P1-41 does not start before it.
2. P1T-01 green.
3. For P1-41: the step set of its `Needs` (index §7: 33, 20, 25, 06, 02, 08, 10, 12, 14) done, which transitively
   requires the linker, assembler, store loader, refresh, engine and clients; P1T-01's report records which exist.
4. For P1-42: P1-41 green or its reds explained, P1-26, and P1T-38 (the proving-system build).
5. For P1T-26 to P1T-29: P1-31 and P1-40's `Needs`.
6. For the blind runs: the "Before the record" actions of index §7 done, the records committed, and no uncommitted
   change under `cpg/`.

**Exit.** Every `P1T-NN` has an outcome (pass, fail, blocked with the missing subject, skipped with the reason) and
its evidence in the report; every red row is listed by row id with the layer that failed; no vacuous, skipped or
unrun result is reported as a pass; each AC of §5 has a line in P1T-37's closure inputs with executor, evidence
section and verdict, "reported not closed, because" where the closing rule says so, and AC-10 with its pending
status; the report records the commit and the counts of P1T-01.

**Report** (`docs/test-reports/connected-code-graph-p1-report.md`): summary with the required `CPG:` line, results
table by `P1T-NN`, defects (title, severity by user impact, steps, expected, actual, evidence), coverage and gaps,
recommendations. Written by `qa-engineer`, one section per gate step.

## 12. Out of scope and deliberate omissions

- **ACs P1 does not close:** AC-3, 7, 9a, 12, 13, 14, 15, 19, 20, 21a, 22, 23, 26 (§5), and the parts of AC-17
  beyond code location and contract point.
- **Fixture rows not gated at P1:** 41 of 133 (P2 16, P3 21, P4 1, P5 3): `fx-lib` rows, AB-3, AB-4, `OV-*`, `B1S-*`,
  `SQC-*`, `TF-*`, AC6-EX-3, AC6-TOP, AC6-QL. They are not asserted and not counted.
- **The git-fetch path** (`R2`, `R8`, `R9`, proxy listener): not built in P1 (option C). Only the controls and
  vacuous rows of `-egress-spec3.md` §8.2 are in scope.
- **Held-out content.** This plan describes no held-out item, key or path, and the executors of L1 do not read any.
- **Performance and scale** beyond recording wall time and peak RSS in P1T-04 and P1T-38 (a budget is P1-20's brief).
- **Indexer adoption quality** (P1-15) beyond its effect on the gate build; its measurement is `data-scientist`'s.
- **`ownedRegions` on a fixture:** no fixture row plants it; P1-05's synthetic test is its only P1 test (index §4.2).
- **Not re-litigated:** unit-level behaviour owned by each build step's tests (L0).

## 13. Open questions

**Source disagreements:** none open. The architect's rulings (index §5 and §4.2, `-p1-gates.md` P1-33, P1-41, P1-42,
`-p1-extraction.md` P1-05) and the fixtures plan v1.4 resolved the ones this plan's first revision reported; their
text is the authority and this plan cites it.

**Open stakeholder questions** (this plan does not decide them):
- **Q-4 (AC-10).** May AC-10 close on review alone? The requirement ("briefs and reviews at the applicable stages carry
  the impact answer") is observable only in a real `teco` dispatch; P1-27 is "reviewable, not executable". Raised
  through `teco`. Until answered the report says "reviewed at prompt level, behaviour unobserved" (P1T-39, §5).
- **Q-6 (`strace`).** Row 8.16 is the only attestation of the broker's own connections and needs `strace`, a host
  package. Provisioning it needs the user's consent, relayed by `teco` to `devops`. Until then the row is reported
  "unrun" (P1T-29).

