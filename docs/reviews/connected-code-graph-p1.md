# Review: connected-code-graph P1 plan set

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1

### Scope and verdict

**Reviewed (static, pre-implementation, all uncommitted):** `docs/plans/connected-code-graph-p1.md` (index) and
`-p1-extraction.md`, `-p1-linking.md`, `-p1-answers.md`, `-p1-gates.md`, against the proposal §4 P1 row,
`-p0-2.md` §15, the P0 disclosures, the requirements, the ledger, and the live tree under `cpg/`. Read root
`REVIEW.md`; there is no `cpg/REVIEW.md`. Nothing was written to any database; no mutating git command was run.

**Verdict: needs changes.** Two blockers (RV-1, RV-2): the plan's headline flow decision cannot be executed as
written, and the build path never calls three of its own extractors. Ten majors follow; most are seam defects
between sub-plans and the tree, cheap to fix in the plan. The obligation walk itself is good (RV-0).

CPG: considered, not relevant — no graph is loaded for `cpg/` (the plan says so, §2); `cpg/ccg/` and the spikes were read directly from the tree.

**Growth-stops check:** these are new documents, not a revision of a prior version, so there is no growth to
classify. The set already complies on the neighbouring rules: current design only, no finding-to-resolution
table, a live spec is not restated (`cpg/` has none, §11). No growth-stops finding.

Finding ids are `RV-n` (not `P1-n`, which would collide with the plan's step ids). Each says *verified* (ran or
traced) or *inferred*.

### Blockers

**RV-1 (blocker, verified) — The recommended flow path (§5, P1-09, P1-34) cannot run, and as specified cannot pass its own entry check.**
(a) *Scope.* `docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md` §a2.2.2 gives python's 173
misses as projection 111 = 67 (R1) + 17 (R3-only) + 27 (comprehension 12, call-as-CDG-source 10, dict/array/spread
4, alias 1); join 40 = 36 (`for` header) + 4 (the `not (self._open or ...)` contains-join); frontend 9,
definition 9, undetermined 4. I re-derived the plan's ceiling and it holds: (308+151)/481 = 459/481 = 0.954, and
js (61+22)/152 = 83/152 = 0.546. But index §5 option 1 and P1-09 (1)-(3) port R1, R2, the 27 unverified causes
and the `for`-header join only. **R3 (the 17) and the 4 contains-join misses are in neither.** What P1-09 as
written reaches: 308+67+27+36 = 438/481 = 0.911; adding R3 gives 455/481 = 0.946. Passing 0.95 needs 457/481, i.e.
149 of the 151 fixable misses fixed, on the very 30 methods the entry check measures (the 481 pairs *are* those 30
methods). So the 0.93 entry check is unreachable by a faithful P1-09 (0.911 < 0.93), and P1-34 step 1 ends the
step by construction. R3 is a scoring-time relaxation (report a2.2.5 note "R3 is not a projection"), so porting it
needs its own design line in P1-09. (b) *Tooling.* `cpg/spikes/frontend-fitness/s28_run.py` hard-wires
`EXTRACT = CPG / "spikes" / "vertical-slice" / "extract.sc"` (the old projection) or the guarded copy of it;
P1-34 step 3 says to run it "with the fixed projection from P1-09" but no step re-points it, and D7 deletes the
file it points at (see RV-9). **Fix:** P1-09 names R3 and the contains-join (or records "not fixed, because") and
states the dev ceiling it can reach; P1-34 owns the `s28_run.py` re-point (or a `--extract` flag) with a test that
the run used the P1-09 script (sha256 recorded in the output); D7 waits on it.

**RV-2 (blocker, verified) — No step runs the shell, config and Markdown extractors in a build.**
P1-03 says shell/config/Markdown units are "handled by the non-Joern extractors"; P1-10, 11, 12 each write bundles
through `write_bundle`. P1-20 (1) builds a unit as "extract → normalize → bundle (P1-04/05)" only; P1-15 (SCIP)
merges inside the Joern normalizer. Nothing dispatches by unit kind, and no step Needs 10 or 12. Nothing in the
index Needs column depends on 10, 12, 08, 14 (they are leaves feeding only P1-38 "all"), so a topological schedule
reaches P1-33 without them, yet P1-33's rows (AC-1 shell lifting, prompt mentions, example configs, `.mcp.json`
chain) need their bundles. **Fix:** P1-20 gains the dispatch (kind → extractor, shell/config/Markdown per file,
failure isolated per unit as in its behaviour (1)) with a test per kind, and Needs 04, 05, 10, 11, 12.

### Majors

**RV-3 (major, verified) — Edits to `cpg/ccg/answer.py` and `answer-v1.schema.json` have no owning step.**
P1-00 adds `cacheBypass` to provenance, a new `algorithm.version`, hop kinds for `LAUNCHES` and prohibitions for
the rest, and says it touches `ccg/answer.py` "only through its owner (P1-21)". P1-21's file list is
`engine/*` only; P1-22 says `ENGINE_REASON_CODES` "then drops" `function-level-continuation`
(`ccg/answer.py`: `ENGINE_REASON_CODES = {"function-level-continuation": "possible"}`) but lists no `answer.py`.
`Provenance` in the schema has `additionalProperties: false` and no `cacheBypass` (properties end at `cacheKey`,
`cacheHit`), so P1-26's bypass provenance cannot validate. **Fix:** name the editing step for `answer.py`, the
schema, `reasons.yaml` and `tests/test_answer.py` (failing validator test first, as P0 did) and make P1-21, 22 and
26 Need it.

**RV-4 (major, verified by reading all four sub-plans) — The Needs column omits dependencies the step text uses.**
P1-25 behaviour (3) calls P1-20 `refresh`; Needs has 23, 24, 19 only. P1-17 (4) compares on `bindParts` (P1-14) and
(9) follows the `-p1-graph.md` O-9 ruling; Needs has 16, 11, 07 and the notes table blocks only 18, 19, 21. P1-32b
"builds at `EVAL_COMMIT` through P1-20", P1-33 "built through P1-20": neither Needs 20. P1-33 needs 08 (SkippedFile
rows), 10, 12, 14 (RV-2). P1-37's prerequisite order (pre-registration record, then build) is carried by prose only.
**Fix:** recompute Needs from the behaviours (or add a one-line "reads output of" per step) and re-check the
critical path; no cycle was found (I walked every Needs entry; 20 and 14 attach below 19 and 11).

**RV-5 (major, verified) — No pack recognizes the shell `mcp-call` consumer, though LNK-19 and LNK-20 gate at P1.**
`docs/test-plans/connected-code-graph-fixtures.md` LNK-19 (`mcp-call reindex`, confirmed `key-match`) and LNK-20
(`eval "mcp-call $TOOL"`, `possible/shell-dynamic`) are P1 rows; `cpg/fixtures/system1/fx-svc/scripts/refresh.sh`
contains `mcp-call reindex`. `grep -rn 'mcp-call' cpg/packs cpg/ccg` returns nothing; the `fastmcp` `tool-call`
rule matches `callee: call_tool` under `scope.imports: mcp`, which a shell script never satisfies. P1-16 tests
only the three existing packs; no step authors or extends a pack. **Fix:** add a pack-authoring unit (a `sh`
consumer rule, plus its `negatives.yaml` per `ccg.rulepack.load_pack`) before P1-17, with its rows named; decide
whether the Compose and `.mcp.json` launch rows need pack entries or are linker code, and say which.

**RV-6 (major, verified for the facts, inferred for the consequence) — System scope contradicts the census scope the keys were written under.**
P1-03: `cpg/ccg/system.yaml` excludes `deprecated/` and "**does not copy** `cpg/`'s excluded paths ... (evaluation
scope, not system scope)". `cpg/eval/eval-manifest.yaml` `exclusions.paths` lists `cpg/`, `deprecated/`,
`docs/**/connected-code-graph*`, `docs/BACKLOG.md`, and its header says "Census scope = evaluated scope". With `cpg/`
in system scope the proving-repo graph gains `cpg/`'s Python and the fixture units (`find cpg/fixtures/system1 -name
pyproject.toml -o -name package.json` gives four `pyproject.toml` under `fx-svc` and one `package.json` under `fx-web`), duplicate `mcp_tool` providers (LNK-20 says an
`unknown-key` consumer links to *every* provider of the kind), and files the kit deliberately deleted. Items outside
the key then land in the noise and wrong-possible measures of P1-36/P1-37. **Fix:** decide it in the plan: either
copy the exclusions into `system.yaml` for the proving-repo system (the gate then measures what the key assumed), or
state that out-of-scope items are filtered before scoring and say where; add the choice to N-ML question 5.

**RV-7 (major, verified) — P1-01's done-conditions contradict the tree.**
(1) Behaviour (1) requires `import extractors, clients` to resolve, but `cpg/` has neither directory and P1-01
creates none (`packages.find` needs `__init__.py`). (2) `cpg/infra/ccg-falkordb.sh status` marks drift when
`TIMEOUT_MAX != 0` or any ACL user besides `default` exists ("§5.1 exit hygiene"), and `infra/tests/test-ccg-falkordb.sh`
pins it (`st_case "TIMEOUT_MAX left"`). P1-01 sets `TIMEOUT_MAX` 10000 and P1-19 creates the four ACL users on this
instance, so `status` exits non-zero permanently; neither step lists `status` or its test. The script header and the
`reset-p0` verb are also P0-scoped, and the instance has no persistence (`save ""`, no volume): a restart empties the
store, so say that the first query after a restart is a rebuild. (3) The `slice` marker the equivalence tests use is
registered nowhere (only `live` is; D8 gives `pyproject.toml` to P1-01). **Fix:** P1-01 creates the two package
stubs, lists the `status`/test edit (hygiene becomes "the P1 steady state"), and registers `slice`.

**RV-8 (major, verified by reading P1-28/29/30, index §6, `-egress-spec.md` §3.3) — The Q1 split is not as clean as §6 claims, and the default is self-contradictory.**
(a) Index §6 says P1-29 builds `R0, R1, R3-R5`; P1-29's own text builds the rule table `R0-R7`, `R99`, first match
wins. The spec orders `R2-R4` before `R5/R6` so local-only can use them, so `policy.py` must decide `R2`'s
admission ahead of Q1. (b) P1-28 is gated (`analyst`, `security-expert`) without the `R2` section, P1-29 and P1-31
then execute against it, and the `R2` section is "amended in place before P1-30": that is an approved, executed-against
document edited in place (root `AGENTS.md` collision rule 5). (c) "A shipped as C" is a fourth state: P1-30 is built
(A) and never enabled (C); gates says "under C this step is not built". Its done-condition calibrates
`gitFetchMaxBytesOut` from "a real fetch ... on the proving repo" while its test "never touches a real host", and
`git remote -v` shows this repo's origin is an SSH-form URL, which a `CONNECT https://host:443` broker cannot
carry. **Fix:** take C for P1 outright and move P1-30 to the first phase where a manifest names an https remote
(P1-29 keeps `R2` as a deny-with-reason row); or keep A and state calibration against a local fake remote with
the margin set as policy. Put the `R2` text in a separate `-egress-spec4.md` (or write P1-28 after Q1) rather than
amending in place. The stakeholder question (Q-A) still stands; this removes the build ambiguity, not the decision.

**RV-9 (major, verified) — Equivalence tests, D7's deletion precondition and the slice's dependents do not fit together.**
(a) P1-05 requires byte-for-byte equality with the stored `s14a` bundles; P1-06 (masks `Cond.code`), P1-08
(`file` records for `i18n`), P1-09 (adds `data` records) then deliberately change that output, and D7 deletes the
slice "once each ported module's equivalence test ... passes" at P1-38, by which time several cannot. Pin each
equivalence test to a named flag set or a delta whitelist, and reword D7 to "green at the step that ported it,
sha256 recorded". (b) The tests are `slice`-marked and "skip when the artifact is absent"; the artifact is in
gitignored `cpg/.ccg-work/`, so on any other checkout they skip green. Only P1-16's gate says "execute, do not
read". Every equivalence gate must record `0 skipped`. (c) Deleting `cpg/spikes/vertical-slice/` breaks tracked
code that D7 says stays: `git grep -n vertical-slice` (unfiltered, tracked, excluding the directory and `docs/`)
hits `spikes/store/bfs.py` (`VS = HERE.parent / "vertical-slice"`), `store/synth.py` and `store/lock_replay.py`
(`sys.path.insert`), `store/tests/test_store_spikes.py` (asserts `p["file"] == "cpg/spikes/vertical-slice/impact.py"`),
`frontend-fitness/s28_run.py` (RV-1), and three `rd*.sc` header comments; the rest are `contact.json` and recorded
results. **Fix:** D7 carries a per-hit disposition (re-point, vendor the three modules into the store spike, or
retire the spike), and P1-38 runs the same sweep as its done-condition.

**RV-10 (major, inferred from `-ml.md` §4.1 and the P1 steps) — The runtime oracle has no data source at P1.**
ML §4.1 defines it as edges observed at runtime: "server access logs or HTTP-client instrumentation during the
suites and a scripted storefront session". P1-36 writes `oracle.py` "per the note", but no step runs the suites or
a session under instrumentation, and `RAN` importers are P3 (index §4.2). Running `falkor-chat`'s suites and the
storefront session writes to the shared `falkordb-dev` unless pointed elsewhere, against the process rule in §10.
**Fix:** name the capture step, its owner and the isolated instance, or report the oracle "not closed at P1, because"
in §4.1 (it is listed there as owned by P1-36/37, which is false until this is decided).

**RV-11 (major, verified for the tests as written) — Two safety checks cannot fail on the shapes that matter, and ACL-user credential custody is unspecified.**
(a) P1-02 (c) enforces "no LLM-backed tool reads a bundle" by grepping headless allow-lists for the bundle
directory. A literal grep passes on a superset glob (`Read(cpg/**)`) and says nothing about interactive agents.
Use the glob-containment logic already in `cpg/eval/tools/audit.py` (`pattern_escapes`) with a planted
`Read(cpg/**)` and `Read(cpg/.ccg-work/**)` as positive controls, and state the residual for unrestricted-`Read` agents.
(b) P1-26 (6) bans five module names by import; `__import__`, `importlib`, `subprocess curl` and `from ccg.broker...`
indirections pass. Name the mutants (`import urllib.request as u` inside a function) and cover `ccg/engine`,
`clients` and `refresh` too, since P1-25 (7) asserts no client opens a socket but names no test. (c) P1-19
`create_users(admin, system)` mints four credentials; no step says where passwords live (mode, path) or how the
`ccg` MCP server gets `ccg_raw_<s>` without it landing in the committed repo-root `.mcp.json` that P1-27 edits.
P1-02 covers the HMAC key only. A deep pass is out of scope; `security-expert` is advisory on P1-02 already.

**RV-12 (major, verified by counting files and behaviours) — Four steps exceed one dispatch.**
P1-23 has four modules, three tests, eight behaviours and a live measurement; P1-31 builds a harness, three
modules and runs a full refresh scenario; P1-33 authors a test plan and executes 91 rows plus twelve ACs; P1-37 is a
loop (miss, fix on dev, move clusters, write a new list with two isolated writers, re-gate). The index promises "at
most about five files and three sequential sub-steps". **Fix:** split P1-23 (flow/grouping vs staleness/measurement),
P1-31 (harness vs run), P1-33 (plan vs execution) and state P1-37's re-gate as a new step with its own brief.

### Minor

**RV-13 (minor, verified) — P1-01 puts Node and the SCIP indexers into the sandbox runtime image the Dockerfile keeps free of them.**
`cpg/infra/toolchain/Dockerfile`: stage `toolchain` is "the image the sandbox runs ... no Node, no indexers"; stage
`truth` already carries Node 24.22.2 and the pinned scip tools. Say why `truth` (selected by `CCG_TOOLCHAIN_IMAGE`)
is not reused, since a Node image in the sandbox widens the AC-8 "no dependency download" surface.

**RV-14 (minor) — Index inconsistencies.** §3.1 W0 says "three co-owned notes", §7 says "two notes"; §4.1's row for
P1-32 lists "§11 noise sampling" but the gates plan puts it in P1-36; §2's claim that the only `P1-` strings are
"test-plan and review ids" is wrong in detail (none in `docs/test-plans/`; one-digit review-finding ids `P1-1..4` in
`falkor-chat/docs/reviews/mcp-input-bounds.md` and commit `62b75cc0`, which do not collide with `P1-NN` but match a
`P1-1` substring grep); `ccg/staleness.py` (P1-20) and `ccg/engine/staleness.py` (P1-23) are two modules for one
concept (have the engine import the first); a module named `ccg/hmac.py` invites shadowing stdlib `hmac` when a
script runs with `cpg/ccg` on `sys.path` (use `ccg/valuehash.py`); `extractors` and `clients` become generic
top-level package names.

**RV-15 (minor, verified) — P1-01's O3 test names a fixture that is not there.** It says "test with the stored breach2
stream (`cpg/tests/fixtures/headless`)". That directory holds `capture.*`, `derived.*` and `twoattempt-*` streams; no
file under `cpg/` or any plan contains the string `No such tool` except the reports. Plant a derived
`absent-tool` stream (as `derived.mcp-call.*` was) and name it.

**RV-16 (minor, verified) — P1-35's unverified "24" is derivable.** Report §AC-24: 36 unambiguous and 8 ambiguity items
in total; `cpg/eval/dev/ac24.jsonl` has 12 `unambiguous` and 2 `ambiguity` (counted); so held-out is 24 and 6. The
plan may state it and drop "I did not verify the 24"; the harness still re-counts from the split output.

**RV-17 (minor) — Smaller seams.** (a) `-graph.md` is edited in place (P1-19, N-GRAPH q6) although it is `Extended by:`
`-p0-graph.md` and was executed against; put the corrected `ccg_build` line in `-p1-graph.md`. (b) P1-16's "24 leaves
exercised" is correct (the set in `tests/test_rule_pack_leaf_mutation.py` has 24) but the step lists no edit to that
file; the done-condition is the set becoming empty and `test_each_leaf_behaves_as_classified` passing. (c) Bundle-spec3
F-13 (http `keyNorm` step order; "none may encode the old order") is absent from the walk although P1-17's
`matchers.py` and P1-21's anchors are new consumers; add "uses `ccg.keynorm` only". (d) Parallel live steps (P1-19,
P1-21, P1-25, P1-31, P1-33) share one `ccg-falkordb` with fixed key names and `ACL`/`GRAPH.CONFIG` writes; carry P0's
§5.1 reservation rule or give each a throwaway instance. (e) Q-B: under a false flag the control-dependency effect
(guarded code in the same function) is not a distinct item in a symbol-level, call-level walk; say what the planted
fixture lists, or Q-B's default reading is not testable. (f) N-GRAPH q1 lists `Dependency (with scope)` while P1-00
forbids `DEPENDS_ON`/`FROM_DEP` hops until P3.

### The flow ceiling and the 0.93 guard (brief item 3)

*Ceiling:* re-computed from the report's tables, correct (above). *Reachability:* see RV-1: not reachable as
specified; reachable only with R3 plus the contains-join and essentially no misses left (149 of 151). *The 0.93
guard, judged:* it is a prior with no derivation, and the plan's own source gives the cost of being wrong. ML §8.4's
operating curve (`docs/plans/connected-code-graph-p0-ml.md`, n = 120): P(pass) at true recall 0.95, 0.93, 0.90 is
0.61, 0.26, 0.04 and the table rows (m = 5-8, independent and clustered) give 0.52-0.59, 0.14-0.31 and 0.00-0.08. A dev recall of 0.93 therefore licenses a fresh blind draw whose pass probability is about 0.14 to 0.31,
and dev recall is an optimistic estimate (the 30 methods are contact, and the fix is fitted to their misses). With a
hard ceiling of 0.954 the honest entry value is the dev figure at which P(pass) is acceptable (about 0.96, which the
dev set cannot exceed) so the draw should be expected not to run. Recommend: keep P1-09 (R1 is verified to lose no
pair), state in §5 that python's flag is expected to stay false, make P1-34 conditional on a dev figure derived from
the curve by the method owner, and stop budgeting the fresh draw in the critical path. TS/JS: agreed (0.546 ceiling).

### Egress Q1 (brief item 4)

Options are fairly stated; the recommendation (A shipped as C) is sound in spirit and RV-8 trims it to a build
decision. Steps that truly do not depend on Q1: P1-26's model path, P1-31 except the 8.2 `R2` clause and the
`git-fetch` `toolArgv` check, sandboxed offline builds, P1-02. Steps claimed independent that are not: P1-28 (its
`R2` section) and P1-29 (the `R2` row of the policy table).

### What's solid (RV-0)

- **Obligation walk (item 1), re-derived independently from `-p0-2.md` §15 and the P0 disclosures:** every named
  handoff, F-1/6/7/9/10/12, C1-C9, O-3/5/9/10/12/13/15, the deferred measurements and all thirteen P0 disclosures
  have an owner or a justified "reported not closed". Gaps found are only RV-10 (oracle source), RV-17(c) (F-13) and
  the omitted-but-later `security-expert` HMAC/canary review before P5 (not P1; one line in §4.2 would close it).
- **Counts verified:** 39 steps P1-00..38 (5+12+5+7+10 rows); 91 P1 rows of 132 (fixtures plan §6);
  held-out counting items 26+46+79+8 = 159 with flow 0 of 99 (`cpg/eval/heldout-manifest.json`); the stored `s14a`
  bundles (2,775 and 85,071 records) pass `ccg.bundle.validate`; slice line counts (403, 1,244, 657, 675, 422) and the
  `cpg/ccg/` module list match the tree; every symbol the plan imports exists (`mask_http_segments`, `screen_fails`,
  `name_fails`, `gate_argv`, `mask_identifier`, `fence_data`, `cache_key`, `reason_row`, `load_pack`, `link_id`,
  `h1_symbol_credit`, `E`, `renormalize_check`); the reason codes the plan uses are in `reasons.yaml` except the
  new `stale-kept` (a graph status, not a reason).
- **Step ids:** no `P1-NN` plan step in `git log --all` or the ledger (ledger uses `S01-S59`, `P0-NN`, `U33`).
- **Design strengths:** D2 (flag as data covered by `coreTreeHash`), D5 (prompts as `.txt`, checked against EP §12.3's
  actual exclusion `**/*.md`), F-9 and F-1 before the first non-proving build, equivalence-before-improvement, the
  mutation check on every contract boundary, the "I did not verify" honesty, builder isolation in every brief.
- **Checks with named mutants and positive controls:** P1-09 (4) (no pair lost), P1-19 (2) (old flat ACL pattern as
  the negative control), P1-22 (`<` vs `<=`), P1-31 (injected unlogged connection), P1-32 (fabricated chain).

### Open questions

1. (stakeholder, RV-6) Is `cpg/` in the proving repo's system scope, or excluded as in the census?
2. (stakeholder, via `tico`) Q-A: with RV-8, take C outright for P1 and defer P1-30?
3. (method owner) The dev-set entry value for P1-34, from the §8.4 curve; whether `definition`-side misses (python 9)
   could be fixed by projecting the transitive CDG, which raises the ceiling to 0.973 but changes what a flow pair
   means (§7.3 requires a direct edge). Not taken by me; it is the method owner's rule.
4. (architect) Are Compose build-context and `.mcp.json` launch rows pack entries or linker code (RV-5)?

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: RV-9(b) (equivalence tests that skip when a gitignored oracle is
absent), recurring class with the leaf-harness "unexercised" set in `cpg/tests/test_rule_pack_leaf_mutation.py`.
Addition: "A test that skips when its oracle artifact is absent (a gitignored work area, an external file) is
reported as unrun, not green; a gate that relies on it records the non-skipped run (`0 skipped`)." No existing line
is enforced elsewhere, so no prune.

## Pass 2

Re-gate of the revised set (index plus four sub-plans, 47 active steps P1-00..P1-47, P1-30 retired) against Pass 1
and the two stakeholder answers (egress Q1 = C outright; `cpg/` excluded from scope). Executed against the tree and
the files, not the architect's report. CPG: considered, not relevant (same reason as Pass 1). No DB writes, no
mutating git.

**Verdict: approve with suggestions.** Both Pass 1 blockers are fixed and verified. Two new majors (NF-1, NF-2) and
one new major seam (NF-3) should be folded in before dispatch; none needs a redesign. The rest is minor.

### Disposition of Pass 1

- **RV-1 fixed.** Re-derived from report §a2.2.2: 308+67+27+36 = 438/481 = 0.911; +R3 17 = 455/481 = 0.946; need 457/481
  (0.950; 0.95 x 481 = 456.95); ceiling 459/481 = 0.954; transitive-CDG 468/481 = 0.973. All match the index §5. P1-09
  now ports R3 as a projection rule with a golden and both join fixes. `0.93` appears only as a column of the §8.4
  operating-curve citation (`grep` over all five files); the entry value is the method owner's (N-ML q2) and "not
  run" is a stated outcome. P1-34 step 1 adds `--extract` and a sha256 test; `s28_run.py` line 24 and 73 are as it
  describes.
- **RV-2 fixed.** P1-44 (`build.py`, one dispatch point, test per kind that the extractor is invoked); P1-20 Needs it; D11.
- **RV-3 fixed.** P1-45 owns `answer.py`, the schema, `reasons.yaml`, `test_answer.py`; P1-22 owns the
  `ENGINE_REASON_CODES` removal; P1-21/24/26 Need 45.
- **RV-4 fixed with residue** (NF-4). Needs rewritten; I parsed the 48 index rows: acyclic, every id has exactly one
  sub-plan header and every `P1-nn` reference resolves; Part column matches the file.
- **RV-5 fixed with residue** (NF-3). P1-46 plus D10 (Compose and `.mcp.json` chain are linker code; consistent with
  P1-11 and P1-17).
- **RV-6 fixed.** Index §2 and P1-03 take the census scope; P1-03 asserts its lists equal `eval-manifest.yaml`
  `exclusions` (`paths` and `contentCheck.files`); P1-32 flags out-of-scope items and never scores them.
- **RV-7 fixed.** P1-01 creates the two stubs, changes `status` and its test, registers `slice`, notes no persistence.
- **RV-8 fixed.** Option C throughout; P1-30 a stub; P1-29 `R2/R8/R9` deny rows; `R2` text no longer amended in place;
  `remote` rejected by the manifest loader (P1-03 (6b)); AC-8 detector design in P1-31/P1-40. Residue in NF-5.
- **RV-9 (a), (b) fixed; (c) decided differently, accepted.** `slice` profile and D9 (`0 skipped`, sha256) fix (a),
  (b). Keeping `cpg/spikes/vertical-slice/` frozen is sound for the reasons I verified (`spikes/store/{bfs,synth,
  lock_replay}.py`, `store/tests/test_store_spikes.py`, `s28_run.py`). **The `contact.json` reason is refuted:**
  `tests/test_contact.py::test_contact_json_matches_rederivation_at_its_derivedfrom_tree` calls
  `make_contact.check`, which re-derives from the immutable tree object named by `derivedFromTree` (`git show <tree>:<path>`),
  never from HEAD, so deleting the directory later cannot change a pinned tree. Delete that clause from D7 and from
  the §11 backlog item ("handle `contact.json`'s pinned tree hash"); the decision stands on the other reasons.
- **RV-10 fixed with a safety residue** (NF-2). P1-47 is now the capture step; P1-36 Needs it.
- **RV-11 fixed.** (a) uses `pattern_escapes` with planted controls (see NF-7 for the root); (b) `test_no_network.py`
  is an AST test with named mutants over engine, render, refresh, build, clients, front; (c) ACL credentials in
  `cpg/.ccg-work/keys/<system>.acl`, `.mcp.json` carries none.
- **RV-12 fixed.** P1-23/39, P1-31/40, P1-33/41/42, P1-43 split; P1-01 remains oversized (NF-7).
- **RV-13 fixed** (`indexer` target `FROM toolchain`; the Dockerfile's `toolchain` stage stays free of Node).
- **RV-14 fixed except one accepted nit.** Notes count and noise owner corrected, one `staleness.py`, `valuehash.py`,
  `P1-` id claim corrected. `extractors`/`clients` kept: proposal §3.13 line 989-990 fixes the directories
  (`cpg/extractors/<name>/`, `cpg/clients/{mcp,cli}/`), and the generic top-level names are a consequence of
  `packages.find`; not fixed, accepted (nit).
- **RV-15, RV-16 fixed** (derived `absent-tool` stream; 24 and 6 stated with the re-count kept).
- **RV-17 fixed** (a) graph note not edited, (b) P1-16/P1-46 leaf set, (c) F-13 row and P1-17 (12), (d) §8 rule
  (but see NF-7), (e) §5 AC-25 fixture (but see NF-7), (f) "nodes only".

### New findings

**NF-1 (major, verified) - No core-freeze point before the pre-registration record.**
`coreTreeHash` covers every tracked file under `cpg/` except `cpg/eval/**`, `cpg/packs/**`, `cpg/tests/**` and
`**/*.md` (EP §12.3, read at `-eval-protocol-ml.md` "coreTreeHash"); the first-contact rule needs all four fields
equal and no uncommitted change under `cpg/`. P1-37 requires the record "before the build", but its Needs closure
(computed from the index) omits **P1-15** (indexer adoption: `extractors/indexers/*`, `merge_scip.py`, which change
resolved calls) and **P1-31** (`cpg/sandbox/ac8/*`); P1-27, 35, 40, 43 are outside `cpg/` core or excluded. An edit
to either after the record makes the first run "after adaptation" (the AC-5 result is lost). `cpg/spikes/**` is in
the hash too, which is why P1-34's `s28_run.py` edit must precede it (it does, P1-37 Needs 34). **Fix:** one rule in
index §7: the `ac5-gate-1` record Needs every step that edits non-excluded `cpg/` files (name 15 and 31), and the
record's commit is the freeze; P1-15 either lands before it or is declared out of the first gate.

**NF-2 (major, verified default, safety) - P1-47's fence checks the wrong thing.** The check "refuses a connection
string naming port 6379" does not see an *implicit* default: `falkor-chat/scripts/start_server.sh`, `test_queries.sh`
(lines 12-13), `seed_*.sh`, `verify_catalog.sh` all use `${FALKORDB_PORT:-6379}`. A run of `falkor-chat`'s suites or
server with the variable unset reaches the shared `falkordb-dev`, and those suites wipe state; this is the S50 slip
shape. **Fix:** the capture wrapper exports `FALKORDB_HOST` and `FALKORDB_PORT` explicitly, then asserts the
*effective* port equals the throwaway's own published port and is neither 6379 nor 6390, and that the target container
carries the `ccg-falkordb=1` label (the ownership invariant `ccg-falkordb.sh` already uses). Test: unset
`FALKORDB_PORT` must refuse (the named mutant), and `--throwaway` itself refuses ports 6379 and 6390 (add to P1-01
behaviours, which today list neither). Otherwise no step permits a write to 6379 (`grep 6379|falkordb-dev` over the
set: only prohibitions).

**NF-3 (major, verified) - "Flip the packs to live" and the new pack need a format change nobody owns.**
`cpg/ccg/schemas/rule-pack-v1.schema.json` has `"status": {"enum": ["paper"]}`; the three packs carry `status:
paper`; `tests/test_rule_pack_schema.py` pins them to the spec's blocks. P1-16 ("flips their status to live") and
P1-46 (new pack) list none of the schema, `ccg/rulepack.py` or that pin test as files, and the rule-pack format is
frozen v1.5. **Fix:** either keep `paper` as the only value and drop "flip to live", or give P1-16 the schema/validator/
pin edit (failing test first) and a one-line spec successor note from `architect`.

**NF-4 (minor, verified by closure computation) - Residual Needs gaps.** P1-39 imports `ccg.staleness` (P1-20) and
runs a live measurement on the built proving graph: Needs lacks 20. P1-26 returns answers whose provenance is built
by P1-39 `answer_build.py`: Needs lacks 39 (inferred). P1-47 writes under a mode-0700 work dir (P1-02): Needs lacks 02.
P1-44 merges SCIP "when adopted" (P1-15): state the conditional dependency. Critical-path line lists 19 -> 44 -> 20,
but 44 does not Need 19 (19 and 44 are parallel; the edge is 19 -> 20).

**NF-5 (minor, verified) - Stale and contradictory text left after the revision.** Index §4.2 row for `ownedRegions`
and multi-candidate fixtures still cites **P1-33** (now the test plan; the gate is P1-41). P1-03's interface says
"`remote` optional" two lines before rejecting it. The four sub-plan intros still say "decisions D1-D8" (index has
D1-D11). Index §11 says no proposal edit is planned, but the proposal's P1 row lists AC-8 part 1 as "git-remote fetch
only" and requirement AC-8 says it is "closed only when all of it is built": with git-fetch deferred to an unscheduled
trigger, AC-8 cannot close at P2 exit as the proposal states. Add the pointer and a line to N-TICO asking the
stakeholder how AC-8 closes without the fetch path. Tombstone P1-30 (index row, gates section) is history; one line
in §6 suffices.

**NF-6 (minor, verified) - AC-2 and AC-4 have no gate step.** P1-38's table says "AC-1, AC-2, AC-4 closed"; the old
fixture step named them, P1-41/P1-42 do not (`grep` over the gates plan: AC-2 and AC-4 absent; AC-1 only as a lifting
chain). Name the proving-repo asks (a `falkor-chat` MCP tool's callers and wiring; a `cypher-mcp` tool's scripts)
in P1-42 or point to the P1-37 strata as the evidence.

**NF-7 (minor) - Smaller.** (a) P1-02 (c): `pattern_escapes(root, base_dir, pat)` answers "can the glob name
something outside `root`", so the test must name the root (the headless kit dir); with it, both planted controls are
flagged and any other outside pattern is too (fail closed). (b) P1-19's ACL test runs "on `ccg-falkordb`" while index
§8 lists P1-19 among throwaway-instance steps; the test mutates ACL users and `status` treats a stray user as drift:
run it on a throwaway. (c) P1-01 is about ten files across three areas, over the dispatch bound it states. (d) G-4
(CDG self-edge dropped) is P1-07's, but P1-07's writer rules serve P1-10/11/12; no step applies it in the Joern
normalizer (P1-05/09). (e) Q-B's control-dependency fixture puts guarded code in a different function "reached by a
call from the guarded branch": name the answer-spec row that lists that callee in an impact answer (a callee is a
depends-on item, not an impact item, in a caller-direction walk). (f) `CCG_INDEXER_IMAGE`: `offline-run.sh` reads only
`CCG_TOOLCHAIN_IMAGE` (its header), so `indexers/run.py` must set that variable per call.

### Cross-checks requested

- **Acyclicity and ids (item 2):** verified by script over the index (no cycle; 48 rows; ids, parts and `Needs:`
  lines in the sub-plans agree; no reference to a missing id; every `P1-30` mention is the retirement).
- **Retired behaviour sweep:** `R2` (egress) occurs only as deny rows and "not implemented"; `gitFetch`: none;
  `proxy`: only the detector and the deferral; `0.93`: only the operating curve; deletion: only the D7 disposition
  text; stale hits are NF-5.
- **AC-8 part 1 under C:** satisfiable as the index says (no allowed proxied stream, any `conn` fails, rows 8.2/8.5
  proxied/8.12-8.13 reported vacuous), but see NF-5 for AC-8's closure rule.
- **Pass 1 cleared items re-audited (item 3):** the obligation walk survives the renumbering. Re-mapped owners all
  exist (`P1-39` superset, `P1-41` F-9/F-1 prerequisites, `P1-47` oracle, `P1-45` C-items). One orphan, NF-5's
  P1-33 citation, and NF-6's AC-2/AC-4. Step-id collisions: none for `P1-NN` (re-run `git log --all`, ledger,
  `docs/`; one-digit `P1-1..4` review ids only).
- **Growth-stops (item 4):** 17,074 words in Pass 1 became 19,414 (+2,340). Classified: new scope is P1-44..47, the
  splits (39, 41, 42, 43), D9-D11, the P1-46 pack and the AC-25/scope/egress text; rework is the rewritten Needs
  column and §5, which replaced rather than appended (no finding-to-resolution table, no `RV-n` citations in the plans).
  Findings to report: the P1-30 tombstone and N-ML's "replaces the earlier 0.93 prior" are history, not design
  (one clause each to cut); the D7 "Why" restates five dependents and one is false (above).
- **Safety (item 5):** see NF-2; the throwaway rule for parallel live steps is otherwise sound.

### Proposed REVIEW.md lines

None new. Pass 1's proposal (skipped-oracle tests count as unrun) is now plan rule D9; the one addition worth
making is the same one, unchanged.
