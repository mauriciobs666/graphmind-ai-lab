# Review: the CO-6b pre-registration record (`ccg-prereg-v1` / `ac5-gate-1`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1 close-out, CO-6b)

**Scope.** The uncommitted last line of `cpg/eval/preregistration.jsonl` (id `ccg-prereg-v1`, stratum `ac5-gate-1`) and its note `docs/plans/connected-code-graph-p1-closeout-prereg.md`, against the committed tree (HEAD `273f9858`), `docs/plans/connected-code-graph-p1-closeout.md` (CO-6b row, §3.5, §7, §9), `docs/plans/connected-code-graph-p1-gate-marker-ml.md` (last section), the ledger `docs/plans/connected-code-graph-coordination.md` and the decision log `docs/requirements/connected-code-graph.md`. No held-out file, no `~/.local/share/ccg-eval/`, no `freeze-check`, no `gate.py run`. Dev gold was read (builder-visible). Not run: the full suite, mutants on `gate.py`/`withheld_shapes.py`.

**Verdict: approve with suggestions.** Every pin recomputes, every verbatim string diffs clean, the bound and table recompute. No blocker. Three majors (M1-M3) concern what the record lets a reader believe or a gate skip; the record is uncommitted, so amending it costs nothing before R and a new F after. Fix M1-M3 before R.

CPG: considered, not relevant — a pre-registration record checked by recomputation, diffing and a read of the harness's record binding, not by a code-graph question.

**Stopping condition (falsifiable).** This review closes the record when, at the record commit R: (a) `git diff db7f7f45 R -- cpg` lists only `cpg/eval/preregistration.jsonl`; (b) the Appendix A script prints `True` for every row against `git show R:...`; (c) the record names the no-workaround scan and a re-run reproduces its stated counts (M1); (d) M2 and M3 are either text in the record or a recorded stakeholder answer. It is wrong if any (a)/(b) value differs at R, or if a `gate_run run` dry refusal list shows a freeze-check or noise-bar check I did not find (M3: none exists in `gate.py`/`gate_run.py`).

## Findings

### M1 (major) The no-workaround scan is not reproducible from the record, and its stated count does not match its stated definition
Criterion item 2 says "the scan and its result go in the record". The record has the result (473 literals, 136 files, 3 hits) and no scan: no script, path or sha. I re-ran with several definitions (Appendix B):

- `473` is **not** reproduced by "ids, qnames, paths, contract keys": that gives 341 over http+mcp+scripts, 489 over all five dev strata. It is reproduced exactly (473) only by the union of ids, paths, qnames, locators, **basenames and tails** over http+mcp+scripts+**wiring**. Under that wider definition the hit set is wider than three (`run.sh` in 6 files, `<script>` in 3, `AGENTS.md`).
- Product-file definitions give 136 files (the three directories, as the author), 134 (no `.md`, no tests), 321 (all non-eval/tests/spikes non-`.md` under `cpg/`). Ids: **0 hits** under every definition and under the added-lines diff since `511b14cf` (a superset of "since the dev gold was first run"). Qnames: 0. Strict path literals: 2 (gated strata) to 3 (with wiring): `.mcp.json`, `AGENTS.md`, `test-stamp-wiring.sh`. The author's three are exactly mine. `.mcp.json` comes from the withheld wiring stratum only.
- Spread of extra "hits" with looser literals: `mcp_tool` (a contract-kind name, 8 files), `run.sh`, `<script>`/`<module>`/`<document>` (extractor symbol names), `ensure-environment.sh` (a substring of `test-ensure-environment.sh` in the shell README, a false positive). A non-literal check (63 path segments of 5+ chars from the gated dev paths, whole-token, 134 code files) hits 12 generic words (`server`, `scripts`, `hooks`, `pipeline` = a tree-sitter node type, `claude` in a comment about `.claude/`) and `AGENTS`; none is dev-specific.

**Rulings on the three hits: none is a workaround.** (1) `.mcp.json`: the MCP config file-type name in `cpg/ccg/linker/bind.py`, `cpg/ccg/units.py`, `cpg/extractors/config/{formats,launch}.py`, `common/writer_rules.py`, since `acd3be0a` (2026-11-11); it matches `.mcp.<x>.json` generally; its only dev item is wiring, which is withheld. (2) `AGENTS.md`: the glob `**/AGENTS.md` in `cpg/ccg/system.yaml`, a declared copy of `eval-manifest.yaml`'s census scope with a test asserting equality; unchanged since 2026-11-19 (`8377c326`), no extractor code names it. (3) `test-stamp-wiring.sh`: `cpg/extractors/shell/README.md` lines 83-85, prose listing files tree-sitter rejects; no code.

Two defects remain. The criterion's letter is "finds no ... literal", and the README hit is one; the record adjudicates instead of stating a reading. The scan is a literal bound: it cannot see a shape-specific rule with no literal, which is what the §7 synthetic-tree tests are for (see below).
**Fix before R:** commit the scan (a script under `cpg/eval/` or `docs/`), pin its sha in the record, state the literal kinds and strata it ran over, and state the reading ("hits in prose are disclosed, not excused silently") so the 473 can be re-derived.

**The §7 synthetic-tree tests exist, run and can fail.** `tests/test_script_resolver.py` (45 tests, tmp-path trees), `test_co3_curl_end_to_end.py`, `test_pack_shell_http_client.py`, `test_pack_shell_mcp_call.py`, `test_shell_extractor.py`: 268 passed from `cpg/` with its venv. Mutants on a throwaway copy of HEAD (`ccg/assemble/rows.py`, restored and diffed): own-directory tier off, own-directory tier rooted at the repo, own-directory status `None`, one-segment suffix tier on, ambiguous suffix picked, tier 1 off. All six killed (first failure each time). The real tree was not touched (`git status` unchanged).

### M2 (major) Held-out items of the unbuilt shapes are asked and scored; the record's pass-probability table and the decision log read as if they were withheld
Evidence: `cpg/eval/tools/gate_run.py:623-624,676` passes `records` (after `withhold_strata`, which removes only strata) to `run_asks`; the predicate runs there only as `withheld_counts` (counts) and `check_withheld_cap`; `gate.py:1127` prints the counts beside the hits; `gate.py:1449` is the only place the predicate excuses an item, and it is the dev probe. So a held-out item of shapes B/C/I/J/K/M is asked and a miss fails the gate. That is what the verbatim statement says ("scored `not-listed`, not excused"), but it contradicts the decision-log entry "Withhold them by a build-invariant predicate" (`docs/requirements/connected-code-graph.md:797`), plan §9 ("The withheld script shapes are not a miss", `docs/plans/connected-code-graph-p1-closeout.md:143`) and the gate-marker note ("never scored"). The record follows the harness.
Consequence for the table: `r^c` (0.56 at r = 0.99) is the chance of zero misses **given no held-out item of an unbuilt shape**. On dev 10 of 49 scripts items (20%) are of those shapes, 2 of the 10 were listed only by coincidence (`docs/reviews/connected-code-graph-p1-closeout-co2.md:94`, family I). The held-out analogue count is unknown and the record rightly does not estimate it; the arithmetic is still that a mix like dev's puts about 16 of 79 scripts items there. "A first-spend miss is the likely outcome unless the true recall is above about 0.99" understates it: the shape term dominates. The cap (20 or more refuses) leaves 1-19 shape items to spend a list on.
**Fix before R:** state in the record, next to the table, that it is conditional on no unbuilt-shape item, and that any such held-out item the engine does not list is a miss. Ask the stakeholder to confirm the spend on that basis, or decide that held-out shape items are excused (a harness change: new F, new record, new rehearsal). Do not leave "withheld" in the decision log and "scored" in the harness unreconciled.

### M3 (major) "The order is forced by the harness" holds for one link only
Note §4 says the order F, R, freeze-check, rehearsal, run "is forced by the harness". What the harness enforces: the record is committed and exactly one per id (`gate.py:159-178`), its four fields equal the tree (`gate.py:183`), the record commit is an ancestor of HEAD with nothing under `cpg/` outside `cpg/eval/` and `*.md` changed since (`gate.py:1230-1245`), the dev probe carries R, the current harness shas and the gold hashes (`gate.py:1425-1437`), and the list runs once (marker). What it does not: a `freeze-check`/`check-report` run (no reference in `gate.py` or `gate_run.py`; `grep -n -i "freeze-check\|check_report"` finds none), the dev noise bars (the probe checks hits and key-errors only), the rehearsal count (`rehearsalId` was dropped; the probe is a file-level attestation, `gate.py:1466`), or "a rehearsal miss means a new F" (nothing stops asking the list after a noise miss). A freeze-check hit that moves a cluster to dev is caught only indirectly, through the gold hash.
**Fix:** in `ordering.sequence`, tag each step `harness-enforced` or `procedural`, and have the spend checklist (custodian and `teco`) require the freeze-check result, the rehearsal's noise figures against the bars and the run count before `gate_run run`. Reword note §4.

### m1 (minor) The successor-record rule is real but incomplete
`--gate-id` is required (`gate.py:1577`, `gate_run.py:793`) and the match is exact equality on `stratum` (`gate.py:171`), so `ac5-gate-1-r<k>` cannot collide. But the harness reads `kind` from the committed record (`gate.py:1265-1276`): a successor written as `kind: gate` runs as a first gate (floors 75 and 8), and the Q4 re-gate (floor 35, fresh clusters) needs `kind: re-gate`, usable `movedOut` counts and `--flags` (`gate.py:1516`). Add that to `q4.onFirstSpendMiss`.

### m2 (minor) Item 6 is honest, but its dependencies are not stated
The record does not claim item 6 met. Two gaps: the note says the stakeholder's decision of 2026-11-25 "forbids" the held-out read before freeze; ledger 1238 records only that the stakeholder chose to skip the `qa-engineer` counts read, so the permission to run `freeze-check` at R is open (the note's §6 asks it; the record's ordering step 3 presumes it). And a contact hit moves a cluster to dev after R, which changes the held-out counts (26/46/79, the cap's denominator), the dev gold and so the probe's gold hash. State that `strata` counts are as of R and that the run report carries the final ones.

### m3 (minor) A reading still open to stakeholder objection is pre-registered as settled
`scriptsReporting.hit` ("a distance-2 hit is a hit") is the recall ruling that `docs/requirements/connected-code-graph.md:804` records as "proceeding on this reading, awaiting stakeholder objection". Get the answer before R or mark it in the record.

### m4 (minor) `freeze.moves` is incomplete
A `cpg/tests/**` edit after R does not move either hash but `run_context` refuses it (it allows only `cpg/eval/` and `*.md`; `gate.py:1241-1243`). Add it. Also the `withheld` block gives no dev count for flow (28 dev items, 1 counting) though the CO-6b row asks for counts of the withheld set.

### n1 (nit) `ccg-noise-ratify-v1` — a missing file is not a defect
It is the stakeholder's requirements-log entry of 2026-11-10 (`docs/requirements/connected-code-graph.md:774`; `docs/plans/connected-code-graph-p1.md:426` says so), append-only, so a sha pin would add nothing. The bars the record's reading 3 lists (Q <= 4, possible precision >= 0.20, p90 <= 100, reduction >= 5x) match `-p1-ml.md` §5.3. Cite the log entry, not only the ML note. `s1Disclosure`'s "the record's priorPrints.notCovered" means the decision record `ccg-noise-s1-v1`; say so. The S1 disclosure ("no reduction printed") agrees with what is committed: `cpg/eval/results/co2/summary.json:108` states none was computed and no `reduction` figure appears in committed results; the scratch rehearsal logs stay uncovered, as the record says.

## What's solid (each checked)
- **Pins.** The 12 harness sha256, read from the working tree and from `HEAD:` (blobs): all equal the record. The key-error record sha (`456173…a852`) matches and the file is tracked. The three decision shas match. My import closure from `gate.py`/`gate_run.py` (11 modules plus `EVAL_COMMIT`) equals `gate.HARNESS_FILES` as a set. `coreTreeHash` and `packSetHash`: my own script over `git ls-tree`/`git show` blobs (not the working tree, not `observe_tree`) agrees at F, at HEAD and under `ensure_ascii` true and false (787 core files, 16 pack files). `git diff db7f7f45 HEAD -- cpg` is empty; F is a docs-only commit whose core tree is that of `952468dd`/`60c0cb27`. `prereg_comparison` accepts the record against my values and names `packSetHash` for a planted flip. `algorithm.version` "1.2", depth bound 2: `bfs.algorithm_version()` run in `cpg/.venv`. No ignored importable file under `cpg/ccg`, `cpg/clients`, `cpg/eval/tools`, `cpg/packs`; the only dirty path under `cpg/` is the record file.
- **Plan conformance.** The withheld-shape statement is byte-identical to §3.5 (diffed); the S1 wording and all six entry-criterion items are byte-identical to the gate-marker note; the claim sentence is. Floors 75/8, cap 0.25 (refuses at 20 or more; `gate.py:1301,1335`), gated and withheld counts (http 26/46/79, wiring 8, flow 99; dev 18/20/49, wiring 11) match the plan and the dev files. The predicate run on dev (pure call, no `run`) gives 10 withheld (B2 C2 I2 J2 K1 M1), disjoint from the three key-error ids, so 49 - 10 - 3 = 36 holds. The record keeps `minimumsNotMet` and the Q4 and S1 decisions as the ledger and log state them, and the cluster bound and table follow the user's 2026-11-25 decision (formula, c = 50 and 58).
- **Bound and table.** `c/(c+3.8416)`: 0.9286 (50), 0.9379 (58); item bounds 0.9752, 0.8713, 0.9229, 0.9536; `r^c` table recomputed: all entries agree after rounding.

## Open questions
1. May the custodian run `freeze-check` at R, and who reads its counts (m2)?
2. Confirm the spend knowing held-out unbuilt-shape items are scored (M2), or change the harness?
3. The stakeholder's answer on the recall ruling (m3).

## Appendix

### A. Independent hash script (core and pack set)
Definitions read from `ac24_smoke.py:574-605`: the array of `[relpath, sha256]` sorted by path, canonical JSON (sorted keys, `,`/`:`), sha256; core = tracked under `cpg/` minus top-level `eval|packs|tests` minus `*.md`, path relative to `cpg/`; packs = under `cpg/packs/`, relative to it. My script reads `git ls-tree -r -z REV -- cpg` and `git show REV:path`, never `observe_tree` or the working tree:
`h(pairs) = sha256(json.dumps(sorted(map(list,pairs)), separators=(",",":"), sort_keys=True, ensure_ascii=False))`. Result at `db7f7f45` and at HEAD: both equal the record (`d778b538…03bf`, `4149abd2…5aa6fe`).

### B. Scan spread (full tree at F, `cpg/ccg|extractors|packs` unless noted)
| Literal kind (dev strata) | http+mcp+scripts | + wiring + flow |
|---|---|---|
| item/anchor/impacted ids | 0 | 0 |
| qnames | 0 | 0 |
| full paths | 2 (`AGENTS.md`, `test-stamp-wiring.sh`) | 3 (+ `.mcp.json`, 7 file-pairs) |
| locators (`mcp_tool` etc.) | 8 files, field names | 47 files, field names |
| basenames / tails (>= 6 chars) | 7 / 16 files, generic or false-positive | not enumerated |
| product files 134 (no `.md`/tests) | paths: 1 (`AGENTS.md`) | |
| product files 321 (all non-eval/tests/spikes non-`.md`) | paths: 1 (`AGENTS.md`) | |
Ids also appear in `cpg/eval/results/**`, `cpg/eval/decisions/**`, `cpg/tests/test_ids.py` and `cpg/tests/golden/answer-ac1.json` (the last two as expected values for two dev anchors): outside the criterion's three directories, not product code.
