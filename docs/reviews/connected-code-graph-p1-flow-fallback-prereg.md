# Review: pre-registration record `ccg-p1-flow-fallback-v1`

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-39c)

## Scope & verdict

Reviewed, statically (read-only commands only): line 2 of `cpg/eval/preregistration.jsonl` (uncommitted; the file is 2 lines, `git diff` shows +1 line), against `docs/plans/connected-code-graph-p1-ml.md` §3.5, `docs/plans/connected-code-graph-p1-answers.md` P1-23/P1-39, `docs/plans/connected-code-graph-answer-spec2.md` §1.2/§5, `cpg/eval/dev/ac5-flow.jsonl` (dev only), line 1 (`ccg-p1-flow-python-v1`), and the engine code the record pins (`cpg/ccg/engine/flow.py`, `bfs.py`, `reader.py`). Root `REVIEW.md` read; no `cpg/REVIEW.md` exists. No held-out artifact, `~/.local/share/ccg-eval` or `~/.config/ccg` was opened; no tree-mutating git run.

**Verdict: approve with suggestions.** The statistic, rule, Wilson figure and thresholds are sound and ratified-only. Three majors should be fixed in place before the record is committed (M1 builder unpinned, M2 snapshot-pin phase, M3 wrong tie-break order claim), plus the F4 language-cut mismatch (M4). None biases a result in a direction; all are things the record claims or relies on that do not hold as written.

CPG: considered, not relevant — a pre-registration/method review whose claims are checked against files, shas and engine source read directly; the CPG answers no question about them.

## Findings

### M1 (major, verified) — the builder is unpinned, and the evalCommit tree contains no builder
`pins.snapshot` / `runConditions.snapshot` say the CPG is "built at evalCommit from a clean checkout (git archive)". `git ls-tree -r cf7b2042 cpg` shows no `ccg/engine`, `ccg/build.py`, `assemble/`, `linker/`, `units.py` or `extractors/` (only schemas, DDL, `answer.py`, `bundle.py`, eval tools); `git rev-list --count cf7b2042..HEAD` = 441. The HEAD engine reads `Snapshot`, `Unit.flowAvailable`, `CALLS_NAME`, `TO_UNIT` (`reader.py:78-79`), so the graph can only come from the HEAD-era builder run over the evalCommit *source* tree. The record pins the engine, flag registry, dev file and plans, but not `ccg/build.py`, the assembler, `ids.py`/`keynorm.py` (symbol ids: the dev items' `impactedId`s must equal the loaded Symbol ids), `extract.sc`, or Joern/toolchain image. Two different builders give different graphs and different `s`, with the record's pins still green. 
**Fix:** state which tree is the *corpus* (git archive of evalCommit) and which is the *builder* (a named commit); pin the builder files by sha256 (or the builder commit) and the toolchain image digest in the record, not only in the result block. The "graph content digest" can stay in the `-tool` line (see M2).

### M2 (major, inferred) — the snapshot pin cannot be appended "before any result is read" under a one-shot tool
The record says the `-tool` companion pins the loaded graph's content digest before results are read, yet the tool contract is a single run that builds, walks and emits one JSON, and refuses to start without the `-tool` line (which holds the tool sha). A digest of a graph that does not yet exist cannot be in a line written before the first run. **Fix:** split the contract into two phases with an explicit stop: phase A builds/loads and prints snapshot commit, per-unit built commits, digest, image digest (no walks); a second append-only line (`-snapshot`) pins them; phase B walks and refuses unless that line matches the live graph. Alternatively declare the digest recorded-not-pinned and drop the "pins them before any result is read" claim.

### M3 (major, verified) — the tie-break order is wrong against the code
`rule.tieBreakFixedHere`: "as-built < V-flow < V-both < V-union by table size". Measured with `ccg.engine.flow.candidate_table(n)` row counts: as-built 49, **V-both 53, V-flow 62**, V-union 66. By the record's own criterion V-both precedes V-flow. The last tie-break is reachable (V-flow and V-both are not nested, so equal max `s` and equal burden sum and maximum are possible; burden is otherwise structurally monotone: as-built <= V-flow, V-both <= V-union). **Fix:** replace the clause with an explicit fixed list in the chosen order (state either order, but state it as a list, not as "table size"); say plainly that the fewer-rows criterion yields as-built, V-both, V-flow, V-union.

### M4 (major, verified) — the F4 comparison uses a different language cut than the record's own rule
`alsoRequired` compares the as-built `s` to "10/28 (python 6/8, TS/JS 4/20)". The record's language rule is by the *impacted item's file*. Recomputed from `ac5-flow.jsonl`: impacted `.py` = 9 (8 under `resolve_llm`, plus `storefront_api.py::advance_order` under the `advanceOrder` anchor), `.ts/.tsx/.js` = 19. Pure `return-use` chains (the only ones the as-built walk ends at, F4): 10 total, python 6 of 9, TS/JS 4 of 19. F4's 8/20 is the anchor-language cut. As written the comparison either mismatches or must be silently re-cut after seeing data. **Fix:** compare the pooled 10/28 only, and report language-by-impacted-file with a stated prediction of 6/9 and 4/19 (derivable pre-data from hop kinds), or state that F4's cut is by anchor.

### m1 (minor, verified) — a stale sentence sits inside a pinned file
`docs/plans/connected-code-graph-p1-answers.md:121` still says "P1-23 re-pins it before P1-39 publishes it"; spec2 §1.2 (working tree) now says 1.1 is published and any change needs a new version. Fix it before the re-pin; editing it later voids the `f222f02a…` pin.

### m2 (minor, inferred) — tool-contract gaps an implementer would have to decide
Each is a degree of freedom, none individually biasing, but the contract's purpose is to leave none:
- The item match: `Item` has `(kind, id)` (`bfs.py:638-665`); say `kind == "symbol"` and `id == impactedId`, and that the anchor `Node` is `("Symbol", anchorId)` with a missing anchor an error (not an unlisted cluster).
- How the throwaway is credentialed (`ccg_engine_<s>` role), the system name, and the Joern/extractor invocation (see M1).
- Plan shas and the `test_engine_bfs_rows.py` pin are not in the tool's refuse-to-run list (the engine, dev file, EVAL_COMMIT, registry are); either add them or say they are record-only.
- "Writes ... only after the `-tool` line is committed": say how the tool checks it (e.g. the line's blob is in `git show HEAD:cpg/eval/preregistration.jsonl`), else the check cannot fail. Also name the planted input that turns each refusal red (a one-byte-changed engine file, a mismatched `-tool` sha), in the tool's own test.
- `cpg/eval/results/` does not exist; say the tool creates it, or choose the outside-tree path.
- The `counting: true` dev item (`advance_order`, `callee-arg`,`link`) and the one `link` hop kind are in the 28 but not in F4; harmless, worth a clause so the by-signature cut is reproducible.

### m3 (minor, verified) — `measuredAtHead` lags the tree; suggested handling (item 5)
All 12 recomputed shas match the working tree (list below), but spec2's pinned `6e0b9b1d…` is the *uncommitted* file: `git show HEAD:…spec2.md | sha256sum` = `2ab798b1…`. So the pins hold only on a tree one uncommitted edit ahead of `d02de630`, and the record cannot carry the sha of the commit that contains it. **Fix:** (a) commit the spec2 edit (and the `-p1-answers.md` m1 fix, re-pinned) first; (b) in the record, replace `measuredAtHead` with `pinsVerifiedOnTree: "d02de630 plus the committed spec2/answers edits"` stated as a description, not a pin; the real binding is the file shas, and the tool's environment block records the actual HEAD; (c) the tool refuses on sha mismatch, not HEAD mismatch, which is already its design. This keeps the claim coherent without a self-referential commit sha.

### n1 (nit) — cluster caveat and growth
- With 4 anchor clusters the pooled Wilson 0.879 assumes independence the record concedes it lacks; add the cluster-level figure (4/4, Wilson lower 0.510) beside it so the screen is not read as stronger than its units. Recomputed: 28/(28+1.96²) = 0.87935.
- `retLimit` restates the docstring (`answer_build.py:30`) and test (`test_engine_bfs_rows.py:37`), both of which I confirmed exist; it is a fact with a home elsewhere and could shrink to a citation plus the one-line consequence. Growth-stops check (root `AGENTS.md`/`REVIEW.md`): the JSONL record is not a covered kind; the covered edits are the spec2 diff (+4/−3, rework of the "unpublished" wording, no new scope; it restates the fingerprint literal that has a code home, `bfs.py:202`, which is a minor restated fact) and the coordination ledger (append-only, exempt).

## Checks on the numbered questions

1. **Thresholds:** only n/n, 28/28 with Wilson 0.879, depth bound 2 (`config.py:5`) — all in `-p1-ml.md` §3.5/config. Nothing invented. Operational definitions: summing unweighted `Walk.items()` over 4 anchors is fair and pre-data (it equals the plan's per-anchor burden aggregated), and cannot be tuned post hoc since the maximum is fixed alongside. It leaks nothing, because burden is structurally monotone in the table (see M3) and `s` is decided first. One honest caveat to disclose: V-flow/V-both were designed after the dev chains' hop kinds were known (F4 is derived from them), so 28/28 is in-sample; the record calls it a screen but should say the pick is not validated on held-out (held-out validation stays with the AC-5 gate, out of this record's reach).
2. **Falsifiability:** H0 (as-built s<1) and H1 (some candidate 28/28) can each fail; the pair is not complementary but cover every outcome (as-built 28/28 refutes H0 and satisfies H1). `noWalkAtNN` is explicit. Wilson recomputed (0.8794). Clause "28/28 for the V-both second level" consistent with `-p1-ml.md` §3.5 (two-level, depth bound 2).
3. **Shas:** recomputed all and matched — dev file `a67ca4c1…`, `answer_build.py` `131eb3e1…`, `bfs.py` `a8d0db7d…`, `flow.py` `ac97bbe5…`, `config.py` `bfe015cb…`, `reader.py` `109f5dea…`, `state.py` `7f7a9785…`, `group.py` `7ba8ecf3…`, registry `2fadd855…`, plans `94f43ee0…` / `6e0b9b1d…` / `f222f02a…`, `test_engine_bfs_rows.py` `f2dbd1b9…`; `EVAL_COMMIT` = `cf7b2042…`. spec2 matches only the uncommitted file (m3). Dev file: 28 lines, 4 anchors (10/8/7/3), matches the record.
4. **Tool contract:** adequate in shape (read-only, table via `candidate_table`, `bound` default, twice-run byte-equality, V-union superset assertion, locator existence error), but M1/M2/m2 leave freedom. The `-tool` mechanism (sha of the file pinned by an append-only line, run refuses without it) works only if the check is verifiable (m2) and the digest phase is split (M2).
5. **`measuredAtHead`:** m3.
6. **Custody:** respected: only the dev file is named; held-out/reserve, the custodian root and `~/.config/ccg` are excluded and the tool refuses them. `preregistration.jsonl` is itself in `split_manifest.py:822` `CUSTODIAN_FILES`; line 1 was committed the same way, so no new boundary issue.
7. **Growth-stops:** see n1.

## What's solid
Statistic definition (listed iff `Walk.items()` contains the impacted id; anchor never an item; no documents, per D2); fixed rule order (max `s`, then burden); disclosed operational choices; diagnostics explicitly excluded from the rule; deterministic walks with twice-run equality; throwaway instance and quiet-host conditions; the Ret-limit disclosure (docstring and test verified present).

## Open questions
- Which commit supplies the builder (M1)? That is the method owner's call and changes what is pinned.
- Is the cluster-level figure (n1) to be reported by default?

## Pass 2

Re-checked the revised line 2 (still uncommitted; HEAD `a8b496de` holds the spec2/answers edits). Executed: recomputed 92 pinned sha256 (dev file, 7 engine files, registry, 3 plans, bfs-rows test, 79 `builderFiles`): 0 mismatches. `builderFiles` set equals `git ls-files cpg/ccg cpg/extractors/joern` minus `cpg/ccg/engine/` plus `cpg/infra/toolchain/Dockerfile` and `cpg/infra/ccg-falkordb.sh` (79 = 79, no missing, no extra). `candidate_table` row counts: as-built 49, V-both 53, V-flow 62, V-union 66. Dev file by impacted file: pure return-use python 6 of 9, TS/JS 4 of 19. Wilson 28/28 = 0.8794; 4/4 = 0.5101. `EVAL_COMMIT` = `cf7b2042…`.

**Verdict: approve with suggestions** (no blocker; the residual items below are tool-contract text and can be fixed in place before the commit, or left to the tool unit's own test).

Dispositions:
- M1 builder unpinned: fixed. `pins.corpusAndBuilder` names corpus (git objects at `cf7b2042`) and builder (79 files, base image digest via the Dockerfile, builder commit `a8b496de`). Evidence: the set check above. `ccg/ids.py` and `ccg/keynorm.py` have no commit after `cf7b2042` (`git log cf7b2042..HEAD`, empty), so the dev `impactedId`s are produced by unchanged id code; whether the ids resolve is checked at phase A/B (a missing id is an error), correctly.
- M2 snapshot-pin phase: fixed, with caveats G2/G3 below. Phase A prints and stops; the `-snapshot` line is committed; phase B refuses on mismatch. Not circular: the tool file does not contain the `-tool` line, the `-snapshot` line pins output of a run that produced no result, and each refusal reads the committed `HEAD` blob.
- M3 tie-break order: fixed (explicit list as-built, V-both, V-flow, V-union; matches 49/53/62/66).
- M4 F4 cut: fixed (10/28, python 6/9, TS/JS 4/19; matches my recompute).
- m1 stale sentence: fixed (committed in `a8b496de`; `p1-answers` pin matches).
- m2 contract gaps: fixed (Item match `kind == 'symbol'`, anchor `Node('Symbol', anchorId)`, refuse list incl. plans and bfs-rows test, committed-blob check, one planted input per refusal, results dir created). Remaining: G1-G4.
- m3 `measuredAtHead`: fixed (`pinsVerifiedOnTree`, descriptive; file shas bind).
- n1: fixed (cluster-level 4/4 / 0.510 beside pooled; in-sample disclosure in `rule.safeClaim`; `retLimit` now a citation block). Growth-stops: no covered document edited since Pass 1 beyond the committed spec2/answers rework; the JSONL record is not a covered kind.

Remaining gaps (all minor, none biases a result):
- **G1 (inferred) Joern units and "no checkout".** Verified that non-Joern units read the git tree through `gitsafe` (`ccg/build.py:17`, `units.py:22`). `build_unit` is documented as taking "the git work tree `repo` at `commit`" and the Joern path calls `joern_run.extract_unit(repo, unit, …, sha, …)` (`build.py:406`); I did not trace whether it exports the commit or reads the work tree. The record's "the builder reads the tree through git, no checkout" is true only if it does. Have phase A assert it (e.g. the Joern input is a `git archive` of `EVAL_COMMIT`, not the repo work tree), or restate the claim for non-Joern units only. The corpus, where the dev anchors live (`.ts/.js/.py`), is exactly the Joern path, so this is the one that matters.
- **G2 digest undefined.** "the loaded graph's content digest" has no algorithm; two implementers diverge. Phase B equality only needs it to be reproducible, but define it (e.g. sha256 over a sorted dump of node and edge ids) and say whether phase B reuses phase A's live instance (the instance must outlive the stop; "torn down after" currently reads as after phase B) or rebuilds and requires an identical digest (stronger, and tests determinism).
- **G3 per-unit build status.** `build_units` fails per unit alone and `system.yaml` at HEAD may name units whose roots differ at `cf7b2042`. Say that phase A pins the per-unit build status and refuses when a unit hosting a dev anchor or impacted node did not build, so that a degraded graph cannot reach phase B as "unlisted".
- **G4 where pins are read.** State that the tool reads the record's pins from the committed `HEAD` blob, not the working file (otherwise a working-tree edit of the record moves the bar), and that append-only is checked by the earlier lines being byte-identical in `HEAD`. Also `refuseList` calls the `EVAL_COMMIT` check a "sha256 against the pin" while the pin is the commit id itself; say "equals the pin".

Custody re-check: still only the dev file; `~/.local/share/ccg-eval`, `~/.config/ccg`, :6379 and :6390 excluded; the throwaway uses its own `CCG_CONFIG_DIR`. Respected.
