# P1-34 flow re-measurement: the `extract.sc` pin after P1-05 5d

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-34 steps 3–4) · **Extends:** `docs/plans/connected-code-graph-p1-flow-prereg.md`

## Verdict

1. **The 5d change is inert for the flow measurement.** Verified by code reading and by running both `extract.sc` versions on the pinned python CPGs.
2. **Reconcile with a successor record `ccg-p1-flow-python-v2`**, which pins the current `extract.sc` and carries this ruling. Do not restore the old file. This needs one small edit to `s28_metrics.py` (§3), because the record id is a constant in the pinned tool.
3. **The human's step is the existing custodian draw** (`prereg` §3.2). It happens after `-v2` is committed and gated, and takes roughly 15 minutes of the owner's time (§4).

## 1. Findings

- **Drift.** `coreFreeze["extract.sc"]` of `-v1` is `6cdb498f…ca7d` (`3089951f`). The tree is `02001899344a435ec805f2066f3936e6af6f05424ed25bd43f46f8c9793c00cf` after `69789943`. Re-hashed today: every other pin of `-v1` still matches the tree (`s28_core.py`, `s28_run.py`, `s28_metrics.py`, `run_fitness.py`, `flow_decide.py`, `flow_p1_audit.py`, brief template). `extract.sc` is the only drift. The record is unrun, undrawn and unlabelled: no label exists and no held-out item has been opened.
- **Diff content.** The hunks are `recvTag`, one `"rt"` key set on `cs` facts, and a JavaScript-only export-block branch (`lang != "javascript"` gives `Nil`). Nothing is removed or reordered; `rt` is set after the existing keys.
- **Consumers.** `s28_core.load_facts` keeps every `cs` object whole but reads only `id`, `name`, `l`, `c`, `code`, `m`. There is no `.items()` or `**` over a fact, and no `rt` anywhere in `s28_core.py`. The sample frame comes from tree-sitter (`run_fitness.flow_population`), not from facts. The `stats` row is read only by the descriptive performance block of `s28_metrics` (`loadMs`, `projectionMs`, `histogram`), which is not a decision input.
- **Empirical check (offline, existing toolchain image, about 15 s per run).** The pinned CPGs match the record (`off` `a30c8363…`, `on` `fc06c0c3…`). I ran the `3089951f` file and the current file, via `cpg/sandbox/offline-run.sh` as `s28_run.py` does, on both CPGs, into the scratchpad only. For `off`, both runs give 132,693 fact lines; the only differences are the `rt` key on 22,805 of 23,677 `cs` facts and the timing fields of the `stats` row. Then I ran the frozen `s28_core.load_facts` and `build_method` over all 6,254 methods for each version, and hashed every method's full state. Old and new are identical for `off` (`6f279fa7…`) and for `on` (`cf19ae2e…`).
- **Not tested.** The Joern run was not through `s28_run.py`, so no run record was written. `s28_metrics` scoring end to end was not executed, because labels and the draw do not exist; the join and scoring read only what `build_method` returns, and that is identical.
- **Consequence.** The decision cell (fixed projection × new join, `-v1`/prereg §0.1) and the entry value 465/481 (designed on measured methods) are unaffected by 5d.

## 2. Why a successor and not restoring the file

Restoring `extract.sc` to `3089951f` reverts shipped P1-05 5d code that the normalizer and its tests now depend on. Pointing `EXTRACT_SC` at a frozen copy edits a pinned tool. Editing the `-v1` line rewrites a committed record, and the tools read the first matching line, so two lines with one id would be a silent trap. A successor keeps `-v1` as committed history and states what changed and why.

## 3. What changes, and who does it

1. **`s28_metrics.py`** (`tdd-engineer`, `analyst` gate; behaviour-preserving): `RECORD_ID` becomes `ccg-p1-flow-python-v2`. `load_record(…, RECORD_ID + "-draw")` then resolves to `-v2-draw` by construction. Change only that constant and the two strings in the same file that name the id; leave `flow_decide.py` and `flow_p1_audit.py` untouched (their text only mentions the id in docstrings, and an edit would change two more pinned shas). Update any test that asserts the id string. This is a pre-label tool edit; the core (`s28_core.py`, join, scoring) is not touched.
2. **Record `-v2`** (`teco` appends to `cpg/eval/preregistration.jsonl`, full 64-hex shas verified by script against the tree, after item 1 is committed): `id` `ccg-p1-flow-python-v2`; `supersedes` `ccg-p1-flow-python-v1`; `coreFreeze["extract.sc"]` = `02001899…c00cf`; `tools["s28_metrics.py"]` = the new sha; a field `extractRevision` naming `69789943` and citing this note; every other pin copied from `-v1` unchanged (`s28_core.py`, `JOIN_P0_ONLY`, the other tools, `cpgs`, `shippedFacts`, `seeds`, `frame`, `rules`, `decisionCell`, `entryValue`); `toolingCommit` is the commit of item 1.
3. **Freeze `extract.sc` again** from the `-v2` commit to the decision. `teco` tells every P1-05 follow-up (the 5d audit fixes are pending) that any further edit to `extract.sc` before the run needs a `-v3` and a fresh inertness check. Re-running this note's check costs about 30 s per configuration.
4. **Prereg and review text** (`data-scientist`, on the next ordinary revision of `prereg`): replace the `-v1` record id and `extract.sc` sha with `-v2` where they are cited as pins (§0 finding 3, done-condition checklist). `-v1` is then cited only as superseded.
5. **Gate.** `analyst` checks items 1–2 in the same dispatch: that the diff to `s28_metrics.py` is the constant and strings only, and that the `-v2` pins equal the tree (`s28_metrics.check_record` against a synthetic draw does this mechanically).

## 4. The one stakeholder decision (held-out custodian)

The stakeholder spent the fresh draw on 2026-11-11 and the owner is the custodian. Ask: **will the repo owner run the custodian draw (`prereg` §3.2) once `-v2` is committed and gated?** If yes, concretely:

- **When:** after `-v2` is committed, so that step 0 (pull; check `run_fitness.py` equals the record's pin) runs on the final record. Nothing else gates it. The unit is on the P1 path before P1-35's closing items and the feature close; the labelling and scoring units (U4 to U9) cannot start until the owner returns the counts.
- **What, on the owner's machine:** (a) `run_fitness.py fetch python` reports `reused` or digest `365761fb…4800d6`; (b) write a private file outside the repo with one JSON line `{"path", "defLine"}` per python anchor of the held-out and reserve AC-5 `flow` items; (c) run `flowdraw-p1 python --exclude <private-file> --counts <counts.json> --tier2-out <path outside repo>`; (d) check 30 and 15 lines and `sha256sum` the tier-2 file.
- **Returned, nothing else:** `excludedTier1Frame` (k1), `excludedTier2Frame` (k3), `sample/python-flow-p1.jsonl`, the tier-2 file's sha256, tree-sitter and grammar versions. A `0`/`0` result is normal and is reported as such. The private file, anchors and any prefix count are never returned.

## 5. Risks

- **The entry value has zero margin** (464/480 reproduced, 465/481 with the omission). This note does not change that; the `-v2` run is only valid if the facts are the ones measured here, which §1 supports. Any later projection change breaks both the pin and that margin.
- **This ruling covers python only.** The JavaScript export-block change is not inert for JavaScript constants, which is irrelevant to this record and relevant to any JS flow record.
- **The inertness proof is on the offline CPGs of this machine's toolchain image.** The image id is not pinned before the run (`check_record` doc); `teco` hand-checks it at U7 as already specified.
