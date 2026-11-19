# connected-code-graph P1-35 dispatch 1: AC-24 smoke harness (implementation review)

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-35 (connected-code-graph P1)

## Scope and verdict

Reviewed, uncommitted and new files only: `cpg/eval/tools/ac24_smoke.py` (702 lines) and `cpg/tests/test_ac24_smoke.py` (902 lines, 90 tests). Baselines: `docs/plans/connected-code-graph-p1-gates.md` P1-35, `docs/plans/connected-code-graph-p1-ml.md` §4.1-§4.3 and §6.4, `docs/plans/connected-code-graph-eval-protocol-ml.md` §10.4, §10.5 and §12.3, `docs/reviews/connected-code-graph-p1-26a-ml.md` Pass 3/4, and the code the harness drives (`cpg/ccg/front/*`, `cpg/ccg/engine/anchors.py`, `cpg/clients/common.py`, `cpg/ccg/broker/server.py`). Root `REVIEW.md` read; `cpg/REVIEW.md` does not exist. No held-out item content was read (only `ls` of the custodian directory names). No live model, no FalkorDB.

CPG: considered, not relevant — a pre-registered measuring instrument checked by reading and by running the real Client/Front with a fake broker; call-graph impact is not the question.

**Verdict: needs changes.** One blocker (B1) must be fixed before the held-out run; it does not block the dev entry run. Three majors.

Checks run: the 90 tests in `cpg/tests/test_ac24_smoke.py` pass in the real tree. Mutants and probes ran only against copies in the session scratchpad (`CCG_TOOLS_DIR` points the tests at a copy). `cpg/eval/preregistration.jsonl` and the tool were sha256-identical before and after.

## Findings

### B1 (blocker for the held-out run, verified): the unusable-reply observable collides with a valid answer, so a correct `no-anchor` answer scores "not understood"

`is_unusable` (`ac24_smoke.py:189`) reads "ambiguous, `no-anchor`, `options == []`, provenance present, `cacheHit` false" as an unusable reply and scores it "not understood" for every kind (`:210-213`). A valid, grammar-conformant draft emits the same document. Probe (real `Client`/`Front`/`Interpreter` over `FakeBroker`, draft reply `{"anchors":[{"spans":[{"text":"it","kindHint":"other"}]}], ...}`, item `goldOptions == []`):

`{'understood': False, 'reason': 'unusable-reply', 'status': 'ambiguous', 'nOptions': 0, 'unusable': True, 'ambiguityKindEmitted': 'no-anchor'}`

Why it matters: EP §10.1 requires each of the four `ambiguityKind`s at least once per language in the written set, and EP §10.4 scores a `no-anchor` item as understood on any `status != resolved`. A model answering "it" correctly extracts no anchor, which is the expected behaviour, and the harness fails it. The held-out ambiguity bar is "every item understood" and the run is once-only (a failing run is reported as run, the fix goes on dev), so one such item makes the gate unpassable by a correct model. Dev holds only `several-symbols` and `two-use-cases` ambiguity items (QW-19, QW-47), so the dev entry run cannot expose this. The module docstring discloses the limit and calls it conservative; conservative is not the same as valid at a once-only pass/fail gate whose point is to measure the interpreter.

Fix, no spec change needed: `Front.answer` calls only `self._interpreter.interpret(...)` and reads `result.draft is None` (`ccg/front/__init__.py:111-114`). In `build_live_rig` pass a thin `Interpreter` subclass whose `interpret` appends `result.draft is None` to a per-call list on the `Rig`. `run_items` reads that list between `mark()` and the end of each repeat, and `score_item` takes `unusable` from it instead of inferring from the document. Keep the inferred observable only as the fallback for stubs. Add the planted pair that distinguishes them: not-JSON reply (unusable, not understood) against a valid `other`-only draft on a `goldOptions == []` item (understood).

### M1 (major, verified): EP §10.4's "E1 applies first" is not implemented

`_anchor_key` (`:170`) compares the raw `qname`. EP §10.4 maps every interpreted `symbol` anchor to `E(qname)` before comparing, and gives the example that a lambda enclosing the line scores the same as `ProductList`. The resolver returns the innermost symbol by line (`ccg/engine/anchors.py:196-208`, minimum width), so a file+line item inside an anonymous scope yields `...::ProductList.<anon>#1` against gold `...::ProductList`. Probe: `score_item` returns `wrong-interpretation`. Held-out cell 1 is "a symbol name, or a file and line". The error direction is only to under-credit, never to flatter, so it cannot inflate a pass, but it can fail a correct answer on the once-only run. `E()` already exists at `cpg/eval/tools/match_keys.py:213`.
Fix: apply `E` to the `qname` of interpreted symbol anchors in `request_key` for the interpretation side (gold is already eligible), with the EP §2.3 `#k` caveat recorded as the known limit. If the method owner chooses not to apply it, record that in the pre-registration record as a deviation; today it appears nowhere.

### M2 (major, verified by mutants and trace): the effective-sampler assertion is vacuous in the live path, and its held-out plumbing is untested

- `build_live_rig` wires no observer (`:617-631`). `prereg-record` never passes `sampler=` to `build_prereg_record` (`:685`), so the record's `sampler.effective` is `dev_result["samplerEffective"]`, which is `None` live. At the held-out run `expected_sampler_effective is None` skips the check. The pre-registered requirement (Pass 4 item 4.2: observe before dev, re-observe before held-out, mismatch is a harness error) cannot be satisfied through the CLI.
- If a human hand-adds `effective` to the record, the live held-out run always raises `HarnessError` (`effective` is `None` against a set expectation). That happens after the marker is taken (see m1), so it burns the once-only run.
- Mutants on a copy: replacing `expected_sampler_request=(req or {}).get("params") if req else None` with `None` (a4) and `expected_sampler_effective=rec.get(...)` with `None` (b1) both leave 90/90 green. `run_items` is tested with these arguments; `run_heldout`'s hand-off from the record to them is not. A third silent argument (`cap_floor=rec["capFloor"]`) is an equivalent mutant, because `find_prereg` already forces 5.

Fix: add `--sampler-effective FILE` to `prereg-record` and an observation file or hook to `heldout` (or state in the note that the effective read is taken by hand and compared by `teco` before the run, and remove the half-wired path). Refuse before the marker when the record sets `effective` and the rig has no observer. Add two held-out tests: a request-body sampler mismatch, and an effective mismatch, each through `run_heldout` with a record that carries them.

### M3 (major, verified absent): `coreTreeHash` and `packSetHash` are never recomputed (EP §12.3 gap)

EP §12.3: at the run the harness recomputes `coreTreeHash` and `packSetHash` from the working tree, `promptHash`/`grammarHash` from the prompt and grammar files, and the rest from provenance; any difference makes the run "after adaptation". The harness takes the first two as caller input (`:461`, CLI `--core-tree-hash --pack-set-hash`) and checks nothing against the tree. `promptHash`, `grammarHash`, `samplingHash`, `resolverVersion` and the fingerprint are covered, indirectly: the interpreter computes them from the live files and the provenance equality check (`:358`) compares them with the record. A grep for `coreTreeHash` in `cpg/` finds only this tool and its test, so there is no helper to reuse. Consequence: core or pack code changed after the record, with an unchanged prompt, still scores as the pre-registered run.
Fix: implement the §12.3 construction (a SHA-256 over canonical JSON of sorted `[relpath, sha256]` pairs; `coreTreeHash` over tracked files under `cpg/` minus `eval/**`, `packs/**`, `tests/**`, `**/*.md`), compare in `run_heldout` before the marker, and refuse when it differs (or label the run "after adaptation" and refuse the claim). At minimum, refuse on any uncommitted change under `cpg/` (`git status --porcelain -- cpg`, read-only). A plan row that moves this to another unit must say so; P1-35's text does not.

### m1 (minor, traced): the once-only marker is not unbypassable, and the rig is built after it

- `--custodian-root` is free (`:655`). Copy `heldout/` to another root and a fresh marker is created: a second run, silently. Derive the marker location from something fixed: a path under the real custodian root `custodian_root()` keyed by the record hash, or reject a `--custodian-root` other than the derived one (tests pass the root through `run_heldout`, so only the CLI needs to).
- `rig_factory()` and `run_items`' preflights (`cache_len`, sampler) run after `take_marker` (`:593-604`). Any import or setup error, or an unknown system, burns the run before a single item is read. Build the rig and run the preflights first; the marker still precedes `load_items`.
- Verified sound: `O_CREAT|O_EXCL` is atomic; the marker is taken before `load_items`; an existing marker in any state refuses; Refusal and HarnessError after it write `aborted`; an unexpected exception leaves `started` (fail-safe, requires human removal).

### m2 (minor, traced): `prereg-record --out` truncates its target

`a.out.write_text(...)` (`:688`) overwrites. `--out cpg/eval/preregistration.jsonl` destroys the append-only registry. The docstring says "NEVER appends"; it also never refuses to clobber. Refuse an `--out` that exists or resolves to `preregistration.jsonl`. Add a test with that path.

### m3 (minor): parser-resolved items count in the pooled and per-language k

A held-out question the parsers fully read has no provenance and no model call yet adds to `k_unamb` (`summarize`, `:406-408`). The claim is "the interpreter is not broken", so the headline k/24 should be reported beside an interpreter-only k/n; the tally is shown but is not the denominator. A not-built outcome after a model call (`prov is None`, `posts` present) also skips the provenance and sampler checks (`:349-352`), so a wrong model on that item goes unseen.

### Nits
- The docstring says "counts, qids and paths on stdout"; stdout carries counts and the output path only (`_print_summary`). Qids go to the custodian JSON. Correct the sentence.
- `load_items` echoes a bad `outcome` value (`:149`); echo only the line number.
- A malformed dev-result JSON raises an uncaught `JSONDecodeError` (exit 1, a gate-fail code) instead of a Refusal (4).
- Held-out `dev --dev-items` pointing at the held-out file is stopped only by the 12/2 count; add an explicit refusal of any path under the custodian root.

## Answers to the brief's checks

1. **Scoring vs §10.4.** The understood definition, `CAP_i = max(capFloor, |gold|)` applied to every ambiguous answer, and "every gold option" containment match §10.4 (`score_item`, `:196-242`). E1 is the gap (M1); without it the omission is one-directional (under-credit). Thresholds are exact: `-((-num*n)//den)` gives 23 (n=24, 0.92), 11 (n=12, 0.85), and "all" for ambiguity; they derive from `recount`, not from constants, and a split with other counts is refused (`EXPECTED_HELDOUT`). Wilson is z = 1.959964, tested against the §10.5 figures. Repeat 1 is the scored draw (mutating `_first` to the last repeat is killed).
2. **Held-out guards.** The path `<root>/heldout/ac24.jsonl` matches `split-ac24`'s output (`split_manifest.py:1139`: `heldout_dir / "ac24.jsonl"`), and the file exists at `~/.local/share/ccg-eval/cf7b2042cce7/heldout/`; `<E12>` equals the first 12 hex of `eval/EVAL_COMMIT`. No path prints or writes item text: stdout is counts, the output JSON holds qid, language, class and verdicts, Refusal and HarnessError details are counts or codes. The HEAD check is exact line membership in `git show HEAD:<rel>` (mutation removing it is killed). Bypass and ordering: m1.
3. **Cache and counting.** POST-only count at the broker seam is right: the broker server has no retry of its own (grep), `GET /api/v0/models` is excluded by path and method, a retry is `status is None or non-2xx` with an identical body, and the interpreter's one retry re-enters `request`. A parser-resolved item sends nothing (no run is opened), so it cannot miscount. Gap: m3 (not-built after a model call).
4. **Provenance equality.** Seven keys compared against the record's `interpreter` block, in both `run_items` and the dev-to-record check; mismatch is `HarnessError` (exit 5), not a score. Fail-closed on extra keys in the record. Mutants removing either comparison are killed.
5. **Unusable observable.** B1.
6. **`build_prereg_record`** returns a dict and does not open the registry; `preregistration.jsonl` was sha256-identical across the run. The `--out` hazard is m2.
7. **`coreTreeHash`/`packSetHash`.** M3.
8. **`build_live_rig`** mirrors `Client._default_front` (`common.py:402-406`) with the counting broker and a fresh `Interpreter`, so the product path is what is measured. It is untested; one smoke test with a `FakeBroker` injected through a patched `BrokerClient` would pin the wiring. The `sampler.effective` hook semantics are M2.
9. **Mutants (copy only).** Killed: bypass starts at repeat 2 instead of 3 (`r > 1` to `r > 2`), scored repeat (`[0]` to `[-1]`), per-language k, voided repeats included in draws, `interpreterObserved` guard removed, bypass dropped from the call, qid order reversed, void flag off, HEAD check removed, marker removed, dev `entryPass` check removed, `cacheHit` condition in `is_unusable` removed. **Survived: a4 and b1 (M2).** Equivalent: `cap_floor`.

## What is solid
- The guard order (flag, record present and in HEAD, dev pass, interpreter equality, items file exists, atomic marker, then read) and the fail-closed direction of the marker.
- Counting at the broker seam with per-repeat accounting, which is stricter than EP's pooled ≥ 5 count and agrees with its planted inputs.
- Exact-fraction thresholds from a re-count, and a stop (C5) that cannot be passed.
- The test suite's end-to-end tests run the real `Client`/`Front`/`Interpreter` against a fake broker, including the retry and the duplicate-question C5 case.

## Open questions
1. M1: does the method owner want E1 applied (the EP text says yes) or recorded as a deviation?
2. M3: is the §12.3 recompute owned by P1-35 or by P1-32b/P1-42? The P1-35 text does not say.
3. B1: if the interpreter-side hook is not wanted, the method owner must decide that a `no-anchor` held-out item scores by the document alone and unusable replies are reported only; today's rule is not that.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading "Always check". Source: B1, M2 of this review.
- A measuring instrument that infers an engine event from output ("unusable reply", "cache hit") is probed with a valid output that yields the same observable, and the planted pair must score differently; a collision is a finding.
- A pre-registered value is checked through the call that hands it to the guard: mutate the argument to `None` at that call site, and a test must go red. A guard tested only with its argument supplied directly is a bound.
