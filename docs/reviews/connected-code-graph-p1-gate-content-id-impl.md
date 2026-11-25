# Connected code graph P1 — implementation review: the dev-gold probe vouches for a content snapshot id

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-37)

CPG: not applicable — a harness-only change (`cpg/eval/tools/`, `cpg/tests/`) reviewed by reading the diff against the assembler, linker, at-rest and engine sources and by running/mutating the tests; no `cpg_*` graph answers it.

Note on path: the brief named `docs/reviews/connected-code-graph-p1-gate-content-id.md`; this is a review of an implementation, so the root `AGENTS.md` grammar gives it the `-impl` suffix.

## Scope and verdict

Reviewed: the uncommitted diff (`gate.py`, `gate_run.py`, six test files, untracked `test_gate_content_snapshot_id.py`; the foreign files ignored) against option (b) of `docs/reviews/connected-code-graph-p1-gate-snapshot-rca.md` and the probe's purpose in `docs/plans/connected-code-graph-p1-closeout-prereg.md` §3 item 1 and §4. Root `REVIEW.md` read; `cpg/REVIEW.md` does not exist.

**Verdict: approve with suggestions.** No blocker. The design is sound: the content id keeps every identity input of the stamped id (graph model version, `packSetHash`, `codeHash`, per-unit id/status/flow/failReason, every bundle byte but the keyed fields), its tag differs, and the stamped `snapshotId` is untouched. Two suggestions matter before the R3 rehearsal (F1, F2).

Checks run: `.venv/bin/python -m pytest` on the five touched gate test files plus `test_withheld_shapes.py` (468 passed); `-k "gate or withheld"` over all of `tests` (1836 passed); `git diff --name-only` is `cpg/eval/tools/{gate,gate_run}.py`, `cpg/tests/*` and the foreign `claude/docs/reviews/...`. `coreTreeHash` excludes `eval/`, `packs/`, `tests/` and `*.md` (`cpg/eval/tools/ac24_smoke.py:90`), and no `cpg/ccg` file changed, so `coreTreeHash`, `packSetHash` and the engine fingerprint are unchanged (executed, not inferred).

## Findings

### F1 (major) — `store_content_id` is never run on real at-rest bundles; a surviving mutant sits in its stale/not-mapped branch

Evidence: `cpg/eval/tools/gate.py` `store_content_id` (new). Its tests use `FakeStore` and an injected `read_bundle`, and the default reader is tested only with both `atrest.read_bundle` and `bundle.loads` monkeypatched. I tried the real path (`atrest.write_bundle` of `TA.world()` bundles into a work root, then `G.store_content_id(FakeStore, ..)` with the default reader): the fixture bundles are not schema-valid for `write_bundle` (`'intra-flow' is not one of [...]`, unit id mismatch), so no test in the suite crosses `write_bundle` → `read_bundle` → `store_content_id`. Mutant `records = None if status != "fresh" else ...` (reads no bundle for a `stale-kept` unit) **survived** all 234 tests in the gate files: `TA.world()` has two `fresh` units and no `stale-kept` or `not-mapped` one, so the `not-mapped` branch and the stale-kept read are unexercised.
Why it matters: the self-check turns any wrong read into a `HarnessError`/`AtRestError` after the ~16-20 min build, stranding the store (the exact cost the RCA was about). It fails closed, so the list is never spent, but the first live exercise would be the R3 rehearsal.
Fix: one test, in the style of `tests/test_refresh_scope.py` (a real refresh over valid bundles with a `stale-kept` unit from `test_a_failed_unit_keeps_its_last_good_bundle_marked_stale_kept...` and a `not-mapped` one), asserting `store_content_id(store_of_that_build, system, its work_root)` equals `content_snapshot_id` computed from the rows, and that two refreshes under different keys agree. That also covers the work-root layout (`atrest.bundle_dir`) which the injected reader hides.

### F2 (minor) — the whole `bindParts` object is dropped, though only two of its fields are keyed

Evidence: `KEYED_BUNDLE_FIELDS` drops `bindParts` wholesale (`gate.py`, new). The keyed members are `hostHmac` and `pathHmacs` (`ccg/linker/bind.py:179-183`, `tests/test_gate_content_snapshot_id.py:keyed_world`); `scheme`, `hostKind`, `host`, `port` are clear and read by the binder (`bind.py:382`, `url.host`/`url.port`/`host_kind`). The presence of `bindParts` on an entry is hidden too. The per-field test replaces the whole object, so it cannot show this.
Why it matters: the claim "excluded fields only" is wider than stated. Two builds of one commit cannot differ there (same `builtTree`, same extractor), so it is not exploitable today, but it is exactly the class "two builds that differ in a way that matters get an equal id".
Fix: strip only `hostHmac`/`pathHmacs` inside `bindParts` (keep the object and its clear fields), and add a test that changing `bindParts.port` moves the content id. If kept as is, say in the docstring that the clear `bindParts` fields are covered only through `builtTree`.

### F3 (minor) — "no answer path reads the keyed fields" is true today but guarded by nothing

Verified by search: `valueHash`, `defaultHash`, `bindParts`, `bindKeyCheck` appear in no file of `ccg/engine`, `ccg/front`, `ccg/store`, `clients/` or the gate harness (only `ccg/assemble/rows.py:738` writes `ConfigEntry.valueHash` as a graph property, plus the linker `ccg/linker/bind.py`, `valuehash.py`, `atrest.py`, `build.py`, `refresh.py`, the extractors and `bundle.py`). The `ConfigEntry.valueHash` graph property **does differ between a rehearsal's store and the gate's**; nothing reads it. Binding is key-consistent inside one build: the binder recomputes hashes under the build key (`bind.py:278`, `bind.py:493`) and compares them for equality, iteration orders are by id/example/repo/path (`bind.py:280-322`), and `ConfigEntry` ids are keyless (`ids.config_entry_id(repo_id, path, kpath)`, `extractors/config/extract.py:145`). A `bind-key-mismatch` demotion surfaces as `stale-kept` + failReason, which stay in the id.
Fix: a small static test (a round before any later reader, not in this change's content id test) that fails if any module under `ccg/engine`, `ccg/front`, `ccg/store`, `clients/` names one of the four fields. It passes now and reddens the day someone reads one, which is the only moment the content id becomes unsound.

### F4 (minor) — missing bundle / absent work dir is an uncaught `AtRestError`, not a `HarnessError`

Evidence: `store_content_id` reads every non-`not-mapped` unit via `atrest.read_bundle`; a missing file (resume with the work dir removed, or a unit whose bundle retention pruned) raises `AtRestError` (reproduced in my probe above: a path-bearing `AtRestError`). Run on `--resume` the work dir is the earlier run's `gate-work-<key>`, so resume works when it exists. A unit in the store with no bundle is a hard stop (correct, fails closed); an extra bundle on disk with no unit is ignored (correct). Fix: catch `(atrest.AtRestError, OSError)` and re-raise a value-free `HarnessError("the build's stored bundle of a unit is missing or unreadable ...")` naming the stranded-store remedy. The self-check itself is not vacuous: it recomputes the stamped id from the same bundles and mutant "stamped check removed" is killed; it is order-independent because `manifest_hash` sorts units.

### F5 (minor) — prereg wording and ordering

`docs/plans/connected-code-graph-p1-closeout-prereg.md:48` still says "snapshot id: harness-enforced by the probe"; it should say "content snapshot id (the stamped id carries a per-build key)". The harness files are pinned by `harness_pin` against the record commit, so this change must be committed **before** record commit R3 (and the rehearsal run on it), else the marker reads `afterAdaptation`; this is `teco`'s sequencing, not a code defect.

### Nits

- `store_content_id` recomputes the stamped id with `SN.bundle_hash` over the unstripped records; fine, and equal to `ccg/assemble/rows.py:482`. `unit_rows` coerces `flowAvailable` with `bool()` and `failReason` with `or ""`, as `rows.py:482/499` do for fresh/not-mapped (a null stored property is the stamped `""`).
- The `gate_run.py` module docstring and `gate.py` design comment were updated; no remaining place compares or prints the old `snapshotId` as the probe check. `snapshotId` stays in the probe as informational and still required by `dev_probe_check` (`gate.py:1466`). `docs/plans/connected-code-graph-p1-closeout-prereg.md:60` uses the run's own stamped id as a noise seed, which is correct.

## Answers to the brief's questions

1. **Soundness.** Another commit: `builtCommit`/`builtTree` (not exempt, tested). Pack set, code state: `packSetHash`/`codeHash` from the `:Snapshot` node. Unit set, status, flow, failReason: per-unit rows, all tested (`test_the_other_identity_inputs_stay_in_the_content_id`). Any other bundle byte: tested for a header field, a symbol, a config keyPath, a launchFact port. What the exclusion hides: the four fields plus (F2) the clear members of `bindParts`; by commit/tree identity and the extractor's determinism these are functions of (tree, key). No answer path reads them (F3 list, executed search).
2. **Store reader.** Not vacuous (stamped recomputation, mutant killed); order-independent; fails closed on a missing bundle but with a raw `AtRestError` (F4); real-path coverage missing (F1).
3. **Run wiring.** `_run_gate` computes the content id right after the build on every path (rehearsal and gate; resume recomputes from the existing work dir), records `contentSnapshotId` in `run.json`, and `write_dev_probe` copies it (refuses a run without it). Only hashes are persisted or printed; no key, bundle or held-out value. The stamped id is still used for the engine-reader check and the marker pin.
4. **Probe strictness.** Old probes are refused twice with text naming `gate.py dev-probe` (`dev_probe_check` pre-run, `check_probe_snapshot` post-build); the "rehearse again on this build" text is gone (tested).
5. **Mutation testing** on a copy under the scratchpad (`mut.py`, 18 mutants over `gate.py`/`gate_run.py`): killed — keyed-field set shrunk, filter applied to header only, stamped self-check removed, unit-row sort removed, wrong work root passed to the reader (`Path(work_root).parent`) and wrong work area in `gate_run` (`PRIVATE_AREAS[1]`), flow/status/failReason forced, tag altered, comparison disabled, probe compares stamped id, content id copied from the stamped id, `run.update` field dropped, presence check removed. **Survived:** `status != "fresh"` in `store_content_id` (F1). One more mutant of mine (`and units` on the self-check) is equivalent, not a survivor; the mutant for `_store_content_id` system argument did not apply (text mismatch) and is untested by me.
6. **Drift.** Only F5.

## What's solid

The test file compares two real builds under two keys instead of copying constants; the content id keeps the stamped id's own `manifest_hash` so no input is re-derived; the stamped `snapshotId` is untouched; the refusal texts say what to do; the injection seam (`content_id_factory`) keeps the run verb tests honest about which work root is passed; the `test_withheld_shapes.py` count change (64 items; 14 withheld = B2 C6 I2 J2 K1 M1) matches the prereg §3 item 1 text.

## Open questions

None blocking. For `teco`: confirm this change lands before R3 (F5).

## Proposed `REVIEW.md` lines (proposal only)

- A harness identity that excludes fields "because they vary per build" gets a static test that no answer-path module names them, and each exclusion removes only the varying members of a composite field.
- A store-reader that reads files a real build wrote needs one test over real written files, not only an injected reader.

## Pass 2

Scope: the uncommitted tree after the engineer's fixes (`gate.py`, new `tests/test_gate_content_snapshot_id_real.py`, extended `tests/test_gate_content_snapshot_id.py`). Executed: the five gate/content test files (163 passed); `tests -k "gate or withheld or content"` (1882 passed); 7 mutants on a fresh scratchpad copy (`mut2.py`, never in place).

**Verdict: approve.** No blocker, no major open. F5 is `teco`'s.

- **F1 fixed.** The old survivor `status != "fresh"` is now killed (1 failed); new mutants also killed: self-check disabled (`if False`), wrong work root (`Path(work_root) / "x"`), and the wrap's exception tuple replaced (`except ZeroDivisionError`). The real test (`atrest.write_bundle` → `read_bundle` via a real two-key refresh with a fresh, a stale-kept and a not-mapped unit) passes, and the stale-kept read is asserted.
- **F2 fixed.** `KEYED_BIND_PARTS = ("hostHmac", "pathHmacs")`; `ccg/valuehash.py:bind_parts` produces exactly `scheme`, `port`, `hostKind`, `host` (clear) and `hostHmac`, `pathHmacs` (keyed), so the strip is exact. Mutants adding `port` to the set or removing `hostHmac` are killed; port/scheme/hostKind/host/presence edits move the id (tests lines 104-113).
- **F3 fixed.** `keyed_mentions` scans `ccg/engine`, `ccg/front`, `ccg/store`, `clients` for the six names, passes on the tree, and has planted-mention and coverage tests.
- **F4 fixed.** Missing bundle and absent work dir give a value-free `HarnessError` (no path, no unit id, not an `AtRestError`); the mutant disabling the wrap is killed.
- **Cross-check of the same-revision fixes.** F3's scan and F4's wrap do not interact (different files, no shared state). F2 with F1's two-key equality: the real test equates two keys over real bundles, but the instrumented run shows those bundles carry `valueHash` (6 records) and **no `bindParts`, `bindKeyCheck` or `defaultHash`**, so the narrower `bindParts` strip is proven only on the synthetic world (N1).

### New findings

- **N1 (minor).** The real-bundle two-key test never exercises `bindParts`/`bindKeyCheck`/`defaultHash`, the fields F2 changed. If a real writer shape differs from the synthetic `bindParts` (key order, extra member), only a live rehearsal would show it, at 16-20 min cost. Suggest a config bundle with a URL entry in the `test_refresh` world (or `bind_parts` output spliced via `valuehash.bind_parts` under both keys) in the real test.
- **N2 (nit).** Stripping the whole `pathHmacs` array also hides the number of path segments, which the binder reads (`len(url.path_hmacs)`, `ccg/linker/bind.py:499`). Replacing it with its length keeps that fact; determined by `builtTree` today, same class as F2.
- Survivor accounting: mutant `... and k != "pathHmacs"` survived and is equivalent (already in the tuple).
