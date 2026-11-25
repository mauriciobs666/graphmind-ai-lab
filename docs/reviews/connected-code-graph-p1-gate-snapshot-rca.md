# Connected code graph P1 — RCA: the dev-gold probe never matches the gate build's snapshot id

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-37)

CPG: not applicable — a root-cause analysis of a build identity, read from the rehearsal and gate-run build artifacts, the assembler/atrest sources and a read-only query of the run-2 throwaway store; no `cpg_*` graph answers this.

Scope: gate run 2 (refused, `exit=2`, 916 s) against the t3 rehearsal (snapshot `a965482a84f90cecff7b66a484642e38`). Compared: `cpg/ccg/assemble/snapshot.py`, `cpg/ccg/atrest.py`, `cpg/eval/tools/gate.py`/`gate_run.py`, the t1/t2/t3 rehearsal bundle sets (`<scratchpad>/co9`, `co9-t2`, `co9-t3`, `.../home/.local/share/ccg-eval/rehearsal/gate-work-*/bundles`, 1042 bundles each) and the run-2 store `ccg-falkordb-tw-16640` (read-only queries as `ccg_raw_graphmind-ai-lab`). Not read: `~/.local/share/ccg-eval/` (so run 2's own bundles), `cpg/eval/dev|heldout*` gold. Not run: any gate verb.

## 1. Symptom and impact

`gate_run.py run` builds the gate graph, then `G.check_probe_snapshot(probe_doc, sid)` (`cpg/eval/tools/gate_run.py:651`, `gate.py:1481-1484`) requires `probe.snapshotId == built snapshot id`. Run 1 and run 2 both refused. Each refusal leaves a stranded built store and costs a 16-21 min rebuild; the list is not spent. Run 2 removed the tree difference (both builds are `337910eb`, tree `277a2cb4...`), so the cause is not the commit.

## 2. Evidence

All of the following were computed, not read off a claim.

| Input of `manifestHash` (`snapshot.py:143`) | t3 rehearsal | run 2 | Equal |
|---|---|---|---|
| `graphModelVersion` | constant `1.1` | `1.1` (store) | yes |
| `packSetHash` | `03407085ef3d341a5e88d12d41d6d1ab` (recomputed from `t3/repo/cpg/packs`) | `03407085...` (`:Snapshot`) | yes, and all 10 `:Pack.contentHash` equal |
| `codeHash` | `16ea26b3...153e` (recomputed on `t3/repo/cpg`; also on `gate-run-2/repo/cpg`) | `16ea26b3...153e` | yes |
| unit set | 1042 units | 1042 units, same `unitId`s and `rootPath`s | yes |
| per-unit `buildStatus` / `flowAvailable` / `failReason` | fresh / registry value / "" | all 1042 fresh, `flowAvailable` false, no failReason | yes |
| per-unit `bundleHash` | rebuilt from t3 bundles: reproduces `a965482a...` exactly (method validated) | not obtainable (custodian path off limits) | **differs for at least one unit; see below** |

So `bundleHash` is the only identity input left. Evidence that run 2's bundles equal t3's everywhere except keyed fields, from the store: every `Symbol` (11,654: id, kind, qname, lines, `lastChangedAt/Commit`, complexity), `CallSite` (55,582), `File` (1,327, contentHash and path), `Param`, `Ret`, `Cond`, `Unit` (`builtTree`, `builtCommit`, language, path) and the `ConfigEntry` id set match t3's bundles one-to-one; edge counts equal the t3 header counts (`DATA` 69,983, `CONTROLLED_BY` 18,623). The only mismatch is `ConfigEntry.valueHash`: **0 of 79,462 equal** (plus 14 launch-derived `ConfigEntry` rows with no bundle counterpart).

The same effect is visible in the three earlier builds of the real repository, which all built `EVAL_COMMIT cf7b2042` bundles under different keys. Per-unit `bundleHash`, t1 vs t2 vs t3: **182 of 1042 units differ in every pair**, always the same 182 (all `config`-language: 189 config units exist, 182 hold `configEntry`/`launchFact` records and the other 7 hold none). The difference is only in `configEntry.valueHash` (79,462 records), `configEntry.bindParts` host/path HMACs (66), `launchFact.defaultHash` (6), and header `bindKeyCheck` (9). With those four keyed fields removed before hashing, **all 1042 units are identical across t1, t2 and t3** (0 differ), though those builds ran at three different repo commits.

## 3. Causal chain

1. The snapshot identity covers `bundleHash` per unit (`snapshot.py:126-135`, `rows.py:482`); only `builtAt` and `sandboxAuditIds` are exempt (`snapshot.py:24`).
2. A bundle's config records carry HMAC-keyed fields: `valueHash`, `defaultHash`, `bindParts`, `bindKeyCheck` (`ccg/valuehash.py:31-44`, `ccg/atrest.py:481-483`), keyed by the system HMAC key `atrest.hmac_key(system, keys_root)` (`refresh.py:505`).
3. That key is `secrets.token_bytes(32)`, generated on first use in a keys root (`atrest.py:492-507`; its docstring: "the key changes every `valueHash`").
4. The gate gives every attempt a **fresh private keys root**: `_private_build_areas` creates `gate-keys-<listkey>` and `_clean_leftover` removes any earlier one (`gate_run.py:487-497`, docstring "Fresh per attempt"). Run 2's custodian root and list differ from t3's rehearsal root and list in any case.
5. Therefore every attempt has a new random key, so the 182 units with config entries get new `valueHash`es, new `bundleHash`es, a new `manifestHash` and a new `snapshotId`. The probability that two attempts agree is 2^-256, not "usually". Run 1 (a different commit) and run 2 (the same commit) fail for this one reason; the commit difference in run 1 was additionally sufficient on its own through `codeHash`.

Root cause: the snapshot id is a function of a per-attempt random secret, and the probe protocol (`check_probe_snapshot`) compares that id across two attempts. **Confidence: confirmed** for the chain and for t1/t2/t3 (computed from bundles; method reproduces `a965482a`); **inferred** for run 2's own bundle hashes (its bundles were not read; the store shows `valueHash` ≠ t3's for all 79,462 entries and everything else equal, which is what the chain predicts).

Trigger: the first time a probe from one build was presented to another. Contributing factors:
- The determinism clause (`docs/plans/connected-code-graph-p1.md` ~446, "identical snapshots minus timestamps and `auditSetHash`") and the AC-9 rebuild test (`cpg/tests/test_fixture_gate_p1_ac.py:365`) hold only under one persistent system key (the fixture stack's steady key). No test builds twice under two keys, and `test_assemble_rows.py:1181` varies input order, not the key. The statement "same commit gives the same `snapshotId`" (closure report) is true only under a fixed key.
- The refusal text ("rehearse again on this build") is unfulfillable: the harness refuses to run on a store that already holds the live graph, and the next `run` rebuilds under a new key.
- The pinned design (a probe bound to the build, produced by a prior dev-only run that is a separate build) had never been exercised end to end before run 1.

Ruled out (each by comparison): clone or user path in bundles or code hash (no `/home/mauricio`, `gate-build`, `/tmp/claude` string in any t3 bundle; code/pack hashes equal across `t3/repo` and `gate-run-2/repo`); a unit with another `buildStatus`/`flowAvailable`/`failReason` (run-2 store: 1042 fresh, false, empty); transient extractor or Joern failure (same 12 joern units flow-projected in t3; every `Symbol`/`CallSite`/`DATA`/`CONTROLLED_BY` row equal); timestamps and ordering (exempt, or sorted in `manifest_hash`; t1/t2/t3 equal without the keyed fields); files outside git (the build clones tracked content at `EVAL_COMMIT`); untracked or `__pycache__` files (excluded from `assembly_code_files`; pack dirs hold none in either clone); held-out/custodian inputs (the build never reads the lists); packs reading environment (pack hashes equal).

## 4. Can rehearsal-then-run ever match under the current harness?

No. A dev-only run is a separate attempt with its own fresh key, the held-out run is another, and `run` takes the probe as a file produced by the earlier attempt. The match probability per attempt is negligible, so the check is unsatisfiable, not flaky. Any retry of run 2 as it stands burns a store, ~16 min and one authorisation and refuses again.

## 5. Options

- **(a) Procedural fix to make the builds identical: not available.** The key is created inside the harness between `mkdir` and `refresh` and cleaned on every attempt; no operator step can inject a shared key without editing `gate_run.py` (pinned in the probe's `harnessSha256`), and a shared key between the rehearsal and the held-out custodian root would couple them. A code variant (persist `gate-keys` across attempts) is a harness change anyway, and it also needs a custodian root common to rehearsal and run. Not recommended.
- **(c) Rehearse and run in the same store/build: sound in principle, needs a harness change, worse than (b).** It requires a verb that keeps the rehearsal's store and builds nothing at run time, which contradicts `gate_run.py`'s deliberate refusal on a store that already holds the live graph and the "fresh build at `EVAL_COMMIT` per gate run" property. It also makes the dev rehearsal and the single-use held-out run share one build lifetime, so a harness fault found during the run costs the store.
- **(b) A probe that vouches for the snapshot's content inputs: recommended.** Add a second identity, e.g. `contentSnapshotId`, computed by the same function over the same rows but with a `bundleHash` that excludes the four keyed fields (`valueHash`, `defaultHash`, `bindParts`, `bindKeyCheck`; `BUNDLE_HASH_EXEMPT` is the existing hook). The probe records it and `check_probe_snapshot` compares it; the stamped `snapshotId` stays as is for the engine/answer stamp (C5). Feasibility is shown here: that exact exclusion makes all 1042 units equal across t1/t2/t3 (§2), and the keyed fields feed only binding (`Binder`, `bindKeyCheck`) and stored value hashes, not the dev-gold items' answers (ids are keyless: `ConfigEntry` ids equal across keys). Caveat for the owner to settle before relying on it: confirm that no answer path depends on a key-dependent value (a `bind-key-mismatch` demotion happens only within one build; `BOUND_BY` count 112 in run 2 was not compared to t3's, since the t3 store is gone).

Cost of (b): `gate.py` / `gate_run.py` / `snapshot.py` are pinned in the preregistered record (`harnessSha256`, `ASSEMBLY_CODE` feeds `codeHash`), so the change reopens the prereg record, needs a new dev rehearsal and a new probe on the new harness, and is a decision for `teco` and the user, not a unit this RCA authorises. The held-out list is not spent by any of this.

## 6. Suggested fix and prevention

Owner: an implementer (`tdd-engineer`) for (b), after the stakeholder decision on re-pinning; the re-pin and the rehearsal are `teco`'s.
1. Reproduction test first: build the same fixture bundles under two different `keys_root`s (`atrest.hmac_key`) and assert (i) the existing `snapshotId` differs (documents the cause) and (ii) the new content id is equal. Planted mutants that must turn it red: leave one keyed field in the content hash (equality fails); also drop a non-keyed field such as a unit's `builtTree` from it (a changed commit must still change it).
2. `check_probe_snapshot` compares the content id; a probe from another commit or code state must still be refused (the `codeHash`/`packSetHash` stay in the content id).
3. Replace the refusal text; "rehearse again on this build" cannot be done.
4. Prevention: a test that two builds of one fixture under different keys agree on the content id, so the plan's determinism clause is stated and tested as "minus the keyed hashes".

## 7. Open questions

- Does the user accept re-opening the prereg pins for (b)? If not, no held-out run can pass the probe check, because (a) and (c) need the same pin change.
- Run 2's store is stranded with the live graph; the list is not spent. A new empty store and the rehearsal on the new harness are required before the next attempt (no teardown was done here).
