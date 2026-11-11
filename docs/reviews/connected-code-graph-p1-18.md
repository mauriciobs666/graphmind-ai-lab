# Review: P1-18 assembler (code gate)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed the uncommitted P1-18 build (`cpg/ccg/assemble/{__init__,rows,snapshot}.py`, `cpg/ccg/registry/flow-availability.yaml`, `cpg/ccg/sandbox_audit.py`, `cpg/ccg/schemas/sandbox-record-v1.schema.json`, `cpg/ccg/staleness.py`, `cpg/tests/test_assemble_rows.py`, `cpg/tests/slice/test_assemble_equivalence.py`, three README rows) against `docs/plans/connected-code-graph-p1-linking.md` P1-18, `-p1-graph.md` (G1-G13, §2, §3, §8), `-p0-graph.md` (D5/D6/D8, §1.1), `-egress-spec3.md` §5.3-§5.4, `-bundle-spec6.md`, the committed linker and the slice oracle `spikes/vertical-slice/load.py:build_rows`.

**Verdict: needs changes** (no blocker in the produced rows; three majors, all about unpinned or unowned behaviour, one of them a test red on the current tree).

CPG: considered, not relevant — `cpg_falkorchat` is the only code graph loaded; nothing indexes `cpg/ccg`, so the review read the code directly.

What I ran (all from cwd `cpg/`, `ulimit -v 3000000`, `timeout -s KILL`): `tests/test_assemble_rows.py` 87 passed earlier, 86 passed + 1 failed on the final tree (F2); `tests/slice/test_assemble_equivalence.py -m slice` 11 passed (it ran against the **real** `cpg/.audit/sandbox.jsonl`, which holds every id of both slice bundles: I confirmed with `check_sandbox_ids` directly). A wider run (`test_linker_bind`, `test_linker_link`, `test_atrest`, `test_bundle`, `test_packrt_generic`, assembler tests): 1585 passed, 1 failed (F2). `test_linker_bind.py::test_l1_...` now passes (the P1-44 fix landed during the review). Mutations ran in a scratch copy (`scratchpad/base`) under `ulimit -v 1500000`/`timeout -s KILL 45`; no tree edit, no write to `cpg/.audit`, no DB connection.

## Findings

### F1 (major, verified) — `LAUNCHES` is only partly sourced, and no step owns the rest

`-p1-graph.md` G8/§2 (the note is P1-18's contract) require one `LAUNCHES` edge per target the linker resolves: `build.context` ConfigEntry → Unit, command launch → script `File` → Unit, the `.mcp.json` script hop (`-p1-linking.md` P1-17 (6): "the `LAUNCHES` edge `command → script File → Unit` carrying the script hop"; the script hop has no `BOUND_BY` entry). `rows.py:743-751` writes `LAUNCHES` only from `facts.launch_facts` of `fact == "launchUnit"` (the pack runtime's `targetArg`). The committed linker returns `Link.evidence` with `BOUND_BY` entries only (`link.py:163-175`); `bind.py` holds the chains (`_Chain.unit`, `_launch_chains`, `_service_chains`) but exports no entry→target pair. So the Compose `build.context` → Unit edge, the script `File` edge and the Dockerfile hop have no source in the assembler; `File` as a `LAUNCHES` target is never written.

Judgement: not a P1-18 plan violation as written (Behaviours (1)-(11) and §8's P1-18 tests do not mention `LAUNCHES`; the assembler cannot invent targets), but it is a **linker follow-up (P1-17) that nothing owns**: the script hop's only graph home was to be this edge, so for AC-4 the hop is lost entirely, and P1-17's "Owed at close" list does not record it. Nothing reads `LAUNCHES` in P1 (V19), so there is no wrong answer today; the risk is a silent contract hole that ships. The README row and the `_launches` docstring also don't say it is partial.
Fix: (a) `architect`/`teco` add a P1-17 follow-up: the linker exposes launch targets per chain (`Links.launches`: `(entryId, "File"|"Unit", targetId)`, system-wide unit index); (b) P1-18 wires it plus a test per G8 row (Compose context into another repo, script hop); (c) until then record in `-p1-graph.md` G8 / the step that only `launchUnit` edges exist.

### F2 (major, verified) — the `ASSEMBLY_CODE` import-coverage test is red on the current tree

`test_the_assembly_code_list_covers_every_ccg_module_the_assembler_imports` fails: `Extra items … 'ccg/gitsafe.py'`. The working tree's modified `ccg/units.py` now imports the new untracked `ccg/gitsafe.py` (P1-44's hardening), and `units.py` is in `ASSEMBLY_CODE` (`snapshot.py:ASSEMBLY_CODE`). The guard did its job (positive control, in effect); the gate cannot call P1-18 green while it is red. Owner of the `gitsafe` change adds `"ccg/gitsafe.py"` in the same change.
Two follow-ups for the note (`-p1-graph.md` §3, graph-dba's "ASSEMBLY_CODE amendments", item 4): the implementer added `ccg/__init__.py`, `sandbox_audit`, `staleness`, `units`, `manifest`, `atrest` to the note's expected list. Verified: all six are in the `-I` import closure once `ccg.linker`/`ccg.packrt` are imported (`units`, `manifest`, `atrest` arrive through the linker; the guard imports them because the *test script* imports the linker, not because `rows.py` does), so the additions are required by the rule, and the note's list should be updated to match. Consequence worth stating: every edit of git plumbing or at-rest key code (neither changes a row) now changes `codeHash` and so `snapshotId`; accept it knowingly or have the linker's dependencies split out.

### F3 (major, verified by mutation and a drop-reason probe) — stated behaviours and 12 of 18 drop reasons are unpinned

Mutations that **survive** `test_assemble_rows.py` (each is a stated behaviour):
- `Snapshot.builtAt` over fresh units only (G: "maximum `builtAt` of the `fresh` units"): including `stale-kept` units' `builtAt` survives (`rows.py:477`). No test has a stale-kept unit with a later `builtAt`.
- O-9 tie-break "ties broken by lowest `siteId`" (`_check_link`, `rows.py:699-702`): reversing the site-id order and removing it both survive.
- `flowAvailable` of a **not-mapped** unit (G5): default `True` survives (`rows.py:459, 462`; the value is a `manifestHash` input).
- `Repo.headCommit` source (`repo_head`, fresh-first, first wins; `rows.py:422-427`): `setdefault` → assignment survives. The fresh-first policy is the implementer's, not the note's.
- `link-provider-missing` (item 6 below): changing the drop to a raise survives, i.e. no test reaches it.

Equivalent mutants (not findings): the `int(m[1]) != SUPPORTED_MAJOR` test (the schema's `enum ['1.0','1.1']` already refuses a major 2) and the `MOUNTED_BY` dedup key `(site, path)` vs `(path,)` (the edge identity already holds the call site).

Killed, as the brief asked: `ord` direction (reverse) and tie-break by line (ignored, ascending-line swapped, line-descending), D8 omission on each of `CALLS`/`DATA` and on `status` vs `reason` separately and the default constants, `candidateCount` on `Link` and on `VIA`, the audit's exactly-one rule (`<1`, `>1`, off), the `refused`/`exitCode` conditions each, the 124 boundary (`>`, `-1`, the cap constant), `HOST_LANGS` membership (drop `markdown`, drop `config`, add `c`, add `python`), the in-memory `minItems` patch, the record-vs-header `egressMode`, `_mask_key_path`, `to_unit` known, and the `Name→RESOLVES_TO` set.

The class is enumerable, so close it with a probe rather than a case list: an AST sweep of `b.drop("…")` literals in `rows.py` versus the reasons a test asserts in `rows.dropped`. I ran that probe with a hook on `_Builder.drop` across both test files: **not reached by any test**: `bound-entry-missing`, `calls-target-not-in-snapshot`, `consumer-site-missing`, `controlled-by-endpoint-missing`, `data-endpoint-missing`, `launch-unit-missing`, `lifted-hop-missing`, `link-provider-missing`, `mount-fact-missing`, `mount-path-no-kinds`, `provider-handler-missing`, `via-site-missing`. Ask: a test that parses `rows.py` for every drop literal and asserts a planted input produces exactly that reason with count 1, plus the five survivors above. The probe test belongs in a round that does not also edit `rows.py`.

### F4 (minor, verified) — `link-provider-missing` drops the whole `Link` silently (item 6)

`rows.py:620-624`: a link whose provider symbol is not in the snapshot is dropped (node, `VIA`, `LIFTED_VIA`, `BOUND_BY`, `ON`), counted. The oracle wrote `TO` regardless (a loader `MATCH` would skip the edge and leave a `Link` with neither `TO` nor `TO_UNIT`, which reads as a never-built target: worse). Dropping is the safer of the two, and a provider that is a `ConfigEntry` or an unknown `TO_UNIT` already **raises** here, so the inconsistency is the finding: facts and bundles come from one build, so a missing provider is an inconsistent input, not a data fact. Prefer raising `AssembleError` (ids only) like its two siblings, or keep the drop and require P1-20 to refuse or report a non-empty integrity subset of `dropped` (nothing in P1-20's text mentions `rows.dropped`). Either way F3's probe covers it.

### F5 (minor, verified) — `check_sandbox_ids` echoes a header value

`sandbox_audit.py:88`: `f"auditId {i!r}: not a lowercase UUIDv4"` puts an arbitrary header string into the message (ran: a planted `"SECRET-ish value http://u:p@h"` is echoed; `None`/`5` too). The module docstring claims "no record content", and P1-19 will log these. `classify_bundle` suppresses the text, so the assembler itself is safe; the standalone function is not. Name the index (`sandboxAuditIds[2]`), not the value.

### F6 (minor, inferred) — `classify_bundle` checks only the header

Item 3: `classify_bundle` validates version, header schema and audit ids; a malformed record body raises a raw `KeyError`/`TypeError` from `build_rows` rather than `BundleRefused`. The docstring says the full check is `bundle.read_bundle`'s, which is sound if every caller reads bundles through it (P1-20 will). Add one test that `build_rows` over a bundle with a record missing `id` fails with an `AssembleError` subclass or document "inputs are `read_bundle` output" in the `build_rows` docstring; no code change needed otherwise. (A raw `KeyError` message names a key, never a value.)

### F7 (minor, verified) — `bundle:bind-key-check` has no producer (item 2)

Correct that `classify_bundle` cannot emit it (no key; `Links.bind_key_mismatch` is the linker's channel). It is a member of `FAIL_REASONS` and no step maps it: P1-20's behaviours (1) say the reason is "the one code P1-44 (3) assigned", and P1-44 (3) lists none for a key mismatch. One line owed in P1-20 (or P1-44 (3)): `Links.bind_key_mismatch` → `bundle:bind-key-check`.

### F8 (nit, verified) — row order depends on the linker's site/evidence order

With each link's `sites`/`evidence` shuffled the `VIA`/`LIFTED_VIA` rows come out in a different order (same multiset, same `Snapshot`). With the **real** linker, reversed bundles/packs/facts/links give byte-identical `canonical_rows` (run end to end). Fine for a loader that creates edges; just don't claim identical rows for arbitrary `Link` tuples. Records reversed inside a bundle also change row order (bundles are canonical, so not an input variant).

### Items 4, 5, 7 from the brief

- **(4) `ASSEMBLY_CODE`:** see F2; amendments are required by the rule, not drift.
- **(5) `MOUNTED_BY`:** empty prefix `""` → `/`, a missing leading slash is added, then `mask_http_segments(…, PROVIDER)`; several prefix alternatives give one edge each (`key=(site, path)`), `ord` by `(mountPath, line, site)`. Tests kill reversal, the line tie-break and the empty prefix. **P1-19 must create these edges (not `MERGE` on `(s,t)`)**: two edges between one `Symbol`/`CallSite` pair differ only by `mountPath`/`ord`.
- **(7) `Repo` holding only `repoId` for not-mapped units:** acceptable. No reader in `cpg/ccg` or `clients` reads `headCommit`; G4 gives a not-mapped unit no commit field. One line in G4 or §1 should say a repository with no built unit carries only `repoId` (the graph note's `name` was never written by the slice either).

## What's solid (verified)

- **Equivalence is honest.** Recomputed independently (my own script, not the test): oracle 38,121 nodes / 90,867 edges (matches `-p0-graph.md` §1.1); assembler, `http`-only, 38,124 / 90,966. Group by group the only deltas are `Pack` 0→3 and `MOUNTED_BY` 0→99; node ids equal in all 11 shared labels; node properties equal except `Unit.flowAvailable` and `bundleSchemaVersion` (2 units); edge `(s, t, props)` multisets equal in every oracle group except `PROVIDES` (2 vs 2 rows). `PROVIDES` statuses 57 confirmed / 4 possible vs the oracle's 59 / 2, the two weakened providers being the ones `test_packrt_conformance.py` documents; `CONSUMES`, `Link` and `VIA` identical. Full packs: 38,321 / 91,630 with `dropped == {"consumer-key-none": 3}` only.
- **Strictness matches the real linker.** I wrapped `ccg.linker.link` over the 282 linker tests: 239 real `Link`s (82 with `BOUND_BY`, port/pack-default/Compose chains) all pass `_check_link` (O-9, port evidence, `siteId` membership).
- **Audit check.** Schema admits all 591 real records (0 errors), rejects `lib-launch`; exact-one, 124 refused with the cap comparison, partial last line and garbage handled; every real-slice id passes against the real file. `classify_bundle` returns only a code (planted exception text and `bindValue` appear in no code or `BundleRefused` message; the planted `Evil` path-like object is caught).
- **No DB or network:** an unfiltered grep over `ccg/assemble`, `sandbox_audit.py`, `staleness.py` for `falkordb|redis|socket|requests|urllib|subprocess|connect(|GRAPH.|open(…w)|write_text|mkdir` finds only `os.environ.get("CCG_AUDIT_DIR")`; the only file access is reading the registry, `sandbox.jsonl` and pack/code bytes.
- Determinism (real linker, reversed bundles and packs): identical bytes and `Snapshot`.

## Open questions

1. `docs/plans/connected-code-graph-bundle-spec6.md` §2 puts `HOST_LANGS` in `ccg/bundle.py` (imported by `sandbox_audit`), adds `attestation(...)`, and asks `rows.py` to record `unattested_units`; none of it is in this diff (the brief makes the in-`sandbox_audit` constant a stop-gap). Who builds `unattested_units`: P1-18 follow-up or the P1-44a/spec6 builder? Until then a host `[]` bundle is indistinguishable from an attested one in assembler output, and the existing `test_host_langs_...` imports from `sandbox_audit`, so moving the constant needs a re-export or a test edit.
2. F1's owner: `teco` to route the linker launch-target export to `coder` or `tdd-engineer` under P1-17 (`architect` to confirm the shape).

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, "Always check". Source: F3 (12 of 18 drop reasons never reached, found only by an instrumentation probe).

- A guard or step that counts or drops by named reason (a `dropped[...]` counter, a refusal code, a failure vocabulary) is checked by sweeping the literals in the source and showing a planted input reaches each one; a reason no test reaches is a finding, not a bound.
