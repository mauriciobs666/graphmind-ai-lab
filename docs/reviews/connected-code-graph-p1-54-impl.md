# Connected code graph P1-53b + P1-54 — implementation gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-53b, P1-54 (ccg P1)

## Scope & verdict

Diff gate on two commits: P1-53b `d8c5c815` (three fixture edits, regenerated `PINNED_SHAS`, relaxed `sub` branch of the pin's `edited_root`) and P1-54 `5e022161` (`cpg/ccg/linker/bind.py` `_ENV_READ_TARGETS` / `_is_env_read`, `tests/test_linker_bind.py`, `tests/test_linker_bind_real_shapes.py`, `tests/data/p1_54_environ_reads.json`, README row), against `docs/plans/connected-code-graph-p1-linking.md` P1-53b / P1-54. Root `REVIEW.md` applied; `cpg/REVIEW.md` does not exist.

**Verdict: approve with suggestions.** No blocker. The recognition fix is correct, the golden data is consistent with real extractor output, and every mutant I ran was killed.

CPG: considered, not relevant — the question is a diff-scoped predicate over bundle records plus a grep of `cpg/ccg/` and `cpg/packs/`, answered by running tests and mutants; `cpg_*` graphs describe other components.

**Observed runs.**
- `cd cpg && .venv/bin/python -m pytest tests/test_linker_bind.py tests/test_linker_bind_real_shapes.py tests/test_fixture_binding_offline.py -q` gave 186 passed.
- `tests/test_fixtures.py` gave 78 passed, 1 failed (`test_p1_form_equals_the_plan_table_and_covers_every_flow_dependent_row`). The working tree carries uncommitted edits by the in-flight 53c unit (`fixtures/system1/expected/fx-svc.yaml`, `peer.py`, `compose.example.yaml`, `tests/test_fixtures.py`), so I attribute this red to 53c (inferred, not isolated: I could not run HEAD cleanly without touching the tree). It is not on the known-reds list, so 53c's gate should confirm it is its own.
- Mutants (copy of `bind.py` kept aside, restored afterwards; `git diff --stat -- ccg/linker/bind.py` empty): all killed, with the killing test shown.

| Mutant of `bind.py` | Killed by |
|---|---|
| drop `os.py:<module>.environ.get` | `..._binds_the_compose_set_value[shape_i]` |
| drop `os.py:<module>.getenv` | `test_a_url_the_compose_file_sets_..._lnk_13` |
| drop `os.py:<module>.<member>(environ).get` | `test_a_default_url_read_..._lnk_15` |
| add `__builtin.dict.get` | `..._binds_nothing[shape_e]` |
| add `...(environ).setdefault` | `..._binds_nothing[shape_h]` |
| predicate `startswith("os.py:<module>.")` | `[shape_h]` |
| predicate back to `calleeName in ("get","getenv")` | `[shape_e]` |
| predicate `externalTarget.endswith(("get","getenv"))` | `[shape_e]` |

## Findings

### M1 (minor) — the "no other `calleeName`-keyed record predicate" claim holds, but nothing guards the golden data against extractor drift

*Verified by sweep.* An unfiltered `grep` for `calleeName`, `environ` and `getenv` over `cpg/ccg/` finds no other env-read predicate. The remaining `calleeName` consumers are:
- the pack matcher (`packrt/match.py:180`);
- the `mcp` mention rule (`linker/link.py:251`);
- graph/render plumbing.

Every `callee:` in `cpg/packs/*/pack.yaml` is a bare member (`get`, `include_router`, `callTool`, `publish`, …), with dotted forms carried in `externalTarget` clauses (`python-http-client` pins `requests.py:<module>.get`). They are therefore already in the real shape, and no dotted `callee:` exists to be dead. No same-class defect found. The sweep covered `ccg/` and `packs/`, not the `extractors/` tree.

The residual is that the census (`test_census_*`) proves the table matches records in a **static** golden file. If the Joern/extractor naming changes (for example `<member>(environ)`), the golden stays green while the live predicate goes dead again. This is the P1-17 defect class, one step removed. Suggestion: carry one live-tier assertion (`-m live`, fixture build) that the committed `fx-svc/clients/jobs.py` callsite's `externalTarget` is in `_ENV_READ_TARGETS`. P1-53c's live tier is the natural home; it needs no new infrastructure.

### M2 (minor) — golden provenance is asserted, not reproducible

`tests/data/p1_54_environ_reads.json` is real-shaped: all 22 callsite ids equal `ids.callsite_id(symbolId, calleeName, ordinal)` (I recomputed them), and the shapes carry the observed `<member>(environ).get`, `.get.rstrip`, `.setdefault`, `__builtin.dict.get` targets. I cannot prove it came from a Joern run: the only provenance is the `source` string, and there is no regeneration script or recorded command. Hand-built records could also satisfy the id check. Suggestion: record the extraction command and the scratch commit in the JSON's `source` (or in `tests/data/README`), so a reviewer or 53c can regenerate it.

Separately, `world()` builds the lifted URL fact synthetically (`url_lifted(http["id"], "C", "\x00" + path)`). The records are real, but that one fact is hand-built. This is acceptable, because the lifting has its own real-fact tests, but state it in the module docstring (which currently says the inputs are "never hand-built").

### M3 (minor) — relaxed `sub` branch: weakens nothing the pin relies on, but loosens a typo guard

`edited_root` now asserts `old in text or new in text` (`tests/test_fixture_binding_offline.py:130`). Traced:
- The as-committed tree is pinned separately by `test_the_fixtures_as_committed_meet_the_binding_contract`, which runs on `base_root` with no edits. Reverting a fixture therefore still reddens that test, and the relaxation cannot hide an unedited fixture.
- The `change CMD module` mutant's `old` (`api.app:create_app`) is present in the conforming tree, so the mutant is applied for real, and its test passes with the required message (186 passed).
- What is lost: a mistyped `old` that happens to equal text already in the tree now passes silently as a no-op, instead of failing loudly. For a mutant, that would mean a test asserting red on a no-op edit, which fails rather than lies, so the damage is low.

Suggestion: make the idempotent branch explicit, with `old in text` required for MUTANTS and `old in text or new in text` allowed only for `CONFORMING_EDITS` (e.g. a keyword `idempotent=True`). Not required for the gate.

### M4 (minor) — 53b done-conditions evidenced only by delegate figures

The "exactly five rows differ" claim (LNK-16 `port-shared`; LNK-45 `BOUND_BY` 0/1; LNK-02 and LNK-06 to `binding-confirmed`; LNK-12 loses its `job` link and its `svc` link becomes `binding-confirmed`) is a live-tier (Joern/FalkorDB) before/after multiset diff. I could not reproduce it offline.
- What I verified: the `PINNED_SHAs` diff touches exactly two repos (`fx-job`: 5 commits and the branch; `fx-svc`: 5 commits and 4 branches); `fx-lib` and `fx-web` rows are untouched. 30 pinned values changed in total, consistent with a content change reaching H0 of the two edited repos.
- What I verified for the rows individually: the offline pin (green) asserts LNK-45, 16, 19, 12, 02, 06 and AC6-EX-4 against the conforming tree.
- Not verified: that no *other* row changed. The offline pin does not enforce it, and the fixtures live tier I did not run.

P1-54 done-condition (c) (before/after over 1233 stored bundles, 1578 links unchanged) is likewise only in the coordination ledger, with no artefact in the repository. Both figures are the delegate's and the coordinator's, not mine.

### N1 (nit) — docstring and README match the code

The `bind.py` module docstring and comment block, and the README row, agree with the table. `os.getenv` and `from os import getenv` do share the target `os.py:<module>.getenv`: shapes d and j carry the same string in the golden data. The `_ENV_READS` constant and the suffix test are gone, and `test_the_table_is_exact_strings_not_a_suffix_or_glob` asserts the constant is absent. The README `Tests:` enumeration does not list `tests/test_linker_bind_real_shapes.py` (it appears only inside the `bind.py` clause), which is fine.

The docstring's "No other form of environment read is in any bundle record" is a pre-existing claim. It is not tested for `os.environ["X"]` or aliased imports (`import os as o`). I did not verify it either way.

## What's solid

- `_is_env_read` is a one-line exact-membership test on `externalTarget`; the `calleeName` suffix test is deleted; `.get(...)` chains and `dict.get` are excluded by construction.
- The census is a real check: it fails if a table entry is not in the golden data (mutants 1 to 3 and the widening mutants confirm both directions), and its expected sets are written independently of the constant.
- `shape_e` (`dict.get` reading the same variable name as a Compose entry) is the right negative control: only the target distinguishes it.
- The relaxation in `edited_root` leaves the as-committed pin intact.

## Open questions

None blocking. For the coordinator: is a live-tier assertion for M1 wanted in 53c's scope or deferred?

### Proposed REVIEW.md lines

Target: `cpg/REVIEW.md` (new) under `## Always check`; source: plan P1-54 "Defect-class lesson" and findings M1/M2 here.

- A Binder or pack predicate that reads a bundle record shape has one test whose input is a record extracted from a real build, with the extraction command and commit recorded beside the data; and a census assertion that the predicate matches at least one real record. A census over a static golden file is paired with one live-tier assertion on the committed source file, so extractor drift reddens something.
