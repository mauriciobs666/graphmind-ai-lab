# Review: work-items N5 (public helpers) and P9-2 (rebuild guard vs. reseed) — HEAD a8c7fbe1

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Scope & verdict

Reviewed `git show HEAD` under `skills/work-items/` (check.py, headers.py, rebuild.py, store.py, tests) against `docs/plans/work-item-numbering.md` §3.1/§3.7 and live spec `skills/work-items/docs/SPEC.md` §8. Root `REVIEW.md` read and applied (the spec-update and growth-stops checks do not apply: no spec or plan text changed).

**Verdict: approve with suggestions.** No blocker or major. Suite run from `skills/work-items`: `512 passed`.

CPG: considered, not relevant — a ~120-line Python diff in one small skill with no loaded `cpg_` graph. Callers were swept by unfiltered grep.

## Findings

### 1. [minor] The N5 guard test is narrower than the claim it names (verified by reading the regex)
`tests/test_check.py::test_check_references_no_private_name_of_another_module` checks only `check.py`, and only the attribute pattern `(store|headers|rebuild|alloc|events|project|rules)._x`. An alias (`s = store; s._reconcile()`) or a `from .store import (\n _git)` multi-line import passes it. A private reach remains in the same package: `rebuild.py:15` does `from .store import REGISTRY_REL, Store, _git`. It is outside the N5 brief (check.py), but "no private reaches" is not true repo-wide. Suggestion: either make `_git` public (`git`), or state in the test docstring that scope is check.py and the listed receivers only. If you want it general, walk the AST of every `wilib/*.py` for `ImportFrom` names and `Attribute` nodes starting with `_` on a non-`self` module or class name.

### 2. [minor] `wrote N events` is correct as a registry total, but the wording is ambiguous (verified by running)
In the stale-tip scenario I ran `wi rebuild --write` on a copy. The output was `wrote 7 events`, and journal and tracked each held 7 (I01, I02, F0001–F0005). The count `len(hist)+len(evs)` equals the final registry, because the reseed already wrote the skipped hist events into the journal. So it does **not** overstate the total. It does overstate what `rebuild` itself appended (here 4 of 7). Proposed: leave the count, or say `registry now holds N events`. No test pins the line.

### 3. [minor] Spec §8 wording no longer matches the guard (inferred from the diff)
SPEC §8 says `--write` "is refused when the registry already holds items". The code now refuses when either on-disk copy holds any event (`lost = not tracked and not journal`). Plan §3.7 says "non-empty registry", which matches. The two differ only for an event-bearing copy with zero items, which is not realistic. Suggestion: at feature close, have §8 say "when either copy is non-empty; a ref-tip copy alone (reseed) does not count". The error text still reports `len(tx.state.items)`, which is fine.

### 4. [nit] No test combines a stale tip with header-derived items or `--floor`
`build_events` and `floor_events` run on `hist_state` only, outside the lock. The reseeded events are a subset of `hist` (both read the same path on all refs), so I found no divergence by reading. But `test_rebuild_*stale_tip*` covers only history-only recovery. Optional: one row with `--from-headers` over the `stale_tip` fixture.

## Checks (what I verified)

**(1) N5 is behavior-identical.**
- `check()` previously did `with store._locked(): union = store._reconcile()`. It now calls `read_union()`, which holds the same body. The `_read`→`read_text` rename is an identical body. `read_tracked` and `read_journal` are the old `events.read_events(_read(...))` expressions, with the same `journal_path is None` guard.
- Unfiltered sweep of `skills/work-items` (py, md, sh) for the old names (`_read(`, `_locked`, `_reconcile`, `_family_files`, `_owner`, `_header_index`, `_tokens`) found no stale caller outside `store.py`, which uses `_locked`/`_reconcile` internally. `SKILL.md` and the tests mention none of them. No code outside `skills/work-items` imports `wilib`.

**(2) P9-2.**
- `lost` is computed inside `Store.transaction()`, after `_locked()` is taken and before `_reconcile()` reseeds. It reads exactly the two copies. The `journal_path is None` case gives `read_journal() == []`, so `lost` reduces to "tracked empty", which is correct.
- A live registry is still refused when the journal is non-empty, or when the tracked copy alone is non-empty. Both are pinned by tests, and I confirmed each by mutation (below).
- No race inside the flock: the check and the reseed are under one lock. The only unlocked reads are `history_events` and `collect`, which are git and doc reads done before the lock, as before.
- Final state, run on a scratch copy: after a stale tip with history, the journal had no duplicate evids, sorted by `ts`, and tracked equals journal. `wi check` showed only the informational `rebuilt` finding. The next `alloc.new F` gave `GAL-F0006`, the history maximum, not the stale tip's.
- Dedup: `tx.evids` is taken from the deduplicated union once, before the loop. `hist` is already unique by evid, so a single snapshot is sufficient.

**(3) Mutation results** (copies under the scratchpad only; each run `-x`):

| Mutant | Killed by |
|---|---|
| guard back to `if tx.state.items:` | `test_rebuild_applies_history_when_an_older_copy_survives_on_another_ref` |
| remove the evid dedup | `test_rebuild_over_stale_tip_leaves_no_duplicate_events` |
| `lost` computed after `_reconcile` | the "applies history" test |
| `lost = False` / guard `if False:` | `test_check_reports_rebuilt_...` / `test_rebuild_restores_the_tracked_copy_too_and_is_stable` |
| `lost` ignores tracked | `test_rebuild_still_refuses_when_only_the_tracked_copy_is_present` |
| `lost` ignores journal | `test_rebuild_refuses_when_only_one_copy_is_deleted` |

## What's solid
- Minimal, mechanical N5 with a real new-API test (`read_union` reseed semantics against a lost journal).
- The P9-2 fix puts the decision at the right layer (on-disk copies, under the lock, before the reseed), and every guard branch has a test that goes red when it is broken.
- Shared-state hygiene: all tests run under tmp git repos.

## Open questions
None blocking. Should `_git` be made public (finding 1) as part of N5, or left for a later cleanup?
