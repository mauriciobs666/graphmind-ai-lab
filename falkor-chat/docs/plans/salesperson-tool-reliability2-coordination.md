# Salesperson tool-orchestration reliability — round 2 (K-061) — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-061 (post-M6, not a milestone gate)

Successor to `docs/plans/salesperson-tool-reliability-coordination.md` (archived 2026-08-31,
K-057 + K-058 closed) — ordinal-bumped per root `AGENTS.md`'s collision rule 5 (same kind/topic/
role, `-coordination`, already executed against). Picking up **K-061** (`docs/BACKLOG.md`,
filed 2026-08-31): the K-057+K-058 combined regression pass (U6 of the prior coordination) found
`salesperson@v5` sometimes silently duplicating its own current-turn, legitimately-mentioned
`add_to_cart` call (2/6 reps), plus a related but distinct false "not found" reply despite a
successful add (1/6 reps) — both filed together under K-061 since they share the same repro
shape and were found in the same n=6 pass, with a note to split if either is picked up
independently.

**Prior art (read before dispatching or picking this up cold):**
- `docs/BACKLOG.md` K-061 — the filed item itself: why it exists, why it's distinct from K-058's
  guard (deliberately can't catch a same-turn, legitimately-mentioned duplicate — blocking that
  would incorrectly block a customer's genuine "add another one"), the unproven
  two-consecutive-held-rejections contributing observation, and the suggested test strategy.
- `docs/test-reports/salesperson-tool-reliability-regression-report.md` — U6's report, Defect 1 —
  the actual repro steps, runIds (`rep-2`, `rep-4`, `rep-6`), and ground-truth Cypher/TraceEvent
  evidence for both symptoms at n=6.
- `docs/reviews/salesperson-tool-reliability-ml.md` — the living cross-thread diagnostic note
  (`Status: active`, deliberately never archived — it's the ongoing home for this whole
  investigation thread's sections, currently through §11). New diagnosis for K-061 extends it as
  §12, in place — not a new document (same rule that kept it open through K-057/K-058).
- `server/falkorchat/executor.py` — K-058's shipped guard (`_handle_tool_call`,
  `_target_mentioned_in_turn_text`) for reasoning about why it's structurally unable to catch
  K-061's same-turn variant, and about K-061's own note that this may share a root cause with
  K-059's upcoming `place_order` guard-design work.

**Scope discipline (carried from the prior coordination):** diagnosis first, larger n, before any
fix attempt — this thread has twice shipped a wording guess that didn't hold (K-057's reverted
second iteration) or wasn't warranted by more evidence (K-060 still awaiting root-cause). K-061's
own BACKLOG entry explicitly asks for a rate estimate before a fix shape is chosen.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `data-scientist` | `a4bfd0c3e6bcbf81c` | delivered | `docs/reviews/salesperson-tool-reliability-ml.md` §12 | — → — | 184.5k tok / 83 tools |
| U2 | `tdd-engineer` | `a98eb2cd64adc9f5d` | delivered | `server/falkorchat/executor.py` same-turn write-dedup fix | `analyst` → — | 135.5k tok / 46 tools |
| U3 | `analyst` | `a0b7ab4bc76ae25ad` | delivered | `docs/reviews/salesperson-tool-reliability2-impl.md` | — → approve w/ suggestions | 115.3k tok / 45 tools |
| U4 | `tdd-engineer` | `a98eb2cd64adc9f5d` | in-flight | mutation-coverage test + rename (analyst MAJOR 1 / MINOR 1) | — → — | — |

## Notes

- Single-unit start, same shape as the prior coordination's own U1 (K-057 diagnosis) — that unit
  was verified directly by `teco` rather than dispatched through a separate `analyst` gate, since
  it produces no code change, only a diagnostic note with re-checkable stats/ground-truth. Same
  plan here: `teco` independently re-verified U1's Wilson CIs, ground-truth Cypher, and code refs
  (all reproduced exactly) before accepting.
- **U1 delivered and verified 2026-08-31.** Symptom A (same-turn `add_to_cart` duplicate) confirmed
  at n=24 (pooled 5/30, 16.7%, CI 7.3-33.6%) — real, worth a fix now. Symptom B (false "couldn't
  find" reply) did not reproduce at n=24 (pooled 1/30, 3.3%, CI 0.6-16.7%) — too rare for its own
  track. A new, previously-unflagged text defect (mischaracterized hold-reason, 2/24) was found
  opportunistically and filed separately as **K-062** (not in this coordination's scope). K-059
  shared-root-cause question left open, suggestive not resolved — flagged for whoever picks up
  K-059 next. Findings folded into `docs/BACKLOG.md` K-061 (rewritten in place) and the new K-062
  entry by `teco`, not by the delegate (per this repo's BACKLOG-ownership convention).
- **U2 dispatched 2026-08-31** (`tdd-engineer`): same-turn write-dedup fix for K-061's confirmed
  Symptom A, per the BACKLOG entry's now-updated `Owner`/test-strategy lines. Gated by `analyst`
  on delivery, same double-gate discipline as K-057/K-058.
- No parallel dispatch risk yet (single unit in flight); if further live-eval work is dispatched
  alongside it, re-apply the prior coordination's shared-`reference`-graph sequencing rule before
  assuming disjoint files are enough.
- **U2 delivered and independently re-verified by `teco` 2026-08-31** — diff read in full, all 3
  new tests re-run in isolation, mutation test (RED-without-fix, GREEN-with-fix) independently
  reproduced from a byte-identical copy of the pre-fix file, full offline suite re-run personally
  (2305 passed, 14 deselected — matches the delegate's own report exactly), shared state
  (`reference`/`ws:acme`) re-verified `OK` after re-seeding. Committed as `381c9fc`.

### Operational note (2026-08-31) — a `teco`-caused git-tree mishap during independent verification, self-corrected

While reproducing the mutation test, a chained shell command (`cd server && git stash push ... &&
pytest ... ; git stash pop`) was issued from a cwd that was *already* `server/` (the tool's working
directory persists across calls, unlike a fresh shell) — the `cd server` step failed silently, the
`&&`-chained `stash push`/`pytest` never ran, but the trailing `; git stash pop` ran anyway
(unconditional `;`, not gated on the preceding failure) and popped a **pre-existing, unrelated**
stash entry from an earlier session ("K-028 in-flight work from killed Fable run"), producing a
merge conflict that staged/modified two files with no relationship to this unit
(`server/falkorchat/repository.py`, `docs/plans/workflow-timers-coordination.md`). Caught
immediately via `git status`; both files were cleanly restored to `HEAD`
(`git restore --staged --worktree`) with no loss — the old stash itself was never dropped (`git
stash list` confirmed it survived, conflicted applies are kept, not consumed) and remains exactly
as it was for whoever owns that unrelated work. This unit's own three files were unaffected
throughout (diff stat identical before/after). **Lesson for future verification steps in any
coordination:** never chain a `git stash pop` with `;` after a command that could itself fail —
use `&&` throughout so a failed `cd`/prior step aborts the whole chain instead of running a stash
operation unconditionally, and always re-run `git stash list` before *and* after any stash
operation used for verification purposes, not just `git status` after.
