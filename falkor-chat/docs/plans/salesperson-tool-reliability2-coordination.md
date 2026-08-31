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
| U4 | `tdd-engineer` | `a98eb2cd64adc9f5d` | delivered | mutation-coverage test + rename (analyst MAJOR 1 / MINOR 1) | `teco` (direct) → verified | 169.4k tok / 32 tools |
| U5 | `data-scientist` | `ae55c708663c46839` | delivered | `docs/reviews/salesperson-tool-reliability-ml.md` §15 | `teco` (direct) → accepted | 174.2k tok / 89 tools |
| U6 | `tdd-engineer` | `ad643318819a75668` | delivered | resolved-argument-set keying fix for `executor.py`'s K-061 guard | `analyst` → approve (1 MINOR, closed) | 122.0k tok / 58 tools |
| U7 | `analyst` | `a984f1bfe18d5f36d` | delivered | `salesperson-tool-reliability2-impl.md` Pass 2 | — → approve | 91.5k tok / 36 tools |

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

- **U3 delivered and independently re-verified by `teco` 2026-08-31** — code-line references
  (`executor.py:1003-1042`), the "full argument set" keying claim, and the suite-count claim all
  reproduced exactly against the source. Verdict: approve with suggestions.
- **U4 dispatched 2026-08-31** (same `tdd-engineer` agent, resumed via `SendMessage` by its
  recorded agentId, not a fresh dispatch): closes `analyst`'s two findings (MAJOR 1's genuine
  mutation-testing coverage gap on the "must not poison the dedup set on a failed dispatch"
  invariant; MINOR 1's misleading test name). `teco` handled MAJOR 2 (the BACKLOG/HISTORY
  documentation gap) directly — not this delegate's scope.
- **U4 delivered and independently re-verified by `teco` 2026-08-31** — `executor.py` confirmed
  byte-identical to the shipped fix (no drift); reproduced the MAJOR-1 mutation independently
  (moved the dedup-set seed to before the dispatch attempt) and confirmed the new test catches it
  while the other 3 same-turn tests stay green; restored and reran clean; full offline suite
  re-run personally (2306 passed, 14 deselected — matches exactly); shared state re-verified `OK`
  after re-seeding. Committed as `381fdb8`.

## 2026-08-31 — U4 close-out (superseded by U5's finding below — kept for history, not re-litigated)

Both of `analyst`'s review findings were closed (U4). K-061's own filed test-strategy still had one
item outstanding at that point — the live n≈20-30 regression pass — deliberately left open rather
than run in this coordination's earlier pass.

- **U5 dispatched 2026-08-31** (`data-scientist`, fresh agent, per the outstanding test-strategy
  item above): live n≈25 regression pass reusing `ml.md` §12.1's exact 3-turn script, ground-truth
  via `Cart`/`CartItem` Cypher + raw `TraceEvent`, comparing against the pre-fix pooled 16.7% (CI
  7.3-33.6%); also instructed to opportunistically screen for K-062's pattern per its own filed
  test-strategy note.
- **U5 delivered and independently re-verified by `teco` 2026-08-31.** All 6 distinct Wilson 95%
  CIs recomputed from scratch and matched exactly (1/25→4.0% CI 0.7-19.5%; 0/25→0.0% CI 0.0-13.3%;
  8/25→32.0% CI 17.2-51.6%; 10/49→20.4% CI 11.5-33.6%; plus the two pre-fix figures reproduced
  unchanged, 5/30 and 2/24). Every code citation behind the rep-20 mechanism claim re-read against
  source and confirmed exactly: `executor.py`'s `dispatch_key = (call.name, _dumps(call.arguments))`
  keying line, its own docstring's deliberate "different quantity, both must dispatch" carve-out
  (distinct from what rep-20 shows), and `tools.py`'s `add_to_cart` schema (`"required":
  ["productName"]`) plus its wrapper-level `arguments.get("quantity") or 1` default applied
  *after* the guard's key is computed — the exact mechanism by which two same-intent, same-turn
  calls (one omitting `quantity`, one supplying the same default value explicitly) produce two
  different dedup keys and both dispatch. `ws:ds-k061-regression` confirmed torn down (absent from
  the live graph list, probed directly); `reference`/`ws:acme` confirmed present, `ws:acme`'s
  `Cart`/`CartItem` state probed directly (no anomaly). Diff confirmed scoped to exactly this
  section (196 insertions, one file, `docs/BACKLOG.md` untouched by the delegate as instructed). A
  durable environment fact (schema-default-vs-dedup-key interaction, plus a separate id-reuse
  gotcha for future live-probe scripts) confirmed written to `kaizen_team` for `cobb` to triage.
  **Verdict: diagnosis accepted as delivered — and it changes K-061's status.** The shipped fix is
  real and substantial (16.7%→4.0% point estimate) but does **not** fully close K-061: a narrower,
  distinct loophole in the guard's own argument-set keying remains, with a named (not yet
  implemented) candidate fix. Findings folded into `docs/BACKLOG.md` directly by `teco`: K-061
  rewritten in place (still 🟡 in-progress, not closed — a follow-up fix unit is now warranted,
  owner `tdd-engineer`) and K-062's severity assessment revised upward (still 🔵 proposed, but no
  longer framed as low-severity/opportunistic-only — pooled 20.4%, CI 11.5-33.6%, well above its
  original 8.3% estimate). This coordination doc stays `active`; U5's own finding is itself the
  next unit's trigger, not a close-out — see the ledger for whatever follows.
- **U6 dispatched 2026-08-31** (`tdd-engineer`, fresh agent): reproduction test for §15.2's exact
  rep-20 shape, then a resolved-argument-set keying fix for the K-061 guard. Brief explicitly
  flagged the `remove_from_cart` design nuance (omitted `quantity` means "whole line," not an
  implicit default — must stay a distinct dedup key from any explicit quantity) so the fix
  wouldn't over-generalize into a new bug.
- **U6 delivered and independently re-verified by `teco` 2026-08-31.** Diff read in full:
  `executor.py` adds a per-tool `_DEDUP_ARG_RESOLVERS` table (not a generic schema-default
  lookup) — `add_to_cart` gets a resolver mirroring its own `run()`'s `arguments.get("quantity")
  or 1` collapse; `remove_from_cart` deliberately gets no entry, falling through to raw
  arguments unchanged, preserving its "omit = whole line" vs. "explicit quantity" distinction.
  Both new tests (`test_executor_agent.py`) re-run in isolation, green (6/6 same-turn dedup
  tests). Mutation independently reproduced from a `cp`-backed-up copy (not `git stash`, per
  this coordination's own U2 operational-note lesson): reverted just the `dispatch_key` line to
  the raw pre-fix form, confirmed the new repro test failed for the predicted reason (both calls
  dispatched) while the other 5 tests — including the new `remove_from_cart` distinctness test —
  stayed green; restored, `md5sum`-confirmed byte-identical, re-ran green. Full offline suite
  re-run personally: **2309 passed, 14 deselected** — matches exactly (+2 over the 2307 baseline,
  precisely the two new tests). Shared state re-verified `OK` after re-seeding
  (`bootstrap_schema.sh acme` → `seed_demo.sh` → `seed_workflows.sh` → `seed_catalog.sh` →
  `seed_salesperson.sh` → all three `verify_*.sh acme` reports `OK`). `HISTORY.md` entry
  independently confirmed accurate against everything re-verified above. Not yet committed —
  held pending `analyst`'s review gate (U7), same double-gate discipline as the original K-061
  fix (U2→U3→U4).
- **U7 dispatched 2026-08-31** (`analyst`, fresh agent): review of U6's diff before it's
  committed.
- **U7 delivered and independently re-verified by `teco` 2026-08-31.** Verdict: approve, no
  blockers/majors, one MINOR (the `_DEDUP_ARG_RESOLVERS`/`_WRITE_TARGET_ARG` guardrail gap for a
  future write tool). Read the Pass 2 diff/section in full: the resolver-mirrors-`run()` claim,
  the `remove_from_cart` omission rationale, and the double-direction mutation testing (revert the
  fix; separately plant a wrong `remove_from_cart` resolver) all independently re-confirmed
  against source. **MINOR 1 closed directly by `teco`** (genuinely trivial single-file no-brainer
  — a one-line-plus-context code comment on `_WRITE_TARGET_ARG` pointing future editors at
  `_DEDUP_ARG_RESOLVERS`, no design judgment, no behavior change): re-ran the 6 same-turn dedup
  tests green after the comment-only edit. Full offline suite and shared-state re-verification
  from U6's own delivery still hold (comment-only change since then). Findings folded into
  `docs/BACKLOG.md` K-061 directly by `teco`: fix + review recorded, MINOR closed, still 🟡
  in-progress — a final live n≈20-25 confirmation pass (same closing discipline as the first fix)
  is the one remaining open item, flagged as lower-urgency given the strong unit-level/mutation
  evidence already in hand. Ready to commit: `docs/HISTORY.md`, `server/falkorchat/executor.py`,
  `server/tests/test_executor_agent.py`, `docs/reviews/salesperson-tool-reliability2-impl.md`,
  `docs/BACKLOG.md`.
