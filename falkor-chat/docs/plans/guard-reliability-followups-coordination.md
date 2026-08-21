# Guard reliability follow-ups — coordination log (K-027 items 4/5 + carried gate findings)

> **Status:** active · **Owner:** `teco` · **Tracks:** K-027 (post-M3 follow-up track, not a
> milestone gate)

Coordinator: `teco`. Started 2026-08-20, on the user's "let's work on K-027".

## Scope

`docs/BACKLOG.md` → `### K-027`. Items 1–3 are already delivered (2026-07-24 / 2026-08-16 /
2026-08-17 → `HISTORY.md`). This run picks up everything still open under that heading:

1. **Carried findings from the 2026-07-19 analyst gate** (`docs/archive/reviews/m3-guard-thread-context-impl.md`),
   recorded in `BACKLOG.md` so they "cannot rot" — none delivered yet:
   - **m-1** — `guards._is_negated`'s 12-char window misses cross-clause negation
     (`"The user did not say; more info is needed."`), which is a **false-advance** — the
     dangerous direction, and the opposite of what the code comment currently claims.
   - **m-2** — `guards._recent_turns` slices `thread[-n:]` **before** filtering malformed/empty
     rows, so malformed rows shrink the evidence window exactly when the judge is on its
     degraded fallback tier. Fix = filter first, slice second.
   - **m-3** — the judge's evidence tier (`understanding` vs `recent_turns`) is invisible in the
     trace — `_select_transition` traces `(transition, guard_text, verdict)` only. Calibration
     work (item 3, now delivered) needs results stratified by tier. Fix = return the tier on
     `GuardVerdict` and fold it into the `guard_judgment` trace payload — additive, no graph
     change.
   - **n-1** — function-local `import json as _json` in `app._render_judge_user` /
     `_build_llm_judge`; every other module imports it at the top.
   - **n-2** — the judge-prompt cap loop re-joins the whole message on each eviction — O(n²) in
     turn count (irrelevant at N=6, a test drives it with 50).
   - **Doc-drift** — the `_drive_loop` byte-identity lock is quoted as SHA `71055f756280` **+
     2844 bytes** at several doc sites. The SHA is correct and reproducible; the byte count is
     wrong (the extraction yielding that hash is 2860 bytes; a third figure, 2839, appears in an
     earlier coordination entry). Correct the figure wherever the lock is quoted, or drop the
     byte count and verify by SHA only.
2. **Item 4 — Golden-set expansion (D11).** `server/tests/eval/golden_guards.jsonl` has 26 rows,
   all one labeler's. A real FAR ≤ 10% bound needs ~30 `clear_suspend` cases at zero failures
   (≈50–60 total), and the item calls for a **second labeler** on the boundary tier.
3. **Item 5 — Ministral re-probe (D13 finding 2).** Ministral-3B beat Qwen at the terminal tool
   call (native `post_message` 3/3 in replay) but lost badly at judging (fence-fixed
   advance-recall 0.364 vs Qwen 0.818) — probed **before** the parse-robustness fix (item 1,
   now delivered) and before the terminal-post engine contract (item 2, now delivered). Both
   preconditions are now met; re-probe against the current code.

**Out of scope for this run:** K-028, K-029, K-030, K-032, K-033, K-035 — separate backlog items,
not part of K-027.

## Entry baselines

| Gate | Baseline (confirmed by `teco`, 2026-08-20) |
|---|---|
| `server` pytest (offline, `--collect-only`) | **1088/1091 collected, 3 deselected** (matches `HISTORY.md`'s last recorded run) |
| `git status` on `falkor-chat/` | clean — no uncommitted changes in this component |
| CPG | `cpg_falkorchat` (note: no hyphen, unlike the `cpg_<component>` convention elsewhere), built
  `2026-08-17T00:40:42Z`, **stale** — 3 commits since to `falkor-chat/server` (`60d6cd5`
  guard-judge calibration, `b6b9b53` must-post engine contract, `f207fbe` K-046/K-047), touching
  files this run's U1 will itself edit (`guards.py`-adjacent). Specialists should read
  `guards.py`/`executor.py`/`app.py` directly rather than lean on the CPG for structural claims
  in this area. |
| LM Studio | reachable at `localhost:1234`; loaded models include `mistralai_ministral-3-3b-instruct-2512`
  and `mistralai/ministral-3-3b` (two similarly-named entries — U3 must confirm which one the
  D13 probe actually used, or treat them as the same model under two catalog ids and say so). |

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `tdd-engineer` | `a8aa198b16a0fca53` | accepted | `guards.py`/`executor.py`/`app.py` fixes + doc corrections | `analyst` → approve |
| U1-G | `analyst` | `a806c0663f21d667d` | delivered | `docs/reviews/guard-carried-findings.md` | approve (0 blocker/major, 1 minor, 1 nit) |
| U4 | `data-scientist` (resumed U3) | `a7b3947d7e94941f9` | accepted | K-027 item 5 delivered marker + new K-048 filing in `BACKLOG.md`/`HISTORY.md` | — (doc-only, coordinator-verified) |
| U2 | `data-scientist` | `a1e7c37a0b3f991e7` | delivered | `docs/plans/golden-set-expansion-ml.md` | — (advisory design note; raises a user decision, see Log) |
| U3 | `data-scientist` | `a7b3947d7e94941f9` | accepted | `docs/plans/ministral-reprobe-ml.md` | `analyst` → approve |
| U3-G | `analyst` | `a62c2169d989ef370` | delivered | `docs/reviews/ministral-reprobe.md` | approve (0 blocker/major, 2 minor, 1 nit) |

U1, U2, U3 are **fully independent** — U1 touches `server/falkorchat/{guards,executor,app}.py` +
their tests + docs; U2/U3 are advisory notes under `docs/plans/`, no shared files with U1 or each
other (U3 may touch `server/tests/eval/` if it extends the existing calibration harness for a
second model — flagged in its brief to avoid colliding with any future U2 implementation).
Dispatched in parallel.

## Documentation-impact scan

| Doc | U1 |
|---|---|
| `docs/HISTORY.md` | entry required |
| `docs/BACKLOG.md` | mark K-027's five carried findings (m-1/m-2/m-3/n-1/n-2/doc-drift) as delivered inline; leave items 4/5 open until U2/U3 conclude |
| `falkor-chat/AGENTS.md` | only if a stated invariant changes (unlikely — additive trace field) |

U2/U3 are advisory notes, not yet implementation — no BACKLOG/HISTORY entry until their
recommendations are acted on.

## Log

- **2026-08-20** — coordination opened. Baselines confirmed. U1/U2/U3 dispatched in parallel.
- **2026-08-20** — **U2 delivered**: `docs/plans/golden-set-expansion-ml.md`. Recomputed the
  backlog's "~30 suspend / ~50-60 total" heuristic via the archived protocol's own Wilson-interval
  method and found it under-shoots its own zero-failure gate (n=30 → 11.4% upper bound, fails the
  ≤10% target; n=40 clears it at 8.8%). Recommends expanding 26 → **85 rows**
  (`clear_advance` 30 / `clear_suspend` 40 / `boundary` 15, oversampling the thin `turns`/fallback
  path). Confirmed `ws:acme` (checked read-only) has no real triage traffic to mine — sourcing
  stays synthetic-but-realistic, same register as the existing 26. Two draft illustrative rows
  included, explicitly marked unverified, not written to the fixture. Blast radius if implemented:
  fixture content change + 5 literal-constant edits in `test_guard_calibration_live.py`
  (lines 223/227/228/233/234) — no other harness change needed.
  **Raises a decision only the user can make (§10 of the note, not resolved by teco or the
  delegate):** who is the `boundary` tier's second independent *human* labeler for ~15 fresh
  scenarios labeled blind to the first labeler's rationale — the note argues an LLM cannot
  substitute here (boundary labels are a bias-to-suspend *policy* stance, not a fact-extraction
  task an LLM can independently verify), while the `clear_advance`/`clear_suspend` tiers are
  lower-risk enough for LLM-drafted-plus-human-spot-check. Secondary, lower-stakes: n=40
  (zero-failure-tolerant) vs. n=53 (tolerates one observed failure) for the suspend stratum — both
  defensible, the note doesn't pick. **No implementation dispatched on this unit; paused pending
  the user's decision.**
- **2026-08-20** — **U3 delivered**: `docs/plans/ministral-reprobe-ml.md`. Real
  `guards.evaluate_guard` run (26 cases × k=3) against Ministral via the workspace-override
  mechanism, no config files touched: **G1 false-advance 0.0% (pass), G2 advance-recall 45.5%
  (fail, gate ≥80%) ⇒ judge verdict BLOCK** — the item-1 parse fix helped (D13's fence-tolerant
  36.4% → 45.5%) but didn't close the reasoning-quality gap to Qwen's 81.8%. Model-identity check
  confirmed live: `mistralai_ministral-3-3b-instruct-2512` and `mistralai/ministral-3-3b` are the
  same weights aliased under two LM Studio catalog ids, not two models.
  **New finding, out of scope for both U1 and U3, not yet filed:** before running the full live
  e2e for the terminal-tool-call measurement, U3 live-verified that `executor._assemble_messages`
  unconditionally appends a trailing `user`-role CONTEXT block after thread turns, producing two
  consecutive `user`-role messages on the very first `intake` call — LM Studio's Mistral-family
  chat template hard-rejects that shape (HTTP 400, Jinja alternation error) while Qwen's template
  tolerates it. Substituted the brief's sanctioned fallback (an isolated `post_message`-schema
  replay, n=5): Ministral called the tool natively 5/5 (reconfirms D13's 3/3); Qwen 0/5
  same-session (reconfirms the persistent Defect-C prose failure) — but since K-039's
  already-shipped implicit-dispatch fallback compensates Qwen's weakness in practice, and
  Ministral's alternation crash means it never reaches that safety net at all, **verdict on the
  agent/step axis: block — worse than "loses," structurally broken** for this message-assembly
  convention. **Both axes verdict: do not wire Ministral, current codebase.** Two durable
  environment facts (alternation-crash mechanism; aliased Ministral catalog ids) confirmed written
  to `kaizen_team` (spot-checked by the coordinator). **The alternation-crash defect itself still
  needs filing as a new backlog item — deliberately held until U1 lands** (U1 is concurrently
  editing `BACKLOG.md`/`HISTORY.md`; a concurrent filing would collide in those files, same lesson
  as `m3-followups-coordination.md`'s K-034 race). U3-G (`analyst` review of the harness/numbers)
  dispatched.
- **2026-08-20** — **U3-G returned `approve`** (0 blocker/major, 2 minor, 1 nit) →
  `docs/reviews/ministral-reprobe.md`. The reviewer independently re-derived G2 (45.5%) and κ
  (0.442) by hand from the note's own disclosed per-case breakdown (exact match), confirmed
  `probe_ministral_judge.py` drives the real unmodified `guards.evaluate_guard` (not a stub),
  confirmed the workspace-override mechanism never touches a real graph selector, and — going
  beyond the brief — **independently live-reproduced the HTTP 400 alternation crash** against LM
  Studio directly (Ministral 400 with the identical Jinja error text; Qwen 200 on the byte-identical
  shape). Also confirmed both kaizen entries exist and confirmed scope discipline (no touches to
  `config/models.json`/`config/opencode.example.json`/`reference`/`ws:acme`, and U1's concurrent
  legitimate territory correctly left alone). **U3 ACCEPTED.** The two minors are about a
  non-git-tracked scratchpad replay script's reproducibility, not about trust in the reported
  numbers.
- **2026-08-20** — **U1 delivered**: all six carried findings fixed, test-first, in
  `guards.py`/`executor.py`/`app.py`. Coordinator-verified independently (not taken on the
  delegate's word): full suite **1098 passed, 3 deselected** (own run, up from the 1088 baseline);
  `_drive_loop` lock re-derived via the `DESIGN.md` §6.2 `awk` extraction — **SHA `71055f756280`,
  2860 bytes**, matching the delegate's own report exactly, unchanged before/after; `guards.py`'s
  m-1 (clause-boundary truncation before the negator scan) and m-2 (filter-before-slice in
  `_recent_turns`) fixes read directly and match the reported approach; `BACKLOG.md`'s six
  carried-finding entries read in full — accurate, correctly left items 4/5 untouched, and the
  doc-drift finding resolved honestly (the wrong byte-count figures existed only inside
  `docs/archive/`, which is never re-edited — nothing in a live doc site needed correcting).
  Coordinator re-seeded `reference` after running the suite (own pytest run wiped it again) —
  `verify_workflows.sh acme` → exit 0, 2 defs in sync. U1-G (`analyst`, diff review) dispatched.
  Also: **item 5 (Ministral re-probe) is now a completed, accepted result, not merely an advisory
  note** — its "don't wire, current codebase" verdict needs no downstream implementation, so
  `BACKLOG.md`'s K-027 item 5 bullet should flip to delivered, and the new alternation-crash
  defect U3 surfaced can now be filed (U1's BACKLOG.md/HISTORY.md edits are done, so the collision
  risk that held this back is gone). Both delegated to `data-scientist` (U3's own agent, resumed
  by id — it already holds the full technical narrative, avoiding a paraphrase) as a follow-up,
  assigned the next free backlog number **K-048** for the new defect.
- **2026-08-20** — **U1-G returned `approve`** (0 blocker/major, 1 minor, 1 nit) →
  `docs/reviews/guard-carried-findings.md`. The reviewer independently reproduced every claim
  rather than trusting the summary: traced m-1's exact bug through live execution (confirmed
  false-advance, confirmed the fix flips it), confirmed m-2's new test places malformed rows in
  the tail (the shape that actually exercises the ordering bug), independently re-derived that
  `_select_transition`/`_trace_step` sit outside the SHA lock, wrote its own independent
  differential fuzz harness for n-2 (5000 trials, 0 mismatches — stronger than the delegate's own
  2000-trial claim), confirmed n-1 and the doc-drift claim by direct grep, and reproduced the full
  suite at the identical 1098/3 count. One non-blocking minor (a bare comma in
  `_CLAUSE_BOUNDARY` can also truncate a genuine same-clause negation, but lands on the design's
  already-accepted safe/over-suspend side, not a new false-advance) and one nit (BACKLOG's n-2
  prose undersells the verification strength). **U1 ACCEPTED.** Re-seeded `reference` again after
  its own pytest run (documented hazard); verified in sync.
- **2026-08-20** — **U4 delivered and coordinator-verified**: `docs/BACKLOG.md`'s K-027 heading
  now names item 4 as the sole remaining open item and why (blocked on the user's §10 decision);
  item 5 flipped to `✅ delivered`, citing both artifacts; new `### K-048` filed (message-alternation
  defect, model-agnostic, independently confirmed outside the `_drive_loop` SHA lock — re-derived
  the lock hash again, still `71055f756280`) matching the K-033/034/035 filing convention exactly;
  dated `HISTORY.md` entry added. Coordinator read all three edits directly (not taken on word) —
  accurate. `verify_workflows.sh acme` → exit 0, still in sync (U4 was doc-only, as expected).

## ✅ RUN STATUS — 2026-08-20

**Accepted (independently reviewed, coordinator-verified):**
- **U1** — six carried findings fixed. `analyst` approve (0 blocker/major). Suite 1088 → 1098
  passed, 3 deselected (reproduced independently by both the coordinator and the reviewer).
- **U3** — Ministral re-probe: BLOCK on both axes (judge G2 45.5% < 80%; agent/step axis blocked by
  a newly-found structural crash). `analyst` approve (0 blocker/major).
- **U4** — K-027 item 5 marked delivered + K-048 filed (the alternation-crash defect from U3).
  Doc-only, coordinator-verified directly.

**Delivered, decision pending (not a defect, not blocked on review — blocked on the user):**
- **U2** — golden-set expansion design (`docs/plans/golden-set-expansion-ml.md`). Recommends
  26 → 85 rows with a corrected (Wilson-derived) target size. **Blocks on: who is the `boundary`
  tier's second independent human labeler, and n=40 vs n=53 for `clear_suspend`.** No
  implementation dispatched.

**K-027 status:** items 1–3, 5, and all six carried findings ✅ delivered. Item 4 is the only
open item, gated on the user's decision above. **New backlog item filed: K-048**
(`_assemble_messages`'s alternation-unsafe message shape — 🔵 proposed, unowned/undispatched).

**Not committed.** The commit decision is the user's, per standing practice on this component.
