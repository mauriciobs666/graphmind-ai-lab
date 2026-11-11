# Review: P1-48b follow-up, the indexer wall-clock watchdog

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1) · **Reviews:** `docs/plans/connected-code-graph-egress-spec4.md` §1, §2

## Scope & verdict

Scope: the uncommitted diff of `cpg/sandbox/offline-run.sh`, `cpg/sandbox/tests/test_sandbox.sh`, `cpg/sandbox/tests/mutants.sh` and `cpg/sandbox/README.md`, read against spec4 §1 (watchdog bullet) and §2 (test 27, first-wins, env-cannot-set-cap). Static review. I did not run containers, `test_sandbox.sh` or `mutants.sh`. The delegate's 25/25, 5/5 killed and 903-passed figures are unverified. What I did run: `bash -n` on the runner (clean), and a memory-safe, container-free probe of the real `sbx_wd_arm`/`sbx_wd_disarm` functions, extracted from the runner, under `ulimit -v 1500000` and `timeout -s KILL 60`.

**Verdict: approve with suggestions.** No blocker. The design conforms to spec4. The findings are test-strength and hardening items for `devops`.

CPG: considered, not relevant — a shell script and its shell test harness; the Joern CPG does not model them.

## Findings

### Major

None.

### Minor

**m1. K04 breaks two lines at once, and the test can only see one of them (`mutants.sh` K04; test 27(d)).**
K04 deletes both `trap '' USR2` and `sbx_wd_disarm` from `sbx_on_signal`. Test 27(d) sends USR2 by hand, so only the `trap ''` is exercised. Dropping `sbx_wd_disarm` alone survives, and it is functionally redundant there anyway, because the handler's `jobs -p` KILL loop ends the timer subshell. The row's text, "does not disarm", therefore overstates what it proves. The hand-sent USR2 is a sound substitute for the one guarantee the design actually rests on: a USR2 arriving after handler entry is ignored. It does not show that the real timer is cancelled.
Fix: split K04 into K04a (drop `trap '' USR2` in `sbx_on_signal`) and K04b (drop `trap '' USR2` in `sbx_on_wall`). The second is the only first-wins direction left unexercised. If the disarm call is kept in `sbx_on_signal`, add a check that no `sleep $SBX_INDEXER_WALL_S` orphan survives. See m3.

**m2. The pending-USR2 window and the stale-PID kill in `sbx_on_wall` (`offline-run.sh` ~L747-757, L772-781, L786).**
- (a) **Late fire.** Normal completion runs `wait` → `sbx_wd_disarm` → `trap - INT TERM HUP` → `trap '' USR2`. A timer that already sent USR2 just before `pkill`/`kill` lands has its handler run inside `sbx_launch`. That gives exit 124 plus a 124 record for a tool that had just exited. It is only reachable within microseconds of the cap and arguably correct (the run was at the cap). It is undocumented, though. Either state it in README, or set `trap '' USR2` before `sbx_wd_disarm`, so a finished child can never be reclassified.
- (b) **Stale PID.** On expiry the timer subshell has already exited and been reaped. `sbx_on_wall` then calls `sbx_wd_disarm`, which runs `pkill -KILL -P "$SBX_WD_PID"` and `kill -KILL "$SBX_WD_PID"` on a PID that no longer names our job. PID reuse within that window is very improbable, but it is the one place the design can hit a process that is not ours.
  - Fix: in `sbx_on_wall`, set `SBX_WD_PID=""` instead of calling `sbx_wd_disarm`. The timer has demonstrably finished, so there is nothing to cancel. Alternatively, disarm with the bash jobspec (`kill %n`).
  - In `sbx_wd_disarm`'s normal path the subshell is still alive and unreaped, so there is no reuse risk. I confirmed the normal path.

**m3. Disarm correctness is not tested, and `pkill` is not preflighted.**
I ran the extracted functions. After `sbx_wd_disarm`, no `sleep` orphan remained and no USR2 arrived. With a 1 s cap, USR2 arrived at about 1 s, as designed. `pkill -P` is safe in the runner's trust model: the argument is a numeric PID we created, there is no pattern, and the argument is quoted. It cannot widen to unrelated processes. If `pkill` is missing, though, the subshell is still killed (so no USR2), but the `sleep 1800` is orphaned for up to 30 min. No test or mutant (e.g. "disarm leaves the sleep") covers this, and test 27(b) would not notice an orphaned sleep.
Fix: add a check to 27(b) that `pgrep -f "^sleep <cap>"`, filtered to children of the test's own runner PID, is empty after a normal exit. Optionally, have `sbx_wd_disarm` fall back to a bash `kill` of the sleep PID if `pkill` is absent.

**m4. Cosmetic: a bash "Killed" job notice reaches stderr.**
In my probe, the disarmed subshell produced `line N: PID Killed ( sleep ... )`, the whole subshell body, on stderr. The runner will emit this on every normal indexer run, into whatever P1-15 captures. Silence it by disarming with `kill ... ; wait "$SBX_WD_PID" 2>/dev/null`. The delegate avoided `wait` after seeing a hang, so test that variant first, or reap through `disown` after the kill.

**m5. `trap '' USR2` is left set after `sbx_launch`.**
The disposition is "ignored", and ignored signals are inherited across exec. Later children of the runner (the record writers and `python3`) start with USR2 ignored. This is harmless today. Reset it with `trap - USR2` once the timer is cancelled, if `SBX_WD_PID` is empty and no handler can still fire. That is not safe before disarm (see m2a), so a comment is enough.

**m6. Untested asymmetry.**
`sbx_on_wall` begins with `trap - INT TERM HUP`. An INT or TERM arriving during the watchdog's removal (up to ~40 s) takes the default action and ends the runner with 130/143 and no cleanup. `sbx_on_signal` has the same pre-existing property in reverse, and spec4's first-wins sentence only asks that a signalled run never write a watchdog record. That holds. State the behaviour in README, or ignore INT/TERM/HUP in `sbx_on_wall` instead of resetting them.

## Checks by brief item

1. **Conformance to spec4 §1/§2: conforms, by reading.**
   - `SBX_INDEXER_WALL_S=1800` is assigned unconditionally at `offline-run.sh` ~L51 (the diff shows it at L49). No `:-` form exists, and it is not read from the environment.
   - Arming is gated on `SBX_INDEXER=1`. `SBX_INDEXER` is set from `sbx_is_indexer` and reset in `sbx_init`, so Joern frontends are untouched.
   - The expiry path is `sbx_on_wall`: KILL `jobs -p`, `sbx_remove_own` (container backend), `sbx_write_record 124 ""` (unrefused), `sbx_cleanup`, `trap - EXIT`, `exit 124`.
   - `sbx_write_record` has a `SBX_RECORD_WRITTEN` once-only guard, so a second record cannot be written by construction. That is stronger than the test.
   - First-wins: both handlers set `trap '' USR2` as their first act.
2. **Race analysis.**
   - There is no `wait` on the timer. `pkill -P` (children of the subshell: the sleep) then `kill -KILL` on the subshell means the subshell's `wait "$s"` fails and `&&` never signals.
   - There is no `set -e` (`set -uo pipefail`, L877), so the failed `wait` and the `kill ... 2>/dev/null` calls are harmless.
   - The subshell's TERM trap is dead code: nothing sends TERM, and it is KILLed. That is harmless but misleading.
   - The timer subshell does not inherit the parent's traps, since traps reset in a subshell.
   - Double handlers: handled by the `trap ''` first line plus the once-only record. The residual windows are m2 and m6.
   - A timer firing between disarm and exit is covered by m2(a).
3. **Does test 27 prove first-wins?** Partially. See m1. As to "exported 3 leaves 1800": yes, it tests the unconditional assignment. It expects the literal 1800, not the constant. It sources the runner with the variable exported, so K05's `${...:-1800}` yields 3 and the check reddens. The extra grep check is redundant, but it is a text check and cheap.
4. **Mutant table.**
   - K01 (indexers only), K02 (KILL), K03 (removal), K04 (first-wins) and K05 (env) each map to a distinct spec §2 clause. K04 is compound (m1).
   - YD and YF re-anchoring is legitimate. `mutants.sh` replaces the first occurrence, and the old anchors would have matched `sbx_on_wall`'s identical lines first. The new `old` strings are prefixed with the `sbx_on_signal`-only comment line, so they match only `sbx_on_signal`. The replacement text is otherwise unchanged. I read the diff and checked this, and did not run the mutants. They are not weakened.
5. **README.** The exit-124 bullet, the watchdog section and test 27 match the code. Wording corrections: the README says the handler's KILL "already ends the real timer" (it does, via `jobs -p`). It documents the library-mode hook accurately. Nothing contradicts the runner. m2(a), m4 and m6 are worth adding to the watchdog section.
6. **Outer bound.** It is stated in three places: README L187-188 ("cap plus 60 s"), spec4 §1 (with the 25 s / 40 s `sbx_remove_own` rationale), and `docs/plans/connected-code-graph-p1-extraction.md` L353. The extraction plan only cites "the margin spec4 states", with no number. That is adequate for the P1-15 reader, who is directed to spec4, but P1-15's step would be safer with the 60 s stated inline.

## What's solid

- The shape is minimal: one timer subshell and one signal. First-wins relies on the once-only record guard as a backstop.
- Cleanup and the record sit in the right order. The record is written after removal and before `sbx_cleanup`.
- The test uses a TERM-ignoring tool, and checks the record's fields, container, volume and scratch separately. Controls cover a non-indexer launch and a normal exit.
- The decision not to `wait` on the timer is documented in the code.

## Open questions

- Is the late-fire reclassification (m2a) acceptable as is? If so, document it.

## Routing

All to `devops`: m1 (split K04), m2 (both items), m3 (orphan check, `pkill` fallback), m4 (stderr noise), m5/m6 (document or harden). Then re-run test 27 and K01–K05; those results remain unverified. After the fixes, `teco` can close the follow-up. P1-15's plan gains the inline "60 s" figure (`architect`, minor).

## Pass 2

Scope: the uncommitted fix batch (runner, README, tests, mutants). Run: `bash -n` clean; `MUTANT_DRY=1` (all K01-K10, YD, YF apply; 0 NOT APPLIED); `test_sandbox.sh --only 27` (36 passed, 0 failed, 1 m 16 s); mutants K06, K07, K09, K10, YD, YF all killed; one reviewer-written mutant (below) and a bash probe. The full suite and K01-K05 were not re-run. Verdict: **approve with suggestions** (no blocker).

CPG: considered, not relevant — a shell script and its shell test harness.

Dispositions (analyst m1-m6):
- **m1 (split K04):** fixed (K04a/K04b, both dry-apply). My "drop `sbx_wd_disarm` is redundant" premise is now false, see N1; the implementer's rejection is not upheld.
- **m2a (late fire):** fixed. `trap '' USR2` now precedes `sbx_wd_disarm` in `sbx_launch`; the cap-boundary 124 is documented in README.
- **m2b (stale PID):** fixed. `sbx_on_wall` clears `SBX_WD_PID` and no longer kills it (read in the code).
- **m3 (orphan check, pkill):** fixed. No `pkill` in the runner (disarm is `kill -TERM` plus the timer's TERM trap); test (b2) checks no `sleep 29` after a normal exit; K10 killed on exactly that check.
- **m4 ("Killed" notice):** fixed. The TERM trap makes the timer exit 0; test checks stderr for "Killed" (passes).
- **m5 (USR2 left ignored):** accepted as a documented property. Verified in bash: an ignored signal IS inherited across exec (`trap '' USR2; bash -c 'kill -USR2 $$'` survives); a handled one resets to default (the child died with "User defined signal 2"). The README says exactly that and does not claim ignoring resets: the client starts under the handler, so it gets the default; later children (record writer) inherit "ignored", which is harmless.
- **m6 / security m1 (second signal during cleanup):** fixed with `trap '' INT TERM HUP` in both handlers; tests (e), (e2), K08, K09. Checked against the second-signal tests: no conflict, they now assert the ignored behaviour. Trade-off, see N3.
- **Security M1 (stale timer):** fixed. `/proc/<pid>/stat` field 22 (`${20}` after stripping through `") "`, correct for comm with spaces) recorded at arm, rechecked after every sleep step (at most 30 s) and at the cap, which is the last check before `kill -USR2`; timer closes `SBX_LOCK_FD`. Tests: seam-based mismatch/match (K06 killed), SIGKILLed runner frees the lock and leaves no `sleep 7` (K07 killed). The identity test arms a mismatching start time by seam; the real arm path is covered only by the match control, adequate.
- **Security m2/m3 (124 meaning, boundary race):** documented in README. 124 meaning is spec4/P1-15's (architect) to state.

Cross-checks: `sbx_kill_jobs` vs anchors: K02's `kill -KILL "$j" ...; wait "$j"` line occurs once; YD/YF now anchor on `sbx_kill_jobs "$wd"` ... `sbx_cleanup\n  exit`, which matches only `sbx_on_signal` (`sbx_on_wall` has `trap - EXIT` between); both killed. Identity check vs PID reuse: sound (start time in ticks plus PID). Trap-ignoring vs second-signal tests: consistent.

### New findings

**N1 (minor): the disarm in `sbx_on_signal` is no longer redundant and is not tested.**
The implementer's rejection of m1 rested on the `jobs -p` loop ending the timer. `sbx_kill_jobs "$wd"` now skips the timer, so `sbx_wd_disarm` is the only thing that ends it on a caller signal. Evidence: (1) a mutant I wrote, `sbx_on_signal` without `sbx_wd_disarm`, passes test 27 (36/0), i.e. survives; (2) probe: `sbx_wd_arm; sbx_kill_jobs "$wd"` leaves 1 `sleep 23`; after `sbx_wd_disarm`, 0. Consequence of the gap: a timer and its sleep outlive an INT/TERM/HUP-ed runner by up to 30 s (it then exits at its identity check), signalling nothing. Low stakes, but it is exactly the orphan class m3/K10 guard on the normal path.
Fix: in test 27(d), after the runner exits 130, assert no `sleep 6` (cap 6) remains for the run; add K11 (drop `sbx_wd_disarm` in `sbx_on_signal`). Then the rejection is moot.

**N2 (nit): test-side process matching.** `pgrep -fx "sleep 29"`/`"sleep 7"` and the `pkill -fx` clean-ups match by command line only, so another session's identical `sleep` could redden or be killed (same user). Prefer filtering by parent/session or an unusual odd value (`sleep 29.37` via the cap); the K-rows rely on this pattern. Also (e2) sends its hand-made USR2/TERM at `t0+7` s for a 5 s cap: a slow container start pushes the timer's own firing later and the TERM could land before the handler, a flake risk (exit 143), not a masking one.

**N3 (minor, design note): a wedged removal cannot be interrupted by TERM/INT/HUP now.** With both handlers ignoring them until `exit`, a hung `docker rm` (daemon wedged, outside spec scope, but spec bounds `sbx_remove_own` at ~25-40 s) leaves the runner killable only by KILL (P1-15's `timeout -k` covers it). State it in the README paragraph; no code change needed.

### Open question
None; N1's mutant/test is routed to `devops` with N2/N3 as optional. After N1, `teco` can close; the full-suite and K01-K05 results remain unverified by me.
