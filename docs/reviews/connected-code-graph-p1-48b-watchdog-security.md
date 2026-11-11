# Indexer wall-clock watchdog: security review (C-1001 P1-48b follow-up)

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1-48b)

Lens: code/app security (advisory, on request; not a gate). Reviewed: uncommitted diff of `cpg/sandbox/offline-run.sh`,
`cpg/sandbox/tests/test_sandbox.sh`, `cpg/sandbox/tests/mutants.sh`, `cpg/sandbox/README.md` against
`docs/plans/connected-code-graph-egress-spec4.md` §1 (watchdog bullet) and §2. Root `REVIEW.md` applied
(no `cpg/REVIEW.md` exists). No containers or suite run; `bash -n` clean; two small throwaway-dir bash experiments (below).

CPG: not applicable — a shell sandbox runner (bash), no Joern CPG exists for it; traced by reading and bash experiments.

**Verdict: approve with suggestions.** No blocker. The watchdog is not caller-controllable, indexer-forgeable into a
different outcome, or able to hit another user's process by pattern. One major (stale timer after a runner death
that bypasses the disarm) and a few minors.

## Findings

### M1 (major, confirmed by experiment): a timer that outlives its runner signals a recycled PID and pins the `--output` lock
`sbx_wd_arm` (`offline-run.sh`, new) backgrounds a subshell that, after the cap, runs `kill -USR2 "$me"` with no check
that `$me` is still the runner. Disarm happens only on `sbx_launch`'s normal return and in the two handlers. Any runner
death that skips them (SIGKILL, OOM kill, SIGQUIT/USR1/PIPE default action; README already concedes SIGKILL is untrappable)
leaves the timer running. Experiment (`timeout -s KILL`, `ulimit -v`, scratchpad `x/kill9.sh`): after `kill -KILL` of the runner,
the timer subshell was re-parented to init and its `sleep` survived, **both holding the runner's `--output` lock fd (fd 10)**.
Consequences: (a) 30 minutes later the timer sends `USR2` to whatever process now owns that PID. The default action of
USR2 is termination. Same-user only, and PID wrap inside 30 min is improbable, so it is low-probability but it is an
unattended kill of an unrelated process, which spec §1's "a stale timer never signals a runner past its trap" does not
cover (it covers only normal exit). (b) The lock fd stays open in the orphan for up to 1800 s, so a retry on the same
`--output` is refused ("in use by another run") even after the container (which `--rm`s itself) is gone. The container
already survives a SIGKILL by design, so (b) mostly matches existing behaviour, but the lock now outlives it.
Suggested: in the timer, before `kill -USR2`, verify identity: capture `/proc/$me/stat` field 22 (starttime) at arm time
and compare, or at minimum `[ "$(ps -o ppid= -p $BASHPID)" = "$me" ]` / `kill -0 "$me"` plus the starttime compare;
and close the lock fd in the timer subshell (`{SBX_LOCK_FD}<&-`) and for `sleep`. Add a mutant + a test (SIGKILL the
runner, assert no timer survives or that it signals nothing).

### m1 (minor, confirmed by reading): a second caller signal during the watchdog's cleanup window leaves the container running
`sbx_on_wall` runs `trap - INT TERM HUP` first, then spends up to ~40 s in `sbx_remove_own`. A TERM/INT in that window takes
the default action: the runner dies, `sbx_remove_own` never completes, and the container that already exceeded the cap
keeps running (the Docker client was KILLed, so `--rm` fires only when it ends). `sbx_on_signal` has the same shape
(pre-existing), but spec §1's outer bound (P1-15 timeout > cap + 60 s) makes a TERM-ing outer path in exactly this window a
designed-for case. Suggested: keep the handlers installed during cleanup but make them set a flag/ignore (`trap '' INT TERM HUP`)
so a second signal cannot abort the removal; apply to both handlers.

### m2 (minor, inferred): record 124 is not unforgeable
The tool's own exit status passes through `docker run` into `rc`, so an indexer that calls `exit 124` produces the same
`exitCode 124, refused false` record as a real expiry, in far less than 1800 s. Effect is nil in the stated model (spec: 124
is a non-zero indexer exit, no artifact, P1-15 treats it as timeout failure), but P1-15 must not infer "ran the full cap"
or take 124 as evidence the cap policy fired. Suggested: state in spec/README that 124 means "timeout-or-tool-exit-124",
or make the watchdog distinguishable (e.g. `durationMs >= cap*1000` check by the consumer, which already exists in the record).

### m3 (minor, inferred): benign race at the cap boundary
If the indexer exits at the cap, the timer can deliver USR2 after `wait` returned but before `sbx_wd_disarm`;
`sbx_on_wall` then records 124 for a run that succeeded (output exists, but P1-15 must ignore it). Fail-closed and harmless;
document it. `pkill -KILL -P "$SBX_WD_PID"` is safe by scope (exact parent PID; a child of this runner whose job is still
tracked by bash, so reuse before disarm is not realistic) but `kill -KILL` of a stored sleep PID would be tighter than a
parent-PID pattern. A 300-iteration arm/disarm loop left no orphan sleep (experiment), so the arm-then-immediate-disarm
orphan race is theoretical.

### n1 (nit): test hook scope
Library mode lets a sourcing process reassign `SBX_INDEXER_WALL_S`; it is not reachable from the environment (re-exec
`env -i`, unconditional assignment, test and mutant K05 pin it). Fine; just keep that sentence in README.

## Answers to the focus questions

1. **Bypass / turned against the runner.** Cap not settable from the environment (unconditional assignment; library mode only).
   `kill -USR2 "$me"` targets `$BASHPID` captured in the runner (sbx_launch is not called in a subshell, so it is the runner).
   `pkill -P` is scoped to one PID's children, not a name pattern: no cross-user or cross-run hit (see m3 for PID reuse). The
   one real hazard is M1. The indexer cannot outlive the cap through the race: expiry KILLs the client, then removes by label
   until three quiet checks, and the container has `--pids-limit 4096`.
2. **Record integrity.** `sbx_write_record` is guarded by `SBX_RECORD_WRITTEN`, so both-write is impossible; first-wins is
   by `trap ''`/`trap -` on entry. "Neither" is possible only via m1 (a second signal mid-cleanup) or an untrappable kill, both
   fail-closed (no record, as spec3 §5.3). Forge: only the m2 shape (exit 124 by the tool).
3. **Malicious indexer.** `docker run` has no `--pid=host`, no host mounts beyond ro `/src` and the output volume, `--network none`,
   `--cap-drop ALL`; the container cannot signal host PIDs (USR2 to the runner included). TERM-ignoring and fork-bombing processes
   are handled by KILL of the client then `docker rm -f -v` (SIGKILL of the whole cgroup). Memory/CPU remain uncapped (spec says so):
   a memory-exhausting indexer can wedge the daemon so the removal hangs (each docker call has no timeout; spec calls a hung daemon out of scope).
4. **Cleanup trust model.** `sbx_remove_own` filters by the exact label `ccg.sandbox.auditId=<uuid>` and removes only what that
   returns; `sbx_cleanup` removes only paths the runner recorded. No pattern wide enough to hit another run. Solid.
5. **DoS shapes introduced.** Only M1 (stale timer pinning the lock up to 30 min; recycled-PID USR2) and m1. Each timer is one
   subshell plus one sleep per run; no growth.

## What's solid
Constant not environment-readable; KILL-first ordering and exact-label removal reused from the reviewed handler; idempotent
record write; first-wins disarm in both handlers with a test that delivers the late USR2 deterministically; mutants K01-K05
map to the properties. `bash -n` passes.

## Open questions
- Does the stager/P1-15 consumer use `exitCode == 124` alone as the timeout signal (m2)?

## Routing
- **devops:** M1 (identity check in the timer, close lock fd, a SIGKILL-the-runner test + mutant), m1 (ignore, not default,
  further signals during cleanup, both handlers), m3 doc note.
- **architect:** m2 (spec §1 / §5.3 wording of what 124 asserts; consumer rule in P1-15); amend spec §1's "a stale timer never
  signals" to its true scope (normal exit and trapped signals only) unless M1 is fixed.

## Pass 2

Scope: uncommitted fix batch (`offline-run.sh`, README, `tests/test_sandbox.sh`, `tests/mutants.sh`). Run: `test_sandbox.sh --only 27`
(36 passed, 0 failed, hygiene clean) plus my own foreground bash experiments (scratchpad `p2/`). No exploitation; no containers of mine
left (one foreign `ccg-sbx-9a8b8a70-...` was present before and is untouched).

**Verdict: approve with suggestions.** No blocker, no major. Dispositions:

- **M1 (stale timer / recycled PID / lock fd): fixed.** Live: with the real `sbx_pid_start`/`sbx_wd_arm`, cap 65 s, runner SIGKILLed: orphan timer alive at +17 s,
  gone by +34 s (the 30 s step re-check exits it); it holds no fd 10 (`/proc/<timer>/fd` shows none). `sbx_pid_start` is correct on comm values `a) S 1 2 3 (b`,
  `x y) z`, `(((`, `a) ` (each equals an independent `rsplit(')')` parse): `${l##*) }` is greedy to the last `) `, and later fields are numeric so no
  false split; `$20` after `set --` is field 22. Start time is boot-relative ticks, so a reboot cannot alias (the timer dies with the host anyway); a same-tick PID
  wrap within one boot is not realistic. Zombie runner: identity matches, USR2 to a zombie is a no-op.
- **m1 (second signal during cleanup): fixed.** `trap '' INT TERM HUP` in both handlers; test 27 delivers a second TERM and USR2 mid-removal (exit 130/124, container
  gone). Ignored dispositions are inherited across exec (confirmed: `SigIgn 4800` survives `exec`, a *handled* TERM becomes `SigIgn 0`), so the README claim is
  correct, and the sandbox client (started with handlers installed) runs with default dispositions. The only children that inherit "ignored" are the cleanup
  `docker` calls and later, which is intended. Residual window: in `sbx_on_signal` the `trap '' INT TERM HUP` comes after `trap '' USR2`, `local` and the
  disarm, so a second TERM in those microseconds runs a nested handler that finishes the removal and exits 143: no leak, only a different exit code. Not a finding.
- **m2 (124 not unforgeable): fixed (documented).** README states 124 is "watchdog or a tool exit 124" and points the consumer to `durationMs`.
- **m3 (boundary race, `pkill`): fixed.** Race documented as fail-closed; `pkill` removed. The new disarm order (`trap '' USR2` before TERM-ing the timer)
  leaves only the window between `wait` returning and that trap, which records 124 for a just-succeeded run: fail-closed, as documented. No new race
  between disarm and expiry found (a late USR2 after the trap is ignored; the timer's own signal is sent only after its identity check).
- **n1: unchanged, still fine.**

### New findings

**m4 (minor, confirmed by experiment): the identity check fails open when the recorded start time is empty.** `sbx_wd_arm` takes `st=$(sbx_pid_start "$me")`; the
timer then tests `[ "$(sbx_pid_start "$me")" = "$st" ]`. If `st` is empty, a dead runner also yields empty, so the test passes and the timer signals the
(possibly recycled) PID, i.e. the pre-fix behaviour. `st` is empty when `/proc` is unreadable, or when the runner's own `comm` contains a newline: I reproduced that
`sbx_pid_start` returns empty for a process named `a\n) 9 9 9 9` (the single-line `read` cannot see field 22). The runner's comm is its script basename, so this needs
the owner to run a copy/symlink with a newline in its name: theoretical and self-inflicted. Suggested: after arming, `[ -n "$st" ] || { sbx_log "watchdog: runner start time unreadable"; return 1 or arm no timer }`
(or loop on `[ -n "$st" ] && [ "$(...)" = "$st" ]`), and read with `IFS= read -r -d '' l` so a newline in comm cannot truncate the line.

**m5 (minor, inferred): with INT/TERM/HUP ignored, a wedged cleanup can only be stopped by SIGKILL.** `sbx_kill_jobs` does `wait "$j"` after KILL with no bound, and
`sbx_remove_own` has up to 50 docker round trips with no timeout; a hung daemon previously could be interrupted by a second TERM, now only by KILL (which leaves the
container, as the README already concedes). That matches the stated design and P1-15's cap+60 s outer bound, so no change is required; just ensure P1-15 escalates to
KILL after the bound, and that the README says the ignore also removes the second-TERM escape hatch.

**n2 (nit, test coverage): the 30 s step re-check is not exercised.** Test 27 uses a cap under 30 s, so the in-loop identity check coincides with the cap check; my
65 s experiment shows the stepping works, but no mutant removes it (K06 removes the check entirely). A mutant that makes `step` the whole remainder would leave an orphan up to 30 min
and pass. Suggested: run the SIGKILLed-runner test with a cap of ~35 s (one full step) and assert the timer is gone by 31 s, plus a mutant `step=$((end - SECONDS))` unbounded.

Routing: **devops** m4 (small), n2; **architect/devops** m5 note. Open question unchanged: does P1-15 treat `exitCode == 124` alone as the timeout signal?
