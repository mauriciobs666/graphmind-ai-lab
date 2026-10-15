# Review — teco environment auto-start (`SessionStart` hook)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** cobb K-054 (—)

## Scope & verdict

**Reviewed:** cobb's uncommitted K-054 delivery against commit `5c82170f` — new
`claude/teco/hooks/ensure-environment.sh`; diffs to `claude/teco/teco.md` (frontmatter + step 1),
`claude/teco/coordination-techniques.md`, `claude/README.md`, `claude/AGENTS.md`,
`skills/agent-standards/claude-code.md`, `claude/cobb/kaizen/{plan,history}.md`,
`claude/teco/kaizen/history.md`. Spec: K-054 in `git show 5c82170f:claude/cobb/kaizen/plan.md`.
Evidence spot-checked: cobb's scratchpad `dbg6-8.txt`, `run1-3.jsonl`, `tui1.txt`, the two
`hook-wait*.sh` test hooks. Not reviewed: U3/U4 of `claude/docs/plans/teco-env-autostart-coordination.md`.

**How verified.** The hook was run for real only on paths that cannot launch anything (all-up via
the deployment symlink, traced with `bash -x`; three out-of-scope project dirs). Every launch
branch was run on a **sandbox copy**: the script copied into a scratch tree with stub
`falkor-chat/scripts/start_{agent_team,falkordb}.sh`. `REPO_DIR` comes from the script's own path,
so the stubs take the real scripts' place with no edits. Fake listeners ran on :8215-8219/:6398-6399.
The live services were only read: one `GET /health` on :8200 and one `PING` on :6379. All test
processes were stopped afterwards.

**Verdict: approve with suggestions.** The hook is correct on every contract point the brief asked
about: additive, single-flight for the server's whole life, scoped correctly, fail-open and
bounded. The evidence supports the history entry's claims. There are two **major** findings, both
in the prose around the hook rather than the hook itself. First, main-session teco lost the
per-goal probe. Second, the hand-written delegated recipe is a weaker copy of the hook, with no
lock and a log path that may not exist. Fix both in the same pass as U4, which already touches
these files. The minor findings are probe-classification and message-accuracy gaps. None of them
is destructive.

CPG: considered, not relevant — the artifact is a Bash hook plus agent docs. `cpg_falkorchat`
covers the falkor-chat server's Python, and the one server fact needed (`/health` → 503 exactly
when FalkorDB is unreachable) was read directly at `falkor-chat/server/falkorchat/app.py:497-511`
and `api.py:56-64`.

## Findings

### Major

**M1 — teco.md step 1 no longer tells main-session teco to probe when a goal arrives; "if it
reports nothing, both were up" is a start-time fact presented as a current one.**
Evidence: `git diff 5c82170f -- claude/teco/teco.md`, step 1. The old text said "probe once here"
for any goal touching FalkorDB. The new text gives the main session no probe at all: the hook
"has already started" both services, and silence means "both were up". Step 1 runs **per goal**,
but the hook runs once per session (`startup|resume|clear`). The silence premise also fails in
cases the hook cannot see:
- FalkorDB dying mid-session (e.g. a Docker Desktop restart, which leaves `claude` running).
- `python3` missing: the hook launches the service but prints nothing (`ensure-environment.sh:120`, `|| true`).
- A compaction that drops the hook's message.

`coordination-techniques.md` does cover mid-session deaths, but teco reads it only on demand.
Nothing in step 1 sends a main-session teco there. In each of these cases a delegate hits the
down service and reports it, which is the "report it down" outcome step 1 exists to prevent.
**Fix:** keep one unconditional, cheap probe at decomposition in both modes: run the hook script
itself (see M2), which is silent and takes about 20 ms when everything is up. Reword the hook
sentence as "already ran at session start". Drop "if it reports nothing, both were up".

**M2 — The delegated / mid-session recipe in `coordination-techniques.md` re-implements the hook
without its single-flight lock, and breaks on a machine where the hook has never run.**
Evidence: the new `**The ws:agent-team server**` paragraph says to probe, and on "Connection
refused" to run `setsid nohup ./scripts/start_agent_team.sh >>~/.local/state/graphmind-ai-lab/agent-team-8200.log …`.
1. **No `flock`.** While a hook-launched start is in flight (about 20 s), :8200 refuses
   connections. A teco following this recipe at that moment starts a second
   `start_agent_team.sh`, which leads to:
   - two concurrent `pip install -e` runs into the same `server/.venv`;
   - two concurrent bootstrap/seed runs;
   - a second uvicorn that dies on address-in-use.

   The hook's own guarantee ("Do not start a second copy") only holds for launches that take the lock.
2. **Log directory.** `~/.local/state/graphmind-ai-lab/` is created only by the hook's
   `mkdir -p`. For a delegated-only user it may not exist. Bash then fails the `>>` redirect and
   **never runs the command**. The recipe also ignores `XDG_STATE_HOME`, which the hook honours.
3. **A second copy of the logic.** This is the duplication question 3 of the brief asked about. The
   recipe restates the probe, the classification rule and the launch, and the two copies have
   already diverged on points 1 and 2.

**Fix:** make the recipe one line:
`CLAUDE_PROJECT_DIR="$PWD" claude/teco/hooks/ensure-environment.sh </dev/null`, run from the repo root.
- The hook ignores its input, so this is safe outside a hook context.
- `</dev/null` is required. With stdin left open, the script's `cat` blocks until stdin closes; measured: 3.0 s under `sleep 3 | …`.
- The hook's JSON output is the report.

Keep the polling for 200, the `/mcp` → Reconnect step and the "never restart; route to `devops`"
rule as prose. Delete the hand-written launch command.

### Minor

**m1 — The FalkorDB probe treats every answer other than `PONG` as "down". That contradicts the
header's contract and produces false "was down — starting it" messages.**
Evidence: `ensure-environment.sh:84`. The header (`:16-18`) says only "connection refused" counts
as down. That sentence is written about the server, but it reads as the hook's whole contract.
Sandbox: a listener that answers `-LOADING …` (FalkorDB loading its RDB after a restart) is
classified as down, and `start_falkordb.sh -d` is launched. The same happens when `redis-cli` is
missing (exit 127), on `NOAUTH`, or on a 2 s timeout. The launch itself is harmless: in reality it
fails with a `docker run --name falkordb-dev` name conflict. But the user and teco are told
something false, and when `redis-cli` is missing, **every** session start does this and the hook is
never silent.
**Fix:** classify the way the server probe already does. Treat FalkorDB as down only when
`redis-cli` prints `Connection refused` (verified as its output on a free port). Report anything
else as "answers but not ready" or "cannot probe (redis-cli missing)". Test the change with a
**coverage probe over both probe axes**: curl outcome (0+200 / 0+non-200 / 7 / 28 / other) ×
redis outcome (PONG / refused / error reply / timeout / tool missing). Assert, for every cell,
which script is launched and which message is shown. Use the sandbox-copy technique above. A list
of hand-picked cases is not enough.

**m2 — "Server answers 503" and "FalkorDB down" are reported as two separate problems, though they
are one.** `/health` returns 503 exactly when FalkorDB does not answer (`app.py:507-511`). When
the server is up and FalkorDB is down (sandbox case B, with a 404 stand-in), the hook emits both:
- "server unhealthy … NOT restarted … If it stays unhealthy, route it to devops", and
- "FalkorDB was down — starting it".

This invites a pointless `devops` dispatch.
**Fix:** when `falkor_up=0` and `code=503`, say that 503 is expected while FalkorDB is down and
that FalkorDB is being started, then re-probe `/health`. Escalate only a non-200 that persists
after that.

**m3 — The hook decides "FalkorDB up" by port, but `start_agent_team.sh` decides it by container
name.** `start_agent_team.sh` step 1 checks
`docker inspect falkordb-dev … | grep -q running`. When anything else serves :6379 (e.g. the
`falkor-chat/compose.yaml` stack, which has no `container_name` and binds the same port), the
following happens:
- the hook sees `PONG` and launches the server script;
- the script runs `start_falkordb.sh -d`, which fails to bind :6379;
- `set -e` aborts the whole bring-up;
- the hook has already told the user "starting it in the background".

The FalkorDB lock is also held only for the ~1 s `docker run -d` CLI call, not until FalkorDB is
ready. A second session probing in that window repeats the launch, and that ends in a name conflict.
I traced this by reading; it was not run against Docker. It is harmless, but it is noise. This is
the answer to brief question 2: the two branches never double-start on their own, because only a
different server state between two probes can send two sessions down different branches, and
every resulting collision is a non-destructive `docker run` failure.
**Fix (owner: falkor-chat, via `teco` → `coder`/`devops`):** make `start_agent_team.sh` step 1
probe by port (`redis-cli ping`) before trying `start_falkordb.sh`. That fixes the compose case
and the race together.

**m4 — The always-on server inherits whatever environment the session that launched it had.** The
hook launches `start_agent_team.sh` with Claude Code's environment, and `setsid`/`nohup`/`flock`
all pass it through. The script honours `FALKORCHAT_WS_ID`, `UVICORN_ARGS`, `EMBEDDING_DIM` and
`FALKORCHAT_MODEL_CONFIG`. If a user launches `claude` from a shell that exported one of them for
other falkor-chat work (e.g. `FALKORCHAT_WS_ID=demo`), the shared :8200 server comes up on the
wrong workspace. Every agent's `ingest_document` call is then affected until someone notices.
Unlike a manual start, nobody sees the startup banner.
This is inferred from reading the script, not run.
**Fix:** in `launch_detached`'s server call, pin `FALKORCHAT_WS_ID=agent-team` and unset the
other three (`env -u …`). Keep `AGENT_TEAM_PORT`, which the probe uses too.

### Nits

**n1 — Matcher drift.** The script header (`:2`) says `startup|resume`. The frontmatter and cobb's
history say `startup|resume|clear`. **Ruling:** the frontmatter is the operative one, and `clear`
is right: after `/clear` the hook's message is teco's only signal, and the hook costs about 20 ms
when everything is up. Fix the header. Also decide on `fork`, a fifth `SessionStart` matcher
(`/fork`, `/branch`, `--fork-session`) per `code.claude.com/docs/en/hooks`. It was not
considered; add it, or note in the header why it is left out.

**n2 — Dependency list.** `claude/README.md` says the hook needs `redis-cli`, `curl`, `flock`,
`setsid`. It also needs `python3` (without it the hook is silently mute, see M1) and `timeout`.

**n3 — Evidence wording.** `skills/agent-standards/claude-code.md` says the hook's
`additionalContext` "reaches the model". The logs prove the harness accepted it
(`dbg8.txt:115`, "provided additionalContext (607 chars)"), and the docs say it reaches Claude.
But no run asked the model to repeat it: all three runs replied "OK". Say "accepted by the
harness (debug log); reaches the model per docs". Separately, cobb's "all-up path … silent,
17ms" test could not tell "all up" apart from "scoped out", because both exit silently. I re-ran
it with `bash -x` through the symlink: `REPO_DIR`/`proj` resolve to the repo, the script reaches
both probes, and it exits on the all-up branch. So the claim holds, but only with that trace.

## What's solid

- **Additive and bounded.** It starts something only on curl exit 7. An HTTP 404 and a timeout
  (exit 28) were both reported and left alone (sandbox B and D). No code path stops or restarts
  anything. Probes are capped at 2 s each (`bash -x` timestamps: 2.0 s on an accept-but-never-reply
  listener). Both launch paths returned in about 31 ms, with stdout reaching EOF. The detached
  child keeps none of the hook's pipes, so session start cannot be held past the 15 s timeout.
- **Single-flight really lasts for the server's life.** `flock` forks the command and waits for it.
  `start_agent_team.sh` `exec`s uvicorn, so the same child becomes the server, and it also inherits
  the lock fd. Sandbox check: the lock is `HELD` after the stub's `exec`, `/proc/<pid>/fd/3` points
  at the lock file, PGID/SID equal the `flock` pid, and a concurrent second launch logged its
  header but never ran the stub.
- **Scoping.** `readlink -f "$0"` resolves the `~/.claude/agents/teco` symlink. Out of scope, the
  hook exits 0 before probing: for `CLAUDE_PROJECT_DIR=/tmp`, for an unset variable with
  `PWD=/tmp`, and for the sibling-prefix `…/graphmind-ai-lab-other` (the trailing `/` in the
  `case` pattern blocks it).
- **Fail-open.** No `set -e`. The state dir falls back to `$TMPDIR`. The JSON emitter is wrapped
  in `|| true`. Bash 5.2, so the empty-array/`set -u` pitfall does not apply.
- **Evidence matches the claims** (Appendix A).
- **Docs placement.** The generic facts are in the right skill sections, and `claude/AGENTS.md`
  adds a short sentence rather than a paragraph. No personal-info leak was introduced: the
  `audit-team.sh` FAILs are pre-existing, and a grep of the diff found nothing.

## Open questions

1. K-054's spec lists "the `cypher` MCP" among the services to bring up. The delivery handles it
   implicitly: it is a stdio container that Claude Code spawns per session, so there is nothing
   to start, and it recovers once FalkorDB is up. Nothing in the delivery says so. Should one
   clause in the history entry record that this part of the ask was considered?
2. M1/M2 and U4 (the CPG-freshness report) all touch `ensure-environment.sh` and the step 1 text.
   I recommend handing M1/M2/m1/m2/m4/n1 to cobb with U4, so the file is revised once and
   re-gated once. m3 belongs to falkor-chat, not cobb.

## Appendix A — evidence spot-check

| Claim (cobb history 2026-10-14 / claude-code.md) | Log evidence | Holds? |
|---|---|---|
| MCP connect does not wait for `SessionStart` | `dbg7.txt`: `hook-wait-slow.sh` (sleeps 10 s) ran from ~18.55 to 32.17 (13 618 ms). MCP's first attempt was at 18.98, retries at 19.99 / 22.00 / 26.01 | yes |
| 3 retries at 1/2/4 s, ~7 s window, then gives up | `dbg7.txt:170,231,250` "retry n/3 in 1000/2000/4000ms", final failure at :267; the next connect is 00:18:28 (the manual Reconnect) | yes |
| Warm start (~3.5 s) connects unaided | `dbg6.txt` hook done 32.999 → connected 36.387; `dbg8.txt` 04.393 → 07.756 (on the attempt after "retry 2/3") | yes |
| Hook is fast (82-88 ms) | `dbg6.txt:112` 88 ms, `dbg8.txt:112` 82 ms | yes |
| `systemMessage` shown in the TUI | `tui1.txt` `SessionStart:startup says: …` (an earlier message version, same mechanism) | yes |
| `-p` init reports `failed` even with a synchronous wait-until-healthy hook | `run2.jsonl`: hook "healthy after 7 polls"; `init` `falkor-chat-agent-team: failed` | yes |
| `additionalContext` reaches the model | harness acceptance only (`dbg8.txt:115`); model replied "OK" | partly (n3) |

`dbg6.txt` and `tui1.txt` exercised an earlier message text ("~20-30s … Once healthy, run /mcp");
`dbg8.txt` is the shipped version. The behaviour under test is the same.

## Pass 2

2026-10-14. **Scope:** cobb's revision, uncommitted against `5c82170f`: the rewritten hook (now
also the step-1 probe, plus the U4 CPG check) and the doc/kaizen files listed in the newest entry
of `claude/cobb/kaizen/history.md`. The hook is checked against `start_agent_team.sh` as of
`e7a020cc`. Harness: `…/scratchpad/matrix/run_matrix.py`.

**How verified:**
- Ran cobb's 25-cell matrix 3 times.
- Built my own scratch FalkorDB (`analyst-k054-scratch`, port 6397, `--rm`; stopped afterwards)
  seeded with 4 marker shapes. I ran a sandbox copy of the hook against it. The copy reads the
  real repo's git objects through a `.git` file in the sandbox, with no writes.
- Ran four data mutants of the CPG check against those markers.
- Ran an env-leak probe and a 1,500-iteration timing loop.
- Live services were read only: `GET /health` on :8200, `PING`/`GRAPH.LIST` on :6379, and one
  `mcp__cypher__query` read of `cpg_falkorchat`. All three answered normally afterwards.

**Verdict: approve with suggestions — one last revision against the done-condition below, then
close.** Every Pass 1 finding is fixed except m4, which is only partly fixed. The new findings come
from the fixes themselves:
- my own Pass 1 m4 remedy was a list, and a list can miss a variable;
- the m1 change and the m3 fix interact;
- the new U4 code is not durably tested.

The one exception, P2-m1, is a host fact that was there before this change and that Pass 1 also
hit without explaining. No design defect remains.

CPG: considered, not relevant — a Bash hook plus agent docs. The CPG *check* was reviewed against
`skills/cpg-analysis/references/freshness.md`, not through a graph.

### Pass 1 dispositions

- **M1 fixed.** `teco.md` step 1 now says to run the script "for every goal, in both modes" and calls
  the start-time report "from session start, not now".
- **M2 fixed.** The recipe is "run the script; never hand-roll its launch commands"
  (`coordination-techniques.md`). The stdin drain is bounded (`:73`), but it now spawns
  `timeout`; see P2-m1.
- **m1 fixed.** `:112` treats only `Connection refused` as down. In the matrix, the launch
  classification passed in all 25 cells on all 3 runs. The only failures were timing (P2-m1).
- **m2 fixed.** `:145-146` sends this case to context only.
- **m3 fixed** in `e7a020cc` (port `PING` first, then container name).
- **m4 partly fixed.** Five variables are unset and `FALKORCHAT_WS_ID` is pinned. Every other
  `FALKORCHAT_*` variable still leaks; see P2-m2.
- **n1 fixed.** The header (`:3`, `:52`) and the frontmatter both say `startup|resume|clear|fork`.
- **n2 fixed.** Header `:49` and `claude/README.md:130` list the dependencies; `python3` is gone.
- **n3 fixed.** `run6.jsonl` shows the model quoting `additionalContext` back.
- **Open question 1 answered** (header `:24-28`). **Open question 2 done**: U4 was folded into the
  same pass.

### New findings

**P2-M1 (major) — The CPG check has no durable test, and the one data read that alone separates
"hand-authored" from "fresh" is not pinned by any shape that was tested.**
- `run_matrix.py` cannot reach `cpg_check`. Its fake redis answers every command with `+PONG`, so
  `GRAPH.LIST` returns no `cpg_` lines, `:168` returns, and the all-up cell is asserted silent
  (`run_matrix.py:47-53,117`). The matrix therefore also certifies a `GRAPH.LIST` failure as "no
  graphs to flag".
- cobb's 14-shape CPG test exists only as a claim: there is no such script in the scratchpad (only
  `fakeredis.py`, a PONG-only server).

My mutants, run against seeded markers:

| Mutant | Effect | Caught by |
|---|---|---|
| `b.MARKER_ORIGIN`→`b.MARKER_ORIGN` | A hand-authored marker with `PROVENANCE:'source-origin'` and a matching tree drops out of the report: **reported fresh** | Only that marker shape |
| `$dirty != "false"`→`= "true"` | Would let a marker with `SOURCE_DIRTY` absent through | Only a marker without `SOURCE_DIRTY` |
| `SOURCE_TREE`/`SOURCE_DIRTY` key typos | Noticed (fail closed) | — |

cobb's listed shapes do not include either of the shapes that catch the first two mutants. Its
hand-written (`unknown`) and `hand-backfilled` markers are caught first by the `BUILT_AT` check and
by the closed provenance set, so the MARKER_ORIGIN mutant would likely survive them. That is
inferred: the test cannot be re-run. `freshness.md:35` says `markerOrigin` is *the* reliable
hand-authored tell. A silent `GRAPH.LIST` failure also contradicts the header's "never as fresh"
(`:22-23`).

**Fix:** track a re-runnable test next to the hook, e.g. `claude/teco/hooks/test-ensure-environment.sh`.
The precedent is `skills/joern-cpg/scripts/test-stamp-wiring.sh`. Fold the matrix in, and add:
- a real-FalkorDB shape pass that includes a hand-authored marker carrying a pipeline provenance
  and a marker with no `SOURCE_DIRTY`;
- a `GRAPH.LIST` error reply, which must be reported "could not list graphs", not silence.

Each of the three mutants above must turn the test red.

**P2-m1 — GNU `timeout` on this host sometimes stalls for 13.17 s, so a run can blow the 15 s hook
timeout. cobb's "harness artifact" ruling rests on a test that could not have seen it.**
`timeout 2 true` took 13.17 s in 1 of 1,500 runs; plain `true` took under 1 s in all 1,500
(coreutils 9.4, kernel 5.15.167.4-microsoft-standard-WSL2). Inside the hook, phase timestamps put
the stall in two places: before the FalkorDB probe (the `timeout 2 cat` drain) and in the probe
itself (`timeout 2 redis-cli`). Cobb's matrix reproduced it in its **all-up 200/PONG cell**, where
nothing hangs (13.21 s), and the 15.2 s cell is the same stall plus a 2 s probe. That explains my
unexplained 15.2 s in Pass 1.

cobb's dismissal re-ran only the timeout/timeout cell, 10 times. At this rate 10 tries almost
certainly show nothing, and the stall is not specific to that cell.

Budget with no stall:

| Step | Max |
|---|---|
| curl probe | 2 s |
| `GRAPH.LIST` | 2 s |
| CPG loop: 6 s budget, checked only before each graph, plus one last 2 s query | 8 s |
| **Total** | **about 12 s** — fits the 15 s timeout |

One stall overruns it. The script is fail-open, so the cost is a lost report, not harm.
**Fix:**
- replace the drain with a bash builtin loop (`while read -r -t 1 _; do :; done`), so no `timeout`
  spawns on every run;
- raise the frontmatter `timeout` to 30. At launch, `SessionStart` runs in the background
  (`code.claude.com/docs/en/hooks`).

**P2-m2 — m4 still leaks every `FALKORCHAT_*` variable not on the list.** I exported
`FALKORCHAT_EMBEDDING_MODEL`, `FALKORCHAT_STOREFRONT_ENABLED` and `FALKORCHAT_AGENT_ID`, and the
launched stub saw all three. The server reads about 25 such variables (`config.py`). One leaked
legacy model variable makes the server refuse to start (`config.py:57-75`). Others, such as the
storefront, workflow or thread-limit switches, silently reshape the always-on server. The matrix
asserts exactly the six names it was given, so it cannot see a seventh.
This finding comes from my own Pass 1 remedy, which was a list.
**Fix:**
- before pinning, unset every `FALKORCHAT_*` variable by enumeration
  (`compgen -e | grep '^FALKORCHAT_'`, then one `env -u` per name);
- make the matrix export a sentinel `FALKORCHAT_ZZ_SENTINEL`;
- assert that the only `FALKORCHAT_*` variable the stub sees is `WS_ID=agent-team`.

**P2-m3 — After the m3 fix, the m1 "FalkorDB unclear → don't launch" gate costs a start and buys
nothing. Separately, every launch call reports a start even when the lock refused it.**
Since `e7a020cc`, `start_agent_team.sh` waits for a running `falkordb-dev` that answers
`-LOADING`. When `redis-cli` is missing or another container answers, the script fails harmlessly
under the lock (traced, not run). Meanwhile the gate (`:127`) leaves the server down until teco's
next step 1.

That is also the likely state for a second session started a few seconds after a first one began
the bring-up. FalkorDB is up but still loading, and the second session is told "not started"
while the first session's launch is running. Conversely, a step-1 run during a boot that is
already in flight says "has been started detached", although `flock -n` refused it.
**Fix:**
- launch on curl exit 7 regardless of FalkorDB's state, under the lock (or keep the gate and
  document why);
- before any launch, test the lock (`flock -n "$SERVER_LOCK" true`). If it is held, report
  "already being started (log: …)" instead of claiming a start.

**Nit — the `date -d "$bt"` parse test (`:193,200`) accepts `''`, `now` and bare digits.** A
pipeline marker with an empty `BUILT_AT` was reported fresh against my scratch FalkorDB. That
shape is outside the recipe's list, but the hook's own "unparseable on a pipeline marker" rule
(`:201`) is meant to catch it. Match an ISO-8601 pattern instead.

**Answered, no finding:**
- **Q1, `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_ENABLE_AGENT` unsets.** These are safe.
  `FALKORCHAT_OPENCODE_CONFIG` falls through to the script's documented local-override →
  shared-default resolution, and `FALKORCHAT_ENABLE_AGENT` defaults to the required `1`.
- **Q1, per-goal vs. start-time run.** These cannot double-launch, because both take the same
  `flock` lock. They can double-report; see P2-m3.
- **Q2, CPG classification.** It matches `freshness.md` on every shape I seeded and read: zero or
  several markers, `markerOrigin` non-null, null provenance, `none`, a value outside the closed
  set, dirty, null tree or origin, an origin that doesn't resolve, the `HEAD:./` + `--verify` form,
  and stale with a correct commit count (5, matching `git log`).
- **Q2, budget.** It fits 15 s apart from P2-m1. When FalkorDB is not up, the script skips the
  check and says so.
- **Q2, sourcing.** The recipe is cited (header `:16-23`), not copied.
- **Q2, step 3.** It now says "This dispatch-time check is the authoritative one".

### Done-condition for the last revision (falsifiable; meeting it closes the gate without a Pass 3 review)

1. `grep -c 'timeout 2 cat' claude/teco/hooks/ensure-environment.sh` → `0`, and teco's frontmatter
   `SessionStart` timeout is ≥ `30`.
2. A tracked test script exists beside the hook and exits 0 on the shipped script. It exits
   non-zero on each of three mutants:
   - `MARKER_ORIGIN`→`MARKER_ORIGN`;
   - `!= "false"`→`= "true"`;
   - the `GRAPH.LIST` error reply treated as "no graphs".
3. In that test, with a sentinel exported, the server-launch cell records exactly one
   `FALKORCHAT_*` variable: `FALKORCHAT_WS_ID=agent-team`.
4. In that test, with the server lock held by a dummy `flock`, a refused `/health` produces a
   message containing "already being started" and no "has been started".
5. A pipeline marker with `BUILT_AT:''` is reported "not assessed".

