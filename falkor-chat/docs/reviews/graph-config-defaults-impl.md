# GRAPH.CONFIG defaults — implementation review (U5, deployment config)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-098

## Scope & verdict

**Reviewed:** the uncommitted U5 diff by `devops`. That is `falkor-chat/config/falkordb.env` (new),
`scripts/start_falkordb.sh`, `compose.yaml`, `README.md`, `AGENTS.md` and
`server/tests/test_start_falkordb_script.py` (new). The baseline is
`falkor-chat/docs/plans/graph-config-defaults-graph.md` §5 Steps 1–3 and their done-condition,
plus the Step 4 rollback lines that depend on the script. The plan's review is
`falkor-chat/docs/reviews/graph-config-defaults.md` (Pass 3, approve). The evidence is under
`falkor-chat/docs/reviews/graph-config-defaults-evidence/devops-u5/`. I did not review the other
modified files.

**Verdict: approve with suggestions.** There are no blockers and no majors. The diff matches
Steps 1–3. The three hardening departures leave the rollback path untouched. Two of them close
real holes, and one of those holes is in the parser the note itself prescribes (see *What's
solid*). The four minors are listed below. One of them (m1) is a done-condition sweep that missed
two stale compose invocations.

CPG: considered, not relevant — the diff is bash, YAML, a dotenv file and a subprocess-driven
pytest. `cpg_falkorchat` models the Python package, and none of this diff's behaviour lives there.

### What I ran (all verified, none inferred unless marked)

- **pytest.** `tests/test_start_falkordb_script.py` from `falkor-chat/server`: **19 passed**. It is
  FalkorDB-free: docker is stubbed, and compose runs only `config`. It does not wipe `reference`.
- **Live readback, through the edited script on my own throwaway containers.** I used
  `k098-analyst-u5-*` on ports 16381–16383, with `FALKORDB_VOLUME` pointing at scratch dirs.
  - Default start: all eight keys read back as Step 1's values (`ASYNC_DELETE yes` reads back as `1`).
  - Positive control, `FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"`:
    the image's values came back (`OMP_THREAD_COUNT 8`, `QUERY_MEM_CAPACITY 0`, `MAX_QUEUED_QUERIES 25`).
  - Misspelled key (`QUERY_MEM_CAPCITY`): the server aborted, and the script still exited 0.
  - No leftovers. `falkordb-dev` stayed `Up 45 hours` and was never touched.
- **devops's evidence**, read and spot-checked: `live_throwaway.txt` includes the `ASYNC_DELETE 1`
  abort and its kept log (`Can't load module … server aborting`). `mutants.py` is applied to
  copies only, and its 20 mutants are well-formed (each `old` string is asserted to be present).
  I did not re-run all 20. I ran two of my own against copies (m2).
- **Compose.** I ran 20 subcommands, with neither variable set, on a scratch copy under a separate
  project name (`k098-analyst-probe`). Results are under *Compose `:?`*, below.
- **Parser edge probes**: 16 shapes against the script copy, and the same shapes through
  `docker compose --env-file … config` (m2, Appendix A).

## Findings

### m1 (minor) — two compose invocations still lack `--env-file`, and both now refuse

An unfiltered `git grep -n "docker compose"` over the repo (excluding reviews, plans, history,
test reports, backlog and archive) finds two hits that U5 did not edit:
- `falkor-chat/docs/manuals/operations.md:58-59`, `docker compose up --build` / `docker compose down`.
  This is the end-user operations manual, path A.
- `falkor-chat/Dockerfile:9`, `docker compose up            # preferred …`.

Each of these now fails with `required variable FALKORDB_THREAD_COUNT is missing a value`. That
failure is loud and self-explaining, so it is minor. But the manual is the document an operator
follows. The done-condition named only `compose.yaml:3-4` and `README.md:103-104`, so the gap is
in the note's sweep, not in devops's execution. **Fix:** `devops` edits `Dockerfile:9` now, as a
one-line comment change in the same unit. Route `operations.md` §1 path A to `tico` as part of
K-098's close (it is a manual, not live spec; see *Close-list additions*).

### m2 (minor) — the script and compose read the same env file under different grammars

The script reads each line literally. Compose applies dotenv rules. The two agree on the shipped
file, and the parity test pins that. They disagree on these shapes (Appendix A):

| Shape | Script | Compose |
|---|---|---|
| duplicate key | first wins | last wins |
| `"…"` around the value | quotes kept | quotes stripped |
| `X=8 # cores` | `# cores` kept | comment stripped |
| `$HOME` | literal | expanded |
| trailing `=` | stripped by `read` | kept |

The duplicate-key case is **silent drift**. On a copy with `FALKORDB_THREAD_COUNT=4` appended,
the script still passes `8` and compose renders `4`. Only `test_compose_renders_the_same_string…`
goes red (1 failed), and that test skips when the docker CLI is missing. It also guards only the
shipped file, not a deployment's edited copy. The quote and comment cases make the script hand
FalkorDB a junk token, and FalkorDB then aborts (an unknown key aborts, as measured above). Under
`-d` that abort is invisible (m3). The diff's comment claims "a key compose would honor … cannot
drift". That holds for key names, not for these shapes.

**Fix (prototyped on a copy, verified):** read whole lines with `IFS= read -r line`, strip CR
from the line rather than from the value, and split on the first `=`. Then refuse three things: a
line with no `=`, a value outside `^[A-Za-z0-9_. -]*$`, and a repeated key. With that rule, any
file the script accepts renders identically in compose. Appendix B shows the prototype and its
results. 18 of the 19 tests still pass. `test_env_file_is_parsed_not_sourced` must change from
"passed through verbatim" to "refused, marker absent", because `$(…)` is now out of grammar.

**Test (coverage probe, not a list of shapes):** add one parametrized test over the dotenv grammar
axes compose varies on: quoting (`"`/`'`), inline `#`, `$` expansion, `export` prefix, a repeated
key, trailing `=`, leading whitespace, CRLF (blank and comment lines included), and a missing final
newline. For each axis, assert that the script either refuses before `docker run` or produces the
same string as `docker compose --env-file <same file> config`. The assertion that catches a
wrong fix is the equality branch: a parser that accepts a shape and diverges goes red there.

### m3 (minor, runbook note — not a code defect of this unit) — `-d` hides an in-container abort

Reproduced. On a bad value or key, `start_falkordb.sh -d` exits 0. The container aborts, and
`--rm` deletes it together with its log (`no such object`). This behaviour is older than the diff:
any in-container start failure under `-d` has always looked like this. U5 makes such a failure
newly likely by adding a user-edited input. The note already covers it for the runbook: Step 4.1's
dry run runs in the foreground, against the same env file, *before* the live `-d` start. So a bad
value is caught with its log on screen, provided the file is not edited between step 1 and step 6.
The plan does not ask the script to detect the abort, so this is not a done-condition miss.

What is **not** covered: the three automated `-d` callers, `start_server.sh:173`,
`start_demo.sh:221` and `start_agent_team.sh:195`. On a bad file each of them reports only
`FalkorDB did not respond after 30s`. **Fix:**
- Runbook (now): step 6 adds "no `PONG` within N s means the container exited; re-run step 1's
  foreground dry run to see the abort log".
- Backlog follow-up (`devops`, not U5): in the `-d` branch, poll for a few seconds. If the container
  disappears, exit non-zero with "container exited during start — run without `-d` to see the
  log". Pin it with a stubbed `docker inspect` that reports the container missing.

### m4 (minor) — "every compose command refuses" is slightly overstated

The safety property holds. Every subcommand that can create or start a container refuses with
neither variable set: `up`, `create`, `start`, `restart`, `run`, `pull`, plus `config`, `ps`,
`logs`, `stop`, `down`, `rm` and `kill`. Four do not refuse: `images`, `top`, `events` and `exec`
read project labels and never interpolate the model. `config --no-interpolate` is an explicit
opt-out. **Fix:** in the `compose.yaml` and README sentences, say "every compose command that
loads the file" or "compose never starts FalkorDB without it". The plan's §5 Step 3 sentence has
the same wording and can stay as it is (the plan is approved and frozen).

### Nits

- n1. A CRLF file with a blank line is refused as `unknown key ''`. The key is actually `\r`, which
  makes the message misleading. m2's fix (strip CR from the whole line) resolves this.
- n2. `[ ! -r ]` reports "not found" for an unreadable file (`/etc/shadow`). Say "not found or not
  readable".
- n3. `falkor-chat/AGENTS.md` was already over the ~2,500-word smell (2,846 → 2,869). The row was
  rewritten in place, not appended to, and is under the 700-char bar. Nothing to do in U5.

## Questions the brief asked

- **Do the hardening departures change behaviour the note relies on?** No. The note relies on four
  behaviours: the env-file default, the `FALKORDB_THREAD_COUNT` override, a full `FALKORDB_ARGS`
  used as-is, and a missing file failing before `docker run`.
  - All three hardenings live inside `if [ -z "${FALKORDB_ARGS:-}" ]`. The rollback line (Step 4.8,
    the image's own args) never opens the file. I verified this with `FALKORDB_ENV_FILE=/nonexistent`
    (probe, plus `test_full_falkordb_args_used_as_is_without_env_file`) and live through the
    positive control.
  - `FALKORDB_ARGS=""` still falls through to composing from the file, as the note's `:-` form does.
  - The missing-file check only replaces bash's redirect error with a clear message. Both exit
    before `docker run`.
  - The 2769a99c pin still holds: `git log -- scripts/start_falkordb.sh` lists it as the last commit
    before the edit.
- **Is the parse-not-source block safe?** Yes, as shipped.
  - `case` restricts `key` to two literal names *before* `${!key+x}` and `printf -v "$key"`
    evaluate it, so neither ever sees an attacker-chosen name.
  - Values go through `printf -v … '%s'` and are never expanded or eval'd: `$(touch …)` stays
    literal and the marker is absent.
  - `IFS='='` with `read -r` keeps spaces and backslashes in values intact, and `-e
    FALKORDB_ARGS="${FALKORDB_ARGS}"` is one quoted word. Glob characters do not expand.
  - What is left is the grammar divergence in m2.
- **Does compose `:?` refuse in every command?** It refuses in every command that loads the file,
  each variable on its own (`test_compose_refuses_without_each_var`). `images`, `top`, `events` and
  `exec` are the exceptions, and none of them can start FalkorDB (m4). The shell environment wins
  over `--env-file`, the same way it does in the script.
- **The `-d` hazard?** It is a runbook note plus a backlog follow-up, not a U5 code defect (m3).

## What's solid

- **The allowlist hardening fixes a hole in the note's own parser.** The Step 2 block as written
  in the note runs `${!key+x}` on an arbitrary key. I checked it with a planted key,
  `a[$(touch …/pwned)]=1`: bash evaluated the array subscript, ran the command and created the
  marker. The shipped `case` refuses that key (`unknown key`), with no marker. So "parsed, not
  sourced" was false for the note's own block and is true for the shipped code.
- **The tests' expected values are independent.** `_EXPECTED_ARGS` is a hand copy of the note's
  values, never read from the env file, so value mutants M13–M17 and teco's typo mutant go red.
- **There is a positive control.** The image's own args read back as the image's defaults, which
  proves the readback can tell old from new.
- **The parity test runs the real script against the real compose renderer.**
- **Both `docker run` branches are pinned**, by M1/M2 and the parametrized test.
- **Documentation edits** replace the existing rows and blocks rather than appending.
- **The `config/falkordb.env` header comment** states its booleans rule and the two-key limit.

## Close-list additions (live spec, note §6)

This unit's diff adds one live-spec sentence to the §6 list:
- **`falkor-chat/docs/SERVER.md` §1.5 Layout.** The tree line
  `config/{models.json, opencode.example.json, logging.json}` gains `falkordb.env`.

Optional, if the DESIGN §10 tuning-bullet writer wants the contract complete: add the clause "compose refuses to
interpolate the file without `--env-file`". This is not a live-spec item: the
`falkor-chat/docs/manuals/operations.md` §1 path A compose lines go to `tico` at the same close
(m1).

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: m2 of this review. The class is two readers of
one config file that drift on a grammar shape that only one of them handles.

```markdown
- When two programs read the same config file (a shell parser and compose's dotenv, a script and a loader), probe each grammar axis either reader treats specially (quoting, inline comments, `$` expansion, `export`, repeated keys, CRLF, trailing delimiters) and require that each one either refuses or renders identically in both. Parity on the shipped file alone is a bound.
```

No prune is proposed. No existing line in root `REVIEW.md` is now enforced by a test or hook.

## Appendix

### A. Parser divergence probes (script copy with stubbed docker vs `docker compose --env-file … config`, scratch project)

| Input (env file) | shipped script | compose |
|---|---|---|
| `…THREAD_COUNT=8 … THREAD_COUNT=4` (dup) | `THREAD_COUNT 8 …` rc 0 | `THREAD_COUNT 4 …` |
| `MODULE_ARGS="TIMEOUT 1000 CACHE_SIZE 25"` | `THREAD_COUNT 8 "TIMEOUT 1000 CACHE_SIZE 25"` rc 0 | quotes stripped |
| `THREAD_COUNT=8 # cores` | `THREAD_COUNT 8 # cores TIMEOUT 1000` rc 0 | `THREAD_COUNT 8 TIMEOUT 1000` |
| `MODULE_ARGS=TIMEOUT $HOME` | literal `$HOME` | `/home/<user>` |
| `MODULE_ARGS=TIMEOUT 1000=` | `TIMEOUT 1000` | `TIMEOUT 1000=` |
| `export THREAD_COUNT=8` | refused (unknown key) | accepted |
| CRLF + blank line | refused, `unknown key ''` | accepted |
| `THREAD_COUNT = 8`, `  # c` | refused | (not compared) |
| `FALKORDB_THREAD_COUNT=` exported empty | refused by `:?` | refused by `:?` |
| `FALKORDB_ENV_FILE=/tmp` (dir) | `read error`, then `key: unbound variable`, rc 1 | n/a |

Mutants on copies of the shipped env file: appended duplicate → 1 test failed (the compose parity
test only); value wrapped in quotes → 5 failed.

### B. m2 prototype (scratch copy only; the repo is untouched)

```bash
  seen=" "
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    key="${line%%=*}" value="${line#*=}"
    case "$key" in
      FALKORDB_THREAD_COUNT|FALKORDB_MODULE_ARGS) ;;
      *) echo "start_falkordb.sh: unknown key '$key' in $FALKORDB_ENV_FILE" >&2; exit 1 ;;
    esac
    if [ "$line" = "$key" ] || [[ ! "$value" =~ ^[A-Za-z0-9_.\ -]*$ ]]; then
      echo "start_falkordb.sh: $key in $FALKORDB_ENV_FILE: KEY=value, value of letters, digits, '_', '.', '-' and spaces only" >&2; exit 1
    fi
    case "$seen" in *" $key "*) echo "start_falkordb.sh: duplicate key '$key' in $FALKORDB_ENV_FILE" >&2; exit 1 ;; esac
    seen="$seen$key "
    if [ -z "${!key+x}" ]; then printf -v "$key" '%s' "$value"; fi
  done < "$FALKORDB_ENV_FILE"
```

Results:
- `bash -n` is clean, and 18 of 19 tests pass. The one failure is
  `test_env_file_is_parsed_not_sourced`, which expected `$(…)` to be passed through and now sees it
  refused.
- The shipped file, CRLF with a blank line, a missing final newline, an env override, and the
  rollback with no env file all compose as before.
- Each of these is refused before `docker run`: a duplicate key, quotes, an inline `#`, `$`, a
  trailing `=`, a line with no `=`, and the subscript-injection key (marker absent).

## Pass 2

**Verdict: approve.** Pass 2 found nothing new above nit level.

**Re-run:**
- The test file, from `falkor-chat/server`: **46 passed, 0 skipped**. The compose-parity tests and
  the 25-axis test ran.
- A live readback through the Pass-2 script on throwaway `k098-analyst-u5-p2` (port 16384,
  scratch volume): all eight Step 1 values were read back. No leftovers, and `falkordb-dev` was
  untouched.

**Dispositions:**
- m1 — fixed for `Dockerfile:9`. `docs/manuals/operations.md:58-59` is still unedited, by design:
  it routes to `tico` at close.
- m2 — fixed. devops overruled my character class, and devops is right. Probe on a scratch compose
  project: compose trims ` 8` and `8 ` to `8`. My class `^[A-Za-z0-9_. -]*$` would have accepted
  both, so the two readers' strings would have differed. The shipped class
  `^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$` refuses both.
- m3 — runbook note and backlog follow-up, unchanged (not U5 code).
- m4 — fixed in `compose.yaml` and `README.md`. The commands that do not load the file
  (`images`/`top`/`events`/`exec`) are now named.
- n1 — fixed. A CRLF file with a blank line is accepted (axis `crlf-with-blank-and-comment`).
- n2 — fixed. A directory or an unreadable path gets "not found or not readable" (M34, M35 killed).
- n3 — no change, as expected.

**25-axis grammar test.** It is a real coverage probe. Each axis either refuses before
`docker run` or matches `docker compose config` for the same file. I checked it against nine
axes of my own, script copy against compose:

| Axis | Compose | Script |
|---|---|---|
| double inner space | keeps `TIMEOUT  1000` | refuses |
| tab | keeps `\t` | refuses |
| backslash | keeps `10\n00` | refuses |
| backtick | keeps `` `id` `` | refuses |
| `\r\r\n` | trims | refuses |
| UTF-8 BOM | accepts | refuses (`unknown key`) |
| empty value | `:?` | refuses |
| leading/trailing space | trims | refuses |
| shipped file | — | matches compose |

No shape is accepted by the script and rendered differently by compose.

**Equivalence claims (executed on copies):**
- **M11 `eval` is confirmed equivalent.**
  - I tried all 255 bytes against the shipped class under `C` and `C.UTF-8`. The accepted set is
    exactly ASCII alphanumerics, `_`, `.`, `-` and space. None of `$`, `` ` ``, `\`, `"` or `!` is in
    it, and those are the only characters a double-quoted `eval` treats specially.
  - 500 random accepted values gave 0 differences between `eval "K=\"$v\""` and `printf -v`.
  - Bound: `en_US.UTF-8` is not installed on this box, so a locale whose regex ranges collate
    differently was not tested. The shipped code uses `printf -v`, so this only matters if someone
    later swaps in `eval` and also widens the class. M28 (class admits `$`) is killed.
- **M32 (split on the last `=`) is outcome-equivalent, not message-equivalent.** I ran five
  two-`=` lines (`A=B`, trailing `=`, `KEY=8=`, a leading `=`, `KEY==8`). The shipped script and
  the mutant both refuse every one before `docker run`. On four of them the shipped script says
  `write KEY=value` and the mutant says `unknown key`. That is a nit, and acceptable: no test pins
  the diagnostic for a two-`=` line.
- **M33 (`IFS='='` with one variable) is confirmed equivalent.** `K=a 1=`, `=K=a=` and `K==` are
  read back unchanged. The same five probes give the same outcome and message as the shipped
  script.

### Proposed REVIEW.md lines (final, supersedes Pass 1's)

Target: root `REVIEW.md`, `## Always check`. Source: m2 of this review and its Pass 2 edge-space
reversal.

```markdown
- When two programs read the same config file (a shell parser and compose's dotenv, a script and a loader), probe each grammar axis either reader treats specially (quoting, inline comments, `$` expansion, `export`, repeated keys, CRLF, trailing delimiters, edge whitespace that one reader trims) and require that each one either refuses or renders identically in both. Parity on the shipped file alone is a bound, and so is a reviewer-proposed character class until it has been run through the same probe.
```

No prune.
