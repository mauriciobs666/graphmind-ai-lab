# cpg/sandbox — offline build sandbox

> **Status:** active · **Owner:** `devops` · **Tracks:** C-1001 (P0, S08 / P0-07)

`offline-run.sh` runs a Joern frontend, `joern-parse` or a `joern --script` extractor with no
route out. Its contract is `docs/plans/connected-code-graph-egress-spec.md` v1.3 (§7, §5.3, §10).
Where that spec and the P0 plan's §2 D6 disagree, the spec controls. The two review items from
`docs/reviews/connected-code-graph-p0-impl.md` § S04 Pass 3 are built in:

- **P3-m1:** the `//>` refusal strips all ASCII whitespace.
- **P3-n1:** `/scripts` is a directory bind.

| File | What it is |
|---|---|
| `offline-run.sh` | The runner. Sourced, it only defines functions; the tests use those. |
| `forbidden-flags.txt`, `allowed-flags.txt` | The §7.3.3 flag lists, normative: the Joern v4.0.579 entries, plus the `scip-typescript` entries the indexer sweep (test 24) classifies |
| `tests/test_sandbox.sh` | §10 tests 1–10 on both backends, plus tests 11–15 (below) and the indexer tests 21–26 (egress-spec3 §10.1) |
| `tests/probe.pl` | The escape-axis probe, axes A1–A7. It is Perl because the image has no Python. |
| `tests/mutants.sh` | The mutation harness: each mutant must turn its test red |
| `scripts/cpg-content-check.sc` | **The** CPG content check (plan `docs/plans/connected-code-graph-p0-2.md` §2 D15, P3-7). S11, S25, S28 and S39 run it through the runner; only S08 edits it (plan §5.2). |

## Using the runner

```bash
cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern-parse \
    --source <repo dir> --output <out dir> --codebase <repoId> -- /src --language pythonsrc
cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern \
    --script-file extract.sc --source <dir> --output <cpg dir> \
    -- --script /scripts/extract.sc --param cpgFile=/out/<auditId>.cpg.bin
```

**Indexer entries** (`docs/plans/connected-code-graph-egress-spec3.md` §7.4–§7.8, P1-48b):

```bash
CCG_TOOLCHAIN_IMAGE="$(cpg/infra/toolchain/build.sh --indexer --print-tag)" \
cpg/sandbox/offline-run.sh --backend container --mode offline --frontend scip-python \
    --source <staged dir> --output <out dir> -- index --cwd /src --environment /src/env.json
```

- `scip-typescript` and `scip-python` run from `/opt/indexer/bin/<name>` (a fixed path, not `PATH`),
  `--mode offline`, container backend only. `controlled` is `unknown-frontend`; `--backend
  namespace` is `backend-unavailable` (and `auto` means the container).
- The image must carry the label `ccg-toolchain-indexer=1`; otherwise `backend-unavailable`, exit 69.
- `toolArgv` is `index` then `--cwd <p>` (mandatory), `--environment <p>` (`scip-python`, mandatory),
  `--project-version <v>` (`scip-python`, mandatory) and `--project-name <n>` (`scip-python`), each at
  most once, in any order, option and value as two elements. `<v>` and `<n>` match
  `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`. `<p>` is `/src`, `/src/` or `/src/` plus components that are
  non-empty, not `.` or `..`, and match `^[A-Za-z0-9._@+-]+$`. A forbidden element, including a missing
  leading `index`, is `forbidden-flag`; a missing mandatory element alone is `usage`.
- **Containment (`sbx_idx_walk`).** After the grammar passes, each `--cwd`/`--environment` value is
  walked on the host component by component from the staged tree. Refused as `forbidden-flag`, and
  ahead of any `usage` refusal: a symlink with an **absolute** target (inside the container it
  resolves against the image, so even a host path that lies in the tree is refused), a `..` reached
  through a relative link that leaves the tree, a component that does not exist (a dangling link, a
  missing `env.json`), and a `--cwd` that is not a directory. A relative in-tree link is followed.
  The stager's own symlink rejection (P1-15) stays; this is defence in depth, and a tree changed
  after the check is the stager's and operator's concern.
- **The image is launched by its `sha256:` ID**, for every container launch: the ID the runner
  inspected and recorded in `toolchain.imageId` is the one that runs, so a retag between inspect and
  launch changes nothing. The label `ccg-toolchain-indexer=1` guards against a wrong image only; it
  is not an integrity control, since `CCG_TOOLCHAIN_IMAGE` is caller-chosen.
- The runner appends `--output /out/<auditId>.scip` and refuses a caller `-o`/`--output`. A non-zero
  indexer exit means there is no artifact: a partial `<auditId>.scip` may remain in `/out`, and the
  consumer (P1-15) must not read it.
- The pass-list gains `PYTHONSAFEPATH=1`: `scip-python` runs `python3 -c` with the staged
  tree as its working directory, and this keeps it off `sys.path` (test 26).
- `scip-python` 0.6.8 exits 1 with neither a git revision nor `--project-version`; the mandatory
  `--project-version` is why no `.git` is staged. With the flag it runs no `git` child.

**Exit codes.**
- The tool's own exit code, when the tool ran.
- `64`: refused before launch (argv, frontend, mode, directive or usage).
- `69`: the backend refused or is unavailable, or the runner's Docker-config directory failed
  its check (see **Environment**).
- `128+n`: the runner got signal *n* (INT, TERM or HUP) while the sandbox ran. It removed every
  container and volume labelled with its `auditId` and wrote no record (§5.3: a killed runner
  leaves none).
- `124`: an indexer entry ran past its wall-clock cap, `SBX_INDEXER_WALL_S` (1800 s). See
  **Indexer wall-clock watchdog** below. The one record is written, `refused: false`, `exitCode: 124`.
  124 means "watchdog timeout **or** a tool that exited 124 itself": the tool's own status passes through,
  so a consumer must not read 124 as proof that the cap fired (`durationMs` against the cap tells them apart).
  A race at the cap boundary can record 124 for a run that had just succeeded; that is fail-closed (no
  artifact is used).

**Audit record.** Every invocation, refusals included, appends one `sandbox` record (§5.3) to
`$CCG_AUDIT_DIR/sandbox.jsonl`. The default is `cpg/.audit/`, which is gitignored.

**Environment** (`docs/plans/connected-code-graph-egress-spec2.md` §1.7, a closed rule). Only
two caller variables are inputs: `CCG_AUDIT_DIR` and `CCG_TOOLCHAIN_IMAGE` (default:
`cpg/infra/toolchain/build.sh --print-tag`, run under `env -i`). The runner enforces it:
- **The shebang is `#!/bin/bash -p`.** bash then ignores `BASH_ENV`, `ENV`, `SHELLOPTS` and
  exported functions at startup.
- **`sbx_main` re-execs itself under `env -i`** unless its environment is exactly these
  variables. Keys and values are read from `/proc/$$/environ`, the block the runner was exec'd
  with, which is the caller's input. Two other sources are rejected. `compgen -e` lists only
  identifier names, so an entry such as `A-B=1` or `BASH_FUNC_f%%` would slip past it; such
  entries now force the re-exec (review S08 p3-n1). The live shell variables are rejected too:
  on this host a hook was observed appending `~/.local/bin` to `PATH` inside an `env -i` bash
  after exec. That is not caller input, and if it happened in every bash the re-exec would
  never settle. The allowed set is:
  - `PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`;
  - `DOCKER_HOST=unix:///var/run/docker.sock`, which is pinned: a different daemon is a runner
    edit, not an input;
  - `DOCKER_CONFIG=<home>/.local/state/ccg/docker-config`, so no user `config.json` is read
    (contexts, `proxies`, `credsStore`);
  - `HOME` from the passwd entry of the runner's uid;
  - the two `CCG_*` names.
- **The Docker-config directory is checked at every run.** It is created if absent (mode
  `0700`), and must then be a plain directory (not a symlink), owned by the runner's uid, mode
  `0700`, **empty**, not under `/tmp`, `/var/tmp` or `/dev/shm`, and under no world-writable
  ancestor. Any failure refuses with **69 `backend-precondition`** and one record, before any
  `docker` call. A 64 still takes precedence. The Docker CLI writes nothing there during
  `info`, `image inspect` or `run`, so the directory stays empty run after run (verified).
- **Scope:** the contract covers the runner launched through its `#!` line, which is how S11 and
  the tests call it. `bash offline-run.sh` starts bash explicitly, so a hostile `BASH_ENV` runs
  before line 1. A caller doing that already executes arbitrary code, so the case is out of
  scope (§1.7).

**Conventions the runner owns.** §5.3 says only that the tool's output files carry the
`auditId` as their stem; these two rules make that true:
- For a parse frontend, the runner appends `--output /out/<auditId>.cpg.bin`, and it refuses a
  caller-supplied `-o`/`--output`.
- In any tool argument, the token `@AUDIT_ID@` is replaced by this invocation's `auditId`.
  For example, a script's `--param out=/out/@AUDIT_ID@.json`. A launched record's `toolArgv`
  is the argv as executed, after substitution and with the runner's `--output`. A refused
  record's `toolArgv` is the argv the caller passed after `--`, verbatim
  (`docs/plans/connected-code-graph-egress-spec2.md` §1.5).
- `--help` exits 0 and writes no record. It is the one invocation outside "exactly one per
  invocation".

**Content check (`scripts/cpg-content-check.sc`).** `joern-parse` can exit 0 with an empty CPG
(egress §9 H12, §11 R11), so a parse is certified by its content. Run the check in script mode:

```bash
cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern \
    --script-file cpg/sandbox/scripts/cpg-content-check.sc --source <any dir> --output <cpg dir> \
    -- --script /scripts/cpg-content-check.sc --param 'cpgFiles=/out/<a>.cpg.bin;/out/<b>.cpg.bin'
```

For each CPG (`;`-separated), it prints:
- `METHODS <cpg>=<sorted distinct non-external METHOD names>`, the line test 3 always read,
  unchanged;
- `METHOD_COUNT <cpg>=<n>`;
- one `FILE_METHODS <cpg>\t<file>=<n>` per `FILE` node;
- `CONTENT <cpg>=ok|EMPTY`.

The verdict is the unit-level floor, as before: `EMPTY` iff the CPG has no non-external
`METHOD`. The check exits 1 if any CPG is `EMPTY`. The per-file counts are reported, not judged:
whether a per-file rule holds is S11's to measure and adopt (D15).

Test 3 runs the check on all seven FR-1 CPGs (all `ok`). It also runs it on a real exit-0 empty
CPG, `jimple2cpg` (a bytecode frontend) on Java **source**, per §9 H13: 5.6 KB, 0 methods. That
one must be `EMPTY`, with a non-zero exit. Mutant M43 (accept an empty CPG) is killed. The
record's `script.sha256` ties each run to the reviewed bytes.

**Heap.** `JAVA_OPTS` is fixed at `-Xmx2g`. To raise a frontend's heap, pass `--frontend-args
-J-Xmx<n>` (or `-J-Xmx<n>` to a direct frontend): the frontend's own launcher consumes it. A
`joern-parse -J-Xmx<n>` *before* `--frontend-args` sizes only the `joern-parse` JVM, never the
frontend it spawns (§7.3.5).

**One run per `--output`.** The runner takes `flock -n` on the output directory's own file
descriptor, so the lock creates no file. A second run into a directory that is in use is
refused (exit 64, `usage`). Test 3 runs two invocations concurrently and expects exactly one
refusal (egress-spec2 §1.3). Without the lock, three concurrent runs into one `--output` all
exited 0 (S08 review m2). On this 5.15 kernel overlayfs only warns about a shared upperdir, and
the result is undefined. The lock lives outside the upperdir, as §1.3 requires, so a tool cannot
unlink it.

**Signals** (S08 review p2-n1, p3-m1). The runner waits on the launcher as a child. On INT, TERM
or HUP, `sbx_on_signal` does three things, in this order:
1. **It kills the Docker client (`SIGKILL`) and waits for it.** A TERMed `docker run` proxies the
   signal to the container and stays up. The client is found with `jobs -p`, so a signal that
   lands just after the fork still finds it.
2. **It removes every container and volume labelled `ccg.sandbox.auditId=<auditId>`**, and
   repeats until three checks 0.2 s apart find none (`sbx_remove_own`, capped at about 10 s). A
   create request the dead client already sent can still complete on the daemon, and `--rm`
   never fires for a container that never started. So removing first (the p2-n1 shape), or
   killing first with one removal pass, both leak a `Created` container and its volumes. Test 13
   measured that: 11 of 20 and 17 of 40 trials.
3. **It cleans its scratch and exits `128+n`.** The `--output` lock is held throughout, so it is
   never free while our container still writes.

**Indexer wall-clock watchdog** (egress-spec4 §1). For `scip-typescript` and `scip-python` only (Joern
frontends are unchanged), `sbx_launch` starts a timer: a background subshell that waits
`SBX_INDEXER_WALL_S` (in steps of at most 30 s) and then sends the runner `USR2`. The constant is `1800`,
assigned unconditionally; no environment variable sets it (a test sources the runner and reassigns it,
which is the only hook). On expiry `sbx_on_wall` takes `sbx_on_signal`'s steps (KILL the client,
`sbx_remove_own`, `sbx_cleanup`), then writes the invocation's one record with `exitCode: 124`,
`refused: false`, and exits 124. A bare `timeout` would not do: a hung node may ignore the TERM that
`docker run` proxies.
- **Cancelled on normal exit.** USR2 is set to ignored first, then the timer gets TERM (its trap ends it
  and its sleep); the runner never `wait`s on the timer (a wait hung intermittently). The disposition stays
  "ignored" afterwards, which is harmless: the sandbox client is started while the handler is set, and a
  child resets a handled signal to the default.
- **First wins.** Each handler ignores USR2 on entry, so a signalled run never also writes a watchdog
  record. Both handlers also **ignore** INT/TERM/HUP while they remove containers: a second signal cannot
  abort the removal and leave the container running. On expiry the timer has already ended, so the
  handler does not signal its (possibly recycled) PID.
- **A runner killed without a trap** (SIGKILL, OOM) cannot cancel its timer. The timer therefore closes the
  runner's `--output` lock fd (it never pins the lock), checks every 30 s and at the cap that the runner is
  still the same process (`/proc/<pid>/stat` start time, recorded at arm; a bare `kill -0` would not catch
  PID reuse), and exits silently if not. It sends USR2 only to the runner it armed for. No `pkill`
  is used, so nothing is needed beyond bash and procfs.
- With INT/TERM/HUP ignored during cleanup, a wedged cleanup (for example a hung Docker daemon) is ended
  only by SIGKILL, which leaves the container. P1-15 must therefore run the runner under `timeout -k` (or
  otherwise escalate to KILL) after cap + 60 s, and compare `durationMs` with the cap, since exit 124 alone
  is ambiguous.
- If the runner's start time cannot be read (`/proc` unreadable), no timer can be armed safely, and an
  indexer must not run uncapped: the launch is refused (69, `backend-unavailable`). The start time is read
  with `read -d ''`, so a newline in the runner's process name cannot hide field 22.
- A 124 record is not a successful indexer run: it has no artifact, and P1-15's own timeout must exceed
  the cap plus 60 s.

A `SIGKILL` to the runner cannot be trapped: the container then runs to its end under `--rm`,
and the lock is free meanwhile. The launch stays one `docker run`. Splitting it into `docker
create` + `docker start -a` would make the container's existence known without polling, but it
changes the launcher that §7.1 and the record's `launcherArgv` specify, which is a spec change.

**Container launch.** The shape is §7.1's: `docker run --rm --pull=never --network none
--hostname localhost --read-only --cap-drop ALL --security-opt no-new-privileges --user
<uid>:<gid> --pids-limit 4096 --tmpfs /tmp:rw,exec,nosuid,nodev --tmpfs
/home/ccg:rw,noexec,nosuid,nodev --entrypoint /usr/bin/env <image> -i <13 pass-list pairs> <tool>…`.
Each container is named `ccg-sbx-<auditId>`. Each container and its anonymous volumes carry
the label `ccg.sandbox=s08`; `--rm` removes them.

**How the binds get `noexec`** (§7.1.2, §11 R2: achieved, and proven from mountinfo by tests 1
and 5). Docker's bind mounts have no `noexec`. The local volume driver's `o=bind,…noexec` applies
the flags only when `ro` is present: moby re-mounts a bind only for `MS_RDONLY` (verified: a
`bind,noexec` volume mounts `rw` *without* `noexec`). So:
- **`/src` and `/scripts`** are local-driver `type=none` volumes with
  `o=bind,ro,noexec,nosuid,nodev`.
- **`/out`** is a local-driver **overlay** volume. Its `upperdir` is the output directory, its
  `lowerdir` is empty, and its options are `noexec,nosuid,nodev`. Writes land directly in the
  output directory, owned by the invoking user.
  - Overlay needs its `workdir` on the same filesystem as the output directory. The runner
    creates `.ccg-ovl-<auditId>.*` beside `--output` and removes it after the run. So
    `--output`'s parent directory must be writable.
  - Concurrent runs into the **same** `--output` are refused by the lock above. Give each run
    its own directory.

## Backends on this host

| | Container | Namespace (bwrap) |
|---|---|---|
| Through the runner | runs; all tests pass | **refuses, exit 69** (`backend-precondition`). N1: `WSLInterop-late` is enabled (§7.1.1 scan). N2: `/dev/vsock` exists. Test 1 and test 8's Host row assert this refusal. |
| A1–A3, A5, A6 | probe passes (test 1) | probe passes, on the bwrap template run **directly** from the test, per §11 R5. It is not run through the runner, so the runner has no bypass. |
| A4, A7 | probe passes | **not exercised**: the backend refuses on every WSL2 host, and on a non-WSL host neither N1 nor N2 would fire. They stay unexercised until a non-WSL host exists (§11 R5). |
| `--help` sweep (test 6) | 16 hits: 13 forbidden, 3 allowed, 0 unclassified | the same 16 / 13 / 3, using the host Joern (`~/joern/joern-cli`, v4.0.579) under the template |

## Deviations from the S04 spec (for the S08 gate)

1. **Namespace `/usr` bind set.** The spec's `--ro-bind /usr /usr --tmpfs /usr/lib/wsl` fails
   A5(b) on this host.
   - Why: bwrap's `--ro-bind` is always recursive, so mountinfo still lists the 9p
     `/usr/lib/wsl/drivers` and the overlay `/usr/lib/wsl/lib` under the tmpfs mask.
   - Instead, the runner binds `/usr` entry by entry and skips every host mount point.
     `/usr/lib/wsl` is left out entirely, not masked.
   - It then adds `--remount-ro /`, so the root is read-only as in the container. It also
     generates `/etc/group` and `/etc/nsswitch.conf` (`hosts: files`), so `getent hosts
     localhost` answers from `/etc/hosts`.
   - Mutant M13 (the spec-literal template) is killed by test 1's namespace A5b.
2. **`/out` is an overlay volume**, not a `bind,ro,noexec` local volume: see above. The overlay
   scratch directory is the new cost.
3. **`refusalReason: usage`** is an added value. It covers runner-level errors: no arguments, an
   unknown runner option, a missing or invalid `--backend`/`--mode`/`--`, a path the mount syntax
   cannot express (`, " : =` or a newline), a malformed flag list, `--script-file` given without
   `--frontend joern`, and an `--output` in use by another run. The
   spec's enum is closed and has no fitting value. The spec owner should adopt `usage` or name
   another.
4. **`network`/`networkName`** are always `none`/`null` in P0, including on a refused
   `--mode controlled` record. No network is ever created, so the spec's "`none` iff
   `offline`" would misstate a refusal. An invalid `--mode` value is never recorded:
   `egressMode` keeps the valid requested mode, else `offline`.
5. **`preconditions.netnsDistinct`** is `true` only once a launcher exists, by construction:
   `--network none` and `--unshare-all` always create a new network namespace. A1(c) verifies
   it at run time, by comparing the sandbox's `/proc/self/ns/net` with the host's. A refusal
   record says `false`. The other preconditions come from a read-only scan at start, so even a
   usage refusal records measured values.
6. **A1(b)** ignores a `::/0` route flagged `RTF_REJECT`. That is the kernel's
   `ip6_null_entry` (metric `ffffffff`, dev `lo`), which every network namespace carries. It
   rejects traffic rather than routing it.
7. **The directive check follows replpp, not the spec's whitespace list** (S08 review M1).
   replpp's `UsingDirectives.scanFor` is `line.trim().startsWith("//> using")`, and Java's
   `trim()` strips every char ≤ U+0020. So a `//>` that follows only bytes 0x00–0x20 from a line
   start is refused (`[[:cntrl:] ]` in the C locale). That set also includes 0x7f, which fails
   closed. Lines split at `\n` and `\r`, and `\f` and `\v` count as breaks too (fail closed).
   That replpp splits at `\r` is **verified**, by running replpp through the launch function:
   `x\r//> using dep …` and `val x = 1 \r//> using dep …` both attempted the download, while
   `val x = 1 //> using dep …` ran normally (egress-spec2 §1.1 had inferred it). Test 10(b) probes every leading byte
   0x00–0x20 plus 0x7f through the runner. Its negative controls (BOM-led, mid-line, `///>`)
   are not refused. The spec errata are the architect's: E1's set `[ \t\r\n\f\v]` should
   read "every char ≤ U+0020", and the "indented is inert (H15)" sentence is false.
8. **Order of checks.** Backend resolution runs first, read-only, so every record names a
   backend and its preconditions. Its refusal applies only after the argv checks, so 64 takes
   precedence over 69.
9. **Test-only entry points.** The probe, the sweep and test 9 run their command through
   `sbx_launch`, the same function production uses after its argv audit (§7.2: "exactly as a
   tool is launched"). Their records go to a separate audit file, with `frontend: lib-launch`.
10. **A6(e)** reads `/bin/bash -c 'cat /proc/self/environ'`. That is the environment the shell
    hands its descendants, which is what a launcher's JVM sees: exactly `PWD`, `SHLVL` and `_`
    are added. Reading `/proc/$$/environ` of the shell itself would only echo the parent's.
11. **`JAVA_OPTS` is fixed at `-Xmx2g`.** A caller can still pass `-J-Xmx<n>` or `-J-Xss<n>`
    (full-match exceptions; see **Heap** above).
12. **A6 checks values, not just keys** (S08 review M3). Every §7.3.4 value is asserted. Most
    are literals; `PATH`, `JAVA_HOME`, `GOROOT` and `DOTNET_ROOT` are checked against the
    image's `Config.Env`. The architect owes the spec line.

## Tests

```bash
bash cpg/sandbox/tests/test_sandbox.sh               # all tests (1-15, 21-26)
cpg/sandbox/tests/test_sandbox.sh --only 21,22,23,24,26   # the indexer tests (need the indexer image)
cpg/sandbox/tests/test_sandbox.sh --only 2,10        # a subset
cpg/sandbox/tests/mutants.sh                         # all mutants, each must be killed
```

Each check asserts content, never exit 0 alone:
- the probe must print `PROBE-DONE` with its full check count, and its `ENVV` values must match
  §7.3.4;
- the probe's launch template must equal a production record's `launcherArgv`, modulo ids, paths,
  `/scripts` and the tool (review m5);
- each FR-1 CPG must contain the fixture's method, as printed by a `joern --script` run through
  the runner;
- each `--help` must print its `Usage:` line;
- test 9 must see the Joern directory itself.

Test 2 and test 10(a, b) bracket their refusals with `docker events --filter event=create`. They
count only `ccg-sbx-<auditId>` containers whose `auditId` belongs to this run's own records, so
other agents' concurrent containers cannot fail them.

Two tests were added beyond §10's numbering:
- **Test 11, the polluted environment** (egress-spec2 §1.7). The runner is run through its `#!`
  line with every polluting input at once:
  - a fake `docker` first on `PATH`;
  - `DOCKER_HOST`, `DOCKER_CONTEXT` and `DOCKER_CONFIG` pointed elsewhere, the last at a
    `config.json` carrying `currentContext`, `credsStore` and `proxies`;
  - `HOME` with the same `config.json`;
  - `SHELLOPTS=xtrace`;
  - exported functions (`docker`, `sbx_dcfg_check`);
  - `BASH_ENV` and `ENV` naming marker-writing files.

  The record and `launcherArgv` must equal a clean run's, modulo `auditId`, timings and paths. No
  marker may be written, and no xtrace may appear. Then the runner's own config directory is
  tested, but only after the test checks that it is empty:
  - pre-populated with that `config.json` → 69, under both a polluted and a clean caller
    environment;
  - with a forbidden flag as well → 64 (precedence);
  - mode `0755` → 69;
  - replaced by a symlink → 69;
  - absent → recreated `0700` and empty.

  The suite's EXIT trap restores the directory. **Don't run the suite while a production run
  (S11) parses on the same host:** for test 11's duration, every concurrent run there refuses 69
  (review S08 p3-n2).
- **Test 12, signals and the live environment.** A `joern --script` that sleeps 60 s is started
  under a polluted environment (an extra variable, `TMPDIR`, a caller `DOCKER_CONFIG` whose
  `config.json` sets `proxies`). While it runs, the test asserts two things:
  - the runner's own `/proc/<pid>/environ` is exactly the closed set, with the pinned values;
  - the container's `Config.Env` carries no proxy variable. The Docker client would inject
    them from a `config.json` it read.

  This is what kills M29 (no re-exec) and M39 (`DOCKER_CONFIG` inherited). The runner is then
  sent SIGTERM while its container runs. The runner must exit 143 within 10 s. Its container, volumes and overlay scratch must be
  gone, it must write no record, and the `--output` lock must be free.

- **Test 13, an early SIGTERM** (review S08 p3-m1). Twenty trials (`S08_EARLY_TERM_TRIALS`
  overrides the count) each start the sleeping script and send SIGTERM 0–0.10 s after the overlay
  scratch appears, while `docker run` may be between its create and start calls. After a 2 s
  settle, no container or volume may carry any trial's `auditId`; every trial must exit 143,
  write no record and leave no overlay scratch. Today's runner before the fix leaked in 11 of 20
  trials; the fix is green over 40.
- **Test 14, `sbx_dcfg_check`'s tree rules, in library mode** (review S08 p3-m3). The runner is
  sourced, `sbx_home` is redefined to return a fixture home, and `sbx_dcfg_check` is called, so
  the real config directory is never touched. There are four fixtures: a home under `/tmp`
  (refused, "temporary tree"); `.local/state/ccg` as a symlink to a `0700` dir under `/tmp`
  (refused, "temporary tree": the check resolves the path, so an ancestor symlink cannot hide
  it); a world-writable (`0777`) ancestor **outside** `/tmp` (refused, naming it); and a clean
  control (accepted, config dir created `0700` and empty). The world-writable fixture lives at
  `<passwd home>/.ccg-s08-ww-<pid>`, which the EXIT trap removes. Under `/tmp` the `/tmp` rule
  would fire first and hide the other one.
- **Test 15, the closed environment's key list, in library mode** (review S08 p3-n1).
  `sbx_clean_env` is called with `SBX_SELF` pointed at a stub that prints `REEXEC`, under an
  environment that is exactly the closed set: no re-exec. With `A-B=1` or an exported function
  added, or a different `PATH` value in the block, it must re-exec. With a closed block whose
  shell `PATH`/`HOME` are changed after exec, it must not.

- **Test 27, the indexer wall-clock watchdog** (library mode: the test reassigns `SBX_INDEXER_WALL_S`
  to a few seconds after sourcing). A planted indexer that ignores TERM is KILLed: exit 124, one record
  (`exitCode: 124`, `refused: false`), no container, volume or scratch left. Controls: a tool that exits
  before the cap exits 0 with a 0 record, leaves no `Killed` notice on stderr and no orphan `sleep`; a
  non-indexer launch that outlives the cap is not capped; an exported `SBX_INDEXER_WALL_S=3` leaves the
  constant at `1800`. First wins: an INT while the sandbox runs, then the timer's `USR2` sent by hand
  while the INT handler's removal (slowed to 10 s) is running, gives exit 130 and no record (the real
  timer is already ended by then, so the signal is delivered directly). A second TERM during that removal
  is ignored (exit 130, container removed); a TERM and a second USR2 during the watchdog's removal change
  nothing (exit 124, one record, one expiry log line). A SIGKILLed runner: once its container is removed,
  the `--output` lock is free and no timer `sleep` survives the cap. Identity check without a container: a
  timer armed with a start time that does not match a live process signals nothing; with the right start
  time it does. With no readable start time `sbx_wd_arm` arms nothing and fails; a process name holding a
  newline and `) ` still yields the right start time. The SIGKILL test uses a 47 s cap, so the timer's
  first 30 s step ends with the runner gone and the orphan exits then (about 30 s of extra waiting).
  After a signalled exit (INT) no timer `sleep` remains. The timer sleeps use odd values (23, 27, 47) so
  the tests' `pgrep -fx` matches cannot be confused with another session's `sleep`.

**Hygiene** (S08 review p2-m1, p2-n2, p3-m2). The suite fails if any of the following is left at
the end. Each check counts only what this run made, so another session's sandbox work cannot
turn it red:
- a container or volume carrying one of this run's `auditId`s: those in its records, plus those
  of the signalled runs in tests 12 and 13, which write none;
- a `ccg-sbx-*` container or `ccg.sandbox` volume that `docker events --since <suite start>`
  shows created during the run, that still exists, and that is attributably this run's. It counts
  if its `inspect` JSON names a path under the suite's `$T`, since every launch mounts paths
  there, or, for a volume, if its `auditId` is one of this run's. A `/scripts` volume's device is
  the runner's own scratch, not a `$T` path. Others are counted as foreign and ignored. Checked:
  a foreign, never-started `ccg-sbx-*` container and a foreign `ccg.sandbox` volume created
  mid-run leave the suite green;
- a `/tmp/ccg-sbx.<auditId>.XXXXXX` scratch dir whose `auditId` is one of this run's. The runner
  names its scratch by `auditId` (review S08 p4-m1), so a leak is attributed the way a container
  is; any other new `/tmp/ccg-sbx.*` is foreign and ignored (checked: a foreign
  `/tmp/ccg-sbx.<uuid>.*` created mid-run leaves the suite green and untouched). The EXIT trap
  removes scratch that carries one of this run's `auditId`s, and nothing else;
- for `--help`, a `/tmp/ccg-sbx.*` scratch dir. `--help` writes no record, so its `auditId` is
  unknown to the suite; test 2 runs it under a private `/tmp` (bwrap, with `$T` and the runner's
  directory bound back), so a leak lands in the suite's own `$T` and is attributable to that
  invocation alone.

**Indexer tests 21–26** (egress-spec3 §10.1, P1-48b). They launch the `indexer` image
(`build.sh --indexer`; `CCG_TEST_INDEXER_IMAGE` overrides the tag). A test that needs it and
cannot find it prints `NOT RUN` (never `ok`) and the suite exits 3. No test runs `npm`, `pip` or
`uv`; the fixtures are plain files written by the suite.
- **21**: `SBX_FRONTENDS` equals the twelve names in order; the indexer names map to the fixed
  `/opt/indexer/bin` paths; `scip-java` is `unknown-frontend`.
- **22**: the grammar cases through the production runner, each an exact refusal (reason, exit, one
  record, no container) or a launch checked against its record (container, indexer image, argv plus
  the runner's `--output /out/<auditId>.scip`, `PYTHONSAFEPATH` last in `envKeys`); the
  `--project-version` cases mirror `--project-name`'s; the component-class cases (newline, `*`,
  space, `:`, `$`, `;`, non-ASCII, `=`); the symlink fixtures (`lnk -> /`, `../..`, `/src/sub`, a
  sibling directory sharing the tree's name prefix, a link that is absolute but resolves into the
  tree by its host path, a dangling link, a relative link to an absolute one, an absolute link
  below the root), each refused, with in-tree links as positive controls that launch and exit 0;
  the precedence cases (an escaping path outranks a missing `--project-version`); the image and
  mode cases (default image, `controlled`, `namespace`, `auto`); and the image-by-ID check
  (`launcherArgv`'s image element equals `toolchain.imageId`, for a parse and the indexer launches).
  `scip-typescript` and `scip-python` must each exit 0 and leave a non-empty `.scip`, the tool must
  not exit 126/127, and its output must carry `Indexing <cwd> with version`.
- **23**: the §7.3 launcher clause on the launched indexer records, the production
  `launcherArgv` equal to the probe launch's (modulo ids, paths and the tool tail), and probe A1–A7
  with A6 on the indexer pass-list on the `indexer` image.
- **24**: `index --help` of both tools through `lib-launch` (the grammar refuses `--help`); the
  sweep regex gains `pnpm|yarn|npm|pip|workspace|infer` for the indexers. It finds the four
  `scip-typescript` flags, which are in `forbidden-flags.txt`. Three doctored list copies make it
  fail for a named reason (the output must name it), and a control shows that the unwidened regex
  sees nothing, which is why the widening exists. It is an alarm; the grammar is the control.
- **25**: not duplicated; `ccg-versions` on the indexer image is
  `cpg/infra/toolchain/tests/test_toolchain.sh` section 5.
- **26**: through `lib-launch` under the indexer pass-list, `python3 -c 'import json'` runs with
  working directory `/src`, which holds a `json.py` that writes a marker into `/out`. The stdlib
  `json` must be imported and the marker absent; the control launch with `PYTHONSAFEPATH` unset
  inside the sandbox (`env -u`) must import `/src/json.py` and write the marker, so the absence
  cannot be vacuous. Test 7 and A6 pin the variable in the pass-list.

`run_r` bounds each production run with `timeout` (`RUN_TIMEOUT`, default 300 s): a hung run is a
red test, and the runner's own signal trap removes its container.

Test 5 also pins the labels the scoped check relies on: `--label ccg.sandbox.auditId=<auditId>`,
`--name ccg-sbx-<auditId>`, and the label on each volume.

### Mutants

`tests/mutants.sh` runs each mutant on a copy of the runner, the lists and `scripts/`, never on the real files. Every edit must apply. M42 breaks cleanup on purpose; its scratch lands in the suite's private `/tmp` under `$T`, which the suite removes. Each mutant's test run is bounded by `MUTANT_TIMEOUT` (default 900 s; the whole suite takes about 3 min). A run that hits it is reported as `killed (timeout)`: YC's runner re-execs forever, so it is killed that way. The harness never removes anything else: a new `/tmp/ccg-sbx.*` seen during a mutant is reported, not deleted, because it may be another session's. Last full run 2026-11-10 (P1-48b): every mutant killed (M01–YH, I01–I41, J01–J17; the earlier ones that quote runner lines were re-anchored after the edits). That covers:
- the first-round mutants M01–M21;
- the S08-gate round: M22–M31, X1–X6, X8–X10 and M32–M35;
- the ES2 v1.2 / Pass 2 follow-up: M37–M42 (`-p` dropped is M30, now also run against test 11);
- P3-7's M43;
- the Pass 3 follow-up: YA–YB (p3-m3, test 14), YC (p3-n1, test 15), and YD–YF (p3-m1; YD and YF re-anchored on the `sbx_on_signal` lines once `sbx_on_wall` held identical ones). YD is the rejected design (remove first, then kill), YE kills first but removes only once, and YF sends the client TERM;
- P1-48b's indexer mutants I01–I41 (the grammar, output, image label, backend and mode rules, the pass-list's `PYTHONSAFEPATH`, the indexer flag lists; the "Tests run" column names the test that must turn each red). A path-check mutant that admits `--cwd /` makes `scip-python` index the whole image: the suite's per-run `timeout` turns that hang into a red test;
- P1-48b's fix round J01–J17 (`--project-version`, the path component class, symlink containment, launch by image ID, containment-before-usage precedence);
- the indexer wall-clock watchdog K01–K14 (all test 27): K01 cap on a Joern frontend, K02 TERM for KILL,
  K03 removal skipped, K04a/K04b either handler no longer ignores USR2, K05 constant read from the
  environment, K06 no runner-identity check in the timer, K07 the timer keeps the lock fd, K08/K09 either
  handler resets INT/TERM/HUP to the default during removal, K10 the disarm does not stop the timer, K11 `sbx_on_signal` does not disarm it, K12 no check
  that the start time was read, K13 the stat file read line by line, K14 the 30 s step is the whole remainder;
- the Pass 4 follow-up: YG (p4-m1, `sbx_cleanup` stops removing the scratch, test 2) and YH
  (closed-env values read from live shell variables, test 15). M38 and M39 were re-anchored to
  the values-from-block lines, with the same breaks.

Outside the harness, because it leaves residue by design: **M36**, which drops `--rm`, was run once by hand (S08 fix round). The scoped hygiene checks went red, and its containers were removed. Re-run by hand after p3-m2, against test 5: the scoped hygiene still went red (4 checks, 0 foreign), and its 3 containers were removed.

| Mutant | Break | Tests run | Result | First failing check |
|---|---|---|---|---|
| M01 | drop --cap-drop ALL | 5 | killed | launcher clause |
| M02 | mount /tmp noexec | 1,3,5 | killed | pythonsrc parse |
| M03 | remove env -i (container) | 1,5 | killed | container A6a: probe's own environ keys == envKeys (from launcherArgv) |
| M04 | accept a \v-led //> (whitespace set without \v) | 10 | killed | 10b: leading byte 0x00 + //> using dep |
| M05 | allow a forbidden flag in --flag=value form | 2 | killed | javasrc2cpg --fetch-dependencies=value |
| M06 | drop --network none | 1,5 | killed | container probe: PROBE-DONE checks=122 fail=71 |
| M07 | drop --hostname localhost | 3,5 | killed | pythonsrc parse |
| M08 | /src as a plain ro bind (no noexec) | 1,5 | killed | /src is noexec (ro,relatime) |
| M09 | /out as a plain rw bind instead of the noexec overlay | 5 | killed | /out is noexec (rw,relatime) |
| M10 | N1 matches only the literal WSLInterop name (P0 plan D6 path) | 8 | killed | F13 |
| M11 | N1 ignores the magic rule (a) | 8 | killed | F3 |
| M12 | -J-Xmx exception by prefix, not full match | 2 | killed | launcher option '-J-Xmx1g-x' |
| M13 | namespace: the spec-literal /usr bind (recursive, wsl masked by tmpfs) | 1 | killed | namespace template (direct, R5): PROBE-DONE checks=114 fail=1 |
| M14 | JAVA_OPTS carries a -D property | 1 | killed | container A6c: JAVA_OPTS matches ^-Xmx[0-9]+[mg]( -Xss[0-9]+[km])?$ |
| M15 | a forbidden name on the pass-list | 1 | killed | container A6b: envKeys == offline pass-list |
| M16 | map the refused java alias | 2 | killed | joern-parse --language java |
| M17 | the tool output stem is not the auditId | 4 | killed | test 4: a launched parse writes /out/<auditId>.cpg.bin (exit 0, 0 B) |
| M18 | -v file bind for /scripts (no noexec) | 5,10 | killed | /scripts is noexec (ro,relatime) |
| M19 | C2 accepts any seccomp profile | 8 | killed | C2 '[name=seccomp,profile=/etc/docker/seccomp.json name=cgroupns]' |
| M20 | forbidden flags still launch (no pre-launch check) | 2 | killed | javasrc2cpg --fetch-dependencies |
| M21 | add -e to docker run | 1,5 | killed | launcher clause |
| M22 | M1: directive check strips only [[:space:]] (control bytes admitted) | 10 | killed | 10b: leading byte 0x00 + //> using dep |
| M23 | M2: SBX_BACKEND_REFUSAL not initialised (crash, no record) | 2 | killed | M2: no arguments at all |
| X1 | M4: jimple2cpg --full-resolver moved allowed -> forbidden | 2,6 | killed | M4: reviewed allowed flag jimple2cpg --full-resolver is accepted and recorded (hits=1 allowed=) |
| X2 | M4: javasrc2cpg --delombok-mode moved allowed -> forbidden | 2,6 | killed | M4: reviewed allowed flag javasrc2cpg --delombok-mode is accepted and recorded (hits=1 allowed=) |
| X3 | M4: c2cpg --compilation-database moved allowed -> forbidden | 2,6 | killed | M4: reviewed allowed flag c2cpg --compilation-database is accepted and recorded (hits=1 allowed=) |
| X4 | M3: GOTOOLCHAIN dropped from the pass-list | 1,7 | killed | offline pass-list (container) is the 13 §7.3.4 keys |
| X5 | M3: GOTOOLCHAIN=auto | 1,7 | killed | M3 (text): fixed §7.3.4 values (GOTOOLCHAIN=local, DOTNET_*=1, ...) |
| X6 | M3: DOTNET_CLI_TELEMETRY_OPTOUT=0 | 1,7 | killed | M3 (text): fixed §7.3.4 values (GOTOOLCHAIN=local, DOTNET_*=1, ...) |
| X8 | m3: @AUDIT_ID@ substitution removed | 3,10 | killed | m3: --param out=/out/@AUDIT_ID@.txt wrote /out/<auditId>.txt (rc 0) |
| X9 | m3: record the pre-substitution argv | 3,10 | killed | m3: record toolArgv carries the substituted value |
| X10 | m3: refusal records carry toolArgv [] | 2,10 | killed | javasrc2cpg --fetch-dependencies |
| M24 | m4: netnsDistinct claimed true before any launch | 2 | killed | javasrc2cpg --fetch-dependencies |
| M25 | m4: invalid --mode lands in egressMode | 2 | killed | m4: invalid --mode |
| M26 | m2: no lock on --output | 3 | killed | m2: --output held by a concurrent run is refused |
| M27 | m6: namespace ro binds not recorded in mounts | 1 | killed | m6: namespace record mounts list every --ro-bind/--bind/--tmpfs target (90 binds, 87 unlisted ['/usr/bin', '/u |
| M28 | m5: production launches with a flag the probe template lacks | 1,5 | killed | m5: probe and production templates differ |
| M29 | n1: no env -i re-exec (DOCKER_HOST reaches docker) | 2,12 | killed | §1.7: the running runner's environ keys are the closed set (AGENT_WORKDIR AI_AGENT CCG_AUDIT_DIR CCG_TOOLCHAI |
| M30 | n1 / §1.7: shebang without -p (BASH_ENV/ENV run) | 2,11 | killed |   BASH_ENV file was not executed |
| M31 | m1: auto never falls back to namespace | 8 | killed | auto falls back to namespace when the container image is missing (69) |
| M32 | ES2 §1.1: no \r line split (x\r//> admitted) | 10 | killed | 10b: directive shape #6 $'println(1)\r//> using dep a:b' |
| M33 | ES2 §1.5: refused toolArgv substituted, not the caller's argv | 2 | killed | javasrc2cpg --fetch-dependencies |
| M34 | ES2 §1.5: --help writes a record | 2 | killed | --help exits 0 and writes no record (rc 0, 1 new) |
| M37 | §1.7: skip the Docker-config dir check | 11 | killed | pre-populated Docker-config dir (credsStore/proxies/currentContext) -> 69, one record (rc 0) |
| M38 | §1.7: DOCKER_HOST left inherited (not pinned) | 11 | killed | polluted run launches like the clean one (rc 69, 1 record) |
| M39 | §1.7: DOCKER_CONFIG left inherited (caller config.json read) | 11,12 | killed | §1.7: pinned values (DOCKER_HOST, DOCKER_CONFIG, HOME, PATH) |
| M40 | p2-m1: container launched without the auditId label | 5 | killed | launcher clause |
| M41 | p2-n1: launcher in the foreground (signal trap deferred, container survives) | 12 | killed | runner still alive 10 s after SIGTERM |
| M42 | p2-n2: scratch made before --help, with no trap at the mktemp | 2 | killed | --help leaves no /tmp/ccg-sbx.* scratch (private /tmp: ccg-sbx.74573e16-f376-4748-a9b4-630a8758d169.mUl7fw) |
| M43 | P3-7: the content check accepts an empty CPG | 3 | killed | P3-7: the empty CPG is CONTENT=EMPTY, the real one ok, and the check exits non-zero (rc 0) |
| M35 | M4: the reviewed allowed list swapped (one out, one look-alike in) | 2 | killed | M4: reviewed allowed flag jimple2cpg --full-resolver is accepted and recorded (hits=0 allowed=) |
| YA | p3-m3: the /tmp\|/var/tmp\|/dev/shm tree rule deleted | 14 | killed | home under /tmp -> refused, 'under a temporary tree' (rc=1 fail=/tmp/ccg-s08-test.rXzIbQ/h-tmp/.local/state/cc |
| YB | p3-m3: the world-writable ancestor test disabled | 14 | killed | world-writable ancestor outside /tmp -> refused, names it (rc=0 fail=) |
| YC | p3-n1: env keys from compgen -e (non-identifier entries unseen) | 15 | killed (timeout) | test run exceeded 60 s (`MUTANT_TIMEOUT=60`; its runner re-execs forever) |
| YD | p3-m1: the rejected design -- rm the container by name, then kill the client | 13 | killed | every trial exited 143 and left no container or volume (20 auditIds) |
| YE | p3-m1: client killed first, but one removal pass (no settle loop) | 13 | killed | every trial exited 143 and left no container or volume (20 auditIds) |
| YF | p3-m1: the client gets TERM, not KILL (docker run proxies it and stays up) | 12 | killed | runner still alive 10 s after SIGTERM |
| YG | p4-m1: sbx_cleanup stops removing SBX_SCRATCH (every invocation leaks it) | 2 | killed | no /tmp/ccg-sbx.<auditId>.* scratch of this run left behind (121 new on the host; 121 ours) |
| YH | p4: closed-env values read from live shell variables, not the exec'd block | 15 | killed | control: exactly the closed set -> no re-exec (REEXEC) |
| I01 | (21) scip-python dropped from SBX_FRONTENDS | 21,22 | killed | (21) SBX_FRONTENDS == §7.4's twelve names in order (got: pysrc2cpg jssrc2cpg javasrc2cpg jimple2cpg kotlin2cp |
| I02 | (21) the two indexer names swapped (order is the contract) | 21 | killed | (21) SBX_FRONTENDS == §7.4's twelve names in order (got: pysrc2cpg jssrc2cpg javasrc2cpg jimple2cpg kotlin2cp |
| I03 | (22) --cwd is no longer mandatory | 22 | killed | 22 usage: scip-typescript without --cwd |
| I04 | (22) --environment is no longer mandatory for scip-python | 22 | killed | 22 usage: scip-python without --environment |
| I05 | (22) --environment accepted for scip-typescript | 22 | killed | 22 scip-typescript --environment |
| I06 | (22) --project-name accepted for scip-typescript | 22 | killed | 22 scip-typescript --project-name |
| I07 | (22) a repeated --cwd is accepted | 22 | killed | 22 scip-typescript repeated --cwd |
| I08 | (22) a repeated --environment is accepted | 22 | killed | 22 scip-python repeated --environment |
| I09 | (22) a repeated --project-name is accepted | 22 | killed | 22 scip-python repeated --project-name |
| I10 | (22) path check: `..` components allowed | 22 | killed | 22 scip-typescript --cwd '/src/../x' |
| I11 | (22) path check: `.` components allowed | 22 | killed | 22 scip-typescript --cwd '/src/./a' |
| I12 | (22) path check: empty components allowed | 22 | killed | 22 scip-typescript --cwd '/src//a' |
| I13 | (22) path check: any /src prefix (/srcfoo, /src2) allowed | 22 | killed | 22 scip-typescript --cwd '/' |
| I14 | (22) --project-name pattern relaxed (a leading - or . allowed) | 22 | killed | 22 scip-python --project-name '-x' |
| I15 | (22) --project-name length bound dropped | 22 | killed | 22 scip-python --project-name 'aaaaaaaaaaaa' |
| I16 | (22) first element need not be `index` | 22 | killed | 22 scip-typescript first element 'npm' (the rest is a valid grammar) |
| I17 | (22) an unknown element is ignored (abbreviations, --opt=value pass) | 22 | killed | 22 scip-typescript --cwd=/src form |
| I18 | (22) a caller --output/-o is accepted into the grammar and the output refusal removed | 22 | killed | 22 scip-typescript caller --output |
| I19 | (22) the runner appends the wrong output path | 22 | killed | 22 launch: scip-typescript index --cwd /src: record |
| I20 | (22) the indexer output keeps the parse-frontend suffix | 22 | killed | 22 launch: scip-typescript index --cwd /src: record |
| I21 | (22) a missing mandatory element outranks a forbidden one (usage before forbidden-flag) | 22 | killed | 22 scip-typescript bare --version |
| I22 | (22) the ccg-toolchain-indexer=1 label check removed | 22 | killed | 22 indexer entry on the default toolchain image: backend-unavailable (69) |
| I23 | (22) controlled mode no longer unknown-frontend for an indexer | 22 | killed | 22 controlled: scip-typescript is unknown-frontend |
| I24 | (22) the namespace backend is tried for an indexer (precondition, not unavailable) | 22 | killed | 22 namespace backend: backend-unavailable (exit 69) |
| I25 | (22) auto may fall back to namespace for an indexer | 22 | killed | 22 auto + toolchain image: still backend-unavailable (never the namespace precondition) |
| I26 | (21)/(22) the indexer executable is resolved through PATH, not the fixed path | 21,22 | killed | (21) indexer names map to fixed absolute paths under /opt/indexer; c2cpg stays a $JOERN_HOME file |
| I27 | (22) the indexer entries point at a directory that has no tool | 22 | killed |   scip-typescript exit 0 and /out/<auditId>.scip is non-empty (rc 127) |
| I28 | (22) scip-python is not recognised as an indexer (parse path, no env, no label check) | 21,22 | killed | (21) indexer names map to fixed absolute paths under /opt/indexer; c2cpg stays a $JOERN_HOME file |
| I29 | (22)/(23) an indexer launch carries a flag the probe template lacks | 22,23 | killed | (23) indexer production and probe launchers differ |
| I30 | (23) an indexer launch without --cap-drop ALL (launcher clause) | 23 | killed | (23) launcher clause on indexer records |
| I31 | (26) PYTHONSAFEPATH missing from the indexer pass-list | 7,22,23,26 | killed | indexer pass-list is the 13 offline keys plus PYTHONSAFEPATH |
| I32 | (26) PYTHONSAFEPATH=0 on the indexer pass-list | 7,22,23,26 | killed | indexer PYTHONSAFEPATH has the value 1 |
| I33 | (26) PYTHONSAFEPATH on every entry, not only the indexers | 7,1 | killed | offline pass-list (container) is the 13 §7.3.4 keys |
| I34 | (26) a forbidden indexer name (PYTHONPATH) on the indexer pass-list | 7 | killed | indexer pass-list is the 13 offline keys plus PYTHONSAFEPATH |
| I35 | (24) scip-typescript --pnpm-workspaces dropped from the forbidden list | 24 | killed | (24) indexer sweep |
| I36 | (24) scip-typescript --infer-tsconfig entry made stale | 24 | killed | (24) indexer sweep |
| I37 | (24) scip-typescript --yarn-workspaces listed in both files | 24 | killed | (24) indexer sweep |
| I38 | (26) PYTHONSAFEPATH missing from the indexer pass-list (test 26 alone) | 26 | killed | (26) with PYTHONSAFEPATH the stdlib json is imported, not /src/json.py |
| I39 | (23) PYTHONSAFEPATH missing from the indexer pass-list (test 23 alone) | 23 | killed | 23 launch: scip-typescript (a launched record for the clause and equivalence checks): record |
| I40 | (23) NODE_OPTIONS on the indexer pass-list (A6) | 23 | killed | 23 launch: scip-typescript (a launched record for the clause and equivalence checks): record |
| I41 | (26) the indexer pass-list is never applied to a lib launch (SBX_INDEXER reset after use) | 26 | killed | (26) with PYTHONSAFEPATH the stdlib json is imported, not /src/json.py |
| J01 | --project-version is no longer mandatory for scip-python | 22 | killed | 22 usage: scip-python without --project-version |
| J02 | --project-version accepted for scip-typescript | 22 | killed | 22 scip-typescript --project-version |
| J03 | a repeated --project-version is accepted | 22 | killed | 22 scip-python repeated --project-version |
| J04 | --project-version pattern relaxed (a leading - or . allowed) | 22 | killed | 22 scip-python --project-version '-x' |
| J05 | path component class dropped (only non-empty, not . or ..) | 22 | killed | 22 scip-typescript --cwd component class: $'/src/a\nb' |
| J06 | path component class admits a space | 22 | killed | 22 usage: scip-typescript without --cwd |
| J07 | symlink containment check removed | 22 | killed | 22 symlink: scip-typescript --cwd /src/lnk_root is refused |
| J08 | containment checks --cwd only, not --environment | 22 | killed | 22 symlink: scip-python --environment /src/lnk_root/etc/hostname is refused |
| J09 | containment: a `..` reached through a relative link may leave the tree | 22 | killed | 22 symlink: scip-typescript --cwd /src/lnk_up is refused |
| J10 | containment does not require --cwd to be a directory | 22 | killed | 22 symlink: scip-typescript --cwd /src/env.json is refused |
| J11 | containment accepts a component that does not exist (dangling link, missing file) | 22 | killed | 22 containment: a link through a missing component and back by .. (--cwd /src/lnk_mid_dd) is refused |
| J12 | the image is launched by name, not by the inspected ID | 22 | killed | 22 image by ID: parse launch |
| J13 | scip-python alone resolves to a wrong path (rc 127; scip-typescript unaffected) | 22 | killed |   scip-python: the tool started (rc 127 is not 126/127) |
| J14 | containment walks from the wrong base (the host root) | 22 | killed | 22 usage: scip-python without --cwd |
| J15 | containment follows an absolute symlink by its host path (the realpath-only rule) | 22 | killed | 22 symlink: scip-typescript --cwd /src/lnk_hostabs is refused |
| J16 | a containment refusal no longer outranks a missing mandatory element (usage first) | 22 | killed | 22 precedence: an escaping --cwd outranks a missing --project-version |
| J17 | --environment no longer required to be a regular file (the -f check dropped: a directory or a FIFO passes) | 22 | killed | 22 --environment names a directory (/src/sub): refused |
| K01 | (27) the wall-clock cap is applied to every frontend, not only the indexers | 27 | killed | (27) a non-indexer launch is not capped: it outlives the cap and exits 0 (rc 124) |
| K02 | (27) the watchdog (and the signal handler) send TERM instead of KILL to the client | 27 | killed (timeout) | test run exceeded 150 s |
| K03 | (27) the watchdog skips the removal of the run's containers | 27 | killed | (27) watchdog: no container labelled with its auditId remains |
| K04a | (27) first wins: sbx_on_signal no longer ignores USR2 (a late timer signal re-enters as a watchdog) | 27 | killed | (27) first wins: INT, then the timer's USR2 during the handler's removal: exit 130, not 124 (rc 124) |
| K04b | (27) first wins: sbx_on_wall no longer ignores USR2 (a second USR2 re-enters the watchdog handler) | 27 | killed | (27) the watchdog logged its expiry exactly once (a second USR2 did not re-enter the handler) |
| K05 | (27) the wall-clock constant is read from the environment | 27 | killed | (27) library mode: an exported SBX_INDEXER_WALL_S=3 does not change the constant (got 3) |
| K06 | (27) the timer signals without checking that the runner is the same process (PID reuse) | 27 | killed | (27) timer identity: a recorded start time that does not match the process: no signal is sent (got DEAD) |
| K07 | (27) the timer keeps the runner's --output lock fd open | 27 | killed | (27) SIGKILLed runner: the --output lock is free once its container is gone (the orphan timer does not pin it) |
| K08 | (27) sbx_on_signal resets INT/TERM/HUP to the default during the removal (a second TERM aborts it) | 27 | killed | (27) a second TERM during the INT handler's removal is ignored: exit 130 (rc 143), no record |
| K09 | (27) sbx_on_wall resets INT/TERM/HUP to the default during the removal | 27 | killed | (27) a TERM and a second USR2 during the watchdog's removal change nothing: exit 124 (rc 143), one record |
| K10 | (27) the disarm does not stop the timer (its sleep outlives a normal exit) | 27 | killed | (27) control: after a normal exit (rc 0) no 'sleep 29' timer process remains |
| K11 | (27) sbx_on_signal does not disarm the timer (its sleep outlives a signalled exit) | 27 | killed | (27) after a signalled exit the timer is ended too: no 'sleep 27' remains |
| K12 | (27) no check that the runner's start time was read (the identity check fails open) | 27 | killed | (27) no readable start time: sbx_wd_arm arms no timer and fails (got rc=0 pid=295820) |
| K13 | (27) /proc stat read line by line (a newline in the process name hides field 22) | 27 | killed | (27) start time read for a process whose name contains a newline and ') ' (got '', want '939397') |
| K14 | (27) the timer's step is the whole remainder (an orphan outlives a dead runner up to the cap) | 27 | killed | (27) SIGKILLed runner: one 30 s step after arming, the orphan timer has exited (no 'sleep 47' or 'sleep 17') |
