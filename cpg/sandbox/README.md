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
| `forbidden-flags.txt`, `allowed-flags.txt` | The §7.3.3 flag lists: 13 forbidden, 3 allowed, for Joern v4.0.579 |
| `tests/test_sandbox.sh` | §10 tests 1–10 on both backends |
| `tests/probe.pl` | The escape-axis probe, axes A1–A7. It is Perl because the image has no Python. |
| `tests/mutants.sh` | The mutation harness: each mutant must turn its test red |

## Using the runner

```bash
cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern-parse \
    --source <repo dir> --output <out dir> --codebase <repoId> -- /src --language pythonsrc
cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern \
    --script-file extract.sc --source <dir> --output <cpg dir> \
    -- --script /scripts/extract.sc --param cpgFile=/out/<auditId>.cpg.bin
```

**Exit codes.**
- The tool's own exit code, when the tool ran.
- `64`: refused before launch (argv, frontend, mode, directive or usage).
- `69`: the backend refused or is unavailable.

**Audit record.** Every invocation, refusals included, appends one `sandbox` record (§5.3) to
`$CCG_AUDIT_DIR/sandbox.jsonl`. The default is `cpg/.audit/`, which is gitignored.

**Environment.** The runner reads only two variables: `CCG_AUDIT_DIR`, and
`CCG_TOOLCHAIN_IMAGE` (default: `cpg/infra/toolchain/build.sh --print-tag`, run under `env -i`).
It enforces that itself:
- If the environment holds anything else, `sbx_main` re-execs under `env -i`. It keeps only
  those two variables, a fixed `PATH`, and the `HOME` from the passwd database. So
  `DOCKER_HOST`, `DOCKER_CONTEXT` and the like never reach `docker`; the daemon is the one your
  default Docker context names.
- The shebang `#!/bin/bash -p` makes bash ignore `BASH_ENV`, `ENV` and exported functions.
- **Run the runner directly, not as `bash offline-run.sh`.** That form bypasses the shebang, and
  `BASH_ENV` would run before line 1, which no script can prevent.

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

**Heap.** `JAVA_OPTS` is fixed at `-Xmx2g`. To raise a frontend's heap, pass `--frontend-args
-J-Xmx<n>` (or `-J-Xmx<n>` to a direct frontend): the frontend's own launcher consumes it. A
`joern-parse -J-Xmx<n>` *before* `--frontend-args` sizes only the `joern-parse` JVM, never the
frontend it spawns (§7.3.5).

**One run per `--output`.** The runner takes `flock -n` on the output directory's own file
descriptor, so the lock creates no file. A second run into a directory that is in use is
refused (exit 64, `usage`). Test 3 runs two invocations concurrently and expects exactly one
refusal (egress-spec2 §1.3). Without the lock, three concurrent runs into one `--output` all
exited 0 (S08 review m2). On this 5.15 kernel overlayfs only warns about a shared upperdir, and
the result is undefined.

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
cpg/sandbox/tests/test_sandbox.sh                    # all tests, ~80 s
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

### Mutants

`tests/mutants.sh` runs each mutant on a copy of the runner and lists, never on the real files. Every edit must apply. Last run 2026-10-29: **44 of 44 killed**. That covers the 21 first-round mutants, one or more per S08-gate finding (M22–M31), the reviewer's X1–X6 and X8–X10, and the egress-spec2 additions (M32–M35). X7 is not re-run: the reviewer killed it, and its code path is unchanged.

Outside the harness, because it leaves residue by design: **M36**, which drops `--rm`, was run once by hand with `--only 9`. The scoped hygiene checks went red, and its two containers were removed afterwards.

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
| M29 | n1: no env -i re-exec (DOCKER_HOST reaches docker) | 2 | killed |   docker info still answered (seccomp=builtin, backend container) |
| M30 | n1: shebang without -p (BASH_ENV runs) | 2 | killed |   BASH_ENV file was not executed |
| M31 | m1: auto never falls back to namespace | 8 | killed | auto falls back to namespace when the container image is missing (69) |
| M32 | ES2 §1.1: no \r line split (x\r//> admitted) | 10 | killed | 10b: directive shape #6 $'println(1)\r//> using dep a:b' |
| M33 | ES2 §1.5: refused toolArgv substituted, not the caller's argv | 2 | killed | javasrc2cpg --fetch-dependencies |
| M34 | ES2 §1.5: --help writes a record | 2 | killed | --help exits 0 and writes no record (rc 0, 1 new) |
| M35 | M4: the reviewed allowed list swapped (one out, one look-alike in) | 2 | killed | M4: reviewed allowed flag jimple2cpg --full-resolver is accepted and recorded (hits=0 allowed=) |
